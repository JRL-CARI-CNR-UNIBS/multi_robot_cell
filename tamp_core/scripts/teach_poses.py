#!/usr/bin/env python3
"""Teach a scene's object and place poses by hand, and write them into its YAML.

Move the arm by hand (TIAGo: gravity-compensation mode) until the gripper is where it will grasp an
object -- at its spawn, or at a place pose -- then capture. The tool pose read from TF is turned
into the OBJECT pose with the scene's own grasp: ``object = T_ee * grasp^-1`` (``grasp`` is the
tool pose in the object frame, the convention of every stage of the pipeline). The YAML is patched
in place, comments and layout kept, after a ``.bak`` copy.

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    ros2 launch tiago_cell bringup.launch.py use_mock_hardware:=false   # the real robot up
    ros2 run multi_robot_cell_tamp teach_poses.py <scene.yaml> --table-top <z>

Commands at the prompt:

    list                       every teachable item: objects (spawn), slots and tasks (place)
    teach <item> <robot>       capture <robot>'s gripper now as <item>'s pose (shown, kept)
    show                       the captured poses not yet written
    write                      write the captured poses into the YAML (and a .bak)
    quit

What is written, and why:

* x, y from the gripper. z by default (``--z table``) is NOT the taught height: the table top
  (``--table-top <z>``, measured once on the robot; without it, the origin of the scene's
  ``support_surface`` TF frame -- not the upper surface for TIAGo's ``table_top_link``) and the
  object sits on it with the scenes' 2 mm clearance (``z = top + size_z / 2 + 0.002``) -- a hand-held gripper is rarely within a few
  millimetres of the right height, and a box inside the table makes every plan fail.
  ``--z taught`` keeps the measured height (objects on a fixture, stacked levels).
* Orientation: ``yaw`` only (``--orientation yaw``, default), since every scene object rests on a
  horizontal face; a capture more than ``--tilt-warn`` degrees off level is reported (the gripper
  was not vertical). ``--orientation full`` writes roll, pitch and yaw.
* A slot's place pose uses the grasp and size of its first candidate (noted if candidates differ).
* Before every capture the joints shared by all robots' ``home`` (TIAGo: ``torso_lift_joint``) are
  compared with ``/joint_states``: the pipeline plans at that frozen value, and teaching at another
  height changes nothing in the pose but everything in what the arm can reach.

After writing, re-run the whole offline pipeline: every trajectory depends on these poses.
``--fake-ee x,y,z,roll,pitch,yaw`` replaces the TF lookup (no ROS needed) to try the YAML patching.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
import threading
import time

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# pose algebra (same conventions as vamp_collision_engine.pose_from_yaml / tf2 setRPY)
# --------------------------------------------------------------------------- #
def pose_from_yaml(p: dict) -> np.ndarray:
    r, pt, y = (float(p.get(k, 0.0)) for k in ("roll", "pitch", "yaw"))
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(pt), math.sin(pt),
                              math.cos(y), math.sin(y))
    T = np.eye(4)
    T[:3, :3] = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
                 @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                 @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
    T[:3, 3] = [float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))]
    return T


def rpy_of(R: np.ndarray):
    """(roll, pitch, yaw) with R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    return math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])


def quat_to_R(x, y, z, w) -> np.ndarray:
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def object_from_tool(T_ee: np.ndarray, grasp: dict) -> np.ndarray:
    return T_ee @ np.linalg.inv(pose_from_yaml(grasp))


def level_yaw(T_obj: np.ndarray) -> float:
    """Yaw of a box resting on a horizontal face: the heading of its x axis projected on the
    table. (A raw Euler yaw is wrong whenever the capture is not exactly level.)"""
    x = T_obj[:3, 0]
    return math.atan2(x[1], x[0])


# --------------------------------------------------------------------------- #
# the scene
# --------------------------------------------------------------------------- #
class Scene:
    def __init__(self, path: str):
        self.path = path
        self.text = open(path).read()
        self.spec = yaml.safe_load(self.text)
        self.objects = {o["id"]: o for o in self.spec.get("objects") or []}

    def items(self):
        """{name: (kind, grasp, size, current pose)} -- kind 'spawn' (object) or 'place'."""
        out = {}
        for oid, o in self.objects.items():
            out[oid] = ("spawn", o.get("grasp", {}), o["size"], o.get("spawn", {}))
        for s in self.spec.get("slots") or []:
            cands = [self.objects[c] for c in s.get("candidates", []) if c in self.objects]
            if not cands:
                continue
            if any(c.get("grasp") != cands[0].get("grasp") or c["size"] != cands[0]["size"]
                   for c in cands):
                print(f"  note: slot {s['id']}: candidates differ in grasp/size; "
                      f"using {cands[0]['id']}'s")
            out[s["id"]] = ("place", cands[0].get("grasp", {}), cands[0]["size"], s.get("place", {}))
        for t in self.spec.get("tasks") or []:
            if "place" in t and t.get("object") in self.objects:
                o = self.objects[t["object"]]
                out[t["id"]] = ("place", o.get("grasp", {}), o["size"], t["place"])
        return out

    def patch(self, name: str, kind: str, pose: dict) -> None:
        """Replace ``<kind>: {...}`` of the item ``name`` in the YAML text, layout kept."""
        flow = "{" + ", ".join(f"{k}: {v:.4f}" for k, v in pose.items()) + "}"
        lines = self.text.split("\n")
        # block form:   - id: name   ...   <kind>: {...}   (before the next list item)
        for i, line in enumerate(lines):
            m = re.match(r"^(\s*)-\s+id:\s*" + re.escape(name) + r"\s*(#.*)?$", line)
            if not m:
                continue
            ind = len(m.group(1))
            for j in range(i + 1, len(lines)):
                if lines[j].strip() and (len(lines[j]) - len(lines[j].lstrip())) <= ind:
                    break
                k = re.match(r"^(\s*" + kind + r":\s*)\{[^}]*\}(.*)$", lines[j])
                if k:
                    lines[j] = k.group(1) + flow + k.group(2)
                    self.text = "\n".join(lines)
                    return
        # inline form:  - {id: name, ..., place: {...}}
        for i, line in enumerate(lines):
            if re.search(r"\{\s*id:\s*" + re.escape(name) + r"\s*,", line):
                new = re.sub(kind + r":\s*\{[^}]*\}", f"{kind}: {flow}", line, count=1)
                if new != line:
                    lines[i] = new
                    self.text = "\n".join(lines)
                    return
        raise KeyError(f"{name}: no `{kind}: {{...}}` found to replace in {self.path}")

    def write(self) -> str:
        yaml.safe_load(self.text)          # refuse to write something that no longer parses
        bak = self.path + ".bak"
        shutil.copyfile(self.path, bak)
        with open(self.path, "w") as f:
            f.write(self.text)
        return bak


# --------------------------------------------------------------------------- #
# where the poses come from
# --------------------------------------------------------------------------- #
class TfSource:
    """Live robot: TF lookups and /joint_states."""

    def __init__(self, scene: Scene):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from tf2_ros import Buffer, TransformListener

        rclpy.init()
        self.node = Node("teach_poses")
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self.node)
        self.joints: dict[str, float] = {}
        self.node.create_subscription(JointState, "/joint_states", self._js, 10)
        threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True).start()
        self.base = scene.spec.get("base_frame", "base_footprint")
        homes = [set((r.get("home") or {})) for r in scene.spec["robots"].values()]
        shared = set.intersection(*homes) if homes else set()
        first = next(iter(scene.spec["robots"].values()))
        self.frozen = {j: float(first["home"][j]) for j in shared}

    def _js(self, msg):
        for n, p in zip(msg.name, msg.position):
            self.joints[n] = p

    def frozen_mismatch(self, tol: float = 0.01) -> dict:
        return {j: (self.joints.get(j), v) for j, v in self.frozen.items()
                if self.joints.get(j) is None or abs(self.joints[j] - v) > tol}

    def lookup(self, target: str, samples: int = 10, period: float = 0.05) -> np.ndarray:
        """Mean of ``samples`` TF lookups base -> target (the hand is held still)."""
        from rclpy.time import Time
        Ts = []
        deadline = time.time() + 5.0
        while len(Ts) < samples and time.time() < deadline:
            try:
                t = self.tf.lookup_transform(self.base, target, Time())
            except Exception:  # noqa: BLE001  (not available yet)
                time.sleep(0.1)
                continue
            q, p = t.transform.rotation, t.transform.translation
            T = np.eye(4)
            T[:3, :3] = quat_to_R(q.x, q.y, q.z, q.w)
            T[:3, 3] = [p.x, p.y, p.z]
            Ts.append(T)
            time.sleep(period)
        if not Ts:
            raise RuntimeError(f"no TF {self.base} -> {target} within 5 s")
        T = np.eye(4)
        T[:3, 3] = np.mean([x[:3, 3] for x in Ts], axis=0)
        U, _, Vt = np.linalg.svd(sum(x[:3, :3] for x in Ts))   # mean rotation (chordal)
        T[:3, :3] = U @ Vt
        spread = max(np.linalg.norm(x[:3, 3] - T[:3, 3]) for x in Ts)
        if spread > 0.002:
            print(f"  note: the hand moved {spread * 1000:.1f} mm during the capture")
        return T


class FakeSource:
    """``--fake-ee``: a fixed tool pose and a table top, to try the patching without ROS."""

    def __init__(self, ee: str, table_top: float, base: str):
        x, y, z, r, p, w = (float(v) for v in ee.split(","))
        self.T = pose_from_yaml({"x": x, "y": y, "z": z, "roll": r, "pitch": p, "yaw": w})
        self.top, self.base = table_top, base

    def frozen_mismatch(self, tol: float = 0.01) -> dict:
        return {}

    def lookup(self, target: str, samples: int = 10, period: float = 0.05) -> np.ndarray:
        if target == "__support__":
            T = np.eye(4)
            T[2, 3] = self.top
            return T
        return self.T


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="scene YAML to patch (e.g. tiago_cell/scenes/tamp_task_numbers.yaml)")
    ap.add_argument("--z", choices=["table", "taught"], default="table",
                    help="table: z from the table top (default; see --table-top); taught: the measured height")
    ap.add_argument("--table-top", type=float, default=None,
                    help="z of the table's UPPER surface in the scene's base frame (m), measured on the "
                         "robot (e.g. fingertip on the table, tf2_echo). Recommended on a real robot: "
                         "without it the support_surface link is looked up in TF, whose origin is the "
                         "link's frame (TIAGo's table_top_link: the slab centre, 2.5 cm below the "
                         "surface) and which is not published at all by a use_mock_hardware:=false "
                         "bringup")
    ap.add_argument("--orientation", choices=["yaw", "full"], default="yaw")
    ap.add_argument("--clearance", type=float, default=0.002, help="gap above the table (m), --z table")
    ap.add_argument("--tilt-warn", type=float, default=5.0, help="degrees off level that are reported")
    ap.add_argument("--fake-ee", default=None, help="x,y,z,roll,pitch,yaw: skip ROS (testing)")
    ap.add_argument("--fake-table-top", type=float, default=0.792,
                    help="table top z with --fake-ee when --table-top is not given")
    args = ap.parse_args(argv)

    scene = Scene(os.path.abspath(args.scene))
    items = scene.items()
    robots = scene.spec["robots"]
    base = scene.spec.get("base_frame", "base_footprint")
    src = (FakeSource(args.fake_ee, args.fake_table_top, base) if args.fake_ee else TfSource(scene))
    support = scene.spec.get("support_surface")
    if args.z == "table" and not support:
        print("  note: the scene names no support_surface: falling back to --z taught")
        args.z = "taught"
    pending: dict[str, tuple[str, dict]] = {}
    print(f"scene {scene.path}: {len(items)} teachable item(s), robots {list(robots)}, frame {base}")
    for r, c in robots.items():
        print(f"  {r}: tool frame {c['ee_link']}")
    if args.z == "table" and args.table_top is not None:
        print(f"  object height from the table top z = {args.table_top} + size/2 + {args.clearance} m")
    elif args.z == "table":
        print(f"  object height from the top of '{support}' + size/2 + {args.clearance} m")
        if not args.fake_ee:
            print(f"  WARNING: this is the ORIGIN of TF frame '{support}', which may not be the upper "
                  f"surface (TIAGo: the slab centre). Prefer --table-top <z>.")

    while True:
        try:
            cmd = input("teach> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            print()
            cmd = ["quit"]
        if not cmd:
            continue
        if cmd[0] == "list":
            for n, (kind, _g, _s, cur) in items.items():
                print(f"  {'*' if n in pending else ' '} {n:28s} {kind:5s} {cur}")
        elif cmd[0] == "teach" and len(cmd) == 3:
            name, robot = cmd[1], cmd[2]
            if name not in items or robot not in robots:
                print(f"  unknown item or robot (items: `list`; robots: {list(robots)})")
                continue
            bad = src.frozen_mismatch()
            if bad:
                print("  WARNING: joints the pipeline keeps frozen are off their planned value: "
                      + ", ".join(f"{j} is {a} (planned {b})" for j, (a, b) in bad.items()))
            kind, grasp, size, cur = items[name]
            T_obj = object_from_tool(src.lookup(robots[robot]["ee_link"]), grasp)
            roll, pitch, _ = rpy_of(T_obj[:3, :3])
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, T_obj[2, 2]))))
            x, y, z = (float(v) for v in T_obj[:3, 3])
            if args.z == "table":
                if args.table_top is not None:
                    top = args.table_top
                else:
                    top = src.lookup("__support__" if args.fake_ee else support, samples=1)[2, 3]
                z = float(top) + float(size[2]) / 2 + args.clearance
            pose = {"x": x, "y": y, "z": z}
            if args.orientation == "full":
                pose.update({"roll": roll, "pitch": pitch, "yaw": rpy_of(T_obj[:3, :3])[2]})
            else:
                if tilt > args.tilt_warn:
                    print(f"  note: the object would be {tilt:.1f} deg off level (the gripper is "
                          f"not vertical); only its heading is kept")
                yaw = level_yaw(T_obj)
                if abs(yaw) > 1e-3:
                    pose["yaw"] = yaw
            dxy = math.hypot(x - float(cur.get("x", x)), y - float(cur.get("y", y)))
            print(f"  {name} ({kind}) <- {robot}: "
                  + ", ".join(f"{k} {v:+.4f}" for k, v in pose.items())
                  + f"   ({dxy * 1000:.0f} mm from the YAML's)")
            pending[name] = (kind, pose)
        elif cmd[0] == "show":
            for n, (kind, pose) in pending.items():
                print(f"  {n:28s} {kind}: {pose}")
        elif cmd[0] == "write":
            if not pending:
                print("  nothing captured")
                continue
            for n, (kind, pose) in pending.items():
                scene.patch(n, kind, pose)
            bak = scene.write()
            print(f"  wrote {len(pending)} pose(s) to {scene.path} (previous version {bak}); "
                  f"re-run the offline pipeline before executing")
            pending.clear()
        elif cmd[0] in ("quit", "exit", "q"):
            if pending and input(f"  {len(pending)} captured pose(s) not written -- quit anyway? "
                                 f"[y/N] ").strip().lower() != "y":
                continue
            return 0
        else:
            print("  commands: list | teach <item> <robot> | show | write | quit")


if __name__ == "__main__":
    sys.exit(main())
