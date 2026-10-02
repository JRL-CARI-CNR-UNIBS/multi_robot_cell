#!/usr/bin/env python3
"""The carried object's pose: VAMP engine == FCL engine == trajectory generator.

Run with the ROS Python (it needs moveit_py), after a build, from anywhere:

    python3 scripts/test_object_pose.py [--traj T --task Y]

WHAT IT CHECKS (ADR-0005 addendum, 2026-09-21)

The generator attaches a carried object while the EE sits at ``object (x) grasp``, so the
object rides the EE at ``grasp^-1``. Both mu engines must reproduce that pose:

1. Synthetic, exact: for each robot and two grasps -- ``{z: 0.16, roll: pi}`` and the same
   with ``yaw: pi/2`` -- a random grasp configuration q1 defines the spawn an exact IK would
   have solved (``spawn = FK_ee(q1) (x) grasp^-1``). MoveIt attaches a box there exactly as
   ``setObjectState(Attached)`` does, then the arm moves to random q2. The box centre MoveIt
   reports is the REFERENCE; the FCL engine (``collision_generator`` with
   ``dump_object_centres``, its own ``setAttached``) and the VAMP engine
   (``VampCollisionEngine.object_sphere``) must match it to 1e-6 m, and FCL's orientation
   must match too (up to the box's own symmetry is NOT allowed here: exact rotation).
2. Real data (optional ``--traj/--task``): at every pick (first GripClose sample) and place
   (first GripOpen sample) of a trajectory artifact, ``FK_ee(q) (x) grasp^-1`` against the
   YAML spawn / place centre -- the IK + Cartesian-descent residual the model inherits --
   and FCL vs VAMP on every 10th carried sample.

The three interpreters stay separate processes: this ROS Python drives MoveIt and the C++
node; the VAMP engine runs under ``.venv_vamp`` via a subprocess.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell_registry import scene_path  # noqa: E402
import subprocess
import sys
import tempfile

import numpy as np
import yaml

PKG = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
VENV = os.path.join(PKG, ".venv_vamp", "bin", "python")
GRASPS = {
    "obj_roll": {"x": 0.0, "y": 0.0, "z": 0.16, "roll": math.pi},
    "obj_yaw": {"x": 0.0, "y": 0.0, "z": 0.16, "roll": math.pi, "yaw": 1.5708},
}
SIZE = [0.05, 0.03, 0.07]      # deliberately not a cube, so a wrong rotation shows
TOL = 1e-6


def pose_from_yaml(p):
    """Same semantics as the C++ ``poseFromYaml`` (tf2 setRPY: Rz Ry Rx)."""
    r, pt, y = (float(p.get(k, 0.0)) for k in ("roll", "pitch", "yaw"))
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(pt), 0, math.sin(pt)], [0, 1, 0], [-math.sin(pt), 0, math.cos(pt)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = [p["x"], p["y"], p["z"]]
    return T


def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def model_files(tmp):
    import xacro
    from ament_index_python.packages import get_package_share_directory
    share = os.path.join(get_package_share_directory("multi_robot_moveit_config"), "config")
    urdf = xacro.process_file(os.path.join(share, "multi_robot_cell.urdf.xacro")).toxml()
    with open(os.path.join(share, "multi_robot_cell.srdf")) as f:
        srdf = f.read()
    up, sp = os.path.join(tmp, "cell.urdf"), os.path.join(tmp, "cell.srdf")
    open(up, "w").write(urdf)
    open(sp, "w").write(srdf)
    return up, sp, urdf, srdf


def run_fcl_dump(tmp, urdf, srdf, traj_file, task_file):
    params = os.path.join(tmp, "params.yaml")
    out = os.path.join(tmp, "fcl_centres.json")
    with open(params, "w") as f:
        yaml.safe_dump({"collision_generator": {"ros__parameters": {
            "robot_description": urdf, "robot_description_semantic": srdf,
            "traj_file": traj_file, "task_file": task_file, "out_file": os.path.join(tmp, "unused.json"),
            "dump_object_centres": out}}}, f)
    subprocess.run(["ros2", "run", "multi_robot_cell_tamp", "collision_generator",
                    "--ros-args", "--params-file", params],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.load(open(out))


VAMP_SNIPPET = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import vamp, yaml
from vamp_collision_engine import (CELL_BASE, CELL_MOUNT_YAW, CELL_N_STRUCTURAL, CELL_MARGIN,
                                   ObjectGeom, VampCollisionEngine)
art = json.load(open(sys.argv[2]))
objects = {o["id"]: ObjectGeom.from_yaml(o) for o in yaml.safe_load(open(sys.argv[3]))["objects"]}
eng = VampCollisionEngine(vamp.ur10e_rail, objects, base_transforms=CELL_BASE,
                          mount_yaws=CELL_MOUNT_YAW, n_structural=CELL_N_STRUCTURAL,
                          sphere_margin=CELL_MARGIN)
out = []
for t in art["trajectories"]:
    if not t["object"]:
        continue
    for k, (q, s) in enumerate(zip(t["positions"], t["object_state"])):
        if s == 1 and k % 10 == 0:
            c, r = eng.object_sphere(t["robot"], q, objects[t["object"]])
            out.append({"robot": t["robot"], "task": t["task"], "k": k,
                        "centre": [float(v) for v in c], "radius": float(r)})
json.dump(out, open(sys.argv[4], "w"))
"""


def run_vamp(tmp, traj_file, task_file):
    snip, out = os.path.join(tmp, "vamp_snip.py"), os.path.join(tmp, "vamp_centres.json")
    open(snip, "w").write(VAMP_SNIPPET)
    subprocess.run([VENV, snip, os.path.join(PKG, "scripts"), traj_file, task_file, out], check=True)
    return json.load(open(out))


def synthetic(tmp, up, sp, urdf, srdf, rng):
    from moveit.core.robot_model import RobotModel
    from moveit.core.robot_state import RobotState
    from moveit.core.planning_scene import PlanningScene
    from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
    from shape_msgs.msg import SolidPrimitive
    from geometry_msgs.msg import Pose

    model = RobotModel(urdf_xml_path=up, srdf_xml_path=sp)
    base = yaml.safe_load(open(scene_path("tower")))
    task = {k: base[k] for k in ("robots",)}
    task["objects"] = [{"id": oid, "size": SIZE, "grasp": g,
                        "spawn": {"x": 0.0, "y": 0.0, "z": 0.8}} for oid, g in GRASPS.items()]
    task_file = os.path.join(tmp, "task.yaml")
    yaml.safe_dump(task, open(task_file, "w"))

    refs, trajs = {}, []
    for robot, cfg in task["robots"].items():
        group, ee, attach = cfg["planning_group"], cfg["ee_link"], cfg["attach_link"]
        joints = list(cfg["home"].keys())
        for oid, g in GRASPS.items():
            q1 = np.concatenate([[rng.uniform(-1, 1)], rng.uniform(-3, 3, 6)])
            scene = PlanningScene(model)
            st = RobotState(model)
            st.set_to_default_values()
            st.set_joint_group_positions(group, q1)
            st.update()
            scene.current_state = st
            spawn = np.asarray(st.get_global_link_transform(ee)) @ np.linalg.inv(pose_from_yaml(g))
            co = CollisionObject()
            co.header.frame_id = scene.planning_frame
            co.id = oid
            prim = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(SIZE))
            co.primitives = [prim]
            p = Pose()
            p.position.x, p.position.y, p.position.z = (float(v) for v in spawn[:3, 3])
            from scipy.spatial.transform import Rotation  # noqa: PLC0415
            qx, qy, qz, qw = Rotation.from_matrix(spawn[:3, :3]).as_quat()
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = qx, qy, qz, qw
            co.pose = p
            co.primitive_poses = [Pose()]
            co.operation = CollisionObject.ADD
            aco = AttachedCollisionObject(link_name=attach, object=co)
            scene.process_attached_collision_object(aco)
            q2s = [np.concatenate([[rng.uniform(-1, 1)], rng.uniform(-3, 3, 6)]) for _ in range(4)]
            positions, states = [], []
            for i, q2 in enumerate(q2s):
                s2 = scene.current_state
                s2.set_joint_group_positions(group, q2)
                s2.update()
                refs[(robot, oid, 10 * i)] = np.asarray(s2.get_frame_transform(oid))
                positions += [q2.tolist()] * 10
                states += [1] * 10
            trajs.append({"robot": robot, "task": oid, "slot": oid, "object": oid,
                          "joint_names": joints, "positions": positions,
                          "phase": [0] * len(states), "object_state": states})
    art = {"delta_t": 0.05, "robots": list(task["robots"]), "tasks": list(GRASPS),
           "precedences": [], "homes": {r: list(c["home"].values()) for r, c in task["robots"].items()},
           "trajectories": trajs}
    traj_file = os.path.join(tmp, "traj.json")
    json.dump(art, open(traj_file, "w"))

    fcl = {(d["robot"], d["task"], d["k"]): d for d in run_fcl_dump(tmp, urdf, srdf, traj_file, task_file)}
    vmp = {(d["robot"], d["task"], d["k"]): d for d in run_vamp(tmp, traj_file, task_file)}
    worst = {"fcl_centre": 0.0, "fcl_rot": 0.0, "vamp_centre": 0.0}
    for key, T in refs.items():
        f, v = fcl[key], vmp[key]
        worst["fcl_centre"] = max(worst["fcl_centre"], np.abs(np.array(f["centre"]) - T[:3, 3]).max())
        worst["fcl_rot"] = max(worst["fcl_rot"], np.abs(quat_to_R(*f["quat_xyzw"]) - T[:3, :3]).max())
        worst["vamp_centre"] = max(worst["vamp_centre"], np.abs(np.array(v["centre"]) - T[:3, 3]).max())
        print(f"  {key[0]} {key[1]:9s} q2#{key[2] // 10}: ref centre {np.round(T[:3, 3], 6).tolist()}"
              f"  |FCL-ref| {np.abs(np.array(f['centre']) - T[:3, 3]).max():.1e}"
              f"  |VAMP-ref| {np.abs(np.array(v['centre']) - T[:3, 3]).max():.1e}")
    print(f"synthetic ({len(refs)} poses): max |FCL-ref| centre {worst['fcl_centre']:.2e} m, "
          f"rotation {worst['fcl_rot']:.2e}; max |VAMP-ref| centre {worst['vamp_centre']:.2e} m")
    return all(v <= TOL for v in worst.values())


def real(tmp, up, sp, urdf, srdf, traj_file, task_file):
    from moveit.core.robot_model import RobotModel
    from moveit.core.robot_state import RobotState
    model = RobotModel(urdf_xml_path=up, srdf_xml_path=sp)
    task = yaml.safe_load(open(task_file))
    art = json.load(open(traj_file))
    objs = {o["id"]: o for o in task["objects"]}
    place = {t["id"]: t["place"] for t in task.get("tasks", [])}
    for s in task.get("slots", []) or []:
        for oid in s.get("candidates", []):
            place[f"{s['id']}__{oid}"] = s["place"]
    worst_pick = worst_place = 0.0
    n = 0
    for t in art["trajectories"]:
        if not t["object"]:
            continue
        cfg = task["robots"][t["robot"]]
        ginv = np.linalg.inv(pose_from_yaml(objs[t["object"]]["grasp"]))
        st = RobotState(model)
        st.set_to_default_values()

        def centre(k):
            st.set_joint_group_positions(cfg["planning_group"], np.array(t["positions"][k]))
            st.update()
            return (np.asarray(st.get_global_link_transform(cfg["ee_link"])) @ ginv)[:3, 3]
        k0 = t["phase"].index(1)
        k1 = next(k for k in range(k0, len(t["phase"])) if t["phase"][k] == 3)
        sp_ = objs[t["object"]]["spawn"]
        worst_pick = max(worst_pick, np.linalg.norm(centre(k0) - [sp_["x"], sp_["y"], sp_["z"]]))
        if t["task"] in place:
            pl = place[t["task"]]
            worst_place = max(worst_place, np.linalg.norm(centre(k1) - [pl["x"], pl["y"], pl["z"]]))
        n += 1
    print(f"real ({n} carried trajectories): max |FK_ee (x) grasp^-1 - spawn| at pick "
          f"{worst_pick:.2e} m, at place {worst_place:.2e} m")
    fcl = {(d["robot"], d["task"], d["k"]): d for d in run_fcl_dump(tmp, urdf, srdf, traj_file, task_file)}
    vmp = {(d["robot"], d["task"], d["k"]): d for d in run_vamp(tmp, traj_file, task_file)}
    assert fcl.keys() == vmp.keys(), "FCL and VAMP dumped different samples"
    d = max(np.abs(np.array(fcl[k]["centre"]) - vmp[k]["centre"]).max() for k in fcl)
    print(f"real: FCL vs VAMP centre on {len(fcl)} carried samples: max diff {d:.2e} m")
    return d <= TOL and worst_pick < 5e-3 and worst_place < 5e-3


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--traj")
    ap.add_argument("--task")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        up, sp, urdf, srdf = model_files(tmp)
        ok = synthetic(tmp, up, sp, urdf, srdf, np.random.default_rng(a.seed))
        if a.traj:
            ok = real(tmp, up, sp, urdf, srdf, a.traj, a.task) and ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
