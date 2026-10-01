#!/usr/bin/env python3
"""Do the VAMP gripper spheres cover the Robotiq 2F-85 finger collision meshes?

    .venv_vamp/bin/python scripts/check_finger_patch.py --urdf /tmp/fabricator4.urdf \
        [--robot robot1] [--configs 6] [--no-patch]

End to end, in the WORLD frame: the gripper robot's VAMP spheres as the engine computes them
(``vamp_collision_engine.make_cell_engines`` for the scene's cell, finger patch included unless
``--no-patch``) against the cell URDF's finger collision meshes (robotiq_description STLs,
forward kinematics with the mimic joints) at the SAME arm configuration -- HOME plus
``--configs`` random ones -- over the finger sweep open -> ``gripper_close`` (every 0.05 rad).
Reports, per finger state, the worst mesh vertex's distance OUTSIDE the union of spheres,
without the margin (negative = inside) and with it. Exit 1 if any vertex is outside the
spheres once the margin is counted. ``--urdf``: the xacro-expanded cell URDF (world frame).
"""
import argparse
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def _T(xyz, r):
    M = np.eye(4)
    M[:3, :3] = _rpy(*r)
    M[:3, 3] = xyz
    return M


def _axis(ax, q, prismatic=False):
    M = np.eye(4)
    ax = np.asarray(ax, float) / np.linalg.norm(ax)
    if prismatic:
        M[:3, 3] = ax * q
        return M
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    M[:3, :3] = np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K
    return M


class Urdf:
    def __init__(self, path):
        self.root = ET.parse(path).getroot()
        self.J = {}
        for j in self.root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            r = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax, m = j.find("axis"), j.find("mimic")
            self.J[j.find("child").get("link")] = dict(
                name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                T=_T(xyz, r), axis=[float(v) for v in ax.get("xyz").split()] if ax is not None else [1, 0, 0],
                mimic=(m.get("joint"), float(m.get("multiplier", 1)), float(m.get("offset", 0)))
                if m is not None else None)

    def fk(self, link, q):
        M = np.eye(4)
        while link in self.J:
            j = self.J[link]
            v = 0.0
            if j["type"] in ("revolute", "continuous", "prismatic"):
                v = (q.get(j["mimic"][0], 0.0) * j["mimic"][1] + j["mimic"][2]) if j["mimic"] \
                    else q.get(j["name"], 0.0)
            M = j["T"] @ _axis(j["axis"], v, j["type"] == "prismatic") @ M
            link = j["parent"]
        return M


def _stl(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return a["v"].reshape(-1, 3).astype(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--task", default=os.path.join(PKG, "config", "tamp_task_fabricator.yaml"))
    ap.add_argument("--robot", default="robot1")
    ap.add_argument("--configs", type=int, default=6)
    ap.add_argument("--mesh-dir", default=None)
    ap.add_argument("--no-patch", action="store_true")
    args = ap.parse_args()
    if args.no_patch:
        os.environ["VAMP_FINGER_PATCH"] = "0"
    import vamp
    from vamp_collision_engine import ObjectGeom, make_cell_engines
    ty = yaml.safe_load(open(args.task))
    objs = {o["id"]: ObjectGeom.from_yaml(o) for o in ty["objects"]}
    eng = make_cell_engines(vamp, ty, objs, list(ty["robots"]))[args.robot]
    cfg = ty["robots"][args.robot]
    close = float(cfg["gripper_close"])
    names = list(cfg["home"].keys())            # rail + 6 arm joints, artifact order
    home = [float(v) for v in cfg["home"].values()]
    mesh_dir = args.mesh_dir
    if mesh_dir is None:
        import subprocess
        pre = subprocess.run(["ros2", "pkg", "prefix", "robotiq_description"], capture_output=True,
                             text=True, check=True).stdout.strip()
        mesh_dir = os.path.join(pre, "share", "robotiq_description", "meshes", "collision", "2f_85")
    U = Urdf(args.urdf)
    links = []
    for l in U.root.findall("link"):
        n = l.get("name")
        c = l.find("collision")
        if n.startswith(f"{args.robot}_robotiq_85") and c is not None:
            links.append((n, _stl(os.path.join(mesh_dir, c.find("geometry").find("mesh").get("filename").split("/")[-1]))))
    rng = np.random.default_rng(0)
    configs = [home] + [[float(rng.uniform(-0.5, 0.5))] + list(rng.uniform(-math.pi, math.pi, 6))
                        for _ in range(args.configs)]
    finger = f"{args.robot}_robotiq_85_left_knuckle_joint"
    margin = eng.sphere_margin
    worst_all = -1e9
    for fq in np.round(np.arange(0.0, close + 1e-9, 0.05), 4):
        worst = -1e9
        for q in configs:
            c, r = eng.robot_spheres(args.robot, q)
            ts = eng.traj_spheres(args.robot, [q], [0], "")
            c, r = ts.centres[0][:-1].astype(float), ts.radii[0][:-1].astype(float) - margin
            jq = dict(zip(names, q))
            jq[finger] = float(fq)
            for n, V in links:
                M = U.fk(n, jq)
                W = V @ M[:3, :3].T + M[:3, 3]
                d = (np.linalg.norm(W[:, None, :] - c[None], axis=2) - r[None]).min(axis=1)
                worst = max(worst, float(d.max()))
        worst_all = max(worst_all, worst)
        print(f"fingers at {fq:.2f} rad: worst vertex {worst * 1000:+7.2f} mm outside the spheres "
              f"without the margin, {(worst - margin) * 1000:+7.2f} mm with it "
              f"({len(configs)} arm configurations)")
    ok = worst_all - margin <= 0
    print(f"{'OK' if ok else 'NOT COVERED'}: worst {worst_all * 1000:+.2f} mm without the "
          f"{margin * 1000:.0f} mm margin (patch {'on' if eng.n_patch else 'off'}, {eng.n_patch} spheres)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
