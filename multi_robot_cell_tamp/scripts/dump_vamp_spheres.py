#!/usr/bin/env python3
"""Pre-compute the VAMP collision spheres for RViz overlay (viz-only, not the seam).

Runs in ``.venv_vamp`` (needs ``import vamp``) and dumps, for every (robot, task)
trajectory in ``tamp_trajectories.json``, the **world-frame** collision spheres at
every sample -- exactly the spheres ``collision_generator_vamp.py`` checks (same cell
placement: y-translation + shoulder_pan mount yaw, structural rail spheres excluded,
2 cm margin). A lightweight rclpy node (``vamp_sphere_overlay``) then replays these as
an RViz ``MarkerArray`` synced to the schedule -- it stays pure ROS Python and never
imports vamp.

This file is a VISUALISATION artifact, not the geometry-free seam: it deliberately
carries geometry and is never fed to ``tamp_scheduler`` (ADR-0002 is about the seam).

    .venv_vamp/bin/python scripts/dump_vamp_spheres.py

Defaults read/write the persistent ``artifacts/`` dir
(``artifacts/tamp_trajectories.json`` -> ``artifacts/vamp/tamp_spheres.npz``).

Output ``.npz`` (load with ``numpy.load(path)``):
  * ``manifest``               : JSON string -> {"delta_t": float, "keys":
                                 ["robot|task", ...], "margin": float,
                                 "n_structural": int, "n_spheres": int}
  * ``<robot>__<task>__centres`` : float32 (K, M, 3) world-frame sphere centres
  * ``<robot>__<task>__radii``   : float32 (K, M)    sphere radii
  (the manifest ``keys`` are "robot|task"; the array names swap '|'->'__'.) M =
  n_spheres + 1 object slot; a detached object has radius -inf -- the node skips any
  non-finite radius (draw nothing) so the object shows only while carried.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Keep BLAS single-threaded (before numpy) — this is FK-only and single-process, but
# stay consistent with collision_generator_vamp.py so no run can oversubscribe the CPU.
for _threadvar in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_threadvar, "1")

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vamp_collision_engine import (  # noqa: E402
    CELL_BASE,
    CELL_MARGIN,
    CELL_MOUNT_YAW,
    CELL_N_STRUCTURAL,
    ObjectGeom,
    VampCollisionEngine,
)


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.dirname(here)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj", default=os.path.join(pkg, "artifacts", "tamp_trajectories.json"))
    p.add_argument("--task", default=os.path.join(pkg, "config", "tamp_task.yaml"))
    p.add_argument("--out", default=os.path.join(pkg, "artifacts", "vamp", "tamp_spheres.npz"))
    p.add_argument("--robot", default="ur10e_rail")
    args = p.parse_args(argv)

    import vamp
    if not hasattr(vamp, args.robot):
        print(f"vamp has no robot '{args.robot}'; available: {vamp.robots}", file=sys.stderr)
        return 1

    with open(args.traj) as f:
        art = json.load(f)
    with open(args.task) as f:
        objects = {o["id"]: ObjectGeom.from_size(o["size"]) for o in yaml.safe_load(f)["objects"]}

    engine = VampCollisionEngine(
        getattr(vamp, args.robot), objects,
        base_transforms=CELL_BASE, mount_yaws=CELL_MOUNT_YAW,
        n_structural=CELL_N_STRUCTURAL, sphere_margin=CELL_MARGIN)

    arrays: dict[str, np.ndarray] = {}
    keys: list[str] = []
    for tr in art["trajectories"]:
        key = f"{tr['robot']}|{tr['task']}"
        name = key.replace("|", "__")            # npz array names can't hold '|'
        ts = engine.traj_spheres(tr["robot"], tr["positions"], tr["object_state"], tr["object"])
        arrays[f"{name}__centres"] = ts.centres.astype(np.float32)
        arrays[f"{name}__radii"] = ts.radii.astype(np.float32)
        keys.append(key)
        print(f"  {key}: {ts.K} samples x {ts.centres.shape[1]} spheres")

    # `sample_counts` is the fingerprint that ties this npz to ONE trajectory artifact.
    # Task ids are not enough on their own: `swap` and `tower` both use t_box_1..4, so a
    # tower npz passes a name check and then draws tower shells over swap motion. Sample
    # counts differ whenever the trajectories do, which is exactly the condition that makes
    # an overlay wrong, so the consumer compares these rather than the names.
    manifest = json.dumps({
        "delta_t": float(art["delta_t"]),
        "keys": keys,
        "sample_counts": {k: int(arrays[k.replace("|", "__") + "__centres"].shape[0])
                          for k in keys},
        "traj_file": os.path.basename(os.path.abspath(args.traj)),
        "task_file": os.path.basename(os.path.abspath(args.task)),
        "margin": CELL_MARGIN,
        "n_structural": CELL_N_STRUCTURAL,
        "n_spheres": engine.n_spheres,
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, manifest=np.array(manifest), **arrays)
    print(f"wrote {len(keys)} trajectories of spheres -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
