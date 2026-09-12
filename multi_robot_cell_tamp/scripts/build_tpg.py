#!/usr/bin/env python3
"""Build the temporal plan graph from a solved schedule -> artifacts/<engine>/tpg.json.

    .venv_vamp/bin/python scripts/build_tpg.py --task config/tamp_task_tower.yaml

Reads the trajectory artifact (geometry) and the solver's schedule (assignment + start
slots), recomputes ``mu`` for the four SCHEDULED trajectory pairs, and emits a
geometry-free graph of cross-robot precedences.

Recomputing ``mu`` rather than persisting it is the cheap option: the SIMD kernel returns
a full matrix in ~0.2 s per pair, so the whole stage costs about a second, and the seam
stays a small file of indices. See :mod:`tpg` for why the graph exists and why it has one
edge per node.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Tuple

for _threadvar in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_threadvar, "1")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vamp_collision_engine import (  # noqa: E402
    CELL_BASE, CELL_MARGIN, CELL_MOUNT_YAW, CELL_N_STRUCTURAL,
    ObjectGeom, VampCollisionEngine,
)
from vamp_link_groups import DEFAULT_SPHERIZED_URDF, link_groups  # noqa: E402
from mu_kernel import MuKernel  # noqa: E402
from vamp_reference import collision_matrix  # noqa: E402
import tpg as tpg_mod  # noqa: E402


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.dirname(here)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj", default=os.path.join(pkg, "artifacts", "tamp_trajectories.json"))
    p.add_argument("--solution", default=os.path.join(pkg, "artifacts", "vamp", "tamp_solution.json"))
    p.add_argument("--problem", default=os.path.join(pkg, "artifacts", "vamp", "tamp_problem.json"),
                   help="the seam -- supplies the precedences and pick/place milestones "
                        "that become the graph's non-geometric edges")
    p.add_argument("--task", default=os.path.join(pkg, "config", "tamp_task_tower.yaml"))
    p.add_argument("--out", default=os.path.join(pkg, "artifacts", "vamp", "tpg.json"))
    p.add_argument("--robot", default="ur10e_rail")
    p.add_argument("--no-kernel", action="store_true",
                   help="use the numpy reference for mu (much slower; for cross-checking)")
    args = p.parse_args(argv)

    import vamp
    art = json.load(open(args.traj))
    sol = json.load(open(args.solution))
    prob = json.load(open(args.problem))
    robots = list(art["robots"])
    objects = {o["id"]: ObjectGeom.from_size(o["size"])
               for o in yaml.safe_load(open(args.task))["objects"]}

    engine = VampCollisionEngine(
        getattr(vamp, args.robot), objects,
        base_transforms=CELL_BASE, mount_yaws=CELL_MOUNT_YAW,
        n_structural=CELL_N_STRUCTURAL, sphere_margin=CELL_MARGIN,
        groups=link_groups(DEFAULT_SPHERIZED_URDF, CELL_N_STRUCTURAL))

    kernel = None if args.no_kernel else MuKernel()
    if kernel is not None and not kernel.available:
        kernel = None
    print(f"mu backend: {'SIMD kernel' if kernel else 'numpy reference'}")

    trs = {(t["robot"], t["task"]): t for t in art["trajectories"]}
    spheres: Dict[Tuple[str, str], object] = {}
    packed: Dict[Tuple[str, str], object] = {}

    def geom(robot: str, task: str):
        key = (robot, task)
        if key not in spheres:
            t = trs[key]
            spheres[key] = engine.traj_spheres(robot, t["positions"], t["object_state"], t["object"])
            if kernel is not None:
                packed[key] = kernel.pack(spheres[key], "A" if robot == robots[0] else "B")
        return key

    def mu_of(r: str, ti: str, s: str, tj: str) -> np.ndarray:
        a, b = geom(r, ti), geom(s, tj)
        if kernel is not None:
            return kernel.matrix(packed[a], packed[b])
        return collision_matrix(spheres[a], spheres[b])

    t0 = time.time()
    graph = tpg_mod.build(sol, robots, mu_of, delta_t=float(art["delta_t"]), problem=prob)
    dt = time.time() - t0

    nodes = {r: graph.n_nodes(r) for r in robots}
    print(f"TPG: {sum(nodes.values())} nodes ({', '.join(f'{r}={n}' for r, n in nodes.items())}), "
          f"{graph.n_edges} cross-robot edges, built in {dt:.1f}s")
    print(f"     {graph.n_precedence_edges} of them come from task precedences "
          f"(de-stack gate + re-stack order), the rest from geometry")
    print(f"     nominal makespan {graph.nominal_makespan} slots "
          f"= {graph.nominal_makespan * graph.delta_t:.2f} s")
    if graph.delay_margin >= 0:
        print(f"     rigid delay margin {graph.delay_margin} slots = "
              f"{graph.delay_margin * graph.delta_t:.2f} s  "
              f"(what the OLD executor could absorb, by luck -- the TPG removes the limit)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    graph.to_json(args.out)
    print(f"wrote TPG -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
