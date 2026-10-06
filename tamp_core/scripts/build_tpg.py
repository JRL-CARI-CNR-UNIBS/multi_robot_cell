#!/usr/bin/env python3
"""Build the temporal plan graph from a solved schedule -> artifacts/<engine>/tpg.json.

    .venv_vamp/bin/python scripts/build_tpg.py --task config/tamp_task_tower.yaml

Reads the trajectory artifact (geometry) and the solver's schedule (assignment + start
slots), recomputes ``mu`` for every SCHEDULED trajectory pair of every robot pair, and
emits a geometry-free graph of cross-robot precedences. Any number of robots: the robots
are the trajectory artifact's ``robots`` list, every pair of them gets its edges.

Recomputing ``mu`` rather than persisting it keeps the seam a small file of indices, but it
is the cost of this stage: a full matrix has no early exit, 0.2-0.8 s per pair (two-robot
cell: a few seconds in all; fabricator, 252 scheduled pairs: ~3 min single-threaded). So the
matrices are computed by ``--jobs`` workers, streamed to :func:`tpg.build` in the order it
asks for them, and a pair the seam already proves empty -- no forbidden offset at all, from
the SAME engine (``--seam-engine vamp``) -- is an all-false matrix that is not recomputed.
See :mod:`tpg` for why the graph exists and why it has one edge per node.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell_registry import scene_path  # noqa: E402
import sys
import time
from typing import Dict, Tuple

# Filled in the parent before the pool forks; workers inherit it copy-on-write.
_STATE: Dict[str, object] = {}


def _matrix_job(key):
    """One scheduled pair -> its mu matrix, bit-packed for the trip back to the parent."""
    m = np.asarray(_STATE["mu"](*key), dtype=bool)
    return key, np.packbits(m, axis=None), m.shape

for _threadvar in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_threadvar, "1")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vamp_collision_engine import HoldExemption, ObjectGeom, make_cell_engines, objects_of_scene  # noqa: E402
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
    p.add_argument("--task", default=scene_path("tower"))
    p.add_argument("--out", default=os.path.join(pkg, "artifacts", "vamp", "tpg.json"))
    p.add_argument("--robot", default="ur10e_rail")
    p.add_argument("--no-kernel", action="store_true",
                   help="use the numpy reference for mu (much slower; for cross-checking)")
    p.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                   help="worker processes computing the mu matrices (1 = in-process)")
    p.add_argument("--seam-engine", choices=["", "vamp", "fcl"], default="",
                   help="the engine that wrote --problem. Only with 'vamp' (this stage's own "
                        "engine) does a pair with no forbidden offset mean an all-false mu, "
                        "which is then not recomputed; FCL forbids a subset of VAMP's cells")
    args = p.parse_args(argv)

    import vamp
    art = json.load(open(args.traj))
    sol = json.load(open(args.solution))
    prob = json.load(open(args.problem))
    robots = list(art["robots"])
    task_yaml = yaml.safe_load(open(args.task))
    objects = objects_of_scene(task_yaml)

    # One engine per robot: the scene's `cell:` picks the layout (absent = the dual cell,
    # exactly the one engine `--robot` with CELL_BASE / CELL_MOUNT_YAW built before).
    engines = make_cell_engines(vamp, task_yaml, objects, robots, dual_module=args.robot)

    kernel = None if args.no_kernel else MuKernel()
    if kernel is not None and not kernel.available:
        kernel = None
    print(f"mu backend: {'SIMD kernel' if kernel else 'numpy reference'}")

    trs = {(t["robot"], t["task"]): t for t in art["trajectories"]}
    spheres: Dict[Tuple[str, str], object] = {}
    # The synchronous-hold exemption, the SAME function as the seam's (HoldExemption): the
    # graph orders exactly the cells the solver was told collide. No-op without a hold.
    hx = HoldExemption(art)
    packed: Dict[tuple, object] = {}

    def geom(robot: str, task: str):
        key = (robot, task)
        if key not in spheres:
            t = trs[key]
            spheres[key] = engines[robot].traj_spheres(robot, t["positions"], t["object_state"],
                                                       t["object"])
        return key

    def pack(key, side: str, n_sph: int, n_grp: int, mode: str = ""):
        # The kernel's two sides park dead lanes at opposite sentinels, so the sides of ONE
        # matrix must differ. tpg.build calls mu_of(r, ., s, .) with r before s in robot
        # order: r is side A of that pair, s side B. With two robots that is robots[0] A
        # and robots[1] B, as always; with more a robot can be A of one pair and B of
        # another, so each side is packed (once) when first needed. Both sides of a pair
        # are packed to the pair's larger sphere / group count (two robot modules -- a
        # gripper and a torch -- differ); for one module that is each side's own count.
        k = key + (side, n_sph, n_grp, mode)
        if k not in packed:
            packed[k] = kernel.pack(hx.spheres(spheres[key], key[0], key[1], mode), side,
                                    n_sph=n_sph, n_grp=n_grp)
        return packed[k]

    def mu_of(r: str, ti: str, s: str, tj: str) -> np.ndarray:
        a, b = geom(r, ti), geom(s, tj)
        xa, xb = hx.mode(r, ti, s, tj), hx.mode(s, tj, r, ti)
        if kernel is not None:
            ns = max(spheres[a].centres.shape[1], spheres[b].centres.shape[1])
            ng = max(spheres[a].gcen.shape[1], spheres[b].gcen.shape[1])
            return kernel.matrix(pack(a, "A", ns, ng, xa), pack(b, "B", ns, ng, xb))
        return collision_matrix(hx.spheres(spheres[a], r, ti, xa), hx.spheres(spheres[b], s, tj, xb))

    # Which scheduled pairs the seam proves empty. The seam lists every computed pair with at
    # least one forbidden offset (it skips only mutually exclusive candidates, which are never
    # both scheduled), so a scheduled pair missing from it collides nowhere -- for
    # the engine that wrote it, and only if it was written from THESE trajectories (checked
    # on every duration; a mismatch turns the shortcut off, never the graph).
    forbidden = prob.get("forbidden_offsets", {})
    durations = prob.get("durations", {})
    fresh = all(durations.get(f"{t['robot']}|{t['task']}") == t["num_samples"]
                for t in art["trajectories"])
    skip_ok = args.seam_engine == "vamp" and fresh
    if args.seam_engine == "vamp" and not fresh:
        print("seam durations differ from the trajectories: every mu is recomputed")

    def proven_empty(r: str, ti: str, s: str, tj: str) -> bool:
        return (skip_ok and f"{r}|{ti}|{s}|{tj}" not in forbidden
                and f"{s}|{tj}|{r}|{ti}" not in forbidden)

    # tpg.build asks for the pairs robot pair by robot pair (r before s in `robots`), and
    # within one in the two timelines' order: compute them in that order, in parallel, and
    # hand each over as it arrives -- never all ~3 MB matrices in memory at once.
    segs = tpg_mod.timelines(sol, robots)
    order = [(r, gi.task, s, gj.task) for a, r in enumerate(robots) for s in robots[a + 1:]
             for gi in segs[r] for gj in segs[s]]
    todo = [k for k in order if not proven_empty(*k)]
    for r, ti, s, tj in todo:              # FK in the parent, shared by every worker
        geom(r, ti), geom(s, tj)
    _STATE["mu"] = mu_of
    jobs = max(1, min(args.jobs, len(todo)))
    pool = mp.get_context("fork").Pool(jobs) if jobs > 1 else None
    stream = pool.imap(_matrix_job, todo) if pool else map(_matrix_job, todo)

    def mu_streamed(r: str, ti: str, s: str, tj: str) -> np.ndarray:
        if proven_empty(r, ti, s, tj):
            return np.zeros((trs[(r, ti)]["num_samples"], trs[(s, tj)]["num_samples"]), dtype=bool)
        key, bits, shape = next(stream)
        if key != (r, ti, s, tj):
            raise RuntimeError(f"mu stream out of order: got {key}, tpg.build asked for "
                               f"{(r, ti, s, tj)}")
        return np.unpackbits(bits, count=shape[0] * shape[1]).reshape(shape).astype(bool)

    t0 = time.time()
    try:
        graph = tpg_mod.build(sol, robots, mu_streamed, delta_t=float(art["delta_t"]),
                              problem=prob, rest_home=tpg_mod.home_rest_ends(art))
    finally:
        if pool:
            pool.close()
            pool.join()
    dt = time.time() - t0
    print(f"mu: {len(todo)} of {len(order)} scheduled pairs computed "
          f"({len(order) - len(todo)} proven empty by the seam), {jobs} worker(s)")

    zero_delay = tpg_mod.zero_delay_ticks(graph)
    nodes = {r: graph.n_nodes(r) for r in robots}
    print(f"TPG: {sum(nodes.values())} nodes ({', '.join(f'{r}={n}' for r, n in nodes.items())}), "
          f"{graph.n_edges} cross-robot edges, built in {dt:.1f}s")
    print(f"     {graph.n_precedence_edges} of them come from task precedences "
          f"(de-stack gate + re-stack order; slot precedences resolved to the scheduled "
          f"winners), the rest from geometry")
    print(f"     schedule makespan {graph.nominal_makespan} slots "
          f"= {graph.nominal_makespan * graph.delta_t:.2f} s  "
          f"(the SOLVER's, idle gaps included -- not what the graph executes)")
    print(f"     graph makespan at zero delay {zero_delay} slots = "
          f"{zero_delay * graph.delta_t:.2f} s  "
          f"(every robot advances as soon as its edges allow; `simulate_tpg.py` `tpg` row "
          f"at stall p = 0 reproduces it)")
    if graph.home_rest_masked:
        print(f"     {graph.home_rest_masked} mu cells against a robot resting at HOME dropped "
              f"(sphere conservatism: HOME clearance is certified by the trajectory stage on the "
              f"exact geometry, ADR-0003; replay the plan with plan_oracle.py to confirm)")
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
