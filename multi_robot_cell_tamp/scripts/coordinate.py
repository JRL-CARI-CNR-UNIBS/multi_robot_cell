#!/usr/bin/env python3
"""The coordination diagram: what the best possible execution of a fixed plan costs.

    .venv_vamp/bin/python scripts/coordinate.py --traj artifacts/tamp_trajectories.json

Both other views of the plan answer a narrower question. The scheduler asks "what is the
best *rigid* schedule", i.e. the best single relative start offset, which shifts each whole
trajectory as one block. The TPG asks "is this execution safe under delay". Neither answers
"could any execution of these paths have been faster", and that is what this does.

Lay the two robots' node sequences on the axes of a grid and block every cell whose pair of
configurations collides, plus every cell that would violate a task precedence. Executing the
pair is then exactly a **monotone walk** from one corner to the other: each robot only ever
advances, either may pause, and the walk must avoid the blocked cells. The number of steps
in the shortest such walk is the optimal makespan for those paths -- over every possible
coordination, not just over start offsets. It is computed exactly, by a dynamic program
along anti-diagonals (each depends only on the previous two), in about a second.

TWO THINGS IT IS USED FOR
-------------------------
**Cross-checking the optimum.** On every scene it has been run on, the walk costs exactly the
CP-SAT optimum, to the slot -- two independent formulations, one a MILP over start offsets
and one a shortest path over a grid, landing on the same number. It also says the rigid
schedule leaves nothing on the table there: no cleverer coordination of those paths exists.

**Detecting plans that cannot be executed at all.** A plan can be collision-free
sample-by-sample and still admit no monotone walk, because escaping would need a robot to
*reverse*. No amount of waiting fixes that, so a TPG built on such a plan would deadlock.
That is exactly how the first attempt at commute-shortening was caught (ADR-0008): the
spliced chains had no walk, and the diagram said so in a second rather than the executor
discovering it on the cell.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

for _threadvar in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
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
from tpg import timelines  # noqa: E402

UNREACHABLE = np.int32(np.iinfo(np.int32).max // 4)


def diagram(art: dict, prob: dict, order: Dict[str, List[str]], task_file: str,
            robot: str = "ur10e_rail") -> Tuple[np.ndarray, np.ndarray, Dict[str, List[int]]]:
    """Blocked cells of the coordination diagram: collisions, and precedence violations.

    Returned separately so a blockage can be attributed. They interact -- on the refined
    tower chains each alone admitted a walk and only their union did not.
    """
    import vamp
    objects = {o["id"]: ObjectGeom.from_size(o["size"])
               for o in yaml.safe_load(open(task_file))["objects"]}
    engine = VampCollisionEngine(
        getattr(vamp, robot), objects, base_transforms=CELL_BASE, mount_yaws=CELL_MOUNT_YAW,
        n_structural=CELL_N_STRUCTURAL, sphere_margin=CELL_MARGIN,
        groups=link_groups(DEFAULT_SPHERIZED_URDF, CELL_N_STRUCTURAL))
    kern = MuKernel()
    if not kern.available:
        raise RuntimeError("needs the SIMD kernel; build it with scripts/build_mu_kernel.sh")

    trs = {(t["robot"], t["task"]): t for t in art["trajectories"]}
    r, s = art["robots"]
    off, n_nodes = {}, {}
    packed = {}
    for q in (r, s):
        cursor = 0
        for t in order[q]:
            tr = trs[(q, t)]
            off[(q, t)] = cursor
            cursor += len(tr["positions"])
            packed[(q, t)] = kern.pack(
                engine.traj_spheres(q, tr["positions"], tr["object_state"], tr["object"]),
                "A" if q == r else "B")
        n_nodes[q] = cursor

    coll = np.zeros((n_nodes[r], n_nodes[s]), dtype=bool)
    for i in order[r]:
        for j in order[s]:
            m = kern.matrix(packed[(r, i)], packed[(s, j)])
            a, b = off[(r, i)], off[(s, j)]
            coll[a:a + m.shape[0], b:b + m.shape[1]] = m

    def milestone(task: str, which: str) -> Tuple[str, int]:
        for q in (r, s):
            if task in order[q]:
                # "start" gates DEPARTURE. A robot already occupies its first node at tick
                # zero, so the condition is that it has LEFT it -- hence the +1, without
                # which the very first cell is spuriously blocked.
                extra = 1 if which == "start" else int(prob[f"{which}_offsets"][f"{q}|{task}"])
                return q, off[(q, task)] + extra
        raise KeyError(task)

    n1 = np.arange(n_nodes[r])[:, None]
    n2 = np.arange(n_nodes[s])[None, :]
    prec = np.zeros_like(coll)
    for i, j in prob.get("precedences", []):
        for wi, wj in (("pick", "start"), ("place", "place")):
            (ri, mi), (rj, mj) = milestone(i, wi), milestone(j, wj)
            if ri == rj:
                continue      # one robot's own node order already enforces it
            reached_j = (n2 >= mj) if rj == s else (n1 >= mj)
            behind_i = (n1 < mi) if ri == r else (n2 < mi)
            prec |= reached_j & behind_i
    return coll, prec, {r: [off[(r, t)] for t in order[r]], s: [off[(s, t)] for t in order[s]]}


def walk(blocked: np.ndarray) -> np.ndarray:
    """Ticks to reach each cell by a monotone walk from (0, 0). ``UNREACHABLE`` if none.

    Every move -- advance either robot, or both -- costs one tick, so this is a shortest
    path. Cell (i, j) depends only on (i-1, j), (i, j-1) and (i-1, j-1), all of which lie on
    the two preceding anti-diagonals; sweeping by anti-diagonal therefore vectorises the
    whole thing and keeps a million-cell diagram to about a second.
    """
    n1, n2 = blocked.shape
    if blocked[0, 0]:
        raise AssertionError(
            "the starting cell is blocked: both robots at their first configuration "
            "already collide, so HOME is not the non-blocking pose the model assumes")
    T = np.full((n1, n2), UNREACHABLE, dtype=np.int32)
    T[0, 0] = 0
    for d in range(1, n1 + n2 - 1):
        i = np.arange(max(0, d - n2 + 1), min(d, n1 - 1) + 1)
        j = d - i
        best = np.full(i.size, UNREACHABLE, dtype=np.int32)
        for di, dj in ((1, 0), (0, 1), (1, 1)):
            pi, pj = i - di, j - dj
            ok = (pi >= 0) & (pj >= 0) & (pj < n2)
            if ok.any():
                prev = np.full(i.size, UNREACHABLE, dtype=np.int32)
                prev[ok] = T[pi[ok], pj[ok]]
                best = np.minimum(best, prev)
        T[i, j] = np.where(blocked[i, j] | (best >= UNREACHABLE), UNREACHABLE, best + 1)
    return T


def stall_report(T: np.ndarray, blocked: np.ndarray) -> str:
    """Where a plan with no monotone walk gets stuck, in node indices."""
    reach = T < UNREACHABLE
    i = int(np.max(np.flatnonzero(reach.any(axis=1))))
    j = int(np.max(np.flatnonzero(reach.any(axis=0))))
    return (f"furthest either robot can get: {i} of {blocked.shape[0] - 1} and "
            f"{j} of {blocked.shape[1] - 1}; beyond that each blocks the other and "
            f"neither can reverse")


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.dirname(here)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj", default=os.path.join(pkg, "artifacts", "tamp_trajectories.json"))
    p.add_argument("--problem", default=os.path.join(pkg, "artifacts", "vamp", "tamp_problem.json"))
    p.add_argument("--solution",
                   default=os.path.join(pkg, "artifacts", "vamp", "tamp_solution.json"))
    p.add_argument("--task", default=os.path.join(pkg, "config", "tamp_task_tower.yaml"))
    p.add_argument("--robot", default="ur10e_rail")
    args = p.parse_args(argv)

    art = json.load(open(args.traj))
    prob = json.load(open(args.problem))
    sol = json.load(open(args.solution))
    robots = list(art["robots"])
    order = {q: [g.task for g in segs] for q, segs in timelines(sol, robots).items()}

    coll, prec, _ = diagram(art, prob, order, args.task, args.robot)
    blocked = coll | prec
    dt = float(art["delta_t"])
    print(f"diagram {blocked.shape[0]} x {blocked.shape[1]} cells: "
          f"{100 * coll.mean():.1f}% blocked by collision, "
          f"{100 * (prec & ~coll).mean():.1f}% by precedence alone "
          f"({100 * blocked.mean():.1f}% total)")

    T = walk(blocked)
    if T[-1, -1] >= UNREACHABLE:
        print("\nNO monotone walk exists -- these paths CANNOT be executed together, at any "
              "timing.\n" + stall_report(T, blocked))
        # Attribute the blockage: it is routinely the interaction, not either alone.
        for name, mask in (("collisions", coll), ("precedences", prec)):
            t = walk(mask)
            ok = t[-1, -1] < UNREACHABLE
            print(f"  with only {name}: "
                  + (f"a walk exists ({int(t[-1, -1]) + 1} slots)" if ok else "still none"))
        return 1

    best = int(T[-1, -1]) + 1
    bound = max(blocked.shape)
    print(f"\noptimal coordinated makespan {best} slots = {best * dt:.2f} s")
    print(f"  lower bound (longest path alone, no waiting) {bound} = {bound * dt:.2f} s")
    print(f"  waiting accounts for {best - bound} slots = {(best - bound) * dt:.2f} s")
    if "makespan_slots" in sol:
        rigid = int(sol["makespan_slots"])
        gap = rigid - best
        print(f"  the rigid schedule achieves {rigid} = {rigid * dt:.2f} s, "
              + ("which is optimal for these paths -- no coordination can beat it"
                 if gap == 0 else f"i.e. {gap} slots ({gap * dt:.2f} s) above this bound"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
