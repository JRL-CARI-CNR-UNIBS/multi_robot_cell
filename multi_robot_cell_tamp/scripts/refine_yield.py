#!/usr/bin/env python3
"""Shorten the commute between two consecutive tasks, without coupling the robots.

    .venv_vamp/bin/python scripts/refine_yield.py --plan   ...   # -> transit spec
    <ROS: trajectory_generator with transit_file=...>            # -> transits.json
    .venv_vamp/bin/python scripts/refine_yield.py --splice ...   # -> refined artifact

WHAT IS BEING SHORTENED, AND WHY IT IS NOT SIMPLY DELETED
---------------------------------------------------------
Every task is planned home -> pick -> place -> home, so a robot doing two tasks in a row
flies home in between. That commute is **most of all motion** (ADR-0008 records the figures),
and the obvious refinement is to cut it out and go straight from one task to the next.

That does not work, and the reason is the whole point of this module. Each task is planned
with the OTHER robot parked at home (ADR-0003) -- that decoupling is what lets the
trajectories be generated once, independently of the schedule. The plan survives it because
every task begins and ends at home, so any two conflicting motions can always be separated
*in time*: going home is how an arm gets out of the way. Splice the commute out and the arms
are permanently in the shared workspace; measured, the result is not merely slower but
**unexecutable** -- no interleaving of the two chains exists at all, however long either
robot waits (see ADR-0008, and ``coordinate.py`` which is what detects it).

So home is not overhead, it is a *yield pose*. The question is not whether to go home but
how far towards home one has to go, and the answer is measurable: a configuration is a valid
place to **park** when it is collision-free against every configuration the other robot
passes through in its whole plan. An arm usually reaches that condition well before home, and
the rest of the way is pure overshoot at both ends of the commute.

TWO POSES, TWO TESTS
--------------------
Parking and passing through are different jobs and get different tests (ADR-0008 addendum).

A parked arm holds one configuration for an *interval*, while the offset model checks a
configuration at exactly one offset -- the one where rigid execution makes it simultaneous.
Nothing downstream can see the difference, so the pose the arm STOPS at must be clear of
everything, exactly as ADR-0004 requires of home.

Every other node is only passed through, and there the whole-plan test is wasted
conservatism: it rejects motions that provably never coincide with the configurations it
checks them against. APEX-MR's test is used instead -- collision-check a shortcut only
against the other robot's nodes that are neither predecessors nor successors of the span it
replaces, since nodes ordered by the graph are never occupied simultaneously. The window
comes from the BASELINE graph via :meth:`tpg.TPG.independent_window`.

WHY THIS IS SAFE, AND WHY THE MAKESPAN CANNOT GET WORSE
-------------------------------------------------------
Think of the pair of plans as a coordination diagram: one axis per robot, one cell per pair
of configurations, a cell blocked iff that pair collides. Executing the pair is a monotone
walk from corner to corner avoiding blocked cells.

The nodes this module removes are free *for the partial order they were checked against* --
in the coordination diagram their rows and columns carry no blocked cell that the graph does
not already order. Deleting free rows and inserting free rows cannot destroy a monotone walk
(any walk in the old diagram maps to one in the new), so a plan that could be executed still
can be, and the walk is no longer than before.

Under the whole-plan test that argument is a proof, because the rows are free against every
schedule. Under the independence window it is a proof about the BASELINE order only, and the
re-solve may produce another -- so it degrades to a **check**: the refined makespan is
compared against the baseline every run. Safety does not rest on it either way, since `mu` is
recomputed on the refined trajectories and any real collision becomes an edge.

The refined plan still keeps the property ADR-0004 relies on -- an idle robot sits somewhere
that blocks nobody -- because the parking test above was not relaxed. Home was only ever one
particular choice of such a pose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

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
from tpg import TPG  # noqa: E402

PHASE_TO_PICK, PHASE_TO_HOME = 0, 4


def chains(solution: dict, robots: Sequence[str] = ()) -> Dict[str, List[str]]:
    """Each robot's scheduled tasks, in the order the solver put them.

    Every robot in ``robots`` gets an entry, empty if the solver gave it nothing. A robot
    with no tasks is a legitimate schedule -- it is what an allocation objective with no
    load-balancing term produces -- and it must not fall out of the mapping, or every
    lookup keyed by "the other robot" raises.
    """
    out: Dict[str, List[str]] = {r: [] for r in robots}
    for task, a in sorted(solution["assignments"].items(), key=lambda kv: kv[1]["start_slot"]):
        out.setdefault(a["robot"], []).append(task)
    return out


class Clearance:
    """Which samples of a trajectory are clear of everything the other robot ever does."""

    def __init__(self, task_file: str, robot: str = "ur10e_rail"):
        import vamp
        objects = {o["id"]: ObjectGeom.from_size(o["size"])
                   for o in yaml.safe_load(open(task_file))["objects"]}
        self.engine = VampCollisionEngine(
            getattr(vamp, robot), objects,
            base_transforms=CELL_BASE, mount_yaws=CELL_MOUNT_YAW,
            n_structural=CELL_N_STRUCTURAL, sphere_margin=CELL_MARGIN,
            groups=link_groups(DEFAULT_SPHERIZED_URDF, CELL_N_STRUCTURAL))
        self.kernel = MuKernel()
        if not self.kernel.available:
            raise RuntimeError("the SIMD kernel is required here; run scripts/build_mu_kernel.sh")
        self._packed: Dict[Tuple[str, str], object] = {}

    def pack(self, side: str, key, positions, object_state, obj):
        if key not in self._packed:
            spheres = self.engine.traj_spheres(key[0], positions, object_state, obj)
            self._packed[key] = self.kernel.pack(spheres, side)
        return self._packed[key]

    def free(self, side_a: str, key_a, packed_a, others) -> np.ndarray:
        """Boolean per sample of A: True where it collides with NO sample of any `others`."""
        hit = None
        for packed_b in others:
            m = (self.kernel.matrix(packed_a, packed_b) if side_a == "A"
                 else self.kernel.matrix(packed_b, packed_a).T)
            row = m.any(axis=1)
            hit = row if hit is None else (hit | row)
        return ~hit if hit is not None else np.ones(1, dtype=bool)


def _pack_all(cl: Clearance, art: dict, order: Dict[str, List[str]], robots):
    trs = {(t["robot"], t["task"]): t for t in art["trajectories"]}
    packed = {}
    for r in robots:
        side = "A" if r == robots[0] else "B"
        for t in order[r]:
            tr = trs[(r, t)]
            packed[(r, t)] = cl.pack(side, (r, t), tr["positions"], tr["object_state"], tr["object"])
    return trs, packed


class Against:
    """One trajectory's collisions against the other robot's WHOLE timeline, prefix-summed.

    Columns are the other robot's TPG node indices -- its tasks concatenated in schedule
    order, which is exactly how ``tpg.timelines`` numbers them. The prefix sum turns "does
    sample ``a`` collide with anything in node window ``[lo, hi)``" into two lookups, which
    is what makes an exhaustive scan of the candidate grid affordable (ADR-0008 addendum).
    """

    def __init__(self, cl: Clearance, side: str, packed_a, packed_others: List):
        cols = [cl.kernel.matrix(packed_a, pb) if side == "A"
                else cl.kernel.matrix(pb, packed_a).T for pb in packed_others]
        self.cum = np.cumsum(np.hstack(cols).astype(np.int32), axis=1)

    def hits(self, a: int, lo: int, hi: int) -> bool:
        """Does sample ``a`` collide with any other-robot node in ``[lo, hi)``?"""
        if hi <= lo:
            return False
        before = int(self.cum[a, lo - 1]) if lo > 0 else 0
        return int(self.cum[a, hi - 1]) - before > 0

    def anywhere(self, a: int) -> bool:
        """Does sample ``a`` collide with ANY configuration of the other robot's plan?"""
        return int(self.cum[a, -1]) > 0


class Windows:
    """The independence window of a shortcut span, read off the baseline graph.

    Both bounds are separable -- the left one depends only on where the shortcut leaves this
    robot's trajectory, the right one only on where it rejoins -- which is what lets the
    candidate grid be scanned as a rectangle instead of a search.
    """

    def __init__(self, tpg: TPG, robot: str, windowed: bool = True):
        self.tpg, self.robot, self.windowed = tpg, robot, windowed
        self.pref_self = np.maximum.accumulate(tpg.deps[robot])
        self.pref_other = np.maximum.accumulate(tpg.deps[tpg.other(robot)])
        self.n_other = int(tpg.deps[tpg.other(robot)].shape[0])
        self.base = {seg.task: seg.start_node for seg in tpg.segments[robot]}

    def node(self, task: str, sample: int) -> int:
        return self.base[task] + sample

    def lo(self, task: str, sample: int) -> int:
        """First other-robot node that is NOT already a predecessor of this one."""
        if not self.windowed:
            return 0                     # ablation: fall back to the whole-plan test
        return int(self.pref_self[self.node(task, sample)]) + 1

    def hi(self, task: str, sample: int) -> int:
        """First other-robot node that is a successor of this one."""
        if not self.windowed:
            return self.n_other          # ablation: fall back to the whole-plan test
        return int(np.searchsorted(self.pref_other, self.node(task, sample)))


def yield_points(art, sol, cl, tpg: TPG, robots, n_candidates: int = 16,
                 windowed: bool = True):
    """Candidate shortcuts for each splice, most aggressive first.

    Returns ``{(robot, task_i, task_j): [(out_in_i, in_in_j), ...]}`` -- node indices into
    the two ORIGINAL trajectories. ``out`` is the sample of task i's ToHome leg the arm
    stops at; ``in`` is the sample of task j's ToPick leg it rejoins at.

    Two different tests, because the two poses do different jobs:

    * ``out`` is where the arm **parks** between the tasks. An idle robot occupies one
      configuration for an interval, and the offset model checks a configuration at exactly
      one offset, so nothing downstream can see that. It must therefore be clear of the
      other robot's entire plan -- ADR-0004's rule, unchanged.
    * ``in`` is only **passed through**, so it needs to be clear of the other robot's nodes
      that could actually be concurrent: the independence window of the span (APEX-MR's
      test). Everything outside the window is ordered against the span by the graph, and
      ordered nodes are never occupied simultaneously.

    The admissible pairs are then scanned exhaustively -- the grid is one small rectangle
    and each cell is two lookups -- and the ladder is drawn from the best of them.
    """
    order = chains(sol, robots)
    trs, packed = _pack_all(cl, art, order, robots)
    out = {}
    for r in robots:
        other = [q for q in robots if q != r][0]
        side = "A" if r == robots[0] else "B"
        others = [packed[(other, u)] for u in order[other]]
        if not others:
            print(f"  {r}: {other} has no scheduled tasks; nothing to yield to, "
                  f"keeping every commute whole")
            continue
        win = Windows(tpg, r, windowed)
        for i, j in zip(order[r], order[r][1:]):
            ag_i = Against(cl, side, packed[(r, i)], others)
            ag_j = Against(cl, side, packed[(r, j)], others)
            home_leg = np.flatnonzero(np.asarray(trs[(r, i)]["phase"]) == PHASE_TO_HOME)
            pick_leg = np.flatnonzero(np.asarray(trs[(r, j)]["phase"]) == PHASE_TO_PICK)
            park = [int(o) for o in home_leg if not ag_i.anywhere(int(o))]
            if not park:
                print(f"  {r}: {i} -> {j}: nowhere clear to park on the commute; keeping it whole")
                continue

            # The grid: every (park, rejoin) pair the independence test admits.
            end_i, start_j = int(home_leg[-1]), int(pick_leg[0])
            grid = []
            for o in park:
                lo = win.lo(i, o)
                for n in (int(x) for x in pick_leg):
                    if not ag_j.hits(n, lo, win.hi(j, n)):
                        grid.append(((end_i - o) + (n - start_j), o, n))
            if not grid:
                print(f"  {r}: {i} -> {j}: no rejoin point admitted; keeping the commute whole")
                continue
            grid.sort(reverse=True)
            best = grid[0][0]

            # A ladder, boldest first: both endpoints being clear does not make the motion
            # between them clear, and that is only knowable once planned. The last rung is
            # the original commute, so the ladder always has a valid bottom.
            #
            # The ladder is FINE (16 rungs, not 5), because a rejected rung costs the whole
            # gap to the next one, and rejections are common: a span one sample away often
            # succeeds where this one failed. Transit planning is also stochastic (OMPL is
            # unseeded), so a coarse ladder converts one unlucky draw into a third of the
            # saving lost. Rungs are cheap -- tens of milliseconds of planning each -- and
            # measured, the ladder is the more consistent of this refinement's two effects.
            cand, wanted = [], np.linspace(1.0, 0.0, n_candidates)
            for frac in wanted:
                target = frac * best
                _, o, n = min(grid, key=lambda g: abs(g[0] - target))
                if (o, n) not in cand:
                    cand.append((o, n))
            if (end_i, start_j) not in cand:
                cand.append((end_i, start_j))
            out[(r, i, j)] = cand

            widest = win.hi(j, int(pick_leg[-1])) - win.lo(i, park[0])
            print(f"  {r}: {i} -> {j}: {len(park)}/{home_leg.size} parking poses, "
                  f"{len(grid)} admissible pairs, best saves {best} slots "
                  f"({best * float(art['delta_t']):.2f} s); the independence window is at "
                  f"most {max(widest, 0)} of {win.n_other} {other} nodes -- "
                  f"{len(cand)} candidate shortcuts")
    return out


def cmd_plan(args) -> int:
    art = json.load(open(args.traj))
    sol = json.load(open(args.solution))
    robots = list(art["robots"])
    cl = Clearance(args.task, args.robot)
    tpg = TPG.from_json(args.tpg)
    rule = ("independence window" if args.test == "window" else "the whole plan (ablation)")
    print(f"choosing yield poses (park: clear of everything; rejoin: {rule}; "
          f"{args.rungs} rungs):")
    pts = yield_points(art, sol, cl, tpg, robots, args.rungs, args.test == "window")
    trs = {(t["robot"], t["task"]): t for t in art["trajectories"]}

    spec = {"transits": [], "index": {}}
    for (r, i, j), cand in sorted(pts.items()):
        spec["index"][f"{r}|{i}|{j}"] = cand
        for rank, (o, n) in enumerate(cand):
            spec["transits"].append({
                "id": f"{r}|{i}|{j}|{rank}", "robot": r, "task": j,
                "from": list(trs[(r, i)]["positions"][o]),
                "to": list(trs[(r, j)]["positions"][n]),
            })
    with open(args.out, "w") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")
    print(f"wrote transit spec ({len(spec['transits'])} splices) -> {args.out}")
    return 0


def cmd_splice(args) -> int:
    """Assemble the refined artifact and VERIFY every inserted node is clear."""
    art = json.load(open(args.traj))
    sol = json.load(open(args.solution))
    spec = json.load(open(args.spec))
    transits = json.load(open(args.transits))["transits"]
    robots = list(art["robots"])
    order = chains(sol, robots)
    trs = {(t["robot"], t["task"]): t for t in art["trajectories"]}
    cl = Clearance(args.task, args.robot)
    tpg = TPG.from_json(args.tpg)
    _, packed = _pack_all(cl, art, order, robots)

    got = {t["id"]: t for t in transits}
    keep_head: Dict[Tuple[str, str], int] = {}     # task -> first sample to keep
    keep_tail: Dict[Tuple[str, str], int] = {}     # task -> last sample to keep
    prepend: Dict[Tuple[str, str], list] = {}

    for key, cand in sorted(spec["index"].items()):
        r, i, j = key.split("|")
        other = [q for q in robots if q != r][0]
        side = "A" if r == robots[0] else "B"
        others = [packed[(other, u)] for u in order[other]]
        win = Windows(tpg, r, args.test == "window")
        for rank, (o, n) in enumerate(cand):
            tr = got.get(f"{r}|{i}|{j}|{rank}")
            if tr is None or not tr["planned"]:
                continue
            pos = np.asarray(tr["positions"], dtype=float)
            old = (len(trs[(r, i)]["positions"]) - 1 - o) + n
            new = len(pos) - 2
            if new >= old:
                continue                       # not actually a shortcut
            # The transit must be clear of everything that can run WHILE it runs -- the
            # independence window of the span it replaces. Verified, not trusted. Nodes
            # outside the window are ordered against this span by the graph, so checking
            # them would reject motions that are provably never concurrent.
            lo, hi = win.lo(i, o), win.hi(j, n)
            pk = cl.kernel.pack(
                cl.engine.traj_spheres(r, tr["positions"], [0] * len(pos),
                                       trs[(r, j)]["object"]), side)
            ag = Against(cl, side, pk, others)
            touched = [a for a in range(len(pos)) if ag.hits(a, lo, hi)]
            if touched:
                print(f"  {r}: {i} -> {j}: rung {rank} touches {other} nodes {lo}..{hi} at "
                      f"{len(touched)}/{len(pos)} samples -- rejected")
                continue
            keep_tail[(r, i)] = o
            keep_head[(r, j)] = n
            prepend[(r, j)] = pos[1:-1].tolist()   # endpoints duplicate the samples joined
            print(f"  {r}: {i} -> {j}: rung {rank} accepted -- {old} slots of commute "
                  f"replaced by {new}, saving {old - new} "
                  f"(checked against {other} nodes {lo}..{hi})")
            break
        else:
            print(f"  {r}: {i} -> {j}: no rung verified; keeping the full commute")

    out = []
    for r in robots:
        for t in order[r]:
            tr = trs[(r, t)]
            lo = keep_head.get((r, t), 0)
            hi = keep_tail.get((r, t), len(tr["positions"]) - 1)
            pre = prepend.get((r, t), [])
            positions = pre + [list(p) for p in tr["positions"][lo:hi + 1]]
            phase = [PHASE_TO_PICK] * len(pre) + list(tr["phase"][lo:hi + 1])
            objst = [0] * len(pre) + list(tr["object_state"][lo:hi + 1])
            out.append({
                "robot": r, "task": t, "object": tr["object"],
                "num_samples": len(positions), "used_velocities": tr["used_velocities"],
                "max_joint_step": tr["max_joint_step"], "joint_names": tr["joint_names"],
                "positions": positions, "phase": phase, "object_state": objst,
            })

    refined = {k: art[k] for k in ("delta_t", "gripper_dwell_slots", "robots", "tasks",
                                   "precedences", "homes")}
    refined["chains"] = order
    refined["trajectories"] = out
    with open(args.out, "w") as f:
        json.dump(refined, f)
        f.write("\n")
    before = sum(len(trs[(r, t)]["positions"]) for r in robots for t in order[r])
    after = sum(t["num_samples"] for t in out)
    print(f"\nmotion {before} -> {after} slots ({100 * (after - before) / before:+.1f}%), "
          f"{(before - after) * art['delta_t']:.1f} s removed")
    print(f"wrote refined trajectories -> {args.out}")
    return 0


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.dirname(here)
    art = os.path.join(pkg, "artifacts")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", action="store_true", help="choose yield poses -> transit spec")
    p.add_argument("--splice", action="store_true", help="assemble + verify the refined artifact")
    p.add_argument("--traj", default=os.path.join(art, "tamp_trajectories.json"))
    p.add_argument("--solution", default=os.path.join(art, "vamp", "tamp_solution.json"))
    p.add_argument("--spec", default=os.path.join(art, "transit_spec.json"))
    p.add_argument("--transits", default=os.path.join(art, "transits.json"))
    p.add_argument("--tpg", default=os.path.join(art, "vamp", "tpg.json"),
                   help="the BASELINE temporal plan graph, whose partial order says which "
                        "of the other robot's nodes a shortcut can possibly meet")
    p.add_argument("--test", choices=["window", "whole"], default="window",
                   help="what a shortcut is checked against: 'window' is the independence "
                        "window of the span it replaces (APEX-MR's test, the default), "
                        "'whole' is the other robot's entire plan. An ablation knob -- the "
                        "two differ only in Windows.lo/.hi.")
    p.add_argument("--rungs", type=int, default=16,
                   help="candidate shortcuts per splice, boldest first (default 16)")
    p.add_argument("--task", default=os.path.join(pkg, "config", "tamp_task_tower.yaml"))
    p.add_argument("--robot", default="ur10e_rail")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    if args.plan == args.splice:
        p.error("pass exactly one of --plan / --splice")
    if args.out is None:
        args.out = args.spec if args.plan else os.path.join(art, "tamp_trajectories_refined.json")
    return cmd_plan(args) if args.plan else cmd_splice(args)


if __name__ == "__main__":
    raise SystemExit(main())
