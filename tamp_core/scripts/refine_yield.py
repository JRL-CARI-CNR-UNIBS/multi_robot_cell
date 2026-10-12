#!/usr/bin/env python3
"""Shorten the commute between two consecutive tasks, without coupling the robots.

    .venv_vamp/bin/python scripts/refine_yield.py --plan   ...   # -> transit spec
    <ROS: trajectory_generator with transit_file=...>            # -> transits.json
    .venv_vamp/bin/python scripts/refine_yield.py --splice ...   # -> refined artifact

or LAZILY, the pipeline's default (``--lazy-first K``): plan only the first K rungs of every
ladder, splice, and plan the rest only for the splices none of those rungs resolved:

    refine_yield.py --plan --lazy-first 2 --spec-a S_a ...        # full spec + round-A spec
    <trajectory_generator transit_file=S_a>                      # -> T_a
    refine_yield.py --splice --transits T_a --next-spec S_b ...   # round-B spec (maybe empty)
    <trajectory_generator transit_file=S_b>                      # -> T_b   (only if S_b non-empty)
    refine_yield.py --splice --transits T_a T_b ...               # final refined artifact

The splice takes, per splice, the boldest rung that PLANNED AND VERIFIED; a rung missing from
every transit file counts as not planned, and every rung below K is in round A. So a splice
round A resolves is final (no later rung is ever looked at), and one it does not resolve sees
exactly the full ladder in round B: with the same transit trajectories the result is
identical to planning the whole ladder at once -- only rungs nobody would use are skipped.

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
re-derived on the refined trajectories and any real collision becomes an edge. (Re-derived,
not necessarily recomputed: a refined sample identical to an original one reuses that
sample's cached cells, mu_cache.py -- a cell depends on its two samples and nothing else.)

``--plan`` reads the baseline graph's mu blocks (``tpg.mu.npz`` next to ``--tpg``) instead
of recomputing every scheduled matrix, and builds each task's :class:`Against` once (the
task j of one splice is the task i of the next).

The refined plan still keeps the property ADR-0004 relies on -- an idle robot sits somewhere
that blocks nobody -- because the parking test above was not relaxed. Home was only ever one
particular choice of such a pose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell_registry import scene_path  # noqa: E402
import sys
from typing import Dict, List, Sequence, Tuple

for _threadvar in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_threadvar, "1")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vamp_collision_engine import make_cell_engines, objects_of_scene  # noqa: E402
from mu_kernel import MuKernel  # noqa: E402
from mu_cache import MuBlocks, MuSource, row_digests  # noqa: E402
from tpg import TPG  # noqa: E402

PHASE_TO_PICK, PHASE_TO_HOME = 0, 4


def require_two_robots(robots: Sequence[str], what: str = "refine_yield.py") -> None:
    """Refinement is two-robot by construction (ADR-0008): a parking pose is chosen clear
    of THE other robot's plan and a shortcut is checked against ITS independence window.
    A plan with more robots is refused here rather than refined against one of them."""
    if len(robots) != 2:
        raise ValueError(
            f"{what}: yield-pose refinement (ADR-0008) is defined for exactly two robots; "
            f"this plan has {len(robots)} ({list(robots)}). Run the pipeline with "
            f"refine:=false (the plan graph itself handles N robots).")


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

    def __init__(self, task_file: str, robots: Sequence[str], robot: str = "ur10e_rail",
                 mu_cache: str = ""):
        import vamp
        task = yaml.safe_load(open(task_file))
        # One engine per robot from the scene's cell (ADR-0012): the dual cell's is the one
        # shared `robot` module with CELL_BASE / CELL_MOUNT_YAW it always was; TIAGo's arms are
        # two modules. Both sides of a kernel matrix are packed to the larger sphere / group
        # count (equal for one module, so the dual packing is unchanged).
        self.engines = make_cell_engines(vamp, task, objects_of_scene(task), robots,
                                         dual_module=robot)
        engs = list({id(e): e for e in self.engines.values()}.values())
        self.n_sph = max(e.n_spheres + e.n_patch + 1 for e in engs)
        self.n_grp = max(e.n_groups for e in engs)
        self.kernel = MuKernel()
        if not self.kernel.available:
            raise RuntimeError("the SIMD kernel is required here; run scripts/build_mu_kernel.sh")
        self._packed: Dict[Tuple[str, str], object] = {}
        self._geo: Dict[Tuple[str, str], "Geo"] = {}
        # The baseline graph's mu blocks (mu_cache.py), keyed by geometry: every matrix of a
        # scheduled pair is served from there instead of recomputed; rows it does not know
        # (a transit) are computed. Absent or stale -> everything is computed, as before.
        store = MuBlocks.load(mu_cache) if mu_cache else None
        if store is not None:
            print(f"mu cache: {len(store)} blocks from {mu_cache}")
        self.src = MuSource(self.kernel, store, min_known=0.5)

    def pack(self, side: str, key, positions, object_state, obj):
        return self.geo(side, key, positions, object_state, obj).packed

    def geo(self, side: str, key, positions, object_state, obj) -> "Geo":
        """One trajectory's spheres, row digests and packed planes (cached per key)."""
        if key not in self._geo:
            spheres = self.engines[key[0]].traj_spheres(key[0], positions, object_state, obj)
            self._geo[key] = self.geo_of(spheres, side)
            self._packed[key] = self._geo[key].packed
        return self._geo[key]

    def geo_of(self, spheres, side: str) -> "Geo":
        return Geo(spheres, row_digests(spheres) if self.src.store is not None else None,
                   self.pack_spheres(spheres, side))

    def mu(self, side_a: str, a: "Geo", b: "Geo") -> np.ndarray:
        """``mu`` with ``a``'s samples as rows, whatever its side; the kernel's A side is
        always the first robot, as in the baseline graph (so cached blocks line up)."""
        if side_a == "A":
            m = None if a.dig is None else self.src.assemble(
                a.spheres, a.dig, b.spheres, b.dig, self.n_sph, self.n_grp)
            return m if m is not None else self.kernel.matrix(a.packed, b.packed)
        m = None if a.dig is None else self.src.assemble(
            b.spheres, b.dig, a.spheres, a.dig, self.n_sph, self.n_grp)
        return (m if m is not None else self.kernel.matrix(b.packed, a.packed)).T

    def pack_spheres(self, spheres, side: str):
        return self.kernel.pack(spheres, side, n_sph=self.n_sph, n_grp=self.n_grp)

    def free(self, side_a: str, key_a, packed_a, others) -> np.ndarray:
        """Boolean per sample of A: True where it collides with NO sample of any `others`."""
        hit = None
        for packed_b in others:
            m = (self.kernel.matrix(packed_a, packed_b) if side_a == "A"
                 else self.kernel.matrix(packed_b, packed_a).T)
            row = m.any(axis=1)
            hit = row if hit is None else (hit | row)
        return ~hit if hit is not None else np.ones(1, dtype=bool)


class Geo:
    """A trajectory as the collision checks need it: spheres, row digests, packed planes."""

    __slots__ = ("spheres", "dig", "packed")

    def __init__(self, spheres, dig, packed):
        self.spheres, self.dig, self.packed = spheres, dig, packed


def _pack_all(cl: Clearance, art: dict, order: Dict[str, List[str]], robots):
    trs = {(t["robot"], t["task"]): t for t in art["trajectories"]}
    geos = {}
    for r in robots:
        side = "A" if r == robots[0] else "B"
        for t in order[r]:
            tr = trs[(r, t)]
            geos[(r, t)] = cl.geo(side, (r, t), tr["positions"], tr["object_state"], tr["object"])
    return trs, geos


class Against:
    """One trajectory's collisions against the other robot's WHOLE timeline, prefix-summed.

    Columns are the other robot's TPG node indices -- its tasks concatenated in schedule
    order, which is exactly how ``tpg.timelines`` numbers them. The prefix sum turns "does
    sample ``a`` collide with anything in node window ``[lo, hi)``" into two lookups, which
    is what makes an exhaustive scan of the candidate grid affordable (ADR-0008 addendum).
    """

    def __init__(self, cl: Clearance, side: str, geo_a: Geo, geo_others: List[Geo]):
        cols = [cl.mu(side, geo_a, gb) for gb in geo_others]
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
    require_two_robots(robots)
    order = chains(sol, robots)
    trs, geos = _pack_all(cl, art, order, robots)
    out = {}
    against: Dict[Tuple[str, str], Against] = {}   # task j of one splice is task i of the next

    def ag(r: str, t: str, side: str, others: List[Geo]) -> Against:
        if (r, t) not in against:
            against[(r, t)] = Against(cl, side, geos[(r, t)], others)
        return against[(r, t)]

    for r in robots:
        other = [q for q in robots if q != r][0]
        side = "A" if r == robots[0] else "B"
        others = [geos[(other, u)] for u in order[other]]
        if not others:
            print(f"  {r}: {other} has no scheduled tasks; nothing to yield to, "
                  f"keeping every commute whole")
            continue
        win = Windows(tpg, r, windowed)
        for i, j in zip(order[r], order[r][1:]):
            ag_i = ag(r, i, side, others)
            ag_j = ag(r, j, side, others)
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
    require_two_robots(robots, "refine_yield.py --plan")
    cl = Clearance(args.task, robots, args.robot, mu_cache_path(args))
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
    write_spec(spec, args.out)
    print(f"wrote transit spec ({len(spec['transits'])} candidate transits, "
          f"{len(spec['index'])} splices) -> {args.out}")
    if args.lazy_first > 0:
        # Round A of the lazy ladder: the boldest K rungs of every splice. The full spec
        # above stays the reference -- the splice reads its index, and the rungs left out
        # here are planned later only for the splices round A does not resolve.
        first = sub_spec(spec, {t["id"] for t in spec["transits"]
                                if rank_of(t["id"]) < args.lazy_first})
        write_spec(first, args.spec_a)
        print(f"lazy ladder: round A plans the first {args.lazy_first} rung(s) of each "
              f"splice, {len(first['transits'])} of {len(spec['transits'])} transits "
              f"-> {args.spec_a}")
    return 0


def rank_of(transit_id: str) -> int:
    """The rung of a transit id ``robot|task_i|task_j|rank``."""
    return int(transit_id.rsplit("|", 1)[1])


def splice_of(transit_id: str) -> str:
    """The splice key ``robot|task_i|task_j`` of a transit id."""
    return transit_id.rsplit("|", 1)[0]


def sub_spec(spec: dict, ids) -> dict:
    """``spec`` restricted to the transits in ``ids`` (spec order kept; index restricted to
    the splices that still have a transit, for whoever reads the file)."""
    ts = [t for t in spec["transits"] if t["id"] in ids]
    keys = {splice_of(t["id"]) for t in ts}
    return {"transits": ts, "index": {k: v for k, v in spec["index"].items() if k in keys}}


def write_spec(spec: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")


def load_transits(paths: Sequence[str]) -> Tuple[dict, list]:
    """The planned transits of one or several rounds, by id. A rung in two files would make
    "which trajectory was verified" ambiguous, so it is an error, not a merge."""
    got, delta_t = {}, None
    for path in paths:
        doc = json.load(open(path))
        delta_t = doc.get("delta_t", delta_t)
        for t in doc["transits"]:
            if t["id"] in got:
                raise ValueError(f"transit {t['id']} appears in more than one of {list(paths)}")
            got[t["id"]] = t
    return got, delta_t


def cmd_splice(args) -> int:
    """Assemble the refined artifact and VERIFY every inserted node is clear."""
    art = json.load(open(args.traj))
    sol = json.load(open(args.solution))
    spec = json.load(open(args.spec))
    got, transit_dt = load_transits(args.transits)
    robots = list(art["robots"])
    require_two_robots(robots, "refine_yield.py --splice")
    order = chains(sol, robots)
    trs = {(t["robot"], t["task"]): t for t in art["trajectories"]}
    # No mu cache here: the only matrices this stage computes are the transits', whose rows
    # are new by definition.
    cl = Clearance(args.task, robots, args.robot)
    tpg = TPG.from_json(args.tpg)
    _, geos = _pack_all(cl, art, order, robots)

    keep_head: Dict[Tuple[str, str], int] = {}     # task -> first sample to keep
    keep_tail: Dict[Tuple[str, str], int] = {}     # task -> last sample to keep
    prepend: Dict[Tuple[str, str], list] = {}

    resolved = set()                                # splice keys a rung was accepted for
    for key, cand in sorted(spec["index"].items()):
        r, i, j = key.split("|")
        other = [q for q in robots if q != r][0]
        side = "A" if r == robots[0] else "B"
        others = [geos[(other, u)] for u in order[other]]
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
            pk = cl.geo_of(
                cl.engines[r].traj_spheres(r, tr["positions"], [0] * len(pos),
                                           trs[(r, j)]["object"]), side)
            ag = Against(cl, side, pk, others)
            touched = [a for a in range(len(pos)) if ag.hits(a, lo, hi)]
            if touched:
                print(f"  {r}: {i} -> {j}: rung {rank} touches {other} nodes {lo}..{hi} at "
                      f"{len(touched)}/{len(pos)} samples -- rejected")
                continue
            resolved.add(key)
            keep_tail[(r, i)] = o
            keep_head[(r, j)] = n
            prepend[(r, j)] = pos[1:-1].tolist()   # endpoints duplicate the samples joined
            print(f"  {r}: {i} -> {j}: rung {rank} accepted -- {old} slots of commute "
                  f"replaced by {new}, saving {old - new} "
                  f"(checked against {other} nodes {lo}..{hi})")
            break
        else:
            unplanned = sum(f"{key}|{rank}" not in got for rank in range(len(cand)))
            if unplanned:
                print(f"  {r}: {i} -> {j}: no rung verified yet; {unplanned} rung(s) not "
                      f"planned in this round, keeping the full commute for now")
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
                "robot": r, "task": t, "slot": tr.get("slot", t), "object": tr["object"],
                "num_samples": len(positions), "used_velocities": tr["used_velocities"],
                "max_joint_step": tr["max_joint_step"], "joint_names": tr["joint_names"],
                "positions": positions, "phase": phase, "object_state": objst,
            })

    # A refined artifact is a COMMITTED plan: it holds one chain per robot, so the only
    # tasks it can describe are the ones the schedule runs. That is every task in a scene
    # without interchangeable slots -- `kept` is then the full list and nothing below
    # changes -- but with slots three candidates per slot lost, have no refined
    # trajectory, and so no candidate robot. Carrying them into the refined seam makes
    # the re-solve reject the problem as trivially infeasible.
    kept = {t for r in robots for t in order[r]}
    live = [k for k, pair in enumerate(art["precedences"])
            if pair[0] in kept and pair[1] in kept]
    # Key order matches the C++ writer's, so a refined artifact still diffs cleanly
    # against an unrefined one.
    refined = {
        "delta_t": art["delta_t"],
        "gripper_dwell_slots": art["gripper_dwell_slots"],
        "robots": art["robots"],
        "tasks": [t for t in art["tasks"] if t in kept],
        "precedences": [art["precedences"][k] for k in live],
        "homes": art["homes"],
    }
    # NOT taken wholesale on purpose: an artifact from before precedence modes has no
    # such key, and one that has it must keep it -- dropping it here would silently turn
    # every `gate` back into `pipeline` in the refined plan and let a weld strike before
    # its part lands, with no error anywhere downstream.
    if "precedence_modes" in art:
        refined["precedence_modes"] = [art["precedence_modes"][k] for k in live]
    # Interchangeable slots, restricted the same way. The slot structure itself stays --
    # the re-solve still has to keep the levels in order -- but each slot is now down to
    # its winner, so exactly-one and at-most-one-per-object are satisfied by construction.
    if "slot_of" in art:
        refined["slot_of"] = {t: s for t, s in art["slot_of"].items() if t in kept}
        live_slots = set(refined["slot_of"].values())
        if "object_of" in art:
            refined["object_of"] = {t: o for t, o in art["object_of"].items() if t in kept}
        sp = art.get("slot_precedences", [])
        modes = art.get("slot_precedence_modes")
        live_sp = [k for k, pair in enumerate(sp)
                   if pair[0] in live_slots and pair[1] in live_slots]
        refined["slot_precedences"] = [sp[k] for k in live_sp]
        if modes is not None:
            refined["slot_precedence_modes"] = [modes[k] for k in live_sp]
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

    if args.next_spec:
        # The lazy ladder's next round: every rung not planned yet, of the splices no rung
        # resolved. A resolved splice is final -- its accepted rung is the boldest that
        # planned and verified, and nothing below it is ever looked at.
        todo = {t["id"] for t in spec["transits"]
                if t["id"] not in got and splice_of(t["id"]) not in resolved}
        nxt = sub_spec(spec, todo)
        write_spec(nxt, args.next_spec)
        print(f"lazy ladder: {len(spec['index']) - len(nxt['index'])} of "
              f"{len(spec['index'])} splices settled; {len(nxt['transits'])} rung(s) of "
              f"{len(nxt['index'])} splice(s) left to plan -> {args.next_spec}")
    if args.transits_out:
        # Every transit planned so far, in the full spec's order: the record a snapshot of
        # the run keeps as `transits.json` (re-splicing it reproduces this result).
        order_ids = [t["id"] for t in spec["transits"]]
        known = set(order_ids)
        merged = [got[k] for k in order_ids if k in got] + [
            t for k, t in got.items() if k not in known]
        with open(args.transits_out, "w") as f:
            json.dump({"delta_t": transit_dt if transit_dt is not None else art["delta_t"],
                       "transits": merged}, f)
            f.write("\n")
    return 0


def mu_cache_path(args) -> str:
    """The baseline graph's mu blocks: ``--mu-cache``, or by default the file build_tpg.py
    writes next to ``--tpg`` (``tpg.json`` -> ``tpg.mu.npz``); '' = none."""
    if args.mu_cache is not None:
        return args.mu_cache
    return os.path.splitext(args.tpg)[0] + ".mu.npz"


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
    p.add_argument("--transits", nargs="+", default=[os.path.join(art, "transits.json")],
                   help="--splice: the planned transits, one file per round of the lazy "
                        "ladder (a rung in none of them counts as not planned)")
    p.add_argument("--lazy-first", type=int, default=0,
                   help="--plan: also write the round-A spec (--spec-a) holding only the "
                        "first K rungs of every splice. 0 (default): no lazy ladder")
    p.add_argument("--spec-a", default=None,
                   help="--plan with --lazy-first: where the round-A spec goes "
                        "(default: --out with an _a suffix)")
    p.add_argument("--next-spec", default=None,
                   help="--splice: write the next round's spec -- the rungs not planned yet "
                        "of every splice no rung resolved (empty transits list if none)")
    p.add_argument("--transits-out", default=None,
                   help="--splice: write every transit read (all rounds) to one file, in "
                        "spec order")
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
    p.add_argument("--task", default=scene_path("tower"))
    p.add_argument("--robot", default="ur10e_rail")
    p.add_argument("--out", default=None)
    p.add_argument("--mu-cache", default=None,
                   help="the baseline graph's mu blocks (default: next to --tpg, as "
                        "build_tpg.py writes them); '' computes every matrix")
    args = p.parse_args(argv)
    if args.plan == args.splice:
        p.error("pass exactly one of --plan / --splice")
    if args.out is None:
        args.out = args.spec if args.plan else os.path.join(art, "tamp_trajectories_refined.json")
    if args.plan and args.lazy_first > 0 and args.spec_a is None:
        stem, ext = os.path.splitext(args.out)
        args.spec_a = f"{stem}_a{ext}"
    return cmd_plan(args) if args.plan else cmd_splice(args)


if __name__ == "__main__":
    raise SystemExit(main())
