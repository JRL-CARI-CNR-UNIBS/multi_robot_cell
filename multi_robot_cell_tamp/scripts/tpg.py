"""Temporal plan graph: the execution layer that survives delays.

THE PROBLEM THIS SOLVES
-----------------------
The offset identity (ADR-0001) is exact, but it holds only if trajectories are executed
**rigidly** -- one sample per slot, never pausing. The scheduler therefore guarantees
collision-freedom for the offsets it chose, and for nothing else. The moment a controller
lags, the realised offset ``d`` becomes some ``d'`` the solver never approved, and the
guarantee does not degrade -- it disappears. Open-loop execution has no way to notice.

A temporal plan graph (TPG) removes the assumption instead of trying to enforce it. It
replaces *timing* with *precedence*: rather than "robot 2 starts at slot 314", it says
"robot 2 may not enter this configuration until robot 1 has left that one". Delay then
costs time and nothing else -- a late robot makes others wait, and every wait is one the
graph already proves safe. This is the construction of APEX-MR (Huang et al.), adapted to
our seam.

WHY WE ALREADY HAVE THE EXPENSIVE PART
--------------------------------------
APEX-MR calls TPG construction's bottleneck "the number of collision checks, which scales
quadratically with the number of robots and nodes". That check is exactly ``mu`` -- which
this pipeline already computes, in well under a second. The offset reduction
``D = {k - l}`` is a LOSSY projection of ``mu``: it keeps the difference between colliding
indices and discards which pairs collide. The TPG needs precisely the discarded half, so
the two are different consumers of one computation, not two computations.

ONE EDGE PER NODE, NOT ONE PER COLLIDING PAIR
---------------------------------------------
``mu`` holds well over a million colliding pairs on a four-task scene; enumerating an edge for each and
then running a transitive reduction (as APEX-MR does) would be wasteful. It is unnecessary:
a robot traverses its node sequence **monotonically**, so if it has completed node ``n`` it
has completed every earlier node. For a node ``m`` of the other robot, only

    dep[m] = max{ n : mu says n and m collide, and n is nominally earlier than m }

can bind -- waiting past that node implies waiting past every earlier colliding one. That
is the transitive reduction, computed in closed form, and it leaves exactly **one edge per
node**. Construction is O(K^2) in the collision scan we already pay for and O(K) in edges.

WHY THE GRAPH IS ACYCLIC (AND THEREFORE DEADLOCK-FREE)
------------------------------------------------------
Every edge is oriented from the nominally earlier node to the nominally later one, and the
within-robot sequence also advances in nominal time. So every edge in the graph increases
nominal time, and a cycle would need to return to its start -- impossible. Deadlock-freedom
is thus inherited from the solver's schedule, which is why the CP-SAT stage is still
required: it decides the assignment and the order, and its timing orients the graph.

Correction (2026-09-21): an edge never DEcreases nominal time, but it can keep it equal --
the "+1" makes the required node nominally simultaneous with the waiting one whenever the
colliding pair sits one slot off the scheduled offset. Two such edges can close a cycle
(a one-slot hole in the forbidden offsets at the scheduled offset: ``delta`` free, both
``delta +- 1`` colliding). :func:`assert_acyclic` therefore also runs an exact O(nodes)
cycle check, so such a schedule fails at build time instead of hanging the executor.

A collision at *equal* nominal time cannot occur, and :func:`build` asserts it: two
configurations colliding at the same instant would mean the realised offset was in ``D``,
i.e. the solver returned a schedule violating its own constraints.

TWO KINDS OF EDGE
-----------------
Collisions are not the only reason one robot must wait for another. The solver also
imposes **task precedences** -- de-stack a tower top-down, re-stack it bottom-up -- and
those are semantic, not geometric: picking a box that still has another box on top of it
is wrong even when the two arms are nowhere near each other. Timing alone used to enforce
them; once timing stops being the contract they must become edges too, or a delayed run
can reorder the task plan while remaining perfectly collision-free. See
:func:`precedence_edges`.

WHAT IS ASSUMED
---------------
That a robot resting at HOME blocks nobody. Every trajectory begins and ends at HOME and
the model never constrains an *idle* robot, so this must hold independently -- it is
verified for this cell (0 colliding samples against a parked HOME, across every task pair).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# A node with no cross-robot dependency. Chosen as -1 so ``done >= dep`` is trivially true
# at start of execution, where ``done`` is "index of last completed node" and begins at -1.
FREE = -1

# Date stamp of the graph construction, written into every tpg.json. Bump it whenever a
# change to this module changes what a graph built from the same inputs contains, so a
# campaign driver can refuse graphs built by an older builder (:func:`check_fresh`).
# 2026-09-21: slot precedences as edges (2026-09-19), the empty-robot and equal-start
# guards, the exact cycle check, and `precedence_edges_expected`. Graphs archived before
# then carry no stamp.
BUILDER_VERSION = "2026-09-21"


@dataclass
class Segment:
    """One scheduled task occupying a contiguous run of a robot's nodes."""

    task: str
    start_node: int
    n: int
    start_slot: int  # nominal, from the solver -- used to orient edges, not to execute


@dataclass
class TPG:
    """Two robots' timelines plus the cross-robot precedences between them.

    ``deps[r][n]`` is the node the *other* robot must have **reached** before robot ``r``
    may enter its node ``n`` (``FREE`` when unconstrained), i.e. execution requires
    ``reached[other] >= deps[r][n]``. One entry per node is sufficient -- see the module
    docstring.

    For a COLLISION edge the stored index is the colliding node's SUCCESSOR, not the
    colliding node itself. That +1 is the whole safety argument: requiring the other robot
    merely to have *reached* the colliding configuration would leave it sitting exactly
    there, which is the collision we are trying to forbid. Requiring it to have reached the
    next one means it has LEFT the colliding configuration.

    A PRECEDENCE edge stores the milestone node itself, with no +1, because the solver's
    constraints are non-strict (``start[j] >= pick[i]`` permits equality). The two
    conventions differ because the underlying conditions differ: "has left" for geometry,
    "has reached" for task order.
    """

    robots: Tuple[str, str]
    segments: Dict[str, List[Segment]]
    deps: Dict[str, np.ndarray]
    delta_t: float
    nominal_makespan: int
    n_edges: int = field(default=0)
    # How many of the edges come from task precedences rather than geometry. Reported
    # separately because they are the ones a collision-only test can never catch.
    n_precedence_edges: int = field(default=0)
    # Slots of differential delay the RIGID schedule could absorb before some pair of
    # trajectories reaches a forbidden offset. Recorded because it is the quantity this
    # graph makes irrelevant -- and because it is an accident. The solver minimises
    # makespan and has no notion of a robustness margin, so this number is whatever the
    # optimum happened to leave; nothing stops it being one slot on a denser scene.
    delay_margin: int = field(default=-1)
    # How many precedence edges the SEAM calls for (cross-robot milestone conditions of
    # the scheduled tasks, :func:`expected_precedence_edges`), computed without looking at
    # the edges. ``None`` when built without the seam. :func:`check_fresh` compares it with
    # ``n_precedence_edges``.
    precedence_edges_expected: "int | None" = field(default=None)
    builder_version: "str | None" = field(default=BUILDER_VERSION)

    def other(self, robot: str) -> str:
        return self.robots[1] if robot == self.robots[0] else self.robots[0]

    def n_nodes(self, robot: str) -> int:
        return int(self.deps[robot].shape[0])

    def locate(self, robot: str, node: int) -> Tuple[str, int]:
        """Node index -> (task, sample index within that task's trajectory)."""
        for seg in self.segments[robot]:
            if seg.start_node <= node < seg.start_node + seg.n:
                return seg.task, node - seg.start_node
        raise IndexError(f"{robot} has no node {node}")

    def independent_window(self, robot: str, m: int, n: int) -> Tuple[int, int]:
        """Other-robot nodes that could run *concurrently* with this robot's nodes ``m..n``.

        Returns the half-open interval ``[lo, hi)`` of the other robot's node indices that
        are neither predecessors of ``m`` nor successors of ``n`` in the graph. Everything
        outside it is ordered against the span, and ordered nodes can never be occupied at
        the same time -- so a motion replacing ``m..n`` only has to be collision-checked
        against this window. That is APEX-MR's shortcutting test (ADR-0008's addendum), and
        it is strictly weaker than "clear of the other robot's entire plan".

        Both bounds are PREFIX MAXIMA rather than lookups, which is what makes one stored
        entry per node sufficient here: a robot traverses its nodes monotonically, so a
        constraint attached to node ``q`` binds every node after ``q`` as well.
        """
        dep_r, dep_o = self.deps[robot], self.deps[self.other(robot)]
        if not 0 <= m <= n < dep_r.shape[0]:
            raise IndexError(f"{robot} has no node span {m}..{n}")
        # Predecessors: entering m required the other robot to have COMPLETED this node,
        # hence every node up to it as well.
        lo = int(np.maximum.accumulate(dep_r[:m + 1])[-1]) + 1
        # Successors: the first node the other robot cannot reach until this robot has
        # completed n or later; monotonicity carries it to every node after that.
        later = np.flatnonzero(np.maximum.accumulate(dep_o) >= n)
        hi = int(later[0]) if later.size else int(dep_o.shape[0])
        return max(lo, 0), max(hi, max(lo, 0))

    # -- serialisation: indices only, no geometry (ADR-0002) ------------------- #
    def to_json(self, path: str) -> None:
        obj = {
            "builder_version": self.builder_version,
            "precedence_edges_expected": self.precedence_edges_expected,
            "delta_t": self.delta_t,
            "robots": list(self.robots),
            "nominal_makespan_slots": self.nominal_makespan,
            "n_edges": self.n_edges,
            "n_precedence_edges": self.n_precedence_edges,
            "rigid_delay_margin_slots": self.delay_margin,
            "segments": {
                r: [{"task": s.task, "start_node": s.start_node, "n": s.n,
                     "start_slot": s.start_slot} for s in self.segments[r]]
                for r in self.robots
            },
            "deps": {r: self.deps[r].tolist() for r in self.robots},
        }
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")

    @staticmethod
    def from_json(path: str) -> "TPG":
        with open(path) as f:
            o = json.load(f)
        robots = (o["robots"][0], o["robots"][1])
        return TPG(
            robots=robots,
            segments={r: [Segment(s["task"], s["start_node"], s["n"], s["start_slot"])
                          for s in o["segments"][r]] for r in robots},
            deps={r: np.asarray(o["deps"][r], dtype=np.int64) for r in robots},
            delta_t=float(o["delta_t"]),
            nominal_makespan=int(o["nominal_makespan_slots"]),
            n_edges=int(o.get("n_edges", 0)),
            n_precedence_edges=int(o.get("n_precedence_edges", 0)),
            delay_margin=int(o.get("rigid_delay_margin_slots", -1)),
            precedence_edges_expected=o.get("precedence_edges_expected"),
            # Absent on graphs built before 2026-09-21: None, never the current stamp.
            builder_version=o.get("builder_version"),
        )


def check_fresh(tpg_json, problem: dict | None = None) -> None:
    """Refuse a plan graph that an older builder produced, or whose edges miss the seam's.

    ``tpg_json`` is a path to a ``tpg.json`` or its loaded dict. Raises ``ValueError`` when

    * ``builder_version`` is missing (every graph built before 2026-09-21 -- among them the
      archived ones built before slot precedences became edges) or older than
      :data:`BUILDER_VERSION`;
    * ``precedence_edges_expected`` is missing (built without the seam) or differs from
      ``n_precedence_edges``;
    * ``problem`` (the seam) is given and the precedence edges it calls for, recomputed
      here from the graph's scheduled tasks, differ from the stored count.

    Meant for a campaign driver, before any number is read off the graph.
    """
    if isinstance(tpg_json, (str, bytes)) or hasattr(tpg_json, "__fspath__"):
        with open(tpg_json) as f:
            o = json.load(f)
        where = str(tpg_json)
    else:
        o, where = tpg_json, "tpg"
    ver = o.get("builder_version")
    if ver is None:
        raise ValueError(f"{where}: no builder_version -- built before {BUILDER_VERSION} "
                         f"by a builder that may lack slot-precedence edges; rebuild it "
                         f"(build_tpg.py)")
    if str(ver) < BUILDER_VERSION:
        raise ValueError(f"{where}: builder_version {ver} is older than {BUILDER_VERSION}; "
                         f"rebuild it (build_tpg.py)")
    exp = o.get("precedence_edges_expected")
    if exp is None:
        raise ValueError(f"{where}: precedence_edges_expected is missing -- the graph was "
                         f"built without the seam, so its task precedences are not edges")
    got = int(o.get("n_precedence_edges", -1))
    if int(exp) != got:
        raise ValueError(f"{where}: {got} precedence edges, the seam calls for {exp}")
    if problem is not None:
        segs = {r: [Segment(s["task"], s["start_node"], s["n"], s["start_slot"])
                    for s in o["segments"][r]] for r in o["robots"]}
        again = expected_precedence_edges(problem, segs, o["robots"])
        if again != got:
            raise ValueError(f"{where}: {got} precedence edges, but this seam calls for "
                             f"{again} -- graph and seam are from different runs")


def timelines(solution: dict, robots: Sequence[str]) -> Dict[str, List[Segment]]:
    """Lay each robot's assigned tasks out in schedule order, back to back as nodes.

    Node indices are contiguous per robot: idle time between two tasks occupies no node,
    because under a TPG waiting is a runtime outcome rather than a planned quantity. The
    solver's ``start_slot`` is retained only to orient edges.

    Guards (2026-09-21), each a ``ValueError`` naming the tasks: an assignment to a robot
    not in ``robots``; a non-positive duration; and two tasks of ONE robot at the same
    ``start_slot`` or overlapping in time. A robot runs its tasks one after another, so
    either is a schedule no executor can follow -- and with equal starts the node order
    would silently follow the JSON key order. A robot with no task gets an empty timeline.
    """
    out: Dict[str, List[Segment]] = {r: [] for r in robots}
    for task, a in sorted(solution["assignments"].items(),
                          key=lambda kv: (kv[1]["start_slot"], kv[0])):
        if a["robot"] not in out:
            raise ValueError(f"task {task} is assigned to robot {a['robot']!r}, which is not "
                             f"one of {list(robots)}")
        if a["end_slot"] <= a["start_slot"]:
            raise ValueError(f"task {task} has end_slot {a['end_slot']} <= start_slot "
                             f"{a['start_slot']}")
        out[a["robot"]].append(Segment(task, 0, a["end_slot"] - a["start_slot"], a["start_slot"]))
    for r in robots:
        cursor = 0
        prev = None
        for seg in out[r]:
            if prev is not None:
                if seg.start_slot == prev.start_slot:
                    raise ValueError(
                        f"{r}: tasks {prev.task} and {seg.task} both start at slot "
                        f"{seg.start_slot}; one robot cannot run two tasks at once, and the "
                        f"node order between them would be arbitrary")
                if seg.start_slot < prev.start_slot + prev.n:
                    raise ValueError(
                        f"{r}: task {seg.task} starts at slot {seg.start_slot}, before "
                        f"{prev.task} ends at {prev.start_slot + prev.n}")
            seg.start_node = cursor
            cursor += seg.n
            prev = seg
    return out


def build(
    solution: dict,
    robots: Sequence[str],
    mu_of: "callable[[str, str, str, str], np.ndarray]",
    delta_t: float,
    problem: dict | None = None,
) -> TPG:
    """Construct the TPG from the solved schedule and the collision matrices.

    ``mu_of(r, task_i, s, task_j) -> (K_i, K_j) bool`` supplies ``mu`` for a scheduled
    pair; it is a callback so this module stays free of any geometry import (ADR-0002).

    ``problem`` is the seam artifact. It is optional only so the geometric core can be
    exercised alone; a scene WITH precedences and no ``problem`` would silently drop them,
    so passing it is required whenever the seam declares any.
    """
    if len(robots) != 2:
        raise ValueError("the TPG construction here assumes exactly two robots")
    r, s = robots[0], robots[1]
    segs = timelines(solution, robots)

    deps = {q: np.full(sum(g.n for g in segs[q]), FREE, dtype=np.int64) for q in robots}
    margin = None

    for gi in segs[r]:
        for gj in segs[s]:
            mu = np.asarray(mu_of(r, gi.task, s, gj.task), dtype=bool)
            if mu.shape != (gi.n, gj.n):
                raise ValueError(
                    f"mu for {gi.task} x {gj.task} is {mu.shape}, expected {(gi.n, gj.n)}")

            # Nominal instants: node k of gi happens at gi.start_slot + k. So node k of r
            # is earlier than node l of s exactly when k - l < delta.
            delta = gj.start_slot - gi.start_slot
            k = np.arange(gi.n)[:, None]
            l = np.arange(gj.n)[None, :]

            # A collision at EQUAL nominal time would mean the realised offset is in D --
            # the solver contradicting itself. Check before relying on the orientation.
            simultaneous = mu & (k - l == delta)
            if simultaneous.any():
                bad = np.argwhere(simultaneous)[:5].tolist()
                raise AssertionError(
                    f"schedule is self-inconsistent: {gi.task} x {gj.task} collide at the "
                    f"scheduled offset {delta} (sample pairs {bad}) -- the solver returned "
                    f"a schedule violating its own forbidden offsets")

            # How far this pair's realised offset sits from the nearest colliding one --
            # the delay the rigid schedule could have absorbed here.
            ks, ls = np.nonzero(mu)
            if ks.size:
                gap = int(np.abs(np.unique(ks - ls) - delta).min())
                margin = gap if margin is None else min(margin, gap)

            # s waits for r: for each l, the latest nominally-earlier colliding k.
            _tighten(deps[s], gj.start_node, mu & (k - l < delta), axis=0, base=gi.start_node)
            # r waits for s: for each k, the latest nominally-earlier colliding l.
            _tighten(deps[r], gi.start_node, mu & (k - l > delta), axis=1, base=gj.start_node)

    n_prec = precedence_edges(deps, segs, robots, problem) if problem is not None else 0
    expected = expected_precedence_edges(problem, segs, robots) if problem is not None else None
    n_edges = int(sum(int((deps[q] != FREE).sum()) for q in robots))
    # `default=0`: a schedule may leave one robot, or both, without a task (an allocation
    # objective with no balancing term can). Both empty is the empty graph, makespan 0.
    makespan = max((g.start_slot + g.n for q in robots for g in segs[q]), default=0)
    tpg = TPG(robots=(r, s), segments=segs, deps=deps, delta_t=delta_t,
              nominal_makespan=makespan, n_edges=n_edges, n_precedence_edges=n_prec,
              delay_margin=-1 if margin is None else margin,
              precedence_edges_expected=expected)
    assert_acyclic(tpg)
    return tpg


def _tighten(dep: np.ndarray, node_offset: int, valid: np.ndarray, axis: int, base: int) -> None:
    """Fold ``max{index along axis where valid} + 1`` into ``dep``, in place.

    ``valid`` is the collision matrix already masked down to the pairs whose dependency
    runs in this direction. Taking the maximum is what makes one edge per node sufficient;
    the ``+ 1`` is what makes it safe (see :class:`TPG`).
    """
    any_hit = valid.any(axis=axis)
    if not any_hit.any():
        return
    # argmax over the reversed axis gives the LAST True.
    n_along = valid.shape[axis]
    rev = valid[::-1, :] if axis == 0 else valid[:, ::-1]
    last = n_along - 1 - np.argmax(rev, axis=axis)

    target = np.arange(dep.shape[0])[node_offset:node_offset + any_hit.shape[0]]
    cand = np.where(any_hit, base + last + 1, FREE)
    np.maximum.at(dep, target, cand)


def expand_precedences(problem: dict, scheduled: Iterable[str]) -> List[Tuple[str, str, str]]:
    """The seam's task-ordering constraints as ``[(i, j, mode), ...]`` over SCHEDULED tasks.

    One place for the precedence semantics, shared by the graph builder
    (:func:`precedence_edges`), the simulator's ground-truth order check and the
    coordination diagram, so none of them can go blind to a kind of precedence the others
    see.

    Two sources, in this order:

    * ``precedences`` (+ ``precedence_modes``, absent means all ``pipeline``): pairs of
      TASKS, taken as they are.
    * ``slot_precedences`` (+ ``slot_precedence_modes``, same default): pairs of SLOTS --
      "whichever candidate wins slot ``a`` runs before whichever wins slot ``b``"
      (``model.SchedulingProblem.slot_precedences``). A scene with interchangeable slots
      carries its whole build order there and has ``precedences == []``. Once the schedule
      has chosen the winners each slot pair is an ordinary task pair, with the same mode,
      and goes through exactly the same edge construction. Without this translation such a
      scene's graph has no ordering edge at all (found 2026-09-19 on ``tower_ic_A/B``,
      ``tower_wall_A/B``: ``n_precedence_edges == 0``).

    ``scheduled`` is the set of task ids the schedule assigns (a solution's
    ``assignments`` keys, or the graph's segments). ``slot_of`` maps task -> slot id; when
    absent, task ids double as slot ids (the solver's own convention). Every slot must have
    EXACTLY one scheduled member -- the solver's exactly-one-per-slot constraint, which a
    solution violating it would be silently mis-ordered under -- else ``ValueError``.
    """
    precedences = problem.get("precedences", [])
    modes = problem.get("precedence_modes") or ["pipeline"] * len(precedences)
    if len(modes) != len(precedences):
        raise ValueError(f"{len(modes)} precedence_modes for {len(precedences)} precedences")
    out = [(i, j, m) for (i, j), m in zip(precedences, modes)]

    slot_prec = problem.get("slot_precedences", [])
    if not slot_prec:
        return out
    smodes = problem.get("slot_precedence_modes") or ["pipeline"] * len(slot_prec)
    if len(smodes) != len(slot_prec):
        raise ValueError(f"{len(smodes)} slot_precedence_modes for {len(slot_prec)} "
                         f"slot_precedences")
    scheduled = set(scheduled)
    slot_of = problem.get("slot_of") or {t: t for t in problem.get("tasks", scheduled)}
    members: Dict[object, List[str]] = {}
    for task, slot in slot_of.items():
        members.setdefault(slot, []).append(task)
    winner: Dict[object, str] = {}
    for slot, tasks in members.items():
        won = sorted(t for t in tasks if t in scheduled)
        if len(won) != 1:
            raise ValueError(
                f"slot {slot!r} has {len(won)} scheduled task(s) {won} out of its "
                f"{len(tasks)} candidate(s); the slot precedences need exactly one winner "
                f"per slot (the schedule and the seam are from different runs, or the "
                f"schedule is not a solution of this seam)")
        winner[slot] = won[0]
    for (a, b), mode in zip(slot_prec, smodes):
        for slot in (a, b):
            if slot not in winner:
                raise KeyError(f"slot precedence ({a!r}, {b!r}) names slot {slot!r}, which "
                               f"slot_of does not define")
        out.append((winner[a], winner[b], mode))
    return out


def expected_precedence_edges(
    problem: dict,
    segs: Dict[str, List[Segment]],
    robots: Sequence[str],
) -> int:
    """How many precedence edges the seam calls for, counted without building any.

    One per cross-robot milestone condition of the scheduled tasks: two per ``pipeline``
    pair (de-stack gate, re-stack order), one per ``gate`` pair; a pair on one robot adds
    none (its node order enforces it). This is what :func:`precedence_edges` must have
    added, so a graph whose ``n_precedence_edges`` differs was built by a builder that
    dropped some -- the 2026-09-19 bug, where every slot scene had 0.
    """
    robot_of = {g.task: q for q in robots for g in segs[q]}
    n = 0
    for i, j, mode in expand_precedences(problem, robot_of):
        if i not in robot_of or j not in robot_of:
            raise KeyError(f"precedence ({i}, {j}) names a task the schedule never assigns")
        if robot_of[i] != robot_of[j]:
            n += 2 if mode == "pipeline" else 1
    return n


def precedence_edges(
    deps: Dict[str, np.ndarray],
    segs: Dict[str, List[Segment]],
    robots: Sequence[str],
    problem: dict,
) -> int:
    """Add the solver's task-ordering constraints to the graph. Returns how many.

    Geometry is not the only thing that orders two robots. A tower must be taken apart
    top-down and rebuilt bottom-up, and the scheduler encodes that as two conditions per
    precedence ``(i, j)``:

    * ``start[j] >= pick[i]`` -- the de-stack gate. Task ``j`` may not begin until ``i``
      has been *lifted off* the stack, because until then ``j``'s box is buried.
    * ``place[i] <= place[j]`` -- the re-stack order. At the destination ``i`` must land
      before ``j`` does, or ``j`` is placed into thin air.

    The scheduler enforced both with *arithmetic on start slots*. That is exactly the
    contract the TPG dissolves, so without this function a delayed run can satisfy every
    collision edge and still execute the task plan out of order -- opening a gripper over
    a box that is not there yet. Nothing geometric would flag it: the failure is that the
    arms are in the RIGHT places at the WRONG times, which is precisely what a
    collision-only check cannot see.

    Within one robot both conditions hold by construction (nodes advance monotonically and
    the solver ordered the tasks), and that is asserted rather than assumed.

    Since 2026-09-13 a precedence also carries a MODE, from the seam's optional
    ``precedence_modes`` (parallel to ``precedences``; absent means every pair is
    ``pipeline``, which is what every seam before then was). Since 2026-09-19 the seam's
    ``slot_precedences`` are resolved to their scheduled winners and added here too
    (:func:`expand_precedences`):

    * ``pipeline`` -- the two conditions above.
    * ``gate`` -- ``pick[j] >= place[i]``: ``j`` may not reach its ACQUIRE milestone
      (a weld striking its arc) until ``i`` has reached its RELEASE milestone (the part
      has been let go). It does not imply the pipeline pair; a scene that needs both
      lists the pair twice, once per mode, and each entry adds its own edges.
    """
    # .get: a schedule with no task at all needs no milestone (the empty-robot guard).
    pick, place = problem.get("pick_offsets", {}), problem.get("place_offsets", {})
    node = {}
    for q in robots:
        for g in segs[q]:
            node[g.task] = (q, g.start_node,
                            g.start_node + int(pick[f"{q}|{g.task}"]),
                            g.start_node + int(place[f"{q}|{g.task}"]))

    n = 0
    for i, j, mode in expand_precedences(problem, node):
        if i not in node or j not in node:
            raise KeyError(f"precedence ({i}, {j}) names a task the schedule never assigns")
        (ri, _, pick_i, place_i), (rj, start_j, pick_j, place_j) = node[i], node[j]
        if mode == "pipeline":
            pairs = ((pick_i, start_j), (place_i, place_j))
        elif mode == "gate":
            pairs = ((place_i, pick_j),)
        else:
            raise ValueError(f"precedence ({i}, {j}) has unknown mode {mode!r}")
        for milestone, waiter in pairs:
            if ri == rj:
                if milestone > waiter:
                    raise AssertionError(
                        f"schedule violates {mode} precedence ({i}, {j}) on {ri}: node "
                        f"{milestone} must come before node {waiter} but does not")
                continue      # same robot: its own node order already enforces it
            # "has REACHED", not "has left" -- the solver's conditions permit equality.
            deps[rj][waiter] = max(int(deps[rj][waiter]), milestone)
            n += 1
    return n


def zero_delay_ticks(tpg: TPG) -> int:
    """Slots the graph takes to execute with no stall at all: the TPG's own makespan.

    Same rule as ``simulate_tpg.run_tpg`` -- each tick every robot (in ``tpg.robots``
    order) advances one node iff its incoming edge is satisfied by what the other has
    reached, the second robot seeing the first's move of the same tick -- so the two agree
    tick for tick. It is NOT ``tpg.nominal_makespan``: that is the SOLVER's makespan, which
    keeps the idle gaps between tasks (a turn-based schedule starts each task only when the
    previous one has ended, and the graph does not wait for a clock, only for its edges).
    Raises ``RuntimeError`` on a deadlock, which :func:`assert_acyclic` should make
    impossible.
    """
    r, s = tpg.robots
    reached = {r: -1, s: -1}
    n = {q: tpg.n_nodes(q) for q in (r, s)}
    ticks = 0
    while reached[r] < n[r] - 1 or reached[s] < n[s] - 1:
        ticks += 1
        moved = False
        for q in (r, s):
            nxt = reached[q] + 1
            if nxt >= n[q]:
                continue
            d = int(tpg.deps[q][nxt])
            if d == FREE or reached[tpg.other(q)] >= d:
                reached[q] = nxt
                moved = True
        if not moved:
            raise RuntimeError(f"TPG deadlocks at tick {ticks}: {reached}")
    return ticks


def assert_acyclic(tpg: TPG) -> None:
    """Deadlock-freedom check.

    Two ways a TPG can deadlock, both checked here:

    * **A cycle.** Orienting every edge by nominal time already precludes one, so this
      re-derives each edge's nominal instants and confirms the dependency does not
      arrive after the node that waits on it.
    * **An unreachable dependency.** A node whose ``dep`` is the other robot's node count
      can never be satisfied -- that robot stops at its last node and never "reaches" the
      one past it. This arises only if a robot's FINAL configuration collides with
      something, i.e. if HOME is blocking; the cell is verified otherwise, but relying on
      that silently would turn a modelling assumption into a hang.
    """
    for q in tpg.robots:
        o = tpg.other(q)
        dep, n_other = tpg.deps[q], tpg.n_nodes(o)
        for node in np.flatnonzero(dep != FREE):
            d = int(dep[node])
            if d >= n_other:
                task, k = tpg.locate(q, int(node))
                raise AssertionError(
                    f"unsatisfiable dependency: {q}#{node} ({task}[{k}]) waits for {o} to "
                    f"reach node {d}, but {o} has only {n_other} nodes. Its final "
                    f"configuration collides -- HOME is not a non-blocking pose here.")
            if _instant(tpg, o, d) > _instant(tpg, q, int(node)):
                raise AssertionError(
                    f"TPG edge {o}#{d} -> {q}#{node} runs backwards in nominal time; "
                    f"the graph may contain a cycle and could deadlock")
    _assert_no_cycle(tpg)


def _assert_no_cycle(tpg: TPG) -> None:
    """Exact cycle check, O(nodes): the nominal-time test above is necessary, not sufficient.

    An edge's source is never nominally LATER than its target, but it can be nominally
    SIMULTANEOUS, and two such edges can close a cycle (found 2026-09-21 by
    ``test_tpg.py`` on synthetic instances). The geometric case: with the scheduled offset
    ``delta`` collision-free but both ``delta - 1`` and ``delta + 1`` colliding -- a
    one-slot hole in the forbidden set, which the solver may legally pick -- nodes
    ``(n, m)`` and ``(n + 1, m - 1)`` both collide, so ``r`` may not enter ``n + 1``
    until ``s`` has left ``m - 1`` and ``s`` may not enter ``m`` until ``r`` has left
    ``n``. The rigid schedule passes through that crossing in lock-step; a graph, which
    only knows "after", cannot, and would deadlock on it at any delay.

    With one incoming edge per node plus each robot's own chain, a greedy sweep that
    advances either robot whenever its next node's edge is satisfied visits every node iff
    the graph is acyclic (if both robots are stuck, each one's next node needs a node of
    the other at or beyond that robot's next node: a cycle).
    """
    r, s = tpg.robots
    n = {q: tpg.n_nodes(q) for q in (r, s)}
    reached = {r: -1, s: -1}
    while True:
        moved = False
        for q in (r, s):
            o = tpg.other(q)
            while reached[q] + 1 < n[q]:
                d = int(tpg.deps[q][reached[q] + 1])
                if d != FREE and reached[o] < d:
                    break
                reached[q] += 1
                moved = True
        if reached[r] == n[r] - 1 and reached[s] == n[s] - 1:
            return
        if not moved:
            a, b = reached[r] + 1, reached[s] + 1
            ta, ka = tpg.locate(r, a)
            tb, kb = tpg.locate(s, b)
            raise AssertionError(
                f"TPG has a cycle: {r}#{a} ({ta}[{ka}]) waits for {s} to reach "
                f"{int(tpg.deps[r][a])} and {s}#{b} ({tb}[{kb}]) waits for {r} to reach "
                f"{int(tpg.deps[s][b])} -- nominally simultaneous edges closing a loop "
                f"(e.g. a one-slot hole in the forbidden offsets at the scheduled offset). "
                f"The graph would deadlock at any delay.")


def _instant(tpg: TPG, robot: str, node: int) -> int:
    for seg in tpg.segments[robot]:
        if seg.start_node <= node < seg.start_node + seg.n:
            return seg.start_slot + (node - seg.start_node)
    raise IndexError(f"{robot} has no node {node}")
