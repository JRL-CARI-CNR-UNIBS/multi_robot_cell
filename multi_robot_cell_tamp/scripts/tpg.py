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
from typing import Dict, List, Sequence, Tuple

import numpy as np

# A node with no cross-robot dependency. Chosen as -1 so ``done >= dep`` is trivially true
# at start of execution, where ``done`` is "index of last completed node" and begins at -1.
FREE = -1


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
        )


def timelines(solution: dict, robots: Sequence[str]) -> Dict[str, List[Segment]]:
    """Lay each robot's assigned tasks out in schedule order, back to back as nodes.

    Node indices are contiguous per robot: idle time between two tasks occupies no node,
    because under a TPG waiting is a runtime outcome rather than a planned quantity. The
    solver's ``start_slot`` is retained only to orient edges.
    """
    out: Dict[str, List[Segment]] = {r: [] for r in robots}
    for task, a in sorted(solution["assignments"].items(), key=lambda kv: kv[1]["start_slot"]):
        out[a["robot"]].append(Segment(task, 0, a["end_slot"] - a["start_slot"], a["start_slot"]))
    for r in robots:
        cursor = 0
        for seg in out[r]:
            seg.start_node = cursor
            cursor += seg.n
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
    n_edges = int(sum(int((deps[q] != FREE).sum()) for q in robots))
    makespan = max(g.start_slot + g.n for q in robots for g in segs[q])
    tpg = TPG(robots=(r, s), segments=segs, deps=deps, delta_t=delta_t,
              nominal_makespan=makespan, n_edges=n_edges, n_precedence_edges=n_prec,
              delay_margin=-1 if margin is None else margin)
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
    """
    pick, place = problem["pick_offsets"], problem["place_offsets"]
    node = {}
    for q in robots:
        for g in segs[q]:
            node[g.task] = (q, g.start_node,
                            g.start_node + int(pick[f"{q}|{g.task}"]),
                            g.start_node + int(place[f"{q}|{g.task}"]))

    n = 0
    for i, j in problem.get("precedences", []):
        if i not in node or j not in node:
            raise KeyError(f"precedence ({i}, {j}) names a task the schedule never assigns")
        (ri, _, pick_i, place_i), (rj, start_j, _, place_j) = node[i], node[j]
        for milestone, waiter in ((pick_i, start_j), (place_i, place_j)):
            if ri == rj:
                if milestone > waiter:
                    raise AssertionError(
                        f"schedule violates precedence ({i}, {j}) on {ri}: node {milestone} "
                        f"must come before node {waiter} but does not")
                continue      # same robot: its own node order already enforces it
            # "has REACHED", not "has left" -- the solver's conditions permit equality.
            deps[rj][waiter] = max(int(deps[rj][waiter]), milestone)
            n += 1
    return n


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


def _instant(tpg: TPG, robot: str, node: int) -> int:
    for seg in tpg.segments[robot]:
        if seg.start_node <= node < seg.start_node + seg.n:
            return seg.start_slot + (node - seg.start_node)
    raise IndexError(f"{robot} has no node {node}")
