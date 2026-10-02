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

N ROBOTS (2026-09-26)
---------------------
Nothing above needs two robots: every argument is about ONE ordered pair of robots. With
``N`` robots the graph keeps one incoming edge per node **per other robot** --
``deps_on[r][s][n]``, the node robot ``s`` must have reached before robot ``r`` may enter
its node ``n`` -- and a node is ready when every other robot satisfies its entry. Each pair
``(r, s)`` gets exactly the two-robot construction (collision edges from ``mu`` of the
pair, precedence edges from the seam), so the safety argument holds pair by pair. What does
NOT carry over pairwise is deadlock-freedom: a cycle can now run through three robots
along nominally simultaneous edges, so the exact cycle check is Kahn's algorithm over the
``N`` chains plus every cross edge (:func:`_assert_no_cycle`), not a two-chain argument.
For ``N = 2`` the graph, and its ``tpg.json``, are byte-identical to the two-robot builder's;
:attr:`TPG.deps` and :meth:`TPG.other` remain as the two-robot view and raise for ``N > 2``
(ADR-0008 refinement and ``coordinate.py`` are two-robot by construction).

WHAT IS ASSUMED
---------------
That a robot resting at HOME blocks nobody. Every trajectory begins and ends at HOME and
the model never constrains an *idle* robot, so this must hold independently. It does, BY
CONSTRUCTION of the trajectory stage (ADR-0003): every trajectory is planned and every one
of its samples validated with every other robot parked at HOME, on the exact geometry
(MoveIt/FCL, collision meshes, the carried object attached). ``mu`` is a conservative sphere
model and may still flag a HOME sample (a carried plate is ONE circumscribing sphere; each
sphere carries a 2 cm margin): on fabricator4 it does, where the exact replay
(``plan_oracle.py``) finds no robot-robot contact at all. So when the caller certifies which
trajectory ends ARE the robot's HOME (``rest_home``, from the artifact's ``homes``), those
samples are dropped from ``mu`` before any edge is built (:func:`mask_home_rest`) and only
counted. A resting pose that is NOT certified HOME (a refined chain's parking pose) is still
checked against ``mu`` and refused by name (:func:`_refuse_blocking_rest`).
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
    """The robots' timelines plus the cross-robot precedences between every pair of them.

    ``deps_on[r][s][n]`` is the node robot ``s`` must have **reached** before robot ``r``
    may enter its node ``n`` (``FREE`` when unconstrained), i.e. execution requires
    ``reached[s] >= deps_on[r][s][n]`` for EVERY other robot ``s``. One entry per node and
    per other robot is sufficient -- see the module docstring. ``deps_on[r]`` has one array
    per other robot, in ``robots`` order.

    For a COLLISION edge the stored index is the colliding node's SUCCESSOR, not the
    colliding node itself. That +1 is the whole safety argument: requiring the other robot
    merely to have *reached* the colliding configuration would leave it sitting exactly
    there, which is the collision we are trying to forbid. Requiring it to have reached the
    next one means it has LEFT the colliding configuration.

    A PRECEDENCE edge stores the milestone node itself, with no +1, because the solver's
    constraints are non-strict (``start[j] >= pick[i]`` permits equality). The two
    conventions differ because the underlying conditions differ: "has left" for geometry,
    "has reached" for task order.

    With two robots, :attr:`deps` (``deps[r][n]``, the entry against "the other" robot)
    and :meth:`other` are the historical view of the same arrays; both raise for ``N > 2``.
    """

    robots: Tuple[str, ...]
    segments: Dict[str, List[Segment]]
    deps_on: Dict[str, Dict[str, np.ndarray]]
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
    # With N robots: the minimum over every robot pair.
    delay_margin: int = field(default=-1)
    # How many precedence edges the SEAM calls for (cross-robot milestone conditions of
    # the scheduled tasks, :func:`expected_precedence_edges`), computed without looking at
    # the edges. ``None`` when built without the seam. :func:`check_fresh` compares it with
    # ``n_precedence_edges``.
    precedence_edges_expected: "int | None" = field(default=None)
    builder_version: "str | None" = field(default=BUILDER_VERSION)
    # Colliding mu cells dropped because they involve a certified HOME sample (build's
    # `rest_home`). Reported by build_tpg; not part of tpg.json.
    home_rest_masked: int = field(default=0)

    # -- robots ---------------------------------------------------------------- #
    def others(self, robot: str) -> Tuple[str, ...]:
        """Every robot but ``robot``, in ``robots`` order."""
        if robot not in self.robots:
            raise KeyError(f"{robot!r} is not one of {list(self.robots)}")
        return tuple(q for q in self.robots if q != robot)

    def other(self, robot: str) -> str:
        """THE other robot -- defined for a two-robot graph only."""
        if len(self.robots) != 2:
            raise ValueError(f"other() is defined for two robots, this graph has "
                             f"{len(self.robots)} ({list(self.robots)}); use others() / "
                             f"deps_on[r][s]")
        return self.robots[1] if robot == self.robots[0] else self.robots[0]

    @property
    def deps(self) -> Dict[str, np.ndarray]:
        """Two-robot view: ``deps[r]`` is ``deps_on[r][other(r)]`` (the same array, not a
        copy). Raises for a graph of more than two robots."""
        return {q: self.deps_on[q][self.other(q)] for q in self.robots}

    def n_nodes(self, robot: str) -> int:
        return int(sum(g.n for g in self.segments[robot]))

    def count_edges(self) -> int:
        """Stored cross-robot entries, i.e. what ``n_edges`` records."""
        return int(sum(int((d != FREE).sum()) for q in self.robots
                       for d in self.deps_on[q].values()))

    # -- execution rule, shared by every executor of the graph ----------------- #
    def blockers(self, robot: str, node: int, reached: Dict[str, int]) -> List[Tuple[str, int]]:
        """``[(s, dep), ...]``: the other robots that keep ``robot`` out of ``node`` now."""
        out = []
        for s, dep in self.deps_on[robot].items():
            d = int(dep[node])
            if d != FREE and reached[s] < d:
                out.append((s, d))
        return out

    def ready(self, robot: str, node: int, reached: Dict[str, int]) -> bool:
        """May ``robot`` enter ``node`` given what every robot has ``reached``?"""
        for s, dep in self.deps_on[robot].items():
            d = int(dep[node])
            if d != FREE and reached[s] < d:
                return False
        return True

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

        Two robots only (ADR-0008 refinement is two-robot): with a third robot, a
        predecessor can also be reached THROUGH it, which prefix maxima of one pair miss.
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
        """Write ``tpg.json``.

        ``deps`` is ``{r: [..]}`` (the entry against THE other robot) for two robots --
        the historical format, byte for byte -- and ``{r: {s: [..] for every s != r}}``
        for any other number of robots. Nothing else differs.
        """
        if len(self.robots) == 2:
            deps = {r: self.deps_on[r][self.other(r)].tolist() for r in self.robots}
        else:
            deps = {r: {s: self.deps_on[r][s].tolist() for s in self.others(r)}
                    for r in self.robots}
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
            "deps": deps,
        }
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")

    @staticmethod
    def from_json(path: str) -> "TPG":
        with open(path) as f:
            o = json.load(f)
        robots = tuple(o["robots"])
        segments = {r: [Segment(s["task"], s["start_node"], s["n"], s["start_slot"])
                        for s in o["segments"][r]] for r in robots}
        return TPG(
            robots=robots,
            segments=segments,
            deps_on=deps_from_json(o, where=str(path)),
            delta_t=float(o["delta_t"]),
            nominal_makespan=int(o["nominal_makespan_slots"]),
            n_edges=int(o.get("n_edges", 0)),
            n_precedence_edges=int(o.get("n_precedence_edges", 0)),
            delay_margin=int(o.get("rigid_delay_margin_slots", -1)),
            precedence_edges_expected=o.get("precedence_edges_expected"),
            # Absent on graphs built before 2026-09-21: None, never the current stamp.
            builder_version=o.get("builder_version"),
        )


def deps_from_json(o: dict, where: str = "tpg") -> Dict[str, Dict[str, np.ndarray]]:
    """A loaded ``tpg.json`` -> ``deps_on[r][s]`` arrays, whichever of the two formats.

    The two-robot file stores ``deps[r]`` as a flat list against the other robot; an
    ``N``-robot file stores ``deps[r][s]``. Each array must have one entry per node of
    ``r`` (the sum of its segments), else ``ValueError``.
    """
    robots = list(o["robots"])
    if len(set(robots)) != len(robots):
        raise ValueError(f"{where}: repeated robot in {robots}")
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for r in robots:
        n = sum(int(g["n"]) for g in o["segments"][r])
        raw = o["deps"][r]
        if isinstance(raw, dict):
            missing = [s for s in robots if s != r and s not in raw]
            if missing:
                raise ValueError(f"{where}: deps[{r}] has no entry for {missing}")
            out[r] = {s: np.asarray(raw[s], dtype=np.int64) for s in robots if s != r}
        else:
            if len(robots) != 2:
                raise ValueError(f"{where}: deps[{r}] is a flat list, the two-robot format, "
                                 f"but the graph has {len(robots)} robots")
            out[r] = {next(s for s in robots if s != r): np.asarray(raw, dtype=np.int64)}
        for s, d in out[r].items():
            if d.shape != (n,):
                raise ValueError(f"{where}: deps[{r}][{s}] has shape {d.shape}, but {r} has "
                                 f"{n} nodes")
    return out


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


def home_rest_ends(art: dict, tol: float = 1e-9) -> Dict[Tuple[str, str], Tuple[bool, bool]]:
    """For every trajectory of a trajectory artifact: (first sample is HOME, last sample is
    HOME), from the artifact's own ``homes`` -- the certification :func:`build` and
    ``simulate_tpg`` use to drop HOME samples from ``mu`` (module docstring, WHAT IS ASSUMED)."""
    homes = {r: np.asarray(v, dtype=float) for r, v in (art.get("homes") or {}).items()}
    out = {}
    for t in art["trajectories"]:
        h = homes.get(t["robot"])
        if h is None or not t["positions"]:
            out[(t["robot"], t["task"])] = (False, False)
            continue
        first = np.asarray(t["positions"][0], dtype=float)
        last = np.asarray(t["positions"][-1], dtype=float)
        out[(t["robot"], t["task"])] = (bool(np.abs(first - h).max() <= tol),
                                        bool(np.abs(last - h).max() <= tol))
    return out


def mask_home_rest(mu: np.ndarray, ends_i: Tuple[bool, bool],
                   ends_j: Tuple[bool, bool]) -> Tuple[np.ndarray, int]:
    """``mu`` (K_i, K_j) with the certified HOME rows / columns cleared, and how many
    colliding cells that removed. Untouched (same object, 0) when nothing is certified or
    nothing collides there -- so a plan without such hits is built byte for byte as before."""
    rows = [0] * ends_i[0] + [mu.shape[0] - 1] * ends_i[1]
    cols = [0] * ends_j[0] + [mu.shape[1] - 1] * ends_j[1]
    if not (rows or cols):
        return mu, 0
    hit = np.zeros_like(mu, dtype=bool)
    if rows:
        hit[rows, :] = True
    if cols:
        hit[:, cols] = True
    n = int(np.count_nonzero(mu & hit))
    if n == 0:
        return mu, 0
    out = np.array(mu, dtype=bool, copy=True)
    out[hit] = False
    return out, n


def build(
    solution: dict,
    robots: Sequence[str],
    mu_of: "callable[[str, str, str, str], np.ndarray]",
    delta_t: float,
    problem: dict | None = None,
    rest_home: Dict[Tuple[str, str], Tuple[bool, bool]] | None = None,
) -> TPG:
    """Construct the TPG from the solved schedule and the collision matrices.

    ``mu_of(r, task_i, s, task_j) -> (K_i, K_j) bool`` supplies ``mu`` for a scheduled
    pair; it is a callback so this module stays free of any geometry import (ADR-0002).
    It is called for every robot pair with ``r`` BEFORE ``s`` in ``robots`` order (for two
    robots: always ``robots[0], robots[1]``, as before), so a caller that packs sides per
    pair (the SIMD kernel's A/B) knows which side each robot is on.

    ``problem`` is the seam artifact. It is optional only so the geometric core can be
    exercised alone; a scene WITH precedences and no ``problem`` would silently drop them,
    so passing it is required whenever the seam declares any.
    """
    robots = tuple(robots)
    if not robots or len(set(robots)) != len(robots):
        raise ValueError(f"the TPG needs distinct robot names, got {list(robots)}")
    segs = timelines(solution, robots)

    n_of = {q: sum(g.n for g in segs[q]) for q in robots}
    deps_on = {q: {o: np.full(n_of[q], FREE, dtype=np.int64) for o in robots if o != q}
               for q in robots}
    margin = None
    resting: List[Tuple[str, str, str, str, int, int, int]] = []
    home_masked = [0]

    for a, r in enumerate(robots):
        for s in robots[a + 1:]:
            margin = _pair_edges(segs, r, s, mu_of, deps_on, margin, resting,
                                 rest_home, home_masked)
    if resting:
        _refuse_blocking_rest(resting)

    n_prec = precedence_edges(deps_on, segs, robots, problem) if problem is not None else 0
    expected = expected_precedence_edges(problem, segs, robots) if problem is not None else None
    # `default=0`: a schedule may leave a robot, or all of them, without a task (an
    # allocation objective with no balancing term can). All empty is the empty graph,
    # makespan 0.
    makespan = max((g.start_slot + g.n for q in robots for g in segs[q]), default=0)
    tpg = TPG(robots=robots, segments=segs, deps_on=deps_on, delta_t=delta_t,
              nominal_makespan=makespan, n_edges=0, n_precedence_edges=n_prec,
              delay_margin=-1 if margin is None else margin,
              precedence_edges_expected=expected)
    tpg.n_edges = tpg.count_edges()
    tpg.home_rest_masked = home_masked[0]
    assert_acyclic(tpg)
    return tpg


def _pair_edges(segs: Dict[str, List[Segment]], r: str, s: str, mu_of,
                deps_on: Dict[str, Dict[str, np.ndarray]], margin: "int | None",
                resting: "list | None" = None,
                rest_home: "Dict[Tuple[str, str], Tuple[bool, bool]] | None" = None,
                home_masked: "list | None" = None) -> "int | None":
    """The collision edges of ONE robot pair, both directions, folded into ``deps_on``.

    Exactly the two-robot construction: ``s`` waits for ``r`` where ``r``'s colliding
    configuration comes first, and vice versa. Returns the running rigid delay margin.

    A trajectory's first and last samples are where its robot RESTS -- HOME (ADR-0004), or
    a refined chain's parking pose, which ADR-0008 already makes clear of the other
    robot's whole plan. Any colliding pair involving one is appended to ``resting`` as
    ``(resting robot, its task, "first"/"last", other robot, other task, first and last
    colliding sample of the other task, count)``; see :func:`_refuse_blocking_rest`.
    """
    for gi in segs[r]:
        for gj in segs[s]:
            mu = np.asarray(mu_of(r, gi.task, s, gj.task), dtype=bool)
            if mu.shape != (gi.n, gj.n):
                raise ValueError(
                    f"mu for {gi.task} x {gj.task} is {mu.shape}, expected {(gi.n, gj.n)}")
            if rest_home is not None:
                # Certified HOME samples: clear by construction of the trajectory stage
                # (module docstring, WHAT IS ASSUMED) -- dropped, counted, never an edge.
                mu, n_home = mask_home_rest(mu, rest_home.get((r, gi.task), (False, False)),
                                            rest_home.get((s, gj.task), (False, False)))
                if home_masked is not None:
                    home_masked[0] += n_home

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

            if resting is not None:
                for who, task, other, otask, rows in (
                        (r, gi.task, s, gj.task, (("first", mu[0, :]), ("last", mu[-1, :]))),
                        (s, gj.task, r, gi.task, (("first", mu[:, 0]), ("last", mu[:, -1])))):
                    for end, hit in rows:
                        idx = np.flatnonzero(hit)
                        if idx.size:
                            resting.append((who, task, end, other, otask, int(idx[0]),
                                            int(idx[-1]), int(idx.size)))

            # How far this pair's realised offset sits from the nearest colliding one --
            # the delay the rigid schedule could have absorbed here.
            ks, ls = np.nonzero(mu)
            if ks.size:
                gap = int(np.abs(np.unique(ks - ls) - delta).min())
                margin = gap if margin is None else min(margin, gap)

            # s waits for r: for each l, the latest nominally-earlier colliding k.
            _tighten(deps_on[s][r], gj.start_node, mu & (k - l < delta), axis=0,
                     base=gi.start_node)
            # r waits for s: for each k, the latest nominally-earlier colliding l.
            _tighten(deps_on[r][s], gi.start_node, mu & (k - l > delta), axis=1,
                     base=gj.start_node)
    return margin


def _refuse_blocking_rest(resting: list) -> None:
    """Refuse a plan in which a RESTING robot collides with another robot's motion.

    The graph orders nodes, and a robot that is idle -- before its first task, between
    two tasks, after its last -- occupies no node of its own: it sits at the last sample
    of the task it finished (or at HOME before any). ``mu`` against that pose therefore
    produces edges that cannot be honoured: waiting for the resting robot to "leave" its
    last node means waiting for the first node of its NEXT task, nominally later (the
    "runs backwards in nominal time" error), or past its node count (the "unsatisfiable
    dependency" error); a robot at HOME before its first task cannot be waited for at
    all, and nothing is raised -- the graph is silently unsafe. The rigid schedule is no
    better: the solver's forbidden offsets relate two tasks' samples, never a task against
    an idle robot, so a rigid replay parks the robot in the way.

    So "a robot at rest blocks nobody" (module docstring, WHAT IS ASSUMED) is checked
    here for every scheduled pair rather than assumed: the fix is geometric (a HOME or
    layout clear of the other robots' reach, or a finer object model in mu), not in the
    graph.
    """
    # Every task starts and ends at the same HOME, so one blocking pose shows up at every
    # task end: report each (resting robot, other robot's task) once.
    grouped: Dict[Tuple[str, str, str], list] = {}
    for who, task, end, other, otask, l0, l1, n in resting:
        g = grouped.setdefault((who, other, otask), [l0, l1, n, 0])
        g[0], g[1], g[2], g[3] = min(g[0], l0), max(g[1], l1), max(g[2], n), g[3] + 1
    lines = [f"{who} at rest (HOME: {k} task ends) collides with {other} {otask} "
             f"samples {l0}..{l1} ({n} samples)"
             for (who, other, otask), (l0, l1, n, k) in list(grouped.items())[:12]]
    more = f"\n  ... {len(grouped) - 12} more" if len(grouped) > 12 else ""
    raise AssertionError(
        "a resting robot blocks another robot's motion -- the plan graph (and a rigid "
        "replay) assume that a robot at HOME or at a parking pose collides with nothing "
        "(ADR-0007):\n  " + "\n  ".join(lines) + more
        + "\nFix the geometry (HOME / layout / carried-object model), not the graph.")


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
            n += {"pipeline": 2, "gate": 1, "hold": 3}.get(mode, 1)
    return n


def precedence_edges(
    deps: "Dict[str, Dict[str, np.ndarray]] | Dict[str, np.ndarray]",
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

    * ``hold`` (fabricator v2, a pick-place ``i`` HOLDING its part while a process ``j``
      tacks it) -- three edges, matching the solver's ``start[j] >= m0[i]``,
      ``h[i] <= m0[j]``, ``e[j] <= m1[i]``:
        1. ``j``'s first node waits for ``i``'s pick node (the pick gate, as ``pipeline``'s);
        2. ``j``'s acquire node (the arc strike) waits for ``i``'s hold node ``h``
           (``hold_offsets``: the part has arrived and is held still);
        3. ``i``'s release node (GripOpen) waits for ``j``'s node ``e - 1``, its last
           ProcessOff sample (``process_end_offsets``): the part is let go only once the
           last arc is out. This edge runs from the process robot to the handler.

    ``deps`` is ``deps_on`` (``deps[rj][ri]`` receives the edge of a pair on two robots);
    the two-robot flat ``{r: array}`` is still accepted.
    """
    # .get: a schedule with no task at all needs no milestone (the empty-robot guard).
    pick, place = problem.get("pick_offsets", {}), problem.get("place_offsets", {})
    hold, pend = problem.get("hold_offsets", {}), problem.get("process_end_offsets", {})
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
        (ri, start_i, pick_i, place_i), (rj, start_j, pick_j, place_j) = node[i], node[j]
        # (milestone robot, milestone node, waiter robot, waiter node)
        if mode == "pipeline":
            pairs = ((ri, pick_i, rj, start_j), (ri, place_i, rj, place_j))
        elif mode == "gate":
            pairs = ((ri, place_i, rj, pick_j),)
        elif mode == "hold":
            h_i = start_i + int(hold[f"{ri}|{i}"])
            e_j = start_j + int(pend[f"{rj}|{j}"])
            pairs = ((ri, pick_i, rj, start_j), (ri, h_i, rj, pick_j), (rj, e_j - 1, ri, place_i))
        else:
            raise ValueError(f"precedence ({i}, {j}) has unknown mode {mode!r}")
        for rm, milestone, rw, waiter in pairs:
            if rm == rw:
                if milestone > waiter:
                    raise AssertionError(
                        f"schedule violates {mode} precedence ({i}, {j}) on {rm}: node "
                        f"{milestone} must come before node {waiter} but does not")
                continue      # same robot: its own node order already enforces it
            # "has REACHED", not "has left" -- the solver's conditions permit equality.
            row = deps[rw][rm] if isinstance(deps[rw], dict) else deps[rw]
            row[waiter] = max(int(row[waiter]), milestone)
            n += 1
    return n


def zero_delay_ticks(tpg: TPG) -> int:
    """Slots the graph takes to execute with no stall at all: the TPG's own makespan.

    Same rule as ``simulate_tpg.run_tpg`` -- each tick every robot (in ``tpg.robots``
    order) advances one node iff its entry is satisfied by what every other robot has
    reached, a robot seeing the moves of the robots before it in the same tick -- so the
    two agree tick for tick. It is NOT ``tpg.nominal_makespan``: that is the SOLVER's
    makespan, which keeps the idle gaps between tasks (a turn-based schedule starts each
    task only when the previous one has ended, and the graph does not wait for a clock,
    only for its edges). Raises ``RuntimeError`` on a deadlock, which
    :func:`assert_acyclic` should make impossible.
    """
    robots = tpg.robots
    reached = {q: -1 for q in robots}
    n = {q: tpg.n_nodes(q) for q in robots}
    ticks = 0
    while any(reached[q] < n[q] - 1 for q in robots):
        ticks += 1
        moved = False
        for q in robots:
            nxt = reached[q] + 1
            if nxt >= n[q]:
                continue
            if tpg.ready(q, nxt, reached):
                reached[q] = nxt
                moved = True
        if not moved:
            raise RuntimeError(f"TPG deadlocks at tick {ticks}: {reached}")
    return ticks


def assert_acyclic(tpg: TPG) -> None:
    """Deadlock-freedom check.

    Two ways a TPG can deadlock, both checked here, for every ordered robot pair:

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
        for o in tpg.others(q):
            dep, n_other = tpg.deps_on[q][o], tpg.n_nodes(o)
            for node in np.flatnonzero(dep != FREE):
                d = int(dep[node])
                if d >= n_other:
                    task, k = tpg.locate(q, int(node))
                    raise AssertionError(
                        f"unsatisfiable dependency: {q}#{node} ({task}[{k}]) waits for {o} "
                        f"to reach node {d}, but {o} has only {n_other} nodes. Its final "
                        f"configuration collides -- HOME is not a non-blocking pose here.")
                if _instant(tpg, o, d) > _instant(tpg, q, int(node)):
                    raise AssertionError(
                        f"TPG edge {o}#{d} -> {q}#{node} runs backwards in nominal time; "
                        f"the graph may contain a cycle and could deadlock")
    _assert_no_cycle(tpg)


def _assert_no_cycle(tpg: TPG) -> None:
    """Exact cycle check, O(nodes x robots): the nominal-time test above is necessary, not
    sufficient.

    An edge's source is never nominally LATER than its target, but it can be nominally
    SIMULTANEOUS, and two such edges can close a cycle (found 2026-09-21 by
    ``test_tpg.py`` on synthetic instances). The geometric case: with the scheduled offset
    ``delta`` collision-free but both ``delta - 1`` and ``delta + 1`` colliding -- a
    one-slot hole in the forbidden set, which the solver may legally pick -- nodes
    ``(n, m)`` and ``(n + 1, m - 1)`` both collide, so ``r`` may not enter ``n + 1``
    until ``s`` has left ``m - 1`` and ``s`` may not enter ``m`` until ``r`` has left
    ``n``. The rigid schedule passes through that crossing in lock-step; a graph, which
    only knows "after", cannot, and would deadlock on it at any delay. With three robots
    or more such edges can also close a cycle through several robots, each pair of which
    is acyclic on its own.

    This is Kahn's algorithm on the node graph: the ``N`` per-robot chains plus every
    cross edge. Within a chain only the head can have all its predecessors done (its own
    predecessor is the previous node), so "remove every node whose in-edges are all
    satisfied" is the sweep below, which advances any robot whose next node is ready. It
    visits every node iff the graph is acyclic: if every unfinished robot is stuck, each
    one's next node waits for a node of another robot at or beyond THAT robot's next
    node, and following those waits among finitely many robots closes a cycle.
    """
    robots = tpg.robots
    n = {q: tpg.n_nodes(q) for q in robots}
    reached = {q: -1 for q in robots}
    while True:
        moved = False
        for q in robots:
            while reached[q] + 1 < n[q]:
                if not tpg.ready(q, reached[q] + 1, reached):
                    break
                reached[q] += 1
                moved = True
        if all(reached[q] == n[q] - 1 for q in robots):
            return
        if not moved:
            waits = []
            for q in robots:
                a = reached[q] + 1
                if a >= n[q]:
                    continue
                t, k = tpg.locate(q, a)
                waits += [f"{q}#{a} ({t}[{k}]) waits for {o} to reach {d}"
                          for o, d in tpg.blockers(q, a, reached)]
            raise AssertionError(
                f"TPG has a cycle: {' and '.join(waits)} -- nominally simultaneous edges "
                f"closing a loop (e.g. a one-slot hole in the forbidden offsets at the "
                f"scheduled offset). The graph would deadlock at any delay.")


def _instant(tpg: TPG, robot: str, node: int) -> int:
    for seg in tpg.segments[robot]:
        if seg.start_node <= node < seg.start_node + seg.n:
            return seg.start_slot + (node - seg.start_node)
    raise IndexError(f"{robot} has no node {node}")
