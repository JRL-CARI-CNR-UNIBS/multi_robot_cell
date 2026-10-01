#!/usr/bin/env python3
"""Test battery for the N-robot temporal plan graph (``tpg.py``, 2026-09-26). No pytest.

    .venv_vamp/bin/python scripts/test_tpg_nrobot.py [--seeds 12] [--traces 20] \
        [--runs RUN_DIR ...]

PASS/FAIL lines and a summary; exit status 1 if anything failed. The synthetic part needs
only numpy (plus the modules it exercises: ``simulate_tpg``, ``plan_oracle``, which import
the VAMP engine and yaml -- hence the VAMP interpreter); ``--runs`` also needs the SIMD
kernel. ``test_tpg.py`` stays the two-robot battery; this one covers what N robots add.

Synthetic instances: seeded random 3- and 4-robot plans -- random task durations, random
rectangles of colliding sample pairs between tasks of different robots (never touching a
trajectory's first or last sample: HOME blocks nobody), random cross-robot precedences
(``pipeline`` and ``gate``), and a RIGID schedule built by a greedy that places each task
at the first slot whose offset against every already-placed task of another robot is at
least ``margin`` away from every forbidden offset and that meets the precedences.

1. **Structure.** ``build`` is acyclic; every cross edge points strictly backward in
   schedule time (margin 2: no nominally simultaneous edge); ``deps_on[r][s]`` equals the
   TWO-robot graph of the pair ``(r, s)`` built alone (the N-robot graph is the union of
   the pairwise graphs); ``deps``/``other`` refuse N > 2; JSON round trip is exact.
2. **Zero delay.** With release times (a robot may not start a task before its scheduled
   slot) the graph's execution IS the rigid schedule, tick for tick -- so no edge ever
   blocks the nominal plan -- and its makespan is the rigid makespan. Without them (the
   graph's own ASAP execution) ``tpg.zero_delay_ticks`` <= the rigid makespan and equals
   ``simulate_tpg.run_tpg`` at zero stall and ``plan_oracle.run_graph``.
3. **Reduction equivalence.** The unreduced graph (one requirement per colliding pair,
   oriented by absolute instants, plus every precedence milestone; executed with ``all``,
   never a maximum) and the reduced graph give identical traces under zero delay and
   seeded stall-burst traces.
4. **Delays.** Random stall bursts on every robot (several seeds and rates) and an
   adversarial stall of each robot at each of its task starts: never two colliding
   samples in the same tick (checked against ``mu`` directly, every pair), never a
   precedence milestone out of order, every robot finishes (no deadlock). The rigid
   executor, fed the same traces, is reported for contrast.
5. **Cycles and resting robots.** A hand-built circular wait A -> B -> C -> A, each pair of which is acyclic
   on its own and whose edges are all nominally simultaneous (so the nominal-time test
   passes), is refused by the cycle check; and on margin-1 instances (the scheduled
   offset may sit one slot from a forbidden one) ``build`` refuses exactly the instances
   whose unreduced graph deadlocks. A turn-based-shaped plan in which a robot resting
   at HOME collides with another robot's motion (between two of its tasks, or before
   its first) is refused by name (regression, fabricator4 turn-based plan 2026-09-26).
7. **Synchronous hold** (fabricator v2): 3-robot plans where a handler's pick-place HOLDS
   its part for a tack on another robot (precedence ``hold``, ``hold_offsets`` /
   ``process_end_offsets``, mu empty on the handler's hold tail x the tack as the seam's
   exemption makes it). The graph has 3 edges per hold; under random stall bursts the
   graph never collides, never deadlocks, never lets the tack strike before the handler
   holds (h) or start before its pick, and never releases the part (GripOpen) before the
   tack's last ProcessOff sample (e - 1): 0 early releases.
6. **Two robots, byte for byte** (``--runs``). For each run directory the graph is rebuilt
   with ``build_tpg.py`` and compared with the archived ``tpg.json``; ``from_json`` ->
   ``to_json`` of the archived file is byte-identical as well.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import sys
import tempfile
import time
import traceback
from typing import Callable, Dict, List, Tuple

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tpg as T  # noqa: E402

RESULTS: List[Tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    sys.stdout.flush()
    return ok


# --------------------------------------------------------------------------- #
# synthetic N-robot instances
# --------------------------------------------------------------------------- #
class Instance:
    def __init__(self, name, robots, tasks, mu, start, precedences, modes, margin):
        self.name, self.robots, self.tasks, self.mu = name, tuple(robots), tasks, mu
        self.start, self.margin = start, margin
        self.solution = {"assignments": {
            t: {"robot": a["robot"], "start_slot": start[t], "end_slot": start[t] + a["n"]}
            for t, a in tasks.items()}}
        self.problem = {
            "tasks": list(tasks),
            "precedences": [list(p) for p in precedences],
            "precedence_modes": list(modes),
            "pick_offsets": {f"{a['robot']}|{t}": a["pick"] for t, a in tasks.items()},
            "place_offsets": {f"{a['robot']}|{t}": a["place"] for t, a in tasks.items()},
        }
        self.makespan = max(start[t] + a["n"] for t, a in tasks.items())
        # Own node layout, independent of tpg.timelines.
        self.base, self.n = {}, {}
        for q in self.robots:
            cur = 0
            for t in sorted((t for t, a in tasks.items() if a["robot"] == q),
                            key=lambda t: start[t]):
                self.base[t] = cur
                cur += tasks[t]["n"]
            self.n[q] = cur
        self.node_task = {q: [] for q in self.robots}      # node -> (task, k)
        for q in self.robots:
            for t in sorted((t for t, a in tasks.items() if a["robot"] == q),
                            key=lambda t: start[t]):
                self.node_task[q] += [(t, k) for k in range(tasks[t]["n"])]

    def mu_of(self, r, ti, s, tj):
        if self.robots.index(r) >= self.robots.index(s):
            raise AssertionError(f"mu_of called with {r} not before {s}")
        return self.mu[(ti, tj)]

    def instant(self, q: str, node: int) -> int:
        t, k = self.node_task[q][node]
        return self.start[t] + k

    def collides(self, pos: Dict[str, int]) -> List[Tuple[str, str]]:
        """Pairs of robots whose current samples collide -- straight from mu."""
        out = []
        for a, r in enumerate(self.robots):
            for s in self.robots[a + 1:]:
                if pos[r] < 0 or pos[s] < 0:
                    continue
                ti, k = self.node_task[r][pos[r]]
                tj, l = self.node_task[s][pos[s]]
                if self.mu[(ti, tj)][k, l]:
                    out.append((r, s))
        return out

    def sub(self, r: str, s: str) -> "Instance":
        """The two-robot instance of the pair (r, s): their tasks, mu and precedences."""
        keep = {t: a for t, a in self.tasks.items() if a["robot"] in (r, s)}
        prec = [(i, j, m) for (i, j), m in zip(self.problem["precedences"],
                                              self.problem["precedence_modes"])
                if i in keep and j in keep]
        return Instance(f"{self.name}[{r},{s}]", (r, s), keep,
                        {k: v for k, v in self.mu.items() if k[0] in keep and k[1] in keep},
                        {t: self.start[t] for t in keep}, [(i, j) for i, j, _ in prec],
                        [m for _, _, m in prec], self.margin)


def synthetic(seed: int, n_robots: int, margin: int = 2, p_collide: float = 0.55,
              stripes: bool = False) -> Instance:
    """``stripes``: colliding pairs on every OTHER diagonal of a window (offsets c, c+-2,
    c+-4), i.e. one-slot holes in the forbidden set -- where a margin-1 schedule can sit
    in lock-step between two forbidden offsets, which a graph cannot execute."""
    rng = np.random.default_rng(1000 * n_robots + seed + (0 if margin == 2 else 500_000))
    robots = tuple(f"robot{i + 1}" for i in range(n_robots))
    tasks: Dict[str, dict] = {}
    per_robot: Dict[str, List[str]] = {q: [] for q in robots}
    for q in robots:
        for _ in range(int(rng.integers(1, 5))):
            n = int(rng.integers(8, 31))
            t = f"t{len(tasks)}"
            tasks[t] = {"robot": q, "n": n, "pick": int(rng.integers(1, n // 2)),
                        "place": int(rng.integers(n // 2, n - 1))}
            per_robot[q].append(t)

    mu: Dict[Tuple[str, str], np.ndarray] = {}
    for ti, a in tasks.items():
        for tj, b in tasks.items():
            if robots.index(a["robot"]) >= robots.index(b["robot"]):
                continue
            m = np.zeros((a["n"], b["n"]), dtype=bool)
            if rng.random() < p_collide:
                for _ in range(int(rng.integers(1, 4))):
                    k0 = int(rng.integers(1, a["n"] - 1))
                    l0 = int(rng.integers(1, b["n"] - 1))
                    k1 = min(a["n"] - 1, k0 + int(rng.integers(1, 9)))   # never the last sample
                    l1 = min(b["n"] - 1, l0 + int(rng.integers(1, 9)))
                    if stripes:
                        kk, ll = np.meshgrid(np.arange(k0, k1), np.arange(l0, l1),
                                             indexing="ij")
                        c = int(rng.integers(-3, 4)) + (k0 - l0)
                        m[k0:k1, l0:l1] |= ((kk - ll - c) % 2 == 0) & (np.abs(kk - ll - c) <= 4)
                    else:
                        m[k0:k1, l0:l1] = True
            mu[(ti, tj)] = m

    # Round-robin placement order; precedences only from an earlier-placed task.
    order = []
    for rank in range(max(len(v) for v in per_robot.values())):
        order += [per_robot[q][rank] for q in robots if rank < len(per_robot[q])]
    pos = {t: i for i, t in enumerate(order)}
    precedences, modes = [], []
    for _ in range(int(rng.integers(0, 2 * n_robots))):
        i, j = (str(x) for x in rng.choice(order, 2, replace=False))
        if pos[i] > pos[j]:
            i, j = j, i
        if tasks[i]["robot"] == tasks[j]["robot"] or (i, j) in precedences:
            continue
        precedences.append((i, j))
        modes.append("pipeline" if rng.random() < 0.6 else "gate")

    # The solver's precedence constraints are non-strict; margin 2 keeps them strict so
    # that no edge is nominally simultaneous, margin 1 allows equality as the solver does.
    strict = 1 if margin >= 2 else 0
    start: Dict[str, int] = {}
    ready = {q: 0 for q in robots}
    for t in order:
        a = tasks[t]
        lo = ready[a["robot"]] + int(rng.integers(0, 4))
        for (i, j), mode in zip(precedences, modes):
            if j != t:
                continue
            si, bi = start[i], tasks[i]
            if mode == "pipeline":
                lo = max(lo, si + bi["pick"] + strict,                     # start[j] >= pick[i]
                         si + bi["place"] + strict - a["place"])          # place[i] <= place[j]
            else:
                lo = max(lo, si + bi["place"] + strict - a["pick"])       # pick[j] >= place[i]
        x = lo
        while not _clear(t, x, tasks, mu, start, robots, margin):
            x += 1
        start[t] = x
        ready[a["robot"]] = x + a["n"]
    return Instance(f"N{n_robots}#{seed}{'' if margin == 2 else ' m1'}", robots, tasks, mu,
                    start, precedences, modes, margin)


def _clear(t, x, tasks, mu, start, robots, margin) -> bool:
    """Is start slot ``x`` for ``t`` at least ``margin`` from every forbidden offset?"""
    a = tasks[t]
    for u, su in start.items():
        b = tasks[u]
        if b["robot"] == a["robot"]:
            continue
        first, second, s_first, s_second = (t, u, x, su) \
            if robots.index(a["robot"]) < robots.index(b["robot"]) else (u, t, su, x)
        ks, ls = np.nonzero(mu[(first, second)])
        if ks.size == 0:
            continue
        # sample k of `first` and l of `second` are simultaneous when k - l == delta
        delta = s_second - s_first
        if np.abs((ks - ls) - delta).min() < margin:
            return False
    return True


# --------------------------------------------------------------------------- #
# the unreduced graph and one executor for every graph
# --------------------------------------------------------------------------- #
class FullGraph:
    """Every colliding sample pair a requirement, every precedence milestone one too.

    ``req[q][m]`` maps each other robot to the array of its node indices that must ALL
    have been reached before ``q`` enters ``m``. Oriented by absolute instants, "+1 = has
    left" for geometry, "has reached" for precedences; nothing from ``tpg``.
    """

    def __init__(self, inst: Instance):
        self.robots = inst.robots
        lists: Dict[str, Dict[int, Dict[str, List[int]]]] = {q: {} for q in inst.robots}

        def need(q, m, o, v):
            lists[q].setdefault(int(m), {}).setdefault(o, []).append(int(v))

        for (ti, tj), m in inst.mu.items():
            r, s = inst.tasks[ti]["robot"], inst.tasks[tj]["robot"]
            for k, l in zip(*np.nonzero(m)):
                tr, ts = inst.start[ti] + k, inst.start[tj] + l
                if tr == ts:
                    raise AssertionError(f"{ti} x {tj}: collision at a simultaneous instant")
                if tr < ts:
                    need(s, inst.base[tj] + l, r, inst.base[ti] + k + 1)
                else:
                    need(r, inst.base[ti] + k, s, inst.base[tj] + l + 1)
        for (i, j), mode in zip(inst.problem["precedences"], inst.problem["precedence_modes"]):
            qi, qj = inst.tasks[i]["robot"], inst.tasks[j]["robot"]
            if qi == qj:
                continue
            node = {w: {t: inst.base[t] + (0 if w == "start" else inst.tasks[t][w])
                        for t in (i, j)} for w in ("start", "pick", "place")}
            conds = (("pick", "start"), ("place", "place")) if mode == "pipeline" \
                else (("place", "pick"),)
            for wi, wj in conds:
                need(qj, node[wj][j], qi, node[wi][i])
        self.req = {q: {m: {o: np.array(v) for o, v in d.items()} for m, d in lists[q].items()}
                    for q in inst.robots}

    def allows(self, q: str, m: int, reached: Dict[str, int]) -> bool:
        return all(bool(np.all(reached[o] >= v)) for o, v in self.req[q].get(m, {}).items())


def execute(robots, n: Dict[str, int], allows: Callable[[str, int, Dict[str, int]], bool],
            stall: Callable[[int, str, Dict[str, int]], bool] | None = None,
            release: Dict[str, Dict[int, int]] | None = None,
            max_ticks: int | None = None) -> Tuple[List[Tuple[int, ...]], bool]:
    """Each tick each robot, in order, advances one node iff not stalled, not before its
    release tick (``release[q][node]``, optional) and allowed by what the others have
    reached (seeing the moves of robots before it this tick). Returns (trace, deadlock)."""
    reached = {q: -1 for q in robots}
    trace: List[Tuple[int, ...]] = []
    t = 0
    limit = max_ticks or 60 * (sum(n.values()) + 10)
    while any(reached[q] < n[q] - 1 for q in robots):
        t += 1
        if t > limit:
            return trace, True
        moved, waiting = False, False
        for q in robots:
            nxt = reached[q] + 1
            if nxt >= n[q]:
                continue
            if stall is not None and stall(t, q, reached):
                waiting = True
                continue
            if release is not None and t < release[q].get(nxt, 0):
                waiting = True
                continue
            if allows(q, nxt, reached):
                reached[q] = nxt
                moved = True
        trace.append(tuple(reached[q] for q in robots))
        if not moved and not waiting:
            return trace, True
    return trace, False


def burst_stall(seed: int, robots, p: float, burst_max: int):
    """Seeded stall bursts, drawn once per tick for every robot (so two executions of the
    same length see the same trace): with probability ``p`` a robot not already stalled
    enters a stall of 1..``burst_max`` ticks."""
    rng = np.random.default_rng(seed)
    remaining = {q: 0 for q in robots}
    cur = {"t": 0, "stalled": {q: False for q in robots}}

    def stall(t, q, _reached):
        if t != cur["t"]:
            cur["t"] = t
            for r in robots:
                if remaining[r] == 0 and rng.random() < p:
                    remaining[r] = int(rng.integers(1, burst_max + 1))
                cur["stalled"][r] = remaining[r] > 0
                if remaining[r] > 0:
                    remaining[r] -= 1
        return cur["stalled"][q]
    return stall


def milestone_violations(inst: Instance, trace: List[Tuple[int, ...]]) -> int:
    """Precedence conditions whose milestones were first reached in the wrong order."""
    idx = {q: i for i, q in enumerate(inst.robots)}

    def first_tick(task, which):
        q = inst.tasks[task]["robot"]
        node = inst.base[task] + (0 if which == "start" else inst.tasks[task][which])
        for t, row in enumerate(trace):
            if row[idx[q]] >= node:
                return t
        return None

    bad = 0
    for (i, j), mode in zip(inst.problem["precedences"], inst.problem["precedence_modes"]):
        conds = (("pick", "start"), ("place", "place")) if mode == "pipeline" \
            else (("place", "pick"),)
        for wi, wj in conds:
            a, b = first_tick(i, wi), first_tick(j, wj)
            if a is not None and b is not None and a > b:
                bad += 1
    return bad


def collisions_in(inst: Instance, trace: List[Tuple[int, ...]]) -> int:
    return sum(1 for row in trace if inst.collides(dict(zip(inst.robots, row))))


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_structure(inst: Instance, g: T.TPG) -> None:
    # every cross edge strictly backward in schedule time (independent instants)
    n_cross, bad = 0, []
    for q in inst.robots:
        for o in inst.robots:
            if o == q:
                continue
            dep = g.deps_on[q][o]
            for node in np.flatnonzero(dep != T.FREE):
                n_cross += 1
                d = int(dep[node])
                if not inst.instant(o, d) < inst.instant(q, int(node)):
                    bad.append((q, int(node), o, d))
    record(f"[1] {inst.name}: every cross edge points strictly backward in schedule time",
           not bad, f"{n_cross} cross entries, {g.n_precedence_edges} from precedences"
           + (f"; offending {bad[:3]}" if bad else ""))

    # the N-robot graph is the union of the pairwise two-robot graphs
    diff = []
    for a, r in enumerate(inst.robots):
        for s in inst.robots[a + 1:]:
            sub = inst.sub(r, s)
            g2 = T.build(sub.solution, sub.robots, sub.mu_of, 0.01, problem=sub.problem)
            if not (np.array_equal(g2.deps[r], g.deps_on[r][s])
                    and np.array_equal(g2.deps[s], g.deps_on[s][r])):
                diff.append((r, s))
    record(f"[1] {inst.name}: deps_on[r][s] == the two-robot graph of (r, s), every pair",
           not diff, f"{len(inst.robots) * (len(inst.robots) - 1) // 2} pairs"
           + (f"; differing {diff}" if diff else ""))

    # two-robot views refuse, JSON round trip exact
    refused = 0
    for fn in (lambda: g.deps, lambda: g.other(inst.robots[0]),
               lambda: g.independent_window(inst.robots[0], 0, 0)):
        try:
            fn()
        except ValueError:
            refused += 1
    with tempfile.TemporaryDirectory() as d:
        p1, p2 = os.path.join(d, "a.json"), os.path.join(d, "b.json")
        g.to_json(p1)
        back = T.TPG.from_json(p1)
        back.to_json(p2)
        same = filecmp.cmp(p1, p2, shallow=False)
        raw = json.load(open(p1))
    nested = all(isinstance(raw["deps"][q], dict) and set(raw["deps"][q]) == set(g.others(q))
                 for q in inst.robots)
    eq = all(np.array_equal(back.deps_on[q][o], g.deps_on[q][o])
             for q in inst.robots for o in g.others(q))
    record(f"[1] {inst.name}: deps/other/independent_window refuse N>2; JSON round trip",
           refused == 3 and same and nested and eq and back.robots == g.robots
           and back.n_edges == g.n_edges == g.count_edges(),
           f"refused {refused}/3, nested deps {nested}, byte-identical re-write {same}")


def test_zero_delay(inst: Instance, g: T.TPG, sim, oracle_mod) -> None:
    robots = inst.robots
    n = {q: g.n_nodes(q) for q in robots}
    reduced = g.ready
    # released: no task before its scheduled slot -> must be the rigid schedule itself
    release = {q: {} for q in robots}
    for t, a in inst.tasks.items():
        release[a["robot"]][inst.base[t]] = inst.start[t] + 1        # tick 1 = slot 0
    tr, dead = execute(robots, n, reduced, release=release)
    rigid = []
    for tick in range(1, inst.makespan + 1):
        row = []
        for q in robots:
            p = -1
            for t in sorted((t for t, a in inst.tasks.items() if a["robot"] == q),
                            key=lambda t: inst.start[t]):
                s0, nt = inst.start[t], inst.tasks[t]["n"]
                if tick - 1 >= s0 + nt:
                    p = inst.base[t] + nt - 1
                elif tick - 1 >= s0:
                    p = inst.base[t] + tick - 1 - s0
                    break
                else:
                    break
            row.append(p)
        rigid.append(tuple(row))
    record(f"[2] {inst.name}: released zero-delay execution == rigid schedule, tick for tick",
           not dead and tr == rigid and len(tr) == inst.makespan,
           f"rigid makespan {inst.makespan}, released graph {len(tr)} ticks")

    # ASAP: graph's own zero-delay makespan, three implementations agree, <= rigid
    zd = T.zero_delay_ticks(g)
    mu_tasks = {(ti, tj): m for (ti, tj), m in inst.mu.items()}
    orc = sim.Oracle(g, mu_tasks)
    order = sim.Order(g, inst.problem)
    horizon = 4 * sum(n.values()) + 10
    no_stall = [{q: False for q in robots}] * horizon
    res = sim.run_tpg(g, orc, order, no_stall)
    rig = sim.run_rigid(g, orc, order, no_stall)
    plan = type("P", (), {})()
    plan.robots, plan.n = list(robots), n
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tpg.json")
        g.to_json(p)
        ticks, dead_o = oracle_mod.run_graph(plan, json.load(open(p)))
    record(f"[2] {inst.name}: graph zero-delay makespan <= rigid; tpg / simulate_tpg / "
           f"plan_oracle agree",
           zd <= inst.makespan and res["ticks"] == zd and not res["deadlock"]
           and res["collisions"] == 0 and len(ticks) - 1 == zd and not dead_o
           and rig["collisions"] == 0 and not rig["deadlock"],
           f"graph {zd}, simulate_tpg {res['ticks']}, plan_oracle {len(ticks) - 1}, rigid "
           f"{inst.makespan} (simulate_tpg rigid {rig['ticks']}, {rig['collisions']} coll.)")


def test_equivalence_and_delays(inst: Instance, g: T.TPG, n_traces: int, sim) -> None:
    robots = inst.robots
    n = {q: g.n_nodes(q) for q in robots}
    full = FullGraph(inst)
    specs = [None] + [(sd, p, b) for sd in range(n_traces)
                      for p, b in [((0.01, 0.03, 0.08)[sd % 3], (10, 25, 50)[sd % 3])]]
    same, coll, ooo, dead = 0, 0, 0, 0
    rigid_coll = 0
    for spec in specs:
        mk = (lambda: None) if spec is None else (lambda sp=spec: burst_stall(sp[0], robots,
                                                                             sp[1], sp[2]))
        tr_red, d1 = execute(robots, n, g.ready, mk())
        tr_full, d2 = execute(robots, n, full.allows, mk())
        same += tr_red == tr_full and d1 == d2
        coll += collisions_in(inst, tr_red)
        ooo += milestone_violations(inst, tr_red)
        dead += d1
        if spec is not None:
            # rigid executor on the same kind of trace, for contrast
            rng = np.random.default_rng(spec[0])
            trace = sim._delay_trace(rng, 4 * sum(n.values()) + 10, robots, spec[1], spec[2])
            rig = sim.run_rigid(g, sim.Oracle(g, inst.mu), sim.Order(g, inst.problem), trace)
            rigid_coll += rig["collisions"] > 0
    record(f"[3] {inst.name}: full vs reduced graph, {len(specs)} executions identical",
           same == len(specs), f"{same}/{len(specs)} identical")
    record(f"[4] {inst.name}: {len(specs) - 1} random stall-burst traces",
           coll == 0 and ooo == 0 and dead == 0,
           f"{coll} colliding ticks, {ooo} out of order, {dead} deadlocks "
           f"(rigid executor collided in {rigid_coll}/{len(specs) - 1})")

    # adversarial: each robot stalls at each of its task starts for 2x the longest task
    longest = max(a["n"] for a in inst.tasks.values())
    runs = coll = ooo = dead = 0
    for victim in robots:
        for t, a in inst.tasks.items():
            if a["robot"] != victim:
                continue
            at = inst.base[t]
            state = {"from": None}

            def stall(tick, q, reached, victim=victim, at=at, state=state):
                if q != victim:
                    return False
                if state["from"] is None and reached[q] + 1 == at:
                    state["from"] = tick
                return state["from"] is not None and tick < state["from"] + 2 * longest
            tr, d = execute(robots, n, g.ready, stall)
            runs += 1
            coll += collisions_in(inst, tr)
            ooo += milestone_violations(inst, tr)
            dead += d
    record(f"[4] {inst.name}: {runs} adversarial stalls ({2 * longest} ticks at every task "
           f"start, each robot)", coll == 0 and ooo == 0 and dead == 0,
           f"{coll} colliding ticks, {ooo} out of order, {dead} deadlocks")


def test_circular_wait() -> None:
    """A -> B -> C -> A: pairwise acyclic, nominally simultaneous, refused."""
    robots = ("A", "B", "C")
    segs = {q: [T.Segment(f"t{q}", 0, 4, 0)] for q in robots}
    deps_on = {q: {o: np.full(4, T.FREE, dtype=np.int64) for o in robots if o != q}
               for q in robots}
    deps_on["A"]["B"][2] = 2         # A may enter 2 once B has reached 2
    deps_on["B"]["C"][2] = 2         # B ... once C has reached 2
    deps_on["C"]["A"][2] = 2         # C ... once A has reached 2
    g = T.TPG(robots=robots, segments=segs, deps_on=deps_on, delta_t=0.01,
              nominal_makespan=4)
    g.n_edges = g.count_edges()
    try:
        T.assert_acyclic(g)
        raised, msg = False, ""
    except AssertionError as e:
        raised, msg = "cycle" in str(e), str(e)
    # each pair alone is acyclic: the cycle needs all three robots
    pair_ok = 0
    for a, r in enumerate(robots):
        for s in robots[a + 1:]:
            sub = T.TPG(robots=(r, s), segments={q: segs[q] for q in (r, s)},
                        deps_on={r: {s: deps_on[r][s].copy()}, s: {r: deps_on[s][r].copy()}},
                        delta_t=0.01, nominal_makespan=4)
            try:
                T.assert_acyclic(sub)
                pair_ok += 1
            except AssertionError:
                pass
    try:
        T.zero_delay_ticks(g)
        hangs = False
    except RuntimeError:
        hangs = True
    record("[5] circular wait A->B->C->A refused by the cycle check (Kahn over 3 chains)",
           raised and pair_ok == 3 and hangs,
           f"each of the 3 pairs acyclic alone: {pair_ok == 3}; executor deadlocks: {hangs}; "
           f"message: {msg[:150]}...")

    # The same loop from GEOMETRY, through build(): three one-task robots, all starting at
    # slot 0, each pair colliding at exactly one sample pair one slot off the scheduled
    # offset -- so each pair yields one nominally simultaneous edge and no 2-cycle.
    tasks = {f"t{q}": {"robot": q, "n": 5, "pick": 1, "place": 3} for q in robots}
    mu = {k: np.zeros((5, 5), dtype=bool) for k in (("tA", "tB"), ("tB", "tC"), ("tA", "tC"))}
    mu[("tA", "tB")][2, 1] = True    # A#2 x B#1, B's earlier: A#2 waits for B to reach 2
    mu[("tB", "tC")][2, 1] = True    # B#2 x C#1: B#2 waits for C to reach 2
    mu[("tA", "tC")][1, 2] = True    # A#1 x C#2, A's earlier: C#2 waits for A to reach 2
    inst = Instance("A->B->C->A (geometry)", robots, tasks, mu, {t: 0 for t in tasks}, [], [], 1)
    try:
        T.build(inst.solution, robots, inst.mu_of, 0.01, problem=inst.problem)
        refused, msg = False, ""
    except AssertionError as e:
        refused, msg = "cycle" in str(e), str(e)
    pairs = 0
    for a, r in enumerate(robots):
        for s in robots[a + 1:]:
            sub = inst.sub(r, s)
            T.build(sub.solution, sub.robots, sub.mu_of, 0.01, problem=sub.problem)
            pairs += 1
    _, dead = execute(robots, inst.n, FullGraph(inst).allows)
    record("[5] the same circular wait from mu (build(), 3 robots) refused; every pair builds "
           "alone; the unreduced graph deadlocks", refused and pairs == 3 and dead,
           msg[:120] + "...")

    # the same loop, but one edge satisfiable earlier: accepted
    deps_on["C"]["A"][2] = 1
    try:
        T.assert_acyclic(g)
        ok = T.zero_delay_ticks(g) > 0
    except AssertionError:
        ok = False
    record("[5] the same chain with the loop broken (C waits for A#1) is accepted", ok)


def test_cycle_detection(n_inst: int) -> None:
    """Margin-1 instances: build refuses exactly the ones whose unreduced graph deadlocks."""
    total = refused = deadlocking = agree = multi = 0
    for nr in (3, 4):
        for seed in range(n_inst):
            inst = synthetic(seed, nr, margin=1, p_collide=0.8, stripes=seed % 2 == 1)
            total += 1
            full = FullGraph(inst)
            _, dead = execute(inst.robots, inst.n, full.allows)
            deadlocking += dead
            try:
                T.build(inst.solution, inst.robots, inst.mu_of, 0.01, problem=inst.problem)
                raised = False
            except AssertionError as e:
                if "cycle" not in str(e):
                    raise
                raised = True
            refused += raised
            agree += raised == dead
            if raised:
                # does every pair build on its own? then the cycle runs through >= 3 robots
                pairwise = True
                for a, r in enumerate(inst.robots):
                    for s in inst.robots[a + 1:]:
                        sub = inst.sub(r, s)
                        try:
                            T.build(sub.solution, sub.robots, sub.mu_of, 0.01,
                                    problem=sub.problem)
                        except AssertionError:
                            pairwise = False
                multi += pairwise
    record(f"[5] cycle check on {total} margin-1 instances (3 and 4 robots): refused == "
           f"deadlocking", agree == total,
           f"{refused} refused, {deadlocking} deadlocking, {multi} of the refused have every "
           f"robot pair acyclic on its own (a cycle through >= 3 robots)")


def test_resting_robot() -> None:
    """Regression (2026-09-26, fabricator4): a robot at REST (HOME) that collides with
    another robot's motion must be refused by name, whatever the schedule's shape.

    Turn-based shape, three robots, one moving at a time: C's HOME (the first and last
    sample of each of its tasks) collides with A's task ``a1`` mid-motion. With C
    resting BETWEEN two tasks the old build failed with a misleading "runs backwards in
    nominal time" (the edge's target is the first node of C's next task); with C resting
    BEFORE its first task it built without complaint -- a graph that lets A pass through
    C's HOME. Both must now raise the resting-robot error.
    """
    robots = ("A", "B", "C")
    tasks = {"a0": {"robot": "A", "n": 10, "pick": 2, "place": 7},
             "a1": {"robot": "A", "n": 12, "pick": 3, "place": 8},
             "b0": {"robot": "B", "n": 9, "pick": 2, "place": 6},
             "c0": {"robot": "C", "n": 8, "pick": 2, "place": 5},
             "c1": {"robot": "C", "n": 8, "pick": 2, "place": 5}}

    def mu_for(order):
        mu = {}
        for ti, a in tasks.items():
            for tj, b in tasks.items():
                if robots.index(a["robot"]) < robots.index(b["robot"]):
                    mu[(ti, tj)] = np.zeros((a["n"], b["n"]), dtype=bool)
        for c in ("c0", "c1"):                   # C at HOME x a1 samples 4..6
            mu[("a1", c)][4:7, 0] = True
            mu[("a1", c)][4:7, -1] = True
        return mu

    cases = {
        # turn-based: a0, c0, b0, a1 (C idle at HOME between c0 and c1), c1
        "C resting between tasks": {"a0": 0, "c0": 10, "b0": 18, "a1": 27, "c1": 39},
        # C's first task after a1: C waits at HOME before any node of its own
        "C resting before its first task": {"a0": 0, "b0": 10, "a1": 19, "c0": 31, "c1": 39},
    }
    for label, start in cases.items():
        inst = Instance(f"turn-based {label}", robots, tasks, mu_for(start), start, [], [], 2)
        try:
            T.build(inst.solution, robots, inst.mu_of, 0.01, problem=inst.problem)
            ok, msg = False, "built without complaint"
        except AssertionError as e:
            msg = str(e)
            ok = "resting robot" in msg and "C at rest" in msg and "a1 samples 4..6" in msg
        record(f"[5] {label} at a colliding HOME: build refuses it by name", ok,
               msg.splitlines()[1].strip() if "\n" in msg else msg[:160])


def hold_instance(seed: int) -> Instance:
    """A 3-robot plan with two holds: robot1's pick-places pp0/pp1 hold their parts for the
    tacks t0/t1 (on robot2 or robot3), which then gate a seam each; mu random elsewhere,
    empty on each hold tail [h, m1 + 2) x its tack (the exemption), never on first/last."""
    for attempt in range(200):
        rng = np.random.default_rng(90_000 + 1000 * seed + attempt)
        robots = ("robot1", "robot2", "robot3")
        tasks: Dict[str, dict] = {}
        for k in range(2):
            n = int(rng.integers(45, 60))
            pick = int(rng.integers(3, 8))
            hold = int(rng.integers(12, 18))
            place = hold + int(rng.integers(20, 26))
            tasks[f"pp{k}"] = {"robot": "robot1", "n": n, "pick": pick, "place": place, "hold": hold}
        for k in range(2):
            n = int(rng.integers(20, 30))
            pick = int(rng.integers(3, 6))
            end = pick + int(rng.integers(6, 12))           # e: one past the last ProcessOff
            tasks[f"t{k}"] = {"robot": str(rng.choice(["robot2", "robot3"])), "n": n,
                              "pick": pick, "place": end - 2, "pend": end}
        for k in range(2):
            n = int(rng.integers(15, 30))
            tasks[f"w{k}"] = {"robot": str(rng.choice(["robot2", "robot3"])), "n": n,
                              "pick": int(rng.integers(1, n // 2)),
                              "place": int(rng.integers(n // 2, n - 1))}
        mu: Dict[Tuple[str, str], np.ndarray] = {}
        for ti, a in tasks.items():
            for tj, b in tasks.items():
                if robots.index(a["robot"]) >= robots.index(b["robot"]):
                    continue
                m = np.zeros((a["n"], b["n"]), dtype=bool)
                if rng.random() < 0.6:
                    for _ in range(int(rng.integers(1, 4))):
                        k0 = int(rng.integers(1, a["n"] - 1))
                        l0 = int(rng.integers(1, b["n"] - 1))
                        m[k0:min(a["n"] - 1, k0 + int(rng.integers(1, 9))),
                          l0:min(b["n"] - 1, l0 + int(rng.integers(1, 9)))] = True
                if ti.startswith("pp") and tj == "t" + ti[2:]:
                    m[a["hold"]:a["place"] + 2, :] = False     # the exemption
                mu[(ti, tj)] = m
        precedences = [("pp0", "t0"), ("pp1", "t1"), ("t0", "w0"), ("t1", "w1")]
        modes = ["hold", "hold", "gate", "gate"]
        # greedy rigid schedule, strict margin 2 (no nominally simultaneous edge)
        start: Dict[str, int] = {}
        ready = {q: 0 for q in robots}
        ok = True
        for t in ("pp0", "t0", "pp1", "t1", "w0", "w1"):
            a = tasks[t]
            lo, hi = ready[a["robot"]] + int(rng.integers(0, 4)), 10 ** 6
            if t.startswith("t"):
                i = "pp" + t[1:]
                si, bi = start[i], tasks[i]
                lo = max(lo, si + bi["pick"] + 1, si + bi["hold"] + 1 - a["pick"])
                hi = si + bi["place"] - 1 - a["pend"]
            if t.startswith("w"):
                i = "t" + t[1:]
                lo = max(lo, start[i] + tasks[i]["place"] + 1 - a["pick"])
            x = lo
            while x <= hi and not _clear(t, x, tasks, mu, start, robots, 2):
                x += 1
            if x > hi:
                ok = False
                break
            start[t] = x
            ready[a["robot"]] = x + a["n"]
        if not ok:
            continue
        inst = Instance(f"hold#{seed}", robots, tasks, mu, start, precedences, modes, 2)
        inst.problem["hold_offsets"] = {f"robot1|pp{k}": tasks[f"pp{k}"]["hold"] for k in range(2)}
        inst.problem["process_end_offsets"] = {f"{tasks[f't{k}']['robot']}|t{k}": tasks[f"t{k}"]["pend"]
                                               for k in range(2)}
        return inst
    raise RuntimeError(f"hold#{seed}: no feasible synthetic schedule")


def test_hold(n_inst: int, n_traces: int) -> None:
    """[7] the synchronous hold, synthetic: 3 edges per hold, and under random stalls no
    collision, no deadlock, no strike before the hold, no early release."""
    for seed in range(n_inst):
        inst = hold_instance(seed)
        g = T.build(inst.solution, inst.robots, inst.mu_of, 0.01, problem=inst.problem)
        cross = sum(1 for i, j in (("pp0", "t0"), ("pp1", "t1"))
                    if inst.tasks[i]["robot"] != inst.tasks[j]["robot"])
        exp = 3 * cross + sum(1 for i, j in (("t0", "w0"), ("t1", "w1"))
                              if inst.tasks[i]["robot"] != inst.tasks[j]["robot"])
        record(f"[7] {inst.name}: hold edges", g.n_precedence_edges == exp
               and g.precedence_edges_expected == exp,
               f"{g.n_precedence_edges} precedence edges, expected {exp}")
        idx = {q: i for i, q in enumerate(inst.robots)}

        def first(trace, task, node_off):
            q = inst.tasks[task]["robot"]
            node = inst.base[task] + node_off
            return next((t for t, row in enumerate(trace) if row[idx[q]] >= node), None)

        bad = []
        n = {q: g.n_nodes(q) for q in inst.robots}
        for tr in range(n_traces):
            stall = burst_stall(7000 + 97 * seed + tr, inst.robots, [0.0, 0.01, 0.05, 0.15][tr % 4], 25)
            trace, dead = execute(inst.robots, n, g.ready, stall)
            col = collisions_in(inst, trace)
            early = strike = gate = 0
            for k in range(2):
                pp, tk = inst.tasks[f"pp{k}"], inst.tasks[f"t{k}"]
                t_rel = first(trace, f"pp{k}", pp["place"])
                t_out = first(trace, f"t{k}", tk["pend"] - 1)
                early += t_rel is not None and (t_out is None or t_out > t_rel)
                t_h, t_s = first(trace, f"pp{k}", pp["hold"]), first(trace, f"t{k}", tk["pick"])
                strike += t_s is not None and (t_h is None or t_h > t_s)
                t_p, t_0 = first(trace, f"pp{k}", pp["pick"]), first(trace, f"t{k}", 0)
                gate += t_0 is not None and (t_p is None or t_p > t_0)
            if dead or col or early or strike or gate:
                bad.append((tr, dead, col, early, strike, gate))
        record(f"[7] {inst.name}: {n_traces} stall traces: 0 collisions, 0 deadlocks, "
               f"0 early releases, 0 strikes before the hold, 0 starts before the pick",
               not bad, f"failing (trace, deadlock, collisions, early, strike, gate): {bad[:3]}"
               if bad else "")


def test_guards() -> None:
    try:
        import refine_yield
        import coordinate
    except Exception as e:  # noqa: BLE001
        record("[1] refine_yield / coordinate refuse N>2", False, f"import: {e}")
        return
    n = 0
    for fn in (refine_yield.require_two_robots, coordinate.require_two_robots):
        try:
            fn(["robot1", "robot2", "robot3"])
        except ValueError:
            n += 1
        fn(["robot1", "robot2"])
    record("[1] refine_yield / coordinate refuse a 3-robot plan, accept a 2-robot one", n == 2)


def test_runs(run_dirs: List[str]) -> None:
    import build_tpg
    for d in run_dirs:
        name = os.path.basename(os.path.normpath(d))
        ymls = [f for f in os.listdir(d) if f.startswith("tamp_task") and f.endswith(".yaml")]
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "tpg.json")
            rt = os.path.join(tmp, "roundtrip.json")
            t0 = time.time()
            build_tpg.main(["--traj", os.path.join(d, "tamp_trajectories.json"),
                            "--task", os.path.join(d, ymls[0]),
                            "--solution", os.path.join(d, "tamp_solution.json"),
                            "--problem", os.path.join(d, "tamp_problem.json"),
                            "--out", out])
            same = filecmp.cmp(out, os.path.join(d, "tpg.json"), shallow=False)
            T.TPG.from_json(os.path.join(d, "tpg.json")).to_json(rt)
            rt_same = filecmp.cmp(rt, os.path.join(d, "tpg.json"), shallow=False)
            g = T.TPG.from_json(out)
        record(f"[6] {name}: rebuilt tpg.json byte-identical to the archived one; "
               f"from_json -> to_json byte-identical", same and rt_same,
               f"rebuild {same}, round trip {rt_same}; {g.n_edges} edges, zero-delay "
               f"{T.zero_delay_ticks(g)} slots; {time.time() - t0:.1f}s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10, help="synthetic instances per N (3, 4)")
    ap.add_argument("--traces", type=int, default=15, help="random stall traces per instance")
    ap.add_argument("--cycle-instances", type=int, default=40,
                    help="margin-1 instances per N for the cycle check")
    ap.add_argument("--runs", nargs="*", default=[],
                    help="run directories (tamp_trajectories/solution/problem.json, "
                         "tamp_task*.yaml, tpg.json) for the two-robot byte-identity test")
    args = ap.parse_args(argv)
    t_start = time.time()

    def guarded(name, fn, *a):
        try:
            fn(*a)
        except Exception as e:  # noqa: BLE001
            record(name, False, f"{type(e).__name__}: {e}")
            traceback.print_exc()

    import simulate_tpg as sim
    import plan_oracle as oracle_mod

    print("== guards and hand-built cycles")
    guarded("[1] guards", test_guards)
    guarded("[5] circular wait", test_circular_wait)
    guarded("[5] cycle detection", test_cycle_detection, args.cycle_instances)
    guarded("[5] resting robot", test_resting_robot)
    print("== synchronous hold")
    guarded("[7] hold", test_hold, 6, args.traces)
    print("== synthetic 3- and 4-robot instances")
    for nr in (3, 4):
        for seed in range(args.seeds):
            inst = synthetic(seed, nr)
            try:
                g = T.build(inst.solution, inst.robots, inst.mu_of, 0.01, problem=inst.problem)
            except Exception as e:  # noqa: BLE001
                record(f"{inst.name}: build", False, f"{type(e).__name__}: {e}")
                traceback.print_exc()
                continue
            print(f"INFO  {inst.name}: {len(inst.tasks)} tasks, nodes "
                  f"{'+'.join(str(inst.n[q]) for q in inst.robots)}, {g.n_edges} cross "
                  f"entries ({g.n_precedence_edges} precedence), rigid makespan "
                  f"{inst.makespan}, rigid delay margin {g.delay_margin}")
            guarded(f"[1] {inst.name}", test_structure, inst, g)
            guarded(f"[2] {inst.name}", test_zero_delay, inst, g, sim, oracle_mod)
            guarded(f"[3/4] {inst.name}", test_equivalence_and_delays, inst, g,
                    args.traces, sim)
    if args.runs:
        print("== two-robot runs: byte identity")
        guarded("[6] runs", test_runs, args.runs)

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed, {n_fail} failed "
          f"({time.time() - t_start:.0f}s)")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAIL {name}: {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
