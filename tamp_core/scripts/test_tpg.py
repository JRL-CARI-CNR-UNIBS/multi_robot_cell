#!/usr/bin/env python3
"""Test battery for the temporal plan graph (``tpg.py``). No pytest.

    .venv_vamp/bin/python scripts/test_tpg.py [--quick] [--runs DIR ...]

PASS/FAIL lines and a summary; exit status 1 if anything failed. Synthetic instances
(``mu`` built by hand, seeded) run in a second; the archived runs recompute ``mu`` with the
CURRENT VAMP engine (the SIMD kernel) and rebuild the graph from it, so every test on a run
compares things built from one and the same ``mu`` -- the archived ``tpg.json`` is only
compared for information (its ``mu`` may predate an engine change).

1. **Reduction equivalence.** ``tpg.build`` keeps ONE edge per node, the closed-form
   maximum over nominally-earlier colliding nodes (+1). This builds the graph APEX-MR
   builds -- one edge per colliding pair, each oriented by the nominal instants computed
   from absolute slots, plus one per precedence milestone condition, no reduction -- and
   executes it by checking EVERY incoming edge (``np.all``), never a maximum. The two
   graphs are run by one executor on the zero-delay trace and on seeded stall-burst traces;
   the executed traces (both robots' node at every tick) must be identical. It is an
   independent check of the reduction because nothing in the full graph is derived from
   ``_tighten`` (no delta mask, no reversed argmax, no base offset) and because the
   executor never forms a maximum: equality of traces is exactly the claim that the
   dropped edges were implied by the kept one and the robot's own node order.
2. **Zero-delay optimality.** For a parallel optimum (B runs) the graph's zero-delay
   makespan equals the coordination-diagram optimum of ``coordinate.py`` on the same
   paths, within 1 slot; for turn-based (A runs) it is >= the optimum. Numbers reported.
   On the synthetic instances: zero-delay >= optimum always (the graph's execution is a
   walk of the diagram), and the executed trace never enters a blocked cell.
3. **Adversarial delays.** For every milestone (start, pick, place) of every scheduled
   task, and for either robot: a stall of 2x the longest task that begins exactly when the
   milestone's robot reaches it. 0 collisions (``mu``), 0 out of order
   (``simulate_tpg.Order``), 0 deadlocks.
4. **Guards.** A robot without tasks (one or both) gives a defined graph; two tasks of one
   robot at the same ``start_slot`` (or overlapping) are a clear ``ValueError``.
5. **Version.** ``tpg.json`` carries ``builder_version`` and ``precedence_edges_expected``;
   ``check_fresh`` accepts a fresh graph and refuses a stale or tampered one.
6. **Cycle check.** On margin-1 schedules ``build`` refuses exactly the instances whose
   unreduced graph deadlocks.
7. **Offset dilation** (2026-10-26). On seeded random N-robot (2..4) instances with
   pipeline / gate / hold precedences and tight schedules, a plan with ``offset_dilation:
   1`` (collisions >= 2 slots from the scheduled offset) never has a cycle, with and
   without collisions; the undilated control (1 slot off) does. Unit cases: the new
   build-time assertions refuse a margin-1 collision, a precedence that does not advance
   the acquire instant, and a hold reverse edge that is not strictly forward.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from typing import Callable, Dict, List, Sequence, Tuple

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import tpg as T  # noqa: E402

RUNS_DIR = os.path.join(PKG, "artifacts", "runs")
DEFAULT_RUNS = ["rep_tower_interchangeable_A_1", "rep_tower_interchangeable_B_1",
                "rep_tower_wall_A_1", "rep_tower_wall_B_1", "tower_verify_transit"]

RESULTS: List[Tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    sys.stdout.flush()
    return ok


# --------------------------------------------------------------------------- #
# instances
# --------------------------------------------------------------------------- #
class Instance:
    """A schedule, its mu, its seam; enough to build and to execute a graph."""

    def __init__(self, name, robots, solution, mu, problem, policy=None, art=None, task=None):
        self.name, self.robots, self.solution = name, tuple(robots), solution
        self.mu = mu                 # {(task_r, task_s): (K_r, K_s) bool}
        self.problem, self.policy, self.art, self.task = problem, policy, art, task

    def mu_of(self, r, ti, s, tj):
        return self.mu[(ti, tj)]


def synthetic(seed: int, n_tasks=(3, 3), with_prec=True, margin: int = 2) -> Instance:
    """Random two-robot schedule with hand-built blob mu, consistent by construction:
    no pair collides within ``margin - 1`` slots of its scheduled offset (``margin=1``:
    only the scheduled offset itself is free, which is all a solver guarantees), and a
    task's first and last samples (HOME) collide with nothing."""
    rng = np.random.default_rng(seed)
    robots = ("ra", "rb")
    assign, lens = {}, {}
    for q, nt in zip(robots, n_tasks):
        t = int(rng.integers(0, 15))
        for i in range(nt):
            name = f"{q}_t{i}"
            n = int(rng.integers(8, 40))
            assign[name] = {"robot": q, "start_slot": t, "end_slot": t + n}
            lens[name] = n
            t += n + int(rng.integers(0, 6))
    tasks = {q: [k for k, a in assign.items() if a["robot"] == q] for q in robots}
    mu = {}
    for ti in tasks["ra"]:
        for tj in tasks["rb"]:
            m = np.zeros((lens[ti], lens[tj]), dtype=bool)
            for _ in range(int(rng.integers(0, 4))):
                a0, b0 = int(rng.integers(0, lens[ti])), int(rng.integers(0, lens[tj]))
                m[a0:a0 + int(rng.integers(1, 12)), b0:b0 + int(rng.integers(1, 12))] = True
            m[0, :] = m[-1, :] = False
            m[:, 0] = m[:, -1] = False
            delta = assign[tj]["start_slot"] - assign[ti]["start_slot"]
            k = np.arange(lens[ti])[:, None]
            l = np.arange(lens[tj])[None, :]
            m &= np.abs(k - l - delta) >= margin
            mu[(ti, tj)] = m
    pick = {f"{a['robot']}|{t}": int(rng.integers(1, lens[t] // 2)) for t, a in assign.items()}
    place = {f"{a['robot']}|{t}": int(rng.integers(lens[t] // 2, lens[t] - 1))
             for t, a in assign.items()}
    precs, modes, sprecs = [], [], []
    if with_prec:
        # Only precedences the nominal schedule satisfies, cross- and same-robot.
        abs_ = lambda t, w: assign[t]["start_slot"] + (0 if w == "start" else
                                                       (pick if w == "pick" else place)[f"{assign[t]['robot']}|{t}"])
        names = list(assign)
        for _ in range(12):
            i, j = rng.choice(names, 2, replace=False)
            if abs_(i, "pick") <= abs_(j, "start") and abs_(i, "place") <= abs_(j, "place"):
                if rng.random() < 0.5:
                    precs.append([str(i), str(j)])
                    modes.append("pipeline")
                else:
                    sprecs.append([str(i), str(j)])        # via slot_precedences
            elif abs_(i, "place") <= abs_(j, "pick"):
                precs.append([str(i), str(j)])
                modes.append("gate")
    problem = {"precedences": precs, "precedence_modes": modes, "pick_offsets": pick,
               "place_offsets": place}
    if sprecs:
        problem["slot_precedences"] = sprecs
        problem["slot_of"] = {t: t for t in assign}
    return Instance(f"synthetic#{seed}", robots, {"assignments": assign}, mu, problem)


def real(run: str, kernel_required=True) -> Instance:
    import yaml
    import vamp
    from vamp_collision_engine import (CELL_BASE, CELL_MARGIN, CELL_MOUNT_YAW,
                                       CELL_N_STRUCTURAL, ObjectGeom, VampCollisionEngine)
    from vamp_link_groups import DEFAULT_SPHERIZED_URDF, link_groups
    from mu_kernel import MuKernel
    d = os.path.join(RUNS_DIR, run)
    art = json.load(open(os.path.join(d, "tamp_trajectories.json")))
    sol = json.load(open(os.path.join(d, "tamp_solution.json")))
    prob = json.load(open(os.path.join(d, "tamp_problem.json")))
    task = [os.path.join(d, f) for f in os.listdir(d) if f.startswith("tamp_task") and f.endswith(".yaml")][0]
    mk = getattr(ObjectGeom, "from_yaml", None)
    objects = {o["id"]: (mk(o) if mk else ObjectGeom.from_size(o["size"]))
               for o in yaml.safe_load(open(task))["objects"]}
    engine = VampCollisionEngine(
        getattr(vamp, "ur10e_rail"), objects, base_transforms=CELL_BASE,
        mount_yaws=CELL_MOUNT_YAW, n_structural=CELL_N_STRUCTURAL, sphere_margin=CELL_MARGIN,
        groups=link_groups(DEFAULT_SPHERIZED_URDF, CELL_N_STRUCTURAL))
    kern = MuKernel()
    if kernel_required and not kern.available:
        raise RuntimeError("SIMD kernel missing: scripts/build_mu_kernel.sh")
    robots = list(art["robots"])
    trs = {(t["robot"], t["task"]): t for t in art["trajectories"]}
    segs = T.timelines(sol, robots)
    packed = {}
    for q in robots:
        for g in segs[q]:
            t = trs[(q, g.task)]
            packed[g.task] = kern.pack(engine.traj_spheres(q, t["positions"], t["object_state"],
                                                           t["object"]),
                                       "A" if q == robots[0] else "B")
    mu = {(gi.task, gj.task): np.asarray(kern.matrix(packed[gi.task], packed[gj.task]), dtype=bool)
          for gi in segs[robots[0]] for gj in segs[robots[1]]}
    policy = sol.get("policy")
    try:
        policy = policy or json.load(open(os.path.join(d, "plan.json"))).get("policy")
    except OSError:
        pass
    return Instance(run, robots, sol, mu, prob, policy=policy, art=art, task=task)


# --------------------------------------------------------------------------- #
# the unreduced graph (APEX-MR's, before transitive reduction)
# --------------------------------------------------------------------------- #
class FullGraph:
    """Every colliding pair an edge; every precedence milestone condition an edge.

    ``req[q][m]`` is the array of other-robot node indices that must ALL have been
    reached before ``q`` enters node ``m``. Built from absolute nominal instants
    (``start_slot + k``), with the "+1 = has left" rule for geometry and "has reached" for
    precedences (the TPG docstring's two conventions), and nothing from ``tpg._tighten``.
    """

    def __init__(self, inst: Instance):
        r, s = inst.robots
        self.robots = (r, s)
        # Own layout of the node sequence, from the assignments (not tpg.timelines).
        lay = {q: [] for q in (r, s)}
        for task, a in inst.solution["assignments"].items():
            lay[a["robot"]].append((a["start_slot"], task, a["end_slot"] - a["start_slot"]))
        self.base, self.slot0, self.n = {}, {}, {}
        for q in (r, s):
            cur = 0
            for st, task, n in sorted(lay[q]):
                self.base[task], self.slot0[task] = cur, st
                cur += n
            self.n[q] = cur
        self.robot_of = {t: a["robot"] for t, a in inst.solution["assignments"].items()}
        waiters = {q: [] for q in (r, s)}
        needs = {q: [] for q in (r, s)}
        self.n_geom = 0
        for (ti, tj), m in inst.mu.items():
            ks, ls = np.nonzero(m)
            if ks.size == 0:
                continue
            tr = self.slot0[ti] + ks            # absolute nominal instant of r's sample
            ts = self.slot0[tj] + ls
            if np.any(tr == ts):
                raise AssertionError(f"{ti} x {tj}: collision at a simultaneous instant")
            s_waits = tr < ts                  # r's colliding config comes first: s waits
            waiters[s].append(self.base[tj] + ls[s_waits])
            needs[s].append(self.base[ti] + ks[s_waits] + 1)
            waiters[r].append(self.base[ti] + ks[~s_waits])
            needs[r].append(self.base[tj] + ls[~s_waits] + 1)
            self.n_geom += int(ks.size)
        self.n_prec = 0
        prob = inst.problem
        if prob is not None:
            def node(task, which):
                q = self.robot_of[task]
                off = 0 if which == "start" else int(prob[f"{which}_offsets"][f"{q}|{task}"])
                return q, self.base[task] + off
            for i, j, mode in T.expand_precedences(prob, list(self.robot_of)):
                conds = (("pick", "start"), ("place", "place")) if mode == "pipeline" \
                    else (("place", "pick"),)
                for wi, wj in conds:
                    (qi, mi), (qj, mj) = node(i, wi), node(j, wj)
                    if qi == qj:
                        continue
                    waiters[qj].append(np.array([mj]))
                    needs[qj].append(np.array([mi]))
                    self.n_prec += 1
        self.req = {}
        for q in (r, s):
            w = np.concatenate(waiters[q]) if waiters[q] else np.zeros(0, dtype=np.int64)
            v = np.concatenate(needs[q]) if needs[q] else np.zeros(0, dtype=np.int64)
            order = np.argsort(w, kind="stable")
            w, v = w[order], v[order]
            cuts = np.searchsorted(w, np.arange(self.n[q] + 1))
            self.req[q] = [v[cuts[m]:cuts[m + 1]] for m in range(self.n[q])]

    def allows(self, q: str, m: int, other_reached: int) -> bool:
        need = self.req[q][m]
        return bool(np.all(other_reached >= need)) if need.size else True


# --------------------------------------------------------------------------- #
# one executor for every graph
# --------------------------------------------------------------------------- #
def execute(robots, n: Dict[str, int], allows: Callable[[str, int, int], bool],
            stall: Callable[[int, str, Dict[str, int]], bool] | None = None,
            max_ticks: int | None = None) -> Tuple[List[Tuple[int, int]], bool]:
    """``simulate_tpg.run_tpg``'s rule: each tick each robot, in order, advances one node
    iff not stalled and its entry is allowed by what the other has reached (the second
    robot sees the first's move). Returns the per-tick positions and a deadlock flag."""
    r, s = robots
    other = {r: s, s: r}
    reached = {r: -1, s: -1}
    trace = []
    t = 0
    limit = max_ticks or 50 * (n[r] + n[s] + 10)
    while reached[r] < n[r] - 1 or reached[s] < n[s] - 1:
        t += 1
        if t > limit:
            return trace, True
        moved, stalled_any = False, False
        for q in (r, s):
            nxt = reached[q] + 1
            if nxt >= n[q]:
                continue
            if stall is not None and stall(t, q, reached):
                stalled_any = True
                continue
            if allows(q, nxt, reached[other[q]]):
                reached[q] = nxt
                moved = True
        trace.append((reached[r], reached[s]))
        if not moved and not stalled_any:
            return trace, True
    return trace, False


def reduced_allows(g: T.TPG):
    def f(q, m, other_reached):
        d = int(g.deps[q][m])
        return d == T.FREE or other_reached >= d
    return f


def burst_stall(seed: int, robots, p: float, burst: int):
    """The trace generator of simulate_tpg._delay_trace, drawn lazily per tick."""
    rng = np.random.default_rng(seed)
    remaining = {q: 0 for q in robots}
    rows: Dict[int, Dict[str, bool]] = {}

    def row(t):
        while len(rows) < t:
            rr = {}
            for q in robots:
                if remaining[q] == 0 and rng.random() < p:
                    remaining[q] = burst
                rr[q] = remaining[q] > 0
                if remaining[q] > 0:
                    remaining[q] -= 1
            rows[len(rows) + 1] = rr
        return rows[t]
    return lambda t, q, reached: row(t)[q]


class Lookup:
    """Global node -> (task, sample) per robot, for mu lookups at executed positions."""

    def __init__(self, g: T.TPG, inst: Instance):
        self.g, self.inst = g, inst
        self.tab = {}
        for q in g.robots:
            tasks, ks = [], []
            for seg in g.segments[q]:
                tasks += [seg.task] * seg.n
                ks += list(range(seg.n))
            self.tab[q] = (tasks, ks)

    def collides(self, a: int, b: int) -> bool:
        r, s = self.g.robots
        if a < 0 or b < 0:
            return False
        ti, k = self.tab[r][0][a], self.tab[r][1][a]
        tj, l = self.tab[s][0][b], self.tab[s][1][b]
        return bool(self.inst.mu[(ti, tj)][k, l])


def audit(trace, look: Lookup, order) -> Tuple[int, int]:
    """(colliding ticks by mu, out-of-order conditions by simulate_tpg.Order)."""
    r, s = look.g.robots
    order.reset()
    coll = 0
    for t, (a, b) in enumerate(trace, start=1):
        order.observe(t, {r: a, s: b})
        coll += look.collides(a, b)
    return coll, order.violations()


def order_checker(g: T.TPG, problem: dict):
    from simulate_tpg import Order
    return Order(g, problem)


# --------------------------------------------------------------------------- #
# the tests
# --------------------------------------------------------------------------- #
def test_equivalence(inst: Instance, g: T.TPG, n_traces: int) -> None:
    t0 = time.time()
    full = FullGraph(inst)
    n = {q: g.n_nodes(q) for q in g.robots}
    if n != full.n:
        record(f"[1] {inst.name}: node layout", False, f"reduced {n} vs full {full.n}")
        return
    look = Lookup(g, inst)
    order = order_checker(g, inst.problem) if inst.problem else None
    runs = [("zero-delay", None)]
    longest = max((seg.n for q in g.robots for seg in g.segments[q]), default=10)
    for sd in range(n_traces):
        p = [0.0005, 0.002, 0.005, 0.02][sd % 4] if inst.art is not None else [0.02, 0.05, 0.1, 0.2][sd % 4]
        burst = int([0.05, 0.2, 0.5, 1.0][(sd // 4) % 4] * longest) + 1
        runs.append((f"seed{sd}", (sd, p, burst)))
    diffs, colls, ooo, dead = [], 0, 0, 0
    for label, spec in runs:
        mk = (lambda: None) if spec is None else (lambda sp=spec: burst_stall(sp[0], g.robots, sp[1], sp[2]))
        tr_red, d1 = execute(g.robots, n, reduced_allows(g), mk())
        tr_full, d2 = execute(g.robots, n, full.allows, mk())
        if tr_red != tr_full or d1 != d2:
            first = next((i for i, (a, b) in enumerate(zip(tr_red, tr_full)) if a != b),
                         min(len(tr_red), len(tr_full)))
            diffs.append(f"{label}: first divergence at tick {first + 1}")
        c, o = audit(tr_red, look, order) if order else (audit_mu_only(tr_red, look), 0)
        colls += c
        ooo += o
        dead += d1
    zd = T.zero_delay_ticks(g)
    tr0, _ = execute(g.robots, n, reduced_allows(g))
    record(f"[1] {inst.name}: full vs reduced graph, {len(runs)} executions identical",
           not diffs and zd == len(tr0),
           f"full graph {full.n_geom} geometric + {full.n_prec} precedence edges, reduced "
           f"{g.n_edges} ({g.n_precedence_edges} precedence); zero-delay {zd} ticks; "
           f"{time.time() - t0:.1f}s" + (f"; DIVERGED: {diffs[:3]}" if diffs else "")
           + ("" if zd == len(tr0) else f"; zero_delay_ticks {zd} != executor {len(tr0)}"))
    record(f"[1] {inst.name}: those executions safe (mu), in order, deadlock-free",
           colls == 0 and ooo == 0 and dead == 0,
           f"{colls} colliding ticks, {ooo} out-of-order conditions, {dead} deadlocks")


def audit_mu_only(trace, look):
    return sum(look.collides(a, b) for a, b in trace)


def test_simulate_agrees(inst: Instance, g: T.TPG) -> None:
    """Our executor == simulate_tpg.run_tpg on the same trace (so [1]/[3] test THE executor)."""
    from simulate_tpg import Oracle, _delay_trace, run_tpg
    n = {q: g.n_nodes(q) for q in g.robots}
    horizon = 8 * sum(n.values())
    oracle = Oracle(g, inst.mu)
    order = order_checker(g, inst.problem)
    ok, notes = True, []
    for sd in range(3):
        rng = np.random.default_rng(sd)
        trace = _delay_trace(rng, horizon, g.robots, 0.002 if inst.art else 0.05, 40)
        res = run_tpg(g, oracle, order, trace)
        mine, dead = execute(g.robots, n, reduced_allows(g),
                             lambda t, q, reached, tr=trace: tr[t - 1][q] if t - 1 < len(tr) else False)
        ok &= res["ticks"] == len(mine) and res["collisions"] == 0 and not res["deadlock"]
        notes.append(f"{res['ticks']}/{len(mine)}")
    record(f"[1] {inst.name}: test executor == simulate_tpg.run_tpg", ok,
           "ticks (run_tpg/ours) " + ", ".join(notes))


def diagram_synthetic(inst: Instance, full: FullGraph) -> np.ndarray:
    """Blocked cells: collisions + precedences, with coordinate.diagram's semantics."""
    r, s = inst.robots
    blocked = np.zeros((full.n[r], full.n[s]), dtype=bool)
    for (ti, tj), m in inst.mu.items():
        a, b = full.base[ti], full.base[tj]
        blocked[a:a + m.shape[0], b:b + m.shape[1]] |= m
    n1 = np.arange(full.n[r])[:, None]
    n2 = np.arange(full.n[s])[None, :]
    prob = inst.problem
    for i, j, mode in T.expand_precedences(prob, list(full.robot_of)):
        conds = (("pick", "start"), ("place", "place")) if mode == "pipeline" else (("place", "pick"),)
        for wi, wj in conds:
            def ms(task, which):
                q = full.robot_of[task]
                extra = 1 if which == "start" else int(prob[f"{which}_offsets"][f"{q}|{task}"])
                return q, full.base[task] + extra
            (qi, mi), (qj, mj) = ms(i, wi), ms(j, wj)
            if qi == qj:
                continue
            reached_j = (n2 >= mj) if qj == s else (n1 >= mj)
            behind_i = (n1 < mi) if qi == r else (n2 < mi)
            blocked |= reached_j & behind_i
    return blocked


def test_optimality_synthetic(inst: Instance, g: T.TPG) -> None:
    from coordinate import UNREACHABLE, walk
    full = FullGraph(inst)
    blocked = diagram_synthetic(inst, full)
    Tm = walk(blocked)
    opt = int(Tm[-1, -1]) + 1 if Tm[-1, -1] < UNREACHABLE else None
    tr, _ = execute(g.robots, {q: g.n_nodes(q) for q in g.robots}, reduced_allows(g))
    zd = len(tr)
    entered = sum(bool(blocked[a, b]) for a, b in tr if a >= 0 and b >= 0)
    record(f"[2] {inst.name}: zero-delay >= diagram optimum, no blocked cell entered",
           opt is not None and zd >= opt and entered == 0,
           f"zero-delay {zd}, optimum {opt}, blocked cells entered {entered}")


def test_optimality_real(inst: Instance, g: T.TPG) -> None:
    from coordinate import UNREACHABLE, diagram, walk
    order = {q: [seg.task for seg in g.segments[q]] for q in g.robots}
    t0 = time.time()
    coll, prec, _ = diagram(inst.art, inst.problem, order, inst.task)
    Tm = walk(coll | prec)
    opt = int(Tm[-1, -1]) + 1 if Tm[-1, -1] < UNREACHABLE else None
    zd = T.zero_delay_ticks(g)
    sched = int(inst.solution.get("makespan_slots", g.nominal_makespan))
    turn = (inst.policy or "").startswith("turn")
    if opt is None:
        record(f"[2] {inst.name}: diagram admits a walk", False, "no monotone walk")
        return
    ok = zd >= opt if turn else abs(zd - opt) <= 1
    record(f"[2] {inst.name} ({'turn-based' if turn else 'parallel'}): zero-delay "
           f"{'>=' if turn else '=='} diagram optimum" + ("" if turn else " (+-1)"), ok,
           f"zero-delay {zd}, diagram optimum {opt}, schedule makespan {sched}, "
           f"gap {zd - opt} slots; {time.time() - t0:.1f}s")


def test_adversarial(inst: Instance, g: T.TPG) -> None:
    t0 = time.time()
    prob = inst.problem
    n = {q: g.n_nodes(q) for q in g.robots}
    look = Lookup(g, inst)
    order = order_checker(g, prob)
    longest = max((seg.n for q in g.robots for seg in g.segments[q]), default=1)
    L = 2 * longest
    milestones = []
    for q in g.robots:
        for seg in g.segments[q]:
            for w in ("start", "pick", "place"):
                off = 0 if w == "start" else int(prob[f"{w}_offsets"][f"{q}|{seg.task}"])
                milestones.append((q, seg.start_node + off, f"{seg.task}.{w}"))
    bad = []
    runs = 0
    for q, node, label in milestones:
        for victim in g.robots:
            state = {"at": None}

            def stall(t, who, reached, q=q, node=node, victim=victim, state=state):
                # The stall window opens at the first tick at which `q` is seen at (or past)
                # the milestone node, and holds the victim for L ticks from there.
                if state["at"] is None and reached[q] >= node:
                    state["at"] = t
                return who == victim and state["at"] is not None and t - state["at"] < L
            tr, dead = execute(g.robots, n, reduced_allows(g), stall)
            c, o = audit(tr, look, order)
            runs += 1
            if c or o or dead:
                bad.append(f"{label} stall {victim}: {c} coll, {o} ooo, dead={dead}")
    record(f"[3] {inst.name}: {runs} adversarial stalls ({L} ticks at every milestone, "
           f"either robot)", not bad, (f"{len(bad)} failing: {bad[:4]}" if bad else
                                       f"0 collisions, 0 out of order, 0 deadlocks") +
           f"; {time.time() - t0:.1f}s")


def test_cycle_detection(n: int) -> None:
    """[6] With only the scheduled offset free (``margin=1``) a one-slot hole can close a
    nominally simultaneous 2-cycle. ``build`` must refuse exactly the instances whose
    UNREDUCED graph deadlocks at zero delay (an independent witness)."""
    agree, refused, total, bad = 0, 0, 0, []
    for sd in range(n):
        inst = synthetic(1000 + sd, n_tasks=(3, 3), margin=1)
        full = FullGraph(inst)
        _, dead = execute(inst.robots, full.n, full.allows)
        try:
            T.build(inst.solution, inst.robots, inst.mu_of, 0.01, problem=inst.problem)
            raised = False
        except AssertionError as e:
            raised = "cycle" in str(e)
        total += 1
        refused += raised
        if raised == dead:
            agree += 1
        else:
            bad.append(f"seed {1000 + sd}: build {'refused' if raised else 'accepted'}, "
                       f"full graph {'deadlocks' if dead else 'completes'}")
    record(f"[6] cycle check: build refuses exactly the deadlocking schedules ({total} "
           f"margin-1 instances)", agree == total,
           f"{refused} refused, all confirmed by the unreduced graph deadlocking" if not bad
           else "; ".join(bad[:4]))


def test_guards() -> None:
    def expect(name, fn, exc, needle):
        try:
            fn()
        except exc as e:
            return record(name, needle in str(e), f"{type(e).__name__}: {str(e)[:110]}")
        except Exception as e:  # noqa: BLE001
            return record(name, False, f"wrong exception {type(e).__name__}: {e}")
        return record(name, False, "no exception")

    mu = lambda r, ti, s, tj: np.zeros((5, 5), dtype=bool)  # noqa: E731
    one = {"assignments": {"a1": {"robot": "ra", "start_slot": 0, "end_slot": 5},
                           "a2": {"robot": "ra", "start_slot": 7, "end_slot": 12}}}
    offs = {"ra|a1": 2, "ra|a2": 2}
    g = T.build(one, ("ra", "rb"), mu, 0.01, problem={"precedences": [["a1", "a2"]],
                                                      "pick_offsets": offs,
                                                      "place_offsets": {"ra|a1": 3, "ra|a2": 3}})
    record("[4] one robot without tasks: defined graph",
           g.n_nodes("rb") == 0 and g.n_nodes("ra") == 10 and T.zero_delay_ticks(g) == 10
           and g.nominal_makespan == 12,
           f"nodes ra={g.n_nodes('ra')} rb={g.n_nodes('rb')}, zero-delay {T.zero_delay_ticks(g)}, "
           f"makespan {g.nominal_makespan}")
    g0 = T.build({"assignments": {}}, ("ra", "rb"), mu, 0.01, problem={"precedences": []})
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "g.json")
        g0.to_json(p)
        back = T.TPG.from_json(p)
    record("[4] both robots without tasks: empty graph, makespan 0",
           g0.n_nodes("ra") == g0.n_nodes("rb") == 0 and g0.nominal_makespan == 0
           and T.zero_delay_ticks(g0) == 0 and back.n_nodes("ra") == 0,
           f"makespan {g0.nominal_makespan}, zero-delay {T.zero_delay_ticks(g0)}, JSON round trip ok")
    same = {"assignments": {"a1": {"robot": "ra", "start_slot": 3, "end_slot": 8},
                            "a2": {"robot": "ra", "start_slot": 3, "end_slot": 8}}}
    expect("[4] same robot, same start_slot -> ValueError",
           lambda: T.build(same, ("ra", "rb"), mu, 0.01), ValueError, "both start at slot 3")
    overlap = {"assignments": {"a1": {"robot": "ra", "start_slot": 0, "end_slot": 5},
                               "a2": {"robot": "ra", "start_slot": 4, "end_slot": 9}}}
    expect("[4] same robot, overlapping tasks -> ValueError",
           lambda: T.build(overlap, ("ra", "rb"), mu, 0.01), ValueError, "before a1 ends")
    stranger = {"assignments": {"a1": {"robot": "rz", "start_slot": 0, "end_slot": 5}}}
    expect("[4] task on an unknown robot -> ValueError",
           lambda: T.build(stranger, ("ra", "rb"), mu, 0.01), ValueError, "not one of")


def test_version(inst: Instance | None) -> None:
    src = synthetic(3) if inst is None else inst
    g = T.build(src.solution, src.robots, src.mu_of, 0.01, problem=src.problem)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tpg.json")
        g.to_json(p)
        o = json.load(open(p))
        ok_fields = (o.get("builder_version") == T.BUILDER_VERSION and
                     o.get("precedence_edges_expected") == g.n_precedence_edges)
        record(f"[5] {src.name}: tpg.json carries builder_version and precedence_edges_expected",
               ok_fields, f"builder_version {o.get('builder_version')}, expected "
                          f"{o.get('precedence_edges_expected')}, n_precedence_edges "
                          f"{o['n_precedence_edges']}")
        try:
            T.check_fresh(p)
            T.check_fresh(p, problem=src.problem)
            record(f"[5] {src.name}: check_fresh accepts the fresh graph (also against its seam)", True)
        except ValueError as e:
            record(f"[5] {src.name}: check_fresh accepts the fresh graph", False, str(e))
        refused = []
        for label, mutate in (("no stamp", lambda o: o.pop("builder_version")),
                              ("older stamp", lambda o: o.__setitem__("builder_version", "2026-09-19")),
                              ("no expected count", lambda o: o.pop("precedence_edges_expected")),
                              ("edge count off", lambda o: o.__setitem__(
                                  "n_precedence_edges", o["n_precedence_edges"] + 1))):
            o2 = json.load(open(p))
            mutate(o2)
            try:
                T.check_fresh(o2)
                refused.append(f"{label}: ACCEPTED")
            except ValueError:
                pass
        record(f"[5] {src.name}: check_fresh refuses stale/tampered graphs", not refused,
               "; ".join(refused) or "no stamp, older stamp, no expected count, edge count off")


def dilation_instance(rng, gap: int, p: float, dilation: int):
    """Random N-robot (2..4) instance for [7]: tasks with milestones 0 < m0 < h <= m1, a
    random DAG of pipeline / gate / hold precedences, and a TIGHT schedule (earliest starts
    by fixed-point relaxation of every constraint, so many precedence conditions hold with
    equality: many equal-time precedence edges). ``mu`` is random with density ``p`` but
    keeps every colliding pair at least ``gap`` slots from the scheduled offset (``gap 2``
    = what a solve with dilation 1 guarantees; ``gap 1`` = only the scheduled offset
    free). ``None`` when the relaxation finds no schedule (a positive cycle through hold
    back-edges)."""
    n_rob = rng.choice([2, 3, 4])
    robots = [f"r{q}" for q in range(n_rob)]
    tasks = [f"t{i}" for i in range(rng.randint(4, 9))]
    rob = {t: rng.choice(robots) for t in tasks}
    K, m0, m1, h, e = {}, {}, {}, {}, {}
    for t in tasks:
        K[t] = rng.randint(8, 16)
        m0[t] = rng.randint(1, K[t] - 5)
        m1[t] = rng.randint(m0[t] + 1, K[t] - 2)
        h[t] = rng.randint(m0[t] + 1, m1[t])
        e[t] = rng.randint(m0[t] + 1, K[t] - 1)
    order = tasks[:]
    rng.shuffle(order)
    prec, modes = [], []
    for a in range(len(tasks)):
        for b in range(a + 1, len(tasks)):
            if rng.random() < 0.35:
                i, j = order[a], order[b]
                modes.append(rng.choice(["pipeline", "gate", "hold"] if rob[i] != rob[j]
                                        else ["pipeline", "gate"]))
                prec.append([i, j])
    rank = {t: order.index(t) for t in tasks}
    seq = {q: sorted([t for t in tasks if rob[t] == q], key=lambda t: rank[t] + rng.random() * 3)
           for q in robots}
    S = {t: 0 for t in tasks}
    for _ in range(500):
        old = dict(S)
        for q in robots:
            for a, b in zip(seq[q], seq[q][1:]):
                S[b] = max(S[b], S[a] + K[a])
        for (i, j), mode in zip(prec, modes):
            if mode == "pipeline":
                S[j] = max(S[j], S[i] + m0[i], S[i] + m1[i] - m1[j])
            elif mode == "gate":
                S[j] = max(S[j], S[i] + m1[i] - m0[j])
            else:   # hold: S_j >= alpha_i, chi_i <= alpha_j, omega_j <= beta_i
                S[j] = max(S[j], S[i] + m0[i], S[i] + h[i] - m0[j])
                S[i] = max(S[i], S[j] + e[j] - m1[i])
        if S == old:
            break
    else:
        return None
    if any(v > 10_000 for v in S.values()):
        return None
    sol = {"assignments": {t: {"robot": rob[t], "start_slot": S[t], "end_slot": S[t] + K[t]}
                           for t in tasks}}
    if dilation:
        sol["offset_dilation"] = dilation
    problem = {"precedences": prec, "precedence_modes": modes,
               "pick_offsets": {f"{rob[t]}|{t}": m0[t] for t in tasks},
               "place_offsets": {f"{rob[t]}|{t}": m1[t] for t in tasks},
               "hold_offsets": {f"{rob[t]}|{t}": h[t] for t in tasks},
               "process_end_offsets": {f"{rob[t]}|{t}": e[t] for t in tasks}}
    mus = {}

    def mu_of(r, ti, s, tj):
        if (ti, tj) not in mus:
            k = np.arange(K[ti])[:, None]
            l = np.arange(K[tj])[None, :]
            m = (np.array([[rng.random() < p for _ in range(K[tj])] for _ in range(K[ti])])
                 & (np.abs(k - l - (S[tj] - S[ti])) >= gap))
            m[0, :] = m[-1, :] = False      # a robot resting at HOME blocks nobody
            m[:, 0] = m[:, -1] = False
            mus[(ti, tj)] = m
        return mus[(ti, tj)]

    return sol, robots, mu_of, problem


def test_dilation(n: int) -> None:
    """[7] Offset dilation (2026-10-26): acyclicity by construction.

    Stress: ``n`` random N-robot instances per variant (seeded, see
    :func:`dilation_instance`). With every collision >= 2 slots from the scheduled offset
    and ``offset_dilation: 1`` -- precedences only, and with collisions -- every graph
    builds: 0 cycles, 0 other refusals, and equal-time precedence edges are present (so
    the case the argument has to handle is exercised). The control with collisions 1 slot
    off and no dilation must hit cycles, which proves the stress is sensitive. Units: a
    dilated toy schedule passes; ``offset_dilation: 1`` with a margin-1 collision, a
    precedence that does not advance the acquire instant, and a hold whose reverse edge
    is not strictly forward are refused by name; ``offset_dilation`` round-trips through
    ``tpg.json`` and is absent there for an undilated plan."""
    import random

    def run(label, gap, p, dilation, seed):
        rng = random.Random(seed)
        st = {"built": 0, "cycle": 0, "other": 0, "skipped": 0}
        eq, geo, first = 0, 0, ""
        for _ in range(n):
            inst = dilation_instance(rng, gap, p, dilation)
            if inst is None:
                st["skipped"] += 1
                continue
            sol, robots, mu_of, problem = inst
            try:
                g = T.build(sol, robots, mu_of, 0.01, problem=problem)
            except AssertionError as ex:
                st["cycle" if "cycle" in str(ex) else "other"] += 1
                first = first or str(ex)[:160]
                continue
            st["built"] += 1
            geo += g.n_edges
            for q in g.robots:
                for o in g.others(q):
                    dep = g.deps_on[q][o]
                    for node in np.flatnonzero(dep != T.FREE):
                        eq += T._instant(g, o, int(dep[node])) == T._instant(g, q, int(node))
        return st, eq, geo, first

    t0 = time.time()
    for label, p in (("precedences only", 0.0), ("with collisions", 0.3)):
        st, eq, geo, first = run(label, 2, p, 1, 7)
        record(f"[7] dilation 1, {label}: 0 cycles in {st['built'] + st['cycle'] + st['other']} "
               f"N-robot graphs", st["cycle"] == 0 and st["other"] == 0 and st["built"] > 50
               and eq > 0,
               f"{st}, {eq} equal-time edges, {geo} stored edges" + (f"; {first}" if first else ""))
    st, eq, geo, first = run("control", 1, 0.3, 0, 7)
    record("[7] control (collisions 1 slot off, undilated): the cycle check fires",
           st["cycle"] > 0, f"{st}")
    print(f"INFO  [7] stress {time.time() - t0:.1f}s")

    def expect(name, fn, needle):
        try:
            fn()
        except AssertionError as ex:
            return record(name, needle in str(ex), str(ex)[:120])
        except Exception as ex:  # noqa: BLE001
            return record(name, False, f"wrong exception {type(ex).__name__}: {ex}")
        return record(name, False, "no exception")

    def toy(cell, dilation, picks=(1, 3)):
        sol = {"assignments": {"a": {"robot": "ra", "start_slot": 0, "end_slot": 8},
                               "b": {"robot": "rb", "start_slot": 0, "end_slot": 8}},
               "offset_dilation": dilation}
        mu = np.zeros((8, 8), dtype=bool)
        mu[cell] = True
        prob = {"precedences": [["a", "b"]], "precedence_modes": ["gate"],
                "pick_offsets": {"ra|a": picks[0], "rb|b": picks[1]},
                "place_offsets": {"ra|a": 3, "rb|b": 5}}
        return lambda: T.build(sol, ("ra", "rb"), lambda *_: mu, 0.01, problem=prob)

    # delta = 0: cell (k, l) is |k - l| slots from the scheduled offset. Gate a -> b needs
    # pick[b] >= place[a] = 3; picks (1, 3) advance the acquire instant, (3, 3) do not.
    g = toy((5, 3), 1, picks=(1, 3))()
    with tempfile.TemporaryDirectory() as d:
        pth = os.path.join(d, "tpg.json")
        g.to_json(pth)
        o = json.load(open(pth))
        back = T.TPG.from_json(pth)
        g0 = toy((5, 3), 0, picks=(1, 3))()
        g0.to_json(pth)
        o0 = json.load(open(pth))
    record("[7] dilated toy schedule (collision 2 slots off) builds; offset_dilation in tpg.json",
           g.delay_margin == 2 and o.get("offset_dilation") == 1 and back.offset_dilation == 1
           and "offset_dilation" not in o0,
           f"margin {g.delay_margin}, json {o.get('offset_dilation')}, undilated json "
           f"{'has' if 'offset_dilation' in o0 else 'lacks'} the key")
    expect("[7] offset_dilation 1 with a margin-1 collision -> refused",
           toy((4, 3), 1, picks=(1, 3)), "violates its dilated forbidden offsets")
    try:
        toy((4, 3), 0, picks=(1, 3))()
        record("[7] same margin-1 collision, undilated plan: built as before", True)
    except Exception as ex:  # noqa: BLE001
        record("[7] same margin-1 collision, undilated plan: built as before", False, str(ex))
    expect("[7] offset_dilation 1, precedence not advancing the acquire instant -> refused",
           toy((5, 3), 1, picks=(3, 3)), "does not strictly advance the acquire")
    hold_sol = {"assignments": {"i": {"robot": "ra", "start_slot": 0, "end_slot": 10},
                                "j": {"robot": "rb", "start_slot": 0, "end_slot": 10}},
                "offset_dilation": 1}
    hold_prob = {"precedences": [["i", "j"]], "precedence_modes": ["hold"],
                 "pick_offsets": {"ra|i": 1, "rb|j": 2}, "place_offsets": {"ra|i": 5, "rb|j": 6},
                 "hold_offsets": {"ra|i": 2, "rb|j": 3},
                 "process_end_offsets": {"ra|i": 9, "rb|j": 8}}
    expect("[7] hold whose reverse edge is not strictly forward -> refused",
           lambda: T.build(hold_sol, ("ra", "rb"), lambda *_: np.zeros((10, 10), bool), 0.01,
                           problem=hold_prob), "not strictly forward")


def survey_archived() -> None:
    """INFO: which archived graphs lack precedence edges the seam calls for."""
    stale = []
    total = 0
    for d in sorted(os.listdir(RUNS_DIR)):
        p = os.path.join(RUNS_DIR, d)
        try:
            o = json.load(open(os.path.join(p, "tpg.json")))
            prob = json.load(open(os.path.join(p, "tamp_problem.json")))
        except (OSError, ValueError):
            continue
        total += 1
        segs = {r: [T.Segment(s["task"], s["start_node"], s["n"], s["start_slot"])
                    for s in o["segments"][r]] for r in o["robots"]}
        try:
            exp = T.expected_precedence_edges(prob, segs, o["robots"])
        except (KeyError, ValueError) as e:
            stale.append(f"{d}(seam mismatch: {str(e)[:40]})")
            continue
        if exp != int(o.get("n_precedence_edges", 0)):
            stale.append(f"{d}({o.get('n_precedence_edges', 0)}/{exp})")
    print(f"INFO  archived graphs whose precedence edges differ from the seam: {len(stale)} of "
          f"{total}: {', '.join(stale)}")


def compare_archived(inst: Instance, g: T.TPG) -> None:
    p = os.path.join(RUNS_DIR, inst.name, "tpg.json")
    try:
        old = T.TPG.from_json(p)
    except OSError:
        return
    same = all(np.array_equal(old.deps[q], g.deps[q]) for q in g.robots)
    diff = {q: int((old.deps[q] != g.deps[q]).sum()) for q in g.robots if old.deps[q].shape == g.deps[q].shape}
    print(f"INFO  {inst.name}: rebuilt graph {'==' if same else '!='} archived tpg.json "
          f"(deps differing per robot {diff}; archived {old.n_edges} edges / "
          f"{old.n_precedence_edges} precedence, zero-delay {T.zero_delay_ticks(old)}; rebuilt "
          f"{g.n_edges} / {g.n_precedence_edges}, zero-delay {T.zero_delay_ticks(g)})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--synthetic", type=int, default=12, help="number of synthetic instances")
    ap.add_argument("--traces", type=int, default=20, help="random delay traces per instance")
    ap.add_argument("--dilation-n", type=int, default=600,
                    help="random N-robot instances per variant of the [7] dilation stress")
    ap.add_argument("--quick", action="store_true", help="synthetic and guards only")
    args = ap.parse_args(argv)
    t_start = time.time()

    def guarded(name, fn, *a):
        try:
            fn(*a)
        except Exception as e:  # noqa: BLE001
            record(name, False, f"{type(e).__name__}: {e}")
            traceback.print_exc()

    print("== guards and version")
    guarded("[4] guards", test_guards)
    guarded("[5] version", test_version, None)
    guarded("[6] cycle check", test_cycle_detection, 40)
    guarded("[7] offset dilation", test_dilation, args.dilation_n)
    print("== synthetic instances")
    for sd in range(args.synthetic):
        inst = synthetic(sd, n_tasks=((3, 3), (2, 4), (4, 1), (3, 2))[sd % 4])
        try:
            g = T.build(inst.solution, inst.robots, inst.mu_of, 0.01, problem=inst.problem)
        except Exception as e:  # noqa: BLE001
            record(f"{inst.name}: build", False, f"{type(e).__name__}: {e}")
            continue
        guarded(f"[1] {inst.name}", test_equivalence, inst, g, args.traces)
        guarded(f"[2] {inst.name}", test_optimality_synthetic, inst, g)
        guarded(f"[3] {inst.name}", test_adversarial, inst, g)
    if not args.quick:
        print("== archived runs (mu recomputed with the current engine)")
        survey_archived()
        for run in args.runs:
            try:
                t0 = time.time()
                inst = real(run)
                g = T.build(inst.solution, inst.robots, inst.mu_of,
                            float(inst.art["delta_t"]), problem=inst.problem)
                print(f"INFO  {run}: mu + graph in {time.time() - t0:.1f}s; nodes "
                      f"{g.n_nodes(g.robots[0])}+{g.n_nodes(g.robots[1])}, policy {inst.policy}")
            except Exception as e:  # noqa: BLE001
                record(f"{run}: load + build", False, f"{type(e).__name__}: {e}")
                traceback.print_exc()
                continue
            compare_archived(inst, g)
            guarded(f"[1] {run}", test_simulate_agrees, inst, g)
            guarded(f"[1] {run}", test_equivalence, inst, g, args.traces)
            guarded(f"[2] {run}", test_optimality_real, inst, g)
            guarded(f"[3] {run}", test_adversarial, inst, g)
            guarded(f"[5] {run}", test_version, inst)

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed, {n_fail} failed "
          f"({time.time() - t_start:.0f}s)")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAIL {name}: {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
