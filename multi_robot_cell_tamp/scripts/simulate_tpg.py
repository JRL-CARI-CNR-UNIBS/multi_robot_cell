#!/usr/bin/env python3
"""Execute the TPG against injected delays, and check it never collides.

    .venv_vamp/bin/python scripts/simulate_tpg.py --delay 0.3 --trials 20

This is the experiment behind the delay-robustness claim, and it is deliberately
adversarial: the collision check does NOT consult the graph. At every tick it looks up the
two robots' current configurations in ``mu`` directly, so a wrong edge shows up as a
collision rather than being masked by the same reasoning that produced it.

Two executors are compared under identical delay traces:

* **rigid** -- what the pipeline does today. Each robot replays its trajectory one sample
  per slot from its scheduled start. A delayed robot simply falls behind, the realised
  offset drifts away from the one the solver approved, and nothing detects it.
* **tpg** -- a robot advances only when its incoming edge is satisfied. Delay costs time
  and nothing else.

The expected result is not that the TPG is faster: under delay it is usually slower,
because waiting is the mechanism. The result is that rigid execution *collides* and the
TPG does not.
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
from tpg import FREE, TPG  # noqa: E402


class Oracle:
    """Ground-truth collision lookup between two robots' current configurations."""

    def __init__(self, graph: TPG, mu: Dict[Tuple[str, str], np.ndarray]):
        self.graph = graph
        self.mu = mu

    def collides(self, node_a: int, node_b: int) -> bool:
        r, s = self.graph.robots
        ti, k = self.graph.locate(r, node_a)
        tj, l = self.graph.locate(s, node_b)
        return bool(self.mu[(ti, tj)][k, l])


class Order:
    """Ground-truth check that the TASK plan was executed in a legal order.

    Collision-freedom is not the whole contract. A run can keep the arms perfectly clear
    of one another and still be wrong -- picking a box while another still sits on top of
    it, or placing one where its support has not landed yet. That failure is invisible to
    ``mu``: the arms are in the right places, at the wrong times.

    So this watches the milestones directly, in REALISED ticks, and re-derives the
    scheduler's two conditions per precedence ``(i, j)`` from scratch:
    ``start[j] >= pick[i]`` and ``place[i] <= place[j]``. Like :class:`Oracle` it never
    consults the graph, so a missing precedence edge shows up as a violation rather than
    being excused by the reasoning that produced the graph.
    """

    def __init__(self, graph: TPG, problem: dict):
        self.precedences = problem.get("precedences", [])
        self.milestone = {}
        for r in graph.robots:
            for g in graph.segments[r]:
                self.milestone[g.task] = {
                    "robot": r, "start": g.start_node,
                    "pick": g.start_node + int(problem["pick_offsets"][f"{r}|{g.task}"]),
                    "place": g.start_node + int(problem["place_offsets"][f"{r}|{g.task}"]),
                }

        # Milestone nodes per robot, ascending: a robot advances monotonically, so the
        # first-reach tick is found by walking a cursor rather than rescanning.
        self.watch = {r: sorted({m[w] for m in self.milestone.values()
                                 if m["robot"] == r for w in ("start", "pick", "place")})
                      for r in graph.robots}

    def reset(self) -> None:
        self.at = {}                                      # (robot, node) -> first tick
        self.cursor = {r: 0 for r in self.watch}

    def observe(self, tick: int, position: Dict[str, int]) -> None:
        for r, node in position.items():
            watch, c = self.watch[r], self.cursor[r]
            while c < len(watch) and node >= watch[c]:
                self.at[(r, watch[c])] = tick
                c += 1
            self.cursor[r] = c

    def violations(self) -> int:
        """Count precedence conditions whose milestones came out in the wrong order."""
        def when(task, which):
            m = self.milestone[task]
            return self.at.get((m["robot"], m[which]))

        bad = 0
        for i, j in self.precedences:
            for a, b in ((("pick", i), ("start", j)), (("place", i), ("place", j))):
                ta, tb = when(a[1], a[0]), when(b[1], b[0])
                if ta is None or tb is None:
                    continue      # the run never got that far; incompleteness is reported
                bad += ta > tb
        return bad


def _delay_trace(rng, n_ticks: int, robots, p: float, burst: int) -> List[Dict[str, bool]]:
    """One stall/go decision per robot per tick, shared by both executors.

    Delays are modelled as BURSTS, not as independent per-tick coin flips: with
    probability ``p`` per tick a robot enters a stall lasting ``burst`` ticks. This is
    both the realistic failure (a controller hiccup, a gripper closing slowly, a comms
    stall) and the revealing one. Independent per-tick stalls slow both arms by the same
    expected factor, so the realised offset barely drifts and even an unsafe executor
    looks fine; a burst on ONE arm shifts the offset by ``burst`` slots and keeps it
    shifted, which is exactly the condition the rigid schedule was never checked against.

    Both executors consume the SAME trace, so any difference in outcome is the executor's
    doing, not the sampling.
    """
    remaining = {r: 0 for r in robots}
    trace = []
    for _ in range(n_ticks):
        row = {}
        for r in robots:
            if remaining[r] == 0 and rng.random() < p:
                remaining[r] = burst
            row[r] = remaining[r] > 0
            if remaining[r] > 0:
                remaining[r] -= 1
        trace.append(row)
    return trace


def run_tpg(graph: TPG, oracle: Oracle, order: Order, trace) -> dict:
    r, s = graph.robots
    reached = {r: -1, s: -1}          # last node index reached
    n = {q: graph.n_nodes(q) for q in (r, s)}
    collisions, ticks, blocked = 0, 0, 0
    order.reset()

    for stalls in trace:
        if reached[r] == n[r] - 1 and reached[s] == n[s] - 1:
            break
        ticks += 1
        moved = False
        for q in (r, s):
            o = graph.other(q)
            nxt = reached[q] + 1
            if nxt >= n[q] or stalls[q]:
                continue
            d = int(graph.deps[q][nxt])
            if d == FREE or reached[o] >= d:
                reached[q] = nxt
                moved = True
            else:
                blocked += 1   # the graph actively held this robot back
        order.observe(ticks, reached)
        if not moved and not any(stalls.values()) and (
                reached[r] < n[r] - 1 or reached[s] < n[s] - 1):
            return {"deadlock": True, "ticks": ticks, "collisions": collisions,
                    "blocked": blocked, "out_of_order": order.violations()}
        if reached[r] >= 0 and reached[s] >= 0 and oracle.collides(reached[r], reached[s]):
            collisions += 1

    done = reached[r] == n[r] - 1 and reached[s] == n[s] - 1
    return {"deadlock": not done, "ticks": ticks, "collisions": collisions,
            "blocked": blocked, "out_of_order": order.violations()}


def _rigid_position(graph: TPG, robot: str, vt: int) -> int:
    """Where a rigid replay has got to at its own (delay-shifted) clock ``vt``.

    Faithful to what the executor really does, which matters: each TASK starts at its own
    scheduled slot, so between two tasks the robot sits IDLE at home rather than running
    them back to back. Modelling the timeline as one contiguous block would invent an
    executor nobody has -- and a much less safe one, since it would start the next task
    early. Returns -1 before the first task has begun.
    """
    segs = graph.segments[robot]
    if not segs:
        return -1          # no tasks: parked at home for the whole run, never in the way
    if vt < segs[0].start_slot:
        return -1
    pos = segs[0].start_node
    for seg in segs:
        if vt >= seg.start_slot + seg.n:
            pos = seg.start_node + seg.n - 1      # finished: parked at its last node (home)
        elif vt >= seg.start_slot:
            return seg.start_node + (vt - seg.start_slot)
        else:
            break                                  # in the gap before this task: stay parked
    return pos


def run_rigid(graph: TPG, oracle: Oracle, order: Order, trace) -> dict:
    """Today's executor: replay the scheduled timeline, absorbing delay by falling behind.

    It consults nothing at runtime, so a stall shifts this robot's whole remaining timeline
    later while the other robot's keeps running -- which is exactly how the realised offset
    drifts away from the one the solver approved.
    """
    r, s = graph.robots
    stalled = {r: 0, s: 0}
    n = {q: graph.n_nodes(q) for q in (r, s)}
    # -1 for a robot the solver gave no tasks: it is finished before the run starts. That
    # is a legitimate schedule (an allocation objective with no balancing term produces
    # one), and it must not index off the end of an empty segment list.
    last = {q: (graph.segments[q][-1].start_slot + graph.segments[q][-1].n - 1
                if graph.segments[q] else -1) for q in (r, s)}
    collisions, ticks = 0, 0
    order.reset()

    for t, stalls in enumerate(trace):
        vt = {q: t - stalled[q] for q in (r, s)}
        if all(vt[q] >= last[q] for q in (r, s)):
            break
        ticks += 1
        for q in (r, s):
            if stalls[q]:
                stalled[q] += 1
        vt = {q: t - stalled[q] for q in (r, s)}
        a, b = _rigid_position(graph, r, vt[r]), _rigid_position(graph, s, vt[s])
        order.observe(ticks, {r: a, s: b})
        if a >= 0 and b >= 0 and oracle.collides(a, b):
            collisions += 1

    done = all(t - stalled[q] >= last[q] for q in (r, s))
    return {"deadlock": not done, "ticks": ticks, "collisions": collisions, "blocked": 0,
            "out_of_order": order.violations()}


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.dirname(here)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tpg", default=os.path.join(pkg, "artifacts", "vamp", "tpg.json"))
    p.add_argument("--traj", default=os.path.join(pkg, "artifacts", "tamp_trajectories.json"))
    p.add_argument("--problem", default=os.path.join(pkg, "artifacts", "vamp", "tamp_problem.json"))
    p.add_argument("--task", default=os.path.join(pkg, "config", "tamp_task_tower.yaml"))
    p.add_argument("--robot", default="ur10e_rail")
    p.add_argument("--delay", type=float, nargs="*", default=[0.0, 0.0005, 0.002, 0.005],
                   help="per-tick probability that a robot ENTERS a stall burst")
    p.add_argument("--burst", type=int, default=0,
                   help="stall duration in slots; 0 (default) picks 1.2x the schedule's "
                        "rigid delay margin, i.e. just enough to matter")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import vamp
    graph = TPG.from_json(args.tpg)
    art = json.load(open(args.traj))
    objects = {o["id"]: ObjectGeom.from_size(o["size"])
               for o in yaml.safe_load(open(args.task))["objects"]}
    engine = VampCollisionEngine(
        getattr(vamp, args.robot), objects,
        base_transforms=CELL_BASE, mount_yaws=CELL_MOUNT_YAW,
        n_structural=CELL_N_STRUCTURAL, sphere_margin=CELL_MARGIN,
        groups=link_groups(DEFAULT_SPHERIZED_URDF, CELL_N_STRUCTURAL))
    kern = MuKernel()

    r, s = graph.robots
    trs = {(t["robot"], t["task"]): t for t in art["trajectories"]}
    mu: Dict[Tuple[str, str], np.ndarray] = {}
    for gi in graph.segments[r]:
        for gj in graph.segments[s]:
            a, b = trs[(r, gi.task)], trs[(s, gj.task)]
            A = engine.traj_spheres(r, a["positions"], a["object_state"], a["object"])
            B = engine.traj_spheres(s, b["positions"], b["object_state"], b["object"])
            mu[(gi.task, gj.task)] = kern.matrix(kern.pack(A, "A"), kern.pack(B, "B"))
    oracle = Oracle(graph, mu)
    order = Order(graph, json.load(open(args.problem)))

    horizon = 8 * (graph.n_nodes(r) + graph.n_nodes(s))
    dt = graph.delta_t
    print(f"TPG: {graph.n_nodes(r)}+{graph.n_nodes(s)} nodes, {graph.n_edges} edges; "
          f"nominal makespan {graph.nominal_makespan} slots = "
          f"{graph.nominal_makespan * dt:.2f} s\n")
    burst = args.burst or max(1, int(1.2 * graph.delay_margin)) if graph.delay_margin > 0 \
        else (args.burst or 40)
    print(f"rigid delay margin: {graph.delay_margin} slots = {graph.delay_margin * dt:.2f} s "
          f"-- how much differential delay the OLD executor absorbs before it collides.\n"
          f"The solver never optimised that number; it is whatever the makespan optimum left.\n")
    print(f"delay bursts of {burst} slots = {burst * dt:.2f} s\n")
    print(f"{'stall p':>8}  {'executor':<8}  {'makespan (s)':>13}  "
          f"{'collided':>10}  {'out of order':>13}  {'waits':>8}  {'deadlocks':>10}")
    print("-" * 82)

    failures = 0
    for pd in args.delay:
        stats = {"rigid": [], "tpg": []}
        for trial in range(args.trials):
            rng = np.random.default_rng(args.seed + trial)
            trace = _delay_trace(rng, horizon, (r, s), pd, burst)
            stats["rigid"].append(run_rigid(graph, oracle, order, trace))
            stats["tpg"].append(run_tpg(graph, oracle, order, trace))
        for name in ("rigid", "tpg"):
            res = stats[name]
            bad = sum(1 for x in res if x["collisions"] > 0)
            ooo = sum(1 for x in res if x["out_of_order"] > 0)
            dead = sum(1 for x in res if x["deadlock"])
            mk = np.mean([x["ticks"] for x in res]) * dt
            print(f"{pd:8.4f}  {name:<8}  {mk:13.2f}  {bad:>7}/{len(res):<3}  "
                  f"{ooo:>10}/{len(res):<3}  "
                  f"{int(np.mean([x['blocked'] for x in res])):>8}  {dead:>10}")
            if name == "tpg" and (bad or dead or ooo):
                failures += 1
        print()

    if failures:
        print(f"FAIL: the TPG collided, deadlocked or ran out of order in "
              f"{failures} configuration(s)")
        return 1
    print("PASS: the TPG never collided, never ran the tasks out of order, and never "
          "deadlocked, at any delay rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
