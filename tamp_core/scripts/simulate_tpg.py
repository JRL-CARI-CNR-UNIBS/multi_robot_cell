#!/usr/bin/env python3
"""Execute the TPG against injected delays, and check it never collides.

    .venv_vamp/bin/python scripts/simulate_tpg.py --delay 0.3 --trials 20

This is the experiment behind the delay-robustness claim, and it is deliberately
adversarial: the collision check does NOT consult the graph. At every tick it looks up
every robot pair's current configurations in ``mu`` directly, so a wrong edge shows up as
a collision rather than being masked by the same reasoning that produced it. Any number of
robots (the graph's); a tick with several colliding pairs counts once.

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
from vamp_collision_engine import HoldExemption, ObjectGeom, make_cell_engines  # noqa: E402
from mu_kernel import MuKernel  # noqa: E402
from tpg import FREE, TPG, expand_precedences, home_rest_ends, mask_home_rest  # noqa: E402


class Oracle:
    """Ground-truth collision lookup between the robots' current configurations.

    ``mu[(ti, tj)]`` is keyed by task pair, ``ti`` the task of the EARLIER robot of the
    pair in ``graph.robots`` order (task ids are unique across robots).
    """

    def __init__(self, graph: TPG, mu: Dict[Tuple[str, str], np.ndarray]):
        self.graph = graph
        self.mu = mu
        self.pairs = [(r, s) for a, r in enumerate(graph.robots) for s in graph.robots[a + 1:]]

    def collides(self, node_a: int, node_b: int, r: str | None = None,
                 s: str | None = None) -> bool:
        """Do robot ``r`` at ``node_a`` and robot ``s`` at ``node_b`` collide? ``r, s``
        default to the two robots of a two-robot graph; ``r`` must precede ``s``."""
        if r is None:
            r, s = self.graph.robots
        ti, k = self.graph.locate(r, node_a)
        tj, l = self.graph.locate(s, node_b)
        return bool(self.mu[(ti, tj)][k, l])

    def any_collision(self, pos: Dict[str, int]) -> bool:
        """Does any robot pair collide at positions ``pos`` (-1 = at home, never)?"""
        return any(pos[r] >= 0 and pos[s] >= 0 and self.collides(pos[r], pos[s], r, s)
                   for r, s in self.pairs)


class Order:
    """Ground-truth check that the TASK plan was executed in a legal order.

    Collision-freedom is not the whole contract. A run can keep the arms perfectly clear
    of one another and still be wrong -- picking a box while another still sits on top of
    it, or placing one where its support has not landed yet. That failure is invisible to
    ``mu``: the arms are in the right places, at the wrong times.

    So this watches the milestones directly, in REALISED ticks, and re-derives the
    scheduler's conditions per precedence ``(i, j)`` from scratch:
    ``start[j] >= pick[i]`` and ``place[i] <= place[j]`` for a ``pipeline`` pair,
    ``pick[j] >= place[i]`` for a ``gate`` pair (slot precedences included, resolved to the
    scheduled winners). Like :class:`Oracle` it never
    consults the graph, so a missing precedence edge shows up as a violation rather than
    being excused by the reasoning that produced the graph.
    """

    def __init__(self, graph: TPG, problem: dict):
        # Task pairs AND slot pairs resolved to their scheduled winners (tpg.expand_precedences).
        # Reading only `precedences` left this check blind on every scene with interchangeable
        # slots -- whose whole build order is in `slot_precedences` -- so it reported 0 out of
        # order for a graph that enforced no order at all (fixed 2026-09-19). It derives the
        # pairs from the seam and the graph's SCHEDULED TASKS only, never from the graph's
        # edges, so it stays an independent witness.
        self.precedences = expand_precedences(
            problem, [g.task for r in graph.robots for g in graph.segments[r]])
        self.milestone = {}
        # Synchronous hold milestones, only where the seam has them: `hold` = the handler's
        # h (part arrived and held still), `pend` = the process's e - 1 (its last ProcessOff
        # sample: the arc is out for good).
        hold, pend = problem.get("hold_offsets", {}), problem.get("process_end_offsets", {})
        for r in graph.robots:
            for g in graph.segments[r]:
                m = self.milestone[g.task] = {
                    "robot": r, "start": g.start_node,
                    "pick": g.start_node + int(problem["pick_offsets"][f"{r}|{g.task}"]),
                    "place": g.start_node + int(problem["place_offsets"][f"{r}|{g.task}"]),
                }
                if f"{r}|{g.task}" in hold:
                    m["hold"] = g.start_node + int(hold[f"{r}|{g.task}"])
                if f"{r}|{g.task}" in pend:
                    m["pend"] = g.start_node + int(pend[f"{r}|{g.task}"]) - 1

        # Milestone nodes per robot, ascending: a robot advances monotonically, so the
        # first-reach tick is found by walking a cursor rather than rescanning.
        self.watch = {r: sorted({m[w] for m in self.milestone.values()
                                 if m["robot"] == r for w in ("start", "pick", "place", "hold", "pend")
                                 if w in m})
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

    @staticmethod
    def conditions(i: str, j: str, mode: str):
        """The milestone conditions of one precedence, as ((which, task), (which, task)):
        the first must be reached no later than the second."""
        if mode == "pipeline":
            return ((("pick", i), ("start", j)), (("place", i), ("place", j)))
        if mode == "hold":
            # pick gate; the arc strikes on a held part; the part is let go after the arc
            return ((("pick", i), ("start", j)), (("hold", i), ("pick", j)),
                    (("pend", j), ("place", i)))
        return ((("place", i), ("pick", j)),)

    def early_releases(self) -> int:
        """Holds whose handler reached GripOpen BEFORE the held process's last arc-out sample
        -- the part let go while it is still being tacked. Must be 0."""
        bad = 0
        for i, j, mode in self.precedences:
            if mode != "hold":
                continue
            mi, mj = self.milestone[i], self.milestone[j]
            ta, tb = self.at.get((mj["robot"], mj["pend"])), self.at.get((mi["robot"], mi["place"]))
            if tb is not None and (ta is None or ta > tb):
                bad += 1
        return bad

    def details(self) -> List[tuple]:
        """Every violated condition as ``(i, j, mode, cond, lead)``, ``lead`` in ticks.

        ``lead`` is by how many ticks the dependent milestone (``b``) was reached BEFORE the
        milestone it should have waited for (``a``): ``lead = t(a) - t(b) > 0``.
        """
        def when(task, which):
            m = self.milestone[task]
            return self.at.get((m["robot"], m[which]))

        out = []
        for i, j, mode in self.precedences:
            for a, b in self.conditions(i, j, mode):
                ta, tb = when(a[1], a[0]), when(b[1], b[0])
                if ta is not None and tb is not None and ta > tb:
                    out.append((i, j, mode, f"{a[0]}[{i}]<={b[0]}[{j}]", ta - tb))
        return out

    def violations(self) -> int:
        """Count precedence conditions whose milestones came out in the wrong order."""
        def when(task, which):
            m = self.milestone[task]
            return self.at.get((m["robot"], m[which]))

        bad = 0
        for i, j, mode in self.precedences:
            # pipeline: start[j] >= pick[i] and place[i] <= place[j]
            # gate:     pick[j] >= place[i] -- the ONLY condition; it does not imply the
            #           pipeline pair, which a scene needing both lists as a second entry.
            #   hold:     start[j] >= pick[i], pick[j] >= hold[i], place[i] >= pend[j]
            for a, b in self.conditions(i, j, mode):
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
    robots = graph.robots
    reached = {q: -1 for q in robots}          # last node index reached
    n = {q: graph.n_nodes(q) for q in robots}
    collisions, ticks, blocked = 0, 0, 0
    order.reset()

    def finished() -> bool:
        return all(reached[q] == n[q] - 1 for q in robots)

    for stalls in trace:
        if finished():
            break
        ticks += 1
        moved = False
        for q in robots:
            nxt = reached[q] + 1
            if nxt >= n[q] or stalls[q]:
                continue
            if graph.ready(q, nxt, reached):
                reached[q] = nxt
                moved = True
            else:
                blocked += 1   # the graph actively held this robot back
        order.observe(ticks, reached)
        if not moved and not any(stalls.values()) and not finished():
            return {"deadlock": True, "ticks": ticks, "collisions": collisions,
                    "blocked": blocked, "out_of_order": order.violations(),
                    "early_release": order.early_releases(), "order_details": order.details()}
        if oracle.any_collision(reached):
            collisions += 1

    done = finished()
    return {"deadlock": not done, "ticks": ticks, "collisions": collisions,
            "blocked": blocked, "out_of_order": order.violations(),
            "early_release": order.early_releases(), "order_details": order.details()}


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
    robots = graph.robots
    stalled = {q: 0 for q in robots}
    # -1 for a robot the solver gave no tasks: it is finished before the run starts. That
    # is a legitimate schedule (an allocation objective with no balancing term produces
    # one), and it must not index off the end of an empty segment list.
    last = {q: (graph.segments[q][-1].start_slot + graph.segments[q][-1].n - 1
                if graph.segments[q] else -1) for q in robots}
    collisions, ticks = 0, 0
    order.reset()

    for t, stalls in enumerate(trace):
        vt = {q: t - stalled[q] for q in robots}
        if all(vt[q] >= last[q] for q in robots):
            break
        ticks += 1
        for q in robots:
            if stalls[q]:
                stalled[q] += 1
        pos = {q: _rigid_position(graph, q, t - stalled[q]) for q in robots}
        order.observe(ticks, pos)
        if oracle.any_collision(pos):
            collisions += 1

    done = all(t - stalled[q] >= last[q] for q in robots)
    return {"deadlock": not done, "ticks": ticks, "collisions": collisions, "blocked": 0,
            "out_of_order": order.violations(), "early_release": order.early_releases(),
            "order_details": order.details()}


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
    p.add_argument("--order-detail", action="store_true",
                   help="after the table, list each executor's violated precedence "
                        "conditions in the first trial of every delay rate, with the lead "
                        "in ticks (how early the dependent milestone came)")
    args = p.parse_args(argv)

    import vamp
    graph = TPG.from_json(args.tpg)
    art = json.load(open(args.traj))
    task_yaml = yaml.safe_load(open(args.task))
    objects = {o["id"]: ObjectGeom.from_yaml(o) for o in task_yaml["objects"]}
    # One engine per robot, from the scene's `cell:` (absent = the dual cell, as before).
    engines = make_cell_engines(vamp, task_yaml, objects, graph.robots, dual_module=args.robot)
    kern = MuKernel()

    robots = graph.robots
    trs = {(t["robot"], t["task"]): t for t in art["trajectories"]}
    spheres = {}

    def geom(q, task):
        if (q, task) not in spheres:
            t = trs[(q, task)]
            spheres[(q, task)] = engines[q].traj_spheres(q, t["positions"], t["object_state"],
                                                         t["object"])
        return spheres[(q, task)]

    # The same certified-HOME drop as the graph builder (tpg.mask_home_rest): an idle robot
    # sits at its last node, HOME, whose clearance the trajectory stage certified on the
    # exact geometry; mu's sphere conservatism there must not count as a collision.
    rest = home_rest_ends(art)
    # The synchronous-hold exemption of the seam, the same function (HoldExemption): the graph
    # must be checked against the mu the solver was given.
    hx = HoldExemption(art)
    mu: Dict[Tuple[str, str], np.ndarray] = {}
    for r, s in [(r, s) for a, r in enumerate(robots) for s in robots[a + 1:]]:
        for gi in graph.segments[r]:
            for gj in graph.segments[s]:
                A = hx.spheres(geom(r, gi.task), r, gi.task, hx.mode(r, gi.task, s, gj.task))
                B = hx.spheres(geom(s, gj.task), s, gj.task, hx.mode(s, gj.task, r, gi.task))
                ns = max(A.centres.shape[1], B.centres.shape[1])   # two modules: pad to the pair
                ng = max(A.gcen.shape[1], B.gcen.shape[1])
                m = kern.matrix(kern.pack(A, "A", n_sph=ns, n_grp=ng),
                                kern.pack(B, "B", n_sph=ns, n_grp=ng))
                mu[(gi.task, gj.task)], _ = mask_home_rest(
                    m, rest.get((r, gi.task), (False, False)), rest.get((s, gj.task), (False, False)))
    oracle = Oracle(graph, mu)
    order = Order(graph, json.load(open(args.problem)))

    horizon = 8 * sum(graph.n_nodes(q) for q in robots)
    dt = graph.delta_t
    print(f"TPG: {'+'.join(str(graph.n_nodes(q)) for q in robots)} nodes, "
          f"{graph.n_edges} edges; "
          f"{graph.n_precedence_edges} of them task precedences; "
          f"SCHEDULE makespan {graph.nominal_makespan} slots = "
          f"{graph.nominal_makespan * dt:.2f} s (the solver's, with idle gaps -- what `rigid` "
          f"replays; `tpg` at zero delay is what the graph really executes, below)\n")
    burst = args.burst or max(1, int(1.2 * graph.delay_margin)) if graph.delay_margin > 0 \
        else (args.burst or 40)
    print(f"rigid delay margin: {graph.delay_margin} slots = {graph.delay_margin * dt:.2f} s "
          f"-- how much differential delay the OLD executor absorbs before it collides.\n"
          f"The solver never optimised that number; it is whatever the makespan optimum left.\n")
    print(f"delay bursts of {burst} slots = {burst * dt:.2f} s\n")
    holds = any(m == "hold" for _, _, m in order.precedences)
    print(f"{'stall p':>8}  {'executor':<8}  {'makespan (s)':>13}  "
          f"{'collided':>10}  {'out of order':>13}  {'waits':>8}  {'deadlocks':>10}"
          + (f"  {'early release':>14}" if holds else ""))
    print("-" * (82 + (16 if holds else 0)))

    failures = 0
    for pd in args.delay:
        stats = {"rigid": [], "tpg": []}
        for trial in range(args.trials):
            rng = np.random.default_rng(args.seed + trial)
            trace = _delay_trace(rng, horizon, robots, pd, burst)
            stats["rigid"].append(run_rigid(graph, oracle, order, trace))
            stats["tpg"].append(run_tpg(graph, oracle, order, trace))
        for name in ("rigid", "tpg"):
            res = stats[name]
            bad = sum(1 for x in res if x["collisions"] > 0)
            ooo = sum(1 for x in res if x["out_of_order"] > 0)
            dead = sum(1 for x in res if x["deadlock"])
            mk = np.mean([x["ticks"] for x in res]) * dt
            early = sum(1 for x in res if x["early_release"] > 0)
            print(f"{pd:8.4f}  {name:<8}  {mk:13.2f}  {bad:>7}/{len(res):<3}  "
                  f"{ooo:>10}/{len(res):<3}  "
                  f"{int(np.mean([x['blocked'] for x in res])):>8}  {dead:>10}"
                  + (f"  {early:>11}/{len(res):<3}" if holds else ""))
            if name == "tpg" and (bad or dead or ooo or early):
                failures += 1
        if args.order_detail:
            for name in ("rigid", "tpg"):
                d = stats[name][0]["order_details"]
                worst = max((x[4] for x in d), default=0)
                print(f"  [{name}, trial 0] {len(d)} violated condition(s)"
                      + (f", worst lead {worst} ticks = {worst * dt:.2f} s" if d else ""))
                for i, j, mode, cond, lead in sorted(d, key=lambda x: -x[4])[:5]:
                    print(f"      {mode:<8} {cond}  early by {lead} ticks = {lead * dt:.2f} s")
        print()

    if failures:
        print(f"FAIL: the TPG collided, deadlocked, ran out of order or released a held part "
              f"early in "
              f"{failures} configuration(s)")
        return 1
    print("PASS: the TPG never collided, never ran the tasks out of order, and never "
          "deadlocked, at any delay rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
