#!/usr/bin/env python3
"""VAMP collision generator -- the ADR-0005 Phase-2 drop-in for the C++ FCL stage.

Reads the SAME ``tamp_trajectories.json`` the C++ ``collision_generator`` reads and
writes the SAME geometry-free seam (``tamp_problem.json``): durations, pick/place
offsets, precedences, and the forbidden offsets ``D = {k - l : mu[k,l]}`` per
cross-robot trajectory pair. The only difference from the C++ stage is the collision
engine underneath -- VAMP's ``fk`` -> world-frame spheres -> a sphere-set intersection
test, instead of MoveIt/FCL. No ``tamp_scheduler`` import (ADR-0002).

    .venv_vamp/bin/python scripts/collision_generator_vamp.py --task config/tamp_task_tower.yaml

Every path defaults into the persistent ``artifacts/`` dir
(``artifacts/tamp_trajectories.json`` -> ``artifacts/vamp/tamp_problem.json``), so the
task file is normally the only argument. Diff a run against the FCL reference with
``scripts/vamp_fcl_mu_diff.py``.

The collision test comes from :mod:`mu_kernel` (SIMD) when ``libmu_kernel.so`` is built,
and from :mod:`vamp_reference` (numpy) otherwise or under ``--no-kernel``. Both produce
the same seam, byte for byte.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, List, Tuple

# Pin BLAS/OpenMP to ONE thread per process BEFORE numpy is imported (via
# vamp_collision_engine below). We parallelise across trajectory pairs with an
# mp.Pool of `--jobs` workers; if each worker also let its numpy matmul spin up a
# full BLAS thread pool, that is jobs x n_cores threads oversubscribing the CPU,
# and the per-thread scratch buffers can exhaust RAM and hard-freeze the machine
# (seen on WSL2 with the tight tower scene). setdefault so an explicit env override
# still wins. This is why the pipeline is safe run directly, no env vars needed.
for _threadvar in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_threadvar, "1")

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vamp_collision_engine import (  # noqa: E402
    CELL_BASE,
    CELL_MARGIN,
    CELL_MOUNT_YAW,
    CELL_N_STRUCTURAL,
    PHASE_GRIP_CLOSE,
    PHASE_GRIP_OPEN,
    PHASE_PROCESS_OFF,
    PHASE_PROCESS_ON,
    ObjectGeom,
    TrajSpheres,
    VampCollisionEngine,
    HoldExemption,
    finger_patch_enabled,
    make_cell_engines,
)
from vamp_link_groups import DEFAULT_SPHERIZED_URDF, link_groups  # noqa: E402
from vamp_reference import forbidden_offsets_pair  # noqa: E402
from mu_kernel import MuKernel, PackedTraj  # noqa: E402

# Filled in the parent before the pool forks; workers inherit it copy-on-write.
_SPHERES: Dict[Tuple[str, str], TrajSpheres] = {}
# SIMD path: the same trajectories repacked into planes, plus the loaded kernel, keyed
# (robot, task, side). With two robots each trajectory is consumed from one side only
# (robot1 always A, robot2 always B); with more, a robot is A against the robots after it
# and B against those before, so it is packed on both sides -- decided once here.
_PACKED: Dict[Tuple[str, str, str], PackedTraj] = {}
_KERNEL: MuKernel | None = None
# Synchronous hold: the exemption of the hold task PAIRS (vamp_collision_engine.HoldExemption,
# shared with the plan graph) and the kernel packing of the exempted geometry, keyed
# (robot, task, side, mode). Empty without a `hold`: nothing changes.
_HOLD: HoldExemption | None = None
_PACKED_X: Dict[Tuple[str, str, str, str], PackedTraj] = {}


def load_objects(task_yaml: str) -> Dict[str, ObjectGeom]:
    with open(task_yaml) as f:
        root = yaml.safe_load(f)
    return {o["id"]: ObjectGeom.from_yaml(o) for o in root["objects"]}


_ACQUIRE = (PHASE_GRIP_CLOSE, PHASE_PROCESS_ON)
_RELEASE = (PHASE_GRIP_OPEN, PHASE_PROCESS_OFF)

# Fixed-point scale for the seam's `transit_distances` -- MUST match
# tamp_scheduler/model.py's `TRANSIT_SCALE` and collision_generator.cpp's own copy
# exactly, and MIRRORS collision_generator.cpp's `TRANSIT_SCALE`.
TRANSIT_SCALE = 1000


def _l1(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def transit_distance(home: List[float], pick_cfg: List[float], place_cfg: List[float]) -> int:
    """`SchedulingProblem.transit_distances`'s value for one (robot, task): the seam's
    own transit cost, NOT APEX-MR's formula (see model.py's `transit_distances`
    docstring for the full rationale -- summarised: two legs anchored at
    home->pick and pick->place, matching the topology of the
    home->pick->place->home trajectory this work actually plans, rather than
    APEX-MR's home->pick + home->place). L1 (sum of absolute per-DOF
    differences) over all 7 DOF (rail + 6 joints), scaled by TRANSIT_SCALE and
    rounded, same fixed-point convention `durations`/`delta_t` already use.
    MIRRORS collision_generator.cpp's `transitDistance` -- keep the two identical."""
    c = _l1(home, pick_cfg) + _l1(pick_cfg, place_cfg)
    return int(round(c * TRANSIT_SCALE))


def pick_offset(robot: str, task: str, phase: List[int]) -> int:
    """The acquire milestone m0: first GripClose (a pick) or ProcessOn (a weld strikes).

    Seam key stays ``pick_offsets`` so archived seams load; a pick-and-place value is
    unchanged. MIRRORS C++ ``pickOffset`` -- keep the two identical."""
    for k, p in enumerate(phase):
        if p in _ACQUIRE:
            return k
    raise ValueError(f"trajectory {robot}|{task} has no GripClose or ProcessOn sample "
                     f"-- no acquire milestone (m0)")


def place_offset(robot: str, task: str, phase: List[int], pick: int) -> int:
    """The release milestone m1: first GripOpen or ProcessOff at/after m0 (mirrors C++
    ``placeOffset``). A process task of several legs (a tack of two spots) cuts its arc once
    per leg: its m1 is the first sample of the LAST ProcessOff run (with one leg, the first
    ProcessOff sample, as always)."""
    if phase[pick] == PHASE_PROCESS_ON:
        offs = [k for k in range(pick, len(phase)) if phase[k] == PHASE_PROCESS_OFF]
        if offs:
            k = offs[-1]
            while k - 1 >= pick and phase[k - 1] == PHASE_PROCESS_OFF:
                k -= 1
            return k
    else:
        for k in range(pick, len(phase)):
            if phase[k] in _RELEASE:
                return k
    raise ValueError(f"trajectory {robot}|{task} has no GripOpen or ProcessOff sample at or "
                     f"after its m0 -- no release milestone (m1)")


def hold_offset(robot: str, task: str, positions: List[List[float]], place: int,
                hold_slots: int | None) -> int:
    """The hold milestone h of a holding pick-place (precedence mode ``hold``): m1 minus its
    ``hold_slots``, the first sample of the hold tail the generator puts before GripOpen.
    Checked, not trusted: every sample from h to m1 must be the place configuration (1e-6
    rad), since the solver lets the held process run from h on. Mirrors C++ ``holdOffset``."""
    if not hold_slots or hold_slots <= 0:
        raise ValueError(f"trajectory {robot}|{task} is the pick-place of a `hold` but carries "
                         f"no hold_slots")
    h = place - int(hold_slots)
    if h < 0:
        raise ValueError(f"trajectory {robot}|{task}: hold longer than the trajectory")
    ref = positions[place]
    for k in range(h, place):
        if max(abs(a - b) for a, b in zip(positions[k], ref)) > 1e-6:
            raise ValueError(f"trajectory {robot}|{task}: sample {k} of its hold tail is not at "
                             f"the place configuration")
    return h


def process_end_offset(robot: str, task: str, phase: List[int]) -> int:
    """The arc-end milestone e of a held process task: one past its last ProcessOff sample
    (mirrors C++ ``processEndOffset``)."""
    for k in range(len(phase) - 1, -1, -1):
        if phase[k] == PHASE_PROCESS_OFF:
            return k + 1
    raise ValueError(f"trajectory {robot}|{task} is held but has no ProcessOff sample")


def _pair_worker(args: Tuple[str, str, str, str]) -> Tuple[str, List[int]]:
    """One trajectory pair -> its forbidden offsets.

    The SIMD kernel and the numpy path compute the SAME set (the self-test asserts it),
    so which one runs is invisible to the seam -- it is purely a speed choice.
    """
    r, i, s, j = args
    xa = _HOLD.mode(r, i, s, j) if _HOLD else ""
    xb = _HOLD.mode(s, j, r, i) if _HOLD else ""
    if _KERNEL is not None and _KERNEL.available:
        offs = _KERNEL.forbidden_offsets(
            _PACKED_X[(r, i, "A", xa)] if xa else _PACKED[(r, i, "A")],
            _PACKED_X[(s, j, "B", xb)] if xb else _PACKED[(s, j, "B")])
    else:
        offs = forbidden_offsets_pair(
            _HOLD.spheres(_SPHERES[(r, i)], r, i, xa) if xa else _SPHERES[(r, i)],
            _HOLD.spheres(_SPHERES[(s, j)], s, j, xb) if xb else _SPHERES[(s, j)])
    return f"{r}|{i}|{s}|{j}", offs


def write_problem(
    path: str,
    delta_t: float,
    robots: List[str],
    tasks: List[str],
    precedences: List[List[str]],
    durations: Dict[str, int],
    picks: Dict[str, int],
    places: Dict[str, int],
    transit: Dict[str, int],
    forbidden: Dict[str, List[int]],
    chains: Dict[str, List[str]] | None = None,
    precedence_modes: List[str] | None = None,
    slots: Dict[str, object] | None = None,
    holds: Dict[str, Dict[str, int]] | None = None,
    eligibility: Dict[str, Dict[str, str]] | None = None,
) -> None:
    """Write the geometry-free seam, key-for-key compatible with the C++ writeProblem.

    Consumed by ``tamp_scheduler.artifact.load_problem`` via ``json.load``, so the
    parsed structure -- not byte layout -- is what must match; we still mirror the
    C++ key order and one-key-per-line offset maps for a clean textual diff."""
    obj = {
        "delta_t": delta_t,
        "robots": robots,
        "tasks": tasks,
        "precedences": [list(p) for p in precedences],
    }
    # Passed through verbatim, and only when the trajectory artifact has it: this stage
    # gives the modes no meaning, and an artifact from before they existed must produce
    # the same seam it always did (same rule as the C++ writer).
    if precedence_modes is not None:
        if len(precedence_modes) != len(precedences):
            raise ValueError(f"{len(precedence_modes)} precedence_modes for "
                             f"{len(precedences)} precedences")
        obj["precedence_modes"] = list(precedence_modes)
    # Interchangeable slots, passed through verbatim from the trajectory artifact and
    # only when it carries them (a scene with a `slots:` block). Four id-to-id
    # relations, no geometry. `slot_of`'s mere PRESENCE is what tells the solver this
    # is a slot problem -- objective="apex" and the Gurobi backends refuse one -- so an
    # ordinary scene must not carry it, and its seam stays byte-identical.
    for key in ("slot_of", "object_of", "slot_precedences", "slot_precedence_modes"):
        if slots and slots.get(key) is not None:
            obj[key] = slots[key]
    # auto eligibility (v3): what became of every candidate pair, passed through verbatim for
    # traceability, only when the trajectory artifact carries it (same rule as the C++ writer).
    if eligibility:
        obj["eligibility"] = eligibility
    obj |= {
        "durations": durations,
        "pick_offsets": picks,
        "place_offsets": places,
    }
    # Synchronous hold: `hold_offsets` (h of each holding pick-place trajectory) and
    # `process_end_offsets` (e of each held process trajectory), ONLY when the artifact has a
    # `hold` precedence -- every other seam is byte-identical (same rule as the C++ writer).
    if holds:
        obj |= {"hold_offsets": holds["hold_offsets"],
                "process_end_offsets": holds["process_end_offsets"]}
    obj |= {
        "transit_distances": transit,
        "forbidden_offsets": forbidden,
    }
    # A refined artifact's trajectories are CHAINED: each one begins where the previous
    # ended, so they are only valid in that order. Carrying it into the seam lets the
    # re-solve pin the order instead of being free to swap two tasks and leave the robot
    # starting a trajectory from a pose it is not standing in (ADR-0008).
    if chains:
        obj["chains"] = {r: list(seq) for r, seq in chains.items()}
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def hold_window_report(precedences, modes, robots, picks, places, hold_offs, pend_offs, forbidden):
    """For every `hold` (i, j): per (holder robot, process robot) candidate pair, how many start
    offsets delta = start[j] - start[i] the solver may use -- the hold constraints
    (start[j] >= m0[i], h[i] <= m0[j], e[j] <= m1[i]) intersected with the complement of the
    pair's forbidden offsets. Returns [(i, j, rh, rw, lo, hi, free)]."""
    out = []
    for (i, j), m in zip(precedences, modes or []):
        if m != "hold":
            continue
        for kh in [k for k in hold_offs if k.split("|")[1] == i]:
            rh = kh.split("|")[0]
            for kw in [k for k in pend_offs if k.split("|")[1] == j]:
                rw = kw.split("|")[0]
                lo = max(picks[kh], hold_offs[kh] - picks[kw])
                hi = places[kh] - pend_offs[kw]
                if robots.index(rh) < robots.index(rw):
                    D = set(forbidden.get(f"{rh}|{i}|{rw}|{j}", ()))
                    free = sum(1 for d in range(lo, hi + 1) if d not in D)
                else:
                    D = set(forbidden.get(f"{rw}|{j}|{rh}|{i}", ()))
                    free = sum(1 for d in range(lo, hi + 1) if -d not in D)
                out.append((i, j, rh, rw, lo, hi, free))
    return out


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.dirname(here)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj", default=os.path.join(pkg, "artifacts", "tamp_trajectories.json"),
                   help="trajectory artifact (same file the FCL stage reads)")
    p.add_argument("--task", default=os.path.join(pkg, "config", "tamp_task_tower.yaml"),
                   help="task yaml (object sizes) -- must match the scene the trajectory "
                        "artifact was generated from; tower is the current reference")
    p.add_argument("--out", default=os.path.join(pkg, "artifacts", "vamp", "tamp_problem.json"),
                   help="VAMP geometry-free seam (default artifacts/vamp/)")
    p.add_argument("--robot", default="ur10e_rail",
                   help="vamp robot module -- the codegen'd cell robot (vamp_codegen/)")
    p.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--sphere-margin", type=float, default=CELL_MARGIN,
                   help=f"robot-sphere safety margin (m) for soundness vs FCL "
                        f"(default {CELL_MARGIN}; see ADR-0005)")
    p.add_argument("--spherized-urdf", default=DEFAULT_SPHERIZED_URDF,
                   help="foam-spherized URDF the per-link broad-phase groups are read "
                        "from (its link order is the codegen sphere order)")
    p.add_argument("--no-kernel", action="store_true",
                   help="force the numpy path even when libmu_kernel.so is available "
                        "(for benchmarking and for diffing the two implementations)")
    args = p.parse_args(argv)

    import vamp
    if not hasattr(vamp, args.robot):
        print(f"vamp has no robot '{args.robot}'; available: {vamp.robots}", file=sys.stderr)
        return 1
    robot_module = getattr(vamp, args.robot)

    with open(args.traj) as f:
        art = json.load(f)
    delta_t = float(art["delta_t"])
    robots = list(art["robots"])
    tasks = list(art["tasks"])
    precedences = art["precedences"]
    homes = art["homes"]
    objects = load_objects(args.task)
    with open(args.task) as f:
        task_yaml = yaml.safe_load(f) or {}
    if str(task_yaml.get("cell") or "dual") == "dual":
        # The two-robot cell: exactly the engine this stage always built (and --robot /
        # --spherized-urdf keep their meaning), shared by both robots.
        if len(robots) != 2:
            print("the dual cell has exactly two robots", file=sys.stderr)
            return 1
        engine = VampCollisionEngine(
            robot_module, objects,
            base_transforms=CELL_BASE, mount_yaws=CELL_MOUNT_YAW,
            n_structural=CELL_N_STRUCTURAL, sphere_margin=args.sphere_margin,
            groups=link_groups(args.spherized_urdf, CELL_N_STRUCTURAL),
            finger_patch=finger_patch_enabled(False))
        engines = {r: engine for r in robots}
    else:
        # Any other cell: its layout picks base, mount yaw and VAMP module per robot
        # (vamp_collision_engine.make_cell_engines, ADR-0010).
        engines = make_cell_engines(vamp, task_yaml, objects, robots, sphere_margin=args.sphere_margin)
    for eng in {id(e): e for e in engines.values()}.values():
        who = [r for r in robots if engines[r] is eng]
        print(f"engine: {eng.robot.__name__.split('.')[-1]} for {', '.join(who)}  dim={eng.dim}  "
              f"n_spheres={eng.n_spheres}+{eng.n_patch} finger patch  sphere_margin={args.sphere_margin}  "
              f"broad-phase groups={eng.n_groups}")

    # -- durations / pick / place (copied straight from the artifact) + FK ----- #
    durations: Dict[str, int] = {}
    picks: Dict[str, int] = {}
    places: Dict[str, int] = {}
    transit: Dict[str, int] = {}
    modes = art.get("precedence_modes") or []
    holders = {p[0] for p, m in zip(precedences, modes) if m == "hold"}
    held = {p[1] for p, m in zip(precedences, modes) if m == "hold"}
    hold_offs: Dict[str, int] = {}
    pend_offs: Dict[str, int] = {}
    t_fk = time.time()
    for tr in art["trajectories"]:
        r, task = tr["robot"], tr["task"]
        key = f"{r}|{task}"
        phase = list(tr["phase"])
        positions = tr["positions"]
        durations[key] = len(positions)
        pk = pick_offset(r, task, phase)
        picks[key] = pk
        pl = place_offset(r, task, phase, pk)
        places[key] = pl
        transit[key] = transit_distance(homes[r], positions[pk], positions[pl])
        if task in holders:
            hold_offs[key] = hold_offset(r, task, positions, pl, tr.get("hold_slots"))
        if task in held:
            pend_offs[key] = process_end_offset(r, task, phase)
        _SPHERES[(r, task)] = engines[r].traj_spheres(r, tr["positions"], tr["object_state"], tr["object"])
    print(f"FK: {len(art['trajectories'])} trajectories in {time.time() - t_fk:.1f}s")
    global _HOLD
    _HOLD = HoldExemption(art)
    for (rr, task), (h, g, held) in _HOLD.tails.items():
        for j in sorted(held):
            hmodes = sorted({_HOLD.mode(rr, task, t["robot"], j) for t in art["trajectories"]
                            if t["task"] == j})
            print(f"hold: {rr}|{task} exempted on the hold tail [{h}, {g}) against {j} only: "
                  f"{'/'.join(hmodes)} ({'whole handler' if hmodes == ['full'] else 'part'} off)")

    # -- SIMD kernel (optional) ------------------------------------------------ #
    # A pair (r, s) takes r from side A and s from side B, r before s in `robots`: the
    # first robot is only ever A, the last only ever B, the others both. Every trajectory
    # is packed to ONE shape (the largest module's sphere and group counts), which the
    # kernel needs for any pair; with a single module that is each trajectory's own shape,
    # so the two-robot packing is exactly what it always was.
    global _KERNEL
    if not args.no_kernel:
        kernel = MuKernel()
        if kernel.available:
            t_pack = time.time()
            n_sph = max(ts.centres.shape[1] for ts in _SPHERES.values())
            n_grp = max(ts.gcen.shape[1] for ts in _SPHERES.values())
            for (rr, task), ts in _SPHERES.items():
                idx = robots.index(rr)
                for side in (("A",) if idx < len(robots) - 1 else ()) + (("B",) if idx > 0 else ()):
                    _PACKED[(rr, task, side)] = kernel.pack(ts, side, n_sph, n_grp)
                    for mode in (("object", "full") if (rr, task) in _HOLD.tails else ()):
                        _PACKED_X[(rr, task, side, mode)] = kernel.pack(
                            _HOLD.spheres(ts, rr, task, mode), side, n_sph, n_grp)
            _KERNEL = kernel
            print(f"engine: SIMD kernel, {kernel.simd_width} float32 lanes "
                  f"(packed in {time.time() - t_pack:.1f}s)")
        else:
            print(f"engine: numpy path -- no SIMD kernel at {kernel.path} "
                  f"(build it with scripts/build_mu_kernel.sh)")
    else:
        print("engine: numpy path (--no-kernel)")

    # -- mu -> forbidden offsets for every cross-robot pair (r=robots[0], s=[1]) - #
    # Only over trajectories the artifact actually contains. A refined artifact holds one
    # chain per robot rather than the full grid, so the pair list shrinks with it -- and
    # the resulting seam names one candidate robot per task, which is exactly the
    # allocation the refinement was planned from (ADR-0008).
    #
    # Mutually exclusive candidates are dropped from the pair list: the solver runs
    # exactly one candidate per slot and consumes each physical object at most once, so
    # two such tasks are never in one plan and there is no pair of simultaneous samples
    # to forbid an offset between (ADR-0003 addendum). Empty maps -- every scene without
    # `slots:` -- leave the full grid, so those seams are byte-identical.
    slot_of = art.get("slot_of") or {}
    object_of = art.get("object_of") or {}

    def exclusive(i: str, j: str) -> bool:
        if i in slot_of and j in slot_of and slot_of[i] == slot_of[j]:
            return True
        return i in object_of and j in object_of and object_of[i] == object_of[j]

    have = {(t["robot"], t["task"]) for t in art["trajectories"]}
    # Every robot pair r < s (in the artifact's robot order), every task pair; with two
    # robots this is the single (robot1, robot2) pair, key order unchanged. A task is
    # never run by two robots, so (r, i, s, i) is a real pair only for the SAME task
    # planned for two candidates -- kept, exactly as the two-robot stage always did.
    pairs = [(r, i, s, j) for a, r in enumerate(robots) for s in robots[a + 1:]
             for i in tasks for j in tasks
             if (r, i) in have and (s, j) in have and not exclusive(i, j)]
    t_mu = time.time()
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            results = pool.map(_pair_worker, pairs)
    else:
        results = [_pair_worker(pr) for pr in pairs]

    forbidden: Dict[str, List[int]] = {}
    total = 0
    for key, offs in results:
        if offs:
            forbidden[key] = offs
            total += len(offs)
        print(f"  {key}: {len(offs)} forbidden offsets")
    print(f"mu: {len(pairs)} pairs in {time.time() - t_mu:.1f}s, {total} forbidden offsets total")

    # Synchronous holds: does every hold leave the solver at least one (holder, welder) pair
    # with a start offset that is both inside its hold window and collision-free? None means
    # the plan is infeasible whatever the solver does -- say so here, naming the hold.
    if holders:
        rep = hold_window_report(precedences, modes, robots, picks, places, hold_offs,
                                 pend_offs, forbidden)
        print("hold windows (free start offsets per holder x process robot):")
        dead = []
        for (i, j) in dict.fromkeys((i, j) for i, j, *_ in rep):
            rows = [x for x in rep if x[0] == i and x[1] == j]
            print(f"  {i} -> {j}: " + ", ".join(
                f"{rh}x{rw} [{lo},{hi}] {free} free" for _, _, rh, rw, lo, hi, free in rows))
            if not any(x[6] > 0 for x in rows):
                dead.append(f"{i} -> {j}")
        if dead:
            print(f"hold windows: NO free offset for {', '.join(dead)} -- no welder can run the "
                  f"held process inside the hold without colliding; the plan is infeasible "
                  f"(raise planning.hold_slack_s, or check the process trajectories)", file=sys.stderr)
            return 3

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_problem(args.out, delta_t, robots, tasks, precedences, durations, picks, places,
                  transit, forbidden, art.get("chains"), art.get("precedence_modes"),
                  {k: art.get(k) for k in
                   ("slot_of", "object_of", "slot_precedences", "slot_precedence_modes")},
                  {"hold_offsets": hold_offs, "process_end_offsets": pend_offs} if holders else None,
                  art.get("eligibility"))
    print(f"wrote seam -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
