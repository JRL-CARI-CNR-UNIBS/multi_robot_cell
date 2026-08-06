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
    ObjectGeom,
    TrajSpheres,
    VampCollisionEngine,
)
from vamp_link_groups import DEFAULT_SPHERIZED_URDF, link_groups  # noqa: E402
from vamp_reference import forbidden_offsets_pair  # noqa: E402
from mu_kernel import MuKernel, PackedTraj  # noqa: E402

# Filled in the parent before the pool forks; workers inherit it copy-on-write.
_SPHERES: Dict[Tuple[str, str], TrajSpheres] = {}
# SIMD path: the same trajectories repacked into planes, plus the loaded kernel. Each
# trajectory is always consumed from the same side of the pair (robot1 is always A,
# robot2 always B), so the side-dependent sentinel packing can be decided once here.
_PACKED: Dict[Tuple[str, str], PackedTraj] = {}
_KERNEL: MuKernel | None = None


def load_objects(task_yaml: str) -> Dict[str, ObjectGeom]:
    with open(task_yaml) as f:
        root = yaml.safe_load(f)
    return {o["id"]: ObjectGeom.from_size(o["size"]) for o in root["objects"]}


def pick_offset(robot: str, task: str, phase: List[int]) -> int:
    """First GripClose sample -- the pick milestone (mirrors C++ pickOffset)."""
    for k, p in enumerate(phase):
        if p == PHASE_GRIP_CLOSE:
            return k
    raise ValueError(f"trajectory {robot}|{task} has no GripClose sample -- not pick-and-place")


def place_offset(robot: str, task: str, phase: List[int], pick: int) -> int:
    """First GripOpen at/after the pick -- the place milestone (mirrors C++ placeOffset)."""
    for k in range(pick, len(phase)):
        if phase[k] == PHASE_GRIP_OPEN:
            return k
    raise ValueError(f"trajectory {robot}|{task} has no GripOpen after its pick -- not pick-and-place")


def _pair_worker(args: Tuple[str, str, str, str]) -> Tuple[str, List[int]]:
    """One trajectory pair -> its forbidden offsets.

    The SIMD kernel and the numpy path compute the SAME set (the self-test asserts it),
    so which one runs is invisible to the seam -- it is purely a speed choice.
    """
    r, i, s, j = args
    if _KERNEL is not None and _KERNEL.available:
        offs = _KERNEL.forbidden_offsets(_PACKED[(r, i)], _PACKED[(s, j)])
    else:
        offs = forbidden_offsets_pair(_SPHERES[(r, i)], _SPHERES[(s, j)])
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
    forbidden: Dict[str, List[int]],
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
        "durations": durations,
        "pick_offsets": picks,
        "place_offsets": places,
        "forbidden_offsets": forbidden,
    }
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


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
    if len(robots) != 2:
        print("this stage assumes exactly two robots", file=sys.stderr)
        return 1

    objects = load_objects(args.task)
    engine = VampCollisionEngine(
        robot_module, objects,
        base_transforms=CELL_BASE, mount_yaws=CELL_MOUNT_YAW,
        n_structural=CELL_N_STRUCTURAL, sphere_margin=args.sphere_margin,
        groups=link_groups(args.spherized_urdf, CELL_N_STRUCTURAL))
    print(f"engine: vamp.{args.robot}  dim={engine.dim}  n_spheres={engine.n_spheres}  "
          f"sphere_margin={args.sphere_margin}  broad-phase groups={engine.n_groups}")

    # -- durations / pick / place (copied straight from the artifact) + FK ----- #
    durations: Dict[str, int] = {}
    picks: Dict[str, int] = {}
    places: Dict[str, int] = {}
    t_fk = time.time()
    for tr in art["trajectories"]:
        r, task = tr["robot"], tr["task"]
        key = f"{r}|{task}"
        phase = list(tr["phase"])
        durations[key] = len(tr["positions"])
        pk = pick_offset(r, task, phase)
        picks[key] = pk
        places[key] = place_offset(r, task, phase, pk)
        _SPHERES[(r, task)] = engine.traj_spheres(r, tr["positions"], tr["object_state"], tr["object"])
    print(f"FK: {len(art['trajectories'])} trajectories in {time.time() - t_fk:.1f}s")

    # -- SIMD kernel (optional) ------------------------------------------------ #
    # robots[0] is always the A side of every pair and robots[1] always the B side, so
    # each trajectory packs once, with the sentinel sign its side requires.
    global _KERNEL
    if not args.no_kernel:
        kernel = MuKernel()
        if kernel.available:
            t_pack = time.time()
            for (rr, task), ts in _SPHERES.items():
                _PACKED[(rr, task)] = kernel.pack(ts, "A" if rr == robots[0] else "B")
            _KERNEL = kernel
            print(f"engine: SIMD kernel, {kernel.simd_width} float32 lanes "
                  f"(packed in {time.time() - t_pack:.1f}s)")
        else:
            print(f"engine: numpy path -- no SIMD kernel at {kernel.path} "
                  f"(build it with scripts/build_mu_kernel.sh)")
    else:
        print("engine: numpy path (--no-kernel)")

    # -- mu -> forbidden offsets for every cross-robot pair (r=robots[0], s=[1]) - #
    r, s = robots[0], robots[1]
    pairs = [(r, i, s, j) for i in tasks for j in tasks]
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

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_problem(args.out, delta_t, robots, tasks, precedences, durations, picks, places, forbidden)
    print(f"wrote seam -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
