#!/usr/bin/env python3
"""Validate the fk -> broadcast mu MATH against brute force (ADR-0005).

This does NOT test cell geometry (the shipped ur5 is not the cell robot). It tests
that the vectorised sphere-broadcast collision verdict -- the core of the VAMP engine
-- equals a naive brute-force sphere-pair loop, EXACTLY, for two opposed ur5 across a
sweep of configurations. Two checks:

  1. broadcast verdict (min over cross sphere-pairs of dist-(rA+rB) <= 0)
     == brute force (nested python loop)                          -- must match exactly
  2. broad-phase-accelerated collision_matrix()
     == no-broad-phase collision_matrix()                         -- broad phase is sound

Run:  .venv_vamp/bin/python scripts/test_vamp_fk_broadcast.py
Exit 0 = all pass.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vamp_collision_engine import (  # noqa: E402
    AT_SPAWN,
    ObjectGeom,
    VampCollisionEngine,
    collision_matrix,
    yaw_translation,
)


def brute_force_collide(cA, rA, cB, rB) -> bool:
    """Naive reference: True iff SOME sphere of A touches SOME sphere of B."""
    for a in range(len(rA)):
        for b in range(len(rB)):
            d = math.dist(cA[a], cB[b])
            if d <= rA[a] + rB[b]:
                return True
    return False


def broadcast_collide(cA, rA, cB, rB) -> bool:
    """The engine's vectorised verdict, standalone."""
    cA = np.asarray(cA); cB = np.asarray(cB)
    dist = np.linalg.norm(cA[:, None, :] - cB[None, :, :], axis=2)  # (nA,nB)
    rsum = np.asarray(rA)[:, None] + np.asarray(rB)[None, :]
    return bool((dist <= rsum).any())


def main() -> int:
    import vamp
    r = vamp.ur5
    rng = np.random.default_rng(7)

    # Two opposed ur5, base B facing base A. Separation chosen so the sweep produces
    # BOTH colliding and clear configs (exercises both branches).
    Tb = yaw_translation(0.0, 0.55, 0.0, math.pi)

    def spheres(module, q, T=None):
        s = module.fk(list(q))
        c = np.array([[p.x, p.y, p.z] for p in s], dtype=np.float64)
        rad = np.array([p.r for p in s], dtype=np.float64)
        if T is not None:
            c = c @ T[:3, :3].T + T[:3, 3]
        return c, rad

    n_trials = 300
    n_collide = 0
    for _ in range(n_trials):
        qa = rng.uniform(-math.pi, math.pi, r.dimension())
        qb = rng.uniform(-math.pi, math.pi, r.dimension())
        cA, rA = spheres(r, qa)
        cB, rB = spheres(r, qb, Tb)
        bf = brute_force_collide(cA, rA, cB, rB)
        bc = broadcast_collide(cA, rA, cB, rB)
        if bf != bc:
            print(f"FAIL: broadcast {bc} != brute force {bf}")
            return 1
        n_collide += int(bf)
    print(f"[1] broadcast == brute force on {n_trials} config pairs "
          f"({n_collide} collide, {n_trials - n_collide} clear) -- EXACT MATCH")
    if n_collide == 0 or n_collide == n_trials:
        print("FAIL: sweep did not exercise both branches (all same verdict)")
        return 1

    # -- broad phase soundness: accelerated mu == exhaustive mu ---------------- #
    engine = VampCollisionEngine(r, {"none": ObjectGeom.from_size([0.0, 0.0, 0.0])},
                                 base_transforms={"A": np.eye(4), "B": Tb})
    K = 40
    QA = rng.uniform(-math.pi, math.pi, (K, r.dimension()))
    QB = rng.uniform(-math.pi, math.pi, (K, r.dimension()))
    state = [AT_SPAWN] * K  # object detached -> pure robot-vs-robot
    A = engine.traj_spheres("A", QA.tolist(), state, "none")
    B = engine.traj_spheres("B", QB.tolist(), state, "none")

    mu_broad = collision_matrix(A, B, use_broad=True)
    mu_full = collision_matrix(A, B, use_broad=False)
    if not np.array_equal(mu_broad, mu_full):
        d = np.argwhere(mu_broad != mu_full)
        print(f"FAIL: broad-phase mu differs from exhaustive at {len(d)} cell(s), e.g. {d[:3].tolist()}")
        return 1
    print(f"[2] broad-phase mu == exhaustive mu on {K}x{K} grid "
          f"({int(mu_full.sum())} colliding cells) -- broad phase is SOUND")

    # Cross-check the mu diagonal against the standalone brute force verdict.
    for k in range(K):
        cA, rA = spheres(r, QA[k])
        cB, rB = spheres(r, QB[k], Tb)
        if brute_force_collide(cA, rA, cB, rB) != bool(mu_full[k, k]):
            print(f"FAIL: mu[{k},{k}] disagrees with brute force")
            return 1
    print(f"[3] mu diagonal == brute force on {K} paired configs -- MATCH")

    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
