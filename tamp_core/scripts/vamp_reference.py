"""Reference (numpy) implementation of the robot-vs-robot collision test.

This is what :mod:`mu_kernel` replaced. It is kept, deliberately, for two reasons:

1. **It is the oracle.** The SIMD kernel is validated by requiring it to produce exactly
   the same forbidden-offset sets as this module (self-test check [5], and a
   byte-identical seam on the full scene). A fast implementation with no independent
   reference is a fast implementation nobody can check.
2. **It is the fallback.** ``collision_generator_vamp.py`` uses it automatically when
   ``libmu_kernel.so`` is absent or will not load, and on demand via ``--no-kernel``.
   A machine without a compiler still runs the pipeline -- roughly 40x slower.

The two algorithmic ideas of ADR-0005 Phase 2a live here and are mirrored in the kernel:

* the **per-link broad phase** (:func:`_group_overlap`), which rejects ~61 % of
  configuration pairs that a single whole-robot bounding sphere could not touch; and
* the **diagonal traversal with early exit** (:func:`forbidden_offsets_pair`), which
  exploits the fact that the seam consumes only ``D = {k - l}`` and never ``mu`` itself.

Both are EXACT: they change which work is done, never the answer. :func:`collision_matrix`
materialises ``mu`` in full and exists only as the exhaustive cross-check the self-test
diffs the other two against.
"""

from __future__ import annotations

from typing import List

import numpy as np

from vamp_collision_engine import DTYPE, TrajSpheres  # noqa: F401


def _group_overlap(
    ga_c: np.ndarray, ga_r: np.ndarray, gb_c: np.ndarray, gb_r: np.ndarray
) -> np.ndarray:
    """(S,) bool: for each aligned pair, do ANY two link-group bounds overlap?

    ``ga_*`` and ``gb_*`` are (S, G, 3)/(S, G) -- one entry per candidate config pair.
    Squared distances via the same matmul form as the narrow phase; the ``rsum > 0``
    guard drops the object slot when it is absent (``-inf`` radius), which the squared
    comparison alone would not, since ``(-inf)**2`` is ``+inf``.
    """
    a2 = np.einsum("sgk,sgk->sg", ga_c, ga_c)[:, :, None]          # (S,G,1)
    b2 = np.einsum("shk,shk->sh", gb_c, gb_c)[:, None, :]          # (S,1,G)
    d2 = a2 + b2 - 2.0 * np.matmul(ga_c, gb_c.transpose(0, 2, 1))  # (S,G,G)
    rsum = ga_r[:, :, None] + gb_r[:, None, :]                     # (S,G,G)
    return ((d2 <= rsum * rsum) & (rsum > 0.0)).any(axis=(1, 2))


def _narrow_collide(
    A: TrajSpheres, B: TrajSpheres, ks: np.ndarray, ls: np.ndarray, batch: int = 2048
) -> np.ndarray:
    """Exact sphere-set collision for S candidate sample pairs ``(ks[i], ls[i])``.

    Returns (S,) bool: pair collides iff SOME cross sphere-pair touches
    (``dist <= rA + rB``), mirroring the C++ ACM mask -- (A robot spheres + A object)
    x (B robot spheres + B object). The robot x robot block (the dominant cost) uses
    the ``|a-b|^2 = |a|^2 + |b|^2 - 2 a.b`` matmul form (BLAS-accelerated); the object
    is one sphere per side, checked with direct distances and masked by presence
    (``isfinite`` radius) so an absent object never registers. Batched to bound RAM.
    """
    n = A.centres.shape[1] - 1               # robot sphere count (last slot = object)
    rA = A.radii[0, :n]                       # robot radii are config-independent
    rB = B.radii[0, :n]
    rsum2 = (rA[:, None] + rB[None, :]) ** 2  # (n,n)

    S = ks.shape[0]
    out = np.zeros(S, dtype=bool)
    for lo in range(0, S, batch):
        hi = min(lo + batch, S)
        ka, la = ks[lo:hi], ls[lo:hi]
        ac = A.centres[ka, :n]                # (b,n,3)
        bc = B.centres[la, :n]

        # robot x robot: squared distances via matmul, compare to (rA+rB)^2.
        a2 = np.einsum("bik,bik->bi", ac, ac)[:, :, None]   # (b,n,1)
        b2 = np.einsum("bjk,bjk->bj", bc, bc)[:, None, :]   # (b,1,n)
        d2 = a2 + b2 - 2.0 * np.matmul(ac, bc.transpose(0, 2, 1))  # (b,n,n)
        coll = (d2 <= rsum2[None, :, :]).any(axis=(1, 2))   # (b,)

        # object(A) x robot(B)
        oa_c, oa_r = A.centres[ka, n], A.radii[ka, n]       # (b,3),(b,)
        pa = np.isfinite(oa_r)
        if pa.any():
            d = np.linalg.norm(oa_c[:, None, :] - bc, axis=2)          # (b,n)
            coll |= (d <= (oa_r[:, None] + rB[None, :])).any(axis=1) & pa

        # robot(A) x object(B)
        ob_c, ob_r = B.centres[la, n], B.radii[la, n]
        pb = np.isfinite(ob_r)
        if pb.any():
            d = np.linalg.norm(ob_c[:, None, :] - ac, axis=2)          # (b,n)
            coll |= (d <= (ob_r[:, None] + rA[None, :])).any(axis=1) & pb

        # object(A) x object(B)
        both = pa & pb
        if both.any():
            d = np.linalg.norm(oa_c - ob_c, axis=1)                    # (b,)
            coll |= (d <= (oa_r + ob_r)) & both

        out[lo:hi] = coll
    return out


def collision_matrix(A: TrajSpheres, B: TrajSpheres, use_broad: bool = True,
                     block: int = 16) -> np.ndarray:
    """The boolean ``mu[k,l]`` for one trajectory pair (A = robot r task i, B = s task j).

    ``mu[k,l]`` is True iff robot A at sample k collides with robot B at sample l.
    The per-link broad phase prunes the K x K grid; the exact narrow check then runs
    only on survivors -- identical in result to ``use_broad=False``, just faster.

    This is the reference/debug path and the one the self-test diffs against. The
    pipeline itself uses :func:`forbidden_offsets_pair`, which never materialises mu.
    ``block`` bounds the (block, Kj, G, G) broad-phase temporary.
    """
    Ki, Kj = A.K, B.K
    mu = np.zeros((Ki, Kj), dtype=bool)
    all_l = np.arange(Kj)

    for lo in range(0, Ki, block):
        hi = min(lo + block, Ki)
        ks = np.repeat(np.arange(lo, hi), Kj)
        ls = np.tile(all_l, hi - lo)
        if use_broad:
            keep = _group_overlap(A.gcen[ks], A.grad[ks], B.gcen[ls], B.grad[ls])
            ks, ls = ks[keep], ls[keep]
        if ks.size:
            hit = _narrow_collide(A, B, ks, ls)
            mu[ks[hit], ls[hit]] = True
    return mu


def forbidden_offsets(mu: np.ndarray) -> List[int]:
    """Exact reduction ``D = {k - l : mu[k,l]}`` (ADR-0001), sorted.

    Inlined (not imported from tamp_scheduler) so no vamp code path ever touches the
    solver package -- byte-identical logic to collisions.forbidden_offsets_from_matrices.
    """
    ks, ls = np.where(mu)
    return sorted(set((ks - ls).tolist()))


def forbidden_offsets_pair(A: TrajSpheres, B: TrajSpheres, chunk: int = 128) -> List[int]:
    """``D = {k - l : mu[k,l]}`` computed directly, without ever building mu.

    WHY DIAGONAL-MAJOR
    ------------------
    The seam only ever consumes ``D``; mu is scaffolding. An offset ``d`` belongs to
    ``D`` iff AT LEAST ONE pair on the diagonal ``k - l = d`` collides -- so once that
    diagonal has produced a hit, every remaining pair on it is wasted work. Walking
    diagonal-major makes that early exit expressible. Measured on the tower scene:
    511 of 1382 diagonals are forbidden, and skipping their tails cuts the pairs
    actually visited by **1.91x**.

    The result is exactly equal to ``forbidden_offsets(collision_matrix(A, B))`` -- the
    self-test asserts this -- and is produced in ascending ``d``, so it is already sorted.

    CHUNK SIZE SETS THE EARLY-EXIT GRANULARITY
    ------------------------------------------
    ``chunk`` is how far past the first collision on a diagonal we are still willing to
    compute. A hit anywhere in a chunk is only noticed after the whole chunk has been
    evaluated, so larger chunks overshoot. Measured on one trajectory pair (746x637,
    tower scene), counting the pairs actually reaching the narrow phase:

        chunk        32      64     128      256      512
        narrow pairs 38785   46497  47448    99841    180379
        time (f32)   4.52s   4.87s  3.97s    9.02s    14.81s

    Time tracks PAIRS, not bytes -- the same curve appears in float64 with the knee at
    the same chunk, which rules out a cache-residency explanation. Below 128 the work
    keeps falling but numpy's per-call overhead takes over (1732 narrow calls at 32 vs
    1015 at 128), so 128 is the minimum of that U.

    For scale: 47448 narrow pairs against the 475202-pair grid is a 10x reduction, of
    which the per-link broad phase and the diagonal early exit each supply roughly half.

    ``D`` is chunk-invariant (chunking is pure loop tiling), so this is safe to retune
    per machine; the sweep asserts the invariance.
    """
    Ki, Kj = A.K, B.K
    out: List[int] = []

    for d in range(-(Kj - 1), Ki):
        k0, k1 = max(0, d), min(Ki, Kj + d)
        if k1 <= k0:
            continue

        # Walk the diagonal in chunks, broad THEN narrow within each, so a hit near the
        # head abandons the rest of the diagonal without paying either phase for it.
        # Both slices are contiguous (l = k - d), so the broad phase indexes without
        # copying and only the survivors are ever gathered.
        for ka in range(k0, k1, chunk):
            kb = min(ka + chunk, k1)
            la, lb = ka - d, kb - d
            cand = np.flatnonzero(
                _group_overlap(A.gcen[ka:kb], A.grad[ka:kb], B.gcen[la:lb], B.grad[la:lb]))
            if cand.size and _narrow_collide(A, B, cand + ka, cand + la).any():
                out.append(d)
                break
    return out
