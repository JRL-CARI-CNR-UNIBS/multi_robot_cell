"""ctypes binding + SoA packing for the robot-vs-robot SIMD kernel (src/mu_kernel.cc).

WHY ctypes AND NOT nanobind
---------------------------
``.venv_vamp`` has no nanobind, pybind11, Cython or setuptools -- vamp was built from
source elsewhere and only its runtime landed here. The whole surface is ONE function
taking flat float32 buffers, so a binding framework would be a new build dependency for
nothing. ctypes is stdlib and the build stays a single ``g++`` line.

WHY THE ARRAYS ARE REPACKED
---------------------------
The numpy engine stores spheres as ``(K, M, 3)`` -- xyz interleaved. SIMD wants
struct-of-arrays so that 8 consecutive spheres load as one vector, so this module
repacks to ``(K, 4, M)``: an x-plane, a y-plane, a z-plane and an r-plane per sample.
Cost is ~1 MB per trajectory, which is nothing against the runtime it buys.

THE PADDING CONTRACT (mirrored in mu_kernel.cc -- change both or neither)
------------------------------------------------------------------------
Lanes with no real sphere are parked at radius 0 and a centre ``+1e6`` for robot A /
``-1e6`` for robot B, so padded-vs-real and padded-vs-padded are ~1e6 m apart and can
never register a hit. This lets the kernel run without a single lane mask or tail
branch. It also retires the ``-inf`` radius the numpy path used for a detached carried
object: absent entries are simply parked the same way.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import List

import numpy as np

# Distance (metres) at which a padded/absent lane is parked. Large enough that no real
# geometry can reach it, small enough that its square (1e12) is far inside float32 range.
PARK = 1.0e6

_LIB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib", "libmu_kernel.so")

_F32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, flags="C_CONTIGUOUS")
_I32 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
_U8_2D = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS")


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


@dataclass
class PackedTraj:
    """One trajectory in the layout the kernel reads."""

    spheres: np.ndarray  # (K, 4, n_sph)     float32 -- x, y, z, r planes
    groups: np.ndarray   # (K, 4, n_grp_pad) float32 -- per-link bounding spheres
    n_sph: int
    n_grp_real: int
    n_grp_pad: int

    @property
    def K(self) -> int:
        return self.spheres.shape[0]


class MuKernel:
    """The compiled kernel, or a graceful absence.

    ``available`` is False when the ``.so`` was never built or will not load (no AVX2,
    wrong ABI, ...). Callers fall back to the numpy path, which computes exactly the
    same ``D`` -- just slower -- so nothing downstream can tell the difference.
    """

    def __init__(self, path: str = _LIB_PATH):
        self.path = path
        self.available = False
        self.simd_width = 0
        self._lib = None
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            return

        lib.mu_kernel_simd_width.restype = ctypes.c_int
        lib.mu_kernel_simd_width.argtypes = []
        lib.mu_forbidden_offsets.restype = ctypes.c_int
        lib.mu_forbidden_offsets.argtypes = [
            _F32, _F32, ctypes.c_int,        # A spheres, A groups, KA
            _F32, _F32, ctypes.c_int,        # B spheres, B groups, KB
            ctypes.c_int,                    # n_sph
            ctypes.c_int, ctypes.c_int,      # n_grp_real, n_grp_pad
            _I32,                            # out
        ]
        lib.mu_matrix.restype = None
        lib.mu_matrix.argtypes = [
            _F32, _F32, ctypes.c_int,
            _F32, _F32, ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            _U8_2D,                          # out (KA, KB)
        ]
        self._lib = lib
        self.simd_width = int(lib.mu_kernel_simd_width())
        self.available = self.simd_width > 0

    # -- packing --------------------------------------------------------------- #
    def pack(self, traj, side: str) -> PackedTraj:
        """Repack a :class:`TrajSpheres` into planes. ``side`` is 'A' or 'B'.

        The two sides park their dead lanes at OPPOSITE sentinels so that even a
        padded-vs-padded comparison is far apart; parking both at the same point would
        put them at zero distance with zero radii, which the ``<=`` test would call a
        touch.
        """
        if side not in ("A", "B"):
            raise ValueError(f"side must be 'A' or 'B', got {side!r}")
        sign = 1.0 if side == "A" else -1.0
        w = self.simd_width or 8

        spheres = _pack_planes(traj.centres, traj.radii, _round_up(traj.centres.shape[1], w), sign)
        n_grp_real = traj.gcen.shape[1]
        groups = _pack_planes(traj.gcen, traj.grad, _round_up(n_grp_real, w), sign)
        return PackedTraj(spheres, groups, spheres.shape[2], n_grp_real, groups.shape[2])

    # -- the call -------------------------------------------------------------- #
    def forbidden_offsets(self, A: PackedTraj, B: PackedTraj) -> List[int]:
        """``D = {k - l}`` for one trajectory pair, ascending. Same result as the numpy
        :func:`vamp_collision_engine.forbidden_offsets_pair`, asserted by the self-test."""
        if not self.available:
            raise RuntimeError(f"mu kernel unavailable ({self.path}); build it with "
                               f"scripts/build_mu_kernel.sh")
        if A.n_sph != B.n_sph or A.n_grp_pad != B.n_grp_pad:
            raise ValueError("the two trajectories were packed with different shapes")

        out = np.empty(A.K + B.K - 1, dtype=np.int32)
        n = self._lib.mu_forbidden_offsets(
            A.spheres, A.groups, A.K,
            B.spheres, B.groups, B.K,
            A.n_sph, A.n_grp_real, A.n_grp_pad,
            out)
        return out[:n].tolist()

    def matrix(self, A: PackedTraj, B: PackedTraj) -> np.ndarray:
        """The full ``mu[k, l]`` for one trajectory pair, as a (KA, KB) bool array.

        The temporal plan graph needs which pairs collide, not just their difference, so
        this keeps the matrix the offset reduction throws away. Costs a few times a
        :meth:`forbidden_offsets` call -- there is no diagonal early exit, since every
        entry is part of the answer.
        """
        if not self.available:
            raise RuntimeError(f"mu kernel unavailable ({self.path})")
        if A.n_sph != B.n_sph or A.n_grp_pad != B.n_grp_pad:
            raise ValueError("the two trajectories were packed with different shapes")

        out = np.empty((A.K, B.K), dtype=np.uint8)
        self._lib.mu_matrix(
            A.spheres, A.groups, A.K,
            B.spheres, B.groups, B.K,
            A.n_sph, A.n_grp_real, A.n_grp_pad,
            out)
        return out.astype(bool)


def _pack_planes(centres: np.ndarray, radii: np.ndarray, n_pad: int, sign: float) -> np.ndarray:
    """(K, n, 3) + (K, n) -> (K, 4, n_pad) float32 planes, dead lanes parked."""
    K, n, _ = centres.shape
    out = np.empty((K, 4, n_pad), dtype=np.float32)
    for axis in range(3):
        out[:, axis, :n] = centres[:, :, axis]
    out[:, 3, :n] = radii

    # Entries the numpy engine marks absent with a -inf radius (a detached carried
    # object, and its group bound) become parked lanes -- same treatment as padding.
    absent = ~np.isfinite(radii)
    if absent.any():
        for axis in range(3):
            plane = out[:, axis, :n]
            plane[absent] = sign * PARK
        rad = out[:, 3, :n]
        rad[absent] = 0.0

    if n_pad > n:
        out[:, 0:3, n:] = sign * PARK
        out[:, 3, n:] = 0.0

    return np.ascontiguousarray(out)
