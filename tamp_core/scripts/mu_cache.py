"""Content-addressed ``mu`` blocks, so the stages after the solve stop recomputing them.

WHY
---
The baseline plan graph (``build_tpg.py``) computes the full ``mu`` matrix of every
SCHEDULED trajectory pair. Refinement then needed the same matrices again -- ``--plan``
built each task against the other robot's whole timeline, every intermediate task twice --
and after the splice the refined seam and the refined graph recomputed pairs that are
mostly the old ones: a refined trajectory is ``transit[1:-1] + original[lo:hi+1]``, so its
``mu`` against the other refined trajectory is sub-blocks of the old matrices plus the
transit rows and columns, which are the only genuinely new cells.

WHY THE KEY IS THE GEOMETRY ITSELF
----------------------------------
A cell ``mu[k, l]`` is a pure function of sample k's spheres and sample l's spheres -- the
SIMD kernel reads nothing else, and its padding lanes are parked where they cannot touch
(mu_kernel.py). So every sample (row) is keyed by a digest of exactly what the kernel
consumes: its sphere centres and radii and its link-group bounds, float32, after the
synchronous-hold exemption (:class:`vamp_collision_engine.HoldExemption`). Anything that
changes ``mu`` changes those numbers -- other trajectories (OMPL is unseeded), another
scene, sphere margin, finger patch, object sizes, grasp, a re-codegen'd module -- and a
row whose numbers changed simply does not match. A stale or foreign cache therefore costs
a lookup, never a wrong cell, and no provenance has to travel between the stages: a
refined trajectory's original samples find their rows by content, its transit samples
find nothing and are computed. The only thing a digest cannot see is the collision TEST
itself; :data:`FORMAT` names it, and must be bumped whenever ``mu_kernel.cc`` changes
what counts as a collision (``sets_touch``, the broad phase, the padding contract).

Blocks are stored with the robot that comes first in the artifact's ``robots`` as side A
(rows), as ``tpg.build`` asks for them; a lookup uses a block only in that orientation.
Pairs the seam proved empty are NOT stored: that proof rests on the seam being fresh, and
the cache must hold only matrices that were actually computed.

FILE (``np.savez``, uncompressed): ``format``; ``rows`` (N, 16) uint8 row digests of every
stored trajectory, ``ptr`` their offsets; per block ``blk_a`` / ``blk_b`` (trajectory
indices) and ``blk_ptr`` into ``bits``, the matrix bit-packed row by row
(``np.packbits(axis=1)``); ``meta`` (JSON, labels for humans, never used for lookup).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# The collision semantics the stored bits were computed under. Bump on any change to what
# mu_kernel.cc (or the numpy reference) calls a collision; a cache of another format is
# ignored, loudly.
FORMAT = "mu-cache/1 (sphere-set touch |c_a-c_b| <= r_a+r_b, per-link broad phase, float32)"

_PERSON = b"mu-row/1"


def row_digests(ts) -> np.ndarray:
    """(K, 16) uint8: a 128-bit BLAKE2b digest per sample of a TrajSpheres, over exactly
    what the kernel reads for that sample (centres, radii, group centres and radii, float32,
    with the array shapes in the key)."""
    K = ts.centres.shape[0]
    parts = [np.ascontiguousarray(a, dtype=np.float32).reshape(K, -1)
             for a in (ts.centres, ts.radii, ts.gcen, ts.grad)]
    flat = np.ascontiguousarray(np.concatenate(parts, axis=1)).view(np.uint8)
    base = hashlib.blake2b(digest_size=16, person=_PERSON)
    base.update(np.asarray([ts.centres.shape[1], ts.gcen.shape[1]], dtype=np.int64).tobytes())
    out = np.empty((K, 16), dtype=np.uint8)
    prev = None
    for k in range(K):
        row = flat[k].tobytes()
        if row != prev:                                  # frozen samples: same digest
            h = base.copy()
            h.update(row)
            d = np.frombuffer(h.digest(), dtype=np.uint8)
            prev = row
        out[k] = d
    return out


def _keys(dig: np.ndarray) -> List[bytes]:
    return np.ascontiguousarray(dig).view("V16").ravel().tolist()


def _traj_key(dig: np.ndarray) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(dig).tobytes(), digest_size=16).digest()


def _sl(idx: np.ndarray):
    """A slice for an ascending contiguous index array (cheap views), else the array."""
    if idx.size and int(idx[-1]) - int(idx[0]) == idx.size - 1 and (
            idx.size < 2 or bool((np.diff(idx) == 1).all())):
        return slice(int(idx[0]), int(idx[-1]) + 1)
    return idx


class MuBlocks:
    """The store: trajectories (as row digests) and the bit-packed blocks between them."""

    def __init__(self):
        self.trajs: List[np.ndarray] = []           # slot -> (K, 16) row digests
        self.labels: List[str] = []
        self._slot: Dict[bytes, int] = {}           # whole-trajectory key -> slot
        self.blocks: Dict[Tuple[int, int], Tuple[np.ndarray, Tuple[int, int]]] = {}
        self._index: Dict[str, Dict[bytes, Tuple[int, int]]] = {}
        self._maps: Dict[Tuple[str, bytes], Tuple[np.ndarray, np.ndarray]] = {}

    # -- building ------------------------------------------------------------- #
    def slot(self, dig: np.ndarray, label: str = "") -> int:
        key = _traj_key(dig)
        s = self._slot.get(key)
        if s is None:
            s = self._slot[key] = len(self.trajs)
            self.trajs.append(np.ascontiguousarray(dig, dtype=np.uint8))
            self.labels.append(label)
            self._index.clear()
            self._maps.clear()
        return s

    def add(self, dig_a: np.ndarray, dig_b: np.ndarray, bits: np.ndarray,
            shape: Tuple[int, int], label_a: str = "", label_b: str = "") -> None:
        """Store a block: A rows ``dig_a`` x B columns ``dig_b``, ``bits`` the matrix as
        ``np.packbits(mu, axis=1)``."""
        if tuple(shape) != (dig_a.shape[0], dig_b.shape[0]) or \
                bits.shape != (shape[0], (shape[1] + 7) // 8):
            raise ValueError(f"block {tuple(shape)} / bits {bits.shape} do not match digests "
                             f"{(dig_a.shape[0], dig_b.shape[0])}")
        a, b = self.slot(dig_a, label_a), self.slot(dig_b, label_b)
        self.blocks[(a, b)] = (np.ascontiguousarray(bits, dtype=np.uint8), tuple(shape))
        self._index.clear()
        self._maps.clear()

    def add_matrix(self, dig_a: np.ndarray, dig_b: np.ndarray, mu: np.ndarray,
                   label_a: str = "", label_b: str = "") -> None:
        self.add(dig_a, dig_b, np.packbits(np.asarray(mu, dtype=bool), axis=1), mu.shape,
                 label_a, label_b)

    def __len__(self) -> int:
        return len(self.blocks)

    # -- persistence ---------------------------------------------------------- #
    def save(self, path: str) -> None:
        keys = sorted(self.blocks)
        rows = (np.concatenate(self.trajs) if self.trajs else np.zeros((0, 16), np.uint8))
        ptr = np.cumsum([0] + [t.shape[0] for t in self.trajs]).astype(np.int64)
        bits = [self.blocks[k][0].ravel() for k in keys]
        blk_ptr = np.cumsum([0] + [b.size for b in bits]).astype(np.int64)
        meta = {"labels": self.labels}
        tmp = path + ".tmp.npz"
        np.savez(tmp, format=np.array(FORMAT), rows=rows, ptr=ptr,
                 blk_a=np.array([k[0] for k in keys], dtype=np.int32),
                 blk_b=np.array([k[1] for k in keys], dtype=np.int32),
                 blk_ptr=blk_ptr,
                 bits=(np.concatenate(bits) if bits else np.zeros(0, np.uint8)),
                 meta=np.array(json.dumps(meta)))
        os.replace(tmp, path)                   # never a half-written cache under the name

    @classmethod
    def load(cls, path: str, log=print) -> Optional["MuBlocks"]:
        """The cache at ``path``, or None (absent, unreadable, or another FORMAT)."""
        if not path or not os.path.isfile(path):
            return None
        try:
            z = np.load(path, allow_pickle=False)
            fmt = str(z["format"])
            if fmt != FORMAT:
                log(f"mu cache {path}: format '{fmt}' is not '{FORMAT}' -- ignored")
                return None
            self = cls()
            rows, ptr = z["rows"], z["ptr"]
            labels = json.loads(str(z["meta"])).get("labels", [])
            for i in range(len(ptr) - 1):
                self.slot(rows[ptr[i]:ptr[i + 1]], labels[i] if i < len(labels) else "")
            bits, bp = z["bits"], z["blk_ptr"]
            for n, (a, b) in enumerate(zip(z["blk_a"].tolist(), z["blk_b"].tolist())):
                ka, kb = self.trajs[a].shape[0], self.trajs[b].shape[0]
                blk = bits[bp[n]:bp[n + 1]].reshape(ka, (kb + 7) // 8)
                self.blocks[(a, b)] = (blk, (ka, kb))
            return self
        except Exception as e:                  # a cache is an optimisation, never a failure
            log(f"mu cache {path}: unreadable ({e}) -- ignored")
            return None

    # -- lookup --------------------------------------------------------------- #
    def _side_index(self, side: str) -> Dict[bytes, Tuple[int, int]]:
        idx = self._index.get(side)
        if idx is None:
            slots = sorted({k[0] if side == "A" else k[1] for k in self.blocks})
            idx = {}
            for s in slots:
                for r, key in enumerate(_keys(self.trajs[s])):
                    idx.setdefault(key, (s, r))
            self._index[side] = idx
        return idx

    def _map(self, dig: np.ndarray, side: str) -> Tuple[np.ndarray, np.ndarray]:
        """Per row: (slot, row in that slot), slot -1 where no stored row has this digest.
        Memoised per (side, trajectory): a trajectory takes part in many pairs."""
        tk = _traj_key(dig)
        hit = self._maps.get((side, tk))
        if hit is None:
            hit = self._maps[(side, tk)] = self._map_rows(dig, side, tk)
        return hit

    def _map_rows(self, dig: np.ndarray, side: str, tk: bytes) -> Tuple[np.ndarray, np.ndarray]:
        K = dig.shape[0]
        s = self._slot.get(tk)
        if s is not None and any((k[0] if side == "A" else k[1]) == s for k in self.blocks):
            return np.full(K, s, dtype=np.int64), np.arange(K, dtype=np.int64)
        idx = self._side_index(side)
        slot = np.full(K, -1, dtype=np.int64)
        row = np.zeros(K, dtype=np.int64)
        for k, key in enumerate(_keys(dig)):
            hit = idx.get(key)
            if hit is not None:
                slot[k], row[k] = hit
        return slot, row

    def plan(self, dig_a: np.ndarray, dig_b: np.ndarray):
        """How a (KA, KB) request decomposes: ``(fills, todo, known_cells)``.

        ``fills``: [(rows, cols, block, block_rows, block_cols)] served from stored blocks;
        ``todo``: [(rows, cols)] to compute (per row group, the union of its uncovered
        column groups -- one kernel call each)."""
        sa, ra = self._map(dig_a, "A")
        sb, rb = self._map(dig_b, "B")
        row_groups = [(t, np.flatnonzero(sa == t)) for t in np.unique(sa).tolist()]
        col_groups = [(t, np.flatnonzero(sb == t)) for t in np.unique(sb).tolist()]
        fills, todo, known = [], [], 0
        for t1, rows in row_groups:
            missing = []
            for t2, cols in col_groups:
                blk = self.blocks.get((t1, t2)) if t1 >= 0 and t2 >= 0 else None
                if blk is None:
                    missing.append(cols)
                    continue
                fills.append((rows, cols, blk, ra[rows], rb[cols]))
                known += rows.size * cols.size
            if missing:
                todo.append((rows, np.sort(np.concatenate(missing))))
        return fills, todo, known


def _put(out: np.ndarray, rows: np.ndarray, cols: np.ndarray, val: np.ndarray) -> None:
    r, c = _sl(rows), _sl(cols)
    if isinstance(r, slice) or isinstance(c, slice):
        out[r, c] = val                    # at most one advanced axis: a plain block write
    else:
        out[np.ix_(rows, cols)] = val


def fill(out: np.ndarray, fills) -> None:
    """Copy the stored cells of ``plan`` into ``out`` (bool, (KA, KB))."""
    for rows, cols, (bits, (ka, kb)), brows, bcols in fills:
        sub = np.unpackbits(bits[_sl(brows)], axis=1, count=kb).view(bool)
        _put(out, rows, cols, sub[:, _sl(bcols)])


def sub_spheres(ts, idx):
    """The TrajSpheres of a subset of samples (a cell needs only its own two samples)."""
    from vamp_collision_engine import TrajSpheres
    if isinstance(idx, slice) and idx == slice(0, ts.centres.shape[0]):
        return ts
    return TrajSpheres(centres=ts.centres[idx], radii=ts.radii[idx],
                       gcen=ts.gcen[idx], grad=ts.grad[idx])


class MuSource:
    """``mu`` for a pair: from the cache where its rows are known, the kernel elsewhere.

    ``min_known``: the fraction of cells the cache must serve for assembly to be used at
    all; below it the pair is computed whole, exactly as without a cache (a stale cache
    still matches the HOME samples every trajectory shares, and assembling around a few
    rows would only fragment the kernel calls). The answer is the same either way.
    """

    def __init__(self, kernel, store: Optional[MuBlocks], min_known: float = 0.0):
        self.kernel, self.store, self.min_known = kernel, store, min_known
        self.served = 0          # cells served from the cache
        self.computed = 0        # cells computed by the kernel (as requested, before dedupe)

    def assemble(self, ts_a, dig_a, ts_b, dig_b, n_sph: int, n_grp: int) -> Optional[np.ndarray]:
        """The (KA, KB) bool matrix with the stored cells copied and the rest computed, or
        None when the cache serves less than ``min_known`` of it (compute it whole)."""
        if self.store is None or not len(self.store):
            return None
        fills, todo, known = self.store.plan(dig_a, dig_b)
        KA, KB = dig_a.shape[0], dig_b.shape[0]
        if known == 0 or known < self.min_known * KA * KB:
            return None
        out = np.zeros((KA, KB), dtype=bool)
        fill(out, fills)
        for rows, cols in todo:
            r, c = _sl(rows), _sl(cols)
            pa = self.kernel.pack(sub_spheres(ts_a, r), "A", n_sph=n_sph, n_grp=n_grp)
            pb = self.kernel.pack(sub_spheres(ts_b, c), "B", n_sph=n_sph, n_grp=n_grp)
            m = self.kernel.matrix(pa, pb)
            _put(out, rows, cols, m)
            self.computed += m.size
        self.served += known
        return out


def offsets_of(mu: np.ndarray) -> List[int]:
    """``D = {k - l : mu[k, l]}``, ascending -- what ``mu_forbidden_offsets`` returns."""
    KA, KB = mu.shape
    ks, ls = np.nonzero(mu)
    if ks.size == 0:
        return []
    hit = np.zeros(KA + KB - 1, dtype=bool)
    hit[ks - ls + (KB - 1)] = True
    return (np.flatnonzero(hit) - (KB - 1)).tolist()
