"""VAMP collision engine -- the ADR-0005 Phase-2 drop-in for FCL, collisions only.

This is the robot-agnostic runtime half of the VAMP contribution. It consumes the
SAME ``tamp_trajectories.json`` the C++ ``collision_generator`` reads, and it must
emit the SAME geometry-free seam (``tamp_problem.json``) so the ROS-free solver in
``thesis_material_tamp/tamp_scheduler`` cannot tell whether its numbers came from
MoveIt/FCL or from VAMP (ADR-0002: zero vamp import ever crosses into the solver).

WHAT VAMP GIVES US
------------------
A compiled robot module (``vamp.ur5`` now; a codegen'd ``vamp.ur10e_rail`` in Phase
B) exposes ``fk(q) -> list[Sphere(x,y,z,r)]``: the robot's whole collision geometry
as world-frame spheres, in the robot's base frame, for one configuration. Two robots'
sphere sets touch iff some cross sphere-pair satisfies ``||c_k - c_l|| <= r_k + r_l``.
That reduces the inter-robot check ``mu[k,l]`` to a numpy broadcast -- no FCL, no
MoveIt, no ROS.

WHAT THIS ENGINE MIRRORS FROM THE C++ (collision_generator.cpp)
--------------------------------------------------------------
* ONLY robot-vs-robot (+ each mover's carried object) enters ``mu``. Self- and
  world-collisions were settled in the trajectory stage; re-checking them would
  corrupt ``mu`` with hits that have nothing to do with the two robots' timing.
* The CARRIED OBJECT is part of the mover's geometry (ADR-0005): when
  ``object_state[k] == ATTACHED`` the box rides the gripper and can strike the other
  robot. We add it as one bounding sphere at the end-effector, offset +0.10 m along
  the tool z-axis (mirrors the ``p.position.z = 0.10`` grasp offset in ``setAttached``).
* A per-sample bounding sphere broad phase prunes the K x K grid before the exact
  narrow check -- the same two-phase structure as the C++ (``computeBounds`` + FCL).

TWO ROBOTS (ur5 stand-in vs the codegen'd ur10e_rail)
-----------------------------------------------------
* ``vamp.ur5`` (Phase-A) is a 6-DOF bare arm that does NOT match the cell; its ``mu``
  only exercises the pipeline plumbing (use ``--base-layout cell``).
* ``vamp.ur10e_rail`` (Phase-B, realised 2026-07-23) is the codegen'd 7-DOF rail+arm
  +gripper. Its ``mu`` is a faithful 1:1 match to FCL -- verified 0-missing against the
  FCL reference seam (``--base-layout ur10e_rail``). Cell placement is exact via:
    - a pure y-translation to each rail (+/-0.80 m), no sphere rotation (both rails run
      along world-x), and
    - a per-robot mount yaw (robot1 -pi/2, robot2 +pi/2) folded into the shoulder_pan
      column of the fed config -- coaxial with the arm's first joint, so it reproduces
      the URDF carriage->arm yaw exactly (``mount_yaws``);
  plus ``n_structural=18`` (the rail/support/carriage spheres, which contribute nothing
  to mu) and ``sphere_margin=0.02`` (foam under-covers the arm mesh by up to 2 cm, so a
  2 cm margin restores conservatism vs FCL: 0 missing, ~+20 % extra offsets).

No ``tamp_scheduler`` import appears anywhere here (ADR-0002). The offset reduction
``D = {k - l : mu[k,l]}`` is inlined rather than imported for the same reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Object-state codes, identical to resample.hpp's ObjectState enum.
AT_SPAWN = 0
ATTACHED = 1
AT_PLACE = 2

# Phase codes, identical to resample.hpp's Phase enum.
PHASE_GRIP_CLOSE = 1  # pick milestone
PHASE_GRIP_OPEN = 3   # place milestone

# The gripper-frame grasp offset baked into the C++ setAttached(): the carried box's
# volume travels 0.10 m out along the tool approach axis, not at the wrist origin.
GRASP_OFFSET_Z = 0.10


def yaw_translation(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    """4x4 homogeneous transform: rotate ``yaw`` about world z, then translate."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0, x],
            [s, c, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


# Phase-A stand-in: place the two identical ur5 instances in the cell's opposed rail
# layout (rails at y = +/-0.80, table top at z = 0.75, robots facing each other along
# y) so their sphere sets are actually separated. NOT cell-accurate geometry -- a
# placeholder until the Phase-B ur10e_rail bakes its own base (then: identity).
CELL_BASE_TRANSFORMS: Dict[str, np.ndarray] = {
    "robot1": yaw_translation(0.0, 0.80, 0.75, math.pi),  # +y rail, facing -y
    "robot2": yaw_translation(0.0, -0.80, 0.75, 0.0),     # -y rail, facing +y
}

# Phase-B cell-accurate placement for the codegen'd ``vamp.ur10e_rail``.
#
# The codegen robot is baked CANONICAL: rail along world-x, arm mount yaw = 0, support
# frame already at the cell rail height (guide_origin_z = table_height - rail_height =
# 0.75 - 0.05 = 0.70). So placing it into the cell needs, per robot:
#
#   * a PURE Y-TRANSLATION to the rail's cell offset (guide_y_offset = table_width/2 +
#     0.35 = 0.45 + 0.35 = 0.80), no rotation -- both cell rails run along x, so
#     rotating the sphere set would swing the rail off-axis (wrong);
#   * a MOUNT YAW of robot_yaw (robot1 = -pi/2, robot2 = +pi/2, from the cell macro).
#     robot_yaw is applied in the URDF at the carriage->arm mount, AFTER the rail joint
#     and COAXIAL with shoulder_pan (both rotate about the vertical z through the arm
#     base). A fixed R_z(yaw) followed by shoulder_pan(theta) equals shoulder_pan(theta
#     + yaw) with a zero mount -- so the yaw folds EXACTLY into the shoulder_pan column
#     of the fed config. This is why it is a config offset, not a sphere rotation.
#
# Result: geometry identical to the cell's robot1_*/robot2_* FK, so mu matches FCL.
CELL_UR10E_RAIL_BASE: Dict[str, np.ndarray] = {
    "robot1": yaw_translation(0.0, 0.80, 0.0, 0.0),   # +y rail, pure translation
    "robot2": yaw_translation(0.0, -0.80, 0.0, 0.0),  # -y rail, pure translation
}
CELL_UR10E_RAIL_MOUNT_YAW: Dict[str, float] = {
    "robot1": -math.pi / 2,   # faces -y
    "robot2": math.pi / 2,    # faces +y
}
# The codegen prepends 18 structural rail spheres (support 8 + rail 8 + carriage 2,
# gen_inputs.py) ahead of the 79 arm+gripper spheres. They are excluded from mu (see
# n_structural in VampCollisionEngine): verified 0-missing vs FCL, and it removes the
# r=0.40 support spheres that otherwise defeat the broad phase.
CELL_UR10E_RAIL_N_STRUCTURAL: int = 18
# Minimal robot-sphere safety margin (metres) that drives the FCL diff to 0-missing on
# every trajectory pair: foam's decomposition under-covers the arm mesh by up to ~2 cm
# at a few configurations, so 0.02 restores soundness. Empirically determined (the
# 11 false-negatives at margin 0 clear at 0.02; see the diff report).
CELL_UR10E_RAIL_MARGIN: float = 0.02


@dataclass
class ObjectGeom:
    """A graspable box, reduced to its circumscribing sphere (an over-approximation
    of the box -- sound: it never under-covers the true volume)."""

    size: Tuple[float, float, float]
    radius: float

    @staticmethod
    def from_size(size: Sequence[float]) -> "ObjectGeom":
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
        return ObjectGeom((sx, sy, sz), 0.5 * math.sqrt(sx * sx + sy * sy + sz * sz))


@dataclass
class TrajSpheres:
    """One trajectory's whole geometry, pre-FK'd once, ready for the mu broadcast.

    Every sample carries exactly ``M = n_spheres + 1`` spheres: the robot's spheres
    plus one object slot. When the object is not attached, its radius is ``-inf`` so
    it can never register a collision (``dist <= r1 + r2`` is false against -inf) and
    never inflates the broad-phase bound.
    """

    centres: np.ndarray  # (K, M, 3) world-frame sphere centres
    radii: np.ndarray    # (K, M)    sphere radii (object slot -inf when detached)
    bcen: np.ndarray     # (K, 3)    per-sample bounding-sphere centre (broad phase)
    brad: np.ndarray     # (K,)      per-sample bounding-sphere radius

    @property
    def K(self) -> int:
        return self.centres.shape[0]


class VampCollisionEngine:
    """Robot-agnostic FK -> world-frame spheres -> broadcast ``mu``.

    ``robot_module`` is a vamp robot submodule (``vamp.ur5``, later
    ``vamp.ur10e_rail``); ``base_transforms`` maps each artifact robot name to a 4x4
    world transform applied to that robot's FK spheres (Phase-A separation shim).
    """

    def __init__(
        self,
        robot_module,
        objects: Dict[str, ObjectGeom],
        base_transforms: Dict[str, np.ndarray] | None = None,
        mount_yaws: Dict[str, float] | None = None,
        n_structural: int = 0,
        sphere_margin: float = 0.0,
    ):
        self.robot = robot_module
        self.objects = objects
        self.base_transforms = base_transforms or {}
        # Per-robot mount yaw folded into the shoulder_pan column of the fed config
        # (see CELL_UR10E_RAIL_MOUNT_YAW). Empty for the ur5 stand-in, which carries
        # its facing in the base_transform rotation instead.
        self.mount_yaws = mount_yaws or {}
        self.dim = int(robot_module.dimension())
        # The first ``n_structural`` FK spheres are structural links (rail/support/
        # carriage for ur10e_rail: 18) that are geometrically separated from the OTHER
        # robot and provably contribute nothing to mu (verified: dropping them leaves
        # the FCL diff at 0 missing). Excluding them shrinks the narrow check and --
        # more importantly -- tightens the broad-phase bounding sphere, which the big
        # r=0.40 support spheres otherwise blow up across the whole workspace (the
        # ADR-0005 "exclude the structural links" prune). Kept in mu only if 0.
        self.n_structural = int(n_structural)
        # Safety margin (metres) added to every robot sphere radius. VAMP's foam
        # decomposition is NOT guaranteed to over-approximate the FCL collision mesh --
        # a coarse sphere can leave a thin gap the mesh fills, producing a false
        # NEGATIVE (an offset FCL forbids that VAMP allows -- unsound). Inflating radii
        # by ``sphere_margin`` restores conservatism: it can only add EXTRA offsets,
        # never remove a true one. The minimal margin that drives the FCL diff to
        # 0-missing is an empirical property of the decomposition (see the diff report).
        self.sphere_margin = float(sphere_margin)
        # Sphere radii are fixed robot geometry (config-independent); read once, minus
        # the excluded structural prefix, plus the safety margin.
        ref = robot_module.fk([0.0] * self.dim)
        self._robot_radii = (
            np.array([s.r for s in ref], dtype=np.float64)[self.n_structural:] + self.sphere_margin)
        self.n_spheres = int(robot_module.n_spheres()) - self.n_structural

    # -- configuration mapping ------------------------------------------------ #
    def map_config(self, row: Sequence[float]) -> List[float]:
        """Map a 7-vector artifact sample to this robot's config.

        * ``dim == len(row)``      -> identity (a matched 7-DOF robot, e.g. Phase-B
                                      ur10e_rail: [rail, pan, lift, elbow, w1, w2, w3]);
        * ``dim == len(row) - 1``  -> drop the rail column (index 0), leaving the 6
                                      arm joints in UR order (the shipped ur5).
        Anything else is a genuine mismatch and fails loud.
        """
        n = len(row)
        if self.dim == n:
            return [float(v) for v in row]
        if self.dim == n - 1:
            return [float(v) for v in row[1:]]
        raise ValueError(
            f"robot config dim {self.dim} is incompatible with a {n}-DOF artifact "
            f"sample (expected {n} or {n - 1})"
        )

    def _config(self, robot_name: str, q_row: Sequence[float]) -> List[float]:
        """map_config + fold this robot's mount yaw into the shoulder_pan column.

        A UR arm's last six joints are [pan, lift, elbow, w1, w2, w3], so shoulder_pan
        is at index ``dim - 6`` (index 1 for the 7-DOF rail+arm, 0 for the bare ur5).
        Adding the fixed mount yaw there reproduces the URDF's carriage->arm yaw exactly
        (it is coaxial with shoulder_pan), with no sphere rotation.
        """
        q = self.map_config(q_row)
        yaw = self.mount_yaws.get(robot_name)
        if yaw:
            q[self.dim - 6] += yaw
        return q

    # -- per-sample world-frame spheres --------------------------------------- #
    def _apply(self, T: np.ndarray | None, pts: np.ndarray) -> np.ndarray:
        if T is None:
            return pts
        return pts @ T[:3, :3].T + T[:3, 3]

    def robot_spheres(self, robot_name: str, q_row: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """World-frame robot spheres for one artifact sample: centres (n,3), radii (n,)."""
        q = self._config(robot_name, q_row)
        sph = self.robot.fk(q)
        centres = np.array([[s.x, s.y, s.z] for s in sph], dtype=np.float64)[self.n_structural:]
        centres = self._apply(self.base_transforms.get(robot_name), centres)
        return centres, self._robot_radii

    def object_sphere(self, robot_name: str, q_row: Sequence[float], obj: ObjectGeom) -> Tuple[np.ndarray, float]:
        """World-frame bounding sphere of the carried object at one sample.

        Placed at the end-effector, offset +0.10 m along the tool z-axis -- the same
        grasp offset the C++ ``setAttached`` applies to the attached box.
        """
        q = self._config(robot_name, q_row)
        ee = np.asarray(self.robot.eefk(q), dtype=np.float64)  # (4,4), robot base frame
        centre_local = ee[:3, 3] + GRASP_OFFSET_Z * ee[:3, 2]
        centre = self._apply(self.base_transforms.get(robot_name), centre_local[None, :])[0]
        return centre, obj.radius + self.sphere_margin

    def traj_spheres(
        self,
        robot_name: str,
        positions: Sequence[Sequence[float]],
        object_state: Sequence[int],
        object_id: str,
    ) -> TrajSpheres:
        """Pre-FK a whole trajectory into the (K, M, 3)/(K, M) sphere arrays + bounds."""
        K = len(positions)
        M = self.n_spheres + 1
        centres = np.zeros((K, M, 3), dtype=np.float64)
        radii = np.empty((K, M), dtype=np.float64)
        radii[:, : self.n_spheres] = self._robot_radii[None, :]
        radii[:, self.n_spheres] = -np.inf  # object slot, off unless attached

        obj = self.objects[object_id]
        for k in range(K):
            rc, _ = self.robot_spheres(robot_name, positions[k])
            centres[k, : self.n_spheres, :] = rc
            if object_state[k] == ATTACHED:
                oc, orad = self.object_sphere(robot_name, positions[k], obj)
                centres[k, self.n_spheres, :] = oc
                radii[k, self.n_spheres] = orad

        bcen, brad = _bounding_spheres(centres, radii, self.n_spheres, object_state)
        return TrajSpheres(centres=centres, radii=radii, bcen=bcen, brad=brad)


def _bounding_spheres(
    centres: np.ndarray, radii: np.ndarray, n_robot: int, object_state: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """One bounding sphere per sample, over the robot spheres (+ object when present).

    Centre = mean of the robot sphere centres; radius = max over covered spheres of
    ``dist(centre, c) + r`` -- so the sphere fully CONTAINS every covered sphere and
    the broad phase can never wrongly reject a true collision (mirrors computeBounds)."""
    rc = centres[:, :n_robot, :]                       # (K, n, 3)
    rr = radii[:, :n_robot]                             # (K, n)
    bcen = rc.mean(axis=1)                              # (K, 3)
    rad = (np.linalg.norm(rc - bcen[:, None, :], axis=2) + rr).max(axis=1)  # (K,)

    attached = np.asarray(object_state) == ATTACHED
    if attached.any():
        oc = centres[:, n_robot, :]                    # (K, 3)
        orad = radii[:, n_robot]                        # (K,)
        odist = np.linalg.norm(oc - bcen, axis=1) + np.where(np.isfinite(orad), orad, 0.0)
        rad = np.where(attached, np.maximum(rad, odist), rad)
    return bcen, rad


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


def collision_matrix(A: TrajSpheres, B: TrajSpheres, use_broad: bool = True) -> np.ndarray:
    """The boolean ``mu[k,l]`` for one trajectory pair (A = robot r task i, B = s task j).

    ``mu[k,l]`` is True iff robot A at sample k collides with robot B at sample l.
    Broad phase (bounding spheres) prunes the K x K grid; the exact narrow check then
    runs only on survivors -- identical in result to ``use_broad=False``, just faster.
    """
    Ki, Kj = A.K, B.K
    mu = np.zeros((Ki, Kj), dtype=bool)

    if use_broad:
        dcen = np.linalg.norm(A.bcen[:, None, :] - B.bcen[None, :, :], axis=2)  # (Ki,Kj)
        survive = dcen <= (A.brad[:, None] + B.brad[None, :])
        ks, ls = np.where(survive)
    else:
        ks, ls = np.meshgrid(np.arange(Ki), np.arange(Kj), indexing="ij")
        ks, ls = ks.ravel(), ls.ravel()

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
