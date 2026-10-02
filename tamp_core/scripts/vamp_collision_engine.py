"""VAMP collision engine -- geometry only: configurations -> world-frame spheres.

The ADR-0005 Phase-2 drop-in for FCL. It consumes the SAME ``tamp_trajectories.json``
the C++ ``collision_generator`` reads and feeds the SAME geometry-free seam, so the
ROS-free solver in ``thesis_material_tamp/tamp_scheduler`` cannot tell which engine ran
(ADR-0002: zero vamp import ever crosses into the solver).

This module does ONE job: turn a trajectory into the sphere geometry a collision test
needs. The tests themselves live elsewhere:

* :mod:`mu_kernel`     -- the SIMD kernel (ADR-0005 Phase 2b). The production path.
* :mod:`vamp_reference` -- the numpy implementation it replaced. Kept as the oracle the
  self-test diffs the kernel against, and as a fallback where the kernel will not build.

WHAT VAMP GIVES US
------------------
``vamp.ur10e_rail.fk(q) -> list[Sphere(x,y,z,r)]``: the robot's whole collision geometry
as world-frame spheres for one configuration. Two robots touch iff some cross sphere-pair
satisfies ``||c_a - c_b|| <= r_a + r_b`` -- no FCL, no MoveIt, no ROS.

WHAT THIS MIRRORS FROM THE C++ (collision_generator.cpp)
--------------------------------------------------------
* ONLY robot-vs-robot (+ each mover's carried object) enters ``mu``. Self- and
  world-collisions were settled in the trajectory stage; re-checking them would corrupt
  ``mu`` with hits that have nothing to do with the two robots' timing.
* The CARRIED OBJECT is part of the mover's geometry: when ``object_state[k] ==
  ATTACHED`` the box rides the gripper and can strike the other robot. It is added as one
  sphere about the box's TRUE centre, ``tool0 (x) grasp^-1`` (the pose the trajectory
  generator attaches it at, and the C++ ``setAttached`` models), with the box's
  half-diagonal plus ``sphere_margin`` as radius -- so it covers the whole box whatever
  its orientation. Before 2026-09-21 it sat at a fixed +0.10 m from VAMP's end-effector
  frame, 9.7 cm short of the real centre (ADR-0005 addendum).
* The broad-phase bound is HIERARCHICAL: one sphere per robot LINK
  (:mod:`vamp_link_groups`), not one per robot. A single whole-robot sphere prunes 0.6 %
  here -- the arm spans ~0.9 m against a 1.6 m rail separation, so the two bounds always
  overlap -- while per-link groups prune 60.9 %.

CELL PLACEMENT IS A CONFIG TRICK, NOT A RIGID TRANSFORM
-------------------------------------------------------
``vamp.ur10e_rail`` is codegen'd CANONICAL: rail along world-x, arm mount yaw 0, support
frame already at the cell rail height. Placing it into the cell therefore needs only

* a PURE Y-TRANSLATION to each rail (+/-0.80 m) -- no rotation, because both cell rails
  run along x and rotating the sphere set would swing the rail off-axis; and
* a per-robot MOUNT YAW (robot1 -pi/2, robot2 +pi/2) folded into the shoulder_pan column
  of the fed config. The URDF applies that yaw at the carriage->arm mount, COAXIAL with
  shoulder_pan, so ``R_z(yaw)`` then ``shoulder_pan(theta)`` equals ``shoulder_pan(theta +
  yaw)`` with a zero mount -- an exact config offset, not a sphere rotation.

Plus ``n_structural=18`` (rail/support/carriage spheres, which contribute nothing to mu)
and ``sphere_margin=0.02`` (foam under-covers the arm mesh by up to 2 cm, so the margin
restores conservatism vs FCL: 0 missing on both scenes tested).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from cell_registry import get_cell

# Working precision for all geometry. The narrow phase is memory-bound (it
# materialises an (S, n, n) squared-distance array per batch), so halving the element
# width is close to a straight 2x. Soundness is unaffected: |x| < 2.5 m in this cell
# and float32 carries ~1.5e-7 m of absolute error there -- five orders of magnitude
# below the 0.02 m sphere_margin that already guards against foam's under-coverage.
DTYPE = np.float32

# Object-state codes, identical to resample.hpp's ObjectState enum.
AT_SPAWN = 0
ATTACHED = 1
AT_PLACE = 2

# Phase codes, identical to resample.hpp's Phase enum.
PHASE_GRIP_CLOSE = 1  # acquire milestone m0 of a pick-and-place
PHASE_GRIP_OPEN = 3   # release milestone m1 of a pick-and-place
PHASE_PROCESS_ON = 5  # acquire milestone m0 of a process (weld) task
PHASE_PROCESS_OFF = 7  # release milestone m1 of a process (weld) task

# ``tool0``'s pose in the frame ``vamp.ur10e_rail.eefk`` returns. The codegen's
# end-effector is its own ``robotiq_85_base_link``, which cricket's spherized UR10e+2F85
# mounts 0.037 m BEHIND tool0 (``robotiq_85_base_joint`` xyz 0 0 -0.037 in
# vamp_codegen/inputs/ur10e_rail_spherized.urdf); the cell URDF mounts it ON tool0.
# Measured against MoveIt FK of ``robotN_tool0`` (2026-09-21): translation (0,0,-0.037)
# to 1e-6, rotation identity, at five random configurations of each robot.
TOOL0_IN_EEFK = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.037], [0.0, 0.0, 0.0, 1.0]])


# Fingertip PATCH spheres of the 2F-85 (fabricator v3, 2026-09-29): (x, y, z, r) in the
# tool0 frame, r WITHOUT the margin (``sphere_margin`` is added on top, as for every robot
# sphere). The spherized CoMMALab gripper of the ur10e_rail module sits 24-37 mm closer to
# the flange than the cell's 2F-85 and its fingers are frozen open: the cell's finger
# collision meshes stuck out of the module's spheres by up to 25.0 mm open and 44.4 mm at
# gripper_close (5.0 / 24.4 mm beyond the 20 mm margin). These 21 spheres cover every mesh
# vertex the module's spheres do not, over the whole finger sweep open -> gripper_close
# (knuckle 0 .. 0.5, sampled every 0.05 rad): worst vertex 0.01-0.04 mm INSIDE before the
# margin. Fitted and re-checked by scripts/check_finger_patch.py. Placed through eefk like
# the carried object, so the module needs no regeneration. On for the gripper robots of a
# cell declared with ``cell:`` (make_cell_engines); off in the dual cell unless
# VAMP_FINGER_PATCH=1 (a diagnostic: the two-robot seams stay byte-identical by default).
FINGER_PATCH_TOOL0 = np.array([
    [0.02494, -0.00079, 0.15669, 0.01407],
    [0.05730, -0.01388, 0.05836, 0.01370],
    [0.02079, 0.01682, 0.07891, 0.01906],
    [0.06364, -0.00010, 0.10291, 0.01962],
    [0.04264, -0.01348, 0.11319, 0.01647],
    [0.00847, -0.01656, 0.06504, 0.00050],
    [0.04600, 0.01498, 0.06781, 0.01896],
    [0.02516, -0.01216, 0.08684, 0.01563],
    [0.04048, -0.00078, 0.15008, 0.01411],
    [0.02745, 0.00093, 0.12134, 0.01757],
    [-0.02509, 0.00004, 0.15670, 0.01413],
    [-0.05016, 0.01349, 0.05790, 0.00812],
    [-0.02326, -0.01271, 0.08699, 0.01721],
    [-0.06284, -0.00986, 0.10405, 0.01557],
    [-0.03720, 0.01249, 0.11664, 0.01571],
    [-0.03112, 0.01255, 0.09262, 0.01351],
    [-0.04043, 0.00002, 0.14964, 0.01399],
    [-0.05137, -0.01352, 0.05722, 0.00866],
    [-0.02639, -0.00118, 0.11908, 0.01732],
    [-0.06233, 0.01208, 0.10379, 0.01581],
    [-0.01678, 0.01206, 0.09784, 0.01383]], dtype=np.float64)


def finger_patch_enabled(default: bool) -> bool:
    """``VAMP_FINGER_PATCH`` = 1 / 0 overrides ``default`` (the dual-cell diagnostic)."""
    v = os.environ.get("VAMP_FINGER_PATCH")
    return default if v is None or v == "" else v not in ("0", "false", "no")


def pose_from_yaml(p: Dict[str, float]) -> np.ndarray:
    """4x4 transform of a task-YAML pose {x,y,z,roll,pitch,yaw}, with the semantics of
    trajectory_generator.cpp's ``poseFromYaml`` (tf2 ``setRPY``: R = Rz(yaw) Ry(pitch)
    Rx(roll); missing angles are 0)."""
    r, pt, y = (float(p.get(k, 0.0)) for k in ("roll", "pitch", "yaw"))
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(pt), math.sin(pt),
                              math.cos(y), math.sin(y))
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = [float(p["x"]), float(p["y"]), float(p["z"])]
    return T


def object_in_ee(grasp: Dict[str, float], approach_axis: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """The carried object's pose in the EE frame (``ee_link``, tool0 on the UR cell):
    ``grasp^-1``, translation AND rotation.

    Why this is the truth: the generator's IK puts the EE at ``object (x) grasp`` and
    ``setObjectState(Attached)`` then attaches the object at its world pose, so the
    object rides the EE at ``grasp^-1``. The IK may instead return the grasp turned by pi
    about the approach axis (``ikTo``'s flip), and the artifact does not say which; the
    volume is the same only if the object's centre is on that axis and the box is
    symmetric under the half-turn. Fail loud when it is not (same guard as the C++
    ``checkGraspSymmetry``)."""
    T = np.linalg.inv(pose_from_yaml(grasp))
    a = np.asarray(approach_axis, dtype=float)
    t = T[:3, 3]
    off_axis = np.linalg.norm(t - t.dot(a) * a)
    align = np.abs(T[:3, :3].T @ a).max()
    if off_axis > 1e-6 or align < 1.0 - 1e-6:
        raise ValueError(
            f"grasp {grasp} is not symmetric under the IK's half-turn flip about the tool "
            "approach axis: the carried pose is ambiguous")
    return T


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


# Cell placement of the codegen'd ``vamp.ur10e_rail`` -- see the module docstring for why
# this is a y-translation plus a config offset, and not a rigid transform.
CELL_BASE: Dict[str, np.ndarray] = {
    "robot1": yaw_translation(0.0, 0.80, 0.0, 0.0),   # +y rail, pure translation
    "robot2": yaw_translation(0.0, -0.80, 0.0, 0.0),  # -y rail, pure translation
}
CELL_MOUNT_YAW: Dict[str, float] = {
    "robot1": -math.pi / 2,   # faces -y
    "robot2": math.pi / 2,    # faces +y
}
# The codegen prepends 18 structural rail spheres (support 8 + rail 8 + carriage 2, see
# vamp_codegen/gen_inputs.py) ahead of the 79 arm+gripper spheres. They are excluded from
# mu: verified 0-missing vs FCL either way, and dropping them removes the r=0.40 support
# spheres that would otherwise defeat the broad phase.
CELL_N_STRUCTURAL: int = 18
# Smallest robot-sphere safety margin (metres) that drives the FCL diff to 0-missing.
# foam's decomposition under-covers the arm mesh by up to ~2 cm at a few configurations,
# which produced 11 false NEGATIVES (unsound) at margin 0. Empirical, not derived.
CELL_MARGIN: float = 0.02


@dataclass
class ObjectGeom:
    """A graspable box, reduced to its circumscribing sphere (an over-approximation
    of the box -- sound: it never under-covers the true volume), and where that sphere
    rides while carried.

    ``radius`` is the box's half-diagonal (``size`` is the bounding box, so a ``mesh:``
    object is covered too); ``centre_in_ee`` is the box centre in the EE frame,
    ``grasp^-1`` (see :func:`object_in_ee`). ``None`` only for a placeholder that is
    never attached.
    """

    size: Tuple[float, float, float]
    radius: float
    centre_in_ee: np.ndarray | None = None

    @staticmethod
    def from_size(size: Sequence[float], ee_T_obj: np.ndarray | None = None) -> "ObjectGeom":
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
        centre = None if ee_T_obj is None else np.asarray(ee_T_obj, dtype=float)[:3, 3].copy()
        return ObjectGeom((sx, sy, sz), 0.5 * math.sqrt(sx * sx + sy * sy + sz * sz), centre)

    @staticmethod
    def from_yaml(entry: Dict) -> "ObjectGeom":
        """One ``objects[]`` entry of a task YAML: ``size`` and ``grasp``. The ONE place
        the VAMP engine's consumers (seam, refinement, plan graph, simulation,
        coordination, sphere dump) turn a scene object into geometry."""
        return ObjectGeom.from_size(entry["size"], object_in_ee(entry["grasp"]))


@dataclass
class TrajSpheres:
    """One trajectory's whole geometry, pre-FK'd once, ready for the mu broadcast.

    Every sample carries exactly ``M = n_spheres + 1`` spheres: the robot's spheres
    plus one object slot. When the object is not attached, its radius is ``-inf`` so
    it can never register a collision (``dist <= r1 + r2`` is false against -inf) and
    never inflates the broad-phase bound.

    ``gcen``/``grad`` are the hierarchical broad phase: one bounding sphere per robot
    LINK plus one for the object slot (see :mod:`vamp_link_groups`). The single
    whole-robot bounding sphere this replaced pruned 0.6 % of the grid; per-link groups
    prune 60.9 %.
    """

    centres: np.ndarray  # (K, M, 3) world-frame sphere centres
    radii: np.ndarray    # (K, M)    sphere radii (object slot -inf when detached)
    gcen: np.ndarray     # (K, G, 3) per-link bounding-sphere centres (broad phase)
    grad: np.ndarray     # (K, G)    per-link bounding-sphere radii (-inf = absent)

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
        groups: Sequence[Tuple[str, int, int]] | None = None,
        finger_patch: bool = False,
    ):
        self.robot = robot_module
        self.objects = objects
        self.base_transforms = base_transforms or {}
        # Per-robot mount yaw folded into the shoulder_pan column of the fed config
        # (see CELL_MOUNT_YAW). Empty for the ur5 stand-in, which carries
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
            np.array([s.r for s in ref], dtype=DTYPE)[self.n_structural:] + self.sphere_margin)
        self.n_spheres = int(robot_module.n_spheres()) - self.n_structural

        # Hierarchical broad phase: sphere index ranges, one per robot link. Falls back
        # to a single whole-robot group (the old, near-useless bound) when no group
        # table is supplied, so an ungrouped robot still runs -- just slowly.
        self._groups: List[Tuple[int, int]] = (
            [(a, b) for _, a, b in groups] if groups else [(0, self.n_spheres)])
        if self._groups[-1][1] != self.n_spheres:
            raise ValueError(
                f"group table covers {self._groups[-1][1]} spheres but the robot has "
                f"{self.n_spheres} after excluding {self.n_structural} structural")
        # Fingertip patch spheres (FINGER_PATCH_TOOL0): after the robot's own, before the
        # object slot (which stays LAST), as one more broad-phase group. None by default.
        self.n_patch = len(FINGER_PATCH_TOOL0) if finger_patch else 0
        if self.n_patch:
            self._robot_radii = np.concatenate(
                [self._robot_radii, FINGER_PATCH_TOOL0[:, 3].astype(DTYPE) + self.sphere_margin])
            self._groups = self._groups + [(self.n_spheres, self.n_spheres + self.n_patch)]

    @property
    def n_groups(self) -> int:
        """Broad-phase groups per sample: one per robot link (+ the finger patch), plus the
        object slot."""
        return len(self._groups) + 1

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
        centres = np.array([[s.x, s.y, s.z] for s in sph], dtype=DTYPE)[self.n_structural:]
        centres = self._apply(self.base_transforms.get(robot_name), centres)
        return centres, self._robot_radii

    def object_sphere(self, robot_name: str, q_row: Sequence[float], obj: ObjectGeom) -> Tuple[np.ndarray, float]:
        """World-frame bounding sphere of the carried object at one sample.

        Centred on the box's true centre, ``tool0 (x) grasp^-1`` -- the pose the C++
        ``setAttached`` gives the attached box -- with its half-diagonal plus the margin.
        """
        if obj.centre_in_ee is None:
            raise ValueError("object has no carried pose: build it with ObjectGeom.from_yaml()")
        q = self._config(robot_name, q_row)
        ee = np.asarray(self.robot.eefk(q), dtype=np.float64)  # (4,4), robot base frame
        centre_local = (ee @ TOOL0_IN_EEFK)[:3, :3] @ obj.centre_in_ee + (ee @ TOOL0_IN_EEFK)[:3, 3]
        centre = self._apply(self.base_transforms.get(robot_name), centre_local[None, :])[0]
        return centre.astype(DTYPE), obj.radius + self.sphere_margin

    def traj_spheres(
        self,
        robot_name: str,
        positions: Sequence[Sequence[float]],
        object_state: Sequence[int],
        object_id: str,
    ) -> TrajSpheres:
        """Pre-FK a whole trajectory into the (K, M, 3)/(K, M) sphere arrays + bounds."""
        K = len(positions)
        n_rob = self.n_spheres + self.n_patch          # robot spheres (+ finger patch)
        M = n_rob + 1
        centres = np.zeros((K, M, 3), dtype=DTYPE)
        radii = np.empty((K, M), dtype=DTYPE)
        radii[:, :n_rob] = self._robot_radii[None, :]
        radii[:, n_rob] = -np.inf  # object slot, off unless attached

        # A process (weld) task carries nothing: ``"object": ""``. Its object slot stays
        # parked (radius -inf) for the whole trajectory, exactly like a detached object,
        # so both the numpy reference and the SIMD kernel skip it.
        obj = self.objects[object_id] if object_id else None
        if obj is None and any(int(s) == ATTACHED for s in object_state):
            raise ValueError(f"{robot_name}: an objectless trajectory has an ATTACHED sample")
        for k in range(K):
            rc, _ = self.robot_spheres(robot_name, positions[k])
            centres[k, : self.n_spheres, :] = rc
            if self.n_patch:
                ee = np.asarray(self.robot.eefk(self._config(robot_name, positions[k])),
                                dtype=np.float64) @ TOOL0_IN_EEFK
                loc = FINGER_PATCH_TOOL0[:, :3] @ ee[:3, :3].T + ee[:3, 3]
                centres[k, self.n_spheres:n_rob, :] = self._apply(
                    self.base_transforms.get(robot_name), loc)
            if obj is not None and object_state[k] == ATTACHED:
                oc, orad = self.object_sphere(robot_name, positions[k], obj)
                centres[k, n_rob, :] = oc
                radii[k, n_rob] = orad

        gcen, grad = _group_bounds(centres, radii, self._groups, n_rob)
        return TrajSpheres(centres=centres, radii=radii, gcen=gcen, grad=grad)


# ---------------------------------------------------------------------------- #
# Synchronous hold (precedence mode `hold`, fabricator v2): the mu exemption
# ---------------------------------------------------------------------------- #
#
# A pick-place i HOLDS its part at the place pose for its last ``hold_slots`` samples before
# GripOpen (the hold tail [h, m1)) while a process task j (a tack) runs on it. For the task
# PAIR (i, j) only, the handler's geometry is switched off on the tail samples [h, m1) --
# the part frozen at its place pose, the fingers closed (checked here). NOT on the GripOpen
# dwell after it: there the fingers open, a motion the certification world (fingers closed)
# never saw, so GripOpen keeps its ordinary mu (coordinator decision, 2026-09-28):
# during the tail the part is exactly at its place pose (the generator checks the tail is
# frozen there), which is the world the trajectory stage validated EVERY sample of j in, on
# the exact geometry, with the part at its place pose (ADR-0003: the hold prunes the part's
# spawn copy away from j's world). The circumscribing sphere of a 0.18 m plate would
# otherwise forbid the torch from ever coming near the plate it tacks. The handler's arm
# and gripper spheres stay, every other task pair keeps the normal geometry, and the tail
# of i against any OTHER task keeps the object slot. ONE implementation, used by the seam
# (collision_generator_vamp.py) and by the plan graph (build_tpg.py, simulate_tpg.py), so the
# graph orders exactly what the solver did.

def hold_tails(art: dict) -> Dict[Tuple[str, str], Tuple[int, int, frozenset]]:
    """(robot, holding pick-place) -> (h, m1, the held tasks) from a trajectory artifact.

    h = m1 - the trajectory's ``hold_slots``, m1 the first GripOpen sample: [h, m1) is the
    exempted hold tail, every sample of it checked to be the place configuration (1e-6
    rad). Empty for an artifact with no ``hold`` precedence (every scene before fabricator
    v2)."""
    partners: Dict[str, set] = {}
    for (i, j), m in zip(art.get("precedences") or [], art.get("precedence_modes") or []):
        if m == "hold":
            partners.setdefault(i, set()).add(j)
    out: Dict[Tuple[str, str], Tuple[int, int, frozenset]] = {}
    for t in art.get("trajectories") or []:
        if t["task"] not in partners:
            continue
        hs = int(t.get("hold_slots") or 0)
        if hs <= 0:
            raise ValueError(f"{t['robot']}|{t['task']} holds for {sorted(partners[t['task']])} "
                             f"but its trajectory has no hold_slots")
        ph = [int(x) for x in t["phase"]]
        m1 = ph.index(PHASE_GRIP_OPEN)
        g = m1                    # the hold tail only: GripOpen is not exempted
        h = m1 - hs
        ref = np.asarray(t["positions"][m1], dtype=float)
        if h < 0 or np.abs(np.asarray(t["positions"][h:g], dtype=float) - ref).max() > 1e-6:
            raise ValueError(f"{t['robot']}|{t['task']}: the hold tail [{h}, {g}) is "
                             f"not frozen at the place configuration")
        out[(t["robot"], t["task"])] = (h, g, frozenset(partners[t["task"]]))
    return out


def hold_exempt(tails, robot: str, task: str, other_task: str) -> bool:
    """Whether (robot, task)'s hold tail is exempted against ``other_task``."""
    x = tails.get((robot, task))
    return x is not None and other_task in x[2]


def exempt_hold_tail(ts: "TrajSpheres", h: int, g: int, full: bool = False) -> "TrajSpheres":
    """A copy of ``ts`` with the hold tail [h, g) exempted: the carried-object slot
    (and its broad-phase group) off, or -- ``full`` -- every sphere of the handler off."""
    radii = ts.radii.copy()
    grad = ts.grad.copy()
    if full:
        radii[h:g, :] = -np.inf
        grad[h:g, :] = -np.inf
    else:
        radii[h:g, -1] = -np.inf
        grad[h:g, -1] = -np.inf
    return TrajSpheres(centres=ts.centres, radii=radii, gcen=ts.gcen, grad=grad)


class HoldExemption:
    """The mu exemption of every hold of an artifact, for the seam AND the plan graph.

    ``mode(r, i, s, j)`` says how (r, i) is exempted in the pair against (s, j):
      * ``""``      -- not at all (not a hold pair; every scene without a hold);
      * ``"full"``  -- the whole handler on the hold tail [h, m1): the held process's trajectory (s, j) was
        planned and validated by the generator with the handler STANDING at the hold
        configuration, part in hand, on the exact geometry (its ``held_by`` names (r, i)).
        Every cell (k in [h, m1), any l) is collision-free by construction -- the clearance
        is certified by the trajectory stage exactly as HOME's is (ADR-0003).
      * ``"object"`` -- only the part (a held trajectory without ``held_by``, from a generator
        that planned the process against the part alone at its place pose).
    ``spheres(ts, r, i, mode)`` returns the exempted copy (cached per mode)."""

    def __init__(self, art: dict):
        self.tails = hold_tails(art)
        self.held_by = {(t["robot"], t["task"]): t.get("held_by")
                        for t in art.get("trajectories") or [] if t.get("held_by")}
        self._cache: Dict[tuple, TrajSpheres] = {}

    def __bool__(self) -> bool:
        return bool(self.tails)

    def mode(self, r: str, i: str, s: str, j: str) -> str:
        if not hold_exempt(self.tails, r, i, j):
            return ""
        return "full" if self.held_by.get((s, j)) == f"{r}|{i}" else "object"

    def spheres(self, ts: "TrajSpheres", r: str, i: str, mode: str) -> "TrajSpheres":
        if not mode:
            return ts
        key = (r, i, mode)
        if key not in self._cache:
            h, g, _ = self.tails[(r, i)]
            self._cache[key] = exempt_hold_tail(ts, h, g, full=(mode == "full"))
        return self._cache[key]


def _group_bounds(
    centres: np.ndarray, radii: np.ndarray, groups: Sequence[Tuple[int, int]], n_robot: int
) -> Tuple[np.ndarray, np.ndarray]:
    """One bounding sphere per link group, plus the object slot as its own group.

    Centre = mean of the group's sphere centres; radius = max over its spheres of
    ``dist(centre, c) + r`` -- so the group sphere fully CONTAINS every sphere it
    covers, and the broad phase can never wrongly reject a true collision.

    The object slot is passed through verbatim (it is already a single sphere), so a
    detached object keeps its ``-inf`` radius. Both consumers reject it: the numpy
    reference via its ``rsum > 0`` guard, the SIMD kernel by repacking it as a parked
    lane (see :mod:`mu_kernel`).
    """
    K = centres.shape[0]
    G = len(groups) + 1
    gcen = np.zeros((K, G, 3), dtype=DTYPE)
    grad = np.empty((K, G), dtype=DTYPE)

    for g, (a, b) in enumerate(groups):
        c = centres[:, a:b, :]                                    # (K, m, 3)
        m = c.mean(axis=1)
        gcen[:, g, :] = m
        grad[:, g] = (np.linalg.norm(c - m[:, None, :], axis=2) + radii[:, a:b]).max(axis=1)

    gcen[:, -1, :] = centres[:, n_robot, :]
    grad[:, -1] = radii[:, n_robot]
    return gcen, grad


# ---------------------------------------------------------------------------- #
# Cells with more than the dual layout (fabricator4, ADR-0010)
# ---------------------------------------------------------------------------- #
#
# A scene names its cell with a top-level ``cell:`` key (absent = the dual cell, whose
# placement is the CELL_BASE / CELL_MOUNT_YAW constants above -- unchanged). Any other cell
# has a layout YAML named by its cell.yaml (four_robot_cell/urdf/fabricator4_layout.yaml),
# the ONE file both the URDF and this engine read (written by
# thesis_material_tamp/tools/fabricator_layout.py). Per robot it gives the rail support frame
# (x, y, z, rail_yaw), the arm's mount_yaw and the tool, which picks the VAMP module:
#
#   gripper -> vamp.ur10e_rail        (UR10e + rail + Robotiq 2F-85, 97 spheres)
#   torch   -> vamp.ur10e_rail_torch  (UR10e + rail + MIG torch, 63 spheres)
#
# Both modules are codegen'd CANONICAL (rail along x, support frame at z = 0.70), so a robot
# whose rail runs along y is the canonical robot ROTATED by rail_yaw about z and moved to
# (x, y, z - 0.70): a rigid transform, exact for the whole robot (rail and arm turn together).
# The arm's mount yaw stays a shoulder_pan offset, as for the dual cell. For rail_yaw = 0 the
# transform is the dual cell's pure translation.

CANONICAL_RAIL_Z = 0.70
MODULE_OF_TOOL: Dict[str, str] = {"gripper": "ur10e_rail", "torch": "ur10e_rail_torch"}


def _spherized(cell: str, tool: str) -> str:
    """``<cell>/<vamp_modules dir>/inputs/<module>_spherized.urdf`` (the cell.yaml names the dir)."""
    d = get_cell(cell).vamp_module_dir(tool)
    return os.path.join(d, "inputs", f"{os.path.basename(d)}_spherized.urdf")


SPHERIZED_URDF: Dict[str, str] = {
    "ur10e_rail": _spherized("dual", "gripper"),
    "ur10e_rail_torch": _spherized("fabricator4", "torch"),
}


def load_cell_layout(cell: str) -> Dict:
    return get_cell(cell).layout()


def make_cell_engines(
    vamp_mod, task_yaml: Dict, objects: Dict[str, ObjectGeom], robot_names: Sequence[str],
    sphere_margin: float = CELL_MARGIN, dual_module: str = "ur10e_rail",
) -> Dict[str, "VampCollisionEngine"]:
    """One engine per robot NAME (robots sharing a module share an engine).

    ``task_yaml`` without ``cell:`` (or ``cell: dual``) -> exactly the dual-cell engine every
    script built before: ``dual_module`` with CELL_BASE / CELL_MOUNT_YAW, the ur10e_rail link
    groups, one instance for both robots. Otherwise the cell's layout decides base, mount yaw
    and module per robot.
    """
    from vamp_link_groups import link_groups  # local: vamp_link_groups imports nothing heavy

    cell = str(task_yaml.get("cell", "dual") or "dual")
    if cell == "dual":
        engine = VampCollisionEngine(
            getattr(vamp_mod, dual_module), objects, base_transforms=CELL_BASE,
            mount_yaws=CELL_MOUNT_YAW, n_structural=CELL_N_STRUCTURAL, sphere_margin=sphere_margin,
            groups=link_groups(SPHERIZED_URDF["ur10e_rail"], CELL_N_STRUCTURAL),
            finger_patch=finger_patch_enabled(False))
        return {r: engine for r in robot_names}

    layout = load_cell_layout(cell)["robots"]
    missing = [r for r in robot_names if r not in layout]
    if missing:
        raise ValueError(f"cell '{cell}': robots {missing} are not in its layout")
    by_module: Dict[str, List[str]] = {}
    for r in robot_names:
        tool = layout[r].get("tool", "gripper")
        if tool not in MODULE_OF_TOOL:
            raise ValueError(f"cell '{cell}', {r}: unknown tool '{tool}'")
        by_module.setdefault(MODULE_OF_TOOL[tool], []).append(r)
    engines: Dict[str, VampCollisionEngine] = {}
    for mod, names in by_module.items():
        if not hasattr(vamp_mod, mod):
            raise ImportError(f"vamp has no module '{mod}': build it (vamp_codegen/README.md)")
        bases = {r: yaw_translation(float(layout[r]["x"]), float(layout[r]["y"]),
                                    float(layout[r]["z"]) - CANONICAL_RAIL_Z,
                                    float(layout[r]["rail_yaw"])) for r in names}
        yaws = {r: float(layout[r]["mount_yaw"]) for r in names}
        eng = VampCollisionEngine(
            getattr(vamp_mod, mod), objects, base_transforms=bases, mount_yaws=yaws,
            n_structural=CELL_N_STRUCTURAL, sphere_margin=sphere_margin,
            groups=link_groups(SPHERIZED_URDF[mod], CELL_N_STRUCTURAL),
            finger_patch=finger_patch_enabled(mod == "ur10e_rail"))
        for r in names:
            engines[r] = eng
    return engines
