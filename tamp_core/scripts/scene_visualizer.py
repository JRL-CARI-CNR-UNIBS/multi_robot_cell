#!/usr/bin/env python3
"""Render and animate the TAMP scene in RViz while the schedule replays.

Helper for ``schedule_executor.py``. The executor drives the *arms*; this module
drives the *world*: it publishes the tray/lid/boxes into the live planning scene
and then, on the shared Delta-t clock, attaches each object to the picking
gripper and lands it at its place pose -- so RViz shows the objects being carried,
not just the arms waving.

The single source of geometry is ``config/tamp_task.yaml`` -- the SAME file the
offline generators read. Nothing here recomputes a pose: sizes and spawn/place
poses come straight from that YAML, and the pick/place *instants* come straight
from the trajectory artifact's per-sample ``phase[]`` (the ``mrct`` ``Phase``
enum, encoded as ints): the object attaches when the gripper closes and is
released when it opens. So the whole pipeline -- planning,
collision, execution, visualization -- shares one geometry description.

Attach carries EXPLICIT geometry: the ``AttachedCollisionObject`` includes the
object's box and its pose in the link frame, set to ``grasp^-1`` (the object
expressed in the attach link) from the SAME ``grasp`` the generators use. This
pins the transport pose deterministically, rather than letting MoveIt derive the
object->link transform from the live robot state -- that capture lags the
commanded controller state and left the object riding at a wrong vertical offset
for the whole carry. Release still detaches and re-adds the box at ``task.place``.

MESHES (optional)
-----------------
A fixture or object may carry ``mesh: {file: meshes/x.stl, scale: 1.0}`` next to its
mandatory ``size`` (which stays the bounding box every other stage reads). The mesh ORIGIN is
the centre of its bounding box, so ``spawn`` / ``place`` / ``grasp`` mean exactly what they
mean for a box. Wherever a box was published -- static fixture, static object at spawn,
attach in the link frame at ``grasp^-1``, place -- a ``CollisionObject`` with ``meshes`` /
``mesh_poses`` (``shape_msgs/Mesh``) is published instead, at the very same pose. ``file`` is
resolved against the YAML's own directory and, failing that, against the installed
``config/`` (same rule as ``trajectory_generator``). The STL is read with numpy only (binary,
ASCII as a fallback) and cached per (file, scale). A scene without ``mesh:`` is byte-for-byte
what it was.

COLOURS (optional)
------------------
A fixture or object may also carry ``color: [r, g, b, a]`` (each in 0..1; ``a`` defaults to 1
when only three are given). A ``CollisionObject`` has no colour field, so colours travel as
``moveit_msgs/PlanningScene.object_colors`` in a diff (``is_diff``) on ``/planning_scene``, which
move_group's monitor applies and forwards on ``/monitored_planning_scene`` (RViz).

A colour ALONE does not stick in move_group (MoveIt 2 Jazzy, read in planning_scene.cpp and
measured): the monitor keeps a diff scene over a parent and, on every publish cycle -- several
per second while an arm moves -- calls ``pushDiffs`` then ``clearDiffs``; ``pushDiffs`` copies a
colour to the parent only for an object whose geometry changed in that cycle, or for an
attached body when the robot state changed. A colour-only diff therefore vanishes at the next
cycle (it survived in a static test only because nothing moved), and attaching an object
deletes its colour from the parent. So every colour message carries, for an object lying in the
world, a re-ADD of that object's own last published geometry and pose (``_world_co``) in the
SAME diff: colour and geometry change in one cycle and are pushed together. An attached object
gets the colour alone, pushed as an attached body while the arm moves. Colours are sent after
every burst of the static scene, and after each attach and each place -- then again after
``RECOLOR_DELAYS_S``, because the attach / place REMOVE -> ADD travel on other topics whose order
relative to ``/planning_scene`` is not guaranteed (a REMOVE drops the colour; ``clear_world``
relies on that, so the next scene starts uncoloured). Uncoloured entries fall back to RViz's
"Scene Color". A scene without ``color:`` creates no ``/planning_scene`` publisher and publishes
exactly what it did before.
"""

from __future__ import annotations

import math
import os
import re
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    ObjectColor,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import GetPlanningScene
from rclpy.duration import Duration
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from std_msgs.msg import ColorRGBA

# Must mirror include/multi_robot_cell_tamp/resample.hpp::Phase. The object is
# grasped when the gripper closes and released when it opens, so the PICK sample
# is the first GripClose and the PLACE sample is the first GripOpen. (Keying off
# object_state instead releases the object at ToHome -- i.e. only once the arm has
# already retreated from the place pose -- which is not when the gripper opens.)
PHASE_GRIP_CLOSE = 1
PHASE_GRIP_OPEN = 3

# Pause after every message published to /collision_object, so a scene with many objects
# does not overflow move_group's subscriber queue (see `publish_static`).
PUBLISH_PACE_S = 0.02

# After a PICK or PLACE of a coloured object, its colour is re-sent this long after (see COLOURS
# in the module docstring): late enough that the attach / REMOVE -> ADD, which erase the colour,
# have certainly been applied.
RECOLOR_DELAYS_S = (0.2, 1.0)


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Roll/pitch/yaw (radians, intrinsic XYZ) -> geometry_msgs/Quaternion."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


# --------------------------------------------------------------------------- #
# Meshes: a minimal STL reader on numpy, and the YAML `mesh:` resolution.
# --------------------------------------------------------------------------- #

_STL_HEADER = 80
_STL_BINARY_RECORD = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
_VERTEX_RE = re.compile(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", re.IGNORECASE)


def read_stl(path: str) -> np.ndarray:
    """STL file -> (M, 3, 3) float32 array of triangle corners (a "soup", unscaled).

    Binary is recognised by its size: an 80-byte header, a uint32 triangle count N and
    exactly 50 bytes per triangle. That test, not the leading ``solid``, decides -- many
    exporters write ``solid`` into a binary header. Anything else is parsed as ASCII.
    """
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= _STL_HEADER + 4:
        n = int(np.frombuffer(data, dtype="<u4", count=1, offset=_STL_HEADER)[0])
        if len(data) == _STL_HEADER + 4 + 50 * n:
            rec = np.frombuffer(data, dtype=_STL_BINARY_RECORD, count=n, offset=_STL_HEADER + 4)
            return np.ascontiguousarray(rec["v"], dtype=np.float32)
    found = _VERTEX_RE.findall(data)
    if not found or len(found) % 3:
        raise ValueError(
            f"{path}: not a valid STL (neither binary of the exact size nor ASCII with "
            f"a whole number of facets; {len(found)} ASCII vertices found)")
    return np.array(found, dtype=np.float64).astype(np.float32).reshape(-1, 3, 3)


class MeshAsset:
    """One STL, scaled, as a ``shape_msgs/Mesh`` plus what a caller wants to check about it.

    ``msg`` shares vertices (STL repeats each corner once per facet; the message does not need
    to) but keeps EVERY facet, so ``n_triangles`` equals the file's count. ``lo``/``hi`` are the
    bounding-box corners after scaling, in the mesh's own frame.
    """

    def __init__(self, path: str, scale: float):
        tri = read_stl(path)
        corners = tri.reshape(-1, 3)
        uniq, inv = np.unique(corners, axis=0, return_inverse=True)
        idx = np.asarray(inv).reshape(-1, 3)
        verts = uniq.astype(np.float64) * float(scale)
        msg = Mesh()
        msg.vertices = [Point(x=float(v[0]), y=float(v[1]), z=float(v[2])) for v in verts]
        msg.triangles = [MeshTriangle(vertex_indices=[int(a), int(b), int(c)]) for a, b, c in idx]
        self.file = path
        self.scale = float(scale)
        self.msg = msg
        self.n_triangles = int(idx.shape[0])
        self.n_vertices = int(verts.shape[0])
        self.lo = verts.min(axis=0)
        self.hi = verts.max(axis=0)
        area2 = np.linalg.norm(np.cross(verts[idx[:, 1]] - verts[idx[:, 0]],
                                        verts[idx[:, 2]] - verts[idx[:, 0]]), axis=1)
        self.n_degenerate = int(np.count_nonzero(area2 < 1e-18))

    @property
    def extents(self) -> np.ndarray:
        return self.hi - self.lo

    @property
    def centre(self) -> np.ndarray:
        return 0.5 * (self.lo + self.hi)

    @property
    def resource(self) -> str:
        """``file://`` URI, as RViz's ``Marker.MESH_RESOURCE`` wants it (unscaled file)."""
        return "file://" + self.file


_MESH_CACHE: dict[tuple[str, float], MeshAsset] = {}


def resolve_mesh_file(name: str, yaml_dir: str) -> str:
    """Absolute path of a ``mesh: {file: ...}``: next to the YAML, else the installed config/.

    The generator resolves it the same way. The fallback is what lets an installed scene
    (``config/tamp_task_x.yaml``) and a copied one (``artifacts/runs/<n>/``) both find
    ``meshes/x.stl``; a saved run carries its own copy, found first.
    """
    if os.path.isabs(name):
        cands = [name]
    else:
        cands = [os.path.join(yaml_dir, name)]
        try:
            from ament_index_python.packages import get_package_share_directory

            cands.append(os.path.join(
                get_package_share_directory("multi_robot_cell_tamp"), "config", name))
        except Exception:  # noqa: BLE001 -- no ament index: the YAML's own folder only
            pass
    for c in cands:
        if os.path.isfile(c):
            return os.path.realpath(c)
    raise FileNotFoundError(
        f"mesh '{name}' not found; looked in: {', '.join(cands)}")


def load_mesh(path: str, scale: float = 1.0) -> MeshAsset:
    """Cached ``MeshAsset`` per (file, scale): the scene is rebuilt often, the STL is read once."""
    key = (path, float(scale))
    if key not in _MESH_CACHE:
        _MESH_CACHE[key] = MeshAsset(path, scale)
    return _MESH_CACHE[key]


def mesh_spec(entry: dict) -> tuple[str, float] | None:
    """``(file, scale)`` from a YAML fixture/object entry, or None when it has no ``mesh:``.

    Accepts ``mesh: {file: x.stl, scale: 0.001}`` and the shorthand ``mesh: x.stl``.
    """
    m = entry.get("mesh")
    if not m:
        return None
    if isinstance(m, str):
        return m, 1.0
    return str(m["file"]), float(m.get("scale", 1.0))


def color_spec(entry: dict) -> ColorRGBA | None:
    """``color: [r, g, b(, a)]`` of a YAML fixture/object entry, or None when it has none.

    Raises ValueError on anything else (wrong length, a value outside 0..1), so a typo fails
    the executor at construction rather than drawing the default colour silently.
    """
    c = entry.get("color")
    if c is None:
        return None
    try:
        vals = [float(v) for v in c]
    except (TypeError, ValueError):
        vals = []
    if len(vals) == 3:
        vals.append(1.0)
    if len(vals) != 4 or not all(0.0 <= v <= 1.0 for v in vals):
        raise ValueError(
            f"'{entry.get('id')}': color must be [r, g, b] or [r, g, b, a] with every value "
            f"in 0..1, got {c!r}")
    return ColorRGBA(r=vals[0], g=vals[1], b=vals[2], a=vals[3])


class SceneVisualizer:
    """Publishes the scene and animates pick/place on the executor's node."""

    def __init__(self, node, task_yaml_path: str):
        self.node = node
        self.log = node.get_logger()

        with open(task_yaml_path) as f:
            spec = yaml.safe_load(f)

        self.base_frame = spec.get("base_frame", "world")
        self._yaml_dir = os.path.dirname(os.path.realpath(task_yaml_path))

        # id -> MeshAsset, for the fixtures / objects that declare `mesh:`. Kept beside the
        # (size, pose) tuples below rather than inside them, so a scene without meshes has
        # exactly the structures it always had. Loaded here, before any motion: a missing
        # or unreadable file must fail the executor at construction, not mid-carry.
        self.fixture_mesh: dict[str, MeshAsset] = {}
        self.object_mesh: dict[str, MeshAsset] = {}
        # id -> ColorRGBA, for the fixtures / objects that declare `color:` (either kind: the
        # ids share one planning-scene namespace).
        self.colors: dict[str, ColorRGBA] = {}

        # id -> (size[3], pose_dict)
        self.fixtures: dict[str, tuple[list, dict]] = {}
        for fx in spec.get("fixtures") or []:
            self.fixtures[fx["id"]] = (list(fx["size"]), dict(fx["pose"]))
            self._load_entry_mesh(fx, self.fixture_mesh, "fixture")
            self._load_entry_color(fx)

        # id -> (size[3], spawn_pose_dict, grasp_dict)
        # `grasp` is the attach-link pose in the OBJECT frame (T_object->EE); we
        # keep it so PICK can attach with the SAME explicit offset the trajectory
        # generator used, instead of relying on move_group's live-state capture.
        self.objects: dict[str, tuple[list, dict, dict]] = {}
        for ob in spec.get("objects") or []:
            self.objects[ob["id"]] = (
                list(ob["size"]),
                dict(ob["spawn"]),
                dict(ob.get("grasp") or {}),
            )
            self._load_entry_mesh(ob, self.object_mesh, "object")
            self._load_entry_color(ob)

        # task id -> (object id, place_pose_dict)
        self.tasks: dict[str, tuple[str, dict]] = {}
        for tk in spec.get("tasks") or []:
            self.tasks[tk["id"]] = (tk["object"], dict(tk["place"]))

        # A `slots:` entry (interchangeable-candidate scenes) expands into one
        # internal task per candidate, id `<slot_id>__<object_id>` -- same
        # convention trajectory_generator.cpp uses when it builds TaskDef from a
        # slot (loadTaskSpec). Only the candidate that was actually scheduled ever
        # shows up in `solution["assignments"]`, so populating every candidate
        # here is harmless -- the rest are simply never looked up.
        for sl in spec.get("slots") or []:
            place_d = dict(sl["place"])
            for obj_id in sl.get("candidates") or []:
                self.tasks[f'{sl["id"]}__{obj_id}'] = (obj_id, place_d)

        # robot name -> (attach_link, touch_links[])
        self.robots: dict[str, tuple[str, list]] = {}
        for name, cfg in (spec.get("robots") or {}).items():
            self.robots[name] = (cfg["attach_link"], list(cfg.get("touch_links") or []))

        # Latched (transient_local) so the PlanningSceneMonitor gets every message
        # even if it happens to subscribe a moment after we publish. Reliable, and
        # deep enough to keep all static objects + a burst of animation events.
        qos = QoSProfile(
            depth=50,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._co_pub = node.create_publisher(CollisionObject, "/collision_object", qos)
        self._aco_pub = node.create_publisher(
            AttachedCollisionObject, "/attached_collision_object", qos
        )
        # Only a coloured scene gets this publisher: an uncoloured one stays exactly as it was.
        self._ps_pub = (node.create_publisher(PlanningScene, "/planning_scene", qos)
                        if self.colors else None)
        self._recolor_timers: list = []
        # id -> the last ADD published for a coloured object that lies in the WORLD (not attached):
        # what a colour message re-sends so move_group keeps the colour (COLOURS, docstring).
        self._world_co: dict[str, CollisionObject] = {}

        self._events: list[dict] = []
        self._next = 0
        self._timer = None
        self.last_event_time = None  # rclpy.time.Time of the final event, once scheduled

    # ---- meshes ------------------------------------------------------------- #

    def _load_entry_mesh(self, entry: dict, into: dict, kind: str) -> None:
        """Load ``entry['mesh']`` (if any) into ``into[entry['id']]``; warn on odd geometry.

        The convention is that the mesh origin is its bounding-box centre and ``size`` its
        bounding box. Neither is enforced here (the generator does not enforce it either),
        but a mesh that violates them would be drawn away from the pose the planner used,
        so it is worth saying once.
        """
        spec = mesh_spec(entry)
        if spec is None:
            return
        name, scale = spec
        asset = load_mesh(resolve_mesh_file(name, self._yaml_dir), scale)
        into[entry["id"]] = asset
        size = np.array([float(v) for v in entry["size"]])
        off = np.abs(asset.centre)
        dev = np.abs(asset.extents - size)
        msg = (f"scene: {kind} '{entry['id']}' mesh {asset.file} "
               f"x{asset.scale:g}: {asset.n_triangles} triangles")
        if asset.n_degenerate:
            msg += f", {asset.n_degenerate} degenerate"
        self.log.info(msg)
        if off.max() > 2e-3 or dev.max() > max(2e-3, 0.02 * size.max()):
            self.log.warn(
                f"scene: {kind} '{entry['id']}': mesh bbox extents "
                f"{np.round(asset.extents, 4).tolist()} centre {np.round(asset.centre, 4).tolist()}"
                f" disagree with size {size.tolist()} / origin-at-bbox-centre; the mesh will be "
                f"drawn off the pose the planner used")

    # ---- colours ------------------------------------------------------------ #

    def _load_entry_color(self, entry: dict) -> None:
        c = color_spec(entry)
        if c is not None:
            self.colors[entry["id"]] = c

    def publish_colors(self, ids=None) -> None:
        """One ``PlanningScene`` diff carrying ``object_colors`` for ``ids`` (default: all).

        For every id that lies in the world, the same diff re-ADDs its last published geometry
        (``_world_co``), which is what makes move_group keep the colour (COLOURS, docstring); an
        attached id gets the colour alone. A no-op for ids without a colour, and for a scene
        without any.
        """
        if self._ps_pub is None:
            return
        ids = list(self.colors) if ids is None else [i for i in ids if i in self.colors]
        if not ids:
            return
        ps = PlanningScene()
        ps.is_diff = True
        ps.robot_state.is_diff = True
        ps.object_colors = [ObjectColor(id=i, color=self.colors[i]) for i in ids]
        stamp = self.node.get_clock().now().to_msg()
        for i in ids:
            co = self._world_co.get(i)
            if co is not None:
                co.header.stamp = stamp
                ps.world.collision_objects.append(co)
        self._ps_pub.publish(ps)

    def _recolor_later(self, obj_id: str) -> None:
        """Re-send ``obj_id``'s colour after each of ``RECOLOR_DELAYS_S`` (one-shot timers)."""
        if obj_id not in self.colors:
            return
        for delay in RECOLOR_DELAYS_S:
            holder = []

            def once(holder=holder):
                timer = holder[0]
                timer.cancel()
                self.publish_colors([obj_id])
                if timer in self._recolor_timers:
                    self._recolor_timers.remove(timer)
                self.node.destroy_timer(timer)

            timer = self.node.create_timer(delay, once)
            holder.append(timer)
            self._recolor_timers.append(timer)

    # ---- geometry helpers --------------------------------------------------- #

    def pose_from(self, d: dict) -> Pose:
        """{x,y,z,roll,pitch,yaw} (each defaulting to 0) -> geometry_msgs/Pose."""
        p = Pose()
        p.position.x = float(d.get("x", 0.0))
        p.position.y = float(d.get("y", 0.0))
        p.position.z = float(d.get("z", 0.0))
        p.orientation = quaternion_from_rpy(
            float(d.get("roll", 0.0)),
            float(d.get("pitch", 0.0)),
            float(d.get("yaw", 0.0)),
        )
        return p

    def grasp_inverse_pose(self, grasp_d: dict) -> Pose:
        """Pose of the OBJECT expressed in the attach-link frame = grasp^-1.

        ``grasp`` in tamp_task.yaml is the tool pose in the OBJECT frame
        (T_object->EE): the generator composes ``world_EE = spawn (x) grasp``
        (see trajectory_generator.cpp ``compose``). An AttachedCollisionObject
        carrying explicit geometry wants the object expressed IN the link frame,
        i.e. T_EE->object = grasp^-1 = (R^T, -R^T t). The gripper mount makes
        ee_link (tool0) and attach_link (robotiq_85_base_link) coincident (the
        fixed robotiq_85_base_joint has a zero origin), so this inverse is exactly
        the pose to publish under ``header.frame_id = attach_link``.

        For the boxes/lid (grasp z=0.16, roll=pi) this yields translation
        (0, 0, 0.16), roll=pi -- Rx(pi) is its own inverse and t is along z.
        """
        tx = float(grasp_d.get("x", 0.0))
        ty = float(grasp_d.get("y", 0.0))
        tz = float(grasp_d.get("z", 0.0))
        q = quaternion_from_rpy(
            float(grasp_d.get("roll", 0.0)),
            float(grasp_d.get("pitch", 0.0)),
            float(grasp_d.get("yaw", 0.0)),
        )
        # Inverse rotation = conjugate; inverse translation = -R^T t = rotate(-t)
        # by the conjugate quaternion (v' = v + 2 w (u x v) + 2 u x (u x v)).
        w, ux, uy, uz = q.w, -q.x, -q.y, -q.z  # conjugate
        vx, vy, vz = -tx, -ty, -tz
        # u x v
        cx = uy * vz - uz * vy
        cy = uz * vx - ux * vz
        cz = ux * vy - uy * vx
        # u x (u x v)
        ccx = uy * cz - uz * cy
        ccy = uz * cx - ux * cz
        ccz = ux * cy - uy * cx
        p = Pose()
        p.position.x = vx + 2.0 * w * cx + 2.0 * ccx
        p.position.y = vy + 2.0 * w * cy + 2.0 * ccy
        p.position.z = vz + 2.0 * w * cz + 2.0 * ccz
        p.orientation = Quaternion(w=w, x=ux, y=uy, z=uz)
        return p

    @staticmethod
    def add_shape(co: CollisionObject, size, pose: Pose, mesh: MeshAsset | None) -> None:
        """Put the object's geometry at ``pose`` (in ``co``'s frame): its mesh if it has one,
        else its bounding box. The single place where box-vs-mesh is decided, so the static
        scene, the attach and the place cannot disagree about what an object looks like."""
        if mesh is not None:
            co.meshes.append(mesh.msg)
            co.mesh_poses.append(pose)
            return
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(size[0]), float(size[1]), float(size[2])]
        co.primitives.append(prim)
        co.primitive_poses.append(pose)

    def _shape_co(self, obj_id: str, size, pose: Pose, operation,
                  mesh: MeshAsset | None = None) -> CollisionObject:
        co = CollisionObject()
        co.header.frame_id = self.base_frame
        co.header.stamp = self.node.get_clock().now().to_msg()
        co.id = obj_id
        self.add_shape(co, size, pose, mesh)
        co.operation = operation
        return co

    # ---- static scene ------------------------------------------------------- #

    def _static_scene(self) -> list[CollisionObject]:
        """Every fixture + every object (at its spawn pose) as ADD messages.

        Single source of geometry: sizes and poses come straight from the shared
        ``tamp_task.yaml``. Built once, published repeatedly by ``publish_static``.
        """
        msgs = [
            self._shape_co(fid, size, self.pose_from(pose_d), CollisionObject.ADD,
                           self.fixture_mesh.get(fid))
            for fid, (size, pose_d) in self.fixtures.items()
        ]
        msgs += [
            self._shape_co(oid, size, self.pose_from(spawn_d), CollisionObject.ADD,
                           self.object_mesh.get(oid))
            for oid, (size, spawn_d, _grasp_d) in self.objects.items()
        ]
        return msgs

    def _query_scene(self, wait_timeout: float):
        """World objects and attached objects from ``/get_planning_scene``, or None (warned)."""
        client = self.node.create_client(GetPlanningScene, "/get_planning_scene")
        try:
            if not client.wait_for_service(timeout_sec=wait_timeout):
                self.log.warn(
                    "scene: /get_planning_scene unavailable; cannot clear leftovers from a "
                    "previous run (objects from another task may still be in the scene)"
                )
                return None
            request = GetPlanningScene.Request()
            request.components.components = (
                PlanningSceneComponents.WORLD_OBJECT_NAMES
                | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
            )
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=wait_timeout)
            if not future.done() or future.result() is None:
                self.log.warn("scene: /get_planning_scene did not answer; not clearing")
                return None
            return future.result().scene
        finally:
            self.node.destroy_client(client)

    def clear_world(self, wait_timeout: float = 3.0, rounds: int = 3) -> int:
        """Remove everything already in the planning scene. Returns how many.

        The planning scene is LIVE and outlives any one run, while this class only ever
        published ADDs -- so objects accumulated across scenes. Running `nominal` after
        `tower` left `box_4` sitting on the table (it exists in tower and not in nominal);
        the reverse leaves the lid and four tray walls. The stale geometry is real to
        move_group, so it is not merely a rendering artifact: a subsequent *online* plan
        would avoid an obstacle that is not there.

        Nothing here can be inferred from the task file, precisely because the leftovers
        are the ids the new task does NOT mention. So ask the scene what it is holding and
        remove all of it, rather than removing what we expect to find. Attached objects are
        cleared too: a run interrupted mid-carry leaves one welded to a gripper.

        The REMOVEs are sent once, so the first one can be lost to the same late-subscriber
        race ``publish_static`` works around (measured: the first REMOVE of a fresh publisher
        dropped on 3 runs out of 3). Hence up to ``rounds`` passes: after each, ask the scene
        again and remove what is still there. A REMOVE also drops the object's colour, which
        is what lets an uncoloured scene follow a coloured one.

        A missing move_group only warns -- the executor must never hang on the visualiser.
        """
        removed = []
        for rnd in range(max(1, rounds)):
            if rnd:
                time.sleep(0.2)          # let move_group apply the previous pass
            scene = self._query_scene(wait_timeout)
            if scene is None:
                break
            attached_objs = scene.robot_state.attached_collision_objects
            world_objs = scene.world.collision_objects
            if not attached_objs and not world_objs:
                break
            if rnd + 1 == max(1, rounds):
                self.log.warn(
                    f"scene: {len(attached_objs) + len(world_objs)} object(s) survived "
                    f"{rnd} clearing pass(es): "
                    f"{', '.join(sorted(o.object.id for o in attached_objs))} "
                    f"{', '.join(sorted(o.id for o in world_objs))}; removing once more")
            for attached in attached_objs:
                msg = AttachedCollisionObject()
                msg.link_name = attached.link_name
                msg.object.id = attached.object.id
                msg.object.operation = CollisionObject.REMOVE
                self._aco_pub.publish(msg)
                removed.append(f"{attached.object.id}(attached)")
            for obj in world_objs:
                msg = CollisionObject()
                msg.header.frame_id = self.base_frame
                msg.header.stamp = self.node.get_clock().now().to_msg()
                msg.id = obj.id
                msg.operation = CollisionObject.REMOVE
                self._co_pub.publish(msg)
                time.sleep(PUBLISH_PACE_S)   # same queue overflow as the ADDs, see publish_static
                removed.append(obj.id)

        removed = sorted(set(removed))
        if removed:
            self.log.info(f"scene: cleared {len(removed)} leftover object(s): "
                          f"{', '.join(removed)}")
        return len(removed)

    def publish_static(
        self,
        wait_timeout: float = 5.0,
        bursts: int = 5,
        burst_interval: float = 0.15,
    ) -> None:
        """Add every fixture and every object (at its spawn pose) to the world.

        Defeats a publish-before-subscriber race: move_group's PlanningSceneMonitor
        subscribes to ``/collision_object`` with a *volatile* QoS, so a single latched
        publish that lands before it connects is lost -- the object would only appear
        later, at its place-time ADD. So we first wait (up to ``wait_timeout`` s) for at
        least one subscriber, then republish the whole static scene ``bursts`` times
        ~``burst_interval`` s apart. A missing monitor only warns and continues -- the
        generators each own a PlanningScene and the executor must not hang on RViz.
        """
        msgs = self._static_scene()
        self._world_co = {co.id: co for co in msgs if co.id in self.colors}

        # Wait for the PlanningSceneMonitor (or any subscriber) to connect. A latched
        # message published into a void with zero volatile subscribers is never seen.
        deadline = time.monotonic() + max(0.0, wait_timeout)
        while self._co_pub.get_subscription_count() == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._co_pub.get_subscription_count() == 0:
            self.log.warn(
                f"scene: no /collision_object subscriber after {wait_timeout:.1f} s; "
                "publishing anyway (objects may not reach the planning scene)"
            )

        # Wipe first, so the scene ends up being exactly what this task file describes
        # rather than it plus whatever the last run left. Ordering is safe: REMOVE and ADD
        # go out on the same RELIABLE publisher, so an id present in both scenes is removed
        # and re-added in that order.
        self.clear_world()

        # Republish a handful of times over a short window: robust against a monitor
        # that connects a beat late, and harmless (repeated ADD of the same id is a
        # no-op once present).
        #
        # Each message is followed by a short pause (`PUBLISH_PACE_S`): published back to back,
        # a scene of ~20+ objects overflows the subscriber's queue in move_group and some ADDs
        # are silently dropped, and which ones varies from run to run (measured: 24 objects
        # gave 14, 14 and 19 in the planning scene). Pacing every publish, not just every
        # burst, is what lets the whole scene arrive.
        for i in range(max(1, bursts)):
            for co in msgs:
                co.header.stamp = self.node.get_clock().now().to_msg()
                self._co_pub.publish(co)
                time.sleep(PUBLISH_PACE_S)
            # Colours after the burst's ADDs; a burst of the next round re-sends them, which
            # also covers the first burst's colours landing before clear_world's REMOVEs.
            if self.colors:
                self.publish_colors()
                time.sleep(PUBLISH_PACE_S)
            if i + 1 < max(1, bursts):
                time.sleep(max(0.0, burst_interval))

        self.log.info(
            f"scene: published {len(self.fixtures)} fixture(s) + "
            f"{len(self.objects)} object(s) "
            f"({len(self.fixture_mesh) + len(self.object_mesh)} as meshes) to /collision_object "
            f"({max(1, bursts)}x, {self._co_pub.get_subscription_count()} subscriber(s))"
            + (f"; {len(self.colors)} colour(s) to /planning_scene "
               f"({self._ps_pub.get_subscription_count()} subscriber(s))" if self.colors else "")
        )

    # ---- animation schedule ------------------------------------------------- #

    @staticmethod
    def _first_index(states, value, start=0):
        for i in range(start, len(states)):
            if states[i] == value:
                return i
        return None

    def schedule_from(self, artifact: dict, solution: dict, start_time,
                      node_base: dict | None = None) -> None:
        """Build the time-sorted pick/place event list.

        ``start_time`` is the shared t=0 instant (an ``rclpy.time.Time``); an event
        at local sample ``k`` of a task starting at ``start_slot`` fires at
        ``start_time + (start_slot + k) * delta_t``.

        ``node_base`` maps ``(robot, task)`` to that task's first TPG node index. Pass it
        under graph dispatch (ADR-0007): every event then also carries ``node``, and
        ``tick_nodes`` fires it when the arm REACHES that node rather than at a wall-clock
        instant. The two triggers are not interchangeable -- under the graph a sample's
        time is not known in advance, since a robot may be held at any node, so keying the
        scene to the schedule's slots would attach a box while the arm is still waiting
        for it.
        """
        delta_t = float(artifact["delta_t"])
        traj_by_key = {(t["robot"], t["task"]): t for t in artifact["trajectories"]}

        events: list[dict] = []
        for task_id, a in solution["assignments"].items():
            robot = a["robot"]
            start_slot = int(a["start_slot"])
            key = (robot, task_id)
            tr = traj_by_key.get(key)
            if tr is None:
                self.log.warn(f"scene: no trajectory for {key}; task {task_id} not animated")
                continue
            if robot not in self.robots:
                self.log.warn(f"scene: robot '{robot}' not in task YAML; task {task_id} not animated")
                continue

            # A process (weld) task moves no object: its trajectory writes "object": ""
            # and has no GripClose/GripOpen, so there is nothing to attach or place.
            # process_commander animates it instead.
            if not tr["object"]:
                continue

            obj_id = tr["object"]
            phases = tr["phase"]
            attach_link, touch_links = self.robots[robot]

            # Attach when the gripper closes; release when it opens (at the place
            # pose, before the arm retreats) -- not at the ToHome/AtPlace flip.
            pick_k = self._first_index(phases, PHASE_GRIP_CLOSE)
            place_k = None
            if pick_k is not None:
                place_k = self._first_index(phases, PHASE_GRIP_OPEN, start=pick_k)

            if pick_k is None or place_k is None:
                self.log.warn(
                    f"scene: task {task_id} phase[] has no GripClose->GripOpen "
                    f"transition (pick={pick_k}, place={place_k}); not animated"
                )
                continue

            place_pose_d = self.tasks.get(task_id, (None, None))[1]
            obj_entry = self.objects.get(obj_id)
            if place_pose_d is None or obj_entry is None:
                self.log.warn(
                    f"scene: task {task_id} / object {obj_id} missing in YAML; not animated"
                )
                continue
            size, _spawn_d, grasp_d = obj_entry

            base = None if node_base is None else node_base.get((robot, task_id))
            events.append(
                {
                    "time": start_time + Duration(seconds=(start_slot + pick_k) * delta_t),
                    "node": None if base is None else base + pick_k,
                    "kind": "pick",
                    "object": obj_id,
                    "attach_link": attach_link,
                    "touch_links": touch_links,
                    # Explicit attach geometry so the transport pose is exact and
                    # independent of move_group's live-state timing: the object's
                    # own size, placed in the link frame at grasp^-1.
                    "size": size,
                    "mesh": self.object_mesh.get(obj_id),
                    "obj_in_link": self.grasp_inverse_pose(grasp_d),
                    "task": task_id,
                    "robot": robot,
                }
            )
            events.append(
                {
                    "time": start_time + Duration(seconds=(start_slot + place_k) * delta_t),
                    "node": None if base is None else base + place_k,
                    "kind": "place",
                    "object": obj_id,
                    "attach_link": attach_link,
                    "size": size,
                    "mesh": self.object_mesh.get(obj_id),
                    "place_pose": self.pose_from(place_pose_d),
                    "task": task_id,
                    "robot": robot,
                }
            )

        events.sort(key=lambda e: e["time"].nanoseconds)
        self._events = events
        self._next = 0
        self.last_event_time = events[-1]["time"] if events else start_time
        self.log.info(f"scene: scheduled {len(events)} animation event(s)")

    def start(self) -> None:
        """Create the ~50 ms timer that fires due events. Idempotent."""
        if self._timer is None:
            self._timer = self.node.create_timer(0.05, self.tick)

    def tick(self) -> None:
        now = self.node.get_clock().now()
        while self._next < len(self._events) and self._events[self._next]["time"] <= now:
            self._fire(self._events[self._next])
            self._next += 1

    def _fire(self, ev: dict) -> None:
        if ev["kind"] == "pick":
            # Attach with EXPLICIT geometry: the object's box (or mesh), placed in the link
            # frame at grasp^-1 (object-in-EE). This pins the transport pose to the
            # same grasp the trajectory planner used, instead of letting move_group
            # derive it from the live robot state (which lags the commanded state
            # and made the object ride at a wrong vertical offset the whole way).
            aco = AttachedCollisionObject()
            aco.link_name = ev["attach_link"]
            aco.object.id = ev["object"]
            aco.object.header.frame_id = ev["attach_link"]
            aco.object.operation = CollisionObject.ADD
            # A box, or the object's mesh when it has one -- at the same pose either way.
            self.add_shape(aco.object, ev["size"], ev["obj_in_link"], ev.get("mesh"))
            aco.touch_links = ev["touch_links"]
            self._aco_pub.publish(aco)
            # Attached now: its colour travels alone (COLOURS, docstring); re-sent after the
            # attach, which deletes the colour, has certainly been applied.
            self._world_co.pop(ev["object"], None)
            self.publish_colors([ev["object"]])
            self._recolor_later(ev["object"])
            self.log.info(
                f"scene: PICK  {ev['robot']}/{ev['task']} attach '{ev['object']}' "
                f"-> {ev['attach_link']}"
            )
        else:
            # Detach (returns the object to the world), clear it, then re-add it as a
            # fresh world box (or mesh) at the place pose. Mirrors the generator's REMOVE->ADD.
            det = AttachedCollisionObject()
            det.link_name = ev["attach_link"]
            det.object.id = ev["object"]
            det.object.operation = CollisionObject.REMOVE
            self._aco_pub.publish(det)

            rm = CollisionObject()
            rm.id = ev["object"]
            rm.header.frame_id = self.base_frame
            rm.operation = CollisionObject.REMOVE
            self._co_pub.publish(rm)

            add = self._shape_co(ev["object"], ev["size"], ev["place_pose"], CollisionObject.ADD,
                                 ev.get("mesh"))
            self._co_pub.publish(add)
            # The REMOVE above erased the colour; re-send it with the placed geometry now and,
            # since the REMOVE travels on another topic, again shortly after (COLOURS, docstring).
            if ev["object"] in self.colors:
                self._world_co[ev["object"]] = add
            self.publish_colors([ev["object"]])
            self._recolor_later(ev["object"])
            self.log.info(
                f"scene: PLACE {ev['robot']}/{ev['task']} place '{ev['object']}'"
            )

    # ---- introspection for the executor ------------------------------------- #

    def pending(self) -> bool:
        """True while animation events remain unfired."""
        if any("fired" in e for e in self._events):     # node-keyed run
            return any(not e.get("fired") for e in self._events)
        return self._next < len(self._events)

    def tick_nodes(self, reached: dict) -> None:
        """Node-keyed twin of ``tick``: fire what the arms have actually reached.

        Scans every unfired event rather than walking a prefix, because the list is
        sorted by nominal TIME and arrival order under the graph is not that order --
        a robot held at a type-2 edge lets the other overtake it.
        """
        for ev in self._events:
            if ev.get("fired") or ev.get("node") is None:
                continue
            if reached.get(ev["robot"], -1) >= ev["node"]:
                self._fire(ev)
                ev["fired"] = True
