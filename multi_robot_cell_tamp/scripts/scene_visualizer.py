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
"""

from __future__ import annotations

import math
import time

import rclpy
import yaml
from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from rclpy.duration import Duration
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from shape_msgs.msg import SolidPrimitive

# Must mirror include/multi_robot_cell_tamp/resample.hpp::Phase. The object is
# grasped when the gripper closes and released when it opens, so the PICK sample
# is the first GripClose and the PLACE sample is the first GripOpen. (Keying off
# object_state instead releases the object at ToHome -- i.e. only once the arm has
# already retreated from the place pose -- which is not when the gripper opens.)
PHASE_GRIP_CLOSE = 1
PHASE_GRIP_OPEN = 3


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


class SceneVisualizer:
    """Publishes the scene and animates pick/place on the executor's node."""

    def __init__(self, node, task_yaml_path: str):
        self.node = node
        self.log = node.get_logger()

        with open(task_yaml_path) as f:
            spec = yaml.safe_load(f)

        self.base_frame = spec.get("base_frame", "world")

        # id -> (size[3], pose_dict)
        self.fixtures: dict[str, tuple[list, dict]] = {}
        for fx in spec.get("fixtures") or []:
            self.fixtures[fx["id"]] = (list(fx["size"]), dict(fx["pose"]))

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

        # task id -> (object id, place_pose_dict)
        self.tasks: dict[str, tuple[str, dict]] = {}
        for tk in spec.get("tasks") or []:
            self.tasks[tk["id"]] = (tk["object"], dict(tk["place"]))

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

        self._events: list[dict] = []
        self._next = 0
        self._timer = None
        self.last_event_time = None  # rclpy.time.Time of the final event, once scheduled

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

    def _box_co(self, obj_id: str, size, pose: Pose, operation) -> CollisionObject:
        co = CollisionObject()
        co.header.frame_id = self.base_frame
        co.header.stamp = self.node.get_clock().now().to_msg()
        co.id = obj_id
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(size[0]), float(size[1]), float(size[2])]
        co.primitives.append(prim)
        co.primitive_poses.append(pose)
        co.operation = operation
        return co

    # ---- static scene ------------------------------------------------------- #

    def _static_scene(self) -> list[CollisionObject]:
        """Every fixture + every object (at its spawn pose) as ADD messages.

        Single source of geometry: sizes and poses come straight from the shared
        ``tamp_task.yaml``. Built once, published repeatedly by ``publish_static``.
        """
        msgs = [
            self._box_co(fid, size, self.pose_from(pose_d), CollisionObject.ADD)
            for fid, (size, pose_d) in self.fixtures.items()
        ]
        msgs += [
            self._box_co(oid, size, self.pose_from(spawn_d), CollisionObject.ADD)
            for oid, (size, spawn_d, _grasp_d) in self.objects.items()
        ]
        return msgs

    def clear_world(self, wait_timeout: float = 3.0) -> int:
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

        A missing move_group only warns -- the executor must never hang on the visualiser.
        """
        client = self.node.create_client(GetPlanningScene, "/get_planning_scene")
        try:
            if not client.wait_for_service(timeout_sec=wait_timeout):
                self.log.warn(
                    "scene: /get_planning_scene unavailable; cannot clear leftovers from a "
                    "previous run (objects from another task may still be in the scene)"
                )
                return 0
            request = GetPlanningScene.Request()
            request.components.components = (
                PlanningSceneComponents.WORLD_OBJECT_NAMES
                | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
            )
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=wait_timeout)
            if not future.done() or future.result() is None:
                self.log.warn("scene: /get_planning_scene did not answer; not clearing")
                return 0
            scene = future.result().scene
        finally:
            self.node.destroy_client(client)

        removed = []
        for attached in scene.robot_state.attached_collision_objects:
            msg = AttachedCollisionObject()
            msg.link_name = attached.link_name
            msg.object.id = attached.object.id
            msg.object.operation = CollisionObject.REMOVE
            self._aco_pub.publish(msg)
            removed.append(f"{attached.object.id}(attached)")
        for obj in scene.world.collision_objects:
            msg = CollisionObject()
            msg.header.frame_id = self.base_frame
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.id = obj.id
            msg.operation = CollisionObject.REMOVE
            self._co_pub.publish(msg)
            removed.append(obj.id)

        if removed:
            self.log.info(f"scene: cleared {len(removed)} leftover object(s): "
                          f"{', '.join(sorted(removed))}")
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
        for i in range(max(1, bursts)):
            for co in msgs:
                co.header.stamp = self.node.get_clock().now().to_msg()
                self._co_pub.publish(co)
            if i + 1 < max(1, bursts):
                time.sleep(max(0.0, burst_interval))

        self.log.info(
            f"scene: published {len(self.fixtures)} fixture(s) + "
            f"{len(self.objects)} object(s) to /collision_object "
            f"({max(1, bursts)}x, {self._co_pub.get_subscription_count()} subscriber(s))"
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
            # Attach with EXPLICIT geometry: the object's box, placed in the link
            # frame at grasp^-1 (object-in-EE). This pins the transport pose to the
            # same grasp the trajectory planner used, instead of letting move_group
            # derive it from the live robot state (which lags the commanded state
            # and made the object ride at a wrong vertical offset the whole way).
            aco = AttachedCollisionObject()
            aco.link_name = ev["attach_link"]
            aco.object.id = ev["object"]
            aco.object.header.frame_id = ev["attach_link"]
            aco.object.operation = CollisionObject.ADD
            prim = SolidPrimitive()
            prim.type = SolidPrimitive.BOX
            prim.dimensions = [
                float(ev["size"][0]),
                float(ev["size"][1]),
                float(ev["size"][2]),
            ]
            aco.object.primitives.append(prim)
            aco.object.primitive_poses.append(ev["obj_in_link"])
            aco.touch_links = ev["touch_links"]
            self._aco_pub.publish(aco)
            self.log.info(
                f"scene: PICK  {ev['robot']}/{ev['task']} attach '{ev['object']}' "
                f"-> {ev['attach_link']}"
            )
        else:
            # Detach (returns the object to the world), clear it, then re-add it as a
            # fresh world box at the place pose. Mirrors the generator's REMOVE->ADD.
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

            self._co_pub.publish(
                self._box_co(ev["object"], ev["size"], ev["place_pose"], CollisionObject.ADD)
            )
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
