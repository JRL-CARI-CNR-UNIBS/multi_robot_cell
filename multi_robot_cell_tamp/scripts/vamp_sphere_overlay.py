#!/usr/bin/env python3
"""Overlay the VAMP collision spheres on the mesh arms in RViz during a replay.

A sibling to ``scene_visualizer.py``: where that module animates the *world*
(tray/lid/boxes), this node animates the *collision model* -- the exact
sphere-shells the VAMP engine (ADR-0005 Phase 2) checks -- riding along with both
arms while ``execute_schedule.launch.py`` replays a solved TAMP schedule. You see,
overlaid on the robots, precisely the geometry that produced the forbidden
offsets.

Unlike ``scene_visualizer`` (which runs *on the executor's node*, so it shares the
executor's exact shared-clock ``start`` instant), this is a *standalone* rclpy
node launched alongside the executor. It therefore reconstructs t=0 itself:

  * Provisionally, ``t0 = now() + start_delay`` -- the SAME convention
    ``schedule_executor`` uses for its shared ``header.stamp`` (both launched
    together by the one launch file).
  * Then, best-effort, it RE-ANCHORS ``t0`` to the first real arm motion seen on
    ``/joint_states`` (``anchor_on_motion``, default true): the executor's actual
    dispatch trails its node start by its controller/gripper/scene setup, so the
    wall-clock provisional would run a beat ahead. Anchoring on observed motion
    removes that skew and locks the shells to the mesh arms. Gripper joints are
    ignored so the executor's pre-dispatch ``init_open()`` is not mistaken for the
    start of the schedule.

The spheres are PRE-COMPUTED -- this node NEVER imports vamp. Run the prep step
once (in ``.venv_vamp``, viz-only, optional) to (re)generate the ``.npz``:

    .venv_vamp/bin/python scripts/dump_vamp_spheres.py

(defaults read/write the persistent ``artifacts/`` dir:
``artifacts/tamp_trajectories.json`` -> ``artifacts/vamp/tamp_spheres.npz``)

That ``.npz`` is a VISUALISATION artifact (it carries geometry) and is NOT the
geometry-free seam -- it never touches ``tamp_scheduler`` (ADR-0002).

npz format (see ``dump_vamp_spheres.py``): ``manifest`` (JSON: delta_t, keys,
margin, n_structural, n_spheres) plus, per key ``robot|task``, arrays
``<robot>__<task>__centres`` float32 (K, M, 3) WORLD-FRAME centres and
``<robot>__<task>__radii`` float32 (K, M). M = n_spheres + 1 object slot (last);
a DETACHED object has radius ``-inf`` -- non-finite radii are skipped (drawn as a
DELETE), so the carried-box shell appears only while grasped.

Markers: ``visualization_msgs/MarkerArray`` of SPHERE markers in the ``world``
frame (centres are already world-frame -- no TF). One stable marker id per sphere,
namespaced ``vamp_spheres/<robot>``, so each tick overwrites the last. robot1 and
robot2 get distinct colours; the carried-object slot gets a highlight colour.

Add a ``MarkerArray`` display on ``/vamp_collision_spheres`` in RViz to see it.

    ros2 launch multi_robot_cell_bringup start.launch.py            # terminal A
    ros2 launch multi_robot_cell_tamp execute_schedule.launch.py \\   # terminal B
        visualize_spheres:=true
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

# Joint-name fragments that identify an ARM/rail joint (as opposed to a gripper
# joint). Only these are watched for the motion re-anchor, so the executor's
# pre-dispatch gripper open() is not mistaken for the schedule starting.
ARM_JOINT_KEYS = ("linear_guide", "shoulder", "elbow", "wrist")

# Persistent (non-/tmp) artifacts dir under the package SOURCE tree, for the node's
# standalone `ros2 run` defaults; the launch file passes explicit paths that override
# these. realpath() resolves the --symlink-install symlink back to source.
_ARTIFACTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "artifacts", "vamp"
)

# Per-robot RGB for the arm sphere shells (alpha applied from the `alpha` param).
ROBOT_COLORS = {
    "robot1": (0.20, 0.55, 1.00),  # blue
    "robot2": (1.00, 0.55, 0.10),  # orange
}
FALLBACK_COLOR = (0.65, 0.65, 0.65)  # any other robot name
# The carried-object slot (last sphere) -- a highlight so the grasped box pops.
OBJECT_COLOR = (0.20, 1.00, 0.35)  # green
OBJECT_ALPHA_BOOST = 0.20


class VampSphereOverlay(Node):
    def __init__(self):
        super().__init__("vamp_sphere_overlay")

        self.declare_parameter("spheres_file", os.path.join(_ARTIFACTS, "tamp_spheres.npz"))
        self.declare_parameter("solution_file", os.path.join(_ARTIFACTS, "tamp_solution.json"))
        # Declared for symmetry with the executor / launch (one shared set of
        # args). The overlay is self-sufficient from the npz + solution; these are
        # accepted but not required for rendering.
        self.declare_parameter("traj_file", "")
        self.declare_parameter("task_file", "")
        # Same convention as schedule_executor: t0 = now() + start_delay.
        self.declare_parameter("start_delay", 2.0)
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("alpha", 0.35)
        self.declare_parameter("publish_rate", 20.0)
        # Re-anchor t0 to the first arm motion on /joint_states (removes the
        # executor's setup-lag skew). Falls back to the wall-clock provisional if
        # no motion is detected.
        self.declare_parameter("anchor_on_motion", True)
        self.declare_parameter("motion_threshold", 0.02)  # rad
        # Shut the node down once the schedule has fully played out, so
        # `visualize_spheres:=true` exits with the executor instead of spinning
        # forever. `end_settle` keeps the final (home) shells on screen a moment
        # past the makespan before exiting.
        self.declare_parameter("auto_stop", True)
        self.declare_parameter("end_settle", 2.0)  # seconds shown past makespan

        spheres_file = self.get_parameter("spheres_file").value
        solution_file = self.get_parameter("solution_file").value
        self.start_delay = float(self.get_parameter("start_delay").value)
        self.frame_id = self.get_parameter("frame_id").value
        self.alpha = float(self.get_parameter("alpha").value)
        rate = float(self.get_parameter("publish_rate").value)
        self.anchor_on_motion = bool(self.get_parameter("anchor_on_motion").value)
        self.motion_threshold = float(self.get_parameter("motion_threshold").value)
        self.auto_stop = bool(self.get_parameter("auto_stop").value)
        self.end_settle = float(self.get_parameter("end_settle").value)

        # ---- load the pre-computed spheres + the solved schedule ------------- #
        data = np.load(spheres_file)
        manifest = json.loads(str(data["manifest"]))
        self.delta_t = float(manifest["delta_t"])

        with open(solution_file) as f:
            sol = json.load(f)
        sol_dt = float(sol["delta_t"])
        if abs(self.delta_t - sol_dt) > 1e-9:
            raise ValueError(
                f"delta_t mismatch: spheres npz says {self.delta_t}, solution says "
                f"{sol_dt} -- these artifacts are from different runs"
            )

        # Slot past which the schedule is done and the node self-terminates.
        self.stop_slot = int(sol["makespan_slots"]) + int(round(self.end_settle / self.delta_t))

        # robot -> {"tasks": [(start_slot, K, centres, radii)], "home_c", "home_r"}
        self.by_robot: dict[str, dict] = {}
        used_start_slots: list[int] = []
        missing: list[str] = []
        for task_id, a in sol["assignments"].items():
            robot = a["robot"]
            start_slot = int(a["start_slot"])
            name = f"{robot}__{task_id}"  # npz array names swap '|' -> '__'
            ck, rk = f"{name}__centres", f"{name}__radii"
            if ck not in data.files or rk not in data.files:
                missing.append(f"{robot}|{task_id}")
                continue
            centres = data[ck]
            radii = data[rk]
            K = int(centres.shape[0])
            self.by_robot.setdefault(robot, {"tasks": []})["tasks"].append(
                (start_slot, K, centres, radii)
            )
            used_start_slots.append(start_slot)

        # A stale npz (spheres from a different scene than the solution) is the classic
        # silent failure: the assigned tasks are missing, so those arms would freeze at
        # home while the others animate -- looking like the robots and their shells are
        # swapped. Fail LOUD and tell the user to regenerate, rather than mis-overlay.
        if missing:
            raise ValueError(
                f"sphere npz is missing {len(missing)} task(s) the solution assigns: "
                f"{missing}. The npz ({spheres_file}) is stale -- it was built from a "
                f"different scene than this schedule (npz has {manifest.get('keys')}). "
                f"Regenerate it for THIS scene:  .venv_vamp/bin/python "
                f"scripts/dump_vamp_spheres.py --task <the scene's tamp_task*.yaml>"
            )
        if not self.by_robot:
            raise ValueError(
                f"no (robot, task) spheres matched the solution assignments; "
                f"npz keys={manifest.get('keys')}"
            )

        # Sort each robot's tasks and cache its home shells (sample 0 of the
        # earliest task -- every task starts and ends at home).
        self.M = 0
        for robot, d in self.by_robot.items():
            d["tasks"].sort(key=lambda t: t[0])
            _, _, c0, r0 = d["tasks"][0]
            d["home_c"] = c0[0]
            d["home_r"] = r0[0]
            self.M = max(self.M, int(c0.shape[1]))

        # Earliest slot at which any arm moves -> the re-anchor reference slot.
        self.first_motion_slot = min(used_start_slots) if used_start_slots else 0

        # ---- clock / t0 ------------------------------------------------------ #
        # Provisional shared start, same convention as schedule_executor. May be
        # overwritten once real motion is seen (see _joint_state_cb).
        self.t0 = self.get_clock().now() + Duration(seconds=self.start_delay)
        self.anchored = not self.anchor_on_motion
        self._ref_positions: dict[str, float] | None = None

        # ---- pub / sub / timer ---------------------------------------------- #
        qos = QoSProfile(
            depth=10,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._pub = self.create_publisher(MarkerArray, "/vamp_collision_spheres", qos)

        if self.anchor_on_motion:
            self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)

        # Clear any stale markers from a previous run before we start drawing.
        self._delete_all()

        period = 1.0 / rate if rate > 0 else 0.05
        self.create_timer(period, self._tick)
        self.get_logger().info(
            f"vamp sphere overlay: {sum(len(d['tasks']) for d in self.by_robot.values())} "
            f"task(s) across {len(self.by_robot)} robot(s), M={self.M} spheres, "
            f"delta_t={self.delta_t}s; frame '{self.frame_id}', "
            f"start_delay {self.start_delay:.1f}s, anchor_on_motion={self.anchor_on_motion}"
        )

    # ---- t0 re-anchor ------------------------------------------------------- #

    def _joint_state_cb(self, msg: JointState) -> None:
        """Lock t0 to the first observed arm motion (best-effort, once)."""
        if self.anchored:
            return
        current = {
            n: p
            for n, p in zip(msg.name, msg.position)
            if any(k in n for k in ARM_JOINT_KEYS)
        }
        if not current:
            return
        if self._ref_positions is None:
            # First reading: the arms are holding home. Take it as the reference.
            self._ref_positions = current
            return
        moved = max(
            (abs(p - self._ref_positions[n]) for n, p in current.items() if n in self._ref_positions),
            default=0.0,
        )
        if moved > self.motion_threshold:
            # Motion has begun. The current slot is first_motion_slot, so
            # t0 = now - first_motion_slot * delta_t.
            self.t0 = self.get_clock().now() - Duration(
                seconds=self.first_motion_slot * self.delta_t
            )
            self.anchored = True
            self.get_logger().info(
                f"anchored t0 to observed motion ({moved:.3f} rad) at "
                f"slot {self.first_motion_slot}"
            )

    # ---- rendering ---------------------------------------------------------- #

    def _current_slot(self) -> int:
        dt = (self.get_clock().now() - self.t0).nanoseconds / 1e9
        return int(round(dt / self.delta_t))

    def _color_for(self, robot: str) -> ColorRGBA:
        r, g, b = ROBOT_COLORS.get(robot, FALLBACK_COLOR)
        c = ColorRGBA()
        c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(self.alpha)
        return c

    def _object_color(self) -> ColorRGBA:
        r, g, b = OBJECT_COLOR
        c = ColorRGBA()
        c.r, c.g, c.b = float(r), float(g), float(b)
        c.a = float(min(1.0, self.alpha + OBJECT_ALPHA_BOOST))
        return c

    def _tick(self) -> None:
        slot = self._current_slot()
        # Schedule finished (plus the end settle): stop publishing and exit, so the
        # launch tears down with the executor instead of the timer spinning forever.
        if self.auto_stop and slot > self.stop_slot:
            self.get_logger().info(
                f"schedule complete (slot {slot} > {self.stop_slot}); overlay shutting down"
            )
            rclpy.shutdown()
            return
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()
        for robot, d in self.by_robot.items():
            centres, radii = self._sample_for(d, slot)
            arr.markers.extend(self._markers_for(robot, centres, radii, stamp))
        if arr.markers:
            self._pub.publish(arr)

    def _sample_for(self, d: dict, slot: int):
        """The (centres, radii) for a robot at a global slot, or its home shells.

        Idle (before/after the schedule, or between tasks -> arm at home) shows the
        home sample -- mirroring how scene_visualizer leaves idle objects at rest,
        so the shells sit on the mesh arm rather than blinking out.
        """
        for start_slot, K, centres, radii in d["tasks"]:
            if start_slot <= slot < start_slot + K:
                k = slot - start_slot
                return centres[k], radii[k]
        return d["home_c"], d["home_r"]

    def _markers_for(self, robot: str, centres, radii, stamp) -> list[Marker]:
        ns = f"vamp_spheres/{robot}"
        arm_color = self._color_for(robot)
        obj_color = self._object_color()
        markers: list[Marker] = []
        n = int(radii.shape[0])
        for i in range(n):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = ns
            m.id = i
            r = float(radii[i])
            c = centres[i]
            # A detached object (radius -inf) -> draw nothing: DELETE this id so a
            # previously-carried box shell disappears the instant it is released.
            if not math.isfinite(r) or r <= 0.0 or not np.all(np.isfinite(c)):
                m.action = Marker.DELETE
                markers.append(m)
                continue
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(c[0])
            m.pose.position.y = float(c[1])
            m.pose.position.z = float(c[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 2.0 * r  # diameter
            # The last slot is the carried-object sphere -> highlight colour.
            m.color = obj_color if i == n - 1 else arm_color
            markers.append(m)
        return markers

    def _delete_all(self) -> None:
        """Clear stale markers (a previous run) in each robot's namespace."""
        arr = MarkerArray()
        for robot in self.by_robot:
            m = Marker()
            m.ns = f"vamp_spheres/{robot}"
            m.action = Marker.DELETEALL
            arr.markers.append(m)
        if arr.markers:
            self._pub.publish(arr)


def main():
    rclpy.init()
    node = VampSphereOverlay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
