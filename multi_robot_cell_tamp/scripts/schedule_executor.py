#!/usr/bin/env python3
"""Replay a computed TAMP schedule on the cell's trajectory controllers.

The last stage of the pipeline: read the trajectory artifact (joint positions per
(robot, task), on the shared Delta-t clock) and the solved schedule (which robot
does which task, starting at which slot), build ONE joint trajectory per robot,
and send both to the 7-DOF ``..._linear_guide_joint_trajectory_controller``s so
the schedule plays out in RViz.

Reads only JSON, so it needs no OR-Tools -- the solve happened in the thesis venv
(``solve.py``) and handed this side a plain schedule. This keeps the seam intact:
no geometry and no solver crosses into the ROS runtime.

The shared clock is what makes the anti-collision offsets mean anything. Both
goals carry the SAME ``trajectory.header.stamp`` (a common start instant a moment
in the future), so both controllers begin slot 0 at exactly the same wall-clock
time. Dispatch skew between the two action calls therefore does not desynchronise
them: the timestamp, not the call time, sets t=0.

Alongside the arms it (optionally) animates the scene on the SAME clock:
``scene_visualizer`` attaches/releases the tray objects and ``gripper_commander``
opens/closes the fingers, both keyed to each task's GripClose/GripOpen dwell -- so
a box is grasped as the gripper closes and left at its place pose as it opens.
Geometry and gripper bindings come from the same ``tamp_task.yaml`` the offline
generators use (``visualize`` / ``actuate_grippers`` toggle each; both default on).

    ros2 launch multi_robot_cell_tamp execute_schedule.launch.py

Defaults read the persistent ``artifacts/`` dir (shared ``tamp_trajectories.json`` +
the VAMP schedule ``artifacts/vamp/tamp_solution.json``).

Needs the cell up (``start.launch.py``), because it commands the live controllers.
"""

from __future__ import annotations

import json
import os
import sys

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# scene_visualizer.py is installed next to this script (same lib/<pkg> dir). Put
# that dir on the path so the sibling import works when run as an installed node.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Persistent (non-/tmp) artifacts dir under the package SOURCE tree, for standalone
# `ros2 run` fallback defaults; execute_schedule.launch.py passes explicit paths that
# override these. realpath() resolves the --symlink-install symlink back to source.
_ARTIFACTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "artifacts"
)


def _duration(seconds: float) -> Duration:
    d = Duration()
    d.sec = int(seconds)
    d.nanosec = int(round((seconds - d.sec) * 1e9))
    return d


class ScheduleExecutor(Node):
    def __init__(self):
        super().__init__("schedule_executor")
        self.declare_parameter("traj_file", os.path.join(_ARTIFACTS, "tamp_trajectories.json"))
        self.declare_parameter("solution_file", os.path.join(_ARTIFACTS, "vamp", "tamp_solution.json"))
        self.declare_parameter("controller_suffix", "_linear_guide_joint_trajectory_controller")
        # Lead time before t=0, so both goals are accepted and queued before the
        # shared start instant arrives. Too short and one controller misses it.
        self.declare_parameter("start_delay", 2.0)
        # Scene visualization: render the tray/lid/boxes and animate attach/place
        # in RViz from the SAME tamp_task.yaml the offline generators use. Empty
        # task_file -> the installed config/tamp_task.yaml.
        self.declare_parameter("task_file", "")
        self.declare_parameter("visualize", True)
        # Send GripperCommand open/close in sync with each pick/place. Independent
        # of `visualize`: the fingers should actuate even with the scene render off.
        self.declare_parameter("actuate_grippers", True)

        traj_file = self.get_parameter("traj_file").value
        solution_file = self.get_parameter("solution_file").value
        self.suffix = self.get_parameter("controller_suffix").value
        self.start_delay = self.get_parameter("start_delay").value
        self.visualize = self.get_parameter("visualize").value
        self.actuate_grippers = self.get_parameter("actuate_grippers").value
        task_file = self.get_parameter("task_file").value

        with open(traj_file) as f:
            self.art = json.load(f)
        with open(solution_file) as f:
            self.sol = json.load(f)

        self.delta_t = float(self.art["delta_t"])
        if abs(self.delta_t - float(self.sol["delta_t"])) > 1e-12:
            raise ValueError(
                f"delta_t mismatch: trajectories say {self.delta_t}, "
                f"schedule says {self.sol['delta_t']} -- these artifacts are from different runs"
            )

        # (robot, task) -> trajectory entry, for quick lookup.
        self.traj = {(t["robot"], t["task"]): t for t in self.art["trajectories"]}
        self.robots = list(self.art["robots"])

        # The schedule's slots and the trajectories' lengths must describe the SAME plan.
        #
        # This is not a formality. Refinement (ADR-0008) produces a second, SHORTER set of
        # trajectories with its own schedule, so it is easy to pair one run's schedule with
        # another run's motions. Nothing downstream notices: `build_trajectory` drops points
        # whose slot has already passed (to remove the shared home knot between back-to-back
        # tasks), so trajectories that are longer than the schedule expects silently lose
        # their first hundred-odd points. The arm then LEAPS from home to the middle of the
        # next approach, and every relative timing the collision proof rests on is wrong.
        # Both symptoms look like a planning bug and are not, so refuse the pair outright.
        bad = []
        for task, a in self.sol["assignments"].items():
            key = (a["robot"], task)
            if key not in self.traj:
                raise ValueError(
                    f"the schedule assigns {task} to {a['robot']}, but the trajectory "
                    f"artifact has no such trajectory -- these are different plans"
                )
            n = len(self.traj[key]["positions"])
            want = a["end_slot"] - a["start_slot"]
            if n != want:
                bad.append(f"{a['robot']}/{task}: schedule says {want} slots, "
                           f"trajectories have {n}")
        # Second guard: whatever the plan says, the arm must be able to FOLLOW it.
        #
        # Every plan here is a sequence of configurations one control slot apart, so
        # consecutive commanded points differ by at most a slot's worth of joint motion.
        # A larger gap means the timeline was assembled wrongly -- tasks reordered relative
        # to the trajectories that were planned for them, an artifact spliced at the wrong
        # index -- and the controller would answer it by slewing across the gap at whatever
        # speed it can, through space nobody collision-checked. That is unsafe in a way no
        # per-slot collision matrix can see, because mu only ever looks AT the samples.
        # Checked here, at the last point before motion, so it holds for any future
        # producer of a plan and not just the ones that exist today.
        for robot in self.robots:
            tasks = sorted(
                (a["start_slot"], a["end_slot"], task)
                for task, a in self.sol["assignments"].items()
                if a["robot"] == robot
            )
            if not tasks:
                continue
            budget = 4.0 * max(
                float(self.traj[(robot, t)]["max_joint_step"]) for _, _, t in tasks
            )
            # The reference home comes from the ARTIFACT, never from the plan's own first
            # sample: anchoring to the plan would make the check self-referential and blind
            # to the one failure it most needs to catch -- a timeline that starts somewhere
            # other than home.
            home = self.art["homes"][robot]
            timeline = [("home", home)]
            for _s, _e, task in tasks:
                timeline += [(task, q) for q in self.traj[(robot, task)]["positions"]]
            timeline.append(("home", home))
            for (n0, q0), (n1, q1) in zip(timeline, timeline[1:]):
                gap = max(abs(a - b) for a, b in zip(q0, q1))
                if gap > budget:
                    bad.append(
                        f"{robot}: a {gap:.3f} rad jump between {n0} and {n1} "
                        f"(one slot allows about {budget / 4.0:.3f} rad)"
                    )
                    break

        if bad:
            raise ValueError(
                "schedule and trajectories describe different plans:\n  "
                + "\n  ".join(bad)
                + f"\n\ntraj_file      = {traj_file}"
                + f"\nsolution_file  = {solution_file}"
                + "\n\nA refined schedule must be replayed against the REFINED "
                  "trajectories (tamp_trajectories_refined.json), and a baseline schedule "
                  "against the baseline ones. Pass refined:=true to select both together."
            )

        # Scene visualizer + gripper commander (both optional), built here so a bad
        # YAML / geometry mismatch fails before we command any motion. Both read the
        # SAME tamp_task.yaml -- resolve it once.
        if (self.visualize or self.actuate_grippers) and not task_file:
            from ament_index_python.packages import get_package_share_directory

            task_file = os.path.join(
                get_package_share_directory("multi_robot_cell_tamp"),
                "config",
                "tamp_task.yaml",
            )

        self.viz = None
        if self.visualize:
            from scene_visualizer import SceneVisualizer

            self.viz = SceneVisualizer(self, task_file)

            # One geometry source: every object the artifact moves must be described
            # in the YAML, or the scene we render would diverge from what was planned.
            art_objs = {t["object"] for t in self.art["trajectories"]}
            missing = art_objs - set(self.viz.objects)
            if missing:
                raise ValueError(
                    f"objects referenced by the trajectory artifact are absent from "
                    f"{task_file}: {sorted(missing)} -- the artifact and the scene YAML "
                    f"are from different geometry"
                )
            self.get_logger().info(f"scene visualization on, geometry from {task_file}")

        self.gripper = None
        if self.actuate_grippers:
            from gripper_commander import GripperCommander

            self.gripper = GripperCommander(self, task_file)
            self.get_logger().info(f"gripper actuation on, bindings from {task_file}")

        # Everything driven by the shared-clock timer + finish loop.
        self.animators = [a for a in (self.viz, self.gripper) if a is not None]

    def build_trajectory(self, robot: str) -> JointTrajectory:
        """One joint trajectory for a robot covering its whole timeline.

        The robot's assigned tasks are laid out at their absolute slot times. Each
        task trajectory already starts and ends exactly at home, so:

        * a gap between two tasks needs no filler -- the controller interpolates
          from the previous task's final home to the next task's initial home,
          i.e. it holds home;
        * a task not starting at slot 0 gets a single home waypoint at t=0, so the
          arm holds home until the task begins rather than lurching from wherever
          the controller thinks it is.

        Back-to-back tasks share an instant (one ends at slot S, the next starts at
        slot S, both at home). That duplicate timestamp is dropped -- a
        non-monotonic time would make the controller reject the goal.
        """
        tasks = sorted(
            (
                (a["start_slot"], a["end_slot"], task)
                for task, a in self.sol["assignments"].items()
                if a["robot"] == robot
            ),
        )

        traj = JointTrajectory()
        if not tasks:
            return traj  # this robot is idle for the whole schedule

        traj.joint_names = list(self.traj[(robot, tasks[0][2])]["joint_names"])
        home = self.traj[(robot, tasks[0][2])]["positions"][0]

        points: list[JointTrajectoryPoint] = []
        last_slot = -1

        if tasks[0][0] > 0:
            pt = JointTrajectoryPoint()
            pt.positions = list(home)
            pt.time_from_start = _duration(0.0)
            points.append(pt)
            last_slot = 0

        for start_slot, _end_slot, task in tasks:
            positions = self.traj[(robot, task)]["positions"]
            for k, q in enumerate(positions):
                slot = start_slot + k
                if slot <= last_slot:
                    continue  # drop the shared home knot between back-to-back tasks
                pt = JointTrajectoryPoint()
                pt.positions = list(q)
                pt.time_from_start = _duration(slot * self.delta_t)
                points.append(pt)
                last_slot = slot

        traj.points = points
        self.get_logger().info(
            f"{robot}: {len(tasks)} task(s), {len(points)} waypoints, "
            f"ends at {points[-1].time_from_start.sec}."
            f"{points[-1].time_from_start.nanosec // 1_000_000:03d} s"
        )
        return traj

    def run(self) -> bool:
        clients = {}
        for robot in self.robots:
            name = f"/{robot}{self.suffix}/follow_joint_trajectory"
            c = ActionClient(self, FollowJointTrajectory, name)
            self.get_logger().info(f"waiting for {name} ...")
            if not c.wait_for_server(timeout_sec=10.0):
                self.get_logger().error(f"controller action {name} not available")
                return False
            clients[robot] = c

        # Gripper action servers up? Robots without one are dropped (arms still run).
        # Bringup default is CLOSED -- open now, once, before dispatch, so fingers
        # are clear for the first pick's approach.
        if self.gripper is not None:
            self.gripper.wait_for_servers()
            self.gripper.init_open()

        # Drop the static scene in now, a beat before dispatch. The controller action
        # servers answering does NOT imply move_group's PlanningSceneMonitor has
        # subscribed to /collision_object, so publish_static() waits for a subscriber
        # and republishes a few times to defeat the publish-before-subscriber race.
        if self.viz is not None:
            self.viz.publish_static()

        # One shared start instant for both robots. The whole point of the common
        # clock: slot 0 is the same wall-clock time for every controller. Scene
        # animation AND gripper actuation are scheduled against the SAME instant, so
        # an object attaches / a gripper closes exactly when the arm reaches the
        # corresponding sample.
        start = self.get_clock().now() + rclpy.duration.Duration(seconds=self.start_delay)
        stamp = start.to_msg()

        for anim in self.animators:
            anim.schedule_from(self.art, self.sol, start)
            anim.start()  # ~50 ms timer, serviced by the spins below

        makespan_s = self.sol["makespan_slots"] * self.delta_t
        self.get_logger().info(
            f"dispatching schedule: makespan {makespan_s:.2f} s, starting in "
            f"{self.start_delay:.1f} s"
        )

        futures = {}
        for robot in self.robots:
            traj = self.build_trajectory(robot)
            if not traj.points:
                self.get_logger().info(f"{robot}: idle, nothing to send")
                continue
            traj.header.stamp = stamp
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj
            futures[robot] = clients[robot].send_goal_async(goal)

        for robot, fut in futures.items():
            rclpy.spin_until_future_complete(self, fut)
            handle = fut.result()
            if not handle.accepted:
                self.get_logger().error(f"{robot}: goal REJECTED by controller")
                return False
            self.get_logger().info(f"{robot}: goal accepted")
            futures[robot] = handle.get_result_async()

        # Wait for both to finish and report each controller's error code.
        ok = True
        for robot, fut in futures.items():
            rclpy.spin_until_future_complete(self, fut)
            result = fut.result().result
            code = result.error_code
            if code == FollowJointTrajectory.Result.SUCCESSFUL:
                self.get_logger().info(f"{robot}: trajectory completed")
            else:
                self.get_logger().error(
                    f"{robot}: controller returned error {code} ({result.error_string})"
                )
                ok = False

        # Controllers are done, but the final place event may not have fired yet (or
        # its scene update may still be in flight). Keep the node spinning so the
        # animation timer completes, then a short settle so the last place is shown
        # before we tear down. Without this the node could shut down mid-animation.
        if self.animators:
            self._finish_animation()
        return ok

    def _finish_animation(self, settle: float = 1.0) -> None:
        """Spin until every animator's events have fired, then settle.

        Controllers can finish before the last place / gripper event (or its scene
        update is still in flight), so without this the node could tear down
        mid-animation and the final place or gripper-open would be lost.
        """
        # Guard against an event whose time is far past controller completion.
        deadline = None
        last = [a.last_event_time for a in self.animators if a.last_event_time is not None]
        if last:
            latest = max(last, key=lambda t: t.nanoseconds)
            deadline = latest + rclpy.duration.Duration(seconds=settle)

        while rclpy.ok() and any(a.pending() for a in self.animators):
            rclpy.spin_once(self, timeout_sec=0.05)
            if deadline is not None and self.get_clock().now() > deadline:
                self.get_logger().warn("animation: giving up on unfired events past deadline")
                break

        # Let the final scene update / gripper goal propagate before shutdown.
        settle_end = self.get_clock().now() + rclpy.duration.Duration(seconds=settle)
        while rclpy.ok() and self.get_clock().now() < settle_end:
            rclpy.spin_once(self, timeout_sec=0.05)


def main():
    rclpy.init()
    node = ScheduleExecutor()
    try:
        ok = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
