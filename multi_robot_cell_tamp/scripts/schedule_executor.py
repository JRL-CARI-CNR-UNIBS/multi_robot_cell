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

    ros2 launch multi_robot_cell_tamp execute_schedule.launch.py \\
        traj_file:=/tmp/tamp_trajectories.json \\
        solution_file:=/tmp/tamp_solution.json

Needs the cell up (``start.launch.py``), because it commands the live controllers.
"""

from __future__ import annotations

import json

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def _duration(seconds: float) -> Duration:
    d = Duration()
    d.sec = int(seconds)
    d.nanosec = int(round((seconds - d.sec) * 1e9))
    return d


class ScheduleExecutor(Node):
    def __init__(self):
        super().__init__("schedule_executor")
        self.declare_parameter("traj_file", "/tmp/tamp_trajectories.json")
        self.declare_parameter("solution_file", "/tmp/tamp_solution.json")
        self.declare_parameter("controller_suffix", "_linear_guide_joint_trajectory_controller")
        # Lead time before t=0, so both goals are accepted and queued before the
        # shared start instant arrives. Too short and one controller misses it.
        self.declare_parameter("start_delay", 2.0)

        traj_file = self.get_parameter("traj_file").value
        solution_file = self.get_parameter("solution_file").value
        self.suffix = self.get_parameter("controller_suffix").value
        self.start_delay = self.get_parameter("start_delay").value

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

        # One shared start instant for both robots. The whole point of the common
        # clock: slot 0 is the same wall-clock time for every controller.
        start = self.get_clock().now() + rclpy.duration.Duration(seconds=self.start_delay)
        stamp = start.to_msg()

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
        return ok


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
