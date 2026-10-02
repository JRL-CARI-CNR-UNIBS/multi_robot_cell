#!/usr/bin/env python3
"""Actuate grippers driven by a JointTrajectoryController, in sync with the replay.

The counterpart of ``gripper_commander.py`` (a ``GripperActionController`` / ``GripperCommand``,
the UR cells' Robotiq) for a gripper that is an ordinary
``joint_trajectory_controller/JointTrajectoryController`` -- TIAGo's PAL parallel gripper, see
``tiago_cell/vendor/pal_pro_gripper/pal_pro_gripper_controller_configuration/config/
gripper_controller.yaml`` -- so it sends a one-point ``FollowJointTrajectory`` goal instead.
Everything else (timing off the shared Delta-t clock, firing at the task's GripClose/GripOpen
dwell, fire-and-forget dispatch, node-keyed ``tick_nodes`` for the plan-graph executor) mirrors
``gripper_commander.py``. Selected per scene by ``cell_hardware.make_gripper_commander``: a
robot with ``gripper_joint`` uses this one.

Bindings (action name, joint name, open/close positions) come from the SAME scene YAML the
rest of the pipeline uses.
"""
from __future__ import annotations

import yaml
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Must mirror include/multi_robot_cell_tamp/resample.hpp::Phase.
PHASE_GRIP_CLOSE = 1
PHASE_GRIP_OPEN = 3


def _first(seq, value, start=0):
    for i in range(start, len(seq)):
        if seq[i] == value:
            return i
    return None


def _seconds_to_msg(seconds: float) -> DurationMsg:
    d = DurationMsg()
    d.sec = int(seconds)
    d.nanosec = int(round((seconds - d.sec) * 1e9))
    return d


class TrajectoryGripperCommander:
    """Sends gripper open/close FollowJointTrajectory goals timed to the schedule."""

    def __init__(self, node, task_yaml_path: str):
        self.node = node
        self.log = node.get_logger()

        with open(task_yaml_path) as f:
            spec = yaml.safe_load(f)

        # robot -> (action_name, joint_name, open_position, close_position)
        self.grippers: dict[str, tuple[str, str, float, float]] = {}
        for name, cfg in (spec.get("robots") or {}).items():
            action = cfg.get("gripper_action")
            joint = cfg.get("gripper_joint")
            if action is None or joint is None:
                continue
            self.grippers[name] = (
                action, joint, float(cfg["gripper_open"]), float(cfg["gripper_close"])
            )

        self._clients: dict[str, ActionClient] = {}
        self._events: list[dict] = []
        self._futures: list = []  # keep goal futures alive so they are not GC'd
        self._next = 0
        self._timer = None
        self.last_event_time = None
        # How long a goal is given to reach position -- the dwell IS the time
        # budget (set once schedule_from sees the artifact's discretisation).
        self._dwell_seconds = 1.0

    # ---- setup -------------------------------------------------------------- #

    def wait_for_servers(self, timeout: float = 5.0) -> None:
        available: dict[str, tuple[str, str, float, float]] = {}
        for robot, (action, joint, open_pos, close_pos) in self.grippers.items():
            client = ActionClient(self.node, FollowJointTrajectory, action)
            if client.wait_for_server(timeout_sec=timeout):
                self._clients[robot] = client
                available[robot] = (action, joint, open_pos, close_pos)
            else:
                self.log.warn(
                    f"gripper: action server '{action}' unavailable; "
                    f"{robot}'s gripper will not actuate"
                )
        self.grippers = available

    def init_open(self) -> None:
        """Open every gripper once, at startup (mirrors the UR commander's init_open)."""
        for robot, (_action, joint, open_pos, _close_pos) in self.grippers.items():
            self._send(robot, joint, open_pos)
            self.log.info(f"gripper: INIT  {robot} -> {open_pos:.3f} (open)")

    # ---- schedule ----------------------------------------------------------- #

    def schedule_from(self, artifact: dict, solution: dict, start_time,
                      node_base: dict | None = None) -> None:
        """Build the time-sorted close/open event list on the shared clock.

        Same convention as the UR commander / scene visualizer: an event at local
        sample ``k`` of a task starting at ``start_slot`` fires at
        ``start_time + (start_slot + k) * dt``.
        """
        delta_t = float(artifact["delta_t"])
        dwell_slots = int(artifact.get("gripper_dwell_slots", 40))
        self._dwell_seconds = dwell_slots * delta_t
        traj_by_key = {(t["robot"], t["task"]): t for t in artifact["trajectories"]}

        events: list[dict] = []
        for task_id, a in solution["assignments"].items():
            robot = a["robot"]
            if robot not in self.grippers:
                continue
            tr = traj_by_key.get((robot, task_id))
            if tr is None:
                continue

            phases = tr["phase"]
            start_slot = int(a["start_slot"])
            _action, joint, open_pos, close_pos = self.grippers[robot]

            close_k = _first(phases, PHASE_GRIP_CLOSE)
            open_k = _first(phases, PHASE_GRIP_OPEN, close_k or 0)
            base = None if node_base is None else node_base.get((robot, task_id))
            if close_k is not None:
                events.append({
                    "time": start_time + Duration(seconds=(start_slot + close_k) * delta_t),
                    "node": None if base is None else base + close_k,
                    "robot": robot, "joint": joint, "pos": close_pos,
                    "what": "close", "task": task_id,
                })
            if open_k is not None:
                events.append({
                    "time": start_time + Duration(seconds=(start_slot + open_k) * delta_t),
                    "node": None if base is None else base + open_k,
                    "robot": robot, "joint": joint, "pos": open_pos,
                    "what": "open", "task": task_id,
                })

        events.sort(key=lambda e: e["time"].nanoseconds)
        self._events = events
        self._next = 0
        self.last_event_time = events[-1]["time"] if events else start_time
        self.log.info(f"gripper: scheduled {len(events)} actuation event(s)")

    def start(self) -> None:
        if self._timer is None:
            self._timer = self.node.create_timer(0.05, self.tick)

    def tick(self) -> None:
        now = self.node.get_clock().now()
        while self._next < len(self._events) and self._events[self._next]["time"] <= now:
            self._fire(self._events[self._next])
            self._next += 1

    def _send(self, robot: str, joint: str, position: float) -> None:
        traj = JointTrajectory()
        traj.joint_names = [joint]
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start = _seconds_to_msg(self._dwell_seconds)
        traj.points = [point]
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        # Fire-and-forget: keep the future referenced so it isn't garbage collected
        # mid-flight, but don't block -- the frozen dwell covers the actuation.
        self._futures.append(self._clients[robot].send_goal_async(goal))

    def _fire(self, ev: dict) -> None:
        self._send(ev["robot"], ev["joint"], ev["pos"])
        self.log.info(
            f"gripper: {ev['what'].upper():5s} {ev['robot']}/{ev['task']} -> {ev['pos']:.3f}"
        )

    # ---- introspection for the executor ------------------------------------- #

    def pending(self) -> bool:
        if any("fired" in e for e in self._events):
            return any(not e.get("fired") for e in self._events)
        return self._next < len(self._events)

    def tick_nodes(self, reached: dict) -> None:
        for ev in self._events:
            if ev.get("fired") or ev.get("node") is None:
                continue
            if reached.get(ev["robot"], -1) >= ev["node"]:
                self._fire(ev)
                ev["fired"] = True
