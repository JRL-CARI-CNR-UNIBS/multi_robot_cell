#!/usr/bin/env python3
"""Actuate the grippers in sync with the schedule replay.

Companion to ``schedule_executor.py`` / ``scene_visualizer.py``. The executor
drives the arms and the visualizer drives the world objects; this drives the
*grippers*. On the shared Delta-t clock it sends a ``GripperCommand`` **close** at
each task's ``GripClose`` dwell and an **open** at its ``GripOpen`` dwell -- the
same two instants at which the visualizer attaches and releases the object -- so
the fingers actually close on the box and open to let it go. Each dwell is a
frozen window (``gripper_dwell_slots``), which is exactly the time budget the
actuation needs, so the command is sent at the *start* of the dwell.

Gripper bindings (action name, open/close positions) come from the SAME
``config/tamp_task.yaml`` the rest of the pipeline uses -- one geometry/task
source across planning, collision, execution, visualization, and actuation.

Goals are fire-and-forget: we do not block on the result (the dwell covers the
motion), and the arms keep running even if a gripper server is missing.
"""

from __future__ import annotations

import yaml
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from rclpy.duration import Duration

# Must mirror include/multi_robot_cell_tamp/resample.hpp::Phase.
PHASE_GRIP_CLOSE = 1
PHASE_GRIP_OPEN = 3

# Matches multi_robot_cell_scene/pick_place.cpp's commandGripper().
MAX_EFFORT = 100.0


def _first(seq, value, start=0):
    for i in range(start, len(seq)):
        if seq[i] == value:
            return i
    return None


class GripperCommander:
    """Sends gripper open/close goals timed to the schedule, on the executor's node."""

    def __init__(self, node, task_yaml_path: str):
        self.node = node
        self.log = node.get_logger()

        with open(task_yaml_path) as f:
            spec = yaml.safe_load(f)

        # robot -> (action_name, open_position, close_position)
        self.grippers: dict[str, tuple[str, float, float]] = {}
        for name, cfg in (spec.get("robots") or {}).items():
            action = cfg.get("gripper_action")
            if action is None:
                continue
            self.grippers[name] = (
                action, float(cfg["gripper_open"]), float(cfg["gripper_close"])
            )

        self._clients: dict[str, ActionClient] = {}
        self._events: list[dict] = []
        self._futures: list = []  # keep goal futures alive so they are not GC'd
        self._next = 0
        self._timer = None
        self.last_event_time = None

    # ---- setup -------------------------------------------------------------- #

    def wait_for_servers(self, timeout: float = 5.0) -> None:
        """Create and wait for each robot's gripper action server.

        A robot whose server never appears is dropped (with a warning) rather than
        aborting the run: the arms still play, the fingers just won't move.
        """
        available: dict[str, tuple[str, float, float]] = {}
        for robot, (action, open_pos, close_pos) in self.grippers.items():
            client = ActionClient(self.node, GripperCommand, action)
            if client.wait_for_server(timeout_sec=timeout):
                self._clients[robot] = client
                available[robot] = (action, open_pos, close_pos)
            else:
                self.log.warn(
                    f"gripper: action server '{action}' unavailable; "
                    f"{robot}'s gripper will not actuate"
                )
        self.grippers = available

    def init_open(self) -> None:
        """Open every gripper once, at startup.

        Bringup's default gripper state is fully CLOSED. First task's approach
        assumes an open gripper (phase ToPick, hands empty) -- closed fingers foul
        the object on the very first pick. Fire this once, before dispatch, so
        fingers are open by the time the arm arrives (start_delay is the buffer).
        """
        for robot, (_action, open_pos, _close_pos) in self.grippers.items():
            goal = GripperCommand.Goal()
            goal.command.position = open_pos
            goal.command.max_effort = MAX_EFFORT
            self._futures.append(self._clients[robot].send_goal_async(goal))
            self.log.info(f"gripper: INIT  {robot} -> {open_pos:.3f} (open)")

    # ---- schedule ----------------------------------------------------------- #

    def schedule_from(self, artifact: dict, solution: dict, start_time,
                      node_base: dict | None = None) -> None:
        """Build the time-sorted close/open event list on the shared clock.

        Same convention as the visualizer: an event at local sample ``k`` of a task
        starting at ``start_slot`` fires at ``start_time + (start_slot + k) * dt``.
        ``node_base`` switches the trigger to TPG node arrival -- see the visualizer.
        """
        delta_t = float(artifact["delta_t"])
        traj_by_key = {(t["robot"], t["task"]): t for t in artifact["trajectories"]}

        events: list[dict] = []
        for task_id, a in solution["assignments"].items():
            robot = a["robot"]
            if robot not in self.grippers:
                continue  # no server / not configured
            tr = traj_by_key.get((robot, task_id))
            if tr is None:
                continue

            phases = tr["phase"]
            start_slot = int(a["start_slot"])
            _action, open_pos, close_pos = self.grippers[robot]

            close_k = _first(phases, PHASE_GRIP_CLOSE)
            open_k = _first(phases, PHASE_GRIP_OPEN, close_k or 0)
            base = None if node_base is None else node_base.get((robot, task_id))
            if close_k is not None:
                events.append({
                    "time": start_time + Duration(seconds=(start_slot + close_k) * delta_t),
                    "node": None if base is None else base + close_k,
                    "robot": robot, "pos": close_pos, "what": "close", "task": task_id,
                })
            if open_k is not None:
                events.append({
                    "time": start_time + Duration(seconds=(start_slot + open_k) * delta_t),
                    "node": None if base is None else base + open_k,
                    "robot": robot, "pos": open_pos, "what": "open", "task": task_id,
                })

        events.sort(key=lambda e: e["time"].nanoseconds)
        self._events = events
        self._next = 0
        self.last_event_time = events[-1]["time"] if events else start_time
        self.log.info(f"gripper: scheduled {len(events)} actuation event(s)")

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
        goal = GripperCommand.Goal()
        goal.command.position = ev["pos"]
        goal.command.max_effort = MAX_EFFORT
        # Fire-and-forget: keep the future referenced so it isn't garbage collected
        # mid-flight, but don't block -- the frozen dwell covers the actuation.
        self._futures.append(self._clients[ev["robot"]].send_goal_async(goal))
        self.log.info(
            f"gripper: {ev['what'].upper():5s} {ev['robot']}/{ev['task']} -> {ev['pos']:.3f}"
        )

    # ---- introspection for the executor ------------------------------------- #

    def pending(self) -> bool:
        """True while actuation events remain unfired."""
        if any("fired" in e for e in self._events):     # node-keyed run
            return any(not e.get("fired") for e in self._events)
        return self._next < len(self._events)

    def tick_nodes(self, reached: dict) -> None:
        """Node-keyed twin of ``tick``. See ``SceneVisualizer.tick_nodes``."""
        for ev in self._events:
            if ev.get("fired") or ev.get("node") is None:
                continue
            if reached.get(ev["robot"], -1) >= ev["node"]:
                self._fire(ev)
                ev["fired"] = True
