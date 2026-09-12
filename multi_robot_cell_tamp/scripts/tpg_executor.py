#!/usr/bin/env python3
"""Execute a plan under the temporal plan graph, on the cell's real controllers.

The open item of ADR-0007. ``schedule_executor.py`` replays the plan RIGIDLY: one
``FollowJointTrajectory`` per robot, both stamped with the same t=0, so slot ``k`` of one
arm is the same instant as slot ``k`` of the other. That is what makes the solver's
anti-collision offsets mean something -- and it is also its weakness, because the guarantee
holds only while both controllers keep up. Any lag and the realised offset drifts away from
the one that was approved, silently.

This node replaces the shared clock with the graph. Robot ``r`` may enter node ``n`` only
once the other robot has REACHED node ``deps[r][n]``; a delay then costs time and nothing
else. Same rule as ``simulate_tpg.run_tpg``, against controllers instead of a delay trace.

DISPATCH IS BY CONTIGUOUS RUN, NOT BY NODE
------------------------------------------
A node is one control slot (25 ms), so commanding node by node would mean 40 goals per
second per arm -- not something to run on hardware. Instead each arm is sent the LONGEST
PREFIX of its remaining nodes that is already unblocked, as one trajectory, and the next
run is dispatched when that one completes. Blocking points are the only place the arm
stops.

That costs nothing when the graph is slack and everything when it is not, which is exactly
the difference this node exists to show. Measured on the ``swap`` scene:

    this work (parallel + makespan)     1 run per arm, 0 stops       -- the graph is idle
    APEX-MR   (turn_based + apex)      10 and 21 runs, 29 stops      -- the graph IS the plan

The cost of dispatching in runs rather than continuously is a brief stop at each blocking
point: an arm that is unblocked *while* its predecessor run is still executing does not
learn so until that run ends. The alternative -- pre-empting the active goal to extend it --
buys back a few tenths of a second and needs the controller to re-time from wherever the arm
currently is, through space nobody collision-checked. Not worth it: the executed makespan is
reported, so the cost is measured rather than assumed.

PROGRESS COMES FROM ``/joint_states``, NOT FROM THE CLOCK
--------------------------------------------------------
``reached[r]`` is derived by matching the measured configuration against the nodes of the
run in flight, searching forward only. Deriving it from elapsed time instead would make the
executor assume the arm is where it was told to be -- which is the very assumption the graph
exists to drop, and it would report a guarantee it never checked.

Time does appear, but only as an upper bound on the search: a controller never runs faster
than the trajectory it was handed, so nodes further ahead than the elapsed time allows are
not the arm however well they match. Without that bound a pick fires its gripper at the
approach pose, because the descent to the box and the retreat from it retrace the same
configurations and the nearest match jumps across the grasp -- see ``_on_joint_state``.

    ros2 launch multi_robot_cell_tamp execute_schedule.launch.py mode:=tpg

Needs the cell up (``multi_robot_cell_bringup start.launch.py``).
"""

from __future__ import annotations

import json
import os
import sys
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Siblings are installed next to this script in lib/<pkg>.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tpg import FREE, TPG  # noqa: E402

_ARTIFACTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "artifacts"
)

# Slack on the rate bound in _on_joint_state, in nodes (4 = 0.1 s of plan). Absorbs
# dispatch latency and /joint_states jitter without widening the window enough to let a
# retraced configuration back in -- the ambiguity it exists to exclude is ~150 nodes away.
_RATE_SLACK = 4


def _duration(seconds: float) -> Duration:
    d = Duration()
    d.sec = int(seconds)
    d.nanosec = int(round((seconds - d.sec) * 1e9))
    return d


class TpgExecutor(Node):
    def __init__(self):
        super().__init__("tpg_executor")
        self.declare_parameter("traj_file", os.path.join(_ARTIFACTS, "tamp_trajectories_refined.json"))
        self.declare_parameter("solution_file", os.path.join(_ARTIFACTS, "vamp", "tamp_solution_refined.json"))
        self.declare_parameter("tpg_file", os.path.join(_ARTIFACTS, "vamp", "tpg_refined.json"))
        self.declare_parameter("controller_suffix", "_linear_guide_joint_trajectory_controller")
        self.declare_parameter("task_file", "")
        self.declare_parameter("visualize", True)
        self.declare_parameter("actuate_grippers", True)
        # Label printed in the summary, so two runs can be told apart in a terminal.
        self.declare_parameter("label", "")

        traj_file = self.get_parameter("traj_file").value
        solution_file = self.get_parameter("solution_file").value
        tpg_file = self.get_parameter("tpg_file").value
        self.suffix = self.get_parameter("controller_suffix").value
        self.label = self.get_parameter("label").value
        task_file = self.get_parameter("task_file").value

        with open(traj_file) as f:
            self.art = json.load(f)
        with open(solution_file) as f:
            self.sol = json.load(f)
        self.graph = TPG.from_json(tpg_file)

        self.delta_t = float(self.art["delta_t"])
        self.traj = {(t["robot"], t["task"]): t for t in self.art["trajectories"]}
        self.robots = list(self.graph.robots)

        self._check_artifacts(traj_file, solution_file, tpg_file)

        # node table: robot -> [joint positions per node], laid out by the graph's segments
        # so index n here is the same n the graph's deps[] talk about.
        self.nodes: dict[str, list[list[float]]] = {}
        self.joint_names: dict[str, list[str]] = {}
        self.node_base: dict[tuple[str, str], int] = {}
        for r in self.robots:
            table: list[list[float]] = []
            for seg in self.graph.segments[r]:
                entry = self.traj[(r, seg.task)]
                self.joint_names[r] = list(entry["joint_names"])
                self.node_base[(r, seg.task)] = seg.start_node
                if len(table) != seg.start_node:
                    raise ValueError(
                        f"{r}: segment {seg.task} claims start_node {seg.start_node} but "
                        f"{len(table)} nodes precede it -- graph and trajectories disagree"
                    )
                table.extend(list(q) for q in entry["positions"])
            self.nodes[r] = table
            if len(table) != self.graph.n_nodes(r):
                raise ValueError(
                    f"{r}: {len(table)} trajectory samples against {self.graph.n_nodes(r)} "
                    f"graph nodes -- these artifacts are from different runs"
                )

        # Live state. reached[r] is the last node the ARM has been observed at (-1: none).
        self.reached = {r: -1 for r in self.robots}
        self.active: dict[str, dict] = {}     # r -> {"from","to","handle_future","result"}
        self.runs = {r: 0 for r in self.robots}
        self.blocked_since = {r: None for r in self.robots}
        self.blocked_total = {r: 0.0 for r in self.robots}
        self._last_state: dict[str, list[float]] = {}

        self._setup_animators(task_file)

        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)

    # ------------------------------------------------------------------ setup ----- #

    def _check_artifacts(self, traj_file, solution_file, tpg_file) -> None:
        """Refuse a mismatched triple before commanding anything.

        Same reasoning as the rigid executor, extended to the graph: refinement writes a
        second, shorter set of everything, and a graph built for one plan indexes nodes
        that do not exist in another. Caught here rather than as an IndexError halfway
        through a motion.
        """
        bad = []
        if abs(self.delta_t - float(self.sol["delta_t"])) > 1e-12:
            bad.append(f"delta_t {self.delta_t} vs schedule's {self.sol['delta_t']}")
        if abs(self.delta_t - float(self.graph.delta_t)) > 1e-12:
            bad.append(f"delta_t {self.delta_t} vs graph's {self.graph.delta_t}")
        for r in self.robots:
            for seg in self.graph.segments[r]:
                if (r, seg.task) not in self.traj:
                    bad.append(f"graph gives {seg.task} to {r}, trajectories have no such pair")
                elif len(self.traj[(r, seg.task)]["positions"]) != seg.n:
                    bad.append(f"{r}/{seg.task}: graph says {seg.n} samples, "
                               f"trajectories have {len(self.traj[(r, seg.task)]['positions'])}")
        if bad:
            raise ValueError(
                "graph, schedule and trajectories describe different plans:\n  "
                + "\n  ".join(bad)
                + f"\n\ntraj_file     = {traj_file}\nsolution_file = {solution_file}"
                + f"\ntpg_file      = {tpg_file}\n\nA refined plan needs the refined "
                  "trajectories AND tpg_refined.json; pass refined:= to select all three."
            )

    def _setup_animators(self, task_file: str) -> None:
        visualize = self.get_parameter("visualize").value
        actuate = self.get_parameter("actuate_grippers").value
        if (visualize or actuate) and not task_file:
            from ament_index_python.packages import get_package_share_directory

            task_file = os.path.join(
                get_package_share_directory("multi_robot_cell_tamp"), "config", "tamp_task.yaml"
            )
        self.viz = None
        if visualize:
            from scene_visualizer import SceneVisualizer

            self.viz = SceneVisualizer(self, task_file)
        self.gripper = None
        if actuate:
            from gripper_commander import GripperCommander

            self.gripper = GripperCommander(self, task_file)
        self.animators = [a for a in (self.viz, self.gripper) if a is not None]

    # -------------------------------------------------------------- progress ----- #

    def _on_joint_state(self, msg: JointState) -> None:
        """Advance ``reached`` to the furthest node of the run in flight that matches.

        Forward-only: a configuration is evidence of progress just once. Every task begins
        and ends at home (ADR-0004), so home alone recurs in each run and a free search
        would read a later occurrence of it as progress the arm has not made.

        Forward-only is NOT enough on its own, because a pick retraces its own approach.
        The arm descends to the box and comes back up the same way, so while it is on the
        way DOWN there is a node on the way UP holding almost exactly the same
        configuration -- measured on `swap`, node 123 (approach) sits 6e-5 rad from node
        280 (retreat), and a nearest-match over a wide window jumps 157 nodes forward,
        straight past the GripClose at node 182. The gripper then fires at the approach
        pose instead of at the box.

        The cure is a rate bound. Elapsed time cannot say where the arm IS -- that is the
        assumption the graph exists to drop -- but it bounds where the arm can be: a
        controller does not run faster than the trajectory it was given, so it may lag and
        never lead. Candidates past that bound are therefore not the arm, whatever they
        measure. Position still comes from the measurement; time only rules out the
        impossible.
        """
        index = {n: i for i, n in enumerate(msg.name)}
        now = time.monotonic()
        for r, run in list(self.active.items()):
            names = self.joint_names[r]
            if not all(n in index for n in names):
                continue
            q = [msg.position[index[n]] for n in names]
            self._last_state[r] = q
            table = self.nodes[r]
            lo = max(self.reached[r], run["from"] - 1) + 1
            # Node `from + k` is commanded at (k+1)*delta_t after dispatch, and `t0` is
            # taken before the goal is even accepted, so this errs on the generous side --
            # which is the right direction for an upper bound.
            furthest = run["from"] + int((now - run["t0"]) / self.delta_t) + _RATE_SLACK
            hi = min(run["to"], furthest)
            best, best_d = None, None
            for n in range(lo, hi + 1):
                d = max(abs(a - b) for a, b in zip(q, table[n]))
                if best_d is None or d < best_d:
                    best, best_d = n, d
            # A tolerance, not an exact match: the controller interpolates between the
            # samples it was given, so the arm is almost never exactly ON a node.
            if best is not None and best_d is not None and best_d <= self.tol(r):
                if best > self.reached[r]:
                    self.reached[r] = best

    def tol(self, robot: str) -> float:
        """Half a slot of motion: enough to claim the nearest node, not the next one."""
        steps = [float(self.traj[(robot, s.task)]["max_joint_step"])
                 for s in self.graph.segments[robot]]
        return 0.75 * max(steps) if steps else 0.05

    # -------------------------------------------------------------- dispatch ----- #

    def _ready_upto(self, robot: str) -> int:
        """Furthest node reachable right now: the longest already-unblocked prefix.

        Returns ``reached[robot]`` itself when the very next node is blocked, which is the
        signal to hold. Stopping at the FIRST blocked node (rather than skipping it) is the
        graph's semantics: nodes are entered in order.
        """
        dep = self.graph.deps[robot]
        other = self.graph.other(robot)
        n_max = self.graph.n_nodes(robot) - 1
        n = self.reached[robot]
        while n < n_max:
            d = int(dep[n + 1])
            if d != FREE and self.reached[other] < d:
                break
            n += 1
        return n

    def _send(self, robot: str, first: int, last: int) -> None:
        """Command nodes ``first..last`` as one trajectory, one slot apart.

        The first point sits at ``delta_t``, not at 0: the arm is standing at node
        ``first - 1`` (the end of the previous run, or home), so a point at t=0 would ask
        the controller to be somewhere else instantly. One slot is exactly the planned
        spacing, so the run continues at the speed it was planned for.
        """
        traj = JointTrajectory()
        traj.joint_names = list(self.joint_names[robot])
        traj.points = []
        for k, n in enumerate(range(first, last + 1)):
            pt = JointTrajectoryPoint()
            pt.positions = list(self.nodes[robot][n])
            pt.time_from_start = _duration((k + 1) * self.delta_t)
            traj.points.append(pt)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        self.runs[robot] += 1
        self.active[robot] = {
            "from": first,
            "to": last,
            "t0": time.monotonic(),      # anchors the rate bound in _on_joint_state
            "goal": self._traj_clients[robot].send_goal_async(goal),
            "result": None,
        }

    def _pump(self) -> bool:
        """One decision round: retire finished runs, dispatch what is unblocked.

        Returns False on deadlock -- both arms unfinished, neither moving, neither able to
        move. The graph is acyclic by construction (ADR-0007), so this should be
        unreachable; it is checked anyway, because hanging silently would be the worst way
        to find out otherwise.
        """
        for r in list(self.active):
            run = self.active[r]
            if run["result"] is None:
                if run["goal"].done():
                    handle = run["goal"].result()
                    if not handle.accepted:
                        self.get_logger().error(f"{r}: goal REJECTED by controller")
                        return False
                    run["result"] = handle.get_result_async()
                continue
            if run["result"].done():
                res = run["result"].result().result
                if res.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
                    self.get_logger().error(
                        f"{r}: controller error {res.error_code} ({res.error_string})")
                    return False
                # The controller says it is there; believe it for the run's last node so a
                # match tolerance never leaves an arm one node short of its own goal.
                self.reached[r] = max(self.reached[r], run["to"])
                del self.active[r]

        now = time.monotonic()
        for r in self.robots:
            if r in self.active:
                continue
            if self.reached[r] >= self.graph.n_nodes(r) - 1:
                continue
            upto = self._ready_upto(r)
            if upto <= self.reached[r]:
                if self.blocked_since[r] is None:
                    self.blocked_since[r] = now
                    dep = int(self.graph.deps[r][self.reached[r] + 1])
                    self.get_logger().info(
                        f"{r}: held at node {self.reached[r] + 1} -- waits for "
                        f"{self.graph.other(r)} to reach {dep} (now at "
                        f"{self.reached[self.graph.other(r)]})")
                continue
            if self.blocked_since[r] is not None:
                self.blocked_total[r] += now - self.blocked_since[r]
                self.blocked_since[r] = None
            self._send(r, self.reached[r] + 1, upto)

        if not self.active and not self._done():
            return False
        return True

    def _done(self) -> bool:
        return all(self.reached[r] >= self.graph.n_nodes(r) - 1 for r in self.robots)

    # ------------------------------------------------------------------- run ----- #

    def run(self) -> bool:
        self._traj_clients = {}
        for robot in self.robots:
            name = f"/{robot}{self.suffix}/follow_joint_trajectory"
            c = ActionClient(self, FollowJointTrajectory, name)
            self.get_logger().info(f"waiting for {name} ...")
            if not c.wait_for_server(timeout_sec=10.0):
                self.get_logger().error(f"controller action {name} not available")
                return False
            self._traj_clients[robot] = c

        if self.gripper is not None:
            self.gripper.wait_for_servers()
            self.gripper.init_open()
        if self.viz is not None:
            self.viz.publish_static()

        # Events are built exactly as for rigid replay and then re-keyed to node arrival:
        # under the graph the schedule's slot times are not when things happen.
        start = self.get_clock().now()
        for anim in self.animators:
            anim.schedule_from(self.art, self.sol, start, node_base=self.node_base)

        nominal = self.sol["makespan_slots"] * self.delta_t
        self.get_logger().info(
            f"TPG dispatch{' [' + self.label + ']' if self.label else ''}: "
            f"{sum(int((self.graph.deps[r] != FREE).sum()) for r in self.robots)} type-2 "
            f"edges, nominal makespan of the schedule {nominal:.2f} s")

        t0 = time.monotonic()
        ok = True
        while rclpy.ok() and not self._done():
            if not self._pump():
                self.get_logger().error(
                    "DEADLOCK: both arms unfinished and neither can advance. The graph is "
                    "acyclic by construction, so this means the artifacts do not match.")
                ok = False
                break
            for anim in self.animators:
                anim.tick_nodes(self.reached)
            rclpy.spin_once(self, timeout_sec=0.02)
        elapsed = time.monotonic() - t0

        # Close out any wait still open when the last arm finished.
        now = time.monotonic()
        for r in self.robots:
            if self.blocked_since[r] is not None:
                self.blocked_total[r] += now - self.blocked_since[r]
                self.blocked_since[r] = None

        for anim in self.animators:
            anim.tick_nodes(self.reached)
        self._settle()
        self._report(elapsed, nominal)
        return ok

    def _settle(self, seconds: float = 1.0) -> None:
        """Let the last place / gripper goal propagate before tearing the node down."""
        end = self.get_clock().now() + rclpy.duration.Duration(seconds=seconds)
        while rclpy.ok() and self.get_clock().now() < end:
            for anim in self.animators:
                anim.tick_nodes(self.reached)
            rclpy.spin_once(self, timeout_sec=0.05)

    def _report(self, elapsed: float, nominal: float) -> None:
        tag = f" [{self.label}]" if self.label else ""
        lines = [
            "",
            f"=== TPG execution{tag} =======================================",
            f"  policy in the schedule      {self.sol.get('policy', '?')} / "
            f"{self.sol.get('objective', '?')}",
            f"  nominal makespan (schedule) {nominal:8.2f} s",
            f"  EXECUTED makespan           {elapsed:8.2f} s",
        ]
        for r in self.robots:
            lines.append(
                f"  {r}: {self.runs[r]:3d} run(s) dispatched, "
                f"{self.blocked_total[r]:6.2f} s held by the graph, "
                f"{self.reached[r] + 1}/{self.graph.n_nodes(r)} nodes")
        lines.append("=" * 62)
        self.get_logger().info("\n".join(lines))


def main():
    rclpy.init()
    node = TpgExecutor()
    try:
        ok = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
