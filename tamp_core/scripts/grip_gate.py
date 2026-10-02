"""Make an arm wait for its own gripper: the plan-graph executor's grip gate.

The plan holds each arm still for a grip dwell (``gripper_dwell_slots``) at every GripClose and
GripOpen, and the gripper is commanded at the start of it. Timing alone does not guarantee the
fingers are done when the dwell ends: a real gripper may be slower than planned, or close on
nothing. Under the plan graph (``tpg_executor.py``) each grip event therefore has a GATE, the last
node of its dwell: the arm is not dispatched past it until the event is confirmed.

Confirmed means: the action reported success, or the finger joint has stopped moving (a stall:
the fingers are holding the part, which a position-controlled gripper reports as a tolerance
failure, or never reports at all). Failed means: the fingers ended at or below the robot's
``gripper_min_closed`` on a CLOSE (nothing between them: a missed grasp; only checked when the
scene sets it, since a simulated gripper always closes fully), or no confirmation within
``gripper_timeout_s`` (scene, per robot; default 5 s). A failure stops the execution.

Mixed into ``gripper_commander.GripperCommander`` and
``trajectory_gripper_commander.TrajectoryGripperCommander``; each supplies ``_judge``.
"""
from __future__ import annotations

import time
from collections import deque

STALL_WINDOW_S = 0.25      # the fingers count as stopped after this long ...
STALL_EPS = 1e-3           # ... within this much travel (joint units)
DEFAULT_TIMEOUT_S = 5.0


def dwell_end(phases, k: int) -> int:
    """Last index of the run of identical phases starting at ``k`` (the end of a dwell)."""
    j = k
    while j + 1 < len(phases) and phases[j + 1] == phases[k]:
        j += 1
    return j


class GripGate:
    """Event bookkeeping shared by both commanders. Expects ``self._events`` (dicts with
    ``robot``, ``what``, ``task``, and, for a node-keyed run, ``gate``), ``self.node``,
    ``self.log``, ``self._gate_cfg`` (robot -> {"timeout_s", "min_closed", "joint"})."""

    def _gate_init(self, spec: dict) -> None:
        self._gate_cfg = {}
        for name, cfg in (spec.get("robots") or {}).items():
            self._gate_cfg[name] = {
                "timeout_s": float(cfg.get("gripper_timeout_s", DEFAULT_TIMEOUT_S)),
                "min_closed": (None if cfg.get("gripper_min_closed") is None
                               else float(cfg["gripper_min_closed"])),
                "joint": cfg.get("gripper_joint"),
            }
        self._finger: dict[str, deque] = {}      # joint -> deque[(t, position)]
        self._js_sub = None
        self.failures: list[str] = []

    def watch_fingers(self) -> None:
        """Track the finger joints on /joint_states (for the stall test). Idempotent."""
        if self._js_sub is not None:
            return
        from sensor_msgs.msg import JointState
        joints = {c["joint"] for c in self._gate_cfg.values() if c["joint"]}
        for j in joints:
            self._finger[j] = deque(maxlen=200)

        def cb(msg):
            now = time.monotonic()
            for n, p in zip(msg.name, msg.position):
                if n in self._finger:
                    self._finger[n].append((now, p))
        self._js_sub = self.node.create_subscription(JointState, "/joint_states", cb, 10)

    def finger(self, robot: str):
        """(position, stopped) of ``robot``'s finger joint, (None, False) if unknown."""
        j = self._gate_cfg.get(robot, {}).get("joint")
        hist = self._finger.get(j) if j else None
        if not hist:
            return None, False
        now, pos = time.monotonic(), hist[-1][1]
        recent = [p for t, p in hist if now - t <= STALL_WINDOW_S]
        covered = hist[0][0] <= now - STALL_WINDOW_S
        stopped = covered and len(recent) >= 2 and (max(recent) - min(recent)) <= STALL_EPS
        return pos, stopped

    # ---- the gate ------------------------------------------------------------ #

    def blocks(self, robot: str, node: int) -> bool:
        """True when ``robot`` may not be commanded to ``node``: some grip of it whose gate
        precedes ``node`` is not confirmed yet."""
        return any(ev["robot"] == robot and ev.get("gate") is not None and ev["gate"] < node
                   and ev.get("state") != "done" for ev in self._events)

    def waiting_on(self, robot: str, node: int):
        for ev in self._events:
            if (ev["robot"] == robot and ev.get("gate") is not None and ev["gate"] < node
                    and ev.get("state") != "done"):
                return ev
        return None

    def poll(self) -> None:
        """Advance every fired event: resolve its goal, judge it, time it out."""
        now = time.monotonic()
        for ev in self._events:
            if ev.get("state") not in ("sent",):
                continue
            fut = ev.get("goal")
            if ev.get("result") is None and fut is not None and fut.done():
                handle = fut.result()
                if not handle.accepted:
                    self._fail(ev, "goal rejected by the gripper controller")
                    continue
                ev["result"] = handle.get_result_async()
            res = ev["result"].result().result if (ev.get("result") is not None
                                                    and ev["result"].done()) else None
            verdict = self._judge(ev, res, now - ev["t_fired"])
            if verdict is True:
                ev["state"] = "done"
                ev["t_done"] = now
                self.log.info(f"gripper: {ev['what'].upper():5s} {ev['robot']}/{ev['task']} "
                              f"confirmed after {now - ev['t_fired']:.2f} s"
                              + (f" ({ev['how']})" if ev.get("how") else ""))
            elif isinstance(verdict, str):
                self._fail(ev, verdict)
            elif now - ev["t_fired"] > self._gate_cfg.get(ev["robot"], {}).get(
                    "timeout_s", DEFAULT_TIMEOUT_S):
                pos, _ = self.finger(ev["robot"])
                self._fail(ev, f"no confirmation within the timeout (finger at {pos})")

    def _check_missed(self, ev, pos):
        """A CLOSE that ends at or below ``gripper_min_closed`` held nothing."""
        mc = self._gate_cfg.get(ev["robot"], {}).get("min_closed")
        if ev["what"] == "close" and mc is not None and pos is not None and pos <= mc:
            return (f"missed grasp: fingers at {pos:.4f}, at or below gripper_min_closed {mc} "
                    f"(nothing between them)")
        return None

    def _fail(self, ev, why: str) -> None:
        ev["state"] = "failed"
        msg = f"{ev['robot']}/{ev['task']} {ev['what']}: {why}"
        self.failures.append(msg)
        self.log.error(f"gripper: {msg}")

    def gate_report(self) -> dict:
        """robot -> seconds between a grip being fired and confirmed, summed (diagnostics)."""
        out: dict[str, float] = {}
        for ev in self._events:
            if ev.get("t_done") is not None:
                out[ev["robot"]] = out.get(ev["robot"], 0.0) + ev["t_done"] - ev["t_fired"]
        return out
