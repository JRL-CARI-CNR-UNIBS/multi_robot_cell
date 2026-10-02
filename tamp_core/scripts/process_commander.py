#!/usr/bin/env python3
"""Emulate the process (weld) interlock in sync with plan execution.

Third animator next to ``scene_visualizer`` (the world objects) and ``gripper_commander``
(the fingers). The cell carries grippers, not torches, so the process is EMULATED: the
kinematics, trajectories, collisions and timing of a weld are real; the arc is not. What
this module drives is the honest analogue of what a PLC would: the weld-enable signal.

Events come from the trajectory artifact's per-sample ``phase[]``, exactly like the gripper
events: ARC ON at the first sample of each ``ProcessOn`` run (the start of an arc-strike
dwell) and ARC OFF at the first sample of the ``ProcessOff`` run that follows it (the start
of the crater-fill dwell). A weld of one pass has one of each, as always; a tack of several
spots (``legs:``, fabricator v2) strikes and cuts once per spot, the arc OFF in between.
Rigid replay keys them to the shared clock; graph execution (``node_base``) to the node the
arm has REACHED, via ``tick_nodes`` -- same convention, same reasons, as the visualizer.

On each event, in order of value:

* ``/<robot>/process_active`` (``std_msgs/Bool``, depth 1, TRANSIENT_LOCAL) -- latched, so
  ``ros2 topic echo`` started at any moment shows the current interlock state;
* ``/weld_seams`` (``visualization_msgs/MarkerArray``, frame ``base_frame``) -- one
  ``LINE_STRIP`` per seam, grey while pending and orange once welded, plus a ``SPHERE`` at
  the arc while it is on, advanced along the seam with the traverse. Every message carries
  the whole state, so the latched last one is enough for a late-joining RViz;
* a log line: ``weld: ARC OFF robot2/w_b1_t (25.0 mm @ 50 mm/s, 0.50 s)`` -- the duration
  is the planned traverse (``Processing`` samples x Delta-t), next to the length and speed
  it should equal.

A SPOT weld is a ``welds:`` entry whose seam is (nearly) a point -- ``start`` and ``end`` a
couple of millimetres apart -- and whose process is the two dwells around a near-zero traverse.
A line strip of that length is degenerate (invisible in RViz), so a spot is drawn instead as a
fixed-size ``SPHERE`` at ``start``: grey while pending, orange once welded. The arc sphere at
the point still lights during the process, and no arc fraction is ever divided by a zero
traverse. A spot whose trajectory has no ``Processing`` samples at all (the traverse shorter
than a sample) is still emulated, ARC ON at the strike and ARC OFF at the cut; for a real seam
that same shape is still the warning it always was.

Seam geometry comes from the ``welds:`` block of the SAME task YAML the generators read --
one geometry source, the rule the other two animators follow. A weld trajectory never
produces a gripper command: its phases hold no GripClose/GripOpen.
"""

from __future__ import annotations

import math

import yaml
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

# Must mirror include/multi_robot_cell_tamp/resample.hpp::Phase. Values are frozen.
PHASE_PROCESS_ON = 5
PHASE_PROCESSING = 6
PHASE_PROCESS_OFF = 7

# Matches the generator's default (trajectory_generator.cpp, planning.process_speed).
DEFAULT_PROCESS_SPEED = 0.05

_SEAM_NS = "weld_seams"
_ARC_NS = "weld_arc"
_SEAM_WIDTH = 0.006       # m, line width in RViz
_ARC_DIAMETER = 0.025     # m
# A seam shorter than this is a spot weld: drawn as a fixed-size sphere, not a line strip.
_SPOT_MAX_LENGTH = 0.005  # m
_SPOT_DIAMETER = 0.012    # m
_PENDING = ColorRGBA(r=0.55, g=0.55, b=0.55, a=1.0)
_DONE = ColorRGBA(r=1.0, g=0.45, b=0.0, a=1.0)
_ARC = ColorRGBA(r=1.0, g=0.95, b=0.6, a=1.0)


def _first(seq, value, start=0):
    for i in range(start, len(seq)):
        if seq[i] == value:
            return i
    return None


def _xyz(d: dict) -> tuple[float, float, float]:
    return (float(d.get("x", 0.0)), float(d.get("y", 0.0)), float(d.get("z", 0.0)))


def _seam_legs(w: dict) -> list[list[tuple[float, float, float]]]:
    """A ``welds:`` entry's passes: one polyline per leg of ``legs:``, else the one seam."""
    if w.get("legs"):
        return [[_xyz(p) for p in leg] for leg in w["legs"]]
    return [_seam_points(w)]


def _seam_points(w: dict) -> list[tuple[float, float, float]]:
    """A ``welds:`` entry's waypoints: its ``path:`` polyline, or ``start``/``end``. A tack
    of several spots (``legs:``) is flattened, spot after spot (per-spot arc events: open)."""
    if w.get("legs"):
        return [_xyz(p) for leg in w["legs"] for p in leg]
    if w.get("path"):
        return [_xyz(p) for p in w["path"]]
    return [_xyz(w["start"]), _xyz(w["end"])]


def _along(points: list, frac: float) -> tuple[float, float, float]:
    """The point at ``frac`` of a polyline's arc length (the tool runs it at one speed)."""
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    left = max(0.0, min(1.0, frac)) * sum(lengths)
    for (a, b), seg in zip(zip(points, points[1:]), lengths):
        if left <= seg and seg > 0.0:
            u = left / seg
            return tuple(p + u * (q - p) for p, q in zip(a, b))
        left -= seg
    return points[-1]


class ProcessCommander:
    """Drives the emulated weld interlock and its RViz rendering, on the executor's node."""

    def __init__(self, node, task_yaml_path: str):
        self.node = node
        self.log = node.get_logger()

        with open(task_yaml_path) as f:
            spec = yaml.safe_load(f)

        self.base_frame = spec.get("base_frame", "world")
        default_speed = float(
            (spec.get("planning") or {}).get("process_speed", DEFAULT_PROCESS_SPEED)
        )

        # seam id -> {index, points, start, end, length, speed}; `points` is the polyline
        # (`path:`), two points for a `start`/`end` seam, and `length` its arc length.
        self.seams: dict[str, dict] = {}
        for i, w in enumerate(spec.get("welds") or []):
            points = _seam_points(w)
            legs = _seam_legs(w)
            start, end = points[0], points[-1]
            length = sum(math.dist(a, b) for leg in legs for a, b in zip(leg, leg[1:]))
            self.seams[w["id"]] = {
                "index": i,
                "points": points,
                "legs": legs,
                "start": start,
                "end": end,
                "length": length,
                "spot": length < _SPOT_MAX_LENGTH,
                "speed": float(w.get("speed", default_speed)),
            }

        # Latched: the interlock state is a level, not an edge, so whoever subscribes late
        # must still learn it. Depth 1 -- only the current level means anything.
        self._qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._active_pubs: dict = {}
        self._arc_ids: dict[str, int] = {}
        for robot in (spec.get("robots") or {}):
            self._robot(robot)
        self._marker_pub = node.create_publisher(MarkerArray, "/weld_seams", self._qos)

        self._events: list[dict] = []
        self._next = 0
        self._timer = None
        self._done: set[str] = set()
        self._arcs: dict[str, dict] = {}   # robot -> {"ev", "frac"} while its arc is on
        self.last_event_time = None

    def _robot(self, robot: str):
        """The robot's interlock publisher, created on first use."""
        if robot not in self._active_pubs:
            self._active_pubs[robot] = self.node.create_publisher(
                Bool, f"/{robot}/process_active", self._qos
            )
            self._arc_ids[robot] = len(self._arc_ids)
        return self._active_pubs[robot]

    # ---- schedule ----------------------------------------------------------- #

    def schedule_from(self, artifact: dict, solution: dict, start_time,
                      node_base: dict | None = None) -> None:
        """Build the time-sorted ARC ON / ARC OFF event list and publish the initial state.

        Same convention as the gripper: an event at local sample ``k`` of a task starting
        at ``start_slot`` fires at ``start_time + (start_slot + k) * dt``; with ``node_base``
        it also carries ``node = node_base[(robot, task)] + k`` for ``tick_nodes``.

        Raises if a weld trajectory has no seam in the YAML: the artifact and the scene
        would then be different geometry. Both executors call this before sending any
        trajectory goal, so nothing has moved yet.
        """
        delta_t = float(artifact["delta_t"])
        traj_by_key = {(t["robot"], t["task"]): t for t in artifact["trajectories"]}

        events: list[dict] = []
        missing = []
        for task_id, a in solution["assignments"].items():
            robot = a["robot"]
            tr = traj_by_key.get((robot, task_id))
            if tr is None:
                continue
            phases = tr["phase"]
            if _first(phases, PHASE_PROCESS_ON) is None:
                continue  # a pick-and-place task: nothing to emulate
            if task_id not in self.seams:
                missing.append(task_id)
                continue
            # One ON/OFF pair per pass: each ProcessOn run and the ProcessOff that follows it.
            passes = []
            k = 0
            while True:
                on_k = _first(phases, PHASE_PROCESS_ON, k)
                if on_k is None:
                    break
                off_k = _first(phases, PHASE_PROCESS_OFF, on_k)
                proc_k = _first(phases, PHASE_PROCESSING, on_k)
                if proc_k is not None and off_k is not None and proc_k > off_k:
                    proc_k = None
                if (proc_k is None and off_k is not None
                        and self.seams.get(task_id, {}).get("spot")):
                    proc_k = off_k    # spot: no traverse samples, the dwells are the whole process
                if proc_k is None or off_k is None:
                    self.log.warn(
                        f"weld: {robot}/{task_id} has ProcessOn but no Processing->ProcessOff "
                        f"(processing={proc_k}, off={off_k}); not emulated"
                    )
                    passes = []
                    break
                passes.append((on_k, proc_k, off_k))
                k = off_k
                while k < len(phases) and phases[k] == PHASE_PROCESS_OFF:
                    k += 1
            if not passes:
                continue
            self._robot(robot)

            start_slot = int(a["start_slot"])
            base = None if node_base is None else node_base.get((robot, task_id))
            n_legs = len(self.seams[task_id]["legs"])
            for leg, (on_k, proc_k, off_k) in enumerate(passes):
                common = {
                    "robot": robot, "task": task_id,
                    "proc_k": proc_k, "off_k": off_k, "base": base,
                    # which pass of the seam (the leg its arc marker runs along), and whether
                    # it is the last one (the seam is done at ITS arc-out)
                    "leg": min(leg, n_legs - 1), "n_passes": len(passes),
                    "last": leg == len(passes) - 1,
                    # Wall-clock start of the traverse, for the rigid-replay arc marker.
                    "t_proc": start_time + Duration(seconds=(start_slot + proc_k) * delta_t),
                    "traverse_s": (off_k - proc_k) * delta_t,
                }
                for what, kk in (("on", on_k), ("off", off_k)):
                    events.append(dict(
                        common, what=what,
                        time=start_time + Duration(seconds=(start_slot + kk) * delta_t),
                        node=None if base is None else base + kk,
                    ))

        if missing:
            raise ValueError(
                f"weld trajectories reference seams absent from the task YAML's welds: "
                f"{sorted(missing)} -- the artifact and the scene YAML are from different "
                f"geometry"
            )

        # Stable sort: an ON and OFF of one task never share a time, but keep ON first.
        events.sort(key=lambda e: e["time"].nanoseconds)
        self._events = events
        self._next = 0
        self._done = set()
        self._arcs = {}
        self.last_event_time = events[-1]["time"] if events else start_time

        # Initial state: every interlock off, every seam pending, anything a previous run
        # left in RViz cleared.
        for pub in self._active_pubs.values():
            pub.publish(Bool(data=False))
        self._publish_markers(clear=True)
        self.log.info(
            f"weld: scheduled {len(events)} process event(s) over "
            f"{len({e['task'] for e in events})} weld task(s) ({len(events) // 2} arc(s)); "
            f"{len(self.seams)} seam(s) in the YAML"
        )

    def start(self) -> None:
        """Create the ~50 ms timer that fires due events. Idempotent."""
        if self._timer is None:
            self._timer = self.node.create_timer(0.05, self.tick)

    def tick(self) -> None:
        now = self.node.get_clock().now()
        while self._next < len(self._events) and self._events[self._next]["time"] <= now:
            self._fire(self._events[self._next])
            self._next += 1
        # Rigid replay is time-keyed by design, so the arc marker follows the clock here.
        self._advance_arcs({
            r: (now - arc["ev"]["t_proc"]).nanoseconds * 1e-9 / arc["ev"]["traverse_s"]
            for r, arc in self._arcs.items() if arc["ev"]["traverse_s"] > 0
        })

    def _fire(self, ev: dict) -> None:
        robot, task = ev["robot"], ev["task"]
        seam = self.seams[task]
        n = ev.get("n_passes", 1)
        which = f" pass {ev.get('leg', 0) + 1}/{n}" if n > 1 else ""
        if ev["what"] == "on":
            self._active_pubs[robot].publish(Bool(data=True))
            self._arcs[robot] = {"ev": ev, "frac": 0.0}
            self._publish_markers()
            self.log.info(f"weld: ARC ON  {robot}/{task}{which}")
        else:
            self._active_pubs[robot].publish(Bool(data=False))
            self._arcs.pop(robot, None)
            if ev.get("last", True):
                self._done.add(task)
            self._publish_markers()
            leg = seam["legs"][ev.get("leg", 0)] if n > 1 else None
            length = (sum(math.dist(a, b) for a, b in zip(leg, leg[1:])) if leg
                      else seam["length"])
            self.log.info(
                f"weld: ARC OFF {robot}/{task}{which} ({length * 1e3:.1f} mm @ "
                f"{seam['speed'] * 1e3:.0f} mm/s, {ev['traverse_s']:.2f} s)"
                + (" [spot]" if seam["spot"] else "")
            )

    def release(self, timeout: float = 1.0) -> None:
        """Drop every interlock to False and the arc markers with it. Idempotent.

        The executors call this on EVERY exit -- normal, exception, SIGINT/SIGTERM -- while
        the node can still publish. Without it an executor stopped mid-weld leaves
        ``process_active`` True in every live subscriber: the last level they received is
        the arc on, and nothing would ever say otherwise. Seams keep their state (an
        interrupted seam stays grey: it was not completed). Waits up to ``timeout`` per
        publisher for the reliable delivery to be acknowledged, since the node is about to
        be destroyed. (SIGKILL cannot be caught; nothing in-process can cover it.)
        """
        interrupted = sorted(f"{r}/{a['ev']['task']}" for r, a in self._arcs.items())
        self._arcs = {}
        for pub in self._active_pubs.values():
            pub.publish(Bool(data=False))
        self._publish_markers()
        for pub in (*self._active_pubs.values(), self._marker_pub):
            pub.wait_for_all_acked(Duration(seconds=timeout))
        if interrupted:
            self.log.warn(f"weld: ARC FORCED OFF {', '.join(interrupted)} -- the executor "
                          f"stopped mid-weld; interlock released")

    def _advance_arcs(self, fracs: dict) -> None:
        """Move each active arc marker to ``frac`` of its seam; republish only on change."""
        changed = False
        for robot, f in fracs.items():
            f = min(1.0, max(0.0, f))
            if abs(f - self._arcs[robot]["frac"]) > 1e-3:
                self._arcs[robot]["frac"] = f
                changed = True
        if changed:
            self._publish_markers()

    # ---- markers ------------------------------------------------------------ #

    def _marker(self, ns: str, mid: int, mtype: int) -> Marker:
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    def _publish_markers(self, clear: bool = False) -> None:
        """Publish the WHOLE weld state, so the latched last message is self-sufficient."""
        arr = MarkerArray()
        if clear:
            wipe = Marker()
            wipe.header.frame_id = self.base_frame
            wipe.action = Marker.DELETEALL
            arr.markers.append(wipe)
        for sid, seam in self.seams.items():
            if seam["spot"]:
                # start ~ end: a line strip would be degenerate, so a point-sized sphere.
                m = self._marker(_SEAM_NS, seam["index"], Marker.SPHERE)
                m.pose.position.x, m.pose.position.y, m.pose.position.z = seam["start"]
                m.scale.x = m.scale.y = m.scale.z = _SPOT_DIAMETER
            elif len(seam["legs"]) > 1:
                # several passes (a tack of spots): one segment pair per piece of each leg,
                # never a line across the gap the torch jumps with the arc off
                m = self._marker(_SEAM_NS, seam["index"], Marker.LINE_LIST)
                m.scale.x = _SEAM_WIDTH
                m.points = [Point(x=p[0], y=p[1], z=p[2]) for leg in seam["legs"]
                            for a, b in zip(leg, leg[1:]) for p in (a, b)]
            else:
                m = self._marker(_SEAM_NS, seam["index"], Marker.LINE_STRIP)
                m.scale.x = _SEAM_WIDTH
                m.points = [Point(x=p[0], y=p[1], z=p[2]) for p in seam["points"]]
            m.color = _DONE if sid in self._done else _PENDING
            arr.markers.append(m)
        for robot, aid in self._arc_ids.items():
            m = self._marker(_ARC_NS, aid, Marker.SPHERE)
            arc = self._arcs.get(robot)
            if arc is None:
                if clear:
                    continue  # DELETEALL already removed it
                m.action = Marker.DELETE
            else:
                seam = self.seams[arc["ev"]["task"]]
                f = arc["frac"]
                pts = (seam["legs"][arc["ev"].get("leg", 0)] if len(seam["legs"]) > 1
                       else seam["points"])
                m.pose.position.x, m.pose.position.y, m.pose.position.z = _along(pts, f)
                m.scale.x = m.scale.y = m.scale.z = _ARC_DIAMETER
                m.color = _ARC
            arr.markers.append(m)
        self._marker_pub.publish(arr)

    # ---- introspection for the executor ------------------------------------- #

    def pending(self) -> bool:
        """True while process events remain unfired."""
        if any("fired" in e for e in self._events):     # node-keyed run
            return any(not e.get("fired") for e in self._events)
        return self._next < len(self._events)

    def tick_nodes(self, reached: dict) -> None:
        """Node-keyed twin of ``tick``. See ``SceneVisualizer.tick_nodes``.

        The arc marker follows ``reached`` too -- measured progress along the traverse,
        never elapsed time.
        """
        for ev in self._events:
            if ev.get("fired") or ev.get("node") is None:
                continue
            if reached.get(ev["robot"], -1) >= ev["node"]:
                self._fire(ev)
                ev["fired"] = True
        fracs = {}
        for robot, arc in self._arcs.items():
            ev = arc["ev"]
            if ev["base"] is None or ev["off_k"] <= ev["proc_k"]:
                continue
            fracs[robot] = ((reached.get(robot, -1) - (ev["base"] + ev["proc_k"]))
                            / (ev["off_k"] - ev["proc_k"]))
        self._advance_arcs(fracs)
