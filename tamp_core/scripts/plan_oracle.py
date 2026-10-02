#!/usr/bin/env python3
"""Independent geometric oracle: replay a FINAL plan tick by tick on the real geometry.

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    python3 scripts/plan_oracle.py --run artifacts/runs/rep_tower_interchangeable_B_1 \
        --mode graph rigid --out /tmp/oracle.json

(ROS Python: it needs `xacro` and the built `plan_oracle_check`; nothing from the VAMP venv.)

WHY
---
``simulate_tpg.py`` checks an execution against ``mu`` -- the same sphere model and the same
matrices that produced the plan. A modelling error the planner and that checker share is
therefore invisible to it: until 2026-09-21 both mu engines placed the carried object 6 cm
short of where it really rides, and every "0 collisions" was measured against that same
mistake. This oracle shares nothing with mu but the robot model. It rebuilds the cell from
the URDF collision meshes and the SRDF, the scene YAML's fixtures and objects (box or mesh),
and replays BOTH robots together, tick by tick, with every object in the state it is
really in at that tick; the check is MoveIt's FCL on that scene (``src/plan_oracle.cpp``).

WHAT IS REPLAYED
----------------
* ``graph`` -- the plan graph's own execution: each tick every robot (in ``tpg.robots``
  order) advances one node iff its incoming edges are satisfied by what the others have
  reached (``deps[r][n]`` against the other robot in a two-robot ``tpg.json``,
  ``deps[r][s][n]`` per other robot ``s`` in an N-robot one), each robot seeing the moves
  of those before it -- the rule of
  ``tpg.zero_delay_ticks`` / ``simulate_tpg.run_tpg``, re-implemented here from the JSON
  (no import of the graph code). With ``--delay p`` it runs under a seeded stall-burst
  trace (same generator as ``simulate_tpg``).
* ``rigid`` -- the solver's shared-clock schedule: each task starts at its ``start_slot``,
  a robot idles at its last node between tasks, at home before its first.

Objects: a scheduled task's object stands at its spawn until its robot enters the task,
then follows the trajectory's ``object_state`` (spawn -> attached -> place), and stays at
its place once the robot has left the task. Objects no scheduled task moves (slot
candidates that lost) stand at spawn throughout. A carried object is fixed to the attach
link where it stands at the first attached sample (the realised grasp, not a model of it),
with ``touch_links`` from the YAML; its geometry is the YAML box or mesh. Places come from
the YAML task or slot (``<slot>__<object>`` ids resolve to the slot's place).

WHAT IS EXCLUDED (and only this): the SRDF's disabled pairs; the gripper's ``touch_links``
against the object it holds, while held; an object on the ``support_surface`` while it
stands, and only a touching contact (<= ``--support-tol``) while carried; a carried object
seated on another object/fixture within a declared fit-up gap (``--fitup-gap`` or the
YAML's ``fitup_gap:``; default 0, i.e. reported).

OUTPUT: JSON with ticks checked, time, and every contact as an interval of ticks with the
two bodies, the penetration depth (FCL's, max over the interval) and what each robot was
doing at its first tick. Exit status 1 if any contact, 2 on an error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

PHASES = ["ToPick", "GripClose", "Carrying", "GripOpen", "ToHome",
          "ProcessOn", "Processing", "ProcessOff"]
AT_SPAWN, ATTACHED, AT_PLACE = 0, 1, 2


# --------------------------------------------------------------------------- #
# poses
# --------------------------------------------------------------------------- #
def quat_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """tf2 ``setRPY`` (R = Rz(yaw) Ry(pitch) Rx(roll)) as (x, y, z, w)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def pose_str(p: dict) -> str:
    q = quat_rpy(p.get("roll", 0.0), p.get("pitch", 0.0), p.get("yaw", 0.0))
    return " ".join(f"{v:.9g}" for v in (p["x"], p["y"], p["z"], *q))


# --------------------------------------------------------------------------- #
# the scene
# --------------------------------------------------------------------------- #
def resolve_mesh(file: str, yaml_path: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (os.path.dirname(os.path.abspath(yaml_path)),
                 os.path.join(os.path.dirname(here), "config")):
        cand = os.path.normpath(os.path.join(base, file))
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"mesh {file!r} not found next to {yaml_path} nor in config/")


def shape_str(entry: dict, yaml_path: str) -> str:
    if "mesh" in entry and entry["mesh"]:
        m = entry["mesh"]
        return f"MESH {resolve_mesh(m['file'], yaml_path)} {float(m.get('scale', 1.0)):.9g}"
    sx, sy, sz = entry["size"]
    return f"BOX {sx:.9g} {sy:.9g} {sz:.9g}"


def robot_model_files(workdir: str, cell: str = "dual") -> Tuple[str, str]:
    """The cell's URDF (xacro-expanded) and SRDF (xacro-expanded when it is one), as MoveIt
    loads them for planning. ``cell`` is the scene's ``cell:``; files come from its cell.yaml."""
    import xacro
    from cell_registry import get_cell
    c = get_cell(cell)
    urdf = os.path.join(workdir, "cell.urdf")
    with open(urdf, "w") as f:
        f.write(xacro.process_file(c.moveit("urdf"), mappings=c.mappings()).toxml())
    srdf_src = c.moveit("srdf")
    if not srdf_src.endswith(".xacro"):
        return urdf, srdf_src
    srdf = os.path.join(workdir, "cell.srdf")
    with open(srdf, "w") as f:
        f.write(xacro.process_file(srdf_src, mappings=c.mappings()).toxml())
    return urdf, srdf


def checker_binary() -> str:
    from ament_index_python.packages import get_package_prefix
    exe = os.path.join(get_package_prefix("multi_robot_cell_tamp"), "lib",
                       "multi_robot_cell_tamp", "plan_oracle_check")
    if not os.path.exists(exe):
        raise FileNotFoundError(f"{exe} missing: colcon build --packages-select multi_robot_cell_tamp")
    return exe


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
class Plan:
    """Scheduled trajectories laid out per robot as contiguous node sequences."""

    def __init__(self, art: dict, sol: dict, task_yaml: dict, tpg: Optional[dict]):
        self.robots: List[str] = list(tpg["robots"] if tpg else art["robots"])
        # `hold` precedences (fabricator v2): holding pick-place -> the process tasks it holds for
        self.holds: Dict[str, set] = {}
        for (i, j), m in zip(art.get("precedences") or [], art.get("precedence_modes") or []):
            if m == "hold":
                self.holds.setdefault(i, set()).add(j)
        trs = {(t["robot"], t["task"]): t for t in art["trajectories"]}
        # Segments come from the GRAPH when there is one (graph mode must replay exactly
        # its node layout), else from the schedule; the two are cross-checked below.
        from_sol: Dict[str, List[Tuple[int, str, int]]] = {r: [] for r in self.robots}
        for task, a in sol["assignments"].items():
            if a["robot"] not in from_sol:
                raise ValueError(f"task {task} assigned to unknown robot {a['robot']}")
            from_sol[a["robot"]].append((int(a["start_slot"]), task, int(a["end_slot"])))
        for r in self.robots:
            from_sol[r].sort()
        self.segments: Dict[str, List[dict]] = {}
        for r in self.robots:
            if tpg:
                segs = [dict(task=s["task"], start_node=int(s["start_node"]), n=int(s["n"]),
                             start_slot=int(s["start_slot"])) for s in tpg["segments"][r]]
                if [s["task"] for s in segs] != [t for _, t, _ in from_sol[r]]:
                    raise ValueError(f"{r}: graph task order {[s['task'] for s in segs]} != "
                                     f"schedule order {[t for _, t, _ in from_sol[r]]}")
            else:
                segs, cursor = [], 0
                for st, t, en in from_sol[r]:
                    segs.append(dict(task=t, start_node=cursor, n=en - st, start_slot=st))
                    cursor += en - st
            for s in segs:
                tr = trs.get((r, s["task"]))
                if tr is None:
                    raise KeyError(f"no trajectory for ({r}, {s['task']}) in the artifact")
                if len(tr["positions"]) != s["n"]:
                    raise ValueError(f"({r}, {s['task']}): {len(tr['positions'])} samples, "
                                     f"the plan says {s['n']}: trajectories and plan are "
                                     f"from different runs")
                s["traj"] = tr
            self.segments[r] = segs
        self.n = {r: sum(s["n"] for s in self.segments[r]) for r in self.robots}

        cfg = task_yaml["robots"]
        self.home = {r: (list(cfg[r]["home"].keys()), [float(v) for v in cfg[r]["home"].values()])
                     for r in self.robots}
        # Fingers (v3, as the trajectory and collision stages model them): a gripper robot's
        # finger joint open with empty hands, closed while it holds its part, and moving
        # linearly across the GripClose / GripOpen dwells. Only when the scene models them
        # (`planning.model_fingers`, default on iff it declares `tool:`); otherwise the
        # fingers stay at the model default, as every replay before.
        pl = task_yaml.get("planning") or {}
        tools = any("tool" in (c or {}) for c in cfg.values())
        self.fingers: Dict[str, Tuple[str, float, float]] = {}
        if bool(pl.get("model_fingers", tools)):
            for r in self.robots:
                c = cfg[r]
                if "gripper_close" in c:
                    self.fingers[r] = (c.get("gripper_joint", f"{r}_robotiq_85_left_knuckle_joint"),
                                       float(c.get("gripper_open", 0.0)), float(c["gripper_close"]))
        # Scene objects and where each scheduled task puts its object.
        self.objects = {o["id"]: o for o in task_yaml.get("objects", [])}
        place_of_slot = {s["id"]: s["place"] for s in task_yaml.get("slots", []) or []}
        place_of_task = {t["id"]: t["place"] for t in task_yaml.get("tasks", []) or []}
        self.task_object: Dict[str, Tuple[str, str, dict]] = {}   # task -> (robot, obj, place)
        for r in self.robots:
            for s in self.segments[r]:
                obj = s["traj"].get("object") or ""
                if not obj:
                    continue
                slot = s["traj"].get("slot")
                place = place_of_task.get(s["task"]) or place_of_slot.get(slot) \
                    or place_of_slot.get(s["task"].split("__")[0])
                if place is None:
                    raise KeyError(f"no place pose for task {s['task']} in the YAML")
                if obj in {o for _, o, _ in self.task_object.values()}:
                    raise ValueError(f"object {obj} is moved by two scheduled tasks")
                self.task_object[s["task"]] = (r, obj, place)

    def locate(self, r: str, node: int) -> Tuple[Optional[dict], int]:
        for s in self.segments[r]:
            if s["start_node"] <= node < s["start_node"] + s["n"]:
                return s, node - s["start_node"]
        return None, -1

    def config(self, r: str, node: int) -> Tuple[List[str], List[float]]:
        if node < 0 or not self.segments[r]:
            return self.home[r]
        s, k = self.locate(r, node)
        return list(s["traj"]["joint_names"]), [float(v) for v in s["traj"]["positions"][k]]

    def object_states(self, pos: Dict[str, int]) -> Dict[str, tuple]:
        """Every object's realised state for the robots at node ``pos``."""
        out = {o: ("spawn",) for o in self.objects}
        for task, (r, obj, place) in self.task_object.items():
            s = next(g for g in self.segments[r] if g["task"] == task)
            node = pos[r]
            if node < s["start_node"]:
                continue
            if node >= s["start_node"] + s["n"]:
                out[obj] = ("place", task)
                continue
            st = int(s["traj"]["object_state"][node - s["start_node"]])
            out[obj] = {AT_SPAWN: ("spawn",), ATTACHED: ("attached", r),
                        AT_PLACE: ("place", task)}[st]
        return out

    def finger(self, r: str, node: int) -> Optional[float]:
        """The finger joint value of robot ``r`` at ``node`` (None: fingers not modelled)."""
        if r not in self.fingers:
            return None
        _, op, cl = self.fingers[r]
        if node < 0 or not self.segments[r]:
            return op
        seg, k = self.locate(r, node)
        ph = seg["traj"]["phase"]
        p = int(ph[k])
        if p in (PHASES.index("GripClose"), PHASES.index("GripOpen")):
            a = k
            while a > 0 and int(ph[a - 1]) == p:
                a -= 1
            b = k
            while b + 1 < len(ph) and int(ph[b + 1]) == p:
                b += 1
            frac = (k - a + 1) / (b - a + 1)
            return op + frac * (cl - op) if p == PHASES.index("GripClose") else cl - frac * (cl - op)
        return cl if int(seg["traj"]["object_state"][k]) == ATTACHED else op

    def context(self, r: str, node: int) -> str:
        if node < 0 or not self.segments[r]:
            return "home"
        s, k = self.locate(r, node)
        ph = s["traj"].get("phase")
        phase = PHASES[int(ph[k])] if ph else "?"
        return f"{s['task']}[{k}/{s['n']}] {phase} (node {node})"


# --------------------------------------------------------------------------- #
# executions: a list of per-tick positions (node index per robot, -1 = home)
# --------------------------------------------------------------------------- #
def delay_trace(seed: int, n_ticks: int, robots: Sequence[str], p: float, burst: int):
    """Seeded stall bursts; the generator of ``simulate_tpg._delay_trace``."""
    rng = np.random.default_rng(seed)
    remaining = {r: 0 for r in robots}
    trace = []
    for _ in range(n_ticks):
        row = {}
        for r in robots:
            if remaining[r] == 0 and rng.random() < p:
                remaining[r] = burst
            row[r] = remaining[r] > 0
            if remaining[r] > 0:
                remaining[r] -= 1
        trace.append(row)
    return trace


def graph_deps(plan: Plan, tpg: dict) -> Dict[str, Dict[str, List[int]]]:
    """``deps[r][s]`` lists from a ``tpg.json`` of either format (flat against THE other
    robot for two robots, per other robot for N), checked against the plan's node counts."""
    robots = list(plan.robots)
    if list(tpg["robots"]) != robots:
        raise ValueError(f"graph robots {tpg['robots']} != plan robots {robots}")
    deps: Dict[str, Dict[str, List[int]]] = {}
    for q in robots:
        raw = tpg["deps"][q]
        if isinstance(raw, dict):
            deps[q] = {o: [int(d) for d in raw[o]] for o in robots if o != q}
        elif len(robots) == 2:
            deps[q] = {next(o for o in robots if o != q): [int(d) for d in raw]}
        else:
            raise ValueError(f"{q}: flat deps (two-robot format) in a {len(robots)}-robot graph")
        for o, row in deps[q].items():
            if len(row) != plan.n[q]:
                raise ValueError(f"{q}: graph has {len(row)} nodes (against {o}), "
                                 f"plan {plan.n[q]}")
    return deps


def run_graph(plan: Plan, tpg: dict, trace=None) -> Tuple[List[Dict[str, int]], bool]:
    robots = list(plan.robots)
    deps = graph_deps(plan, tpg)
    reached = {q: -1 for q in robots}
    ticks = [dict(reached)]                                # tick 0: all at home
    t = 0
    no_stall = {q: False for q in robots}
    while any(reached[q] < plan.n[q] - 1 for q in robots):
        stalls = trace[t] if trace is not None and t < len(trace) else no_stall
        t += 1
        moved = False
        for q in robots:
            nxt = reached[q] + 1
            if nxt >= plan.n[q] or stalls[q]:
                continue
            if all(row[nxt] == -1 or reached[o] >= row[nxt] for o, row in deps[q].items()):
                reached[q] = nxt
                moved = True
        ticks.append(dict(reached))
        if not moved and not any(stalls.values()):
            return ticks, True                             # deadlock
        if trace is not None and t > len(trace) + sum(plan.n.values()):
            return ticks, True
    return ticks, False


def run_rigid(plan: Plan) -> List[Dict[str, int]]:
    horizon = max((g["start_slot"] + g["n"] for q in plan.robots for g in plan.segments[q]),
                  default=0)
    ticks = []
    for t in range(horizon):
        pos = {}
        for q in plan.robots:
            p = -1
            for g in plan.segments[q]:
                if t >= g["start_slot"] + g["n"]:
                    p = g["start_node"] + g["n"] - 1          # done: parked at its last node
                elif t >= g["start_slot"]:
                    p = g["start_node"] + t - g["start_slot"]
                    break
                else:
                    break                                    # idle before this task
            pos[q] = p
        ticks.append(pos)
    return ticks


# --------------------------------------------------------------------------- #
# replay file and the checker
# --------------------------------------------------------------------------- #
def write_replay(path: str, plan: Plan, ticks: List[Dict[str, int]], task_yaml: dict,
                 yaml_path: str, urdf: str, srdf: str, support: str, support_tol: float,
                 fitup_gap: float) -> List[Tuple[int, int, Dict[str, int]]]:
    groups: List[Tuple[int, int, Dict[str, int]]] = []
    for t, pos in enumerate(ticks):
        if groups and groups[-1][2] == pos:
            groups[-1] = (groups[-1][0], t, pos)
        else:
            groups.append((t, t, dict(pos)))

    with open(path, "w") as f:
        w = f.write
        w(f"URDF {urdf}\nSRDF {srdf}\nSUPPORT {support}\n")
        w(f"SUPPORT_TOL {support_tol:.9g}\nFITUP_GAP {fitup_gap:.9g}\n")
        for r in plan.robots:
            cfg = task_yaml["robots"][r]
            touch = list(cfg.get("touch_links", []))
            w(f"ROBOT {r} {cfg['attach_link']} {len(touch)} {' '.join(touch)}\n")
            names = plan.home[r][0] + ([plan.fingers[r][0]] if r in plan.fingers else [])
            w(f"JOINTS {r} {len(names)} {' '.join(names)}\n")
        for oid, o in plan.objects.items():
            w(f"SHAPE {oid} {shape_str(o, yaml_path)}\n")
        for fx in task_yaml.get("fixtures", []) or []:
            w(f"FIXTURE {fx['id']} {shape_str(fx, yaml_path)} {pose_str(fx['pose'])}\n")
        w("BEGIN\n")

        cur_q: Dict[str, List[float]] = {}
        cur_o: Dict[str, tuple] = {}
        # Every object starts at its spawn: a task whose first sample already holds its
        # object is grasped from there.
        for oid, o in plan.objects.items():
            w(f"SPAWN {oid} {pose_str(o['spawn'])}\n")
            cur_o[oid] = ("spawn",)
        for first, last, pos in groups:
            states = plan.object_states(pos)
            # Releases BEFORE the arm moves on: the carried pose is compared with the
            # planned place at the last attached configuration.
            for oid, st in states.items():
                old = cur_o.get(oid)
                if old is not None and old[0] == "attached" and st[0] != "attached":
                    if st[0] != "place":
                        raise ValueError(f"{oid}: attached -> {st[0]} at tick {first}")
                    w(f"PLACE {oid} {pose_str(plan.task_object[st[1]][2])}\n")
                    cur_o[oid] = st
            for r in plan.robots:
                names, q = plan.config(r, pos[r])
                if names != plan.home[r][0]:
                    raise ValueError(f"{r}: trajectory joint order {names} != YAML home order")
                fv = plan.finger(r, pos[r])
                if fv is not None:
                    q = list(q) + [fv]
                if cur_q.get(r) != q:
                    w(f"Q {r} {' '.join(f'{v:.12g}' for v in q)}\n")
                    cur_q[r] = q
            for oid, st in states.items():
                old = cur_o.get(oid)
                if old == st:
                    continue
                if st[0] == "spawn":
                    if old is not None:
                        raise ValueError(f"{oid}: {old[0]} -> spawn at tick {first}")
                    w(f"SPAWN {oid} {pose_str(plan.objects[oid]['spawn'])}\n")
                elif st[0] == "place":
                    w(f"PLACE {oid} {pose_str(plan.task_object[st[1]][2])}\n")
                else:
                    if old is None or old[0] != "spawn":
                        raise ValueError(f"{oid}: attached from {old} at tick {first}")
                    w(f"ATTACH {oid} {st[1]}\n")
                cur_o[oid] = st
            w(f"CHECK {first} {last}\n")
        w("END\n")
    return groups


def parse_contacts(path: str) -> dict:
    out = {"contacts": [], "attach": [], "release": [], "stats": None, "error": None}
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == "CONTACT":
                out["contacts"].append(dict(first=int(p[1]), last=int(p[2]), body1=p[3],
                                            type1=p[4], body2=p[5], type2=p[6],
                                            depth=float(p[7])))
            elif p[0] == "ATTACH":
                out["attach"].append(dict(object=p[1], robot=p[2],
                                          centre_in_attach_link=[float(v) for v in p[3:6]]))
            elif p[0] == "RELEASE":
                out["release"].append(dict(object=p[1], robot=p[2], offset_m=float(p[3]),
                                           angle_rad=float(p[4])))
            elif p[0] == "STATS":
                out["stats"] = dict(checks=int(p[1]), ticks=int(p[2]), contact_ticks=int(p[3]),
                                    seconds=float(p[4]), excused_support=int(p[5]),
                                    excused_fitup=int(p[6]))
            elif p[0] == "ERROR":
                out["error"] = line.strip()
    return out


def merge_intervals(contacts: List[dict]) -> List[dict]:
    """Consecutive tick groups of the same pair -> one interval, deepest depth kept."""
    by_pair: Dict[tuple, List[dict]] = {}
    for c in sorted(contacts, key=lambda c: c["first"]):
        key = (c["body1"], c["type1"], c["body2"], c["type2"])
        runs = by_pair.setdefault(key, [])
        if runs and c["first"] == runs[-1]["last"] + 1:
            runs[-1]["last"] = c["last"]
            if c["depth"] > runs[-1]["depth"]:
                runs[-1]["depth"], runs[-1]["deepest_tick"] = c["depth"], c["first"]
        else:
            runs.append(dict(c, deepest_tick=c["first"]))
    return sorted((r for runs in by_pair.values() for r in runs), key=lambda c: c["first"])


def owner(plan: Plan, body: str, btype: str, states: Dict[str, tuple]) -> str:
    """Which robot a contact body belongs to ("world" for a standing object/fixture)."""
    if btype == "world_object":
        return "world"
    if btype == "attached_object":
        st = states.get(body, ("?",))
        return st[1] if st[0] == "attached" else "?"
    for r in plan.robots:
        if body.startswith(r + "_"):
            return r
    return "cell"          # a robot-model link of neither arm (table, legs)


_PROCESS_PHASES = ("ProcessOn", "Processing", "ProcessOff")


def process_excusable(task_yaml: dict) -> Dict[Tuple[str, str], Tuple[set, set]]:
    """(robot, weld) -> (the robot's ``process_links``, the weld's ``touch:`` ids): the contact
    the trajectory stage deliberately allows while the arc is on (a torch nozzle at a 15 mm
    stickout touches the joint's proxies; CONTEXT.md, Phase 2b)."""
    links = {r: set(cfg.get("process_links") or []) for r, cfg in (task_yaml.get("robots") or {}).items()}
    out = {}
    for w in task_yaml.get("welds") or []:
        touch = set(w.get("touch") or [])
        if not touch:
            continue
        for r, ls in links.items():
            if ls:
                out[(r, w["id"])] = (ls, touch)
    return out


def process_excused(plan: "Plan", ticks: List[Dict[str, int]], c: dict,
                    excusable: Dict[Tuple[str, str], Tuple[set, set]]) -> bool:
    """True iff at EVERY tick of the contact interval ``c`` the robot body is one of its
    robot's process links and the other body is in the ``touch:`` of the weld that robot is
    executing, and at least one of those ticks is in ProcessOn / Processing / ProcessOff.
    The generator's allowance also covers the last / first ``process_contact_depth`` (15 mm)
    of the descent and retreat around the arc, so the contact may begin just before
    ProcessOn and end just after ProcessOff -- but never leave that weld."""
    if not excusable:
        return False
    for i, j in ((1, 2), (2, 1)):
        if c[f"type{i}"] != "robot_link":
            continue
        link, other = c[f"body{i}"], c[f"body{j}"]
        robot = next((r for r in plan.robots if link.startswith(r + "_")), None)
        if robot is None:
            continue
        ok, arc = True, False
        for t in range(c["first"], c["last"] + 1):
            seg, k = plan.locate(robot, ticks[t][robot])
            if seg is None:
                ok = False
                break
            ex = excusable.get((robot, seg["task"]))
            ph = seg["traj"].get("phase")
            if ex is None or link not in ex[0] or other not in ex[1] or not ph:
                ok = False
                break
            arc = arc or PHASES[int(ph[k])] in _PROCESS_PHASES
        if ok and arc:
            return True
    return False


def hold_tail(seg: dict) -> Tuple[int, int]:
    """[h, g) of a holding pick-place's trajectory: its hold (``hold_slots`` before the first
    GripOpen sample) and the GripOpen dwell -- the part frozen at its place pose."""
    ph = [int(x) for x in seg["traj"]["phase"]]
    m1 = ph.index(PHASES.index("GripOpen"))
    g = m1
    while g < len(ph) and ph[g] == PHASES.index("GripOpen"):
        g += 1
    return m1 - int(seg["traj"].get("hold_slots") or 0), g


def hold_excused(plan: "Plan", ticks: List[Dict[str, int]], c: dict,
                 excusable: Dict[Tuple[str, str], Tuple[set, set]]) -> bool:
    """The synchronous hold's process contact (fabricator v2): True iff at EVERY tick of the
    contact interval one body is a process link of a welder W executing a process task j,
    the other the part ATTACHED to a handler H executing the pick-place i that holds for j
    (precedence ``hold``), H is on its hold tail + GripOpen (the part frozen at its place
    pose), and the part is in j's ``touch:``. It is the contact the trajectory stage allowed
    when it planned j with the part at its place pose (``process_excused``'s allowance),
    met here with the part still in the handler's fingers."""
    if not plan.holds:
        return False
    for a, b in ((1, 2), (2, 1)):
        if c[f"type{a}"] != "robot_link" or c[f"type{b}"] != "attached_object":
            continue
        link, part = c[f"body{a}"], c[f"body{b}"]
        w = next((r for r in plan.robots if link.startswith(r + "_")), None)
        if w is None:
            continue
        ok = True
        for t in range(c["first"], c["last"] + 1):
            seg_w, _ = plan.locate(w, ticks[t][w])
            if seg_w is None:
                ok = False
                break
            ex = excusable.get((w, seg_w["task"]))
            if ex is None or link not in ex[0] or part not in ex[1]:
                ok = False
                break
            holder = next(((q, g) for q in plan.robots if q != w
                           for g in [plan.locate(q, ticks[t][q])]
                           if g[0] is not None and seg_w["task"] in plan.holds.get(g[0]["task"], ())
                           and (g[0]["traj"].get("object") or "") == part), None)
            if holder is None:
                ok = False
                break
            (seg_h, k_h) = holder[1]
            h, gend = hold_tail(seg_h)
            if not (h <= k_h < gend):
                ok = False
                break
        if ok:
            return True
    return False


def check(plan: Plan, ticks: List[Dict[str, int]], task_yaml: dict, yaml_path: str,
          urdf: str, srdf: str, workdir: str, support_tol: float, fitup_gap: float,
          label: str, keep: bool) -> dict:
    t0 = time.time()
    replay = os.path.join(workdir, f"replay_{label}.txt")
    result = os.path.join(workdir, f"contacts_{label}.txt")
    support = task_yaml.get("support_surface", "table_top")
    groups = write_replay(replay, plan, ticks, task_yaml, yaml_path, urdf, srdf, support,
                          support_tol, fitup_gap)
    proc = subprocess.run([checker_binary(), replay, result], capture_output=True, text=True)
    parsed = parse_contacts(result) if os.path.exists(result) else {"error": proc.stderr}
    if proc.returncode != 0 or parsed.get("error"):
        raise RuntimeError(f"plan_oracle_check failed ({proc.returncode}): "
                           f"{parsed.get('error') or proc.stderr[-2000:]}")
    intervals = merge_intervals(parsed["contacts"])
    for c in intervals:
        pos = ticks[c["first"]]
        c["context"] = {r: plan.context(r, pos[r]) for r in plan.robots}
        c["ticks"] = c["last"] - c["first"] + 1
        states = plan.object_states(pos)
        c["owner1"], c["owner2"] = (owner(plan, c[f"body{i}"], c[f"type{i}"], states)
                                    for i in (1, 2))
        # robot-robot: the two robots (links or what they carry) -- mu's responsibility.
        # robot-world: a robot or its load against a standing object or fixture -- the
        # planning scene's (ADR-0003) and the task order's. self: one robot with itself.
        o1, o2 = c["owner1"], c["owner2"]
        c["kind"] = ("robot-world" if "world" in (o1, o2)
                     else "self" if o1 == o2 else "robot-robot")
    # The weld process allowance (a scene's `process_links` x a weld's `touch:`, arc on):
    # excused and counted, exactly like the support and fit-up excuses.
    excusable = process_excusable(task_yaml)
    kept = [c for c in intervals if not (c["kind"] == "robot-world"
                                         and process_excused(plan, ticks, c, excusable))]
    n_process = len(intervals) - len(kept)
    # The synchronous hold: a welder's process links on the part its handler holds, the part
    # frozen at its place pose (hold tail + GripOpen). Excused, counted, depths kept.
    held = [c for c in kept if c["kind"] == "robot-robot" and hold_excused(plan, ticks, c, excusable)]
    intervals = [c for c in kept if c not in held]
    if not keep:
        os.remove(replay)
        os.remove(result)
    return {
        "ticks_checked": len(ticks),
        "distinct_states_checked": len(groups),
        "contact_ticks": parsed["stats"]["contact_ticks"],
        "n_contacts": len(intervals),
        "contacts": intervals,
        "excused": {"support_touch_checks": parsed["stats"]["excused_support"],
                    "fitup_checks": parsed["stats"]["excused_fitup"],
                    "process_contact_intervals": n_process,
                    "hold_contact_intervals": len(held),
                    "hold_contacts": [dict(first=c["first"], last=c["last"], body1=c["body1"],
                                           body2=c["body2"], depth_m=c["depth"],
                                           context=c["context"]) for c in held]},
        "attach": parsed["attach"],
        "release": parsed["release"],
        "max_release_offset_m": max((x["offset_m"] for x in parsed["release"]), default=0.0),
        "checker_seconds": parsed["stats"]["seconds"],
        "seconds": time.time() - t0,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="a run directory holding tamp_trajectories.json, "
                                 "tamp_solution.json, tpg.json and the task YAML; any of the "
                                 "explicit paths below overrides it")
    p.add_argument("--traj")
    p.add_argument("--solution")
    p.add_argument("--tpg")
    p.add_argument("--task")
    p.add_argument("--mode", nargs="+", default=["graph", "rigid"], choices=["graph", "rigid"])
    p.add_argument("--delay", type=float, default=0.0,
                   help="graph mode also under a stall-burst trace with this per-tick rate")
    p.add_argument("--burst", type=int, default=160)
    p.add_argument("--seed", type=int, nargs="*", default=[0])
    p.add_argument("--support-tol", type=float, default=2e-3)
    p.add_argument("--fitup-gap", type=float, default=None)
    p.add_argument("--out")
    p.add_argument("--workdir")
    p.add_argument("--keep-replay", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    t_start = time.time()
    try:
        def pick(explicit, name):
            if explicit:
                return explicit
            if not args.run:
                raise ValueError(f"--{name} or --run is required")
            return os.path.join(args.run, name)
        traj_p = pick(args.traj, "tamp_trajectories.json")
        sol_p = pick(args.solution, "tamp_solution.json")
        tpg_p = args.tpg or (os.path.join(args.run, "tpg.json") if args.run else None)
        task_p = args.task
        if not task_p:
            ymls = [f for f in os.listdir(args.run) if f.startswith("tamp_task") and f.endswith(".yaml")]
            if len(ymls) != 1:
                raise ValueError(f"cannot pick the task YAML in {args.run}: {ymls}")
            task_p = os.path.join(args.run, ymls[0])
        art = json.load(open(traj_p))
        sol = json.load(open(sol_p))
        task_yaml = yaml.safe_load(open(task_p))
        tpg = json.load(open(tpg_p)) if tpg_p and os.path.exists(tpg_p) else None
        if "graph" in args.mode and tpg is None:
            raise ValueError("graph mode needs --tpg")
        fitup = args.fitup_gap if args.fitup_gap is not None else float(task_yaml.get("fitup_gap", 0.0))

        workdir = args.workdir or tempfile.mkdtemp(prefix="plan_oracle_")
        os.makedirs(workdir, exist_ok=True)
        urdf, srdf = robot_model_files(workdir, str(task_yaml.get("cell") or "dual"))

        report = {"inputs": dict(traj=traj_p, solution=sol_p, tpg=tpg_p, task=task_p),
                  "support_tol_m": args.support_tol, "fitup_gap_m": fitup, "runs": {}}
        for mode in args.mode:
            plan = Plan(art, sol, task_yaml, tpg if mode == "graph" else None)
            variants = [("rigid", None, None)] if mode == "rigid" else [("graph", None, None)]
            if mode == "graph" and args.delay > 0:
                horizon = 8 * sum(plan.n.values())
                variants += [(f"graph_delay{args.delay}_seed{sd}",
                              delay_trace(sd, horizon, plan.robots, args.delay, args.burst), sd)
                             for sd in args.seed]
            for label, trace, _ in variants:
                if mode == "rigid":
                    ticks, dead = run_rigid(plan), False
                else:
                    ticks, dead = run_graph(plan, tpg, trace)
                res = check(plan, ticks, task_yaml, task_p, urdf, srdf, workdir,
                            args.support_tol, fitup, label, args.keep_replay)
                res["deadlock"] = dead
                res["makespan_ticks"] = len(ticks) - 1 if mode == "graph" else len(ticks)
                report["runs"][label] = res
        report["seconds"] = time.time() - t_start
        report["total_contacts"] = sum(r["n_contacts"] for r in report["runs"].values())
    except Exception as e:  # noqa: BLE001 -- any failure is a non-verdict, exit 2
        print(f"plan_oracle: ERROR: {e}", file=sys.stderr)
        return 2

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
    if not args.quiet:
        for label, res in report["runs"].items():
            print(f"[{label}] {res['ticks_checked']} ticks ({res['distinct_states_checked']} "
                  f"distinct states) in {res['seconds']:.1f} s; makespan {res['makespan_ticks']}; "
                  f"deadlock {res['deadlock']}; {res['n_contacts']} contact interval(s); "
                  f"excused: {res['excused']['process_contact_intervals']} process, "
                  f"{res['excused']['hold_contact_intervals']} hold"
                  + (f" (max depth {1000 * max(x['depth_m'] for x in res['excused']['hold_contacts']):.1f} mm)"
                     if res['excused']['hold_contacts'] else "") + "; "
                  f"max release offset {1000 * res['max_release_offset_m']:.2f} mm")
            for c in res["contacts"][:40]:
                print(f"    [{c['kind']}] ticks {c['first']}-{c['last']}: {c['body1']} ({c['type1']}) x "
                      f"{c['body2']} ({c['type2']}), depth {1000 * c['depth']:.2f} mm "
                      f"@ {c['deepest_tick']}; " +
                      "; ".join(f"{r}: {v}" for r, v in c["context"].items()))
            if len(res["contacts"]) > 40:
                print(f"    ... {len(res['contacts']) - 40} more")
        print(f"total {report['seconds']:.1f} s")
    return 1 if report["total_contacts"] or any(r["deadlock"] for r in report["runs"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
