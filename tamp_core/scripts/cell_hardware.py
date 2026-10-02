"""Which controller and which gripper driver each robot of a scene uses (the executors' side).

The scene YAML is the one source: a robot may name its arm action (``controller_action``; the UR
cells' default is ``/<robot><suffix>/follow_joint_trajectory``) and its gripper interface --
``gripper_joint`` present means a JointTrajectoryController gripper (TIAGo's PAL gripper,
``trajectory_gripper_commander``), absent the ``GripperCommand`` one (``gripper_commander``).
"""
from __future__ import annotations

import yaml


def _robots(task_file: str) -> dict:
    with open(task_file) as f:
        return (yaml.safe_load(f) or {}).get("robots", {}) or {}


def controller_action(task_file: str, robot: str, suffix: str) -> str:
    action = (_robots(task_file).get(robot) or {}).get("controller_action")
    return action or f"/{robot}{suffix}/follow_joint_trajectory"


def make_gripper_commander(node, task_file: str):
    if any("gripper_joint" in (cfg or {}) for cfg in _robots(task_file).values()):
        from trajectory_gripper_commander import TrajectoryGripperCommander
        return TrajectoryGripperCommander(node, task_file)
    from gripper_commander import GripperCommander
    return GripperCommander(node, task_file)
