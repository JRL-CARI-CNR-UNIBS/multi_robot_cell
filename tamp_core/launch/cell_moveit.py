"""Which robot cell a TAMP scene runs on, and the MoveIt configuration of that cell.

Shared by pipeline.launch.py, generate_trajectories.launch.py and generate_collisions.launch.py
(imported from this directory; not a launch file itself). The cell is a property of the SCENE:
its task YAML may carry a top-level ``cell:`` key -- absent means ``dual``, the two-robot cell
every scene before fabricator4 was written for -- so a scene cannot be run on a cell it was not
built for, and no launch argument has to be kept in step with it.

tamp_core knows no cell by name: each cell directory next to it carries a ``cell.yaml`` naming
its MoveIt files (see ``scripts/cell_registry.py``).
"""
import os
import sys

import yaml
from launch_ros.substitutions import FindPackageShare  # noqa: F401  (kept for importers)
from moveit_configs_utils import MoveItConfigsBuilder

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "scripts"))
from cell_registry import cell_of_task, get_cell, scene_path  # noqa: E402,F401


def robot_count(task_file):
    with open(task_file) as f:
        return len((yaml.safe_load(f) or {}).get("robots", {}))


def cell_files(cell, context=None):
    """(urdf xacro, srdf file, srdf_is_xacro, joint_limits, kinematics or None, ompl or None,
    moveit controllers) of ``cell``."""
    c = get_cell(cell)
    srdf = c.moveit("srdf")
    return (c.moveit("urdf"), srdf, srdf.endswith(".xacro"), c.moveit("joint_limits"),
            c.moveit("kinematics"), c.moveit("ompl"), c.moveit("controllers"))


def moveit_config_for(cell, context):
    """MoveItConfigs of ``cell`` for the offline MoveIt nodes (trajectory generator)."""
    urdf, srdf, _, limits, kin, ompl, controllers = cell_files(cell, context)
    mappings = get_cell(cell).mappings()
    b = (MoveItConfigsBuilder("manipulator", package_name=get_cell(cell).package)
         .robot_description(file_path=urdf, mappings=mappings)
         .robot_description_semantic(file_path=srdf, mappings=mappings))
    if kin:
        b = b.robot_description_kinematics(file_path=kin)
    cfg = (b.trajectory_execution(file_path=controllers)
           .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
           .joint_limits(file_path=limits)
           .to_moveit_configs())
    if ompl:
        # planning_pipelines() only reads config/<pipeline>_planning.yaml: swap in the cell's
        # file (same planner configs, one group entry per robot), as the bringup does.
        with open(ompl) as f:
            cfg.planning_pipelines["ompl"] = yaml.safe_load(f)
    return cfg


def description_and_semantic(cell, context):
    """(robot_description XML, robot_description_semantic text) of ``cell`` -- what the FCL
    collision node takes."""
    import xacro

    urdf, srdf, srdf_xacro, *_ = cell_files(cell, context)
    mappings = get_cell(cell).mappings()
    if srdf_xacro:
        semantic = xacro.process_file(srdf, mappings=mappings).toxml()
    else:
        with open(srdf) as f:
            semantic = f.read()
    return xacro.process_file(urdf, mappings=mappings).toxml(), semantic
