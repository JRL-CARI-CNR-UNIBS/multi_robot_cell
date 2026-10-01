"""Which robot cell a TAMP scene runs on, and the MoveIt configuration of that cell.

Shared by pipeline.launch.py, generate_trajectories.launch.py and generate_collisions.launch.py
(imported from this directory; not a launch file itself). The cell is a property of the SCENE:
its task YAML may carry a top-level ``cell:`` key -- absent means ``dual``, the two-robot cell
every scene before fabricator4 was written for -- so a scene cannot be run on a cell it was not
built for, and no launch argument has to be kept in step with it.

    dual         multi_robot_moveit_config/config/multi_robot_cell.{urdf.xacro,srdf}, config/
                 joint_limits.yaml, the bringup's moveit_controllers.yaml -- byte for byte the
                 parameters these launch files built before this helper existed.
    fabricator4  multi_robot_moveit_config/config/fabricator4/: the wrapper URDF, the SRDF (a
                 xacro, generated from the description's fabricator4_layout.yaml), kinematics,
                 joint_limits and OMPL files; the bringup's fabricator4_moveit_controllers.yaml.
"""
import yaml
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder

CELLS = ("dual", "fabricator4")


def cell_of_task(task_file):
    """The ``cell:`` of a task YAML ('dual' when absent)."""
    with open(task_file) as f:
        data = yaml.safe_load(f) or {}
    cell = str(data.get("cell") or "dual")
    if cell not in CELLS:
        raise RuntimeError(f"{task_file}: unknown cell '{cell}' (known: {', '.join(CELLS)})")
    return cell


def robot_count(task_file):
    with open(task_file) as f:
        return len((yaml.safe_load(f) or {}).get("robots", {}))


def cell_files(cell, context):
    """(urdf xacro, srdf file, srdf_is_xacro, joint_limits, kinematics or None, ompl or None,
    moveit controllers) for ``cell``."""
    share = FindPackageShare("multi_robot_moveit_config").perform(context)
    bringup = FindPackageShare("multi_robot_cell_bringup").perform(context)
    if cell == "dual":
        return (f"{share}/config/multi_robot_cell.urdf.xacro", f"{share}/config/multi_robot_cell.srdf",
                False, f"{share}/config/joint_limits.yaml", None, None,
                # multi_robot_moveit_config's own moveit_controllers.yaml is fully commented out
                # and to_moveit_configs() chokes on it; the bringup copy is the working one.
                PathJoinSubstitution([bringup, "config", "moveit_controllers.yaml"]).perform(context))
    cfg = f"{share}/config/{cell}"
    return (f"{cfg}/{cell}_cell.urdf.xacro", f"{cfg}/{cell}_cell.srdf.xacro", True,
            f"{cfg}/joint_limits.yaml", f"{cfg}/kinematics.yaml", f"{cfg}/ompl_planning.yaml",
            f"{bringup}/config/{cell}_moveit_controllers.yaml")


def moveit_config_for(cell, context):
    """MoveItConfigs of ``cell`` for the offline MoveIt nodes (trajectory generator)."""
    urdf, srdf, _, limits, kin, ompl, controllers = cell_files(cell, context)
    b = (MoveItConfigsBuilder("manipulator", package_name="multi_robot_moveit_config")
         .robot_description(file_path=urdf)
         .robot_description_semantic(file_path=srdf))
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
    if srdf_xacro:
        semantic = xacro.process_file(srdf).toxml()
    else:
        with open(srdf) as f:
            semantic = f.read()
    return xacro.process_file(urdf).toxml(), semantic
