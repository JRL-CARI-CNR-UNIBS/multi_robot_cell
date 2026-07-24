"""Offline trajectory generation — plans one trajectory per (robot, task) pair.

This node does NOT need the cell running. It owns its own PlanningScene and
planning pipeline, so there is no move_group to talk to and nothing is executed:

    ros2 launch multi_robot_cell_tamp generate_trajectories.launch.py

The artifact it writes is the geometry-free seam — the Python scheduler reads it
and never sees a robot model.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder

# Persistent, non-/tmp output dir under the package SOURCE tree (survives reboots).
# realpath() resolves the install/ symlink back to source when built with
# --symlink-install, so the artifact lands next to the package, not in install/.
# The shared trajectory artifact feeds BOTH the FCL and VAMP collision stages.
ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "artifacts"
)


def launch_setup(context):
    share = FindPackageShare("multi_robot_moveit_config")
    urdf = PathJoinSubstitution([share, "config", "multi_robot_cell.urdf.xacro"]).perform(context)
    srdf = PathJoinSubstitution([share, "config", "multi_robot_cell.srdf"]).perform(context)
    limits = PathJoinSubstitution([share, "config", "joint_limits.yaml"]).perform(context)
    # The moveit_config package's moveit_controllers.yaml is fully commented out and
    # to_moveit_configs() chokes on it; the bringup copy is the working one. (The
    # generator never executes anything, but the builder still loads the file.)
    controllers = PathJoinSubstitution(
        [FindPackageShare("multi_robot_cell_bringup"), "config", "moveit_controllers.yaml"]
    ).perform(context)

    moveit_config = (
        MoveItConfigsBuilder("manipulator", package_name="multi_robot_moveit_config")
        .robot_description(file_path=urdf)
        .robot_description_semantic(file_path=srdf)
        .trajectory_execution(file_path=controllers)
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
        .joint_limits(file_path=limits)
        .to_moveit_configs()
    )

    task_file = PathJoinSubstitution(
        [FindPackageShare("multi_robot_cell_tamp"), "config", "tamp_task.yaml"]
    ).perform(context)

    # Make sure the output dir exists before the C++ node opens the file for writing.
    os.makedirs(os.path.dirname(LaunchConfiguration("out_file").perform(context)), exist_ok=True)

    return [
        Node(
            package="multi_robot_cell_tamp",
            executable="trajectory_generator",
            output="screen",
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.planning_pipelines,
                moveit_config.joint_limits,
                {
                    "task_file": LaunchConfiguration("task_file").perform(context) or task_file,
                    "out_file": LaunchConfiguration("out_file").perform(context),
                },
            ],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("task_file", default_value=""),
            DeclareLaunchArgument(
                "out_file",
                default_value=os.path.join(ARTIFACTS_DIR, "tamp_trajectories.json"),
                description="Where the offline trajectory artifact is written "
                "(shared by the FCL and VAMP collision stages).",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
