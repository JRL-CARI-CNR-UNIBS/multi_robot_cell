"""Offline trajectory generation — plans one trajectory per (robot, task) pair.

This node does NOT need the cell running. It owns its own PlanningScene and
planning pipeline, so there is no move_group to talk to and nothing is executed:

    ros2 launch multi_robot_cell_tamp generate_trajectories.launch.py

The artifact it writes is the geometry-free seam — the Python scheduler reads it
and never sees a robot model.
"""

import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from cell_moveit import cell_of_task, moveit_config_for, scene_path  # noqa: E402

# Persistent, non-/tmp output dir under the package SOURCE tree (survives reboots).
# realpath() resolves the install/ symlink back to source when built with
# --symlink-install, so the artifact lands next to the package, not in install/.
# The shared trajectory artifact feeds BOTH the FCL and VAMP collision stages.
ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "artifacts"
)


def launch_setup(context):
    task_file = LaunchConfiguration("task_file").perform(context) or scene_path("nominal")
    # URDF, SRDF and MoveIt files of the scene's cell (`cell:` in its YAML, default dual):
    # launch/cell_moveit.py. The generator never executes anything, but the builder still
    # loads a controllers file -- the bringup's (moveit_config's own is commented out).
    moveit_config = moveit_config_for(cell_of_task(task_file), context)

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
                    "task_file": task_file,
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
