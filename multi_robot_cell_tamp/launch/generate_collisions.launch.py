"""Offline inter-robot collision stage — trajectory artifact -> SchedulingProblem.

Reads the trajectories from the previous stage, computes the inter-robot collision
matrices via FCL on the loaded robot model, reduces them to forbidden start-offset
sets, and writes the geometry-free problem the Python scheduler solves.

    ros2 launch multi_robot_cell_tamp generate_collisions.launch.py \\
        traj_file:=/tmp/tamp_trajectories.json \\
        out_file:=/tmp/tamp_problem.json

Needs the robot model (for FK + collision geometry) but not the running cell.
"""

import xacro
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context):
    share = FindPackageShare("multi_robot_moveit_config")
    urdf = PathJoinSubstitution([share, "config", "multi_robot_cell.urdf.xacro"]).perform(context)
    srdf = PathJoinSubstitution([share, "config", "multi_robot_cell.srdf"]).perform(context)

    # The collision node only needs the model's geometry — no controllers, no
    # planning pipeline — so the URDF/SRDF are loaded directly. MoveItConfigsBuilder
    # is avoided on purpose: its to_moveit_configs() auto-loads the package's
    # moveit_controllers.yaml, which is fully commented out and makes it throw.
    robot_description = xacro.process_file(urdf).toxml()
    with open(srdf) as f:
        robot_description_semantic = f.read()

    task_file = PathJoinSubstitution(
        [FindPackageShare("multi_robot_cell_tamp"), "config", "tamp_task.yaml"]
    ).perform(context)

    return [
        Node(
            package="multi_robot_cell_tamp",
            executable="collision_generator",
            output="screen",
            parameters=[
                {"robot_description": robot_description},
                {"robot_description_semantic": robot_description_semantic},
                {
                    "traj_file": LaunchConfiguration("traj_file").perform(context),
                    "task_file": LaunchConfiguration("task_file").perform(context) or task_file,
                    "out_file": LaunchConfiguration("out_file").perform(context),
                },
            ],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("traj_file", default_value="/tmp/tamp_trajectories.json"),
            DeclareLaunchArgument("task_file", default_value=""),
            DeclareLaunchArgument("out_file", default_value="/tmp/tamp_problem.json"),
            OpaqueFunction(function=launch_setup),
        ]
    )
