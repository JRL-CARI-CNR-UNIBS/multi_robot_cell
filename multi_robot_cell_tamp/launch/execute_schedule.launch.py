"""Replay a solved schedule on the running cell's controllers.

Needs the cell up in another terminal:

    ros2 launch multi_robot_cell_bringup start.launch.py

then:

    ros2 launch multi_robot_cell_tamp execute_schedule.launch.py \\
        traj_file:=/tmp/tamp_trajectories.json \\
        solution_file:=/tmp/tamp_solution.json

Both arms play their trajectories on the shared clock; watch it in RViz. With
``visualize:=true`` (default) the tray/lid/boxes are also rendered and animated
(attach on pick, land at the place pose on place), read from the SAME
``config/tamp_task.yaml`` the offline generators use.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def launch_setup(context):
    # Empty task_file -> the installed config/tamp_task.yaml (mirrors
    # generate_collisions.launch.py). One geometry source across the pipeline.
    task_default = PathJoinSubstitution(
        [FindPackageShare("multi_robot_cell_tamp"), "config", "tamp_task.yaml"]
    ).perform(context)
    task_file = LaunchConfiguration("task_file").perform(context) or task_default

    return [
        Node(
            package="multi_robot_cell_tamp",
            executable="schedule_executor.py",
            output="screen",
            parameters=[
                {
                    "traj_file": LaunchConfiguration("traj_file"),
                    "solution_file": LaunchConfiguration("solution_file"),
                    "start_delay": LaunchConfiguration("start_delay"),
                    "task_file": task_file,
                    "visualize": ParameterValue(
                        LaunchConfiguration("visualize"), value_type=bool
                    ),
                    "actuate_grippers": ParameterValue(
                        LaunchConfiguration("actuate_grippers"), value_type=bool
                    ),
                }
            ],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("traj_file", default_value="/tmp/tamp_trajectories.json"),
            DeclareLaunchArgument("solution_file", default_value="/tmp/tamp_solution.json"),
            DeclareLaunchArgument("start_delay", default_value="2.0"),
            DeclareLaunchArgument(
                "task_file",
                default_value="",
                description="Scene geometry YAML; empty -> installed config/tamp_task.yaml.",
            ),
            DeclareLaunchArgument(
                "visualize",
                default_value="true",
                description="Render + animate the scene objects (tray/lid/boxes) in RViz.",
            ),
            DeclareLaunchArgument(
                "actuate_grippers",
                default_value="true",
                description="Open/close the grippers in sync with each pick/place.",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
