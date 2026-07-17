"""Replay a solved schedule on the running cell's controllers.

Needs the cell up in another terminal:

    ros2 launch multi_robot_cell_bringup start.launch.py

then:

    ros2 launch multi_robot_cell_tamp execute_schedule.launch.py \\
        traj_file:=/tmp/tamp_trajectories.json \\
        solution_file:=/tmp/tamp_solution.json

Both arms play their trajectories on the shared clock; watch it in RViz.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("traj_file", default_value="/tmp/tamp_trajectories.json"),
            DeclareLaunchArgument("solution_file", default_value="/tmp/tamp_solution.json"),
            DeclareLaunchArgument("start_delay", default_value="2.0"),
            Node(
                package="multi_robot_cell_tamp",
                executable="schedule_executor.py",
                output="screen",
                parameters=[
                    {
                        "traj_file": LaunchConfiguration("traj_file"),
                        "solution_file": LaunchConfiguration("solution_file"),
                        "start_delay": LaunchConfiguration("start_delay"),
                    }
                ],
            ),
        ]
    )
