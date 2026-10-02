from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    robot1_ns = LaunchConfiguration("robot1_ns")
    robot2_ns = LaunchConfiguration("robot2_ns")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot1_ns", default_value="robot1"),
            DeclareLaunchArgument("robot2_ns", default_value="robot2"),
            DeclareLaunchArgument("use_fake_hardware", default_value="true"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("multi_robot_cell_description"),
                            "launch",
                            "multi_robot_cell_control.launch.py",
                        ]
                    )
                ),
                launch_arguments={
                    "robot1_ns": robot1_ns,
                    "robot2_ns": robot2_ns,
                    "use_fake_hardware": use_fake_hardware,
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("multi_robot_cell_description"),
                            "launch",
                            "multi_robot_cell_moveit.launch.py",
                        ]
                    )
                ),
                launch_arguments={
                    "robot1_ns": robot1_ns,
                    "robot2_ns": robot2_ns,
                    "use_fake_hardware": use_fake_hardware,
                }.items(),
            ),
        ]
    )
