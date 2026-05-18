from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace

from moveit_configs_utils import MoveItConfigsBuilder


def _moveit_group(ns, use_fake_hardware, robotiq_com_port):
    moveit_config = (
        MoveItConfigsBuilder(
            robot_name="ur10e_2f85_linear_robot",
            package_name="multi_robot_cell_description",
        )
        .robot_description(
            file_path="urdf/ur10e_2f85_linear_robot.urdf.xacro",
            mappings={
                "use_fake_hardware": use_fake_hardware,
                "robotiq_com_port": robotiq_com_port,
            },
        )
        .robot_description_semantic(file_path="srdf/ur10e_2f85_linear_robot.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .planning_pipelines(load_all=False)
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )

    return GroupAction(
        [
            PushRosNamespace(ns),
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                output="screen",
                parameters=[moveit_config.to_dict()],
                remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
            ),
        ]
    )


def generate_launch_description():
    robot1_ns = LaunchConfiguration("robot1_ns")
    robot2_ns = LaunchConfiguration("robot2_ns")

    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    robot1_robotiq_port = LaunchConfiguration("robot1_robotiq_com_port")
    robot2_robotiq_port = LaunchConfiguration("robot2_robotiq_com_port")

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot1_ns", default_value="robot1"),
            DeclareLaunchArgument("robot2_ns", default_value="robot2"),
            DeclareLaunchArgument("use_fake_hardware", default_value="true"),
            DeclareLaunchArgument("robot1_robotiq_com_port", default_value="/dev/robotiq1"),
            DeclareLaunchArgument("robot2_robotiq_com_port", default_value="/dev/robotiq2"),
            _moveit_group(robot1_ns, use_fake_hardware, robot1_robotiq_port),
            _moveit_group(robot2_ns, use_fake_hardware, robot2_robotiq_port),
        ]
    )
