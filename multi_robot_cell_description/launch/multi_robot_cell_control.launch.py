from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import FindExecutable


def _robot_description(use_fake_hardware, robotiq_com_port):
    xacro_path = PathJoinSubstitution(
        [
            FindPackageShare("multi_robot_cell_description"),
            "urdf",
            "ur10e_2f85_linear_robot.urdf.xacro",
        ]
    )

    return Command(
        [
            FindExecutable(name="xacro"),
            " ",
            xacro_path,
            " ",
            "use_fake_hardware:=",
            use_fake_hardware,
            " ",
            "robotiq_com_port:=",
            robotiq_com_port,
        ]
    )


def _robot_control_group(ns, controllers_yaml, use_fake_hardware, robotiq_com_port):
    robot_description = _robot_description(use_fake_hardware, robotiq_com_port)

    return GroupAction(
        [
            PushRosNamespace(ns),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
                remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
            ),
            Node(
                package="controller_manager",
                executable="ros2_control_node",
                output="screen",
                parameters=[
                    {"robot_description": robot_description},
                    controllers_yaml,
                ],
                remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                output="screen",
                arguments=[
                    "joint_state_broadcaster",
                    "--controller-manager",
                    "controller_manager",
                ],
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                output="screen",
                arguments=[
                    "joint_trajectory_controller",
                    "--controller-manager",
                    "controller_manager",
                ],
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                output="screen",
                arguments=[
                    "gripper_action_controller",
                    "--controller-manager",
                    "controller_manager",
                ],
            ),
        ]
    )


def generate_launch_description():
    robot1_ns = LaunchConfiguration("robot1_ns")
    robot2_ns = LaunchConfiguration("robot2_ns")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")

    robot1_robotiq_port = LaunchConfiguration("robot1_robotiq_com_port")
    robot2_robotiq_port = LaunchConfiguration("robot2_robotiq_com_port")

    pkg_share = FindPackageShare("multi_robot_cell_description")

    robot1_controllers = PathJoinSubstitution([pkg_share, "controllers", "robot1_controllers.yaml"])
    robot2_controllers = PathJoinSubstitution([pkg_share, "controllers", "robot2_controllers.yaml"])

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot1_ns", default_value="robot1"),
            DeclareLaunchArgument("robot2_ns", default_value="robot2"),
            DeclareLaunchArgument("use_fake_hardware", default_value="true"),
            DeclareLaunchArgument("robot1_robotiq_com_port", default_value="/dev/robotiq1"),
            DeclareLaunchArgument("robot2_robotiq_com_port", default_value="/dev/robotiq2"),
            _robot_control_group(
                ns=robot1_ns,
                controllers_yaml=robot1_controllers,
                use_fake_hardware=use_fake_hardware,
                robotiq_com_port=robot1_robotiq_port,
            ),
            _robot_control_group(
                ns=robot2_ns,
                controllers_yaml=robot2_controllers,
                use_fake_hardware=use_fake_hardware,
                robotiq_com_port=robot2_robotiq_port,
            ),
        ]
    )
