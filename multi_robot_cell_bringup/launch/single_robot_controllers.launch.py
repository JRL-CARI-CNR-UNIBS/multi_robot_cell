from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    SetLaunchConfiguration,
)
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile


ACTIVE_CONTROLLERS = [
    "joint_trajectory_controller",
    "gripper_action_controller",
]

INACTIVE_CONTROLLERS = [
    # "scaled_joint_trajectory_controller",
    # "joint_trajectory_controller",
    ""
]


def launch_setup(context, *args, **kwargs):

    prefix = LaunchConfiguration("prefix").perform(context)

    controllers_file = PathJoinSubstitution([
        FindPackageShare("multi_robot_cell_bringup"),
        "config",
        "controllers.yaml"
    ])

    # controller_params = ParameterFile(
    #     controllers_file,
    #     allow_substs=True
    # )


    nodes = []

    #
    # active controllers
    #
    for controller in ACTIVE_CONTROLLERS:

        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    f"{prefix}_{controller}",
                    "--controller-manager",
                    "/controller_manager",
                    "-p",
                    controllers_file,
                ],
                output="screen",
            )
        )

    #
    # inactive controllers
    #
    for controller in INACTIVE_CONTROLLERS:

        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    f"{prefix}_{controller}",
                    "--controller-manager",
                    "/controller_manager",
                    "-p",
                    controllers_file,
                    "--inactive",
                ],
                output="screen",
            )
        )

    return nodes


def generate_launch_description():

    return LaunchDescription([

        DeclareLaunchArgument(
            "prefix",
        ),

        OpaqueFunction(
            function=launch_setup
        )
    ])