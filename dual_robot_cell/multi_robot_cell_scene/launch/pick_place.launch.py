from launch import LaunchDescription
from launch.actions import OpaqueFunction
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context):
    srdf_path = PathJoinSubstitution(
        [FindPackageShare("multi_robot_moveit_config"), "config", "multi_robot_cell.srdf"]
    ).perform(context)
    joint_limits_path = PathJoinSubstitution(
        [FindPackageShare("multi_robot_moveit_config"), "config", "joint_limits.yaml"]
    ).perform(context)

    # Same config the move_group launch builds, so the interface agrees with
    # the running move_group. Kinematics is loaded automatically and is needed
    # for pose-goal IK.
    urdf_path = PathJoinSubstitution(
        [FindPackageShare("multi_robot_moveit_config"), "config", "multi_robot_cell.urdf.xacro"]
    ).perform(context)
    # The moveit_config package ships a fully-commented moveit_controllers.yaml,
    # which to_moveit_configs() would try to load and choke on (None). Use the
    # working controllers file from the bringup package instead.
    controllers_path = PathJoinSubstitution(
        [FindPackageShare("multi_robot_cell_bringup"), "config", "moveit_controllers.yaml"]
    ).perform(context)

    moveit_config = (
        MoveItConfigsBuilder("manipulator", package_name="multi_robot_moveit_config")
        .robot_description(file_path=urdf_path)
        .robot_description_semantic(file_path=srdf_path)
        .trajectory_execution(file_path=controllers_path)
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
        .joint_limits(file_path=joint_limits_path)
        .to_moveit_configs()
    )

    task_file = PathJoinSubstitution(
        [FindPackageShare("multi_robot_cell_scene"), "config", "task.yaml"]
    ).perform(context)

    pick_place_node = Node(
        package="multi_robot_cell_scene",
        executable="pick_place",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {"task_file": task_file},
        ],
    )

    return [pick_place_node]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=launch_setup)])
