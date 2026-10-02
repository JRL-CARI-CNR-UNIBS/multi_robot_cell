from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder
from launch.conditions import IfCondition
import yaml

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(name="cell", default_value="dual", choices=["dual", "fabricator4"],
                              description="Cell variant: dual (2 robots) or fabricator4 (4 robots)"),
        DeclareLaunchArgument(name="fake", default_value="true", description="Use fake hardware"),
        DeclareLaunchArgument(name="rviz", default_value="true", description="Start RViz"),
        OpaqueFunction(function=launch_setup)
    ])

def launch_setup(context):
    if LaunchConfiguration("cell").perform(context) == "fabricator4":
        return fabricator4_setup(context)

    srdf_path = PathJoinSubstitution([FindPackageShare("multi_robot_moveit_config"),"config","multi_robot_cell.srdf"]).perform(context)
    joint_limits_path = PathJoinSubstitution([FindPackageShare("multi_robot_moveit_config"),"config","joint_limits.yaml"]).perform(context)
    moveit_controllers_path = PathJoinSubstitution([FindPackageShare("multi_robot_cell_bringup"),"config","moveit_controllers.yaml"]).perform(context)

    moveit_config = (
        MoveItConfigsBuilder("manipulator", package_name="multi_robot_moveit_config")
        .robot_description_semantic(file_path=srdf_path)
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
            publish_planning_scene=True
        )
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
        .joint_limits(file_path=joint_limits_path)
        .trajectory_execution(file_path=moveit_controllers_path)
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    rviz_config = PathJoinSubstitution([FindPackageShare("multi_robot_cell_bringup"),"rviz","rviz_config.rviz"]).perform(context)

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["--display-config", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("rviz"))
    )

    return [
      rviz_node,
      move_group_node
    ]

def fabricator4_setup(context):
    """cell:=fabricator4 -- MoveIt files from four_robot_cell/config/ (the SRDF is a xacro generated
    from four_robot_cell/urdf/fabricator4_layout.yaml), the controller map included.

    The URDF is the DESCRIPTION's cell, exactly what the dual branch gets implicitly (the
    moveit_config's .setup_assistant points MoveItConfigsBuilder at
    multi_robot_cell_description/urdf/multi_robot_cell.urdf.xacro). move_group republishes it on
    /robot_description, which ros2_control_node also listens to, so it must be the same model
    robot_state_publisher publishes: the moveit_config wrapper (extra FakeSystem over the same
    joints) makes the controller manager fail with duplicate command interfaces."""
    cfg = FindPackageShare("four_robot_cell").perform(context) + "/config"
    urdf = PathJoinSubstitution([FindPackageShare("four_robot_cell"), "urdf", "fabricator4_cell.urdf.xacro"]).perform(context)
    moveit_controllers_path = f"{cfg}/fabricator4_moveit_controllers.yaml"

    moveit_config = (
        MoveItConfigsBuilder("manipulator", package_name="four_robot_cell")
        .robot_description(file_path=urdf, mappings={
            "use_fake_hardware": LaunchConfiguration("fake").perform(context),
            "robotiq_com_port_robot1": LaunchConfiguration("robotiq_com_port_robot1", default="/dev/robotiq1").perform(context),
        })
        .robot_description_semantic(file_path=f"{cfg}/fabricator4_cell.srdf.xacro")
        .robot_description_kinematics(file_path=f"{cfg}/kinematics.yaml")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
            publish_planning_scene=True
        )
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
        .joint_limits(file_path=f"{cfg}/joint_limits.yaml")
        .trajectory_execution(file_path=moveit_controllers_path)
        .to_moveit_configs()
    )
    # planning_pipelines() only looks in config/<pipeline>_planning.yaml: swap in the
    # fabricator4 OMPL file (same planner_configs, per-robot group entries).
    with open(f"{cfg}/ompl_planning.yaml") as f:
        moveit_config.planning_pipelines["ompl"] = yaml.safe_load(f)

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    rviz_config = PathJoinSubstitution([FindPackageShare("multi_robot_cell_bringup"),"rviz","rviz_config.rviz"]).perform(context)

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["--display-config", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("rviz"))
    )

    return [
      rviz_node,
      move_group_node
    ]
