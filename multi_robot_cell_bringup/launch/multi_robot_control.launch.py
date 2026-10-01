from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution, Command, FindExecutable, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)

import os
import yaml

# cell:=dual (default, the original two-robot cell) | fabricator4 (four robots, layout YAML).
CELLS = ("dual", "fabricator4")


def cell_robots(cell):
  """[(robot, has_gripper)] of a cell. fabricator4 reads its layout YAML, the single source
  of which robots exist and which tool each carries."""
  if cell == "dual":
    return [("robot1", True), ("robot2", True)]
  layout_path = os.path.join(FindPackageShare("multi_robot_cell_description").find("multi_robot_cell_description"),
                             "urdf", "fabricator4_layout.yaml")
  with open(layout_path) as f:
    layout = yaml.safe_load(f)
  return [(name, spec["tool"] == "gripper") for name, spec in layout["robots"].items()]

ACTIVE_CONTROLLERS = [
    "linear_guide_joint_trajectory_controller",
    "gripper_action_controller",
]

INACTIVE_CONTROLLERS = [
   "joint_trajectory_controller",
   "scaled_joint_trajectory_controller",
]

INACTIVE_WHOLE_SYSTEM_CONTROLLERS = [
    "joint_trajectory_controller",
    "robot_gripper_action_controller",
]

def launch_setup(context, *args, **kwargs):

  cell = LaunchConfiguration("cell").perform(context)
  if cell not in CELLS:
    raise RuntimeError(f"cell:={cell} is not one of {CELLS}")

  if cell == "dual":
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        " ", 
        PathJoinSubstitution([
            FindPackageShare("multi_robot_cell_description"), 
            "urdf", "multi_robot_cell.urdf.xacro"
        ]),
        " ", 
        "use_fake_hardware:=", LaunchConfiguration("fake"),
        " ",
        "robotiq_com_port_robot1:=", LaunchConfiguration("robotiq_com_port_robot1"),
        " ",
        "robotiq_com_port_robot2:=", LaunchConfiguration("robotiq_com_port_robot2"),

    ])
  else:
    # Only robot1 (the handler) has a gripper, hence a com port.
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("multi_robot_cell_description"),
            "urdf", "fabricator4_cell.urdf.xacro"
        ]),
        " ",
        "use_fake_hardware:=", LaunchConfiguration("fake"),
        " ",
        "robotiq_com_port_robot1:=", LaunchConfiguration("robotiq_com_port_robot1"),
    ])
  robot_description = {
      'robot_description': ParameterValue(robot_description_content, value_type=str)
  }
  
  controller_manager_config = PathJoinSubstitution([FindPackageShare("multi_robot_cell_bringup"),
    "config", "controller_manager.yaml"])

  controller_manager_node = Node(
    package="controller_manager",
    executable="ros2_control_node",
    parameters=[controller_manager_config, 
                robot_description],
    output="screen",
  )

  robot_controller_nodes = []

  for robot, has_gripper in cell_robots(cell):
    for controller in ACTIVE_CONTROLLERS:
        if controller == "gripper_action_controller" and not has_gripper:
            continue
        controller_config_file = PathJoinSubstitution([
            FindPackageShare("multi_robot_cell_bringup"),
            "config",
            f"{robot}_controllers.yaml"
        ])

        robot_controller_nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    f"{robot}_{controller}",
                    "--controller-manager",
                    "/controller_manager",
                    "-p",
                    controller_config_file,
                ],
                output="screen",
            )
        )
    for controller in INACTIVE_CONTROLLERS:
        controller_config_file = PathJoinSubstitution([
            FindPackageShare("multi_robot_cell_bringup"),
            "config",
            f"{robot}_controllers.yaml"
        ])

        robot_controller_nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    f"{robot}_{controller}",
                    "--controller-manager",
                    "/controller_manager",
                    "-p",
                    controller_config_file,
                    "--inactive",
                ],
                output="screen",
            )
        )

  # for controller in INACTIVE_WHOLE_SYSTEM_CONTROLLERS:
  #       controller_config_file = PathJoinSubstitution([
  #           FindPackageShare("multi_robot_cell_bringup"),
  #           "config",
  #           f"whole_system_controllers.yaml"
  #       ])

  #       robot_controller_nodes.append(
  #           Node(
  #               package="controller_manager",
  #               executable="spawner",
  #               arguments=[
  #                   "controller",
  #                   "--controller-manager",
  #                   "/controller_manager",
  #                   "-p",
  #                   controller_config_file,
  #               ],
  #               output="screen",
  #           )
  #       )


  robot_state_publisher_node = Node(
    package="robot_state_publisher",
    executable="robot_state_publisher",
    output="screen",
    parameters=[robot_description]
  )

  joint_state_broadcaster_node = Node(
    package="controller_manager",
    executable="spawner",
    output="screen",
    arguments=[
      'joint_state_broadcaster',
      '--controller-manager',
      '/controller_manager',          
    ]
  )

  what_to_launch = [
    controller_manager_node,
    robot_state_publisher_node,
    joint_state_broadcaster_node,
    *robot_controller_nodes,
    ]

  return what_to_launch

def generate_launch_description():
  launch_args = []
  launch_args.append(DeclareLaunchArgument(name="cell", default_value="dual", choices=list(CELLS),
                                           description="Cell variant: dual (2 robots) or fabricator4 (4 robots, fabricator4_layout.yaml)"))
  launch_args.append(DeclareLaunchArgument(name="fake", default_value="true", description="use fake hardware"))
  launch_args.append(DeclareLaunchArgument(name="robotiq_com_port_robot1", default_value="/dev/robotiq1", description="Gripper com port"))
  launch_args.append(DeclareLaunchArgument(name="robotiq_com_port_robot2", default_value="/dev/robotiq2", description="Gripper com port"))

  ld = LaunchDescription(launch_args+[OpaqueFunction(function=launch_setup)])
    
  return ld

  