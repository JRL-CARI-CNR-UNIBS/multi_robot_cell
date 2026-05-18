from launch.launch_description import LaunchDescription
from launch.actions import OpaqueFunction, IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
  launch_args = [
    DeclareLaunchArgument(name="fake", default_value="true", description="use fake hardware"),
    DeclareLaunchArgument(name="robotiq_com_port_robot1", default_value="/dev/robotiq1", description="Gripper com port"),
    DeclareLaunchArgument(name="robotiq_com_port_robot2", default_value="/dev/robotiq2", description="Gripper com port"),
    DeclareLaunchArgument(name="rviz", default_value="true", description="Start RViz"),
  ]
  return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])

def launch_setup(context):
  launch_moveit_path = PathJoinSubstitution([FindPackageShare('multi_robot_cell_bringup'), 'launch', 'multi_robot_moveit.launch.py'])
  launch_moveit_and_robot_description_launch = IncludeLaunchDescription(
    launch_description_source = PythonLaunchDescriptionSource(launch_moveit_path),
    launch_arguments = [('fake', LaunchConfiguration("fake")),
                        ('robotiq_com_port_robot1', LaunchConfiguration("robotiq_com_port_robot1")),
                        ('robotiq_com_port_robot2', LaunchConfiguration("robotiq_com_port_robot2")),
                        ('rviz', LaunchConfiguration("rviz"))]
  )

  launch_controllers_path = PathJoinSubstitution([FindPackageShare('multi_robot_cell_bringup'), 'launch', 'multi_robot_control.launch.py'])
  launch_controllers_launch  = IncludeLaunchDescription(
    launch_description_source = PythonLaunchDescriptionSource(launch_controllers_path),
    launch_arguments = [('fake', LaunchConfiguration("fake")),
                        ('robotiq_com_port_robot1', LaunchConfiguration("robotiq_com_port_robot1")),
                        ('robotiq_com_port_robot2', LaunchConfiguration("robotiq_com_port_robot2"))]
  )

  # motion_server_path = PathJoinSubstitution([FindPackageShare("drims2_description"), "launch", "ur10e_2f", "ur10e_2f_motion_server.launch.py"])
  # motion_server_launch = IncludeLaunchDescription(
  #     launch_description_source = PythonLaunchDescriptionSource(motion_server_path),
  #     launch_arguments = [('fake', LaunchConfiguration("fake"))]
  # )


  return [
    launch_moveit_and_robot_description_launch,
    launch_controllers_launch,
  ]