# Multi Robot Cell

ROS 2 workspace packages for a dual-robot cell composed of two UR10e manipulators, Robotiq 2F-85 grippers, and linear guides. The repository includes the robot description, MoveIt configuration, ros2_control setup, and launch files for interactive planning in RViz.

## Packages

- `multi_robot_cell_description`  
  URDF/Xacro description of the multi-robot cell, including the UR10e robots, Robotiq grippers, linear guides, and environment geometry.

- `multi_robot_moveit_config`  
  MoveIt configuration for planning groups, kinematics, joint limits, OMPL planning, and semantic robot information.

- `multi_robot_cell_bringup`  
  Launch files and controller configuration for starting MoveIt, RViz, `ros2_control`, robot state publishing, and trajectory controllers.

## Installation

```bash
mkdir -p ~/jazzy_ws/src
cd ~/jazzy_ws/src
git clone https://github.com/JRL-CARI-CNR-UNIBS/multi_robot_cell.git
cd ~/jazzy_ws
````

Install dependencies:

```bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

Build:

```bash
colcon build --symlink-install
source install/setup.bash
```

## Launch

### Start the full system 

* `move_group`
* RViz
* `ros2_control_node`
* `robot_state_publisher`
* joint state broadcaster
* trajectory controllers
* gripper controllers

```bash
ros2 launch multi_robot_cell_bringup start.launch.py
```

## Planning in RViz and Moveit Configuration

After launching, RViz opens with the MoveIt MotionPlanning plugin.

Available planning groups include:

* `manipulator1`
* `manipulator2`
* `manipulator1_on_rail`
* `manipulator2_on_rail`

Use:

* `manipulator1` or `manipulator2` to plan with the 6-DOF arm only.
* `manipulator1_on_rail` or `manipulator2_on_rail` to plan with the linear guide plus the arm.
* `robot_system` to plan for the entire robot system.


## Useful commands

List active controllers:

```bash
ros2 control list_controllers
```

Check available MoveIt actions:

```bash
ros2 action list -t
```

Inspect joint states:

```bash
ros2 topic echo /joint_states
```

Inspect the planned trajectory shown by MoveIt:

```bash
ros2 topic echo /display_planned_path
```

Inspect controller state:

```bash
ros2 topic echo /robot1_linear_guide_joint_trajectory_controller/controller_state
```

or, when using the arm-only controller:

```bash
ros2 topic echo /robot1_joint_trajectory_controller/controller_state
```