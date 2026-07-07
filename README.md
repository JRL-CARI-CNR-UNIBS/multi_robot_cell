# Multi Robot Cell

[![jazzy](https://github.com/JRL-CARI-CNR-UNIBS/multi_robot_cell/actions/workflows/jazzy.yml/badge.svg)](https://github.com/JRL-CARI-CNR-UNIBS/multi_robot_cell/actions/workflows/jazzy.yml)

ROS 2 workspace packages for a dual-robot cell composed of two UR10e manipulators, Robotiq 2F-85 grippers, and linear guides. The repository includes the robot description, MoveIt configuration, ros2_control setup, and launch files for interactive planning in RViz.

## Packages

- `multi_robot_cell_description`  
  URDF/Xacro description of the multi-robot cell, including the UR10e robots, Robotiq grippers, linear guides, and environment geometry.

- `multi_robot_moveit_config`  
  MoveIt configuration for planning groups, kinematics, joint limits, OMPL planning, and semantic robot information.

- `multi_robot_cell_bringup`  
  Launch files and controller configuration for starting MoveIt, RViz, `ros2_control`, robot state publishing, and trajectory controllers.

## Installation

Create a ROS 2 Jazzy workspace and clone this repository:

```bash
mkdir -p ~/jazzy_ws/src
cd ~/jazzy_ws/src
git clone https://github.com/JRL-CARI-CNR-UNIBS/multi_robot_cell.git
cd ~/jazzy_ws
```

Import external dependencies listed in dependencies.repos:

```bash
vcs import src < src/multi_robot_cell/dependencies.repos
```

`dependencies.repos` pulls in `ros2_robotiq_gripper`, which itself depends on the `serial` library that is not available as a system/rosdep package. Import its nested `.repos` file as well:

```bash
vcs import src < src/ros2_robotiq_gripper/ros2_robotiq_gripper-not-released.rolling.repos
```

If `rosdep` is not installed yet:

```bash
sudo apt install python3-rosdep
sudo rosdep init   # only needed once per machine, skip if already initialized
```

Update and install ROS dependencies:

```bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

Build the workspace:

```bash
colcon build --symlink-install
source install/setup.bash
```

> **Troubleshooting**
> - `rosdep: command not found` → install it with `sudo apt install python3-rosdep` (the older `python3-rosdep2` package name is obsolete on newer Ubuntu/ROS releases).
> - `rosdep update` fails with `HTTP Error 429: Too Many Requests` → this is a transient GitHub rate limit, just retry after a short wait.
> - CMake error `Could not find a package configuration file provided by "serial"` during `colcon build` → the `serial` dependency was not imported; run the `vcs import` step above for `ros2_robotiq_gripper-not-released.rolling.repos`, then re-run `rosdep install` and `colcon build`.

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
* `robot_system`

Use:

* `manipulator1` or `manipulator2` to plan with the 6-DOF arm only.
* `manipulator1_on_rail` or `manipulator2_on_rail` to plan with the linear guide plus the arm.
* `robot_system` to plan for the entire robot system.

> **Note**  
> The active `ros2_control` controller must be able to command all joints of the selected MoveIt planning group.
>
> For example, when planning with `manipulator1`, the active trajectory controller must command the six UR joints of `manipulator1`.
>
> When planning with `manipulator1_on_rail`, the active trajectory controller must command both the six UR joints and the linear guide joint belonging to that group.
>
> Controllers that claim overlapping command interfaces cannot be active at the same time. Switch controllers according to the planning group you want to execute.

### Controller commands

List active controllers:

```bash
ros2 control list_controllers
```

Switch controllers:

```bash
ros2 control switch_controllers \
  --deactivate <current_controller_name> \
  --activate <target_controller_name>
```

If the controller manager is namespaced, specify it explicitly:

```bash
ros2 control switch_controllers \
  -c /<namespace>/controller_manager \
  --deactivate <current_controller_name> \
  --activate <target_controller_name>
```