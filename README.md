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

- `multi_robot_cell_scene`  
  A YAML-driven pick-and-place demo for robot1: spawns fixtures/objects into the planning scene and executes an ordered pick-and-place task via the `move_group` C++ interface. Single robot, fixed order — the **pre-TAMP baseline**.

- `multi_robot_cell_tamp`  
  The **multi-robot TAMP pipeline**: given a set of pick-and-place tasks, it decides which robot does each one and when it starts, then runs both arms concurrently in RViz or on the controllers — objects and grippers animated along with them. Where `multi_robot_cell_scene` replays a fixed plan on one arm, this works out *who does what, and when*, and guarantees the arms never collide. See [the TAMP pipeline](#multi-robot-tamp-pipeline-multi_robot_cell_tamp) below.

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

## Pick-and-place demo (`multi_robot_cell_scene`)

`multi_robot_cell_scene` runs an automated pick-and-place task for `robot1`, driven entirely by [config/task.yaml](multi_robot_cell_scene/config/task.yaml). For each entry in `task_plan` it plans and executes: pre-grasp → approach → close gripper → attach → retreat → pre-place → lower → open gripper → detach → retreat.

### Prerequisites

1. The full system must already be running (`move_group`, `ros2_control`, and the gripper action server):

   ```bash
   ros2 launch multi_robot_cell_bringup start.launch.py
   ```

2. `task.yaml` plans for `manipulator1_on_rail`, so the active trajectory controller must command both `robot1`'s linear guide and its arm joints. **The default already does** — `start.launch.py` activates `robot1_linear_guide_joint_trajectory_controller` (7 DOF: rail + arm) plus the gripper controller, and spawns the 6-DOF and scaled variants *inactive*. No switch is needed. To confirm:

   ```bash
   ros2 control list_controllers
   ```

   You should see `robot1_linear_guide_joint_trajectory_controller ... active`. Only switch if you deliberately want a different group — and note that controllers claiming overlapping joints cannot both be active, so activating `robot1_scaled_joint_trajectory_controller` requires deactivating the linear-guide one in the same call.

### Run the demo

In a new terminal (with the workspace sourced):

```bash
ros2 launch multi_robot_cell_scene pick_place.launch.py
```

This spawns the tray fixtures and objects from `task.yaml` into the MoveIt planning scene, then runs the task plan step by step. Watch RViz to follow the motion; the node logs each planning/execution stage to the terminal.

To change what gets picked/placed, edit `multi_robot_cell_scene/config/task.yaml` (object poses, grasp offsets, or the `task_plan` order) and re-run the launch file — no rebuild is required since it's read at runtime.

### Standalone scene demo (`spawn_object`)

`spawn_object` is a minimal example node that spawns a single 5 cm collision box onto the table via the planning scene interface. It only needs `move_group` running (not the full task setup):

```bash
ros2 run multi_robot_cell_scene spawn_object
```

## Multi-robot TAMP pipeline (`multi_robot_cell_tamp`)

Given a scene and a list of pick-and-place tasks, this decides **which robot performs each task and when it starts** — minimising the total time while guaranteeing the two arms never collide — and then runs the result on both robots at once.

You describe *what* to move and *where*. Choosing who does it, and in what order, is the pipeline's job.

### How it works

```
config/tamp_task*.yaml          the scene: objects, tasks, precedences
        │
        │  1. trajectories      plan home → pick → place → home, for every (robot, task)
        │  2. collisions        find which relative start times would make the arms collide
        │  3. solve             pick the assignment and start times with the shortest makespan
        │  4. execute           replay both robots on a shared clock
        ▼
   both arms moving concurrently in RViz / on the controllers
```

Stages 1–3 are **offline**: they need the built workspace but not the running cell, and their
results are cached in `artifacts/`. Only stage 4 needs the cell up.

### Run it

Stages 1–3 are a single command:

```bash
ros2 launch multi_robot_cell_tamp pipeline.launch.py task:=tower
```

| argument | default | |
| --- | --- | --- |
| `task` | `tower` | which scene — `tower`, `swap`, `nominal`, or a path to your own task file |
| `engine` | `vamp` | how collisions are checked: `vamp` (seconds) or `fcl` (minutes, the reference) |
| `backend` | `cp-sat` | solver to use — `cp-sat` or `gurobi` |
| `time_limit` | `120.0` | how long the solver may take, in seconds |

It runs the three stages in order, stops immediately if one fails, and prints the execute
command for you when it finishes. The tower scene takes about **6 seconds** end to end.

Then, to watch it run — with the cell started in another terminal:

```bash
ros2 launch multi_robot_cell_bringup start.launch.py          # terminal 1
```
```bash
ros2 launch multi_robot_cell_tamp execute_schedule.launch.py \
    task_file:=<pkg>/config/tamp_task_tower.yaml \
    solution_file:=<pkg>/artifacts/vamp/tamp_solution.json    # terminal 2
```

Execution is kept separate on purpose: it needs the cell running and it is something you
watch, rather than a step that finishes on its own.

### Scenes

Three scenes ship in `config/`. Pass the one you want as `task:=`.

| Scene | What it is |
| --- | --- |
| `nominal` | a lid plus three boxes; the lid must come off before any box moves |
| `swap` | four boxes crossing to the opposite side of the table, in any order |
| `tower` | four boxes restacked into an inverted tower, in a fixed order |

To use your own, copy one of the files and pass its path. A task says what to move and
where to put it, plus any ordering constraints:

```yaml
tasks:
  - id: t_box_1
    object: box_1
    place: {x: -0.8, y: -0.15, z: 0.775}

precedences:
  - [t_lid, t_box_1]        # the lid comes off before box_1 moves
```

> If a task fails to plan, a missing precedence is the first thing to suspect: objects that
> are not ordered relative to each other are avoided at *both* their start and end positions,
> so a forgotten rule can leave no room to move.

### Where the results go

```
artifacts/
  tamp_trajectories.json     the planned motions       (stage 1)
  vamp/ or fcl/
    tamp_problem.json        what the solver is given  (stage 2)
    tamp_solution.json       the schedule              (stage 3)
```

These persist across reboots and are not committed. Re-running the pipeline overwrites them.

### Tests

```bash
colcon test --packages-select multi_robot_cell_tamp
colcon test-result --verbose
```

### Going deeper

`CONTEXT.md` at the workspace root explains the method and the design decisions behind it;
`docs/adr/` records why each was made.
