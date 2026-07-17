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
  The **multi-robot TAMP pipeline**: plans one trajectory per (robot, task) pair offline, computes the inter-robot collision matrices, reduces them to forbidden start-offset sets, and replays the solved schedule on both robots concurrently. Where `multi_robot_cell_scene` runs a hard-coded plan on one arm, this decides *which robot does what, and when* — and proves the result collision-free. See [the TAMP pipeline](#multi-robot-tamp-pipeline-multi_robot_cell_tamp) below.

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

Given a scene and a set of pick-and-place tasks, this decides **which robot performs each task and when it starts**, minimising the makespan while guaranteeing the two arms never collide — then executes the result on both robots concurrently.

The method is *decouple-then-allocate*: pay the geometric cost **offline**, then solve a purely discrete allocation problem. The bridge is the **offset identity** — if trajectories are executed rigidly, one sample per time slot, then sample `k` of one and sample `l` of another are simultaneous **iff** `b − a = k − l`. So a collision between samples `k` and `l` forbids *exactly* the relative start offset `d = k − l`. Collecting them gives the forbidden-offset sets, and the online problem never sees geometry at all.

### The four stages

```
tamp_task.yaml
   │  trajectory_generator   plans home→pick→place→home per (robot, task), resamples to a
   │                         uniform Δt shared by both robots            → tamp_trajectories.json
   │  collision_generator    μ[k,l] via FCL, robot1×robot2 only          → tamp_problem.json
   │                                                                       (GEOMETRY-FREE — the seam)
   │  solve.py               CP-SAT → which robot, which start slot      → tamp_solution.json
   │  schedule_executor      one trajectory per robot, shared clock      → the controllers / RViz
```

The two generators need the **built workspace but not the running cell** — each owns its own `PlanningScene` and never talks to `move_group`. Only the executor needs the cell up.

### Run it

```bash
# 1. Trajectories (~30 s). One per (robot, task) pair: 2 robots × 4 tasks = 8.
ros2 launch multi_robot_cell_tamp generate_trajectories.launch.py \
     out_file:=/tmp/tamp_trajectories.json

# 2. Inter-robot collisions → forbidden offsets (~10 min: ~9.2M FCL checks).
ros2 launch multi_robot_cell_tamp generate_collisions.launch.py \
     traj_file:=/tmp/tamp_trajectories.json \
     out_file:=/tmp/tamp_problem.json

# 3. Solve. OR-Tools is NOT in the ROS Python, so this runs in the thesis venv.
cd ../thesis_material_tamp && source .venv/bin/activate
python solve.py /tmp/tamp_problem.json /tmp/tamp_solution.json

# 4. Execute on the live cell — needs `start.launch.py` running in another terminal.
ros2 launch multi_robot_cell_tamp execute_schedule.launch.py \
     traj_file:=/tmp/tamp_trajectories.json \
     solution_file:=/tmp/tamp_solution.json
```

Stages 1–2 are expensive and depend only on the scene, the robot model and `Δt` — **cache their output**. `/tmp` is cleared on reboot; point `out_file` elsewhere if you want the artifacts to survive.

The solver lives in a separate repo (`thesis_material_tamp`) on purpose: it is the thesis's reusable contribution and has **no ROS dependency**. The seam between them is a JSON file, which is a stronger guarantee than an API call — geometry physically cannot appear in it.

### Configuring the task (`config/tamp_task.yaml`)

Unlike `multi_robot_cell_scene/config/task.yaml`, this file is **robot-agnostic**: tasks say *what* to move and *where*, never *who* moves it. Choosing the robot is the scheduler's job.

```yaml
tasks:
  - id: t_box_1
    object: box_1
    place: {x: -0.8, y: -0.15, z: 0.775}

precedences:
  - [t_lid, t_box_1]        # lid off before any box
```

Key parameters:

| Parameter | Meaning |
| --- | --- |
| `discretisation.delta_t` | Seconds per slot (**0.025**). The sampling step — see the warning below. |
| `discretisation.gripper_dwell_slots` | Gripper actuation time, as slots. The arm is frozen but the slots are consumed. |
| `planning.vel_scale` / `acc_scale` | Fed to TOTG. They set how *fast* the arm moves, hence how far it travels per slot. |
| `robots.<name>.home` | The **7-DOF** home (rail + arm). The SRDF's `home_r1` only pins the 6 arm joints. |
| `robots.<name>.arm_group` | 6-DOF group used for the straight-line approach; the 7-DOF group is redundant and KDL stalls solving its Cartesian IK. |

> **`delta_t` is set by soundness, not convenience.** The collision matrices are evaluated *only* at slots, so if a robot link travels further than the thinnest obstacle between two samples, two arms could cross with no sample registering it — and the schedule would be "provably" collision-free while not being so. The generator measures the fastest link's travel per slot and warns when it exceeds the threshold. At `delta_t = 0.1` it was 0.066 m/slot against a 0.02 m tray wall (unsound); `0.025` gives 0.0167 m. **If you change `vel_scale`, the scene, or the object sizes, re-check that number.**

> After editing anything in `config/`, **rebuild** (`colcon build --symlink-install`) — the launch reads the *installed* copy, not the source.

### Precedences carry geometric weight

A trajectory is planned before the schedule exists, so the offline stage cannot know which objects have already moved. It uses a **precedence-pruned worst-case environment**: an object whose task must run *before* this one is at its place pose; one that must run *after* is at its spawn pose; one that is unordered appears at **both**.

This makes the environment order-independent, hence sound for any schedule the solver may return. But it means **a missing precedence yields an *infeasible* plan, not a wrong one** — the lid sits over the boxes at its spawn pose, so without `t_lid → t_box_k` every box task is planned against a phantom lid and fails. If a task fails to plan, suspect this first.

### Reading the output

`tamp_problem.json` is the geometry-free seam — durations, forbidden offsets, precedences. `solve.py` prints the schedule:

```
makespan 2048 slots = 51.20 s (64% of fully-sequential)
  robot2   t_lid    slots [   0,  690]
  robot2   t_box_1  slots [ 690, 1371]
  robot1   t_box_3  slots [ 690, 1519]
  robot2   t_box_2  slots [1371, 2048]
```

Everything scheduling-side is in **slots**; multiply by `delta_t` for seconds. 64% of fully-sequential means the schedule is genuinely parallel — both arms work concurrently from slot 690.

The executor sends **one trajectory per robot**, both carrying the *same* `header.stamp`, so both controllers begin slot 0 at the same wall-clock instant. That shared clock is what makes the offsets mean at runtime what they meant when computed; dispatch skew between the two action calls cannot desynchronise them.

### Current limitations

- The executor moves the **arms only** — grippers do not actuate and objects do not attach in the planning scene. The trajectory artifact carries the phase and object-state data needed for both; they are simply not wired up yet.
- The schedule is verified **self-consistent** (0 offset violations, 0 precedence violations) but has not yet been independently FCL-replayed to prove collision-freedom end to end.
- Both robots must be able to reach **every** task: the online model forces exactly one robot per task with no way to express "robot r cannot do task i".

### Tests

The resampler — the piece the offset identity depends on — is unit-tested without ROS:

```bash
colcon test --packages-select multi_robot_cell_tamp
colcon test-result --verbose
```