# Multi Robot Cell

[![jazzy](https://github.com/JRL-CARI-CNR-UNIBS/multi_robot_cell/actions/workflows/jazzy.yml/badge.svg)](https://github.com/JRL-CARI-CNR-UNIBS/multi_robot_cell/actions/workflows/jazzy.yml)

ROS 2 workspace packages for a dual-robot cell composed of two UR10e manipulators, Robotiq 2F-85 grippers, and linear guides. The repository includes the robot description, MoveIt configuration, ros2_control setup, launch files for interactive planning in RViz, and a multi-robot task and motion planning (TAMP) pipeline.

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
  The **multi-robot TAMP pipeline**: given a set of pick-and-place tasks, it decides which robot does each one and when it starts, guarantees the two arms never collide, and runs both arms concurrently — in RViz or on the controllers, with grippers and objects animated. Where `multi_robot_cell_scene` replays a fixed plan on one arm, this works out *who does what, and when*. See [the TAMP pipeline](#multi-robot-tamp-pipeline-multi_robot_cell_tamp) below.

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
> - A change to C++ code seems to have no effect → `colcon build` was probably run from inside a package directory, which builds a separate workspace there. Always build from the workspace root.

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

You describe *what* to move and *where*. The pipeline decides **which robot performs each task and when it starts**, minimising the total time (the makespan) while guaranteeing the two arms never collide, and then runs both arms at once.

### The idea

All the geometry is computed **before** anything is scheduled. Every robot gets a trajectory for every task, sampled on a clock shared by both arms: one sample every Δt = 25 ms. If robot 1 starts a task at slot `a` and robot 2 starts one at slot `b`, sample `k` of the first and sample `ℓ` of the second happen at the same instant exactly when `b − a = k − ℓ`. So each pair of samples that collide forbids exactly one relative start offset. Collision checking reduces to a list of forbidden offsets, and scheduling becomes an integer problem that never sees a pose or a mesh.

### The stages

```
scene YAML → 1 plan → 2 collide → 3 solve → 4 refine → 5 plan graph → 6 execute
             └─────────────── offline, one command ───────────────┘   needs the cell
```

1. **Plan** — `trajectory_generator` (C++, MoveIt/OMPL). Both robots plan every task, because choosing the robot is the solver's job. A trajectory runs home → pick → place → home and is resampled to Δt. Objects whose order relative to the task is not fixed count as obstacles at both their start and their goal, so the motion stays valid in whatever order the solver picks.
2. **Collide** — `collision_generator_vamp.py` (VAMP, about a second) or `collision_generator` (FCL, minutes). Every sample of robot 1 is checked against every sample of robot 2, carried boxes included, and only the forbidden offsets are kept. The output, `tamp_problem.json`, holds durations, forbidden offsets and ordering constraints and no geometry: it is all the solver sees. FCL is the exact reference; VAMP is checked against it and may forbid more offsets, never fewer.
3. **Solve** — `solve.py` from `thesis_material_tamp`, a separate ROS-free library (CP-SAT). One robot and one start slot per task: shortest makespan, one task at a time per robot, ordering constraints kept, no forbidden offset used.
4. **Refine** — `refine_yield.py` plus MoveIt. Going home between two tasks is how an arm gets out of the other's way, so the trip cannot simply be dropped. It can, however, stop at the first pose that is already clear of the other arm and head straight for the next task. The shortened plan goes through stages 2 and 3 again.
5. **Plan graph** — `build_tpg.py`. A timed schedule is collision-free only while both arms keep exact time. The graph replaces the timing with waits: an arm enters a pose only after the other arm has left every pose that collides with it, so a late arm costs time, never a collision.
6. **Execute** — `execute_schedule.launch.py`. Runs the plan on the controllers, opening and closing the grippers and moving the objects in RViz.

### Before the first run

The stages need three Python environments, because OR-Tools is not in the ROS Python and VAMP must stay out of it:

- this workspace, built from its root with `colcon build --symlink-install`;
- `thesis_material_tamp/` cloned next to `src/`, with its `.venv` (see that repository's README);
- `multi_robot_cell_tamp/.venv_vamp` with `vamp-planner` and the `ur10e_rail` robot module ([recipe](multi_robot_cell_tamp/vamp_codegen/README.md)), then `./scripts/build_mu_kernel.sh` once per machine. Without the kernel VAMP falls back to numpy: same result, about 40× slower. This environment is needed even with `engine:=fcl`, because refinement and the plan graph run in it.

### Run it

```bash
# offline stages 1–5; the cell does not need to be running
ros2 launch multi_robot_cell_tamp pipeline.launch.py task:=tower save_as:=tower
```

```bash
ros2 launch multi_robot_cell_bringup start.launch.py                               # terminal 1
ros2 launch multi_robot_cell_tamp execute_schedule.launch.py mode:=tpg run:=tower  # terminal 2
```

Each stage starts only if the previous one succeeded, and all of them run every time: the motion planner is not seeded, so reusing an old result next to new trajectories would mix two different plans. The offline part takes about half a minute on the tower scene. `save_as` stores the matching trajectories, schedule, graph and scene under `artifacts/runs/<name>/`, and `run:=` replays exactly that set.

| `pipeline.launch.py` | default | |
| --- | --- | --- |
| `task` | `tower` | a scene name (below) or a path to a task file |
| `engine` | `vamp` | `vamp` (seconds) or `fcl` (minutes, the exact reference) |
| `refine` | `true` | `false` stops after the first solve |
| `save_as` | — | keep the result under `artifacts/runs/<name>/` |
| `backend` | `cp-sat` | `cp-sat` or `gurobi` |
| `time_limit` | `120.0` | solver time limit, in seconds |

`schedule`, `objective`, `balance_weight`, `solver_seed`, `refine_test` and `refine_rungs` exist for the APEX-MR comparison and the refinement ablation; the launch file documents them.

| `execute_schedule.launch.py` | default | |
| --- | --- | --- |
| `mode` | `rigid` | `rigid` plays both arms on one shared clock: safe only while both controllers keep up. `tpg` moves each arm when the plan graph allows, so a delay costs time instead of safety. |
| `run` | — | replay a plan saved with `save_as` |
| `refined` | `true` | without `run`: the refined plan or the baseline, from the latest pipeline run |
| `task_file` | `tamp_task.yaml` | without `run`: the scene to animate — pass the one you planned |
| `visualize`, `actuate_grippers` | `true` | animate the objects; open and close the grippers |
| `visualize_spheres` | `false` | draw VAMP's collision spheres on the arms |

The executor refuses to move if the trajectories and the schedule come from different runs, or if the plan does not start and end at home.

### Scenes

Every `config/tamp_task_<name>.yaml` can be passed as `task:=<name>`; `nominal` is `tamp_task.yaml`.

| Scene | What it is |
| --- | --- |
| `nominal` | a lid and three boxes; the lid must come off before any box moves |
| `swap` | four boxes, each moving to the opposite end of the table and to the other robot's side; any order |
| `swap_x` | the original `swap`: boxes change end but stay on their side (kept so earlier results reproduce) |
| `tower`, `tower6`, `tower8`, `tower10` | 4–10 boxes restacked into an inverted tower, in a fixed order |
| `cont05`, `cont08`, `cont14`, `cont22` | boxes in two columns at x = ±5…22 cm — the closer to the centre, the more the arms contend; any order |
| `size6`, `size8` | 6 or 8 boxes; any order |
| `seq6`, `seq8`, `seq10`, `seq12` | 6–12 boxes under a total order |
| `precchain`, `precpart` | the `cont14` boxes under a total and a partial order |

To write your own, copy a scene and edit its objects and tasks. A task says what to move and where; precedences say what must happen first:

```yaml
tasks:
  - id: t_box_1
    object: box_1
    place: {x: -0.8, y: -0.15, z: 0.775}

precedences:
  - [t_lid, t_box_1]        # the lid comes off before box_1 moves
```

> If a task fails to plan, suspect a missing precedence first: objects that are not ordered relative to each other are avoided at *both* their start and their goal positions, so a forgotten rule can leave no room to move.

### Where the results go

```
artifacts/
  tamp_trajectories.json            1  planned motions
  tamp_trajectories_refined.json    4  the same, with shorter trips between tasks
  vamp/  (or fcl/)
    tamp_problem.json               2  the solver's input
    tamp_solution.json              3  the schedule
    tpg.json                        5  plan graph of the unrefined plan
    tamp_problem_refined.json       4  stages 2, 3 and 5 again, on the refined motions
    tamp_solution_refined.json
    tpg_refined.json
    tamp_spheres*.npz                  collision spheres for visualize_spheres
  runs/<name>/                         plans kept with save_as
```

Every run overwrites the working files; none of them is committed.

### Tests

```bash
colcon test --packages-select multi_robot_cell_tamp
colcon test-result --verbose
```

### Going deeper

`scripts/inspect_tpg.py` (the plan graph, plain `python3`), `scripts/simulate_tpg.py` (rigid versus graph execution under delays), `scripts/coordinate.py` (the best possible coordination of two fixed paths) and `scripts/vamp_fcl_mu_diff.py` (VAMP against FCL).
