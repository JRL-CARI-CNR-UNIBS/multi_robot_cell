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
  The **multi-robot TAMP pipeline**: plans one trajectory per (robot, task) pair offline, computes the inter-robot collision matrices (via **FCL** *or* the SIMD **VAMP** engine), reduces them to forbidden start-offset sets, and replays the solved schedule on both robots concurrently — animating the objects and grippers, and optionally overlaying the VAMP collision spheres, in RViz. Where `multi_robot_cell_scene` runs a hard-coded plan on one arm, this decides *which robot does what, and when* — and proves the result collision-free. See [the TAMP pipeline](#multi-robot-tamp-pipeline-multi_robot_cell_tamp) below.

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
config/tamp_task*.yaml
   │  trajectory_generator      plan home→pick→place→home per (robot, task), resample to a
   │                            uniform Δt shared by both robots      → artifacts/tamp_trajectories.json
   │  collision stage           μ[k,l], robot1×robot2 only, reduced to forbidden offsets
   │    ├ collision_generator    FCL, C++/MoveIt                      → artifacts/fcl/tamp_problem.json
   │    └ collision_generator_vamp  VAMP spheres, Python/SIMD         → artifacts/vamp/tamp_problem.json
   │                                                                    (GEOMETRY-FREE — the seam)
   │  solve.py                  CP-SAT → which robot, which start slot → artifacts/{fcl,vamp}/tamp_solution.json
   │  schedule_executor         one trajectory per robot, shared clock → the controllers / RViz
```

The collision stage has **two interchangeable engines** behind the same geometry-free seam (ADR-0005): **FCL** (the C++/MoveIt reference) and **VAMP** (a SIMD `fk`→sphere-broadcast engine, ~2× faster, verified sound against FCL — see [Collision engines](#collision-engines-fcl-and-vamp)). Both write the *identical* `tamp_problem.json` schema, so the scheduler cannot tell which ran.

The two generators need the **built workspace but not the running cell** — each owns its own `PlanningScene` and never talks to `move_group`. Only the executor needs the cell up. All artifacts default to the persistent, gitignored `multi_robot_cell_tamp/artifacts/` directory (see [Artifacts layout](#artifacts-layout)).

### Run it

All stages default to the persistent, gitignored `multi_robot_cell_tamp/artifacts/`
directory: a **shared** `tamp_trajectories.json`
plus per-pipeline `fcl/` and `vamp/` subfolders. Every path below is a *default* — pass
`out_file:=`/`--out` etc. only to override.

```bash
# 1. Trajectories (~2 s). Shared by both pipelines -> artifacts/tamp_trajectories.json.
ros2 launch multi_robot_cell_tamp generate_trajectories.launch.py

# 2. Inter-robot collisions -> forbidden offsets. Pick ONE engine:
#    FCL (~10 min, ~9.2M checks)     -> artifacts/fcl/tamp_problem.json
ros2 launch multi_robot_cell_tamp generate_collisions.launch.py
#    VAMP (~4 min, sound vs FCL)     -> artifacts/vamp/tamp_problem.json   (needs .venv_vamp)
cd multi_robot_cell_tamp && .venv_vamp/bin/python scripts/collision_generator_vamp.py \
     --robot ur10e_rail --base-layout ur10e_rail

# 3. Solve. OR-Tools is NOT in the ROS Python, so this runs in the thesis venv.
A=$PWD/multi_robot_cell_tamp/artifacts        # from the multi_robot_cell repo root
cd ../thesis_material_tamp
.venv/bin/python solve.py "$A/vamp/tamp_problem.json" "$A/vamp/tamp_solution.json"   # or fcl/
cd ../src/multi_robot_cell

# 4. Execute on the live cell — needs `start.launch.py` running in another terminal.
#    Defaults replay the VAMP schedule (artifacts/vamp/tamp_solution.json); for FCL pass
#    solution_file:=<pkg>/artifacts/fcl/tamp_solution.json.
ros2 launch multi_robot_cell_tamp execute_schedule.launch.py
```

**Optional — VAMP collision-sphere overlay (viz-only).** To *watch* the exact
sphere-shells the VAMP engine (ADR-0005 Phase 2) checks ride along with both arms
during the replay, first pre-compute them once (needs `.venv_vamp`; `vamp` is never
imported by the ROS runtime). Defaults read/write `artifacts/`:

```bash
# viz prep -> artifacts/vamp/tamp_spheres.npz
cd multi_robot_cell_tamp && .venv_vamp/bin/python scripts/dump_vamp_spheres.py 
```

then add `visualize_spheres:=true` to step 4 (opt-in, default false):

```bash
ros2 launch multi_robot_cell_tamp execute_schedule.launch.py visualize_spheres:=true
```

Add a **MarkerArray** display on `/vamp_collision_spheres` in RViz. robot1 and
robot2 get distinct colours; the carried-object shell is highlighted and appears
only while grasped, and the overlay node self-terminates once the schedule
finishes. The `.npz` is a *visualisation* artifact (it carries geometry) and is
**not** the geometry-free seam — it never touches `tamp_scheduler` (ADR-0002).

> The sphere `.npz` is **per-scene** — regenerate it (`dump_vamp_spheres.py --task <the scene's yaml>`) whenever you change `task_file`, exactly like the trajectories/problem/solution. If it is stale the overlay **fails loudly** and tells you the command to run, rather than silently showing the wrong arms.

### Artifacts layout

Every stage reads/writes `multi_robot_cell_tamp/artifacts/` by default — persistent across reboots and **git-ignored** (only the empty folder skeleton is tracked):

```
artifacts/
  tamp_trajectories.json          # shared by BOTH engines (stage 1)
  fcl/  tamp_problem.json  tamp_solution.json          # FCL pipeline (stages 2–3)
  vamp/ tamp_problem.json  tamp_solution.json  tamp_spheres.npz   # VAMP pipeline (+ overlay)
```

The path is resolved from the package **source** tree (via `realpath` of the launch/script file, which the `--symlink-install` build resolves back to source), so outputs land next to the package, not in `install/`. Override any path with `out_file:=` / `--out` / `solution_file:=`. The committed reference seams for the FCL↔VAMP diff live separately in `config/tamp_problem.{fcl_reference,vamp}.json`.

### Collision engines: FCL and VAMP

Stage 2 has two engines behind the same seam (ADR-0005). Both compute `μ[k,l]` for **robot1 × robot2 only** (plus each robot's carried object) and reduce it to forbidden offsets.

| | **FCL** (`collision_generator`, C++) | **VAMP** (`collision_generator_vamp.py`, Python) |
| --- | --- | --- |
| Method | MoveIt/FCL on the full robot model, ACM masked to cross-robot pairs | `vamp.ur10e_rail.fk(q)` → world-frame spheres → numpy **broadcast** between the two arms (SIMD) |
| Role | the **reference** (same collision model as the planner) | a **measured benchmark / contribution** |
| Cost (nominal scene) | ~8 min, ~9.2M checks | **~4 min** μ (FK ~0.5 s) |
| vs FCL | — | **sound: 0 missing** offsets; +~20 % extra (conservative); **~1.9×** faster |
| Runs in | the ROS build | its own `.venv_vamp` (no `vamp` import ever reaches the ROS runtime or `tamp_scheduler`) |

**Using VAMP.** It needs a one-time build of `vamp-planner` + a codegen'd `vamp.ur10e_rail` robot module; the recipe and generated header live in `multi_robot_cell_tamp/vamp_codegen/`, and the venv in `multi_robot_cell_tamp/.venv_vamp`. Then:

```bash
cd multi_robot_cell_tamp
.venv_vamp/bin/python scripts/collision_generator_vamp.py --robot ur10e_rail --base-layout ur10e_rail
```

`--robot ur10e_rail --base-layout ur10e_rail` are **mandatory** (the defaults are a geometry-mismatched `ur5` plumbing stand-in). A 2 cm `sphere_margin` (default for `ur10e_rail`) restores soundness where foam under-covers the arm mesh. BLAS threads are pinned internally so the parallel run can't oversubscribe the CPU.

**Diffing the two engines** (the soundness certificate):

```bash
.venv_vamp/bin/python scripts/vamp_fcl_mu_diff.py <fcl>/tamp_problem.json <vamp>/tamp_problem.json
# reports EXTRA (VAMP false-positives, expected) and MISSING (unsound — must be 0)
```

### Timing

Measured for the reference `tamp_task.yaml` scene (2 robots, 4 tasks, `Δt = 0.025`) on a 4-core / 7.7 GB WSL2 box:

| Stage | Time | Dominated by |
| --- | --- | --- |
| 1. Trajectory generation | **~2 s** | 8 OMPL plans + TOTG + resample + per-sample re-validation |
| 2. Collision analysis — **FCL** | **~8 min** | ~9.2M FCL config-pair checks |
| 2. Collision analysis — **VAMP** | **~4 min** (FK ~0.5 s) | numpy sphere-broadcast, 4 workers — ~1.9× faster, sound vs FCL |
| 3. Solve (CP-SAT) | **~1 s** | ~7.6k forbidden offsets → OPTIMAL |
| 4. Execution | **~43 s (FCL) / ~45 s (VAMP)** | the makespan itself — real robot motion (VAMP slightly higher: +20 % conservative offsets) |

**Stage 2 dominates the offline cost; everything else is noise.** So caching is really about stage 2 — it depends only on the scene, the robot model and `Δt`, and re-running it is the entire cost of an iteration. Stages 1 and 3 are cheap enough to just redo. This is exactly why the VAMP engine (~2× on stage 2) is worth having.

Stage 2 scales as `K² × tasks²`, so `Δt` hits it **quadratically**: moving from 0.1 s to the sound 0.025 s multiplied the workload by ~16. That is the real price of the discretisation.

Outputs persist in `artifacts/` across reboots (see [Artifacts layout](#artifacts-layout)), so a cached stage 2 survives a restart.

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

### Scenes

Three task files ship in `config/`, selected with `task_file:=<abs path>` on **every** stage (①②④) — keep it the *same* across stages so trajectories, collisions, and the rendered scene agree. The VAMP engine and sphere prep take the matching `--task <yaml>`.

| Scene | What it is | Exercises |
| --- | --- | --- |
| `tamp_task.yaml` (default) | lid + 3 boxes; `t_lid` precedes every box | the nominal end-to-end pipeline |
| `tamp_task_swap.yaml` | 4 free boxes crossing to the opposite table half; **no precedences** | the **anti-collision core** — both arms cross concurrently |
| `tamp_task_tower.yaml` | 4 boxes restacked into an inverted tower; full precedence chain | the **precedence machinery** + stacked geometry (near-serial) |

### Precedences carry geometric weight

A trajectory is planned before the schedule exists, so the offline stage cannot know which objects have already moved. It uses a **precedence-pruned worst-case environment**: an object whose task must run *before* this one is at its place pose; one that must run *after* is at its spawn pose; one that is unordered appears at **both**.

This makes the environment order-independent, hence sound for any schedule the solver may return. But it means **a missing precedence yields an *infeasible* plan, not a wrong one** — the lid sits over the boxes at its spawn pose, so without `t_lid → t_box_k` every box task is planned against a phantom lid and fails. If a task fails to plan, suspect this first.

### Reading the output

`tamp_problem.json` is the geometry-free seam — durations, forbidden offsets, precedences, and the pick/place milestone offsets. `solve.py` prints the schedule (FCL, nominal scene):

```
makespan 1717 slots = 42.93 s (53% of fully-sequential)
  robot1   t_lid    slots [   0,  813]
  robot2   t_box_2  slots [ 293,  970]
  robot1   t_box_3  slots [ 888, 1717]
  robot2   t_box_1  slots [ 970, 1651]
```

Everything scheduling-side is in **slots**; multiply by `delta_t` for seconds. 53% of fully-sequential means the schedule is genuinely parallel — both arms work concurrently (robot2 starts at slot 293, while robot1 is still on `t_lid`). The VAMP seam solves to a very slightly higher makespan (~45 s) — the price of its +20 % conservative offsets.

The executor sends **one trajectory per robot**, both carrying the *same* `header.stamp`, so both controllers begin slot 0 at the same wall-clock instant. That shared clock is what makes the offsets mean at runtime what they meant when computed; dispatch skew between the two action calls cannot desynchronise them.

### Execution: scene + grippers

The executor commands the **arms** on the shared clock and, on the same clock, drives the *world* to match: `scene_visualizer` publishes the tray/lid/boxes and attaches each object to the picking gripper at the correct grasp pose (from the object's `grasp` in the task YAML) then lands it at its place pose, and `gripper_commander` opens/closes the fingers — both keyed to each task's `GripClose`/`GripOpen` dwell. Toggle with `visualize:=` and `actuate_grippers:=` (both default `true`). So RViz shows the objects being carried, not just the arms waving.

### Current limitations

- The schedule is verified **self-consistent** (0 offset violations, 0 precedence violations) but has not yet been independently FCL/VAMP-replayed to prove collision-freedom of the *concurrent* motion end to end.
- Both robots must be able to reach **every** task: the online model forces exactly one robot per task with no way to express "robot r cannot do task i".
- Each task is a full pick-and-place returning **home** (ADR-0004); back-to-back tasks round-trip home, which the makespan pays for.

### Tests

The resampler — the piece the offset identity depends on — is unit-tested without ROS:

```bash
colcon test --packages-select multi_robot_cell_tamp
colcon test-result --verbose
```