# tamp_core — the multi-robot TAMP pipeline

ROS package `multi_robot_cell_tamp`. It works for any cell: the cell comes from the scene's `cell:` key and is looked up in `<cell>/cell.yaml` next to this folder (`scripts/cell_registry.py`). Setup of the workspace and the Python environments: [top-level README](../README.md#workspace-setup).

## The idea

All geometry is computed **before** anything is scheduled. Every robot gets a trajectory for every task it can reach, sampled on a clock shared by all arms: one sample every Δt = 25 ms. If robot A starts a task at slot `a` and robot B starts one at slot `b`, sample `k` of the first and sample `ℓ` of the second happen at the same instant exactly when `b − a = k − ℓ`. So each colliding pair of samples forbids exactly one relative start offset. Collision checking becomes a list of forbidden offsets, and scheduling becomes an integer problem that never sees a pose or a mesh.

## The stages

```
scene YAML → 1 plan → 2 collide → 3 solve → 4 refine → 5 plan graph → 6 execute
             └──────────── pipeline.launch.py, offline ─────────────┘   execute_schedule.launch.py, needs the cell
```

1. **Plan**: `trajectory_generator` (C++, MoveIt/OMPL). Every robot plans every task, since choosing the robot is the solver's job. A trajectory runs home → pick → place → home and is resampled to Δt. Objects whose order relative to the task is not fixed count as obstacles at both their start and their goal, so the motion stays valid in whatever order the solver picks.
2. **Collide**: `collision_generator_vamp.py` (VAMP, about a second) or `collision_generator` (FCL, minutes). Every sample of each robot is checked against every sample of every other robot, carried objects included, and only the forbidden offsets are kept. The output, `tamp_problem.json`, holds durations, forbidden offsets and ordering constraints and no geometry. It is all the solver sees. FCL is the exact reference; VAMP is checked against it and may forbid more offsets, never fewer.
3. **Solve**: `solve.py` from `thesis_material_tamp` (CP-SAT, or Gurobi). One robot and one start slot per task: shortest makespan, one task at a time per robot, ordering constraints kept, no forbidden offset used.
4. **Refine** (two robots only): `refine_yield.py` plus MoveIt. Between two tasks an arm returns home, which is how it gets out of the other arm's way. Refinement stops it at the first pose already clear of the other arm and sends it straight to its next task. The shortened plan goes through stages 2 and 3 again.
5. **Plan graph**: `build_tpg.py`. A timed schedule is collision-free only while all arms keep exact time. The graph replaces timing with waits: an arm enters a pose only after the other arms have left every pose that collides with it, so a late arm costs time, never a collision.
6. **Execute**: `execute_schedule.launch.py`. Runs the plan on the controllers, opens and closes the grippers, and moves the objects in RViz.

Each stage starts only if the previous one succeeded, and every stage runs every time. The motion planner is not seeded, so reusing an old result next to new trajectories would mix two different plans.

## Run it

```bash
# offline stages 1–5; the cell does not need to be running
ros2 launch multi_robot_cell_tamp pipeline.launch.py task:=tower save_as:=tower

# execution: bring the cell up, then replay the saved plan
ros2 launch multi_robot_cell_bringup start.launch.py                               # terminal 1
ros2 launch multi_robot_cell_tamp execute_schedule.launch.py mode:=tpg run:=tower  # terminal 2
```

At the end the pipeline prints the exact replay command. Per-cell commands (fabricator4, TIAGo) are in the [top-level README](../README.md#quick-start).

| `pipeline.launch.py` | default | |
| --- | --- | --- |
| `task` | `tower` (dual), `nominal` (other cells) | scene name or path to a scene YAML |
| `cell` | `dual` | which cell's `scenes/` a scene name is looked up in first; `tiago_cell_tamp`'s launch files set `tiago` |
| `engine` | `vamp` | `vamp` (seconds) or `fcl` (minutes, the exact reference) |
| `refine` | `true` | `false` stops after the first solve; must be `false` for more than two robots |
| `save_as` | — | keep the result under `artifacts/runs/<name>/` |
| `backend` | `cp-sat` | `cp-sat` or `gurobi` |
| `time_limit` | `120.0` | solver time limit, in seconds |

The other arguments (`schedule`, `objective`, `balance_weight`, `solver_seed`, `solver_workers`, `refine_test`, `refine_rungs`, `stop_after`, `from_pool`, `artifacts_dir`) serve the benchmark studies; the launch file documents each one.

| `execute_schedule.launch.py` | default | |
| --- | --- | --- |
| `mode` | `rigid` | `rigid`: all arms on one shared clock, safe only while every controller keeps up. `tpg`: each arm moves when the plan graph allows, so a delay costs time instead of safety. |
| `run` | — | replay a plan saved with `save_as` (trajectories, schedule, graph and scene together) |
| `refined` | `auto` | without `run`: the refined plan if the last pipeline run refined, else the baseline |
| `task_file` | the cell's `nominal` scene | without `run`: the scene to animate. Pass the one you planned, or use `run`. |
| `cell` | `dual` | whose `nominal` scene an empty `task_file` means |
| `visualize`, `actuate_grippers`, `process_events` | `true` | animate the objects; open and close the grippers; emulate the welding interlock of process tasks |
| `visualize_spheres` | `false` | draw VAMP's collision spheres on the arms |

The executor refuses to move if the trajectories and the schedule come from different runs, or if the plan does not start and end at home.

## Scenes

A scene is `<cell>/scenes/tamp_task_<name>.yaml`, passed as `task:=<name>` (`nominal` is `tamp_task.yaml`). The scene lists are in each cell's README. To preview a scene in RViz without planning (cell running):

```bash
ros2 run multi_robot_cell_tamp show_scene.py tower
```

To write a new scene, copy one from the same cell and edit its objects and tasks. `dual_robot_cell/scenes/tamp_task.yaml` explains every key in its comments. A task says what to move and where; precedences say what must happen first:

```yaml
tasks:
  - id: t_box_1
    object: box_1
    place: {x: -0.8, y: -0.15, z: 0.775}

precedences:
  - [t_lid, t_box_1]        # the lid comes off before box_1 moves
```

> If a task fails to plan, suspect a missing precedence first. Objects not ordered relative to each other are avoided at *both* their start and their goal, so a forgotten rule can leave no room to move.

On a real cell the object and place poses can be taught by hand instead of measured: move the gripper onto each pose and capture it, and the scene YAML is updated in place (a `.bak` copy is kept). Re-run the pipeline afterwards.

```bash
ros2 run multi_robot_cell_tamp teach_poses.py <scene.yaml>      # the robot must be up; commands: list, teach, show, write
```

## Outputs

```
artifacts/                          (in the source tree, not committed)
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

Every run overwrites the working files. Archive with `save_as` before comparing plans.

## On the real robots

With `mode:=tpg`, each grip waits for confirmation before the arm moves on: the gripper action succeeded, or the fingers stopped on the part. A close that ends fully shut (nothing grasped, checked only when the scene sets `gripper_min_closed`) or no confirmation within `gripper_timeout_s` (default 5 s) stops the execution. The arm and gripper interfaces come from the scene (`controller_action`, `gripper_joint`), so the same executors drive the UR cells and TIAGo.

## Tests and tools

```bash
colcon test --packages-select multi_robot_cell_tamp && colcon test-result --verbose   # C++ resampling
cd src/multi_robot_cell/tamp_core
.venv_vamp/bin/python scripts/test_tpg.py --quick          # plan graph: construction and execution checks
.venv_vamp/bin/python scripts/test_tpg_nrobot.py           # the same for more than two robots
python3 scripts/regress_seam.py artifacts/runs/<name>      # re-run a saved plan's seam and solve, compare
```

| Script | Interpreter | What it does |
| --- | --- | --- |
| `inspect_tpg.py` | `python3` | summarise or draw a plan graph |
| `simulate_tpg.py` | `.venv_vamp` | rigid versus graph execution under random delays |
| `coordinate.py` | `.venv_vamp` | the best possible coordination of two fixed paths |
| `vamp_fcl_mu_diff.py` | `.venv_vamp` | compare the VAMP seam with the FCL reference (missing offsets must be 0) |

Robot collision models for VAMP: [`vamp_codegen/README.md`](vamp_codegen/README.md). Design decisions: `docs/adr/` at the workspace root.
