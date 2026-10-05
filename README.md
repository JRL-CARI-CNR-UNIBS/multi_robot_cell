# Multi Robot Cell

[![jazzy](https://github.com/JRL-CARI-CNR-UNIBS/multi_robot_cell/actions/workflows/jazzy.yml/badge.svg)](https://github.com/JRL-CARI-CNR-UNIBS/multi_robot_cell/actions/workflows/jazzy.yml)

ROS 2 Jazzy packages for **multi-robot task and motion planning (TAMP)**. You give a scene: objects, where they go, and what must happen first. The pipeline decides **which robot does each task and when it starts**, minimises the total time (the makespan), guarantees the arms never collide, and runs all arms at once in RViz or on the controllers.

The same pipeline runs on three robot cells:

| Cell | Robots | Folder |
| --- | --- | --- |
| `dual` | two UR10e on linear rails, Robotiq 2F-85 grippers | [`dual_robot_cell/`](dual_robot_cell/README.md) |
| `fabricator4` | four UR10e on rails: one handler with a gripper, three MIG welders | [`four_robot_cell/`](four_robot_cell/README.md) |
| `tiago` | a stationary TIAGo Pro, two arms, PAL grippers | [`tiago_cell/`](https://github.com/CNR-STIIMA-IRAS/tiago_cell) (separate repository) |

## Repository layout

```
multi_robot_cell/
  tamp_core/          the pipeline, independent of any cell (ROS package multi_robot_cell_tamp)
  dual_robot_cell/    UR description, MoveIt config, bringup, pick-place demo, dual scenes
  four_robot_cell/    fabricator4 description, MoveIt config, controllers, its scene
  tiago_cell/         imported by vcs (own repository, gitignored here)
  dependencies.repos
```

A cell is a folder with a `cell.yaml`: its MoveIt files, its `scenes/` and its collision model for VAMP. `tamp_core` knows no cell by name, so a new cell is a new folder and needs no change in `tamp_core`. A scene picks its cell with its `cell:` key (no key means `dual`). The pipeline itself is documented in [`tamp_core/README.md`](tamp_core/README.md).

## Workspace setup

The pipeline expects this layout. The paths are fixed: the launch files find the solver and the VAMP environment from them.

```
<ws>/                            e.g. ~/ws_thesis
  src/
    multi_robot_cell/            this repository
      tiago_cell/                vcs import (dependencies.repos)
        vendor/                  PAL packages, vcs import (tiago_cell/dependencies.repos)
      tamp_core/.venv_vamp/      Python environment for VAMP (created below)
    ros2_robotiq_gripper/        vcs import (dependencies.repos)
    serial/                      vcs import (the Robotiq driver's dependency)
  thesis_material_tamp/          the solver (git clone, NOT under src/)
    .venv/                       Python environment for the solver (created below)
```

### 1. Clone and import

```bash
mkdir -p ~/ws_thesis/src && cd ~/ws_thesis
git clone https://github.com/JRL-CARI-CNR-UNIBS/multi_robot_cell.git src/multi_robot_cell
vcs import src < src/multi_robot_cell/dependencies.repos
vcs import src < src/ros2_robotiq_gripper/ros2_robotiq_gripper-not-released.rolling.repos
vcs import src/multi_robot_cell/tiago_cell/vendor < src/multi_robot_cell/tiago_cell/dependencies.repos
git clone git@github.com:CNR-STIIMA-IRAS/thesis_material_tamp.git
```

`tiago_cell` and `thesis_material_tamp` are private repositories: you need access to them on GitHub. Without `tiago_cell` the dual and four-robot cells still work; skip the third `vcs import`.

### 2. Build the ROS packages

```bash
source /opt/ros/jazzy/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Always build **from the workspace root**. Run from a package directory, colcon builds a separate workspace there and the real binaries stay stale.

### 3. The two Python environments

The pipeline uses three Python interpreters and never mixes them: the ROS Python (nodes and launch files), the solver's (OR-Tools, Gurobi) and VAMP's. The launch files pick the right one for each stage.

```bash
# the solver: OR-Tools CP-SAT (Gurobi optional, needs a license)
cd ~/ws_thesis/thesis_material_tamp
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# VAMP, the collision engine; also used by refinement and the plan graph
cd ~/ws_thesis/src/multi_robot_cell/tamp_core
python3 -m venv .venv_vamp && .venv_vamp/bin/pip install numpy pyyaml
./vamp_codegen/build_vamp.sh      # builds VAMP with every cell's robot models (~8 min, ~1.6 GB RAM)
./scripts/build_mu_kernel.sh      # the SIMD collision kernel for this CPU (once per machine)
```

Run `build_vamp.sh` again after adding a cell or changing a robot's collision model. Without the kernel, VAMP falls back to numpy: same result, about 40 times slower. If `python3 -m venv` gives an environment without pip, bootstrap it with [`get-pip.py`](https://bootstrap.pypa.io/get-pip.py).

> **Troubleshooting**
> - `Could not find a package configuration file provided by "serial"`: the second `vcs import` (the Robotiq `.repos` file) was skipped.
> - `rosdep update` fails with `HTTP Error 429`: a GitHub rate limit; retry after a short wait.
> - A C++ change has no effect: colcon was run from inside a package directory. Build from the workspace root.
> - After moving a package directory, delete its folders in `build/` and `install/` once (stale CMake cache).

## Quick start

Two terminals per cell. The offline pipeline does not need the cell running; execution does.

**Dual cell** (two UR10e):

```bash
ros2 launch multi_robot_cell_tamp pipeline.launch.py task:=tower save_as:=tower              # offline, ~30 s
ros2 launch multi_robot_cell_bringup start.launch.py                                         # terminal 1
ros2 launch multi_robot_cell_tamp execute_schedule.launch.py mode:=tpg run:=tower            # terminal 2
```

**Four-robot cell** (fabricator4). Refinement handles two robots only, so it is off:

```bash
ros2 launch multi_robot_cell_tamp pipeline.launch.py task:=fabricator refine:=false save_as:=fab
ros2 launch multi_robot_cell_bringup start.launch.py cell:=fabricator4                       # terminal 1
ros2 launch multi_robot_cell_tamp execute_schedule.launch.py mode:=tpg run:=fab              # terminal 2
```

**TIAGo cell**:

```bash
ros2 launch tiago_cell_tamp pipeline.launch.py task:=numbers save_as:=numbers
ros2 launch tiago_cell bringup.launch.py                                                     # terminal 1
ros2 launch tiago_cell_tamp execute_schedule.launch.py mode:=tpg run:=numbers                # terminal 2
```

Always use `save_as:=<name>` and replay with `run:=<name>`. Every pipeline run overwrites the working files in `tamp_core/artifacts/`. A saved run keeps the trajectories, schedule, plan graph and scene together, so the replay animates exactly the scene that was planned.

`mode:=tpg` moves each arm as soon as the plan graph allows, so a late arm only costs time. `mode:=rigid` plays all arms on one shared clock, which is safe only while every controller keeps up. All arguments are in [`tamp_core/README.md`](tamp_core/README.md).

## Documentation

- [`tamp_core/README.md`](tamp_core/README.md): pipeline stages, launch arguments, scene format, outputs, tests.
- [`dual_robot_cell/README.md`](dual_robot_cell/README.md): the UR cell, interactive planning in RViz, the pick-and-place demo, the scenes.
- [`four_robot_cell/README.md`](four_robot_cell/README.md): the fabricator4 welding cell.
- `tiago_cell/README.md`: the TIAGo cell, its table and the real robot.
- `thesis_material_tamp/README.md`: the solver library (no ROS) and its 2D validation.
