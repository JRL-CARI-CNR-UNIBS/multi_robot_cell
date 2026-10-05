# dual_robot_cell — two UR10e on linear rails

Two UR10e arms, each on a linear guide (7 DOF: rail + arm), with Robotiq 2F-85 grippers, facing each other across a table. This is the cell `dual` of the TAMP pipeline (`cell.yaml`), and the default one: a scene without a `cell:` key runs here.

| Package / folder | Content |
| --- | --- |
| `multi_robot_cell_description` | URDF/Xacro: UR10e, Robotiq grippers, linear guides, environment |
| `multi_robot_moveit_config` | MoveIt: SRDF planning groups, kinematics, joint limits, OMPL |
| `multi_robot_cell_bringup` | `start.launch.py`: `ros2_control`, controllers, `move_group`, RViz. Also brings up the four-robot cell (`cell:=fabricator4`). |
| `multi_robot_cell_scene` | single-robot pick-and-place demo, fixed order (the baseline before TAMP) |
| `scenes/` | the TAMP scenes of this cell |
| `vamp_modules/ur10e_rail/` | the arm's sphere model for VAMP (also used by the four-robot cell's gripper robots) |

## Bring up the cell

```bash
ros2 launch multi_robot_cell_bringup start.launch.py
```

Starts `ros2_control_node`, `robot_state_publisher`, the joint state broadcaster, the trajectory and gripper controllers, `move_group` and RViz.

| Argument | Default | |
| --- | --- | --- |
| `fake` | `true` | mock hardware; `false` for the real robots |
| `robotiq_com_port_robot1`, `robotiq_com_port_robot2` | `/dev/robotiq1`, `/dev/robotiq2` | gripper serial ports (real hardware) |
| `rviz` | `true` | start RViz |
| `cell` | `dual` | `fabricator4` for the [four-robot cell](../four_robot_cell/README.md) |

## Run the TAMP pipeline here

```bash
ros2 launch multi_robot_cell_tamp pipeline.launch.py task:=tower save_as:=tower
ros2 launch multi_robot_cell_tamp execute_schedule.launch.py mode:=tpg run:=tower     # with the cell up
```

Details: [`tamp_core/README.md`](../tamp_core/README.md).

### Scenes

Pass `scenes/tamp_task_<name>.yaml` as `task:=<name>`; `nominal` is `tamp_task.yaml`. The header of each file explains it.

| Scene | What it is |
| --- | --- |
| `nominal` | a lid and three boxes; the lid comes off before any box moves |
| `swap` | four boxes, each to the opposite end of the table and to the other robot's side; any order |
| `swap_x` | four boxes crossing along the rail only (the original `swap`, kept so earlier results reproduce) |
| `tower`, `tower6`, `tower8`, `tower10` | 4–10 boxes restacked in reverse order, fixed sequence |
| `tower_interchangeable` | a six-level tower built from a stock of identical cubes: any cube can fill any level |
| `int_cluttered` | the same tower under an earlier, more cluttered rule for which objects count as obstacles |
| `tower_wall` | `tower_interchangeable` with two static fixtures in the way |
| `slotmini` | two tower levels, two candidate cubes each: small enough for the FCL reference |
| `numbers` | five blocks laid out as the digit "2", rearranged into a "3" |
| `cont05`, `cont08`, `cont14`, `cont22` | boxes in two columns at x = ±5…22 cm: the closer to the centre, the more the arms contend; any order |
| `size6`, `size8` | 6 or 8 boxes, any order |
| `seq6`, `seq8`, `seq10`, `seq12` | 6–12 boxes under a total order |
| `precchain`, `precpart` | the `cont14` boxes under a total and a partial order |
| `biw`, `biw_v09`, `biw_panel`, `biw_door` | industrial cases: a body-in-white welding station, at reduced speed, a framing/respot panel, a real car door at 1:2 |
| `weldprobe` | one bracket, one weld seam: the smallest process task |
| `meshprobe`, `p2probe` | test scenes for mesh geometry and for the four-robot generator path |

`holdprobe` is also stored here but runs on the four-robot cell (`cell: fabricator4`).

## Interactive planning in RViz

After `start.launch.py`, RViz opens with the MoveIt MotionPlanning plugin. Planning groups:

- `manipulator1`, `manipulator2`: the 6-DOF arm only;
- `manipulator1_on_rail`, `manipulator2_on_rail`: the linear guide plus the arm;
- `robot_system`: the whole cell.

The active controller must command every joint of the group you execute. By default `start.launch.py` activates `robot<N>_linear_guide_joint_trajectory_controller` (rail + arm) and the gripper controllers; the 6-DOF and scaled variants are spawned inactive. Controllers that claim the same joints cannot be active together, so switch in one call:

```bash
ros2 control list_controllers
ros2 control switch_controllers --deactivate <current_controller> --activate <target_controller>
```

## Pick-and-place demo (`multi_robot_cell_scene`)

A fixed, single-robot pick-and-place on `robot1`, driven by [`multi_robot_cell_scene/config/task.yaml`](multi_robot_cell_scene/config/task.yaml). For each entry in `task_plan`: pre-grasp, approach, close, attach, retreat, pre-place, lower, open, detach, retreat. With the cell up:

```bash
ros2 launch multi_robot_cell_scene pick_place.launch.py
```

It plans with `manipulator1_on_rail`, which the default controller already covers. Edit `task.yaml` (poses, grasp offsets, order) and relaunch; no rebuild needed. `ros2 run multi_robot_cell_scene spawn_object` adds a single 5 cm box to the planning scene (needs only `move_group`).
