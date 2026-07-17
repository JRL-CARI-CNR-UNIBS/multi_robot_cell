# Deprecated files in this package

**`urdf/` is live and load-bearing.** Everything else listed below is not.

The `launch/`, `controllers/`, `config/` and `srdf/` directories in this package are an
**earlier parallel copy** of the cell's launch and configuration, superseded by
`multi_robot_cell_bringup` (launch + runtime controllers) and `multi_robot_moveit_config`
(SRDF, kinematics, joint limits, OMPL). Nothing in the workspace references them.

They are kept for reference only. **Do not use them, and do not update them** — they have
drifted from the live model and several are now simply broken:

| Path | Why it is dead |
| --- | --- |
| `launch/multi_robot_cell_moveit.launch.py` | Loads `urdf/ur10e_2f85_linear_robot.urdf.xacro` — **a file that does not exist**. Cannot run. |
| `launch/multi_robot_cell{,_control,_view}.launch.py` | Superseded by `multi_robot_cell_bringup/launch/`. Referenced by nothing. |
| `controllers/robot{1,2}_controllers.yaml` | Use **unprefixed** joint names (`shoulder_pan_joint`, not `robot1_shoulder_pan_joint`) — they pre-date the `robot1_`/`robot2_` namespacing the live URDF and controllers use. |
| `srdf/ur10e_2f85_linear_robot.srdf` | A **single-robot** model with groups `manipulator`/`gripper`. The live SRDF is `multi_robot_moveit_config/config/multi_robot_cell.srdf` (`manipulator1`, `manipulator2`, `manipulator1_on_rail`, `manipulator2_on_rail`, `robot_system`). |
| `config/{kinematics,joint_limits,ompl_planning,moveit_controllers}.yaml` | Key on group `manipulator:`, which no longer exists. Live equivalents are in `multi_robot_moveit_config/config/`. |
| `config/multi_robot_cell_view.rviz`, `scripts/publish_robot_description.py` | Only reachable from the dead launch files above. |

## The live path

```
ros2 launch multi_robot_cell_bringup start.launch.py
  └── multi_robot_moveit.launch.py   (move_group + RViz)
  └── multi_robot_control.launch.py  (ros2_control, robot_state_publisher, spawners)
```

Controller YAMLs: `multi_robot_cell_bringup/config/`.
MoveIt config: `multi_robot_moveit_config/config/`.
Kinematic model: **`multi_robot_cell_description/urdf/`** — this package's one live contribution.
