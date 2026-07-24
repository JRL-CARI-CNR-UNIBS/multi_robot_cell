"""Replay a solved schedule on the running cell's controllers.

Needs the cell up in another terminal:

    ros2 launch multi_robot_cell_bringup start.launch.py

then:

    ros2 launch multi_robot_cell_tamp execute_schedule.launch.py

Defaults read the persistent ``artifacts/`` dir (shared ``tamp_trajectories.json`` +
the VAMP schedule ``artifacts/vamp/tamp_solution.json``). Replay the FCL schedule with
``solution_file:=<pkg>/artifacts/fcl/tamp_solution.json``.

Both arms play their trajectories on the shared clock; watch it in RViz. With
``visualize:=true`` (default) the tray/lid/boxes are also rendered and animated
(attach on pick, land at the place pose on place), read from the SAME
``config/tamp_task.yaml`` the offline generators use.

With ``visualize_spheres:=true`` (opt-in, default false) the VAMP collision
spheres (ADR-0005 Phase 2) are overlaid on the mesh arms, synced to the same
shared clock -- a `MarkerArray` on ``/vamp_collision_spheres``. This needs the
pre-computed ``.npz`` (``spheres_file``, default ``/tmp/tamp_spheres.npz``), a
viz-only prep step run in ``.venv_vamp`` (see ``scripts/vamp_sphere_overlay.py``).
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

# Persistent (non-/tmp) artifacts dir under the package SOURCE tree; realpath()
# resolves the --symlink-install symlink to source. Defaults target the VAMP
# pipeline (artifacts/vamp/), which pairs with the sphere overlay; replay the FCL
# schedule with solution_file:=<...>/artifacts/fcl/tamp_solution.json.
ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "artifacts"
)


def launch_setup(context):
    # Empty task_file -> the installed config/tamp_task.yaml (mirrors
    # generate_collisions.launch.py). One geometry source across the pipeline.
    task_default = PathJoinSubstitution(
        [FindPackageShare("multi_robot_cell_tamp"), "config", "tamp_task.yaml"]
    ).perform(context)
    task_file = LaunchConfiguration("task_file").perform(context) or task_default

    nodes = [
        Node(
            package="multi_robot_cell_tamp",
            executable="schedule_executor.py",
            output="screen",
            parameters=[
                {
                    "traj_file": LaunchConfiguration("traj_file"),
                    "solution_file": LaunchConfiguration("solution_file"),
                    "start_delay": LaunchConfiguration("start_delay"),
                    "task_file": task_file,
                    "visualize": ParameterValue(
                        LaunchConfiguration("visualize"), value_type=bool
                    ),
                    "actuate_grippers": ParameterValue(
                        LaunchConfiguration("actuate_grippers"), value_type=bool
                    ),
                }
            ],
        )
    ]

    # Opt-in VAMP collision-sphere overlay: replays the pre-computed sphere shells
    # (ADR-0005 Phase 2) on the mesh arms, synced to the same shared clock. Needs
    # the .npz prepped in .venv_vamp (see vamp_sphere_overlay.py / README).
    if LaunchConfiguration("visualize_spheres").perform(context).lower() in ("true", "1"):
        nodes.append(
            Node(
                package="multi_robot_cell_tamp",
                executable="vamp_sphere_overlay.py",
                output="screen",
                parameters=[
                    {
                        "spheres_file": LaunchConfiguration("spheres_file"),
                        "solution_file": LaunchConfiguration("solution_file"),
                        "traj_file": LaunchConfiguration("traj_file"),
                        "task_file": task_file,
                        "start_delay": LaunchConfiguration("start_delay"),
                    }
                ],
            )
        )

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "traj_file",
                default_value=os.path.join(ARTIFACTS_DIR, "tamp_trajectories.json"),
            ),
            DeclareLaunchArgument(
                "solution_file",
                default_value=os.path.join(ARTIFACTS_DIR, "vamp", "tamp_solution.json"),
                description="Schedule to replay; default the VAMP pipeline's. Use "
                "artifacts/fcl/tamp_solution.json for the FCL schedule.",
            ),
            DeclareLaunchArgument("start_delay", default_value="2.0"),
            DeclareLaunchArgument(
                "task_file",
                default_value="",
                description="Scene geometry YAML; empty -> installed config/tamp_task.yaml.",
            ),
            DeclareLaunchArgument(
                "visualize",
                default_value="true",
                description="Render + animate the scene objects (tray/lid/boxes) in RViz.",
            ),
            DeclareLaunchArgument(
                "actuate_grippers",
                default_value="true",
                description="Open/close the grippers in sync with each pick/place.",
            ),
            DeclareLaunchArgument(
                "visualize_spheres",
                default_value="false",
                description="Overlay the VAMP collision spheres on the arms in RViz "
                "(opt-in; needs the .npz prepped in .venv_vamp -- see the README).",
            ),
            DeclareLaunchArgument(
                "spheres_file",
                default_value=os.path.join(ARTIFACTS_DIR, "vamp", "tamp_spheres.npz"),
                description="Pre-computed VAMP sphere .npz (dump_vamp_spheres.py).",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
