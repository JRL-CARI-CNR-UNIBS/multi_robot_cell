"""Replay a solved schedule on the running cell's controllers.

Needs the cell up in another terminal:

    ros2 launch multi_robot_cell_bringup start.launch.py

then:

    ros2 launch multi_robot_cell_tamp execute_schedule.launch.py

Defaults replay the REFINED plan from the persistent ``artifacts/`` dir
(``tamp_trajectories_refined.json`` + ``artifacts/vamp/tamp_solution_refined.json``).
``refined:=false`` replays the unrefined baseline instead.

The trajectories and the schedule are chosen TOGETHER by that one argument, because only
matching pairs are valid: refinement produces its own shorter trajectories *and* its own
schedule, and crossing them replays a plan nobody solved. The executor checks the pair and
refuses to move if it does not match. Replay the FCL schedule with
``refined:=false solution_file:=<pkg>/artifacts/fcl/tamp_solution.json``.

Both arms play their trajectories on the shared clock; watch it in RViz. With
``visualize:=true`` (default) the tray/lid/boxes are also rendered and animated
(attach on pick, land at the place pose on place), read from the SAME
``config/tamp_task.yaml`` the offline generators use.

With ``visualize_spheres:=true`` (opt-in, default false) the VAMP collision
spheres (ADR-0005 Phase 2) are overlaid on the mesh arms, synced to the same
shared clock -- a `MarkerArray` on ``/vamp_collision_spheres``. This needs the
``.npz`` the pipeline regenerates every run, selected by the same ``refined:=``
argument so it always matches the motion it decorates.
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

    # The trajectories and the schedule must come from the SAME run, so one argument
    # picks both. Refinement (ADR-0008) writes a shorter, separate set of each; pairing a
    # refined schedule with baseline motions replays a plan that was never solved, and the
    # executor rejects it. Explicit traj_file / solution_file still override, for replaying
    # e.g. the FCL schedule.
    refined = LaunchConfiguration("refined").perform(context).lower() in ("true", "1", "yes")
    suffix = "_refined" if refined else ""
    mode = LaunchConfiguration("mode").perform(context).lower()
    # A plan archived by `pipeline.launch.py save_as:=<name>`. Its three files were copied
    # together and are matched by construction, which is the point: the working artifacts
    # hold only the most recent run, so comparing two schedules needs both saved first.
    run = LaunchConfiguration("run").perform(context).strip()
    run_dir = os.path.join(ARTIFACTS_DIR, "runs", run) if run else ""
    if run and not os.path.isdir(run_dir):
        saved = os.path.join(ARTIFACTS_DIR, "runs")
        have = sorted(os.listdir(saved)) if os.path.isdir(saved) else []
        raise RuntimeError(
            f"no saved plan named {run!r} in {saved}"
            + (f" -- have: {', '.join(have)}" if have else " -- none saved yet; plan one "
               "with pipeline.launch.py save_as:=<name>"))

    traj_file = (LaunchConfiguration("traj_file").perform(context)
                 or (os.path.join(run_dir, "tamp_trajectories.json") if run
                     else os.path.join(ARTIFACTS_DIR, f"tamp_trajectories{suffix}.json")))
    solution_file = (LaunchConfiguration("solution_file").perform(context)
                     or (os.path.join(run_dir, "tamp_solution.json") if run
                         else os.path.join(ARTIFACTS_DIR, "vamp", f"tamp_solution{suffix}.json")))
    tpg_file = (LaunchConfiguration("tpg_file").perform(context)
                or (os.path.join(run_dir, "tpg.json") if run
                    else os.path.join(ARTIFACTS_DIR, "vamp", f"tpg{suffix}.json")))
    if run and not LaunchConfiguration("task_file").perform(context):
        # The scene is part of the saved plan: replaying it against the installed default
        # would animate different geometry from the one that was solved.
        import glob

        yamls = [y for y in glob.glob(os.path.join(run_dir, "*.yaml"))]
        if yamls:
            task_file = yamls[0]
    # The sphere shells are computed FROM the trajectories, so they belong to the same
    # matched set and follow the same switch. The overlay verifies the pairing anyway.
    spheres_file = (LaunchConfiguration("spheres_file").perform(context)
                    or os.path.join(ARTIFACTS_DIR, "vamp", f"tamp_spheres{suffix}.npz"))

    common = {
        "traj_file": traj_file,
        "solution_file": solution_file,
        "task_file": task_file,
        "visualize": ParameterValue(LaunchConfiguration("visualize"), value_type=bool),
        "actuate_grippers": ParameterValue(
            LaunchConfiguration("actuate_grippers"), value_type=bool),
    }
    if mode == "tpg":
        # Graph dispatch: no shared start instant, because there is no shared clock -- each
        # arm advances when its incoming edge is satisfied (ADR-0007).
        return [
            Node(
                package="multi_robot_cell_tamp",
                executable="tpg_executor.py",
                output="screen",
                parameters=[dict(common, tpg_file=tpg_file, label=run or "working artifacts")],
            )
        ]

    nodes = [
        Node(
            package="multi_robot_cell_tamp",
            executable="schedule_executor.py",
            output="screen",
            parameters=[
                {
                    "traj_file": traj_file,
                    "solution_file": solution_file,
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
                        "spheres_file": spheres_file,
                        "solution_file": solution_file,
                        "traj_file": traj_file,
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
                "refined", default_value="true", choices=["true", "false"],
                description="Replay the refined plan (ADR-0008) or the "
                "baseline. Selects the trajectory artifact and the schedule TOGETHER, "
                "which is the only pairing that is valid.",
            ),
            DeclareLaunchArgument(
                "traj_file", default_value="",
                description="Override the trajectory artifact. Empty: chosen by `refined`.",
            ),
            DeclareLaunchArgument(
                "solution_file", default_value="",
                description="Override the schedule. Empty: chosen by `refined`. Use "
                "artifacts/fcl/tamp_solution.json for the FCL schedule (with refined:=false).",
            ),
            DeclareLaunchArgument(
                "mode", default_value="rigid", choices=["rigid", "tpg"],
                description="rigid: one trajectory per arm on a shared clock -- the "
                "offsets hold only while both controllers keep up. tpg: each arm advances "
                "when its incoming edge is satisfied, dispatched in contiguous runs "
                "(ADR-0007). Under `tpg` the executed makespan is reported.",
            ),
            DeclareLaunchArgument(
                "run", default_value="",
                description="Replay a plan archived by pipeline.launch.py save_as:=<name>, "
                "from artifacts/runs/<name>/. Selects trajectories, schedule, graph AND "
                "scene together. Empty: the working artifacts, chosen by `refined`.",
            ),
            DeclareLaunchArgument(
                "tpg_file", default_value="",
                description="Override the plan graph (mode:=tpg). Empty: chosen by `run` "
                "or `refined`.",
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
                "spheres_file", default_value="",
                description="Override the VAMP sphere .npz. Empty: chosen by `refined`, "
                "matching the trajectories being replayed. The pipeline regenerates it "
                "every run (dump_vamp_spheres.py).",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
