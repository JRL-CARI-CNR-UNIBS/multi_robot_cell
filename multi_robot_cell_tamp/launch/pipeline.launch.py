"""The whole offline TAMP pipeline in one command.

    ros2 launch multi_robot_cell_tamp pipeline.launch.py task:=tower

Runs the three offline stages in order, each starting only if the previous one
succeeded:

    1. trajectory_generator   one trajectory per (robot, task) pair   ~2 s
    2. collision analysis     mu -> forbidden offsets                 ~1 s (vamp) / ~9 min (fcl)
    3. solve.py               CP-SAT: who does what, and when         ~1 s

Execution is deliberately NOT part of this: it needs the cell up
(``multi_robot_cell_bringup start.launch.py``) in another terminal, and it is a
long-running interactive process rather than a pipeline stage. The launch prints the
exact command to run when it finishes.

WHY A LAUNCH FILE HAS TO WORK THIS HARD HERE
--------------------------------------------
The three stages do not share a Python environment, and cannot:

* stages 1 and 2-fcl are ROS C++ nodes needing the MoveIt robot model;
* stage 2-vamp runs in ``.venv_vamp``, because ``vamp`` is not (and per ADR-0002 must
  never be) importable from the ROS runtime;
* stage 3 runs in ``thesis_material_tamp/.venv``, because OR-Tools is not in the ROS
  Python and ``tamp_scheduler`` must stay ROS-free.

So the venv stages are ``ExecuteProcess`` with an explicit interpreter rather than
``Node``. Sequencing uses ``OnProcessExit`` handlers, and each handler checks the exit
code: a failed stage stops the pipeline instead of feeding the next one a stale or
missing artifact.

EVERY STAGE ALWAYS RUNS
-----------------------
There is no staleness check, on purpose. OMPL is not seeded (``seed`` in the task YAML
is parsed but never applied), so regenerating trajectories produces *different* ones and
silently invalidates every downstream artifact. Running all three together keeps them
consistent by construction. Run a single stage on its own if you want to reuse an
artifact deliberately.
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder

# realpath() resolves the --symlink-install symlink back to the SOURCE tree, so
# artifacts land next to the package rather than inside install/.
PKG_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ARTIFACTS_DIR = os.path.join(PKG_DIR, "artifacts")
WS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(PKG_DIR)))

VAMP_PYTHON = os.path.join(PKG_DIR, ".venv_vamp", "bin", "python")
SOLVER_DIR = os.path.join(WS_DIR, "thesis_material_tamp")
SOLVER_PYTHON = os.path.join(SOLVER_DIR, ".venv", "bin", "python")


def resolve_task(name: str) -> str:
    """Accept either a scene shorthand or a path.

    ``tower`` -> ``config/tamp_task_tower.yaml``; ``nominal`` -> ``config/tamp_task.yaml``;
    anything containing a separator or ending in .yaml is taken as a path as given.
    """
    if os.sep in name or name.endswith((".yaml", ".yml")):
        return os.path.abspath(name)
    stem = "tamp_task.yaml" if name == "nominal" else f"tamp_task_{name}.yaml"
    return os.path.join(PKG_DIR, "config", stem)


def launch_setup(context):
    task = resolve_task(LaunchConfiguration("task").perform(context))
    engine = LaunchConfiguration("engine").perform(context).lower()
    if engine not in ("vamp", "fcl"):
        raise RuntimeError(f"engine must be 'vamp' or 'fcl', got {engine!r}")
    if not os.path.isfile(task):
        raise RuntimeError(f"task file not found: {task}")

    traj = os.path.join(ARTIFACTS_DIR, "tamp_trajectories.json")
    out_dir = os.path.join(ARTIFACTS_DIR, engine)
    problem = os.path.join(out_dir, "tamp_problem.json")
    solution = os.path.join(out_dir, "tamp_solution.json")
    os.makedirs(out_dir, exist_ok=True)

    share = FindPackageShare("multi_robot_moveit_config")
    urdf = PathJoinSubstitution([share, "config", "multi_robot_cell.urdf.xacro"]).perform(context)
    srdf = PathJoinSubstitution([share, "config", "multi_robot_cell.srdf"]).perform(context)
    limits = PathJoinSubstitution([share, "config", "joint_limits.yaml"]).perform(context)
    # multi_robot_moveit_config's own moveit_controllers.yaml is fully commented out and
    # to_moveit_configs() chokes on it; the bringup copy is the working one.
    controllers = PathJoinSubstitution(
        [FindPackageShare("multi_robot_cell_bringup"), "config", "moveit_controllers.yaml"]
    ).perform(context)

    moveit_config = (
        MoveItConfigsBuilder("manipulator", package_name="multi_robot_moveit_config")
        .robot_description(file_path=urdf)
        .robot_description_semantic(file_path=srdf)
        .trajectory_execution(file_path=controllers)
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
        .joint_limits(file_path=limits)
        .to_moveit_configs()
    )

    # -- stage 1: trajectories ------------------------------------------------- #
    stage_traj = Node(
        package="multi_robot_cell_tamp",
        executable="trajectory_generator",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {"task_file": task, "out_file": traj},
        ],
    )

    # -- stage 2: collision analysis ------------------------------------------- #
    if engine == "fcl":
        # xacro is processed here rather than reusing moveit_config so the collision
        # node gets exactly the two parameters it wants and nothing else.
        import xacro

        with open(srdf) as f:
            srdf_text = f.read()
        stage_coll = Node(
            package="multi_robot_cell_tamp",
            executable="collision_generator",
            output="screen",
            parameters=[
                {"robot_description": xacro.process_file(urdf).toxml()},
                {"robot_description_semantic": srdf_text},
                {"traj_file": traj, "task_file": task, "out_file": problem},
            ],
        )
    else:
        stage_coll = ExecuteProcess(
            cmd=[VAMP_PYTHON, os.path.join(PKG_DIR, "scripts", "collision_generator_vamp.py"),
                 "--traj", traj, "--task", task, "--out", problem],
            output="screen",
        )

    # -- stage 3: solve -------------------------------------------------------- #
    stage_solve = ExecuteProcess(
        cmd=[SOLVER_PYTHON, os.path.join(SOLVER_DIR, "solve.py"), problem, solution,
             "--backend", LaunchConfiguration("backend"),
             "--time-limit", LaunchConfiguration("time_limit")],
        output="screen",
    )

    def chain(previous, nxt, label):
        """Run ``nxt`` when ``previous`` exits 0; abort the pipeline otherwise."""
        return RegisterEventHandler(
            OnProcessExit(
                target_action=previous,
                on_exit=lambda event, ctx: (
                    [nxt] if event.returncode == 0
                    else [LogInfo(msg=f"\n*** pipeline aborted: {label} failed "
                                      f"(exit {event.returncode}) ***\n"),
                          Shutdown(reason=f"{label} failed")]
                ),
            )
        )

    done = RegisterEventHandler(
        OnProcessExit(
            target_action=stage_solve,
            on_exit=lambda event, ctx: [
                LogInfo(msg=(
                    f"\n=== pipeline complete ({engine}) ===\n"
                    f"  trajectories : {traj}\n"
                    f"  seam         : {problem}\n"
                    f"  schedule     : {solution}\n\n"
                    f"To execute it, with the cell running "
                    f"(ros2 launch multi_robot_cell_bringup start.launch.py):\n"
                    f"  ros2 launch multi_robot_cell_tamp execute_schedule.launch.py \\\n"
                    f"      task_file:={task} solution_file:={solution}\n"
                )) if event.returncode == 0 else
                LogInfo(msg=f"\n*** pipeline aborted: solve failed "
                            f"(exit {event.returncode}) ***\n"),
            ],
        )
    )

    return [
        LogInfo(msg=f"\n=== TAMP pipeline: task={os.path.basename(task)} engine={engine} ===\n"),
        stage_traj,
        chain(stage_traj, stage_coll, "trajectory generation"),
        chain(stage_coll, stage_solve, "collision analysis"),
        done,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "task", default_value="tower",
                description="Scene: a shorthand (tower, swap, nominal) resolved against "
                            "config/, or a path to a task YAML."),
            DeclareLaunchArgument(
                "engine", default_value="vamp", choices=["vamp", "fcl"],
                description="Collision engine. vamp: the SIMD sphere engine (~1 s). "
                            "fcl: the exact MoveIt/FCL reference (~9 min)."),
            DeclareLaunchArgument(
                "backend", default_value="cp-sat", choices=["cp-sat", "gurobi"],
                description="Solver backend for the allocation stage."),
            DeclareLaunchArgument(
                "time_limit", default_value="120.0",
                description="Solver time limit in seconds."),
            OpaqueFunction(function=launch_setup),
        ]
    )
