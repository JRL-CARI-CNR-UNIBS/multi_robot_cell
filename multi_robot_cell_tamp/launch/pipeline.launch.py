"""The whole offline TAMP pipeline in one command.

    ros2 launch multi_robot_cell_tamp pipeline.launch.py task:=tower

Runs the offline stages in order, each starting only if the previous one succeeded:

    1. trajectory_generator   one trajectory per (robot, task) pair   ~2 s
    2. collision analysis     mu -> forbidden offsets                 ~1 s (vamp) / ~9 min (fcl)
    3. solve.py               CP-SAT: who does what, and when         ~1 s
    4. build_tpg.py           the delay-robust execution graph        ~1 s

With ``refine:=true`` (the default) three more stages run after the solve and then
stages 2-4 repeat on the shortened plan:

    3a. refine_yield --plan   pick yield poses clear of the other arm  ~1 s
    3b. transit planning      MoveIt: the connecting motions           ~1 s
    3c. refine_yield --splice verify and assemble                      ~1 s

Refinement shortens the commute between two tasks on the same robot -- most of all
motion, since every task returns home. It needs the schedule to know which tasks are
consecutive, which is why it runs after the solve and why the seam and schedule are then
rebuilt. Its measured effect is recorded in ADR-0008, not here.

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

import json
import os
import shutil

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
    refine = LaunchConfiguration("refine").perform(context).lower() in ("true", "1", "yes")
    # Ablation knobs (ADR-0008 addendum): which test a shortcut is checked against, and how
    # many rungs the ladder has. Defaults are the production configuration.
    refine_test = LaunchConfiguration("refine_test").perform(context).lower()
    refine_rungs = LaunchConfiguration("refine_rungs").perform(context)
    # Comparison arm (APEX-MR): schedule turn-based, so concurrency is recovered by the
    # graph at execution instead of being decided by the solver. Applies to BOTH solves,
    # since a policy that changed halfway through would not be either architecture.
    turn_based = LaunchConfiguration("schedule").perform(context).lower() == "turn_based"
    solver_objective = LaunchConfiguration("objective").perform(context).lower()
    balance_weight = LaunchConfiguration("balance_weight").perform(context)
    solver_seed = LaunchConfiguration("solver_seed").perform(context).strip()
    # Archive the finished plan under a name, so two plans can exist at once. Without this
    # every run overwrites artifacts/, and comparing two schedules in RViz is impossible:
    # by the time the second is planned the first no longer exists.
    save_as = LaunchConfiguration("save_as").perform(context).strip()
    if save_as and (os.sep in save_as or save_as.startswith(".")):
        raise RuntimeError(f"save_as must be a plain name, got {save_as!r}")
    if engine not in ("vamp", "fcl"):
        raise RuntimeError(f"engine must be 'vamp' or 'fcl', got {engine!r}")
    if not os.path.isfile(task):
        raise RuntimeError(f"task file not found: {task}")

    traj = os.path.join(ARTIFACTS_DIR, "tamp_trajectories.json")
    out_dir = os.path.join(ARTIFACTS_DIR, engine)
    problem = os.path.join(out_dir, "tamp_problem.json")
    solution = os.path.join(out_dir, "tamp_solution.json")
    os.makedirs(out_dir, exist_ok=True)
    # Refinement rewrites the plan, so its outputs get their own names: the unrefined
    # artifacts stay on disk as the baseline the improvement is measured against.
    refined_traj = os.path.join(ARTIFACTS_DIR, "tamp_trajectories_refined.json")
    refined_problem = os.path.join(out_dir, "tamp_problem_refined.json")
    refined_solution = os.path.join(out_dir, "tamp_solution_refined.json")
    transit_spec = os.path.join(ARTIFACTS_DIR, "transit_spec.json")
    transits = os.path.join(ARTIFACTS_DIR, "transits.json")
    spheres = os.path.join(out_dir, "tamp_spheres_refined.npz" if refine else "tamp_spheres.npz")

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

    def motion_node(extra):
        """The MoveIt node, in whichever of its modes `extra` selects."""
        return Node(
            package="multi_robot_cell_tamp",
            executable="trajectory_generator",
            output="screen",
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.planning_pipelines,
                moveit_config.joint_limits,
                {"task_file": task, **extra},
            ],
        )

    def collision_stage(traj_in, problem_out):
        if engine == "fcl":
            # xacro is processed here rather than reusing moveit_config so the collision
            # node gets exactly the two parameters it wants and nothing else.
            import xacro

            with open(srdf) as f:
                srdf_text = f.read()
            return Node(
                package="multi_robot_cell_tamp",
                executable="collision_generator",
                output="screen",
                parameters=[
                    {"robot_description": xacro.process_file(urdf).toxml()},
                    {"robot_description_semantic": srdf_text},
                    {"traj_file": traj_in, "task_file": task, "out_file": problem_out},
                ],
            )
        return ExecuteProcess(
            cmd=[VAMP_PYTHON, os.path.join(PKG_DIR, "scripts", "collision_generator_vamp.py"),
                 "--traj", traj_in, "--task", task, "--out", problem_out],
            output="screen",
        )

    def solve_stage(problem_in, solution_out):
        cmd = [SOLVER_PYTHON, os.path.join(SOLVER_DIR, "solve.py"), problem_in, solution_out,
               "--backend", LaunchConfiguration("backend"),
               "--time-limit", LaunchConfiguration("time_limit")]
        if turn_based:
            cmd += ["--turn-based"]
        if solver_seed:
            cmd += ["--seed", solver_seed]
        if solver_objective in ("work", "apex"):
            cmd += ["--objective", solver_objective, "--balance-weight", balance_weight]
        return ExecuteProcess(cmd=cmd, output="screen")

    def refine_stage(mode, out):
        return ExecuteProcess(
            cmd=[VAMP_PYTHON, os.path.join(PKG_DIR, "scripts", "refine_yield.py"), mode,
                 "--traj", traj, "--solution", solution, "--task", task,
                 "--tpg", os.path.join(out_dir, "tpg.json"),
                 "--test", refine_test, "--rungs", refine_rungs,
                 "--spec", transit_spec, "--transits", transits, "--out", out],
            output="screen",
        )

    # -- stage 1: trajectories ------------------------------------------------- #
    stage_traj = motion_node({"out_file": traj})

    # -- stage 2: collision analysis ------------------------------------------- #
    stage_coll = collision_stage(traj, problem)

    # -- stage 3: solve -------------------------------------------------------- #
    stage_solve = solve_stage(problem, solution)

    # -- stages 3a-3c: refinement ---------------------------------------------- #
    # Shorten each robot's commute between consecutive tasks. Split across three
    # processes because choosing where to rejoin needs `mu` (the vamp venv) while
    # planning the connecting motion needs MoveIt (the ROS runtime); see ADR-0008.
    refine_plan = refine_stage("--plan", transit_spec)
    refine_transits = motion_node({"transit_file": transit_spec, "out_file": transits})
    refine_splice = refine_stage("--splice", refined_traj)

    # -- stage 4: temporal plan graph ------------------------------------------ #
    # Turns the timed schedule into a partial order, so execution survives delay
    # (ADR-0007). Needs the schedule, hence it runs last.
    final_traj = refined_traj if refine else traj
    final_problem = refined_problem if refine else problem
    final_solution = refined_solution if refine else solution
    tpg_file = os.path.join(out_dir, "tpg_refined.json" if refine else "tpg.json")
    baseline_tpg = os.path.join(out_dir, "tpg.json")
    stage_recoll = collision_stage(refined_traj, refined_problem)
    stage_resolve = solve_stage(refined_problem, refined_solution)
    stage_tpg = ExecuteProcess(
        cmd=[VAMP_PYTHON, os.path.join(PKG_DIR, "scripts", "build_tpg.py"),
             "--traj", final_traj, "--task", task, "--solution", final_solution,
             "--problem", final_problem, "--out", tpg_file],
        output="screen",
    )

    # -- stage 5: collision-sphere overlay data (viz only) --------------------- #
    # Regenerated every run so it cannot drift from the plan it decorates: the shells are
    # computed FROM the trajectories, and task ids alone do not identify a scene (swap and
    # tower share t_box_1..4), so a stale npz silently draws another scene's arms.
    stage_spheres = ExecuteProcess(
        cmd=[VAMP_PYTHON, os.path.join(PKG_DIR, "scripts", "dump_vamp_spheres.py"),
             "--traj", final_traj, "--task", task, "--out", spheres],
        output="screen",
    )

    # The BASELINE graph is built before refinement, because refinement now READS it: its
    # partial order is what says which of the other robot's nodes a shortcut can possibly
    # meet (ADR-0008 addendum). It doubles as the comparison artifact -- `tpg.json` and
    # `tpg_refined.json` then always describe the same run. Otherwise a refining run leaves
    # the previous `tpg.json` untouched: same task names, a different plan, no way to tell
    # -- the artifact-crossing trap the executor guards already exist for.
    stage_tpg_baseline = ExecuteProcess(
        cmd=[VAMP_PYTHON, os.path.join(PKG_DIR, "scripts", "build_tpg.py"),
             "--traj", traj, "--task", task, "--solution", solution,
             "--problem", problem, "--out", baseline_tpg],
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

    def archive(ctx):
        """Copy the finished plan into artifacts/runs/<name>/, flat and self-contained.

        A plan is a TRIPLE -- trajectories, schedule, graph -- and only matching triples are
        valid (ADR-0008 crossed one pair and produced a schedule that started 1.27 rad from
        home). Copying them together under one name is what keeps a saved plan replayable
        after the next run has overwritten the working artifacts. The task YAML goes in too:
        the scene is part of the plan, and a saved plan replayed against different geometry
        is not the plan that was solved.
        """
        if not save_as:
            return []
        dest = os.path.join(ARTIFACTS_DIR, "runs", save_as)
        os.makedirs(dest, exist_ok=True)
        wanted = [(final_traj, "tamp_trajectories.json"),
                  (final_solution, "tamp_solution.json"),
                  (tpg_file, "tpg.json"),
                  (final_problem, "tamp_problem.json"),
                  (task, os.path.basename(task))]
        for src, name in wanted:
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(dest, name))
        with open(os.path.join(dest, "plan.json"), "w") as f:
            json.dump({"task": os.path.basename(task), "engine": engine, "refined": refine,
                       "policy": "turn-based" if turn_based else "parallel",
                       "objective": solver_objective,
                       "balance_weight": balance_weight, "solver_seed": solver_seed or None},
                      f, indent=2)
        return [LogInfo(msg=f"  saved as    : {dest}\n"
                            f"  replay it   : ros2 launch multi_robot_cell_tamp "
                            f"execute_schedule.launch.py mode:=tpg run:={save_as}\n")]

    done = RegisterEventHandler(
        OnProcessExit(
            target_action=stage_spheres,
            on_exit=lambda event, ctx: archive(ctx) + [
                LogInfo(msg=(
                    f"\n=== pipeline complete ({engine}"
                    f"{', refined' if refine else ''}) ===\n"
                    f"  trajectories : {final_traj}\n"
                    f"  seam         : {final_problem}\n"
                    f"  schedule     : {final_solution}\n"
                    f"  plan graph   : {tpg_file}\n"
                    + (f"  baseline graph: {baseline_tpg}\n" if refine else "")
                    + f"  sphere shells: {spheres}\n\n"
                    + (f"  baseline schedule (unrefined): {solution}\n\n" if refine else "")
                    + (f"  policy: {'turn-based' if turn_based else 'parallel'}, "
                       f"objective: {solver_objective}\n\n"
                       if (turn_based or solver_objective != "makespan") else "")
                    + f"To execute it, with the cell running "
                    f"(ros2 launch multi_robot_cell_bringup start.launch.py):\n"
                    f"  ros2 launch multi_robot_cell_tamp execute_schedule.launch.py \\\n"
                    f"      task_file:={task} solution_file:={final_solution}\n"
                )) if event.returncode == 0 else
                LogInfo(msg=f"\n*** pipeline aborted: sphere overlay data failed "
                            f"(exit {event.returncode}) ***\n"),
            ],
        )
    )

    stages = [stage_traj, stage_coll, stage_solve]
    labels = ["trajectory generation", "collision analysis", "solve"]
    if refine:
        # The baseline graph comes FIRST now: refinement reads its partial order.
        stages += [stage_tpg_baseline, refine_plan, refine_transits, refine_splice,
                   stage_recoll, stage_resolve]
        labels += ["baseline TPG", "yield-pose selection", "transit planning",
                   "shortcut verification", "collision analysis (refined)",
                   "solve (refined)"]
    stages.append(stage_tpg)
    labels.append("TPG construction")
    stages.append(stage_spheres)
    labels.append("sphere overlay data")

    return [
        LogInfo(msg=f"\n=== TAMP pipeline: task={os.path.basename(task)} engine={engine}"
                    f"{' refine' if refine else ''} ===\n"),
        stages[0],
        *[chain(a, b, lbl) for a, b, lbl in zip(stages, stages[1:], labels)],
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
                "refine", default_value="true", choices=["true", "false"],
                description="Shorten each robot's commute between consecutive tasks after "
                            "solving, then rebuild the seam, schedule and graph (ADR-0008). "
                            "false keeps the unrefined plan as the baseline."),
            DeclareLaunchArgument(
                "schedule", default_value="parallel", choices=["parallel", "turn_based"],
                description="Scheduling policy. parallel: this work -- concurrency is "
                            "decided by the solver against the collision constraints. "
                            "turn_based: APEX-MR's -- one arm moves at a time and "
                            "concurrency is recovered afterwards by the graph."),
            DeclareLaunchArgument(
                "objective", default_value="makespan",
                choices=["makespan", "work", "apex"],
                description="What the solver minimises. makespan: this work. work: "
                            "APEX-MR's -- sum of assigned durations plus load balancing, "
                            "then packed lexicographically. Independent of `schedule`, so "
                            "their objective can run under parallel scheduling."),
            DeclareLaunchArgument(
                "solver_seed", default_value="",
                description="CP-SAT random seed. Empty: the backend default. It perturbs the "
                            "search but does not by itself decide which of several tied "
                            "optima comes back -- the solver's 8-worker portfolio does, so "
                            "even a fixed seed varies run to run. Measure tie-break "
                            "sensitivity by repeating a run, not by sweeping this."),
            DeclareLaunchArgument(
                "balance_weight", default_value="1",
                description="turn_based only: weight of the load-balancing term. 0 puts "
                            "every task on one arm, leaving the graph nothing to overlap."),
            DeclareLaunchArgument(
                "refine_test", default_value="window", choices=["window", "whole"],
                description="Ablation: what a candidate shortcut is checked against. "
                            "window: the independence window of the span it replaces "
                            "(APEX-MR's test). whole: the other robot's entire plan."),
            DeclareLaunchArgument(
                "refine_rungs", default_value="16",
                description="Ablation: candidate shortcuts per splice, boldest first."),
            DeclareLaunchArgument(
                "save_as", default_value="",
                description="Archive the finished plan under artifacts/runs/<name>/ "
                            "(trajectories + schedule + graph + task YAML, together). "
                            "Empty: don't. Needed to compare two plans, since every run "
                            "otherwise overwrites the working artifacts."),
            DeclareLaunchArgument(
                "backend", default_value="cp-sat", choices=["cp-sat", "gurobi"],
                description="Solver backend for the allocation stage."),
            DeclareLaunchArgument(
                "time_limit", default_value="120.0",
                description="Solver time limit in seconds."),
            OpaqueFunction(function=launch_setup),
        ]
    )
