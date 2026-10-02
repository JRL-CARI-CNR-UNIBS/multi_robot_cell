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

A SHARED TRAJECTORY POOL (paired comparisons)
---------------------------------------------
Because trajectories differ run to run, comparing two solvers fairly means solving the
SAME trajectories and seam with both. Stages 1-2 are the expensive, random part; they
can be run once and reused:

    ... task:=tower stop_after:=seam save_as:=pool1 artifacts_dir:=/tmp/cmp
    ... from_pool:=/tmp/cmp/runs/pool1 save_as:=A1 schedule:=turn_based ... artifacts_dir:=/tmp/cmp
    ... from_pool:=/tmp/cmp/runs/pool1 save_as:=B1 artifacts_dir:=/tmp/cmp

``from_pool`` copies the pool's trajectories, seam and scene (with its meshes) into
``artifacts_dir`` and starts at the solve; ``plan.json`` records the sha256 of what it
copied. The pool's scene is authoritative: a ``task:=`` naming a different scene is an
error. ``artifacts_dir`` moves every file the pipeline writes (and ``runs/``) elsewhere,
so a campaign never touches the working ``artifacts/``. Every run writes
``<artifacts_dir>/stage_times.json`` (wall-clock start/end of each stage), archived with
the run.
"""

import hashlib
import json
import os
import shutil
import sys
import time

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from cell_moveit import cell_of_task, description_and_semantic, moveit_config_for, robot_count, scene_path  # noqa: E402

# realpath() resolves the --symlink-install symlink back to the SOURCE tree, so
# artifacts land next to the package rather than inside install/.
PKG_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ARTIFACTS_DIR = os.path.join(PKG_DIR, "artifacts")
WS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(PKG_DIR)))

VAMP_PYTHON = os.path.join(PKG_DIR, ".venv_vamp", "bin", "python")
SOLVER_DIR = os.path.join(WS_DIR, "thesis_material_tamp")
SOLVER_PYTHON = os.path.join(SOLVER_DIR, ".venv", "bin", "python")


def resolve_task(name: str) -> str:
    """A scene shorthand (``tower`` -> tamp_task_tower.yaml, ``nominal`` -> tamp_task.yaml, looked
    up in every cell's ``scenes/``) or a path to a task YAML."""
    return scene_path(name)


def scene_mesh_files(task_yaml):
    """[(relative_path, source_path)] for every ``mesh: {file: ...}`` in a scene YAML.

    A fixture or object may carry an STL (``mesh:``, origin at its bounding-box centre).
    ``file`` is relative to the YAML's own folder and, failing that, to the installed
    ``config/`` -- the rule ``trajectory_generator`` and ``scene_visualizer`` share -- so the
    SOURCE is whichever of the two exists. Duplicates (one STL, many objects) collapse.

    Raises ``RuntimeError`` for a file that is nowhere to be found, or whose relative path
    climbs out of the folder (``../x.stl``) and so cannot be reproduced under a run's folder.
    An absolute path is left alone: it resolves the same wherever the YAML sits, so there is
    nothing to copy.
    """
    with open(task_yaml) as f:
        spec = yaml.safe_load(f) or {}
    here = os.path.dirname(os.path.realpath(task_yaml))
    out, seen = [], set()
    for kind in ("fixtures", "objects"):
        for entry in spec.get(kind) or []:
            m = entry.get("mesh")
            name = m if isinstance(m, str) else (m or {}).get("file")
            if not name or os.path.isabs(name) or name in seen:
                continue
            seen.add(name)
            norm = os.path.normpath(name)
            if norm.startswith(os.pardir + os.sep) or norm == os.pardir:
                raise RuntimeError(
                    f"{kind[:-1]} '{entry.get('id')}': mesh file '{name}' climbs out of the "
                    f"scene's folder, so a saved run cannot carry it at the same relative "
                    f"path. Put the mesh under the YAML's folder (e.g. meshes/).")
            candidates = [os.path.join(here, norm)]
            try:
                candidates.append(os.path.join(
                    get_package_share_directory("multi_robot_cell_tamp"), "config", norm))
            except Exception:  # noqa: BLE001
                pass
            src = next((c for c in candidates if os.path.isfile(c)), None)
            if src is None:
                raise RuntimeError(
                    f"{kind[:-1]} '{entry.get('id')}': mesh '{name}' not found; looked in "
                    f"{', '.join(candidates)}")
            out.append((norm, src))
    return out


def copy_scene_meshes(task_yaml, dest):
    """Copy the scene's mesh files under ``dest`` keeping their relative paths.

    ``artifacts/runs/<name>/`` then holds ``tamp_task_x.yaml`` and ``meshes/x.stl`` side by
    side, which is where ``execute_schedule.launch.py run:=<name>`` (whose task YAML is the
    one in that folder) resolves them from. Returns the relative paths copied.
    """
    copied = []
    for rel, src in scene_mesh_files(task_yaml):
        target = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(src, target)
        copied.append(rel)
    return copied


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pool(pool_dir, engine, task_arg, artifacts_dir, traj_dst, problem_dst):
    """Copy a pool's trajectories, seam and scene into ``artifacts_dir``.

    A pool is a run archived by ``save_as`` from an UNREFINED plan with the same engine --
    normally one made with ``stop_after:=seam``. Its seam sits at ``tamp_problem.json`` in
    the run's root (``archive()`` flattens ``<engine>/``). Returns ``(task_yaml, record)``:
    the copied scene YAML, which every later stage reads, and the ``pool`` entry for
    ``plan.json`` (source path plus sha256 of each copied file, checked against the
    source after copying).
    """
    if not os.path.isdir(pool_dir):
        raise RuntimeError(f"from_pool: no such folder: {pool_dir}")
    meta_path = os.path.join(pool_dir, "plan.json")
    if not os.path.isfile(meta_path):
        raise RuntimeError(f"from_pool: {pool_dir} has no plan.json, so it is not a run "
                           f"archived by save_as")
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("refined"):
        raise RuntimeError(f"from_pool: {pool_dir} is a REFINED run -- its trajectories carry "
                           f"one schedule's transits and are not a pool. Use an unrefined "
                           f"run (stop_after:=seam, or refine:=false).")
    if meta.get("engine", engine) != engine:
        raise RuntimeError(f"from_pool: the pool's seam was computed by "
                           f"engine:={meta.get('engine')}, this run asks for engine:={engine}")
    yamls = sorted(y for y in os.listdir(pool_dir) if y.endswith((".yaml", ".yml")))
    if len(yamls) != 1:
        raise RuntimeError(f"from_pool: expected exactly one scene YAML in {pool_dir}, "
                           f"found {yamls or 'none'}")
    pool_yaml = os.path.join(pool_dir, yamls[0])
    sources = {"trajectories": (os.path.join(pool_dir, "tamp_trajectories.json"), traj_dst),
               "seam": (os.path.join(pool_dir, "tamp_problem.json"), problem_dst)}
    for what, (src, _) in sources.items():
        if not os.path.isfile(src):
            raise RuntimeError(f"from_pool: {pool_dir} has no {os.path.basename(src)} ({what})")

    # The pool's scene is authoritative: its trajectories were planned against it. A task:=
    # naming another scene is a mistake worth stopping for, not something to guess about.
    if task_arg:
        asked = resolve_task(task_arg)
        if os.path.basename(asked) != yamls[0]:
            raise RuntimeError(f"from_pool: task:={task_arg} ({os.path.basename(asked)}) is not "
                               f"the pool's scene ({yamls[0]}). Drop task:= -- the pool "
                               f"decides the scene.")
        if os.path.isfile(asked) and sha256_of(asked) != sha256_of(pool_yaml):
            print(f"[pipeline] WARNING: {asked} differs from the pool's copy of {yamls[0]}; "
                  f"using the pool's, which the trajectories were planned against.")

    if os.path.realpath(pool_dir) == os.path.realpath(artifacts_dir):
        raise RuntimeError("from_pool: the pool folder cannot be artifacts_dir itself")
    os.makedirs(artifacts_dir, exist_ok=True)
    task_yaml = os.path.join(artifacts_dir, yamls[0])
    shutil.copyfile(pool_yaml, task_yaml)
    meshes = copy_scene_meshes(pool_yaml, artifacts_dir)
    record = {"path": os.path.realpath(pool_dir), "scene": yamls[0],
              "scene_sha256": sha256_of(task_yaml)}
    for what, (src, dst) in sources.items():
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        digest = sha256_of(dst)
        if digest != sha256_of(src):
            raise RuntimeError(f"from_pool: copy of {src} does not match its source")
        record[what] = {"file": os.path.basename(src), "sha256": digest}
    print(f"[pipeline] from_pool: {pool_dir} -> {artifacts_dir} "
          f"(trajectories {record['trajectories']['sha256'][:12]}, "
          f"seam {record['seam']['sha256'][:12]}, {len(meshes)} mesh file(s))")
    return task_yaml, record


def launch_setup(context):
    # Where every stage writes. Default: the package's artifacts/ (the working folder).
    artifacts_dir = LaunchConfiguration("artifacts_dir").perform(context).strip()
    artifacts_dir = (os.path.abspath(os.path.expanduser(artifacts_dir)) if artifacts_dir
                     else ARTIFACTS_DIR)
    stop_after = LaunchConfiguration("stop_after").perform(context).strip().lower()
    if stop_after not in ("", "seam"):
        raise RuntimeError(f"stop_after must be empty or 'seam', got {stop_after!r}")
    from_pool = LaunchConfiguration("from_pool").perform(context).strip()
    from_pool = os.path.abspath(os.path.expanduser(from_pool)) if from_pool else ""
    if from_pool and stop_after:
        raise RuntimeError("from_pool and stop_after:=seam exclude each other: a pool run "
                           "starts where stop_after:=seam stops")
    solver_workers = LaunchConfiguration("solver_workers").perform(context).strip()
    if solver_workers and not (solver_workers.isdigit() and int(solver_workers) >= 1):
        raise RuntimeError(f"solver_workers must be a positive integer, got {solver_workers!r}")
    task_arg = LaunchConfiguration("task").perform(context).strip()
    task = resolve_task(task_arg or "tower")
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
    if stop_after and refine:
        # Refinement needs a schedule; a pool has none. Not an error: refine defaults true.
        refine = False
    # EVERY path a stage reads or writes derives from artifacts_dir; ARTIFACTS_DIR is only
    # its default. One path built from it would put a campaign's output in the working folder.
    traj = os.path.join(artifacts_dir, "tamp_trajectories.json")
    out_dir = os.path.join(artifacts_dir, engine)
    problem = os.path.join(out_dir, "tamp_problem.json")
    solution = os.path.join(out_dir, "tamp_solution.json")
    pool_record = None
    if from_pool:
        # Validates the pool completely before writing anything into artifacts_dir.
        task, pool_record = load_pool(from_pool, engine, task_arg, artifacts_dir, traj, problem)
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isfile(task):
        raise RuntimeError(f"task file not found: {task}")
    if save_as:
        # Fail now, not after the pipeline: a mesh the archive cannot carry would make the
        # saved run unreplayable.
        scene_mesh_files(task)
    # Refinement rewrites the plan, so its outputs get their own names: the unrefined
    # artifacts stay on disk as the baseline the improvement is measured against.
    refined_traj = os.path.join(artifacts_dir, "tamp_trajectories_refined.json")
    refined_problem = os.path.join(out_dir, "tamp_problem_refined.json")
    refined_solution = os.path.join(out_dir, "tamp_solution_refined.json")
    transit_spec = os.path.join(artifacts_dir, "transit_spec.json")
    transits = os.path.join(artifacts_dir, "transits.json")
    spheres = os.path.join(out_dir, "tamp_spheres_refined.npz" if refine else "tamp_spheres.npz")
    stage_times_file = os.path.join(artifacts_dir, "stage_times.json")

    # The cell is the scene's (`cell:` in its YAML, default dual): URDF, SRDF and MoveIt
    # files follow it (launch/cell_moveit.py).
    cell = cell_of_task(task)
    if refine and robot_count(task) > 2:
        raise RuntimeError(
            f"refine:=true needs exactly 2 robots: the commute refinement (ADR-0008, "
            f"refine_yield.py / coordinate.py) is a two-robot construction, and this scene has "
            f"{robot_count(task)}. Run it with refine:=false.")
    moveit_config = moveit_config_for(cell, context)

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
            urdf_xml, srdf_text = description_and_semantic(cell, context)
            return Node(
                package="multi_robot_cell_tamp",
                executable="collision_generator",
                output="screen",
                parameters=[
                    {"robot_description": urdf_xml},
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
        if solver_workers:
            cmd += ["--workers", solver_workers]
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

    # -- stage timing ----------------------------------------------------------- #
    # Wall-clock start/end of every stage, written after each event so an aborted run keeps
    # what it measured. The instants are those of the launch's own ProcessStarted /
    # ProcessExited events -- the same ones launch.log timestamps.
    stage_times = []
    by_action = {}

    def flush_times():
        with open(stage_times_file, "w") as f:
            json.dump(stage_times, f, indent=2)
            f.write("\n")

    def mark_start(stage, event):
        entry = {"stage": stage, "cmd_name": event.process_name, "start": time.time(),
                 "end": None, "seconds": None, "exit_code": None}
        stage_times.append(entry)
        by_action[id(event.action)] = entry
        flush_times()

    def mark_exit(event):
        """Idempotent: both the timing handler and the chaining one call it."""
        entry = by_action.get(id(event.action))
        if entry is None or entry["end"] is not None:
            return
        entry["end"] = time.time()
        entry["seconds"] = round(entry["end"] - entry["start"], 3)
        entry["exit_code"] = event.returncode
        flush_times()

    def chain(previous, nxt, label):
        """Run ``nxt`` when ``previous`` exits 0; abort the pipeline otherwise."""
        def on_exit(event, ctx):
            mark_exit(event)
            if event.returncode == 0:
                return [nxt]
            return [LogInfo(msg=f"\n*** pipeline aborted: {label} failed "
                                f"(exit {event.returncode}) ***\n"),
                    Shutdown(reason=f"{label} failed")]
        return RegisterEventHandler(OnProcessExit(target_action=previous, on_exit=on_exit))

    def timed(action, stage):
        return [
            RegisterEventHandler(OnProcessStart(
                target_action=action, on_start=lambda event, ctx: mark_start(stage, event))),
            RegisterEventHandler(OnProcessExit(
                target_action=action, on_exit=lambda event, ctx: mark_exit(event))),
        ]

    seam_only = stop_after == "seam"

    def archive(ctx):
        """Copy the finished plan into <artifacts_dir>/runs/<name>/, flat and self-contained.

        A plan is a TRIPLE -- trajectories, schedule, graph -- and only matching triples are
        valid (ADR-0008 crossed one pair and produced a schedule that started 1.27 rad from
        home). Copying them together under one name is what keeps a saved plan replayable
        after the next run has overwritten the working artifacts. The task YAML goes in too:
        the scene is part of the plan, and a saved plan replayed against different geometry
        is not the plan that was solved. So do its mesh files, if it has any.

        With ``stop_after:=seam`` the run is a POOL: trajectories + seam + scene only. The
        schedule and graph are left out on purpose -- whatever sits in artifacts_dir from an
        earlier run does not belong to these trajectories.
        """
        if not save_as:
            return []
        dest = os.path.join(artifacts_dir, "runs", save_as)
        os.makedirs(dest, exist_ok=True)
        wanted = [(final_traj, "tamp_trajectories.json"),
                  (final_problem, "tamp_problem.json"),
                  (task, os.path.basename(task)),
                  (stage_times_file, "stage_times.json")]
        if not seam_only:
            wanted += [(final_solution, "tamp_solution.json"), (tpg_file, "tpg.json")]
        for src, name in wanted:
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(dest, name))
        # The scene's STL meshes, at the same relative path: the copied YAML resolves
        # `mesh: file:` against ITS folder, so the run must carry them.
        meshes = copy_scene_meshes(task, dest)
        meta = {"task": os.path.basename(task), "engine": engine, "refined": refine,
                "stage": "seam" if seam_only else "full"}
        if not seam_only:
            meta.update({"policy": "turn-based" if turn_based else "parallel",
                         "objective": solver_objective,
                         "balance_weight": balance_weight, "solver_seed": solver_seed or None,
                         "solver_workers": int(solver_workers) if solver_workers else None})
        if pool_record is not None:
            meta["pool"] = pool_record
        with open(os.path.join(dest, "plan.json"), "w") as f:
            json.dump(meta, f, indent=2)
        if seam_only:
            return [LogInfo(msg=f"  pool saved  : {dest}\n"
                                f"  solve it    : ros2 launch multi_robot_cell_tamp "
                                f"pipeline.launch.py from_pool:={dest}"
                                + (f" artifacts_dir:={artifacts_dir}"
                                   if artifacts_dir != ARTIFACTS_DIR else "") + "\n")]
        replay = (f"run:={save_as}" if artifacts_dir == ARTIFACTS_DIR else
                  f"task_file:={os.path.join(dest, os.path.basename(task))} "
                  f"traj_file:={os.path.join(dest, 'tamp_trajectories.json')} "
                  f"solution_file:={os.path.join(dest, 'tamp_solution.json')} "
                  f"tpg_file:={os.path.join(dest, 'tpg.json')}")
        return [LogInfo(msg=f"  saved as    : {dest}"
                            + (f" (+ {len(meshes)} mesh file(s))" if meshes else "") + "\n"
                            f"  replay it   : ros2 launch multi_robot_cell_tamp "
                            f"execute_schedule.launch.py mode:=tpg {replay}\n")]

    def summary():
        if seam_only:
            return (f"\n=== pipeline stopped after the seam ({engine}) ===\n"
                    f"  trajectories : {traj}\n"
                    f"  seam         : {problem}\n"
                    f"  stage times  : {stage_times_file}\n")
        return (f"\n=== pipeline complete ({engine}"
                f"{', refined' if refine else ''}) ===\n"
                + (f"  pool         : {from_pool}\n" if from_pool else "")
                + f"  trajectories : {final_traj}\n"
                f"  seam         : {final_problem}\n"
                f"  schedule     : {final_solution}\n"
                f"  plan graph   : {tpg_file}\n"
                + (f"  baseline graph: {baseline_tpg}\n" if refine else "")
                + f"  sphere shells: {spheres}\n"
                f"  stage times  : {stage_times_file}\n\n"
                + (f"  baseline schedule (unrefined): {solution}\n\n" if refine else "")
                + (f"  policy: {'turn-based' if turn_based else 'parallel'}, "
                   f"objective: {solver_objective}\n\n"
                   if (turn_based or solver_objective != "makespan") else "")
                + f"To execute it, with the cell running "
                f"(ros2 launch multi_robot_cell_bringup start.launch.py):\n"
                f"  ros2 launch multi_robot_cell_tamp execute_schedule.launch.py \\\n"
                f"      task_file:={task} solution_file:={final_solution}\n")

    # (action, stable stage name for stage_times.json, human label for abort messages)
    plan = []
    if not from_pool:
        plan += [(stage_traj, "trajectories", "trajectory generation"),
                 (stage_coll, "collisions", "collision analysis")]
    if not seam_only:
        plan.append((stage_solve, "solve", "solve"))
        if refine:
            # The baseline graph comes FIRST now: refinement reads its partial order.
            plan += [(stage_tpg_baseline, "tpg_baseline", "baseline TPG"),
                     (refine_plan, "refine_plan", "yield-pose selection"),
                     (refine_transits, "refine_transits", "transit planning"),
                     (refine_splice, "refine_splice", "shortcut verification"),
                     (stage_recoll, "recollisions", "collision analysis (refined)"),
                     (stage_resolve, "resolve", "solve (refined)")]
        plan += [(stage_tpg, "tpg", "TPG construction"),
                 (stage_spheres, "spheres", "sphere overlay data")]
    stages = [a for a, _, _ in plan]
    labels = [lbl for _, _, lbl in plan]

    def on_last_exit(event, ctx):
        mark_exit(event)
        if event.returncode != 0:
            return [LogInfo(msg=f"\n*** pipeline aborted: {labels[-1]} failed "
                                f"(exit {event.returncode}) ***\n")]
        return archive(ctx) + [LogInfo(msg=summary())]

    done = RegisterEventHandler(OnProcessExit(target_action=stages[-1], on_exit=on_last_exit))

    flush_times()  # a fresh, empty record: never a previous run's
    return [
        LogInfo(msg=f"\n=== TAMP pipeline: task={os.path.basename(task)} engine={engine}"
                    f"{' refine' if refine else ''}"
                    f"{' stop_after=seam' if seam_only else ''}"
                    f"{' from_pool=' + from_pool if from_pool else ''}"
                    f" artifacts_dir={artifacts_dir} ===\n"),
        *[h for action, stage, _ in plan for h in timed(action, stage)],
        stages[0],
        *[chain(a, b, lbl) for a, b, lbl in zip(stages, stages[1:], labels)],
        done,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "task", default_value="",
                description="Scene: a shorthand (tower, swap, nominal) resolved against "
                            "config/, or a path to a task YAML. Empty: tower -- or, with "
                            "from_pool, the pool's scene (a different scene is an error)."),
            DeclareLaunchArgument(
                "artifacts_dir", default_value="",
                description="Folder every stage writes into, with save_as runs under its "
                            "runs/. Empty: the package's working artifacts/. Point a "
                            "campaign elsewhere so it never touches the working folder."),
            DeclareLaunchArgument(
                "stop_after", default_value="", choices=["", "seam"],
                description="seam: run only the trajectory and collision stages (a "
                            "trajectory POOL), then archive with save_as. Empty: the "
                            "whole pipeline."),
            DeclareLaunchArgument(
                "from_pool", default_value="",
                description="Folder of a saved unrefined run (normally stop_after:=seam): "
                            "copy its trajectories, seam and scene (with meshes) into "
                            "artifacts_dir and start at the solve. plan.json records their "
                            "sha256 under `pool`. Empty: plan from scratch."),
            DeclareLaunchArgument(
                "solver_workers", default_value="",
                description="CP-SAT workers for both solves (solve.py --workers). 1 makes "
                            "the returned optimum deterministic. Empty: the backend "
                            "default (8)."),
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
                description="Archive the finished plan under <artifacts_dir>/runs/<name>/ "
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
