#!/usr/bin/env python3
"""Regression check for the collision and solve stages on ARCHIVED trajectories.

    python3 scripts/regress_seam.py <run_dir> [<run_dir> ...] [--pkg DIR] [--solver-dir DIR]

A run directory is what `pipeline.launch.py save_as:=<name>` writes: `tamp_trajectories.json`,
`tamp_task_<scene>.yaml` (+ `meshes/`), `tamp_problem.json`, `tamp_solution.json`. The trajectory
generator is not deterministic (OMPL is unseeded), so a regression of the LATER stages has to start
from the archived trajectories: this re-runs `collision_generator_vamp.py` on them, compares the
new seam with the archived one (sha256 of the canonical JSON, and which keys differ), then solves
the new seam with `solve.py` and compares the makespan and the solver status.

Interpreters (never mixed, CONTEXT.md): the collision stage runs under `<pkg>/.venv_vamp`, the
solve under `<solver-dir>/.venv`; this driver itself is stdlib only (any python3). Nothing is
written outside a temporary directory unless `--keep DIR` is given.

Exit status 0 when every run reproduces seam and makespan, 1 otherwise.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
WS = os.path.dirname(os.path.dirname(os.path.dirname(PKG)))


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def digest(obj):
    return hashlib.sha256(canon(obj).encode()).hexdigest()


def diff_keys(a, b):
    keys = sorted(set(a) | set(b))
    return [k for k in keys if canon(a.get(k)) != canon(b.get(k))]


def check(run, pkg, solver_dir, out_dir):
    traj = os.path.join(run, "tamp_trajectories.json")
    yamls = glob.glob(os.path.join(run, "tamp_task_*.yaml"))
    if not os.path.exists(traj) or len(yamls) != 1:
        return False, f"{run}: needs tamp_trajectories.json and exactly one tamp_task_*.yaml"
    old_seam = json.load(open(os.path.join(run, "tamp_problem.json")))
    old_sol = json.load(open(os.path.join(run, "tamp_solution.json")))
    seam = os.path.join(out_dir, "tamp_problem.json")
    sol = os.path.join(out_dir, "tamp_solution.json")
    vamp_py = os.path.join(pkg, ".venv_vamp", "bin", "python")
    r = subprocess.run([vamp_py, os.path.join(pkg, "scripts", "collision_generator_vamp.py"),
                        "--traj", traj, "--task", yamls[0], "--out", seam],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False, f"collision stage failed:\n{r.stderr[-2000:]}"
    new_seam = json.load(open(seam))
    same_seam = digest(new_seam) == digest(old_seam)
    r = subprocess.run([os.path.join(solver_dir, ".venv", "bin", "python"),
                        os.path.join(solver_dir, "solve.py"), seam, sol],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False, f"solve failed:\n{r.stderr[-2000:]}"
    new_sol = json.load(open(sol))
    same_ms = (new_sol["makespan_slots"] == old_sol["makespan_slots"]
               and new_sol["status"] == old_sol["status"])
    lines = [f"seam {'same' if same_seam else 'DIFFERENT'} "
             f"({digest(new_seam)[:12]} vs {digest(old_seam)[:12]})"
             + ("" if same_seam else f", keys differing: {diff_keys(new_seam, old_seam)}"),
             f"makespan {new_sol['makespan_slots']} ({new_sol['status']}) vs "
             f"{old_sol['makespan_slots']} ({old_sol['status']}) -> {'same' if same_ms else 'DIFFERENT'}"]
    return same_seam and same_ms, "\n    ".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--pkg", default=PKG, help="multi_robot_cell_tamp source dir (scripts, .venv_vamp)")
    ap.add_argument("--solver-dir", default=os.path.join(WS, "thesis_material_tamp"))
    ap.add_argument("--keep", default=None, help="write the new seams/solutions under this dir")
    args = ap.parse_args()
    ok_all = True
    for run in args.runs:
        name = os.path.basename(os.path.normpath(run))
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(args.keep, name) if args.keep else tmp
            os.makedirs(out, exist_ok=True)
            ok, msg = check(run, args.pkg, args.solver_dir, out)
        ok_all &= ok
        print(f"[{'OK' if ok else 'FAIL'}] {name}\n    {msg}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
