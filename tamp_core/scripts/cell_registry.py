"""Which robot cells exist, and where their files and scenes live.

tamp_core knows no cell by name. A cell is a directory next to ``tamp_core`` (dual_robot_cell,
four_robot_cell, tiago_cell, ...) that carries a ``cell.yaml`` descriptor: its MoveIt files, its
layout, its ``scenes/`` and its VAMP modules (see four_robot_cell/cell.yaml for the schema). The
scene's top-level ``cell:`` key names one; absent means ``dual``.

Found from the SOURCE tree (the three interpreters -- ROS Python, solver venv, VAMP venv -- share
no ament index), so this module is stdlib + PyYAML only and every interpreter can import it.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

DEFAULT_CELL = "dual"
# .../src/multi_robot_cell/tamp_core/scripts/cell_registry.py -> .../src/multi_robot_cell
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


@dataclass
class Cell:
    name: str
    dir: str
    spec: dict

    def path(self, rel: str) -> str:
        return os.path.join(self.dir, rel)

    @property
    def package(self) -> str:
        return self.spec["package"]

    def moveit(self, key: str) -> Optional[str]:
        """Absolute path of a MoveIt file of the cell (``urdf``, ``srdf``, ...), None if absent."""
        rel = (self.spec.get("moveit") or {}).get(key)
        return self.path(rel) if rel else None

    @property
    def scenes_dir(self) -> Optional[str]:
        rel = self.spec.get("scenes")
        return self.path(rel) if rel else None

    @property
    def layout_path(self) -> Optional[str]:
        rel = self.spec.get("layout")
        return self.path(rel) if rel else None

    def layout(self) -> Dict:
        if not self.layout_path or not os.path.exists(self.layout_path):
            raise FileNotFoundError(f"cell '{self.name}': no layout at {self.layout_path}")
        with open(self.layout_path) as f:
            return yaml.safe_load(f)



def cells() -> Dict[str, Cell]:
    out: Dict[str, Cell] = {}
    for f in sorted(glob.glob(os.path.join(REPO_DIR, "*", "cell.yaml"))):
        with open(f) as fh:
            spec = yaml.safe_load(fh)
        out[spec["name"]] = Cell(spec["name"], os.path.dirname(f), spec)
    return out


def get_cell(name: str) -> Cell:
    known = cells()
    if name not in known:
        raise RuntimeError(f"unknown cell '{name}' (known: {', '.join(known)})")
    return known[name]


def cell_of_task(task_file: str) -> str:
    """The ``cell:`` of a task YAML ('dual' when absent), validated against the registry."""
    with open(task_file) as f:
        data = yaml.safe_load(f) or {}
    name = str(data.get("cell") or DEFAULT_CELL)
    get_cell(name)
    return name


def scene_dirs() -> List[str]:
    return [c.scenes_dir for c in cells().values() if c.scenes_dir and os.path.isdir(c.scenes_dir)]


def scene_path(name: str, cell: str = DEFAULT_CELL) -> str:
    """A scene shorthand (``tower`` -> tamp_task_tower.yaml, ``nominal`` -> tamp_task.yaml) or a
    path. A path (separator or .yaml) is taken as given. A shorthand is looked up in ``cell``'s
    ``scenes/`` first (the dual cell by default: ``nominal`` and ``numbers`` exist for several
    cells), then in every other cell's, where it must be unique."""
    if os.sep in name or name.endswith((".yaml", ".yml")):
        return os.path.abspath(name)
    stem = "tamp_task.yaml" if name == "nominal" else f"tamp_task_{name}.yaml"
    known = cells()
    own = known[cell].scenes_dir if cell in known else None
    if own and os.path.isfile(os.path.join(own, stem)):
        return os.path.join(own, stem)
    hits = [os.path.join(d, stem) for d in scene_dirs() if os.path.isfile(os.path.join(d, stem))]
    if not hits:
        raise FileNotFoundError(f"scene '{name}': {stem} not in any cell's scenes/ ({scene_dirs()})")
    if len(hits) > 1:
        raise RuntimeError(f"scene '{name}' is ambiguous: {hits}")
    return hits[0]


def module_spec(name: str) -> Dict:
    """A VAMP module's spec, searched in every cell's ``vamp.modules`` (a cell may use another
    cell's module, e.g. fabricator4's gripper robots use dual_robot_cell's ur10e_rail). Keys:
    ``dir`` (absolute), ``urdf`` (the spherized URDF, absolute; default
    ``<dir>/inputs/<name>_spherized.urdf``), ``n_structural`` (leading spheres dropped from mu,
    default 0), ``ee_in_eefk`` (xyz of the scene's ee_link in the module's eefk frame, absent =
    the UR default)."""
    for c in cells().values():
        m = ((c.spec.get("vamp") or {}).get("modules") or {}).get(name)
        if m is not None:
            d = c.path(m["dir"])
            return {**m, "dir": d, "n_structural": int(m.get("n_structural", 0)),
                    "urdf": os.path.join(d, m.get("urdf", f"inputs/{name}_spherized.urdf"))}
    raise KeyError(f"no cell declares VAMP module '{name}'")
