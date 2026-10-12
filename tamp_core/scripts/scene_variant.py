#!/usr/bin/env python3
"""Write a copy of a scene YAML with some `planning:` keys set, for A/B runs that must not edit
the tracked scene files.

    python3 scripts/scene_variant.py fabricator OUT_DIR/fab_p2.yaml \\
        --set unwrap_joints=true --set ik_nearest=8 --set direct_first=true --set shortcut=200
    ros2 launch multi_robot_cell_tamp pipeline.launch.py task:=OUT_DIR/fab_p2.yaml ...

The scene is a shorthand (resolved like `pipeline.launch.py task:=`, `--cell` as there) or a
path. The copy is the original TEXT with each key's line replaced inside the `planning:` block,
or appended at its end, so comments and layout survive. Every mesh folder the scene references
(`mesh: {file: meshes/...}`) is symlinked next to the copy, so the generator, the visualiser and
`save_as` resolve them as they would beside the original. The copy is re-parsed and checked: the
set keys must read back with the given values and nothing else may change.

`--set` values are YAML scalars (true, 8, 0.5, RRTConnect). `--unset key` drops a key.
Standard library + PyYAML (the ROS interpreter has it).
"""
import argparse
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from cell_registry import scene_path  # noqa: E402


def edit_planning(text, sets, unsets):
    lines = text.splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines) if re.match(r"^planning:\s*(#.*)?$", l)), None)
    if start is None:
        raise SystemExit("no top-level `planning:` block")
    end = start + 1
    indent = None
    while end < len(lines):
        l = lines[end]
        if l.strip() == "" or l.lstrip().startswith("#"):
            end += 1
            continue
        if not l[0].isspace():
            break
        if indent is None:
            indent = l[: len(l) - len(l.lstrip())]
        end += 1
    indent = indent or "  "
    # trailing blank/comment lines belong to whatever follows
    last = end
    while last > start + 1 and (lines[last - 1].strip() == "" or lines[last - 1].lstrip().startswith("#")):
        last -= 1
    body = lines[start + 1:last]
    done = set()
    out = []
    for l in body:
        m = re.match(r"^(\s+)([A-Za-z_][A-Za-z0-9_]*)\s*:", l)
        if m and len(m.group(1)) == len(indent):
            k = m.group(2)
            if k in unsets:
                continue
            if k in sets:
                out.append(f"{indent}{k}: {sets[k]}   # scene_variant\n")
                done.add(k)
                continue
        out.append(l)
    for k, v in sets.items():
        if k not in done:
            out.append(f"{indent}{k}: {v}   # scene_variant\n")
    return "".join(lines[:start + 1] + out + lines[last:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("out")
    ap.add_argument("--cell", default="dual")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--unset", action="append", default=[], metavar="KEY")
    a = ap.parse_args()

    src = scene_path(a.scene, a.cell)
    sets = {}
    for kv in a.set:
        k, sep, v = kv.partition("=")
        if not sep or not k:
            raise SystemExit(f"--set wants KEY=VALUE, got {kv!r}")
        sets[k.strip()] = v.strip()
    with open(src) as f:
        text = f.read()
    new = edit_planning(text, sets, set(a.unset))

    old_doc = yaml.safe_load(text)
    new_doc = yaml.safe_load(new)
    want = dict(old_doc["planning"])
    for k in a.unset:
        want.pop(k, None)
    for k, v in sets.items():
        want[k] = yaml.safe_load(v)
    if new_doc["planning"] != want:
        raise SystemExit(f"planning block did not read back as intended: {new_doc['planning']} vs {want}")
    rest_old = {k: v for k, v in old_doc.items() if k != "planning"}
    rest_new = {k: v for k, v in new_doc.items() if k != "planning"}
    if rest_old != rest_new:
        raise SystemExit("the edit touched something outside `planning:`")

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# scene_variant of {src}\n")
        f.write(new)
    here = os.path.dirname(os.path.realpath(src))
    tops = set()
    for kind in ("fixtures", "objects"):
        for e in old_doc.get(kind) or []:
            m = e.get("mesh")
            name = m if isinstance(m, str) else (m or {}).get("file")
            if name and not os.path.isabs(name):
                tops.add(os.path.normpath(name).split(os.sep)[0])
    for t in sorted(tops):
        link = os.path.join(os.path.dirname(out), t)
        target = os.path.join(here, t)
        if os.path.islink(link) and os.path.realpath(link) == os.path.realpath(target):
            continue
        if os.path.exists(link):
            raise SystemExit(f"{link} exists and is not a link to {target}")
        os.symlink(target, link)
    print(f"{out}  (from {src}; planning {sets or ''}{' unset ' + str(a.unset) if a.unset else ''}; "
          f"mesh dirs linked: {sorted(tops) or 'none'})")


if __name__ == "__main__":
    main()
