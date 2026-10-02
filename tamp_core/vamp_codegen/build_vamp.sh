#!/usr/bin/env bash
# Build VAMP with this cell's robot modules into tamp_core/.venv_vamp -- everything
# from this repository, nothing from an external source tree.
#
#   ./vamp_codegen/build_vamp.sh                 # from tamp_core/
#   JOBS=2 ./vamp_codegen/build_vamp.sh          # more parallel compile jobs (more RAM)
#   EXTRA_HEADERS=/path/dir ./vamp_codegen/build_vamp.sh   # extra robot headers (*.hh) to register
#
# What it does:
#   1. clones KavrakiLab/vamp at the pinned release (v0.6.4, commit 8fd768f) into
#      vamp_codegen/.vamp_src (gitignored), or reuses it -- VAMP itself is used UNMODIFIED;
#   2. copies every cell's robot headers (<cell>/vamp_modules/<module>/<module>.hh: dual_robot_cell
#      ur10e_rail = UR10e + rail + Robotiq 2F-85; four_robot_cell ur10e_rail_torch = UR10e + rail
#      + MIG torch; tiago_cell's arms) into its robots/ directory, plus any EXTRA_HEADERS;
#   3. registers them in its pyproject.toml (VAMP_ROBOT_MODULES / VAMP_ROBOT_STRUCTS; the
#      header stem is the module name, `struct <name>` the struct);
#   4. builds a WHEEL first; only if that succeeds, backs up the installed vamp package and
#      installs the wheel into .venv_vamp (--no-deps: numpy/PyYAML stay as they are), then
#      imports every module and prints its sphere count.
#
# The headers are regenerated only when the robot geometry changes, with cricket's fkcc_gen
# (see README.md, "Generate ..."); building VAMP needs none of that toolchain.
# Peak RAM ~1.6 GB with JOBS=1 (measured 2026-09-26, 465 s), ~4.9 GB with JOBS=2.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(dirname "$HERE")"
VENV="${VENV:-$PKG/.venv_vamp}"
SRC="${VAMP_SRC:-$HERE/.vamp_src}"
JOBS="${JOBS:-1}"
VAMP_URL="https://github.com/KavrakiLab/vamp.git"
VAMP_TAG="v0.6.4"
VAMP_COMMIT="8fd768f9974a185a8aa2b4f370e595a8c3ca13cf"
BUILTIN_MODULES="sphere;ur5;panda;fetch;baxter"
BUILTIN_STRUCTS="Sphere;UR5;Panda;Fetch;Baxter"

[ -x "$VENV/bin/python" ] || { echo "no venv at $VENV (see README.md, Toolchain)"; exit 1; }

# -- 1. VAMP source, pinned -------------------------------------------------------------- #
if [ ! -d "$SRC/.git" ]; then
  git clone --quiet --branch "$VAMP_TAG" --depth 1 "$VAMP_URL" "$SRC"
fi
head="$(git -C "$SRC" rev-parse HEAD)"
if [ "$head" != "$VAMP_COMMIT" ]; then
  echo "$SRC is at $head, expected $VAMP_TAG ($VAMP_COMMIT); remove it to re-clone"; exit 1
fi
# start from the pristine release every time: our additions are re-applied below
git -C "$SRC" checkout --quiet -- pyproject.toml
git -C "$SRC" clean --quiet -f -- src/impl/vamp/robots

# -- 2. robot headers -------------------------------------------------------------------- #
# every cell registers its modules: <cell>/vamp_modules/[<module>/]<module>.hh (dual_robot_cell,
# four_robot_cell, tiago_cell, ...); tamp_core ships none.
REPO="$(dirname "$PKG")"
headers=()
for h in "$REPO"/*/vamp_modules/*.hh "$REPO"/*/vamp_modules/*/*.hh; do [ -e "$h" ] && headers+=("$h"); done
[ -n "$EXTRA_HEADERS" ] && headers+=("$EXTRA_HEADERS"/*.hh)
modules="$BUILTIN_MODULES"; structs="$BUILTIN_STRUCTS"
for h in "${headers[@]}"; do
  name="$(basename "$h" .hh)"
  grep -q "struct $name" "$h" || { echo "$h: no 'struct $name'"; exit 1; }
  cp "$h" "$SRC/src/impl/vamp/robots/"
  modules="$modules;$name"; structs="$structs;$name"
  echo "robot module: $name  ($h)"
done

# -- 3. register -------------------------------------------------------------------------- #
sed -i -E "s|^VAMP_ROBOT_MODULES=\"[^\"]*\"|VAMP_ROBOT_MODULES=\"$modules\"|; \
           s|^VAMP_ROBOT_STRUCTS=\"[^\"]*\"|VAMP_ROBOT_STRUCTS=\"$structs\"|" "$SRC/pyproject.toml"
grep -q "VAMP_ROBOT_MODULES=\"$modules\"" "$SRC/pyproject.toml" || { echo "pyproject.toml not updated"; exit 1; }

# -- 4. wheel, then install ---------------------------------------------------------------- #
WHEELS="$(mktemp -d)"
trap 'rm -rf "$WHEELS"' EXIT
echo "building the wheel (JOBS=$JOBS) ..."
env -u PYTHONPATH CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" \
  "$VENV/bin/pip" wheel --no-deps --wheel-dir "$WHEELS" "$SRC"
old="$(env -u PYTHONPATH "$VENV/bin/python" -c 'import vamp, os; print(os.path.dirname(vamp.__file__))' 2>/dev/null || true)"
if [ -n "$old" ] && [ -d "$old" ]; then
  backup="$HERE/.vamp_backup_$(date +%Y%m%d_%H%M%S)"
  cp -a "$old" "$backup"
  echo "previous vamp package backed up to $backup"
fi
env -u PYTHONPATH "$VENV/bin/pip" install --no-deps --force-reinstall "$WHEELS"/vamp*.whl
env -u PYTHONPATH "$VENV/bin/python" - "$modules" <<'EOF'
import sys, vamp
for m in sys.argv[1].split(";"):
    mod = getattr(vamp, m)
    print(f"  vamp.{m:18s} dimension {mod.dimension()}  spheres {mod.n_spheres()}")
EOF
echo "done: vamp installed in $VENV"
