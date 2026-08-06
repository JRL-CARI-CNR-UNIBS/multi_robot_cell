#!/usr/bin/env bash
# Build the robot-vs-robot SIMD collision kernel (src/mu_kernel.cc -> libmu_kernel.so).
#
# Deliberately NOT part of the ament/colcon build: the kernel is loaded by ctypes from
# the ROS-free `.venv_vamp` pipeline, never by a ROS node, so wiring it into
# CMakeLists.txt would drag a vamp include path into the ROS build for no reason
# (ADR-0002 -- no vamp import reaches the ROS runtime or tamp_scheduler).
#
# The vamp headers come from `.venv_vamp` itself (site-packages/include), NOT from the
# out-of-tree vamp source checkout: `vamp/vector.hh` is header-only and needs no Eigen,
# so the venv is a sufficient and self-contained source for it.
#
#   ./scripts/build_mu_kernel.sh          # -march=native
#   MARCH=x86-64-v3 ./scripts/build_mu_kernel.sh    # portable AVX2 baseline
#
# If the .so is absent or unloadable the engine silently falls back to the numpy path,
# so a machine without AVX2 still runs -- just ~20x slower on mu.
set -euo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$PKG/.venv_vamp"
INC="$VENV/lib/python3.12/site-packages/include"
OUT="$PKG/lib/libmu_kernel.so"
MARCH="${MARCH:-native}"

if [[ ! -d "$INC/vamp" ]]; then
  echo "error: vamp headers not found at $INC/vamp" >&2
  echo "       expected them inside .venv_vamp (pip install of vamp-planner ships them)" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

# -w: vamp's AVX headers trip -Wignored-attributes on every __m256 template argument,
# which is upstream noise we cannot fix and would otherwise bury a real diagnostic.
g++ -O3 -march="$MARCH" -std=c++17 -fPIC -shared -w \
    -fno-math-errno -fno-trapping-math \
    -I"$INC" \
    "$PKG/src/mu_kernel.cc" \
    -o "$OUT"

echo "built $OUT"
"$VENV/bin/python" - "$OUT" <<'PY'
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
lib.mu_kernel_simd_width.restype = ctypes.c_int
print(f"  SIMD width: {lib.mu_kernel_simd_width()} float32 lanes")
PY
