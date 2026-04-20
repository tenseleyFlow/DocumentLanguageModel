#!/usr/bin/env bash
# Build the vendored llama.cpp with ROCm (HIP) acceleration for
# `dlm export` on AMD hosts. Default `scripts/bump-llama-cpp.sh build`
# produces a CPU-only build; ROCm users rerun this script to replace
# the `llama-quantize` / `llama-imatrix` binaries with ROCm-accelerated
# ones.
#
# Toolchain prerequisites (sprint 22 docs/hardware/rocm.md):
#   - ROCm >= 5.7 (6.0+ preferred; we test against 6.0 / 6.2)
#   - hipcc on PATH
#   - cmake >= 3.22
#   - AMDGPU_TARGETS env var set to the arch(es) you want to build for.
#     Common values: gfx90a (MI200), gfx942 (MI300), gfx1100 (RDNA3).
#     Example: `export AMDGPU_TARGETS="gfx1100"`
#
# Usage:
#   scripts/build-llama-cpp-rocm.sh
#
# Idempotent. Re-running rebuilds the same targets in
# `vendor/llama.cpp/build-rocm/` without touching the CPU build dir.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
VENDOR_DIR="$REPO_ROOT/vendor/llama.cpp"
BUILD_DIR="$VENDOR_DIR/build-rocm"

if [[ ! -d "$VENDOR_DIR" ]]; then
  echo "ERROR: vendored llama.cpp not found at $VENDOR_DIR." >&2
  echo "Run 'git submodule update --init --recursive' first." >&2
  exit 1
fi

if ! command -v hipcc >/dev/null 2>&1; then
  echo "ERROR: hipcc not on PATH. Install ROCm and re-try." >&2
  exit 1
fi

if [[ -z "${AMDGPU_TARGETS:-}" ]]; then
  echo "ERROR: AMDGPU_TARGETS is empty. Set it to your GPU arch, e.g.:" >&2
  echo "  export AMDGPU_TARGETS=\"gfx1100\"   # RDNA3" >&2
  echo "  export AMDGPU_TARGETS=\"gfx90a\"    # MI200" >&2
  echo "  export AMDGPU_TARGETS=\"gfx942\"    # MI300" >&2
  exit 2
fi

echo "--> configuring ROCm build for AMDGPU_TARGETS=$AMDGPU_TARGETS"
cmake -S "$VENDOR_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIPBLAS=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_C_COMPILER="$(command -v hipcc)" \
  -DCMAKE_CXX_COMPILER="$(command -v hipcc)"

echo "--> building llama-quantize + llama-imatrix (ROCm)"
cmake --build "$BUILD_DIR" --target llama-quantize llama-imatrix -- -j

echo
echo "ROCm-accelerated binaries in: $BUILD_DIR/bin/"
echo "Point DLM_LLAMA_CPP_BUILD=$BUILD_DIR before running \`dlm export\`"
echo "to prefer this build over the default CPU build."
