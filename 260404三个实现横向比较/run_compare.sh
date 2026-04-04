#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yy404nf/FSS-Work/260404三个实现横向比较"
N="${1:-131072}"
BIN="${2:-20}"

cmake -S "$ROOT" -B "$ROOT/build_local" \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 >/dev/null
cmake --build "$ROOT/build_local" -j >/dev/null

cmake -S /home/yy404nf/FSS-Work/FSS -B "$ROOT/build_orca" \
  -DCUDA_VERSION=13.1 \
  -DGPU_ARCH=86 >/dev/null
cmake --build "$ROOT/build_orca" -j >/dev/null

"$ROOT/build_local/myl7_batch_bench" dpf "$N"
"$ROOT/build_local/myl7_batch_bench" dcf "$N"
"$ROOT/build_local/libfss_batch_bench" dpf "$BIN" "$N"
"$ROOT/build_local/libfss_batch_bench" dcf "$BIN" "$N"
"$ROOT/build_orca/dpf_benchmark" "$BIN" "$N"
"$ROOT/build_orca/dcf_benchmark" "$BIN" 1 "$N"
