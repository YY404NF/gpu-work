#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
DEPS_ROOT="$ROOT/.deps"
N="${1:-131072}"
BIN="${2:-20}"
GPU_ARCH="${GPU_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')}"
FSS_COMMIT="204d982c3516d1f8e53a35f13efbf912faf855a4"
LIBFSS_COMMIT="03c7df90f8a999734f90bcde2438830988358962"
FSS_ROOT="$DEPS_ROOT/FSS-$FSS_COMMIT"
LIBFSS_ROOT="$DEPS_ROOT/libfss-$LIBFSS_COMMIT"
MYL7_BIN="$ROOT/build_local_myl7_batch_bench_sm${GPU_ARCH}"
LIBFSS_BIN="$ROOT/build_local_libfss_batch_bench"
ORCA_DPF_BIN="$ROOT/build_orca_dpf_sm${GPU_ARCH}"
ORCA_DCF_BIN="$ROOT/build_orca_dcf_sm${GPU_ARCH}"

mkdir -p "$DEPS_ROOT"

ensure_tarball_dep() {
  local url="$1"
  local tar_path="$2"
  local out_dir="$3"
  local prefix="$4"
  if [[ -d "$out_dir" ]]; then
    return 0
  fi
  python3 - "$url" "$tar_path" "$out_dir" "$prefix" <<'PY'
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

url = sys.argv[1]
tar_path = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
prefix = sys.argv[4]
tar_path.parent.mkdir(parents=True, exist_ok=True)
if not tar_path.exists():
    with urllib.request.urlopen(url, timeout=60) as resp, tar_path.open("wb") as fout:
        shutil.copyfileobj(resp, fout)
if out_dir.exists():
    raise SystemExit(0)
with tarfile.open(tar_path, "r:gz") as archive:
    archive.extractall(out_dir.parent)
matches = [p for p in out_dir.parent.iterdir() if p.is_dir() and p.name.startswith(prefix)]
if not matches:
    raise SystemExit(f"failed to extract {url}")
matches.sort()
src = matches[-1]
if src != out_dir:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    src.rename(out_dir)
PY
}

ensure_tarball_dep \
  "https://codeload.github.com/YY404NF/FSS/tar.gz/$FSS_COMMIT" \
  "$DEPS_ROOT/FSS-$FSS_COMMIT.tar.gz" \
  "$FSS_ROOT" \
  "FSS-"

ensure_tarball_dep \
  "https://codeload.github.com/frankw2/libfss/tar.gz/$LIBFSS_COMMIT" \
  "$DEPS_ROOT/libfss-${LIBFSS_COMMIT:0:7}.tar.gz" \
  "$LIBFSS_ROOT" \
  "libfss-"

nvcc \
  -std=c++20 \
  -O3 \
  --extended-lambda \
  --generate-code="arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH}" \
  -I"$REPO_ROOT/fss/include" \
  -Xcompiler=-maes,-msse4.2,-fopenmp \
  "$ROOT/myl7_batch_bench.cu" \
  -o "$MYL7_BIN" \
  -lcrypto

g++ \
  -O3 \
  -std=c++20 \
  -maes \
  -msse4.2 \
  -I"$LIBFSS_ROOT/cpp" \
  -DAESNI \
  -DOPENSSL_AES_H \
  -Daesni_set_encrypt_key=AES_set_encrypt_key \
  -Daesni_set_decrypt_key=AES_set_decrypt_key \
  -Daesni_encrypt=AES_encrypt \
  -Daesni_decrypt=AES_decrypt \
  "$ROOT/libfss_batch_bench.cpp" \
  "$LIBFSS_ROOT/cpp/fss-client.cpp" \
  "$LIBFSS_ROOT/cpp/fss-server.cpp" \
  "$LIBFSS_ROOT/cpp/fss-common.cpp" \
  -lcrypto \
  -lgmpxx \
  -lgmp \
  -lpthread \
  -ldl \
  -lssl \
  -o "$LIBFSS_BIN"

nvcc \
  -O3 \
  -m64 \
  -std=c++17 \
  --generate-code="arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH}" \
  -I"$FSS_ROOT" \
  -Xcompiler=-O3,-w,-fpermissive,-fpic,-pthread,-fopenmp,-march=native \
  -lcuda \
  -lcudart \
  -lcurand \
  "$FSS_ROOT/dpf_benchmark.cu" \
  "$FSS_ROOT/gpu/gpu_mem.cu" \
  -o "$ORCA_DPF_BIN"

nvcc \
  -O3 \
  -m64 \
  -std=c++17 \
  --generate-code="arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH}" \
  -I"$FSS_ROOT" \
  -Xcompiler=-O3,-w,-fpermissive,-fpic,-pthread,-fopenmp,-march=native \
  -lcuda \
  -lcudart \
  -lcurand \
  "$FSS_ROOT/dcf_benchmark.cu" \
  "$FSS_ROOT/gpu/gpu_mem.cu" \
  -o "$ORCA_DCF_BIN"

"$MYL7_BIN" dpf "$N"
"$MYL7_BIN" dcf "$N"
"$LIBFSS_BIN" dpf "$BIN" "$N"
"$LIBFSS_BIN" dcf "$BIN" "$N"
"$ORCA_DPF_BIN" "$BIN" "$N"
"$ORCA_DCF_BIN" "$BIN" 1 "$N"
