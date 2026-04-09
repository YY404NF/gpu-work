#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"

if git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$REPO_ROOT" submodule sync --recursive || true
  git -C "$REPO_ROOT" submodule update --init --recursive FSS || true
fi

python3 - "$ROOT" <<'PY'
import shutil
import tarfile
import urllib.request
from pathlib import Path
import sys

root = Path(sys.argv[1])
deps = root / ".deps"
deps.mkdir(parents=True, exist_ok=True)

def ensure_tarball(url: str, tar_name: str, out_name: str, prefix: str, marker: str):
    out_dir = deps / out_name
    if (out_dir / marker).exists():
        print(f"ready: {out_dir}")
        return
    tar_path = deps / tar_name
    if not tar_path.exists():
        print(f"download: {url}")
        with urllib.request.urlopen(url, timeout=120) as resp, tar_path.open("wb") as fout:
            shutil.copyfileobj(resp, fout)
    with tarfile.open(tar_path, "r:gz") as archive:
        archive.extractall(deps)
    matches = sorted(p for p in deps.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not matches:
        raise SystemExit(f"failed to extract {url}")
    src = matches[-1]
    if src != out_dir:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        src.rename(out_dir)
    print(f"ready: {out_dir}")

ensure_tarball(
    "https://codeload.github.com/YY404NF/FSS/tar.gz/aa69884ae4411c96b2ec8afb89aefa3bbbb77f67",
    "FSS-aa69884ae4411c96b2ec8afb89aefa3bbbb77f67.tar.gz",
    "FSS",
    "FSS-",
    "Orca/gpu/gpu_mem.cu",
)
ensure_tarball(
    "https://codeload.github.com/myl7/fss/tar.gz/7b241e8add8df81a42fb269683db3f9e82c94029",
    "fss-7b241e8add8df81a42fb269683db3f9e82c94029.tar.gz",
    "fss",
    "fss-",
    "include/fss/dcf.cuh",
)
ensure_tarball(
    "https://codeload.github.com/frankw2/libfss/tar.gz/03c7df90f8a999734f90bcde2438830988358962",
    "libfss-03c7df90f8a999734f90bcde2438830988358962.tar.gz",
    "libfss",
    "libfss-",
    "cpp/fss-client.cpp",
)
PY

echo "Next:"
echo "  cmake -S \"$ROOT\" -B \"$ROOT/.build\""
echo "  cmake --build \"$ROOT/.build\" -j"
