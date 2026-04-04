#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path


WORKDIR = Path(__file__).resolve().parent
ROOT = WORKDIR.parent
SRC = WORKDIR / "orca_compute_only_benchmark.cu"
BIN = WORKDIR / "orca_compute_only_benchmark"
OUT = WORKDIR / "orca_compute_only_results.json"
FSS_TARBALL_URL = "https://codeload.github.com/YY404NF/FSS/tar.gz/refs/heads/main"
FSS_CACHE_ROOT = Path(tempfile.gettempdir()) / "orca_fss_source"
FSS_EXTRACT_PARENT = FSS_CACHE_ROOT / "src"
FSS_EXTRACT_ROOT = FSS_EXTRACT_PARENT / "FSS-main"


def is_valid_fss_root(path: Path) -> bool:
    return (path / "gpu" / "gpu_mem.cu").is_file() and (path / "runtime" / "standalone_runtime.h").is_file()


def download_fss_tarball() -> Path:
    tarball = FSS_CACHE_ROOT / "FSS-main.tar.gz"
    FSS_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if not tarball.exists():
        with urllib.request.urlopen(FSS_TARBALL_URL, timeout=60) as response, tarball.open("wb") as fout:
            shutil.copyfileobj(response, fout)

    if is_valid_fss_root(FSS_EXTRACT_ROOT):
        return FSS_EXTRACT_ROOT

    shutil.rmtree(FSS_EXTRACT_PARENT, ignore_errors=True)
    FSS_EXTRACT_PARENT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as archive:
        archive.extractall(FSS_EXTRACT_PARENT)

    if not is_valid_fss_root(FSS_EXTRACT_ROOT):
        raise RuntimeError(f"Downloaded tarball does not contain a usable FSS tree: {FSS_EXTRACT_ROOT}")
    return FSS_EXTRACT_ROOT


def resolve_fss_root(cli_value: Path | None) -> Path:
    candidates = []
    if cli_value is not None:
        candidates.append(cli_value.expanduser().resolve())
    env_value = os.environ.get("FSS_ROOT")
    if env_value:
        candidates.append(Path(env_value).expanduser().resolve())
    candidates.append((ROOT / "FSS").resolve())
    candidates.append(FSS_EXTRACT_ROOT.resolve())

    for candidate in candidates:
        if is_valid_fss_root(candidate):
            return candidate

    return download_fss_tarball()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile and run compute-only Orca/FSS DPF/DCF scaling benchmarks."
    )
    parser.add_argument(
        "--totals",
        type=int,
        nargs="+",
        default=[100_000, 200_000, 500_000, 1_000_000],
        help="Total element counts to benchmark. Defaults to a WSL-safe range.",
    )
    parser.add_argument(
        "--chunk-n",
        type=int,
        default=1_000_000,
        help="Chunk size used when total_n exceeds on-device comfort range.",
    )
    parser.add_argument(
        "--mode",
        choices=["dpf", "dcf", "both"],
        default="both",
        help="Which primitive(s) to benchmark.",
    )
    parser.add_argument(
        "--direct-check-n",
        type=int,
        default=0,
        help="Optional single-launch direct check size. 0 disables it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT,
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--fss-root",
        type=Path,
        default=None,
        help="Optional FSS source root. Defaults to $FSS_ROOT, ../FSS, or an auto-downloaded tarball cache.",
    )
    return parser.parse_args()


def compile_binary(fss_root: Path) -> None:
    cmd = [
        "nvcc",
        "-O3",
        "-m64",
        "-std=c++17",
        "-I",
        str(fss_root),
        "-Xcompiler=-O3,-w,-fpermissive,-fpic,-pthread,-fopenmp,-march=native",
        "-lcuda",
        "-lcudart",
        "-lcurand",
        str(SRC),
        str(fss_root / "gpu" / "gpu_mem.cu"),
        "-o",
        str(BIN),
    ]
    subprocess.run(cmd, cwd=WORKDIR, check=True)


def parse_output(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        match = re.match(r"\s*([a-zA-Z0-9_]+):\s+(.+)", line)
        if not match:
            continue
        key = match.group(1)
        value = match.group(2).strip()
        if value.endswith(" us"):
            result[key] = float(value[:-3])
        elif value.endswith(" elem"):
            result[key] = float(value[:-5])
        elif value.endswith(" bit"):
            result[key] = int(value[:-4])
        elif re.fullmatch(r"-?\d+\.\d+", value):
            result[key] = float(value)
        elif re.fullmatch(r"-?\d+", value):
            result[key] = int(value)
        else:
            result[key] = value
    return result


def run_case(mode: str, total_n: int, chunk_n: int, bout: int = 1) -> dict:
    actual_chunk_n = min(total_n, chunk_n)
    if total_n % actual_chunk_n != 0:
        raise ValueError(f"total_n={total_n} must be divisible by chunk_n={actual_chunk_n}.")
    chunks = total_n // actual_chunk_n
    cmd = [str(BIN), mode, "64", str(actual_chunk_n), str(chunks)]
    if mode == "dcf":
        cmd.append(str(bout))
    completed = subprocess.run(
        cmd,
        cwd=WORKDIR,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    parsed = parse_output(completed.stdout)
    return {
        "mode": mode,
        "total_n": total_n,
        "chunk_n": actual_chunk_n,
        "chunks": chunks,
        "metrics": parsed,
        "raw_output": completed.stdout,
    }


def run_case_median(mode: str, total_n: int, chunk_n: int, bout: int = 1, attempts: int = 3) -> dict:
    runs = [run_case(mode, total_n, chunk_n, bout) for _ in range(attempts)]
    eval_values = [item["metrics"]["eval_avg_kernel"] for item in runs]
    target = statistics.median(eval_values)
    selected = min(runs, key=lambda item: abs(item["metrics"]["eval_avg_kernel"] - target))
    selected["attempt_eval_avg_kernel_us"] = eval_values
    return selected


def main() -> None:
    args = parse_args()
    fss_root = resolve_fss_root(args.fss_root)
    compile_binary(fss_root)
    totals = args.totals
    chunk_n = args.chunk_n
    results = []
    direct_checks = []

    modes = ("dpf", "dcf") if args.mode == "both" else (args.mode,)
    for mode in modes:
        for total_n in totals:
            results.append(run_case(mode, total_n, chunk_n, 1))
        if args.direct_check_n > 0:
            direct_checks.append(run_case_median(mode, args.direct_check_n, args.direct_check_n, 1))

    payload = {
        "fss_root": str(fss_root),
        "chunk_n": chunk_n,
        "totals": totals,
        "results": results,
        "direct_checks": direct_checks,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
