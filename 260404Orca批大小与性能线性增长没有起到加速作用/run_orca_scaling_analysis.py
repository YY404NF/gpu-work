#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path


WORKDIR = Path(__file__).resolve().parent
ROOT = WORKDIR.parent
FSS_TARBALL_URL = "https://codeload.github.com/YY404NF/FSS/tar.gz/refs/heads/main"
FSS_CACHE_ROOT = Path(tempfile.gettempdir()) / "orca_fss_source"
FSS_EXTRACT_PARENT = FSS_CACHE_ROOT / "src"
FSS_EXTRACT_ROOT = FSS_EXTRACT_PARENT / "FSS-main"
BUILD_ROOT = FSS_CACHE_ROOT / "build"


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


def compile_benchmark(fss_root: Path, source_name: str, output_name: str) -> Path:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    output = BUILD_ROOT / output_name
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
        str(fss_root / source_name),
        str(fss_root / "gpu" / "gpu_mem.cu"),
        "-o",
        str(output),
    ]
    subprocess.run(cmd, cwd=WORKDIR, check=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Orca/FSS DPF/DCF scaling benchmarks and estimate key/output sizes."
    )
    parser.add_argument(
        "--mode",
        choices=["dpf", "dcf", "both"],
        default="both",
        help="Which benchmark(s) to run.",
    )
    parser.add_argument("--bin", type=int, default=64, help="Input bit width.")
    parser.add_argument("--bout", type=int, default=1, help="Output bit width for DCF.")
    parser.add_argument(
        "--n",
        dest="ns",
        type=int,
        nargs="+",
        default=[100000, 200000, 500000, 1000000],
        help="Batch sizes to test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("orca_scaling_results.json"),
        help="JSON output path.",
    )
    parser.add_argument(
        "--fss-root",
        type=Path,
        default=None,
        help="Optional FSS source root. Defaults to $FSS_ROOT, ../FSS, or an auto-downloaded tarball cache.",
    )
    return parser.parse_args()


def micros_per_elem(us: int, n: int) -> float:
    return us / n


def mib(value: int) -> float:
    return value / (1024 * 1024)


def dpf_layout(bin_bits: int, n: int) -> dict:
    mem_scw = n * (bin_bits - 7) * 16
    mem_l = n * 16
    mem_t = (((n - 1) // 32) + 1) * 4 * (bin_bits - 7)
    key_bytes = 12 + mem_scw + 2 * mem_l + mem_t
    out_bytes = (((n - 1) // 32) + 1) * 4
    return {
        "mem_scw_bytes": mem_scw,
        "mem_l0_bytes": mem_l,
        "mem_l1_bytes": mem_l,
        "mem_t_bytes": mem_t,
        "key_blob_bytes_per_party": key_bytes,
        "output_bytes_per_party": out_bytes,
        "key_blob_mib_per_party": round(mib(key_bytes), 3),
        "output_mib_per_party": round(mib(out_bytes), 6),
    }


def dcf_layout(bin_bits: int, bout_bits: int, n: int) -> dict:
    elems_per_block = 128 // bout_bits
    new_bin = bin_bits - int(math.log2(elems_per_block))
    mem_scw = n * new_bin * 16
    mem_l = 2 * n * 16
    mem_vcw = (((bout_bits * n - 1) // 32) + 1) * 4 * (new_bin - 1)
    key_bytes = 12 + mem_scw + mem_l + mem_vcw
    out_bytes = (((bout_bits * n - 1) // 32) + 1) * 4
    return {
        "new_bin": new_bin,
        "mem_scw_bytes": mem_scw,
        "mem_l_bytes": mem_l,
        "mem_vcw_bytes": mem_vcw,
        "key_blob_bytes_per_party": key_bytes,
        "output_bytes_per_party": out_bytes,
        "key_blob_mib_per_party": round(mib(key_bytes), 3),
        "output_mib_per_party": round(mib(out_bytes), 6),
    }


def parse_benchmark_output(text: str) -> dict:
    fields = {}
    for line in text.splitlines():
        match = re.match(r"\s*([a-zA-Z0-9_]+):\s+([0-9]+)", line)
        if match:
            fields[match.group(1)] = int(match.group(2))
    return fields


def run_command(cmd: list[str]) -> tuple[dict, str]:
    completed = subprocess.run(
        cmd,
        cwd=WORKDIR,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    parsed = parse_benchmark_output(completed.stdout)
    if not parsed:
        raise RuntimeError(f"Could not parse benchmark output:\n{completed.stdout}")
    return parsed, completed.stdout


def run_dpf(exe: Path, bin_bits: int, ns: list[int]) -> list[dict]:
    results = []
    for n in ns:
        parsed, raw = run_command([str(exe), str(bin_bits), str(n)])
        layout = dpf_layout(bin_bits, n)
        parsed["eval_avg"] = round((parsed["eval_p0"] + parsed["eval_p1"]) / 2, 3)
        parsed["keygen_us_per_elem"] = round(micros_per_elem(parsed["keygen"], n), 6)
        parsed["eval_avg_us_per_elem"] = round(micros_per_elem(parsed["eval_avg"], n), 6)
        parsed["transfer_avg"] = round((parsed["transfer_p0"] + parsed["transfer_p1"]) / 2, 3)
        parsed["transfer_avg_us_per_elem"] = round(micros_per_elem(parsed["transfer_avg"], n), 6)
        results.append(
            {
                "mode": "dpf",
                "n": n,
                "layout": layout,
                "metrics": parsed,
                "raw_output": raw,
            }
        )
    return results


def run_dcf(exe: Path, bin_bits: int, bout_bits: int, ns: list[int]) -> list[dict]:
    results = []
    for n in ns:
        parsed, raw = run_command([str(exe), str(bin_bits), str(bout_bits), str(n)])
        layout = dcf_layout(bin_bits, bout_bits, n)
        parsed["eval_avg"] = round((parsed["eval_p0"] + parsed["eval_p1"]) / 2, 3)
        parsed["keygen_us_per_elem"] = round(micros_per_elem(parsed["keygen"], n), 6)
        parsed["eval_avg_us_per_elem"] = round(micros_per_elem(parsed["eval_avg"], n), 6)
        parsed["transfer_avg"] = round((parsed["transfer_p0"] + parsed["transfer_p1"]) / 2, 3)
        parsed["transfer_avg_us_per_elem"] = round(micros_per_elem(parsed["transfer_avg"], n), 6)
        results.append(
            {
                "mode": "dcf",
                "n": n,
                "layout": layout,
                "metrics": parsed,
                "raw_output": raw,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    fss_root = resolve_fss_root(args.fss_root)
    payload = {
        "fss_root": str(fss_root),
        "bin": args.bin,
        "bout": args.bout,
        "n_list": args.ns,
        "results": [],
    }
    dpf_exe = compile_benchmark(fss_root, "dpf_benchmark.cu", "dpf_benchmark")
    dcf_exe = compile_benchmark(fss_root, "dcf_benchmark.cu", "dcf_benchmark")
    if args.mode in ("dpf", "both"):
        payload["results"].extend(run_dpf(dpf_exe, args.bin, args.ns))
    if args.mode in ("dcf", "both"):
        payload["results"].extend(run_dcf(dcf_exe, args.bin, args.bout, args.ns))

    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
