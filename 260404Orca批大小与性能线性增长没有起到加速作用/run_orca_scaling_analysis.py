#!/usr/bin/env python3
import argparse
import json
import math
import re
import subprocess
from pathlib import Path


ROOT = Path("/home/yy404nf/FSS-Work")
FSS_BUILD = ROOT / "FSS" / "build"


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
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    parsed = parse_benchmark_output(completed.stdout)
    if not parsed:
        raise RuntimeError(f"Could not parse benchmark output:\n{completed.stdout}")
    return parsed, completed.stdout


def run_dpf(bin_bits: int, ns: list[int]) -> list[dict]:
    results = []
    exe = FSS_BUILD / "dpf_benchmark"
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


def run_dcf(bin_bits: int, bout_bits: int, ns: list[int]) -> list[dict]:
    results = []
    exe = FSS_BUILD / "dcf_benchmark"
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
    payload = {
        "bin": args.bin,
        "bout": args.bout,
        "n_list": args.ns,
        "results": [],
    }
    if args.mode in ("dpf", "both"):
        payload["results"].extend(run_dpf(args.bin, args.ns))
    if args.mode in ("dcf", "both"):
        payload["results"].extend(run_dcf(args.bin, args.bout, args.ns))

    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
