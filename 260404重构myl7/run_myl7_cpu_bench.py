#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = ROOT / "myl7_cpu_bench_template.cu.in"
BUILD_DIR = ROOT / ".build"


def resolve_fss_include() -> Path | None:
    candidates = (
        ROOT.parent / "FSS" / "myl7" / "include",
        ROOT.parent / "fss" / "include",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


FSS_INCLUDE = resolve_fss_include()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, build, and run a myl7 CPU benchmark aligned with the GPU harness."
    )
    subparsers = parser.add_subparsers(dest="scheme", required=True)

    dpf = subparsers.add_parser("dpf", help="Run DPF CPU benchmark")
    dpf.add_argument("bin", type=int, help="Input bit width; only 64 is supported here")
    dpf.add_argument("n", type=int, help="Total number of test items")
    dpf.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="Chunk size per host batch; 0 means auto",
    )
    dpf.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild even if the cached binary already exists",
    )

    dcf = subparsers.add_parser("dcf", help="Run DCF CPU benchmark")
    dcf.add_argument("bin", type=int, help="Input bit width; only 64 is supported here")
    dcf.add_argument("bout", type=int, help="Output bit width; only 1 is supported here")
    dcf.add_argument("n", type=int, help="Total number of test items")
    dcf.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="Chunk size per host batch; 0 means auto",
    )
    dcf.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild even if the cached binary already exists",
    )

    return parser.parse_args()


def compute_capability() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        )
        first = out.strip().splitlines()[0].strip()
        major, minor = first.split(".", 1)
        return f"{major}{minor}"
    except Exception:
        return "89"


def pick_host_compiler() -> str | None:
    for candidate in (
        "/usr/bin/g++-13",
        "/usr/bin/g++-12",
        "/usr/bin/g++-11",
        "/usr/bin/g++-10",
        "/usr/bin/g++",
    ):
        if Path(candidate).exists():
            return candidate
    return None


def scheme_config(scheme: str) -> dict[str, str]:
    if scheme == "dpf":
        return {
            "SCHEME_HEADER": "dpf",
            "SCHEME_NAME": "DPF",
            "PRG_MUL": "2",
            "SCHEME_TYPE": "fss::Dpf<kInBits, GroupType, PrgType, InType>",
            "BUILD_ALPHA_BODY": "    return static_cast<InType>(10ULL + 2ULL * globalIdx);",
            "BUILD_QUERY_BODY": (
                "    return globalIdx % 3 == 0 ? alpha : static_cast<InType>(alpha + 1);"
            ),
            "EXPECT_EXPR": "x == alpha",
        }

    return {
        "SCHEME_HEADER": "dcf",
        "SCHEME_NAME": "DCF",
        "PRG_MUL": "4",
        "SCHEME_TYPE": (
            "fss::Dcf<kInBits, GroupType, PrgType, InType, fss::DcfPred::kLt>"
        ),
        "BUILD_ALPHA_BODY": "    return static_cast<InType>(20ULL + 2ULL * globalIdx);",
        "BUILD_QUERY_BODY": (
            "    if (globalIdx % 4 == 0) return alpha;\n"
            "    if (globalIdx % 4 == 1) return static_cast<InType>(alpha - 1);\n"
            "    return static_cast<InType>(alpha + 1);"
        ),
        "EXPECT_EXPR": "x < alpha",
    }


def render_source(args: argparse.Namespace) -> str:
    template = TEMPLATE_PATH.read_text()
    values = scheme_config(args.scheme)
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"@{key}@", value)
    return rendered


def ensure_built(args: argparse.Namespace) -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    source_name = f"myl7_cpu_{args.scheme}_bin64_bout1.cu"
    binary_name = f"myl7_cpu_{args.scheme}_bin64_bout1"
    source_path = BUILD_DIR / source_name
    binary_path = BUILD_DIR / binary_name

    source_text = render_source(args)
    needs_rebuild = args.force_rebuild or not binary_path.exists()
    if not source_path.exists() or source_path.read_text() != source_text:
        source_path.write_text(source_text)
        needs_rebuild = True

    if needs_rebuild:
        capability = compute_capability()
        host_compiler = pick_host_compiler()
        cmd = [
            "nvcc",
            "-O3",
            "-std=c++20",
            f"--generate-code=arch=compute_{capability},code=sm_{capability}",
            f"-I{FSS_INCLUDE}",
            "-Xcompiler=-maes,-msse4.2,-fopenmp",
        ]
        if host_compiler is not None:
            cmd.append(f"--compiler-bindir={host_compiler}")
        cmd += [
            str(source_path),
            "-o",
            str(binary_path),
            "-lcrypto",
        ]
        print("Compiling:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    return binary_path


def main() -> int:
    args = parse_args()

    if args.bin != 64:
        raise SystemExit(f"Only bin=64 is supported in this CPU harness, got {args.bin}")
    if args.scheme == "dcf" and args.bout != 1:
        raise SystemExit(f"Only bout=1 is supported in this CPU harness, got {args.bout}")
    if args.n <= 0:
        raise SystemExit(f"n must be positive, got {args.n}")
    if args.chunk < 0:
        raise SystemExit(f"chunk must be >= 0, got {args.chunk}")

    if not TEMPLATE_PATH.exists():
        raise SystemExit(f"Template not found: {TEMPLATE_PATH}")
    if FSS_INCLUDE is None:
        raise SystemExit("FSS include directory not found in either /FSS/myl7/include or /fss/include")

    binary_path = ensure_built(args)
    run_cmd = [str(binary_path), str(args.n)]
    if args.chunk:
        run_cmd.append(str(args.chunk))

    print("Running:", " ".join(run_cmd), flush=True)
    completed = subprocess.run(run_cmd)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
