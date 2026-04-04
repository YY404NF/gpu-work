#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = ROOT / "myl7_gpu_bench_template.cu.in"
BUILD_DIR = ROOT / ".build"
FSS_INCLUDE = ROOT.parent / "fss" / "include"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, build, and run a parameterized myl7 GPU benchmark."
    )
    subparsers = parser.add_subparsers(dest="scheme", required=True)

    dpf = subparsers.add_parser("dpf", help="Run DPF benchmark")
    dpf.add_argument("bin", type=int, help="Input bit width")
    dpf.add_argument("n", type=int, help="Total number of test items")
    dpf.add_argument(
        "--bout",
        type=int,
        default=1,
        help="Output bit width for the uint group (default: 1)",
    )
    dpf.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="Chunk size per kernel launch; 0 means auto",
    )
    dpf.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild even if the cached binary already exists",
    )

    dcf = subparsers.add_parser("dcf", help="Run DCF benchmark")
    dcf.add_argument("bin", type=int, help="Input bit width")
    dcf.add_argument("bout", type=int, help="Output bit width")
    dcf.add_argument("n", type=int, help="Total number of test items")
    dcf.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="Chunk size per kernel launch; 0 means auto",
    )
    dcf.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild even if the cached binary already exists",
    )

    return parser.parse_args()


def validate_positive(name: str, value: int, upper: int) -> None:
    if value <= 0 or value > upper:
        raise SystemExit(f"{name} must be in [1, {upper}], got {value}")


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
        return "86"


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


def group_type_for_bout(bout: int) -> str:
    if bout == 64:
        return "fss::group::Uint<std::uint64_t>"
    return "fss::group::Uint<std::uint64_t, (static_cast<std::uint64_t>(1) << kBout)>"


def scheme_config(args: argparse.Namespace) -> dict[str, str]:
    if args.scheme == "dpf":
        if args.bin == 64:
            build_alpha_body = """\
    return static_cast<InType>(10ULL + 2ULL * globalIdx);"""
            build_query_body = """\
    return globalIdx % 3 == 0 ? alpha : static_cast<InType>(alpha + 1);"""
        else:
            build_alpha_body = """\
    const InType limit = static_cast<InType>(1ULL << kInBits);
    const InType mask = limit - 1;
    return static_cast<InType>((10ULL + globalIdx * kStride) & mask);"""
            build_query_body = """\
    const InType limit = static_cast<InType>(1ULL << kInBits);
    return (globalIdx % 3 == 0 || alpha + 1 >= limit) ? alpha : static_cast<InType>(alpha + 1);"""

        return {
            "SCHEME_HEADER": "dpf",
            "SCHEME_NAME": "DPF",
            "PRG_MUL": "2",
            "SCHEME_TYPE": "fss::Dpf<kInBits, GroupType, PrgType, InType>",
            "BUILD_ALPHA_BODY": build_alpha_body,
            "BUILD_QUERY_BODY": build_query_body,
            "EXPECT_EXPR": "x == alpha",
        }

    if args.bin == 64:
        build_alpha_body = """\
    return static_cast<InType>(20ULL + 2ULL * globalIdx);"""
        build_query_body = """\
    if (globalIdx % 4 == 0) return alpha;
    if (globalIdx % 4 == 1) return static_cast<InType>(alpha - 1);
    return static_cast<InType>(alpha + 1);"""
    else:
        build_alpha_body = """\
    const InType limit = static_cast<InType>(1ULL << kInBits);
    const InType span = limit - 1;
    return static_cast<InType>(1ULL + ((19ULL + globalIdx * kStride) % span));"""
        build_query_body = """\
    const InType limit = static_cast<InType>(1ULL << kInBits);
    if (globalIdx % 4 == 0) return alpha;
    if (globalIdx % 4 == 1) return static_cast<InType>(alpha - 1);
    return alpha + 1 < limit ? static_cast<InType>(alpha + 1) : alpha;"""

    return {
        "SCHEME_HEADER": "dcf",
        "SCHEME_NAME": "DCF",
        "PRG_MUL": "4",
        "SCHEME_TYPE": (
            "fss::Dcf<kInBits, GroupType, PrgType, InType, fss::DcfPred::kLt>"
        ),
        "BUILD_ALPHA_BODY": build_alpha_body,
        "BUILD_QUERY_BODY": build_query_body,
        "EXPECT_EXPR": "x < alpha",
    }


def render_source(args: argparse.Namespace) -> str:
    template = TEMPLATE_PATH.read_text()
    bout = args.bout
    config = scheme_config(args)
    values = {
        "IN_BITS": str(args.bin),
        "BOUT": str(bout),
        "GROUP_TYPE": group_type_for_bout(bout),
        **config,
    }
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"@{key}@", value)
    return rendered


def ensure_built(args: argparse.Namespace) -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    source_name = f"myl7_{args.scheme}_bin{args.bin}_bout{args.bout}.cu"
    binary_name = f"myl7_{args.scheme}_bin{args.bin}_bout{args.bout}"
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
            "--extended-lambda",
            f"--generate-code=arch=compute_{capability},code=sm_{capability}",
            f"-I{FSS_INCLUDE}",
        ]
        if host_compiler is not None:
            cmd.append(f"--compiler-bindir={host_compiler}")
        cmd += [
            str(source_path),
            "-o",
            str(binary_path),
        ]
        print("Compiling:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    return binary_path


def main() -> int:
    args = parse_args()
    validate_positive("bin", args.bin, 64)
    validate_positive("bout", args.bout, 64)
    if args.n <= 0:
        raise SystemExit(f"n must be positive, got {args.n}")
    if args.chunk < 0:
        raise SystemExit(f"chunk must be >= 0, got {args.chunk}")

    if not TEMPLATE_PATH.exists():
        raise SystemExit(f"Template not found: {TEMPLATE_PATH}")
    if not FSS_INCLUDE.is_dir():
        raise SystemExit(f"FSS include directory not found: {FSS_INCLUDE}")

    binary_path = ensure_built(args)
    run_cmd = [str(binary_path), str(args.n)]
    if args.chunk:
        run_cmd.append(str(args.chunk))

    print("Running:", " ".join(run_cmd), flush=True)
    completed = subprocess.run(run_cmd)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
