#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/.build}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
TIMEOUT_SECS="${TIMEOUT_SECS:-60}"
EVAL_ITERS="${EVAL_ITERS:-3}"
N_LIST="${N_LIST:-10000 100000 1000000}"

mkdir -p "$RESULTS_DIR/raw"

MEM_TOTAL_KB="$(grep MemTotal /proc/meminfo | awk '{print $2}')"
MEM_TOTAL_GIB="$(free -h | awk '/^Mem:/ {print $2}')"
GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -1 || true)"
GPU_NAME="$(printf '%s' "$GPU_INFO" | awk -F', ' '{print $1}')"
GPU_MEM_TOTAL="$(printf '%s' "$GPU_INFO" | awk -F', ' '{print $2}')"
GPU_CC="$(printf '%s' "$GPU_INFO" | awk -F', ' '{print $3}')"
GPU_ARCH="$(printf '%s' "$GPU_CC" | sed -E 's/^([0-9]+)\.([0-9]+)$/\1\2/')"
TIMEOUT_US="$(( TIMEOUT_SECS * 1000000 ))"

declare -A KEYGEN_MAP
declare -A EVAL_MAP

format_n_label() {
  case "$1" in
    10) printf '10' ;;
    100) printf '10^2' ;;
    1000) printf '10^3' ;;
    10000) printf '10^4' ;;
    100000) printf '10^5' ;;
    1000000) printf '10^6' ;;
    10000000) printf '10^7' ;;
    *) printf '%s' "$1" ;;
  esac
}

run_and_capture() {
  local label="$1"
  shift
  local outfile="$RESULTS_DIR/raw/${label}.log"
  local status=0
  set +e
  timeout "${TIMEOUT_SECS}s" "$@" >"$outfile" 2>&1
  status=$?
  set -e

  if [[ $status -eq 0 ]]; then
    printf '%s\n' "$outfile"
    return 0
  fi

  if [[ $status -eq 124 ]]; then
    printf '>TIMEOUT<\n' >"$outfile"
    printf '%s\n' "$outfile"
    return 124
  fi

  printf 'ERROR(exit=%d)\n' "$status" >"$outfile"
  printf '%s\n' "$outfile"
  return "$status"
}

extract_field() {
  local logfile="$1"
  local field="$2"
  if grep -qx '>TIMEOUT<' "$logfile"; then
    printf '>TIMEOUT<\n'
    return 0
  fi
  if grep -q '^ERROR' "$logfile"; then
    sed -n '1p' "$logfile"
    return 0
  fi
  grep -E "^[[:space:]]+${field}: " "$logfile" | tail -1 | sed -E "s/^[[:space:]]+${field}: //"
}

format_us_value() {
  local raw="$1"
  if [[ "$raw" == ">TIMEOUT<" ]]; then
    printf '>%s us' "$TIMEOUT_US"
    return 0
  fi
  if [[ "$raw" == ERROR* ]]; then
    printf '%s' "$raw"
    return 0
  fi

  python3 - "$raw" <<'PY'
import sys
raw = sys.argv[1].strip()
if raw.endswith("us"):
    raw = raw[:-2].strip()
value = float(raw)
print(f"{int(value + 0.5)} us")
PY
}

record_metric() {
  local map_name="$1"
  local key="$2"
  local label="$3"
  local field="$4"
  shift 4

  local logfile=""
  local metric=""
  local rendered=""
  local -n map_ref="$map_name"

  if logfile="$(run_and_capture "$label" "$@")"; then
    metric="$(extract_field "$logfile" "$field")"
  else
    local status=$?
    if [[ $status -eq 124 ]]; then
      metric=">TIMEOUT<"
    else
      metric="ERROR(exit=${status})"
    fi
  fi

  rendered="$(format_us_value "$metric")"
  map_ref["$key"]="$rendered"
}

print_environment() {
  cat <<EOF
## 测试环境

- 主机内存：\`${MEM_TOTAL_KB} kB\`（约 \`${MEM_TOTAL_GIB}\`）
- 显卡：\`${GPU_NAME}\`
- 显存：\`${GPU_MEM_TOTAL}\`
- \`GPU-ARCH\`：\`${GPU_ARCH}\`（\`sm_${GPU_ARCH}\`）
- 超时阈值：\`${TIMEOUT_SECS} s\`

EOF
}

print_table() {
  local title="$1"
  local map_name="$2"
  local -n map_ref="$map_name"

  printf '### %s\n\n' "$title"
  printf '| 实现 |'
  for n in $N_LIST; do
    printf ' `%s` |' "$(format_n_label "$n")"
  done
  printf '\n| --- |'
  for _ in $N_LIST; do
    printf ' ---: |'
  done
  printf '\n'

  for impl in "Orca" "myl7-gpu" "myl7-cpu" "libfss"; do
    printf '| `%s` |' "$impl"
    for n in $N_LIST; do
      printf ' `%s` |' "${map_ref["${impl}|${n}"]}"
    done
    printf '\n'
  done
  printf '\n'
}

for n in $N_LIST; do
  record_metric KEYGEN_MAP "Orca|${n}" "orca_dcf_batch_keygen_${n}" "keygen" \
    "$BUILD_DIR/orca_dcf_batch_keygen_bench" "$n"
  record_metric KEYGEN_MAP "myl7-gpu|${n}" "myl7_gpu_dcf_batch_keygen_${n}" "keygen" \
    "$BUILD_DIR/myl7_dcf_batch_keygen_bench" "$n"
  record_metric KEYGEN_MAP "myl7-cpu|${n}" "myl7_cpu_dcf_batch_keygen_${n}" "keygen" \
    "$BUILD_DIR/myl7_cpu_dcf_batch_keygen_bench" "$n"
  record_metric KEYGEN_MAP "libfss|${n}" "libfss_dcf_batch_keygen_${n}" "keygen" \
    "$BUILD_DIR/libfss_dcf_batch_keygen_bench" "$n"
done

for n in $N_LIST; do
  record_metric EVAL_MAP "Orca|${n}" "orca_dcf_singlekey_${n}" "eval" \
    "$BUILD_DIR/orca_dcf_singlekey_bench" "$n" "$EVAL_ITERS"
  record_metric EVAL_MAP "myl7-gpu|${n}" "myl7_gpu_dcf_singlekey_${n}" "eval" \
    "$BUILD_DIR/myl7_dcf_singlekey_bench" "$n" "$EVAL_ITERS"
  record_metric EVAL_MAP "myl7-cpu|${n}" "myl7_cpu_dcf_singlekey_${n}" "eval" \
    "$BUILD_DIR/myl7_cpu_dcf_singlekey_bench" "$n" "$EVAL_ITERS"
  record_metric EVAL_MAP "libfss|${n}" "libfss_dcf_singlekey_${n}" "eval" \
    "$BUILD_DIR/libfss_dcf_singlekey_bench" "$n" "$EVAL_ITERS"
done

{
  print_environment
  print_table "DCF - 批量 keygen" KEYGEN_MAP
  print_table "DCF - 单 key，多 \`x\`" EVAL_MAP
} | tee "$RESULTS_DIR/dcf_bench_latest.md"

printf 'Saved markdown table to %s\n' "$RESULTS_DIR/dcf_bench_latest.md"
