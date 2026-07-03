#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAUNCHER=${LAUNCHER:-$SCRIPT_DIR/run-ds4-v9-server.sh}
BENCH=${BENCH:-/root/llm-inference-bench/llm_decode_bench.py}
OUT=${OUT:-/root/bench-results/ds4-v9-$(date -u +%Y%m%d-%H%M%S)}
IMAGE=${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-ds4dspark-v9-ve72ad00-b12x57422ad-cu132-20260703}
PROGRESS_FILE=${PROGRESS_FILE:-/root/vllm/prubezne_vysledky}

TPS=${TPS:-2,4}
BACKENDS=${BACKENDS:-b12x-a16,b12x-a8,b12x-a8-dglin,lucifer-default,lucifer-cutlass}
MODES=${MODES:-dspark}
DECODE_CONCURRENCY=${DECODE_CONCURRENCY:-1,16,32,64}
DECODE_CONTEXTS=${DECODE_CONTEXTS:-0}
DECODE_DURATION=${DECODE_DURATION:-30}
DECODE_MAX_TOKENS=${DECODE_MAX_TOKENS:-8192}
DECODE_TOKEN_BUDGET=${DECODE_TOKEN_BUDGET:-2000000}
PREFILL_CONTEXTS=${PREFILL_CONTEXTS:-8k,64k,128k}
PREFILL_DURATION=${PREFILL_DURATION:-10}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
PORT_BASE=${PORT_BASE:-7100}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1800}

mkdir -p "$OUT"

split_csv() {
  local input=$1
  input=${input//,/ }
  printf '%s\n' $input
}

gpu_groups_for_tp() {
  case "$1" in
    2) printf '%s\n' 0,1 2,3 4,5 6,7 8,9 10,11 12,13 14,15 ;;
    4) printf '%s\n' 0,1,2,3 4,5,6,7 8,9,10,11 12,13,14,15 ;;
    *) echo "Unsupported TP=$1" >&2; return 2 ;;
  esac
}

model_name_for_mode() {
  case "$1" in
    standard-*) printf 'DeepSeek-V4-Flash\n' ;;
    dspark) printf 'DeepSeek-V4-Flash-DSpark\n' ;;
    *) return 2 ;;
  esac
}

wait_for_server() {
  local name=$1
  local port=$2
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$name"; then
      docker logs --tail 240 "$name" || true
      return 1
    fi
    sleep 5
  done
  docker logs --tail 300 "$name" || true
  echo "Timed out waiting for $name on port $port" >&2
  return 1
}

append_case_summary() {
  local status=$1 label=$2 case_dir=$3
  python3 - "$status" "$label" "$case_dir" "$PROGRESS_FILE" <<'PY'
import datetime as dt
import json
import math
import pathlib
import sys

status, label, case_dir, progress_file = sys.argv[1:]
case_path = pathlib.Path(case_dir)
now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

def load(name):
    path = case_path / name
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)

def fmt(value):
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(value):
        return "NA"
    return f"{value:.1f}"

decode = load("decode.json")
prefill = load("prefill.json")

cc = {}
for row in decode.get("results", []):
    if int(row.get("context_tokens", -1)) == 0:
        cc[int(row.get("concurrency", 0))] = row.get("aggregate_tps")

coding_summary = decode.get("coding_peak", {}).get("summary", {})
pref = prefill.get("prefill", {})

line = (
    f"{now} {status} {label} "
    f"decode cc1={fmt(cc.get(1))} cc16={fmt(cc.get(16))} "
    f"cc32={fmt(cc.get(32))} cc64={fmt(cc.get(64))} "
    f"coding_median={fmt(coding_summary.get('median_generation_tok_s'))} "
    f"cjk={coding_summary.get('cjk_runs', 'NA')} "
    f"prefill 8k={fmt(pref.get('8192', {}).get('tok_per_sec'))} "
    f"64k={fmt(pref.get('65536', {}).get('tok_per_sec'))} "
    f"128k={fmt(pref.get('131072', {}).get('tok_per_sec'))} "
    f"dir={case_path}"
)
with open(progress_file, "a", encoding="utf-8") as f:
    f.write(line + "\n")
print(line)
PY
}

validate_case_results() {
  local case_dir=$1
  python3 - "$case_dir" "$DECODE_CONCURRENCY" "$PREFILL_CONTEXTS" <<'PY'
import json
import math
import pathlib
import sys

case_dir = pathlib.Path(sys.argv[1])
decode_concurrency = [int(x) for x in sys.argv[2].replace(",", " ").split()]
prefill_contexts = []
for item in sys.argv[3].replace(",", " ").split():
    item = item.lower()
    if item.endswith("k"):
        prefill_contexts.append(str(int(float(item[:-1]) * 1024)))
    else:
        prefill_contexts.append(str(int(item)))

def fail(msg: str) -> None:
    print(f"invalid benchmark results: {msg}", file=sys.stderr)
    raise SystemExit(1)

def load(name: str):
    path = case_dir / name
    if not path.exists():
        fail(f"missing {name}")
    with path.open() as f:
        return json.load(f)

def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

decode = load("decode.json")
rows = {}
for row in decode.get("results", []):
    try:
        context = int(row.get("context_tokens", -1))
        concurrency = int(row.get("concurrency", 0))
    except (TypeError, ValueError):
        continue
    if context == 0 and finite(row.get("aggregate_tps")):
        rows[concurrency] = row["aggregate_tps"]

missing_decode = [cc for cc in decode_concurrency if cc not in rows]
if missing_decode:
    fail(f"missing decode aggregate_tps for concurrency {missing_decode}")

prefill = load("prefill.json")
prefill_rows = prefill.get("prefill", {})
missing_prefill = [
    ctx for ctx in prefill_contexts
    if ctx not in prefill_rows or not finite(prefill_rows[ctx].get("tok_per_sec"))
]
if missing_prefill:
    fail(f"missing prefill tok_per_sec for contexts {missing_prefill}")
PY
}

run_case() {
  local tp=$1 backend=$2 mode=$3 gpus=$4 port=$5
  local label name case_dir model_name
  label="tp${tp}-${backend}-${mode}"
  name="ds4-v9-${label}"
  case_dir="$OUT/$label"
  model_name=$(model_name_for_mode "$mode")
  mkdir -p "$case_dir"

  echo "==> launch $label on GPUs=$gpus port=$port"
  if ! IMAGE="$IMAGE" \
    NAME="$name" \
    PORT="$port" \
    GPUS="$gpus" \
    TP="$tp" \
    BACKEND="$backend" \
    MODE="$mode" \
    MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    CACHE="$case_dir/cache" \
    CONTAINER_TMP="$case_dir/tmp" \
    "$LAUNCHER" 2>&1 | tee "$case_dir/launch.log"; then
    return 1
  fi

  wait_for_server "$name" "$port" || return 1
  curl -fsS "http://127.0.0.1:${port}/version" > "$case_dir/version.json" || true
  curl -fsS "http://127.0.0.1:${port}/v1/models" > "$case_dir/models.json" || return 1

  echo "==> decode $label"
  if ! python3 "$BENCH" \
    --host localhost \
    --port "$port" \
    --model "$model_name" \
    --concurrency "$DECODE_CONCURRENCY" \
    --contexts "$DECODE_CONTEXTS" \
    --duration "$DECODE_DURATION" \
    --max-tokens "$DECODE_MAX_TOKENS" \
    --max-total-tokens "$DECODE_TOKEN_BUDGET" \
    --skip-prefill \
    --display-mode plain \
    --no-hw-monitor \
    --coding-peak \
    --coding-peak-runs 5 \
    --output "$case_dir/decode.json" \
    2>&1 | tee "$case_dir/decode.log"; then
    return 1
  fi

  echo "==> prefill $label"
  if ! python3 "$BENCH" \
    --host localhost \
    --port "$port" \
    --model "$model_name" \
    --concurrency 1 \
    --contexts 0 \
    --prefill-only \
    --standalone-prefill \
    --prefill-contexts "$PREFILL_CONTEXTS" \
    --prefill-duration "$PREFILL_DURATION" \
    --max-tokens "$DECODE_MAX_TOKENS" \
    --max-total-tokens "$DECODE_TOKEN_BUDGET" \
    --display-mode plain \
    --no-hw-monitor \
    --output "$case_dir/prefill.json" \
    2>&1 | tee "$case_dir/prefill.log"; then
    return 1
  fi

  validate_case_results "$case_dir" || return 1

  curl -fsS "http://127.0.0.1:${port}/metrics" > "$case_dir/final-metrics.prom" || true
  docker logs "$name" > "$case_dir/server.log" 2>&1 || true
  docker rm -f "$name" >/dev/null 2>&1 || true
  append_case_summary DONE "$label" "$case_dir"
}

run_case_guarded() {
  local tp=$1 backend=$2 mode=$3 gpus=$4 port=$5
  local label="tp${tp}-${backend}-${mode}"
  if run_case "$tp" "$backend" "$mode" "$gpus" "$port"; then
    return 0
  fi
  mkdir -p "$OUT/$label"
  docker logs "ds4-v9-${label}" > "$OUT/$label/server.log" 2>&1 || true
  docker rm -f "ds4-v9-${label}" >/dev/null 2>&1 || true
  append_case_summary FAILED "$label" "$OUT/$label" || true
  return 1
}

run_tp_matrix() {
  local tp=$1
  local -a groups cases wave_pids
  mapfile -t groups < <(gpu_groups_for_tp "$tp")
  cases=()
  for backend in $(split_csv "$BACKENDS"); do
    for mode in $(split_csv "$MODES"); do
      cases+=("$tp:$backend:$mode")
    done
  done

  local idx=0 wave=0 failures=0
  while (( idx < ${#cases[@]} )); do
    wave_pids=()
    local slot=0
    while (( slot < ${#groups[@]} && idx < ${#cases[@]} )); do
      IFS=: read -r c_tp c_backend c_mode <<<"${cases[$idx]}"
      local port=$((PORT_BASE + tp * 100 + wave * 20 + slot))
      run_case_guarded "$c_tp" "$c_backend" "$c_mode" "${groups[$slot]}" "$port" &
      wave_pids+=("$!")
      idx=$((idx + 1))
      slot=$((slot + 1))
    done
    for pid in "${wave_pids[@]}"; do
      if ! wait "$pid"; then
        failures=$((failures + 1))
      fi
    done
    wave=$((wave + 1))
  done

  if (( failures > 0 )); then
    echo "TP${tp} completed with $failures failed case(s)" >&2
    return 1
  fi
}

chmod +x "$LAUNCHER"
{
  printf 'image=%s\nout=%s\nmax_num_seqs=%s\ndecode_concurrency=%s\n' \
    "$IMAGE" "$OUT" "$MAX_NUM_SEQS" "$DECODE_CONCURRENCY"
  printf 'backends=%s\nmodes=%s\ntps=%s\n' "$BACKENDS" "$MODES" "$TPS"
} | tee "$OUT/run-config.txt"
printf '%s sweep_start image=%s out=%s backends=%s modes=%s tps=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$IMAGE" "$OUT" "$BACKENDS" "$MODES" "$TPS" \
  >> "$PROGRESS_FILE"
for tp in $(split_csv "$TPS"); do
  run_tp_matrix "$tp"
done
printf '%s sweep_done out=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OUT" >> "$PROGRESS_FILE"
