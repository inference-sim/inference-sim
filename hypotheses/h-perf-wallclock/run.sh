#!/bin/bash
# h-perf-wallclock: Wall-clock performance optimization for prefix-affinity routing
# Measures wall-clock time of optimized binary across 3 workloads.
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$REPO_ROOT/blis"
MODEL="meta-llama/llama-3.1-8b-instruct"

# Build binary
if [[ "${1:-}" == "--rebuild" ]] || [[ ! -x "$BINARY" ]]; then
  echo "Building blis..." >&2
  (cd "$REPO_ROOT" && go build -o blis main.go)
fi

RESULTS_DIR=$(mktemp -d)
trap 'rm -rf "$RESULTS_DIR"' EXIT

# -- Configuration -----------------------------------------------------------
WORKLOADS=(
  "$REPO_ROOT/workload-mert/workload_v2_cache_warmup.yaml"
  "$REPO_ROOT/workload-mert/workload_v2_load_spikes.yaml"
  "$REPO_ROOT/workload-mert/workload_v2_multiturn.yaml"
)
NUM_INSTANCES=4
SEED=42
NUM_RUNS=5

# Policy: prefix-affinity + load-balance (exercises the hot path)
POLICY_PA="$RESULTS_DIR/policy_pa.yaml"
cat > "$POLICY_PA" <<'YAML'
admission:
  policy: always-admit
priority:
  policy: constant
routing:
  policy: weighted
  scorers:
  - name: prefix-affinity
    weight: 1.0
  - name: load-balance
    weight: 1.0
scheduler: fcfs
YAML

# Negative control policy: load-balance only (no prefix-affinity)
POLICY_LB="$RESULTS_DIR/policy_lb.yaml"
cat > "$POLICY_LB" <<'YAML'
admission:
  policy: always-admit
priority:
  policy: constant
routing:
  policy: weighted
  scorers:
  - name: load-balance
    weight: 1.0
scheduler: fcfs
YAML

# -- Helper ------------------------------------------------------------------
measure_total_ms() {
  local policy_file="$1"
  local total_ms=0
  for w in "${WORKLOADS[@]}"; do
    local out="$RESULTS_DIR/tmp_out.txt"
    local start_ns end_ns elapsed_ms
    start_ns=$(python3 -c 'import time; print(int(time.time_ns()))')
    "$BINARY" run --model "$MODEL" --num-instances "$NUM_INSTANCES" --seed "$SEED" \
      --policy-config "$policy_file" --workload-spec "$w" --log error > "$out" 2>/dev/null
    end_ns=$(python3 -c 'import time; print(int(time.time_ns()))')
    elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
    total_ms=$((total_ms + elapsed_ms))
  done
  echo "$total_ms"
}

# -- INV-6 Check: byte-identical output --------------------------------------
echo "=== INV-6 Determinism Check ===" >&2
inv6_out1="$RESULTS_DIR/inv6_run1.txt"
inv6_out2="$RESULTS_DIR/inv6_run2.txt"
"$BINARY" run --model "$MODEL" --num-instances "$NUM_INSTANCES" --seed "$SEED" \
  --policy-config "$POLICY_PA" --workload-spec "${WORKLOADS[0]}" --log error > "$inv6_out1" 2>/dev/null
"$BINARY" run --model "$MODEL" --num-instances "$NUM_INSTANCES" --seed "$SEED" \
  --policy-config "$POLICY_PA" --workload-spec "${WORKLOADS[0]}" --log error > "$inv6_out2" 2>/dev/null

if diff -q "$inv6_out1" "$inv6_out2" > /dev/null 2>&1; then
  echo "INV-6: PASS (byte-identical output)" >&2
else
  echo "INV-6: FAIL (output differs between runs!)" >&2
  exit 1
fi

# -- Main experiment: prefix-affinity enabled --------------------------------
echo "=== Prefix-Affinity Enabled ($NUM_RUNS runs) ===" >&2
PA_RESULTS="$RESULTS_DIR/pa_times.txt"
for i in $(seq 1 "$NUM_RUNS"); do
  ms=$(measure_total_ms "$POLICY_PA")
  echo "$ms" >> "$PA_RESULTS"
  echo "  Run $i: ${ms}ms" >&2
done

# -- Negative control: load-balance only ------------------------------------
echo "=== Negative Control: Load-Balance Only ($NUM_RUNS runs) ===" >&2
LB_RESULTS="$RESULTS_DIR/lb_times.txt"
for i in $(seq 1 "$NUM_RUNS"); do
  ms=$(measure_total_ms "$POLICY_LB")
  echo "$ms" >> "$LB_RESULTS"
  echo "  Run $i: ${ms}ms" >&2
done

# -- Analyze -----------------------------------------------------------------
python3 "$SCRIPT_DIR/analyze.py" "$PA_RESULTS" "$LB_RESULTS"
