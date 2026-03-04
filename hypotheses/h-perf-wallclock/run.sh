#!/bin/bash
# h-perf-wallclock: Wall-clock performance optimization for prefix-affinity routing
# Measures wall-clock time before/after optimizations across 3 workloads and 3 seeds.
# Usage: ./run.sh [--rebuild]
#
# Experiment classification:
#   Type 1 (Deterministic): INV-6 byte-identical output, single seed sufficient
#   Type 2 (Statistical/Dominance): Wall-clock reduction, 3 seeds (42, 123, 456)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$REPO_ROOT/blis"
BINARY_BASELINE="$REPO_ROOT/blis_baseline"
MODEL="meta-llama/llama-3.1-8b-instruct"

# Record provenance
echo "=== Experiment Provenance ===" >&2
echo "  Commit: $(cd "$REPO_ROOT" && git rev-parse HEAD)" >&2
echo "  Branch: $(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD)" >&2
echo "  Go version: $(go version)" >&2
echo "  Platform: $(uname -ms)" >&2

# Build optimized binary (current branch)
echo "Building optimized blis..." >&2
(cd "$REPO_ROOT" && go build -o blis main.go)

# Build baseline binary from main (pre-optimization) via worktree
BASELINE_REF="main"
BASELINE_WORKTREE=$(mktemp -d)
echo "Building baseline blis from $BASELINE_REF (worktree: $BASELINE_WORKTREE)..." >&2
(cd "$REPO_ROOT" && git worktree add "$BASELINE_WORKTREE" "$BASELINE_REF" 2>/dev/null)
(cd "$BASELINE_WORKTREE" && go build -o "$REPO_ROOT/blis_baseline" main.go)
(cd "$REPO_ROOT" && git worktree remove "$BASELINE_WORKTREE" 2>/dev/null)

RESULTS_DIR=$(mktemp -d)
trap 'rm -rf "$RESULTS_DIR" "$BINARY_BASELINE"' EXIT

# -- Configuration -----------------------------------------------------------
WORKLOADS=(
  "$REPO_ROOT/workload-mert/workload_v2_cache_warmup.yaml"
  "$REPO_ROOT/workload-mert/workload_v2_load_spikes.yaml"
  "$REPO_ROOT/workload-mert/workload_v2_multiturn.yaml"
)
NUM_INSTANCES=4
BLOCK_SIZE=16  # explicit for reproducibility (default)
SEEDS=(42 123 456)
NUM_RUNS=3  # runs per seed for wall-clock variance

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

# -- Precondition checks (ED-3) ---------------------------------------------
echo "=== Precondition Checks (ED-3) ===" >&2
# Verify prefix-affinity is in policy
if ! grep -q "prefix-affinity" "$POLICY_PA"; then
  echo "ERROR: precondition failed — prefix-affinity not in policy" >&2; exit 1
fi
# Verify workload files exist
for w in "${WORKLOADS[@]}"; do
  if [[ ! -f "$w" ]]; then
    echo "ERROR: precondition failed — workload file missing: $w" >&2; exit 1
  fi
done
# Verify binaries exist
for b in "$BINARY" "$BINARY_BASELINE"; do
  if [[ ! -x "$b" ]]; then
    echo "ERROR: precondition failed — binary not found: $b" >&2; exit 1
  fi
done
echo "  All preconditions passed." >&2

# -- Helper ------------------------------------------------------------------
measure_total_ms() {
  local binary="$1"
  local policy_file="$2"
  local seed="$3"
  local total_ms=0
  for w in "${WORKLOADS[@]}"; do
    local out="$RESULTS_DIR/tmp_out.txt"
    local start_ns end_ns elapsed_ms
    start_ns=$(python3 -c 'import time; print(int(time.time_ns()))')
    "$binary" run --model "$MODEL" --num-instances "$NUM_INSTANCES" --seed "$seed" \
      --block-size-in-tokens "$BLOCK_SIZE" --latency-model blackbox \
      --policy-config "$policy_file" --workload-spec "$w" --log error > "$out" 2>/dev/null
    end_ns=$(python3 -c 'import time; print(int(time.time_ns()))')
    elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
    total_ms=$((total_ms + elapsed_ms))
  done
  echo "$total_ms"
}

# -- INV-6 Check: byte-identical output (Type 1, single seed) ---------------
echo "=== INV-6 Determinism Check ===" >&2
inv6_baseline="$RESULTS_DIR/inv6_baseline.txt"
inv6_optimized="$RESULTS_DIR/inv6_optimized.txt"
"$BINARY_BASELINE" run --model "$MODEL" --num-instances "$NUM_INSTANCES" --seed 42 \
  --block-size-in-tokens "$BLOCK_SIZE" --latency-model blackbox \
  --policy-config "$POLICY_PA" --workload-spec "${WORKLOADS[0]}" --log error > "$inv6_baseline" 2>/dev/null
"$BINARY" run --model "$MODEL" --num-instances "$NUM_INSTANCES" --seed 42 \
  --block-size-in-tokens "$BLOCK_SIZE" --latency-model blackbox \
  --policy-config "$POLICY_PA" --workload-spec "${WORKLOADS[0]}" --log error > "$inv6_optimized" 2>/dev/null

if diff -q "$inv6_baseline" "$inv6_optimized" > /dev/null 2>&1; then
  echo "INV-6: PASS (baseline vs optimized byte-identical)" >&2
else
  echo "INV-6: FAIL (output differs between baseline and optimized!)" >&2
  exit 1
fi

# -- Baseline measurement (reproducible, same environment) -------------------
echo "=== Baseline (pre-optimization, PA enabled, 3 seeds × $NUM_RUNS runs) ===" >&2
BASELINE_RESULTS="$RESULTS_DIR/baseline_times.txt"
for seed in "${SEEDS[@]}"; do
  for i in $(seq 1 "$NUM_RUNS"); do
    ms=$(measure_total_ms "$BINARY_BASELINE" "$POLICY_PA" "$seed")
    echo "$ms" >> "$BASELINE_RESULTS"
    echo "  Seed $seed, Run $i: ${ms}ms" >&2
  done
done

# -- Optimized measurement (PA enabled, 3 seeds × NUM_RUNS runs) ------------
echo "=== Optimized (PA enabled, 3 seeds × $NUM_RUNS runs) ===" >&2
PA_RESULTS="$RESULTS_DIR/pa_times.txt"
for seed in "${SEEDS[@]}"; do
  for i in $(seq 1 "$NUM_RUNS"); do
    ms=$(measure_total_ms "$BINARY" "$POLICY_PA" "$seed")
    echo "$ms" >> "$PA_RESULTS"
    echo "  Seed $seed, Run $i: ${ms}ms" >&2
  done
done

# -- Negative control: load-balance only (3 seeds × NUM_RUNS runs) ----------
echo "=== Negative Control: LB-only optimized vs LB-only baseline (3 seeds × $NUM_RUNS runs) ===" >&2
LB_OPT_RESULTS="$RESULTS_DIR/lb_opt_times.txt"
LB_BASE_RESULTS="$RESULTS_DIR/lb_base_times.txt"
for seed in "${SEEDS[@]}"; do
  for i in $(seq 1 "$NUM_RUNS"); do
    ms_opt=$(measure_total_ms "$BINARY" "$POLICY_LB" "$seed")
    ms_base=$(measure_total_ms "$BINARY_BASELINE" "$POLICY_LB" "$seed")
    echo "$ms_opt" >> "$LB_OPT_RESULTS"
    echo "$ms_base" >> "$LB_BASE_RESULTS"
    echo "  Seed $seed, Run $i: opt=${ms_opt}ms base=${ms_base}ms" >&2
  done
done

# -- Analyze -----------------------------------------------------------------
python3 "$SCRIPT_DIR/analyze.py" "$BASELINE_RESULTS" "$PA_RESULTS" "$LB_OPT_RESULTS" "$LB_BASE_RESULTS"
