#!/bin/bash
# H26: Admission latency causal ordering
# Tests whether adding admission latency delays E2E by exactly that amount under low load.
#
# Family: Structural model
# VV&UQ: Verification (deterministic)
# Reference: None (first experiment in this area)
#
# Design:
#   - 3 configurations: admission-latency=0, 10000 (10ms), 50000 (50ms) in ticks/us
#   - Low rate (10 req/s) → no queuing confound
#   - CONSTANT distributions for input/output → minimal noise
#   - 4 instances, 50 requests, seed=42
#   - Both TTFT and E2E measured from original ArrivalTime (verified in sim/simulator.go:564)
#   - Expected: TTFT and E2E mean both increase by exactly 10ms with admission latency

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKTREE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$WORKTREE_DIR/blis"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR"

MODEL="meta-llama/llama-3.1-8b-instruct"
NUM_INSTANCES=4
NUM_REQUESTS=50
SEED=42
HORIZON=60000000  # 60s in ticks (us) — plenty for 50 requests at 10 req/s

# Workload: CONSTANT input=128, CONSTANT output=32, rate=10
# Low rate avoids queuing; constant distributions eliminate variance.
# Field names verified against sim/workload/spec.go struct tags.
make_workload() {
    local outfile=$1
    cat > "$outfile" << YAMLEOF
version: "1"
seed: 42
category: language
aggregate_rate: 10.0
num_requests: 50
clients:
  - id: "h26-client"
    tenant_id: "default"
    slo_class: "batch"
    rate_fraction: 1.0
    streaming: false
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 128
    output_distribution:
      type: constant
      params:
        value: 32
YAMLEOF
}

echo "=== H26: Admission Latency Causal Ordering ==="
echo ""

# --- Config A: No admission latency (baseline) ---
echo "--- Config A: admission-latency=0 (baseline) ---"
WORKLOAD_FILE_A="$RESULTS_DIR/workload_a.yaml"
make_workload "$WORKLOAD_FILE_A"

"$BINARY" run \
  --model "$MODEL" \
  --num-instances "$NUM_INSTANCES" \
  --seed "$SEED" \
  --horizon "$HORIZON" \
  --admission-latency 0 \
  --routing-latency 0 \
  --routing-policy least-loaded \
  --workload-spec "$WORKLOAD_FILE_A" \
  --results-path "$RESULTS_DIR/results_a.json" \
  > "$RESULTS_DIR/stdout_a.txt" 2>"$RESULTS_DIR/stderr_a.txt"

echo "Config A done."

# --- Config B: 10ms admission latency ---
echo "--- Config B: admission-latency=10000 (10ms) ---"
WORKLOAD_FILE_B="$RESULTS_DIR/workload_b.yaml"
make_workload "$WORKLOAD_FILE_B"

"$BINARY" run \
  --model "$MODEL" \
  --num-instances "$NUM_INSTANCES" \
  --seed "$SEED" \
  --horizon "$HORIZON" \
  --admission-latency 10000 \
  --routing-latency 0 \
  --routing-policy least-loaded \
  --workload-spec "$WORKLOAD_FILE_B" \
  --results-path "$RESULTS_DIR/results_b.json" \
  > "$RESULTS_DIR/stdout_b.txt" 2>"$RESULTS_DIR/stderr_b.txt"

echo "Config B done."

# --- Config C: 50ms admission latency (linearity check) ---
echo "--- Config C: admission-latency=50000 (50ms) ---"
WORKLOAD_FILE_C="$RESULTS_DIR/workload_c.yaml"
make_workload "$WORKLOAD_FILE_C"

"$BINARY" run \
  --model "$MODEL" \
  --num-instances "$NUM_INSTANCES" \
  --seed "$SEED" \
  --horizon "$HORIZON" \
  --admission-latency 50000 \
  --routing-latency 0 \
  --routing-policy least-loaded \
  --workload-spec "$WORKLOAD_FILE_C" \
  --results-path "$RESULTS_DIR/results_c.json" \
  > "$RESULTS_DIR/stdout_c.txt" 2>"$RESULTS_DIR/stderr_c.txt"

echo "Config C done."

echo ""
echo "=== Running analysis ==="
python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR"
