#!/bin/bash
# H29: Stale Routing Snapshots Degrade Tail Latency Under High Request Rates
#
# Hypothesis: Increasing the snapshot refresh interval from 1ms to 100ms degrades
# TTFT p99 by at least 20% for weighted routing at high request rates (>80%
# saturation, 4 instances), because stale load signals cause the router to
# repeatedly select already-loaded instances, creating transient load imbalance.
#
# CRITICAL DESIGN NOTE: The --snapshot-refresh-interval flag only controls
# KVUtilization staleness (sim/cluster/snapshot.go:39-48). QueueDepth and
# BatchSize are ALWAYS Immediate regardless of the interval. Therefore:
#   - kv-utilization scorer IS affected (reads KVUtilization directly)
#   - queue-depth scorer is NOT affected (reads EffectiveLoad = QueueDepth +
#     BatchSize + PendingRequests, all Immediate/synchronous)
#
# The experiment tests kv-utilization:1 as the primary config, with queue-depth:1
# as a negative control (should show zero effect from interval change).
#
# Usage: ./run.sh [--rebuild]
#   --rebuild  Force rebuild of the binary
#
# Requires: Go 1.24+, Python 3

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"

setup_experiment "${1:-}"

# Must run from repo root so defaults.yaml is found by the simulator
cd "$REPO_ROOT"

# ── Parameters ──────────────────────────────────────────────────────────────
# Capacity: 4 instances, 512 input + 512 output tokens
#   step_time = beta0 + beta1*512 + beta2*512 = 6910 + 9047 + 1454 = 17411 us
#   per-instance capacity = 1e6 / 17411 = ~57.4 req/s
#   cluster capacity = 4 * 57.4 = ~229.7 req/s
#   85% saturation = 195 req/s
RATE=195
NUM_REQUESTS=500
NUM_INSTANCES=4
PROMPT_TOKENS=512
OUTPUT_TOKENS=512
SEEDS=(42 123 456)

# Snapshot refresh intervals (microseconds)
INTERVAL_FRESH=1000     # 1ms — frequent refresh
INTERVAL_STALE=100000   # 100ms — stale signals (~5.7 step times between refreshes)

echo "============================================================================"
echo "  H29: Stale Routing Snapshots Degrade Tail Latency"
echo "  Rate: ${RATE} req/s (85% of 4-instance capacity)"
echo "  Snapshot intervals: ${INTERVAL_FRESH}us (fresh) vs ${INTERVAL_STALE}us (stale)"
echo "============================================================================"
echo ""

# ── Experiment 1: KV-utilization scorer (AFFECTED by staleness) ─────────────
echo "=== Experiment 1: kv-utilization:1 (staleness-sensitive scorer) ==="
echo ""

for SEED in "${SEEDS[@]}"; do
    echo "  Seed ${SEED}: fresh interval (${INTERVAL_FRESH}us)..."
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/exp1_kv_fresh_s${SEED}.txt" \
        --model "$MODEL" \
        --num-instances $NUM_INSTANCES \
        --num-requests $NUM_REQUESTS \
        --rate $RATE \
        --prompt-tokens $PROMPT_TOKENS \
        --output-tokens $OUTPUT_TOKENS \
        --routing-policy weighted \
        --routing-scorers "kv-utilization:1" \
        --snapshot-refresh-interval $INTERVAL_FRESH \
        --seed "$SEED" \
        --results-path "$RESULTS_DIR/exp1_kv_fresh_s${SEED}.json" \
        --log error

    echo "  Seed ${SEED}: stale interval (${INTERVAL_STALE}us)..."
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/exp1_kv_stale_s${SEED}.txt" \
        --model "$MODEL" \
        --num-instances $NUM_INSTANCES \
        --num-requests $NUM_REQUESTS \
        --rate $RATE \
        --prompt-tokens $PROMPT_TOKENS \
        --output-tokens $OUTPUT_TOKENS \
        --routing-policy weighted \
        --routing-scorers "kv-utilization:1" \
        --snapshot-refresh-interval $INTERVAL_STALE \
        --seed "$SEED" \
        --results-path "$RESULTS_DIR/exp1_kv_stale_s${SEED}.json" \
        --log error
done

echo ""

# ── Experiment 2: Queue-depth scorer (NOT affected — negative control) ──────
echo "=== Experiment 2: queue-depth:1 (negative control — always-fresh scorer) ==="
echo ""

for SEED in "${SEEDS[@]}"; do
    echo "  Seed ${SEED}: fresh interval (${INTERVAL_FRESH}us)..."
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/exp2_qd_fresh_s${SEED}.txt" \
        --model "$MODEL" \
        --num-instances $NUM_INSTANCES \
        --num-requests $NUM_REQUESTS \
        --rate $RATE \
        --prompt-tokens $PROMPT_TOKENS \
        --output-tokens $OUTPUT_TOKENS \
        --routing-policy weighted \
        --routing-scorers "queue-depth:1" \
        --snapshot-refresh-interval $INTERVAL_FRESH \
        --seed "$SEED" \
        --results-path "$RESULTS_DIR/exp2_qd_fresh_s${SEED}.json" \
        --log error

    echo "  Seed ${SEED}: stale interval (${INTERVAL_STALE}us)..."
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/exp2_qd_stale_s${SEED}.txt" \
        --model "$MODEL" \
        --num-instances $NUM_INSTANCES \
        --num-requests $NUM_REQUESTS \
        --rate $RATE \
        --prompt-tokens $PROMPT_TOKENS \
        --output-tokens $OUTPUT_TOKENS \
        --routing-policy weighted \
        --routing-scorers "queue-depth:1" \
        --snapshot-refresh-interval $INTERVAL_STALE \
        --seed "$SEED" \
        --results-path "$RESULTS_DIR/exp2_qd_stale_s${SEED}.json" \
        --log error
done

echo ""

# ── Experiment 3: Composite scorer (mitigation test) ────────────────────────
echo "=== Experiment 3: kv-utilization:2,queue-depth:2 (mitigation via composite) ==="
echo ""

for SEED in "${SEEDS[@]}"; do
    echo "  Seed ${SEED}: fresh interval (${INTERVAL_FRESH}us)..."
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/exp3_combo_fresh_s${SEED}.txt" \
        --model "$MODEL" \
        --num-instances $NUM_INSTANCES \
        --num-requests $NUM_REQUESTS \
        --rate $RATE \
        --prompt-tokens $PROMPT_TOKENS \
        --output-tokens $OUTPUT_TOKENS \
        --routing-policy weighted \
        --routing-scorers "kv-utilization:2,queue-depth:2" \
        --snapshot-refresh-interval $INTERVAL_FRESH \
        --seed "$SEED" \
        --results-path "$RESULTS_DIR/exp3_combo_fresh_s${SEED}.json" \
        --log error

    echo "  Seed ${SEED}: stale interval (${INTERVAL_STALE}us)..."
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/exp3_combo_stale_s${SEED}.txt" \
        --model "$MODEL" \
        --num-instances $NUM_INSTANCES \
        --num-requests $NUM_REQUESTS \
        --rate $RATE \
        --prompt-tokens $PROMPT_TOKENS \
        --output-tokens $OUTPUT_TOKENS \
        --routing-policy weighted \
        --routing-scorers "kv-utilization:2,queue-depth:2" \
        --snapshot-refresh-interval $INTERVAL_STALE \
        --seed "$SEED" \
        --results-path "$RESULTS_DIR/exp3_combo_stale_s${SEED}.json" \
        --log error
done

echo ""

# ── Experiment 4: Interval sweep (dose-response for kv-utilization:1) ───────
echo "=== Experiment 4: Interval sweep (kv-utilization:1, seed=42) ==="
echo ""

for INTERVAL in 0 1000 5000 10000 50000 100000 500000; do
    echo "  Interval: ${INTERVAL}us..."
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/exp4_sweep_i${INTERVAL}.txt" \
        --model "$MODEL" \
        --num-instances $NUM_INSTANCES \
        --num-requests $NUM_REQUESTS \
        --rate $RATE \
        --prompt-tokens $PROMPT_TOKENS \
        --output-tokens $OUTPUT_TOKENS \
        --routing-policy weighted \
        --routing-scorers "kv-utilization:1" \
        --snapshot-refresh-interval "$INTERVAL" \
        --seed 42 \
        --results-path "$RESULTS_DIR/exp4_sweep_i${INTERVAL}.json" \
        --log error
done

echo ""

# ── Analysis ────────────────────────────────────────────────────────────────
echo "============================================================================"
echo "  Analysis"
echo "============================================================================"
echo ""

python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR"
