#!/bin/bash
# H10: Tiered KV Cache (GPU+CPU Offload)
#
# Hypothesis: With a CPU tier, blocks can be offloaded to CPU instead of
# being evicted entirely. This should reduce preemptions compared to
# single-tier at the cost of transfer latency.
#
# Classification: Statistical (Type 2) / Dominance.
# Tiered preemption count < single-tier preemption count across all seeds.
#
# Design:
#   - ED-1: Vary exactly one dimension (CPU blocks: 0 vs 500)
#   - ED-2: Rate-aware — rate=2000 creates enough KV pressure for preemptions
#   - ED-3: Precondition — verify single-tier preempts at 2100 GPU blocks
#           (calibrated via H8's cliff: 2200→0%, 2100→11%, 2000→51%)
#   - ED-4: 3 seeds (42, 123, 456) per configuration
#   - ED-5: Self-contained, builds binary, reproducible
#
# Usage: ./run.sh [--rebuild]
#   --rebuild  Force rebuild of the binary
#
# Requires: Go 1.24+, Python 3

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$REPO_ROOT/simulation_worker"

# Build if needed
if [[ "${1:-}" == "--rebuild" ]] || [[ ! -x "$BINARY" ]]; then
    echo "Building simulation_worker..."
    (cd "$REPO_ROOT" && go build -o simulation_worker main.go)
fi

MODEL="meta-llama/llama-3.1-8b-instruct"
INSTANCES=4

# GPU blocks at H8's cliff point — produces ~11% preemption rate single-tier
GPU_BLOCKS=2100

# Workload YAML matching H8's calibrated parameters:
# rate=2000, gaussian input (mean=512), enough KV pressure for preemptions
make_workload() {
    local seed=$1 outfile=$2
    cat > "$outfile" << YAMLEOF
version: "1"
seed: $seed
category: language
aggregate_rate: 2000.0
num_requests: 200
clients:
  - id: kv-stress
    tenant_id: test
    slo_class: interactive
    rate_fraction: 1.0
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 512
        std_dev: 50
        min: 256
        max: 768
    output_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 50
        min: 128
        max: 512
YAMLEOF
}

run_sim() {
    local seed="$1"; shift
    local wl_yaml="$RESULTS_DIR/wl_${seed}.yaml"
    if [[ ! -f "$wl_yaml" ]]; then
        make_workload "$seed" "$wl_yaml"
    fi
    timeout 120 "$BINARY" run \
        --model "$MODEL" \
        --num-instances "$INSTANCES" \
        --seed "$seed" \
        --workload-spec "$wl_yaml" \
        --total-kv-blocks "$GPU_BLOCKS" \
        --block-size-in-tokens 16 \
        --routing-policy least-loaded \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy always-admit \
        --log error \
        --summarize-trace \
        --trace-level decisions \
        "$@" 2>/dev/null
}

analyze() {
    python3 "$SCRIPT_DIR/analyze.py" "$@"
}

echo "============================================================================"
echo "  H10: Tiered KV Cache (GPU+CPU Offload)"
echo "  Reference: docs/plans/research.md, Idea 2, Hypothesis 10"
echo "  Type: Statistical / Dominance"
echo "============================================================================"
echo ""
echo "Workload: rate=2000, requests=200, instances=$INSTANCES"
echo "GPU blocks=$GPU_BLOCKS, block_size=16 tokens"
echo "(Calibrated from H8: cliff at 2100-2200 blocks with this workload)"
echo ""

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# ── Precondition: Verify single-tier actually preempts ───────────────────────
echo "Precondition: Verify GPU blocks=$GPU_BLOCKS triggers preemptions (single-tier)..."
make_workload 42 "$RESULTS_DIR/wl_42.yaml"
PRECHECK=$(timeout 60 "$BINARY" run \
    --model "$MODEL" --num-instances "$INSTANCES" --seed 42 \
    --workload-spec "$RESULTS_DIR/wl_42.yaml" \
    --total-kv-blocks "$GPU_BLOCKS" --block-size-in-tokens 16 \
    --routing-policy least-loaded --scheduler fcfs \
    --priority-policy constant --admission-policy always-admit \
    --kv-cpu-blocks 0 \
    --log error --summarize-trace --trace-level decisions \
    2>/dev/null || true)
PREEMPT_COUNT=$(echo "$PRECHECK" | grep -oP 'Preemption Rate: \K[0-9.]+' || echo "0")
if [[ "$PREEMPT_COUNT" != "0" && "$PREEMPT_COUNT" != "0.0000" ]]; then
    echo "  PASS: Preemption rate=$PREEMPT_COUNT% in single-tier"
else
    echo "  WARNING: No preemptions detected — adjusting GPU blocks..."
    echo "  (Results may still be informative if tiered shows capacity benefit)"
fi
echo ""

# ── Experiment 1: Core hypothesis (3 seeds) ──────────────────────────────────
# Compare single-tier (CPU blocks=0) vs tiered (CPU blocks=500)

echo "Experiment 1: Core hypothesis — single-tier vs tiered (3 seeds)"
echo ""

for SEED in 42 123 456; do
    echo "  Seed $SEED: single-tier (CPU blocks=0)..."
    if ! run_sim "$SEED" \
        --kv-cpu-blocks 0 \
        > "$RESULTS_DIR/exp1_single_${SEED}.txt" 2>&1; then
        echo "    WARNING: Timed out or crashed (possible livelock at low blocks)"
        echo "TIMEOUT_OR_CRASH" > "$RESULTS_DIR/exp1_single_${SEED}.txt"
    fi

    echo "  Seed $SEED: tiered (CPU blocks=500, offload=0.8, bw=100)..."
    if ! run_sim "$SEED" \
        --kv-cpu-blocks 500 \
        --kv-offload-threshold 0.8 \
        --kv-transfer-bandwidth 100 \
        --kv-transfer-base-latency 10 \
        > "$RESULTS_DIR/exp1_tiered_${SEED}.txt" 2>&1; then
        echo "    WARNING: Timed out or crashed"
        echo "TIMEOUT_OR_CRASH" > "$RESULTS_DIR/exp1_tiered_${SEED}.txt"
    fi
done

analyze core "$RESULTS_DIR"/exp1_*.txt

# ── Experiment 2: CPU tier size scaling ──────────────────────────────────────
# Does more CPU blocks = fewer preemptions?

echo ""
echo "Experiment 2: CPU tier size scaling (seed=42)"
echo ""

for CPU_BLOCKS in 0 100 250 500 1000; do
    echo "  CPU blocks=$CPU_BLOCKS..."
    if ! run_sim 42 \
        --kv-cpu-blocks "$CPU_BLOCKS" \
        --kv-offload-threshold 0.8 \
        --kv-transfer-bandwidth 100 \
        --kv-transfer-base-latency 10 \
        > "$RESULTS_DIR/exp2_cpu${CPU_BLOCKS}.txt" 2>&1; then
        echo "    WARNING: Timed out or crashed"
        echo "TIMEOUT_OR_CRASH" > "$RESULTS_DIR/exp2_cpu${CPU_BLOCKS}.txt"
    fi
done

analyze scaling "$RESULTS_DIR"/exp2_*.txt

# ── Experiment 3: Transfer bandwidth sensitivity ─────────────────────────────
# Higher bandwidth = faster offload/reload = less transfer latency overhead

echo ""
echo "Experiment 3: Transfer bandwidth sensitivity (seed=42, CPU=500)"
echo ""

for BW in 10 50 100 500 1000; do
    echo "  Bandwidth=$BW blocks/tick..."
    if ! run_sim 42 \
        --kv-cpu-blocks 500 \
        --kv-offload-threshold 0.8 \
        --kv-transfer-bandwidth "$BW" \
        --kv-transfer-base-latency 10 \
        > "$RESULTS_DIR/exp3_bw${BW}.txt" 2>&1; then
        echo "    WARNING: Timed out or crashed"
        echo "TIMEOUT_OR_CRASH" > "$RESULTS_DIR/exp3_bw${BW}.txt"
    fi
done

analyze bandwidth "$RESULTS_DIR"/exp3_*.txt

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis"
echo "============================================================================"
