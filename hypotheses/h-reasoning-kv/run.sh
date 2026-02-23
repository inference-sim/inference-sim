#!/bin/bash
# H-Reasoning-KV: Reasoning Context Accumulation Shifts KV Pressure Cliff
# Tests whether multi-turn reasoning workloads trigger the preemption cliff
# at higher block counts than standard workloads due to demand heterogeneity.
# Also validates #386 fix (DroppedUnservable) at extreme low block counts.
# Usage: ./run.sh [--rebuild]

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
HORIZON=100000000  # 100s, sufficient for all requests to complete
RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

run_sim() {
    local workload_name="$1"
    local workload_yaml="$SCRIPT_DIR/${workload_name}_workload.yaml"
    local blocks="$2"
    local seed="$3"
    local outfile="$RESULTS_DIR/${workload_name}_${blocks}_${seed}.txt"

    "$BINARY" run --model "$MODEL" \
        --seed "$seed" \
        --workload-spec "$workload_yaml" \
        --total-kv-blocks "$blocks" \
        --block-size-in-tokens 16 \
        --max-num-running-reqs 32 \
        --max-num-scheduled-tokens 2048 \
        --num-instances 1 \
        --horizon "$HORIZON" \
        --results-path "$RESULTS_DIR/${workload_name}_${blocks}_${seed}.json" \
        --log warn \
        2>/dev/null > "$outfile" || true

    echo "$outfile"
}

echo "============================================================"
echo "H-Reasoning-KV: KV Pressure Cliff with Reasoning Workloads"
echo "============================================================"
echo ""

# -- Precondition check (ED-3) -----------------------------------------------
echo "=== Precondition Check ==="
echo "Verifying reasoning workload generates expected token patterns..."

PRECOND_OUT=$(run_sim "reasoning" 100000 42)
python3 "$SCRIPT_DIR/analyze.py" --precondition "$RESULTS_DIR/reasoning_100000_42.json"

echo ""

# -- Main sweep ---------------------------------------------------------------
echo "=== KV Pressure Sweep ==="

BLOCK_LEVELS="5000 3000 2000 1500 1200 1000 800 600 400 100"
SEEDS="42 123 456"
WORKLOADS="reasoning standard_matched_throughput standard_matched_sessions"

total_runs=$(echo "$BLOCK_LEVELS" | wc -w)
total_runs=$((total_runs * 3 * 3))
current=0

for BLOCKS in $BLOCK_LEVELS; do
    for SEED in $SEEDS; do
        for WORKLOAD in $WORKLOADS; do
            current=$((current + 1))
            printf "\r  [%d/%d] %s blocks=%s seed=%s" "$current" "$total_runs" "$WORKLOAD" "$BLOCKS" "$SEED"
            run_sim "$WORKLOAD" "$BLOCKS" "$SEED" > /dev/null
        done
    done
done
echo ""
echo ""

# -- Analysis -----------------------------------------------------------------
echo "=== Analysis ==="
python3 "$SCRIPT_DIR/analyze.py" \
    --results-dir "$RESULTS_DIR" \
    --block-levels $BLOCK_LEVELS \
    --seeds $SEEDS
