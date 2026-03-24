#!/usr/bin/env bash
# run.sh — Arrival Burstiness Experiment (H1)
#
# Tests whether bursty (Gamma CV=3) vs smooth (Poisson CV=1) arrivals produce
# significantly different TTFT and E2E latencies at equivalent throughput.
#
# Produces 18 JSON result files:
#   output/{smooth,bursty}_rate{50,150,300}_seed{42,123,456}.json
#
# Usage: ./run.sh [--skip-build]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BLIS="$PROJECT_ROOT/blis"
RESULTS_DIR="$SCRIPT_DIR/output"
WORKLOADS_DIR="$SCRIPT_DIR/workloads"

SEEDS=(42 123 456)
# Saturation rate for qwen3-14b/H100 with this workload profile is ~22.5 req/s.
# Test at sub-saturation utilization levels to observe the queueing effect:
#   5 req/s  → ρ ≈ 0.22  (low)
#  12 req/s  → ρ ≈ 0.53  (medium)
#  18 req/s  → ρ ≈ 0.80  (high — Kingman predicts 5x effect at CV_s=1)
#  21 req/s  → ρ ≈ 0.93  (very high — strong queueing amplification)
RATES=(5 12 18 21)
MODEL="qwen/qwen3-14b"
NUM_INSTANCES=1
NUM_REQUESTS=3000
HORIZON=600000000  # 600 seconds simulated time (covers even bursty queuing at ρ=0.93)

mkdir -p "$RESULTS_DIR"

# Build unless skipped
if [[ "${1:-}" != "--skip-build" ]]; then
    echo "Building blis..."
    cd "$PROJECT_ROOT"
    go build -o blis main.go
    echo "Build complete."
fi

# run_condition <smooth|bursty> <rate> <seed>
# Generates a rate-substituted workload YAML, runs the simulator, saves results.
run_condition() {
    local condition="$1"
    local rate="$2"
    local seed="$3"

    local template="$WORKLOADS_DIR/${condition}_template.yaml"
    local tmp_yaml="$RESULTS_DIR/tmp_${condition}_${rate}_${seed}.yaml"
    local results_file="$RESULTS_DIR/${condition}_rate${rate}_seed${seed}.json"

    # Substitute rate placeholder
    sed "s/__RATE__/${rate}.0/g" "$template" > "$tmp_yaml"

    # Skip if already done
    if [[ -f "$results_file" ]]; then
        rm -f "$tmp_yaml"
        printf "  %-8s rate=%-4s seed=%s  [skipped — already exists]\n" "$condition" "$rate" "$seed"
        return 0
    fi

    # blis must run from PROJECT_ROOT so it can locate defaults.yaml
    (cd "$PROJECT_ROOT" && "$BLIS" run \
        --model "$MODEL" \
        --workload-spec "$tmp_yaml" \
        --num-instances "$NUM_INSTANCES" \
        --num-requests "$NUM_REQUESTS" \
        --seed "$seed" \
        --horizon "$HORIZON" \
        --results-path "$results_file" \
        2>/dev/null)

    rm -f "$tmp_yaml"

    # Print key metrics inline for progress monitoring
    local summary
    summary=$(python3 -c "
import json, sys
try:
    d = json.load(open('$results_file'))
    completed = d.get('completed_requests', 0)
    timed_out = d.get('timed_out_requests', 0)
    ttft_mean = d.get('ttft_mean_ms', 0)
    ttft_p99  = d.get('ttft_p99_ms', 0)
    e2e_mean  = d.get('e2e_mean_ms', 0)
    e2e_p99   = d.get('e2e_p99_ms', 0)
    rps       = d.get('responses_per_sec', 0)
    print(f'completed={completed} timed_out={timed_out} TTFT mean={ttft_mean:.0f}ms p99={ttft_p99:.0f}ms | E2E mean={e2e_mean:.0f}ms p99={e2e_p99:.0f}ms | rps={rps:.1f}')
except Exception as e:
    print(f'parse error: {e}', file=sys.stderr)
    print('N/A')
" 2>/dev/null || echo "N/A")

    printf "  %-8s rate=%-4s seed=%s  %s\n" "$condition" "$rate" "$seed" "$summary"
}

echo "========================================================"
echo " H1: Arrival Burstiness Experiment"
echo " Model: $MODEL | Instances: $NUM_INSTANCES"
echo " Rates: ${RATES[*]} req/s | Seeds: ${SEEDS[*]}"
echo " 24 total runs (4 rates × 2 conditions × 3 seeds)"
echo "========================================================"
echo ""

TOTAL_RUNS=0
FAILED_RUNS=0

for rate in "${RATES[@]}"; do
    echo "--- Rate: ${rate} req/s ---"
    for seed in "${SEEDS[@]}"; do
        for condition in smooth bursty; do
            if run_condition "$condition" "$rate" "$seed"; then
                TOTAL_RUNS=$((TOTAL_RUNS + 1))
            else
                echo "  ERROR: $condition rate=$rate seed=$seed failed"
                FAILED_RUNS=$((FAILED_RUNS + 1))
            fi
        done
    done
    echo ""
done

echo "========================================================"
echo " Completed: $TOTAL_RUNS runs, $FAILED_RUNS failures"
echo " Results: $RESULTS_DIR/"
echo " Next: python3 analyze.py"
echo "========================================================"
