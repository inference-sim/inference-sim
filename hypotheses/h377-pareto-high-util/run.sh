#!/bin/bash
# H377: Pareto Frontier at High Utilization
# Tests whether a within-workload Pareto frontier emerges at high utilization
# where load-balance routing prevents queueing overload while cache-heavy routing
# wins on TTFT mean via locality.
#
# Extends H17 (which found NO within-workload Pareto frontier at rate=500).
# Config diff vs H17 (ED-6):
#   - SAME: 5 weight configurations, 4 instances, 3 seeds, model
#   - SAME: multiturn-chat-demo workload structure (5 rounds, context accumulate)
#   - CHANGED: rate=1000 (high utilization, ~3x capacity) vs rate=500 in H17
#   - ADDED: rate=300 moderate control (should reproduce H17: no Pareto frontier)
#   - CHANGED: C3 uses queue-depth:3,kv-utilization:2 (matches issue spec) vs queue-depth:3,kv-utilization:3 in H17
#   - CHANGED: C5 uses kv-utilization:5,queue-depth:1 (matches issue spec) vs kv-utilization:5,prefix-affinity:1 in H17
#   - CHANGED: num_requests=500 at each rate (adequate for Pareto analysis)
#
# Reference: hypotheses/h17-pareto-frontier/run.sh
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"

setup_experiment "${1:-}"

SEEDS=(42 123 456)

# 5 weight configurations (from issue #377, matching H17 structure):
# C1: cache-heavy    — prefix-affinity:5,queue-depth:1
# C2: llm-d default  — prefix-affinity:3,queue-depth:2,kv-utilization:2
# C3: load-only      — queue-depth:3,kv-utilization:2
# C4: load-heavy     — prefix-affinity:1,queue-depth:5
# C5: kv-heavy       — kv-utilization:5,queue-depth:1
CONFIGS=(
    "prefix-affinity:5,queue-depth:1"
    "prefix-affinity:3,queue-depth:2,kv-utilization:2"
    "queue-depth:3,kv-utilization:2"
    "prefix-affinity:1,queue-depth:5"
    "kv-utilization:5,queue-depth:1"
)
CONFIG_NAMES=(
    "cache-heavy"
    "llmd-default"
    "load-only"
    "load-heavy"
    "kv-heavy"
)

# Rate levels:
#   high: 1000 req/s (~3x capacity for 4 instances at ~85 req/s each)
#   moderate: 300 req/s (~88% utilization — should reproduce H17 dominance pattern)
RATES=(1000 300)
RATE_NAMES=("high" "moderate")

generate_workload_yaml() {
    local rate="$1"
    local outfile="$2"
    cat > "$outfile" << YAMLEOF
version: "1"
seed: 42
category: reasoning
aggregate_rate: ${rate}.0
num_requests: 500
clients:
  - id: "multi-turn-chat"
    tenant_id: "chat-users"
    slo_class: "interactive"
    rate_fraction: 1.0
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 30
        min: 32
        max: 512
    output_distribution:
      type: gaussian
      params:
        mean: 64
        std_dev: 20
        min: 16
        max: 256
    reasoning:
      reason_ratio_distribution:
        type: gaussian
        params:
          mean: 0
          std_dev: 0
          min: 0
          max: 0
      multi_turn:
        max_rounds: 5
        think_time_us: 500000
        context_growth: accumulate
YAMLEOF
}

run_rate_sweep() {
    local rate_idx="$1"
    local rate="${RATES[$rate_idx]}"
    local rate_name="${RATE_NAMES[$rate_idx]}"

    # Select timeout based on rate level
    local timeout=$TIMEOUT_STANDARD
    if [[ "$rate_name" == "high" ]]; then
        timeout=$TIMEOUT_EXTENDED
    fi

    local workload_yaml="$RESULTS_DIR/${rate_name}_workload.yaml"
    generate_workload_yaml "$rate" "$workload_yaml"

    local subdir="$RESULTS_DIR/$rate_name"
    mkdir -p "$subdir"

    echo "  Rate: $rate_name ($rate req/s, timeout=${timeout}s)"
    for i in "${!CONFIGS[@]}"; do
        name="${CONFIG_NAMES[$i]}"
        scorers="${CONFIGS[$i]}"
        echo "    Config $((i+1)): $name ($scorers)"
        for seed in "${SEEDS[@]}"; do
            outfile="$subdir/${name}_seed${seed}.txt"
            blis_run "$timeout" "$outfile" \
                --model "$MODEL" \
                --num-instances 4 \
                --seed "$seed" \
                --log error \
                --summarize-trace \
                --trace-level decisions \
                --workload-spec "$workload_yaml" \
                --routing-policy weighted \
                --routing-scorers "$scorers" \
                || true
            echo "      seed=$seed done"
        done
    done
}

echo "============================================================================"
echo "  H377: Pareto Frontier at High Utilization"
echo "  Family: Cross-policy comparative | Type: Statistical/Pareto"
echo "  Seeds: ${SEEDS[*]}"
echo "  Rates: ${RATES[*]} req/s"
echo "============================================================================"
echo ""

total=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} * ${#RATES[@]} ))
echo "Running ${#CONFIGS[@]} configs x ${#SEEDS[@]} seeds x ${#RATES[@]} rates = $total simulations..."
echo ""

# ── High rate (treatment) ──────────────────────────────────────────────
run_rate_sweep 0

echo ""

# ── Moderate rate (control) ────────────────────────────────────────────
run_rate_sweep 1

echo ""
echo "=== Analysis ==="
echo ""

python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR" "${CONFIG_NAMES[@]}" "${SEEDS[@]}"
