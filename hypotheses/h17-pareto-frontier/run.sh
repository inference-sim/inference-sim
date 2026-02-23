#!/bin/bash
# H17: Multi-scorer Pareto Frontier
# Different scorer weight combinations should produce a Pareto frontier:
# no single configuration dominates all metrics simultaneously.
#
# Three workloads tested (ED-2: test where effect is expected AND where it should vanish):
#   Workload A: multiturn-chat-demo.yaml (prefix-heavy, multi-turn context accumulation)
#   Workload B: independent requests (no prefix reuse, no multi-turn)
#   Workload C: mixed (50% prefix-heavy + 50% independent) — Round 2 addition per Reviewer B
#
# Reference: hypotheses/prefix-affinity/run.sh (same workload A, single seed)
# Config diff vs reference: This experiment sweeps 5 weight configurations
# across 3 seeds on 2 workloads; reference tested 4 routing policies at seed=42 only.
#
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
SEEDS=(42 123 456)

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# 5 weight configurations:
# C1: cache-heavy    — prefix-affinity:5,queue-depth:1
# C2: llm-d default  — prefix-affinity:3,queue-depth:2,kv-utilization:2
# C3: load-balance   — queue-depth:3,kv-utilization:3
# C4: queue-heavy    — prefix-affinity:1,queue-depth:5
# C5: kv-heavy       — kv-utilization:5,prefix-affinity:1

CONFIGS=(
    "prefix-affinity:5,queue-depth:1"
    "prefix-affinity:3,queue-depth:2,kv-utilization:2"
    "queue-depth:3,kv-utilization:3"
    "prefix-affinity:1,queue-depth:5"
    "kv-utilization:5,prefix-affinity:1"
)
CONFIG_NAMES=(
    "cache-heavy"
    "llmd-default"
    "load-balance"
    "queue-heavy"
    "kv-heavy"
)

run_workload() {
    local workload_name="$1"
    local workload_file="$2"
    local subdir="$RESULTS_DIR/$workload_name"
    mkdir -p "$subdir"

    echo "  Workload: $workload_name"
    for i in "${!CONFIGS[@]}"; do
        name="${CONFIG_NAMES[$i]}"
        scorers="${CONFIGS[$i]}"
        echo "    Config $((i+1)): $name ($scorers)"
        for seed in "${SEEDS[@]}"; do
            outfile="$subdir/${name}_seed${seed}.txt"
            "$BINARY" run \
                --model "$MODEL" \
                --num-instances 4 \
                --seed "$seed" \
                --log error \
                --summarize-trace \
                --trace-level decisions \
                --workload-spec "$workload_file" \
                --routing-policy weighted \
                --routing-scorers "$scorers" \
                2>/dev/null \
                > "$outfile"
            echo "      seed=$seed done"
        done
    done
}

echo "============================================================================"
echo "  H17: Multi-Scorer Pareto Frontier"
echo "  Family: Cross-policy comparative | Type: Statistical/Pareto"
echo "  Seeds: ${SEEDS[*]}"
echo "============================================================================"
echo ""

total=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} * 3 ))
echo "Running ${#CONFIGS[@]} configs x ${#SEEDS[@]} seeds x 3 workloads = $total simulations..."
echo ""

# ── Workload A: Prefix-heavy multi-turn ────────────────────────────────────
run_workload "prefix-heavy" "$REPO_ROOT/examples/multiturn-chat-demo.yaml"

# ── Workload B: Independent requests (no prefix reuse) ────────────────────
# Control workload: same rate/request count, but no multi-turn and no prefix groups.
# This isolates the load-balancing effect from cache locality.
cat > "$RESULTS_DIR/independent.yaml" << 'YAMLEOF'
version: "1"
seed: 42
category: language
aggregate_rate: 500.0
num_requests: 500
clients:
  - id: "independent-requests"
    tenant_id: "users"
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
YAMLEOF

echo ""
run_workload "independent" "$RESULTS_DIR/independent.yaml"

# ── Workload C: Mixed (50% prefix-heavy + 50% independent) ─────────────
# Round 2 addition per Reviewer B feedback: tests whether a genuine
# within-workload Pareto frontier exists when both prefix-heavy and
# independent traffic coexist.
cat > "$RESULTS_DIR/mixed.yaml" << 'YAMLEOF'
version: "1"
seed: 42
category: reasoning
aggregate_rate: 500.0
num_requests: 500
clients:
  - id: "prefix-heavy-chat"
    tenant_id: "chat-users"
    slo_class: "interactive"
    rate_fraction: 0.5
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
  - id: "independent-requests"
    tenant_id: "users"
    slo_class: "interactive"
    rate_fraction: 0.5
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
YAMLEOF

echo ""
run_workload "mixed" "$RESULTS_DIR/mixed.yaml"

echo ""
echo "=== Analysis ==="
echo ""

python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR" "${CONFIG_NAMES[@]}" "${SEEDS[@]}"
