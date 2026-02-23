#!/bin/bash
# H21: Extreme Scorer Weights
# Test whether extreme weight ratio (100:1) behaves identically to single-scorer routing
# Round 2: Added control configs C (weight-sensitivity) and D (zero-prefix-sharing)
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"

setup_experiment "${1:-}"

# Generate prefix-heavy workload inline (need to set seed per-run, so generate per seed)
# 80% shared prefix (multi-turn chat), 20% unique — mirrors prefix-affinity-demo.yaml
generate_workload() {
    local seed=$1
    local outfile=$2
    cat > "$outfile" <<YAML
version: "1"
seed: ${seed}
category: language
aggregate_rate: 500.0
num_requests: 200

clients:
  - id: "chat-with-system-prompt"
    tenant_id: "tenant-A"
    slo_class: "interactive"
    rate_fraction: 0.8
    streaming: true
    prefix_group: "long-system-prompt"
    prefix_length: 256
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 64
        std_dev: 20
        min: 16
        max: 256
    output_distribution:
      type: exponential
      params:
        mean: 128

  - id: "unique-requests"
    tenant_id: "tenant-B"
    slo_class: "batch"
    rate_fraction: 0.2
    streaming: false
    arrival:
      process: poisson
    input_distribution:
      type: exponential
      params:
        mean: 512
    output_distribution:
      type: exponential
      params:
        mean: 256
YAML
}

# Generate zero-prefix-sharing workload (no prefix_group at all — all requests unique)
# Same total rate, distribution, and request count as prefix workload
generate_noprefix_workload() {
    local seed=$1
    local outfile=$2
    cat > "$outfile" <<YAML
version: "1"
seed: ${seed}
category: language
aggregate_rate: 500.0
num_requests: 200

clients:
  - id: "all-unique"
    tenant_id: "tenant-A"
    slo_class: "interactive"
    rate_fraction: 1.0
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 40
        min: 16
        max: 512
    output_distribution:
      type: exponential
      params:
        mean: 128
YAML
}

SEEDS=(42 123 456)

echo "================================================================" >&2
echo "H21: Extreme Scorer Weights (100:1 vs single-scorer)" >&2
echo "  Round 2: +control C (weight-sensitivity) +control D (no-prefix)" >&2
echo "================================================================" >&2

# Common flags (constant across all configs)
COMMON_FLAGS=(
    --model "$MODEL"
    --num-instances 4
    --routing-policy weighted
    --log error
    --trace-level decisions
    --summarize-trace
)

# =====================================================================
# Config A: Extreme weight ratio — prefix-affinity:100,queue-depth:1
# Normalized: prefix-affinity=0.9901, queue-depth=0.0099
# =====================================================================
echo "" >&2
echo "--- Config A: prefix-affinity:100,queue-depth:1 ---" >&2

for seed in "${SEEDS[@]}"; do
    WORKLOAD_YAML="$RESULTS_DIR/workload_${seed}.yaml"
    generate_workload "$seed" "$WORKLOAD_YAML"

    echo "  Seed $seed..." >&2
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/config_a_seed${seed}.txt" \
        "${COMMON_FLAGS[@]}" \
        --routing-scorers "prefix-affinity:100,queue-depth:1" \
        --seed "$seed" \
        --workload-spec "$WORKLOAD_YAML"
done

# =====================================================================
# Config B: Single scorer — prefix-affinity:1 (no load-balancing signal)
# =====================================================================
echo "" >&2
echo "--- Config B: prefix-affinity:1 (single scorer) ---" >&2

for seed in "${SEEDS[@]}"; do
    WORKLOAD_YAML="$RESULTS_DIR/workload_${seed}.yaml"

    echo "  Seed $seed..." >&2
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/config_b_seed${seed}.txt" \
        "${COMMON_FLAGS[@]}" \
        --routing-scorers "prefix-affinity:1" \
        --seed "$seed" \
        --workload-spec "$WORKLOAD_YAML"
done

# =====================================================================
# Config C (Round 2 control — weight sensitivity):
# prefix-affinity:100,queue-depth:0.001
# Normalized: PA=0.99999, QD=0.00001
# If mechanism is correct: even 0.001 prevents concentration (similar to A)
# =====================================================================
echo "" >&2
echo "--- Config C: prefix-affinity:100,queue-depth:0.001 (weight-sensitivity control) ---" >&2

for seed in "${SEEDS[@]}"; do
    WORKLOAD_YAML="$RESULTS_DIR/workload_${seed}.yaml"

    echo "  Seed $seed..." >&2
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/config_c_seed${seed}.txt" \
        "${COMMON_FLAGS[@]}" \
        --routing-scorers "prefix-affinity:100,queue-depth:0.001" \
        --seed "$seed" \
        --workload-spec "$WORKLOAD_YAML"
done

# =====================================================================
# Config D (Round 2 control — zero-prefix-sharing):
# Both A-style (2 scorers) and B-style (1 scorer) with no prefix_group
# If mechanism is correct: no prefix cascade, so both degenerate to
# positional tie-breaking and produce identical all-to-one distribution
# =====================================================================
echo "" >&2
echo "--- Config D1: prefix-affinity:100,queue-depth:1 + no-prefix workload ---" >&2

for seed in "${SEEDS[@]}"; do
    NOPREFIX_YAML="$RESULTS_DIR/noprefix_workload_${seed}.yaml"
    generate_noprefix_workload "$seed" "$NOPREFIX_YAML"

    echo "  Seed $seed..." >&2
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/config_d1_seed${seed}.txt" \
        "${COMMON_FLAGS[@]}" \
        --routing-scorers "prefix-affinity:100,queue-depth:1" \
        --seed "$seed" \
        --workload-spec "$NOPREFIX_YAML"
done

echo "" >&2
echo "--- Config D2: prefix-affinity:1 + no-prefix workload ---" >&2

for seed in "${SEEDS[@]}"; do
    NOPREFIX_YAML="$RESULTS_DIR/noprefix_workload_${seed}.yaml"

    echo "  Seed $seed..." >&2
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/config_d2_seed${seed}.txt" \
        "${COMMON_FLAGS[@]}" \
        --routing-scorers "prefix-affinity:1" \
        --seed "$seed" \
        --workload-spec "$NOPREFIX_YAML"
done

# =====================================================================
# Analysis
# =====================================================================
echo "" >&2
echo "--- Running analysis ---" >&2

python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR"
