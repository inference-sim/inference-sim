#!/bin/bash
# H6: Counterfactual Regret — Round-Robin vs Weighted Routing
#
# Hypothesis: "Counterfactual regret should be higher for round-robin than
# weighted routing under variable load."
#
# Classification: Statistical / Dominance
# Family: Cross-policy comparative
# VV&UQ: Validation
# Tier: 2
#
# Design:
#   - ED-1: Vary exactly one dimension (routing policy: round-robin vs weighted/queue-depth)
#   - ED-2: Low-rate control (rate=100) where regret should be minimal for both
#   - ED-3: Precondition — trace-level decisions + counterfactual-k 3 enables regret computation
#   - ED-4: 3 seeds (42, 123, 456) per configuration
#   - ED-5: Self-contained, builds binary, reproducible
#   - ED-6: No prior experiment reference — first counterfactual/regret experiment
#
# Rate sizing rationale:
#   Mixed workload: 70% short (input=128, output=64), 30% long (input=1024, output=256)
#   Weighted mean step time:
#     short: 6910 + 17.67*128 + 2.84*64 = 6910 + 2262 + 182 = 9354 us
#     long:  6910 + 17.67*1024 + 2.84*256 = 6910 + 18094 + 727 = 25731 us
#     weighted: 0.7*9354 + 0.3*25731 = 6548 + 7719 = 14267 us = ~14.3ms
#   4 instances: 4/0.01427 = ~280 req/s capacity
#   Rate 200 = ~0.71x utilization => moderate load, some queueing asymmetry
#   Rate 100 = ~0.36x utilization => light load control (ED-2), minimal queueing
#
# Usage: ./run.sh [--rebuild]
#
# Requires: Go 1.24+, Python 3

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"

setup_experiment "${1:-}"

INSTANCES=4
SEEDS=(42 123 456)
NUM_REQS=500

# -- Generate workload YAML with two clients for load asymmetry ----------------
make_workload_yaml() {
    local seed=$1
    local rate=$2
    local num_reqs=$3
    local outfile=$4
    cat > "$outfile" << YAMLEOF
version: "1"
seed: $seed
category: language
aggregate_rate: ${rate}.0
num_requests: $num_reqs
clients:
  - id: "short-requests"
    tenant_id: "test"
    slo_class: "interactive"
    rate_fraction: 0.7
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 30
        min: 32
        max: 256
    output_distribution:
      type: gaussian
      params:
        mean: 64
        std_dev: 15
        min: 16
        max: 128
  - id: "long-requests"
    tenant_id: "test"
    slo_class: "batch"
    rate_fraction: 0.3
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 1024
        std_dev: 200
        min: 512
        max: 2048
    output_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 50
        min: 64
        max: 512
YAMLEOF
}

# Run one comparison: RR vs weighted at given rate/seed
run_comparison() {
    local label=$1
    local seed=$2
    local rate=$3
    local num_reqs=$4
    local prefix=$5

    local wl_yaml="$RESULTS_DIR/${prefix}_wl_${seed}.yaml"
    make_workload_yaml "$seed" "$rate" "$num_reqs" "$wl_yaml"

    echo -n "  Seed $seed: Round-Robin ... "
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/${prefix}_rr_${seed}.txt" \
        --model "$MODEL" \
        --num-instances $INSTANCES \
        --seed "$seed" \
        --workload-spec "$wl_yaml" \
        --routing-policy round-robin \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy always-admit \
        --trace-level decisions \
        --counterfactual-k 3 \
        --summarize-trace \
        --log error
    echo "done"

    echo -n "  Seed $seed: Weighted (queue-depth) ... "
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/${prefix}_weighted_${seed}.txt" \
        --model "$MODEL" \
        --num-instances $INSTANCES \
        --seed "$seed" \
        --workload-spec "$wl_yaml" \
        --routing-policy weighted \
        --routing-scorers "queue-depth:1" \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy always-admit \
        --trace-level decisions \
        --counterfactual-k 3 \
        --summarize-trace \
        --log error
    echo "done"
}

echo "============================================================================"
echo "  H6: Counterfactual Regret — Round-Robin vs Weighted Routing"
echo "  Reference: docs/plans/research.md"
echo "  Type: Statistical / Dominance"
echo "============================================================================"
echo ""

# -- Experiment 1: Core comparison (rate=200, 500 reqs, 3 seeds) ---------------
echo "=== Experiment 1: Core — RR vs Weighted at rate=200 (moderate load) ==="
echo "    instances=$INSTANCES, requests=$NUM_REQS, rate=200"
echo ""

for SEED in "${SEEDS[@]}"; do
    run_comparison "exp1" "$SEED" 200 $NUM_REQS "exp1"
done

# -- Experiment 2: Low-rate control (rate=100, 500 reqs, 3 seeds) --------------
# ED-2: At low load (~0.36x utilization), queues should be near-empty, so all
# instances look equally loaded. Regret should be minimal for BOTH policies.
echo ""
echo "=== Experiment 2: Low-rate control — rate=100 (0.36x utilization) ==="
echo "    instances=$INSTANCES, requests=$NUM_REQS, rate=100"
echo ""

for SEED in "${SEEDS[@]}"; do
    run_comparison "ctrl" "$SEED" 100 $NUM_REQS "ctrl"
done

# -- Analysis ------------------------------------------------------------------
echo ""
echo "=== Analysis ==="
echo ""

python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR"

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis"
echo "============================================================================"
