#!/bin/bash
# H4: Round-Robin vs Least-Loaded at Low Utilization
#
# Hypothesis: "Round-robin should outperform or match least-loaded for
# uniform workloads at low rates."
#
# Classification: Statistical / Equivalence
# Family: Cross-policy comparative
# VV&UQ: Validation
# Tier: 5 (workload diversity)
#
# Design:
#   - ED-1: Vary exactly one dimension (routing policy: round-robin vs least-loaded)
#   - ED-2: Rate-aware — low rate (100) where policies should be equivalent,
#           plus high rate (1000, ~3x overload) where LL should outperform RR
#   - ED-3: Precondition — at rate=100, utilization ~0.29, queues ~empty
#   - ED-4: 3 seeds (42, 123, 456) per configuration
#   - ED-5: Self-contained, reproducible via ./run.sh
#   - ED-6: No prior experiment reference — first RR vs LL comparison
#
# Rate sizing rationale:
#   Step time = 6910 + 17.67*256 + 2.84*128 = 6910 + 4524 + 364 = 11798 us ~= 11.8ms
#   4 instances: 4/0.0118 ~= 339 req/s capacity
#   Rate 100 ~= 0.29x utilization -> well below saturation (equivalence expected)
#   Rate 1000 ~= 2.95x overload -> queue buildup, LL should outperform RR
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

# -- Generate workload YAMLs --------------------------------------------------
# Constant token distribution: exactly 256 input, 128 output
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
  - id: "uniform-client"
    tenant_id: "test"
    slo_class: "interactive"
    rate_fraction: 1.0
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 256
    output_distribution:
      type: constant
      params:
        value: 128
YAMLEOF
}

# Run one comparison (RR vs LL) for a given seed, rate, num_requests
run_comparison() {
    local seed=$1
    local rate=$2
    local num_reqs=$3
    local prefix=$4
    local timeout=$5

    local wl_yaml="$RESULTS_DIR/${prefix}_wl_${seed}.yaml"
    make_workload_yaml "$seed" "$rate" "$num_reqs" "$wl_yaml"

    echo -n "  Seed $seed: Round-Robin ... "
    blis_run $timeout "$RESULTS_DIR/${prefix}_rr_${seed}.txt" \
        --model "$MODEL" \
        --num-instances $INSTANCES \
        --seed "$seed" \
        --workload-spec "$wl_yaml" \
        --routing-policy round-robin \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy always-admit \
        --log error
    echo "done"

    echo -n "  Seed $seed: Least-Loaded ... "
    blis_run $timeout "$RESULTS_DIR/${prefix}_ll_${seed}.txt" \
        --model "$MODEL" \
        --num-instances $INSTANCES \
        --seed "$seed" \
        --workload-spec "$wl_yaml" \
        --routing-policy least-loaded \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy always-admit \
        --log error
    echo "done"
}

echo "============================================================================"
echo "  H4: Round-Robin vs Least-Loaded at Low Utilization"
echo "  Reference: docs/plans/research.md"
echo "  Type: Statistical / Equivalence"
echo "============================================================================"
echo ""

# -- Experiment 1: Low rate (rate=100, 500 reqs) -- Equivalence expected ------
echo "=== Experiment 1: Low Rate — RR vs LL at rate=100 (0.29x utilization) ==="
echo "    instances=$INSTANCES, requests=500, rate=100"
echo ""

for SEED in "${SEEDS[@]}"; do
    run_comparison "$SEED" 100 500 "exp1" "$TIMEOUT_QUICK"
done

# -- Experiment 2: High rate control (rate=1000, 500 reqs) -- LL should win ---
echo ""
echo "=== Experiment 2: High Rate Control — RR vs LL at rate=1000 (3x overload) ==="
echo "    instances=$INSTANCES, requests=500, rate=1000"
echo ""

for SEED in "${SEEDS[@]}"; do
    run_comparison "$SEED" 1000 500 "exp2" "$TIMEOUT_STANDARD"
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
