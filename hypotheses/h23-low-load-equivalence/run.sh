#!/bin/bash
# H23: Low-Load Routing Policy Equivalence
#
# Hypothesis: "Under very low load (1 req/s, 4 instances), all routing
# policies should produce equivalent TTFT."
#
# Classification: Statistical / Equivalence
# Family: Cross-policy comparative
# VV&UQ: Validation
# Tier: 5 (baseline sanity check)
#
# Design:
#   - ED-1: Vary exactly one dimension (routing policy/scorer config)
#   - ED-2: High-rate control (rate=2000) where policies SHOULD differ
#   - ED-3: Precondition — at rate=1 with 4 instances, utilization ~0.004
#   - ED-4: 3 seeds (42, 123, 456) per configuration
#   - ED-5: Self-contained, builds binary, reproducible
#   - ED-6: No prior experiment reference — first low-load equivalence test
#
# Rate sizing rationale:
#   Step time with 512/512: beta0 + beta1*512 + beta2*512
#     = 6910.42 + 17.67*512 + 2.84*512 = 6910.42 + 9047.04 + 1454.08 = 17411.54 us ~= 17.4ms
#   Single-instance capacity: 1/0.0174 ~= 57.4 req/s
#   4 instances: ~229.6 req/s capacity
#   Rate=1 with 4 instances: rho ~= 1/(4*57.4) ~= 0.004 (essentially zero)
#   Rate=2000 with 4 instances: rho ~= 2000/229.6 ~= 8.7x overload
#
# Equivalence criterion: max deviation < 5% across all 4 policies
# High-rate divergence: >20% difference expected (validates comparison)
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

# Four routing configurations:
#   A: round-robin (simplest; cycles through instances)
#   B: least-loaded (picks instance with shortest queue)
#   C: weighted with default scorers (prefix-affinity:3,queue-depth:2,kv-utilization:2)
#   D: prefix-affinity (standalone routing policy, not weighted scorer)
POLICIES=(
    "round-robin"
    "least-loaded"
    "weighted"
    "prefix-affinity"
)

# Scorer config only needed for weighted policy
SCORER_CONFIG="prefix-affinity:3,queue-depth:2,kv-utilization:2"

# Run a single configuration
run_config() {
    local label=$1
    local policy=$2
    local seed=$3
    local rate=$4
    local num_reqs=$5
    local prefix=$6
    local timeout=$7

    local outfile="$RESULTS_DIR/${prefix}_${label}_${seed}.txt"

    if [[ "$policy" == "weighted" ]]; then
        blis_run "$timeout" "$outfile" \
            --model "$MODEL" \
            --num-instances $INSTANCES \
            --seed "$seed" \
            --rate "$rate" \
            --num-requests "$num_reqs" \
            --prompt-tokens 512 \
            --output-tokens 512 \
            --routing-policy "$policy" \
            --routing-scorers "$SCORER_CONFIG" \
            --scheduler fcfs \
            --priority-policy constant \
            --admission-policy always-admit \
            --log error
    else
        blis_run "$timeout" "$outfile" \
            --model "$MODEL" \
            --num-instances $INSTANCES \
            --seed "$seed" \
            --rate "$rate" \
            --num-requests "$num_reqs" \
            --prompt-tokens 512 \
            --output-tokens 512 \
            --routing-policy "$policy" \
            --scheduler fcfs \
            --priority-policy constant \
            --admission-policy always-admit \
            --log error
    fi
}

echo "============================================================================"
echo "  H23: Low-Load Routing Policy Equivalence"
echo "  Reference: docs/plans/research.md"
echo "  Type: Statistical / Equivalence"
echo "============================================================================"
echo ""

# -- Experiment 1: Low-load comparison (rate=1, 50 requests, 3 seeds) ----------
echo "=== Experiment 1: Low-load — rate=1, 50 requests, 4 instances ==="
echo "    Expected utilization: ~0.004 (essentially zero)"
echo ""

for SEED in "${SEEDS[@]}"; do
    for POLICY in "${POLICIES[@]}"; do
        echo -n "  Seed $SEED: $POLICY ... "
        run_config "$POLICY" "$POLICY" "$SEED" 1 50 "exp1" $TIMEOUT_STANDARD
        echo "done"
    done
done

# -- Experiment 2: High-load control (rate=2000, 500 requests, 3 seeds) --------
# ED-2: At ~8.7x overload, queues build up and routing policy matters.
# Policies should diverge (>20%) — validates the comparison is meaningful.
echo ""
echo "=== Experiment 2: High-load control — rate=2000, 500 requests, 4 instances ==="
echo "    Expected utilization: ~8.7x overload"
echo ""

for SEED in "${SEEDS[@]}"; do
    for POLICY in "${POLICIES[@]}"; do
        echo -n "  Seed $SEED: $POLICY ... "
        run_config "$POLICY" "$POLICY" "$SEED" 2000 500 "exp2" $TIMEOUT_STANDARD
        echo "done"
    done
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
