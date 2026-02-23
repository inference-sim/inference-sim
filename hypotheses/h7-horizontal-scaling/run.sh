#!/bin/bash
# H7: Horizontal Scaling — Instance Count vs TTFT Under Saturation
#
# Hypothesis: "Increasing instances from 4 to 8 should roughly halve TTFT p99
# for saturated workloads."
#
# Classification: Statistical / Dominance
# Family: Performance-regime
# VV&UQ: Validation
# Tier: 3 (system understanding)
#
# Design:
#   - ED-1: Vary exactly one dimension (instance count: 2, 4, 8)
#   - ED-2: Sub-saturation control at rate=100 (~0.44x util for 4 instances)
#   - ED-3: Precondition — 4-instance TTFT p99 >> bare prefill time (~10.6ms)
#   - ED-4: 3 seeds (42, 123, 456) per configuration
#   - ED-5: Self-contained, builds binary, reproducible
#   - ED-6: No prior experiment reference — first horizontal scaling test
#
# Rate sizing rationale (CLI --rate mode, defaults: prompt=512, output=512):
#   Step time = 6910.42 + 17.67*512 + 2.84*512 = 6910.42 + 9047.04 + 1454.08 = 17411.54 us ~ 17.4ms
#   Per-instance capacity ~ 1/0.0174 ~ 57.4 req/s
#   2 instances: ~115 req/s capacity
#   4 instances: ~230 req/s capacity
#   8 instances: ~460 req/s capacity
#
#   Rate=500: ~4.3x overload for 2, ~2.2x for 4, ~1.09x for 8
#   Rate=100: ~0.87x for 2, ~0.44x for 4, ~0.22x for 8 (sub-saturation control)
#
# Usage: ./run.sh [--rebuild]
#
# Requires: Go 1.24+, Python 3

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"

setup_experiment "${1:-}"

SEEDS=(42 123 456)
INSTANCE_COUNTS=(2 4 8)

# Common run wrapper
run_config() {
    local label=$1
    local instances=$2
    local seed=$3
    local rate=$4
    local num_reqs=$5

    echo -n "  instances=$instances seed=$seed ... "
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/${label}_inst${instances}_seed${seed}.txt" \
        --model "$MODEL" \
        --num-instances "$instances" \
        --seed "$seed" \
        --rate "$rate" \
        --num-requests "$num_reqs" \
        --routing-policy least-loaded \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy always-admit \
        --log error
    echo "done"
}

echo "============================================================================"
echo "  H7: Horizontal Scaling — Instance Count vs TTFT Under Saturation"
echo "  Reference: docs/plans/research.md"
echo "  Type: Statistical / Dominance"
echo "  Family: Performance-regime"
echo "============================================================================"
echo ""

# -- Experiment 1: Saturation sweep (rate=500, 500 reqs, 3 instance counts) ----
echo "=== Experiment 1: Saturation — rate=500, num_requests=500 ==="
echo "    instance counts: ${INSTANCE_COUNTS[*]}"
echo ""

for INSTANCES in "${INSTANCE_COUNTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        run_config "sat" "$INSTANCES" "$SEED" 500 500
    done
done

# -- Experiment 2: Sub-saturation control (rate=100, 500 reqs) ----------------
echo ""
echo "=== Experiment 2: Sub-saturation control — rate=100, num_requests=500 ==="
echo "    instance counts: ${INSTANCE_COUNTS[*]}"
echo ""

for INSTANCES in "${INSTANCE_COUNTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        run_config "sub" "$INSTANCES" "$SEED" 100 500
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
