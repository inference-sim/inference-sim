#!/bin/bash
# H7: Horizontal Scaling
#
# Hypothesis: "Increasing instances from 4 to 8 should roughly halve TTFT p99
# for saturated workloads."
#
# Classification: Statistical / Monotonicity (sweep 3+ values: 2, 4, 8 instances)
# Family: Performance-regime (scaling laws)
# VV&UQ: Validation
# Tier: 2 (behavioral comparison)
#
# Design:
#   - ED-1: Vary exactly one dimension (num-instances: 2, 4, 8)
#   - ED-2: Rate-aware — rate=1000 with 4 instances is ~3x overload;
#           sub-saturation control at rate=100 (~0.29x utilization at 4 instances)
#   - ED-3: Precondition — verify saturation (TTFT p99 >> low-rate baseline)
#   - ED-4: 3 seeds (42, 123, 456) per configuration
#   - ED-5: Self-contained, builds binary, reproducible
#   - ED-6: Reference: hypotheses/h16-gamma-vs-poisson/run.sh (similar rate sizing rationale)
#
# Rate sizing rationale:
#   Step time ~= 6910 + 17.67*256 + 2.84*128 ~= 11797 us ~= 11.8ms
#   Capacity per instance ~= 1/0.0118 ~= 85 req/s
#   2 instances: 170 req/s capacity
#   4 instances: 339 req/s capacity
#   8 instances: 678 req/s capacity
#   Rate 1000 ~= 5.9x overload at 2 inst, 2.95x at 4 inst, 1.47x at 8 inst
#   Rate 100  ~= 0.29x utilization at 4 inst (sub-saturation control)
#
# Usage: ./run.sh [--rebuild]
#
# Requires: Go 1.24+, Python 3

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"

setup_experiment "${1:-}"

INSTANCE_COUNTS=(2 4 8)
SEEDS=(42 123 456)

# -- Run a single configuration ------------------------------------------------
run_config() {
    local label=$1
    local instances=$2
    local seed=$3
    local rate=$4
    local num_reqs=$5

    local outfile="$RESULTS_DIR/${label}_inst${instances}_seed${seed}.txt"
    echo -n "  instances=$instances seed=$seed ... "
    blis_run $TIMEOUT_STANDARD "$outfile" \
        --model "$MODEL" \
        --num-instances "$instances" \
        --seed "$seed" \
        --rate "$rate" \
        --num-requests "$num_reqs" \
        --routing-policy least-loaded \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy always-admit \
        --log error \
        --summarize-trace \
        --trace-level decisions
    echo "done"
}

echo "============================================================================"
echo "  H7: Horizontal Scaling"
echo "  Reference: docs/plans/research.md"
echo "  Type: Statistical / Monotonicity"
echo "============================================================================"
echo ""

# -- Experiment 1: Core scaling sweep (rate=1000, 500 reqs, 3 seeds) -----------
echo "=== Experiment 1: Scaling sweep at rate=1000 (saturating) ==="
echo "    instances={2,4,8}, requests=500, rate=1000"
echo ""

for SEED in "${SEEDS[@]}"; do
    for INST in "${INSTANCE_COUNTS[@]}"; do
        run_config "exp1" "$INST" "$SEED" 1000 500
    done
done

# -- Experiment 2: Sub-saturation control (rate=100, 500 reqs, 3 seeds) --------
# ED-2: At ~0.29x utilization (4 inst), queues should not build up.
# Scaling effect should vanish (TTFT p99 should be similar across instance counts).
echo ""
echo "=== Experiment 2: Sub-saturation control at rate=100 (0.29x util at 4 inst) ==="
echo "    instances={2,4,8}, requests=500, rate=100"
echo ""

for SEED in "${SEEDS[@]}"; do
    for INST in "${INSTANCE_COUNTS[@]}"; do
        run_config "ctrl" "$INST" "$SEED" 100 500
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
