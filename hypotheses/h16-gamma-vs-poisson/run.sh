#!/bin/bash
# H16: Gamma vs Poisson Tail Latency
#
# Hypothesis: "Bursty (Gamma) arrivals should produce worse tail latency
# than Poisson at the same average rate."
#
# Classification: Statistical / Dominance
# Family: Workload/arrival
# VV&UQ: Validation
# Tier: 2 (behavioral comparison)
#
# Design:
#   - ED-1: Vary exactly one dimension (arrival process: poisson vs gamma)
#   - ED-2: Rate-aware — rate=1000 with 4 instances, ~3x overload to create queue buildup
#           Round 2: sub-saturation control at rate=200 (~0.59x utilization)
#   - ED-3: Precondition — Gamma CV=3.5 produces highly bursty inter-arrivals
#   - ED-4: 3 seeds (42, 123, 456) per configuration
#   - ED-5: Self-contained, builds binary, reproducible
#   - ED-6: No prior experiment reference — first workload/arrival comparison
#
# Rate sizing rationale:
#   Step time ~= 6910 + 17.67*256 + 2.84*128 ~= 11797 us ~= 11.8ms
#   4 instances: 4/0.0118 ~= 339 req/s capacity
#   Rate 1000 ~= 2.95x overload → sufficient queue buildup for tail effects
#   Rate 200 ~= 0.59x utilization → sub-saturation control (Round 2: ED-2)
#   Gamma CV=3.5 → shape=0.082 → very bursty (large gaps interspersed with clusters)
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
# Parameterized: rate and num_requests passed as arguments
make_poisson_yaml() {
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
  - id: "poisson-client"
    tenant_id: "test"
    slo_class: "interactive"
    rate_fraction: 1.0
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 50
        min: 64
        max: 512
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 30
        min: 32
        max: 256
YAMLEOF
}

make_gamma_yaml() {
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
  - id: "gamma-client"
    tenant_id: "test"
    slo_class: "interactive"
    rate_fraction: 1.0
    streaming: true
    arrival:
      process: gamma
      cv: 3.5
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 50
        min: 64
        max: 512
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 30
        min: 32
        max: 256
YAMLEOF
}

# Common flags for all runs
run_comparison() {
    local label=$1
    local seed=$2
    local rate=$3
    local num_reqs=$4
    local prefix=$5

    local p_yaml="$RESULTS_DIR/${prefix}_poisson_wl_${seed}.yaml"
    local g_yaml="$RESULTS_DIR/${prefix}_gamma_wl_${seed}.yaml"
    make_poisson_yaml "$seed" "$rate" "$num_reqs" "$p_yaml"
    make_gamma_yaml "$seed" "$rate" "$num_reqs" "$g_yaml"

    echo -n "  Seed $seed: Poisson ... "
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/${prefix}_poisson_${seed}.txt" \
        --model "$MODEL" \
        --num-instances $INSTANCES \
        --seed "$seed" \
        --workload-spec "$p_yaml" \
        --routing-policy least-loaded \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy always-admit \
        --log error \
        --summarize-trace \
        --trace-level decisions
    echo "done"

    echo -n "  Seed $seed: Gamma (CV=3.5) ... "
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/${prefix}_gamma_${seed}.txt" \
        --model "$MODEL" \
        --num-instances $INSTANCES \
        --seed "$seed" \
        --workload-spec "$g_yaml" \
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
echo "  H16: Gamma vs Poisson Tail Latency"
echo "  Reference: docs/plans/research.md"
echo "  Type: Statistical / Dominance"
echo "============================================================================"
echo ""

# -- Experiment 1: Core comparison (rate=1000, 500 reqs, 3 seeds) --------------
echo "=== Experiment 1: Core — Poisson vs Gamma at rate=1000 (3x overload) ==="
echo "    instances=$INSTANCES, requests=500, rate=1000"
echo ""

for SEED in "${SEEDS[@]}"; do
    run_comparison "exp1" "$SEED" 1000 500 "exp1"
done

# -- Experiment B2: Sub-saturation control (rate=200, 500 reqs, 3 seeds) -------
# Round 2 addition (ED-2): at ~0.59x utilization, queues should not build up
# and the Gamma tail penalty should vanish.
echo ""
echo "=== Experiment B2: Sub-saturation control — rate=200 (0.59x utilization) ==="
echo "    instances=$INSTANCES, requests=500, rate=200"
echo ""

for SEED in "${SEEDS[@]}"; do
    run_comparison "b2" "$SEED" 200 500 "b2"
done

# -- Experiment C1: Larger sample (rate=1000, 2000 reqs, 3 seeds) --------------
# Round 2 addition: 2000 requests gives ~20 data points per p99 instead of ~5,
# reducing variance that may have caused seed 456's anomalous 9% result.
echo ""
echo "=== Experiment C1: Larger sample — 2000 requests at rate=1000 ==="
echo "    instances=$INSTANCES, requests=2000, rate=1000"
echo ""

for SEED in "${SEEDS[@]}"; do
    run_comparison "c1" "$SEED" 1000 2000 "c1"
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
