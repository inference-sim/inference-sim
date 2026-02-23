#!/bin/bash
# H20: Heavy-Tailed Input Distributions
#
# Hypothesis: "ParetoLogNormal input distributions should produce more
# preemptions and HOL blocking than Gaussian at the same average length."
#
# Classification: Statistical / Dominance
# Family: Workload/arrival
# VV&UQ: Validation
# Tier: 2 (behavioral comparison)
#
# Design:
#   - ED-1: Vary exactly one dimension (input length distribution: gaussian vs pareto_lognormal)
#   - ED-2: Rate-aware — rate=1000 with 4 instances, ~3x overload to create queue pressure
#   - ED-3: Precondition — ParetoLogNormal has infinite variance (alpha=1.5 < 2),
#           producing occasional very long requests that exhaust KV blocks
#   - ED-4: 3 seeds (42, 123, 456) per configuration
#   - ED-5: Self-contained, builds binary, reproducible
#   - ED-6: Reference — H16 (gamma vs poisson) used same cluster setup
#
# ParetoLogNormal parameterization (from sim/workload/distribution.go):
#   Mixture model: with probability mix_weight, draw from Pareto(alpha, xm);
#   otherwise draw from LogNormal(mu, sigma).
#   Params: alpha=1.5, xm=100, mu=5.2, sigma=0.7, mix_weight=0.35
#   Pareto mean = xm*alpha/(alpha-1) = 100*1.5/0.5 = 300
#   LogNormal mean = exp(mu + sigma^2/2) = exp(5.2+0.245) = exp(5.445) = 231
#   Mixture mean = 0.35*300 + 0.65*231 = 105 + 150 = 255 tokens (target: 256)
#   Heavy tail: alpha=1.5 means infinite variance. Pareto can produce
#   inputs of 1000+ tokens (P(X>1000) ~ (100/1000)^1.5 = 3.2% per Pareto draw,
#   ~1.1% overall). These long requests hold KV blocks much longer.
#
# Gaussian control: mean=256, std_dev=50, min=32, max=512
#   Bounded — no extreme outliers. Max 512 tokens.
#
# Rate sizing rationale:
#   Step time ~= 6910 + 17.67*256 + 2.84*128 ~= 11797 us ~= 11.8ms
#   4 instances: 4/0.0118 ~= 339 req/s capacity
#   Rate 1000 ~= 2.95x overload -> sufficient queue buildup
#
# KV pressure experiment:
#   Default blocks (5000 per instance) may be enough to absorb long requests.
#   A constrained-KV experiment (2000 blocks) forces preemptions.
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
make_gaussian_yaml() {
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
  - id: "gaussian-client"
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
        min: 32
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

make_pareto_yaml() {
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
  - id: "pareto-client"
    tenant_id: "test"
    slo_class: "interactive"
    rate_fraction: 1.0
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: pareto_lognormal
      params:
        alpha: 1.5
        xm: 100.0
        mu: 5.2
        sigma: 0.7
        mix_weight: 0.35
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 30
        min: 32
        max: 256
YAMLEOF
}

# Common comparison runner
run_comparison() {
    local label=$1
    local seed=$2
    local rate=$3
    local num_reqs=$4
    local prefix=$5
    local extra_flags="${6:-}"

    local g_yaml="$RESULTS_DIR/${prefix}_gaussian_wl_${seed}.yaml"
    local p_yaml="$RESULTS_DIR/${prefix}_pareto_wl_${seed}.yaml"
    make_gaussian_yaml "$seed" "$rate" "$num_reqs" "$g_yaml"
    make_pareto_yaml "$seed" "$rate" "$num_reqs" "$p_yaml"

    echo -n "  Seed $seed: Gaussian ... "
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/${prefix}_gaussian_${seed}.txt" \
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
        --trace-level decisions \
        $extra_flags
    echo "done"

    echo -n "  Seed $seed: ParetoLogNormal ... "
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/${prefix}_pareto_${seed}.txt" \
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
        --trace-level decisions \
        $extra_flags
    echo "done"
}

echo "============================================================================"
echo "  H20: Heavy-Tailed Input Distributions"
echo "  Reference: docs/plans/research.md"
echo "  Type: Statistical / Dominance"
echo "============================================================================"
echo ""

# -- Experiment 1: Core comparison (default KV, rate=1000, 500 reqs) -----------
echo "=== Experiment 1: Core -- Gaussian vs ParetoLogNormal (default KV) ==="
echo "    instances=$INSTANCES, requests=500, rate=1000, default KV blocks"
echo ""

for SEED in "${SEEDS[@]}"; do
    run_comparison "exp1" "$SEED" 1000 500 "exp1"
done

# -- Experiment 2: KV-constrained (2000 blocks, rate=1000, 500 reqs) -----------
# With constrained KV, long ParetoLogNormal requests should cause more preemptions
echo ""
echo "=== Experiment 2: KV-constrained -- 2000 blocks per instance ==="
echo "    instances=$INSTANCES, requests=500, rate=1000, total-kv-blocks=2000"
echo ""

preflight_kv_check 2000 16 2000
for SEED in "${SEEDS[@]}"; do
    run_comparison "exp2" "$SEED" 1000 500 "exp2" "--total-kv-blocks 2000"
done

# -- Experiment 3: Low-rate control (rate=200, default KV, 500 reqs) -----------
# At sub-saturation, queues should not build up and HOL blocking should vanish
echo ""
echo "=== Experiment 3: Sub-saturation control -- rate=200 (0.59x utilization) ==="
echo "    instances=$INSTANCES, requests=500, rate=200, default KV blocks"
echo ""

for SEED in "${SEEDS[@]}"; do
    run_comparison "exp3" "$SEED" 200 500 "exp3"
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
