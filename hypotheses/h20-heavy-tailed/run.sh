#!/bin/bash
# H20: Heavy-Tailed Input Distributions — ParetoLogNormal vs Gaussian
#
# Hypothesis: "Heavy-tailed input distributions (ParetoLogNormal) should produce
# more preemptions and HOL blocking than Gaussian at the same mean input length."
#
# Classification: Statistical / Dominance
# Family: Workload/arrival
# VV&UQ: Validation
# Tier: 2 (behavioral comparison)
#
# Design:
#   - ED-1: Vary exactly one dimension (input distribution: gaussian vs pareto_lognormal)
#   - ED-2: Sub-saturation control (rate=200) — effect should diminish without queue pressure
#   - ED-2 bonus: KV-abundant control (--total-kv-blocks=100000) — preemption effect should vanish
#   - ED-3: Both distributions have approximately mean ~256 input tokens
#   - ED-4: 3 seeds (42, 123, 456) per configuration
#   - ED-5: Self-contained, builds binary, reproducible
#   - ED-6: Reference: hypotheses/h16-gamma-vs-poisson/run.sh (same rate, instances, output dist)
#            Diff: input_distribution changed from gaussian to pareto_lognormal;
#            arrival process is poisson for both (not gamma); --total-kv-blocks added for pressure
#
# Rate sizing rationale:
#   Step time ~= 6910 + 17.67*256 + 2.84*128 ~= 11797 us ~= 11.8ms
#   4 instances: 4/0.0118 ~= 339 req/s capacity
#   Rate 1000 ~= 2.95x overload -> sufficient queue buildup for tail effects
#   Rate 200 ~= 0.59x utilization -> sub-saturation control (ED-2)
#
# ParetoLogNormal parameters (analytical mean calculation):
#   mix_weight=0.70, alpha=1.5, xm=50, mu=5.5, sigma=1.2
#   Pareto mean (alpha>1): alpha*xm/(alpha-1) = 1.5*50/0.5 = 150
#   LogNormal mean: exp(mu + sigma^2/2) = exp(5.5 + 0.72) = exp(6.22) ~ 502.7
#   Mixture mean: 0.70*150 + 0.30*502.7 = 105 + 150.8 ~ 255.8 tokens
#
# Gaussian parameters: mean=256, std_dev=50, min=32, max=512
#   Mean ~ 256 (clamped gaussian, symmetric around mean within [32,512])
#
# KV pressure calibration:
#   H8 used blocks=(5000,3000,2200,2100,2000) with input mean=512 and block_size=16.
#   For input mean=256: blocks_per_request ~ ceil(256/16) = 16 blocks.
#   ParetoLogNormal tail can produce >1000 tokens, needing >63 blocks.
#   2000 blocks with 4 instances = 500 blocks/instance = ~31 concurrent requests at mean.
#   This should create moderate preemption pressure for heavy-tailed inputs.
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
KV_BLOCKS=2000
BLOCK_SIZE=16

# Advisory KV safety check for ParetoLogNormal tail (~2000 tokens possible)
preflight_kv_check $KV_BLOCKS $BLOCK_SIZE 2000

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
        xm: 50.0
        mu: 5.5
        sigma: 1.2
        mix_weight: 0.70
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 30
        min: 32
        max: 256
YAMLEOF
}

# Common run function
run_comparison() {
    local label=$1
    local seed=$2
    local rate=$3
    local num_reqs=$4
    local prefix=$5
    local kv_blocks=$6

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
        --total-kv-blocks $kv_blocks \
        --block-size-in-tokens $BLOCK_SIZE \
        --routing-policy least-loaded \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy always-admit \
        --log error
    echo "done"

    echo -n "  Seed $seed: ParetoLogNormal ... "
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/${prefix}_pareto_${seed}.txt" \
        --model "$MODEL" \
        --num-instances $INSTANCES \
        --seed "$seed" \
        --workload-spec "$p_yaml" \
        --total-kv-blocks $kv_blocks \
        --block-size-in-tokens $BLOCK_SIZE \
        --routing-policy least-loaded \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy always-admit \
        --log error
    echo "done"
}

echo "============================================================================"
echo "  H20: Heavy-Tailed Input Distributions (ParetoLogNormal vs Gaussian)"
echo "  Reference: docs/plans/research.md"
echo "  Type: Statistical / Dominance"
echo "============================================================================"
echo ""

# -- Experiment 1: Core comparison (rate=1000, 500 reqs, KV=2000) ---------------
echo "=== Experiment 1: Core — Gaussian vs ParetoLogNormal at rate=1000, KV=$KV_BLOCKS ==="
echo "    instances=$INSTANCES, requests=500, rate=1000, kv_blocks=$KV_BLOCKS"
echo ""

for SEED in "${SEEDS[@]}"; do
    run_comparison "exp1" "$SEED" 1000 500 "exp1" $KV_BLOCKS
done

# -- Experiment 2: Sub-saturation control (rate=200, 500 reqs, KV=2000) ----------
# ED-2: At ~0.59x utilization, queues should not build up.
# The heavy-tail preemption effect should diminish (less queued requests competing for KV).
echo ""
echo "=== Experiment 2: Sub-saturation control — rate=200, KV=$KV_BLOCKS ==="
echo "    instances=$INSTANCES, requests=500, rate=200, kv_blocks=$KV_BLOCKS"
echo ""

for SEED in "${SEEDS[@]}"; do
    run_comparison "exp2" "$SEED" 200 500 "exp2" $KV_BLOCKS
done

# -- Experiment 3: KV-abundant control (rate=1000, 500 reqs, KV=100000) ----------
# ED-2 bonus: With abundant KV, preemptions should vanish for both distributions.
# Any remaining TTFT difference is intrinsic (prefill cost), not HOL blocking.
echo ""
echo "=== Experiment 3: KV-abundant control — rate=1000, KV=100000 ==="
echo "    instances=$INSTANCES, requests=500, rate=1000, kv_blocks=100000"
echo ""

for SEED in "${SEEDS[@]}"; do
    run_comparison "exp3" "$SEED" 1000 500 "exp3" 100000
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
