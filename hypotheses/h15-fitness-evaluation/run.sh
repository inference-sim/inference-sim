#!/bin/bash
# H15: Fitness Evaluation Ranks Prefix-Affinity Higher for Prefix Workloads
#
# Hypothesis: "Fitness evaluation should rank prefix-affinity-aware routing
# higher than load-only for prefix-heavy workloads when weights favor TTFT."
#
# Classification: Statistical / Dominance
# Family: Cross-policy comparative
# VV&UQ: Validation
# Tier: 2 (behavioral comparison)
#
# Design:
#   - ED-1: Vary exactly one dimension (routing scorer config)
#           A: prefix-affinity:3,queue-depth:2,kv-utilization:2 (default weighted)
#           B: queue-depth:2,kv-utilization:2 (load-only, no prefix awareness)
#   - ED-2: Control with non-prefix workload (uniform, no prefix_group) — fitness
#           advantage should vanish. Also test throughput-heavy weights where ranking
#           may change.
#   - ED-3: Precondition — prefix-affinity-demo.yaml has prefix_group with 256 tokens
#   - ED-4: 3 seeds (42, 123, 456)
#   - ED-5: Self-contained, builds binary, reproducible
#   - ED-6: No prior experiment reference — first fitness evaluation comparison
#
# Rate sizing rationale (prefix-affinity-demo.yaml):
#   80% requests: input ~320 (64 user + 256 prefix), output 128
#   20% requests: input ~512, output 256
#   Weighted step time ~= 0.8*(6910+17.67*320+2.84*128) + 0.2*(6910+17.67*512+2.84*256)
#                      ~= 0.8*12928 + 0.2*16684 ~= 13679 us ~= 13.7ms
#   4 instances: capacity ~= 4/0.0137 ~= 292 req/s
#   Rate 500: ~1.71x overload (moderate queue buildup for fitness differentiation)
#
# Fitness normalization:
#   Throughput: value / (value + 100)  — higher is better
#   Latency: 1 / (1 + value/1000)     — lower is better (1000 ticks = 1ms reference)
#   See sim/cluster/metrics.go:410-416 for reference scales
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
PREFIX_WORKLOAD="$REPO_ROOT/examples/prefix-affinity-demo.yaml"

# -- Generate non-prefix control workload (same distributions, no prefix_group) --
make_no_prefix_yaml() {
    local outfile=$1
    cat > "$outfile" << 'YAMLEOF'
version: "1"
seed: 42
category: language
aggregate_rate: 500.0
num_requests: 200
clients:
  - id: "no-prefix-client-a"
    tenant_id: "tenant-A"
    slo_class: "interactive"
    rate_fraction: 0.8
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 320
        std_dev: 50
        min: 64
        max: 512
    output_distribution:
      type: exponential
      params:
        mean: 128
  - id: "no-prefix-client-b"
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
YAMLEOF
}

# Common run function
run_config() {
    local label=$1
    local seed=$2
    local scorers=$3
    local workload=$4
    local output_prefix=$5
    local weight_set=$6

    echo -n "  $label seed=$seed scorers=$scorers weights=$weight_set ... "
    blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/${output_prefix}.txt" \
        --model "$MODEL" \
        --num-instances $INSTANCES \
        --seed "$seed" \
        --workload-spec "$workload" \
        --routing-policy weighted \
        --routing-scorers "$scorers" \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy always-admit \
        --fitness-weights "$weight_set" \
        --log error
    echo "done"
}

NO_PREFIX_YAML="$RESULTS_DIR/no_prefix_workload.yaml"
make_no_prefix_yaml "$NO_PREFIX_YAML"

echo "============================================================================"
echo "  H15: Fitness Evaluation Ranks Prefix-Affinity Higher"
echo "  Reference: docs/plans/research.md"
echo "  Type: Statistical / Dominance"
echo "============================================================================"
echo ""

# Weight sets
TTFT_HEAVY="throughput:0.3,p99_ttft:0.5,mean_e2e:0.2"
THROUGHPUT_HEAVY="throughput:0.7,p99_ttft:0.1,mean_e2e:0.2"

# Scorer configs
SCORERS_PREFIX="prefix-affinity:3,queue-depth:2,kv-utilization:2"
SCORERS_LOAD="queue-depth:2,kv-utilization:2"

# -- Experiment 1: Core — prefix workload, TTFT-heavy weights ------------------
echo "=== Experiment 1: Prefix workload, TTFT-heavy weights ==="
echo "    Weights: $TTFT_HEAVY"
echo "    instances=$INSTANCES, requests=200, rate=500"
echo ""

for SEED in "${SEEDS[@]}"; do
    run_config "exp1" "$SEED" "$SCORERS_PREFIX" "$PREFIX_WORKLOAD" \
        "exp1_prefix_aware_${SEED}" "$TTFT_HEAVY"
    run_config "exp1" "$SEED" "$SCORERS_LOAD" "$PREFIX_WORKLOAD" \
        "exp1_load_only_${SEED}" "$TTFT_HEAVY"
done

# -- Experiment 2: Prefix workload, throughput-heavy weights --------------------
echo ""
echo "=== Experiment 2: Prefix workload, throughput-heavy weights ==="
echo "    Weights: $THROUGHPUT_HEAVY"
echo "    instances=$INSTANCES, requests=200, rate=500"
echo ""

for SEED in "${SEEDS[@]}"; do
    run_config "exp2" "$SEED" "$SCORERS_PREFIX" "$PREFIX_WORKLOAD" \
        "exp2_prefix_aware_${SEED}" "$THROUGHPUT_HEAVY"
    run_config "exp2" "$SEED" "$SCORERS_LOAD" "$PREFIX_WORKLOAD" \
        "exp2_load_only_${SEED}" "$THROUGHPUT_HEAVY"
done

# -- Experiment 3: Control — non-prefix workload, TTFT-heavy weights -----------
# ED-2: With no prefix_group, prefix-affinity scorer has no signal to differentiate
# instances. Both configs should produce equivalent fitness scores.
echo ""
echo "=== Experiment 3: Non-prefix workload control, TTFT-heavy weights ==="
echo "    Weights: $TTFT_HEAVY"
echo "    instances=$INSTANCES, requests=200, rate=500"
echo ""

for SEED in "${SEEDS[@]}"; do
    run_config "exp3" "$SEED" "$SCORERS_PREFIX" "$NO_PREFIX_YAML" \
        "exp3_prefix_aware_${SEED}" "$TTFT_HEAVY"
    run_config "exp3" "$SEED" "$SCORERS_LOAD" "$NO_PREFIX_YAML" \
        "exp3_load_only_${SEED}" "$TTFT_HEAVY"
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
