#!/bin/bash
# H-Elastic-Batching: Strategy Evolution Iteration 6 — Elastic Priority Batching
#
# Tests whether large batches (maxRunningReqs=64) with aggressive priority
# preemption (margin=4.0, circuit breaker=10) can achieve BOTH high SLO
# attainment AND high GPU utilization simultaneously.
#
# Configs:
#   small-batch:   maxRunning=8,  margin=5.0, cb=3  (SLO-optimized, Iter 3 setup)
#   large-batch:   maxRunning=64, margin=0,   cb=0  (GPU-utilization-optimized)
#   elastic:       maxRunning=64, margin=4.0, cb=10 (the new mechanism)
#   elastic-adm:   maxRunning=64, margin=4.0, cb=10 + slo-gated admission
#   fast-lane:     1 instance, critical-only, maxRunning=8 (ideal SLO)
#
# All at 120% capacity = 300 req/s, 1500 requests, 4 instances (except fast-lane)
# 3 seeds per config
#
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"

setup_experiment "${1:-}"

# -- Configuration ------------------------------------------------------------

SEEDS="42 123 456"
NUM_REQUESTS=1500
INSTANCES=4
TP=2
HARDWARE="H100"
RATE=300  # 120% capacity

# Common CLI flags for all multi-instance configs
COMMON_FLAGS="--model $MODEL --tp $TP --hardware $HARDWARE --num-instances $INSTANCES --routing-policy weighted --routing-scorers prefix-affinity:3,queue-depth:2 --scheduler priority-fcfs --priority-policy static-class-weight --log error"

# -- Workload YAML generation -------------------------------------------------

generate_workload_yaml() {
    local rate=$1
    local seed=$2
    local outfile=$3
    local slo_mix=${4:-mixed}  # "mixed" or "critical-only"

    if [[ "$slo_mix" == "critical-only" ]]; then
        cat > "$outfile" <<YAMLEOF
version: "2"
aggregate_rate: $rate
seed: $seed
num_requests: 500
clients:
  - id: critical-client
    rate_fraction: 1.0
    slo_class: critical
    arrival:
      process: gamma
      cv: 2.0
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 64
        min: 32
        max: 1024
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 32
        min: 16
        max: 512
    reasoning:
      multi_turn:
        max_rounds: 3
        think_time_us: 500000
        context_growth: accumulate
YAMLEOF
    else
        cat > "$outfile" <<YAMLEOF
version: "2"
aggregate_rate: $rate
seed: $seed
num_requests: $NUM_REQUESTS
clients:
  - id: critical-client
    rate_fraction: 0.2
    slo_class: critical
    arrival:
      process: gamma
      cv: 2.0
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 64
        min: 32
        max: 1024
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 32
        min: 16
        max: 512
    reasoning:
      multi_turn:
        max_rounds: 3
        think_time_us: 500000
        context_growth: accumulate
  - id: standard-client
    rate_fraction: 0.4
    slo_class: standard
    arrival:
      process: gamma
      cv: 2.0
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 64
        min: 32
        max: 1024
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 32
        min: 16
        max: 512
    reasoning:
      multi_turn:
        max_rounds: 3
        think_time_us: 500000
        context_growth: accumulate
  - id: sheddable-client
    rate_fraction: 0.4
    slo_class: sheddable
    arrival:
      process: gamma
      cv: 2.0
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 64
        min: 32
        max: 1024
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 32
        min: 16
        max: 512
    reasoning:
      multi_turn:
        max_rounds: 3
        think_time_us: 500000
        context_growth: accumulate
YAMLEOF
    fi
}

# -- Run helper ----------------------------------------------------------------

run_config() {
    local label=$1
    local max_running=$2
    local preempt_margin=$3
    local max_preemptions=$4
    local admission=$5
    local rate=$6
    local seed=$7
    local instances=$8
    local slo_mix=${9:-mixed}

    local outfile="$RESULTS_DIR/${label}.txt"
    local workload_yaml="$RESULTS_DIR/${label}_workload.yaml"
    local results_json="$RESULTS_DIR/${label}_results.json"

    generate_workload_yaml "$rate" "$seed" "$workload_yaml" "$slo_mix"

    local preempt_flags=""
    if [[ "$preempt_margin" != "0" ]]; then
        preempt_flags="--priority-preemption-margin $preempt_margin --max-priority-preemptions-per-step $max_preemptions"
    fi

    local admission_flags=""
    if [[ "$admission" != "none" ]]; then
        admission_flags="--admission-policy slo-gated --token-bucket-capacity 100"
    fi

    echo -n "  [$label] max_running=$max_running margin=$preempt_margin cb=$max_preemptions adm=$admission seed=$seed ... "

    local exit_code=0
    blis_run $TIMEOUT_EXTENDED "$outfile" \
        --model $MODEL --tp $TP --hardware $HARDWARE \
        --num-instances $instances \
        --max-num-running-reqs $max_running \
        --routing-policy weighted \
        --routing-scorers prefix-affinity:3,queue-depth:2 \
        --scheduler priority-fcfs \
        --priority-policy static-class-weight \
        --seed "$seed" \
        --workload-spec "$workload_yaml" \
        --results-path "$results_json" \
        --log error \
        $preempt_flags \
        $admission_flags \
        || exit_code=$?

    if is_timeout "$outfile"; then
        echo "TIMEOUT/ERROR"
    else
        echo "ok (exit=$exit_code)"
    fi
}

# -- Experiment Execution ------------------------------------------------------

echo "============================================================================"
echo "  Strategy Evolution Iteration 6: Elastic Priority Batching"
echo "  Dual objective: SLO attainment AND GPU utilization"
echo "============================================================================"
echo ""
echo "Config: instances=$INSTANCES, requests=$NUM_REQUESTS, seeds=$SEEDS"
echo "Rate: ${RATE} req/s (120% capacity)"
echo ""

# -- Small-batch (SLO-optimized) -----------------------------------------------
echo "=== Config: small-batch (maxRunning=8, margin=5.0, cb=3) ==="
for SEED in $SEEDS; do
    run_config "small-batch_s${SEED}" 8 5.0 3 "none" $RATE "$SEED" $INSTANCES
done
echo ""

# -- Large-batch (GPU-utilization-optimized) ------------------------------------
echo "=== Config: large-batch (maxRunning=64, margin=0, cb=0) ==="
for SEED in $SEEDS; do
    run_config "large-batch_s${SEED}" 64 0 0 "none" $RATE "$SEED" $INSTANCES
done
echo ""

# -- Elastic (the new mechanism) ------------------------------------------------
echo "=== Config: elastic (maxRunning=64, margin=4.0, cb=10) ==="
for SEED in $SEEDS; do
    run_config "elastic_s${SEED}" 64 4.0 10 "none" $RATE "$SEED" $INSTANCES
done
echo ""

# -- Elastic + admission -------------------------------------------------------
echo "=== Config: elastic+adm (maxRunning=64, margin=4.0, cb=10, slo-gated) ==="
for SEED in $SEEDS; do
    run_config "elastic-adm_s${SEED}" 64 4.0 10 "slo-gated" $RATE "$SEED" $INSTANCES
done
echo ""

# -- Fast-lane (single instance, critical-only) --------------------------------
echo "=== Config: fast-lane (1 instance, critical-only, maxRunning=8) ==="
for SEED in $SEEDS; do
    run_config "fast-lane_s${SEED}" 8 0 0 "none" 60 "$SEED" 1 "critical-only"
done
echo ""

# -- Copy results to experiment directory --------------------------------------
echo "Copying results to $SCRIPT_DIR/results/ ..."
rm -rf "$SCRIPT_DIR/results"
cp -r "$RESULTS_DIR" "$SCRIPT_DIR/results"
echo ""

# -- Analysis ------------------------------------------------------------------
echo "=== Running Analysis ==="
echo ""

python3 "$SCRIPT_DIR/analyze.py" "$SCRIPT_DIR/results"

echo ""
echo "============================================================================"
echo "  Experiment complete. Results in hypotheses/h-elastic-batching/results/"
echo "============================================================================"
