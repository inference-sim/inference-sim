#!/bin/bash
# H-Priority-Preemption: Strategy Evolution Iteration 3 — Priority-Based Preemption
#
# Tests whether priority-based preemption at the batch level (evicting low-priority
# running requests for high-priority waiting ones) produces dramatic critical TTFT
# improvement beyond what scheduling (Iter 1) and admission (Iter 2) achieved.
#
# Arms:
#   H-main:              B2 vs T3 at 80%/120% x 3 seeds (12 runs)
#   H-control-negative:  T3 vs B2 with uniform SLO at 120% x 3 seeds (6 runs)
#
# Total: ~18 runs
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

# Capacity estimate: ~250 req/s for 4 instances with input=256, output=128
RATE_80=200
RATE_120=300

# Priority preemption margin: critical(10) - sheddable(1) = 9 > 5.0 triggers
PREEMPT_MARGIN=5.0

# Batch capacity: use a tight batch limit to create batch pressure.
# With max-num-running-reqs=8 per instance and ~75 req/s/instance at 120%,
# batch fills up when queueing delay accumulates. This is where priority
# preemption becomes relevant — critical requests waiting for batch slots.
MAX_RUNNING_REQS=8

# Common CLI flags
COMMON_FLAGS="--model $MODEL --tp $TP --hardware $HARDWARE --num-instances $INSTANCES --max-num-running-reqs $MAX_RUNNING_REQS --routing-policy weighted --routing-scorers prefix-affinity:3,queue-depth:2 --scheduler priority-fcfs --log error"

# -- Workload YAML generation -------------------------------------------------

generate_workload_yaml() {
    local rate=$1
    local slo_mix=$2  # "mixed" or "uniform"
    local seed=$3
    local outfile=$4

    if [[ "$slo_mix" == "mixed" ]]; then
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
    else
        # Uniform SLO: all requests are "standard" (protected class)
        cat > "$outfile" <<YAMLEOF
version: "2"
aggregate_rate: $rate
seed: $seed
num_requests: $NUM_REQUESTS
clients:
  - id: uniform-client
    rate_fraction: 1.0
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
YAMLEOF
    fi
}

# -- Run helper ----------------------------------------------------------------

run_config() {
    local label=$1
    local preempt_margin=$2
    local rate=$3
    local seed=$4
    local slo_mix=$5

    local outfile="$RESULTS_DIR/${label}.txt"
    local workload_yaml="$RESULTS_DIR/${label}_workload.yaml"

    generate_workload_yaml "$rate" "$slo_mix" "$seed" "$workload_yaml"

    local preempt_flags=""
    if [[ "$preempt_margin" != "0" ]]; then
        preempt_flags="--priority-preemption-margin $preempt_margin"
    fi

    echo -n "  [$label] margin=$preempt_margin rate=$rate seed=$seed ... "

    local exit_code=0
    blis_run $TIMEOUT_EXTENDED "$outfile" \
        $COMMON_FLAGS \
        --seed "$seed" \
        --workload-spec "$workload_yaml" \
        --priority-policy static-class-weight \
        $preempt_flags \
        --summarize-trace \
        --trace-level decisions \
        || exit_code=$?

    if is_timeout "$outfile"; then
        echo "TIMEOUT/ERROR"
    else
        echo "ok (exit=$exit_code)"
    fi
}

# -- Experiment Execution ------------------------------------------------------

echo "============================================================================"
echo "  Strategy Evolution Iteration 3: Priority-Based Preemption"
echo "  Reference: hypotheses/h-priority-preemption/problem.md"
echo "============================================================================"
echo ""
echo "Config: instances=$INSTANCES, requests=$NUM_REQUESTS, seeds=$SEEDS"
echo "Rates: 80%=${RATE_80}, 120%=${RATE_120}"
echo "Preemption margin: $PREEMPT_MARGIN"
echo ""

# -- H-main: B2 vs T3 at 80% and 120% ----------------------------------------
echo "=== H-main: B2 (no preemption) vs T3 (priority preemption) ==="
echo ""

for SEED in $SEEDS; do
    for RATE_LABEL in "80" "120"; do
        if [[ "$RATE_LABEL" == "80" ]]; then RATE=$RATE_80; else RATE=$RATE_120; fi
        run_config "B2_${RATE_LABEL}pct_s${SEED}" "0" "$RATE" "$SEED" "mixed"
        run_config "T3_${RATE_LABEL}pct_s${SEED}" "$PREEMPT_MARGIN" "$RATE" "$SEED" "mixed"
    done
done

echo ""

# -- H-control-negative: Uniform SLO at 120% ----------------------------------
echo "=== H-control-negative: Uniform SLO (all standard) at 120% ==="
echo ""

for SEED in $SEEDS; do
    run_config "B2_uniform_120pct_s${SEED}" "0" "$RATE_120" "$SEED" "uniform"
    run_config "T3_uniform_120pct_s${SEED}" "$PREEMPT_MARGIN" "$RATE_120" "$SEED" "uniform"
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
echo "  Experiment complete. Results in hypotheses/h-priority-preemption/results/"
echo "============================================================================"
