#!/bin/bash
# H-Compound-Strategy: Strategy Evolution Iteration 4 — Compound Strategy
#
# Tests whether combining all three confirmed mechanisms (StaticClassWeight +
# SLOGatedAdmission + PriorityPreemption) produces super-additive improvement.
#
# Arms:
#   B2:          StaticClassWeight only (baseline from Iter 1)
#   T2:          + SLOGatedAdmission(100) (Iter 2 lever)
#   T3:          + PriorityPreemption(5.0) (Iter 3 lever)
#   T4:          All three combined (compound)
#   T4-uniform:  T4 with uniform SLO (control-negative)
#
# Matrix: 5 configs x 3 seeds = 15 runs at 120% capacity (~300 req/s)
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
# 120% = 300 req/s
RATE=300

# Mechanism parameters
QUEUE_THRESHOLD=100        # SLO-gated queue threshold
PREEMPT_MARGIN=5.0         # Priority difference for preemption
MAX_RUNNING=32             # Batch constraint: production vLLM typically 32-128

# Common CLI flags
COMMON_FLAGS="--model $MODEL --tp $TP --hardware $HARDWARE --num-instances $INSTANCES --routing-policy weighted --routing-scorers prefix-affinity:3,queue-depth:2 --scheduler priority-fcfs --max-num-running-reqs 32 --log error"

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
    local admission=$2
    local preempt_margin=$3
    local seed=$4
    local slo_mix=$5

    local outfile="$RESULTS_DIR/${label}.txt"
    local workload_yaml="$RESULTS_DIR/${label}_workload.yaml"

    generate_workload_yaml "$RATE" "$slo_mix" "$seed" "$workload_yaml"

    local admission_flags=""
    if [[ "$admission" == "slo-gated" ]]; then
        admission_flags="--admission-policy slo-gated --token-bucket-capacity $QUEUE_THRESHOLD"
    else
        admission_flags="--admission-policy always-admit"
    fi

    local preempt_flags=""
    if [[ "$preempt_margin" != "0" ]]; then
        preempt_flags="--priority-preemption-margin $preempt_margin"
    fi

    echo -n "  [$label] admission=$admission margin=$preempt_margin seed=$seed ... "

    local exit_code=0
    blis_run $TIMEOUT_EXTENDED "$outfile" \
        $COMMON_FLAGS \
        --seed "$seed" \
        --workload-spec "$workload_yaml" \
        --priority-policy static-class-weight \
        $admission_flags \
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
echo "  Strategy Evolution Iteration 4: Compound Strategy"
echo "  Reference: hypotheses/h-compound-strategy/problem.md"
echo "============================================================================"
echo ""
echo "Config: instances=$INSTANCES, requests=$NUM_REQUESTS, seeds=$SEEDS"
echo "Rate: $RATE req/s (120% capacity)"
echo "Admission threshold: $QUEUE_THRESHOLD, Preemption margin: $PREEMPT_MARGIN"
echo ""

# -- B2: StaticClassWeight only (baseline) ------------------------------------
echo "=== B2: StaticClassWeight only (baseline) ==="
echo ""

for SEED in $SEEDS; do
    run_config "B2_s${SEED}" "always-admit" "0" "$SEED" "mixed"
done

echo ""

# -- T2: StaticClassWeight + SLOGatedAdmission --------------------------------
echo "=== T2: StaticClassWeight + SLOGatedAdmission ==="
echo ""

for SEED in $SEEDS; do
    run_config "T2_s${SEED}" "slo-gated" "0" "$SEED" "mixed"
done

echo ""

# -- T3: StaticClassWeight + PriorityPreemption --------------------------------
echo "=== T3: StaticClassWeight + PriorityPreemption ==="
echo ""

for SEED in $SEEDS; do
    run_config "T3_s${SEED}" "always-admit" "$PREEMPT_MARGIN" "$SEED" "mixed"
done

echo ""

# -- T4: Full compound (all three) --------------------------------------------
echo "=== T4: Full compound (StaticClassWeight + SLOGatedAdmission + PriorityPreemption) ==="
echo ""

for SEED in $SEEDS; do
    run_config "T4_s${SEED}" "slo-gated" "$PREEMPT_MARGIN" "$SEED" "mixed"
done

echo ""

# -- T4-uniform: Control-negative (uniform SLO with all mechanisms) -----------
echo "=== T4-uniform: Control-negative (uniform SLO) ==="
echo ""

for SEED in $SEEDS; do
    run_config "T4uniform_s${SEED}" "slo-gated" "$PREEMPT_MARGIN" "$SEED" "uniform"
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
echo "  Experiment complete. Results in hypotheses/h-compound-strategy/results/"
echo "============================================================================"
