#!/bin/bash
# H-Elastic-Generalization: 12-variant generalization sweep for elastic priority batching
#
# Tests whether the dual-objective breakthrough (SLO attainment + GPU utilization)
# generalizes across 12 workload variants spanning load level, arrival process,
# session structure, and SLO mix.
#
# For each variant, two configs:
#   large-batch: maxRunning=64, margin=0, cb=0 (baseline: no preemption)
#   elastic:     maxRunning=64, margin=4.0, cb=10 (the mechanism)
#
# 12 variants x 2 configs x 3 seeds = 72 runs
#
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"

setup_experiment "${1:-}"

# -- Configuration ------------------------------------------------------------

SEEDS="42 123 456"
NUM_REQUESTS=500
INSTANCES=4
TP=2
HARDWARE="H100"

# -- Workload YAML generation -------------------------------------------------

generate_workload_yaml() {
    local rate=$1
    local seed=$2
    local outfile=$3
    local arrival=$4      # poisson | gamma | constant
    local cv=$5           # gamma cv (ignored for poisson/constant)
    local multi_turn=$6   # yes | no
    local crit_frac=$7
    local std_frac=$8
    local shed_frac=$9

    local arrival_block
    case "$arrival" in
        poisson)
            arrival_block="      process: poisson"
            ;;
        gamma)
            arrival_block="      process: gamma
      cv: ${cv}"
            ;;
        constant)
            arrival_block="      process: constant"
            ;;
    esac

    local multi_turn_block=""
    if [ "$multi_turn" = "yes" ]; then
        multi_turn_block="    reasoning:
      multi_turn:
        max_rounds: 3
        think_time_us: 500000
        context_growth: accumulate"
    fi

    cat > "$outfile" <<YAMLEOF
version: "2"
aggregate_rate: $rate
seed: $seed
num_requests: $NUM_REQUESTS
clients:
  - id: critical-client
    rate_fraction: $crit_frac
    slo_class: critical
    arrival:
${arrival_block}
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
${multi_turn_block}
  - id: standard-client
    rate_fraction: $std_frac
    slo_class: standard
    arrival:
${arrival_block}
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
${multi_turn_block}
  - id: sheddable-client
    rate_fraction: $shed_frac
    slo_class: sheddable
    arrival:
${arrival_block}
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
${multi_turn_block}
YAMLEOF
}

# -- Run helper ----------------------------------------------------------------

run_variant() {
    local variant_id=$1   # W1..W12
    local config=$2       # large | elastic
    local seed=$3
    local rate=$4
    local arrival=$5
    local cv=$6
    local multi_turn=$7
    local crit_frac=$8
    local std_frac=$9
    local shed_frac=${10}

    local label="${variant_id}_${config}_s${seed}"
    local outfile="$RESULTS_DIR/${label}.txt"
    local workload_yaml="$RESULTS_DIR/${label}_workload.yaml"

    generate_workload_yaml "$rate" "$seed" "$workload_yaml" "$arrival" "$cv" "$multi_turn" "$crit_frac" "$std_frac" "$shed_frac"

    local preempt_flags=""
    if [ "$config" = "elastic" ]; then
        preempt_flags="--priority-preemption-margin 4.0 --max-priority-preemptions-per-step 10"
    fi

    # Use extended timeout for multi-turn, standard for single-turn
    local timeout=$TIMEOUT_STANDARD
    if [ "$multi_turn" = "yes" ]; then
        timeout=$TIMEOUT_EXTENDED
    fi

    echo -n "  [$label] rate=$rate arr=$arrival mt=$multi_turn slo=${crit_frac}/${std_frac}/${shed_frac} ... "

    local exit_code=0
    blis_run $timeout "$outfile" \
        --model $MODEL --tp $TP --hardware $HARDWARE \
        --num-instances $INSTANCES \
        --max-num-running-reqs 64 \
        --routing-policy weighted \
        --routing-scorers prefix-affinity:3,queue-depth:2 \
        --scheduler priority-fcfs \
        --priority-policy static-class-weight \
        --seed "$seed" \
        --workload-spec "$workload_yaml" \
        --log error \
        $preempt_flags \
        || exit_code=$?

    if is_timeout "$outfile"; then
        echo "TIMEOUT/ERROR"
    else
        echo "ok"
    fi
}

# -- Variant dispatch ----------------------------------------------------------

# Each variant is defined as: rate arrival cv multi_turn crit_frac std_frac shed_frac
get_variant_params() {
    local vid=$1
    case "$vid" in
        W1)  echo "200 gamma 2 yes 0.20 0.40 0.40" ;;  # Moderate load (80%)
        W2)  echo "300 gamma 2 yes 0.20 0.40 0.40" ;;  # Base case (120%)
        W3)  echo "272 gamma 2 no 0.20 0.40 0.40" ;;   # Single-turn moderate
        W4)  echo "408 gamma 2 no 0.20 0.40 0.40" ;;   # Single-turn overload
        W5)  echo "300 gamma 2 yes 0.05 0.45 0.50" ;;  # Few critical
        W6)  echo "300 gamma 2 yes 0.50 0.30 0.20" ;;  # Many critical
        W7)  echo "300 poisson 0 yes 0.20 0.40 0.40" ;; # Poisson arrivals
        W8)  echo "300 gamma 4 yes 0.20 0.40 0.40" ;;  # Heavy bursts (cv=4)
        W9)  echo "500 gamma 2 yes 0.20 0.40 0.40" ;;  # Extreme overload (200%)
        W10) echo "125 gamma 2 yes 0.20 0.40 0.40" ;;  # Light load (50%)
        W11) echo "300 gamma 2 yes 0.10 0.10 0.80" ;;  # Sheddable-heavy
        W12) echo "300 constant 0 yes 0.20 0.40 0.40" ;; # Constant arrivals
    esac
}

get_variant_desc() {
    local vid=$1
    case "$vid" in
        W1)  echo "Moderate load (80%)" ;;
        W2)  echo "Base case (120%)" ;;
        W3)  echo "Single-turn moderate (80%)" ;;
        W4)  echo "Single-turn overload (120%)" ;;
        W5)  echo "Few critical (5%)" ;;
        W6)  echo "Many critical (50%)" ;;
        W7)  echo "Poisson arrivals" ;;
        W8)  echo "Heavy bursts (cv=4)" ;;
        W9)  echo "Extreme overload (200%)" ;;
        W10) echo "Light load (50%)" ;;
        W11) echo "Sheddable-heavy (80%)" ;;
        W12) echo "Constant arrivals" ;;
    esac
}

# -- Main sweep ----------------------------------------------------------------

echo "============================================================================"
echo "  Elastic Priority Batching: 12-Variant Generalization Sweep"
echo "  72 runs (12 variants x 2 configs x 3 seeds)"
echo "============================================================================"
echo ""

VARIANTS="W1 W2 W3 W4 W5 W6 W7 W8 W9 W10 W11 W12"
CONFIGS="large elastic"

total_runs=72
run_count=0

for V in $VARIANTS; do
    desc=$(get_variant_desc "$V")
    echo "=== $V: $desc ==="
    params=$(get_variant_params "$V")
    # Split params into array-like positional args
    set -- $params
    v_rate=$1; v_arr=$2; v_cv=$3; v_mt=$4; v_c=$5; v_s=$6; v_sh=$7

    for C in $CONFIGS; do
        for SEED in $SEEDS; do
            run_count=$((run_count + 1))
            echo -n "  ($run_count/$total_runs) "
            run_variant "$V" "$C" "$SEED" "$v_rate" "$v_arr" "$v_cv" "$v_mt" "$v_c" "$v_s" "$v_sh"
        done
    done
    echo ""
done

# -- Copy results to experiment directory --------------------------------------

echo "Copying results to $SCRIPT_DIR/results/ ..."
rm -rf "$SCRIPT_DIR/results"
cp -r "$RESULTS_DIR" "$SCRIPT_DIR/results"
echo ""

echo "============================================================================"
echo "  Sweep complete. $run_count/$total_runs runs executed."
echo "  Results in hypotheses/h-elastic-generalization/results/"
echo "============================================================================"
