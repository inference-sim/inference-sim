#!/bin/bash
# H-Elastic-Stress: Strategy Evolution Iteration 8 — Stress Testing Elastic Batching
#
# Tests whether elastic priority batching generalizes across dimensions that
# Iteration 7 held constant: cluster scale, KV pressure, and asymmetric request sizes.
#
# 8 stress variants x 2 configs x 3 seeds = 48 runs
#
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"

setup_experiment "${1:-}"

# -- Configuration ------------------------------------------------------------

SEEDS="42 123 456"
NUM_REQUESTS=500
TP=2
HARDWARE="H100"

# -- Workload YAML generation -------------------------------------------------

generate_standard_workload_yaml() {
    local rate=$1
    local seed=$2
    local outfile=$3
    local crit_frac=$4
    local std_frac=$5
    local shed_frac=$6

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
    rate_fraction: $std_frac
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
    rate_fraction: $shed_frac
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
}

generate_asymmetric_workload_yaml() {
    local rate=$1
    local seed=$2
    local outfile=$3
    local crit_input_mean=$4
    local crit_input_std=$5
    local crit_input_min=$6
    local crit_input_max=$7
    local shed_input_mean=$8
    local shed_input_std=$9
    local shed_input_min=${10}
    local shed_input_max=${11}

    cat > "$outfile" <<YAMLEOF
version: "2"
aggregate_rate: $rate
seed: $seed
num_requests: $NUM_REQUESTS
clients:
  - id: critical-client
    rate_fraction: 0.20
    slo_class: critical
    arrival:
      process: gamma
      cv: 2.0
    input_distribution:
      type: gaussian
      params:
        mean: $crit_input_mean
        std_dev: $crit_input_std
        min: $crit_input_min
        max: $crit_input_max
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
    rate_fraction: 0.40
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
    rate_fraction: 0.40
    slo_class: sheddable
    arrival:
      process: gamma
      cv: 2.0
    input_distribution:
      type: gaussian
      params:
        mean: $shed_input_mean
        std_dev: $shed_input_std
        min: $shed_input_min
        max: $shed_input_max
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
}

generate_pareto_workload_yaml() {
    local rate=$1
    local seed=$2
    local outfile=$3

    cat > "$outfile" <<YAMLEOF
version: "2"
aggregate_rate: $rate
seed: $seed
num_requests: $NUM_REQUESTS
clients:
  - id: critical-client
    rate_fraction: 0.20
    slo_class: critical
    arrival:
      process: gamma
      cv: 2.0
    input_distribution:
      type: pareto_lognormal
      params:
        alpha: 1.5
        xm: 32
        mu: 5.0
        sigma: 1.0
        mix_weight: 0.7
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
    rate_fraction: 0.40
    slo_class: standard
    arrival:
      process: gamma
      cv: 2.0
    input_distribution:
      type: pareto_lognormal
      params:
        alpha: 1.5
        xm: 32
        mu: 5.0
        sigma: 1.0
        mix_weight: 0.7
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
    rate_fraction: 0.40
    slo_class: sheddable
    arrival:
      process: gamma
      cv: 2.0
    input_distribution:
      type: pareto_lognormal
      params:
        alpha: 1.5
        xm: 32
        mu: 5.0
        sigma: 1.0
        mix_weight: 0.7
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
}

# -- Run helper ----------------------------------------------------------------

run_stress_variant() {
    local variant_id=$1
    local config=$2
    local seed=$3
    local rate=$4
    local instances=$5
    local kv_blocks=$6       # "default" or integer
    local workload_type=$7   # standard | asymmetric-small-crit | asymmetric-large-crit | pareto

    local label="${variant_id}_${config}_s${seed}"
    local outfile="$RESULTS_DIR/${label}.txt"
    local workload_yaml="$RESULTS_DIR/${label}_workload.yaml"

    # Generate workload YAML based on type
    case "$workload_type" in
        standard)
            generate_standard_workload_yaml "$rate" "$seed" "$workload_yaml" 0.20 0.40 0.40
            ;;
        asymmetric-small-crit)
            # Critical = short (mean=64), sheddable = long (mean=512)
            generate_asymmetric_workload_yaml "$rate" "$seed" "$workload_yaml" \
                64 16 16 256 \
                512 128 64 2048
            ;;
        asymmetric-large-crit)
            # Critical = long (mean=512), sheddable = short (mean=64)
            generate_asymmetric_workload_yaml "$rate" "$seed" "$workload_yaml" \
                512 128 64 2048 \
                64 16 16 256
            ;;
        pareto)
            generate_pareto_workload_yaml "$rate" "$seed" "$workload_yaml"
            ;;
    esac

    local preempt_flags=""
    if [ "$config" = "elastic" ]; then
        preempt_flags="--priority-preemption-margin 4.0 --max-priority-preemptions-per-step 10"
    fi

    local kv_flags=""
    if [ "$kv_blocks" != "default" ]; then
        kv_flags="--total-kv-blocks $kv_blocks"
    fi

    echo -n "  [$label] inst=$instances kv=$kv_blocks wl=$workload_type ... "

    local exit_code=0
    blis_run $TIMEOUT_EXTENDED "$outfile" \
        --model $MODEL --tp $TP --hardware $HARDWARE \
        --num-instances "$instances" \
        --max-num-running-reqs 64 \
        --routing-policy weighted \
        --routing-scorers prefix-affinity:3,queue-depth:2 \
        --scheduler priority-fcfs \
        --priority-policy static-class-weight \
        --seed "$seed" \
        --workload-spec "$workload_yaml" \
        --log error \
        $preempt_flags \
        $kv_flags \
        || exit_code=$?

    if is_timeout "$outfile"; then
        echo "TIMEOUT/ERROR"
    else
        echo "ok"
    fi
}

# -- Variant dispatch ----------------------------------------------------------

# Returns: rate instances kv_blocks workload_type
get_variant_params() {
    local vid=$1
    case "$vid" in
        S1)  echo "150 2 default standard" ;;              # 2 instances, 120% of ~125 cap
        S2)  echo "600 8 default standard" ;;              # 8 instances, 120% of ~500 cap
        S3)  echo "300 4 5000 standard" ;;                 # Moderate KV pressure
        S4)  echo "300 4 2000 standard" ;;                 # Heavy KV pressure
        S5)  echo "300 4 default asymmetric-small-crit" ;; # Critical = short, sheddable = long
        S6)  echo "300 4 default asymmetric-large-crit" ;; # Critical = long, sheddable = short
        S7)  echo "300 4 default pareto" ;;                # Heavy-tailed ParetoLogNormal
        S8)  echo "1200 16 default standard" ;;            # 16 instances, 120% of ~1000 cap
    esac
}

get_variant_desc() {
    local vid=$1
    case "$vid" in
        S1)  echo "Small cluster (2 inst, 120%)" ;;
        S2)  echo "Large cluster (8 inst, 120%)" ;;
        S3)  echo "Moderate KV pressure (5K blocks)" ;;
        S4)  echo "Heavy KV pressure (2K blocks)" ;;
        S5)  echo "Critical=short, Sheddable=long" ;;
        S6)  echo "Critical=long, Sheddable=short" ;;
        S7)  echo "ParetoLogNormal input sizes" ;;
        S8)  echo "Very large cluster (16 inst, 120%)" ;;
    esac
}

# -- Main sweep ----------------------------------------------------------------

echo "============================================================================"
echo "  Elastic Priority Batching: 8-Variant Stress Test"
echo "  48 runs (8 variants x 2 configs x 3 seeds)"
echo "============================================================================"
echo ""

VARIANTS="S1 S2 S3 S4 S5 S6 S7 S8"
CONFIGS="large elastic"

total_runs=48
run_count=0

for V in $VARIANTS; do
    desc=$(get_variant_desc "$V")
    echo "=== $V: $desc ==="
    params=$(get_variant_params "$V")
    set -- $params
    v_rate=$1; v_inst=$2; v_kv=$3; v_wl=$4

    # KV pressure preflight for S3 and S4
    if [ "$v_kv" != "default" ]; then
        preflight_kv_check "$v_kv" 16 1024 || true
    fi

    for C in $CONFIGS; do
        for SEED in $SEEDS; do
            run_count=$((run_count + 1))
            echo -n "  ($run_count/$total_runs) "
            run_stress_variant "$V" "$C" "$SEED" "$v_rate" "$v_inst" "$v_kv" "$v_wl"
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
echo "  Stress test complete. $run_count/$total_runs runs executed."
echo "  Results in hypotheses/h-elastic-stress/results/"
echo "============================================================================"
