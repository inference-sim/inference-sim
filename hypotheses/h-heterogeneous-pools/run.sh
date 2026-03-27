#!/bin/bash
# H-Heterogeneous-Pools: Strategy Evolution Iteration 5 — Heterogeneous Instance Pools
#
# Tests whether PHYSICAL ISOLATION of critical traffic into a dedicated fast lane
# outperforms any queue-management policy on a shared pool.
#
# Arms:
#   A (Fast Lane):    1 instance, critical only (60 req/s), maxRunningReqs=8
#   B (Bulk Pool):    3 instances, standard+sheddable (240 req/s), maxRunningReqs=64
#   C (Shared Base):  4 instances, all SLO classes (300 req/s), maxRunningReqs=32
#   D (Compound):     4 instances, all SLO classes + admission + preemption
#
# Matrix: 4 configs x 3 seeds = 12 runs at 120% capacity (~300 req/s total)
#
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"

setup_experiment "${1:-}"

# -- Configuration ------------------------------------------------------------

SEEDS="42 123 456"
TP=2
HARDWARE="H100"

# Capacity estimate: ~250 req/s for 4 instances with input=256, output=128
# 120% = 300 req/s total
TOTAL_RATE=300
FAST_RATE=60       # 20% of total — critical traffic only
BULK_RATE=240      # 80% of total — standard + sheddable

# Mechanism parameters (for Sim D compound)
QUEUE_THRESHOLD=100
PREEMPT_MARGIN=5.0

# -- Workload YAML generation -------------------------------------------------

generate_critical_only_yaml() {
    local rate=$1
    local seed=$2
    local num_requests=$3
    local outfile=$4

    cat > "$outfile" <<YAMLEOF
version: "2"
aggregate_rate: $rate
seed: $seed
num_requests: $num_requests
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
}

generate_bulk_yaml() {
    local rate=$1
    local seed=$2
    local num_requests=$3
    local outfile=$4

    cat > "$outfile" <<YAMLEOF
version: "2"
aggregate_rate: $rate
seed: $seed
num_requests: $num_requests
clients:
  - id: standard-client
    rate_fraction: 0.5
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
    rate_fraction: 0.5
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

generate_mixed_yaml() {
    local rate=$1
    local seed=$2
    local num_requests=$3
    local outfile=$4

    cat > "$outfile" <<YAMLEOF
version: "2"
aggregate_rate: $rate
seed: $seed
num_requests: $num_requests
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
}

# -- Run helper ----------------------------------------------------------------

run_sim_a() {
    local seed=$1
    local outfile="$RESULTS_DIR/A_s${seed}.txt"
    local workload_yaml="$RESULTS_DIR/A_s${seed}_workload.yaml"

    generate_critical_only_yaml "$FAST_RATE" "$seed" 500 "$workload_yaml"

    echo -n "  [A_s${seed}] Fast Lane (1 inst, critical only, maxRun=8) ... "

    local exit_code=0
    blis_run $TIMEOUT_EXTENDED "$outfile" \
        --model $MODEL --tp $TP --hardware $HARDWARE \
        --num-instances 1 \
        --max-num-running-reqs 8 \
        --scheduler priority-fcfs \
        --priority-policy static-class-weight \
        --seed "$seed" \
        --workload-spec "$workload_yaml" \
        --summarize-trace \
        --trace-level decisions \
        --log error \
        || exit_code=$?

    if is_timeout "$outfile"; then
        echo "TIMEOUT/ERROR"
    else
        echo "ok (exit=$exit_code)"
    fi
}

run_sim_b() {
    local seed=$1
    local outfile="$RESULTS_DIR/B_s${seed}.txt"
    local workload_yaml="$RESULTS_DIR/B_s${seed}_workload.yaml"

    generate_bulk_yaml "$BULK_RATE" "$seed" 1200 "$workload_yaml"

    echo -n "  [B_s${seed}] Bulk Pool (3 inst, std+shed, maxRun=64) ... "

    local exit_code=0
    blis_run $TIMEOUT_EXTENDED "$outfile" \
        --model $MODEL --tp $TP --hardware $HARDWARE \
        --num-instances 3 \
        --max-num-running-reqs 64 \
        --routing-policy weighted \
        --routing-scorers prefix-affinity:3,queue-depth:2 \
        --scheduler priority-fcfs \
        --priority-policy static-class-weight \
        --seed "$seed" \
        --workload-spec "$workload_yaml" \
        --summarize-trace \
        --trace-level decisions \
        --log error \
        || exit_code=$?

    if is_timeout "$outfile"; then
        echo "TIMEOUT/ERROR"
    else
        echo "ok (exit=$exit_code)"
    fi
}

run_sim_c() {
    local seed=$1
    local outfile="$RESULTS_DIR/C_s${seed}.txt"
    local workload_yaml="$RESULTS_DIR/C_s${seed}_workload.yaml"

    generate_mixed_yaml "$TOTAL_RATE" "$seed" 1500 "$workload_yaml"

    echo -n "  [C_s${seed}] Shared Baseline (4 inst, all SLO, maxRun=32) ... "

    local exit_code=0
    blis_run $TIMEOUT_EXTENDED "$outfile" \
        --model $MODEL --tp $TP --hardware $HARDWARE \
        --num-instances 4 \
        --max-num-running-reqs 32 \
        --routing-policy weighted \
        --routing-scorers prefix-affinity:3,queue-depth:2 \
        --scheduler priority-fcfs \
        --priority-policy static-class-weight \
        --seed "$seed" \
        --workload-spec "$workload_yaml" \
        --summarize-trace \
        --trace-level decisions \
        --log error \
        || exit_code=$?

    if is_timeout "$outfile"; then
        echo "TIMEOUT/ERROR"
    else
        echo "ok (exit=$exit_code)"
    fi
}

run_sim_d() {
    local seed=$1
    local outfile="$RESULTS_DIR/D_s${seed}.txt"
    local workload_yaml="$RESULTS_DIR/D_s${seed}_workload.yaml"

    generate_mixed_yaml "$TOTAL_RATE" "$seed" 1500 "$workload_yaml"

    echo -n "  [D_s${seed}] Compound (4 inst, admission+preemption) ... "

    local exit_code=0
    blis_run $TIMEOUT_EXTENDED "$outfile" \
        --model $MODEL --tp $TP --hardware $HARDWARE \
        --num-instances 4 \
        --max-num-running-reqs 32 \
        --routing-policy weighted \
        --routing-scorers prefix-affinity:3,queue-depth:2 \
        --scheduler priority-fcfs \
        --priority-policy static-class-weight \
        --admission-policy slo-gated \
        --token-bucket-capacity $QUEUE_THRESHOLD \
        --priority-preemption-margin $PREEMPT_MARGIN \
        --seed "$seed" \
        --workload-spec "$workload_yaml" \
        --summarize-trace \
        --trace-level decisions \
        --log error \
        || exit_code=$?

    if is_timeout "$outfile"; then
        echo "TIMEOUT/ERROR"
    else
        echo "ok (exit=$exit_code)"
    fi
}

# -- Experiment Execution ------------------------------------------------------

echo "============================================================================"
echo "  Strategy Evolution Iteration 5: Heterogeneous Instance Pools"
echo "  Reference: hypotheses/h-heterogeneous-pools/"
echo "============================================================================"
echo ""
echo "Config: total_rate=$TOTAL_RATE, fast_rate=$FAST_RATE, bulk_rate=$BULK_RATE"
echo "Seeds: $SEEDS"
echo ""

# -- Sim A: Fast Lane (critical only, 1 instance) ----------------------------
echo "=== Sim A: Fast Lane (1 instance, critical only, maxRunningReqs=8) ==="
echo ""

for SEED in $SEEDS; do
    run_sim_a "$SEED"
done

echo ""

# -- Sim B: Bulk Pool (standard + sheddable, 3 instances) --------------------
echo "=== Sim B: Bulk Pool (3 instances, std+shed, maxRunningReqs=64) ==="
echo ""

for SEED in $SEEDS; do
    run_sim_b "$SEED"
done

echo ""

# -- Sim C: Shared Baseline (all SLO, 4 instances) ---------------------------
echo "=== Sim C: Shared Baseline (4 instances, all SLO, maxRunningReqs=32) ==="
echo ""

for SEED in $SEEDS; do
    run_sim_c "$SEED"
done

echo ""

# -- Sim D: Compound (all mechanisms, 4 instances) ---------------------------
echo "=== Sim D: Compound (4 instances, admission+preemption) ==="
echo ""

for SEED in $SEEDS; do
    run_sim_d "$SEED"
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
echo "  Experiment complete. Results in hypotheses/h-heterogeneous-pools/results/"
echo "============================================================================"
