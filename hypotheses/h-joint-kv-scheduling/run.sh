#!/bin/bash
# H-Joint-KV-Scheduling: Strategy Evolution Iterations 9-10
#
# Tests whether SLO-aware KV eviction (targeting lowest-priority running request
# instead of tail) creates a multiplicative interaction with elastic priority
# batching. The 2x2x4 design varies:
#   - KV blocks: 2000 (abundant), 1200 (moderate), 800 (heavy), 500 (extreme)
#   - Scheduling: large-batch (no preemption), elastic (margin=4.0, cb=10)
#   - KV eviction: tail (default), SLO-aware
#
# With maxRunningReqs=64 and ~24 blocks/request (mean input=256, block_size=16):
#   64 * 24 = ~1536 blocks minimum for full batch
#   2000: 30% headroom — light KV pressure, some preemptions under bursts
#   1200: 22% deficit — moderate pressure, KV preemptions required
#   800:  48% deficit — heavy preemptions
#   500:  67% deficit — extreme, near livelock territory
#
# All at 120% capacity = 300 req/s, 500 requests, 4 instances
# 3 seeds per config = 48 total runs
#
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"

setup_experiment "${1:-}"

# -- Configuration ------------------------------------------------------------

SEEDS="42 123 456"
NUM_REQUESTS=300
INSTANCES=4
TP=2
HARDWARE="H100"
RATE=300  # 120% capacity

# KV pressure levels (per-instance blocks)
# With multi-turn accumulate (3 rounds), effective per-request KV grows:
#   Round 1: ~24 blocks (256+128 tokens / 16)
#   Round 2: ~48 blocks (accumulated context)
#   Round 3: ~72 blocks (accumulated context)
# Average ~48 blocks/request. At max_running=64: ~3072 blocks needed
# 5000=abundant, 2000=moderate (preemptions occur), 1500=heavy, 1200=extreme
KV_LEVELS="5000 2000 1500 1200"

# -- Workload YAML generation -------------------------------------------------

generate_workload_yaml() {
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
        min: 64
        max: 1024
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 32
        min: 32
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
        min: 64
        max: 1024
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 32
        min: 32
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
        min: 64
        max: 1024
    output_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 32
        min: 32
        max: 512
    reasoning:
      multi_turn:
        max_rounds: 3
        think_time_us: 500000
        context_growth: accumulate
YAMLEOF
}

# -- Run helper ----------------------------------------------------------------

run_config() {
    local label=$1
    local kv_blocks=$2
    local max_running=$3
    local preempt_margin=$4
    local max_preemptions=$5
    local slo_kv=$6  # "true" or "false"
    local seed=$7

    local outfile="$RESULTS_DIR/${label}.txt"
    local workload_yaml="$RESULTS_DIR/${label}_workload.yaml"
    local results_json="$RESULTS_DIR/${label}_results.json"

    generate_workload_yaml "$RATE" "$seed" "$workload_yaml"

    local preempt_flags=""
    if [[ "$preempt_margin" != "0" ]]; then
        preempt_flags="--priority-preemption-margin $preempt_margin --max-priority-preemptions-per-step $max_preemptions"
    fi

    local slo_kv_flags=""
    if [[ "$slo_kv" == "true" ]]; then
        slo_kv_flags="--slo-aware-kv-eviction"
    fi

    echo -n "  [$label] kv=$kv_blocks max_run=$max_running margin=$preempt_margin slo_kv=$slo_kv seed=$seed ... "

    local exit_code=0
    blis_run $TIMEOUT_EXTENDED "$outfile" \
        --model $MODEL --tp $TP --hardware $HARDWARE \
        --num-instances $INSTANCES \
        --max-num-running-reqs $max_running \
        --total-kv-blocks $kv_blocks \
        --routing-policy weighted \
        --routing-scorers prefix-affinity:3,queue-depth:2 \
        --scheduler priority-fcfs \
        --priority-policy static-class-weight \
        --seed "$seed" \
        --workload-spec "$workload_yaml" \
        --results-path "$results_json" \
        --log error \
        $preempt_flags \
        $slo_kv_flags \
        || exit_code=$?

    if is_timeout "$outfile"; then
        echo "TIMEOUT/ERROR"
    else
        echo "ok (exit=$exit_code)"
    fi
}

# -- Experiment Execution ------------------------------------------------------

echo "============================================================================"
echo "  Strategy Evolution Iterations 9-10: Joint KV-Scheduling Optimization"
echo "  SLO-aware KV eviction + elastic priority batching"
echo "============================================================================"
echo ""
echo "Config: instances=$INSTANCES, requests=$NUM_REQUESTS, seeds=$SEEDS"
echo "Rate: ${RATE} req/s (120% capacity)"
echo "KV levels: $KV_LEVELS"
echo ""

for KV in $KV_LEVELS; do
    echo "=== KV blocks: $KV ==="

    # Config A: large-batch + tail eviction (baseline)
    for SEED in $SEEDS; do
        run_config "kv${KV}_large-tail_s${SEED}" $KV 64 0 0 "false" "$SEED"
    done

    # Config B: large-batch + SLO-aware eviction (KV-only)
    for SEED in $SEEDS; do
        run_config "kv${KV}_large-slo_s${SEED}" $KV 64 0 0 "true" "$SEED"
    done

    # Config C: elastic + tail eviction (batch-only)
    for SEED in $SEEDS; do
        run_config "kv${KV}_elastic-tail_s${SEED}" $KV 64 4.0 10 "false" "$SEED"
    done

    # Config D: elastic + SLO-aware eviction (JOINT)
    for SEED in $SEEDS; do
        run_config "kv${KV}_elastic-slo_s${SEED}" $KV 64 4.0 10 "true" "$SEED"
    done

    echo ""
done

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
echo "  Experiment complete. Results in hypotheses/h-joint-kv-scheduling/results/"
echo "============================================================================"
