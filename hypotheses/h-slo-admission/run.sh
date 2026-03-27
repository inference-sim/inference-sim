#!/bin/bash
# H-SLO-Admission: Strategy Evolution Iteration 2 — SLO-Gated Admission Control
#
# Tests whether SLO-gated admission (reject sheddable under load) combined with
# StaticClassWeight scheduling produces non-zero-sum improvements over scheduling alone.
#
# Arms:
#   H-main:              B2 vs T2 at 80%/120% x 3 seeds (12 runs)
#   H-zero-sum-broken:   Uses H-main data (no extra runs)
#   H-control-negative:  T2 vs B2 with uniform SLO at 120% x 3 seeds (6 runs)
#   H-threshold-sensitivity: T2 with thresholds [50,100,200] at 120% x 3 seeds (9 runs)
#
# Total: ~27 runs
#
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"

setup_experiment "${1:-}"

# ── Configuration ────────────────────────────────────────────────────────────

SEEDS="42 123 456"
NUM_REQUESTS=1500
INSTANCES=4
TP=2
HARDWARE="H100"

# Capacity estimate: ~250 req/s for 4 instances with input=256, output=128
# step time ~= 6910 + 17.67*256 = ~11434us prefill, decode ~6910+2.84*batchsize
# Conservative: ~62.5 req/s/instance => 250 req/s total
# 80% = 200 req/s, 120% = 300 req/s
RATE_80=200
RATE_120=300

# SLO-gated queue threshold (total across all instances)
QUEUE_THRESHOLD=100

# Common CLI flags
COMMON_FLAGS="--model $MODEL --tp $TP --hardware $HARDWARE --num-instances $INSTANCES --routing-policy weighted --routing-scorers prefix-affinity:3,queue-depth:2 --scheduler priority-fcfs --log error"

# ── Workload YAML generation ────────────────────────────────────────────────

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

# ── Run helper ──────────────────────────────────────────────────────────────

run_config() {
    local label=$1
    local admission=$2
    local threshold=$3
    local rate=$4
    local seed=$5
    local slo_mix=$6

    local outfile="$RESULTS_DIR/${label}.txt"
    local workload_yaml="$RESULTS_DIR/${label}_workload.yaml"

    generate_workload_yaml "$rate" "$slo_mix" "$seed" "$workload_yaml"

    local admission_flags=""
    if [[ "$admission" == "slo-gated" ]]; then
        admission_flags="--admission-policy slo-gated --token-bucket-capacity $threshold"
    else
        admission_flags="--admission-policy always-admit"
    fi

    echo -n "  [$label] admission=$admission rate=$rate seed=$seed ... "

    local exit_code=0
    blis_run $TIMEOUT_EXTENDED "$outfile" \
        $COMMON_FLAGS \
        --seed "$seed" \
        --workload-spec "$workload_yaml" \
        --priority-policy static-class-weight \
        $admission_flags \
        --summarize-trace \
        --trace-level decisions \
        || exit_code=$?

    if is_timeout "$outfile"; then
        echo "TIMEOUT/ERROR"
    else
        echo "ok (exit=$exit_code)"
    fi
}

# ── Experiment Execution ────────────────────────────────────────────────────

echo "============================================================================"
echo "  Strategy Evolution Iteration 2: SLO-Gated Admission Control"
echo "  Reference: hypotheses/h-slo-admission/problem.md"
echo "============================================================================"
echo ""
echo "Config: instances=$INSTANCES, requests=$NUM_REQUESTS, seeds=$SEEDS"
echo "Rates: 80%=${RATE_80}, 120%=${RATE_120}"
echo "Queue threshold: $QUEUE_THRESHOLD"
echo ""

# ── H-main: B2 vs T2 at 80% and 120% ──────────────────────────────────────
echo "=== H-main: B2 (always-admit) vs T2 (slo-gated) ==="
echo ""

for SEED in $SEEDS; do
    for RATE_LABEL in "80" "120"; do
        if [[ "$RATE_LABEL" == "80" ]]; then RATE=$RATE_80; else RATE=$RATE_120; fi
        run_config "B2_${RATE_LABEL}pct_s${SEED}" "always-admit" 0 "$RATE" "$SEED" "mixed"
        run_config "T2_${RATE_LABEL}pct_s${SEED}" "slo-gated" "$QUEUE_THRESHOLD" "$RATE" "$SEED" "mixed"
    done
done

echo ""

# ── H-control-negative: Uniform SLO at 120% ────────────────────────────────
echo "=== H-control-negative: Uniform SLO (all standard) at 120% ==="
echo ""

for SEED in $SEEDS; do
    run_config "B2_uniform_120pct_s${SEED}" "always-admit" 0 "$RATE_120" "$SEED" "uniform"
    run_config "T2_uniform_120pct_s${SEED}" "slo-gated" "$QUEUE_THRESHOLD" "$RATE_120" "$SEED" "uniform"
done

echo ""

# ── H-threshold-sensitivity: Thresholds [50, 100, 200] at 120% ─────────────
echo "=== H-threshold-sensitivity: Thresholds [50, 100, 200] at 120% ==="
echo ""

for SEED in $SEEDS; do
    for THRESH in 50 100 200; do
        run_config "T2_thresh${THRESH}_120pct_s${SEED}" "slo-gated" "$THRESH" "$RATE_120" "$SEED" "mixed"
    done
done

echo ""

# ── Copy results to experiment directory ────────────────────────────────────
echo "Copying results to $SCRIPT_DIR/results/ ..."
rm -rf "$SCRIPT_DIR/results"
cp -r "$RESULTS_DIR" "$SCRIPT_DIR/results"

echo ""

# ── Analysis ────────────────────────────────────────────────────────────────
echo "=== Running Analysis ==="
echo ""

python3 "$SCRIPT_DIR/analyze.py" "$SCRIPT_DIR/results"

echo ""
echo "============================================================================"
echo "  Experiment complete. Results in hypotheses/h-slo-admission/results/"
echo "============================================================================"
