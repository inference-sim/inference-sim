#!/bin/bash
# H24: Combined Pathological Anomalies
# Combining always-busiest routing with inverted-slo scheduling should produce
# maximum measurable anomalies (HOL blocking > 0, priority inversions > 0,
# degraded TTFT) compared to normal policies.
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"

setup_experiment "${1:-}"

# Generate mixed-SLO workload YAML inline (mirrors ScenarioMixedSLO)
WORKLOAD_YAML="$RESULTS_DIR/mixed-slo.yaml"
cat > "$WORKLOAD_YAML" <<'YAML'
version: "1"
seed: 42
category: language
aggregate_rate: 2000
num_requests: 500
clients:
  - id: realtime
    tenant_id: tenant-rt
    slo_class: realtime
    rate_fraction: 0.33
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 64
        std_dev: 20
        min: 10
        max: 256
    output_distribution:
      type: exponential
      params:
        mean: 32
  - id: interactive
    tenant_id: tenant-int
    slo_class: interactive
    rate_fraction: 0.34
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 100
        min: 32
        max: 2048
    output_distribution:
      type: exponential
      params:
        mean: 128
  - id: batch
    tenant_id: tenant-batch
    slo_class: batch
    rate_fraction: 0.33
    arrival:
      process: poisson
    input_distribution:
      type: exponential
      params:
        mean: 1024
    output_distribution:
      type: exponential
      params:
        mean: 512
YAML

# Common flags shared by all runs
COMMON_FLAGS=(
    --model "$MODEL"
    --num-instances 4
    --workload-spec "$WORKLOAD_YAML"
    --num-requests 500
    --scheduler priority-fcfs
    --log error
    --summarize-trace
    --trace-level decisions
)

run_normal() {
    local seed="$1" output="$2"
    blis_run $TIMEOUT_STANDARD "$output" \
        "${COMMON_FLAGS[@]}" \
        --routing-policy least-loaded \
        --priority-policy slo-based \
        --seed "$seed"
}

run_pathological() {
    local seed="$1" output="$2"
    blis_run $TIMEOUT_STANDARD "$output" \
        "${COMMON_FLAGS[@]}" \
        --routing-policy always-busiest \
        --priority-policy inverted-slo \
        --seed "$seed"
}

echo "============================================================================"
echo "  H24: Combined Pathological Anomalies"
echo "  Hypothesis: always-busiest + inverted-slo should produce maximum anomalies"
echo "============================================================================"
echo ""

# -- Experiment 1: Normal vs Pathological across 3 seeds ------------------

echo "Experiment 1: Normal vs Pathological (rate=2000, 500 requests, 3 seeds)"
echo "  Normal:       least-loaded + priority-fcfs + slo-based"
echo "  Pathological: always-busiest + priority-fcfs + inverted-slo"
echo ""

for SEED in 42 123 456; do
    echo "  Running seed=$SEED (normal)..." >&2
    run_normal "$SEED" "$RESULTS_DIR/normal_${SEED}.txt"
    echo "  Running seed=$SEED (pathological)..." >&2
    run_pathological "$SEED" "$RESULTS_DIR/patho_${SEED}.txt"
done

python3 "$SCRIPT_DIR/analyze.py" core "$RESULTS_DIR"

# -- Experiment 2: Decomposed (routing-only vs scheduling-only) -----------

echo ""
echo "Experiment 2: Decomposed — isolate routing vs scheduling contribution (seed 42)"
echo ""

# Routing-only pathological: always-busiest + slo-based (correct priority)
echo "  Running routing-only pathological..." >&2
blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/routing_only_42.txt" \
    "${COMMON_FLAGS[@]}" \
    --routing-policy always-busiest \
    --priority-policy slo-based \
    --seed 42

# Scheduling-only pathological: least-loaded + inverted-slo
echo "  Running scheduling-only pathological..." >&2
blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/sched_only_42.txt" \
    "${COMMON_FLAGS[@]}" \
    --routing-policy least-loaded \
    --priority-policy inverted-slo \
    --seed 42

python3 "$SCRIPT_DIR/analyze.py" decomposed "$RESULTS_DIR"

# -- Experiment 3: Per-SLO class impact -----------------------------------

echo ""
echo "Experiment 3: Per-SLO class impact analysis (seed 42)"
echo "  Which SLO class is hurt most by the pathological combination?"
echo ""

python3 "$SCRIPT_DIR/analyze.py" per_slo "$RESULTS_DIR"

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis and root cause"
echo "============================================================================"
