#!/bin/bash
# H28: Chunked Prefill ITL Impact
# Tests whether enabling chunked prefill (threshold=512) improves mean ITL for
# concurrent decode requests when large-input (2048-token) prefills are present,
# at the cost of higher TTFT for those large-input requests.
#
# Family: Structural model
# VV&UQ: Sensitivity analysis (parameter sweep: threshold)
#
# Design:
#   - Mixed workload: 50% long-input (2048 tokens), 50% short-input (128 tokens)
#   - All requests: 256 output tokens (enough decode steps to measure ITL)
#   - Config A: threshold=0 (chunking disabled — full prefill in one step)
#   - Config B: threshold=512 (chunking enabled — 2048-token prefill split over ~4 steps)
#   - 4 instances, 200 requests, 3 seeds (42, 123, 456)
#   - Rate: 120 req/s (~73% utilization) to ensure batching concurrency
#   - Mechanism: With threshold=0, a 2048-token prefill monopolizes step time
#     (~beta0 + beta1*2048 = ~43.1ms). With threshold=512, each chunk is ~15.9ms,
#     letting decode requests interleave with ~4x shorter steps.
#
# Config diff (ED-6): Only --long-prefill-token-threshold differs between A and B.
# All other parameters (model, instances, routing, KV, workload) are identical.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"

setup_experiment "${1:-}"

# The binary needs defaults.yaml in its cwd
cd "$REPO_ROOT"

# =============================================================================
# Workload: 50% long-input (2048), 50% short-input (128), all 256 output tokens
# Rate: 120 req/s across 4 instances = 30 req/s per instance
# Capacity estimate: mean input = (2048+128)/2 = 1088 tokens
#   step time ~ beta0 + beta1*1088 = 6910 + 17.67*1088 = ~26.1ms
#   capacity per instance ~ 1000/26.1 = ~38 req/s → 4 instances = ~152 req/s
#   120/152 = ~79% utilization → batching will occur but no overload
# =============================================================================
make_workload() {
    local outfile=$1
    local seed=$2
    cat > "$outfile" << YAMLEOF
version: "1"
seed: ${seed}
category: language
aggregate_rate: 120.0
num_requests: 200
clients:
  - id: "long-input"
    tenant_id: "default"
    slo_class: "interactive"
    rate_fraction: 0.5
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 2048
    output_distribution:
      type: constant
      params:
        value: 256
  - id: "short-input"
    tenant_id: "default"
    slo_class: "interactive"
    rate_fraction: 0.5
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: constant
      params:
        value: 128
    output_distribution:
      type: constant
      params:
        value: 256
YAMLEOF
}

echo "================================================================" >&2
echo "H28: Chunked Prefill ITL Impact" >&2
echo "================================================================" >&2

SEEDS=(42 123 456)

# =============================================================================
# Config A: Chunking disabled (threshold=0)
# 2048-token prefills processed in a single step (~43ms step time)
# =============================================================================
echo "" >&2
echo "--- Config A: threshold=0 (chunking disabled) ---" >&2

for seed in "${SEEDS[@]}"; do
    echo "  Seed $seed..." >&2

    WORKLOAD_FILE="$RESULTS_DIR/workload_a_s${seed}.yaml"
    make_workload "$WORKLOAD_FILE" "$seed"

    OUT="$RESULTS_DIR/a_s${seed}_stdout.txt"
    RESULTS_JSON="$RESULTS_DIR/a_s${seed}_results.json"

    blis_run $TIMEOUT_STANDARD "$OUT" \
        --model "$MODEL" \
        --num-instances 4 \
        --seed "$seed" \
        --routing-policy least-loaded \
        --scheduler fcfs \
        --admission-policy always-admit \
        --long-prefill-token-threshold 0 \
        --num-requests 200 \
        --workload-spec "$WORKLOAD_FILE" \
        --results-path "$RESULTS_JSON" \
        --log error || true  # harness writes ERROR:<code> sentinel; analyze.py handles via check_for_timeout()
done

# =============================================================================
# Config B: Chunking enabled (threshold=512)
# 2048-token prefills chunked into ~4 steps of 512 tokens each (~15.9ms each)
# =============================================================================
echo "" >&2
echo "--- Config B: threshold=512 (chunking enabled) ---" >&2

for seed in "${SEEDS[@]}"; do
    echo "  Seed $seed..." >&2

    WORKLOAD_FILE="$RESULTS_DIR/workload_b_s${seed}.yaml"
    make_workload "$WORKLOAD_FILE" "$seed"

    OUT="$RESULTS_DIR/b_s${seed}_stdout.txt"
    RESULTS_JSON="$RESULTS_DIR/b_s${seed}_results.json"

    blis_run $TIMEOUT_STANDARD "$OUT" \
        --model "$MODEL" \
        --num-instances 4 \
        --seed "$seed" \
        --routing-policy least-loaded \
        --scheduler fcfs \
        --admission-policy always-admit \
        --long-prefill-token-threshold 512 \
        --num-requests 200 \
        --workload-spec "$WORKLOAD_FILE" \
        --results-path "$RESULTS_JSON" \
        --log error || true  # harness writes ERROR:<code> sentinel; analyze.py handles via check_for_timeout()
done

# =============================================================================
# Analysis
# =============================================================================
echo "" >&2
echo "--- Running analysis ---" >&2

python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR"
