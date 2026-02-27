#!/bin/bash
# Prefix-Affinity Hypothesis: Prefix-aware routing should outperform
# load-only routing for prefix-heavy workloads
#
# This was the original validated methodology (PR18) that established
# the hypothesis-driven testing approach. It demonstrated that:
# 1. Prefix-affinity routing routes multi-turn sessions to cached instances
# 2. Queue-depth routing actively destroys cache locality (3-4x worse TTFT)
# 3. Round-robin achieves accidental cache reuse from its cyclic pattern
#
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$REPO_ROOT/blis"

# Build if needed
if [[ "${1:-}" == "--rebuild" ]] || [[ ! -x "$BINARY" ]]; then
    echo "Building blis..."
    (cd "$REPO_ROOT" && go build -o blis main.go)
fi

MODEL="meta-llama/llama-3.1-8b-instruct"

run_sim() {
    local extra="$@"
    "$BINARY" run \
        --model "$MODEL" \
        --num-instances 4 \
        --seed 42 \
        --log error \
        --summarize-trace \
        --trace-level decisions \
        $extra \
        2>/dev/null
}

analyze() {
    python3 "$SCRIPT_DIR/analyze.py" "$@"
}

echo "============================================================================"
echo "  Prefix-Affinity Hypothesis"
echo "  Reference: docs/plans/research.md, Validated Methodology (PR18)"
echo "============================================================================"
echo ""

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# ── Experiment 1: Multi-turn chat (context accumulation) ─────────────────────

echo "Experiment 1: Multi-turn chat — cache reuse from session affinity"
echo "  Workload: examples/multiturn-chat-demo.yaml (5 rounds, context accumulates)"
echo ""

run_sim --routing-policy weighted --routing-scorers "prefix-affinity:3,queue-depth:2" \
    --workload-spec "$REPO_ROOT/examples/multiturn-chat-demo.yaml" \
    > "$RESULTS_DIR/exp1_pa.txt"
run_sim --routing-policy weighted --routing-scorers "queue-depth:1" \
    --workload-spec "$REPO_ROOT/examples/multiturn-chat-demo.yaml" \
    > "$RESULTS_DIR/exp1_qd.txt"
run_sim --routing-policy round-robin \
    --workload-spec "$REPO_ROOT/examples/multiturn-chat-demo.yaml" \
    > "$RESULTS_DIR/exp1_rr.txt"
run_sim --routing-policy weighted \
    --workload-spec "$REPO_ROOT/examples/multiturn-chat-demo.yaml" \
    > "$RESULTS_DIR/exp1_llmd.txt"

analyze multi-turn "$RESULTS_DIR"/exp1_*.txt

# ── Experiment 2: Multi-turn at high load ─────────────────────────────────────

echo ""
echo "Experiment 2: Multi-turn at high load (rate=5000, 2000 requests)"
echo ""

# Create high-load variant
cat > "$RESULTS_DIR/multiturn-highload.yaml" << 'YAMLEOF'
version: "1"
seed: 42
category: reasoning
aggregate_rate: 5000.0
num_requests: 2000
clients:
  - id: "multi-turn-chat"
    tenant_id: "chat-users"
    slo_class: "interactive"
    rate_fraction: 1.0
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 30
        min: 32
        max: 512
    output_distribution:
      type: gaussian
      params:
        mean: 64
        std_dev: 20
        min: 16
        max: 256
    reasoning:
      reason_ratio_distribution:
        type: gaussian
        params:
          mean: 0
          std_dev: 0
          min: 0
          max: 0
      multi_turn:
        max_rounds: 5
        think_time_us: 500000
        context_growth: accumulate
YAMLEOF

run_sim --routing-policy weighted --routing-scorers "prefix-affinity:3,queue-depth:2" \
    --workload-spec "$RESULTS_DIR/multiturn-highload.yaml" \
    > "$RESULTS_DIR/exp2_pa.txt"
run_sim --routing-policy weighted --routing-scorers "queue-depth:1" \
    --workload-spec "$RESULTS_DIR/multiturn-highload.yaml" \
    > "$RESULTS_DIR/exp2_qd.txt"
run_sim --routing-policy round-robin \
    --workload-spec "$RESULTS_DIR/multiturn-highload.yaml" \
    > "$RESULTS_DIR/exp2_rr.txt"

analyze multi-turn "$RESULTS_DIR"/exp2_*.txt

# ── Experiment 3: Shared system prompt ────────────────────────────────────────

echo ""
echo "Experiment 3: Shared system prompt — concentration vs distribution"
echo "  Workload: examples/prefix-affinity-demo.yaml (80% shared 256-token prefix)"
echo ""

run_sim --routing-policy weighted --routing-scorers "prefix-affinity:5,queue-depth:1" \
    --workload-spec "$REPO_ROOT/examples/prefix-affinity-demo.yaml" \
    > "$RESULTS_DIR/exp3_pa5.txt"
run_sim --routing-policy weighted --routing-scorers "prefix-affinity:1,queue-depth:1" \
    --workload-spec "$REPO_ROOT/examples/prefix-affinity-demo.yaml" \
    > "$RESULTS_DIR/exp3_pa1.txt"
run_sim --routing-policy weighted --routing-scorers "queue-depth:1" \
    --workload-spec "$REPO_ROOT/examples/prefix-affinity-demo.yaml" \
    > "$RESULTS_DIR/exp3_qd.txt"
run_sim --routing-policy round-robin \
    --workload-spec "$REPO_ROOT/examples/prefix-affinity-demo.yaml" \
    > "$RESULTS_DIR/exp3_rr.txt"

analyze shared-prompt "$RESULTS_DIR"/exp3_*.txt

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis and root cause"
echo "============================================================================"
