#!/bin/bash
# H2-Priority-FCFS: Priority-FCFS with SLO-based priority vs constant+FCFS
#
# Hypothesis: "Priority-FCFS with SLO-based priority should reduce realtime TTFT
# at the cost of batch TTFT"
#
# Config A (baseline): --priority-policy constant --scheduler fcfs
# Config B (prioritized): --priority-policy slo-based --scheduler priority-fcfs
#
# Family: Cross-policy comparative | Type: Statistical (Dominance)
# VV&UQ: Validation | Tier: 5 (workload diversity)
#
# Usage: ./run.sh [--rebuild]
#
# Requires: Go 1.24+, Python 3

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
WORKLOAD="$SCRIPT_DIR/mixed-slo-workload.yaml"

run_sim() {
    local priority="$1" sched="$2" seed="$3" results_file="$4"
    "$BINARY" run \
        --model "$MODEL" \
        --num-instances 4 \
        --workload-spec "$WORKLOAD" \
        --priority-policy "$priority" \
        --scheduler "$sched" \
        --seed "$seed" \
        --log error \
        --results-path "$results_file" \
        2>/dev/null
}

analyze() {
    python3 "$SCRIPT_DIR/analyze.py" "$@"
}

echo "============================================================================"
echo "  H2-Priority-FCFS: Constant+FCFS vs SLO-Based+Priority-FCFS"
echo "  Family: Cross-policy comparative | Type: Statistical/Dominance"
echo "============================================================================"
echo ""

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# ── Experiment: Config A vs Config B across 3 seeds ──────────────────────────
# Config A (baseline): --priority-policy constant --scheduler fcfs
# Config B (prioritized): --priority-policy slo-based --scheduler priority-fcfs
# Controlled: model, num-instances (4), workload, routing (round-robin default)
# Seeds: 42, 123, 456

echo "Experiment: Constant+FCFS vs SLO-Based+Priority-FCFS"
echo "  Workload: Mixed SLO (33% realtime, 34% interactive, 33% batch)"
echo "  All clients: 256 input tokens, 128 output tokens, poisson arrival"
echo "  Rate: 500 req/s, 500 requests, 4 instances"
echo "  Config A: --priority-policy constant --scheduler fcfs"
echo "  Config B: --priority-policy slo-based --scheduler priority-fcfs"
echo ""

for SEED in 42 123 456; do
    echo "  Running seed=$SEED..."
    run_sim "constant" "fcfs" "$SEED" "$RESULTS_DIR/a_${SEED}.json" \
        > "$RESULTS_DIR/a_${SEED}_stdout.txt"
    run_sim "slo-based" "priority-fcfs" "$SEED" "$RESULTS_DIR/b_${SEED}.json" \
        > "$RESULTS_DIR/b_${SEED}_stdout.txt"
done

echo ""
echo "--- Aggregate Comparison (from stdout) ---"
analyze aggregate \
    "$RESULTS_DIR/a_42_stdout.txt" "$RESULTS_DIR/a_123_stdout.txt" "$RESULTS_DIR/a_456_stdout.txt" \
    "$RESULTS_DIR/b_42_stdout.txt" "$RESULTS_DIR/b_123_stdout.txt" "$RESULTS_DIR/b_456_stdout.txt"

echo ""
echo "--- Per-SLO-Class Comparison (from stdout) ---"
analyze per_slo \
    "$RESULTS_DIR/a_42_stdout.txt" "$RESULTS_DIR/a_123_stdout.txt" "$RESULTS_DIR/a_456_stdout.txt" \
    "$RESULTS_DIR/b_42_stdout.txt" "$RESULTS_DIR/b_123_stdout.txt" "$RESULTS_DIR/b_456_stdout.txt"

echo ""
echo "--- Per-Request Analysis (from JSON results) ---"
analyze per_request \
    "$RESULTS_DIR/a_42.json" "$RESULTS_DIR/a_123.json" "$RESULTS_DIR/a_456.json" \
    "$RESULTS_DIR/b_42.json" "$RESULTS_DIR/b_123.json" "$RESULTS_DIR/b_456.json"

echo ""
echo "--- Priority Distribution Analysis ---"
analyze priority \
    "$RESULTS_DIR/a_42.json" "$RESULTS_DIR/a_123.json" "$RESULTS_DIR/a_456.json" \
    "$RESULTS_DIR/b_42.json" "$RESULTS_DIR/b_123.json" "$RESULTS_DIR/b_456.json"

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis and root cause"
echo "============================================================================"
