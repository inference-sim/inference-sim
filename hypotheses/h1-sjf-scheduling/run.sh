#!/bin/bash
# H1-SJF: SJF scheduling should reduce TTFT for mixed-length workloads
#
# Hypothesis: When short and long requests are mixed, SJF scheduling should
# reduce TTFT for short requests compared to FCFS, because short jobs no longer
# wait behind long ones in the queue (the classic SJF result from OS scheduling).
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
WORKLOAD="$SCRIPT_DIR/mixed-workload.yaml"

run_sim() {
    local sched="$1" seed="$2" results_file="$3"
    "$BINARY" run \
        --model "$MODEL" \
        --num-instances 4 \
        --workload-spec "$WORKLOAD" \
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
echo "  H1-SJF: SJF vs FCFS Scheduling for Mixed-Length Workloads"
echo "  Family: Cross-policy comparative | Type: Statistical/Dominance"
echo "============================================================================"
echo ""

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# ── Experiment 1: FCFS vs SJF across 3 seeds ─────────────────────────────────
# Controlled variables: model, num-instances (4), workload, routing (round-robin default)
# Varied variable: scheduler (fcfs vs sjf)
# Seeds: 42, 123, 456

echo "Experiment 1: FCFS vs SJF (rate=3000, 1000 requests, 4 instances, 3 seeds)"
echo "  Workload: 50% short (32 input tokens) + 50% long (1024 input tokens)"
echo "  FCFS: --scheduler fcfs"
echo "  SJF:  --scheduler sjf"
echo ""

for SEED in 42 123 456; do
    echo "  Running seed=$SEED..."
    run_sim "fcfs" "$SEED" "$RESULTS_DIR/fcfs_${SEED}.json" \
        > "$RESULTS_DIR/fcfs_${SEED}_stdout.txt"
    run_sim "sjf" "$SEED" "$RESULTS_DIR/sjf_${SEED}.json" \
        > "$RESULTS_DIR/sjf_${SEED}_stdout.txt"
done

echo ""
echo "--- Aggregate Comparison (from stdout) ---"
analyze aggregate \
    "$RESULTS_DIR/fcfs_42_stdout.txt" "$RESULTS_DIR/fcfs_123_stdout.txt" "$RESULTS_DIR/fcfs_456_stdout.txt" \
    "$RESULTS_DIR/sjf_42_stdout.txt" "$RESULTS_DIR/sjf_123_stdout.txt" "$RESULTS_DIR/sjf_456_stdout.txt"

echo ""
echo "--- Per-SLO-Class Comparison (from stdout) ---"
analyze per_slo \
    "$RESULTS_DIR/fcfs_42_stdout.txt" "$RESULTS_DIR/fcfs_123_stdout.txt" "$RESULTS_DIR/fcfs_456_stdout.txt" \
    "$RESULTS_DIR/sjf_42_stdout.txt" "$RESULTS_DIR/sjf_123_stdout.txt" "$RESULTS_DIR/sjf_456_stdout.txt"

echo ""
echo "--- Per-Request Analysis (from JSON results) ---"
analyze per_request \
    "$RESULTS_DIR/fcfs_42.json" "$RESULTS_DIR/fcfs_123.json" "$RESULTS_DIR/fcfs_456.json" \
    "$RESULTS_DIR/sjf_42.json" "$RESULTS_DIR/sjf_123.json" "$RESULTS_DIR/sjf_456.json"

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis and root cause"
echo "============================================================================"
