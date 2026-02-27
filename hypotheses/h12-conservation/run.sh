#!/bin/bash
# H12: Request Conservation Invariant
#
# Hypothesis: No matter what routing, scheduling, or admission policy is used,
# every injected request must end up completed, queued, or running at simulation
# end: injected == completed + still_queued + still_running.
# With admission control: num_requests == injected + rejected.
#
# Classification: Deterministic (Type 1) — single seed, exact pass/fail.
#
# Usage: ./run.sh [--rebuild]
#   --rebuild  Force rebuild of the binary
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
SEED=42
NUM_REQUESTS=200
RATE=500
INSTANCES=4

run_sim() {
    local label="$1"; shift
    "$BINARY" run \
        --model "$MODEL" \
        --num-instances "$INSTANCES" \
        --num-requests "$NUM_REQUESTS" \
        --rate "$RATE" \
        --seed "$SEED" \
        --log error \
        --summarize-trace \
        --trace-level decisions \
        "$@" 2>/dev/null
}

analyze() {
    python3 "$SCRIPT_DIR/analyze.py" "$@"
}

echo "============================================================================"
echo "  H12: Request Conservation Invariant"
echo "  Reference: docs/plans/research.md, Idea 2, Hypothesis 12"
echo "  Type: Deterministic (single seed, exact pass/fail)"
echo "============================================================================"
echo ""
echo "Workload: rate=$RATE, requests=$NUM_REQUESTS, instances=$INSTANCES, seed=$SEED"
echo ""

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# ── Configuration 1: Baseline (round-robin + FCFS + always-admit) ────────────
echo "  [1/11] round-robin + fcfs + always-admit (baseline)"
run_sim "baseline" \
    --routing-policy round-robin \
    --scheduler fcfs \
    --priority-policy constant \
    --admission-policy always-admit \
    > "$RESULTS_DIR/cfg01_baseline.txt"

# ── Configuration 2: Least-loaded routing ────────────────────────────────────
echo "  [2/11] least-loaded + fcfs + always-admit"
run_sim "least-loaded" \
    --routing-policy least-loaded \
    --scheduler fcfs \
    --priority-policy constant \
    --admission-policy always-admit \
    > "$RESULTS_DIR/cfg02_least_loaded.txt"

# ── Configuration 3: Weighted routing (queue-depth only) ─────────────────────
echo "  [3/11] weighted (queue-depth:1) + fcfs + always-admit"
run_sim "weighted-qd" \
    --routing-policy weighted \
    --routing-scorers "queue-depth:1" \
    --scheduler fcfs \
    --priority-policy constant \
    --admission-policy always-admit \
    > "$RESULTS_DIR/cfg03_weighted_qd.txt"

# ── Configuration 4: Full scorer stack ───────────────────────────────────────
echo "  [4/11] weighted (pa:3,qd:2,kv:2) + fcfs + always-admit"
run_sim "weighted-full" \
    --routing-policy weighted \
    --routing-scorers "prefix-affinity:3,queue-depth:2,kv-utilization:2" \
    --scheduler fcfs \
    --priority-policy constant \
    --admission-policy always-admit \
    > "$RESULTS_DIR/cfg04_weighted_full.txt"

# ── Configuration 5: SJF scheduler ──────────────────────────────────────────
echo "  [5/11] round-robin + sjf + always-admit"
run_sim "sjf" \
    --routing-policy round-robin \
    --scheduler sjf \
    --priority-policy constant \
    --admission-policy always-admit \
    > "$RESULTS_DIR/cfg05_sjf.txt"

# ── Configuration 6: Priority scheduling ─────────────────────────────────────
echo "  [6/11] round-robin + priority-fcfs + slo-based + always-admit"
run_sim "priority" \
    --routing-policy round-robin \
    --scheduler priority-fcfs \
    --priority-policy slo-based \
    --admission-policy always-admit \
    > "$RESULTS_DIR/cfg06_priority.txt"

# ── Configuration 7: Token-bucket admission ──────────────────────────────────
echo "  [7/11] round-robin + fcfs + token-bucket (tests full-pipeline invariant)"
run_sim "token-bucket" \
    --routing-policy round-robin \
    --scheduler fcfs \
    --priority-policy constant \
    --admission-policy token-bucket \
    --token-bucket-capacity 50 \
    --token-bucket-refill-rate 100 \
    > "$RESULTS_DIR/cfg07_token_bucket.txt"

# ── Configuration 8: High-rate stress (deep queues) ─────────────────────────
echo "  [8/11] least-loaded + fcfs + high rate=2000 (queue stress)"
"$BINARY" run \
    --model "$MODEL" \
    --num-instances "$INSTANCES" \
    --num-requests "$NUM_REQUESTS" \
    --rate 2000 \
    --seed "$SEED" \
    --log error \
    --summarize-trace \
    --trace-level decisions \
    --routing-policy least-loaded \
    --scheduler fcfs \
    --priority-policy constant \
    --admission-policy always-admit \
    2>/dev/null > "$RESULTS_DIR/cfg08_high_rate.txt"

# ── Configuration 9: Combined policies ───────────────────────────────────────
echo "  [9/11] weighted (qd:2,kv:2) + priority-fcfs + slo-based + token-bucket"
run_sim "combined" \
    --routing-policy weighted \
    --routing-scorers "queue-depth:2,kv-utilization:2" \
    --scheduler priority-fcfs \
    --priority-policy slo-based \
    --admission-policy token-bucket \
    --token-bucket-capacity 50 \
    --token-bucket-refill-rate 100 \
    > "$RESULTS_DIR/cfg09_combined.txt"

# ── Configuration 10: Pathological ───────────────────────────────────────────
echo "  [10/11] always-busiest + reverse-priority + inverted-slo (pathological)"
run_sim "pathological" \
    --routing-policy always-busiest \
    --scheduler reverse-priority \
    --priority-policy inverted-slo \
    --admission-policy always-admit \
    > "$RESULTS_DIR/cfg10_pathological.txt"

# ── Configuration 11: KV pressure (expected panic — documents bug) ───────────
echo "  [11/11] least-loaded + KV=500 (preemption pressure — expected panic)"
echo "          Testing preemption path: sim.preempt() panics on empty RunningBatch"
if run_sim "kv-constrained" \
    --routing-policy least-loaded \
    --scheduler fcfs \
    --priority-policy constant \
    --admission-policy always-admit \
    --total-kv-blocks 500 \
    > "$RESULTS_DIR/cfg11_kv_panic.txt" 2>&1; then
    echo "          Result: No panic (unexpected — preemption bug may be fixed)"
else
    echo "          Result: PANIC confirmed — preempt() index out of range [-1]"
    echo "          See GitHub issue for fix"
fi

echo ""
echo "Conservation invariant analysis (configs 1-10):"
echo ""

analyze "$NUM_REQUESTS" "$RESULTS_DIR"/cfg{01,02,03,04,05,06,07,08,09,10}_*.txt

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis"
echo "============================================================================"
