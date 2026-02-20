#!/bin/bash
# H3: Signal Freshness — queue-depth vs kv-utilization scorer
#
# Hypothesis: At high request rates, the queue-depth scorer should distribute
# requests more evenly than the kv-utilization scorer, because queue-depth
# updates synchronously at routing time (PendingRequests) while KV utilization
# only changes when batch formation allocates blocks (a lagging indicator).
#
# Usage: ./run.sh [--rebuild]
#   --rebuild  Force rebuild of the binary (otherwise uses existing if found)
#
# Requires: Go 1.24+, Python 3

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$REPO_ROOT/simulation_worker"

# Build if needed
if [[ "${1:-}" == "--rebuild" ]] || [[ ! -x "$BINARY" ]]; then
    echo "Building simulation_worker..."
    (cd "$REPO_ROOT" && go build -o simulation_worker main.go)
fi

MODEL="meta-llama/llama-3.1-8b-instruct"

run_sim() {
    local scorers="$1" rate="$2" requests="$3" seed="$4" refresh="${5:-0}"
    "$BINARY" run \
        --model "$MODEL" \
        --num-instances 4 \
        --num-requests "$requests" \
        --rate "$rate" \
        --routing-policy weighted \
        --routing-scorers "$scorers" \
        --seed "$seed" \
        --log error \
        --summarize-trace \
        --trace-level decisions \
        --snapshot-refresh-interval "$refresh" \
        2>/dev/null
}

analyze() {
    python3 "$SCRIPT_DIR/analyze.py" "$@"
}

echo "============================================================================"
echo "  H3: Signal Freshness — queue-depth vs kv-utilization"
echo "  Reference: docs/plans/research.md, Idea 1, Hypothesis 3"
echo "============================================================================"
echo ""

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# ── Experiment 1: Core hypothesis (3 seeds) ──────────────────────────────────

echo "Experiment 1: Core hypothesis (rate=5000, 1000 requests, 3 seeds)"
echo ""

for SEED in 42 123 456; do
    run_sim "queue-depth:1" 5000 1000 "$SEED" > "$RESULTS_DIR/exp1_qd_${SEED}.txt"
    run_sim "kv-utilization:1" 5000 1000 "$SEED" > "$RESULTS_DIR/exp1_kv_${SEED}.txt"
done
analyze core "$RESULTS_DIR"/exp1_*.txt

# ── Experiment 2: Rate scaling ────────────────────────────────────────────────

echo ""
echo "Experiment 2: Rate scaling — does effect diminish at lower rates?"
echo ""

for RATE_REQS in "100:200" "500:500" "1000:500" "2000:1000" "5000:1000"; do
    RATE="${RATE_REQS%%:*}"
    REQS="${RATE_REQS##*:}"
    run_sim "queue-depth:1" "$RATE" "$REQS" 42 > "$RESULTS_DIR/exp2_qd_r${RATE}.txt"
    run_sim "kv-utilization:1" "$RATE" "$REQS" 42 > "$RESULTS_DIR/exp2_kv_r${RATE}.txt"
done
analyze rate-scaling "$RESULTS_DIR"/exp2_*.txt

# ── Experiment 3: Snapshot refresh interval ───────────────────────────────────

echo ""
echo "Experiment 3: Snapshot refresh interval compounding (kv-utilization:1)"
echo ""

for INTERVAL in 0 500 1000 2000 5000 10000; do
    run_sim "kv-utilization:1" 5000 1000 42 "$INTERVAL" > "$RESULTS_DIR/exp3_i${INTERVAL}.txt"
done
analyze refresh-interval "$RESULTS_DIR"/exp3_*.txt

# ── Experiment 4: Combined scorers ────────────────────────────────────────────

echo ""
echo "Experiment 4: Combined scorers — does queue-depth mitigate kv-util staleness?"
echo ""

run_sim "queue-depth:1" 5000 1000 42 > "$RESULTS_DIR/exp4_qd.txt"
run_sim "kv-utilization:1" 5000 1000 42 > "$RESULTS_DIR/exp4_kv.txt"
run_sim "kv-utilization:2,queue-depth:2" 5000 1000 42 > "$RESULTS_DIR/exp4_equal.txt"
run_sim "kv-utilization:5,queue-depth:1" 5000 1000 42 > "$RESULTS_DIR/exp4_kv_dom.txt"
run_sim "queue-depth:5,kv-utilization:1" 5000 1000 42 > "$RESULTS_DIR/exp4_qd_dom.txt"
run_sim "prefix-affinity:3,queue-depth:2,kv-utilization:2" 5000 1000 42 > "$RESULTS_DIR/exp4_llmd.txt"
analyze combined "$RESULTS_DIR"/exp4_*.txt

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis and root cause"
echo "============================================================================"
