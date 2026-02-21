#!/bin/bash
# H13: Determinism Invariant
#
# Hypothesis: Same seed must produce byte-identical stdout across runs.
# BLIS uses PartitionedRNG for deterministic simulation — running the same
# configuration with the same seed twice should produce identical output.
#
# Classification: Deterministic (Type 1) — single seed, exact pass/fail.
# One failure = non-determinism bug.
#
# Key risk: prefix-affinity scorer's PrefixCacheIndex uses map iteration
# in evictOldest — non-deterministic map iteration is Go's #1 source of
# INV-6 violations (R2: sort map keys).
#
# Usage: ./run.sh [--rebuild]
#   --rebuild  Force rebuild of the binary
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
SEED=42
NUM_REQUESTS=200
RATE=1000
INSTANCES=4

run_sim() {
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
echo "  H13: Determinism Invariant (INV-6)"
echo "  Reference: docs/plans/research.md, Idea 2, Hypothesis 13"
echo "  Type: Deterministic (single seed, exact pass/fail)"
echo "============================================================================"
echo ""
echo "Workload: rate=$RATE, requests=$NUM_REQUESTS, instances=$INSTANCES, seed=$SEED"
echo ""

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# ── Configuration 1: Round-robin + FCFS (simplest path) ─────────────────────
echo "  [1/5] round-robin + fcfs (simplest code path)"
for RUN in 1 2; do
    run_sim \
        --routing-policy round-robin \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy always-admit \
        > "$RESULTS_DIR/cfg01_rr_run${RUN}.txt"
done

# ── Configuration 2: Least-loaded (PendingRequests tracking) ────────────────
echo "  [2/5] least-loaded + fcfs (PendingRequests bookkeeping)"
for RUN in 1 2; do
    run_sim \
        --routing-policy least-loaded \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy always-admit \
        > "$RESULTS_DIR/cfg02_ll_run${RUN}.txt"
done

# ── Configuration 3: Weighted (queue-depth + kv-utilization) ────────────────
echo "  [3/5] weighted (qd:2,kv:2) + fcfs (scorer pipeline)"
for RUN in 1 2; do
    run_sim \
        --routing-policy weighted \
        --routing-scorers "queue-depth:2,kv-utilization:2" \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy always-admit \
        > "$RESULTS_DIR/cfg03_weighted_run${RUN}.txt"
done

# ── Configuration 4: Prefix-affinity (stateful scorer — highest risk) ───────
# This is the MOST LIKELY violation source: PrefixCacheIndex uses map
# iteration in evictOldest. Must use enough requests + prefix diversity
# to trigger LRU eviction.
echo "  [4/5] weighted (pa:3,qd:2,kv:2) + fcfs (stateful scorer — highest risk)"
for RUN in 1 2; do
    run_sim \
        --routing-policy weighted \
        --routing-scorers "prefix-affinity:3,queue-depth:2,kv-utilization:2" \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy always-admit \
        > "$RESULTS_DIR/cfg04_prefix_run${RUN}.txt"
done

# ── Configuration 5: Priority-FCFS + SLO-based (priority ordering) ──────────
echo "  [5/5] least-loaded + priority-fcfs + slo-based (priority ordering)"
for RUN in 1 2; do
    run_sim \
        --routing-policy least-loaded \
        --scheduler priority-fcfs \
        --priority-policy slo-based \
        --admission-policy always-admit \
        > "$RESULTS_DIR/cfg05_priority_run${RUN}.txt"
done

echo ""
echo "Determinism invariant analysis:"
echo ""

analyze "$RESULTS_DIR"

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis"
echo "============================================================================"
