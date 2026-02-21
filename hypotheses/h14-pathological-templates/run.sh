#!/bin/bash
# H14: Pathological Templates — anomaly detection validation
#
# Hypothesis: Pathological policies (always-busiest, reverse-priority,
# inverted-slo) should produce measurably worse behavior than normal
# counterparts, and anomaly detectors should correctly identify the
# degradation (HOL blocking events > 0, priority inversions > 0).
#
# Usage: ./run.sh [--rebuild]
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
WORKLOAD="$SCRIPT_DIR/mixed-slo.yaml"

run_sim() {
    local routing="$1" scheduler="$2" priority="$3" seed="$4" instances="${5:-4}"
    "$BINARY" run \
        --model "$MODEL" \
        --num-instances "$instances" \
        --workload-spec "$WORKLOAD" \
        --num-requests 500 \
        --routing-policy "$routing" \
        --scheduler "$scheduler" \
        --priority-policy "$priority" \
        --seed "$seed" \
        --log error \
        --summarize-trace \
        --trace-level decisions \
        2>/dev/null
}

analyze() {
    python3 "$SCRIPT_DIR/analyze.py" "$@"
}

echo "============================================================================"
echo "  H14: Pathological Templates — Anomaly Detection Validation"
echo "  Reference: docs/plans/research.md, Idea 2, Hypothesis 14"
echo "============================================================================"
echo ""

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# ── Experiment 1: Normal vs Pathological (3 seeds) ──────────────────────────

echo "Experiment 1: Normal vs Pathological (rate=2000, 500 requests, 3 seeds)"
echo "  Normal:       least-loaded + priority-fcfs + slo-based"
echo "  Pathological: always-busiest + priority-fcfs + inverted-slo"
echo ""

for SEED in 42 123 456; do
    run_sim "least-loaded" "priority-fcfs" "slo-based" "$SEED" \
        > "$RESULTS_DIR/normal_${SEED}.txt"
    run_sim "always-busiest" "priority-fcfs" "inverted-slo" "$SEED" \
        > "$RESULTS_DIR/patho_${SEED}.txt"
done
analyze core "$RESULTS_DIR"/normal_*.txt "$RESULTS_DIR"/patho_*.txt

# ── Experiment 2: Decomposed (routing-only vs scheduling-only) ──────────────

echo ""
echo "Experiment 2: Decomposed — isolate routing vs scheduling contribution (seed 42)"
echo ""

# Already have normal and full pathological from Exp 1
cp "$RESULTS_DIR/normal_42.txt" "$RESULTS_DIR/normal.txt"
cp "$RESULTS_DIR/patho_42.txt" "$RESULTS_DIR/patho.txt"

# Routing-only pathological: always-busiest + normal scheduling
run_sim "always-busiest" "priority-fcfs" "slo-based" 42 \
    > "$RESULTS_DIR/routing_only.txt"

# Scheduling-only pathological: normal routing + priority-fcfs + inverted-slo
run_sim "least-loaded" "priority-fcfs" "inverted-slo" 42 \
    > "$RESULTS_DIR/sched_only.txt"

analyze decomposed "$RESULTS_DIR/normal.txt" "$RESULTS_DIR/routing_only.txt" \
    "$RESULTS_DIR/sched_only.txt" "$RESULTS_DIR/patho.txt"

# ── Experiment 3: ED-2 Rate Awareness — scheduling at 1 instance ────────────
# ED-2: "Also run at a rate where the effect should vanish (to confirm the
# mechanism, not just the outcome)." Scheduling effects vanish at 4 instances
# (Exp 2) but should appear at 1 instance where all 500 requests queue up.

echo ""
echo "Experiment 3: ED-2 — scheduling effect at 1 instance (seed 42)"
echo "  Tests whether reverse-priority becomes visible with deeper queues"
echo ""

run_sim "round-robin" "priority-fcfs" "slo-based" 42 1 \
    > "$RESULTS_DIR/sched_1inst_normal.txt"
run_sim "round-robin" "reverse-priority" "inverted-slo" 42 1 \
    > "$RESULTS_DIR/sched_1inst_double.txt"
# Single inversions: only one component pathological
run_sim "round-robin" "priority-fcfs" "inverted-slo" 42 1 \
    > "$RESULTS_DIR/sched_1inst_inv_prio.txt"
run_sim "round-robin" "reverse-priority" "slo-based" 42 1 \
    > "$RESULTS_DIR/sched_1inst_rev_sched.txt"

analyze scheduling "$RESULTS_DIR/sched_1inst_normal.txt" \
    "$RESULTS_DIR/sched_1inst_double.txt" \
    "$RESULTS_DIR/sched_1inst_inv_prio.txt" \
    "$RESULTS_DIR/sched_1inst_rev_sched.txt"

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis and root cause"
echo "============================================================================"
