#!/bin/bash
# H-Overload: 10x overload robustness
#
# Hypothesis: Under stress condition of 10x saturation rate, the system should
# exhibit defined overload behavior (queue growth with always-admit, or rejection
# with token-bucket) and NOT exhibit undefined behavior (panic, deadlock, silent
# data loss). Conservation (INV-1) must hold at all overload levels.
#
# Classification: Deterministic (exact pass/fail on conservation invariant)
# Family: Robustness/failure-mode
# VV&UQ: Verification
#
# Reference: hypotheses/h12-conservation/run.sh (conservation checking pattern)
#
# Design: Use large num_requests (2000) with a fixed time horizon (5 seconds)
# so that at high rates, many requests are generated but few complete,
# creating genuine overload with queue buildup and incomplete requests.
# The horizon (5,000,000 ticks = 5s) is long enough for ~300 requests to
# complete at 1x but short enough that at 10x rate, the queue cannot drain.
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
NUM_REQUESTS=2000
INSTANCES=4

# Use a fixed horizon of 5 seconds (5,000,000 microseconds = 5,000,000 ticks)
# This ensures:
# - At 1x (300 req/s): ~1500 arrivals in 5s, system can process many
# - At 10x (3000 req/s): ~2000 arrivals (capped), massive queue buildup
HORIZON=5000000

# High-load reference rate for 4 instances with default request (input=512, output=512):
# Single-request prefill step: 6910 + 17.67*512 = ~15957 us. Decode step: ~6913 us.
# With batching (max_running_reqs=256), decode cost: 6910 + 2.84*256 = ~7637 us/step.
# Effective throughput ~60-70 req/s/instance, ~250-280 req/s for 4 instances.
# 300 req/s slightly exceeds this, ensuring genuine overload at 1x and above.
SAT_RATE=300

# Rate multipliers for the sweep
MULTIPLIERS="1 2 4 7 10"

# Token bucket parameters:
# Cost model is per-input-token (sim/admission.go:45). With default prompt mean=512,
# each request costs ~512 tokens. Token demand at 1x rate: 300 * 512 = 153,600 tokens/s.
# Set refill_rate to ~1x demand: admits most at 1x, rejects heavily at 10x.
# Capacity = refill_rate * 0.5s = 80,000 (small buffer).
TB_CAPACITY=80000
TB_REFILL_RATE=160000

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

echo "============================================================================"
echo "  H-Overload: 10x Overload Robustness"
echo "  Reference: GitHub Issue #315"
echo "  Type: Deterministic (single seed, exact pass/fail on conservation)"
echo "============================================================================"
echo ""
echo "Configuration: instances=$INSTANCES, requests=$NUM_REQUESTS, seed=$SEED"
echo "Horizon: $HORIZON ticks (5 seconds)"
echo "Saturation rate estimate: $SAT_RATE req/s"
echo "Token bucket: capacity=$TB_CAPACITY tokens, refill_rate=$TB_REFILL_RATE tokens/s"
echo "  (per-input-token cost, mean prompt ~512 tokens => ~312 req/s capacity)"
echo ""

# ── Part 1: Always-Admit rate sweep ──────────────────────────────────────────
echo "=== Part 1: Always-Admit Rate Sweep ==="
echo ""

for MULT in $MULTIPLIERS; do
    RATE=$((SAT_RATE * MULT))
    LABEL="always_admit_${MULT}x"
    OUTFILE="$RESULTS_DIR/${LABEL}.txt"
    ERRFILE="$RESULTS_DIR/${LABEL}_stderr.txt"

    echo -n "  [always-admit ${MULT}x] rate=${RATE} ... "

    EXIT_CODE=0
    "$BINARY" run \
        --model "$MODEL" \
        --num-instances "$INSTANCES" \
        --num-requests "$NUM_REQUESTS" \
        --rate "$RATE" \
        --seed "$SEED" \
        --horizon "$HORIZON" \
        --log error \
        --summarize-trace \
        --trace-level decisions \
        --routing-policy least-loaded \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy always-admit \
        > "$OUTFILE" 2>"$ERRFILE" || EXIT_CODE=$?

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "EXIT_CODE=$EXIT_CODE (PANIC/ERROR)"
        echo "STDERR:" >> "$OUTFILE"
        cat "$ERRFILE" >> "$OUTFILE"
    else
        echo "ok (exit=0)"
    fi

    # Record exit code and stderr for analysis
    echo "---EXIT_CODE=$EXIT_CODE---" >> "$OUTFILE"
    if grep -q "panic" "$ERRFILE" 2>/dev/null; then
        echo "---PANIC_DETECTED---" >> "$OUTFILE"
    fi
done

echo ""

# ── Part 2: Token-Bucket rate sweep ──────────────────────────────────────────
echo "=== Part 2: Token-Bucket Rate Sweep ==="
echo ""

for MULT in $MULTIPLIERS; do
    RATE=$((SAT_RATE * MULT))
    LABEL="token_bucket_${MULT}x"
    OUTFILE="$RESULTS_DIR/${LABEL}.txt"
    ERRFILE="$RESULTS_DIR/${LABEL}_stderr.txt"

    echo -n "  [token-bucket ${MULT}x] rate=${RATE} ... "

    EXIT_CODE=0
    "$BINARY" run \
        --model "$MODEL" \
        --num-instances "$INSTANCES" \
        --num-requests "$NUM_REQUESTS" \
        --rate "$RATE" \
        --seed "$SEED" \
        --horizon "$HORIZON" \
        --log error \
        --summarize-trace \
        --trace-level decisions \
        --routing-policy least-loaded \
        --scheduler fcfs \
        --priority-policy constant \
        --admission-policy token-bucket \
        --token-bucket-capacity "$TB_CAPACITY" \
        --token-bucket-refill-rate "$TB_REFILL_RATE" \
        > "$OUTFILE" 2>"$ERRFILE" || EXIT_CODE=$?

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "EXIT_CODE=$EXIT_CODE (PANIC/ERROR)"
        echo "STDERR:" >> "$OUTFILE"
        cat "$ERRFILE" >> "$OUTFILE"
    else
        echo "ok (exit=0)"
    fi

    # Record exit code and stderr for analysis
    echo "---EXIT_CODE=$EXIT_CODE---" >> "$OUTFILE"
    if grep -q "panic" "$ERRFILE" 2>/dev/null; then
        echo "---PANIC_DETECTED---" >> "$OUTFILE"
    fi
done

echo ""

# ── Analysis ─────────────────────────────────────────────────────────────────
echo "=== Conservation Invariant Analysis ==="
echo ""

python3 "$SCRIPT_DIR/analyze.py" "$NUM_REQUESTS" "$RESULTS_DIR"

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis"
echo "============================================================================"
