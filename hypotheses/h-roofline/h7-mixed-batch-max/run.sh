#!/usr/bin/env bash
# H27 Mixed-Batch Max Combination Validation
#
# Hypothesis: The roofline model's adaptive weighted-average combination for
# mixed prefill+decode steps (roofline_step.go lines 407-430) systematically
# underpredicts mixed-step latency compared to the standard roofline
# max(prefillTime, decodeTime) combination. Replacing with max() should
# improve E2E MAPE for high-QPS workloads by at least 1 percentage point.
#
# Family: Structural model
# VV&UQ: Validation
# Type: Deterministic (unit test style)
#
# This experiment is a UNIT TEST style experiment. It directly calls
# rooflineStepTime with synthetic step configs to compare the current
# weighted-average combination against max(prefillTime, decodeTime),
# computed by running prefill-only and decode-only steps separately
# and taking the max.
#
# The test file (h27_mixed_batch_max_test.go) is placed in sim/ so it can
# access the unexported rooflineStepTime function directly.
#
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Create output directory
OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$OUTPUT_DIR"

# Precondition check: bench_data must exist for MFU database
BENCH_DATA="$REPO_ROOT/bench_data"
if [[ ! -d "$BENCH_DATA" ]]; then
    echo "ERROR: required directory missing: $BENCH_DATA" >&2
    exit 1
fi

echo "=========================================="
echo "  H27: Mixed-Batch Max Combination Test"
echo "=========================================="
echo ""
echo "Baseline:  weighted-average (current adaptive blend)"
echo "Treatment: max(prefillTime, decodeTime)"
echo ""
echo "Test compares rooflineStepTime with mixed batches against"
echo "max of prefill-only and decode-only step times."
echo ""

# Copy test file into sim/latency/ for access to unexported functions.
# Strip //go:build ignore (test file already declares package latency).
TEST_FILE="$REPO_ROOT/sim/latency/h27_mixed_batch_max_test.go"

# Clean up test file on exit (whether success or failure)
cleanup() {
    rm -f "$TEST_FILE"
}
trap cleanup EXIT

grep -v '^//go:build ignore$' "$SCRIPT_DIR/h27_mixed_batch_max_test.go" > "$TEST_FILE"

# Run the test, capturing output
echo "Running H27 mixed-batch comparison test..."
echo ""

cd "$REPO_ROOT"
go test ./sim/latency/... \
    -run "TestH27_MixedBatchMaxComparison" \
    -v \
    -count=1 \
    -timeout 120s \
    2>&1 | tee "$OUTPUT_DIR/test_output.txt"

TEST_EXIT=${PIPESTATUS[0]}

if [[ $TEST_EXIT -ne 0 ]]; then
    echo ""
    echo "ERROR: test exited with code $TEST_EXIT" >&2
    exit $TEST_EXIT
fi

# Check that CSV output was produced
if [[ ! -f "$OUTPUT_DIR/h27_results.csv" ]]; then
    echo "ERROR: expected output file not found: $OUTPUT_DIR/h27_results.csv" >&2
    exit 1
fi

ROWS=$(wc -l < "$OUTPUT_DIR/h27_results.csv")
echo ""
echo "=========================================="
echo "  Test completed. $((ROWS - 1)) test cases written to output/h27_results.csv"
echo "=========================================="
echo ""

# Run analysis
echo "Running analysis..."
python3 "$SCRIPT_DIR/analyze.py" "$OUTPUT_DIR"
