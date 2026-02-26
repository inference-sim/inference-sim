#!/bin/bash
# H29: Prefill Bucketing Overestimates Short Sequences
#
# Tests whether the power-of-2 bucketing in rooflineStepTime (minimum bucket=512)
# causes attention core FLOPs to be overestimated for short prefill sequences,
# producing latency overestimates exceeding 2x for seqLen <= 100.
#
# Methodology:
#   - Sweep seqLen from 1 to 2048 through rooflineStepTime
#   - Compare step times within the same bucket (e.g., seqLen=50 vs seqLen=512)
#   - Isolate compute vs memory bound regime using extreme hardware configs
#   - Compute overestimation ratio relative to linear token-count scaling
#
# The experiment uses Go tests in sim/h29_bucketing_experiment_test.go which
# call rooflineStepTime directly (unexported function, accessible within package).
#
# Tests:
#   TestH29_PrefillBucketingSweep         - Full seqLen sweep with regime identification
#   TestH29_BucketBoundaryComparison      - Direct same-bucket comparisons
#   TestH29_AttentionVsGEMM_Decomposition - Component analysis within bucket 512
#   TestH29_OverestimationRatio           - Quantify overestimation vs linear expectation
#   TestH29_ComputeVsMemoryBound          - Regime analysis per seqLen
#
# Usage: ./run.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Create output directory
OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$OUTPUT_DIR"

echo "=== H29: Prefill Bucketing Overestimates Short Sequences ==="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run Go tests with output directed to CSV files
echo "Running H29 experiment tests..."
echo ""

H29_OUTPUT_DIR="$OUTPUT_DIR" go test -v -run "TestH29_" -count=1 "$REPO_ROOT/sim/..." 2>&1 | tee "$OUTPUT_DIR/test_output.log"

echo ""
echo "=== Test Output Files ==="
ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "  (no CSV files generated)"

echo ""
echo "=== Analysis ==="
python3 "$SCRIPT_DIR/analyze.py" "$OUTPUT_DIR"
