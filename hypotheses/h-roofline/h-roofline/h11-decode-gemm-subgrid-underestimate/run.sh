#!/bin/bash
# H17: Decode GEMM Time Underestimate Below MFU Grid Minimum (formerly H34)
#
# Hypothesis: For decode steps at batch sizes 1-7, computeTransformerGEMMTimes
# predicts GEMM time proportional to batch size (ratio bs/8), whereas actual
# GPU GEMM kernels at these small M values exhibit a near-constant memory-bound
# latency floor, causing up to 8x underestimate at bs=1.
#
# Family: Structural model
# VV&UQ: Validation
# Type: Deterministic (no RNG, pure roofline computation)
#
# Independent variable: batch size (1, 2, 4, 8, 16, 32)
# Controlled variables: model config, hardware config, TP=1
# Dependent variable: GEMM time, ratio relative to bs=8
#
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# --- Build if needed ---
if [[ "${1:-}" == "--rebuild" ]]; then
    echo "Building simulation_worker..." >&2
    (cd "$REPO_ROOT" && go build -o simulation_worker main.go)
fi

# --- Precondition: bench_data must exist (needed for MFU lookups) ---
BENCH_DATA="$REPO_ROOT/bench_data"
if [[ ! -d "$BENCH_DATA" ]]; then
    echo "ERROR: bench_data directory missing: $BENCH_DATA" >&2
    echo "MFU database is required for roofline GEMM time computation." >&2
    exit 1
fi

# --- Output directory ---
OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "  H17: Decode GEMM Time Underestimate Below MFU Grid Minimum"
echo "=========================================="
echo ""
echo "Running Go tests: TestH17_*"
echo "  Models: Llama-3.1-8B + eval suite"
echo "  Hardware: H100 (testHardwareCalib)"
echo "  TP: 1"
echo "  Batch sizes: 1, 2, 4, 8, 16, 32"
echo ""

# --- Copy test files into sim/latency/ for access to unexported functions ---
# Strip //go:build ignore (test files already declare package latency).
# H17 depends on evalSuiteModels() and modelSpec from H15, so copy both.
TEST_SRC="$SCRIPT_DIR/../h17-decode-gemm-subgrid/h17_decode_gemm_subgrid_test.go"
TEST_DST="$REPO_ROOT/sim/latency/h17_decode_gemm_subgrid_test.go"
DEPS_SRC="$SCRIPT_DIR/../h15-missing-lmhead/h15_missing_lmhead_test.go"
DEPS_DST="$REPO_ROOT/sim/latency/h15_missing_lmhead_test.go"

cleanup() {
    rm -f "$TEST_DST" "$DEPS_DST"
}
trap cleanup EXIT

grep -v '^//go:build ignore$' "$TEST_SRC" > "$TEST_DST"
grep -v '^//go:build ignore$' "$DEPS_SRC" > "$DEPS_DST"

# --- Run all H17 experiments via Go test ---
cd "$REPO_ROOT"
go test ./sim/latency/... \
    -run "TestH17_" \
    -v \
    -count=1 \
    2>"$OUTPUT_DIR/test_stderr.log" \
    | tee "$OUTPUT_DIR/raw_output.txt"

echo ""
echo "=========================================="
echo "  Output files:"
echo "    $OUTPUT_DIR/raw_output.txt"
echo "    $OUTPUT_DIR/test_stderr.log"
echo "=========================================="
echo ""

# --- Run analysis ---
echo "Running analysis..."
python3 "$SCRIPT_DIR/analyze.py" "$OUTPUT_DIR"
