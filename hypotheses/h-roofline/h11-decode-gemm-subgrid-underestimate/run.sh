#!/bin/bash
# H34: Decode GEMM Time Underestimate Below MFU Grid Minimum
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
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

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
echo "  H34: Decode GEMM Time Underestimate Below MFU Grid Minimum"
echo "=========================================="
echo ""
echo "Running Go tests: TestH34_*"
echo "  Models: Llama-3.1-8B + eval suite"
echo "  Hardware: H100 (testHardwareCalib)"
echo "  TP: 1"
echo "  Batch sizes: 1, 2, 4, 8, 16, 32"
echo ""

# --- Run all H34 experiments via Go test ---
cd "$REPO_ROOT"
go test ./sim/... \
    -run "TestH34_" \
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
