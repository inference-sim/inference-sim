#!/bin/bash
# H16: Decode Attention MFU Shape Mismatch (formerly H33)
#
# Hypothesis: In heterogeneous decode batches, using maxKVLen for the attention
# MFU lookup while using per-request actual KV lengths for FLOPs systematically
# underestimates decode attention time, because the MFU at maxKVLen is higher
# than the effective per-request MFU at shorter KV lengths.
#
# Family: Structural model
# VV&UQ: Validation
# Type: Deterministic (no RNG, pure roofline computation)
#
# Independent variable: batch composition (KV length heterogeneity)
# Controlled variables: Llama-3.1-8B model config, H100 hardware, TP=1
# Dependent variable: ratio of current method / per-request method attention time
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
    echo "MFU database is required for roofline step time computation." >&2
    exit 1
fi

# --- Output directory ---
OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "  H16: Decode Attention MFU Shape Mismatch"
echo "=========================================="
echo ""
echo "Running Go tests: TestH16_*"
echo "  Model: Llama-3.1-8B (testModelConfig)"
echo "  Hardware: H100 (testHardwareCalib)"
echo "  TP: 1"
echo "  Scenarios: 15 batch compositions (homo, mild, moderate, high, extreme, pathological)"
echo ""

# --- Copy test file into sim/latency/ for access to unexported functions ---
# Strip //go:build ignore (test file already declares package latency).
TEST_SRC="$SCRIPT_DIR/../h16-decode-mfu-shape-mismatch/h16_decode_mfu_shape_mismatch_test.go"
TEST_DST="$REPO_ROOT/sim/latency/h16_decode_mfu_shape_mismatch_test.go"

cleanup() {
    rm -f "$TEST_DST"
}
trap cleanup EXIT

grep -v '^//go:build ignore$' "$TEST_SRC" > "$TEST_DST"

# --- Run the experiment via Go test ---
cd "$REPO_ROOT"
go test ./sim/latency/... \
    -run "TestH16_" \
    -v \
    -count=1 \
    2>"$OUTPUT_DIR/test_stderr.log" \
    | tee "$OUTPUT_DIR/test_output.txt"

echo ""
echo "=========================================="
echo "  Output files:"
echo "    $OUTPUT_DIR/test_output.txt"
echo "    $OUTPUT_DIR/test_stderr.log"
echo "=========================================="
echo ""

# --- Run analysis ---
echo "Running analysis..."
python3 "$SCRIPT_DIR/analyze.py" "$OUTPUT_DIR"
