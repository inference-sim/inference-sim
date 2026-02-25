#!/bin/bash
# H33: Decode Attention MFU Shape Mismatch
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
echo "  H33: Decode Attention MFU Shape Mismatch"
echo "=========================================="
echo ""
echo "Running Go tests: TestH33_*"
echo "  Model: Llama-3.1-8B (testModelConfig)"
echo "  Hardware: H100 (testHardwareCalib)"
echo "  TP: 1"
echo "  Scenarios: 15 batch compositions (homo, mild, moderate, high, extreme, pathological)"
echo ""

# --- Run the experiment via Go test ---
cd "$REPO_ROOT"
go test ./sim/... \
    -run "TestH33_" \
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
