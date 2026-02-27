#!/bin/bash
# H18: Decode Activation Memory Factor Is Inconsequential (formerly H35)
#
# Hypothesis: The decode activation memory factor (0.75) is inconsequential
# because activation bytes constitute less than 0.5% of total memory traffic
# across all evaluation operating points (bs=1..256, kvLen=128..8192), so
# replacing 0.75 with any value in [0.5, 1.5] changes predicted step time
# by less than 0.05%.
#
# Family: Structural model
# VV&UQ: Validation
# Type: Deterministic (no RNG, pure roofline computation)
#
# Independent variable: activation memory discount factor [0.50, 0.75, 1.00, 1.50]
# Controlled variables: Llama-3.1-8B model config, H100 hardware, TP=1
# Dependent variable: activation fraction of dynamic bytes, decode step time delta
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
echo "  H18: Decode Activation Memory Factor"
echo "        Is Inconsequential"
echo "=========================================="
echo ""
echo "Running Go test: TestH18_DecodeActivationDiscountNegligible"
echo "  Model: Llama-3.1-8B (testModelConfig)"
echo "  Hardware: H100 (testHardwareCalib)"
echo "  TP: 1"
echo "  Batch sizes: 1, 4, 8, 16, 32, 64, 128, 256"
echo "  KV lengths: 128, 256, 512, 1024, 2048, 4096, 8192"
echo "  Activation factors: 0.50, 0.75, 1.00, 1.50"
echo ""

# --- Copy test file into sim/latency/ for access to unexported functions ---
# Strip //go:build ignore (test file already declares package latency).
TEST_SRC="$SCRIPT_DIR/../h18-decode-activation-discount/h18_decode_activation_discount_test.go"
TEST_DST="$REPO_ROOT/sim/latency/h18_decode_activation_discount_test.go"

cleanup() {
    rm -f "$TEST_DST"
}
trap cleanup EXIT

grep -v '^//go:build ignore$' "$TEST_SRC" > "$TEST_DST"

# --- Run the experiment via Go test ---
# The test outputs structured data to stdout; stderr has test logs.
cd "$REPO_ROOT"
go test ./sim/latency/... \
    -run TestH18_DecodeActivationDiscountNegligible \
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
