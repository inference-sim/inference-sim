#!/bin/bash
# H28: Decode Attention maxKVLen Overestimation
#
# Hypothesis: In the roofline model's decode phase, attention compute time
# overestimates true per-request-summed attention FLOPs by a factor of
# maxKVLen/meanKVLen for heterogeneous batches. The total decode compute time
# grows superlinearly when adding short-KV requests to a batch containing one
# long-KV request.
#
# Family: Structural model
# VV&UQ: Validation
# Type: Deterministic (no RNG, pure roofline computation)
#
# Independent variable: batch composition — vary number of short-KV (KV=64)
#   decode requests added to a batch anchored by one long-KV (KV=4096) request
# Controlled variables: Llama-3.1-8B model config, H100 hardware, TP=1
# Dependent variable: total decode step time, attention FLOPs, marginal cost
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
echo "  H28: Decode Attention maxKVLen Overestimation"
echo "=========================================="
echo ""
echo "Running Go test: TestH28DecodeMaxKVLenExperiment"
echo "  Model: Llama-3.1-8B (testModelConfig)"
echo "  Hardware: H100 (testHardwareCalib)"
echo "  TP: 1"
echo "  Anchor KV: 4096, Short KV: 64"
echo "  Max added short requests: 15"
echo ""

# --- Run the experiment via Go test ---
# The test outputs CSV to stdout; stderr has test logs and warnings.
cd "$REPO_ROOT"
go test ./sim/... \
    -run TestH28DecodeMaxKVLenExperiment \
    -v \
    -count=1 \
    2>"$OUTPUT_DIR/test_stderr.log" \
    | tee "$OUTPUT_DIR/raw_output.txt"

# --- Extract CSV data (lines not starting with '=' or '---' or '    ' or 'ok') ---
# The CSV data has two sections: main data and attention FLOPs comparison
grep -E '^(batch_type|heterogeneous|homogeneous)' "$OUTPUT_DIR/raw_output.txt" > "$OUTPUT_DIR/step_times.csv"
grep -E '^# [0-9]+,' "$OUTPUT_DIR/raw_output.txt" | sed 's/^# //' > "$OUTPUT_DIR/attn_flops_comparison.csv"

# Add header to attn_flops_comparison
{
    echo "batch_size,roofline_attn_flops,ideal_attn_flops,overestimation_factor"
    cat "$OUTPUT_DIR/attn_flops_comparison.csv"
} > "$OUTPUT_DIR/attn_flops_comparison_tmp.csv"
mv "$OUTPUT_DIR/attn_flops_comparison_tmp.csv" "$OUTPUT_DIR/attn_flops_comparison.csv"

echo ""
echo "=========================================="
echo "  Output files:"
echo "    $OUTPUT_DIR/step_times.csv"
echo "    $OUTPUT_DIR/attn_flops_comparison.csv"
echo "    $OUTPUT_DIR/test_stderr.log"
echo "=========================================="
echo ""

# --- Run analysis ---
echo "Running analysis..."
python3 "$SCRIPT_DIR/analyze.py" "$OUTPUT_DIR"
