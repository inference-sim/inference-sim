#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../../lib/harness.sh"

setup_experiment "${1:-}"

# ============================================================
# H6: MFU Grid-Boundary Discontinuity Test
# ============================================================
#
# This experiment is purely simulator-internal (Part A of H6).
# It sweeps batch size and sequence length, calling the MFU
# lookup functions directly via a Go test program, and measures
# discontinuities in the returned MFU values.
#
# No ground truth data is needed — we are testing properties
# of the lookup algorithm itself.
#
# The experiment:
#   1. Sweeps GEMM MFU across batch sizes 1-256 (step 1)
#      for fixed (K=4096, N=6144) — Llama 3.1 8B Q-projection shape
#   2. Sweeps decode attention MFU across batch sizes 1-256 (step 1)
#      for fixed KV lengths (1024, 4096, 8192)
#   3. Sweeps prefill attention MFU across sequence lengths 512-32768 (step 64)
#   4. Counts discontinuities (>=5% jump between adjacent points)
#   5. Computes summary statistics
#
# Output: CSV files with (input_param, mfu) pairs for analysis
# ============================================================

echo "=== H6: MFU Grid-Boundary Discontinuity Test ===" >&2

# Build and run the Go test program
echo "Building MFU sweep test..." >&2
cd "$REPO_ROOT"

go run "$SCRIPT_DIR/mfu_sweep.go" \
    --bench-data "$REPO_ROOT/InferSim/bench_data" \
    --gpu h100 \
    --output-dir "$RESULTS_DIR"

echo "=== Analyzing results ===" >&2
python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR"
