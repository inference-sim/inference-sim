#!/bin/bash
# Idea 1, H5: LOWO Cross-Validation (2-Regime Piecewise-Linear)
# Tests whether h1's piecewise-linear model generalizes across workloads.
# Usage: ./run.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"

echo "============================================================"
echo "  Idea 1, H5: LOWO Cross-Validation"
echo "============================================================"

cd "$SCRIPT_DIR/../../../shared"
pip3 install -q -r requirements.txt 2>/dev/null || true

echo ""
echo "Running LOWO cross-validation..."
python3 "$SCRIPT_DIR/lowo_cv.py" --output-dir "$OUTPUT_DIR"

echo ""
echo "Done. Results in: $OUTPUT_DIR"
