#!/bin/bash
# Idea 1, H4: LOMO Cross-Validation (2-Regime Piecewise-Linear)
# Tests whether h1's piecewise-linear model generalizes across models.
# Usage: ./run.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"

echo "============================================================"
echo "  Idea 1, H4: LOMO Cross-Validation"
echo "============================================================"

cd "$SCRIPT_DIR/../../../shared"
pip3 install -q -r requirements.txt 2>/dev/null || true

echo ""
echo "Running LOMO cross-validation..."
python3 "$SCRIPT_DIR/lomo_cv.py" --output-dir "$OUTPUT_DIR"

echo ""
echo "Done. Results in: $OUTPUT_DIR"
