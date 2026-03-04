#!/bin/bash
# Idea 1, H1: Piecewise-Linear StepTime with 2 Regimes + KV Features
# Trains per-model piecewise-linear models (decode-only vs mixed-batch),
# evaluates per-step MAPE, exports StepML artifacts.
# Usage: ./run.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"

echo "============================================================"
echo "  Idea 1, H1: Piecewise-Linear StepTime (2 Regimes)"
echo "============================================================"

cd "$SCRIPT_DIR/../../../shared"
pip3 install -q -r requirements.txt 2>/dev/null || true

echo ""
echo "Training piecewise-linear models..."
python3 "$SCRIPT_DIR/train_piecewise.py" --output-dir "$OUTPUT_DIR"

echo ""
echo "Done. Results in: $OUTPUT_DIR"
echo "Artifacts in: $OUTPUT_DIR/artifacts/"
