#!/bin/bash
# Idea 2, H1: Regime-Specific Ridge Regression with KV Features
# Trains 3-regime Ridge models per model, evaluates per-step MAPE,
# exports StepML artifacts for BLIS validation.
# Usage: ./run.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"

echo "============================================================"
echo "  Idea 2, H1: Regime-Specific Ridge with KV Features"
echo "============================================================"

# Ensure we're in the shared directory for imports
cd "$SCRIPT_DIR/../../../shared"

# Install requirements if needed
pip3 install -q -r requirements.txt 2>/dev/null || true

echo ""
echo "Training regime-specific Ridge models..."
python3 "$SCRIPT_DIR/train_regime_ridge.py" --output-dir "$OUTPUT_DIR"

echo ""
echo "Done. Results in: $OUTPUT_DIR"
echo "Artifacts in: $OUTPUT_DIR/artifacts/"
