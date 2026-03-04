#!/bin/bash
# Idea 1, H2: Joint Bayesian Optimization of LatencyModel Methods
# Optimizes overhead + secondary method constants using BLIS E2E as objective.
# Requires h1 artifacts to exist first.
# Usage: ./run.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"
H1_ARTIFACT_DIR="$SCRIPT_DIR/../h1-piecewise-steptime/output/artifacts"

echo "============================================================"
echo "  Idea 1, H2: Joint Bayesian Optimization"
echo "============================================================"

if [ ! -d "$H1_ARTIFACT_DIR" ]; then
    echo "ERROR: H1 artifacts not found at $H1_ARTIFACT_DIR"
    echo "Run h1-piecewise-steptime first."
    exit 1
fi

cd "$SCRIPT_DIR/../../../shared"
pip3 install -q -r requirements.txt 2>/dev/null || true
pip3 install -q scikit-optimize 2>/dev/null || true

echo ""
echo "Running Bayesian optimization (this may take hours)..."
python3 "$SCRIPT_DIR/bo_calibrate.py" \
    --h1-artifact-dir "$H1_ARTIFACT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --max-evals 200

echo ""
echo "Done. Results in: $OUTPUT_DIR"
