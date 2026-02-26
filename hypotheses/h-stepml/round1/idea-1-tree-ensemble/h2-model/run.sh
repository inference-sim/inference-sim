#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHARED_DIR="$(cd "$SCRIPT_DIR/../../shared" && pwd)"
export PYTHONPATH="$SHARED_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$SCRIPT_DIR/output"

echo "=== Idea 1 h2-model: XGBoost with 30 Physics-Informed Features ===" >&2
echo "Step 1/2: Training XGBoost (per-experiment with hyperparameter search)..." >&2
python3 "$SCRIPT_DIR/train_xgboost.py" "$@"

echo "Step 2/2: Evaluating results..." >&2
python3 "$SCRIPT_DIR/analyze.py"

echo "=== Done ===" >&2
