#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHARED_DIR="$(cd "$SCRIPT_DIR/../../shared" && pwd)"
export PYTHONPATH="$SHARED_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$SCRIPT_DIR/output"

echo "=== Idea 1 h1-features: Physics-Informed Feature Set + Ridge ===" >&2
echo "Step 1/3: Engineering 30 features..." >&2
python3 "$SCRIPT_DIR/engineer_features.py" "$@"

echo "Step 2/3: Training Ridge regression..." >&2
python3 "$SCRIPT_DIR/train_ridge.py"

echo "Step 3/3: Evaluating results..." >&2
python3 "$SCRIPT_DIR/analyze.py"

echo "=== Done ===" >&2
