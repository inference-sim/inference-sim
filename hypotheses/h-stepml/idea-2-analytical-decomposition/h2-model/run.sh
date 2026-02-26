#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHARED_DIR="$(cd "$SCRIPT_DIR/../../shared" && pwd)"
export PYTHONPATH="$SHARED_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$SCRIPT_DIR/output"

echo "=== Idea 2 h2-model: Learned Correction Factors ===" >&2
echo "Step 1/2: Fitting correction factors (9, 16, 36 parameters)..." >&2
python3 "$SCRIPT_DIR/fit_corrections.py" "$@"

echo "Step 2/2: Evaluating results..." >&2
python3 "$SCRIPT_DIR/analyze.py"

echo "=== Done ===" >&2
