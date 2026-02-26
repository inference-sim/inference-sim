#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHARED_DIR="$(cd "$SCRIPT_DIR/../../shared" && pwd)"
export PYTHONPATH="$SHARED_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$SCRIPT_DIR/output"

echo "=== Idea 1 h3-generalization: LOMO + LOWO Cross-Validation ===" >&2
echo "Step 1/2: Running cross-validation experiments..." >&2
python3 "$SCRIPT_DIR/cross_validate.py" "$@"

echo "Step 2/2: Analyzing results..." >&2
python3 "$SCRIPT_DIR/analyze.py"

echo "=== Done ===" >&2
