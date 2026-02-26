#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHARED_DIR="$(cd "$SCRIPT_DIR/../../shared" && pwd)"
export PYTHONPATH="$SHARED_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$SCRIPT_DIR/output"

echo "=== Idea 2 h1-features: Analytical Component Decomposition ===" >&2
echo "Step 1/2: Computing analytical FLOPs components..." >&2
python3 "$SCRIPT_DIR/compute_components.py" "$@"

echo "Step 2/2: Analyzing correlations..." >&2
python3 "$SCRIPT_DIR/analyze.py"

echo "=== Done ===" >&2
