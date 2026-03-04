#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
mkdir -p "${OUTPUT_DIR}"

cd "${SCRIPT_DIR}/../../../../.."

echo "=== H2: FairBatching Cycle-Time Regression ==="

# Build the Go binary first
echo "Building simulation_worker..."
go build -o simulation_worker main.go 2>&1

python3 "${SCRIPT_DIR}/run_experiment.py" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/run.log"

echo "=== H2 Complete ==="
