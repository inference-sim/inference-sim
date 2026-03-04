#!/usr/bin/env bash
set -euo pipefail

# H1: Cycle-Time Extraction from Lifecycle Data
# =============================================
# Extract per-step cycle times (inter-token intervals) from lifecycle data
# and correlate with step.duration_us to validate the cycle-time concept.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
mkdir -p "${OUTPUT_DIR}"

cd "${SCRIPT_DIR}/../../../../.."  # repo root

echo "=== H1: Cycle-Time Extraction ==="
echo "Output directory: ${OUTPUT_DIR}"

python3 "${SCRIPT_DIR}/extract_cycle_times.py" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/run.log"

echo "=== H1 Complete ==="
