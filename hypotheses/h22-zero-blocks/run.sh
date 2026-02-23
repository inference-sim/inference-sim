#!/bin/bash
# H22: Zero KV Blocks -- CLI Validation Boundary
#
# Hypothesis: Running with --total-kv-blocks 0 (or other zero/negative KV
# configs) should produce a clean CLI error (logrus.Fatalf), not a panic
# or stack trace from sim/.
#
# Tests R3 (validate CLI flags) and R6 (no Fatalf in library).
#
# Usage: ./run.sh [--rebuild]
#
# Requires: Go 1.24+, Python 3

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$REPO_ROOT/simulation_worker"

# Build if needed
if [[ "${1:-}" == "--rebuild" ]] || [[ ! -x "$BINARY" ]]; then
    echo "Building simulation_worker..."
    (cd "$REPO_ROOT" && go build -o simulation_worker main.go)
fi

MODEL="meta-llama/llama-3.1-8b-instruct"
RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

echo "============================================================================"
echo "  H22: Zero KV Blocks -- CLI Validation Boundary"
echo "  Tests: R3 (validate CLI flags), R6 (no Fatalf in library)"
echo "============================================================================"
echo ""

# ── Helper: run a test case and capture exit code + stderr ──────────────────

run_test_case() {
    local name="$1"
    shift
    local stderr_file="$RESULTS_DIR/${name}.stderr"
    local exit_code_file="$RESULTS_DIR/${name}.exit_code"
    local exit_code=0

    echo "  Running: $name"

    # Run the binary with the given flags, capturing stderr
    "$BINARY" run --model "$MODEL" --num-requests 5 --rate 100 "$@" \
        > /dev/null 2>"$stderr_file" || exit_code=$?

    echo "$exit_code" > "$exit_code_file"
    echo "    Exit code: $exit_code"

    # Quick summary on stdout
    # logrus uses "FATA[" in TTY mode, "level=fatal" when redirected to file
    if grep -qE 'goroutine [0-9]+|^panic\(|runtime error' "$stderr_file" 2>/dev/null; then
        echo "    PANIC DETECTED in stderr"
    elif grep -qE 'FATA\[|level=fatal' "$stderr_file" 2>/dev/null; then
        local msg
        msg=$(grep -E 'FATA\[|level=fatal' "$stderr_file" 2>/dev/null | head -1)
        echo "    Fatalf: $msg"
    elif [ "$exit_code" -eq 0 ]; then
        echo "    Completed successfully"
    else
        echo "    Failed (non-zero exit, no fatalf, no panic)"
    fi
    echo ""
}

# ── Test Case 1: --total-kv-blocks 0 ────────────────────────────────────────

echo "Test 1: --total-kv-blocks 0 (zero GPU blocks)"
run_test_case "zero_kv_blocks" --total-kv-blocks 0

# ── Test Case 2: --block-size-in-tokens 0 ───────────────────────────────────

echo "Test 2: --block-size-in-tokens 0 (zero block size)"
run_test_case "zero_block_size" --block-size-in-tokens 0

# ── Test Case 3: --total-kv-blocks -1 ───────────────────────────────────────

echo "Test 3: --total-kv-blocks -1 (negative blocks)"
run_test_case "negative_kv_blocks" --total-kv-blocks -1

# ── Test Case 4: --total-kv-blocks 0 --kv-cpu-blocks 100 ───────────────────

echo "Test 4: --total-kv-blocks 0 --kv-cpu-blocks 100 (zero GPU + CPU tier)"
run_test_case "zero_gpu_with_cpu" --total-kv-blocks 0 --kv-cpu-blocks 100

# ── Test Case 5: --kv-cpu-blocks -1 (negative CPU blocks) ──────────────────

echo "Test 5: --kv-cpu-blocks -1 (negative CPU blocks)"
run_test_case "negative_cpu_blocks" --kv-cpu-blocks -1

# ── Control: Valid configuration ────────────────────────────────────────────

echo "Control: Valid configuration (should succeed)"
run_test_case "valid_control" --total-kv-blocks 2048 --block-size-in-tokens 16

echo "============================================================================"
echo "  Running analysis..."
echo "============================================================================"
echo ""

# ── Analysis ────────────────────────────────────────────────────────────────

python3 "$SCRIPT_DIR/analyze.py" "$RESULTS_DIR"
