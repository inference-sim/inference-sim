#!/bin/bash
# hypotheses/lib/harness.sh — Shared experiment harness
#
# Source this file at the top of every run.sh:
#   source "$(dirname "$0")/../lib/harness.sh"
#
# Provides:
#   setup_experiment [--rebuild]  — build binary, create temp dir
#   blis_run <timeout> <output_file> [--stderr <file>] [flags...]  — run with mandatory timeout
#   preflight_kv_check <total_blocks> <block_size> <max_input_tokens>  — warn if KV is dangerously low
#   TIMEOUT_QUICK, TIMEOUT_STANDARD, TIMEOUT_EXTENDED  — standard timeout constants
#   BINARY, MODEL, RESULTS_DIR  — standard variables
#
# NOTE: Named blis_run (not run_sim) because 15 existing experiments define
# their own run_sim() with incompatible signatures. Experiments can define
# a local run_sim() that calls blis_run internally.

# Standard timeout tiers (seconds)
TIMEOUT_QUICK=120      # calibration runs, <100 requests
TIMEOUT_STANDARD=300   # main experiment runs, 100-500 requests
TIMEOUT_EXTENDED=600   # stress tests, >500 requests or multi-turn

# Locate repo root relative to lib/
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HARNESS_DIR/../.." && pwd)"
BINARY="$REPO_ROOT/blis"
MODEL="meta-llama/llama-3.1-8b-instruct"
RESULTS_DIR=""

# setup_experiment [--rebuild]
# Builds the binary if needed and creates a temp directory.
# Sets RESULTS_DIR and registers cleanup trap.
setup_experiment() {
    if [[ "${1:-}" == "--rebuild" ]] || [[ ! -x "$BINARY" ]]; then
        echo "Building blis..." >&2
        (cd "$REPO_ROOT" && go build -o blis main.go)
    fi
    RESULTS_DIR=$(mktemp -d) || { echo "ERROR: mktemp failed" >&2; return 1; }
    trap 'rm -rf "$RESULTS_DIR"' EXIT
}

# blis_run <timeout_seconds> <output_file> [--stderr <stderr_file>] [blis flags...]
# Wraps ./blis run with a mandatory timeout.
# On timeout (exit 124): writes "TIMEOUT" to output_file, warns on stderr.
# On other failure: writes "ERROR:<exit_code>" to output_file, warns on stderr.
# Use --stderr <file> to capture stderr (for panic detection in robustness experiments).
# Without --stderr, stderr is discarded (2>/dev/null).
# Returns: 0 on success, 124 on timeout, other on error.
blis_run() {
    local timeout_secs="$1"
    local output_file="$2"
    shift 2

    # Validate timeout is a positive integer (timeout 0 means "no timeout", defeating the harness)
    if ! [[ "$timeout_secs" =~ ^[0-9]+$ ]] || [[ "$timeout_secs" -eq 0 ]]; then
        echo "ERROR: blis_run requires timeout_secs as positive integer, got '$timeout_secs'" >&2
        return 1
    fi

    # Check for optional --stderr flag
    local stderr_target="/dev/null"
    if [[ "${1:-}" == "--stderr" ]]; then
        if [[ -z "${2:-}" ]]; then
            echo "ERROR: --stderr requires a filename argument" >&2
            return 1
        fi
        stderr_target="$2"
        shift 2
    fi

    local exit_code=0
    timeout "$timeout_secs" "$BINARY" run "$@" > "$output_file" 2>"$stderr_target" || exit_code=$?

    if [[ $exit_code -eq 124 ]]; then
        echo "TIMEOUT" > "$output_file"
        echo "  TIMEOUT: simulation exceeded ${timeout_secs}s" >&2
    elif [[ $exit_code -ne 0 ]]; then
        echo "ERROR:${exit_code}" > "$output_file"
        echo "  ERROR: simulation exited with code $exit_code" >&2
    fi

    return $exit_code
}

# preflight_kv_check <total_blocks> <block_size> <max_input_tokens>
# Warns on stderr if KV blocks are dangerously low (below 4x minimum for one request).
# Always returns 0 — safe under set -euo pipefail. Purely advisory; experiments may
# intentionally test KV pressure. block_size <= 0 is a caller bug and returns 1.
preflight_kv_check() {
    local total_blocks=$1
    local block_size=$2
    local max_input=$3

    if [[ "$block_size" -le 0 ]] 2>/dev/null; then
        echo "ERROR: preflight_kv_check block_size must be > 0, got $block_size" >&2
        return 1
    fi

    local blocks_per_request=$(( (max_input + block_size - 1) / block_size ))
    local min_safe=$(( blocks_per_request * 4 ))

    if [[ "$total_blocks" -lt "$min_safe" ]]; then
        echo "WARNING: KV blocks ($total_blocks) < safe minimum ($min_safe)" >&2
        echo "  blocks_per_request=$blocks_per_request (ceil($max_input/$block_size)), need 4x headroom" >&2
        echo "  This may cause preemption cascades. Ensure blis_run has a timeout." >&2
    fi
    return 0
}

# is_timeout <output_file>
# Returns 0 if the output file indicates a timeout or error.
is_timeout() {
    local file="$1"
    [[ ! -s "$file" ]] || head -1 "$file" | grep -qE '^(TIMEOUT|ERROR:)'
}
