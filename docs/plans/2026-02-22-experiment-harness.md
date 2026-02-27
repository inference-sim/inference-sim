# Experiment Harness & Safety Gates Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a shared shell library that every hypothesis `run.sh` sources, providing mandatory timeout wrapping, KV parameter safety checks, and timeout-aware analysis — then update the process doc with new pre-execution safety gates.

**The problem today:** 13 of 23 existing `run.sh` scripts have no timeout protection. When the simulator enters a KV preemption cascade (known issue #373, #349), the simulation runs indefinitely — consuming 5.6 GB RAM and 100% CPU (H25 Config C). Agent sessions burn hours waiting on stuck runs. There is no standard timeout convention (existing scripts use 60s to 600s ad hoc), no pre-flight parameter validation, and no systematic way for `analyze.py` to detect and report timeouts.

**What this PR adds:**
1. **Shared harness** (`hypotheses/lib/harness.sh`) — a shell library providing `blis_run()` (mandatory timeout wrapper with optional stderr capture), `preflight_kv_check()` (parameter safety validation), and standard timeout constants. Every future `run.sh` sources this file instead of reinventing these safeguards. Named `blis_run` (not `run_sim`) because 15 existing experiments define their own `run_sim()` with incompatible signatures — the harness must not shadow them.
2. **Timeout-aware analysis helper** (`hypotheses/lib/analyze_helpers.py`) — a Python module with `check_for_timeout()` that detects exit code 124 sentinel files and `parse_blis_output()` that handles missing/partial output gracefully. Every future `analyze.py` imports this.
3. **Updated process doc** (`docs/process/hypothesis.md`) — adds three new pre-execution safety gate checklist items covering mandatory timeouts, KV safety pre-flight, and timeout handling in analyzers.
4. **Updated template** (`docs/templates/hypothesis.md`) — run.sh template sources the harness; analyze.py template imports the helpers.

**Why this matters:** This is prerequisite infrastructure for running the next batch of 4 parallel hypotheses (H24, H19, H16, H21). Without it, any experiment touching KV-constrained configs risks hanging the entire agent session.

**Architecture:** Two new files in `hypotheses/lib/` (shell library + Python module). Two doc edits (process + template). No Go code changes. The harness uses `timeout` (coreutils) for wall-clock limits and arithmetic for KV safety checks.

**Source:** Discussion in this session about experiment hangs; evidence from H8, H25, H-Overload-KV experiments.

**Closes:** N/A — no linked issues.

**Behavioral Contracts:**

- **BC-1**: `blis_run` wraps every simulation call with `timeout`. If the simulation exceeds the timeout, exit code 124 is captured and a `TIMEOUT` sentinel is written. Supports optional stderr capture via `--stderr <file>` for robustness experiments that need panic detection.
- **BC-2**: `preflight_kv_check` warns on stderr when `total_kv_blocks < 4 * ceil(max_input_tokens / block_size)`. It does NOT abort — the experiment may intentionally test KV pressure.
- **BC-3**: `check_for_timeout` in analyze_helpers.py returns True when a result file contains `TIMEOUT` or is empty, and prints a warning to stderr.
- **BC-4**: `parse_blis_output` returns a dict with default zeros when the output file is empty or contains no cluster JSON block, with a warning to stderr. Includes conservation fields (`still_queued`, `still_running`, `injected`) and `preemption_count` used by 5+ existing experiments.
- **BC-5**: The hypothesis.md pre-execution gates include timeout, KV safety, and analyzer timeout handling as mandatory checklist items.

**Naming decision:** The harness function is `blis_run`, NOT `run_sim`. 15 existing experiments define `run_sim()` with incompatible signatures (positional params for seed, routing, label, etc.). Experiments that source the harness can define their own `run_sim()` that calls `blis_run` internally — the harness provides the building block, not the final wrapper.

**Scope note:** This harness is for NEW experiments only. Existing 23 experiments are NOT migrated — they continue working as-is. The template update ensures all future experiments use the harness from the start.

---

## Task 1: Create `hypotheses/lib/harness.sh`

**Files:**
- Create: `hypotheses/lib/harness.sh`

**Step 0: Create the lib directory**

Run: `mkdir -p hypotheses/lib`

**Step 1: Write the harness shell library**

```bash
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
    RESULTS_DIR=$(mktemp -d)
    trap "rm -rf $RESULTS_DIR" EXIT
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

    # Check for optional --stderr flag
    local stderr_target="/dev/null"
    if [[ "${1:-}" == "--stderr" ]]; then
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
# Warns if KV blocks are dangerously low (below 4x minimum for a single request).
# Does NOT abort — experiments may intentionally test KV pressure.
preflight_kv_check() {
    local total_blocks=$1
    local block_size=$2
    local max_input=$3
    local blocks_per_request=$(( (max_input + block_size - 1) / block_size ))
    local min_safe=$(( blocks_per_request * 4 ))

    if [[ "$total_blocks" -lt "$min_safe" ]]; then
        echo "WARNING: KV blocks ($total_blocks) < safe minimum ($min_safe)" >&2
        echo "  blocks_per_request=$blocks_per_request (ceil($max_input/$block_size)), need 4x headroom" >&2
        echo "  This may cause preemption cascades. Ensure blis_run has a timeout." >&2
        return 1
    fi
    return 0
}

# is_timeout <output_file>
# Returns 0 if the output file indicates a timeout or error.
is_timeout() {
    local file="$1"
    [[ ! -s "$file" ]] || head -1 "$file" | grep -qE '^(TIMEOUT|ERROR:)'
}
```

**Step 2: Verify the file is syntactically valid**

Run: `bash -n .worktrees/experiment-harness/hypotheses/lib/harness.sh`
Expected: No output (no syntax errors)

**Step 3: Commit**

```bash
git add hypotheses/lib/harness.sh
git commit -m "feat(hypotheses): add shared experiment harness with timeout and KV safety"
```

---

## Task 2: Create `hypotheses/lib/analyze_helpers.py`

**Files:**
- Create: `hypotheses/lib/analyze_helpers.py`

**Step 1: Write the Python analysis helper module**

```python
#!/usr/bin/env python3
"""Shared analysis helpers for hypothesis experiments.

Import in analyze.py:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
    from analyze_helpers import parse_blis_output, check_for_timeout

Provides:
    parse_blis_output(filepath) -> dict  — parse BLIS output with timeout/error handling
    check_for_timeout(filepath) -> bool  — detect timeout/error sentinel files
"""

import json
import re
import sys
from pathlib import Path


def check_for_timeout(filepath):
    """Check if an output file indicates a timeout or error.

    Returns True if the file is empty, missing, or starts with TIMEOUT/ERROR.
    Prints a warning to stderr when a timeout/error is detected.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"WARNING: output file missing: {filepath}", file=sys.stderr)
        return True
    content = path.read_text().strip()
    if not content:
        print(f"WARNING: output file empty: {filepath}", file=sys.stderr)
        return True
    first_line = content.split("\n")[0]
    if first_line.startswith("TIMEOUT"):
        print(f"WARNING: simulation timed out: {filepath}", file=sys.stderr)
        return True
    if first_line.startswith("ERROR:"):
        print(f"WARNING: simulation error ({first_line}): {filepath}", file=sys.stderr)
        return True
    return False


def parse_blis_output(filepath):
    """Parse BLIS CLI output into a metrics dict.

    Handles timeout/error/empty files gracefully by returning default zeros
    with a 'timed_out' flag.

    Output format reference:
        - cmd/root.go: text summary + JSON blocks per instance + cluster
        - sim/metrics_utils.go: MetricsOutput JSON struct
        - Cluster JSON: "instance_id": "cluster"
        - KV summary: "Preemption Rate: 0.1750", "Cache Hit Rate: 0.0452"
        - Trace summary: "Rejected Requests: N", "Target Distribution:"
    """
    defaults = {
        "ttft_mean": 0, "ttft_p99": 0,
        "e2e_mean": 0, "e2e_p99": 0,
        "throughput": 0, "completed": 0,
        "preemption_rate": 0.0, "cache_hit_rate": 0.0,
        "rejected": 0,
        # Conservation fields (used by h8, h12, h25, h-overload-kv)
        "injected": 0, "still_queued": 0, "still_running": 0,
        "preemption_count": 0,
        "timed_out": False,
    }

    if check_for_timeout(filepath):
        defaults["timed_out"] = True
        return defaults

    content = Path(filepath).read_text()

    # Extract cluster-level JSON block
    cluster = None
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        try:
            block = json.loads(match.group(1))
            if block.get("instance_id") == "cluster":
                cluster = block
        except json.JSONDecodeError:
            continue

    if cluster is None:
        print(f"WARNING: no cluster JSON block found in {filepath}", file=sys.stderr)
        defaults["timed_out"] = True
        return defaults

    # KV summary metrics (from cmd/root.go text output)
    preemption_rate = 0.0
    m = re.search(r"Preemption Rate: ([0-9.]+)", content)
    if m:
        preemption_rate = float(m.group(1))

    cache_hit_rate = 0.0
    m = re.search(r"Cache Hit Rate: ([0-9.]+)", content)
    if m:
        cache_hit_rate = float(m.group(1))

    rejected = 0
    m = re.search(r"Rejected Requests: (\d+)", content)
    if m:
        rejected = int(m.group(1))

    return {
        "ttft_mean": cluster.get("ttft_mean_ms", 0),
        "ttft_p99": cluster.get("ttft_p99_ms", 0),
        "e2e_mean": cluster.get("e2e_mean_ms", 0),
        "e2e_p99": cluster.get("e2e_p99_ms", 0),
        "throughput": cluster.get("responses_per_sec", 0),
        "completed": cluster.get("completed_requests", 0),
        "preemption_rate": preemption_rate,
        "cache_hit_rate": cache_hit_rate,
        "rejected": rejected,
        # Conservation fields
        "injected": cluster.get("injected_requests", 0),
        "still_queued": cluster.get("still_queued", 0),
        "still_running": cluster.get("still_running", 0),
        "preemption_count": cluster.get("preemption_count", 0),
        "timed_out": False,
    }
```

**Step 2: Verify Python syntax**

Run: `python3 -c "import ast; ast.parse(open('.worktrees/experiment-harness/hypotheses/lib/analyze_helpers.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add hypotheses/lib/analyze_helpers.py
git commit -m "feat(hypotheses): add shared Python analysis helpers with timeout detection"
```

---

## Task 3: Update `docs/process/hypothesis.md` pre-execution gates

**Files:**
- Modify: `docs/process/hypothesis.md:280-285` (Pre-Execution Gates section)

**Step 1: Add three new safety gate items to the Pre-Execution Gates checklist**

Find the existing Pre-Execution Gates section (lines 280-285):

```markdown
### Pre-Execution Gates (check BEFORE running experiments)
- [ ] `run.sh` flags verified against `cmd/root.go` help text
- [ ] `analyze.py` regexes verified against actual output format strings in `cmd/root.go` and `sim/metrics_utils.go`
- [ ] Workload YAML field names verified against `sim/workload/spec.go` struct tags
- [ ] Config diff against referenced experiments documented (ED-6)
- [ ] Code review completed (at least one pass)
```

Replace with:

```markdown
### Pre-Execution Gates (check BEFORE running experiments)
- [ ] `run.sh` sources `hypotheses/lib/harness.sh` and uses `blis_run` for every simulation call
- [ ] Every `blis_run` call has an appropriate timeout tier (`TIMEOUT_QUICK`/`TIMEOUT_STANDARD`/`TIMEOUT_EXTENDED`)
- [ ] KV safety pre-flight: if experiment uses `--total-kv-blocks`, call `preflight_kv_check` with max expected input tokens
- [ ] `analyze.py` imports `analyze_helpers` and uses `parse_blis_output` (handles timeouts gracefully)
- [ ] `run.sh` flags verified against `cmd/root.go` help text
- [ ] `analyze.py` regexes verified against actual output format strings in `cmd/root.go` and `sim/metrics_utils.go`
- [ ] Workload YAML field names verified against `sim/workload/spec.go` struct tags
- [ ] Config diff against referenced experiments documented (ED-6)
- [ ] Code review completed (at least one pass)
```

**Step 2: Verify the edit is correct**

Read the modified file and confirm the checklist has 9 items (4 new + 5 existing).

**Step 3: Commit**

```bash
git add docs/process/hypothesis.md
git commit -m "docs(process): add timeout and KV safety pre-execution gates to hypothesis workflow"
```

---

## Task 4: Update `docs/templates/hypothesis.md` run.sh and analyze.py templates

**Files:**
- Modify: `docs/templates/hypothesis.md:100-134` (run.sh template) and `docs/templates/hypothesis.md:136-195` (analyze.py template)

**Step 1: Update the run.sh template to source the harness**

Find the existing run.sh template (lines 100-134) and replace it with:

```bash
#!/bin/bash
# <Hypothesis name>
# <One-line description>
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"

setup_experiment "${1:-}"

# -- Experiment sections -----------------------------------------------
# Each experiment: use blis_run with appropriate timeout tier.
# NOTE: blis_run (not run_sim) — define your own run_sim() wrapper if needed.
#
# Example (basic):
#   blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/config_a.txt" \
#       --model "$MODEL" --num-instances 4 --seed 42 \
#       --workload-spec "$WORKLOAD_YAML" --log error
#
# Example (with stderr capture for robustness experiments):
#   blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/config_a.txt" \
#       --stderr "$RESULTS_DIR/config_a_stderr.txt" \
#       --model "$MODEL" --num-instances 4 --seed 42 --log error
#
# Example (with per-request JSON):
#   blis_run $TIMEOUT_STANDARD "$RESULTS_DIR/config_a.txt" \
#       --model "$MODEL" --num-instances 4 --seed 42 --log error \
#       --results-path "$RESULTS_DIR/config_a_results.json"
#
# For KV-constrained experiments, add pre-flight check:
#   preflight_kv_check 800 16 512  # total_blocks, block_size, max_input
# ----------------------------------------------------------------------
```

**Step 2: Update the analyze.py template to import helpers**

Find the existing analyze.py template (lines 136-195) and replace it with:

```python
#!/usr/bin/env python3
"""Analysis script for <hypothesis name>.

Parses BLIS multi-block output and produces comparison tables.
"""
import sys
from pathlib import Path

# Import shared helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output, check_for_timeout

# -- Analysis code --------------------------------------------------------
# Use parse_blis_output(filepath) to get metrics dict.
# The dict includes a 'timed_out' flag — check it before computing ratios.
#
# Example:
#   metrics = parse_blis_output(sys.argv[1])
#   if metrics["timed_out"]:
#       print(f"  SKIPPED (timeout)", file=sys.stderr)
#   else:
#       print(f"  TTFT mean: {metrics['ttft_mean']:.2f} ms")
# -------------------------------------------------------------------------
```

**Step 3: Verify the edit is correct**

Read the modified template and confirm both sections are updated.

**Step 4: Commit**

```bash
git add docs/templates/hypothesis.md
git commit -m "docs(templates): update hypothesis template to use shared harness and analysis helpers"
```

---

## Task 5: Smoke test the harness with an inline experiment

**Files:** None (test only)

**Step 1: Build the binary in the worktree**

Run: `cd .worktrees/experiment-harness && go build -o blis main.go`
Expected: Successful build, binary created.

**Step 2: Write a minimal smoke test script**

Create a temporary test script that sources the harness and runs a single simulation:

```bash
#!/bin/bash
set -euo pipefail
source hypotheses/lib/harness.sh
setup_experiment

echo "=== Smoke test: normal run ==="
blis_run $TIMEOUT_QUICK "$RESULTS_DIR/smoke.txt" \
    --model "$MODEL" --num-instances 2 --seed 42 --num-requests 10 --log error
echo "Exit: $?"
head -3 "$RESULTS_DIR/smoke.txt"

echo ""
echo "=== Smoke test: preflight_kv_check (should warn) ==="
preflight_kv_check 50 16 512 || echo "  (returned 1 = unsafe, as expected)"

echo ""
echo "=== Smoke test: preflight_kv_check (should pass) ==="
preflight_kv_check 5000 16 512 && echo "  (returned 0 = safe)"

echo ""
echo "=== Smoke test: is_timeout on good output ==="
is_timeout "$RESULTS_DIR/smoke.txt" && echo "  timeout detected" || echo "  no timeout (correct)"

echo ""
echo "=== Smoke test: is_timeout on timeout sentinel ==="
echo "TIMEOUT" > "$RESULTS_DIR/timedout.txt"
is_timeout "$RESULTS_DIR/timedout.txt" && echo "  timeout detected (correct)" || echo "  no timeout"

echo ""
echo "=== Smoke test: parse_blis_output (Python) ==="
python3 -c "
import sys
sys.path.insert(0, 'hypotheses/lib')
from analyze_helpers import parse_blis_output
m = parse_blis_output('$RESULTS_DIR/smoke.txt')
print(f'  ttft_mean={m[\"ttft_mean\"]:.2f}, timed_out={m[\"timed_out\"]}')
m2 = parse_blis_output('$RESULTS_DIR/timedout.txt')
print(f'  timeout file: timed_out={m2[\"timed_out\"]}')
"

echo ""
echo "All smoke tests passed."
```

Run this script from the worktree directory.
Expected: All 5 smoke tests pass.

**Step 3: Clean up the smoke test script** (delete it — it's temporary).
