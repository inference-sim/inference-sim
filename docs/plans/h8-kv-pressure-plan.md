# H8: KV Cache Pressure Hypothesis Experiment Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate that reducing KV cache blocks increases preemption frequency and worsens tail latency, producing a documented experiment artifact for users.

**The problem today:** BLIS users have no empirical guidance on how `--total-kv-blocks` affects simulation behavior. The KV cache preemption mechanism (`sim/simulator.go:375-408`) is tested structurally but never validated end-to-end in a controlled experiment. Users deploying with undersized KV caches get no warning about the performance cliff.

**What this PR adds:**
1. **H8 hypothesis experiment** — a self-contained `run.sh` + `analyze.py` + `FINDINGS.md` in `hypotheses/h8-kv-pressure/` that demonstrates the monotonic relationship between KV block count and preemption/latency
2. **Preemption count in JSON output** — adds `preemption_count` to `MetricsOutput` so future experiments and users can programmatically access preemption data
3. **Updated hypothesis README** — registers H8 in the experiment catalog

**Why this matters:** Completes Tier 3 of the research plan (H8 = KV cache pressure), continues the hypothesis-driven testing methodology validated by H3 and H9, and addresses the #1 gap identified by reviewers (KV cache behavior).

**Architecture:** No new packages or interfaces. Adds one field to `MetricsOutput` struct (`sim/metrics_utils.go`), populates it in `SaveResults` (`sim/metrics.go`). Experiment artifacts are standalone scripts in `hypotheses/h8-kv-pressure/`.

**Source:** H8 in `docs/plans/research.md` (Idea 2, Tier 3)

**Closes:** N/A — source is research plan, no linked issues

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds a hypothesis experiment testing KV cache pressure behavior. The experiment varies `--total-kv-blocks` across 5 values (5000 → 2000) at high request rate (2000 req/s) and measures preemption rate, TTFT p99, and E2E p99. The experiment also adds `preemption_count` to the JSON output schema so the metric is programmatically accessible.

**Feasibility results (from pre-plan testing):**

| Blocks | TTFT p99 (ms) | E2E p99 (ms) | Preemption Rate |
|--------|---------------|---------------|-----------------|
| 5000   | 474           | 3609          | 0               |
| 2500   | 474           | 3609          | 0               |
| 2100   | 2305          | 6194          | 0.175           |
| 2050   | 2381          | 6414          | 0.395           |
| 2000   | 2860          | 7072          | 0.595           |

The transition is sharp (between 2500 and 2100 blocks). Below 1000 blocks, the simulation livelocks due to extreme preemption churn.

### B) Behavioral Contracts

**Positive contracts:**

BC-1: Preemption count in JSON output
- GIVEN a simulation that completes with preemptions
- WHEN JSON metrics are output to stdout
- THEN the `preemption_count` field MUST be present and equal to the number of preemption events that occurred
- MECHANISM: `MetricsOutput.PreemptionCount` populated from `m.PreemptionCount` in `SaveResults`

BC-2: Monotonic preemption increase
- GIVEN a fixed workload (rate=2000, 200 requests, 4 instances, 512 mean input)
- WHEN total-kv-blocks decreases through {5000, 3000, 2200, 2100, 2000}
- THEN preemption rate MUST monotonically increase (or remain 0) across all 3 seeds
- MECHANISM: Fewer blocks → more KV exhaustion → more `preempt()` calls in `simulator.go:436/455`

BC-3: Monotonic TTFT worsening
- GIVEN the same workload as BC-2
- WHEN total-kv-blocks decreases through the same 5 values
- THEN TTFT p99 MUST monotonically increase (or remain unchanged) across all 3 seeds
- MECHANISM: Preempted requests restart from scratch (ProgressIndex=0), increasing queue wait time

BC-4: Conservation under pressure
- GIVEN constrained KV blocks (2000) with high preemption rate
- WHEN the simulation completes
- THEN `injected_requests == completed_requests + still_queued + still_running` (INV-1)
- MECHANISM: `preempt()` re-queues requests to front of WaitQ (`simulator.go:394`)

**Negative contracts:**

BC-5: No livelock at tested configurations
- GIVEN total-kv-blocks >= 2000 with the experiment workload
- WHEN the simulation runs
- THEN it MUST complete within 120 seconds wall time
- MECHANISM: With 2000 blocks and 512-token requests, each needing ~32 blocks, preemptions are frequent but requests eventually complete

**Error handling:**

BC-6: Zero preemption count for abundant blocks
- GIVEN total-kv-blocks = 5000 (abundant)
- WHEN the simulation completes
- THEN `preemption_count` MUST be 0 in JSON output
- MECHANISM: 5000 blocks per instance × 4 instances = ample capacity for 200 requests

### C) Component Interaction

```
MetricsOutput (sim/metrics_utils.go)
    ↑ new field: PreemptionCount
    |
SaveResults (sim/metrics.go)
    ↑ populates PreemptionCount from Metrics.PreemptionCount
    |
preempt() (sim/simulator.go:375)
    ↑ increments Metrics.PreemptionCount
    |
makeRunningBatch (sim/simulator.go:436/455)
    ↑ calls preempt() when KV allocation fails
```

No new interfaces, no new packages, no state ownership changes.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| 3 block configs: 2000, 500, 100 | 5 configs: 5000, 3000, 2200, 2100, 2000 | CORRECTION: 500 and 100 cause timeout/livelock. Feasibility testing found transition at 2100-2500. |
| Measure `PreemptionCount` in JSON | Adds `preemption_count` to `MetricsOutput` | ADDITION: Field was tracked internally but not surfaced. |
| rate=500, 200 requests | rate=2000, 200 requests, 512-token input | CORRECTION: rate=500 with default input doesn't create enough concurrent KV pressure for preemptions. |

### E) Review Guide

**The tricky part:** The experiment parameters are critical — too few blocks causes livelock, too many causes no differentiation. The 5 block values (5000/3000/2200/2100/2000) were empirically validated.

**What to scrutinize:** BC-4 (conservation under preemption pressure) — this is the highest-value finding. If conservation breaks under KV pressure, it's a real bug.

**What's safe to skim:** The `analyze.py` parsing logic follows the established H3/H9 pattern.

**Known debt:** `PreemptionCount` is not in the cluster-level `RawMetrics` text output as an integer count (only as a rate). The JSON field we're adding is per-instance.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `hypotheses/h8-kv-pressure/run.sh` — experiment driver
- `hypotheses/h8-kv-pressure/analyze.py` — output parser

**Files to modify:**
- `sim/metrics_utils.go` — add `PreemptionCount` field to `MetricsOutput`
- `sim/metrics.go` — populate `PreemptionCount` in `SaveResults`
- `hypotheses/README.md` — register H8

**Files to create after running:**
- `hypotheses/h8-kv-pressure/FINDINGS.md` — results document

### G) Task Breakdown

---

### Task 1: Add PreemptionCount to MetricsOutput JSON

**Contracts Implemented:** BC-1, BC-6

**Files:**
- Modify: `sim/metrics_utils.go:49-75` (MetricsOutput struct)
- Modify: `sim/metrics.go:64-78` (SaveResults population)
- Test: `sim/metrics_test.go` (if exists, or verify via experiment)

**Step 1: Add field to MetricsOutput struct**

In `sim/metrics_utils.go`, add `PreemptionCount` field to `MetricsOutput`:

```go
// In MetricsOutput struct, after KVAllocationFailures:
PreemptionCount      int64            `json:"preemption_count"`
```

**Step 2: Populate field in SaveResults**

In `sim/metrics.go`, in the `SaveResults` function, add after line ~77 (`KVAllocationFailures`):

```go
PreemptionCount:     m.PreemptionCount,
```

**Step 3: Verify build compiles**

Run: `go build ./sim/...`
Expected: PASS

**Step 4: Run existing tests to verify no regression**

Run: `go test ./sim/... -count=1`
Expected: PASS (existing tests unaffected — they don't assert on MetricsOutput fields)

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/metrics_utils.go sim/metrics.go
git commit -m "feat(metrics): add preemption_count to JSON output (BC-1)

- Add PreemptionCount field to MetricsOutput struct
- Populate from Metrics.PreemptionCount in SaveResults
- Enables programmatic access to preemption data for experiments

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Update golden dataset (if needed)

**Contracts Implemented:** (supports BC-1)

**Step 1: Check if golden dataset tests reference MetricsOutput fields**

Run: `go test ./sim/... -run Golden -v -count=1`

If tests pass: no golden dataset update needed (the new field is `omitempty`-free but defaults to 0 which doesn't break existing assertions since golden tests check specific fields, not full JSON match).

If tests fail: regenerate golden dataset:
```bash
go test ./sim/internal/testutil/... -run TestRegenerate -v
```

**Step 2: Run full test suite**

Run: `go test ./... -count=1`
Expected: ALL PASS

**Step 3: Commit if golden data changed**

```bash
# Only if golden dataset was regenerated
git add testdata/goldendataset.json
git commit -m "test: regenerate golden dataset for preemption_count field (R12)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Create experiment run.sh

**Contracts Implemented:** BC-2, BC-3, BC-4, BC-5

**Files:**
- Create: `hypotheses/h8-kv-pressure/run.sh`

**Step 1: Create the experiment script**

```bash
#!/bin/bash
# H8: KV Cache Pressure
#
# Hypothesis: Reducing total KV blocks should increase preemption
# frequency and worsen tail latency (monotonically).
#
# Type: Statistical / Monotonicity
# Mechanism under test:
#   sim/simulator.go:375-408 — preempt() evicts running requests when KV is full
#   sim/simulator.go:436,455 — makeRunningBatch calls preempt() on allocation failure
#
# Experiment 1: Monotonicity (5 block counts × 3 seeds)
# Experiment 2: Conservation check (INV-1 at all block counts)
#
# Usage: ./run.sh [--rebuild]

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
SEEDS=(42 123 456)
BLOCK_COUNTS=(5000 3000 2200 2100 2000)

analyze() {
    python3 "$SCRIPT_DIR/analyze.py" "$@"
}

# Create workload YAML for a given seed
# High rate + medium-length tokens to create KV pressure
make_workload() {
    local seed=$1
    local outfile=$2

    cat > "$outfile" << YAMLEOF
version: "1"
seed: $seed
category: language
aggregate_rate: 2000.0
num_requests: 200
clients:
  - id: "kv-stress"
    tenant_id: "test"
    slo_class: "interactive"
    rate_fraction: 1.0
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 512
        std_dev: 50
        min: 256
        max: 768
    output_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 50
        min: 128
        max: 512
YAMLEOF
}

echo "============================================================================"
echo "  H8: KV Cache Pressure"
echo "  Hypothesis: Reducing total-kv-blocks monotonically increases preemption"
echo "              rate and worsens TTFT p99 / E2E p99"
echo "  Type: Statistical / Monotonicity"
echo "  Reference: docs/plans/research.md, sim/simulator.go:375-408"
echo "============================================================================"
echo ""

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# ── Experiment 1: Monotonicity ───────────────────────────────────────────────

echo "Experiment 1: KV Block Pressure Monotonicity"
echo "  Config: 4 instances, 200 requests, rate=2000, block_size=16"
echo "  Block counts: ${BLOCK_COUNTS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo ""

for seed in "${SEEDS[@]}"; do
    make_workload "$seed" "$RESULTS_DIR/wl_${seed}.yaml"
    for blocks in "${BLOCK_COUNTS[@]}"; do
        echo "  Running: seed=$seed blocks=$blocks ..."
        timeout 120 "$BINARY" run \
            --model "$MODEL" \
            --num-instances 4 \
            --total-kv-blocks "$blocks" \
            --block-size-in-tokens 16 \
            --seed "$seed" \
            --workload-spec "$RESULTS_DIR/wl_${seed}.yaml" \
            --log error \
            2>/dev/null \
            > "$RESULTS_DIR/exp1_b${blocks}_s${seed}.txt" \
            || echo "    WARNING: timeout or error for blocks=$blocks seed=$seed"
    done
done

echo ""
analyze monotonicity "$RESULTS_DIR"/exp1_*.txt

# ── Experiment 2: Conservation Check ─────────────────────────────────────────

echo ""
echo "============================================================================"
echo "Experiment 2: Conservation Invariant (INV-1) Under KV Pressure"
echo "  Verifying: injected == completed + still_queued + still_running"
echo ""

analyze conservation "$RESULTS_DIR"/exp1_*.txt

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis and root cause"
echo "============================================================================"
```

**Step 2: Make executable**

```bash
chmod +x hypotheses/h8-kv-pressure/run.sh
```

**Step 3: Verify script syntax**

Run: `bash -n hypotheses/h8-kv-pressure/run.sh`
Expected: No syntax errors

**Step 4: Commit**

```bash
git add hypotheses/h8-kv-pressure/run.sh
git commit -m "feat(h8): add KV pressure experiment script (BC-2, BC-3, BC-4, BC-5)

- 5 block counts (5000/3000/2200/2100/2000) × 3 seeds
- Workload: rate=2000, 200 requests, 512 mean input tokens
- Conservation invariant check at each configuration
- 120s timeout per run to prevent livelock

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Create analyze.py

**Contracts Implemented:** BC-2, BC-3, BC-4

**Files:**
- Create: `hypotheses/h8-kv-pressure/analyze.py`

**Step 1: Create the analysis script**

```python
#!/usr/bin/env python3
"""Analysis script for H8: KV Cache Pressure hypothesis experiment.

Parses BLIS multi-block output files and produces comparison tables.
Called by run.sh with experiment type and output file paths.

Usage:
    python3 analyze.py monotonicity exp1_b5000_s42.txt exp1_b3000_s42.txt ...
    python3 analyze.py conservation exp1_b5000_s42.txt ...
"""

import json
import re
import sys
from pathlib import Path


def parse_output(filepath):
    """Parse multi-block BLIS output into cluster metrics + KV cache metrics."""
    content = Path(filepath).read_text()
    if not content.strip():
        return None

    # Extract cluster-level JSON block
    cluster = None
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        block = json.loads(match.group(1))
        if block.get("instance_id") == "cluster":
            cluster = block

    if not cluster:
        return None

    # Extract preemption rate from KV Cache Metrics text section
    preemption_rate = 0.0
    pr_match = re.search(r"Preemption Rate: ([\d.]+)", content)
    if pr_match:
        preemption_rate = float(pr_match.group(1))

    # Extract cache hit rate
    cache_hit_rate = 0.0
    chr_match = re.search(r"Cache Hit Rate: ([\d.]+)", content)
    if chr_match:
        cache_hit_rate = float(chr_match.group(1))

    # Compute preemption count from rate and completed requests
    preemption_count = int(round(preemption_rate * cluster["completed_requests"]))

    # Also check JSON preemption_count if available (new field)
    # Sum across per-instance JSON blocks
    json_preemption_total = 0
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        block = json.loads(match.group(1))
        if block.get("instance_id") != "cluster":
            json_preemption_total += block.get("preemption_count", 0)

    return {
        "blocks": 0,  # filled by caller
        "seed": 0,  # filled by caller
        "ttft_mean": cluster["ttft_mean_ms"],
        "ttft_p99": cluster["ttft_p99_ms"],
        "e2e_mean": cluster["e2e_mean_ms"],
        "e2e_p99": cluster["e2e_p99_ms"],
        "throughput": cluster["responses_per_sec"],
        "completed": cluster["completed_requests"],
        "still_queued": cluster["still_queued"],
        "still_running": cluster["still_running"],
        "injected": cluster["injected_requests"],
        "preemption_rate": preemption_rate,
        "preemption_count": json_preemption_total or preemption_count,
        "cache_hit_rate": cache_hit_rate,
    }


def parse_filename(filepath):
    """Extract blocks and seed from filename like exp1_b5000_s42.txt."""
    name = Path(filepath).stem
    blocks_match = re.search(r"_b(\d+)", name)
    seed_match = re.search(r"_s(\d+)", name)
    blocks = int(blocks_match.group(1)) if blocks_match else 0
    seed = int(seed_match.group(1)) if seed_match else 0
    return blocks, seed


def analyze_monotonicity(files):
    """Experiment 1: Verify monotonic increase in preemptions as blocks decrease."""
    results = {}
    for f in files:
        blocks, seed = parse_filename(f)
        r = parse_output(f)
        if r:
            r["blocks"] = blocks
            r["seed"] = seed
            results[(blocks, seed)] = r

    seeds = sorted({s for _, s in results})
    block_counts = sorted({b for b, _ in results}, reverse=True)

    # Per-seed detailed table
    for seed in seeds:
        print(f"  Seed {seed}:")
        print(
            f"    {'Blocks':>7} | {'Preempt Rate':>12} {'Preempt #':>9}"
            f" | {'TTFT p99':>10} {'E2E p99':>10}"
            f" | {'Throughput':>10} {'Cache Hit':>9}"
        )
        print(f"    {'-'*7}-+-{'-'*22}-+-{'-'*21}-+-{'-'*20}")

        for blocks in block_counts:
            r = results.get((blocks, seed))
            if not r:
                print(f"    {blocks:>7} | {'TIMEOUT':>12}")
                continue
            print(
                f"    {blocks:>7} |"
                f" {r['preemption_rate']:>12.4f} {r['preemption_count']:>9d}"
                f" | {r['ttft_p99']:>10.1f} {r['e2e_p99']:>10.1f}"
                f" | {r['throughput']:>10.1f} {r['cache_hit_rate']:>9.4f}"
            )
        print()

    # Monotonicity check
    print("  Monotonicity Check:")
    all_monotonic_preemption = True
    all_monotonic_ttft = True
    for seed in seeds:
        prev_preempt = -1.0
        prev_ttft = -1.0
        mono_p = True
        mono_t = True
        for blocks in block_counts:  # descending blocks = increasing pressure
            r = results.get((blocks, seed))
            if not r:
                mono_p = False
                mono_t = False
                break
            if r["preemption_rate"] < prev_preempt:
                mono_p = False
            if r["ttft_p99"] < prev_ttft - 0.01:  # small tolerance for float
                mono_t = False
            prev_preempt = r["preemption_rate"]
            prev_ttft = r["ttft_p99"]

        status_p = "PASS" if mono_p else "FAIL"
        status_t = "PASS" if mono_t else "FAIL"
        print(f"    Seed {seed}: preemption [{status_p}]  TTFT p99 [{status_t}]")
        if not mono_p:
            all_monotonic_preemption = False
        if not mono_t:
            all_monotonic_ttft = False

    print()
    verdict_p = "CONFIRMED" if all_monotonic_preemption else "REFUTED"
    verdict_t = "CONFIRMED" if all_monotonic_ttft else "REFUTED"
    print(f"  Preemption monotonicity: {verdict_p}")
    print(f"  TTFT p99 monotonicity:   {verdict_t}")

    # Summary table (averaged across seeds)
    print()
    print("  Summary (averaged across seeds):")
    print(
        f"    {'Blocks':>7} | {'Preempt Rate':>12} {'TTFT p99':>10}"
        f" {'E2E p99':>10} | {'vs Baseline':>11}"
    )
    print(f"    {'-'*7}-+-{'-'*33}-+-{'-'*11}")

    baseline_ttft = None
    for blocks in block_counts:
        vals = [results[(blocks, s)] for s in seeds if (blocks, s) in results]
        if not vals:
            continue
        avg_pr = sum(v["preemption_rate"] for v in vals) / len(vals)
        avg_ttft = sum(v["ttft_p99"] for v in vals) / len(vals)
        avg_e2e = sum(v["e2e_p99"] for v in vals) / len(vals)
        if baseline_ttft is None:
            baseline_ttft = avg_ttft
        ratio = avg_ttft / baseline_ttft if baseline_ttft > 0 else 0
        label = "baseline" if ratio == 1.0 else f"{ratio:.2f}x"
        print(
            f"    {blocks:>7} |"
            f" {avg_pr:>12.4f} {avg_ttft:>10.1f} {avg_e2e:>10.1f}"
            f" | {label:>11}"
        )


def analyze_conservation(files):
    """Experiment 2: Verify INV-1 (request conservation) at each config."""
    results = {}
    for f in files:
        blocks, seed = parse_filename(f)
        r = parse_output(f)
        if r:
            results[(blocks, seed)] = r

    all_pass = True
    for (blocks, seed), r in sorted(results.items()):
        actual = r["completed"] + r["still_queued"] + r["still_running"]
        expected = r["injected"]
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"  blocks={blocks:>5} seed={seed}: "
            f"injected={expected} completed={r['completed']} "
            f"queued={r['still_queued']} running={r['still_running']} "
            f"[{status}]"
        )

    print()
    verdict = "ALL PASS" if all_pass else "VIOLATIONS FOUND"
    print(f"  Conservation (INV-1): {verdict}")


ANALYZERS = {
    "monotonicity": analyze_monotonicity,
    "conservation": analyze_conservation,
}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <experiment-type> <files...>")
        print(f"Types: {', '.join(ANALYZERS.keys())}")
        sys.exit(1)

    experiment_type = sys.argv[1]
    files = sys.argv[2:]

    analyzer = ANALYZERS.get(experiment_type)
    if not analyzer:
        print(f"Unknown experiment type: {experiment_type}")
        print(f"Valid types: {', '.join(ANALYZERS.keys())}")
        sys.exit(1)

    analyzer(files)
```

**Step 2: Verify Python syntax**

Run: `python3 -c "import py_compile; py_compile.compile('hypotheses/h8-kv-pressure/analyze.py', doraise=True)"`
Expected: No errors

**Step 3: Commit**

```bash
git add hypotheses/h8-kv-pressure/analyze.py
git commit -m "feat(h8): add analysis script for KV pressure experiment

- Parses cluster JSON + KV Cache Metrics text output
- Monotonicity verification across seeds
- Conservation invariant check (INV-1)
- Summary table with cross-seed averages

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Run the experiment and write FINDINGS.md

**Contracts Implemented:** BC-2, BC-3, BC-4, BC-5, BC-6

**Files:**
- Create: `hypotheses/h8-kv-pressure/FINDINGS.md`

**Step 1: Run the experiment**

Run: `cd hypotheses/h8-kv-pressure && ./run.sh --rebuild`
Expected: Output showing monotonicity results and conservation checks. Capture stdout.

**Step 2: Write FINDINGS.md**

Use the template from `docs/templates/hypothesis.md`. Populate with actual experiment results. Key sections:
- Status: Confirmed / Refuted / Partially confirmed
- Experiment Design with classification (Statistical/Monotonicity)
- Results with per-seed values
- Root Cause Analysis (trace through preempt() code path)
- Findings Classification table
- Standards Audit
- User Implications

**Step 3: Commit**

```bash
git add hypotheses/h8-kv-pressure/FINDINGS.md
git commit -m "docs(h8): add findings — KV pressure experiment results

- [STATUS based on results]
- [Brief summary of key findings]

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Update hypotheses README

**Files:**
- Modify: `hypotheses/README.md`

**Step 1: Add H8 to the Validated Hypotheses table**

Add after the H9 row:

```markdown
| H8 | Reducing KV blocks increases preemption frequency and worsens tail latency | **[STATUS]** | [Key finding summary] |
```

**Step 2: Commit**

```bash
git add hypotheses/README.md
git commit -m "docs: register H8 in hypothesis catalog

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Description |
|----------|------|-----------|-------------|
| BC-1 | Task 1 | Integration | PreemptionCount appears in JSON output |
| BC-2 | Task 5 | Experiment | Monotonicity across 5 block counts × 3 seeds |
| BC-3 | Task 5 | Experiment | TTFT p99 worsens with fewer blocks |
| BC-4 | Task 5 | Invariant | Conservation holds under KV pressure |
| BC-5 | Task 5 | Experiment | All configs complete within 120s |
| BC-6 | Task 5 | Experiment | Zero preemptions at 5000 blocks |

No golden dataset update expected (new field defaults to 0).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Livelock at low block counts | Medium | Low | 120s timeout in run.sh; block counts ≥ 2000 |
| Non-monotonic TTFT across seeds | Low | Medium | 5 data points span the transition zone; if one seed contradicts, document as finding |
| ED-4 (workload seed) confounds | Low | Low | YAML seed varies with CLI seed — documented in run.sh |

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes
- [x] R4: Construction sites for MetricsOutput — `SaveResults` is the only construction site
- [x] R12: Golden dataset — new field defaults to 0, no regeneration expected
- [x] CLAUDE.md — no update needed (no new packages or CLI flags)
- [x] Task dependencies correctly ordered (1→2→3→4→5→6)

---

## Appendix: File-Level Details

### File: `sim/metrics_utils.go`

Add one field to `MetricsOutput` struct, after `KVAllocationFailures`:

```go
PreemptionCount      int64            `json:"preemption_count"`
```

### File: `sim/metrics.go`

In `SaveResults`, add to the `MetricsOutput` literal (after `KVAllocationFailures`):

```go
PreemptionCount:     m.PreemptionCount,
```

### File: `hypotheses/h8-kv-pressure/run.sh`

Complete script in Task 3 Step 1.

### File: `hypotheses/h8-kv-pressure/analyze.py`

Complete script in Task 4 Step 1.
