# H14: Pathological Templates Hypothesis Experiment — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate that BLIS's pathological policy templates (`always-busiest`, `reverse-priority`, `inverted-slo`) produce measurably worse behavior than their normal counterparts, and that the anomaly detection system correctly identifies the degradation.

**The problem today:** BLIS has three pathological policy templates added as adversarial testing tools, and two anomaly detectors (HOL blocking, priority inversions). Neither has been validated end-to-end — we don't know if `always-busiest` actually triggers HOL blocking detection, or if `reverse-priority` + `inverted-slo` triggers priority inversion detection. If the anomaly detectors don't fire for intentionally adversarial inputs, they're useless for real workloads.

**What this PR adds:**
1. **Experiment artifacts** — `hypotheses/h14-pathological-templates/run.sh` and `analyze.py` that run normal vs pathological configurations and compare anomaly counters, TTFT, and distribution uniformity
2. **Workload spec** — `hypotheses/h14-pathological-templates/mixed-slo.yaml` mirroring `ScenarioMixedSLO` (33% realtime / 34% interactive / 33% batch)
3. **Findings document** — `hypotheses/h14-pathological-templates/FINDINGS.md` with results, root cause analysis, findings classification, and standards audit
4. **README update** — Add H14 to the hypothesis experiments table in `hypotheses/README.md`

**Why this matters:** H14 completes Tier 2 (high diagnostic value) and is a prerequisite for H24 (combined worst-case stress test). It validates the anomaly detection feature that users rely on for configuration debugging.

**Architecture:** No Go code changes. This PR creates experiment scripts and documentation only. The experiment uses existing CLI flags (`--routing-policy`, `--scheduler`, `--priority-policy`) with existing pathological templates.

**Source:** H14 in `docs/plans/research.md`, Idea 2 (KV Cache, Tiered Storage, Batch Formation)

**Closes:** N/A — no linked issues

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR creates an H14 hypothesis experiment that validates BLIS's anomaly detection system by running pathological configurations and verifying they trigger the expected anomaly counters. The experiment compares a "normal" configuration (`least-loaded` + `priority-fcfs` + `slo-based`) against a "pathological" configuration (`always-busiest` + `reverse-priority` + `inverted-slo`) using a mixed-SLO workload.

The experiment is classified as **Statistical/Dominance** — the pathological configuration should be strictly worse on every measured dimension.

No Go code is modified. This depends on existing pathological templates (`sim/routing.go:211-238`, `sim/scheduler.go:59-74`, `sim/priority.go:37-48`) and anomaly detection (`sim/cluster/metrics.go:164-247`).

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Pathological routing produces load imbalance
- GIVEN a mixed-SLO workload at rate=2000 with 4 instances
- WHEN routing uses `always-busiest`
- THEN the target distribution standard deviation MUST be >50 (severe imbalance) across all 3 seeds
- MECHANISM: `AlwaysBusiest.Route()` picks max `EffectiveLoad()`, concentrating requests

BC-2: Normal routing produces balanced distribution
- GIVEN the same mixed-SLO workload
- WHEN routing uses `least-loaded`
- THEN the target distribution standard deviation MUST be <10 across all 3 seeds
- MECHANISM: `LeastLoaded.Route()` picks min `EffectiveLoad()`, spreading requests

BC-3: HOL blocking detected for pathological routing
- GIVEN the pathological configuration at rate=2000
- WHEN the experiment completes
- THEN `HOL Blocking Events` in CLI output MUST be >0 for at least 2 of 3 seeds
- MECHANISM: `detectHOLBlocking()` fires when any instance's avg queue depth > 2× mean

BC-4: Priority inversions detected for pathological scheduling
- GIVEN the pathological configuration with `inverted-slo` priority and `reverse-priority` scheduler
- WHEN the experiment completes
- THEN `Priority Inversions` in CLI output MUST be >0 for all 3 seeds
- MECHANISM: `detectPriorityInversions()` counts pairs where earlier-arriving requests have worse E2E

BC-5: Pathological TTFT significantly worse than normal
- GIVEN both configurations run with same seeds
- WHEN comparing cluster `ttft_p99_ms`
- THEN pathological TTFT p99 MUST be >2× normal TTFT p99 across all 3 seeds
- MECHANISM: Load imbalance + priority inversions compound queuing delays

**Negative Contracts:**

BC-6: Normal configuration produces zero anomalies
- GIVEN the normal configuration (`least-loaded` + `priority-fcfs` + `slo-based`)
- WHEN the experiment completes
- THEN `HOL Blocking Events` MUST be 0 AND `Priority Inversions` MUST be 0 for all seeds
- MECHANISM: Proper load balancing and priority scheduling prevent anomalies

BC-7: Experiment is reproducible
- GIVEN the same seed
- WHEN `run.sh` is executed twice
- THEN output MUST be byte-identical (INV-6 determinism)

### C) Component Interaction

```
run.sh
  ├── builds simulation_worker binary
  ├── creates mixed-slo.yaml (workload spec)
  ├── runs normal config (3 seeds) → saves output files
  ├── runs pathological config (3 seeds) → saves output files
  ├── runs decomposed experiments (routing-only, scheduling-only)
  └── calls analyze.py with output files
        ├── parses cluster JSON metrics
        ├── parses anomaly counters
        ├── parses target distribution from trace summary
        └── prints comparison tables
```

No new interfaces, types, or state changes. Pure experiment artifacts.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "Run `./simulation_worker run --help \| grep -E \"always-busiest\|reverse-priority\|inverted-slo\"`" as prerequisite | Verify via running the binary directly with each policy | More reliable — help text may not list all valid values |
| Uses `ScenarioMixedSLO` built-in | Creates equivalent YAML file | `ScenarioMixedSLO` is a Go function, not CLI-accessible; YAML equivalent is needed for `--workload-spec` |
| Single comparison (normal vs pathological) | Adds decomposed sub-experiments (routing-only, scheduling-only) | ADDITION: Isolating each dimension helps diagnose which pathological template contributes what effect |

### E) Review Guide

**The tricky part:** The anomaly detection thresholds are heuristic (2× mean for HOL blocking, E2E ordering for priority inversions). If the workload rate is too low, the effects may not be measurable. Rate=2000 with 500 requests and 4 instances should create enough queuing pressure.

**What to scrutinize:** The `mixed-slo.yaml` workload spec must exactly mirror `ScenarioMixedSLO` in `sim/workload/scenarios.go:59-81`. The analyze.py parsing must handle the case where anomaly counters are absent (they only print when >0).

**What's safe to skim:** The run.sh structure follows the established H3 pattern closely.

**Known debt:** Per-SLO-class TTFT breakdown would add value but is secondary to the anomaly counter validation.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `hypotheses/h14-pathological-templates/mixed-slo.yaml` — workload spec
- `hypotheses/h14-pathological-templates/run.sh` — experiment driver
- `hypotheses/h14-pathological-templates/analyze.py` — output parser

**Files to modify:**
- `hypotheses/README.md` — add H14 row
- `hypotheses/h14-pathological-templates/FINDINGS.md` — results (written after experiments run)

**Key decisions:**
- Rate=2000, 500 requests, 4 instances (enough queuing pressure for anomalies)
- Three seeds (42, 123, 456) per standard
- Decomposed sub-experiments: routing-only (`always-busiest` with normal scheduling) and scheduling-only (normal routing with `reverse-priority` + `inverted-slo`)

### G) Task Breakdown

---

### Task 1: Create workload spec YAML

**Contracts Implemented:** (prerequisite for all BCs)

**Files:**
- Create: `hypotheses/h14-pathological-templates/mixed-slo.yaml`

**Step 1: Create the mixed-SLO workload spec**

Context: We need a YAML equivalent of `ScenarioMixedSLO(seed, rate)` from `sim/workload/scenarios.go:59-81`. The YAML seed will be fixed at 42 (per ED-4, CLI `--seed` varies independently).

```yaml
version: "1"
seed: 42
category: language
aggregate_rate: 2000
num_requests: 500
clients:
  - id: realtime
    tenant_id: tenant-rt
    slo_class: realtime
    rate_fraction: 0.33
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 64
        std_dev: 20
        min: 10
        max: 256
    output_distribution:
      type: exponential
      params:
        mean: 32
  - id: interactive
    tenant_id: tenant-int
    slo_class: interactive
    rate_fraction: 0.34
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 100
        min: 32
        max: 2048
    output_distribution:
      type: exponential
      params:
        mean: 128
  - id: batch
    tenant_id: tenant-batch
    slo_class: batch
    rate_fraction: 0.33
    arrival:
      process: poisson
    input_distribution:
      type: exponential
      params:
        mean: 1024
    output_distribution:
      type: exponential
      params:
        mean: 512
```

**Step 2: Verify the YAML is valid**

Run: `./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --workload-spec hypotheses/h14-pathological-templates/mixed-slo.yaml --num-instances 4 --seed 42 --log error 2>/dev/null | head -5`
Expected: Simulation output (JSON metrics block), no errors

**Step 3: Commit**

```bash
git add hypotheses/h14-pathological-templates/mixed-slo.yaml
git commit -m "feat(hypotheses): add mixed-SLO workload spec for H14

Mirrors ScenarioMixedSLO (33% realtime, 34% interactive, 33% batch)
with rate=2000, 500 requests for anomaly detection experiments.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Create analyze.py

**Contracts Implemented:** BC-1 through BC-6 (provides measurement infrastructure)

**Files:**
- Create: `hypotheses/h14-pathological-templates/analyze.py`

**Step 1: Write the analysis script**

Context: Follows the H3 pattern. Must parse cluster JSON, anomaly counters (which only appear when >0), and target distribution from trace summary. Must handle all four experiment types: core comparison, routing-only, scheduling-only, and per-SLO breakdown.

```python
#!/usr/bin/env python3
"""Analysis script for H14: Pathological Templates hypothesis experiment.

Parses BLIS output files and produces formatted comparison tables
showing anomaly counters, TTFT, and distribution uniformity.

Usage:
    python3 analyze.py core normal_42.txt patho_42.txt ...
    python3 analyze.py decomposed routing_42.txt scheduling_42.txt ...
"""

import json
import math
import re
import sys
from pathlib import Path


def parse_output(filepath):
    """Parse BLIS output into metrics dict."""
    content = Path(filepath).read_text()

    # Extract cluster-level JSON block
    cluster = None
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        block = json.loads(match.group(1))
        if block.get("instance_id") == "cluster":
            cluster = block

    # Extract target distribution from trace summary
    dist = {}
    dist_match = re.search(
        r"Target Distribution:\n((?:\s+instance_\d+: \d+\n?)+)", content
    )
    if dist_match:
        for line in dist_match.group(1).strip().split("\n"):
            parts = line.strip().split(": ")
            dist[parts[0]] = int(parts[1])

    # Distribution standard deviation
    counts = [dist[k] for k in sorted(dist.keys())] if dist else [0]
    mean_d = sum(counts) / len(counts)
    stddev = math.sqrt(sum((x - mean_d) ** 2 for x in counts) / len(counts))

    # Anomaly counters (only present when > 0)
    hol = 0
    hol_match = re.search(r"HOL Blocking Events: (\d+)", content)
    if hol_match:
        hol = int(hol_match.group(1))

    inversions = 0
    inv_match = re.search(r"Priority Inversions: (\d+)", content)
    if inv_match:
        inversions = int(inv_match.group(1))

    rejected = 0
    rej_match = re.search(r"Rejected Requests: (\d+)", content)
    if rej_match:
        rejected = int(rej_match.group(1))

    return {
        "ttft_mean": cluster["ttft_mean_ms"] if cluster else 0,
        "ttft_p99": cluster["ttft_p99_ms"] if cluster else 0,
        "e2e_mean": cluster["e2e_mean_ms"] if cluster else 0,
        "e2e_p99": cluster["e2e_p99_ms"] if cluster else 0,
        "throughput": cluster["responses_per_sec"] if cluster else 0,
        "dist": dist,
        "stddev": stddev,
        "hol": hol,
        "inversions": inversions,
        "rejected": rejected,
    }


def dist_str(dist):
    """Format distribution as compact list."""
    return str([dist[k] for k in sorted(dist.keys())])


def analyze_core(files):
    """Experiment 1: Normal vs Pathological across seeds."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    seeds = sorted({name.split("_")[-1] for name in results})

    print("  Normal vs Pathological (per seed):")
    print()
    hdr = (
        f"  {'Seed':>4} {'Config':<12} | {'TTFT Mean':>10} {'TTFT P99':>10}"
        f" | {'HOL':>4} {'Inv':>4} | {'StdDev':>8} | Distribution"
    )
    print(hdr)
    print(f"  {'-' * 4} {'-' * 12}-+-{'-' * 21}-+-{'-' * 9}-+-{'-' * 8}-+-{'-' * 30}")

    for seed in seeds:
        for config in ["normal", "patho"]:
            key = f"{config}_{seed}"
            r = results.get(key)
            if not r:
                continue
            label = "normal" if config == "normal" else "pathological"
            print(
                f"  {seed:>4} {label:<12} |"
                f" {r['ttft_mean']:>10.1f} {r['ttft_p99']:>10.1f}"
                f" | {r['hol']:>4} {r['inversions']:>4}"
                f" | {r['stddev']:>8.1f} | {dist_str(r['dist'])}"
            )
        # Print ratio
        n = results.get(f"normal_{seed}")
        p = results.get(f"patho_{seed}")
        if n and p and n["ttft_p99"] > 0:
            ratio = p["ttft_p99"] / n["ttft_p99"]
            print(f"       {'':12} | {'Effect:':>10} {ratio:>9.1f}x worse P99")
        print()


def analyze_decomposed(files):
    """Experiment 2: Routing-only vs Scheduling-only (seed 42)."""
    results = {}
    for f in files:
        name = Path(f).stem
        results[name] = parse_output(f)

    configs = [
        ("normal", "Normal (all correct)"),
        ("routing_only", "Pathological routing only"),
        ("sched_only", "Pathological scheduling only"),
        ("patho", "All pathological"),
    ]

    print(
        f"  {'Configuration':<30} | {'TTFT P99':>10}"
        f" | {'HOL':>4} {'Inv':>4} | {'StdDev':>8} | Distribution"
    )
    print(f"  {'-' * 30}-+-{'-' * 10}-+-{'-' * 9}-+-{'-' * 8}-+-{'-' * 30}")

    for key, label in configs:
        r = results.get(key)
        if not r:
            continue
        print(
            f"  {label:<30} |"
            f" {r['ttft_p99']:>10.1f}"
            f" | {r['hol']:>4} {r['inversions']:>4}"
            f" | {r['stddev']:>8.1f} | {dist_str(r['dist'])}"
        )


ANALYZERS = {
    "core": analyze_core,
    "decomposed": analyze_decomposed,
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

**Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('hypotheses/h14-pathological-templates/analyze.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add hypotheses/h14-pathological-templates/analyze.py
git commit -m "feat(hypotheses): add H14 analysis script

Parses anomaly counters, TTFT, and distribution from BLIS output.
Handles core comparison and decomposed (routing-only, scheduling-only).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Create run.sh

**Contracts Implemented:** BC-1 through BC-7

**Files:**
- Create: `hypotheses/h14-pathological-templates/run.sh`

**Step 1: Write the experiment driver**

Context: Two experiments. Experiment 1: normal vs pathological across 3 seeds (validates BC-1 through BC-6). Experiment 2: decomposed comparison with seed 42 only (isolates routing vs scheduling contribution).

```bash
#!/bin/bash
# H14: Pathological Templates — anomaly detection validation
#
# Hypothesis: Pathological policies (always-busiest, reverse-priority,
# inverted-slo) should produce measurably worse behavior than normal
# counterparts, and anomaly detectors should correctly identify the
# degradation (HOL blocking events > 0, priority inversions > 0).
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
WORKLOAD="$SCRIPT_DIR/mixed-slo.yaml"

run_sim() {
    local routing="$1" scheduler="$2" priority="$3" seed="$4"
    "$BINARY" run \
        --model "$MODEL" \
        --num-instances 4 \
        --workload-spec "$WORKLOAD" \
        --num-requests 500 \
        --routing-policy "$routing" \
        --scheduler "$scheduler" \
        --priority-policy "$priority" \
        --seed "$seed" \
        --log error \
        --summarize-trace \
        --trace-level decisions \
        2>/dev/null
}

analyze() {
    python3 "$SCRIPT_DIR/analyze.py" "$@"
}

echo "============================================================================"
echo "  H14: Pathological Templates — Anomaly Detection Validation"
echo "  Reference: docs/plans/research.md, Idea 2, Hypothesis 14"
echo "============================================================================"
echo ""

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# ── Experiment 1: Normal vs Pathological (3 seeds) ──────────────────────────

echo "Experiment 1: Normal vs Pathological (rate=2000, 500 requests, 3 seeds)"
echo "  Normal:      least-loaded + priority-fcfs + slo-based"
echo "  Pathological: always-busiest + reverse-priority + inverted-slo"
echo ""

for SEED in 42 123 456; do
    run_sim "least-loaded" "priority-fcfs" "slo-based" "$SEED" \
        > "$RESULTS_DIR/normal_${SEED}.txt"
    run_sim "always-busiest" "reverse-priority" "inverted-slo" "$SEED" \
        > "$RESULTS_DIR/patho_${SEED}.txt"
done
analyze core "$RESULTS_DIR"/normal_*.txt "$RESULTS_DIR"/patho_*.txt

# ── Experiment 2: Decomposed (routing-only vs scheduling-only) ──────────────

echo ""
echo "Experiment 2: Decomposed — isolate routing vs scheduling contribution (seed 42)"
echo ""

# Already have normal and full pathological from Exp 1
cp "$RESULTS_DIR/normal_42.txt" "$RESULTS_DIR/normal.txt"
cp "$RESULTS_DIR/patho_42.txt" "$RESULTS_DIR/patho.txt"

# Routing-only pathological: always-busiest + normal scheduling
run_sim "always-busiest" "priority-fcfs" "slo-based" 42 \
    > "$RESULTS_DIR/routing_only.txt"

# Scheduling-only pathological: normal routing + reverse-priority + inverted-slo
run_sim "least-loaded" "reverse-priority" "inverted-slo" 42 \
    > "$RESULTS_DIR/sched_only.txt"

analyze decomposed "$RESULTS_DIR/normal.txt" "$RESULTS_DIR/routing_only.txt" \
    "$RESULTS_DIR/sched_only.txt" "$RESULTS_DIR/patho.txt"

echo ""
echo "============================================================================"
echo "  See FINDINGS.md for detailed analysis and root cause"
echo "============================================================================"
```

**Step 2: Make executable and test**

Run: `chmod +x hypotheses/h14-pathological-templates/run.sh && hypotheses/h14-pathological-templates/run.sh`
Expected: Experiment output with comparison tables. No errors.

**Step 3: Commit**

```bash
git add hypotheses/h14-pathological-templates/run.sh
git commit -m "feat(hypotheses): add H14 experiment driver script

Runs normal vs pathological configurations across 3 seeds.
Includes decomposed sub-experiments isolating routing vs scheduling.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Run experiment and write FINDINGS.md

**Contracts Implemented:** BC-1 through BC-7

**Files:**
- Create: `hypotheses/h14-pathological-templates/FINDINGS.md`

**Step 1: Run the full experiment**

Run: `hypotheses/h14-pathological-templates/run.sh`

Capture all output. This is the raw data for the FINDINGS.md.

**Step 2: Write FINDINGS.md**

Context: Follow the template from `docs/templates/hypothesis.md`. Include per-seed values, root cause analysis tracing through the code, findings classification, and standards audit.

The FINDINGS.md structure must include:
- Status (Confirmed/Refuted/Partially confirmed)
- Hypothesis statement
- Experiment Design (classification, configs, controlled variables, seeds, preconditions)
- Results (comparison tables with per-seed values)
- Root Cause Analysis (trace through `AlwaysBusiest.Route()`, `ReversePriority.OrderQueue()`, `detectHOLBlocking()`, `detectPriorityInversions()`)
- Findings Classification table
- Standards Audit (checked against docs/standards/)
- Implications for Users
- Reproducing instructions

**Step 3: Commit**

```bash
git add hypotheses/h14-pathological-templates/FINDINGS.md
git commit -m "docs(hypotheses): add H14 findings — pathological template validation

[Status and key results will be in the commit body based on actual data]

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Update hypotheses README

**Contracts Implemented:** (documentation)

**Files:**
- Modify: `hypotheses/README.md`

**Step 1: Add H14 to the validated hypotheses table**

Add row after H9:
```
| H14 | Pathological templates produce worse behavior; anomaly detectors fire | **[Status]** | [Key finding from experiment] |
```

**Step 2: Commit**

```bash
git add hypotheses/README.md
git commit -m "docs(hypotheses): add H14 to README table

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Verification |
|----------|------|-----------|--------------|
| BC-1 | Task 3-4 | Experiment | Distribution stddev >50 for pathological |
| BC-2 | Task 3-4 | Experiment | Distribution stddev <10 for normal |
| BC-3 | Task 3-4 | Experiment | HOL Blocking Events >0 for pathological |
| BC-4 | Task 3-4 | Experiment | Priority Inversions >0 for pathological |
| BC-5 | Task 3-4 | Experiment | Pathological TTFT p99 >2× normal |
| BC-6 | Task 3-4 | Experiment | Zero anomalies for normal config |
| BC-7 | Task 3 | Experiment | Same seed → same output (run twice) |

No golden dataset changes. No Go test changes.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Anomaly counters not triggered at rate=2000 | Low | High | Decomposed experiments isolate routing vs scheduling; increase rate if needed |
| `detectHOLBlocking` threshold too lenient | Medium | Medium | The 2× threshold is fixed; if it doesn't fire for `always-busiest`, that's a finding (threshold calibration bug) |
| `detectPriorityInversions` suppressed by priority policy name check | Low | High | Verify `inverted-slo` is not in the suppression list (it's not — only "constant" and "" are suppressed) |
| WorkloadSpec YAML doesn't match ScenarioMixedSLO | Low | Medium | Cross-reference YAML fields against `scenarios.go:59-81` |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions (script-only PR)
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes
- [x] No hidden global state impact
- [x] N/A: golangci-lint (no Go code changes)
- [x] N/A: CLAUDE.md update (no new files/packages in sim/)
- [x] Deviation log reviewed — all deviations justified
- [x] Each task produces working artifacts
- [x] Task dependencies correctly ordered (YAML → analyze.py → run.sh → run → findings)
- [x] All contracts mapped to tasks
- [x] N/A: Golden dataset (no output format changes)
- [x] N/A: Construction site audit (no struct changes)
- [x] N/A: Macro plan update (hypothesis experiment, not feature work)

---

## Appendix: File-Level Details

### File: `hypotheses/h14-pathological-templates/mixed-slo.yaml`

Exact mirror of `ScenarioMixedSLO(42, 2000)` from `sim/workload/scenarios.go:59-81` with `num_requests: 500` added.

### File: `hypotheses/h14-pathological-templates/analyze.py`

See Task 2 for complete implementation. Two analyzer functions:
- `analyze_core`: Normal vs pathological across seeds, prints per-seed comparison with TTFT ratio
- `analyze_decomposed`: Four configurations (normal, routing-only, scheduling-only, all pathological) for single seed

### File: `hypotheses/h14-pathological-templates/run.sh`

See Task 3 for complete implementation. Two experiments:
1. Core hypothesis: normal vs pathological × 3 seeds
2. Decomposed: isolate routing (`always-busiest` only) vs scheduling (`reverse-priority` + `inverted-slo` only)

### File: `hypotheses/h14-pathological-templates/FINDINGS.md`

Written after experiments run. Follows `docs/templates/hypothesis.md` structure.
