# Hypothesis Experiment Template

> **For Claude:** Use this template when creating a new hypothesis experiment in `hypotheses/<name>/`.

## FINDINGS.md Structure

Every experiment's `FINDINGS.md` MUST contain these sections:

```
# <Hypothesis Name>

**Status:** Confirmed | Confirmed with nuance | Partially confirmed | Refuted | Inconclusive
**Resolution:** <one of: Clean confirmation | Confirmation with wrong mechanism | Confirmation with bug discovery | Partial confirmation with surprise | Refuted — mechanism not plausible | Refuted — system design flaw | Refuted — wrong mental model | Inconclusive — parameter-dependent | Converged to open question>
**Family:** <one of: Workload/arrival | Scheduler invariants | Performance-regime | Structural model | Robustness/failure-mode | Cross-policy comparative>
**Tier:** <tier number from research.md>
**Type:** Deterministic | Statistical (<subtype>)
**Date:** YYYY-MM-DD
**Rounds:** <number of experiment-review rounds to convergence>

## Hypothesis

> <Quoted hypothesis statement — intuitive claim about system behavior>

## Experiment Design

**Classification:** <Deterministic | Statistical/Dominance | Statistical/Monotonicity | Statistical/Equivalence | Statistical/Pareto>

**Configurations compared:**
- A: <description + exact CLI flags>
- B: <description + exact CLI flags>

**Controlled variables:** <what is held constant>
**Varied variable:** <what differs between A and B>
**Seeds:** <list of seeds used>
**Preconditions verified:** <what was checked before running>

## Results

<Comparison tables with per-seed values>

## Root Cause Analysis

<Why the results are what they are — trace through the code/architecture.
Every causal claim MUST cite file:line (RCV-1).
Every "surprise" MUST include a first-principles calculation (RCV-2).
Must explain the mechanism AND its direction (RCV-3).
If a mechanism is proposed, describe the control experiment that would confirm it (RCV-4).>

## Devil's Advocate (RCV-5)

<Before sending to review, argue the OPPOSITE of your conclusion.>

**If this is "Confirmed," argue why it might be Refuted:**
<2-3 sentences>

**If this is "Refuted," argue why it might be Confirmed:**
<2-3 sentences>

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| <finding 1> | Confirmation / Bug / New rule / New invariant / Design limitation / Surprise / Open question | <issue number or "documented here"> |

## Standards Audit

Findings checked against docs/standards/:
- [ ] Any violations of existing rules? <list or "none found">
- [ ] Any new rules needed? <list or "none">
- [ ] Any new invariants needed? <list or "none">
- [ ] Any existing rules/invariants confirmed? <list or "none">

## Scope and Limitations (RCV-6)

- **Operating point tested:** <blocks, rate, seeds, instances, routing, etc.>
- **Parameters findings depend on:** <what must be true for these results to hold>
- **What was NOT tested:** <parameter ranges, workloads, configs not covered>
- **Generalizability:** <does this finding generalize, or is it specific to this config?>

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| <primary metric> | <value> | High / Medium / Low — <why> |
| Sample size | <seeds × configs × requests> | <assessment> |
| Mechanism | <proposed mechanism> | <confidence + whether control confirms> |

## Implications for Users

<Practical guidance derived from this experiment>

## Reproducing

cd hypotheses/<name>
./run.sh
```

## run.sh Structure

```bash
#!/bin/bash
# <Hypothesis name>
# <One-line description>
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

run_sim() {
    # Wrapper around the binary with common flags
    "$BINARY" run --model "$MODEL" --num-instances 4 \
        --log error --summarize-trace --trace-level decisions \
        "$@" 2>/dev/null
}

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# -- Experiment sections -----------------------------------------------
# Each experiment: run configurations, save to temp files, call analyze.py
# ----------------------------------------------------------------------
```

## analyze.py Structure

```python
#!/usr/bin/env python3
"""Analysis script for <hypothesis name>.

Parses BLIS multi-block output and produces comparison tables.
"""
import json, math, re, sys
from pathlib import Path

def parse_output(filepath):
    """Parse BLIS output -> cluster metrics + distribution + cache hit rate."""
    content = Path(filepath).read_text()
    # Extract cluster JSON block (instance_id == "cluster")
    # Extract target distribution from trace summary
    # Extract KV cache metrics (Cache Hit Rate)
    # Return dict with ttft_mean, ttft_p99, throughput, dist, hit_rate, etc.
    ...
```
