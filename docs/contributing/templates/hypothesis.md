# Hypothesis Experiment Template

> **For Claude:** Use this template when creating a new hypothesis experiment in `hypotheses/<name>/`.

## FINDINGS.md Structure

Every experiment's `FINDINGS.md` MUST contain these sections:

```
# <Hypothesis Name>

**Status:** Confirmed | Confirmed with nuance | Partially confirmed | Refuted | Inconclusive
**Resolution:** <one of: Clean confirmation | Confirmation with wrong mechanism | Confirmation with bug discovery | Partial confirmation with surprise | Refuted — mechanism not plausible | Refuted — system design flaw | Refuted — wrong mental model | Inconclusive — parameter-dependent | Converged to open question>
**Family:** <one of: Workload/arrival | Scheduler invariants | Performance-regime | Structural model | Robustness/failure-mode | Cross-policy comparative>
**VV&UQ:** <one of: Verification | Validation | UQ>
**Tier:** <tier number — see hypotheses/README.md for definitions>
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

Findings checked against docs/contributing/standards/:
- [ ] Any violations of existing rules? <list or "none found">
- [ ] Any new rules needed? <list or "none">
- [ ] Any new invariants needed? <list or "none">
- [ ] Any existing rules/invariants confirmed? <list or "none">

## Scope and Limitations (RCV-6)

- **Operating point tested:** <blocks, rate, seeds, instances, routing, etc.>
- **Parameters findings depend on:** <what must be true for these results to hold>
- **What was NOT tested:** <parameter ranges, workloads, configs not covered>
- **Generalizability:** <does this finding generalize, or is it specific to this config?>
- **Uncertainty quantification:** <for any threshold or boundary finding, report confidence intervals. For any "confirmed" result, estimate the probability of holding under parameter variation. If UQ was not performed, state "UQ not performed — single operating point.">

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
# Example (robustness/stress — non-zero exit expected, use || true under set -e):
#   blis_run $TIMEOUT_EXTENDED "$RESULTS_DIR/stress.txt" \
#       --stderr "$RESULTS_DIR/stress_stderr.txt" \
#       --model "$MODEL" --num-instances 4 --seed 42 --log error || true
#
# For KV-constrained experiments, add pre-flight check (advisory, never aborts):
#   preflight_kv_check 800 16 512  # total_blocks, block_size (default: 16), max_input
# ----------------------------------------------------------------------
```

## analyze.py Structure

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
