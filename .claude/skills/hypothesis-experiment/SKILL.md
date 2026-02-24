---
name: hypothesis-experiment
description: Run the full hypothesis experiment workflow (Steps 0-10) — worktree, classify, design review (5 perspectives), human gate, implement, code review (5 perspectives), run, analyze, FINDINGS review (10 perspectives), self-audit, commit + PR. Enforces convergence protocol and hard gates.
argument-hint: <hypothesis-name>
disable-model-invocation: true
---

# Hypothesis Experiment Workflow

You are running the hypothesis experiment workflow for hypothesis **$ARGUMENTS**.

**Canonical source:** `docs/process/hypothesis.md` (v2.0). Read it NOW before proceeding.
**Standards:** `docs/standards/experiments.md` (ED-1–ED-6, RCV-1–RCV-6).
**Template:** `docs/templates/hypothesis.md` (FINDINGS.md structure).
**Review prompts:** See [review-prompts.md](review-prompts.md) for exact agent dispatch prompts.

---

## Hard Rules (non-negotiable)

1. **Follow the steps in order.** Do not skip steps. Do not reorder.
2. **Human gate at Step 3 is a STOP.** Present the design and WAIT. Do not say "I'll proceed unless you stop me."
3. **Convergence = zero CRITICAL + zero IMPORTANT from ALL reviewers in a round.** Re-run the ENTIRE round after fixes. No exceptions. No shortcuts.
4. **NEVER trust agent self-reported convergence.** Independently verify every finding count. Agents have fabricated "0 CRITICAL, 0 IMPORTANT" when actual review found 3 CRITICAL + 18 IMPORTANT (#390).
5. **Code review BEFORE running experiments** (Step 5). Three of four major bugs in PR #310 would have been caught.
6. **Max 10 rounds per gate.** If not converged by round 10, suspend the experiment.
7. **Cross-gate regression:** If Code or FINDINGS Review finds a design flaw, loop back to Step 2. Max 2 regressions total.

## Lessons Learned (encoded from MEMORY.md)

- **`--stderr` must come BEFORE other flags** in `blis_run` calls — harness checks position 3 only
- **CLI defaults vs workload-spec YAML**: `--rate` mode uses CLI defaults (512/512 tokens). Workload YAMLs define own distributions. Capacity estimates MUST match the actual workload.
- **`--total-kv-blocks` default is context-dependent**: CLI default 1000000 but `defaults.yaml` overrides to 132139 for llama/H100/TP=2. Check defaults.yaml.
- **Conservation formula**: Always 4-term INV-1: `injected == completed + queued + running + dropped_unservable`. `parse_blis_output` does NOT extract `dropped_unservable` — parse it separately.
- **Analyzer verdict must match FINDINGS status**: If `analyze.py` produces a different verdict than FINDINGS.md, acknowledge the discrepancy explicitly.
- **Think before coding calibration**: Compute parameters analytically from alpha/beta coefficients FIRST, then validate with a tiny run.
- **Beta coefficients** (llama-3.1-8b, H100, TP=2): `[6910.42, 17.67, 2.84]` -> stepTime = beta0 + beta1*cacheMissTokens + beta2*decodeTokens
- **Alpha coefficients**: `[1601.35, 3.51, 1805.54]` -> queueDelay = alpha0 + alpha1*inputLen; outputProcessing = alpha2
- **WorkloadSpec YAML format**: `id:` not `client_id:`, `process:` not `type:` in arrival, `aggregate_rate:` top-level, `rate_fraction:` per client, distribution params under `params:` with `std_dev` not `stdev`

---

## Step 0: Create Worktree

Create an isolated workspace FIRST, before any other work.

```
/superpowers:using-git-worktrees h-$ARGUMENTS
```

All subsequent steps happen in the worktree. Set your working directory there.

---

## Step 1: Select and Classify

1. **Check coverage gaps**: Read `hypotheses/README.md` for the coverage table
2. **Classify the hypothesis**:
   - **Family**: Which of the 6 families? (See `docs/standards/experiments.md`)
   - **VV&UQ**: Verification, Validation, or UQ?
   - **Type**: Deterministic or Statistical? If statistical: dominance, monotonicity, equivalence, or Pareto?
3. **Write the hypothesis sentence** using the family-specific pattern from experiments.md
4. **Add diagnostic clause**: "If this fails, it would indicate..."

**WARNING**: Pose the hypothesis WITHOUT reading the code first. Code-grounded hypotheses test implementation, not behavior.

---

## Step 2: Design + Design Review

### 2a: Design the Experiment

Follow ED-1 through ED-6 (see `docs/standards/experiments.md`):
- ED-1: Controlled comparison (vary exactly ONE dimension)
- ED-2: Rate awareness (run where effect expected AND where it should vanish)
- ED-3: Precondition verification (in script, not just prose)
- ED-4: Workload seed independence (3 seeds minimum for statistical: 42, 123, 456)
- ED-5: Reproducibility (everything from `run.sh` alone)
- ED-6: Config diff against referenced experiments

**Compute parameters analytically** from alpha/beta coefficients. Do NOT guess.

### 2b: Design Review (5 perspectives)

Dispatch 5 parallel review agents using the convergence-review skill:

```
/convergence-review h-design
```

Alternatively, dispatch manually. See [review-prompts.md](review-prompts.md) Section A for exact prompts.

**Perspectives**: (1) Hypothesis Quality, (2) ED Rigor, (3) Parameter Calibration, (4) Control Completeness, (5) DES/Domain Fit

The convergence-review skill enforces the protocol automatically. If dispatching manually, apply the **convergence protocol**:
1. Launch all 5 in parallel as background Task agents (subagent_type="general-purpose", model="haiku")
2. Collect all findings classified as CRITICAL / IMPORTANT / SUGGESTION
3. **Zero CRITICAL + zero IMPORTANT = converged** -> proceed to Step 3
4. **Any CRITICAL or IMPORTANT** -> fix all, re-run ENTIRE round (not just failed perspectives)
5. **Independently count findings yourself.** Do not trust agent summaries.

---

## Step 3: Human Approval Gate — STOP HERE

**Present the experiment design to the user.** Include:
- Hypothesis sentence + classification (family, VV&UQ, type)
- Experiment design summary (configurations, controlled variables, seeds)
- Parameter choices with analytical derivation
- Planned controls (one per proposed mechanism)
- Expected outcomes and diagnostic implications

**This is a hard gate. WAIT for explicit human approval before proceeding.**

Use the AskUserQuestion tool:
```
"Do you approve this experiment design?"
Options: "Approve — proceed to implementation", "Revise — I have feedback"
```

---

## Step 4: Implement

Create `hypotheses/h-$ARGUMENTS/run.sh` and `hypotheses/h-$ARGUMENTS/analyze.py`.

### Mandatory harness requirements

**run.sh** MUST:
```bash
#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../lib/harness.sh"
setup_experiment "${1:-}"
```
- Use `blis_run` for EVERY simulation call (not `$BINARY run` directly)
- Every `blis_run` needs a timeout tier: `$TIMEOUT_QUICK` (<100 req), `$TIMEOUT_STANDARD` (100-500), `$TIMEOUT_EXTENDED` (>500)
- If using `--total-kv-blocks`, call `preflight_kv_check` first
- Call `python3 "$SCRIPT_DIR/analyze.py" ...` at the end

**analyze.py** MUST:
```python
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from analyze_helpers import parse_blis_output, check_for_timeout
```
- Use `parse_blis_output()` for all metric extraction
- Check `metrics["timed_out"]` before computing ratios
- Print warnings to stderr, results to stdout
- Verify INV-1 conservation (4-term formula) for every run

---

## Step 5: Code Review (5 perspectives) — BEFORE running

**Every run.sh and analyze.py must be code-reviewed BEFORE running.**

Dispatch 5 parallel review agents using the convergence-review skill:

```
/convergence-review h-code hypotheses/h-$ARGUMENTS/
```

Alternatively, dispatch manually. See [review-prompts.md](review-prompts.md) Section B for exact prompts.

**Perspectives**: (1) Parser-Output Agreement, (2) CLI Flag Correctness, (3) YAML Field Validation, (4) Config Diff (ED-6), (5) Seed and Determinism

Apply the convergence protocol (same rules as Step 2b).

**Cross-gate regression**: If this review finds a design flaw, loop back to Step 2.

---

## Step 6: Run Experiments

After Code Review converges:
- **Deterministic**: Single seed sufficient
- **Statistical**: Minimum 3 seeds (42, 123, 456) per configuration
- Execute: `bash hypotheses/h-$ARGUMENTS/run.sh`
- Verify reproducibility: running twice produces identical output

---

## Step 7: Analyze and Document

1. **Review analyzer output** from Step 6
2. **Trace every causal claim through code** (RCV-1: cite `file:line`)
3. **Compute expected values from first principles** for any "surprises" (RCV-2)
4. **Check mechanism AND direction** (RCV-3)
5. **Write FINDINGS.md** using `docs/templates/hypothesis.md` — ALL sections must be non-empty
6. **Update `hypotheses/README.md`** — add row to the validated hypotheses table

---

## Step 8: FINDINGS Review (10 perspectives)

Dispatch 10 parallel review agents using the convergence-review skill:

```
/convergence-review h-findings hypotheses/h-$ARGUMENTS/FINDINGS.md
```

Alternatively, dispatch manually. See [review-prompts.md](review-prompts.md) Section C for exact prompts.

**Perspectives**: (1) Code Verifier, (2) Experiment Designer, (3) Statistical Rigor, (4) Control Auditor, (5) Standards Compliance, (6) Substance/Logic, (7) DES Mechanism, (8) Reproducibility, (9) Cross-Experiment, (10) User Guidance

Expected: 1-5 rounds (10 perspectives = higher quality bar).

**CRITICAL**: Independently read every agent's output and count findings yourself. Do NOT rely on agent self-reported totals.

**Cross-gate regression**: If design flaw found, loop back to Step 2. Max 2 regressions total.

---

## Step 9: Self-Audit (6 dimensions)

**This is NOT an agent pass.** Stop. Think critically. Answer each question yourself.

1. **Logic bugs in analyzer**: Trace through `analyze.py` mentally. Edge cases? Silent defaults to 0? Integer vs float?
2. **Reproducibility**: Would `./run.sh` again produce identical output? Any non-deterministic dependencies?
3. **FINDINGS.md consistency**: Does Status match Results data? Does Devil's Advocate actually argue against the conclusion?
4. **Cross-experiment contradictions**: Do findings contradict prior experiments or MEMORY.md knowledge?
5. **User guidance**: Would a BLIS user know what to do with these findings?
6. **Issue filing completeness**: Every actionable finding has a planned issue?

Fix all issues found. Then proceed to Step 10.

---

## Step 10: Verify + Commit + PR

### If code fixes were discovered:
```bash
go build ./...
go test ./... -count=1
golangci-lint run ./...
```
All three must pass before committing.

### Commit and PR:
```
/commit-commands:commit-push-pr
```

PR description must include:
- Hypothesis sentence and status
- Key findings (1-3 bullets)
- `Fixes #NNN` for any issues addressed

### Post-PR: File issues per the taxonomy
- **Bug**: `--label bug` — code defects discovered
- **Enhancement**: `--label enhancement` — improvements needed
- **New hypothesis**: `--label hypothesis` — follow-up experiments (use `.github/ISSUE_TEMPLATE/hypothesis.md`)
- **Design limitation**: `--label design` — documented limitations
- **Standards update**: `--label standards` — new rules/invariants

**Issue title format for hypotheses**: behavioral prediction ("X should Y"), NOT a task ("test X").

---

## Convergence Protocol Quick Reference

| Rule | Detail |
|------|--------|
| **Converged** | Zero CRITICAL + zero IMPORTANT from ALL reviewers in current round |
| **Not converged** | Fix all issues, re-run ENTIRE round |
| **Max rounds** | 10 per gate (Design, Code, FINDINGS each independent) |
| **SUGGESTION items** | Do not block convergence |
| **Agent timeout** | 5 min per reviewer; if exceeded, check output and restart |
| **Agent failure** | Fall back to performing that review directly |
| **Severity doubtful?** | If fixing it would change a conclusion → IMPORTANT. If only readability → SUGGESTION |
| **Model for reviewers** | Use haiku for speed (~2-3 min, thorough reviews) |
