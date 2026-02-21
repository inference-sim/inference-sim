# Hypothesis Experiment Process

This document describes the process for running a hypothesis-driven experiment. For experiment standards (rigor, classification, analysis), see [docs/standards/experiments.md](../standards/experiments.md). For the experiment template, see [docs/templates/hypothesis.md](../templates/hypothesis.md).

## When to Run Experiments

- Validating that a new feature works as designed (post-PR confirmation)
- Testing intuitive claims about system behavior (from `docs/plans/research.md`)
- Investigating unexpected behavior observed during development
- Exploring design tradeoffs between configurations

## The Three-Round Protocol

Every hypothesis experiment goes through **at least three rounds** of experimentation interleaved with external LLM review, continuing **until convergence**. This is non-negotiable — it exists because single-pass experiments produce plausible-sounding but incorrect analyses (evidence: H5 and H10 in PR #310 required 4 rounds to reach correct conclusions).

```
Round 1: Design → Run → Analyze → Document draft FINDINGS.md
                ↓
         External Review (Opus 4.6 via /review-plan)
                ↓
Round 2: Address review gaps → Run additional experiments → Update FINDINGS.md
                ↓
         External Review (Opus 4.6 via /review-plan)
                ↓
Round 3: Resolve remaining questions → Final experiments if needed → Finalize FINDINGS.md
                ↓
         External Review (Opus 4.6 — convergence check)
                ↓
         Converged? → Commit and PR
         Not converged? → Round 4+ (repeat until converged)
```

### Convergence Criterion

An experiment **converges** when the reviewer's only remaining items are **"acknowledged as open"** — not **"needs a new experiment."** Specifically:

- **Converged**: "The directional question (why fewer cache hits helps) remains open and would require per-request logging to resolve." → This is acknowledged. No further rounds needed.
- **Not converged**: "There is a confounding variable — you need a control with routing=round-robin." → This is an actionable experiment. Another round required.

The distinction: **"open and requires different tooling"** is a stopping point. **"Open and answerable by running another experiment"** is not.

Three rounds is the **minimum**, not the target. Most experiments will converge in 3 rounds. Some may need 4-5. The process stops when the review produces no new actionable experiments, regardless of round count.

### Round 1: Initial Experiment

1. **Select or pose hypothesis** — from `docs/plans/research.md` or from a new observation
2. **Classify** — deterministic or statistical? If statistical, which subtype? (See [experiments.md](../standards/experiments.md))
3. **Design experiment** — ED-1 through ED-6
4. **Implement** — create `hypotheses/<name>/run.sh`, `analyze.py`
5. **Code review experiment code** — BEFORE running. See [Code Review Before Execution](#code-review-before-execution) below.
6. **Run** — execute across required seeds; verify reproducibility (ED-5)
7. **Analyze** — produce comparison tables, compute effect sizes
8. **Verify root cause** — trace every causal claim through code (RCV-1, RCV-2, RCV-3)
9. **Document draft FINDINGS.md** — results, root cause, classification, standards audit

### Round 1 Review

Run external review: `/review-plan <path-to-findings> aws/claude-opus-4-6`

Focus areas for the reviewer:
- Are causal claims verified against code? (RCV-1)
- Are "surprises" computed from first principles? (RCV-2)
- Does the mechanism explain the direction, not just the correlation? (RCV-3)
- Are there confounding variables? (ED-1, ED-6)
- Are there missing control experiments?

### Round 2: Address Review Gaps

9. **Design additional experiments** to address review feedback:
   - Confound matrices (isolate variables the reviewer flagged)
   - Control experiments (disable proposed mechanism to verify causality)
   - Calibrated parameters (test with corrected/proper values)
10. **Run additional experiments**
11. **Update FINDINGS.md** — incorporate new results, correct or qualify earlier claims

### Round 2 Review

Run external review again on the updated FINDINGS.md.

Focus areas:
- Did the new experiments resolve the Round 1 gaps?
- Are there remaining directional questions (mechanism identified but effect direction unexplained)?
- Are findings appropriately qualified (confirmed vs partially confirmed vs open)?

### Round 3: Resolve or Acknowledge

12. **Final experiments** if Round 2 review identified remaining gaps
13. **Finalize FINDINGS.md** — every open question must be either resolved or explicitly marked as "open, requires future work" with a specific proposed experiment
14. **Classify findings** — confirmation, bug, new rule, new invariant, design limitation, or surprise
15. **Audit against standards** — check findings against `docs/standards/rules.md` and `docs/standards/invariants.md`
16. **File issues** — for any bugs (`--label bug`), design limitations (`--label design`), or new rules/invariants discovered

### Round 3 Review (Confirmation Pass)

Final external review — this is a confirmation pass, not a discovery pass. The reviewer should confirm:
- All previously flagged issues are addressed or explicitly acknowledged
- No new concerns
- FINDINGS.md is ready for merge

17. **Commit and PR** — rebase on upstream/main, push, create PR

## Code Review Before Execution

**Every `run.sh` and `analyze.py` must be code-reviewed BEFORE running experiments.** This is non-negotiable. The reviewer has a unique advantage over the findings reviewer: they can cross-reference experiment code against the simulator codebase to catch parser bugs that are invisible in FINDINGS.md.

Use code review skills (e.g., `/code-review`, `pr-review-toolkit:code-reviewer`, or manual review) on the experiment code. Repeat until clean.

### What to check

1. **Parser–output format agreement**: For every regex or field extraction in `analyze.py`, verify the pattern matches the actual output format in the simulator code.
   - `cmd/root.go` — what text does the CLI print? (e.g., `"Preemption Rate: %.4f"` at line 544)
   - `sim/metrics_utils.go` — what JSON fields exist? (e.g., `preemption_count` vs `Preemption Rate`)
   - Match every regex in `analyze.py` against the format string in the producer code

2. **CLI flag correctness**: For every flag in `run.sh`, verify the flag name and value match `cmd/root.go` defaults and help text. Check for typos that strict parsing would reject.

3. **Workload YAML field names**: Verify against `sim/workload/spec.go` struct tags. `KnownFields(true)` will reject typos at runtime, but catching them at review saves a failed experiment run.

4. **Config diff against referenced experiments (ED-6)**: If the experiment reuses calibration from a prior experiment, diff every flag. The reviewer should explicitly list differences.

5. **Seed and determinism**: Verify `--seed` is passed correctly and workload YAML `seed:` field doesn't conflict.

### Evidence: what this step would have caught

| Bug | Round discovered | Would code review have caught it? |
|-----|-----------------|-----------------------------------|
| YAML `input_dist` vs `input_distribution` (H5) | Round 1 run failure | **Yes** — cross-ref against `spec.go` struct tags |
| Analyzer regex `Preemptions?: (\d+)` vs actual `Preemption Rate: 0.1750` (H10) | Round 4 | **Yes** — cross-ref against `cmd/root.go:544` format string |
| H10 routing policy mismatch with H8 | Round 2 | **Yes** — ED-6 config diff |
| H5 bucket cap=500 < mean_input=512 | Round 2 | **Possibly** — first-principles check on parameters |

Three of the four major bugs in this PR would have been caught by code review before a single experiment ran. The analyzer bug alone cost two rounds of incorrect conclusions.

## Quality Gates

### Pre-Execution Gates (check BEFORE running experiments)
- [ ] `run.sh` flags verified against `cmd/root.go` help text
- [ ] `analyze.py` regexes verified against actual output format strings in `cmd/root.go` and `sim/metrics_utils.go`
- [ ] Workload YAML field names verified against `sim/workload/spec.go` struct tags
- [ ] Config diff against referenced experiments documented (ED-6)
- [ ] Code review completed (at least one pass)

### Per-Round Gates (check after each round)
- [ ] Every causal claim cites `file:line` (RCV-1)
- [ ] Every "surprise" has a first-principles calculation (RCV-2)
- [ ] Root cause explains mechanism AND direction (RCV-3)
- [ ] External review completed and feedback addressed

### Final Gates (check before PR)
- [ ] Hypothesis classified (deterministic or statistical + subtype)
- [ ] Experiment design follows ED-1 through ED-6
- [ ] If reusing prior calibration data, config diff documented (ED-6)
- [ ] Results reproducible via `./run.sh`
- [ ] At least three rounds completed with external reviews
- [ ] **Convergence reached**: reviewer's only remaining items are "acknowledged as open" (requires different tooling), not "needs a new experiment"
- [ ] All review feedback addressed or explicitly acknowledged as open
- [ ] Findings classified per the findings table
- [ ] Standards audit completed
- [ ] Issues filed for all actionable findings

## Why At Least Three Rounds?

Evidence from PR #310 (H5, H10, H13):

| Round | What happened | What was caught |
|-------|---------------|-----------------|
| **1** | Initial experiments + FINDINGS.md | Plausible but wrong root causes published |
| **2** | Code review + external review | H10: "capacity increase" was wrong (NewKVStore doesn't change GPU blocks). H5: "96% surprise" was mathematically inevitable. |
| **3** | Confound matrix + calibrated bucket | H10: `maybeOffload` confirmed as sole mechanism via byte-identical control. H5: calibrated bucket shows <5% effect — original was load shedding. |
| **After 3** | H10 directional question remains | "Why do fewer cache hits improve TTFT?" — requires code instrumentation, not experiments. **Converged**: acknowledged as open. |

Round 1 produced confident but wrong answers. Round 2 identified the errors. Round 3 produced definitive evidence. H10 converged after Round 3 despite having an open question, because the remaining question requires different tooling (per-request logging), not another experiment sweep.

**Three rounds is the minimum, not the maximum.** The stopping criterion is convergence: the reviewer raises no new actionable experiments.

## References

- Standards: [docs/standards/experiments.md](../standards/experiments.md)
- Template: [docs/templates/hypothesis.md](../templates/hypothesis.md)
- Hypothesis catalog: [docs/plans/research.md](../plans/research.md)
- Validated experiments:
  - `hypotheses/h3-signal-freshness/` — signal freshness (Tier 2, statistical dominance)
  - `hypotheses/h8-kv-pressure/` — KV cache pressure (Tier 3, statistical monotonicity)
  - `hypotheses/h9-prefix-caching/` — prefix caching effectiveness (Tier 2, statistical monotonicity)
  - `hypotheses/h12-conservation/` — request conservation invariant (Tier 1, deterministic)
  - `hypotheses/h14-pathological-templates/` — pathological policy templates (Tier 2, statistical dominance)
  - `hypotheses/prefix-affinity/` — prefix-affinity routing (Tier 2, statistical dominance)
  - `hypotheses/h5-token-bucket-burst/` — token-bucket admission under burst (Tier 3, statistical dominance)
  - `hypotheses/h10-tiered-kv/` — tiered KV cache GPU+CPU offload (Tier 3, statistical dominance)
  - `hypotheses/h13-determinism/` — determinism invariant INV-6 (Tier 1, deterministic)
