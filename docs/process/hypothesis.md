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
5. **Run** — execute across required seeds; verify reproducibility (ED-5)
6. **Analyze** — produce comparison tables, compute effect sizes
7. **Verify root cause** — trace every causal claim through code (RCV-1, RCV-2, RCV-3)
8. **Document draft FINDINGS.md** — results, root cause, classification, standards audit

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

## Quality Gates

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
