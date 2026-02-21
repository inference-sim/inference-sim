# Hypothesis Experiment Process

This document describes the process for running a hypothesis-driven experiment. For experiment standards (rigor, classification, analysis), see [docs/standards/experiments.md](../standards/experiments.md). For the experiment template, see [docs/templates/hypothesis.md](../templates/hypothesis.md).

## When to Run Experiments

- Validating that a new feature works as designed (post-PR confirmation)
- Testing intuitive claims about system behavior (from `docs/plans/research.md`)
- Investigating unexpected behavior observed during development
- Exploring design tradeoffs between configurations
- Filling coverage gaps identified in the [family coverage table](../../hypotheses/README.md)

## Generating Hypotheses

Hypotheses can come from **internal** sources (your own experiments and development) or **external** sources (user questions, literature, analytical models). This section provides structured guidance for generating good hypotheses. See also [experiments.md](../standards/experiments.md) for family-specific sentence patterns.

### Sources of hypotheses

| Source | How it works | Example |
|--------|-------------|---------|
| **User intuition** | "I think X should be better than Y because of Z" | "SJF should reduce TTFT for mixed workloads because short jobs finish first" |
| **Coverage gaps** | Check the [family coverage table](../../hypotheses/README.md) for untested families | Workload/arrival family has 0 experiments → "Gamma sampler should match theoretical CV" |
| **Experiment findings** | Surprises and open questions from completed experiments spawn follow-up hypotheses | H10's maybeOffload finding → "test at GPU=1500 for preemption-path offload" |
| **Bug reports** | "This behavior seems wrong" → formalize as a testable claim | H12: preemption panic → "conservation should hold even under preemption pressure" |
| **Analytical models** | Divergence between theory and simulation → "does the DES match M/M/k under matching assumptions?" | "Under Poisson arrivals, queue length should match M/M/k within 5%" |
| **Literature / external** | Published results about inference serving systems | "Prefix caching should reduce TTFT proportional to prefix length (as in vLLM literature)" |
| **Design docs** | Claims made in design documents that have never been validated | "The composable scorer framework should produce Pareto-optimal configurations" |

### What makes a good hypothesis

A good hypothesis is **behavioral** (about observable system behavior), **testable** (with a clear experiment), and **diagnostic** (failure points to something worth investigating).

| Criterion | Good | Bad |
|-----------|------|-----|
| **Behavioral** | "Burst smoothing should reduce tail latency" | "The token bucket decrements currentTokens correctly" |
| **Testable** | "TTFT should decrease monotonically as prefix_length increases" | "The system should be fast" |
| **Diagnostic** | "If this fails, it indicates the cache eviction path has a bug" | "If this fails, something is wrong" |
| **Conceptual** | "Tiered storage should reduce preemptions" | "kvcache_tiered.go:224 should delete the hash" |
| **Intuitive** | "More instances should roughly halve latency under saturation" | "The event queue should process 2x events" |

### Anti-patterns in hypothesis generation

| Anti-pattern | Problem | Fix |
|-------------|---------|-----|
| **Code-grounded hypothesis** | Tests implementation, not behavior. Prevents discovery of design gaps. | Pose the hypothesis WITHOUT reading the code first. |
| **Unfalsifiable hypothesis** | "The system should work correctly" — no way to fail | Specify a concrete metric and direction: "TTFT P99 should be lower for A than B" |
| **Hypothesis that tests the obvious** | "More resources should improve performance" — trivially true | Add a diagnostic clause: "...and the improvement should be proportional to the resource increase (not sub-linear due to contention)" |
| **Hypothesis with no failure action** | Confirmation and refutation both lead to "ok, noted" | Every hypothesis should specify: "If this fails, investigate X" |
| **Over-scoped hypothesis** | "The entire system should be correct under all configurations" | Decompose by family: scheduler invariant + structural model + robustness are separate experiments |

### How to propose a new hypothesis

1. **Check coverage**: Read the [family coverage table](../../hypotheses/README.md). Prioritize families with low coverage.
2. **Choose a family**: Which domain does your claim target? (See [experiments.md](../standards/experiments.md) for the 6 families.)
3. **Write the sentence**: Use the family-specific pattern from experiments.md.
4. **Add the diagnostic clause**: "If this fails, it would indicate..."
5. **Check for redundancy**: Search existing hypotheses in `docs/plans/research.md` and on GitHub: [issues labeled `hypothesis`](https://github.com/inference-sim/inference-sim/labels/hypothesis).
6. **File as a GitHub issue**: Use the [Hypothesis Proposal issue template](../../.github/ISSUE_TEMPLATE/hypothesis.md) on GitHub (click "New Issue" → "Hypothesis Proposal"). This template has fields for family, VV&UQ category, diagnostic value, and experiment design.

External contributors should file a GitHub issue using the Hypothesis Proposal template. Maintainers will triage, prioritize, and run the review protocol.

## The Iterative Review Protocol

Every hypothesis experiment goes through **iterative rounds** of experimentation interleaved with **three parallel external reviews**, continuing **until convergence** (max 10 rounds). There is no minimum round count — if three thorough reviewers all converge in Round 1, the experiment is done.

```
Round N:
  Design → Code Review → Run → Analyze → Document FINDINGS.md
                          ↓
              ┌───────────┼───────────┐
              ↓           ↓           ↓
         Reviewer A   Reviewer B  Reviewer C     (3 parallel Opus 4.6 reviews)
         (mechanism)  (design)    (rigor)
              ↓           ↓           ↓
              └───────────┼───────────┘
                          ↓
                  All 3 converged? → Commit and PR
                  Any has actionable experiment? → Round N+1
                  Round 10 reached? → Stop, document remaining gaps
```

### Three Parallel Reviewers

Each round runs **three Opus 4.6 reviews in parallel**, each with a different (but overlapping) focus area. This catches what single-reviewer sequential rounds miss — different reviewers have different blind spots.

Run all three in parallel: `/review-plan <findings> aws/claude-opus-4-6`

**For external contributors without AI review infrastructure:** Submit your FINDINGS.md via PR. Maintainers will run the three-reviewer protocol on your behalf. You can also conduct the three reviews manually by having three different people review the FINDINGS.md, each focusing on one of the three areas below.

**Reviewer A — Mechanism Verification:**
- Are causal claims traced through code with `file:line` citations? (RCV-1)
- Does the mechanism explain the direction, not just the correlation? (RCV-3)
- Is there a control experiment that disables only the proposed mechanism? (RCV-4)
- Do the code paths cited actually produce the claimed behavior?

**Reviewer B — Experimental Design:**
- Are there confounding variables? (ED-1, ED-6)
- Are there missing control experiments or confound matrix cells?
- Are parameters properly calibrated? (e.g., bucket cap vs mean input)
- Is the config diff against referenced experiments documented? (ED-6)

**Reviewer C — Statistical Rigor & Generalizability:**
- Are "surprises" computed from first principles? (RCV-2)
- Is the sample size adequate (seeds, operating points)?
- Are claims properly scoped (not over-generalized from narrow evidence)?
- Is the evidence quality table complete and honest?

Each reviewer's prompt should include their focus area AND the full FINDINGS.md. The overlapping coverage means any critical issue is likely caught by at least one reviewer.

### Convergence Criterion

An experiment **converges** when **all three reviewers** have no remaining items that require a new experiment. Specifically:

- **Converged**: All three reviewers' remaining items are "acknowledged as open" (requires different tooling) or documentation fixes. No new experiments needed.
- **Not converged**: Any reviewer raises an actionable experiment (confound matrix, control, calibrated parameters, etc.). Another round required.

The distinction: **"open and requires different tooling"** is a stopping point. **"Open and answerable by running another experiment"** is not.

**Max 10 rounds.** If convergence is not reached by Round 10, stop and document remaining gaps as future work. This prevents unbounded iteration on irreducibly complex systems.

### Round Structure (each round)

**Steps 1-4 apply to Round 1 only. Subsequent rounds start at step 5.**

1. **Select or pose hypothesis** — from `docs/plans/research.md` or from a new observation
2. **Classify** — (a) which hypothesis family? (b) Verification, Validation, or UQ? (c) deterministic or statistical? If statistical, which subtype? The family determines design rules; the VV&UQ category determines evidence requirements. (See [experiments.md](../standards/experiments.md))
3. **Design experiment** — ED-1 through ED-6, with family-specific considerations
4. **Implement** — create `hypotheses/<name>/run.sh`, `analyze.py`
5. **Code review experiment code** — BEFORE running. See [Code Review Before Execution](#code-review-before-execution) below.
6. **Run** — execute across required seeds; verify reproducibility (ED-5)
7. **Analyze** — produce comparison tables, compute effect sizes
8. **Verify root cause** — trace every causal claim through code (RCV-1, RCV-2, RCV-3)
9. **Document FINDINGS.md** — results, root cause, classification, standards audit
10. **Three parallel external reviews** — run Reviewers A, B, C simultaneously
11. **Assess convergence** — if all three converge, proceed to finalization. If any has actionable feedback, start next round at step 5.

### Finalization (after convergence)

12. **Classify findings** — confirmation, bug, new rule, new invariant, design limitation, surprise, or open question
13. **Audit against standards** — check findings against `docs/standards/rules.md` and `docs/standards/invariants.md`
14. **Assess promotion to test suite** — see [Promotion of Confirmed Hypotheses](#promotion-of-confirmed-hypotheses) below
15. **Commit and PR** — rebase on upstream/main, push, create PR
16. **File issues** — AFTER PR creation, file structured issues per the [Issue Taxonomy](#issue-taxonomy-after-convergence) below. Reference the PR in each issue. Include promotion issues for any hypotheses identified in step 14.

**Why issues come last:** Findings can change across rounds (H10 went from "untested" to "confirmed" between Rounds 3-4). Filing issues before convergence risks creating wrong issues that need to be closed and re-filed. File once, file right.

### Issue Taxonomy (after convergence)

After convergence and PR creation, walk the findings classification table in FINDINGS.md and file one GitHub issue per actionable finding. Not every hypothesis produces issues — a clean confirmation (like H13) may produce none.

**Issue types and labels:**

| Issue Type | Label | When to file | Title format | Example |
|------------|-------|-------------|--------------|---------|
| **Bug** | `--label bug` | Code defect discovered during experiment | `bug: <component> — <defect>` | `bug: sim/simulator.go — preempt() panics on empty RunningBatch` (H12) |
| **Enhancement** | `--label enhancement` | New feature, rule, or documentation improvement needed | `enhancement: <area> — <improvement>` | `enhancement: CLI — document token-bucket per-input-token cost model` (H5) |
| **New hypothesis** | `--label hypothesis` | Follow-up experiment spawned by current findings | `hypothesis: <claim to test>` | `hypothesis: test tiered KV at GPU=1500 blocks to trigger preemption-path offload` (H10) |
| **Design limitation** | `--label design` | System works as coded but has undocumented behavioral limitation | `design: <limitation>` | `design: no burst-smoothing sweet spot under Gamma CV>3` (H5) |
| **Standards update** | `--label standards` | New rule or invariant discovered that should be added | `standards: <rule/invariant>` | `standards: R17 signal freshness — routing signals have tiered staleness` (H3) |

**Mapping from resolution type to expected issues:**

| Resolution | Expected issues |
|------------|----------------|
| Clean confirmation | Usually none. Optionally: standards update confirming existing rules. |
| Confirmation with wrong mechanism | Enhancement: update documentation with correct mechanism. |
| Confirmation with bug discovery | Bug: one per code defect. Enhancement: if detector/tooling needs improvement. |
| Partial confirmation with surprise | New hypothesis: follow-up experiments to investigate surprise. |
| Refuted — system design flaw | Design: architectural limitation. Enhancement: proposed fix. |
| Refuted — mechanism not plausible | Design: document the limitation. Enhancement: update CLI help or user docs if the mechanism assumption stems from misleading parameter names. |
| Refuted — wrong mental model | Usually none. Optionally: enhancement if CLI help text is misleading. |
| Inconclusive — parameter-dependent | New hypothesis: test at different parameters. |
| Converged to open question | New hypothesis: specific experiment or tooling to resolve. |

**Issue body template:**

```markdown
## Context
Discovered in hypothesis experiment <name> (PR #NNN).

## Finding
<One-paragraph description from FINDINGS.md>

## Evidence
<Key data point or code reference>

## Proposed action
<What should be done — fix, new experiment, documentation update>
```

**What NOT to file:**
- Issues for findings that are "documented here" with no action needed
- Duplicate issues for findings already covered by existing open issues
- Issues for scope limitations that are acknowledged in FINDINGS.md (these are future work, not bugs)

## Promotion of Confirmed Hypotheses

After convergence, assess whether any confirmed findings should be promoted from bash-script experiments to the Go test suite and/or formal invariants. Hypothesis experiments run as bash scripts are NOT in CI — a regression would not be caught by `go test ./...`.

### When to promote

| Condition | Promote to | Why |
|-----------|-----------|-----|
| Confirmed deterministic hypothesis | **Go test** (regression protection in CI) | Deterministic properties are exact — they can be encoded as pass/fail tests. Bash experiments catch them today; Go tests catch them on every commit. |
| Deterministic invariant aspect of a statistical hypothesis | **Go test** for the invariant aspect | Statistical hypotheses often contain deterministic sub-claims (e.g., conservation holds across all configs tested). The invariant aspect is promotable even if the full comparison isn't. |
| New invariant discovered | **`docs/standards/invariants.md`** entry | Codify as a formal system property with verification strategy. |
| New rule discovered | **`docs/standards/rules.md`** entry | Codify as an antipattern check for PR reviews. |

### Promotion assessment for PR #310 hypotheses

| Hypothesis | Promotable aspect | Current Go test coverage | Promotion needed? |
|---|---|---|---|
| H12 (Conservation) | INV-1 across 10 cluster-level policy combinations | Per-instance conservation tested; **cluster-level multi-config NOT in Go tests** | **Yes** — file `--label enhancement` issue |
| H13 (Determinism) | INV-6: run twice with same seed → byte-identical stdout | Golden dataset comparison exists; **"run twice, diff" NOT in Go tests** | **Yes** — file `--label enhancement` issue |
| H10 (Tiered KV) | INV-1 holds across all tiered configurations | Not tested in Go with tiered cache enabled | **Yes** — file `--label enhancement` issue |
| H8 (KV pressure) | INV-1 holds under preemption pressure | Not tested in Go under KV-constrained configs | **Yes** — file `--label enhancement` issue |
| H5 (Token-bucket) | completed + rejected == total for admission configs | Not tested in Go with token-bucket | **Yes** — file `--label enhancement` issue |

### What a promoted test looks like

A promoted hypothesis test is a Go test that encodes the deterministic invariant verified by the experiment:

```go
// TestClusterConservation_AcrossPolicyCombinations tests INV-1 at cluster level.
// Promoted from hypothesis H12 (hypotheses/h12-conservation/).
func TestClusterConservation_AcrossPolicyCombinations(t *testing.T) {
    configs := []struct{ routing, scheduler, admission string }{
        {"round-robin", "fcfs", "always-admit"},
        {"least-loaded", "fcfs", "always-admit"},
        {"weighted", "priority-fcfs", "token-bucket"},
        // ... all 10 H12 configurations
    }
    for _, cfg := range configs {
        t.Run(cfg.routing+"/"+cfg.scheduler+"/"+cfg.admission, func(t *testing.T) {
            // Run cluster simulation
            // Assert: injected == completed + still_queued + still_running
        })
    }
}
```

The bash experiment remains as the full reproducible artifact with analysis. The Go test is the CI-integrated regression guard.

## Code Review Before Execution

**Every `run.sh` and `analyze.py` must be code-reviewed BEFORE running experiments.** This is non-negotiable. The reviewer has a unique advantage over the findings reviewer: they can cross-reference experiment code against the simulator codebase to catch parser bugs that are invisible in FINDINGS.md.

Use code review skills (e.g., `/code-review`, `pr-review-toolkit:code-reviewer`, or manual review) on the experiment code. Repeat until clean.

### What to check

1. **Parser–output format agreement**: For every regex or field extraction in `analyze.py`, verify the pattern matches the actual output format in the simulator code.
   - `cmd/root.go` — what text does the CLI print? (e.g., `"Preemption Rate: %.4f"` at line 544)
   - `sim/metrics_utils.go` — what JSON fields exist? (e.g., `preemption_count` vs `Preemption Rate`)
   - Match every regex in `analyze.py` against the format string in the producer code
   - **Silent defaults**: Verify that when a regex matches nothing, `analyze.py` emits a warning to stderr rather than silently defaulting to 0. A silent default can mask real data for multiple review rounds (H10: analyzer reported 0 preemptions for 2 rounds because the regex never matched).

2. **CLI flag correctness**: For every flag in `run.sh`, verify the flag name and value match `cmd/root.go` defaults and help text. Check for typos that strict parsing would reject.

3. **Workload YAML field names**: Verify against `sim/workload/spec.go` struct tags. `KnownFields(true)` will reject typos at runtime, but catching them at review saves a failed experiment run.

4. **Config diff against referenced experiments (ED-6)**: If the experiment reuses calibration from a prior experiment, diff every flag. The reviewer should explicitly list differences. The `run.sh` must include an explicit `# Reference: hypotheses/<name>/run.sh` comment with the file path, so the diff is easy to perform.

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
- [ ] **Convergence reached**: all three parallel reviewers have no remaining actionable experiments
- [ ] All review feedback addressed or explicitly acknowledged as open
- [ ] Findings classified per the findings table (including resolution type)
- [ ] Standards audit completed

### Post-PR Gates (check after PR creation)
- [ ] Issues filed per [Issue Taxonomy](#issue-taxonomy-after-convergence) — one per actionable finding
- [ ] Each issue has correct label (`bug`, `enhancement`, `hypothesis`, `design`, or `standards`)
- [ ] Each issue references the PR number
- [ ] No issues filed for "documented here" findings with no action needed

## Why Three Parallel Reviewers?

Evidence from PR #310 multi-model reviews:

| Reviewer | Unique insight no other reviewer caught |
|----------|----------------------------------------|
| **Gemini 3 Pro** | Queue-time vs compute-time split for H10; hard-block behavior for H5 |
| **Claude Opus 4** | Process enforcement gaps; analysis paralysis risk |
| **Claude Opus 4.6** | Confound matrix design; cap < mean_input is structural rejection; ED-7 pre-registration |
| **GPT-4o** | H13 scope narrowness; dynamic bucket policies |

No single reviewer caught all issues. Three parallel reviewers with different focus areas (mechanism, design, rigor) maximize coverage per round, potentially converging faster than sequential single-reviewer rounds.

## Why Iterate Until Convergence (Not Fixed Rounds)?

Evidence from PR #310 (H5, H10, H13):

| Round | What happened | What was caught |
|-------|---------------|-----------------|
| **1** | Initial experiments | Wrong root causes for H5 and H10 |
| **2** | Code + external review | Corrected math (H5), identified mechanism (H10), designed confound matrix |
| **3** | Confound matrix + calibrated bucket | H5 burst-smoothing mechanism refuted (directional prediction holds), H10 analyzer bug masked preemptions |
| **4** | Corrected analyzer | H10 confirmed — preemptions DO occur, cache hits INCREASE |

H13 converged in Round 1 (deterministic = pass/fail). H5 converged in Round 3. H10 required Round 4 due to an analyzer bug. Fixed round counts would have either stopped too early (missing the H10 bug) or forced unnecessary work (H13 didn't need Round 2).

**Iterate until convergence, max 10 rounds.** Three parallel reviewers per round. No minimum.

## References

- Standards: [docs/standards/experiments.md](../standards/experiments.md)
- Template: [docs/templates/hypothesis.md](../templates/hypothesis.md)
- Hypothesis catalog: [docs/plans/research.md](../plans/research.md)
- Validated experiments (by family):
  - **Scheduler invariants:** `h12-conservation/` (Tier 1, deterministic), `h13-determinism/` (Tier 1, deterministic)
  - **Structural model:** `h3-signal-freshness/` (Tier 2, dominance), `h9-prefix-caching/` (Tier 2, monotonicity), `h10-tiered-kv/` (Tier 3, dominance)
  - **Performance-regime:** `h8-kv-pressure/` (Tier 3, monotonicity)
  - **Robustness/failure-mode:** `h14-pathological-templates/` (Tier 2, dominance), `h5-token-bucket-burst/` (Tier 3, dominance)
  - **Cross-policy comparative:** `prefix-affinity/` (Tier 2, dominance)
  - **Workload/arrival:** *(none yet)*
