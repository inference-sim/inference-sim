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

Every hypothesis experiment goes through **iterative rounds** of experimentation interleaved with **five parallel internal reviews**, continuing **until convergence** (max 10 rounds). There is no minimum round count — if no reviewer flags any CRITICAL or IMPORTANT item in Round 1, the experiment is done.

```
Round N:
  Design → Code Review → Run → Analyze → Document FINDINGS.md
                          ↓
         ┌────────┬───────┼───────┬────────┐
         ↓        ↓       ↓       ↓        ↓
       Code    Experiment  Stats  Control  Standards     (5 parallel Task agents)
       Verify  Designer   Rigor  Auditor  Compliance
         ↓        ↓       ↓       ↓        ↓
         └────────┴───────┼───────┴────────┘
                          ↓
                  0 CRITICAL + 0 IMPORTANT? → Commit and PR
                  Any has actionable experiment? → Round N+1
                  Round 10 reached? → Stop, document remaining gaps
```

### Five Parallel Internal Reviewers

Each round runs **five internal Task agents in parallel**, each with a different (but overlapping) focus area. Internal agents can read the actual source files, verify `file:line` citations, cross-reference analyzer regexes against `cmd/root.go` format strings, and check workload YAML field names against `sim/workload/spec.go` struct tags — capabilities that external LLM reviews lack.

**How to run:** Launch all five as background Task agents. Each agent receives the FINDINGS.md path and its specific focus area. Collect results and assess convergence when all five complete.

```
# Launch all 5 in parallel (one Task tool call per reviewer):
Task(subagent_type="general-purpose", run_in_background=True,
     prompt="You are Reviewer N (<role>). Read FINDINGS.md at <path>.
             <reviewer-specific checklist from below>
             Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
             Report: (1) list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.")
```

**Timeout:** 5 minutes per reviewer agent. If an agent exceeds this, check its output file and restart if stalled.

**Failure handling:** If a reviewer agent fails or hangs, fall back to performing that review directly (read the FINDINGS.md yourself with that reviewer's checklist). Do not skip a reviewer perspective.

**Expected output format:** Each reviewer produces a list of findings, each classified as CRITICAL/IMPORTANT/SUGGESTION, with a summary count. Convergence assessment: sum CRITICAL and IMPORTANT counts across all 5 reviewers. If the sum is zero, the experiment converges.

**For external contributors without AI review infrastructure:** Submit your FINDINGS.md via PR. Maintainers will run the five-reviewer protocol on your behalf. You can also conduct the reviews manually by having people review the FINDINGS.md, each focusing on one of the five areas below.

**Reviewer 1 — Code Verifier:**
- READ the actual source files cited in the FINDINGS.md. Verify every `file:line` citation against current code.
- Does the code at the cited location actually produce the claimed behavior?
- Are there off-by-one errors in line citations? (Acceptable: ±2 lines. Flag: >2 lines off.)
- Does the mechanism explanation match what the code does, not just what it's named?

**Reviewer 2 — Experiment Designer:**
- Are there confounding variables? Is exactly one dimension varied? (ED-1)
- Was the experiment run at a rate where the effect should vanish, to confirm mechanism dependence? (ED-2)
- Are experiment preconditions verified in the script (e.g., queue depth > batch size for SJF tests)? (ED-3)
- Is workload seed handling correct? Does `--seed` on the CLI properly vary across runs? (ED-4)
- Is the experiment reproducible from `run.sh` alone — binary built, seeds documented, no manual steps? (ED-5)
- Is the config diff against referenced experiments documented? (ED-6)
- Are there missing control experiments or confound matrix cells?
- Are parameters properly calibrated? (e.g., bucket cap vs mean input)
- Cross-reference every CLI flag in `run.sh` against `cmd/root.go` flag definitions.
- Cross-reference every YAML field name against `sim/workload/spec.go` struct tags.

**Reviewer 3 — Statistical Rigor:**
- Are "surprises" computed from first principles? (RCV-2)
- Is the sample size adequate (seeds, operating points)?
- Are claims properly scoped (not over-generalized from narrow evidence)?
- Is the evidence quality table complete and honest?
- Do per-seed effect sizes meet the legacy thresholds (>20% for dominance, <5% for equivalence)?
- Is the status classification consistent with the data? (e.g., "Confirmed" requires >20% in ALL seeds.)

**Reviewer 4 — Control Experiment Auditor:**
- Does every proposed mechanism (RCV-3: a specific code path claimed to cause the observed effect) have a corresponding control experiment (RCV-4: disabling only that mechanism)?
- Were control experiments actually EXECUTED, not just proposed? Look for conditional language ("one would test", "could be confirmed by") vs past tense with data ("the control showed 0.0% difference"). Verify that control results appear in the Results section with actual numbers, not just in Root Cause Analysis as narrative.
- Does each control isolate exactly one variable? Diff the CLI flags between treatment and control runs in `run.sh` — the control should differ by exactly one flag or parameter.
- Do the control results confirm or refute the proposed mechanism?
- Do the control experiment results in the Evidence Quality table accurately reflect the current round (not stale text from a prior round)?
- Does the mechanism explain the direction using experimental evidence (e.g., "disabling the mechanism reversed the effect"), not just code-reading claims? (This complements Reviewer 1, who verifies the mechanism via code; Reviewer 4 verifies it via experimental data.)

**Reviewer 5 — Standards Compliance:**
- Are ALL 12 FINDINGS.md sections present and non-empty? (per `docs/templates/hypothesis.md`)
- Is the hypothesis correctly classified (family, VV&UQ category, type)?
- Does the Devil's Advocate section (RCV-5) argue both directions convincingly?
- Are scope and limitations (RCV-6) complete — operating point, dependencies, what was NOT tested, generalizability, UQ?
- Does the standards audit correctly check findings against `docs/standards/rules.md` and `docs/standards/invariants.md`?
- Are any new rules or invariants warranted by the findings?

The overlap is intentional — multiple perspectives checking the same FINDINGS.md means different reviewers catch different issues. Evidence from PR #385: Reviewer 1 (code) and Reviewer 3 (rigor) both caught H19's stale evidence quality row; Reviewer 2 (design) caught H16's sub-threshold seed that Reviewers 1 and 3 missed; Reviewer 4 (control) caught H21's unexecuted control experiments that Reviewers 2 and 3 didn't flag as actionable.

### Convergence Defined

An experiment **converges** when **no reviewer flags any CRITICAL or IMPORTANT item in the current round**. Convergence is not about agreement between reviewers — it is about the absence of actionable findings.

- **Converged**: Zero CRITICAL items and zero IMPORTANT items across all five reviewers. Remaining items are SUGGESTION-level (documentation nits, cosmetic fixes, minor citation off-by-ones). No new experiments needed.
- **Not converged**: Any reviewer flags a CRITICAL or IMPORTANT item (missing control experiment, sub-threshold effect size, stale claims contradicted by data, unexecuted proposed experiments). Another round required.

**Severity levels** (each reviewer must classify every finding):
- **CRITICAL**: Must fix before merge. Examples: missing control experiment (RCV-4), status classification contradicted by data, silent data loss in analyzer, cross-document contradiction on protocol definition.
- **IMPORTANT**: Should fix before merge. The key test: **would merging with this unfixed item mislead a reader or produce incorrect conclusions?** Examples: sub-threshold effect size in one seed (misleads on significance), stale text contradicting Round 2 results (factual error in FINDINGS), undocumented confound (incomplete experiment), missing ED coverage in reviewer prompt (gap in review process).
- **SUGGESTION**: Does not affect correctness or reader understanding. Examples: off-by-one line citation (±2 lines), cosmetic terminology, additional scoping qualifier, style consistency.

**When in doubt between IMPORTANT and SUGGESTION:** If fixing the item would change any conclusion, metric, or user guidance in FINDINGS.md, it is IMPORTANT. If it would only improve readability or precision without changing any conclusion, it is SUGGESTION. If multiple reviewers classify the same item at different severities, the highest severity applies.

**Note on broadening from the old definition:** The old convergence criterion (pre-v2) was "no remaining items that require a new experiment." The new criterion (zero CRITICAL + zero IMPORTANT) is intentionally broader — it also blocks on factual errors in documentation (e.g., stale claims) even when no new experiment is needed. This prevents shipping FINDINGS.md with incorrect text that readers would trust.

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
10. **Five parallel internal reviews** — launch Reviewers 1-5 as background Task agents simultaneously
11. **Assess convergence** — if zero CRITICAL and zero IMPORTANT items across all five reviewers, proceed to finalization. If any reviewer flags a CRITICAL or IMPORTANT item, start next round at step 5.

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
| **Promotion** | `--label promotion` | Confirmed hypothesis finding promoted from bash experiment to Go test suite (see [Promotion of Confirmed Hypotheses](#promotion-of-confirmed-hypotheses)) | `enhancement: promote <hypothesis> <finding> to Go test suite` | `enhancement: promote H-Overload conservation under 10x to Go test suite` (#337) |

**Mapping from resolution type to expected issues:**

| Resolution | Expected issues |
|------------|----------------|
| Clean confirmation | Usually none. Optionally: promotion to Go test suite, standards update confirming existing rules. |
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
- [ ] `run.sh` sources `hypotheses/lib/harness.sh` and uses `blis_run` for every simulation call
- [ ] Every `blis_run` call has an appropriate timeout tier (`TIMEOUT_QUICK`/`TIMEOUT_STANDARD`/`TIMEOUT_EXTENDED`)
- [ ] KV safety pre-flight: if experiment uses `--total-kv-blocks`, call `preflight_kv_check` with max expected input tokens
- [ ] `analyze.py` imports `analyze_helpers` and uses `parse_blis_output` (handles timeouts gracefully)
- [ ] `run.sh` flags verified against `cmd/root.go` help text
- [ ] `analyze.py` regexes verified against actual output format strings in `cmd/root.go` and `sim/metrics_utils.go`
- [ ] Workload YAML field names verified against `sim/workload/spec.go` struct tags
- [ ] Config diff against referenced experiments documented (ED-6)
- [ ] Code review completed (at least one pass)

### Per-Round Gates (check after each round)
- [ ] Every causal claim cites `file:line` (RCV-1)
- [ ] Every "surprise" has a first-principles calculation (RCV-2)
- [ ] Root cause explains mechanism AND direction (RCV-3)
- [ ] Five internal reviewer assessments completed and all CRITICAL/IMPORTANT items addressed

### Final Gates (check before PR)
- [ ] Hypothesis classified (deterministic or statistical + subtype)
- [ ] Experiment design follows ED-1 through ED-6
- [ ] If reusing prior calibration data, config diff documented (ED-6)
- [ ] Results reproducible via `./run.sh`
- [ ] **Convergence reached**: zero CRITICAL and zero IMPORTANT items across all five internal reviewers in the current round
- [ ] All review feedback addressed or explicitly acknowledged as open
- [ ] Findings classified per the findings table (including resolution type)
- [ ] Standards audit completed

### Post-PR Gates (check after PR creation)
- [ ] Issues filed per [Issue Taxonomy](#issue-taxonomy-after-convergence) — one per actionable finding
- [ ] Each issue has correct label (`bug`, `enhancement`, `hypothesis`, `design`, or `standards`)
- [ ] Each issue references the PR number
- [ ] No issues filed for "documented here" findings with no action needed

## Why Five Internal Reviewers?

### Evolution from external to internal reviews

**v1 (PR #310):** Three external LLM reviews via `/review-plan` (Claude, GPT-4o, Gemini). External models caught different issues, proving the value of parallel review. However, none could read the codebase — citations like `sim/routing.go:180` were taken on faith.

**v2 (PR #385):** Five internal Task agents with codebase access. Each agent reads the actual source files, verifies `file:line` citations, cross-references regexes, and checks struct tags. This caught issues external reviews could never find.

### Evidence from PR #385 (internal reviews)

| Reviewer | Unique insight no other reviewer caught |
|----------|----------------------------------------|
| **Code Verifier** (H21) | Verified all 18 `file:line` citations against actual source. Found that `sim/routing.go:180` strict `>` causes positional bias — confirmed by reading the code, not by trusting the FINDINGS claim. |
| **Experiment Designer** (H16) | Seed 456 at 9% TTFT p99 difference — below the 10% "inconclusive" threshold. Caught by reading per-seed data and cross-referencing against `docs/standards/experiments.md` line 98. |
| **Statistical Rigor** (H16) | p99 from 500 requests = ~5 data points per seed. Flagged as insufficient for precise effect size, leading to the 2000-request Round 2 experiment that revealed the load-duration dependence. |
| **Control Auditor** (H21) | Caught that two control experiments were *proposed* but not *executed*. The FINDINGS said "one would test" — conditional language, not a report. Led to Round 2 with both controls actually run. |
| **Standards Compliance** (H19) | Found stale Round 1 text surviving into Round 2 FINDINGS — "control experiment proposed but not yet run" in the Evidence Quality table, when the control had been executed. |

### Why internal agents beat external LLMs

| Capability | External (`/review-plan`) | Internal (Task agent) |
|-----------|--------------------------|----------------------|
| Read source files | No | Yes — verifies every citation |
| Cross-ref regexes against format strings | No | Yes — catches analyzer bugs before they hide data |
| Check YAML fields against struct tags | No | Yes — catches typos that `KnownFields(true)` would reject at runtime |
| Run `grep` to verify claims | No | Yes — can search for "one would test" vs executed controls |
| API reliability | Fragile (auth, timeouts, rate limits) | Reliable (same process, no external dependency) |
| Cost | External API call per review | Included in session (no additional cost) |

### Why 5 instead of 3

Evidence from PR #385: with 3 reviewers (mechanism, design, rigor), the H19 experiment needed 3 separate review agents (Round 1 + two Round 2 reviews) to fully cover the space. The control experiment auditor perspective (Reviewer 4) and the standards compliance perspective (Reviewer 5) were not distinct roles — they were folded into "mechanism" and "rigor" respectively, where they received less attention. Splitting them into dedicated roles ensures:

- **RCV-4 control experiments** get a dedicated auditor who checks execution status, not just design
- **Template completeness** gets a dedicated checker who verifies all 12 FINDINGS.md sections, not just the ones related to mechanism or statistics

## Why Iterate Until Convergence (Not Fixed Rounds)?

Evidence from PR #310 (H5, H10, H13):

| Round | What happened | What was caught |
|-------|---------------|-----------------|
| **1** | Initial experiments | Wrong root causes for H5 and H10 |
| **2** | Code + external review | Corrected math (H5), identified mechanism (H10), designed confound matrix |
| **3** | Confound matrix + calibrated bucket | H5 burst-smoothing mechanism refuted (directional prediction holds), H10 analyzer bug masked preemptions |
| **4** | Corrected analyzer | H10 confirmed — preemptions DO occur, cache hits INCREASE |

H13 converged in Round 1 (deterministic = pass/fail). H5 converged in Round 3. H10 required Round 4 due to an analyzer bug. Fixed round counts would have either stopped too early (missing the H10 bug) or forced unnecessary work (H13 didn't need Round 2).

**Iterate until convergence, max 10 rounds.** Five parallel internal reviewers per round. No minimum.

## References

- Standards: [docs/standards/experiments.md](../standards/experiments.md)
- Template: [docs/templates/hypothesis.md](../templates/hypothesis.md)
- Hypothesis catalog: [docs/plans/research.md](../plans/research.md)
- Validated experiments (by family):
  - **Scheduler invariants:** `h12-conservation/` (Tier 1, deterministic), `h13-determinism/` (Tier 1, deterministic), `h-liveness/` (Tier 1, deterministic), `h25-integration-stress/` (Tier 1, deterministic)
  - **Structural model:** `h3-signal-freshness/` (Tier 2, dominance), `h9-prefix-caching/` (Tier 2, monotonicity), `h10-tiered-kv/` (Tier 3, dominance), `h-phase-structure/` (Tier 1, monotonicity), `h-mmk-validation/` (Tier 1, validation), `h26-admission-latency/` (Tier 2, dominance), `h-step-quantum/` (Tier 1, validation), `h19-roofline-vs-blackbox/` (Tier B, equivalence)
  - **Performance-regime:** `h8-kv-pressure/` (Tier 3, monotonicity), `h11-token-budget/` (Tier 3, dominance)
  - **Robustness/failure-mode:** `h14-pathological-templates/` (Tier 2, dominance), `h5-token-bucket-burst/` (Tier 3, dominance), `h-overload/` (Tier 1, deterministic), `h-overload-kv/` (Tier 1, deterministic), `h22-zero-blocks/` (Tier 2, deterministic), `h21-extreme-weights/` (Tier B, equivalence), `h24-combined-pathological/` (Tier 2, dominance)
  - **Cross-policy comparative:** `prefix-affinity/` (Tier 2, dominance), `h1-sjf-scheduling/` (Tier 3, dominance), `h2-priority-fcfs/` (Tier 5, dominance), `h17-pareto-frontier/` (Tier 4, Pareto)
  - **Workload/arrival:** `h-arrival-generators/` (Tier 1, statistical), `h16-gamma-vs-poisson/` (Tier 2, dominance)
