# Hypothesis Experiment Process

**Status:** Active (v2.0 — updated 2026-02-23)

This document describes the end-to-end process for running a hypothesis-driven experiment in BLIS. For experiment standards (rigor, classification, analysis), see [docs/standards/experiments.md](standards/experiments.md). For the FINDINGS.md template, see [docs/templates/hypothesis.md](templates/hypothesis.md). For experiment status and coverage gaps, see [hypotheses/README.md](https://github.com/inference-sim/inference-sim/blob/main/hypotheses/README.md).

---

## Prerequisites

This workflow uses the following Claude Code skills. Each has a manual alternative for contributors without AI tools.

| Skill | Purpose | Used In | Manual Alternative |
|-------|---------|---------|--------------------|
| `superpowers:using-git-worktrees` | Create isolated workspace | Step 0 | `git worktree add .worktrees/h-<name> -b h-<name>` |
| `convergence-review` | Dispatch parallel perspectives and enforce convergence | Steps 2, 5, 8 | Sequential manual checklists (one per perspective) using the checklist sections below |
| `commit-commands:commit-push-pr` | Commit, push, create PR | Step 10 | Standard git commands (`git add`, `git commit`, `git push`, `gh pr create`) |

**For external contributors without AI review infrastructure:** Submit your experiment artifacts via PR. Maintainers will run the review protocols on your behalf. You can also conduct reviews manually by having people review with the perspective checklists documented in each gate.

---

## Overview

```
┌──────────────────────────────────┐
│ Step 0: Create Worktree          │ (Isolated workspace)
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Step 1: Select & Classify        │ (Family, VV&UQ, type)
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Step 2: Design + Design Review   │ (5 perspectives → convergence)
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Step 3: Human Approval           │ (Approve design before implementation)
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Step 4: Implement                │ (run.sh + analyze.py using harness)
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Step 5: Code Review              │ (5 perspectives → convergence)
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Step 6: Run Experiments          │ (Execute across seeds)
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Step 7: Analyze & Document       │ (FINDINGS.md)
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Step 8: FINDINGS Review          │ (10 perspectives → convergence)
└──────────┬───────────────────────┘  ↑ (iterate rounds until convergence)
           │
           ▼
┌──────────────────────────────────┐
│ Step 9: Self-Audit               │ (6 dimensions, no agent)
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Step 10: Verify + Commit + PR    │ (verification gate, commit-push-pr)
└──────────────────────────────────┘
```

**Key insights:**
1. **Three review gates** at different lifecycle stages — each catches issues the others cannot
2. **Human gate before implementation** (Step 3) — prevents wasted experiment runs
3. **Universal convergence protocol** — same rules for all three gates
4. **Self-audit** — catches substance issues that pattern-matching agents miss

---

## Quick Reference

| Step | Action |
|------|--------|
| **0. Worktree** | `/superpowers:using-git-worktrees h-<name>` |
| **1. Classify** | Choose family, VV&UQ category, type from [experiments.md](standards/experiments.md) |
| **2. Design** | ED-1–ED-6 compliance, then 5-perspective Design Review |
| **3. Human gate** | Present design for approval — pause until approved |
| **4. Implement** | `run.sh` + `analyze.py` using `hypotheses/lib/` harness |
| **5. Code Review** | 5-perspective code review → convergence |
| **6. Run** | Execute `./run.sh` across required seeds |
| **7. Document** | Write FINDINGS.md using [template](templates/hypothesis.md) |
| **8. FINDINGS Review** | 10-perspective review → convergence (iterate rounds) |
| **9. Self-audit** | 6 dimensions of deliberate critical thinking |
| **10. Commit + PR** | Verification gate (if code fixes) + `/commit-commands:commit-push-pr` |

---

## Step-by-Step Process

### Step 0: Create Isolated Worktree

**Context:** Main repo (inference-sim)

Create an isolated workspace BEFORE any work begins.

```
/superpowers:using-git-worktrees h-<hypothesis-name>
```

This creates `.worktrees/h-<name>/` with a new branch. All subsequent steps happen in the worktree.

**Manual alternative:** `git worktree add .worktrees/h-<name> -b h-<name> && cd .worktrees/h-<name>`

---

### Step 1: Select and Classify Hypothesis

**Context:** Worktree

1. **Select hypothesis** — from `docs/plans/research.md`, coverage gaps in [hypotheses/README.md](https://github.com/inference-sim/inference-sim/blob/main/hypotheses/README.md), or a new observation
2. **Classify:**
   - (a) Which **family**? (See [experiments.md](standards/experiments.md) for the 6 families and sentence patterns)
   - (b) **Verification**, **Validation**, or **UQ**? (Determines evidence requirements)
   - (c) **Deterministic** or **statistical**? If statistical, which subtype (dominance, monotonicity, equivalence, Pareto)?

The family determines design rules; the VV&UQ category determines evidence requirements.

**Tip:** Pose the hypothesis WITHOUT reading the code first. Code-grounded hypotheses test implementation, not behavior. See [Generating Hypotheses](#generating-hypotheses) below.

---

### Step 2: Design Experiment + Design Review

**Context:** Worktree

**Design the experiment** following ED-1 through ED-6 (see [experiments.md](standards/experiments.md)):
- ED-1: Controlled comparison (vary exactly one dimension)
- ED-2: Rate awareness (run at target rate AND where effect should vanish)
- ED-3: Precondition verification (in script, not just prose)
- ED-4: Workload seed independence
- ED-5: Reproducibility (everything from `run.sh` alone)
- ED-6: Config diff against referenced experiments

Then run the **5-perspective Design Review** using the [universal convergence protocol](#universal-convergence-protocol).

**Primary mechanism (Claude Code):**
```
/convergence-review h-design
```

**Manual alternative:** Launch all 5 perspectives as parallel Task agents (or review sequentially with each checklist below), then apply the convergence protocol.

#### Design Review Perspectives

**Perspective 1 — Hypothesis Quality:**
- Is the hypothesis behavioral, testable, and diagnostic?
- Does it follow the family-specific sentence pattern from experiments.md?
- Is the diagnostic clause present ("If this fails, it would indicate...")?
- Is it correctly classified (family, VV&UQ category, type)?

**Perspective 2 — Experiment Design Rigor (ED-1–ED-6):**
- Is exactly one dimension varied? (ED-1)
- Is there a rate where the effect should vanish, to confirm mechanism dependence? (ED-2)
- Are preconditions verified in the script? (ED-3)
- Is seed handling correct? (ED-4)
- Is the experiment reproducible from `run.sh` alone? (ED-5)
- If reusing calibration from a prior experiment, is the config diff documented? (ED-6)

**Perspective 3 — Parameter Calibration:**
- Are parameters computed analytically from known coefficients (alpha/beta), not guessed?
- Are capacity estimates matched to the actual workload mode (CLI defaults vs workload-spec YAML)?
- Is the operating point correct for the intended effect? (e.g., near saturation for queueing effects, sub-saturation for baseline)

**Perspective 4 — Control Completeness:**
- Does every proposed mechanism have a planned control experiment? (RCV-4)
- Does each control isolate exactly one variable?
- Is the baseline configuration clearly defined?

**Perspective 5 — DES and Domain Fit:**
- Will the experiment create the conditions needed for the hypothesis to be testable?
- Are there DES-specific subtleties (event ordering, clock granularity, alpha overhead) that could confound results?
- Is the experiment duration sufficient? Is the warmup period adequate?

**Cross-gate regression:** If a later gate (Code Review, FINDINGS Review) discovers a design-level flaw (e.g., confounding variable, wrong operating point), the workflow loops back to Step 2 for re-design, re-convergence, and re-approval.

---

### Step 3: Human Approval of Experiment Design

**Context:** Worktree (after Design Review convergence)

Present the experiment design for human approval. The human reviews:
- Hypothesis classification (family, VV&UQ, type)
- Experiment design (ED-1–ED-6 compliance)
- Parameter choices (computed from coefficients, not guessed)
- Planned controls (one per proposed mechanism)
- Expected outcomes and diagnostic implications

**This is a hard gate.** Do not proceed to implementation until the human approves. Do not say "I'll proceed unless you stop me." The pause is the point.

**In parallel mode:** Each hypothesis gets its own independent human approval. The team lead presents each design; the human approves each independently.

---

### Step 4: Implement Experiment Code

**Context:** Worktree (after human approval)

Create `hypotheses/<name>/run.sh` and `hypotheses/<name>/analyze.py`.

**Harness requirements (mandatory):**
- `run.sh` MUST source `hypotheses/lib/harness.sh` and use `blis_run` for every simulation call
- Every `blis_run` call MUST have an appropriate timeout tier (`TIMEOUT_QUICK`/`TIMEOUT_STANDARD`/`TIMEOUT_EXTENDED`)
- If using `--total-kv-blocks`, call `preflight_kv_check` with max expected input tokens
- `analyze.py` MUST import `analyze_helpers` and use `parse_blis_output` (handles timeouts gracefully)

**Reference comment:** If reusing calibration from a prior experiment, include `# Reference: hypotheses/<name>/run.sh` with the file path.

---

### Step 5: Code Review (5 perspectives)

**Context:** Worktree (after implementation, BEFORE running experiments)

**Every `run.sh` and `analyze.py` must be code-reviewed BEFORE running experiments.** This is non-negotiable. Three of four major bugs in PR #310 would have been caught by code review before a single experiment ran.

**Cross-gate regression:** If this gate discovers a design-level flaw (e.g., confounding variable, wrong operating point), loop back to [Step 2](#step-2-design-experiment--design-review) for re-design, re-convergence, and re-approval. The experiment-wide limit of 2 cross-gate regressions applies (see [Step 8](#step-8-findings-review-10-perspectives) for the circuit breaker).

Run the **5-perspective Code Review** using the [universal convergence protocol](#universal-convergence-protocol).

**Primary mechanism (Claude Code):**
```
/convergence-review h-code hypotheses/<name>/
```

**Manual alternative:** Launch all 5 perspectives as parallel Task agents (or review sequentially with each checklist below), then apply the convergence protocol.

#### Code Review Perspectives

**Perspective 1 — Parser–Output Format Agreement:**
- For every regex or field extraction in `analyze.py`, verify the pattern matches the actual output format in the simulator code
- `cmd/root.go` — what text does the CLI print? (e.g., `"Preemption Rate: %.4f"`)
- `sim/metrics_utils.go` — what JSON fields exist?
- Match every regex in `analyze.py` against the format string in the producer code
- **Silent defaults**: Verify that when a regex matches nothing, `analyze.py` emits a warning to stderr rather than silently defaulting to 0

**Perspective 2 — CLI Flag Correctness:**
- For every flag in `run.sh`, verify the flag name and value match `cmd/root.go` defaults and help text
- Check for typos that strict parsing would reject
- Cross-reference every CLI flag against `cmd/root.go` flag definitions

**Perspective 3 — YAML Field Validation:**
- Verify workload YAML field names against `sim/workload/spec.go` struct tags
- `KnownFields(true)` will reject typos at runtime, but catching them at review saves a failed experiment run

**Perspective 4 — Config Diff (ED-6):**
- If the experiment reuses calibration from a prior experiment, diff every CLI flag and YAML field between the two experiments
- Explicitly list differences
- Verify the `# Reference:` comment in `run.sh` points to the correct file

**Perspective 5 — Seed and Determinism:**
- Verify `--seed` is passed correctly and workload YAML `seed:` field doesn't conflict
- Verify seeds vary across runs as intended (ED-4)
- Check that `run.sh` builds the binary and is fully self-contained (ED-5)

**Evidence: what code review catches**

| Bug | Round discovered | Would code review have caught it? |
|-----|-----------------|-----------------------------------|
| YAML `input_dist` vs `input_distribution` (H5) | Round 1 run failure | **Yes** — cross-ref against `spec.go` struct tags |
| Analyzer regex `Preemptions?: (\d+)` vs actual `Preemption Rate: 0.1750` (H10) | Round 4 | **Yes** — cross-ref against `cmd/root.go` format string |
| H10 routing policy mismatch with H8 | Round 2 | **Yes** — ED-6 config diff |
| H5 bucket cap=500 < mean_input=512 | Round 2 | **Possibly** — first-principles check on parameters |

---

### Step 6: Run Experiments

**Context:** Worktree (after Code Review convergence)

Execute experiments across required seeds:
- **Deterministic experiments:** Single seed sufficient (determinism is the point)
- **Statistical experiments:** Minimum 3 seeds (42, 123, 456) for each configuration
- Verify reproducibility: running `./run.sh` twice produces identical output (ED-5)

---

### Step 7: Analyze and Document FINDINGS.md

**Context:** Worktree (after experiments complete)

1. **Analyze** — produce comparison tables, compute effect sizes
2. **Verify root cause** — trace every causal claim through code (RCV-1, RCV-2, RCV-3)
3. **Document FINDINGS.md** — use the [template](templates/hypothesis.md). All sections must be present and non-empty.
4. **Update `hypotheses/README.md`** — add a row to the "Validated Hypotheses" table and update "Coverage by Family" if needed

---

### Step 8: FINDINGS Review (10 perspectives)

**Context:** Worktree (after FINDINGS.md documented)

Run the **10-perspective FINDINGS Review** using the [universal convergence protocol](#universal-convergence-protocol).

**Primary mechanism (Claude Code):**
```
/convergence-review h-findings hypotheses/<name>/FINDINGS.md
```

**Manual alternative:** Launch all 10 as parallel Task agents (each receives the FINDINGS.md path and its specific focus area), then apply the convergence protocol.

**Cross-gate regression:** If this gate discovers a design-level flaw (e.g., confounding variable not identified in design), loop back to [Step 2](#step-2-design-experiment--design-review) for re-design, re-convergence, and re-approval. Maximum 2 cross-gate regressions per experiment (across all gates combined) — if the design still has fundamental issues after 2 regressions, suspend the experiment and escalate for a re-scoping decision.

#### FINDINGS Review Perspectives

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
- Does every proposed mechanism (RCV-3) have a corresponding control experiment (RCV-4)?
- Were control experiments actually EXECUTED, not just proposed? Look for conditional language ("one would test", "could be confirmed by") vs past tense with data ("the control showed 0.0% difference"). Verify that control results appear in the Results section with actual numbers, not just in Root Cause Analysis as narrative.
- Does each control isolate exactly one variable? Diff the CLI flags between treatment and control runs in `run.sh` — the control should differ by exactly one flag or parameter.
- Do the control results confirm or refute the proposed mechanism?
- Do the control experiment results in the Evidence Quality table accurately reflect the current round (not stale text from a prior round)?
- Does the mechanism explain the direction using experimental evidence, not just code-reading claims?

**Reviewer 5 — Standards Compliance:**
- Are ALL FINDINGS.md sections present and non-empty? (per `docs/templates/hypothesis.md`)
- Is the hypothesis correctly classified (family, VV&UQ category, type)?
- Does the Devil's Advocate section (RCV-5) argue both directions convincingly?
- Are scope and limitations (RCV-6) complete — operating point, dependencies, what was NOT tested, generalizability, UQ?
- Does the standards audit correctly check findings against `docs/standards/rules.md` and `docs/standards/invariants.md`?
- Are any new rules or invariants warranted by the findings?

**Reviewer 6 — Substance and Logic:**
- Are there logical errors in the conclusions?
- Are there mathematical mistakes in effect size calculations or statistical claims?
- Does the evidence actually support the claims? (Not just "the numbers are close enough")
- Are alternative explanations adequately considered?

**Reviewer 7 — DES Mechanism Expert:**
- Are there event-ordering subtleties that could explain the results differently?
- Are assumptions about DES timing correct (alpha overhead, step quantization, clock granularity)?
- Could the result be an artifact of the simulation architecture rather than the modeled system behavior?

**Reviewer 8 — Reproducibility and Robustness:**
- Can `run.sh` reproduce the results from scratch on a clean checkout?
- Are results fragile to small parameter variations? (Would ±10% on key parameters change the conclusion?)
- Are all intermediate files generated by the script, not checked in as stale artifacts?

**Reviewer 9 — Cross-Experiment Consistency:**
- Do the findings contradict any prior experiment results? If so, is the contradiction acknowledged and explained?
- Are references to prior experiments accurate? (Check specific claims against referenced FINDINGS.md files)
- Are there stale references to prior rounds that should have been updated?

**Reviewer 10 — User Guidance and Actionability:**
- Are the "Implications for Users" practical and specific enough to act on?
- Are proposed issues (bugs, enhancements, follow-up hypotheses) well-scoped?
- Would a BLIS user reading this FINDINGS.md understand what to do differently?

The overlap between reviewers is intentional — different perspectives checking the same FINDINGS.md catch different issues. Evidence from PR #385: Reviewer 1 (code) and Reviewer 3 (rigor) both caught H19's stale evidence quality row; Reviewer 2 (design) caught H16's sub-threshold seed; Reviewer 4 (control) caught H21's unexecuted control experiments.

---

### Step 9: Self-Audit (6 dimensions)

**Context:** Worktree (after FINDINGS Review convergence)

> **For Claude:** This is NOT an agent pass. Stop, think critically, and answer each question from your own reasoning. Do NOT dispatch agents.

**Why this step exists:** In PR9, 3 real bugs were found by critical thinking after 4 automated passes found 0 issues. Automated review perspectives check structure; this step checks substance.

**Self-audit dimensions — think through each one:**

1. **Logic bugs in analyzer code:** Trace through `analyze.py` mentally. Are there edge cases where regex parsing silently defaults to 0? Could integer vs float conversion produce wrong results?
2. **Results determinism/reproducibility:** Would running `./run.sh` again produce identical output? Are there any non-deterministic dependencies (timestamps, system load, file ordering)?
3. **FINDINGS.md internal consistency:** Does the Status match the data in the Results section? Does the Devil's Advocate section actually argue against the conclusion? Are all per-seed values consistent with the summary?
4. **Cross-experiment contradictions:** Do these findings contradict any known BLIS behavior documented in prior experiments or MEMORY.md? If so, is the contradiction explained?
5. **User guidance practicality:** Would a BLIS user know what to do with these findings? Are the implications actionable?
6. **Issue filing completeness:** For each actionable finding, is there a clear issue to file? Are there findings that need issues but don't have them planned?

**Fix all issues found. Then proceed to Step 10.**

---

### Step 10: Verification Gate + Commit + PR

**Context:** Worktree (after self-audit)

**If the experiment discovered code fixes** (e.g., #386 KV livelock, #387 conservation test update), run the verification gate before committing:

```bash
go build ./...          # Build passes
go test ./... -count=1  # All tests pass
golangci-lint run ./... # Zero lint issues
```

**Commit and PR:**

```
/commit-commands:commit-push-pr
```

The PR description should include:
- Hypothesis sentence and status
- Key findings (1-3 bullet points)
- `Fixes #NNN` for any issues this experiment addresses

**Manual alternative:** Standard git commands (`git add`, `git commit`, `git push`, `gh pr create`).

**Post-PR issue filing:** See [Two-Track Issue Filing](#two-track-issue-filing) below.

---

## Universal Convergence Protocol

> **Canonical source:** [`docs/process/convergence.md`](convergence.md). If this section diverges, convergence.md is authoritative.

All three review gates (Design Review, Code Review, FINDINGS Review) use the same convergence protocol: run all N perspectives in parallel, fix any CRITICAL/IMPORTANT findings, re-run until zero CRITICAL and zero IMPORTANT in a round. Max 10 rounds per gate. See [`docs/process/convergence.md`](convergence.md) for the full protocol, severity definitions, agent failure handling, and expected convergence rates.

> **Executable implementation:** The `convergence-review` skill automates this protocol. Invoke with `/convergence-review <gate-type> [artifact-path] [--model opus|sonnet|haiku]`.

---

## Parallel Execution Mode

The #385/#390 pattern: run N hypothesis experiments simultaneously with a team lead coordinating.

### Setup

1. **Team lead creates worktree:** `.worktrees/h-<batch-name>/`
2. **Team lead builds binary once:** `go build -o simulation_worker main.go` (agents reference this shared binary)
3. **Team lead creates team** with N hypothesis agents, each assigned to `hypotheses/<name>/`
4. Each agent runs the full pipeline independently (Steps 1-9)

### Coordination Rules

- Each agent creates files ONLY in its own `hypotheses/<name>/` directory — no file conflicts
- **README.md updates are deferred** to the team lead's consolidation step (not done by individual agents)
- **Team lead MUST independently run convergence review** for each experiment. Do NOT delegate convergence assessment to the same agent that ran the experiment. Evidence from #390: agents self-reported "Round 1 convergence" but actual independent review found 3 CRITICAL + 18 IMPORTANT issues.
- **Step 3 is a synchronization point.** All agents pause at Step 3 until the human has reviewed and approved each design independently. The team lead should batch-present all designs for human review to minimize idle time.
- Solo mode is the degenerate case (team size = 1)

### Consolidation

After all agents complete:
1. Team lead reviews all proposed issues for deduplication
2. Team lead updates `hypotheses/README.md` with all new experiments
3. Cross-experiment consistency check across all N experiments
4. Single PR via `commit-push-pr` skill

---

## Two-Track Issue Filing

### Immediate Track (file as soon as discovered)

**Only for bugs that affect experiment validity:**
- Simulator panics or crashes during experiment
- Conservation violations (INV-1) discovered during analysis
- Livelock or infinite loops (R19) preventing experiment completion
- Data loss or silent failures affecting measured metrics

Example: #386 (KV livelock) was filed immediately because it caused infinite preempt-requeue cycles.

### Post-Convergence Track (file AFTER convergence + PR creation)

**For all findings-derived issues:**
- Bugs found but not affecting current experiment validity
- Enhancements, design limitations, new hypotheses
- Standards updates (new rules, invariants)
- Promotion to Go test suite

**Why issues come last:** Findings can change across rounds (H10 went from "untested" to "confirmed" between Rounds 3-4). Filing issues before convergence risks creating wrong issues. File once, file right.

See [Issue Taxonomy](#issue-taxonomy-after-convergence) for the complete filing guide.

---

## Quality Gates

### Pre-Execution Gates (check BEFORE running experiments — Step 5)
- [ ] `run.sh` sources `hypotheses/lib/harness.sh` and uses `blis_run` for every simulation call
- [ ] Every `blis_run` call has an appropriate timeout tier (`TIMEOUT_QUICK`/`TIMEOUT_STANDARD`/`TIMEOUT_EXTENDED`)
- [ ] KV safety pre-flight: if experiment uses `--total-kv-blocks`, call `preflight_kv_check` with max expected input tokens
- [ ] `analyze.py` imports `analyze_helpers` and uses `parse_blis_output` (handles timeouts gracefully)
- [ ] `run.sh` flags verified against `cmd/root.go` help text
- [ ] `analyze.py` regexes verified against actual output format strings in `cmd/root.go` and `sim/metrics_utils.go`
- [ ] Workload YAML field names verified against `sim/workload/spec.go` struct tags
- [ ] Config diff against referenced experiments documented (ED-6)
- [ ] Code Review (5 perspectives) converged

### Per-Round Gates (check after each FINDINGS Review round — Step 8)
- [ ] Every causal claim cites `file:line` (RCV-1)
- [ ] Every "surprise" has a first-principles calculation (RCV-2)
- [ ] Root cause explains mechanism AND direction (RCV-3)
- [ ] All reviewer assessments completed and all CRITICAL/IMPORTANT items addressed

### Final Gates (check before PR — Step 10)
- [ ] Hypothesis classified (deterministic or statistical + subtype)
- [ ] Experiment design follows ED-1 through ED-6
- [ ] If reusing prior calibration data, config diff documented (ED-6)
- [ ] Results reproducible via `./run.sh`
- [ ] **FINDINGS Review converged**: zero CRITICAL and zero IMPORTANT items across all 10 reviewers in the current round
- [ ] Self-audit completed (6 dimensions)
- [ ] All review feedback addressed or explicitly acknowledged as open
- [ ] Findings classified per the findings table (including resolution type)
- [ ] Standards audit completed
- [ ] Promotion assessment completed (see [Promotion of Confirmed Hypotheses](#promotion-of-confirmed-hypotheses))
- [ ] `hypotheses/README.md` updated with new experiment row(s) and coverage changes
- [ ] If code fixes involved: `go build`, `go test`, `golangci-lint` all pass

### Post-PR Gates (check after PR creation — Step 10)
- [ ] Issues filed per [Issue Taxonomy](#issue-taxonomy-after-convergence) — one per actionable finding
- [ ] Each issue has correct label (`bug`, `enhancement`, `hypothesis`, `design`, or `standards`)
- [ ] Each issue references the PR number
- [ ] No issues filed for "documented here" findings with no action needed

---

## When to Run Experiments

- Validating that a new feature works as designed (post-PR confirmation)
- Testing intuitive claims about system behavior (from `docs/plans/research.md`)
- Investigating unexpected behavior observed during development
- Exploring design tradeoffs between configurations
- Filling coverage gaps identified in the [family coverage table](https://github.com/inference-sim/inference-sim/blob/main/hypotheses/README.md)

---

## Generating Hypotheses

Hypotheses can come from **internal** sources (your own experiments and development) or **external** sources (user questions, literature, analytical models). This section provides structured guidance for generating good hypotheses. See also [experiments.md](standards/experiments.md) for family-specific sentence patterns.

### Sources of hypotheses

| Source | How it works | Example |
|--------|-------------|---------|
| **User intuition** | "I think X should be better than Y because of Z" | "SJF should reduce TTFT for mixed workloads because short jobs finish first" |
| **Coverage gaps** | Check the [family coverage table](https://github.com/inference-sim/inference-sim/blob/main/hypotheses/README.md) for untested families | Workload/arrival family has 0 experiments → "Gamma sampler should match theoretical CV" |
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
| **Conceptual** | "Tiered storage should reduce preemptions" | "tiered.go:224 should delete the hash" |
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

1. **Check coverage**: Read the [family coverage table](https://github.com/inference-sim/inference-sim/blob/main/hypotheses/README.md). Prioritize families with low coverage.
2. **Choose a family**: Which domain does your claim target? (See [experiments.md](standards/experiments.md) for the 6 families.)
3. **Write the sentence**: Use the family-specific pattern from experiments.md.
4. **Add the diagnostic clause**: "If this fails, it would indicate..."
5. **Check for redundancy**: Search existing hypotheses in `docs/plans/research.md` and on GitHub: [issues labeled `hypothesis`](https://github.com/inference-sim/inference-sim/labels/hypothesis).
6. **File as a GitHub issue**: Use the [Hypothesis Proposal issue template](https://github.com/inference-sim/inference-sim/issues/new?template=hypothesis.md) on GitHub (click "New Issue" → "Hypothesis Proposal"). This template has fields for family, VV&UQ category, diagnostic value, and experiment design.

External contributors should file a GitHub issue using the Hypothesis Proposal template. Maintainers will triage, prioritize, and run the review protocol.

---

## Issue Taxonomy (after convergence)

After convergence and PR creation, walk the findings classification table in FINDINGS.md and file one GitHub issue per actionable finding. Not every hypothesis produces issues — a clean confirmation (like H13) may produce none.

**Issue types and labels:**

| Issue Type | Label | When to file | Title format | Example |
|------------|-------|-------------|--------------|---------|
| **Bug** | `--label bug` | Code defect discovered during experiment | `bug: <component> — <defect>` | `bug: sim/simulator.go — preempt() panics on empty RunningBatch` (H12) |
| **Enhancement** | `--label enhancement` | New feature, rule, or documentation improvement needed | `enhancement: <area> — <improvement>` | `enhancement: CLI — document token-bucket per-input-token cost model` (H5) |
| **New hypothesis** | `--label hypothesis` | Follow-up experiment spawned by current findings | `hypothesis: <claim to test>` | `hypothesis: test tiered KV at GPU=1500 blocks to trigger preemption-path offload` (H10) |
| **Design limitation** | `--label design` | System works as coded but has undocumented behavioral limitation | `design: <limitation>` | `design: no burst-smoothing sweet spot under Gamma CV>3` (H5) |
| **Standards update** | `--label standards` | New rule or invariant discovered that should be added | `standards: <rule/invariant>` | `standards: R17 signal freshness — routing signals have tiered staleness` (H3) |
| **Promotion** | `--label promotion` | Confirmed hypothesis finding promoted to Go test suite | `enhancement: promote <hypothesis> <finding> to Go test suite` | `enhancement: promote H-Overload conservation under 10x to Go test suite` (#337) |

**Mapping from resolution type to expected issues:**

| Resolution | Expected issues |
|------------|----------------|
| Clean confirmation | Usually none. Optionally: promotion to Go test suite, standards update confirming existing rules. |
| Confirmation with wrong mechanism | Enhancement: update documentation with correct mechanism. |
| Confirmation with bug discovery | Bug: one per code defect. Enhancement: if detector/tooling needs improvement. |
| Partial confirmation with surprise | New hypothesis: follow-up experiments to investigate surprise. |
| Refuted — system design flaw | Design: architectural limitation. Enhancement: proposed fix. |
| Refuted — mechanism not plausible | Design: document the limitation. Enhancement: update CLI help or user docs if misleading. |
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

---

## Promotion of Confirmed Hypotheses

After convergence, assess whether any confirmed findings should be promoted from bash-script experiments to the Go test suite and/or formal invariants. Hypothesis experiments run as bash scripts are NOT in CI — a regression would not be caught by `go test ./...`.

### When to promote

| Condition | Promote to | Why |
|-----------|-----------|-----|
| Confirmed deterministic hypothesis | **Go test** (regression protection in CI) | Deterministic properties are exact — they can be encoded as pass/fail tests. |
| Deterministic invariant aspect of a statistical hypothesis | **Go test** for the invariant aspect | Statistical hypotheses often contain deterministic sub-claims (e.g., conservation holds across all configs). |
| New invariant discovered | **`docs/standards/invariants.md`** entry | Codify as a formal system property with verification strategy. |
| New rule discovered | **`docs/standards/rules.md`** entry | Codify as an antipattern check for PR reviews. |

### What a promoted test looks like

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

---

## Why Iterate Until Convergence (Not Fixed Rounds)?

Evidence from PR #310 (H5, H10, H13):

| Round | What happened | What was caught |
|-------|---------------|-----------------|
| **1** | Initial experiments | Wrong root causes for H5 and H10 |
| **2** | Code + external review | Corrected math (H5), identified mechanism (H10), designed confound matrix |
| **3** | Confound matrix + calibrated bucket | H5 burst-smoothing mechanism refuted, H10 analyzer bug masked preemptions |
| **4** | Corrected analyzer | H10 confirmed — preemptions DO occur, cache hits INCREASE |

H13 converged in Round 1 (deterministic = pass/fail). H5 converged in Round 3. H10 required Round 4 due to an analyzer bug. Fixed round counts would have either stopped too early (missing the H10 bug) or forced unnecessary work (H13 didn't need Round 2).

### Why internal agents beat external LLMs

| Capability | External (`/review-plan`) | Internal (Task agent) |
|-----------|--------------------------|----------------------|
| Read source files | No | Yes — verifies every citation |
| Cross-ref regexes against format strings | No | Yes — catches analyzer bugs |
| Check YAML fields against struct tags | No | Yes — catches typos |
| Run `grep` to verify claims | No | Yes — can search for executed vs proposed controls |
| API reliability | Fragile (auth, timeouts, rate limits) | Reliable (same process) |

---

## References

- Standards: [docs/standards/experiments.md](standards/experiments.md)
- Template: [docs/templates/hypothesis.md](templates/hypothesis.md)
- Hypothesis catalog: [docs/plans/research.md](../plans/research.md)
- Validated experiments: [hypotheses/README.md](https://github.com/inference-sim/inference-sim/blob/main/hypotheses/README.md)
- PR workflow (structural inspiration): [docs/process/pr-workflow.md](pr-workflow.md)

---

## Appendix: Workflow Evolution

**v1.0 (PR #310):** Three external LLM reviews per round, no design gate, no code review gate, ad-hoc git commands.
**v2.0 (2026-02-23, #392):** Three review gates (Design 5, Code 5, FINDINGS 10) with universal convergence protocol, human approval gate, self-audit, verification gate, parallel execution, two-track issue filing, explicit worktree/commit skill integration. Structural alignment with PR workflow v3.0.
