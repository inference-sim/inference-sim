# PR Development Workflow

**Status:** Active (v4.0 — updated 2026-02-27)

This document describes the complete workflow for implementing a PR from any source: a macro plan section, GitHub issues, a design document, or a feature request. The same steps apply whether you use Claude Code or standard git tools.

---

## Overview

```
┌─────────────────────────────────┐
│ Step 1: Create Isolated Workspace│
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ Step 2: Write Implementation Plan│
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ Step 2.5: Review the Plan        │ (10 perspectives → convergence)
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ Step 3: Human Review             │ (Approve plan)
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ Step 4: Implement the Plan       │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ Step 4.5: Review the Code        │ (10 perspectives → convergence)
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ Step 4.75: Pre-Commit Self-Audit │ (Critical thinking — no agent)
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ Step 5: Commit, Push, Create PR  │
└─────────────────────────────────┘
```

**Key insights:**
1. **Worktree isolation from start** (Step 1) — Create worktree BEFORE any work. Main worktree never touched. Enables parallel work on multiple PRs.
2. **Three-stage quality assurance:**
   - **Plan Review** (Step 2.5) — two-stage: holistic pre-pass, then 10 targeted perspectives. Catches design issues before implementation.
   - **Code Review** (Step 4.5) — two-stage: holistic pre-pass, then 10 targeted perspectives. Catches implementation issues before PR creation.
   - **Self-Audit** (Step 4.75) — deliberate critical thinking across 10 dimensions. Catches substance bugs that pattern-matching agents miss.

---

## Step-by-Step Process

### Step 1: Create an Isolated Workspace

Create a git worktree BEFORE any work begins. This ensures the main worktree stays clean, the plan is committed on a feature branch, and you can work on multiple PRs in parallel.

```bash
git worktree add .worktrees/pr<N>-<feature-name> -b pr<N>-<feature-name>
cd .worktrees/pr<N>-<feature-name>
```

**Why first?** Create isolated workspace BEFORE any work begins. This ensures:
- Main worktree stays clean (no uncommitted plans or code)
- Plan document committed on feature branch (not main)
- Complete isolation for entire PR lifecycle (planning + implementation)
- Ability to work on multiple PRs in parallel

All remaining steps happen in the worktree.

!!! tip "Automation"
    `/superpowers:using-git-worktrees pr<N>-<feature-name>` creates the worktree and switches your shell into it. Optional pre-cleanup: `/commit-commands:clean_gone` removes stale branches. See [Skills & Plugins](../guide/skills-and-plugins.md).

---

### Step 2: Write an Implementation Plan

Write an implementation plan following the [micro-plan template](templates/micro-plan.md). The plan must include:

- **Behavioral contracts** (GIVEN/WHEN/THEN) defining what this PR guarantees
- **TDD task breakdown** (6–12 tasks, each: test → fail → implement → pass → lint → commit)
- **Test strategy** mapping contracts to specific tests

Save the plan to `docs/plans/<feature-name>-plan.md`.

The source of work can be a macro plan section, a design document, one or more GitHub issues, or a feature request.

!!! tip "Automation"
    `/superpowers:writing-plans for <work-item> in @docs/plans/<name>-plan.md using @docs/contributing/templates/micro-plan-prompt.md and @<source-document>` generates the plan automatically. The skill reads the source document and the template, inspects the codebase, and produces behavioral contracts with executable tasks.

---

### Step 2.5: Review the Plan

Review the plan from 10 targeted perspectives, applying the [convergence protocol](convergence.md). Run all 10 perspectives in parallel as a single round. If any perspective finds CRITICAL or IMPORTANT issues, fix them all and re-run the entire round. Repeat until a round produces zero CRITICAL and zero IMPORTANT findings.

**Two-stage review:**

1. **Holistic pre-pass:** Do a single deep review to catch cross-cutting issues before the formal convergence protocol.
2. **Formal convergence:** Run all 10 perspectives below in parallel.

!!! tip "Automation"
    Stage 1: `/pr-review-toolkit:review-pr`. Stage 2: `/convergence-review pr-plan docs/plans/<name>-plan.md`. See [Skills & Plugins](../guide/skills-and-plugins.md).

**Why two stages?** The holistic sweep catches emergent cross-cutting issues (the kind a human reviewer would spot). Fixing those first means the convergence review starts from a cleaner baseline — fewer rounds needed because obvious issues are already addressed.

**Why rounds with multiple perspectives?** Generic "review everything" misses issues that targeted perspectives catch. Different lenses find different bugs: cross-doc consistency catches stale references, architecture catches boundary violations, substance catches design bugs. Running them in parallel maximizes coverage per round. The hypothesis process proved this model: 3 parallel reviewers with different foci caught issues that sequential single-reviewer rounds missed.

#### Perspective 1: Substance & Design

Check for: design bugs, mathematical errors, logical inconsistencies, scale mismatches, missing edge cases. Are the behavioral contracts logically sound? Could the design actually achieve what the contracts promise? Check formulas, thresholds, and edge cases from first principles — not just structural completeness.

**Catches:** Design bugs, mathematical errors, logical inconsistencies, scale mismatches, missing edge cases.

**Why this perspective exists:** In PR9, the fitness normalization formula (`1/(1+value)`) passed all structural checks but was a design bug (500,000x scale imbalance between throughput and latency). A substance-focused review caught what structure-focused reviews missed.

#### Perspective 2: Cross-Document Consistency

Check for: scope mismatch between micro plan and source document, stale file paths, deviation log completeness. Does the deviation log account for all differences between what the source says and what the micro plan does? Check for stale references.

**Catches:** Stale references, scope mismatch, missing deviations, wrong file paths.

#### Perspective 3: Architecture Boundary Verification

Check for: import cycle risks, boundary violations (individual instances accessing cluster-level state), types in wrong packages, multiple construction sites for the same type, high touch-point multipliers (adding one field requires >3 files), library code (`sim/`) calling `logrus.Fatalf`.

**Catches:** Import cycle risks, boundary violations, missing bridge types, wrong abstraction level, construction site proliferation, high touch-point multipliers, error handling boundary violations.

#### Perspective 4: Codebase Readiness

Check for: stale comments ("planned for PR N" where N is completed), pre-existing bugs in files the plan will modify, missing dependencies, unclear insertion points, TODO/FIXME items in the modification zone.

**Catches:** Stale comments, pre-existing bugs, missing dependencies, unclear insertion points.

#### Perspective 5: Plan Structural Validation

Perform these 4 checks directly (no agent needed):

**Check 1: Task Dependencies** — For each task, verify it can actually start given what comes before it. Trace the dependency chain: what files does each task create/modify? Does any task require a file or type that hasn't been created yet? Flag tasks that modify the same file and could conflict.

**Check 2: Template Completeness** — Verify all sections from the [micro-plan template](templates/micro-plan.md) are present and non-empty: Header, Part 1 (A–E), Part 2 (F–I), Part 3 (J), Appendix.

**Check 3: Executive Summary Clarity** — Read the executive summary as if you're a new team member with no context. Is it clear what the PR does and why? Can you understand the scope without reading the rest of the plan?

**Check 4: Under-specified Tasks** — For each task, verify it has complete code in every step (no "add validation" without showing exact code). Verify exact test commands with expected output. Verify exact commit commands. Flag any step that an executing agent would need to figure out on its own.

**Catches:** Broken task ordering, missing template sections, unclear summaries, vague implementation steps that will cause agent confusion.

#### Perspective 6: DES Expert

Check for: event ordering bugs, clock monotonicity violations, stale signal propagation between event types, heap priority errors, event-driven race conditions, work-conserving property violations, incorrect assumptions about DES event processing semantics. Verify that any new events respect the `(timestamp, priority, seqID)` ordering contract.

**Catches:** Event ordering violations, clock regression, stale-signal bugs, priority inversion in event queues.

#### Perspective 7: vLLM/SGLang Expert

Check for: batching semantics that don't match real continuous-batching servers, KV cache eviction policies that differ from vLLM's implementation, chunked prefill behavior mismatches, preemption policy differences, missing scheduling features that real servers have. Flag any assumption about LLM serving that this plan gets wrong.

**Catches:** Batching model inaccuracies, KV cache behavior mismatches, prefill/decode pipeline errors, scheduling assumption violations.

#### Perspective 8: Distributed Inference Platform Expert

Check for: multi-instance coordination bugs, routing load imbalance under high request rates, stale snapshot propagation between instances, admission control edge cases at scale, horizontal scaling assumption violations, prefix-affinity routing correctness across instances.

**Catches:** Load imbalance, stale routing state, admission control failures, scaling assumption violations, cross-instance coordination bugs.

#### Perspective 9: Performance & Scalability

Check for: algorithmic complexity issues (O(n²) where O(n) suffices), unnecessary allocations in hot paths, map iteration in O(n) loops that could grow, benchmark-sensitive changes, memory growth patterns, changes that would degrade performance at 1000+ requests or 10+ instances.

**Catches:** Algorithmic complexity regressions, hot-path allocations, memory growth, scalability bottlenecks.

#### Perspective 10: Security & Robustness

Check for: input validation completeness (all CLI flags, YAML fields, config values), panic paths reachable from user input, resource exhaustion vectors (unbounded loops, unlimited memory growth), degenerate input handling (empty, zero, negative, NaN, Inf), configuration injection risks.

**Catches:** Input validation gaps, user-reachable panics, resource exhaustion, degenerate input failures, injection risks.

---

### Step 3: Human Review of Plan

Final human review of the plan (after automated review).

**Focus areas:**
1. **Part 1 (Design Validation)** — Review behavioral contracts, component interaction, risks
2. **Part 2 (Executable Tasks)** — Verify task breakdown makes sense, no dead code
3. **Deviation Log** — Check if deviations from source document are justified
4. **Appendix** — Spot-check file-level details for accuracy

**Common issues to catch:**
- Behavioral contracts too vague or missing edge cases
- Tasks not properly ordered (dependencies)
- Missing test coverage for contracts
- Deviations from source document not justified
- Dead code or scaffolding

**Outcome:** ✅ Approve plan → proceed to Step 4 (implementation). ❌ Need revisions → iterate, re-review (Step 2.5), then approve.

**Note:** The plan will be committed together with the implementation in Step 5 (single commit for entire PR).

---

### Step 4: Implement the Plan

Implement the tasks from the approved plan using TDD:

For each task:
1. Write the failing test
2. Run test to verify it fails
3. Implement minimal code to pass
4. Run test to verify it passes
5. Run lint: `golangci-lint run ./path/to/package/...`
6. Commit with contract reference

Execute all tasks sequentially. Stop only on test failure, lint failure, or build error.

!!! tip "Automation"
    `/superpowers:executing-plans @docs/plans/<name>-plan.md` executes all tasks continuously without pausing. On failure, use `/superpowers:systematic-debugging` for structured root-cause analysis.

---

### Step 4.5: Review the Code

Review the implementation from 10 targeted perspectives, applying the [convergence protocol](convergence.md). Same two-stage structure as Step 2.5.

**Two-stage review:**

1. **Holistic pre-pass:** Single deep review to catch cross-cutting issues.
2. **Formal convergence:** Run all 10 perspectives below in parallel.

!!! tip "Automation"
    Stage 1: `/pr-review-toolkit:review-pr`. Stage 2: `/convergence-review pr-code`. See [Skills & Plugins](../guide/skills-and-plugins.md).

**Why two stages?** The holistic sweep catches emergent cross-cutting issues. In past PRs, this pre-pass found issues (runtime-breaking regressions, stale panic message prefixes) that individual targeted perspectives missed because they were each focused on their narrow lens. Fixing those first reduces convergence rounds.

**Why 10 perspectives in parallel?** Each catches issues the others miss. In the standards-audit-hardening PR, Perspective 1 (substance) found a runtime-breaking regression, Perspective 3 (tests) found weakened coverage, Perspective 7 (vLLM expert) confirmed CLI validation matches real server semantics, and Perspective 10 (security) found pre-existing factory validation gaps. Domain-specific perspectives (DES, vLLM, distributed platform) catch issues that generic code-quality reviewers miss.

#### Perspective 1: Substance & Design

Check for: logic bugs, design mismatches between contracts and implementation, mathematical errors, silent regressions. Does the implementation actually achieve what the behavioral contracts promise? Check from first principles — not just structural patterns.

**Catches:** Design bugs, formula errors, silent regressions, semantic mismatches between intent and implementation.

#### Perspective 2: Code Quality + Error Handling

Check for:
(1) Any new error paths that use `continue` or early `return` — do they clean up partial state?
(2) Any map iteration that accumulates floats — are keys sorted?
(3) Any struct field added — are all construction sites updated?
(4) Does library code (`sim/`) call `logrus.Fatalf` anywhere in new code?
(5) Any exported mutable maps — should they be unexported with `IsValid*()` accessors?
(6) Any YAML config fields using `float64` instead of `*float64` where zero is valid?
(7) Any division where the denominator derives from runtime state without a zero guard?
(8) Any new interface with methods only meaningful for one implementation?
(9) Any method >50 lines spanning multiple concerns (scheduling + latency + metrics)?
(10) Any changes to `docs/contributing/standards/` files — are CLAUDE.md working copies updated to match?

**Catches:** Logic errors, nil pointer risks, silent failures (discarded return values), panic paths reachable from user input, CLAUDE.md convention violations, dead code, silent `continue` data loss, non-deterministic map iteration, construction site drift, library code calling `os.Exit`, exported mutable maps, YAML zero-value ambiguity, division by zero in runtime computation, leaky interfaces, monolith methods, documentation drift.

#### Perspective 3: Test Behavioral Quality

Check for: Are all tests truly behavioral (testing WHAT, not HOW)? Rate each test as Behavioral, Mixed, or Structural. Would they survive a refactor? Are there golden dataset tests that lack companion invariant tests? Golden tests encode current behavior as "correct" — if the code had a bug when golden values were captured, the test perpetuates the bug. Flag golden tests whose expected values are not independently validated by an invariant test.

**Catches:** Structural tests (Go struct assignment, trivial getters), type assertions in factory tests, exact-formula assertions instead of behavioral invariants, tests that pass even if the feature is broken, golden-only tests that would perpetuate pre-existing bugs (issue #183: codellama golden dataset encoded a silently-dropped request as the expected value since its initial commit).

#### Perspective 4: Getting-Started Experience

Check for: Simulate the journey of (1) a user doing capacity planning with the CLI, and (2) a contributor adding a new algorithm. Where would they get stuck? What's missing? Check for missing example files, undocumented output metrics, incomplete contributor guide, unclear extension points, README not updated for new features.

**Catches:** Missing example files, undocumented output metrics, incomplete contributor guide, unclear extension points, README not updated for new features.

#### Perspective 5: Automated Reviewer Simulation

Check for: What GitHub Copilot, Claude, and Codex would flag. Exported mutable globals, user-controlled panic paths, YAML typo acceptance, NaN/Inf validation gaps, redundant code, style nits.

**Catches:** Exported mutable globals, user-controlled panic paths, YAML typo acceptance, NaN/Inf validation gaps, redundant code, style nits.

#### Perspective 6: DES Expert

Check for: event ordering bugs, clock monotonicity violations, stale signal propagation between event types, heap priority errors, event-driven race conditions, work-conserving property violations.

**Catches:** Event ordering violations, clock regression, stale-signal bugs, work-conserving property gaps.

#### Perspective 7: vLLM/SGLang Expert

Check for: batching semantics that don't match real continuous-batching servers, KV cache eviction mismatches, chunked prefill behavior errors, preemption policy differences, missing scheduling features. Flag any assumption about LLM serving that this code gets wrong.

**Catches:** Batching model inaccuracies, KV cache behavior mismatches, scheduling assumption violations.

#### Perspective 8: Distributed Inference Platform Expert

Check for: multi-instance coordination bugs, routing load imbalance, stale snapshot propagation, admission control edge cases, horizontal scaling assumptions, prefix-affinity routing correctness.

**Catches:** Load imbalance, stale routing state, scaling assumption violations, cross-instance bugs.

#### Perspective 9: Performance & Scalability

Check for: algorithmic complexity issues, unnecessary allocations in hot paths, map iteration in O(n) loops, benchmark-sensitive changes, memory growth patterns, changes that would degrade performance at 1000+ requests or 10+ instances.

**Catches:** Complexity regressions, hot-path allocations, memory growth, scalability bottlenecks.

#### Perspective 10: Security & Robustness

Check for: input validation completeness, panic paths reachable from user input, resource exhaustion vectors, degenerate input handling (empty, zero, NaN, Inf), configuration injection risks.

**Catches:** Validation gaps, user-reachable panics, resource exhaustion, degenerate input failures.

#### Filing Pre-Existing Issues

Review passes naturally surface pre-existing bugs in surrounding code. These are valuable discoveries but outside the current PR's scope.

**Rule:** File a GitHub issue immediately. Do not fix in the current PR.

```bash
gh issue create --title "Bug: <concise description>" --body "<location, impact, discovery context>" --label bug
```

**Why not fix in-PR?**
- **Scope creep** — muddies the diff, makes review harder, risks introducing regressions in unrelated code
- **Attribution** — the fix deserves its own tests and its own commit history
- **Tracking** — issues that aren't filed are issues that are lost

#### After Convergence: Verification Gate

After fixing issues from all passes, run the verification gate to ensure all claims are backed by evidence:

```bash
go build ./...            # Build passes
go test ./... -count=1    # All tests pass (with counts)
golangci-lint run ./...   # Zero lint issues
git status                # Working tree status
```

Report: build exit code, test pass/fail counts, lint issue count, working tree status. Wait for user approval before proceeding.

**Why a skill instead of prose?** In PR9, the manual "run these commands" instruction was easy to skip or half-execute. The skill makes verification non-optional and evidence-based.

!!! tip "Automation"
    `/superpowers:verification-before-completion` enforces running these commands and confirming output before making any success claims.

---

### Step 4.75: Pre-Commit Self-Audit

Stop, think critically, and answer each question below from your own reasoning. Do not delegate to automated tools — review each dimension yourself using critical thinking. Report all issues found. If you find zero issues, explain why you're confident for each dimension.

**Why this step exists:** In PR9, the 4-perspective automated code review (Step 4.5) found 0 new issues in the final perspective. Then the user asked "are you confident?" and Claude found 3 real bugs by thinking critically: a wrong reference scale for token throughput normalization, non-deterministic map iteration in output, and inconsistent comment patterns. Automated review perspectives check structure; this step checks substance.

**Self-audit dimensions — think through each one:**

1. **Logic bugs:** Trace through the core algorithm mentally. Are there edge cases where the math breaks? Division by zero? Off-by-one? Wrong comparisons?
2. **Design bugs:** Does the design actually achieve what the contracts promise? Would a user get the expected behavior? Are there scale mismatches, unit confusions, or semantic errors?
3. **Determinism (R2, INV-6):** Is all output deterministic? Any map iteration used for ordered output? Any floating-point accumulation order dependencies?
4. **Consistency:** Are naming patterns consistent across all changed files? Do comments match code? Do doc strings match implementations? Are there stale references?
5. **Documentation:** Would a new user find everything they need? Would a contributor know how to extend this? Are CLI flags documented everywhere (CLAUDE.md, README, `--help`)?
6. **Defensive edge cases:** What happens with zero input? Empty collections? Maximum values? What if the user passes unusual but valid flag combinations?
7. **Test epistemology (R7, R12):** For every test that compares against a golden value, ask: "How do I know this expected value is correct?" If the answer is "because the code produced it," that test catches regressions but not pre-existing bugs. Verify a corresponding invariant test validates the result from first principles. (See issue #183: a golden test perpetuated a silently-dropped request for months.)
8. **Construction site uniqueness (R4):** Does this PR add fields to existing structs? If so, are ALL construction sites updated? Grep for `StructName{` across the codebase. Are there canonical constructors, or are structs built inline in multiple places?
9. **Error path completeness (R1, R5):** For every error/failure path in new code, what happens to partially-mutated state? Does every `continue` or early `return` clean up what was started? Is there a counter or log so the failure is observable?
10. **Documentation DRY (source-of-truth map):** Does this PR modify content that exists as a working copy elsewhere? Check the source-of-truth map in `docs/contributing/standards/principles.md`. If a canonical source was updated, verify all working copies listed in the map are also updated.

**Why no agent?** Agents are good at pattern-matching (finding style violations, checking structure). They're bad at stepping back and asking "does this actually make sense?" That requires the kind of critical thinking that only happens when you deliberately pause and reflect.

**Fix all issues found. Then wait for user approval before Step 5.**

---

### Step 5: Commit, Push, and Create PR

Stage your changes, commit, push, and create a PR:

```bash
git add <files>
git commit -m "feat(scope): <description>

- Implement BC-1: <brief description>
- Implement BC-2: <brief description>

Co-Authored-By: Claude <noreply@anthropic.com>"
git push -u origin <branch-name>
gh pr create --title "<title>" --body "<description with behavioral contracts>"
```

The PR description should include a summary, behavioral contracts (GIVEN/WHEN/THEN), testing verification, and GitHub closing keywords from the plan's `Closes:` field (e.g., `Fixes #183, fixes #189`).

!!! tip "Automation"
    `/commit-commands:commit-push-pr` handles staging, committing, pushing, and PR creation in one command. It analyzes current git state, creates an appropriate commit message referencing behavioral contracts, and opens the PR.

---

## Workflow Variants

### Subagent-Driven Development (In-Session)

Alternative to Step 4 for simpler PRs where you want tighter iteration. Executes in the current session with a fresh subagent per task and immediate code review after each task.

**Trade-offs:** ✅ Faster for simple PRs (no session switching), better for iterative refinement. ⚠️ Uses current session's context (can grow large), review after every task (vs continuous execution).

!!! tip "Automation"
    Use the `superpowers:subagent-driven-development` skill to implement the plan with fresh subagent per task.

### PR Size Tiers

Not all PRs need the same level of review. Use these objective criteria:

| Tier | Criteria | Plan Review (Step 2.5) | Code Review (Step 4.5) | Self-Audit (Step 4.75) |
|------|----------|----------------------|----------------------|----------------------|
| **Small** | Docs-only with no process/workflow semantic changes (typo fixes, formatting, comment updates, link fixes), OR ≤3 files changed AND only mechanical changes (renames, formatting) AND no behavioral logic changes AND no new interfaces/types AND no new CLI flags | Skip convergence review; single pre-pass sufficient | Skip convergence review; single pre-pass sufficient | Full (all 10 dimensions) |
| **Medium** | 4–10 files changed, OR new policy template behind existing interface | Full two-stage (pre-pass + convergence) | Full two-stage (pre-pass + convergence) | Full (all 10 dimensions) |
| **Large** | >10 files, OR new interfaces/modules, OR architecture changes | Full two-stage (pre-pass + convergence) | Full two-stage (pre-pass + convergence) | Full (all 10 dimensions) |

**Rules:**
- **Steps 1, 2, 3, 4, 5 are always required** — worktree, plan, human review, execution, and commit apply to all tiers.
- **Self-audit is always full** — the 10-dimension critical thinking check catches substance bugs that no automated review can. It costs 5 minutes and has caught 3+ real bugs in every PR where it was applied.
- **When in doubt, tier up** — if you're unsure whether a change is Small or Medium, use Medium.
- **Human reviewer can override** — if the human reviewer at Step 3 believes the tier is wrong, they can request a different tier.

---

## Example Walkthrough

A typical PR from a macro plan section:

1. **Create worktree:** `git worktree add .worktrees/pr8-routing -b pr8-routing && cd .worktrees/pr8-routing`
2. **Write plan:** Follow [micro-plan template](templates/micro-plan.md), save to `docs/plans/pr8-routing-plan.md`
3. **Review plan:** Run all 10 perspectives (Stage 1 pre-pass, then Stage 2 convergence). Fix issues, re-run until converged.
4. **Human review:** Read plan — contracts, tasks, appendix. Approve to proceed.
5. **Implement:** Execute TDD tasks: test → fail → implement → pass → lint → commit.
6. **Review code:** Same two-stage review as plan. Fix issues, re-run until converged. Run verification gate.
7. **Self-audit:** Think through all 10 dimensions. Fix issues.
8. **Commit + PR:** Push branch, create PR with closing keywords.

The workflow is the same regardless of source (macro plan, design doc, GitHub issues). Only the source document passed to the planning step differs.

---

## Tips for Success

1. **Use automated reviews proactively** — run reviews after plan creation and after implementation (don't wait for human review to catch issues)
2. **Fix critical issues immediately** — don't proceed with known critical issues (they compound)
3. **Re-run targeted reviews after fixes** — verify fixes worked
4. **Use worktrees for complex PRs** — avoid disrupting main workspace
5. **Review after execution** — use automated code review (Step 4.5) after all tasks complete
6. **Reference contracts in commits** — makes review easier and more traceable
7. **Update CLAUDE.md immediately** — don't defer documentation
8. **Keep source documents updated** — mark PRs as completed in macro plan; close resolved issues
9. **Don't trust automated passes alone** — the self-audit (Step 4.75) catches substance bugs that pattern-matching agents miss. In PR9, 3 real bugs were found by critical thinking after 4 automated passes found 0 issues.
10. **Checkpoint long sessions** — for PRs with 8+ tasks or multi-round reviews, write a checkpoint summary to `.claude/checkpoint.md` after each major phase. If you hit context limits, read the checkpoint first.

### Headless Mode for Reviews (Context Overflow Workaround)

If multi-agent review passes hit "Prompt is too long" errors during consolidation, switch to headless mode: run each review agent as an isolated invocation that writes findings to a file, then consolidate in a lightweight final pass.

```bash
#!/bin/bash
# headless-review.sh — Run review agents with full context each
BRANCH=$(git branch --show-current)
mkdir -p .review

# Run each pass in its own context (no overflow)
claude -p "Pass 1: Code quality review of branch $BRANCH. Write findings to .review/01-code-quality.md" \
  --allowedTools "Read,Grep,Glob,Bash" &
claude -p "Pass 2: Test behavioral quality review. Write findings to .review/02-test-quality.md" \
  --allowedTools "Read,Grep,Glob,Bash" &
claude -p "Pass 3: Getting-started review. Write findings to .review/03-getting-started.md" \
  --allowedTools "Read,Grep,Glob,Bash" &
wait

# Lightweight consolidation
claude -p "Read .review/*.md files. Produce a consolidated summary sorted by severity." \
  --allowedTools "Read,Glob"
```

When to use: When Step 2.5 or Step 4.5 hits context limits. Not needed for most PRs.

---

## Common Issues and Solutions

### Issue: Plan too generic, agents ask clarifying questions

**Solution:** Add specific guidance in the invocation. Claude reads the full source document and extracts context automatically. If still too generic, add explicit notes about integration points.

### Issue: Tasks miss behavioral contracts during execution

**Solution:** After execution completes, verify all contracts are tested: "Confirm all contracts are tested: BC-1: Show test results. BC-2: Show test results."

### Issue: Lint fails at the end with many issues

**Solution:** Ensure lint runs in each task step: `golangci-lint run ./path/to/modified/package/...`

### Issue: Dead code introduced (unused functions, fields)

**Solution:** In Step 3 plan review, check every struct field is used by end of task or a later task. Every method called by tests or production code. Every parameter actually needed.

### Issue: Review finds many critical issues, overwhelming to fix

**Solution:** Fix issues in priority order: (1) Fix all critical, re-run. (2) Fix important, re-run. (3) Consider suggestions. Use targeted review after fixes.

### Issue: Uncertain if review findings are valid

**Solution:** Review agents provide file:line references. Check the specific code location. If uncertain, ask Claude to explain. If agent is wrong, document why and proceed.

---

<details>
<summary>Appendix: Workflow Evolution</summary>

**v1.0 (pre-2026-02-14):** Manual agent team prompts, separate design/execution plans
**v2.0 (2026-02-14):** Unified planning with `writing-plans` skill, batch execution with `executing-plans` skill, automated two-stage review with `pr-review-toolkit:review-pr`, simplified invocations with @ file references
**v2.1 (2026-02-16):** Same-session worktree workflow (project-local `.worktrees/` no longer requires new session); continuous execution replaces batch checkpoints (tasks run without pausing, stop only on failure)
**v2.2 (2026-02-16):** Focused review passes replace generic review-pr invocations. Step 2.5 expanded to 3 passes (cross-doc consistency, architecture boundary, codebase readiness). Step 4.5 expanded to 4 passes (code quality, test behavioral quality, getting-started experience, automated reviewer simulation). Based on PR8 experience where each focused pass caught issues the others missed.
**v2.3 (2026-02-16):** Step 2.5 expanded to 4 passes — added Pass 4 (structural validation: task dependencies, template completeness, executive summary clarity, under-specified task detection). Based on PR9 experience where deferred items fell through cracks in the macro plan, and an under-specified documentation task would have confused the executing agent.
**v2.4 (2026-02-16):** Four targeted skill integrations addressing real failure modes: (1) `review-plan` as Pass 0 in Step 2.5 — external LLM review catches design bugs that self-review misses (PR9: fitness normalization bug passed 3 focused passes). (2) `superpowers:systematic-debugging` as on-failure handler in Step 4 — structured root-cause analysis instead of ad-hoc debugging. (3) `superpowers:verification-before-completion` replaces manual verification prose after Step 4.5 — makes build/test/lint gate non-skippable. (4) `commit-commands:clean_gone` as pre-cleanup in Step 1 — prevents stale branch accumulation.
**v2.5 (2026-02-16):** Three additions from `/insights` analysis of 212 sessions: (1) Step 4.75 (pre-commit self-audit) — deliberate critical thinking step with no agent, checking logic/design/determinism/consistency/docs/edge-cases. In PR9, this step found 3 real bugs that 4 automated passes missed. (2) Headless mode documentation for review passes. (3) Checkpointing tip for long sessions.
**v2.6 (2026-02-18):** Two additions: (1) "Filing Pre-Existing Issues" subsection to Step 4.5. (2) Antipattern prevention from hardening audit — Step 4.75 expanded to 9 self-audit dimensions; Step 4.5 Pass 1 prompt expanded with 4 antipattern checks.
**v2.7 (2026-02-18):** Generalized workflow from "macro plan only" to any source document (macro plan sections, design docs, GitHub issues, feature requests).
**v2.8 (2026-02-18):** Auto-close issues on PR merge. Added `Closes:` field to micro plan header template.
**v2.9 (2026-02-20):** Convergence re-run protocol for both Step 2.5 and Step 4.5.
**v3.0 (2026-02-23):** Multi-perspective rounds replace sequential passes. External LLM review removed. Convergence redefined as property of a clean round.
**v4.0 (2026-02-27):** Human-first rewrite (#464). Steps describe human actions; skills in admonition callouts. Manual path is primary; automation is additive. Templates split into human-readable format descriptions + agent prompt companions.

</details>
