# PR Development Workflow

**Status:** Active (v2.9 - updated 2026-02-20)

This document describes the complete workflow for implementing a PR from any source: a macro plan section, GitHub issues, a design document, or a feature request.

## Current Template Versions

**Update this section when templates change. All examples below reference these versions.**

- **Design guidelines:** `docs/templates/design-guidelines.md` — DES foundations, module architecture, extension framework. Read before writing any design doc or macro plan.
- **Macro-planning template:** `docs/templates/macro-plan.md` (updated 2026-02-18 — aligned with design guidelines)
- **Micro-planning template:** `docs/templates/micro-plan.md` (updated 2026-02-18)
- **Active macro plan:** `docs/plans/2026-02-19-weighted-scoring-macro-plan.md` (scorer framework)
- **Archived design docs:** `docs/plans/archive/` (completed design docs for reference)

---

## Prerequisites

This workflow requires the following Claude Code skills to be available:

| Skill | Purpose | Used In |
|-------|---------|---------|
| `commit-commands:clean_gone` | Clean up stale branches before worktree creation | Step 1 (pre-cleanup) |
| `superpowers:using-git-worktrees` | Create isolated workspace for PR work | Step 1 |
| `superpowers:writing-plans` | Generate implementation plan from templates | Step 2 |
| `review-plan` | External LLM review of plan (catches blind spots) | Step 2.5 (Pass 0) |
| `pr-review-toolkit:review-pr` | Automated multi-agent review | Step 2.5, Step 4.5 |
| `superpowers:executing-plans` | Execute plan tasks continuously | Step 4 |
| `superpowers:systematic-debugging` | Structured root-cause analysis on failure | Step 4 (on failure) |
| `superpowers:verification-before-completion` | Enforced verification gate before commit | Step 4.5 (after passes) |
| `commit-commands:commit-push-pr` | Commit, push, and create PR | Step 5 |

**Verification:**
```bash
# List all available skills
/agents
```

**If a required skill is unavailable:**
- Check Claude Code version (skills may be added in newer versions)
- Check skill installation in `~/.claude/plugins/` or `~/.claude/skills/`
- For official superpowers skills, ensure plugins are up to date

**Alternative workflows:**
If skills are unavailable, you can implement each step manually:
- Step 1: Use `git worktree add ../repo-prN -b prN-name` directly
- Step 2: Follow `docs/templates/micro-plan.md` template manually
- Step 2.5/4.5: Manual code review or skip automated review
- Step 4: Implement tasks manually following plan; on failure, debug manually
- Step 4.75: Self-audit is always available (no skill required — just critical thinking)
- Step 5: Use standard git commands (`git add`, `git commit`, `git push`, `gh pr create`)

---

## Overview

```
┌─────────────────────────┐
│ Step 1: using-worktrees │ (Create isolated workspace for PR)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Step 2: writing-plans   │ (Create behavioral contracts + executable tasks)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Step 2.5: plan review   │ (5 passes: external + 3 focused + structural)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Step 3: Human Review    │ (Approve plan)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Step 4: executing-plans │ (Implement tasks continuously)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Step 4.5: code review   │ (4 focused review passes — see checklist)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Step 4.75: self-audit   │ (Deliberate critical thinking — no agent)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Step 5: commit-push-pr  │ (Commit, push, create PR - all in one)
└─────────────────────────┘
```

**Key insights:**
1. **Worktree isolation from start** (Step 1) - Create worktree BEFORE any work
   - Entire PR lifecycle (planning + implementation) in isolated workspace
   - Main worktree never touched
   - Enables parallel work on multiple PRs

2. **Three-stage quality assurance:**
   - **Plan Review** (Step 2.5) - 5 passes: external LLM, cross-doc, architecture, codebase, structural
     - Catches design issues before implementation
   - **Code Review** (Step 4.5) - 4 passes: code quality, test quality, getting-started, automated reviewer
     - Catches implementation issues before PR creation
   - **Self-Audit** (Step 4.75) - Deliberate critical thinking across 9 dimensions
     - Catches substance bugs that pattern-matching agents miss

---

## Quick Reference: Simplified Invocations

**No copy-pasting required!** Use @ file references and simple commands:

| Step | Command |
|------|---------|
| **1. Create worktree** | `/superpowers:using-git-worktrees pr<N>-<name>` |
| **2. Create plan** | `/superpowers:writing-plans for <work-item> in @docs/plans/<name>-plan.md using @docs/templates/micro-plan.md and @<source-document>` |
| **2.5. Review plan** | 5 passes: external review, cross-doc consistency, architecture boundary, codebase readiness, structural validation |
| **3. Human review plan** | Review contracts, tasks, appendix, then approve to proceed |
| **4. Execute plan** | `/superpowers:executing-plans @docs/plans/pr<N>-<name>-plan.md` |
| **4.5. Review code** | 4 focused passes: code quality, test behavioral quality, getting-started, automated reviewer sim |
| **4.75. Self-audit** | Deliberate critical thinking: logic, design, determinism, consistency, docs, edge cases, test epistemology, construction sites, error paths |
| **5. Commit, push, PR** | `/commit-commands:commit-push-pr` |

**Example for PR 8 (same-session workflow with `.worktrees/`):**
> **Note:** Examples below use file paths from completed PRs. Referenced plan files may have been archived or removed. Adapt the pattern using current plans from `docs/plans/`.
```bash
# Step 1: Create worktree (stays in same session)
/superpowers:using-git-worktrees pr8-routing-state-and-policy-bundle

# Output: Worktree ready at .worktrees/pr8-routing-state-and-policy-bundle/
# (shell cwd already switched — continue directly)

# Step 2: Create plan
/superpowers:writing-plans for PR8 in @docs/plans/pr8-routing-state-and-policy-bundle-plan.md using @docs/templates/micro-plan.md and @docs/plans/2026-02-11-macro-implementation-plan-v2.md

# Step 2.5: Review plan
/pr-review-toolkit:review-pr

# Step 3: Human review
# [Read plan, verify contracts and tasks, approve to proceed]

# Step 4: Execute implementation
/superpowers:executing-plans @docs/plans/pr8-routing-state-and-policy-bundle-plan.md

# Step 4.5: Review code
/pr-review-toolkit:review-pr

# Step 5: Commit, push, and create PR
/commit-commands:commit-push-pr
```

**Example (separate-session workflow with sibling directory):**
```bash
# Step 1: Create worktree
/superpowers:using-git-worktrees pr6-routing-policy
# Output: Worktree created at ../inference-sim-pr6/

# Open NEW Claude Code session in worktree:
# Terminal: cd ../inference-sim-pr6/ && claude

# Steps 2-5: Same as above, in the new session
```

---

## Step-by-Step Process

### Step 1: Create Isolated Worktree Using `using-git-worktrees` Skill

**Context:** Main repo (inference-sim)

**Pre-cleanup (optional):** Run `commit-commands:clean_gone` to remove stale local branches whose remote tracking branches have been deleted. This prevents `.worktrees/` from accumulating cruft from previous PRs.

```
/commit-commands:clean_gone
```

**Skill:** `superpowers:using-git-worktrees`

**Why first?** Create isolated workspace BEFORE any work begins. This ensures:
- Main worktree stays clean (no uncommitted plans or code)
- Plan document committed on feature branch (not main)
- Complete isolation for entire PR lifecycle (planning + implementation)
- Ability to work on multiple PRs in parallel

**Invocation (simplified):**
```
/superpowers:using-git-worktrees pr<N>-<feature-name>
```

**Example:**
```
/superpowers:using-git-worktrees pr6-routing-policy
```

**What Happens:**
- Creates a new git worktree (project-local in `.worktrees/` or as a sibling directory)
- Creates and checks out a new branch (`pr6-routing-policy`)
- Shell working directory switches to the worktree
- Isolates work from main development

**Output:** Path to worktree directory (e.g., `.worktrees/pr6-routing-policy/`)

**Next — choose one:**

- **Continue in same session (recommended for `.worktrees/`):** The skill already switched your working directory into the worktree. You can proceed directly to Step 2 in the same Claude Code session.

- **Open a new session (required for sibling directories):** If the worktree is outside the project (e.g., `../inference-sim-pr6/`), open a new Claude Code session there:
  ```bash
  cd ../inference-sim-pr6/
  claude
  ```

**All remaining steps happen in the worktree (same session or new session).**

---

### Step 2: Create Implementation Plan Using `writing-plans` Skill

**Context:** Worktree (same or new session)

**Skill:** `superpowers:writing-plans`

**Invocation (simplified):**
```
/superpowers:writing-plans for <work-item> in @docs/plans/<name>-plan.md using @docs/templates/micro-plan.md and @<source-document>
```

The `<source-document>` can be any of:
- A macro plan: `@docs/plans/<active-macro-plan>.md` (e.g., `@docs/plans/2026-02-19-weighted-scoring-macro-plan.md`)
- A design document: `@docs/plans/<design-doc>.md` or `@docs/plans/archive/<design-doc>.md`
- GitHub issues: reference by number in the prompt text (e.g., "for issues #183, #189, #195")

**Examples:**
```
# From macro plan section:
/superpowers:writing-plans for PR6 in @docs/plans/pr6-routing-policy-plan.md using @docs/templates/micro-plan.md and @docs/plans/2026-02-11-macro-implementation-plan-v2.md

# From design document:
/superpowers:writing-plans for hardening PR in @docs/plans/hardening-plan.md using @docs/templates/micro-plan.md and @docs/plans/2026-02-18-hardening-antipattern-refactoring-design.md

# From GitHub issues:
/superpowers:writing-plans for issues #183 #189 #195 in @docs/plans/kv-bugfix-plan.md using @docs/templates/micro-plan.md
```

**What Happens:**
- Claude reads the source document (macro plan section, design doc, or issue descriptions)
- Claude reads `docs/templates/micro-plan.md` as the template
- Claude inspects the codebase (Phase 0: Component Context)
- Claude creates behavioral contracts (Phase 1)
- Claude breaks implementation into 6-12 TDD tasks (Phase 4)
- Claude saves plan to the specified output file **in the worktree**

**Output:**
- Plan file at `docs/plans/<name>-plan.md` (in worktree, on feature branch)
- Contains: behavioral contracts, executable tasks, test strategy, appendix

**Tips:**
- Use @ file references instead of copy-pasting
- Claude automatically extracts the relevant context from the source document
- Template structure is preserved automatically
- For issue-based work, Claude reads issue details via `gh issue view`

---

### Step 2.5: Focused Plan Review (5 Passes)

**Context:** Worktree (same or new session)

> **For Claude:** When the user asks you to execute Step 2.5, run all 5 review passes below
> sequentially. Pass 0 uses `review-plan`. Passes 1-3 use `/pr-review-toolkit:review-pr` with
> the **exact prompt text** shown in the `Prompt:` field. Pass 4 is done directly (no agent
> needed). After each pass, summarize findings and fix all critical/important issues before
> starting the next pass. After all 5 passes complete, report a summary to the user and wait
> for approval to proceed.

**Why focused passes?** Generic "review everything" misses issues that targeted reviews catch. PR8 experience: Pass 1 found 6 categories of stale macro plan content. PR9 experience: the 3 focused passes missed a fitness normalization design bug that was only caught by a separate targeted review. Pass 0 (external LLM review) catches blind spots that self-review misses.

---

#### Pass 0: External Plan Review (Quick Pre-Screen)

Get an independent second opinion on the plan before detailed passes.

**Skill:** `review-plan`

**Invocation:**
```
/review-plan @docs/plans/pr<N>-<name>-plan.md
```

**Catches:** Design bugs, mathematical errors, logical inconsistencies, missing edge cases — issues that focused passes miss because they check structure, not substance.

**Why this pass exists:** In PR9, the fitness normalization formula (`1/(1+value)`) passed all 3 focused passes but was a design bug (500,000× scale imbalance between throughput and latency). An external review caught what self-review missed.

**Fix critical/important issues before Pass 1.**

---

#### Pass 1: Cross-Document Consistency

Verify the micro plan matches the source document and both match the codebase.

**Prompt:**
```
/pr-review-toolkit:review-pr Does the micro plan's scope match the source document? Are file paths consistent? Does the deviation log account for all differences between what the source says and what the micro plan does? Check for stale references. @docs/plans/<name>-plan.md and @<source-document>
```

For issue-based work, replace `@<source-document>` with the issue numbers in the prompt text.

**Catches:** Stale references, scope mismatch, missing deviations, wrong file paths.

**Fix all critical/important issues before Pass 2.**

---

#### Pass 2: Architecture Boundary Verification

Verify the plan respects architectural boundaries and separation of concerns.

**Prompt:**
```
/pr-review-toolkit:review-pr Does this plan maintain architectural boundaries? Are we ensuring individual instances don't have access to cluster-level state? Are types in the right packages? Check the plan against the actual code for boundary violations. Also check: (1) does the plan introduce multiple construction sites for the same type? (2) does adding one field to a new type require >3 files? (3) does library code (sim/) call logrus.Fatalf anywhere in the new code? @docs/plans/pr<N>-<name>-plan.md
```

**Catches:** Import cycle risks, boundary violations, missing bridge types, wrong abstraction level, construction site proliferation, high touch-point multipliers, error handling boundary violations.

**Fix all critical/important issues before Pass 3.**

---

#### Pass 3: Codebase Readiness

Verify the files to be modified are clean and ready for the planned changes.

**Prompt:**
```
/pr-review-toolkit:review-pr We're about to implement this PR. Review the codebase for readiness. Check each file the plan will modify for stale comments, existing bugs, or issues that would complicate implementation. @docs/plans/pr<N>-<name>-plan.md
```

**Catches:** Stale comments ("planned for PR N"), pre-existing bugs, missing dependencies, unclear insertion points.

**Fix all critical/important issues. Then proceed to Pass 4.**

---

#### Pass 4: Plan Structural Validation

Verify the plan is complete, internally consistent, and implementation-ready.

> **For Claude:** Perform these 4 checks directly (no agent needed). Report all issues found.

**Check 1: Task Dependencies**
- For each task, verify it can actually start given what comes before it.
- Trace the dependency chain: what files does each task create/modify? Does any task require a file or type that hasn't been created yet?
- Flag tasks that modify the same file and could conflict.

**Check 2: Template Completeness**
- Verify all sections from `docs/templates/micro-plan.md` are present and non-empty:
  - Header (Goal, Architecture, Source Reference)
  - Part 1: A) Executive Summary, B) Behavioral Contracts, C) Component Interaction, D) Deviation Log, E) Review Guide
  - Part 2: F) Implementation Overview, G) Task Breakdown, H) Test Strategy, I) Risk Analysis
  - Part 3: J) Sanity Checklist
  - Appendix: File-Level Details

**Check 3: Executive Summary Clarity**
- Read the executive summary as if you're a new team member with no context.
- Is it clear what the PR does and why?
- Can you understand the scope without reading the rest of the plan?

**Check 4: Under-specified Tasks**
- For each task, verify it has complete code in every step (no "add validation" without showing exact code).
- Verify exact test commands with expected output.
- Verify exact commit commands.
- Flag any step that an executing agent would need to figure out on its own.

**Catches:** Broken task ordering, missing template sections, unclear summaries, vague implementation steps that will cause agent confusion.

**Fix all issues. Then report summary to user.**

---

### Step 3: Human Review of Plan

**Context:** Worktree (same or new session)

**Action:** Final human review of the plan (after automated review)

**Focus Areas:**
1. **Part 1 (Design Validation)** - Review behavioral contracts, component interaction, risks
2. **Part 2 (Executable Tasks)** - Verify task breakdown makes sense, no dead code
3. **Deviation Log** - Check if deviations from source document are justified
4. **Appendix** - Spot-check file-level details for accuracy

**Common Issues to Catch:**
- Behavioral contracts too vague or missing edge cases
- Tasks not properly ordered (dependencies)
- Missing test coverage for contracts
- Deviations from source document not justified
- Dead code or scaffolding

**Outcome:**
- ✅ Approve plan → proceed to Step 4 (implementation)
- ❌ Need revisions → iterate with Claude, re-review (Step 2.5), then approve

**Note:** The plan will be committed together with the implementation in Step 5 (single commit for entire PR).

---

### Step 4: Execute Plan Using `executing-plans` Skill

**Context:** Worktree (same or new session)

**Skill:** `superpowers:executing-plans`

**Invocation (simplified):**
```
/superpowers:executing-plans @docs/plans/pr<N>-<feature-name>-plan.md
```

**Example:**
```
/superpowers:executing-plans @docs/plans/pr6-routing-policy-plan.md
```

**That's it!** The skill automatically:
- Reads the plan
- Executes all tasks continuously (no pausing for human input)
- Stops only on test failure, lint failure, or build error

**What Happens:**
- Claude reads the plan file
- Claude executes all tasks sequentially without pausing
  - Each task: write test → verify fail → implement → verify pass → lint → commit
- If a failure occurs, Claude stops and reports the issue
- On success, all tasks complete and Claude reports a summary

**On Failure:** If a task fails (test failure, build error, lint error), invoke structured debugging instead of ad-hoc poking:

```
/superpowers:systematic-debugging
```

This skill provides structured root-cause analysis: reproduce → isolate → hypothesize → verify → fix. Prevents the common pattern of making random changes hoping the test passes. After the fix, resume execution from the failing task.

**Output:**
- Implemented code in working directory
- All tests passing (`go test ./...`)
- Lint clean (`golangci-lint run ./...`)
- Commits for each task (referencing contracts)

---

### Step 4.5: Focused Code Review (4 Passes)

**Context:** Worktree (after implementation complete)

> **For Claude:** When the user asks you to execute Step 4.5, run all 4 review passes below
> sequentially. For each pass: invoke `/pr-review-toolkit:review-pr` with the **exact prompt
> text** shown in the `Prompt:` field. After each pass, summarize findings and fix all
> critical/important issues before starting the next pass. After all 4 passes complete, run
> full verification (`go build && go test ./... -count=1 && golangci-lint run`), then report
> a summary to the user and wait for approval to proceed.

**Why 4 passes?** Each catches issues the others miss. PR8 experience: Pass 1 found mutable exported maps and YAML typo acceptance. Pass 2 found 2 structural tests and 3 mixed tests. Pass 3 found missing metrics glossary and contributor guide gaps. Pass 4 found CLI panic paths and NaN validation gaps.

---

#### Pass 1: Code Quality + Error Handling

Find bugs, logic errors, silent failures, and convention violations.

**Prompt:**
```
/pr-review-toolkit:review-pr Also check: (1) any new error paths that use `continue` or early `return` — do they clean up partial state? (2) any map iteration that accumulates floats — are keys sorted? (3) any struct field added — are all construction sites updated? (4) does library code (sim/) call logrus.Fatalf anywhere in new code? (5) any exported mutable maps — should they be unexported with IsValid*() accessors? (6) any YAML config fields using float64 instead of *float64 where zero is valid? (7) any division where the denominator derives from runtime state without a zero guard? (8) any new interface with methods only meaningful for one implementation? (9) any method >50 lines spanning multiple concerns (scheduling + latency + metrics)?
```

**Catches:** Logic errors, nil pointer risks, silent failures (discarded return values), panic paths reachable from user input, CLAUDE.md convention violations, dead code, silent `continue` data loss, non-deterministic map iteration, construction site drift, library code calling os.Exit, exported mutable maps, YAML zero-value ambiguity, division by zero in runtime computation, leaky interfaces, monolith methods.

**Fix all critical/important issues before Pass 2.**

---

#### Pass 2: Test Behavioral Quality

Verify tests are truly behavioral (testing WHAT) not structural (testing HOW).

**Prompt:**
```
/pr-review-toolkit:review-pr Are all the tests well written and truly behavioral? Do they test observable behavior (GIVEN/WHEN/THEN) or just assert internal structure? Would they survive a refactor? Rate each test as Behavioral, Mixed, or Structural. Also check: are there any golden dataset tests (comparing against captured output values) that lack a corresponding invariant test? Golden tests encode current behavior as "correct" — if the code had a bug when the golden values were captured, the test perpetuates the bug. Flag any golden test whose expected values are not independently validated by an invariant test (e.g., request conservation, KV block conservation, causality).
```

**Catches:** Structural tests (Go struct assignment, trivial getters), type assertions in factory tests, exact-formula assertions instead of behavioral invariants, tests that pass even if the feature is broken, golden-only tests that would perpetuate pre-existing bugs (issue #183: codellama golden dataset encoded a silently-dropped request as the expected value since its initial commit).

**Fix: Delete structural tests, replace type assertions with behavior assertions. Add invariant tests alongside any golden-only tests.**

---

#### Pass 3: Getting-Started Experience

Simulate the journey of a new user and a new contributor.

**Prompt:**
```
/pr-review-toolkit:review-pr Is it easy for a user and contributor to get started? Simulate both journeys: (1) a user doing capacity planning with the CLI, and (2) a contributor adding a new algorithm. Where would they get stuck? What's missing?
```

**Catches:** Missing example files, undocumented output metrics, incomplete contributor guide, unclear extension points, README not updated for new features.

**Fix: Add examples, glossaries, contributor docs. Then proceed to Pass 4.**

---

#### Pass 4: Automated Reviewer Simulation

Catch what GitHub Copilot, Claude, and Codex would flag.

**Prompt:**
```
/pr-review-toolkit:review-pr The upstream community uses github copilot, claude, codex apps to perform a review of this PR. Please do a rigorous check (and fix any issues) so that this will pass the review.
```

**Catches:** Exported mutable globals, user-controlled panic paths, YAML typo acceptance, NaN/Inf validation gaps, redundant code, style nits.

**Fix all critical issues. This is the final quality gate.**

---

#### During Any Pass: Filing Pre-Existing Issues

Review passes naturally surface pre-existing bugs in surrounding code. These are valuable discoveries but outside the current PR's scope.

**Rule:** File a GitHub issue immediately. Do not fix in the current PR.

```bash
gh issue create --title "Bug: <concise description>" --body "<location, impact, discovery context>" --label bug
```

**Label guide:** Use `bug` for code defects, `design` for design limitations, `enhancement` for feature gaps, `hardening` for correctness/invariant issues. Every issue must have at least one label — unlabeled issues are invisible in filtered views.

**Why not fix in-PR?**
- **Scope creep** — muddies the diff, makes review harder, risks introducing regressions in unrelated code
- **Attribution** — the fix deserves its own tests and its own commit history
- **Tracking** — issues that aren't filed are issues that are lost

**After filing:** Reference the issue number in the PR description under a "Discovered Issues" section so reviewers know it was found and tracked.

**Example (from #38 log level recalibration):** Code review found that `simulator.go:593` silently drops a request on KV allocation failure — a pre-existing bug predating the PR. Filed as #183 rather than fixing in-scope.

---

#### After All 4 Passes: Convergence Re-Run

**After all passes complete with fixes, re-run all 4 passes from scratch** to verify fixes didn't introduce cross-pass issues. If the re-run finds new CRITICAL or IMPORTANT issues, fix and re-run again. Repeat until convergence: a clean pass with 0 CRITICAL and 0 IMPORTANT findings. Additive-only fixes (e.g., adding a comment without restructuring logic) may skip the re-run.

---

#### After All 4 Passes: Enforced Verification Gate

> **For Claude:** After fixing issues from all passes, invoke the verification skill to ensure
> all claims are backed by evidence. This is non-optional — do NOT skip to Step 5.

**Skill:** `superpowers:verification-before-completion`

```
/superpowers:verification-before-completion
```

This skill enforces running verification commands and confirming output before making any success claims. It requires:
- `go build ./...` — build passes
- `go test ./... -count=1` — all tests pass (with counts)
- `golangci-lint run ./...` — zero lint issues
- `git status` — working tree status reported

**Why a skill instead of prose?** In PR9, the manual "run these commands" instruction was easy to skip or half-execute. The skill makes verification non-optional and evidence-based.

Report: build exit code, test pass/fail counts, lint issue count, working tree status.
Wait for user approval before proceeding to Step 4.75.

---

### Step 4.75: Pre-Commit Self-Audit (No Agent — Deliberate Thinking)

**Context:** Worktree (after verification gate passes)

> **For Claude:** This is NOT an agent pass. Stop, think critically, and answer each question
> below from your own reasoning. Do NOT dispatch agents or read files — you already have the
> full context. Report all issues found. If you find zero issues, explain why you're confident
> for each dimension.

**Why this step exists:** In PR9, the 4-pass automated code review (Step 4.5) found 0 new issues in Pass 4. Then the user asked "are you confident?" and Claude found 3 real bugs by thinking critically: a wrong reference scale for token throughput normalization, non-deterministic map iteration in output, and inconsistent comment patterns. Automated review passes check structure; this step checks substance.

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

**Fix all issues found. Then wait for user approval before Step 5.**

**Why no agent?** Agents are good at pattern-matching (finding style violations, checking structure). They're bad at stepping back and asking "does this actually make sense?" That requires the kind of critical thinking that only happens when you deliberately pause and reflect.

---

### Step 5: Commit, Push, and Create PR Using `commit-commands:commit-push-pr`

**Context:** Worktree (after code review passed and all issues fixed)

**Skill:** `commit-commands:commit-push-pr`

**Invocation (simplified):**
```
/commit-commands:commit-push-pr
```

**What Happens:**
- Reviews git status and staged/unstaged changes
- Creates a commit with appropriate message (or amends if per-task commits exist)
- Pushes branch to origin
- Creates GitHub PR automatically
- All in one command!

**The skill automatically:**
1. Analyzes current git state (per-task commits from Step 4, or uncommitted changes)
2. Creates/amends commit with appropriate message (references behavioral contracts)
3. Pushes branch to origin with `-u` flag
4. Creates PR using `gh pr create` with title and description

**Commit message includes:**
- PR title from source document or work item
- Multi-line description of changes
- List of implemented behavioral contracts (BC-1, BC-2, etc.)
- Co-authored-by line

**PR description includes:**
- Summary from source document
- GitHub closing keywords from the plan's `Closes:` field (e.g., `Fixes #183, fixes #189`) — these auto-close issues on merge
- Behavioral contracts (GIVEN/WHEN/THEN)
- Testing verification
- Checklist of completed items

**Output:**
- Commit(s) pushed to GitHub (per-task commits from Step 4 + plan file)
- PR URL (e.g., `https://github.com/user/repo/pull/123`)

**Note:** If you prefer a single squashed commit, manually squash before Step 5:
```bash
git reset --soft HEAD~N  # N = number of task commits
git commit -m "PR<N>: <title>"
/commit-commands:commit-push-pr
```

---

## Workflow Variants

### Option A: Subagent-Driven Development (In-Session)

**Alternative to Step 4** - Use for simpler PRs where you want tighter iteration:

**Skill:** `superpowers:subagent-driven-development`

**Invocation:**
```
Use the subagent-driven-development skill to implement docs/plans/pr<N>-<feature-name>-plan.md.
```

**Differences:**
- Executes in current session (no separate session needed)
- Fresh subagent per task (better context isolation)
- Immediate code review after each task
- Faster iteration for small changes

**Trade-offs:**
- ✅ Faster for simple PRs (no session switching)
- ✅ Better for iterative refinement
- ⚠️ Uses current session's context (can grow large)
- ⚠️ Review after every task (vs continuous execution in executing-plans)

---

## Skill Reference Quick Guide

| Skill | When to Use | Input | Output |
|-------|-------------|-------|--------|
| `commit-commands:clean_gone` | **Step 1** - Pre-cleanup of stale branches | None | Removed stale branches |
| `using-git-worktrees` | **Step 1** - Create isolated workspace FIRST | Branch name | Worktree directory path |
| `writing-plans` | **Step 2** - Create implementation plan from source document | Source document (macro plan/design doc/issues) + `docs/templates/micro-plan.md` | Plan file with contracts + tasks |
| `review-plan` | **Step 2.5 Pass 0** - External LLM review (catches design bugs) | Plan file path | Independent review feedback |
| `pr-review-toolkit:review-pr` | **Step 2.5 Passes 1-3** - Focused plan review passes | Targeted prompts (see checklist) | Critical/important issues per pass |
| `executing-plans` | **Step 4** - Execute plan tasks continuously | Plan file path | Implemented code + commits |
| `systematic-debugging` | **Step 4 (on failure)** - Structured root-cause analysis | Failing test/error context | Root cause + fix |
| `subagent-driven-development` | **Step 4 (alt)** - Execute plan in-session | Plan file path | Implemented code + commits |
| `pr-review-toolkit:review-pr` | **Step 4.5** - 4 focused code review passes | Targeted prompts (see checklist) | Critical/important issues per pass |
| `verification-before-completion` | **Step 4.5 (gate)** - Enforced build/test/lint verification | None | Evidence-based pass/fail |
| `commit-commands:commit-push-pr` | **Step 5** - Commit, push, create PR (all in one) | Current branch state | Commit + push + PR URL |

---

## Example A: Macro Plan PR Workflow (Same-Session with `.worktrees/`)

> **Note:** Examples below use file paths from completed PRs (PR6, PR8, hardening). Referenced plan files have been archived or removed. Adapt the pattern using current plans from `docs/plans/`.

```bash
# Step 1: Clean up stale branches, then create worktree
/commit-commands:clean_gone
/superpowers:using-git-worktrees pr8-routing-state-and-policy-bundle

# Output: Worktree ready at .worktrees/pr8-routing-state-and-policy-bundle/
# (continue directly — no new session needed)

# Step 2: Create plan (source = macro plan section)
/superpowers:writing-plans for PR8 in @docs/plans/pr8-routing-state-and-policy-bundle-plan.md using @docs/templates/micro-plan.md and @docs/plans/2026-02-11-macro-implementation-plan-v2.md

# Output: Plan created at docs/plans/pr8-routing-state-and-policy-bundle-plan.md

# Step 2.5: Focused plan review (5 passes)
# Pass 0: External LLM review (catches design bugs)
/review-plan @docs/plans/pr8-routing-state-and-policy-bundle-plan.md
# Pass 1: Cross-doc consistency (micro plan vs source document)
/pr-review-toolkit:review-pr Does the micro plan match the source document? @docs/plans/pr8-plan.md and @docs/plans/2026-02-11-macro-implementation-plan-v2.md
# Pass 2: Architecture boundary verification
/pr-review-toolkit:review-pr Does this plan maintain architectural boundaries?
# Pass 3: Codebase readiness scan
/pr-review-toolkit:review-pr Review the codebase for readiness to implement this PR.
# Pass 4: Structural validation (done directly, no agent)
# [Fix issues between passes]

# Step 3: Human review plan
# [Read plan, verify contracts and tasks, approve to proceed]

# Step 4: Execute implementation
/superpowers:executing-plans @docs/plans/pr8-routing-state-and-policy-bundle-plan.md

# Output: Tasks execute continuously → done (stops on failure)

# Step 4.5: Focused code review (4 passes)
# Pass 1: Code quality + error handling
/pr-review-toolkit:review-pr
# Pass 2: Test behavioral quality
/pr-review-toolkit:review-pr Are all the tests truly behavioral?
# Pass 3: Getting-started experience
/pr-review-toolkit:review-pr Is it easy for a user and contributor to get started?
# Pass 4: Automated reviewer simulation
/pr-review-toolkit:review-pr Simulate what Copilot/Claude/Codex would flag.
# [Fix issues between passes]
# Enforced verification gate
/superpowers:verification-before-completion

# Step 4.75: Self-audit (no agent — deliberate critical thinking)
# Think through: logic bugs, design bugs, determinism, consistency, docs, edge cases
# [Fix any issues found, re-verify]

# Step 5: Commit plan + implementation, push, and create PR (all in one!)
/commit-commands:commit-push-pr

# Output:
# - Single commit created (plan + implementation)
# - Branch pushed to origin
# - PR created on GitHub
# - PR URL returned
```

**Key benefit:** No copy-pasting! Just use @ file references and let Claude extract the context. No session switching needed with project-local `.worktrees/`.

---

## Example B: Issue/Design-Doc PR Workflow

```bash
# Step 1: Create worktree for hardening work (source = design doc + issues)
/commit-commands:clean_gone
/superpowers:using-git-worktrees hardening-antipatterns

# Step 2: Create plan (source = design document, not macro plan)
/superpowers:writing-plans for hardening PR in @docs/plans/hardening-plan.md using @docs/templates/micro-plan.md and @docs/plans/2026-02-18-hardening-antipattern-refactoring-design.md

# Step 2.5: Focused plan review (5 passes)
# Pass 0: External LLM review
/review-plan @docs/plans/hardening-plan.md
# Pass 1: Cross-doc consistency (micro plan vs design doc)
/pr-review-toolkit:review-pr Does the micro plan's scope match the design doc? @docs/plans/hardening-plan.md and @docs/plans/2026-02-18-hardening-antipattern-refactoring-design.md
# Passes 2-4: Same as Example A

# Steps 3-5: Identical to Example A
```

**The workflow is the same regardless of source.** The only difference is what you pass as `@<source-document>` in Step 2. The template, review passes, execution, and quality gates are identical.

---

## Tips for Success

1. **Use automated reviews proactively** - Run `review-pr` after plan creation and after implementation (don't wait for human review to catch issues)
2. **Fix critical issues immediately** - Don't proceed with known critical issues (they compound)
3. **Re-run targeted reviews after fixes** - Verify fixes worked: `/pr-review-toolkit:review-pr code tests`
4. **Use worktrees for complex PRs** - Avoid disrupting main workspace
5. **Review after execution** - Use automated code review (Step 4.5) after all tasks complete
6. **Reference contracts in commits** - Makes review easier and more traceable
7. **Update CLAUDE.md immediately** - Don't defer documentation
8. **Keep source documents updated** - Mark PRs as completed in macro plan; close resolved issues
9. **Don't trust automated passes alone** - The self-audit (Step 4.75) catches substance bugs that pattern-matching agents miss. In PR9, 3 real bugs were found by critical thinking after 4 automated passes found 0 issues.
10. **Checkpoint long sessions** - For PRs with 8+ tasks or multi-round reviews, write a checkpoint summary to `.claude/checkpoint.md` after each major phase (planning, implementation, review). If you hit context limits or need to continue in a new session, read the checkpoint first. This prevents losing progress and avoids re-reading the entire conversation history.

### Headless Mode for Reviews (Context Overflow Workaround)

If multi-agent review passes hit "Prompt is too long" errors during consolidation (a recurring friction point), switch to headless mode: run each review agent as an isolated invocation that writes findings to a file, then consolidate in a lightweight final pass.

```bash
#!/bin/bash
# headless-review.sh — Run review agents with full context each
BRANCH=$(git branch --show-current)
PLAN="docs/plans/pr<N>-<name>-plan.md"
mkdir -p .review

# Run each pass in its own context (no overflow)
claude -p "Pass 1: Code quality review of branch $BRANCH. Read all changed Go files. Write findings to .review/01-code-quality.md" \
  --allowedTools "Read,Grep,Glob,Bash" &
claude -p "Pass 2: Test behavioral quality review. Rate each new test. Write findings to .review/02-test-quality.md" \
  --allowedTools "Read,Grep,Glob,Bash" &
claude -p "Pass 3: Getting-started review. Simulate user + contributor journeys. Write findings to .review/03-getting-started.md" \
  --allowedTools "Read,Grep,Glob,Bash" &
wait

# Lightweight consolidation (reads only the small finding files)
claude -p "Read .review/*.md files. Produce a consolidated summary sorted by severity." \
  --allowedTools "Read,Glob"
```

**When to use:** When Step 2.5 or Step 4.5 hits context limits. Not needed for most PRs — only when the conversation history is already long.

### Review Strategy Tips

**Always use focused prompts, not generic invocations.** Each focused review pass catches different issues:

| Pass | What It Catches That Others Miss |
|------|----------------------------------|
| External LLM review | Design bugs, mathematical errors, logical flaws (substance, not structure) |
| Cross-doc consistency | Stale source document references, scope mismatch, wrong file paths |
| Architecture boundary | Import cycles, boundary violations, wrong abstraction level |
| Codebase readiness | Stale comments, pre-existing bugs, missing dependencies |
| Structural validation | Broken task dependencies, missing sections, vague steps, unclear summaries |
| Code quality | Logic errors, silent failures, convention violations |
| Test behavioral quality | Structural tests, type assertions, formula-coupled assertions |
| Getting-started experience | Missing examples, undocumented output, contributor friction |
| Automated reviewer sim | Mutable globals, user-controlled panics, YAML typo acceptance |

**Fix issues between passes, not after all passes.** Fixes from Pass 1 may affect what Pass 2 finds.

**Re-run a pass if you made significant changes** during fixing. Don't assume the fix is correct.

**After all passes complete with fixes, re-run all 5 passes from scratch** to verify fixes didn't introduce cross-pass issues (e.g., a Pass 0 fix breaking Pass 4 structural validation). If the re-run finds new CRITICAL or IMPORTANT issues, fix and re-run again. Repeat until convergence: a clean pass with 0 CRITICAL and 0 IMPORTANT findings. Additive-only fixes (e.g., adding a file to scope without restructuring tasks) may skip the re-run. Evidence: in the Wave 1 parallel PR session, Track B's re-run validated that 3 fixes (including 2 CRITICAL — overwrite-existing-file and incomplete zero-value test) introduced no regressions.

---

## Common Issues and Solutions

### Issue: Plan too generic, agents ask clarifying questions

**Solution:** The simplified invocation with @ references handles this automatically:
```bash
/superpowers:writing-plans for PR6 in @docs/plans/pr6-plan.md using @docs/templates/micro-plan.md and @docs/plans/2026-02-11-macro-implementation-plan-v2.md
```

Claude reads the full macro plan and extracts PR6 context (architecture, dependencies, etc.) automatically.

**If still too generic:** Add specific guidance in the invocation:
```bash
/superpowers:writing-plans for PR6 in @docs/plans/pr6-plan.md using @docs/templates/micro-plan.md and @docs/plans/2026-02-11-macro-implementation-plan-v2.md

Pay special attention to:
- Integration with existing SnapshotProvider (see sim/cluster/snapshot.go)
- Round-robin as default routing policy
```

### Issue: Tasks miss behavioral contracts during execution

**Solution:** After execution completes, verify all contracts are tested:
```
"Confirm all contracts are tested:
- BC-1: Show test results
- BC-2: Show test results"
```

### Issue: Lint fails at the end with many issues

**Solution:** Ensure task Step 5 (lint check) runs in each task:
```
Each task Step 5 must run:
golangci-lint run ./path/to/modified/package/...
```

### Issue: Dead code introduced (unused functions, fields)

**Solution:** In Step 3 plan review, check:
- Every struct field used by end of task or later task?
- Every method called by tests or production code?
- Every parameter actually needed?

**Better Solution:** Run `review-pr` at Step 2.5 to catch dead code in plan:
```
/pr-review-toolkit:review-pr code
# code-reviewer agent catches unused abstractions in task code examples
```

### Issue: Review finds many critical issues, overwhelming to fix

**Solution:** Fix issues in priority order:
1. **First pass**: Fix all critical issues, re-run review
2. **Second pass**: Fix important issues, re-run review
3. **Third pass**: Consider suggestions
4. **Use targeted review**: After fixes, only re-run affected aspects
   ```
   # Example: After fixing error handling
   /pr-review-toolkit:review-pr errors
   ```

### Issue: Uncertain if review findings are valid

**Solution:** Review agents provide file:line references:
1. Check the specific code location mentioned
2. Understand the context (sometimes agents miss context)
3. If uncertain, ask Claude to explain the finding
4. If agent is wrong, document why and proceed
5. Consider adding a comment in code explaining why the pattern is intentional

---

## Appendix: Workflow Evolution

**v1.0 (pre-2026-02-14):** Manual agent team prompts, separate design/execution plans
**v2.0 (2026-02-14):** Unified planning with `writing-plans` skill, batch execution with `executing-plans` skill, automated two-stage review with `pr-review-toolkit:review-pr`, simplified invocations with @ file references
**v2.1 (2026-02-16):** Same-session worktree workflow (project-local `.worktrees/` no longer requires new session); continuous execution replaces batch checkpoints (tasks run without pausing, stop only on failure)
**v2.2 (2026-02-16):** Focused review passes replace generic review-pr invocations. Step 2.5 expanded to 3 passes (cross-doc consistency, architecture boundary, codebase readiness). Step 4.5 expanded to 4 passes (code quality, test behavioral quality, getting-started experience, automated reviewer simulation). Based on PR8 experience where each focused pass caught issues the others missed.
**v2.3 (2026-02-16):** Step 2.5 expanded to 4 passes — added Pass 4 (structural validation: task dependencies, template completeness, executive summary clarity, under-specified task detection). Based on PR9 experience where deferred items fell through cracks in the macro plan, and an under-specified documentation task would have confused the executing agent.
**v2.4 (2026-02-16):** Four targeted skill integrations addressing real failure modes: (1) `review-plan` as Pass 0 in Step 2.5 — external LLM review catches design bugs that self-review misses (PR9: fitness normalization bug passed 3 focused passes). (2) `superpowers:systematic-debugging` as on-failure handler in Step 4 — structured root-cause analysis instead of ad-hoc debugging. (3) `superpowers:verification-before-completion` replaces manual verification prose after Step 4.5 — makes build/test/lint gate non-skippable. (4) `commit-commands:clean_gone` as pre-cleanup in Step 1 — prevents stale branch accumulation.
**v2.5 (2026-02-16):** Three additions from `/insights` analysis of 212 sessions: (1) Step 4.75 (pre-commit self-audit) — deliberate critical thinking step with no agent, checking logic/design/determinism/consistency/docs/edge-cases. In PR9, this step found 3 real bugs (wrong reference scale, non-deterministic output, inconsistent comments) that 4 automated passes missed. (2) Headless mode documentation for review passes — workaround for context overflow during multi-agent consolidation, the #1 recurring friction point across 212 sessions. (3) Checkpointing tip for long sessions — prevents progress loss when hitting context limits mid-PR.
**v2.6 (2026-02-18):** Two additions: (1) "Filing Pre-Existing Issues" subsection to Step 4.5 — file a GitHub issue immediately for pre-existing bugs found during review, do not fix in-PR. Based on #38 experience where #183 was discovered. (2) Antipattern prevention from hardening audit of 20+ issues — Step 4.75 expanded to 9 self-audit dimensions (added test epistemology, construction site uniqueness, error path completeness); Step 4.5 Pass 1 prompt expanded with 4 antipattern checks; Step 2.5 Pass 2 prompt expanded with 3 modularity checks. Companion change: `prmicroplanprompt-v2.md` updated with construction site audit (Phase 0), extension friction assessment (Phase 2), invariant test requirement (Phase 6), and 6 new sanity checklist items (Phase 8).
**v2.7 (2026-02-18):** Generalized workflow from "macro plan only" to any source document (macro plan sections, design docs, GitHub issues, feature requests). Updated template references, review prompts, examples, and invocation patterns. Added Example B showing issue/design-doc workflow alongside macro plan workflow. The same rigor (behavioral contracts, TDD tasks, 5+4 review passes, self-audit) applies regardless of work source.
**v2.8 (2026-02-18):** Auto-close issues on PR merge. Added `**Closes:**` field to micro plan header template (`prmicroplanprompt-v2.md`) that captures GitHub closing keywords (e.g., `Fixes #183, fixes #189`). Updated Step 5 PR description spec to propagate closing keywords into the PR body. GitHub auto-closes referenced issues when the PR merges — no manual cleanup needed.
**v2.9 (2026-02-20):** Convergence re-run protocol for both Step 2.5 and Step 4.5. After all review passes complete with fixes, re-run all passes from scratch to verify fixes didn't introduce cross-pass issues. Repeat until convergence (0 CRITICAL, 0 IMPORTANT). Evidence: Wave 1 parallel PR session — Track B's full re-run validated that 3 fixes (including 2 CRITICAL: overwrite-existing-test-file and incomplete zero-value coverage) introduced no regressions across all 5 passes.

**Key improvements in v2.0:**
- **Simplified invocations:** No copy-pasting! Use @ file references (e.g., `@docs/plans/<macro-plan>.md`)
- **Single planning stage:** Produces both design contracts and executable tasks
- **Automated plan review:** Catches design issues before implementation (Step 2.5)
- **Automated code review:** Catches implementation issues before PR creation (Step 4.5)
- **Built-in checkpoint reviews:** During execution (Step 4) — *replaced by continuous execution in v2.1*
- **Reduced manual overhead:** Skills handle context extraction automatically

**Example workflow brevity:**
- **v1.0:** ~200 words of manual prompts per PR
- **v2.0:** 5 simple commands with @ references

---

**For questions or workflow improvements, discuss with Claude using this document as context.**
