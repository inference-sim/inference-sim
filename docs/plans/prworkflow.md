# PR Development Workflow

**Status:** Active (v2.2 - updated 2026-02-16)

This document describes the complete workflow for implementing a PR from the macro plan.

## Current Template Versions

**Update this section when templates change. All examples below reference these versions.**

- **Micro-planning template:** `docs/plans/prmicroplanprompt-v2.md` (updated 2026-02-16)
- **Macro-planning template:** `docs/plans/2026-02-11-macro-implementation-plan-v2.md` (v2.3)
- **Deprecated micro-planning:** `docs/plans/prmicroplanprompt-v1-deprecated.md` (for reference only)

---

## Prerequisites

This workflow requires the following Claude Code skills to be available:

| Skill | Purpose | Used In |
|-------|---------|---------|
| `superpowers:using-git-worktrees` | Create isolated workspace for PR work | Step 1 |
| `superpowers:writing-plans` | Generate implementation plan from templates | Step 2 |
| `pr-review-toolkit:review-pr` | Automated multi-agent review | Step 2.5, Step 4.5 |
| `superpowers:executing-plans` | Execute plan tasks continuously | Step 4 |
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
- Step 2: Follow `prmicroplanprompt-v2.md` template manually
- Step 2.5/4.5: Manual code review or skip automated review
- Step 4: Implement tasks manually following plan
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
│ Step 2.5: plan review   │ (3 focused review passes — see checklist)
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
│ Step 5: commit-push-pr  │ (Commit, push, create PR - all in one)
└─────────────────────────┘
```

**Key insights:**
1. **Worktree isolation from start** (Step 1) - Create worktree BEFORE any work
   - Entire PR lifecycle (planning + implementation) in isolated workspace
   - Main worktree never touched
   - Enables parallel work on multiple PRs

2. **Two-stage automated review** with `pr-review-toolkit:review-pr`:
   - **Plan Review** (Step 2.5) - Reviews plan markdown file (uncommitted in worktree)
     - Validates behavioral contracts, task breakdown, test strategy
     - Catches design issues before implementation
   - **Code Review** (Step 4.5) - Reviews implementation changes (git diff in worktree)
     - Validates code quality, tests, error handling, types
     - Catches implementation issues before PR creation

---

## Quick Reference: Simplified Invocations

**No copy-pasting required!** Use @ file references and simple commands:

| Step | Command |
|------|---------|
| **1. Create worktree** | `/superpowers:using-git-worktrees pr<N>-<name>` |
| **2. Create plan** | `/superpowers:writing-plans for PR<N> in @docs/plans/pr<N>-<name>-plan.md using @docs/plans/prmicroplanprompt-v2.md and @docs/plans/2026-02-11-macro-implementation-plan-v2.md` |
| **2.5. Review plan** | 3 focused passes: cross-doc consistency, architecture boundary, codebase readiness |
| **3. Human review plan** | Review contracts, tasks, appendix, then approve to proceed |
| **4. Execute plan** | `/superpowers:executing-plans @docs/plans/pr<N>-<name>-plan.md` |
| **4.5. Review code** | 4 focused passes: code quality, test behavioral quality, getting-started, automated reviewer sim |
| **5. Commit, push, PR** | `/commit-commands:commit-push-pr` |

**Example for PR 8 (same-session workflow with `.worktrees/`):**
```bash
# Step 1: Create worktree (stays in same session)
/superpowers:using-git-worktrees pr8-routing-state-and-policy-bundle

# Output: Worktree ready at .worktrees/pr8-routing-state-and-policy-bundle/
# (shell cwd already switched — continue directly)

# Step 2: Create plan
/superpowers:writing-plans for PR8 in @docs/plans/pr8-routing-state-and-policy-bundle-plan.md using @docs/plans/prmicroplanprompt-v2.md and @docs/plans/2026-02-11-macro-implementation-plan-v2.md

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
/superpowers:writing-plans for PR<N> in @docs/plans/pr<N>-<feature-name>-plan.md using @docs/plans/prmicroplanprompt-v2.md and @docs/plans/2026-02-11-macro-implementation-plan-v2.md
```

**Example:**
```
/superpowers:writing-plans for PR6 in @docs/plans/pr6-routing-policy-plan.md using @docs/plans/prmicroplanprompt-v2.md and @docs/plans/2026-02-11-macro-implementation-plan-v2.md
```

**What Happens:**
- Claude reads the macro plan and locates PR<N> section
- Claude reads prmicroplanprompt-v2.md as the template
- Claude inspects the codebase (Phase 0: Component Context)
- Claude creates behavioral contracts (Phase 1)
- Claude breaks implementation into 6-12 TDD tasks (Phase 4)
- Claude saves plan to the specified output file **in the worktree**

**Output:**
- Plan file at `docs/plans/pr<N>-<feature-name>-plan.md` (in worktree, on feature branch)
- Contains: behavioral contracts, executable tasks, test strategy, appendix

**Tips:**
- Use @ file references instead of copy-pasting
- Claude automatically extracts the relevant PR section from macro plan
- Template structure is preserved automatically

---

### Step 2.5: Focused Plan Review (3 Passes)

**Context:** Worktree (same or new session)

> **For Claude:** When the user asks you to execute Step 2.5, run all 3 review passes below
> sequentially. For each pass: invoke `/pr-review-toolkit:review-pr` with the **exact prompt
> text** shown in the `Prompt:` field (substituting `<N>` and `<name>` with the actual PR
> number and plan filename). After each pass, summarize findings and fix all critical/important
> issues before starting the next pass. After all 3 passes complete, report a summary to the
> user and wait for approval to proceed.

**Why focused passes?** Generic "review everything" misses issues that targeted reviews catch. PR8 experience: Pass 1 found 6 categories of stale macro plan content. Pass 2 found priority overwrite semantics. Pass 3 found stale comments in files to be modified. None of these were found by generic review.

---

#### Pass 1: Cross-Document Consistency

Verify the micro plan matches the macro plan and both match the codebase.

**Prompt:**
```
/pr-review-toolkit:review-pr Does the micro plan's scope match the macro plan's PR section? Are file paths consistent? Does the deviation log account for all differences between what the macro plan says and what the micro plan does? Check for stale references in the macro plan. @docs/plans/pr<N>-<name>-plan.md and @docs/plans/2026-02-11-macro-implementation-plan-v2.md
```

**Catches:** Stale macro plan references, scope mismatch, missing deviations, wrong file paths.

**Fix all critical/important issues before Pass 2.**

---

#### Pass 2: Architecture Boundary Verification

Verify the plan respects architectural boundaries and separation of concerns.

**Prompt:**
```
/pr-review-toolkit:review-pr Does this plan maintain architectural boundaries? Are we ensuring individual instances don't have access to cluster-level state? Are types in the right packages? Check the plan against the actual code for boundary violations. @docs/plans/pr<N>-<name>-plan.md
```

**Catches:** Import cycle risks, boundary violations, missing bridge types, wrong abstraction level.

**Fix all critical/important issues before Pass 3.**

---

#### Pass 3: Codebase Readiness

Verify the files to be modified are clean and ready for the planned changes.

**Prompt:**
```
/pr-review-toolkit:review-pr We're about to implement this PR. Review the codebase for readiness. Check each file the plan will modify for stale comments, existing bugs, or issues that would complicate implementation. @docs/plans/pr<N>-<name>-plan.md
```

**Catches:** Stale comments ("planned for PR N"), pre-existing bugs, missing dependencies, unclear insertion points.

**Fix all critical/important issues. Then report summary to user.**

---

### Step 3: Human Review of Plan

**Context:** Worktree (same or new session)

**Action:** Final human review of the plan (after automated review)

**Focus Areas:**
1. **Part 1 (Design Validation)** - Review behavioral contracts, component interaction, risks
2. **Part 2 (Executable Tasks)** - Verify task breakdown makes sense, no dead code
3. **Deviation Log** - Check if deviations from macro plan are justified
4. **Appendix** - Spot-check file-level details for accuracy

**Common Issues to Catch:**
- Behavioral contracts too vague or missing edge cases
- Tasks not properly ordered (dependencies)
- Missing test coverage for contracts
- Deviations from macro plan not justified
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
/pr-review-toolkit:review-pr
```

**Catches:** Logic errors, nil pointer risks, silent failures (discarded return values), panic paths reachable from user input, CLAUDE.md convention violations, dead code.

**Fix all critical/important issues before Pass 2.**

---

#### Pass 2: Test Behavioral Quality

Verify tests are truly behavioral (testing WHAT) not structural (testing HOW).

**Prompt:**
```
/pr-review-toolkit:review-pr Are all the tests well written and truly behavioral? Do they test observable behavior (GIVEN/WHEN/THEN) or just assert internal structure? Would they survive a refactor? Rate each test as Behavioral, Mixed, or Structural.
```

**Catches:** Structural tests (Go struct assignment, trivial getters), type assertions in factory tests, exact-formula assertions instead of behavioral invariants, tests that pass even if the feature is broken.

**Fix: Delete structural tests, replace type assertions with behavior assertions.**

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

#### After All 4 Passes

> **For Claude:** After fixing issues from all passes, run verification before reporting:
> ```bash
> go build ./... && go test ./... -count=1 && golangci-lint run ./...
> ```
> Report: build exit code, test pass/fail counts, lint issue count, working tree status.
> Wait for user approval before proceeding to Step 5.

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
- PR title from macro plan
- Multi-line description of changes
- List of implemented behavioral contracts (BC-1, BC-2, etc.)
- Co-authored-by line

**PR description includes:**
- Summary from macro plan
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
| `using-git-worktrees` | **Step 1** - Create isolated workspace FIRST | Branch name | Worktree directory path |
| `writing-plans` | **Step 2** - Create implementation plan from macro plan | Macro plan PR section + prmicroplanprompt-v2.md | Plan file with contracts + tasks |
| `pr-review-toolkit:review-pr` | **Step 2.5** - 3 focused plan review passes | Targeted prompts (see checklist) | Critical/important issues per pass |
| `pr-review-toolkit:review-pr` | **Step 4.5** - 4 focused code review passes | Targeted prompts (see checklist) | Critical/important issues per pass |
| `executing-plans` | **Step 4** - Execute plan tasks continuously | Plan file path | Implemented code + commits |
| `subagent-driven-development` | **Step 4 (alt)** - Execute plan in-session | Plan file path | Implemented code + commits |
| `commit-commands:commit-push-pr` | **Step 5** - Commit, push, create PR (all in one) | Current branch state | Commit + push + PR URL |

---

## Example: Complete PR Workflow (Same-Session with `.worktrees/`)

```bash
# Step 1: Create worktree (shell cwd switches automatically)
/superpowers:using-git-worktrees pr8-routing-state-and-policy-bundle

# Output: Worktree ready at .worktrees/pr8-routing-state-and-policy-bundle/
# (continue directly — no new session needed)

# Step 2: Create plan (one simple command with @ references)
/superpowers:writing-plans for PR8 in @docs/plans/pr8-routing-state-and-policy-bundle-plan.md using @docs/plans/prmicroplanprompt-v2.md and @docs/plans/2026-02-11-macro-implementation-plan-v2.md

# Output: Plan created at docs/plans/pr8-routing-state-and-policy-bundle-plan.md

# Step 2.5: Focused plan review (3 passes)
# Pass 1: Cross-doc consistency (micro plan vs macro plan)
/pr-review-toolkit:review-pr Does the micro plan match the macro plan? @docs/plans/pr8-plan.md and @docs/plans/macro-plan.md
# Pass 2: Architecture boundary verification
/pr-review-toolkit:review-pr Does this plan maintain architectural boundaries?
# Pass 3: Codebase readiness scan
/pr-review-toolkit:review-pr Review the codebase for readiness to implement this PR.
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

## Tips for Success

1. **Use automated reviews proactively** - Run `review-pr` after plan creation and after implementation (don't wait for human review to catch issues)
2. **Fix critical issues immediately** - Don't proceed with known critical issues (they compound)
3. **Re-run targeted reviews after fixes** - Verify fixes worked: `/pr-review-toolkit:review-pr code tests`
4. **Use worktrees for complex PRs** - Avoid disrupting main workspace
5. **Review after execution** - Use automated code review (Step 4.5) after all tasks complete
6. **Reference contracts in commits** - Makes review easier and more traceable
7. **Update CLAUDE.md immediately** - Don't defer documentation
8. **Keep macro plan updated** - Mark PRs as completed

### Review Strategy Tips

**Always use focused prompts, not generic invocations.** Each focused review pass catches different issues:

| Pass | What It Catches That Others Miss |
|------|----------------------------------|
| Cross-doc consistency | Stale macro plan references, scope mismatch, wrong file paths |
| Architecture boundary | Import cycles, boundary violations, wrong abstraction level |
| Codebase readiness | Stale comments, pre-existing bugs, missing dependencies |
| Code quality | Logic errors, silent failures, convention violations |
| Test behavioral quality | Structural tests, type assertions, formula-coupled assertions |
| Getting-started experience | Missing examples, undocumented output, contributor friction |
| Automated reviewer sim | Mutable globals, user-controlled panics, YAML typo acceptance |

**Fix issues between passes, not after all passes.** Fixes from Pass 1 may affect what Pass 2 finds.

**Re-run a pass if you made significant changes** during fixing. Don't assume the fix is correct.

---

## Common Issues and Solutions

### Issue: Plan too generic, agents ask clarifying questions

**Solution:** The simplified invocation with @ references handles this automatically:
```bash
/superpowers:writing-plans for PR6 in @docs/plans/pr6-plan.md using @docs/plans/prmicroplanprompt-v2.md and @docs/plans/2026-02-11-macro-implementation-plan-v2.md
```

Claude reads the full macro plan and extracts PR6 context (architecture, dependencies, etc.) automatically.

**If still too generic:** Add specific guidance in the invocation:
```bash
/superpowers:writing-plans for PR6 in @docs/plans/pr6-plan.md using @docs/plans/prmicroplanprompt-v2.md and @docs/plans/2026-02-11-macro-implementation-plan-v2.md

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

**Key improvements in v2.0:**
- **Simplified invocations:** No copy-pasting! Use @ file references (e.g., `@docs/plans/macroplan.md`)
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
