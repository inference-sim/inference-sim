# PR Development Workflow

**Status:** Active (v2.0 - updated 2026-02-14)

This document describes the complete workflow for implementing a PR from the macro plan.

## Overview

```
┌─────────────┐
│ Macro Plan  │ (Select PR to implement)
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│ Step 2: writing-plans   │ (Create behavioral contracts + executable tasks)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Step 2.5: review-pr     │ (Automated plan review)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Step 3: Human Review    │ (Approve plan)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Step 4: using-worktrees │ (Optional: isolate work)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Step 5: executing-plans │ (Implement in batches with checkpoints)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Step 5.5: review-pr     │ (Automated code review)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Step 6: finishing-work  │ (Final verification)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Step 7: Create PR       │ (Push and open GitHub PR)
└─────────────────────────┘
```

**Key insight:** `pr-review-toolkit:review-pr` works at two stages:
1. **Plan Review** (Step 2.5) - Reviews plan markdown file (uncommitted)
   - Validates behavioral contracts, task breakdown, test strategy
   - Catches design issues before implementation
2. **Code Review** (Step 5.5) - Reviews implementation changes (git diff)
   - Validates code quality, tests, error handling, types
   - Catches implementation issues before PR creation

---

## Quick Reference: Simplified Invocations

**No copy-pasting required!** Use @ file references and simple commands:

| Step | Command |
|------|---------|
| **2. Create plan** | `/superpowers:writing-plans for PR<N> in @docs/plans/pr<N>-<name>-plan.md using @docs/plans/prmicroplanprompt-v2.md and @docs/plans/2026-02-11-macro-implementation-plan-v2.md` |
| **2.5. Review plan** | `/pr-review-toolkit:review-pr` |
| **4. Create worktree** | `/superpowers:using-git-worktrees pr<N>-<name>` |
| **5. Execute plan** | `/superpowers:executing-plans @docs/plans/pr<N>-<name>-plan.md` |
| **5.5. Review code** | `/pr-review-toolkit:review-pr` |
| **6. Finish** | `/superpowers:finishing-a-development-branch` |

**Example for PR 6:**
```bash
/superpowers:writing-plans for PR6 in @docs/plans/pr6-routing-policy-plan.md using @docs/plans/prmicroplanprompt-v2.md and @docs/plans/2026-02-11-macro-implementation-plan-v2.md
/pr-review-toolkit:review-pr
/superpowers:using-git-worktrees pr6-routing-policy
/superpowers:executing-plans @docs/plans/pr6-routing-policy-plan.md
/pr-review-toolkit:review-pr
/superpowers:finishing-a-development-branch
```

---

## Step-by-Step Process

### Step 1: Select PR from Macro Plan

**Input:** `docs/plans/2026-02-11-macro-implementation-plan-v2.md`

**Action:** Identify the next PR to implement based on:
- Phase sequencing (Phase 1 → Phase 2 → ...)
- PR dependencies (check dependency graph in macro plan)
- Current completion status (marked in macro plan or CLAUDE.md)

**Output:** PR number and title (e.g., "PR 5: Architectural Simplification")

---

### Step 2: Create Implementation Plan Using `writing-plans` Skill

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
- Claude saves plan to the specified output file

**Output:**
- Plan file at `docs/plans/pr<N>-<feature-name>-plan.md`
- Contains: behavioral contracts, executable tasks, test strategy, appendix

**Tips:**
- Use @ file references instead of copy-pasting
- Claude automatically extracts the relevant PR section from macro plan
- Template structure is preserved automatically

---

### Step 2.5: Automated Plan Review Using `pr-review-toolkit:review-pr`

**Skill:** `pr-review-toolkit:review-pr`

**Invocation (simplified):**
```
/pr-review-toolkit:review-pr
```

**That's it!** The skill automatically:
- Detects the newly created plan file
- Determines which agents to run based on content
- Reviews behavioral contracts, task breakdown, code examples

**What Happens:**
- Skill detects changed files (newly created `docs/plans/pr<N>-plan.md`)
- Intelligently selects applicable review agents based on content:
  - **code-reviewer**: Reviews plan structure, task clarity, code examples
  - **pr-test-analyzer**: Reviews test strategy and coverage mapping
  - **comment-analyzer**: Reviews documentation and explanations (if applicable)
- Agents run and return detailed reports

**Review Focus:**
- Behavioral contracts: completeness, clarity, testability
- Task breakdown: proper ordering, no dead code, dependencies correct
- Code examples: syntactically correct, complete (no placeholders)
- Test strategy: contracts mapped to tests, coverage adequate
- Architecture: consistent with macro plan, deviations justified

**Output:**
- Structured review with:
  - **Critical Issues**: Must fix before implementation
  - **Important Issues**: Should fix
  - **Suggestions**: Nice to have
  - **Strengths**: What's well-done

**Action After Review:**
- If critical/important issues found: Fix plan with Claude, re-run review
- If only suggestions: Human judgment on whether to incorporate
- If clean: Proceed to Step 3

**Alternative:** You can specify which aspects to review:
```
/pr-review-toolkit:review-pr code tests
# Reviews only code quality and test strategy in the plan
```

---

### Step 3: Human Review of the Plan

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
- ✅ Approve plan → proceed to Step 4
- ❌ Request revisions → iterate with Claude on specific sections

---

### Step 4: Set Up Isolated Worktree (Recommended)

**Skill:** `superpowers:using-git-worktrees`

**Invocation (simplified):**
```
/superpowers:using-git-worktrees pr<N>-<feature-name>
```

**Example:**
```
/superpowers:using-git-worktrees pr6-routing-policy
```

**What Happens:**
- Creates a new git worktree in a separate directory
- Creates and checks out a new branch
- Isolates work from main development

**Output:** Path to worktree directory (e.g., `/Users/sri/Documents/Projects/inference-sim-pr5/`)

**Note:** This is optional but recommended for complex PRs. For simple PRs, you can work directly in the main worktree.

---

### Step 5: Execute Plan Using `executing-plans` Skill

**Skill:** `superpowers:executing-plans`

**Context:** Open a new Claude Code session in the worktree directory (or current directory if no worktree)

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
- Executes tasks in batches
- Shows checkpoints with test/lint results

**What Happens:**
- Claude reads the plan file
- Claude executes Task 1-3 (first batch)
  - Each task: write test → verify fail → implement → verify pass → lint → commit
- Claude reports batch completion and waits for feedback
- Repeat for remaining batches

**Checkpoints:**
After each batch, Claude shows:
- Summary of implemented contracts
- Test output (all passing)
- Lint output (no new issues)
- Commit log

**Your Response Options:**
- "Continue" → proceed to next batch
- "Fix [issue]" → Claude addresses issue and re-verifies
- "Explain [decision]" → Claude clarifies implementation choice

**Output:**
- Implemented code in working directory
- All tests passing (`go test ./...`)
- Lint clean (`golangci-lint run ./...`)
- Commits for each task (referencing contracts)

---

### Step 5.5: Automated Code Review Using `pr-review-toolkit:review-pr`

**Skill:** `pr-review-toolkit:review-pr`

**Invocation (simplified):**
```
/pr-review-toolkit:review-pr
```

**That's it!** The skill automatically:
- Detects all changed files via git diff
- Selects applicable review agents
- Runs comprehensive review

**What Happens:**
- Skill detects all changed files via `git diff`
- Intelligently selects applicable review agents:
  - **code-reviewer**: General code quality, CLAUDE.md compliance, bugs
  - **pr-test-analyzer**: Test coverage and quality (if test files changed)
  - **silent-failure-hunter**: Error handling review (if error paths changed)
  - **type-design-analyzer**: Type design quality (if new types added)
  - **comment-analyzer**: Comment accuracy (if comments/docs added)
  - **code-simplifier**: Simplification opportunities (after other reviews pass)
- Agents run in sequence and return detailed reports

**Review Focus:**
- **Behavioral contracts**: All contracts from plan implemented and tested?
- **Code quality**: Bugs, style violations, CLAUDE.md compliance
- **Test coverage**: Critical gaps, missing edge cases
- **Error handling**: Silent failures, inadequate logging
- **Type design**: Encapsulation, invariants (if new types)
- **Comments**: Accuracy vs code, documentation completeness
- **Simplicity**: Can code be simplified while preserving functionality?

**Output:**
- Structured review with:
  - **Critical Issues**: Must fix before PR creation
  - **Important Issues**: Should fix
  - **Suggestions**: Nice to have
  - **Strengths**: What's well-done

**Action After Review:**
1. **Fix critical issues first**: Address must-fix items
2. **Address important issues**: Fix should-fix items
3. **Consider suggestions**: Human judgment on nice-to-haves
4. **Re-run review**: Verify fixes resolved issues
   ```
   /pr-review-toolkit:review-pr code errors tests
   # Re-run specific aspects after fixes
   ```
5. **Proceed to Step 6** when all critical/important issues resolved

**Parallel Review Option:**
```
/pr-review-toolkit:review-pr all parallel
# Launches all agents simultaneously for faster review
```

**Targeted Review Option:**
```
/pr-review-toolkit:review-pr errors tests
# Reviews only error handling and test coverage
```

---

### Step 6: Final Verification Using `finishing-a-development-branch` Skill

**Skill:** `superpowers:finishing-a-development-branch`

**Invocation (simplified):**
```
/superpowers:finishing-a-development-branch
```

**That's it!** The skill automatically:
- Runs final verification (tests, lint)
- Presents completion options (PR, merge, etc.)

**What Happens:**
- Claude runs final verification (`go test ./...`, `golangci-lint run ./...`)
- Claude presents options:
  1. Create PR
  2. Merge to main (if allowed by your workflow)
  3. Additional fixes needed

**Output:**
- Verification results
- Structured completion options

---

### Step 7: Create PR and Push

**Action:** Create PR using GitHub CLI or web interface

**Commit message format:**
```
PR <N>: <title> (#<PR-number>)

<Multi-line description from macro plan>

Implements:
- BC-1: <contract name>
- BC-2: <contract name>
...

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

**PR description template:**
```markdown
## Summary
[From macro plan PR section]

## Behavioral Contracts
- BC-1: <GIVEN X WHEN Y THEN Z>
- BC-2: ...

## Testing
- All tests pass: `go test ./...`
- Lint clean: `golangci-lint run ./...`
- CLI exercisable: [example command]

## Checklist
- [ ] All contracts implemented and tested
- [ ] No dead code
- [ ] CLAUDE.md updated (if needed)
- [ ] Golden dataset updated (if needed)
- [ ] No deviations from macro plan (or documented)
```

**Commands:**
```bash
# Push branch
git push -u origin pr<N>-<feature-name>

# Create PR via gh CLI
gh pr create --title "PR <N>: <title>" --body-file pr-description.md

# Or via web interface
# Visit GitHub and create PR from pushed branch
```

---

## Workflow Variants

### Option A: Subagent-Driven Development (In-Session)

**Alternative to Step 5** - Use for simpler PRs where you want tighter iteration:

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
- ⚠️ No built-in checkpoint batching (review after every task)

---

## Skill Reference Quick Guide

| Skill | When to Use | Input | Output |
|-------|-------------|-------|--------|
| `writing-plans` | **Step 2** - Create implementation plan from macro plan | Macro plan PR section + prmicroplanprompt-v2.md | Plan file with contracts + tasks |
| `pr-review-toolkit:review-pr` | **Step 2.5** - Review plan document | Plan file (uncommitted) | Critical/important issues + suggestions |
| `pr-review-toolkit:review-pr` | **Step 5.5** - Review implementation | Code changes (git diff) | Critical/important issues + suggestions |
| `using-git-worktrees` | **Step 4** - Isolate work (optional) | Branch name | Worktree directory path |
| `executing-plans` | **Step 5** - Execute plan in batches | Plan file path | Implemented code + commits |
| `subagent-driven-development` | **Step 5 (alt)** - Execute plan in-session | Plan file path | Implemented code + commits |
| `finishing-a-development-branch` | **Step 6** - Final verification | Current branch | Verification + completion options |

---

## Example: Complete PR 6 Workflow (Simplified Invocations)

```bash
# Step 1: Select PR
"I want to implement PR 6: RoutingPolicy Interface"

# Step 2: Create plan (one simple command with @ references)
/superpowers:writing-plans for PR6 in @docs/plans/pr6-routing-policy-plan.md using @docs/plans/prmicroplanprompt-v2.md and @docs/plans/2026-02-11-macro-implementation-plan-v2.md

# Output: Plan created at docs/plans/pr6-routing-policy-plan.md

# Step 2.5: Automated plan review (no arguments needed)
/pr-review-toolkit:review-pr

# Output: Review report with any issues
# [Fix critical/important issues if found]
# [Re-run: /pr-review-toolkit:review-pr]

# Step 3: Human review of plan
# [Read plan, verify contracts and tasks, approve]

# Step 4: Create worktree (simple branch name)
/superpowers:using-git-worktrees pr6-routing-policy

# Output: Worktree created at ../inference-sim-pr6/

# Step 5: Execute plan (in new session in worktree)
# [Open new terminal in worktree directory]
# [Start new Claude Code session]

/superpowers:executing-plans @docs/plans/pr6-routing-policy-plan.md

# Output: Batch 1 → checkpoint → Batch 2 → checkpoint → Batch 3 → done

# Step 5.5: Automated code review (no arguments needed)
/pr-review-toolkit:review-pr

# Output: Comprehensive review report
# [Fix critical/important issues]
# [Re-run targeted: /pr-review-toolkit:review-pr code tests]

# Step 6: Finish (no arguments needed)
/superpowers:finishing-a-development-branch

# Output: Verification results + completion options

# Step 7: Create PR
"Create PR"
# [Claude uses gh pr create or guides manual PR creation]
```

**Key benefit:** No copy-pasting! Just use @ file references and let Claude extract the context.

---

## Tips for Success

1. **Use automated reviews proactively** - Run `review-pr` after plan creation and after implementation (don't wait for human review to catch issues)
2. **Fix critical issues immediately** - Don't proceed with known critical issues (they compound)
3. **Re-run targeted reviews after fixes** - Verify fixes worked: `/pr-review-toolkit:review-pr code tests`
4. **Use worktrees for complex PRs** - Avoid disrupting main workspace
5. **Don't skip checkpoints** - Review after each batch, fix issues before continuing
6. **Reference contracts in commits** - Makes review easier and more traceable
7. **Update CLAUDE.md immediately** - Don't defer documentation
8. **Keep macro plan updated** - Mark PRs as completed

### Review Strategy Tips

**When to use parallel review:**
```
/pr-review-toolkit:review-pr all parallel
```
- Good for: Large PRs, comprehensive review before PR creation
- Trade-off: Faster but may get overwhelming number of issues at once

**When to use sequential review:**
```
/pr-review-toolkit:review-pr
# (default, runs agents one at a time)
```
- Good for: Incremental fixes, focusing on one aspect at a time
- Trade-off: Slower but more manageable

**When to use targeted review:**
```
/pr-review-toolkit:review-pr errors tests
# Only review error handling and test coverage
```
- Good for: After fixing specific issues, focused verification
- Good for: When you know which aspects need attention

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

**Solution:** In Step 5 checkpoint review, verify:
```
"Before continuing, confirm all contracts from Batch 1 are tested:
- BC-1: Show test results
- BC-2: Show test results"
```

### Issue: Lint fails at the end with many issues

**Solution:** Ensure Step 6 of each task includes lint check:
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

**Key improvements in v2.0:**
- **Simplified invocations:** No copy-pasting! Use @ file references (e.g., `@docs/plans/macroplan.md`)
- **Single planning stage:** Produces both design contracts and executable tasks
- **Automated plan review:** Catches design issues before implementation (Step 2.5)
- **Automated code review:** Catches implementation issues before PR creation (Step 5.5)
- **Built-in checkpoint reviews:** During execution (Step 5)
- **Reduced manual overhead:** Skills handle context extraction automatically

**Example workflow brevity:**
- **v1.0:** ~200 words of manual prompts per PR
- **v2.0:** 6 simple commands with @ references

---

**For questions or workflow improvements, discuss with Claude using this document as context.**
