---
name: design-and-implement-pr
description: Use when designing and implementing an element (PR, subsection, task, objective) of a "macro" plan.
---

# Design and Implement PR

## Overview

Load plan, review critically, execute tasks in batches, report for review between batches.

**Core principle:** Batch execution with checkpoints for architect review.

**Announce at start:** "I'm using the implement-pr skill to implement this plan."

## Input

The user must provide:
- **PR identifier**: The PR name from the macro plan (e.g., `PR1`, `PR5`, `PR12`)
- **Macro plan path**: Path to the plan file (e.g., `docs/plans/2026-02-11-macro-implementation-plan-v2.md`)

## The Process

<!-- 
Assume that a detailed code summary has been generated, 
for example in PROJECT_DETAILS.md and that this has been 
referenced by CLAUDE.md (already loaded in the context).
-->

### Step 1: Load and Review Overall Plan
1. Read and understand the macro plan
2. Review the macro plan critically
    - Is it well-formed?
    - Does it conform to the guidance in `docs/plans/macroplanprompt.md`
3. Identify any questions or concerns about the work plan paying attention to the current implementation
4. If concerns: Raise them with your human partner before starting
5. If no concerns: continue

### Step 2: Create PR Plan
1. Create a detailed implementation plan for the PR that:
    - adheres to `docs/plans/prmicroplanprompt.md`
    - is grounded in the existing codebase
2. Write the PR plan in `docs/plans/<PR>-plan.md`.

### Step 3: Review the PR Plan
1. Thoroughly review the plan to verify that:
    - it is in conformance with `docs/plans/prmicroplanprompt.md`
    - it guarantees the intended behaviors, API contracts described in the macro plan
    - it does not introduce regressions, bugs, dead code, or duplicate code
    - it's tests are truly behavioral
    - proposed changes are minimal

### Step 4: Update the PR Plan
1. Consider the feedback from Step 3's critical review
2. Verify against the PR plan and the macro plan
3. Evaluate the validity of the self-critique
4. If valid: modify the PR plan to incorporate feedback, return to Step 3
5. If not valid: continue

**CHECKPOINT:** Present the PR plan to the user for approval before proceeding.

### Step 5: Execute the Plan
1. Use superpowers:executing-plans to implement the PR plan

### Step 6: Review Changes (Local, Pre-PR)
1. Use superpowers:requesting-code-review to request a thorough code review that, in addition to built in function:
    - validates tests execute correctly
    - verifies that tests are truly behavioral
    - runs lint
2. Use superpowers:receiving-code-review to consider the result of the review.

**Purpose:** Catch issues locally before creating the PR.

### Step 7: Update Documentation
1. Identify and make any needed changes to `CLAUDE.md`
2. Identify and make any needed changes to `README.md`

**CHECKPOINT:** Present review results to the user for approval before creating PR.

### Step 8: Create Pull Request
1. Create a pull request for the changes
2. The pull request title
    - SHOULD be succinct and clear
    - SHOULD summarize the changes
3. The pull request description
    - SHOULD be succinct and clear
    - SHOULD describe the behavioral or functional changes

### Step 9: Review Pull Request (Post-CI)
1. Use pr-review-toolkit:review-pr to request a thorough review of the pull request:
    - validates that all GitHub Actions executed correctly
    - verifies that tests are truly behavioral
2. Use superpowers:receiving-code-review to consider the feedback from the review.

**Purpose:** Verify CI passed and catch issues visible only in the PR context (diff view, CI logs).

### Step 10: Recommend Plan Changes
1. Review the work plan critically in light of the code changes made
2. Identify any new questions or concerns about the work plan
3. Identify any recommended changes
4. If concerns or recommendations: Raise them with your human partner and update the work plan
5. If no concerns or recommendations: Notify user that the PR is ready for merge
6. If work plan was updated: Notify user of the changes made

## When to Stop and Ask for Help

**STOP executing immediately when:**
- Hit a blocker mid-batch (missing dependency, test fails, instruction unclear)
- Plan has critical gaps preventing starting
- You don't understand an instruction
- Verification fails repeatedly

**Ask for clarification rather than guessing.**

## When to Revisit Earlier Steps

**Return to Step 1 (Load and Review Overall Plan) when:**
- Partner updates the plan based on your feedback
- Fundamental approach needs rethinking

**Don't force through blockers** - stop and ask.

## Remember
- Review plan critically first
- Follow plan steps exactly
- Don't skip verifications and reviews
- Reference skills when plan says to
- Between batches: just report and wait
- Stop when blocked, don't guess
- Never start implementation on main/master branch without explicit user consent

