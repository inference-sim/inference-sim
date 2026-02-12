---
name: implement-pr
description: Use when implementing a PR from a macro plan
---

# Implement PR

## Overview

Load plan, review critically, execute tasks in batches, report for review between batches.

**Core principle:** Batch execution with checkpoints for architect review.

**Announce at start:** "I'm using the implement-pr skill to implement this plan."

## Input

The user must provide:
- **PR identifier**: The PR name from the macro plan (e.g., `PR1`, `PR5`, `PR12`)
- **Macro plan path**: Path to the work plan file (e.g., `docs/plans/2026-02-11-macro-implementation-plan-v2.md`)

## The Process

### Step 1: Understand the Current Implementation
1. Review the code relevant to this PR's scope

### Step 2: Load and Review Overall Plan
1. Read work plan file
2. Review work plan critically
    - Is it well-formed?
    - Does it conform to the guidance in `docs/plans/macroplanprompt.md`
3. Identify any questions or concerns about the work plan paying attention to the current implementation
4. If concerns: Raise them with your human partner before starting

### Step 3: Create PR Plan
1. Create a detailed implementation plan for the PR that adheres to `docs/plans/prmicroplanprompt.md`
2. Write the PR plan in `docs/plans/<PR>-plan.md`.

### Step 4: Review the PR Plan
1. Critically review the plan to verify that it:
    - is in conformance with `docs/plans/prmicroplanprompt.md`
    - implements the intended behaviors, API contracts and functions described in the work plan for this PR

### Step 5: Update the PR Plan
1. Consider the feedback
2. Verify against the PR plan
3. Evaluate validity of feedback
4. If feedback is valid: modify the PR plan to incorporate valid feedback, return to Step 4
5. If feedback is not valid: continue

### Step 6: Execute the Plan
1. Use superpowers:executing-plans to implement the PR plan

### Step 7: Review Changes (Local, Pre-PR)
1. Use superpowers:requesting-code-review to request a thorough code review that, in addition to built in function:
    - validates tests execute correctly
    - verifies that tests are truly behavioral
    - runs lint
2. Use superpowers:receiving-code-review to consider the result of the review.

**Purpose:** Catch issues locally before creating the PR.

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

### Step 10: Update Documentation
1. Identify and make any needed changes to `CLAUDE.md`
2. Identify and make any needed changes to `README.md`
3. Push changes to the pull request.

### Step 11: Recommend Plan Changes
1. Review the work plan critically in light of the code changes made
2. Identify any new questions or concerns about the work plan
3. Identify any recommended changes
4. If concerns or recommendations: Raise them with your human partner
5. If no concerns or recommendations: Stop

## When to Stop and Ask for Help

**STOP executing immediately when:**
- Hit a blocker mid-batch (missing dependency, test fails, instruction unclear)
- Plan has critical gaps preventing starting
- You don't understand an instruction
- Verification fails repeatedly

**Ask for clarification rather than guessing.**

## When to Revisit Earlier Steps

**Return to Step 2 (Load and Review Overall Plan) when:**
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

