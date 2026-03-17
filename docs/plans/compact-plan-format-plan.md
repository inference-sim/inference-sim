# Compressed Micro-Plan Format for Small Tier PRs — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a compact plan format so Small tier PRs don't require the full ~150-line template with sections irrelevant to trivial changes.

**The problem today:** Even Small tier PRs (≤3 files, mechanical changes) produce ~150-line plans with executive summary, component interaction diagram, deviation log, review guide, implementation overview, and test strategy table. For a 5-line bug fix, the plan is often 30x longer than the code. This overhead discourages small contributions and wastes review time.

**What this PR adds:**
1. Compact plan format in the micro-plan template — a streamlined format that keeps behavioral contracts and TDD tasks but drops ceremony sections (executive summary, component diagram, deviation log, review guide, implementation overview, test strategy table)
2. Updated PR Size Tiers table — references the compact format for Small tier
3. Updated agent prompt — teaches the writing-plans agent when and how to produce compact plans

**Why this matters:** Reducing process friction for simple changes encourages more frequent, smaller PRs — the healthiest contribution pattern.

**Architecture:** Pure documentation changes across three template/process files. No code changes.

**Source:** GitHub issue #672

**Closes:** Fixes #672

**Behavioral Contracts:** See Part 1, Section B below

---

## Phase 0: Component Context

1. **Building block:** Documentation templates and process definitions
2. **Adjacent blocks:** PR workflow (references micro-plan template), writing-plans skill (reads micro-plan-prompt.md)
3. **Invariants touched:** None (no code changes)
4. **Construction Site Audit:** N/A — no structs modified

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds a compact plan format to the micro-plan template for Small tier PRs. The compact format retains behavioral contracts (GIVEN/WHEN/THEN) and TDD tasks — the substance — while dropping sections that add no value for trivial changes: executive summary, component interaction diagram, deviation log, review guide, implementation overview, and test strategy table. The PR workflow's Size Tiers table is updated to reference the compact format, and the agent prompt gains a compact-mode generation path.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Compact Format Structure
- GIVEN a contributor writing a Small tier PR plan
- WHEN they use the compact format
- THEN the plan contains exactly: header (Goal, Closes, Source), behavioral contracts (GIVEN/WHEN/THEN), TDD tasks (files, test, impl, verify, commit), and sanity checklist — and no other sections
- MECHANISM: Template defines the compact format with clear section list

BC-2: Compact Format Criteria Match Small Tier
- GIVEN the PR Size Tiers table in pr-workflow.md
- WHEN a PR qualifies as Small tier
- THEN the compact plan format is referenced as an option for that tier
- MECHANISM: Size Tiers table updated with compact format reference

BC-3: Agent Prompt Compact Mode
- GIVEN the micro-plan-prompt.md agent instructions
- WHEN the source work qualifies as Small tier
- THEN the prompt instructs the agent to produce a compact plan instead of the full template
- MECHANISM: Conditional section in the prompt with criteria and output format

**Negative Contracts:**

BC-4: Full Format Preserved
- GIVEN a contributor writing a Medium or Large tier PR plan
- WHEN they use the micro-plan template
- THEN the full format (all sections) is still required — compact format is not available
- MECHANISM: Compact format section clearly states criteria; full format sections unchanged

BC-5: No Rigor Reduction
- GIVEN a plan written in compact format
- WHEN reviewed
- THEN behavioral contracts MUST still use GIVEN/WHEN/THEN with observable THEN clauses, and tasks MUST still follow TDD (test → fail → implement → pass → lint → commit)
- MECHANISM: Compact format template explicitly includes these requirements

### C) Component Interaction

N/A — documentation-only change. No code components interact.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue lists 4 scope items including "Update CLAUDE.md if the micro-plan description changes" | CLAUDE.md update included as Task 4 only if needed | SIMPLIFICATION — CLAUDE.md already describes micro plans generically; compact format doesn't change the description |
| Issue problem statement mentions "policy template behind existing interface" as Small tier | Plan scopes compact format to existing Small tier criteria only (docs-only or ≤3 mechanical files) | CORRECTION — "policy template behind existing interface" is Medium tier in pr-workflow.md, not Small. The issue text was imprecise. |
| Issue compact format example omits Lint step | Plan adds Lint step to task template | ADDITION — Lint is part of the standard TDD cycle and should not be skipped even for Small tier |

### E) Review Guide

- **Tricky part:** Getting the compact format criteria right — they must exactly match the existing Small tier definition to avoid confusion
- **Scrutinize:** BC-1 (does the compact format retain all essential rigor?) and BC-3 (does the agent prompt produce the right format for the right tier?)
- **Safe to skim:** The full format sections remain unchanged
- **Known debt:** None

---

## Part 2: Executable Implementation

### F) Implementation Overview

Files to modify:
- `docs/contributing/templates/micro-plan.md` — add compact format section
- `docs/contributing/pr-workflow.md` — update Size Tiers table
- `docs/contributing/templates/micro-plan-prompt.md` — add compact generation mode

Key decisions:
- Compact format is an ALTERNATIVE section at the top of the template, not a replacement
- Criteria are stated once in the template and referenced from pr-workflow.md (single source of truth)
- Sanity checklist is kept in compact format (it's a quality gate, not ceremony)

### G) Task Breakdown

#### Task 1: Add compact format section to micro-plan.md (BC-1, BC-4, BC-5)

**Files:**
- Modify: `docs/contributing/templates/micro-plan.md`

**Step 1: Add compact format section after the template intro, before "Document Header"**

Add the following section after line 16 (after the two-audiences paragraph, before the `---` that precedes "Document Header").

**Important:** The compact format example uses a 4-backtick fence (``````) because it contains markdown formatting. The outer code block in this plan uses 3 backticks. When inserting into micro-plan.md, use 4-backtick fences for the example block.

The content to insert (between the existing `---` separator and `## Document Header`):

````markdown
## Compact Format (Small Tier PRs)

For Small tier PRs (see [PR Size Tiers](../pr-workflow.md#pr-size-tiers)), use this streamlined format instead of the full template below. The compact format retains behavioral rigor (contracts + TDD) while dropping sections that add no value for trivial changes.

**Criteria — use compact format when ALL of these apply:**

- Docs-only with no process/workflow semantic changes (typo fixes, formatting, comment updates, link fixes), OR
- ≤3 files changed AND only mechanical changes AND no behavioral logic AND no new interfaces/types AND no new CLI flags

!!! note "Process/workflow semantic changes"
    Changing how the PR workflow operates, adding new template sections, or modifying review criteria are semantic changes — use the full format even if the PR is docs-only.

**Compact plan structure:**

```
# [Title] Implementation Plan

**Goal:** One sentence a non-contributor could understand.
**Source:** Link to source of work.
**Closes:** GitHub issue numbers (e.g., `Fixes #123`).

## Behavioral Contracts

BC-1: <Name>
- GIVEN <precondition>
- WHEN <action>
- THEN <observable outcome>

[Repeat for each contract. Quality gate: every THEN clause must describe
observable behavior, not internal structure.]

## Tasks

### Task 1: <Name> (BC-1)

**Files:** create/modify `path/to/file`, test `path/to/test`

**Test:**
[Complete test code]

**Impl:**
[Complete implementation code]

**Verify:** `go test ./path/... -run TestName`
**Lint:** `golangci-lint run ./path/...`
**Commit:** `feat(scope): description (BC-1)`

[Repeat for each task.]

## Sanity Checklist

[Same checklist as full format — antipattern rules still apply.]
```

**Sections omitted in compact format** (compared to full template):

- Phase 0: Component Context
- Part 1 Section A: Executive Summary
- Part 1 Section C: Component Interaction
- Part 1 Section D: Deviation Log
- Part 1 Section E: Review Guide
- Part 2 Section F: Implementation Overview
- Part 2 Section H: Test Strategy Table
- Part 2 Section I: Risk Analysis
- Appendix: File-Level Implementation Details

**What's kept and why:**

- **Behavioral contracts** — the substance of what the PR guarantees (non-negotiable)
- **TDD tasks** — executable implementation steps (non-negotiable)
- **Sanity checklist** — quality gate (catches antipatterns regardless of PR size)
````

**Step 2: Verify the full format sections are unchanged**

Visual inspection — no modifications to any existing section below the new compact format section.

**Step 3: Commit**

```bash
git add docs/contributing/templates/micro-plan.md
git commit -m "docs(templates): add compact plan format for Small tier PRs (BC-1, BC-4, BC-5)

- Add Compact Format section to micro-plan template
- Define criteria matching Small tier from pr-workflow.md
- List omitted vs retained sections with rationale

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 2: Update PR Size Tiers table in pr-workflow.md (BC-2)

**Files:**
- Modify: `docs/contributing/pr-workflow.md`

**Step 1: Add a note after the PR Size Tiers table and its Rules section**

After the existing rules list (line ~410, after "Human reviewer can override"), add a MkDocs admonition:

```markdown

!!! note "Compact Plan Format"
    Small tier PRs may use the [compact plan format](templates/micro-plan.md#compact-format-small-tier-prs) instead of the full micro-plan template. The compact format retains behavioral contracts and TDD tasks but drops ceremony sections. See the template for criteria and structure.
```

The Small tier row in the table itself does NOT change — the criteria are already correct. The note provides the cross-reference without cluttering the table.

**Step 2: Commit**

```bash
git add docs/contributing/pr-workflow.md
git commit -m "docs(workflow): reference compact plan format in PR Size Tiers (BC-2)

- Add note after Size Tiers table pointing to compact format
- Small tier description unchanged (criteria defined in template)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 3: Add compact generation mode to micro-plan-prompt.md (BC-3)

**Files:**
- Modify: `docs/contributing/templates/micro-plan-prompt.md`

**Step 1: Add compact mode section before the QUALITY BAR section**

Insert a new section before line 634 (the `======` delimiter before "QUALITY BAR"). Note: the micro-plan-prompt.md file is plain text (not rendered markdown), so nested code fences are not an issue — the triple backticks inside are just example text within the `======` delimited section.

````
======================================================================
COMPACT MODE (SMALL TIER PRs)
======================================================================

If the source of work meets ALL of these criteria, produce a COMPACT
plan instead of the full format above:

- Docs-only with no process/workflow semantic changes, OR
- ≤3 files changed AND only mechanical changes AND no behavioral logic
  AND no new interfaces/types AND no new CLI flags

COMPACT OUTPUT FORMAT:

```markdown
# [Title] Implementation Plan

**Goal:** [One sentence]
**Source:** [Link]
**Closes:** [Issue numbers]

## Behavioral Contracts

BC-1: <Name>
- GIVEN <precondition>
- WHEN <action>
- THEN <observable outcome>

## Tasks

### Task N: <Name> (BC-X)

**Files:** create/modify X, test Y

**Test:**
[Complete test code]

**Impl:**
[Complete implementation code]

**Verify:** `go test ./path/... -run TestName`
**Lint:** `golangci-lint run ./path/...`
**Commit:** `type(scope): description (BC-X)`

## Sanity Checklist

[Include full checklist from Phase 8]
```

DO NOT produce compact format if:
- The PR adds new interfaces, types, or CLI flags
- The PR changes behavioral logic (not just mechanical/formatting)
- The PR touches >3 files
- You are unsure — when in doubt, use the full format
````

**Step 2: Commit**

```bash
git add docs/contributing/templates/micro-plan-prompt.md
git commit -m "docs(templates): add compact generation mode to agent prompt (BC-3)

- Add COMPACT MODE section with criteria and output format
- Explicit guard: when in doubt, use full format

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 4: Verify CLAUDE.md accuracy

**Files:**
- Check: `CLAUDE.md`

**Step 1: Verify micro-plan references in CLAUDE.md**

Grep CLAUDE.md for "micro-plan" references. The existing text says:
> Micro plans (single PR): Full implementation detail with behavioral contracts, TDD tasks, exact code. Written per `docs/contributing/templates/micro-plan.md`

This is still accurate — the compact format is part of the micro-plan template. No CLAUDE.md update needed unless the description has changed.

**Step 2: No commit needed if no changes**

---

### H) Test Strategy

No code tests — this is a documentation-only PR. Verification is structural:

| Contract | Task | Verification |
|----------|------|-------------|
| BC-1 | Task 1 | Compact format section exists with correct structure |
| BC-2 | Task 2 | Size Tiers table references compact format |
| BC-3 | Task 3 | Agent prompt includes compact mode section |
| BC-4 | Task 1 | Full format sections unchanged (visual diff) |
| BC-5 | Task 1 | Compact format explicitly requires GIVEN/WHEN/THEN + TDD |

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Contributors use compact format for Medium/Large PRs | Low | Medium | Clear criteria with "when in doubt, use full format" | Task 1, 3 |
| Drift between criteria in template vs pr-workflow.md | Low | Low | Template is single source of truth; workflow references it | Task 2 |
| Agent produces compact format inappropriately | Low | Medium | Explicit guard in prompt: "DO NOT produce compact format if..." | Task 3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates
- [x] No hidden global state impact
- [x] N/A — no new code (golangci-lint)
- [x] N/A — no test helpers
- [x] CLAUDE.md verified — no update needed (Task 4)
- [x] No stale references left in CLAUDE.md
- [x] Documentation DRY — template is single source of truth for criteria
- [x] Deviation log reviewed — one SIMPLIFICATION documented
- [x] Each task produces working output
- [x] Task dependencies correctly ordered (1 → 2 → 3 → 4)
- [x] All contracts mapped to tasks
- [x] N/A — no golden dataset
- [x] N/A — no struct construction sites
- [x] N/A — not part of a macro plan

**Antipattern rules:** N/A — documentation-only PR, no Go code changes.
