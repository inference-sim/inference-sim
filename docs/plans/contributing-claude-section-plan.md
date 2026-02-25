# CONTRIBUTING.md Claude Code Section Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a unified "Contributing with Claude Code" section to CONTRIBUTING.md that covers the full range of contribution journeys — from idea to design to plan to PR — replacing the current split between "Development Workflow" and "Human Contributor Quick Path."

**The problem today:** CONTRIBUTING.md covers one entry point well ("I have code to contribute"), but doesn't guide contributors through three other journeys: turning an idea into a design doc, turning a design into a macro plan, or using Claude Code skills to orchestrate any of these. The current "Development Workflow" section lists Claude skill names without explaining what they are or when to use them, while "Human Contributor Quick Path" feels like an afterthought. Contributors with Claude Code don't know how to start from an idea; contributors without it don't understand what the automated tools do for them.

**What this PR adds:**
1. **Journey-based navigation** — a decision table that routes contributors to the right workflow based on what they want to do (fix a bug, add a policy, build a feature, run an experiment)
2. **Idea-to-PR coverage** — documents the full pipeline: brainstorming → design doc → macro plan → micro plan → PR, with the exact Claude skill invocations for each phase
3. **Unified audience** — serves both Claude Code users (skill invocations) and manual contributors (fallback steps) in a single section instead of two separate ones

**Why this matters:** BLIS has rich process infrastructure (4 process docs, 3 templates, 2 custom skills) but no single place that shows a new contributor when to use what. This section becomes the map.

**Architecture:** Documentation-only changes to `CONTRIBUTING.md` and `docs/standards/principles.md`. Replace lines 130–155 of CONTRIBUTING.md ("Development Workflow" + "Human Contributor Quick Path") with a new "Contributing with Claude Code" section (~80 lines). Update source-of-truth map in principles.md to register new working copies. No Go code, no new packages, no interface changes.

**Source:** Approved brainstorming design from current conversation session.

**Closes:** N/A — no linked issues.

**Behavioral Contracts:** See Part 1, Section B below.

---

## Part 1: Design Validation

### A) Executive Summary

This PR replaces two sections in CONTRIBUTING.md — "Development Workflow" (a 9-step Claude-focused list) and "Human Contributor Quick Path" (a 5-step manual list) — with a single "Contributing with Claude Code" section organized around 4 contribution journeys. Each journey shows the high-level steps, the Claude skill to invoke at each step, and links to the detailed process doc. A "Without Claude Code" subsection absorbs the manual path. No other sections of CONTRIBUTING.md change.

Adjacent docs: `docs/process/pr-workflow.md` (canonical PR workflow), `docs/process/design.md` (design process), `docs/process/macro-plan.md` (macro planning), `docs/process/hypothesis.md` (experiment process). CONTRIBUTING.md is a working copy of the PR workflow and hypothesis workflow per the source-of-truth map in `docs/standards/principles.md:92`.

No deviations from approved design.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Journey Router Table
- GIVEN a contributor opens CONTRIBUTING.md
- WHEN they want to contribute but don't know which process to follow
- THEN they MUST find a decision table that maps contribution types (bug fix, new policy, new feature, hypothesis experiment) to specific workflow subsections
- MECHANISM: "Choosing Your Journey" table with anchor links

BC-2: Full Idea-to-PR Pipeline
- GIVEN a contributor with Claude Code wants to build a new feature from scratch
- WHEN they follow the "New Feature" journey
- THEN they MUST see the complete pipeline: brainstorm → design doc → macro plan (if multi-PR) → micro plan → PR, with skill invocations for each phase
- MECHANISM: "New Feature" subsection with numbered phases and `/skill` invocations

BC-3: Bug Fix Journey
- GIVEN a contributor with Claude Code wants to fix a bug
- WHEN they follow the "Bug Fix / Small Change" journey
- THEN they MUST see the abbreviated pipeline: worktree → micro plan → implement → PR, with the key skill invocations
- MECHANISM: "Bug Fix / Small Change" subsection referencing pr-workflow.md

BC-4: Extension Journey
- GIVEN a contributor wants to add a new policy or scorer
- WHEN they follow the "New Policy or Extension" journey
- THEN they MUST see a reference to `docs/extension-recipes.md` and the extension type table already in CONTRIBUTING.md
- MECHANISM: "New Policy or Extension" subsection cross-referencing "Adding New Components" section

BC-5: Hypothesis Journey Link
- GIVEN a contributor wants to run or create a hypothesis experiment
- WHEN they reach the hypothesis journey entry in the decision table
- THEN they MUST be directed to the existing "Running or Contributing Hypothesis Experiments" section (not duplicated content)
- MECHANISM: Table row links to existing anchor + mentions `hypothesis-experiment` skill

BC-6: Manual Contributor Path
- GIVEN a contributor without Claude Code
- WHEN they read the "Contributing with Claude Code" section
- THEN they MUST find a "Without Claude Code" subsection describing the 5-step manual workflow and stating that maintainers run the automated review protocols on submitted PRs
- MECHANISM: "Without Claude Code" subsection absorbing lines 146–155

BC-7: Process Doc Links
- GIVEN a contributor following any journey
- WHEN they need more detail than the summary provides
- THEN each journey MUST link to the canonical process doc (`pr-workflow.md`, `design.md`, `macro-plan.md`, or `hypothesis.md`)
- MECHANISM: "Full process:" links after each journey's step list

**Negative Contracts:**

BC-8: No Content Duplication
- GIVEN the source-of-truth map in `docs/standards/principles.md`
- WHEN writing journey descriptions
- THEN CONTRIBUTING.md MUST NOT duplicate detailed workflow steps from the canonical process docs — it MUST remain a summary with links
- MECHANISM: Each journey is 5-10 lines of steps + links, not a reproduction of pr-workflow.md

BC-9: No Unlinked Sections
- GIVEN the existing sections below the replaced content (Engineering Principles, Antipattern Checklist, etc.)
- WHEN replacing lines 130–155
- THEN no existing content outside lines 130–155 MUST be modified or broken
- MECHANISM: Only lines 130–155 are replaced; all other sections stay

### C) Component Interaction

```
CONTRIBUTING.md (this PR)
    │
    ├── "Choosing Your Journey" table ──→ links to subsections below
    │
    ├── "Bug Fix / Small Change" ──→ refs docs/process/pr-workflow.md
    │
    ├── "New Policy or Extension" ──→ refs docs/extension-recipes.md
    │                                    refs "Adding New Components" section (same doc)
    │
    ├── "New Feature (Idea to PR)" ──→ refs docs/process/design.md
    │                                    refs docs/process/macro-plan.md
    │                                    refs docs/process/pr-workflow.md
    │
    ├── "Hypothesis Experiment" ──→ refs "Running or Contributing..." section (same doc)
    │                                refs docs/process/hypothesis.md
    │
    └── "Without Claude Code" ──→ refs docs/templates/micro-plan.md
                                     refs docs/process/pr-workflow.md
```

No new types, no APIs, no state changes. Pure documentation restructure.

Extension friction: N/A (documentation only).

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| No deviations | — | Design was approved in conversation before planning |

### E) Review Guide

1. **THE TRICKY PART:** Getting the right level of detail — enough that a new contributor can follow the journey, but not so much that it duplicates the canonical process docs.
2. **WHAT TO SCRUTINIZE:** BC-8 (no duplication) and BC-7 (all links correct). Verify each linked doc path actually exists.
3. **WHAT'S SAFE TO SKIM:** BC-9 (no unlinked sections) — mechanical verification that lines outside 130–155 are untouched.
4. **KNOWN DEBT:** The "Running or Contributing Hypothesis Experiments" section (lines 249–282) could also benefit from Claude skill invocations, but that's out of scope for this PR.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files:**
- Modify: `CONTRIBUTING.md:130-155` (replace "Development Workflow" + "Human Contributor Quick Path")
- Modify: `docs/standards/principles.md:86-92` (add design/macro-plan working copies to source-of-truth map)

**Key decisions:**
- Journey 4 (hypothesis) is a brief pointer to the existing section, not a full subsection
- "Without Claude Code" subsection reuses most of the existing "Human Contributor Quick Path" text
- Skill invocations shown as inline code, not full command blocks (detail lives in pr-workflow.md)

**Confirmation:** No dead code, no dead links. All content exercisable by following the linked process docs.

---

### G) Task Breakdown

#### Task 1: Replace "Development Workflow" and "Human Contributor Quick Path" with new "Contributing with Claude Code" section

**Contracts Implemented:** BC-1, BC-2, BC-3, BC-4, BC-5, BC-6, BC-7, BC-8, BC-9

**Files:**
- Modify: `CONTRIBUTING.md:130-155`

**Step 1: Write the new section content**

Replace lines 130–155 (from `## Development Workflow` through the end of "Human Contributor Quick Path") with the following content:

```markdown
## Contributing with Claude Code

> **Canonical source:** [`docs/process/pr-workflow.md`](docs/process/pr-workflow.md). If this section diverges, pr-workflow.md is authoritative.

BLIS development workflows are orchestrated through [Claude Code](https://claude.ai/code) skills — structured sequences that handle worktree creation, plan generation, multi-perspective review with convergence enforcement, and PR creation. Contributors with Claude Code get the full automated pipeline. Contributors without it follow the manual path below and still go through the same quality gates (maintainers run the automated reviews on submitted PRs).

**Prerequisites:** Claude Code installed with project skills available (`convergence-review`, `hypothesis-experiment`) and general Claude Code skills (`writing-plans`, `executing-plans`, `commit-push-pr`). See [`docs/process/pr-workflow.md`](docs/process/pr-workflow.md) for the full skill table. Before your first contribution, read [`docs/templates/design-guidelines.md`](docs/templates/design-guidelines.md) — it covers module architecture, extension types, and DES foundations.

### Choosing Your Journey

| You want to... | Journey | Starts with |
|---|---|---|
| Fix a bug or make a small change | [Bug Fix / Small Change](#bug-fix--small-change) | A GitHub issue or observed bug |
| Add a new policy, scorer, or extension | [New Policy or Extension](#new-policy-or-extension) | An existing interface to implement |
| Build a new feature or subsystem | [New Feature (Idea to PR)](#new-feature-idea-to-pr) | An idea or requirement |
| Validate simulator behavior | [Hypothesis Experiment](#running-or-contributing-hypothesis-experiments) | A behavioral prediction |

For hypothesis experiments, see [Running or Contributing Hypothesis Experiments](#running-or-contributing-hypothesis-experiments) below. With Claude Code, the `hypothesis-experiment` skill orchestrates the full Steps 0–10 workflow.

### Bug Fix / Small Change

The lightest path. For bug fixes, docs updates, and single-PR changes that don't introduce new module boundaries.

1. **Create worktree** — `/superpowers:using-git-worktrees fix-<name>`
2. **Write micro plan** — `/superpowers:writing-plans` using `@docs/templates/micro-plan.md`
3. **Review plan** — `/pr-review-toolkit:review-pr` then `/convergence-review pr-plan <plan-path>`
4. **Human approval** — review contracts and tasks, approve to proceed
5. **Implement** — `/superpowers:executing-plans @<plan-path>`
6. **Review code** — `/pr-review-toolkit:review-pr` then `/convergence-review pr-code`
7. **Self-audit + commit** — deliberate critical thinking, then `/commit-commands:commit-push-pr`

Full process: [`docs/process/pr-workflow.md`](docs/process/pr-workflow.md)

### New Policy or Extension

For adding a routing policy, admission policy, scorer, scheduler, priority policy, or tier composition — anything behind an existing interface.

1. **Identify extension type** — see [Adding New Components](#adding-new-components) below
2. **Create worktree** — `/superpowers:using-git-worktrees <extension-name>`
3. **Write micro plan** — `/superpowers:writing-plans` using `@docs/templates/micro-plan.md` and `@docs/extension-recipes.md`
4. **Follow steps 3–7 from Bug Fix** (review → approve → implement → review → commit)

No design doc needed for policy templates. For tier compositions, a design doc is recommended — see the extension type table in [Adding New Components](#adding-new-components). Full process: [`docs/process/pr-workflow.md`](docs/process/pr-workflow.md)

### New Feature (Idea to PR)

The full pipeline for features that introduce new module boundaries, new interfaces, or span multiple PRs.

**Phase 1 — Idea to Design:**
1. **Explore approaches** — discuss design options with Claude, settle on an approach
2. **Write design doc** — following [`docs/templates/design-guidelines.md`](docs/templates/design-guidelines.md)
3. **Review design** — `/convergence-review design <path>` (8 perspectives)
4. **Human approval** — review design doc before planning begins

Full process: [`docs/process/design.md`](docs/process/design.md)

**Phase 2 — Design to Macro Plan** (skip if single-PR):
5. **Write macro plan** — decompose into PRs following [`docs/templates/macro-plan.md`](docs/templates/macro-plan.md)
6. **Review macro plan** — `/convergence-review macro-plan <path>` (8 perspectives)
7. **Human approval** — review PR decomposition and module contracts

Full process: [`docs/process/macro-plan.md`](docs/process/macro-plan.md)

**Phase 3 — Plan to PR** (repeat for each PR):
8. **Follow the Bug Fix journey** (steps 1–7) using the macro plan section or design doc as the source document

Each phase produces an artifact that feeds the next. Human approval gates between phases prevent wasted work.

### Without Claude Code

If you are not using Claude Code, here is the simplified workflow:

1. **Branch** — `git checkout -b feature/my-change`
2. **Plan** — write an implementation plan following `docs/templates/micro-plan.md`. Include behavioral contracts (GIVEN/WHEN/THEN) and a task breakdown. Post the plan as a PR draft or issue comment for review.
3. **Implement** — follow TDD: write a failing test, implement the minimal code to pass it, run `go test ./...`, run `golangci-lint run ./...`, commit. Repeat for each contract.
4. **Self-review** — check the [Antipattern Checklist](#antipattern-checklist) below. Run `go build ./... && go test ./... && golangci-lint run ./...` one final time.
5. **PR** — push your branch and open a PR. Maintainers will run the automated review protocols (convergence-review with 10 perspectives).

The automated review tools (convergence-review, pr-review-toolkit) are run by maintainers — you do not need Claude Code installed. Your PR will go through the same quality gates regardless of tooling.

For design docs and macro plans: follow the same templates ([`docs/templates/design-guidelines.md`](docs/templates/design-guidelines.md), [`docs/templates/macro-plan.md`](docs/templates/macro-plan.md)) and submit for review. Maintainers will run convergence review.

Full process: [`docs/process/pr-workflow.md`](docs/process/pr-workflow.md) (the same workflow applies regardless of tooling)
```

**Step 2: Verify the edit**

Run: Visually confirm the new section in CONTRIBUTING.md:
- Lines before: "Your First Contribution" section ends at line 128 with the `> **Important:**` note
- New section starts with `## Contributing with Claude Code`
- Lines after: `## Engineering Principles` continues unchanged
- All internal anchor links resolve (test by grepping for the heading text)

**Step 3: Verify all links resolve**

Run the following checks:
```bash
# Verify linked files exist
ls docs/process/pr-workflow.md docs/process/design.md docs/process/macro-plan.md docs/templates/design-guidelines.md docs/templates/macro-plan.md docs/templates/micro-plan.md docs/extension-recipes.md
```
Expected: All files exist

```bash
# Verify internal anchor targets exist in CONTRIBUTING.md
grep -c "## Adding New Components" CONTRIBUTING.md
grep -c "## Running or Contributing Hypothesis Experiments" CONTRIBUTING.md
grep -c "## Antipattern Checklist" CONTRIBUTING.md
```
Expected: Each returns 1

**Step 4: Verify no other sections were modified**

```bash
# Compare line counts — new section should be ~90 lines replacing ~25 lines
wc -l CONTRIBUTING.md
```
Expected: ~368 lines (was 303, adding ~90, removing ~25 = ~368)

**Step 5: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs: add unified 'Contributing with Claude Code' section to CONTRIBUTING.md

- Replace split 'Development Workflow' + 'Human Contributor Quick Path'
  with journey-based 'Contributing with Claude Code' section
- Cover 4 contribution journeys: bug fix, new policy, new feature, hypothesis
- New Feature journey chains brainstorming → design → macro plan → PR
- 'Without Claude Code' subsection preserves manual contributor path
- All journeys link to canonical process docs (no content duplication)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 2: Update source-of-truth map in principles.md

**Contracts Implemented:** BC-8 (DRY compliance)

**Files:**
- Modify: `docs/standards/principles.md:86-92`

**Step 1: Add design and macro-plan process rows to source-of-truth map**

In `docs/standards/principles.md`, after the "Extension recipes" row (line 86), add two new rows:

```markdown
| Design process | `docs/process/design.md` | CONTRIBUTING.md (summary) |
| Macro-plan process | `docs/process/macro-plan.md` | CONTRIBUTING.md (summary) |
```

**Step 2: Verify the map is well-formed**

Run: Visually confirm the table has no formatting issues and all rows are aligned.

**Step 3: Commit**

```bash
git add docs/standards/principles.md
git commit -m "docs: register design/macro-plan working copies in source-of-truth map

- CONTRIBUTING.md now summarizes design process and macro-plan process
- Register as working copies per DRY documentation standard

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Verification |
|----------|------|-----------|--------------|
| BC-1     | Task 1 | Manual | "Choosing Your Journey" table present with 4 rows + anchor links |
| BC-2     | Task 1 | Manual | "New Feature" subsection shows brainstorm → design → macro → PR pipeline |
| BC-3     | Task 1 | Manual | "Bug Fix" subsection shows 7 steps with skill invocations |
| BC-4     | Task 1 | Manual | "New Policy" subsection references extension-recipes.md and "Adding New Components" |
| BC-5     | Task 1 | Manual | Table row links to existing hypothesis section anchor |
| BC-6     | Task 1 | Manual | "Without Claude Code" subsection present with 5-step workflow |
| BC-7     | Task 1 | Manual | Each journey subsection contains "Full process:" link |
| BC-8     | Task 1 | Manual | No journey subsection exceeds 15 lines of step descriptions |
| BC-9     | Task 1 | Manual | `git diff` shows changes only in the replaced line range |

No golden dataset updates needed. No Go tests affected.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Broken anchor links | Low | Medium | Step 3 verifies all anchors resolve |
| Content too sparse for newcomers | Medium | Low | Links to detailed process docs provide depth on demand |
| Source-of-truth map drift | Low | Low | CONTRIBUTING.md remains a summary per existing map entry — no new canonical content |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope (only lines 130–155 replaced)
- [x] No unexercised content (all links verifiable)
- [x] No partial implementations
- [x] No breaking changes to existing anchors (checked "Adding New Components", "Antipattern Checklist", "Running or Contributing...")
- [x] No hidden global state impact
- [x] CLAUDE.md File Organization tree unchanged (CONTRIBUTING.md already listed)
- [x] No stale references in CLAUDE.md
- [x] Documentation DRY: CONTRIBUTING.md remains a working copy per source-of-truth map; new working copies (design process, macro-plan process) registered in Task 2
- [x] Task dependencies correctly ordered (single task)
- [x] All contracts mapped to Task 1

---

## Appendix: File-Level Implementation Details

### File: `CONTRIBUTING.md`

**Purpose:** Replace lines 130–155 with the new "Contributing with Claude Code" section.

**What's removed:**
- Lines 130–143: "## Development Workflow" — a 9-step Claude-focused list that referenced skill names without explaining when to use them
- Lines 145–155: "## Human Contributor Quick Path" — a 5-step manual workflow that felt disconnected from the main flow

**What's added:**
- "## Contributing with Claude Code" header with intro paragraph + prerequisites
- "### Choosing Your Journey" decision table (4 rows)
- "### Bug Fix / Small Change" (7 steps with skill invocations, link to pr-workflow.md)
- "### New Policy or Extension" (4 steps, references extension-recipes.md, link to pr-workflow.md)
- "### New Feature (Idea to PR)" (3 phases: idea→design, design→macro, plan→PR, links to all 3 process docs)
- "### Without Claude Code" (5-step manual path, absorbed from old "Human Contributor Quick Path")

**Anchor dependencies verified** (anchors are heading-text-based, line numbers will shift after edit):
- `#adding-new-components` → "## Adding New Components" heading (exists in current file)
- `#running-or-contributing-hypothesis-experiments` → "## Running or Contributing Hypothesis Experiments" heading (exists in current file)
- `#antipattern-checklist` → "## Antipattern Checklist" heading (exists in current file)
- `#bug-fix--small-change` → new heading (within replaced section)
- `#new-policy-or-extension` → new heading (within replaced section)
- `#new-feature-idea-to-pr` → new heading (within replaced section)

**Key constraint:** CONTRIBUTING.md is a working copy in the source-of-truth map (`docs/standards/principles.md:92`). The PR workflow summary and hypothesis workflow summary must remain summaries with `>` canonical-source pointers, not reproductions. The new section achieves this by showing only skill invocations and step names, with "Full process:" links for detail.
