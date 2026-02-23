# Cleanup Stale Plan Files Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove completed/deprecated plan files from `docs/plans/` and archive design docs, reducing 72 files to 14.

**The problem today:** `docs/plans/` has accumulated 72 files over 18 PRs of development. 58 of those are completed micro-plans, deprecated redirects, or one-off agent prompts that no longer serve any purpose. The directory is difficult to navigate and creates confusion about which plans are active.

**What this PR adds:**
1. Removes 58 completed/deprecated files (micro-plans, redirects, agent prompts)
2. Archives 9 completed design docs to `docs/plans/archive/` — these have lasting architectural reference value
3. Updates all cross-references in 7 files: `CLAUDE.md`, `docs/pr-history.md`, `docs/process/pr-workflow.md`, `docs/extension-recipes.md`, `docs/templates/design-guidelines.md`, `CONTRIBUTING.md`

**Why this matters:** Keeps the repository navigable and prevents confusion between active and historical plans.

**Architecture:** Docs-only change. No Go code, no tests, no behavioral changes. File deletions + moves + cross-reference updates.

**Source:** User request to clean up `docs/plans/` directory

**Closes:** N/A — no linked issues

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR removes 58 stale plan files from `docs/plans/`, moves 9 completed design docs to `docs/plans/archive/`, and keeps 5 active files in place. All cross-references in CLAUDE.md, pr-history.md, pr-workflow.md, and extension-recipes.md are updated.

No Go code or tests are affected. The only risk is broken cross-references.

### B) Behavioral Contracts

**BC-1: No broken functional cross-references**
- GIVEN the plan cleanup is complete
- WHEN any `.md` file in the repo references a `docs/plans/` path
- THEN the referenced file MUST exist at that path
- MECHANISM: grep for all `docs/plans/` references and verify targets exist

**BC-2: Active plans remain accessible**
- GIVEN 5 files are classified as active
- WHEN a user navigates to `docs/plans/`
- THEN these 5 files MUST be present at their original paths:
  `2026-02-19-weighted-scoring-macro-plan.md`, `2026-02-21-latency-model-extraction-design.md`,
  `latency-model-extraction-plan.md`, `research.md`, `research-hypotheses-problem.md`

**BC-3: Archived design docs remain accessible**
- GIVEN 9 design docs are archived
- WHEN a user navigates to `docs/plans/archive/`
- THEN all 9 files MUST be present at their new paths

**BC-4: CLAUDE.md file tree reflects new structure**
- GIVEN the `docs/plans/` structure changed
- WHEN CLAUDE.md's file tree section describes `docs/plans/`
- THEN it MUST show the `archive/` subdirectory

### C) Component Interaction

No component interactions — docs-only change.

### D) Deviation Log

No source document to deviate from — this is a user-requested cleanup.

### E) Review Guide

1. **THE TRICKY PART**: Cross-reference completeness — did we miss any `.md` file that links to a removed plan?
2. **WHAT TO SCRUTINIZE**: The `docs/process/pr-workflow.md` updates — this file has many example invocations referencing specific plan files. Distinguish between *functional* references (header section, active links) and *illustrative* examples (showing the pattern with historical file names).
3. **WHAT'S SAFE TO SKIM**: The file deletion list — straightforward removal of completed work.
4. **KNOWN DEBT**: `pr-workflow.md` body examples still reference historical file names like `pr8-routing-state-and-policy-bundle-plan.md`. These are illustrative examples showing the skill invocation pattern, not functional links. Updating them would change ~20 lines of examples for no functional benefit.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `docs/pr-history.md` — update design doc paths, remove micro-plan section
- `docs/process/pr-workflow.md` — update header section + body references to `prmicroplanprompt-v2.md` → `docs/templates/micro-plan.md`
- `docs/extension-recipes.md` — update `pr12-architectural-predesign.md` path to archive
- `docs/templates/design-guidelines.md` — update 2 archived file references in References section
- `CONTRIBUTING.md` — update `prmicroplanprompt-v2.md` reference to `docs/templates/micro-plan.md`
- `CLAUDE.md` — update file tree to show `archive/` subdirectory

**Files to delete:** 58 completed/deprecated plan files (listed below)
**Files to move:** 9 design docs → `docs/plans/archive/`

**Key decisions:**
- pr-workflow.md *example invocations* (showing historical file names like `pr8-routing-state-and-policy-bundle-plan.md`) left as-is — illustrative, not functional
- pr-workflow.md *procedural instructions* referencing `prmicroplanprompt-v2.md` updated to `docs/templates/micro-plan.md` — these are active instructions, not examples
- pr-workflow.md *changelog entries* referencing `prmicroplanprompt-v2.md` left as-is — historical records
- Archived design docs keep their original filenames (no renaming)

### G) Task Breakdown

#### Task 1: Delete stale files and move design docs to archive

**Contracts Implemented:** BC-2, BC-3

**Files:**
- Create: `docs/plans/archive/` directory
- Move: 9 design docs to `docs/plans/archive/`
- Delete: 58 completed/deprecated files

**Step 1: Create archive directory and move design docs**

```bash
mkdir -p docs/plans/archive
mv docs/plans/2026-02-06-evolutionary-policy-optimization-design.md \
   docs/plans/2026-02-13-mock-study-findings.md \
   docs/plans/2026-02-13-mock-study-implementation.md \
   docs/plans/2026-02-13-simplification-assessment.md \
   docs/plans/2026-02-16-workload-generator-design.md \
   docs/plans/2026-02-18-hardening-antipattern-refactoring-design.md \
   docs/plans/2026-02-19-weighted-scoring-evolution-design.md \
   docs/plans/2026-02-20-seed-unification-design.md \
   docs/plans/pr12-architectural-predesign.md \
   docs/plans/archive/
```

**Step 2: Remove deprecated/redirect stubs (6 files)**

```bash
rm docs/plans/prmicroplanprompt-v1-deprecated.md \
   docs/plans/2026-02-11-macro-implementation-plan-v2--deprecated.md \
   docs/plans/2026-02-18-design-guidelines.md \
   docs/plans/macroplanprompt.md \
   docs/plans/prmicroplanprompt-v2.md \
   docs/plans/prworkflow.md
```

**Step 3: Remove completed micro-plans (50 files)**

```bash
rm docs/plans/pr1.md \
   docs/plans/PR2-instance-simulator-microplan.md \
   docs/plans/pr3.md \
   docs/plans/pr4-micro-plan.md \
   docs/plans/pr4-phase1-behavioral-contracts.md \
   docs/plans/pr4-phases2-5-implementation-planning.md \
   docs/plans/pr5-micro-plan.md \
   docs/plans/pr6-routing-plan.md \
   docs/plans/pr7-priority-plus-sched-plan.md \
   docs/plans/pr8-routing-state-and-policy-bundle-plan.md \
   docs/plans/pr9-raw-metrics-anomaly-detection-plan.md \
   docs/plans/pr10-workload-generator-plan.md \
   docs/plans/pr12-tiered-kv-plan.md \
   docs/plans/2026-02-17-pr13-decision-traces.md \
   docs/plans/pr17-scorer-framework-plan.md \
   docs/plans/pr18-prefix-affinity-scorer-plan.md \
   docs/plans/hardening-phase1-plan.md \
   docs/plans/hardening-phase2-correctness-fixes-plan.md \
   docs/plans/hardening-phase3-metric-fixes-plan.md \
   docs/plans/phase4-invariant-tests-plan.md \
   docs/plans/fix-193-horizon-warning-plan.md \
   docs/plans/fix-207-default-log-level-plan.md \
   docs/plans/fix-212-input-validation-plan.md \
   docs/plans/fix-226-suppress-priority-inversion-plan.md \
   docs/plans/fix-231-kv-cli-validation-plan.md \
   docs/plans/fix-234-deployment-base-latency-plan.md \
   docs/plans/fix-236-roofline-validation-plan.md \
   docs/plans/fix-237-silent-drop-counter-plan.md \
   docs/plans/fix-240-stale-comments-plan.md \
   docs/plans/fix-254-257-roofline-dead-code-and-tests-plan.md \
   docs/plans/fix-264-262-silent-data-loss-plan.md \
   docs/plans/fix-267-263-roofline-stale-comments-plan.md \
   docs/plans/fix-cli-metric-observability-plan.md \
   docs/plans/fix-silent-correctness-bugs-plan.md \
   docs/plans/fix-hypothesis-bugs-plan.md \
   docs/plans/fix-audit-analyzer-parsers-318-plan.md \
   docs/plans/pr213-modularity-improvements-plan.md \
   docs/plans/pr-num-requests-readme-plan.md \
   docs/plans/pr-readme-docs-261-plan.md \
   docs/plans/consolidate-standards-plan.md \
   docs/plans/seed-unification-284-plan.md \
   docs/plans/refactor-step-metrics-243-plan.md \
   docs/plans/feat-inference-perf-workload-252-plan.md \
   docs/plans/2026-02-20-docs-stale-output-and-scorer-recipe.md \
   docs/plans/2026-02-19-parallel-dev-plan-234-plus-4.md \
   docs/plans/hypothesis-promotions-plan.md \
   docs/plans/promotions-overload-liveness-plan.md \
   docs/plans/h8-kv-pressure-plan.md \
   docs/plans/h14-pathological-templates-plan.md \
   docs/plans/embed-simconfig-247-plan.md
```

**Step 4: Remove one-off agent prompts (2 files)**

```bash
rm docs/plans/2026-02-13-simplification-assessment-team-prompt.md \
   docs/plans/2026-02-13-v3-update-team-prompt.md
```

**Step 5: Verify final state**

```bash
echo "=== Active ===" && ls docs/plans/*.md
echo "=== Archive ===" && ls docs/plans/archive/*.md
```

Expected: 5 active files + this plan file, 9 archived files.

---

#### Task 2: Update cross-references

**Contracts Implemented:** BC-1, BC-4

**Files:**
- Modify: `docs/pr-history.md` — restructure design docs section (active vs archived), remove micro-plan section
- Modify: `docs/process/pr-workflow.md` — update header references
- Modify: `docs/extension-recipes.md:65` — update `pr12-architectural-predesign.md` path
- Modify: `CLAUDE.md:336` — update file tree to show `archive/` subdirectory

**Step 1: Update pr-history.md**

Replace the Design Documents and Micro-Level sections with active/archived structure. Remove the micro-plan section entirely (those files are deleted).

**Step 2: Update pr-workflow.md header**

Replace stale header references:
- `2026-02-11-macro-implementation-plan-v2.md` → current active macro plan
- `prmicroplanprompt-v1-deprecated.md` → archived design docs reference

**Step 3: Update extension-recipes.md**

Line 65: change `docs/plans/pr12-architectural-predesign.md` → `docs/plans/archive/pr12-architectural-predesign.md`

**Step 4: Update CLAUDE.md file tree**

Change:
```
│   ├── plans/                 # Per-feature implementation plans
```
To:
```
│   ├── plans/                 # Active implementation plans
│   │   └── archive/           # Completed design docs (architectural reference)
```

**Step 5: Update pr-workflow.md line 954 stale example**

Change `@docs/plans/macroplan.md` → `@docs/plans/<macro-plan>.md` (the redirect file was removed)

**Step 6: Commit**

```bash
git add -A docs/plans/ docs/pr-history.md docs/process/pr-workflow.md docs/extension-recipes.md CLAUDE.md
git commit -m "docs: clean up stale plan files and archive design docs

Remove 58 completed micro-plans, deprecated redirects, and one-off
agent prompts from docs/plans/. Archive 9 design docs with lasting
architectural reference value to docs/plans/archive/. Update all
cross-references in pr-history.md, pr-workflow.md, extension-recipes.md,
and CLAUDE.md.

72 files → 14 files (5 active + 9 archived).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 3: Verify BC-1 (no broken references)

**Contracts Implemented:** BC-1

**Step 1: Grep for all docs/plans/ references and verify each target exists**

```bash
grep -rn 'docs/plans/' docs/ CLAUDE.md CONTRIBUTING.md | grep -v archive | grep -v '\.md:.*<' | grep -v 'plans/<'
```

Every match must reference a file that exists in `docs/plans/` (not archive). Archive references must use `docs/plans/archive/` path.

---

### H) Test Strategy

| Contract | Task | Test Type | Verification |
|----------|------|-----------|--------------|
| BC-1 | Task 3 | Manual | grep for broken references |
| BC-2 | Task 1 | Manual | `ls docs/plans/*.md` shows 5 active files |
| BC-3 | Task 1 | Manual | `ls docs/plans/archive/*.md` shows 9 files |
| BC-4 | Task 2 | Manual | Read CLAUDE.md file tree section |

No Go tests needed — docs-only change.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Broken cross-reference missed | Low | Medium | BC-1 grep verification in Task 3 |
| Design doc incorrectly classified as removable | Low | Low | 9 design docs preserved in archive; git history has everything |
| Active plan mistakenly removed | Low | High | Only 5 files kept — each verified as active/recent work |

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] CLAUDE.md updated (file tree section)
- [x] No stale references left in CLAUDE.md
- [x] Deviation log reviewed — N/A (no source document)
- [ ] R15: grep for stale plan references after deletion — Task 3
- N/A: All code-related rules (R1-R14, R16-R20) — docs-only PR
