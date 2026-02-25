# Fix Documentation Standards Drift Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix documentation drift where README.md diverged from canonical sources after #386 (KV livelock fix) and PKG-1/PKG-2 extractions.

**The problem today:** README.md has a wrong conservation formula (missing `dropped_unservable`), an incomplete Project Structure tree (missing 7 files added in recent PRs), and misleading file descriptions. A user checking conservation from the README would use the wrong equation. A contributor reading the README for architecture guidance would miss key files like `config.go` and `batch_formation.go`.

**What this PR adds:**
1. Correct conservation formula — README now shows the full INV-1 equation including `dropped_unservable`, matching `docs/standards/invariants.md`
2. Complete Project Structure tree — all files from canonical CLAUDE.md tree appear in README, including `config.go`, `batch_formation.go`, `latency_model.go`, `metrics_utils.go`, `deployment.go`
3. Accurate file descriptions — `batch.go` correctly described as "Batch struct" (not "Batch formation"), `event.go` includes all 6 event types

**Why this matters:** The source-of-truth map in `docs/standards/principles.md` designates CLAUDE.md as canonical for file organization and `docs/standards/invariants.md` as canonical for invariants. Working copies that diverge silently mislead users — exactly the failure mode the DRY documentation principle was designed to prevent.

**Architecture:** Docs-only changes to `README.md`. No code changes. No test changes.

**Source:** Documentation standards audit findings C1, I1-I4.

**Closes:** N/A — no linked issues.

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes 5 documentation drift issues in `README.md` where the working copy diverged from canonical sources (CLAUDE.md, invariants.md). The README is the primary entry point for new users and contributors — inaccuracies here propagate misunderstanding.

No code changes. No adjacent components affected. All changes are additive text corrections in a single file.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Conservation formula accuracy
- GIVEN the README.md Example Output section
- WHEN a user reads the conservation formula explanation
- THEN the formula MUST match INV-1: `injected == completed + still_queued + still_running + dropped_unservable`
- MECHANISM: Update the text at README.md:484 to include `dropped_unservable`

BC-2: Example JSON completeness
- GIVEN the README.md Example Output JSON block
- WHEN a user examines the example output schema
- THEN the JSON MUST include `preemption_count` and `dropped_unservable` fields (with value 0 for the example scenario)
- MECHANISM: Add the two missing fields to the example JSON at README.md:449-474

BC-3: Project Structure tree completeness
- GIVEN the README.md Project Structure tree
- WHEN a contributor reads the tree to understand file organization
- THEN every source code file in the `sim/`, `sim/kv/`, `sim/latency/`, `sim/cluster/`, `sim/workload/`, and `sim/trace/` sections of the canonical CLAUDE.md tree MUST appear in the README tree (excluding `internal/` test infrastructure)
- MECHANISM: Add missing files (`config.go`, `doc.go`, `batch_formation.go`, `latency_model.go`, `metrics_utils.go`, `deployment.go`, `inference_perf.go`) with correct descriptions

BC-4: File description accuracy
- GIVEN the README.md Project Structure tree
- WHEN a contributor reads file descriptions
- THEN `batch.go` MUST be described as "Batch struct" (not "Batch formation") and `event.go` MUST list all 6 event types including `RequestLeft`
- MECHANISM: Correct the descriptions at README.md:675,677

**Negative Contracts:**

BC-5: No content beyond scope
- GIVEN the README.md file
- WHEN this PR is applied
- THEN no sections outside Example Output and Project Structure MUST be modified
- MECHANISM: Changes limited to the two identified sections

### C) Component Interaction

No component interaction — this is a docs-only PR modifying a single file (`README.md`).

### D) Deviation Log

No deviations from source. All changes align README.md working copy with canonical sources (CLAUDE.md and invariants.md).

### E) Review Guide

- **The tricky part:** Ensuring the Project Structure tree matches CLAUDE.md exactly — easy to miss a file or get a description slightly wrong.
- **What to scrutinize:** Compare the updated README tree against CLAUDE.md line by line. Verify the conservation formula matches invariants.md.
- **What's safe to skim:** The JSON example additions are mechanical.
- **Known debt:** README tree is intentionally less detailed than CLAUDE.md (no interface names, no method signatures). This is by design per the source-of-truth map.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files:**
- Modify: `README.md` (Example Output section + Project Structure tree)

**Key decisions:**
- Add `dropped_unservable` and `preemption_count` to example JSON with value 0 (typical happy-path scenario)
- Match CLAUDE.md tree file-for-file but keep README's shorter descriptions (no interface names)
- Keep `sim/internal/testutil/` out of README tree (internal test infrastructure, not relevant for user/contributor orientation)

### G) Task Breakdown

#### Task 1: Fix conservation formula and example JSON (BC-1, BC-2)

**Files:**
- Modify: `README.md:449-484`

**Step 1: Update example JSON to include missing fields**

In `README.md`, in the Example Output JSON block, add a trailing comma to `scheduling_delay_p99_ms` (currently the last field) and add `preemption_count` and `dropped_unservable`:

```json
  "scheduling_delay_p99_ms": 11.27,
  "preemption_count": 0,
  "dropped_unservable": 0
```

Note: `kv_allocation_failures` (between `scheduling_delay_p99_ms` and `preemption_count` in the struct) uses `omitempty` and is correctly absent from the zero-value example.

**Step 2: Fix conservation formula text**

In `README.md:484`, change:
```
- **Conservation fields**: `still_queued`, `still_running`, and `injected_requests` verify request conservation (`injected == completed + still_queued + still_running`)
```
to:
```
- **Conservation fields**: `still_queued`, `still_running`, `dropped_unservable`, and `injected_requests` verify request conservation (`injected == completed + still_queued + still_running + dropped_unservable`). See [INV-1](docs/standards/invariants.md).
```

**Step 3: Verify by visual inspection**

Run: `grep -n 'dropped_unservable' README.md`
Expected: 2 matches (JSON block + conservation text)

Run: `grep -n 'injected == completed' README.md`
Expected: 1 match with the corrected 4-term formula

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: fix conservation formula and example JSON in README

- Add dropped_unservable and preemption_count to example JSON output
- Fix conservation formula to match INV-1 (was missing dropped_unservable)
- Add cross-reference to docs/standards/invariants.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 2: Fix Project Structure tree (BC-3, BC-4)

**Files:**
- Modify: `README.md:657-731`

**Step 1: Update sim/ section to add missing files and fix descriptions**

In the `sim/` section of the Project Structure tree, add missing files and fix descriptions to match CLAUDE.md. File ordering within sections is not changed (README uses a pre-existing different order). Changes:

1. Add `config.go` — "Module-scoped sub-config types (R16)"
2. Add `doc.go` — "Package reading guide"
3. Fix `batch.go` description: "Batch formation" → "Batch struct"
4. Add `batch_formation.go` — "BatchFormation interface, VLLMBatchFormation"
5. Add `latency_model.go` — "LatencyModel interface and registration"
6. Add `metrics_utils.go` — "MetricsOutput JSON struct, percentile calculations"
7. Fix `event.go` description: add "RequestLeft" to the event type list
8. Fix `distribution.go` description in `sim/workload/` section: add "Constant" to the sampler list (CLAUDE.md has "Gaussian, Exponential, ParetoLogNormal, EmpiricalPDF, Constant")

**Step 2: Update sim/cluster/ section to add missing file**

Add `deployment.go` after `metrics.go` and before `workload.go` (matching CLAUDE.md ordering) — "DeploymentConfig (embeds SimConfig + cluster fields)"

**Step 3: Update sim/workload/ section to add missing file**

Add `inference_perf.go` — "inference-perf format loading and validation"

**Step 4: Verify tree matches CLAUDE.md**

Run: `grep -c '│' README.md` to count tree lines (should increase by ~8)

Spot-check: `grep 'config.go' README.md` — should find the new entry
Spot-check: `grep 'batch_formation.go' README.md` — should find the new entry
Spot-check: `grep 'deployment.go' README.md` — should find the new entry
Spot-check: `grep 'RequestLeft' README.md` — should find the updated event.go description

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: sync README Project Structure tree with CLAUDE.md

- Add 7 missing files: config.go, doc.go, batch_formation.go,
  latency_model.go, metrics_utils.go, deployment.go, inference_perf.go
- Fix batch.go description (Batch struct, not Batch formation)
- Fix event.go description (add RequestLeft event type)
- Fix distribution.go description (add Constant sampler)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Verification |
|----------|------|-----------|--------------|
| BC-1 | Task 1 | Manual | grep for 4-term formula |
| BC-2 | Task 1 | Manual | grep for dropped_unservable in JSON |
| BC-3 | Task 2 | Manual | grep for each added file |
| BC-4 | Task 2 | Manual | grep for corrected descriptions |
| BC-5 | Both | Manual | git diff shows only README.md changes |

No code changes → no `go test` or `golangci-lint` needed. Verification is textual.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Tree misalignment with CLAUDE.md | Low | Medium | Line-by-line comparison after edit |
| Markdown formatting broken | Low | Low | Preview rendering before commit |

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No code changes
- [x] CLAUDE.md not modified (README is the working copy being fixed)
- [x] Documentation DRY: README working copy realigned with canonical sources
- [x] Deviation log reviewed — no deviations
- [x] Each task produces a clean commit

---

## Appendix: Exact Changes

### File: `README.md`

**Change 1 (Task 1):** Example Output JSON — add two fields before closing brace:
- `"preemption_count": 0,`
- `"dropped_unservable": 0`

**Change 2 (Task 1):** Conservation formula text at line 484 — add `dropped_unservable` to formula and field list, add invariants.md cross-reference.

**Change 3 (Task 2):** Project Structure tree `sim/` section — add 5 files, fix 3 descriptions. Note: file ordering within sections is NOT changed in this PR (README already uses a different order from CLAUDE.md; reordering is a separate task). Insertions use README-relative positions:
- Before `simulator.go`: add `config.go` and `doc.go` (matching CLAUDE.md where these precede simulator.go)
- Change `batch.go` description from "Batch formation" to "Batch struct"
- After `batch.go`: add `batch_formation.go`
- After `scheduler.go`: add `latency_model.go` (matching CLAUDE.md where it follows scheduler.go)
- After `metrics.go`: add `metrics_utils.go`
- Change `event.go` description to include "RequestLeft"

**Change 4 (Task 2):** Project Structure tree `sim/cluster/` section — add `deployment.go` after `metrics.go`, before `workload.go` (matching CLAUDE.md ordering).

**Change 5 (Task 2):** Project Structure tree `sim/workload/` section — add `inference_perf.go` after `network.go`, and fix `distribution.go` description to include `Constant`.
