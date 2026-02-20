# fix(sim): Remove Stale Comments — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove two stale code comments that reference work already completed, eliminating misleading information for contributors.

**The problem today:** Two comments in `sim/` reference future work that has already been completed. A contributor reading `sim/priority.go` sees "Full SLO class integration (using TenantState) is deferred to a future PR" — but PR10 already implemented SLO class integration via `SLOClass` field on `Request` (not via `TenantState`). Similarly, `sim/queue.go` has a TODO for preemption re-queuing that's already implemented in `sim/simulator.go:382`.

**What this PR adds:** Accurate comments that reflect the current state of the codebase — no new functionality, pure housekeeping.

**Why this matters:** Stale comments mislead contributors and create confusion about what's implemented vs. what's planned. This is antipattern prevention (CLAUDE.md: "always grep for 'planned for PR N' after completing PR N").

**Architecture:** Comment-only changes in `sim/priority.go` and `sim/queue.go`. No behavioral changes, no new types, no interface modifications.

**Source:** GitHub issue #240

**Closes:** Fixes #240

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR removes two stale comments in the `sim/` package:
1. `sim/priority.go:26` — references abandoned `TenantState` concept; replaced with accurate `SLOClass` reference
2. `sim/queue.go:4` — TODO for preemption re-queuing already implemented at `sim/simulator.go:382`

No code behavior changes. No deviations from the issue description. The issue also asked about `*float64` for workload spec fields — investigation confirms no change is needed (zero is always an error, not ambiguous).

### B) Behavioral Contracts

**BC-1: No Regression**
- GIVEN the codebase passes all tests before this change
- WHEN stale comments are updated/removed
- THEN all existing tests MUST continue to pass with identical output
- MECHANISM: Comment-only changes cannot affect runtime behavior

**BC-2: Accurate Priority Comment**
- GIVEN `sim/priority.go` contains a comment about SLO class integration
- WHEN a contributor reads the `SLOBasedPriority` doc comment
- THEN the comment MUST accurately describe the current state (SLOClass exists but is not used by this scorer)
- MECHANISM: Replace stale TenantState reference with accurate SLOClass availability note

**BC-3: No Stale TODO**
- GIVEN `sim/queue.go` contains a TODO for preemption re-queuing
- WHEN the TODO references work already completed
- THEN the stale TODO MUST be removed entirely (not replaced)
- MECHANISM: Delete the line; preemption re-queuing is implemented at `sim/simulator.go:382`

### C) Component Interaction

No new components. No state changes. No API changes. This PR modifies only comments in two existing files.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Check `spec.go:19,30` for `*float64` | No change | SCOPE_CHANGE: `AggregateRate` and `RateFraction` are validated `> 0` at load time; zero is always an error, not an ambiguous valid value. `*float64` is only needed when zero is a legitimate user-provided value. |

### E) Review Guide

1. **The tricky part:** There is none — this is a pure comment cleanup.
2. **What to scrutinize:** Verify the replacement comment on `SLOBasedPriority` is factually accurate (SLOClass is set by workload generator, not by TenantState).
3. **What's safe to skim:** Everything — comment-only changes.
4. **Known debt:** None.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/priority.go:26` — Replace stale TenantState comment
- `sim/queue.go:4` — Delete stale preemption TODO

**Key decisions:** No `*float64` change for `spec.go` (zero is always invalid, not ambiguous).

**Confirmation:** No dead code, no new code, all changes are comment-only.

### G) Task Breakdown

#### Task 1: Update stale comment in sim/priority.go

**Contracts Implemented:** BC-1, BC-2

**Files:**
- Modify: `sim/priority.go:26`

**Step 1: Update stale comment**

Context: Line 26 says "Full SLO class integration (using TenantState) is deferred to a future PR." PR10 added SLOClass directly on Request. TenantState was abandoned.

In `sim/priority.go`, replace line 26:
```
// Full SLO class integration (using TenantState) is deferred to a future PR.
```
with:
```
// Per-request SLO metadata is available on Request.SLOClass but not yet used by this scorer.
```

**Step 2: Run tests to verify no regression**

Run: `go test ./sim/... -v -count=1 2>&1 | tail -20`
Expected: PASS (all tests pass, identical behavior)

**Step 3: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 4: Commit**

```bash
git add sim/priority.go
git commit -m "fix(sim): update stale SLOBasedPriority comment (BC-2)

Replace outdated TenantState reference with accurate SLOClass
field reference. PR10 implemented SLO class integration via
the SLOClass field on Request, not via TenantState.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 2: Remove stale TODO in sim/queue.go

**Contracts Implemented:** BC-1, BC-3

**Files:**
- Modify: `sim/queue.go:4`

**Step 1: Remove stale TODO**

Context: Line 4 says "TODO: Requests need to be re-queued on preemption." This is already implemented at `sim/simulator.go:382` via `sim.WaitQ.queue = append([]*Request{preemptedRequest}, sim.WaitQ.queue...)`.

In `sim/queue.go`, delete line 4:
```
// TODO: Requests need to be re-queued on preemption.
```

Also delete the preceding blank line (line 3) to avoid a double-blank between the file comment and the package declaration.

**Step 2: Run tests to verify no regression**

Run: `go test ./sim/... -v -count=1 2>&1 | tail -20`
Expected: PASS (all tests pass, identical behavior)

**Step 3: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 4: Commit**

```bash
git add sim/queue.go
git commit -m "fix(sim): remove stale preemption re-queuing TODO (BC-3)

Preemption re-queuing is already implemented in simulator.go:382
via WaitQ.queue prepend. The TODO was stale since the preemption
event handler was added.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|--------------------------|
| BC-1 | Task 1, Task 2 | Regression | All existing `go test ./sim/...` tests pass |
| BC-2 | Task 1 | Manual | Read updated comment for accuracy |
| BC-3 | Task 2 | Manual | Verify TODO line is deleted |

No new tests needed — comment-only changes. No golden dataset updates.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Accidental code deletion | Low | High | Only touch comment lines, verify via `git diff` |
| Inaccurate replacement comment | Low | Low | Verified SLOClass field exists on Request (sim/request.go) |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes
- [x] No hidden global state impact
- [x] All code will pass golangci-lint
- [x] N/A — shared test helpers (no new tests)
- [x] CLAUDE.md — no update needed (no new files/packages/flags)
- [x] No stale references left in CLAUDE.md
- [x] Deviation log reviewed — spec.go `*float64` deviation justified
- [x] Each task produces working code (verified by `go test`)
- [x] Task dependencies correctly ordered (Task 2 independent of Task 1)
- [x] All contracts mapped to tasks
- [x] N/A — golden dataset unchanged
- [x] N/A — no struct construction sites modified
- [x] N/A — no new CLI flags
- [x] N/A — no error paths modified
- [x] N/A — no map iteration modified
- [x] N/A — no logrus.Fatalf in library code
- [x] N/A — no resource allocation loops modified
- [x] N/A — no exported mutable maps
- [x] N/A — no YAML config structs modified
- [x] N/A — no YAML loading modified
- [x] N/A — no division operations modified
- [x] N/A — no new interfaces
- [x] N/A — no methods spanning multiple concerns
- [x] N/A — no configuration parameters added
- [x] Grepped for "planned for PR" — these two are the only stale references found
- [x] N/A — not part of a macro plan

---

## Appendix: File-Level Implementation Details

### File: `sim/priority.go`

**Purpose:** Replace stale comment on line 26.

**Change:** Single line replacement in the `SLOBasedPriority` doc comment block.

Before (line 26):
```go
// Full SLO class integration (using TenantState) is deferred to a future PR.
```

After:
```go
// Per-request SLO metadata is available on Request.SLOClass but not yet used by this scorer.
```

### File: `sim/queue.go`

**Purpose:** Delete stale TODO on line 4.

**Change:** Remove lines 3-4 (blank line + stale TODO comment).

Before (lines 1-6):
```go
// Implements the WaitQueue, which holds all requests waiting to be processed.
// Requests are enqueued on arrival

// TODO: Requests need to be re-queued on preemption.

package sim
```

After (lines 1-4):
```go
// Implements the WaitQueue, which holds all requests waiting to be processed.
// Requests are enqueued on arrival

package sim
```
