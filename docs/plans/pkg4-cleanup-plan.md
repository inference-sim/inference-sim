# PKG-4: Documentation Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update all documentation to reflect the KV cache and latency model package extractions completed in PKG-1 and PKG-2.

**The problem today:** After extracting `sim/kv/` (PKG-1, #421) and `sim/latency/` (PKG-2, #424), several documentation files still reference the old file locations (`sim/kvcache.go`, `sim/kvcache_tiered.go`). Contributors following extension recipes would be directed to non-existent files. The `sim/` package also lacks a package-level doc comment to orient new readers.

**What this PR adds:**
1. Accurate file organization trees — CLAUDE.md and README.md reflect actual directory structure
2. Correct extension recipe paths — KV tier extension guide points to `sim/kv/` files
3. Package reading guide — `sim/doc.go` tells new readers where to start

**Why this matters:** Documentation accuracy is a prerequisite for contributor onboarding. Stale paths create confusion and waste time.

**Architecture:** Documentation-only changes. No code modifications, no behavioral changes. One `sim/doc.go` file adds a Go package reading guide.

**Source:** GitHub issue #412

**Closes:** Fixes #412

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR removes stale file references left behind by PKG-1 (#421, `sim/kv/` extraction) and PKG-2 (#424, `sim/latency/` extraction). Six documentation files need path updates: CLAUDE.md, README.md, `docs/extension-recipes.md`, `docs/standards/experiments.md`, `.github/ISSUE_TEMPLATE/feature_request.md`, and `docs/process/hypothesis.md`. One historical note is added to `hypotheses/README.md` for KV file moves. One new file (`sim/doc.go`) provides a package reading guide.

No code changes. No new dependencies. No behavioral changes.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Zero behavioral change
- GIVEN the full test suite and determinism check
- WHEN this PR is applied
- THEN all tests pass, lint is clean, and deterministic output is byte-identical to pre-PR output
- MECHANISM: Only documentation and doc.go files are modified

BC-2: CLAUDE.md file tree accuracy
- GIVEN the CLAUDE.md File Organization tree
- WHEN a reader looks up any `sim/` file listed in the tree
- THEN the file MUST exist at the listed path, and no files that exist in `sim/` are missing from the tree (excluding test files)
- MECHANISM: Remove stale `kvcache.go` and `kvcache_tiered.go` entries

BC-3: Extension recipe path accuracy
- GIVEN the KV cache extension recipe in `docs/extension-recipes.md`
- WHEN a contributor follows the recipe's file path references
- THEN every referenced file path MUST point to a file that exists
- MECHANISM: Update `sim/kvcache*.go` references to `sim/kv/*.go`

**Negative Contracts:**

BC-4: No stale kvcache references in active docs
- GIVEN the set of active documentation files (excluding `docs/plans/` which are historical snapshots, `docs/plans/archive/`, hypothesis FINDINGS.md, and `learn/research/`)
- WHEN grepping for `sim/kvcache` path patterns
- THEN zero matches MUST be found
- MECHANISM: All references in active docs updated or removed; `.github/ISSUE_TEMPLATE/` included in scope

### C) Component Interaction

No component interaction changes. This is a documentation-only PR.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|---|---|---|
| Create `sim/kv/doc.go` | Skip — already has package doc in `cache.go:1-5` | SIMPLIFICATION: Go convention is one package doc per package; `cache.go` already has `// Package kv implements...` |
| Create `sim/latency/doc.go` | Skip — already has package doc in `latency.go:1-4` | SIMPLIFICATION: Same — `latency.go` already has `// Package latency provides...` |
| Update design-guidelines.md | Skip | CORRECTION: Already current (uses interface names, not file paths) |
| Update `docs/plans/research.md` stale refs | Skip | SIMPLIFICATION: Hypothesis experiment descriptions in research.md are historical (completed experiments); same rationale as hypothesis FINDINGS.md |
| Update `docs/plans/2026-02-22-batch-formation-extraction-design.md` stale refs | Skip | SIMPLIFICATION: Design doc for #242 (batch formation extraction, completed in #371). Historical snapshot. |

### E) Review Guide

- **The tricky part:** Ensuring no stale references remain after edits. The grep verification in Task 4 is the key check.
- **What to scrutinize:** BC-4 (stale reference elimination) — verify the grep in Task 4 catches all variants.
- **What's safe to skim:** The doc.go content (Task 2), README tree edits (Task 1) — these are mechanical.
- **Known debt:** `learn/research/` has stale file refs — intentionally out of scope (historical research notes). `docs/plans/research.md` and `docs/plans/2026-02-22-batch-formation-extraction-design.md` have stale refs — also historical snapshots (see deviation log).

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `CLAUDE.md` — Remove 2 stale lines from File Organization tree
- `README.md` — Remove 2 stale lines from Project Structure tree
- `docs/extension-recipes.md` — Update 6 KV file path references
- `docs/standards/experiments.md` — Update 1 stale `kvcache_tiered.go` reference
- `hypotheses/README.md` — Add KV file move note alongside existing latency note
- `.github/ISSUE_TEMPLATE/feature_request.md` — Update KV cache checkbox path
- `docs/process/hypothesis.md` — Update pedagogical example with stale filename

**Files to create:**
- `sim/doc.go` — Package reading guide

**Key decisions:**
- Skip `sim/kv/doc.go` and `sim/latency/doc.go` (already have package comments)
- Skip `learn/research/` stale refs (historical, out of scope)

### G) Task Breakdown

---

### Task 1: Fix file organization trees in CLAUDE.md and README.md

**Contracts Implemented:** BC-2

**Files:**
- Modify: `CLAUDE.md:209,211` (remove stale kvcache entries)
- Modify: `README.md:676,678` (remove stale kvcache entries)

**Step 1: Remove stale entries and add doc.go to CLAUDE.md**

In `CLAUDE.md`, remove lines 209 and 211 (the `kvcache.go` and `kvcache_tiered.go` entries). These files now live in `sim/kv/cache.go` and `sim/kv/tiered.go` and are already correctly listed under the `sim/kv/` section at lines 221-224.

Remove:
```
│   ├── kvcache.go             # Block-based KV cache with LRU eviction and prefix caching, CacheHits/CacheMisses counters, transactional AllocateKVBlocks with rollbackAllocation on mid-loop failure
```
and:
```
│   ├── kvcache_tiered.go      # TieredKVCache (GPU+CPU composition), cpuTier, offloadedBlock, offload/reload/transfer latency, PendingTransferLatency() (pure query), ConsumePendingTransferLatency() (read-and-clear)
```

Also add `doc.go` to the `sim/` section (after `config.go`):
```
│   ├── doc.go                 # Package reading guide: start with request.go, event.go, simulator.go
```

This satisfies BC-2 (every file in `sim/` listed in the tree).

**Step 2: Remove stale entries and fix description in README.md**

In `README.md`, remove lines 676 and 678. These are already correctly listed under `sim/kv/` at lines 686-689.

Remove:
```
│   ├── kvcache.go          # KV cache modeling (single-tier, implements KVStore)
```
and:
```
│   ├── kvcache_tiered.go   # TieredKVCache: GPU+CPU offload/reload, transfer latency
```

Also update line 677 — after the removals, the `kv_store.go` description will remain. Its current description says "KVStore interface and NewKVStore factory" but the `NewKVStore` factory moved to `sim/kv/register.go` in PKG-1. Update to:
```
│   ├── kv_store.go         # KVStore interface and registration variables
```

**Step 3: Verify trees match disk**

Run: `ls .worktrees/pkg4-cleanup/sim/kvcache.go .worktrees/pkg4-cleanup/sim/kvcache_tiered.go 2>&1`
Expected: Both files should NOT exist (error output confirms they were moved).

Run: `ls .worktrees/pkg4-cleanup/sim/kv/cache.go .worktrees/pkg4-cleanup/sim/kv/tiered.go`
Expected: Both files exist.

**Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: remove stale kvcache.go entries from file trees (BC-2)

Files moved to sim/kv/ in PKG-1 (#421) but old entries remained
in CLAUDE.md and README.md file organization trees.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Create sim/doc.go reading guide

**Contracts Implemented:** BC-1 (zero behavioral change — doc.go only)

**Files:**
- Create: `sim/doc.go`

**Step 1: Create the file**

```go
// Package sim provides the core discrete-event simulation engine for BLIS.
//
// # Reading Guide
//
// Start with these three files to understand the simulation kernel:
//   - request.go: Request lifecycle (queued → running → completed) and state machine
//   - event.go: Event types that drive the simulation (Arrival, Step, Scheduled, etc.)
//   - simulator.go: The event loop, batch formation, and step execution
//
// # Architecture
//
// The sim package defines interfaces and bridge types; implementations live in
// sub-packages:
//   - sim/kv/: KV cache implementations (single-tier GPU, tiered GPU+CPU)
//   - sim/latency/: Latency models (blackbox alpha/beta, roofline FLOPs/bandwidth)
//   - sim/cluster/: Multi-instance cluster orchestration
//   - sim/workload/: Workload generation and trace replay
//   - sim/trace/: Decision trace recording
//
// Sub-packages register their implementations via init() functions that set
// package-level factory variables (NewLatencyModelFunc, NewKVStoreFromConfig).
//
// # Key Interfaces
//
// The extension points are single-method or small interfaces:
//   - LatencyModel: step time, queueing time, scheduling/preemption/output processing overheads
//   - KVStore: block allocation, eviction, prefix caching, capacity queries
//   - RoutingPolicy: select target instance given cluster snapshots
//   - AdmissionPolicy: accept or reject incoming requests
//   - InstanceScheduler: order requests within a single instance's wait queue
//   - PriorityPolicy: compute priority scores for scheduling
//   - BatchFormation: form batches from waiting requests with KV constraints
//
// See docs/extension-recipes.md for step-by-step guides to extend each interface.
package sim
```

**Step 2: Verify build**

Run: `cd .worktrees/pkg4-cleanup && go build ./sim/...`
Expected: Build succeeds.

**Step 3: Verify godoc renders**

Run: `cd .worktrees/pkg4-cleanup && go doc ./sim/ | head -20`
Expected: Package comment appears.

**Step 4: Commit**

```bash
git add sim/doc.go
git commit -m "docs(sim): add package reading guide (sim/doc.go)

Provides entry points for new readers: request.go → event.go → simulator.go.
Documents sub-package architecture and key interfaces.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Update extension-recipes.md KV paths

**Contracts Implemented:** BC-3

**Files:**
- Modify: `docs/extension-recipes.md:53,55,58,59,63,64`

**Step 1: Update file path references**

Six changes in the "Extending KV Cache Tiers" section:

1. Line 53: `sim/kvcache_*.go` → `sim/kv/` (step 1 description)
2. Line 55: `sim/kv_store.go` → `sim/kv/register.go` (step 3 — `NewKVStore` factory moved to `sim/kv/register.go` in PKG-1)
3. Line 58: `sim/kvcache_*_test.go` → `sim/kv/*_test.go` (step 6)
4. Line 59: `sim/kvcache.go` → `sim/kv/cache.go` (step 7 rollback reference)
5. Line 63: `sim/kvcache_tiered.go` → `sim/kv/tiered.go` (example)
6. Line 64: `sim/kvcache.go` → `sim/kv/cache.go` (example)

After edits, the section should read:

Line 53: `1. **Implement the `KVStore` interface** in `sim/kv/` (11 methods: allocate, get cached, release, capacity queries, metrics, `SetClock`, `ConsumePendingTransferLatency`)`

Line 55: `3. **Update `NewKVStore` factory** in `sim/kv/register.go` to instantiate your tier based on `KVCacheConfig` fields (add new fields to `KVCacheConfig` in `sim/config.go`)`

Line 58: `6. **Add behavioral tests** in `sim/kv/*_test.go``

Line 59: `7. **Preserve rollback semantics** — `KVCacheState.AllocateKVBlocks` is transactional: on mid-loop failure, `rollbackAllocation()` undoes all mutations (UsedBlockCnt, CacheMisses, CacheHits, RefCount, InUse, free list, HashToBlock, RequestMap). If your tier adds mutations beyond what delegation to `gpu.AllocateKVBlocks()` handles, you must roll those back too. See `cachedBlockMutation` and `newBlockMutation` types in `sim/kv/cache.go`.`

Line 63: `- See `TieredKVCache` in `sim/kv/tiered.go` for 2-tier GPU+CPU composition`

Line 64: `- See `KVCacheState` in `sim/kv/cache.go` for single-tier baseline (also implements `KVStore`)`

**Step 2: Verify no stale kvcache references remain in extension-recipes.md**

Run: `grep -n 'sim/kvcache' .worktrees/pkg4-cleanup/docs/extension-recipes.md`
Expected: No matches.

**Step 3: Commit**

```bash
git add docs/extension-recipes.md
git commit -m "docs: update KV extension recipe paths to sim/kv/ (BC-3)

Files moved from sim/kvcache*.go to sim/kv/ in PKG-1 (#421).
Updated 6 file path references in the KV tier extension recipe.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Fix stale references in experiments.md, hypotheses/README.md, and issue template

**Contracts Implemented:** BC-4

**Files:**
- Modify: `docs/standards/experiments.md:268`
- Modify: `hypotheses/README.md:19-21`
- Modify: `.github/ISSUE_TEMPLATE/feature_request.md:23`
- Modify: `docs/process/hypothesis.md:624` (pedagogical example with stale filename)

**Step 1: Update experiments.md**

Line 268 currently reads:
```
Evidence: H10 claimed "CPU tier increases total effective capacity" — but `NewKVStore` (`kv_store.go:31-36`) does not change GPU block count. The actual mechanism was `maybeOffload` preserving prefix hashes (`kvcache_tiered.go:214-224`).
```

Update to:
```
Evidence: H10 claimed "CPU tier increases total effective capacity" — but `NewKVStore` (`kv_store.go:31-36`) does not change GPU block count. The actual mechanism was `maybeOffload` preserving prefix hashes (`sim/kv/tiered.go`, formerly `kvcache_tiered.go`).
```

Note: We keep the original line numbers since they refer to historical code state at the time of the experiment. We add the current file path for findability.

**Step 2: Update hypotheses/README.md**

The existing note at lines 19-21 only covers latency file moves. Add KV file moves.

Replace:
```
## Note on File Path References

FINDINGS.md files in individual hypothesis directories may reference `sim/latency_model.go` and `sim/roofline_step.go`. These files were moved to `sim/latency/latency.go` and `sim/latency/roofline.go` respectively in PKG-2 (#406). The file paths in FINDINGS.md reflect the codebase state at the time each experiment was conducted.
```

With:
```
## Note on File Path References

FINDINGS.md files in individual hypothesis directories may reference old file paths. These files were moved during package extraction and the FINDINGS.md paths reflect the codebase state at the time each experiment was conducted:

- `sim/kvcache.go` → `sim/kv/cache.go` (PKG-1, #421)
- `sim/kvcache_tiered.go` → `sim/kv/tiered.go` (PKG-1, #421)
- `sim/latency_model.go` → `sim/latency/latency.go` (PKG-2, #424)
- `sim/roofline_step.go` → `sim/latency/roofline.go` (PKG-2, #424)
```

**Step 3: Update issue template**

In `.github/ISSUE_TEMPLATE/feature_request.md`, line 23 currently reads:
```
- [ ] KV cache (`sim/kvcache*.go`)
```

Update to:
```
- [ ] KV cache (`sim/kv/`)
```

**Step 4: Update hypothesis.md pedagogical example**

In `docs/process/hypothesis.md`, line 624 contains a "bad example" in a table:
```
| **Conceptual** | "Tiered storage should reduce preemptions" | "kvcache_tiered.go:224 should delete the hash" |
```

Update the bad example to use the current filename (keeps the pedagogical purpose — implementation-specific hypotheses are still "bad"):
```
| **Conceptual** | "Tiered storage should reduce preemptions" | "tiered.go:224 should delete the hash" |
```

**Step 5: Verify no stale kvcache references in active docs**

Run a comprehensive grep excluding historical files (plans, archived plans, hypothesis FINDINGS).
Two patterns: one with `sim/` prefix, one bare filename:
```bash
cd .worktrees/pkg4-cleanup && grep -rn -e 'sim/kvcache' -e 'kvcache_tiered\.go' -e 'kvcache\.go' --include='*.md' \
  --exclude-dir='archive' \
  --exclude-dir='hypotheses' \
  --exclude-dir='plans' \
  docs/ CLAUDE.md README.md CONTRIBUTING.md .github/
```
Expected: No matches. (The broader pattern catches bare filename references like `kvcache_tiered.go:214` that lack the `sim/` prefix.)

**Step 6: Commit**

```bash
git add docs/standards/experiments.md hypotheses/README.md .github/ISSUE_TEMPLATE/feature_request.md docs/process/hypothesis.md
git commit -m "docs: fix stale kvcache refs in experiments.md, hypotheses/README, issue template (BC-4)

- Update experiments.md RCV-1 evidence to reference sim/kv/tiered.go
- Expand hypotheses/README.md file path note to cover KV moves from PKG-1
- Update feature_request.md checklist path to sim/kv/
- Update hypothesis.md pedagogical example to use current filename

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Final verification

**Contracts Implemented:** BC-1 (zero behavioral change)

**Step 1: Build**

Run: `cd .worktrees/pkg4-cleanup && go build ./...`
Expected: Success.

**Step 2: Test**

Run: `cd .worktrees/pkg4-cleanup && go test ./...`
Expected: All tests pass.

**Step 3: Lint**

Run: `cd .worktrees/pkg4-cleanup && golangci-lint run ./...`
Expected: No new issues.

**Step 4: Determinism check**

```bash
cd .worktrees/pkg4-cleanup
go build -o simulation_worker main.go
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --seed 42 --num-requests 100 > /tmp/run1.json 2>/dev/null
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --seed 42 --num-requests 100 > /tmp/run2.json 2>/dev/null
diff /tmp/run1.json /tmp/run2.json
```
Expected: No diff (byte-identical).

---

### H) Test Strategy

| Contract | Task | Test Type | Verification |
|---|---|---|---|
| BC-1 | Task 5 | Build + Test + Lint + Determinism | Full suite + diff check |
| BC-2 | Task 1 | Manual | `ls` confirms files exist/don't exist at listed paths |
| BC-3 | Task 3 | Grep | Zero matches for `sim/kvcache` in extension-recipes.md |
| BC-4 | Task 4 | Grep | Zero matches for `sim/kvcache` in active docs |

No golden dataset changes. No new tests needed (documentation-only PR).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Missed stale reference | Low | Low | Comprehensive grep in Task 4 Step 3 |
| doc.go causes build issue | Very Low | Low | Build verification in Task 2 Step 2 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint (doc.go is comment-only)
- [x] CLAUDE.md updated (file tree corrected)
- [x] No stale references left in CLAUDE.md
- [x] Documentation DRY: extension-recipes.md (canonical source for extension guides) updated
- [x] Deviation log reviewed — 5 deviations documented (3 simplifications, 2 historical-snapshot exclusions)
- [x] Each task produces verifiable output
- [x] Task dependencies correctly ordered (1-4 independent, 5 depends on all)
- [x] All contracts mapped to tasks
- [x] No golden dataset changes needed
- [x] No construction site audit needed (no struct changes)
- [x] Not part of a macro plan (standalone cleanup)

---

## Appendix: File-Level Details

### File: `sim/doc.go` (CREATE)

See Task 2 Step 1 for complete content.

### File: `CLAUDE.md` (MODIFY)

Remove lines containing `kvcache.go` and `kvcache_tiered.go` from the `sim/` section of the File Organization tree (around lines 209, 211). The `sim/kv/` section at lines 221-224 already lists these correctly.

### File: `README.md` (MODIFY)

Remove lines containing `kvcache.go` and `kvcache_tiered.go` from the `sim/` section of the Project Structure tree (around lines 676, 678). The `sim/kv/` section at lines 686-689 already lists these correctly.

### File: `docs/extension-recipes.md` (MODIFY)

Update 6 file path references in the "Extending KV Cache Tiers" section. See Task 3 Step 1 for exact changes.

### File: `docs/standards/experiments.md` (MODIFY)

Update line 268 to reference `sim/kv/tiered.go` instead of bare `kvcache_tiered.go`.

### File: `hypotheses/README.md` (MODIFY)

Expand the "Note on File Path References" section to include KV file moves alongside existing latency file moves.

### File: `.github/ISSUE_TEMPLATE/feature_request.md` (MODIFY)

Update line 23 KV cache checkbox from `sim/kvcache*.go` to `sim/kv/`.
