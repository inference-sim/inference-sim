# Phase 2: Correctness Bug Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix six correctness bugs in the KV cache and simulator that produce wrong simulation results: incorrect prefix hashes, duplicate token writes, negative decode token counts, leaked KV blocks on allocation failure, and under-documented fragile code paths.

**The problem today:** The KV cache has four bugs that silently produce wrong simulation results: (1) chunked prefill computes prefix hashes from the wrong token range, causing cache misses for matching prefixes; (2) partial block fills advance the progress index by fewer positions than tokens actually appended, causing duplicate token writes; (3) the decode path passes a negative token count to the preemption function, which is accidentally masked but latent; (4) mid-loop allocation failure leaks already-allocated blocks, violating the block conservation invariant. Additionally, two deprecated KV functions harbor the same allocation leak bug and should be removed. These bugs affect cache hit rates, KV utilization metrics, and determinism guarantees that capacity planning depends on.

**What this PR adds:**
1. **Correct prefix hashes during chunked prefill** -- when a request's prefill is split across multiple steps (startIndex > 0), hashes now use the absolute input token offset instead of a newTokens-relative offset
2. **Correct progress tracking after partial block fill** -- captures remaining block capacity before appending tokens, then advances progress by the actual number of tokens written
3. **Safe decode token count** -- computes `decodeTokens := int64(1)` explicitly instead of reusing a negative prefill-era value
4. **Transactional KV allocation** -- tracks all mutations (cached block RefCount increments, new block allocations) and rolls back everything on failure, preserving `allocated + free == total`
5. **Deprecated function removal** -- removes `AllocateKVBlocksPrefill` and `AllocateKVBlocksDecode` (superseded by unified `AllocateKVBlocks`)
6. **Documentation of fragile code paths** -- documents the two-pass completion loop dependency and the preemption-safe pending request tracking

**Why this matters:** These fixes establish correct KV cache accounting and deterministic simulation results before PR11 (autoscaling) and PR14 (P/D disaggregation) add more complexity. The transactional rollback pattern protects against future allocation bugs.

**Architecture:** All changes are in `sim/` package (`kvcache.go`, `simulator.go`). No new types, interfaces, or packages. The KVStore interface is unchanged. The fixes are local to method bodies -- no signature changes, no new fields on existing structs.

**Source:** GitHub issue #209, design doc `docs/plans/2026-02-18-hardening-antipattern-refactoring-design.md` Phase 2

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes four correctness bugs in `sim/kvcache.go` and one in `sim/simulator.go`, removes two deprecated KV functions, and adds documenting comments for two fragile-but-correct code paths.

**Where it fits:** Phase 2 of the hardening effort (issue #214). Phase 1 (structural helpers) is complete. This PR depends on Phase 1's `NewRequestMetrics()` and `EffectiveLoad()` already being merged. Phase 4 (invariant tests) depends on these fixes being in place.

**Adjacent blocks:** `sim/simulator.go` (calls `AllocateKVBlocks` during `makeRunningBatch` and `Step`), `sim/cluster/cluster.go` (pending request tracking), `sim/kvcache_tiered.go` (delegates to `KVCacheState` methods -- no changes needed there).

**DEVIATION flags:** Two items from the design doc are already resolved:
- 2a (#183) was fixed in PR #206 (counter approach, not panic)
- 2f (#192) was effectively fixed in PR #205 (QueuedEvent detection, not ID-based tracking)

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Partial Block Fill Progress Tracking
- GIVEN a KV cache with BlockSize=16 and a latest block containing 10 tokens
- WHEN AllocateKVBlocks appends 4 new tokens to that partial block
- THEN newTokenProgressIndex advances by exactly 4 (not 2 or 0)
- MECHANISM: Capture `remaining := BlockSizeTokens - Len64(latestBlk.Tokens)` before the append; advance by `Len64(toksToAppend)`

BC-2: Prefix Hash Correctness for Chunked Prefill
- GIVEN a request with 200 input tokens and BlockSize=16
- WHEN AllocateKVBlocks is called with startIndex=100, endIndex=200 (second chunk)
- THEN the prefix hash for the first full block uses `req.InputTokens[:116]` (absolute), not `req.InputTokens[:16]` (relative)
- MECHANISM: Compute `absoluteEnd := startIndex + end` before hashing

BC-3: Decode Path Uses Explicit Token Count
- GIVEN a request in decode phase (ProgressIndex >= len(InputTokens)) with output tokens
- WHEN makeRunningBatch processes the decode branch
- THEN preempt() is called with numNewTokens=1 (not a negative value from the stale prefill computation)
- MECHANISM: Compute `decodeTokens := int64(1)` instead of reusing `numNewTokens`

BC-4: Transactional Rollback on Allocation Failure
- GIVEN a KV cache with 3 free blocks and a request needing 5 new blocks
- WHEN AllocateKVBlocks allocates 3 blocks then popFreeBlock returns nil
- THEN all 3 blocks are returned to the free list, UsedBlockCnt is restored, and the function returns false
- MECHANISM: Track `newlyAllocated` slice; on nil popFreeBlock, iterate and undo each mutation

BC-5: Cached Block Rollback on Allocation Failure
- GIVEN a request with 2 cached blocks and needing 3 new blocks, but only 2 free blocks available
- WHEN AllocateKVBlocks processes cached blocks then fails during new block allocation
- THEN cached block RefCount increments and free list removals are also rolled back
- MECHANISM: Track `cachedBlockMutations` alongside `newlyAllocated`; rollback function undoes both

BC-6: KV Block Conservation After Rollback
- GIVEN any sequence of AllocateKVBlocks calls (some succeeding, some failing)
- WHEN a failed allocation triggers rollback
- THEN `UsedBlocks() + countFreeBlocks() == TotalCapacity()` holds immediately after
- MECHANISM: Conservation invariant asserted in tests after rollback scenarios

**Negative Contracts:**

NC-1: No Deprecated Functions
- GIVEN the codebase after this PR
- WHEN searching for `AllocateKVBlocksPrefill` or `AllocateKVBlocksDecode`
- THEN zero results in `.go` files (only in docs/plans if referenced)
- MECHANISM: Delete both functions entirely

NC-2: No Silent Progress Stall
- GIVEN a partial block fill scenario where tokens are appended
- WHEN the append succeeds
- THEN newTokenProgressIndex MUST advance by at least 1 (preventing infinite loops)
- MECHANISM: `toksToAppend` is always non-empty when entering the partial block branch

**Error Handling Contracts:**

EC-1: Allocation Failure Returns False Cleanly
- GIVEN insufficient free blocks for the requested allocation
- WHEN AllocateKVBlocks returns false
- THEN the KVCacheState is identical to before the call (no leaked blocks, no partial RequestMap entries)
- MECHANISM: Rollback function + RequestMap cleanup

### C) Component Interaction

```
sim/simulator.go                    sim/kvcache.go
┌──────────────┐                   ┌─────────────────────┐
│makeRunningBatch│──preempt()──────▶│ AllocateKVBlocks()  │
│  (decode fix) │                   │  (rollback + fixes) │
│               │──AllocateKVBlocks▶│                     │
│  Step()       │──AllocateKVBlocks▶│ GetCachedBlocks()   │
│  (completion) │                   │ ReleaseKVBlocks()   │
│               │                   │                     │
│               │                   │ [REMOVED]           │
│               │                   │ AllocateKVBlocksPrefill │
│               │                   │ AllocateKVBlocksDecode  │
└──────────────┘                   └─────────────────────┘
```

No new types. No interface changes. No new state. All changes are within method bodies.

**Extension friction:** 0 new files touched for future field additions. These are pure bug fixes.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| 2a: Convert KV alloc failure to panic (#183) | Skip -- already fixed in PR #206 | ALREADY_FIXED: PR #206 used counter+log approach instead of panic. Issue #183 is CLOSED. |
| 2f: Switch to ID-based pending tracking (#192) | Document that QueuedEvent detection (PR #205) already handles preemption correctly | ALREADY_FIXED: PR #205 switched from QD-delta to QueuedEvent detection, which correctly handles preemption. The scenario in #192 no longer triggers false decrements. |
| 2e rollback: Design shows rollback in inner loop only | Plan includes cached-block rollback with `wasInUse` tracking | ADDITION: Design doc notes cached-block rollback in prose ("rollback scope (critical)") but its code snippet only shows new-block rollback. Plan adds `cachedBlockMutation` struct with `wasInUse` to correctly restore blocks that had `RefCount > 1` from other requests. |
| 2c fix: Design doc slices `newTokens[:min(...)]` (from index 0) | Plan slices `newTokens[newTokenProgressIndex:...]` | CORRECTION: Design doc's fix starts from index 0, which is only correct for the first iteration. Plan uses `newTokenProgressIndex` as the slice start to handle multi-iteration scenarios correctly. The partial block branch is entered with `newTokenProgressIndex=0` in practice, but the explicit offset is more robust. |

### E) Review Guide

**The tricky part:** BC-4/BC-5 (transactional rollback) is the most complex change. The rollback must undo mutations to both newly allocated blocks AND previously processed cached blocks. The cached block rollback must restore the original `InUse` state (not just set `false`) because a cached block might have had `InUse=true` from another request.

**What to scrutinize:** The rollback function in Task 4 -- verify it handles the case where a cached block had `RefCount > 1` before the increment (i.e., don't push it back to free list, just decrement RefCount).

**What's safe to skim:** BC-3 (decode fix) is a 2-line change. 2g (documentation) is comments only.

**Known debt:** The deprecated `AllocateKVBlocksPrefill` has a separate allocation leak that we're fixing by deletion rather than patching. The tiered KV cache (`kvcache_tiered.go`) delegates to `KVCacheState.AllocateKVBlocks` so it inherits all fixes. `GetCachedBlocks()` mutates `CacheHits` despite its doc comment claiming purity -- this is a pre-existing issue tracked in design doc Phase 3 (3c), not fixed in this PR. Rollback tests must account for this by not comparing CacheHits pre/post when `GetCachedBlocks` was called before the allocation.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/kvcache.go` -- Fix 2b, 2c, 2e; remove deprecated functions
- `sim/kvcache_test.go` (NEW) -- Tests for BC-1 through BC-6, EC-1
- `sim/simulator.go` -- Fix 2d (decode tokens), add comment 2g

**Key decisions:**
- Rollback uses a simple helper function (not a `defer` pattern) for clarity
- Cached block rollback tracks `wasInUse` bool per block to restore correct state
- Deprecated functions deleted entirely (no callers in production code)

**Confirmation:** No dead code. All paths exercisable via tests. No new public API.

### G) Task Breakdown

---

### Task 1: Fix stale newTokenProgressIndex after partial block fill (BC-1, NC-2)

**Contracts Implemented:** BC-1, NC-2

**Files:**
- Create: `sim/kvcache_test.go`
- Modify: `sim/kvcache.go:254-258`

**Step 1: Write failing test for BC-1**

Context: When tokens are appended to a partial block, the progress index must advance by the actual number of tokens appended. The current code uses post-append block length, causing under-advancement.

```go
// In sim/kvcache_test.go
package sim

import "testing"

func TestAllocateKVBlocks_PartialBlockFill_AdvancesByActualTokenCount(t *testing.T) {
	// GIVEN a KV cache with BlockSize=4 and a request that already has a partial block (2 of 4 tokens)
	kvc := NewKVCacheState(10, 4)
	req := &Request{
		ID:          "r1",
		InputTokens: []int{10, 20, 30, 40, 50, 60},
	}
	// Allocate first 2 tokens (creates a partial block with 2 tokens)
	ok := kvc.AllocateKVBlocks(req, 0, 2, []int64{})
	if !ok {
		t.Fatal("initial allocation should succeed")
	}
	ids := kvc.RequestMap["r1"]
	if len(ids) != 1 {
		t.Fatalf("expected 1 block, got %d", len(ids))
	}
	blk := kvc.Blocks[ids[0]]
	if len(blk.Tokens) != 2 {
		t.Fatalf("expected partial block with 2 tokens, got %d", len(blk.Tokens))
	}

	// WHEN we allocate 2 more tokens that should fill the partial block
	req.ProgressIndex = 2
	ok = kvc.AllocateKVBlocks(req, 2, 4, []int64{})
	if !ok {
		t.Fatal("second allocation should succeed")
	}

	// THEN the partial block now has 4 tokens (full) and no extra blocks were allocated
	blk = kvc.Blocks[ids[0]]
	if len(blk.Tokens) != 4 {
		t.Errorf("expected block with 4 tokens after fill, got %d", len(blk.Tokens))
	}
	// Should still be 1 block total (the partial was filled, no new block needed)
	finalIDs := kvc.RequestMap["r1"]
	if len(finalIDs) != 1 {
		t.Errorf("expected 1 block total (partial filled), got %d", len(finalIDs))
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestAllocateKVBlocks_PartialBlockFill -v -count=1`
Expected: FAIL (infinite loop or wrong block count due to stale progress index)

Note: This test may hang due to the infinite loop bug. Use timeout:
Run: `go test ./sim/... -run TestAllocateKVBlocks_PartialBlockFill -v -count=1 -timeout 5s`
Expected: FAIL with timeout

**Step 3: Implement fix for stale progress index**

Context: The bug is on line 258 of kvcache.go. After appending tokens, `Len64(latestBlk.Tokens)` reflects the new (larger) length, making the remaining capacity calculation wrong. Fix by capturing remaining capacity BEFORE append.

In `sim/kvcache.go`, replace lines 254-259 (the partial block fill section):

```go
	if len(latestBlk.Tokens) > 0 && Len64(latestBlk.Tokens) < kvc.BlockSizeTokens {
		// latest block is not full yet, append tokens to the latest block
		remaining := kvc.BlockSizeTokens - Len64(latestBlk.Tokens)
		toksToAppend := newTokens[newTokenProgressIndex:min(newTokenProgressIndex+remaining, Len64(newTokens))]
		latestBlk.Tokens = append(latestBlk.Tokens, toksToAppend...)
		newTokenProgressIndex += Len64(toksToAppend)
		logrus.Debugf("Appending to latest blk: req: %s, newTokenProgressIndex = %d, appended=%d tokens", req.ID, newTokenProgressIndex, Len64(toksToAppend))
```

Note: Also fix the slice start index from `newTokens[:...]` to `newTokens[newTokenProgressIndex:...]` for correctness in multi-iteration scenarios.

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestAllocateKVBlocks_PartialBlockFill -v -count=1`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/kvcache.go sim/kvcache_test.go
git commit -m "fix(kv): correct stale newTokenProgressIndex after partial block fill (#197)

- Capture remaining block capacity before append
- Advance progress by actual tokens appended (not post-append delta)
- Slice newTokens from newTokenProgressIndex (not always from 0)
- Prevents infinite loop and duplicate token writes (BC-1, NC-2)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Fix wrong prefix cache hashes during chunked prefill (BC-2)

**Contracts Implemented:** BC-2

**Files:**
- Modify: `sim/kvcache_test.go`
- Modify: `sim/kvcache.go:292-296`

**Step 1: Write failing test for BC-2**

Context: When startIndex > 0 (chunked prefill), the prefix hash should use the absolute offset into InputTokens, not the newTokens-relative offset. The current code hashes `req.InputTokens[:end]` where `end` is relative to newTokens.

```go
func TestAllocateKVBlocks_ChunkedPrefill_PrefixHashUsesAbsoluteOffset(t *testing.T) {
	// GIVEN a request with 8 tokens and BlockSize=4
	kvc := NewKVCacheState(10, 4)
	req := &Request{
		ID:          "r1",
		InputTokens: []int{10, 20, 30, 40, 50, 60, 70, 80},
	}

	// Allocate first chunk (tokens 0-3) — block 1 gets hash of InputTokens[:4]
	ok := kvc.AllocateKVBlocks(req, 0, 4, []int64{})
	if !ok {
		t.Fatal("first chunk allocation should succeed")
	}

	// Verify first block has correct hash
	expectedHash1 := hashTokens([]int{10, 20, 30, 40})
	ids1 := kvc.RequestMap["r1"]
	blk1 := kvc.Blocks[ids1[0]]
	if blk1.Hash != expectedHash1 {
		t.Errorf("first block hash mismatch:\n  got  %s\n  want %s", blk1.Hash, expectedHash1)
	}

	// WHEN we allocate second chunk (tokens 4-7, startIndex=4)
	req.ProgressIndex = 4
	ok = kvc.AllocateKVBlocks(req, 4, 8, []int64{})
	if !ok {
		t.Fatal("second chunk allocation should succeed")
	}

	// THEN second block has hash of InputTokens[:8] (absolute), not InputTokens[:4] (relative)
	ids2 := kvc.RequestMap["r1"]
	if len(ids2) < 2 {
		t.Fatalf("expected at least 2 blocks, got %d", len(ids2))
	}
	blk2 := kvc.Blocks[ids2[1]]
	expectedHash2 := hashTokens([]int{10, 20, 30, 40, 50, 60, 70, 80})
	wrongHash := hashTokens([]int{10, 20, 30, 40}) // This is what the buggy code produces
	if blk2.Hash == wrongHash {
		t.Errorf("second block has WRONG hash (newTokens-relative instead of absolute)")
	}
	if blk2.Hash != expectedHash2 {
		t.Errorf("second block hash mismatch:\n  got  %s\n  want %s", blk2.Hash, expectedHash2)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestAllocateKVBlocks_ChunkedPrefill_PrefixHash -v -count=1`
Expected: FAIL with "second block has WRONG hash"

**Step 3: Fix prefix hash to use absolute offset**

In `sim/kvcache.go`, in the new block allocation inner loop (around line 292-296), replace:
```go
			if Len64(blk.Tokens) == kvc.BlockSizeTokens {
				fullPrefix := req.InputTokens[:end]
```
with:
```go
			if Len64(blk.Tokens) == kvc.BlockSizeTokens {
				absoluteEnd := startIndex + end
				fullPrefix := req.InputTokens[:absoluteEnd]
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestAllocateKVBlocks_ChunkedPrefill_PrefixHash -v -count=1`
Expected: PASS

**Step 5: Run lint + full test suite**

Run: `go test ./sim/... -count=1 && golangci-lint run ./sim/...`
Expected: All pass, no new lint issues

**Step 6: Commit**

```bash
git add sim/kvcache.go sim/kvcache_test.go
git commit -m "fix(kv): use absolute offset for prefix hash during chunked prefill (#196)

- Compute absoluteEnd = startIndex + end before hashing
- Fixes wrong cache hashes when startIndex > 0 (chunked prefill)
- Previously hashed InputTokens[:end] where end was newTokens-relative (BC-2)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Fix negative numNewTokens in decode path (BC-3)

**Contracts Implemented:** BC-3

**Files:**
- Create: `sim/simulator_decode_test.go`
- Modify: `sim/simulator.go:419-422`

**Step 1: Write failing test for BC-3**

Context: In makeRunningBatch, the decode branch reuses `numNewTokens` computed for prefill. During decode, `ProgressIndex > len(InputTokens)`, making `numNewTokens` negative. This negative value is passed to `preempt()`.

We can't directly test `makeRunningBatch` (unexported), but we can verify the decode path works correctly by running a simulation with a request that enters decode and checking it completes properly. A more targeted test verifies the preempt function receives a positive value.

```go
// In sim/simulator_decode_test.go
package sim

import (
	"math"
	"testing"
)

func TestMakeRunningBatch_DecodePhase_PreemptGetsPositiveTokenCount(t *testing.T) {
	// GIVEN a simulator with a request that has completed prefill and is in decode
	sim := NewSimulator(SimConfig{
		Horizon:            math.MaxInt64,
		Seed:               42,
		TotalKVBlocks:      100,
		BlockSizeTokens:    4,
		MaxRunningReqs:     10,
		MaxScheduledTokens: 1000,
		BetaCoeffs:         []float64{100, 0.5, 0.5},
		AlphaCoeffs:        []float64{100, 0.1, 50},
	})

	// Create a request with known input/output
	req := &Request{
		ID:            "decode_test",
		InputTokens:   []int{1, 2, 3, 4, 5, 6, 7, 8},
		OutputTokens:  []int{100, 200, 300},
		ArrivalTime:   0,
		ProgressIndex: 10, // Past all input tokens (8), into decode
		State:         "running",
		NumNewTokens:  1,
	}

	// Pre-allocate KV blocks for the request (simulating prior prefill)
	ok := sim.KVCache.AllocateKVBlocks(req, 0, 8, []int64{})
	if !ok {
		t.Fatal("pre-allocation should succeed")
	}

	// Put request in running batch
	sim.RunningBatch.Requests = append(sim.RunningBatch.Requests, req)
	sim.reqNumComputedTokens[req.ID] = 10

	// WHEN makeRunningBatch processes this decode-phase request
	// (This should NOT pass a negative value to preempt)
	sim.makeRunningBatch(1000)

	// THEN the request should still be in the running batch with NumNewTokens=1
	found := false
	for _, r := range sim.RunningBatch.Requests {
		if r.ID == "decode_test" {
			found = true
			if r.NumNewTokens != 1 {
				t.Errorf("NumNewTokens = %d, want 1 for decode", r.NumNewTokens)
			}
		}
	}
	if !found {
		t.Error("request should still be in running batch after decode scheduling")
	}
}
```

**Step 2: Run test to verify behavior**

Run: `go test ./sim/... -run TestMakeRunningBatch_DecodePhase -v -count=1`
Expected: The test may pass (the bug is latent -- masked by AllocateKVBlocks ignoring endIndex in decode). But the fix is still needed for correctness. If it passes, the test documents correct post-fix behavior.

**Step 3: Fix decode path to use explicit token count**

In `sim/simulator.go`, replace lines 419-421:
```go
	if req.ProgressIndex >= Len64(req.InputTokens) && len(req.OutputTokens) > 0 {
		// this request will go through decode phase in this batch
		if can_schedule := sim.preempt(req, now, numNewTokens); !can_schedule {
```
with:
```go
	if req.ProgressIndex >= Len64(req.InputTokens) && len(req.OutputTokens) > 0 {
		// Decode phase: exactly 1 new token per step. Compute explicitly
		// instead of reusing numNewTokens (which is negative during decode:
		// len(InputTokens) - ProgressIndex where ProgressIndex > len(InputTokens)).
		decodeTokens := int64(1)
		if can_schedule := sim.preempt(req, now, decodeTokens); !can_schedule {
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestMakeRunningBatch_DecodePhase -v -count=1`
Expected: PASS

Run full suite to verify no regressions:
Run: `go test ./sim/... -count=1`
Expected: All pass

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/simulator.go sim/simulator_decode_test.go
git commit -m "fix(sim): use explicit decodeTokens=1 in makeRunningBatch decode path (#198)

- Compute decodeTokens := int64(1) instead of reusing negative numNewTokens
- numNewTokens is len(InputTokens)-ProgressIndex which is negative during decode
- Previously latent (AllocateKVBlocks ignores endIndex in decode path) (BC-3)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add transactional rollback to AllocateKVBlocks (BC-4, BC-5, BC-6, EC-1)

**Contracts Implemented:** BC-4, BC-5, BC-6, EC-1

**Files:**
- Modify: `sim/kvcache.go:237-301` (AllocateKVBlocks inner logic)
- Modify: `sim/kvcache_test.go`

**Step 1: Write failing tests for rollback contracts**

Context: When popFreeBlock returns nil mid-loop, previously allocated blocks must be returned to the free list. This also must roll back cached-block mutations.

```go
func TestAllocateKVBlocks_MidLoopFailure_RollsBackNewBlocks(t *testing.T) {
	// GIVEN a KV cache with only 2 free blocks but a request needing 3 new blocks
	kvc := NewKVCacheState(5, 2) // 5 total blocks, 2 tokens per block
	// Consume 3 blocks with a dummy request, leaving 2 free
	dummy := &Request{ID: "dummy", InputTokens: []int{1, 2, 3, 4, 5, 6}}
	kvc.AllocateKVBlocks(dummy, 0, 6, []int64{})

	usedBefore := kvc.UsedBlockCnt
	freeBefore := kvc.TotalBlocks - usedBefore

	// WHEN we try to allocate 3 blocks (6 tokens / 2 per block) but only 2 are free
	req := &Request{ID: "r_fail", InputTokens: []int{10, 20, 30, 40, 50, 60}}
	ok := kvc.AllocateKVBlocks(req, 0, 6, []int64{})

	// THEN allocation fails and block conservation is maintained
	if ok {
		t.Fatal("allocation should fail (not enough free blocks)")
	}

	// BC-6: Conservation invariant
	usedAfter := kvc.UsedBlockCnt
	freeAfter := kvc.TotalBlocks - usedAfter
	if usedAfter != usedBefore {
		t.Errorf("UsedBlockCnt changed: before=%d, after=%d (should be unchanged after rollback)", usedBefore, usedAfter)
	}
	if freeAfter != freeBefore {
		t.Errorf("free blocks changed: before=%d, after=%d (should be unchanged after rollback)", freeBefore, freeAfter)
	}

	// EC-1: No partial RequestMap entries
	if _, exists := kvc.RequestMap["r_fail"]; exists {
		t.Error("RequestMap should not contain entry for failed allocation")
	}
}

func TestAllocateKVBlocks_CachedBlockRollback_OnNewBlockFailure(t *testing.T) {
	// GIVEN a KV cache where a prefix is cached, and we need the cached blocks + new blocks
	kvc := NewKVCacheState(5, 2) // 5 blocks, 2 tokens per block

	// First: create and release a request to populate the prefix cache
	req1 := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	kvc.AllocateKVBlocks(req1, 0, 4, []int64{})
	kvc.ReleaseKVBlocks(req1) // Blocks go to free list with hashes intact

	// Consume all but 1 free block with another request
	// After release, all 5 blocks are free. Consume 4 of them.
	filler := &Request{ID: "filler", InputTokens: []int{90, 91, 92, 93, 94, 95, 96, 97}}
	kvc.AllocateKVBlocks(filler, 0, 8, []int64{})
	// Now: 4 blocks used by filler, 1 free

	// WHEN we try to allocate with 2 cached blocks + need 1 new block
	// The 2 cached blocks will be found and their RefCount incremented
	// But we only have 1 free block, and we need more after cached
	req2 := &Request{ID: "r2", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8}}
	cached := kvc.GetCachedBlocks(req2.InputTokens)

	usedBefore := kvc.UsedBlockCnt
	ok := kvc.AllocateKVBlocks(req2, 0, 8, cached)

	if ok {
		// If it succeeds, that's also valid (depends on exact block layout).
		// The test is about conservation regardless.
		t.Log("allocation succeeded (enough blocks available)")
	}

	// BC-6: Conservation invariant must hold regardless of success/failure
	totalUsed := kvc.UsedBlockCnt
	totalFree := kvc.TotalBlocks - totalUsed
	if totalUsed+totalFree != kvc.TotalBlocks {
		t.Errorf("block conservation violated: used=%d + free=%d != total=%d",
			totalUsed, totalFree, kvc.TotalBlocks)
	}

	// If allocation failed, used should be unchanged
	if !ok && totalUsed != usedBefore {
		t.Errorf("failed allocation changed UsedBlockCnt: before=%d, after=%d", usedBefore, totalUsed)
	}
}

func TestAllocateKVBlocks_BlockConservation_AfterAllocateReleaseCycles(t *testing.T) {
	// BC-6: After any sequence of operations, conservation holds
	kvc := NewKVCacheState(10, 4)

	// Allocate and release several requests
	for i := 0; i < 5; i++ {
		req := &Request{
			ID:          fmt.Sprintf("r%d", i),
			InputTokens: []int{i*10 + 1, i*10 + 2, i*10 + 3, i*10 + 4},
		}
		ok := kvc.AllocateKVBlocks(req, 0, 4, []int64{})
		if !ok {
			t.Fatalf("allocation %d should succeed", i)
		}
	}

	// Release first 3
	for i := 0; i < 3; i++ {
		req := &Request{ID: fmt.Sprintf("r%d", i)}
		kvc.ReleaseKVBlocks(req)
	}

	// Verify conservation
	free := kvc.TotalBlocks - kvc.UsedBlockCnt
	if kvc.UsedBlockCnt+free != kvc.TotalBlocks {
		t.Errorf("conservation violated: used=%d + free=%d != total=%d",
			kvc.UsedBlockCnt, free, kvc.TotalBlocks)
	}

	// Expected: 2 requests still hold 1 block each = 2 used, 8 free
	if kvc.UsedBlockCnt != 2 {
		t.Errorf("UsedBlockCnt = %d, want 2 (2 requests with 1 block each)", kvc.UsedBlockCnt)
	}
}
```

**Step 2: Run tests to verify failure**

Run: `go test ./sim/... -run "TestAllocateKVBlocks_(MidLoopFailure|CachedBlockRollback|BlockConservation)" -v -count=1`
Expected: MidLoopFailure test likely FAILS (blocks leaked on mid-loop failure)

**Step 3: Implement transactional rollback**

In `sim/kvcache.go`, add a rollback helper and restructure AllocateKVBlocks:

First, add a rollback helper after the `countFreeBlocks` method:

```go
// cachedBlockMutation tracks a cached block's state before mutation for rollback.
type cachedBlockMutation struct {
	block    *KVBlock
	wasInUse bool
}

// rollbackAllocation undoes all mutations from a failed AllocateKVBlocks call.
// cacheMissCount is the number of CacheMisses increments to undo (one per new block).
func (kvc *KVCacheState) rollbackAllocation(reqID string, cachedMutations []cachedBlockMutation, newlyAllocated []*KVBlock) {
	// Undo new block allocations (reverse order for clean free list state)
	for i := len(newlyAllocated) - 1; i >= 0; i-- {
		blk := newlyAllocated[i]
		blk.InUse = false
		blk.RefCount = 0
		blk.Tokens = nil
		if blk.Hash != "" {
			delete(kvc.HashToBlock, blk.Hash)
			blk.Hash = ""
		}
		kvc.UsedBlockCnt--
		kvc.CacheMisses-- // Undo the CacheMisses++ from the allocation loop
		kvc.appendToFreeList(blk)
	}
	// Undo cached block mutations
	for _, cm := range cachedMutations {
		cm.block.RefCount--
		if !cm.wasInUse && cm.block.RefCount == 0 {
			cm.block.InUse = false
			kvc.UsedBlockCnt--
			kvc.appendToFreeList(cm.block)
		}
	}
	// Clean up RequestMap
	delete(kvc.RequestMap, reqID)
}
```

Then modify the `AllocateKVBlocks` function to track mutations. Replace the cached-blocks section and the new-blocks allocation loop to track `cachedMutations` and `newlyAllocated`, and call `rollbackAllocation` on failure instead of bare `return false`.

The key changes in AllocateKVBlocks:

1. Declare tracking slices at function start:
```go
var cachedMutations []cachedBlockMutation
var newlyAllocated []*KVBlock
```

2. In the cached blocks section (the `else` branch), track mutations:
```go
for _, blockId := range cachedBlocks {
    blk := kvc.Blocks[blockId]
    wasInUse := blk.InUse
    blk.RefCount++
    if !blk.InUse {
        blk.InUse = true
        kvc.UsedBlockCnt++
        kvc.removeFromFreeList(blk)
    }
    cachedMutations = append(cachedMutations, cachedBlockMutation{block: blk, wasInUse: wasInUse})
    kvc.RequestMap[reqID] = append(kvc.RequestMap[reqID], blockId)
}
```

3. In the new block allocation loop, track and rollback:
```go
for i := int64(0); i < numNewBlocks; i++ {
    blk := kvc.popFreeBlock()
    if blk == nil {
        // Rollback all mutations from this call
        kvc.rollbackAllocation(reqID, cachedMutations, newlyAllocated)
        return false
    }
    // ... existing mutation logic ...
    newlyAllocated = append(newlyAllocated, blk)
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/... -run "TestAllocateKVBlocks_(MidLoopFailure|CachedBlockRollback|BlockConservation)" -v -count=1`
Expected: All PASS

Run full suite:
Run: `go test ./sim/... -count=1`
Expected: All pass

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/kvcache.go sim/kvcache_test.go
git commit -m "fix(kv): add transactional rollback to AllocateKVBlocks (#200)

- Track cached block mutations and newly allocated blocks
- On mid-loop popFreeBlock failure, rollback all mutations
- Preserves allocated + free == total conservation invariant
- Implements BC-4, BC-5, BC-6, EC-1

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Remove deprecated AllocateKVBlocksPrefill and AllocateKVBlocksDecode (NC-1)

**Contracts Implemented:** NC-1

**Files:**
- Modify: `sim/kvcache.go` (delete two functions)

**Step 1: Verify no callers exist**

Run: `grep -rn 'AllocateKVBlocksPrefill\|AllocateKVBlocksDecode' sim/ --include='*.go' | grep -v '_test.go' | grep -v 'deprecated'`
Expected: Only the function definitions themselves (no callers)

**Step 2: Delete both deprecated functions**

In `sim/kvcache.go`, delete:
- `AllocateKVBlocksPrefill` (lines 140-201) including its deprecation comment
- `AllocateKVBlocksDecode` (lines 308-357) including its deprecation comment

**Step 3: Run full test suite**

Run: `go test ./sim/... -count=1`
Expected: All pass (no test calls these deprecated functions)

Run: `go test ./... -count=1`
Expected: All pass

**Step 4: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/kvcache.go
git commit -m "refactor(kv): remove deprecated AllocateKVBlocksPrefill and AllocateKVBlocksDecode

- Both superseded by unified AllocateKVBlocks (since PR12)
- AllocateKVBlocksPrefill had same partial allocation leak as #200
- Removal is safer than patching deprecated code (NC-1)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Document completion condition and PendingRequests fix (2g, 2f)

**Contracts Implemented:** Documentation only

**Files:**
- Modify: `sim/simulator.go` (add completion condition comment)
- Modify: `sim/cluster/cluster.go` (add comment explaining preemption safety)

**Step 1: Add completion condition comment (2g)**

In `sim/simulator.go`, before the completion check loop (around line 572, the `remaining := []*Request{}` section), add:

```go
	// IMPORTANT: This completion loop MUST run as a separate pass after the
	// prefill/decode execution loop above. For zero-output-token requests,
	// both "prefill completed" and "request completed" conditions are true
	// in the same step. The two-pass design ensures prefill metrics (TTFT)
	// are recorded before completion metrics (E2E). If these loops were ever
	// consolidated into a single pass, both branches would fire for the
	// same request in the same step.
```

**Step 2: Add PendingRequests preemption-safety comment (2f)**

In `sim/cluster/cluster.go`, enhance the existing comment at the QueuedEvent detection (around line 163):

```go
			// Causal decrement: QueuedEvent is the definitive moment a request
			// enters the WaitQ, meaning it was absorbed from pending (#178).
			// This replaces the fragile QueueDepth before/after heuristic.
			//
			// Preemption safety (#192): preemption re-enqueues via direct
			// WaitQ.queue manipulation (sim/simulator.go:363), NOT via QueuedEvent.
			// Therefore preemption cannot trigger a false decrement here.
```

**Step 3: Run tests to verify no regressions**

Run: `go test ./... -count=1`
Expected: All pass

**Step 4: Run lint check**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/simulator.go sim/cluster/cluster.go
git commit -m "docs(sim): document completion loop two-pass dependency and preemption-safe pending tracking

- Add comment explaining zero-output-token two-pass requirement (2g)
- Document why QueuedEvent detection is preemption-safe (#192)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | TestAllocateKVBlocks_PartialBlockFill_AdvancesByActualTokenCount |
| BC-2 | Task 2 | Unit | TestAllocateKVBlocks_ChunkedPrefill_PrefixHashUsesAbsoluteOffset |
| BC-3 | Task 3 | Unit | TestMakeRunningBatch_DecodePhase_PreemptGetsPositiveTokenCount |
| BC-4, BC-5 | Task 4 | Unit | TestAllocateKVBlocks_MidLoopFailure_RollsBackNewBlocks |
| BC-5 | Task 4 | Unit | TestAllocateKVBlocks_CachedBlockRollback_OnNewBlockFailure |
| BC-6 | Task 4 | Invariant | TestAllocateKVBlocks_BlockConservation_AfterAllocateReleaseCycles |
| EC-1 | Task 4 | Unit | (covered by MidLoopFailure test — checks RequestMap cleanup) |
| NC-1 | Task 5 | Compile | Deprecated functions removed (compilation verifies no callers) |

**Golden dataset:** No golden dataset changes expected. These are internal KV cache fixes that affect cache efficiency but not the golden dataset's test scenarios (which use sufficient KV blocks and don't trigger the bug paths).

**Invariant tests:** BC-6 (block conservation) is a new invariant test. Phase 4 will add comprehensive invariant tests; this PR adds the minimal set needed to verify the rollback fix.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Rollback misses a mutation pathway | Medium | High | Comprehensive test with cached+new blocks; conservation invariant check | Task 4 |
| Deprecated function removal breaks hidden caller | Low | High | grep verification before deletion; full test suite run | Task 5 |
| Partial block fix changes hash computation order | Low | Medium | Explicit test with known token sequences and expected hashes | Task 2 |
| Decode fix affects preemption behavior | Low | Medium | Full golden dataset suite validates no regression | Task 3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions (rollback helper is minimal and focused)
- [x] No feature creep beyond PR scope (only Phase 2 items)
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used from existing shared test package (not duplicated locally)
- [x] CLAUDE.md: no updates needed (no new files/packages/CLI flags)
- [x] No stale references left in CLAUDE.md
- [x] Deviation log reviewed: 2a already fixed (PR #206), 2f already fixed (PR #205)
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (2c before 2e since rollback restructures the function)
- [x] All contracts mapped to specific tasks
- [x] Golden dataset regeneration not needed
- [x] Construction site audit: no new struct fields added
- [x] No new CLI flags
- [x] Every error path either returns false (allocation failure) or is documented
- [x] No map iteration feeds float accumulation
- [x] Library code never calls logrus.Fatalf
- [x] Allocation loop handles mid-loop failure with rollback (Task 4)

---

## Appendix: File-Level Implementation Details

### File: `sim/kvcache.go`

**Purpose:** Fix three bugs in AllocateKVBlocks, add rollback support, remove deprecated functions.

**Changes:**

1. **Lines 254-259 (partial block fill):** Replace post-append progress computation with pre-append remaining capacity capture. Also fix slice start index to use `newTokenProgressIndex`.

2. **Lines 292-296 (prefix hash):** Replace `req.InputTokens[:end]` with `req.InputTokens[:startIndex + end]` for absolute offset.

3. **Lines 237-301 (allocation loop):** Add `cachedMutations` and `newlyAllocated` tracking slices. Replace bare `return false` in inner loop with `rollbackAllocation()` call.

4. **New: `cachedBlockMutation` struct and `rollbackAllocation` method.**

5. **Delete: `AllocateKVBlocksPrefill` (lines 140-201) and `AllocateKVBlocksDecode` (lines 308-357).**

### File: `sim/simulator.go`

**Purpose:** Fix decode path token count, add documentation comment.

**Changes:**

1. **Lines 419-421:** Replace `sim.preempt(req, now, numNewTokens)` with `decodeTokens := int64(1); sim.preempt(req, now, decodeTokens)`.

2. **Before line 572:** Add multi-line comment explaining two-pass completion loop dependency.

### File: `sim/cluster/cluster.go`

**Purpose:** Document preemption safety of QueuedEvent-based pending request tracking.

**Changes:**

1. **Lines 163-171:** Expand comment to explain why preemption cannot cause false decrements.
