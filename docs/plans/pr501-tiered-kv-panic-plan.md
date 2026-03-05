# PR #501: Fix TieredKVCache Panic on CPU→GPU Reload

- **Goal:** Prevent `slice bounds out of range` panic when CPU→GPU block reload covers the entire requested allocation range.
- **The problem today:** `TieredKVCache.AllocateKVBlocks` panics when `tryReloadFromCPU()` reloads blocks whose prefix covers more tokens than the original `startIndex..endIndex` range, producing `newStart >= endIndex` and an inverted slice at `cache.go:154`.
- **What this PR adds:**
  1. Guard in `tiered.go` that handles `newStart >= endIndex` after CPU reload — commits cached blocks for new requests, returns true for running requests.
  2. New `commitCachedBlocks` method on `KVCacheState` for the tiered guard's use (does NOT replace the inline code in `AllocateKVBlocks` which needs rollback tracking).
  3. Behavioral tests exercising both panic paths (running request + new request) with INV-4 conservation verification.
- **Why this matters:** Without this fix, any sustained workload with tiered KV cache under memory pressure can trigger a hard panic, terminating the simulation.
- **Architecture:** Guard condition in `sim/kv/tiered.go:AllocateKVBlocks` + new unexported method in `sim/kv/cache.go`. No new types, interfaces, or CLI flags.
- **Source:** GitHub issue #501.
- **Closes:** Fixes #501.
- **Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block:** TieredKVCache (GPU+CPU offload/reload)
2. **Adjacent blocks:** KVCacheState (GPU-only, delegates allocation), VLLMBatchFormation (calls AllocateKVBlocks via `preemptForTokens` for running requests, Phase 2 for new requests)
3. **Invariants touched:** INV-4 (KV block conservation: allocated + free = total)
4. **Construction site audit:** No struct fields added. No construction site changes needed.

**Two call sites hit the bug:**
- **`preemptForTokens` (running request):** `cachedBlocks=[]int64{}`, request already in `RequestMap`. Fix: just return true.
- **Phase 2 (new request):** `cachedBlocks` from `GetCachedBlocks`, request NOT in `RequestMap`. Fix: commit cached blocks to `RequestMap` before returning true (otherwise `ReleaseKVBlocks` finds nothing to release → INV-4 violation).

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes a panic in `TieredKVCache.AllocateKVBlocks` where CPU→GPU block reload produces a `newStart` that equals or exceeds `endIndex`, creating an inverted slice. Two call sites are affected: `preemptForTokens` (running requests) and Phase 2 batch formation (new requests). Running requests need only a guard; new requests also need cached-block commit. The fix adds a new `commitCachedBlocks` method to `KVCacheState` for use by the tiered guard. The existing inline commit logic in `AllocateKVBlocks` is left unchanged because it integrates with rollback tracking (`cachedMutations`) that the new method doesn't need.

### B) Behavioral Contracts

**Positive contracts:**

```
BC-1: Full-Range Cache Hit After Reload — Running Request
- GIVEN a running request (already in RequestMap) where GPU allocation fails and CPU→GPU reload succeeds
- WHEN the reloaded blocks cover the entire requested range (newStart >= endIndex)
- THEN AllocateKVBlocks returns true without panic

BC-2: Full-Range Cache Hit After Reload — New Request
- GIVEN a new request (not in RequestMap) where GPU allocation fails and CPU→GPU reload succeeds
- WHEN the reloaded blocks cover the entire requested range (newStart >= endIndex)
- THEN AllocateKVBlocks returns true, cached blocks are committed to RequestMap, and INV-4 conservation holds through subsequent release
```

**Negative contracts:**

```
BC-3: Partial Reload Still Delegates
- GIVEN a TieredKVCache where CPU→GPU reload covers only part of the requested range
- WHEN newStart > startIndex but newStart < endIndex
- THEN AllocateKVBlocks delegates to gpu.AllocateKVBlocks(req, newStart, endIndex, newCached) as before
```

### C) Component Interaction

```
batch_formation.go (two call sites)
  ├── preemptForTokens:  AllocateKVBlocks(req, progressIdx, progressIdx+N, [])    ← running request
  └── Phase 2:           AllocateKVBlocks(req, startIdx, endIdx, cachedBlocks)     ← new request

TieredKVCache.AllocateKVBlocks(req, startIndex, endIndex, cached)
  ├── gpu.AllocateKVBlocks(req, start, end, cached)  → succeeds? return true
  ├── tryReloadFromCPU()
  ├── newCached = gpu.GetCachedBlocks(req.InputTokens)
  ├── newStart = len(newCached) * blockSize
  ├── if newStart >= endIndex:
  │     ├── if new request: gpu.commitCachedBlocks(req.ID, newCached)  ★ NEW
  │     └── return true                                                ★ NEW
  ├── if newStart > startIndex: gpu.AllocateKVBlocks(req, newStart, endIndex, newCached)
  └── else: gpu.AllocateKVBlocks(req, startIndex, endIndex, cached)
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue suggests `return true` | Guard + `commitCachedBlocks` for new requests | CORRECTION — pure `return true` breaks INV-4 for Phase 2 new requests (cached blocks never committed to RequestMap → ReleaseKVBlocks finds nothing to release) |
| Issue suggests modifying only tiered.go | Also add method in cache.go | ADDITION — new `commitCachedBlocks` method for DRY. Does NOT replace inline code in `AllocateKVBlocks` (which integrates with rollback tracking via `cachedMutations`). |

### E) Review Guide

**Scrutinize:**
1. The guard condition at `tiered.go:83` — specifically the `RequestMap[req.ID]` existence check that differentiates running vs new requests.
2. The `commitCachedBlocks` method — verify it does NOT replace the inline code in `AllocateKVBlocks` (the inline version feeds `cachedMutations` for rollback; the new method is only for the tiered guard where rollback is not needed because we return true immediately).

**Safe to skim:** Test setup (block counts, thresholds, offload patterns) — follows existing tiered test conventions.

**Known debt:** The inline cached-block commit in `AllocateKVBlocks` (cache.go:184-197) and the new `commitCachedBlocks` method share logic. A future refactoring could unify them by having `commitCachedBlocks` return `[]cachedBlockMutation` for rollback support. Out of scope for this bug fix.

---

## Part 2: Executable Implementation

### F) Implementation Overview

| File | Action | Purpose |
|------|--------|---------|
| `sim/kv/cache.go` | Modify | Add `commitCachedBlocks` method (new, does NOT replace inline code) |
| `sim/kv/tiered.go` | Modify | Add guard for `newStart >= endIndex` with cached-block commit |
| `sim/kv/tiered_test.go` | Modify | Add tests: running request panic (BC-1), new request conservation (BC-2) |

No dead code. No new files.

### G) Task Breakdown

#### Task 1: Add commitCachedBlocks + fix guard + tests (BC-1 through BC-3)

**Step 1: Write failing tests**

Add to `sim/kv/tiered_test.go`:

```go
func TestTieredKVCache_AllocateKVBlocks_FullRangeReload_RunningRequest_NoPanic(t *testing.T) {
	// BC-1: GIVEN 10 GPU blocks (block_size=2), 10 CPU blocks, threshold=0.3
	// Setup: fill GPU completely, arrange for a running request to need allocation
	// that fails on GPU, succeeds via CPU reload covering full range.
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.3, 100.0, 0)

	// Step 1: Allocate target prefix [1,2,3,4] (2 blocks, IDs 0-1)
	target := &sim.Request{ID: "target", InputTokens: []int{1, 2, 3, 4}}
	if !tiered.AllocateKVBlocks(target, 0, 4, []int64{}) {
		t.Fatal("initial allocation should succeed")
	}

	// Step 2: Allocate 4 more requests to fill all 10 GPU blocks (4 × 2 blocks = 8 more)
	for i := 0; i < 4; i++ {
		other := &sim.Request{
			ID:          fmt.Sprintf("o%d", i),
			InputTokens: []int{i*4 + 10, i*4 + 11, i*4 + 12, i*4 + 13},
		}
		if !tiered.AllocateKVBlocks(other, 0, 4, []int64{}) {
			t.Fatalf("other allocation %d should succeed", i)
		}
	}
	// GPU: 10 used, 0 free

	// Step 3: Release target → blocks become free with hashes → offload to CPU
	// (util 80% > 30%). Offloaded blocks re-added as empty free blocks.
	tiered.ReleaseKVBlocks(target)
	if tiered.offloadCount == 0 {
		t.Fatal("setup error: offload should have triggered")
	}
	// GPU: 8 used, 2 free (empty, no hashes)

	// Step 4: Re-allocate target (puts it back in RequestMap as running request)
	// Uses the 2 empty free blocks. GPU: 10 used, 0 free.
	target2 := &sim.Request{ID: "target", InputTokens: []int{1, 2, 3, 4}}
	if !tiered.AllocateKVBlocks(target2, 0, 4, []int64{}) {
		t.Fatal("re-allocation should succeed (2 free blocks available)")
	}
	// GPU: 10 used, 0 free. Target is running, in RequestMap.

	// Step 5: Release one other to make room for offload→reload cycle
	tiered.ReleaseKVBlocks(&sim.Request{ID: "o3"})
	// GPU: 8 used, 2 free. These 2 free blocks have hashes from o3's allocation.
	// maybeOffload: util 80% > 30% → offloads o3's hashed blocks to CPU.
	// After offload: GPU still 8 used, 2 free (now empty, no hashes).

	// Step 6: Fill the 2 free blocks with fillers
	for i := 0; i < 2; i++ {
		filler := &sim.Request{
			ID:          fmt.Sprintf("f%d", i),
			InputTokens: []int{i*2 + 200, i*2 + 201},
		}
		if !tiered.AllocateKVBlocks(filler, 0, 2, []int64{}) {
			t.Fatalf("filler allocation %d should succeed", i)
		}
	}
	// GPU: 10 used, 0 free. Target ("target") is still running.

	// Step 7: Release one filler to create exactly 1 free block for reload
	tiered.ReleaseKVBlocks(&sim.Request{ID: "f1"})
	// GPU: 9 used, 1 free.

	// WHEN: running request needs chunk [2:4] — GPU has only 1 free block,
	// needs 1 block for tokens [2:4]. But first gpu.AllocateKVBlocks finds
	// request already in RequestMap, tries to extend. This may succeed.
	// Actually — for the running request path (preemptForTokens), the call
	// pattern is AllocateKVBlocks(req, progressIndex, progressIndex+N, []).
	// If gpu allocation succeeds, we never reach the reload path.
	// To force GPU failure: need 0 free blocks.
	// Re-fill: allocate the 1 free block
	extra := &sim.Request{ID: "extra", InputTokens: []int{250, 251}}
	if !tiered.AllocateKVBlocks(extra, 0, 2, []int64{}) {
		t.Fatal("extra allocation should succeed (1 free block)")
	}
	// GPU: 10 used, 0 free.

	// NOW: AllocateKVBlocks for running target with chunk [2:4].
	// gpu.AllocateKVBlocks will fail (0 free blocks).
	// tryReloadFromCPU will reload target's original prefix blocks from CPU.
	// GetCachedBlocks([1,2,3,4]) will find the reloaded hashes.
	// newStart = 2*2 = 4 >= endIndex = 4. Bug path triggered.
	ok := tiered.AllocateKVBlocks(target2, 2, 4, []int64{})

	// THEN: no panic, returns true (BC-1)
	if !ok {
		t.Error("AllocateKVBlocks should succeed for running request when reload covers full range")
	}
}

func TestTieredKVCache_AllocateKVBlocks_FullRangeReload_NewRequest_Conservation(t *testing.T) {
	// BC-2: GIVEN 10 GPU blocks (block_size=2), 10 CPU blocks, threshold=0.3
	// Setup: allocate target, fill GPU, release target (offloads), fill more,
	// then request same prefix as a NEW request (not in RequestMap).
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.3, 100.0, 0)

	// Step 1: Allocate target prefix [1,2,3,4] (2 blocks)
	target := &sim.Request{ID: "target", InputTokens: []int{1, 2, 3, 4}}
	if !tiered.AllocateKVBlocks(target, 0, 4, []int64{}) {
		t.Fatal("initial allocation should succeed")
	}

	// Step 2: Fill GPU: 3 more requests × 2 blocks = 8 used, 2 free
	others := make([]*sim.Request, 3)
	for i := 0; i < 3; i++ {
		others[i] = &sim.Request{
			ID:          fmt.Sprintf("o%d", i),
			InputTokens: []int{i*4 + 10, i*4 + 11, i*4 + 12, i*4 + 13},
		}
		if !tiered.AllocateKVBlocks(others[i], 0, 4, []int64{}) {
			t.Fatalf("other allocation %d should succeed", i)
		}
	}

	// Step 3: Release target → offload cached blocks to CPU (util 60% > 30%)
	tiered.ReleaseKVBlocks(target)
	if tiered.offloadCount == 0 {
		t.Fatal("setup error: offload should have triggered")
	}
	// GPU: 6 used, 4 free (2 from offload as empty + 2 original empty)

	// Step 4: Fill GPU to 9 used, 1 free
	fillers := make([]*sim.Request, 3)
	for i := 0; i < 3; i++ {
		fillers[i] = &sim.Request{
			ID:          fmt.Sprintf("f%d", i),
			InputTokens: []int{i*2 + 100, i*2 + 101},
		}
		if !tiered.AllocateKVBlocks(fillers[i], 0, 2, []int64{}) {
			t.Fatalf("filler allocation %d should succeed", i)
		}
	}
	// GPU: 9 used, 1 free.

	// Step 5: Fill the last free block
	lastFiller := &sim.Request{ID: "lf", InputTokens: []int{150, 151}}
	if !tiered.AllocateKVBlocks(lastFiller, 0, 2, []int64{}) {
		t.Fatal("last filler should succeed")
	}
	// GPU: 10 used, 0 free.

	// Step 6: Release one filler to create exactly 1 free block for reload
	tiered.ReleaseKVBlocks(fillers[2])
	// GPU: 9 used, 1 free (from f2 release; its blocks may have hashes → offload
	// may trigger again if util > 30%). 9/10 = 90% > 30% so offload triggers for
	// f2's hashed block. After offload: still 9 used conceptually but the freed
	// block is empty-free. Actually: f2's blocks become free, maybeOffload checks
	// util. 8/10 = 80% after release of f2 (1 block). Wait — f2 has 1 block
	// (2 tokens / 2 blocksize = 1 block). So release f2: 9 used → 8 used.
	// Util 80% > 30% → offload f2's hashed free block to CPU. Re-add as empty free.
	// GPU: 8 used, 2 free (1 from release, 1 from... no, release freed 1 block,
	// offload moved it to CPU and re-added as empty = still 1 free block total).
	// Actually offload: removes hashed free block from free list, puts in CPU,
	// clears hash, re-appends to free list. Net effect: still 1 free block.
	// 8/10 = 80% > 30%. Are there more hashed free blocks? Only f2's block.
	// After offloading it: 8 used, 2 free? No — only 1 block was freed (f2 had 1 block).
	// 8 used + 1 free = 9. But total is 10. Where's the 10th?
	// Let me recount: target=2, o0=2, o1=2, o2=2, f0=1, f1=1, lf=1 = 11 blocks.
	// But we only have 10 blocks! Something's wrong.
	// Actually: target was released at step 3. So after step 4:
	// o0=2, o1=2, o2=2, f0=1, f1=1, f2=1 = 9 used, 1 free. OK.
	// Step 5: lf=1, so 10 used, 0 free. Good.
	// Step 6: release f2 → 9 used, 1 free.

	// WHEN: NEW request with same prefix [1,2,3,4], partial cached state.
	// GetCachedBlocks on GPU returns nothing (target's hashes were offloaded to CPU).
	// So startIndex=0, and we request allocation for [0:4].
	// gpu.AllocateKVBlocks(newReq, 0, 4, []) needs 2 blocks but only 1 free → fails.
	// tryReloadFromCPU reloads target's prefix blocks from CPU.
	// GetCachedBlocks([1,2,3,4]) now finds 2 blocks.
	// newStart = 2*2 = 4 >= endIndex = 4. Bug path triggered.
	newReq := &sim.Request{ID: "new-req", InputTokens: []int{1, 2, 3, 4}}
	ok := tiered.AllocateKVBlocks(newReq, 0, 4, []int64{})

	// THEN: no panic, returns true (BC-2)
	if !ok {
		t.Error("AllocateKVBlocks should succeed for new request when reload covers full range")
	}

	// AND: INV-4 conservation — release everything, verify used blocks return to 0
	tiered.ReleaseKVBlocks(newReq)
	for _, o := range others {
		tiered.ReleaseKVBlocks(o)
	}
	tiered.ReleaseKVBlocks(fillers[0])
	tiered.ReleaseKVBlocks(fillers[1])
	tiered.ReleaseKVBlocks(lastFiller)
	if tiered.UsedBlocks() != 0 {
		t.Errorf("UsedBlocks() = %d after all releases, want 0 (INV-4 conservation)", tiered.UsedBlocks())
	}
}
```

**Step 2: Run tests to verify they fail**

```bash
go test ./sim/kv/... -run "TestTieredKVCache_AllocateKVBlocks_FullRangeReload" -v
```

Expected: panic or conservation failure

**Step 3: Add commitCachedBlocks method**

In `sim/kv/cache.go`, add method after `rollbackAllocation` (does NOT replace the inline code):

```go
// commitCachedBlocks registers cached blocks for a request's first allocation.
// Increments RefCount, sets InUse, removes from free list, records cache hits,
// and adds block IDs to RequestMap.
//
// NOTE: This method does NOT track mutations for rollback. It is used only by
// TieredKVCache when the entire requested range is cached after reload
// (returning true immediately, so no rollback is possible). The inline
// equivalent in AllocateKVBlocks (lines 184-197) feeds cachedMutations for
// rollback support — do not replace that inline code with this method.
func (kvc *KVCacheState) commitCachedBlocks(reqID string, cachedBlocks []int64) {
	for _, blockID := range cachedBlocks {
		blk := kvc.Blocks[blockID]
		blk.RefCount++
		if !blk.InUse {
			blk.InUse = true
			kvc.UsedBlockCnt++
			kvc.removeFromFreeList(blk)
		}
		kvc.CacheHits++
		kvc.RequestMap[reqID] = append(kvc.RequestMap[reqID], blockID)
	}
}
```

**Step 4: Add guard in TieredKVCache**

In `sim/kv/tiered.go`, replace lines 82-86:

```go
		newStart := int64(len(newCached)) * t.gpu.BlockSize()
		if newStart > startIndex {
			// More cache hits after reload — retry with reduced allocation range
			return t.gpu.AllocateKVBlocks(req, newStart, endIndex, newCached)
		}
```

With:

```go
		newStart := int64(len(newCached)) * t.gpu.BlockSize()
		if newStart > startIndex {
			if newStart >= endIndex {
				// Entire requested range is cached after reload.
				// For new requests, commit cached blocks to RequestMap so
				// ReleaseKVBlocks can track them. Running requests (already
				// in RequestMap from initial allocation) need no action.
				if _, exists := t.gpu.RequestMap[req.ID]; !exists {
					t.gpu.commitCachedBlocks(req.ID, newCached)
				}
				return true
			}
			// More cache hits after reload — retry with reduced allocation range
			return t.gpu.AllocateKVBlocks(req, newStart, endIndex, newCached)
		}
```

**Step 5: Run tests to verify they pass**

```bash
go test ./sim/kv/... -run "TestTieredKVCache_AllocateKVBlocks_FullRangeReload" -v
```

Expected: PASS

**Step 6: Run full test suite + lint**

```bash
go test ./sim/kv/... -v && golangci-lint run ./sim/kv/...
```

Expected: all pass, no lint issues

**Step 7: Run all tests (regression)**

```bash
go test ./...
```

Expected: all pass (BC-3 — existing tests unchanged, partial reload path unmodified)

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Behavioral | TestTieredKVCache_AllocateKVBlocks_FullRangeReload_RunningRequest_NoPanic |
| BC-2 | Task 1 | Behavioral + INV-4 | TestTieredKVCache_AllocateKVBlocks_FullRangeReload_NewRequest_Conservation |
| BC-3 | Task 1 | Regression | Existing cache_test.go + tiered_test.go tests (no inline code replaced) |

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Guard skips cached-block commit for new requests | Medium | High | RequestMap existence check differentiates running vs new requests. Test BC-2 verifies conservation. | Task 1 |
| Guard condition off-by-one (`>` vs `>=`) | Low | High | `endIndex` is non-inclusive, so `newStart == endIndex` means all tokens covered. Both tests verify. | Task 1 |
| Existing partial-reload path regresses | Low | Medium | Guard only activates when `newStart >= endIndex`; partial case (`newStart < endIndex`) unchanged. Full test suite validates. | Task 1 |
| commitCachedBlocks used inside AllocateKVBlocks breaking rollback | Low | High | Method is clearly documented as NOT replacing inline code. Comment explains why. | Task 1 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — `commitCachedBlocks` is the minimum needed for the tiered guard.
- [x] No feature creep beyond PR scope.
- [x] No unexercised flags or interfaces.
- [x] No partial implementations.
- [x] No breaking changes without explicit contract updates.
- [x] No hidden global state impact.
- [x] All new code will pass golangci-lint.
- [x] Shared test helpers used (existing tiered test patterns).
- [x] CLAUDE.md — no updates needed (no new files/packages/CLI flags).
- [x] No stale references.
- [x] Documentation DRY — no canonical sources modified.
- [x] Deviation log reviewed — two deviations, both justified.
- [x] Each task produces working, testable code.
- [x] Task dependencies correctly ordered (single task).
- [x] All contracts mapped to tasks.
- [x] Golden dataset — not affected.
- [x] Construction site audit — no struct fields added.
- [x] Not part of a macro plan.

**Antipattern rules:**
- [x] R1: No silent continue/return — guard commits cached blocks before returning true
- [x] R4: No struct fields added
- [x] R5: Existing rollback in AllocateKVBlocks unchanged; new commitCachedBlocks only used where rollback is not needed
- [x] R6: No logrus.Fatalf in sim/
- [x] R7: INV-4 conservation test (BC-2) alongside behavioral test (BC-1)
- [x] R11: No new division
- [x] R19: No new retry loops

---

## Appendix: File-Level Implementation Details

**File: `sim/kv/cache.go`**
- **Purpose:** Add `commitCachedBlocks` method for TieredKVCache's reload guard
- **New method:** `commitCachedBlocks(reqID string, cachedBlocks []int64)` — unexported, same package
- **Does NOT replace:** Inline code at `AllocateKVBlocks` lines 184-197 (which feeds `cachedMutations` for rollback). Both exist in parallel — the inline version for normal allocation with rollback support, the method for the tiered guard where rollback is not needed.
- **Key note:** No debug log in the new method (the inline version logs with `logrus.Debugf`; the tiered guard is a fast path that doesn't need per-block logging).

**File: `sim/kv/tiered.go`**
- **Purpose:** Guard against inverted slice bounds when CPU→GPU reload covers entire requested range
- **Change:** `if newStart >= endIndex` guard at line 83, with `commitCachedBlocks` for new requests
- **Key note:** `RequestMap[req.ID]` existence check differentiates running (already tracked) vs new (needs commit) requests. Both call sites in batch_formation.go are covered.

**File: `sim/kv/tiered_test.go`**
- **Purpose:** Test both panic paths and conservation
- **Test 1 (BC-1):** Running request path — GPU must be fully packed (0 free blocks) when the triggering call is made, forcing GPU allocation to fail and enter the reload path. Setup: allocate all 10 blocks, release target (offloads to CPU), re-allocate target + fillers to fill all blocks again, then request chunk [2:4] for the running target.
- **Test 2 (BC-2):** New request path — GPU must have insufficient free blocks for a 2-block allocation. Setup: allocate target, fill GPU, release target (offloads), fill remaining free blocks, release one filler (1 free), then NEW request for same prefix needs 2 blocks. GPU fails, reload succeeds, newStart >= endIndex. INV-4 conservation verified via release cycle.
