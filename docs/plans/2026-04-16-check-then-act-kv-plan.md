# Check-Then-Act KV Cache Allocation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate rollback-based KV allocation in favor of vLLM's check-then-act pattern, fixing the #1061 block leak deadlock and aligning BLIS with real vLLM behavior.

**Architecture:** Universal pre-check in `AllocateKVBlocks` gates both prefill and decode paths before any state mutation. Rollback machinery is removed entirely. Direct free-block counter on the doubly-linked free list replaces arithmetic derivation (`TotalBlocks - UsedBlockCnt`). Post-pre-check allocation failure becomes a panic (INV-4 invariant violation, structurally unreachable in single-threaded DES).

**Tech Stack:** Go 1.22+, `sim/kv/` package, testify assertions

**Design doc:** `docs/plans/2026-04-16-check-then-act-kv-design.md`

---

### Task 1: Add Direct Free Block Counter (FreeBlockCnt)

Replace the arithmetic `UsedBlockCnt` with a direct `FreeBlockCnt` counter maintained by the free list itself. This mirrors vLLM's `FreeKVCacheBlockQueue.num_free_blocks`.

**Files:**
- Modify: `sim/kv/cache.go:36-47` (struct), `sim/kv/cache.go:49-70` (constructor), `sim/kv/cache.go:72-87` (appendToFreeList), `sim/kv/cache.go:104-122` (removeFromFreeList), `sim/kv/cache.go:356-359` (countFreeBlocks), `sim/kv/cache.go:476-477` (UsedBlocks accessor)
- Modify: `sim/kv/cache.go:248-252` (cached block commit in AllocateKVBlocks — remove `UsedBlockCnt++`)
- Modify: `sim/kv/cache.go:316-318` (new block alloc in AllocateKVBlocks — remove `UsedBlockCnt++`)
- Modify: `sim/kv/cache.go:437-449` (commitCachedBlocks — remove `UsedBlockCnt++`)
- Modify: `sim/kv/cache.go:451-471` (ReleaseKVBlocks — remove `UsedBlockCnt--`)
- Modify: `sim/kv/cache.go:373-413` (rollbackAllocation — remove `UsedBlockCnt--` lines; full deletion comes in Task 3)
- Test: `sim/kv/cache_test.go`

**Step 1: Write the failing test**

Add a test that verifies `FreeBlockCnt` is maintained directly by free list operations, not derived from arithmetic. This test should pass with the new counter and would have caught the #1061 drift bug.

```go
func TestFreeBlockCnt_DirectCounter_MatchesFreeListLength(t *testing.T) {
	// GIVEN a KV cache where blocks are allocated and released
	kvc := NewKVCacheState(10, 4)

	// WHEN we allocate some blocks
	req := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8}}
	ok := kvc.AllocateKVBlocks(req, 0, 8, []int64{})
	require.True(t, ok)

	// THEN FreeBlockCnt matches an independent walk of the free list
	actualFree := int64(0)
	node := kvc.FreeHead
	for node != nil {
		actualFree++
		node = node.NextFree
	}
	assert.Equal(t, actualFree, kvc.countFreeBlocks(),
		"countFreeBlocks() must match physical free list length")
	assert.Equal(t, kvc.TotalBlocks, kvc.countFreeBlocks()+kvc.UsedBlocks(),
		"INV-4: free + used must equal total")

	// WHEN we release the blocks
	kvc.ReleaseKVBlocks(req)

	// THEN all blocks are free again
	actualFree = 0
	node = kvc.FreeHead
	for node != nil {
		actualFree++
		node = node.NextFree
	}
	assert.Equal(t, kvc.TotalBlocks, kvc.countFreeBlocks(),
		"all blocks should be free after release")
	assert.Equal(t, actualFree, kvc.countFreeBlocks(),
		"countFreeBlocks() must match physical free list length after release")
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/kv/... -run TestFreeBlockCnt_DirectCounter -v`
Expected: FAIL — `FreeBlockCnt` field does not exist yet.

**Step 3: Implement the direct counter**

In `sim/kv/cache.go`:

1. Replace `UsedBlockCnt` with `FreeBlockCnt` in the struct:
```go
type KVCacheState struct {
	TotalBlocks     int64
	BlockSizeTokens int64
	Blocks          []*KVBlock
	RequestMap      map[string][]int64
	HashToBlock     map[string]int64
	FreeHead        *KVBlock
	FreeTail        *KVBlock
	FreeBlockCnt    int64              // Direct count of blocks in free list (vLLM parity)
	CacheHits       int64
	CacheMisses     int64
}
```

2. In `NewKVCacheState`, remove the explicit `UsedBlockCnt: 0` (it was zero-value anyway). The `FreeBlockCnt` will be incremented by `appendToFreeList` during the init loop — no explicit initialization needed.

3. Add counter maintenance to `appendToFreeList`:
```go
func (kvc *KVCacheState) appendToFreeList(block *KVBlock) {
	kvc.FreeBlockCnt++
	// ... existing linked list logic unchanged ...
}
```

4. Add counter maintenance to `removeFromFreeList`:
```go
func (kvc *KVCacheState) removeFromFreeList(block *KVBlock) {
	kvc.FreeBlockCnt--
	// ... existing linked list logic unchanged ...
}
```

5. Update `countFreeBlocks`:
```go
func (kvc *KVCacheState) countFreeBlocks() int64 {
	return kvc.FreeBlockCnt
}
```

6. Update `UsedBlocks` accessor (derived, read-only):
```go
func (kvc *KVCacheState) UsedBlocks() int64 { return kvc.TotalBlocks - kvc.FreeBlockCnt }
```

7. Remove all `kvc.UsedBlockCnt++` and `kvc.UsedBlockCnt--` lines throughout the file:
   - `cache.go:251` (cached block inline commit) — remove `kvc.UsedBlockCnt++`
   - `cache.go:318` (new block allocation) — remove `kvc.UsedBlockCnt++`
   - `cache.go:394` (rollback new blocks) — remove `kvc.UsedBlockCnt--`
   - `cache.go:407` (rollback cached blocks) — remove `kvc.UsedBlockCnt--`
   - `cache.go:443` (commitCachedBlocks) — remove `kvc.UsedBlockCnt++`
   - `cache.go:467` (ReleaseKVBlocks) — remove `kvc.UsedBlockCnt--`

These are all now redundant because `appendToFreeList` and `removeFromFreeList` handle the counter.

8. Update `commitCachedBlocks` comment (line 424-436): remove references to "rollback" since rollback is being eliminated. Simplify to note that `removeFromFreeList` handles the counter.

9. Update package doc comment (line 1-5): replace "transactional allocation with rollback" with "check-then-act allocation (vLLM parity)."

10. Update `KVCacheState` struct comment (line 34-35): replace "tracks the number of used blocks" with "tracks free blocks directly via FreeBlockCnt (vLLM FreeKVCacheBlockQueue parity)."

**Step 4: Run test to verify it passes**

Run: `go test ./sim/kv/... -run TestFreeBlockCnt_DirectCounter -v`
Expected: PASS

**Step 5: Run full test suite to verify no regressions**

Run: `go test ./sim/... -count=1`
Expected: All tests pass. The `UsedBlocks()` accessor returns the same values as before (derived from `TotalBlocks - FreeBlockCnt` instead of `UsedBlockCnt` directly, but the values are identical when no drift exists).

**Step 6: Fix any tests that reference `UsedBlockCnt` directly**

Grep for `UsedBlockCnt` in test files. These are internal field accesses (structural — should use `UsedBlocks()` accessor). Update tiered_test.go lines that reference `gpu.UsedBlockCnt` to use `gpu.UsedBlocks()`.

**Step 7: Commit**

```bash
git add sim/kv/cache.go sim/kv/cache_test.go sim/kv/tiered_test.go
git commit -m "refactor(kv): replace UsedBlockCnt with direct FreeBlockCnt counter

Mirrors vLLM's FreeKVCacheBlockQueue.num_free_blocks pattern.
Counter is maintained by appendToFreeList/removeFromFreeList,
eliminating arithmetic derivation that can drift under partial
mutation bugs (#1061).

UsedBlocks() accessor is now derived as TotalBlocks - FreeBlockCnt
(read-only for callers, not source of truth for allocation)."
```

---

### Task 2: Add Decode Pre-Check

Add the missing decode path pre-check that mirrors vLLM's universal `get_num_blocks_to_allocate` gate. This is the primary bug fix — it prevents the #1061 block leak by returning `false` before any state mutation when the decode path cannot allocate.

**Files:**
- Modify: `sim/kv/cache.go:224-227` (decode branch in AllocateKVBlocks)
- Test: `sim/kv/cache_test.go`

**Step 1: Write the failing tests**

These tests reproduce the #1061 bug: a continuing request in decode fails allocation, and we verify that `RequestMap` is preserved (not deleted by rollback).

```go
func TestAllocateKVBlocks_DecodeFailure_PreservesExistingBlocks(t *testing.T) {
	// GIVEN a request that completed prefill and has 2 blocks allocated
	kvc := NewKVCacheState(3, 4) // 3 blocks total
	req := &sim.Request{
		ID:            "r1",
		InputTokens:   []int{1, 2, 3, 4, 5, 6, 7, 8},
		OutputTokens:  []int{100, 200},
		ProgressIndex: 8, // past prefill, in decode
	}
	// Allocate prefill (2 blocks for 8 tokens, blockSize=4)
	ok := kvc.AllocateKVBlocks(req, 0, 8, []int64{})
	require.True(t, ok)
	require.Len(t, kvc.RequestMap["r1"], 2)

	// Consume the last free block with a filler
	filler := &sim.Request{ID: "filler", InputTokens: []int{90, 91, 92, 93}}
	ok = kvc.AllocateKVBlocks(filler, 0, 4, []int64{})
	require.True(t, ok)
	require.Equal(t, int64(0), kvc.countFreeBlocks())

	// WHEN decode allocation fails (last block is full, 0 free blocks)
	ok = kvc.AllocateKVBlocks(req, 8, 9, []int64{})

	// THEN allocation returns false AND RequestMap is PRESERVED (not deleted)
	assert.False(t, ok, "decode allocation should fail")
	assert.Len(t, kvc.RequestMap["r1"], 2,
		"RequestMap must preserve existing blocks on decode failure (bug #1061: rollback deleted them)")
	assertBlockConservation(t, kvc)
}

func TestAllocateKVBlocks_DecodePreCheck_SucceedsWhenLastBlockHasSpare(t *testing.T) {
	// GIVEN a request in decode whose last block has spare capacity
	kvc := NewKVCacheState(2, 4)
	req := &sim.Request{
		ID:            "r1",
		InputTokens:   []int{1, 2, 3}, // 3 tokens = 1 partial block (3/4)
		OutputTokens:  []int{100},
		ProgressIndex: 3,
	}
	ok := kvc.AllocateKVBlocks(req, 0, 3, []int64{})
	require.True(t, ok)
	require.Equal(t, int64(1), kvc.countFreeBlocks()) // 1 free block remaining

	// Consume the last free block
	filler := &sim.Request{ID: "filler", InputTokens: []int{90, 91, 92, 93}}
	ok = kvc.AllocateKVBlocks(filler, 0, 4, []int64{})
	require.True(t, ok)
	require.Equal(t, int64(0), kvc.countFreeBlocks())

	// WHEN decode allocates 1 token into the partial block (spare=1, no new block needed)
	ok = kvc.AllocateKVBlocks(req, 3, 4, []int64{})

	// THEN succeeds even with 0 free blocks (token fits in existing partial block)
	assert.True(t, ok, "decode should succeed when last block has spare capacity")
	assertBlockConservation(t, kvc)
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/kv/... -run "TestAllocateKVBlocks_Decode(Failure|PreCheck)" -v`
Expected: `TestAllocateKVBlocks_DecodeFailure_PreservesExistingBlocks` FAILS — rollback deletes `RequestMap["r1"]`. The spare test may pass or fail depending on whether the token happens to fit before `popFreeBlock` is needed.

**Step 3: Add the decode pre-check**

In `sim/kv/cache.go`, replace the decode branch (currently lines 224-227):

```go
	} else {
		// request is in decode
		newTokens = append(newTokens, req.OutputTokens[startIndex-util.Len64(req.InputTokens)])
	}
```

With:

```go
	} else {
		// request is in decode
		newTokens = append(newTokens, req.OutputTokens[startIndex-util.Len64(req.InputTokens)])

		// Decode pre-check: if the last block is full and no free blocks exist,
		// fail fast without any state mutation. Mirrors vLLM's universal pre-check
		// (kv_cache_manager.py:334-336 / single_type_kv_cache_manager.py:95-101).
		//
		// For decode, get_num_blocks_to_allocate returns max(cdiv(tokens, blockSize) - len(blocks), 0).
		// This simplifies to: last block full → need 1 new block; last block has spare → need 0.
		// Note: !hasBlocks or len(ids)==0 for a decode request is structurally
		// unreachable — ProgressIndex >= len(InputTokens) requires a prior
		// successful prefill allocation that populated RequestMap. If it occurs,
		// the allocation loop's popFreeBlock panic will fire with a clear message.
		if ids, hasBlocks := kvc.RequestMap[reqID]; hasBlocks && len(ids) > 0 {
			lastBlk := kvc.Blocks[ids[len(ids)-1]]
			if util.Len64(lastBlk.Tokens) == kvc.BlockSizeTokens && kvc.countFreeBlocks() == 0 {
				logrus.Debugf("KV cache full: cannot allocate decode block for req %s (last block full, 0 free)", reqID)
				return false
			}
		}
	}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/kv/... -run "TestAllocateKVBlocks_Decode(Failure|PreCheck)" -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `go test ./sim/... -count=1`
Expected: All tests pass. Existing prefill pre-check tests are unaffected.

**Step 6: Commit**

```bash
git add sim/kv/cache.go sim/kv/cache_test.go
git commit -m "fix(kv): add decode pre-check to prevent #1061 block leak

Mirrors vLLM's universal check-then-act gate
(kv_cache_manager.py:334, single_type_kv_cache_manager.py:95-101).
Returns false before any state mutation when the last block is full
and no free blocks exist. Preserves RequestMap for continuing
requests, preventing the orphaned-block deadlock."
```

---

### Task 3: Remove Rollback Machinery

With both prefill and decode pre-checks in place, rollback is structurally unreachable. Remove it entirely and convert the `popFreeBlock() == nil` path to a panic.

**Files:**
- Modify: `sim/kv/cache.go:89-102` (delete `prependToFreeList`)
- Modify: `sim/kv/cache.go:228-230` (remove rollback tracking vars)
- Modify: `sim/kv/cache.go:245-258` (inline cached block commit — remove mutation tracking)
- Modify: `sim/kv/cache.go:296-306` (replace rollback with panic)
- Delete: `sim/kv/cache.go:361-413` (`cachedBlockMutation`, `newBlockMutation`, `rollbackAllocation`)
- Test: `sim/kv/cache_test.go`

**Step 1: Write the behavioral replacement test**

The old rollback tests (`TestAllocateKVBlocks_MidLoopFailure_RollsBackNewBlocks`, `TestAllocateKVBlocks_CachedBlockRollback_OnNewBlockFailure`, `TestAllocateKVBlocks_Rollback_PreservesFreeListOrder`) verify rollback behavior. Under check-then-act, these scenarios should be caught at the pre-check and never reach the allocation loop. Write a replacement test that verifies the pre-check catches the exact scenarios that previously triggered rollback.

```go
func TestAllocateKVBlocks_PreCheck_CatchesCachedBlockBudgetExhaustion(t *testing.T) {
	// Previously: cached blocks consumed free budget mid-loop, triggering rollback.
	// Now: pre-check must account for cached blocks and reject upfront.
	//
	// Setup: 4 blocks (blockSize=2), cached prefix [1,2,3,4] = 2 blocks.
	// Filler consumes 1 block → 3 free.
	// Request needs 2 cached + 2 new = would consume 4 from free list.
	// But cached blocks with RefCount=0 are in free list, so claiming them
	// reduces free count. Pre-check must account for this.
	kvc := NewKVCacheState(4, 2)
	req1 := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	kvc.AllocateKVBlocks(req1, 0, 4, []int64{})
	kvc.ReleaseKVBlocks(req1)

	filler := &sim.Request{ID: "filler", InputTokens: []int{90, 91}}
	kvc.AllocateKVBlocks(filler, 0, 2, []int64{})

	// WHEN allocating with cached prefix that exhausts free budget
	req2 := &sim.Request{ID: "r2", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8}}
	cached := kvc.GetCachedBlocks(req2.InputTokens)
	require.Len(t, cached, 2)

	freeBefore := kvc.countFreeBlocks()
	ok := kvc.AllocateKVBlocks(req2, 4, 8, cached)

	// THEN fails AND no state was mutated (no CacheHits change, no RequestMap entry)
	assert.False(t, ok)
	assert.Equal(t, freeBefore, kvc.countFreeBlocks(),
		"free block count must be unchanged after pre-check rejection")
	assert.Empty(t, kvc.RequestMap["r2"],
		"no RequestMap entry should exist after pre-check rejection")
	assertBlockConservation(t, kvc)
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/kv/... -run TestAllocateKVBlocks_PreCheck_CatchesCachedBlockBudgetExhaustion -v`
Expected: Currently passes (the pre-check at line 220 catches this for prefill). This test documents the new behavioral contract. Run it to confirm green before removing rollback.

**Step 3: Enhance prefill pre-check to account for cached block free-list consumption**

The existing prefill pre-check at `cache.go:220` computes `numNewBlocks` but does **not** account for cached blocks that sit on the free list with `RefCount == 0`. When these are claimed, they leave the free list, reducing `countFreeBlocks()`. The pre-check must include this.

This mirrors vLLM's `num_evictable_blocks` calculation (`single_type_kv_cache_manager.py:124-127`):
```python
num_evictable_blocks = sum(
    blk.ref_cnt == 0 and not blk.is_null
    for blk in new_computed_blocks[num_skipped_new_computed_blocks:]
)
return num_new_blocks + num_evictable_blocks
```

**Replace** the existing pre-check at lines 219-223 (`if numNewBlocks > kvc.countFreeBlocks()`) with the enhanced version below. The new check adds `cachedFromFreeList` to the budget:

```go
	// Account for cached blocks that will leave the free list when claimed.
	// Mirrors vLLM's num_evictable_blocks (single_type_kv_cache_manager.py:124-127).
	// Invariant: cachedBlocks (from GetCachedBlocks) only contains blocks not yet
	// owned by the calling request — they are prefix-cache hits with RefCount >= 0.
	// Blocks with !InUse (RefCount == 0) sit on the free list and will be removed
	// when claimed, reducing countFreeBlocks(). We must budget for this.
	var cachedFromFreeList int64
	for _, blockID := range cachedBlocks {
		blk := kvc.Blocks[blockID]
		if !blk.InUse {
			cachedFromFreeList++
		}
	}

	// Cannot allocate enough KV cache blocks
	if numNewBlocks+cachedFromFreeList > kvc.countFreeBlocks() {
		logrus.Debugf("KV cache full: cannot allocate %d new + %d cached blocks for req %s",
			numNewBlocks, cachedFromFreeList, req.ID)
		return false
	}
```

**Step 4: Remove rollback machinery**

In `sim/kv/cache.go`:

1. Delete `prependToFreeList` method (lines 89-102) — only used by rollback.

2. Remove rollback tracking variables (lines 228-230):
```go
	// DELETE these lines:
	var cachedMutations []cachedBlockMutation
	var newlyAllocated []newBlockMutation
```

3. In the cached block inline commit loop (lines 245-258), remove mutation tracking. Replace:
```go
			cachedMutations = append(cachedMutations, cachedBlockMutation{block: blk, wasInUse: wasInUse})
```
With nothing (delete the line). The `wasInUse` variable and its tracking are no longer needed.

4. In the new block allocation loop, remove mutation tracking. Replace the `popFreeBlock` nil check (lines 302-305):
```go
			blk := kvc.popFreeBlock()
			if blk == nil {
				kvc.rollbackAllocation(reqID, cachedMutations, newlyAllocated)
				return false
			}
```
With:
```go
			blk := kvc.popFreeBlock()
			if blk == nil {
				panic(fmt.Sprintf("popFreeBlock returned nil after pre-check passed for req %s: INV-4 violation", reqID))
			}
```

5. Remove the `originalHash` tracking (lines 296-301):
```go
			// DELETE these lines:
			var originalHash string
			if kvc.FreeHead != nil {
				originalHash = kvc.FreeHead.Hash
			}
```

6. Remove the `newlyAllocated` append (line 330):
```go
			// DELETE this line:
			newlyAllocated = append(newlyAllocated, newBlockMutation{block: blk, originalHash: originalHash})
```

7. Delete the rollback types and method entirely (lines 361-413):
   - `cachedBlockMutation` struct
   - `newBlockMutation` struct
   - `rollbackAllocation` method

8. Update `commitCachedBlocks` comment (lines 424-436): remove all references to rollback. Simplify to:
```go
// NOTE: With the check-then-act allocation pattern, rollback does not exist.
// This method is used by TieredKVCache before calling gpu.AllocateKVBlocks.
// The inner AllocateKVBlocks pre-check sees the reduced FreeBlockCnt (already
// decremented by removeFromFreeList during commitCachedBlocks), so the pre-check
// correctly accounts for the committed blocks. In BLIS's single-threaded DES,
// FreeBlockCnt cannot decrease between the inner pre-check and allocation loop.
```

**Step 5: Delete obsolete rollback tests, keep behavioral tests**

Delete these tests (they test rollback behavior that no longer exists):
- `TestAllocateKVBlocks_MidLoopFailure_RollsBackNewBlocks`
- `TestAllocateKVBlocks_CachedBlockRollback_OnNewBlockFailure`
- `TestAllocateKVBlocks_Rollback_PreservesFreeListOrder`

**Keep** `TestAllocateKVBlocks_FailedAllocation_CacheHitRateUnchanged` — this verifies a behavioral contract that survives the refactor: failed allocation must not change CacheHitRate. Under check-then-act, the pre-check rejects before any CacheHits/CacheMisses mutations. Update its name to `TestAllocateKVBlocks_PreCheckRejection_CacheHitRateUnchanged` and update its comment to reference the pre-check path instead of rollback.

Keep all other tests — they verify behavioral contracts (conservation, partial blocks, chunked prefill, no-warn output) that still apply.

**Step 6: Run full test suite**

Run: `go test ./sim/... -count=1`
Expected: All remaining tests pass. The deleted tests are gone; the new pre-check tests validate the same scenarios via pre-check rejection instead of rollback.

**Step 7: Run linter**

Run: `golangci-lint run ./sim/kv/...`
Expected: Clean. No unused imports, no dead code.

**Step 8: Commit**

```bash
git add sim/kv/cache.go sim/kv/cache_test.go
git commit -m "refactor(kv): remove rollback machinery, convert to check-then-act

Removes rollbackAllocation, cachedBlockMutation, newBlockMutation,
and prependToFreeList (~60 lines). Pre-check now accounts for
cached-block free-list consumption (mirrors vLLM's
num_evictable_blocks). Post-pre-check popFreeBlock nil is a panic
(INV-4 invariant violation, structurally unreachable in
single-threaded DES).

Fixes #1061 — the block leak was caused by rollbackAllocation
deleting RequestMap for continuing requests. With rollback removed,
the bug class is eliminated entirely."
```

---

### Task 4: Add verifyBlockConservation Debug Assertion

Add a method for step-boundary conservation assertions. This catches any future block accounting bugs at the step they first occur, not thousands of steps later when the system deadlocks.

**Files:**
- Modify: `sim/kv/cache.go` (add method)
- Modify: `sim/kv/tiered.go` (add delegation method, not on interface)
- Test: `sim/kv/cache_test.go`

**Step 1: Write the failing test**

```go
func TestverifyBlockConservation_DetectsOrphanedBlocks(t *testing.T) {
	// GIVEN a KV cache in a consistent state
	kvc := NewKVCacheState(10, 4)
	req := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	kvc.AllocateKVBlocks(req, 0, 4, []int64{})

	// THEN conservation holds
	err := kvc.verifyBlockConservation()
	assert.NoError(t, err)

	// WHEN we artificially orphan a block (simulate the old bug)
	kvc.FreeBlockCnt-- // decrement free count without actually removing from list

	// THEN conservation is violated
	err = kvc.verifyBlockConservation()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "conservation")
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/kv/... -run TestverifyBlockConservation -v`
Expected: FAIL — method does not exist.

**Step 3: Implement verifyBlockConservation**

In `sim/kv/cache.go`:

```go
// verifyBlockConservation walks the free list and RequestMap independently
// to verify INV-4: freeListLen + sum(len(RequestMap[r])) == TotalBlocks.
// Returns nil if conservation holds, or an error describing the violation.
// Intended for debug-mode step-boundary assertions.
func (kvc *KVCacheState) verifyBlockConservation() error {
	// Walk the free list to get an independent count
	freeListLen := int64(0)
	node := kvc.FreeHead
	for node != nil {
		freeListLen++
		node = node.NextFree
	}

	// Count blocks owned by all requests
	ownedBlocks := int64(0)
	for _, ids := range kvc.RequestMap {
		ownedBlocks += int64(len(ids))
	}

	// Note: ownedBlocks may exceed TotalBlocks - freeListLen when blocks
	// are shared via prefix caching (RefCount > 1). Use block-level InUse
	// counting for a precise check.
	inUseCount := int64(0)
	for _, blk := range kvc.Blocks {
		if blk.InUse {
			inUseCount++
		}
	}

	if freeListLen+inUseCount != kvc.TotalBlocks {
		return fmt.Errorf("block conservation violated: freeList=%d + inUse=%d != total=%d",
			freeListLen, inUseCount, kvc.TotalBlocks)
	}

	if freeListLen != kvc.FreeBlockCnt {
		return fmt.Errorf("FreeBlockCnt drift: counter=%d, actual free list=%d",
			kvc.FreeBlockCnt, freeListLen)
	}

	return nil
}
```

**Note:** Do NOT add `verifyBlockConservation` to the `KVStore` interface (R13: interfaces must accommodate 2+ implementations without implementation-specific methods). Keep it as a method on `*KVCacheState` only. Callers that need debug assertions use a type assertion:

```go
if verifier, ok := kvCache.(*kv.KVCacheState); ok {
    if err := verifier.verifyBlockConservation(); err != nil {
        logrus.Errorf("INV-4 violation: %v", err)
    }
}
```

Add a delegation method on `TieredKVCache` for convenience (not on the interface):

```go
func (t *TieredKVCache) verifyBlockConservation() error {
	return t.gpu.verifyBlockConservation()
}
```

**Step 4: Run tests**

Run: `go test ./sim/kv/... -run TestverifyBlockConservation -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `go test ./sim/... -count=1`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add sim/kv/cache.go sim/kv/tiered.go sim/kv/cache_test.go
git commit -m "feat(kv): add verifyBlockConservation debug assertion

Walks free list and block InUse flags independently to verify INV-4.
Also detects FreeBlockCnt drift. Intended for step-boundary
assertions in debug mode."
```

---

### Task 5: Fix Incorrect Comment in processCompletions

The comment at `simulator.go:688-690` claims "AllocateKVBlocks only modifies RequestMap on success." This was false (rollback deleted it), but is now true under check-then-act. Fix the comment to document the new guarantee.

**Files:**
- Modify: `sim/simulator.go:686-690`

**Step 1: Fix the comment**

Replace:

```go
		// ReleaseKVBlocks is safe even when the final-token allocation failed:
		// AllocateKVBlocks only modifies RequestMap on success, so Release
		// frees exactly the blocks from prior successful allocations.
```

With:

```go
		// ReleaseKVBlocks is safe even when the final-token allocation failed:
		// the decode pre-check returns false before any state mutation (check-then-act
		// pattern, matching vLLM kv_cache_manager.py:334-336), so RequestMap is
		// preserved and Release frees all blocks from prior successful allocations.
```

**Step 2: Commit**

```bash
git add sim/simulator.go
git commit -m "docs(sim): fix processCompletions comment to reflect check-then-act

The comment claimed AllocateKVBlocks only modifies RequestMap on
success. This was false under rollback (#1061 root cause) but is
now provably true under check-then-act."
```

---

### Task 6: Add Preemption Retry Regression Test

This test directly reproduces #1061 leak path 1: `preemptForTokens` retry after decode failure. Under the old rollback, the retry would orphan blocks. Under check-then-act, `RequestMap` is preserved through the retry cycle.

**Files:**
- Test: `sim/kv/cache_test.go` or `sim/batch_formation_test.go`

**Step 1: Write the regression test**

```go
func TestPreemptForTokens_RetryPreservesRequestMap(t *testing.T) {
	// GIVEN: Simulates preemptForTokens retry cycle.
	// Request r1 has 2 prefill blocks. Cache is full. Decode fails.
	// Another request is evicted, freeing blocks. Retry succeeds.
	// The retry must see r1's original 2 blocks in RequestMap.
	kvc := NewKVCacheState(4, 4) // 4 blocks, blockSize=4

	r1 := &sim.Request{
		ID:            "r1",
		InputTokens:   []int{1, 2, 3, 4, 5, 6, 7, 8},
		OutputTokens:  []int{100},
		ProgressIndex: 8,
	}
	// Prefill r1: 2 blocks
	ok := kvc.AllocateKVBlocks(r1, 0, 8, []int64{})
	require.True(t, ok)
	require.Len(t, kvc.RequestMap["r1"], 2)

	r2 := &sim.Request{
		ID:          "r2",
		InputTokens: []int{10, 20, 30, 40, 50, 60, 70, 80},
	}
	// Prefill r2: 2 blocks (cache now full)
	ok = kvc.AllocateKVBlocks(r2, 0, 8, []int64{})
	require.True(t, ok)
	require.Equal(t, int64(0), kvc.countFreeBlocks())

	// WHEN decode for r1 fails (last block full, 0 free)
	ok = kvc.AllocateKVBlocks(r1, 8, 9, []int64{})
	assert.False(t, ok)

	// THEN RequestMap["r1"] still has 2 blocks (not deleted)
	assert.Len(t, kvc.RequestMap["r1"], 2,
		"decode failure must not delete RequestMap — this was the #1061 bug")

	// Simulate eviction: release r2
	kvc.ReleaseKVBlocks(r2)
	require.Equal(t, int64(2), kvc.countFreeBlocks())

	// WHEN retry succeeds
	ok = kvc.AllocateKVBlocks(r1, 8, 9, []int64{})
	assert.True(t, ok, "retry after eviction should succeed")

	// THEN r1 has 3 blocks: 2 original + 1 new decode block (not just 1)
	assert.Len(t, kvc.RequestMap["r1"], 3,
		"retry must see original 2 blocks + 1 new decode block")
	assertBlockConservation(t, kvc)
}
```

**Step 2: Run test**

Run: `go test ./sim/kv/... -run TestPreemptForTokens_RetryPreservesRequestMap -v`
Expected: PASS (the decode pre-check preserves RequestMap).

**Step 3: Commit**

```bash
git add sim/kv/cache_test.go
git commit -m "test(kv): add preemption retry regression test for #1061

Reproduces leak path 1: decode failure → eviction → retry.
Verifies RequestMap is preserved through the retry cycle
under check-then-act (would have caught #1061 rollback bug)."
```

---

### Task 7: Final Audit and Full Suite Verification

**Step 1: Grep for any remaining references to removed code**

Run: `grep -rn 'rollback\|UsedBlockCnt\|prependToFreeList\|cachedBlockMutation\|newBlockMutation' sim/`
Expected: No hits in production code. Test comments may reference them for historical context — that's fine.

**Step 2: Run full test suite**

Run: `go test ./sim/... -count=1`
Expected: All tests pass.

**Step 3: Run linter**

Run: `golangci-lint run ./...`
Expected: Clean.

**Step 4: Verify test coverage on the changed file**

Run: `go test ./sim/kv/... -coverprofile=coverage.out && go tool cover -func=coverage.out | grep cache.go`
Expected: `AllocateKVBlocks` function coverage should be ≥ 80%.

**Step 5: Commit any cleanup**

If any remaining references or lint issues are found, fix and commit.

---

## PR Description Template

The PR description must include the code proofs from the design doc (`docs/plans/2026-04-16-check-then-act-kv-design.md`), section "Code Proofs: Behavioral Equivalence with vLLM", with inline vLLM source citations. These are:

1. **Proof 1:** Universal pre-check — `kv_cache_manager.py:325-336` ↔ BLIS prefill+decode pre-checks
2. **Proof 2:** No rollback mechanism — `block_pool.py:299-311` ValueError ↔ BLIS panic
3. **Proof 3:** Preemption loop preserves request state — `scheduler.py:812-863` ↔ BLIS `preemptForTokens`
4. **Proof 4:** processCompletions safety — `free()` contract
5. **Proof 5:** Direct free counter — `FreeKVCacheBlockQueue.num_free_blocks` ↔ BLIS `FreeBlockCnt`
