# Fix TieredKVCache Partial-Improvement Block Commitment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix a bug where CPU-reloaded blocks in the tiered KV cache are left unprotected on the GPU free list and silently lost or given wrong prefix hashes when allocation only partially succeeds.

**The problem today:** When `TieredKVCache.AllocateKVBlocks` reloads blocks from CPU to GPU but the reload only covers part of the requested range, the reloaded blocks are left on the GPU free list with `RefCount=0` and `InUse=false`. The subsequent call to `gpu.AllocateKVBlocks` for the remaining range ignores those blocks (because the request already has a `RequestMap` entry), and `popFreeBlock()` can evict them — destroying their hashes. Even if they survive, the fresh blocks allocated for the tail range chain `prevHash` from the pre-reload state, producing wrong prefix hashes that break future cache lookups.

**What this PR adds:**
1. **Eviction protection for partially-reloaded blocks** — before allocating fresh blocks for the uncached tail `[newStart, endIndex)`, the reloaded prefix blocks `[startIndex, newStart)` are committed (RefCount++, InUse=true, removed from free list) so they cannot be stolen by the subsequent fresh allocation.
2. **Correct prefix hash chain** — committed reloaded blocks are appended to `RequestMap[req.ID]` first, so the fresh blocks' `prevHash` correctly chains from the last reloaded block rather than the stale pre-reload state.

**Why this matters:** This closes the only remaining path where a CPU reload can corrupt a running request's KV state. The full-reload path (line 207–230 of `tiered.go`) already does this correctly; this PR closes the partial-improvement branch to the same standard. The commit-before-allocate pattern mirrors vLLM v1's new-request allocation path (`allocate_new_computed_blocks()` before `allocate_new_blocks()`); the running-request CPU-reload scenario is BLIS-specific (vLLM v1 running requests never re-run `get_computed_blocks()`).

**Architecture:** The fix is entirely in `sim/kv/tiered.go:AllocateKVBlocks`, partial-improvement branch (line 232–233). It reuses the existing `commitCachedBlocks` helper in `cache.go` with the same ceiling-division startBlock logic already present in the full-reload branch (line 222–225). No new types, interfaces, or packages.

**Source:** GitHub issue #640

**Closes:** Fixes #640

**Behavioral Contracts:** See Part 1, Section B below

---

## PART 1: Design Validation

### A) Executive Summary

This PR fixes a single-branch gap in `TieredKVCache.AllocateKVBlocks` (`sim/kv/tiered.go`). The method has three outcome branches after a CPU reload:

1. **Full-reload** (`newStart >= endIndex`): all blocks cached → commit and return true ✅ already correct
2. **Partial-improvement** (`newStart > startIndex && newStart < endIndex`): some blocks cached, some need fresh allocation → **BUG: reloaded blocks not committed** ← this PR fixes
3. **No improvement** (`newStart <= startIndex`): reload freed space only → retry with original params ✅ correct (no commit needed)

The fix in branch 2: commit `newCached[startBlock:newStart/blockSize]` via `commitCachedBlocks` **before** delegating to `gpu.AllocateKVBlocks` for the tail. This closes the R1 silent-data-loss violation. The pattern mirrors vLLM v1's new-request allocation sequence (commit cached blocks before fresh block allocation); the running-request CPU-reload scenario is BLIS-specific and has no direct vLLM v1 analog.

**DEVIATION flags:** None — the fix is fully localized to one branch.

**Adjacent blocks:** `KVCacheState.commitCachedBlocks` (already exists), `KVCacheState.AllocateKVBlocks` (called for fresh tail), `TieredKVCache` tests in `sim/kv/tiered_test.go`.

### B) Behavioral Contracts

**BC-1: Partial-reload eviction protection**
- GIVEN a running request (present in `RequestMap`) whose GPU allocation fails
- WHEN `reloadPrefixFromCPU` reloads blocks covering only `[startIndex, newStart)` where `newStart < endIndex`
- THEN the reloaded blocks are eviction-protected (RefCount ≥ 1, InUse=true, removed from free list) before any fresh block allocation begins
- MECHANISM: `commitCachedBlocks(req.ID, newCached[startBlock:endBlock])` called before `gpu.AllocateKVBlocks`

**BC-2: Partial-reload hash chain integrity**
- GIVEN a running request where partial reload commits blocks `[startBlock, newStart/blockSize)`
- WHEN `gpu.AllocateKVBlocks(req, newStart, endIndex, newCached)` allocates the fresh tail
- THEN fresh blocks' prefix hashes chain from the last committed reloaded block (not the pre-reload state)
- MECHANISM: `commitCachedBlocks` appends block IDs to `RequestMap[req.ID]`; `AllocateKVBlocks` reads `existingIDs[len-1].Hash` as `prevHash` for fresh allocation

**BC-3: KV block conservation (INV-4)**
- GIVEN any call to `AllocateKVBlocks` on `TieredKVCache`
- WHEN the call returns (true or false)
- THEN `UsedBlockCnt + countFreeBlocks() == TotalBlocks`

**BC-4: No double-commit for running requests**
- GIVEN a running request with `startIndex > 0` (partially-filled last block at `ceil(startIndex/blockSize)-1`)
- WHEN partial-reload commits the range `[startBlock, newStart/blockSize)` where `startBlock = ceil(startIndex/blockSize)`
- THEN blocks already in `RequestMap[req.ID]` (indices `< startBlock`) are NOT re-committed (RefCount not double-incremented)
- MECHANISM: ceiling division `startBlock = (startIndex + BlockSize - 1) / BlockSize` skips the partially-filled last block, identical to the full-reload branch

**BC-5: New-request partial reload**
- GIVEN a new request (not yet in `RequestMap`) whose GPU allocation fails
- WHEN partial reload commits `newCached[0:newStart/blockSize]` via `commitCachedBlocks`
- THEN the subsequent `gpu.AllocateKVBlocks(req, newStart, endIndex, newCached)` sees the request already in `RequestMap` and skips the cached-blocks loop (correct — committed blocks already accounted for)
- MECHANISM: `commitCachedBlocks` creates `RequestMap[req.ID]`; `AllocateKVBlocks` enters the `ok` branch

**BC-6: Full-reload path unchanged**
- GIVEN a reload where `newStart >= endIndex` (entire range covered)
- WHEN `AllocateKVBlocks` takes the full-reload branch
- THEN behavior is identical to pre-fix (the full-reload branch is not modified)

**BC-7: No-improvement path unchanged**
- GIVEN a reload where `newStart <= startIndex` (no new cache hits)
- WHEN `AllocateKVBlocks` takes the no-improvement branch
- THEN behavior is identical to pre-fix

### C) Component Interaction

```
TieredKVCache.AllocateKVBlocks
  │
  ├─ gpu.AllocateKVBlocks [first attempt, fails]
  │
  ├─ reloadPrefixFromCPU [reloads 0..N blocks onto GPU free list]
  │
  ├─ gpu.GetCachedBlocks [discovers newCached, newStart]
  │
  ├─ [partial-improvement branch, newStart > startIndex, newStart < endIndex]
  │   ├─ commitCachedBlocks(req.ID, newCached[startBlock:newStart/blockSize])  ← NEW
  │   │     sets RefCount++, InUse=true, removes from free list, appends to RequestMap
  │   └─ gpu.AllocateKVBlocks(req, newStart, endIndex, newCached)
  │         sees req.ID in RequestMap → skips cached-blocks loop
  │         prevHash chains from last committed block → correct hashes
  │
  └─ [full-reload branch unchanged] commitCachedBlocks → return true
```

**State changes:** The partial-improvement branch now mutates `KVCacheState.RequestMap[req.ID]`, `KVBlock.RefCount`, `KVBlock.InUse`, `KVCacheState.UsedBlockCnt`, and free list — same mutations as the full-reload branch, but for a subset of blocks before delegating the tail.

**Extension friction:** 0 additional files needed. The fix is 5 lines in `tiered.go`.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|---|---|---|
| Issue #640: "commit cached blocks covering [startIndex, newStart) before allocating fresh blocks for [newStart, endIndex)" | Uses ceiling division `startBlock = ceil(startIndex/blockSize)` to skip the partially-filled last block | CORRECTION: the issue's description says "startIndex" but the full-reload branch already uses ceiling division to avoid double-committing the partial block. Applying the same logic here is correct. |

### E) Review Guide

**The tricky part:** The `startBlock` ceiling-division logic. For a running request, `startIndex` is the token index where the last allocation ended — which may land in the *middle* of a block (e.g., `startIndex=6, blockSize=4` → the partial block 1 is already in `RequestMap`). Using `ceil(startIndex/blockSize)` correctly starts the commit at block 2, not block 1. Floor division would double-commit block 1.

**What to scrutinize:** BC-4 (no double-commit). Verify the `startBlock` formula matches the full-reload branch exactly (line 222 of `tiered.go`). Also verify BC-5: after `commitCachedBlocks` creates `RequestMap[req.ID]` for a new request, `gpu.AllocateKVBlocks` now enters the `ok` branch — verify it doesn't re-process `newCached` as `cachedBlocks` (it passes `newCached` but the `ok` branch ignores it, which is correct).

**What's safe to skim:** The `cpuTier`, `reloadPrefixFromCPU`, `MirrorToCPU`, and `ReleaseKVBlocks` methods are unchanged.

**Known debt:** `commitCachedBlocks` does not track mutations for rollback. In the partial-improvement branch, if the subsequent `gpu.AllocateKVBlocks` fails **mid-allocation** (i.e., `popFreeBlock` returns nil inside the loop, triggering `rollbackAllocation`), `rollbackAllocation` calls `delete(kvc.RequestMap, reqID)` which wipes all `RequestMap` entries — including the ones `commitCachedBlocks` just appended. Those committed blocks retain `RefCount > 0` and `InUse=true` but are now orphaned (not in any RequestMap, not in the free list). For running requests this is worse: the original pre-existing blocks are also orphaned. This is a **new risk** introduced by this PR — the full-reload branch returns `true` immediately after `commitCachedBlocks` and never calls `AllocateKVBlocks` afterwards. Note: when `gpu.AllocateKVBlocks` fails at the **pre-check** (before any mutations), no rollback is triggered and the committed state is preserved correctly. The mid-allocation failure is a corner case (blocks run out mid-loop, after successfully popping some). Filed as follow-up issue; fixing it requires `commitCachedBlocks` to return its mutations for rollback tracking.

---

## PART 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/kv/tiered.go` — 5-line change in `AllocateKVBlocks` partial-improvement branch
- `sim/kv/tiered_test.go` — 2 new test functions (BC-1/BC-2, BC-3/BC-4)

**Key decisions:**
1. Reuse `commitCachedBlocks` (not inline) — identical semantics to full-reload branch, avoids duplication
2. Apply same `startBlock = ceil(startIndex/blockSize)` guard as full-reload branch for running requests
3. For new requests, `_, exists := t.gpu.RequestMap[req.ID]` check: new requests use `startBlock=0` (commit from block 0), since no partial block exists yet — same as full-reload new-request path (line 227–228)

**No dead code:** All new test functions exercise production paths directly.

### G) Task Breakdown

---

#### Task 1: Write failing tests for BC-1 and BC-2 (partial-reload block commitment)

**Contracts Implemented:** BC-1, BC-2, BC-3, BC-4

**Files:**
- Modify: `sim/kv/tiered_test.go`

**Step 1: Write failing test**

Context: The running-request test demonstrates BC-1/BC-2/BC-3/BC-4 by showing the behavioral difference between before and after the fix:
- **Without fix**: `gpu.AllocateKVBlocks` is called directly, `popFreeBlock` steals the reloaded block (h8 cleared from HashToBlock), allocation returns **true** but the prefix cache is silently corrupted.
- **With fix**: `commitCachedBlocks` protects the reloaded block first; tail allocation fails at pre-check (2 blocks needed, 1 free), returns **false**. h8 is preserved. The state is stable.

Note: the fix changes the return value from true→false. This is correct — a clean failure is better than silent cache corruption.

In `sim/kv/tiered_test.go`, add the following two imports (neither is currently present — the existing file only imports `testify/assert`):
```go
"github.com/inference-sim/inference-sim/sim/internal/hash"
"github.com/stretchr/testify/require"
```

Then add the test:

```go
// TestTieredKVCache_PartialReload_RunningRequest_BlocksCommitted verifies BC-1, BC-2, BC-3, BC-4:
// partial CPU reload for a running request must commit reloaded blocks before fresh allocation.
//
// Without the fix: gpu.AllocateKVBlocks is called directly; popFreeBlock steals the
// reloaded block (clears h8), overall allocation returns true but the prefix cache is
// silently corrupted — future requests cannot find the h8 cache entry.
//
// With the fix: commitCachedBlocks protects the reloaded block; tail allocation fails cleanly
// (returns false) because there aren't enough blocks, but h8 is preserved in HashToBlock.
func TestTieredKVCache_PartialReload_RunningRequest_BlocksCommitted(t *testing.T) {
	// Setup: 8-block GPU, blockSize=4, 10-block CPU
	// req1 holds blocks [0,1] (tokens 0..7)
	// req2 holds blocks [2,3], req3 holds blocks [4,5]
	// Free: [6,7] (2 free blocks)
	// CPU has hash h8 = HashBlock(block1.Hash, tokens[8:12])
	//
	// Call AllocateKVBlocks(req1, 8, 20, cached):
	//   First attempt: need ceil(12/4)=3 blocks, have 2 → fails
	//   After reload: h8 on free list (tail), newStart=12, partial improvement
	//   Without fix: popFreeBlock steals block 6 (h8 cleared), ok=true but cache corrupted
	//   With fix: commitCachedBlocks protects block 6 → 1 free; need 2 for tail → ok=false
	blockSize := int64(4)
	totalBlocks := int64(8)
	gpu := NewKVCacheState(totalBlocks, blockSize)

	tokens := make([]int, 20)
	for i := range tokens {
		tokens[i] = i + 10 // distinct values
	}

	req1 := &sim.Request{ID: "req1", InputTokens: tokens}
	require.True(t, gpu.AllocateKVBlocks(req1, 0, 8, nil)) // blocks [0,1]

	req2 := &sim.Request{ID: "req2", InputTokens: make([]int, 8)}
	require.True(t, gpu.AllocateKVBlocks(req2, 0, 8, nil)) // blocks [2,3]

	req3 := &sim.Request{ID: "req3", InputTokens: make([]int, 8)}
	require.True(t, gpu.AllocateKVBlocks(req3, 0, 8, nil)) // blocks [4,5]
	// Free: [6,7]

	tiered := NewTieredKVCache(gpu, 10, 0, 1.0, 0)

	// Hash for tokens[8:12] chaining from block[1].Hash
	prevHash1 := gpu.Blocks[gpu.RequestMap["req1"][1]].Hash
	h8 := hash.HashBlock(prevHash1, tokens[8:12])
	tiered.cpu.store(h8, tokens[8:12])

	// BC-3 conservation before
	require.Equal(t, totalBlocks, gpu.UsedBlocks()+gpu.countFreeBlocks())

	// WHEN allocating tokens 8..20 for req1 (running request)
	cached := gpu.GetCachedBlocks(tokens)
	ok := tiered.AllocateKVBlocks(req1, 8, 20, cached)

	// THEN overall allocation fails cleanly (pre-check: need 2 tail blocks, 1 free after commit)
	// Without fix: ok=true (this assertion would FAIL), h8 cleared
	require.False(t, ok, "allocation must fail cleanly — not silently corrupt the prefix cache")

	// THEN BC-3: KV conservation holds
	require.Equal(t, totalBlocks, gpu.UsedBlocks()+gpu.countFreeBlocks())

	// THEN BC-1: reloaded block hash is preserved (not stolen by popFreeBlock)
	// Without fix: h8 would be cleared from HashToBlock by popFreeBlock
	_, found := gpu.HashToBlock[h8]
	require.True(t, found, "BC-1: reloaded block hash must be preserved in HashToBlock")

	// THEN BC-1: reloaded block is committed (eviction-protected in RequestMap)
	require.Equal(t, 3, len(gpu.RequestMap["req1"]), "req1 must have original 2 blocks + 1 committed reloaded block")
	reloadedID := gpu.RequestMap["req1"][2]
	reloadedBlk := gpu.Blocks[reloadedID]
	require.True(t, reloadedBlk.InUse, "BC-1: reloaded block must be InUse")
	require.Positive(t, reloadedBlk.RefCount, "BC-1: reloaded block RefCount must be > 0")

	// THEN BC-2: future GetCachedBlocks finds 3 blocks (prefix cache intact for future requests)
	futureCached := gpu.GetCachedBlocks(tokens)
	require.GreaterOrEqual(t, len(futureCached), 3, "BC-2: prefix cache must find all 3 cached blocks")
}
```

**Step 2: Run test to verify it fails**

```bash
cd .worktrees/fix-tiered-partial-improvement
go test ./sim/kv/... -run TestTieredKVCache_PartialReload_RunningRequest -v
```

Expected: FAIL — `require.False(t, ok)` fails because without the fix, `ok=true` (reloaded block stolen by `popFreeBlock`, allocation succeeds with corrupt cache). The h8 assertions also fail since h8 was cleared.

**Step 3: Implement minimal fix**

Context: Add a `commitCachedBlocks` call in the partial-improvement branch of `AllocateKVBlocks`, using the same ceiling-division startBlock logic as the full-reload branch.

In `sim/kv/tiered.go`, replace lines 232–233 (the comment and return statement; do NOT touch line 231 which is the `}` closing the full-reload branch):

```go
			// More cache hits after reload — retry with reduced allocation range
			return t.gpu.AllocateKVBlocks(req, newStart, endIndex, newCached)
```

with:

```go
			// Partial improvement: commit reloaded prefix blocks before allocating tail.
			// Mirrors full-reload branch: commit-before-allocate prevents popFreeBlock
			// from evicting the just-reloaded blocks during fresh allocation.
			// For running requests: startBlock uses ceiling division to skip the
			// partially-filled last block already in RequestMap (same logic as line 222).
			// For new requests: startBlock=0 (no existing partial block).
			newStartBlock := newStart / t.gpu.BlockSize()
			if _, exists := t.gpu.RequestMap[req.ID]; exists {
				startBlock := (startIndex + t.gpu.BlockSize() - 1) / t.gpu.BlockSize()
				if startBlock < newStartBlock {
					t.gpu.commitCachedBlocks(req.ID, newCached[startBlock:newStartBlock])
				}
			} else {
				t.gpu.commitCachedBlocks(req.ID, newCached[:newStartBlock])
			}
			return t.gpu.AllocateKVBlocks(req, newStart, endIndex, newCached)
```

**Step 4: Run test to verify it passes**

```bash
go test ./sim/kv/... -run TestTieredKVCache_PartialReload_RunningRequest -v
```

Expected: PASS

**Step 5: Run lint**

```bash
golangci-lint run ./sim/kv/...
```

Expected: zero new issues

**Step 6: Commit**

```bash
git add sim/kv/tiered.go sim/kv/tiered_test.go
git commit -m "fix(kv): commit reloaded prefix blocks in partial-improvement path (BC-1, BC-2)

- Before fix: partial CPU reload left reloaded blocks on GPU free list
  with RefCount=0; subsequent popFreeBlock could evict them and clear
  their hashes (R1 silent data loss). Fresh blocks also chained prevHash
  from the pre-reload state, producing incorrect prefix hashes.
- After fix: commitCachedBlocks() called for newCached[startBlock:newStartBlock]
  before delegating fresh allocation to gpu.AllocateKVBlocks, mirroring
  the full-reload branch (line 207-230) and vLLM v1 commit-before-allocate.
- Uses same ceiling-division startBlock as full-reload branch (line 222)
  to avoid double-committing the partially-filled last block for running
  requests (BC-4).

Fixes #640

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 2: Write failing test for new-request partial reload and conservation invariant

**Contracts Implemented:** BC-5, BC-3

**Files:**
- Modify: `sim/kv/tiered_test.go`

**Step 1: Write failing test**

Context: For new requests, the partial-improvement bug also manifests but differently. Without the fix, `AllocateKVBlocks` is called with req.ID NOT in RequestMap, so the `cachedBlocks` argument (newCached) IS processed inside the call — the h0 block is committed there. With the fix, `commitCachedBlocks` creates the RequestMap entry first, then `AllocateKVBlocks` skips the cached-blocks loop (req in RequestMap). Both paths commit h0, but with the fix, h0 is committed BEFORE any fresh allocation, eliminating any window where it could be stolen by `popFreeBlock`. The test verifies INV-4 conservation and that h0 is properly committed.

```go
// TestTieredKVCache_PartialReload_NewRequest_Revised tests BC-5: new request partial reload.
func TestTieredKVCache_PartialReload_NewRequest_Revised(t *testing.T) {
	// 7-block GPU, blockSize=4, 10-block CPU
	// Fill 4 blocks (req-filler uses blocks [0,1,2,3]) → 3 free [4,5,6]
	// New request needs tokens 0..11 (3 blocks)
	// First AllocateKVBlocks(req, 0, 12, nil): free=3, numNewBlocks=3 → should PASS directly.
	// Need to set up so first attempt FAILS. Use tighter constraint:
	// Fill 5 blocks → 2 free. Need 3 blocks → fails. Reload gives 1 → 2 free for tail → pass.

	blockSize := int64(4)
	totalBlocks := int64(7)
	gpu := NewKVCacheState(totalBlocks, blockSize)

	tokens := make([]int, 12)
	for i := range tokens {
		tokens[i] = i + 100
	}

	// Fill 5 blocks using two filler requests
	f1 := &sim.Request{ID: "f1", InputTokens: make([]int, 12)}
	require.True(t, gpu.AllocateKVBlocks(f1, 0, 12, nil)) // blocks [0,1,2]
	f2 := &sim.Request{ID: "f2", InputTokens: make([]int, 8)}
	require.True(t, gpu.AllocateKVBlocks(f2, 0, 8, nil)) // blocks [3,4]
	// 2 free: [5,6]

	tiered := NewTieredKVCache(gpu, 10, 0, 1.0, 0)

	// CPU has block for tokens[0:4]
	h0 := hash.HashBlock("", tokens[0:4])
	tiered.cpu.store(h0, tokens[0:4])

	req := &sim.Request{ID: "newreq", InputTokens: tokens}

	// WHEN allocating tokens 0..12 (3 blocks needed, 2 free → fails → reload h0 → commit(1) → 1 free for tail needing 2 → fails)
	// Note: the fix changes NEW-REQUEST behavior: without fix, block h0 is committed inline
	// inside AllocateKVBlocks but rolled back when tail allocation fails; with fix,
	// commitCachedBlocks commits h0 outside rollback tracking — it stays committed (BC-5).
	ok := tiered.AllocateKVBlocks(req, 0, 12, nil)

	// THEN overall allocation fails (tail needs 2 blocks, only 1 free after commit)
	require.False(t, ok)

	// THEN INV-4 conservation holds (BC-3)
	require.Equal(t, totalBlocks, gpu.UsedBlocks()+gpu.countFreeBlocks())

	// THEN BC-5: h0 is preserved in HashToBlock (not cleared)
	_, found := gpu.HashToBlock[h0]
	require.True(t, found, "BC-5: h0 block must be preserved in HashToBlock")

	// THEN BC-5: with fix, the reloaded block is committed in RequestMap (not rolled back)
	// Without fix: rollbackAllocation deletes RequestMap["newreq"] entirely (0 blocks).
	// With fix: commitCachedBlocks commits block outside rollback — RequestMap["newreq"] has 1 block.
	require.Equal(t, 1, len(gpu.RequestMap["newreq"]), "BC-5: with fix, committed block stays in RequestMap even on tail failure")
	committedID := gpu.RequestMap["newreq"][0]
	committedBlk := gpu.Blocks[committedID]
	require.Equal(t, h0, committedBlk.Hash, "BC-5: committed block must have h0 hash")
	require.True(t, committedBlk.InUse, "BC-5: committed block must be InUse")
}
```

**Step 2: Run test (fix from Task 1 is already applied)**

```bash
go test ./sim/kv/... -run TestTieredKVCache_PartialReload_NewRequest_Revised -v
```

Expected: PASS — the fix from Task 1 is already in place. This confirms BC-5 behavior.
Note: Without the fix, `rollbackAllocation` would delete `RequestMap["newreq"]`, so `len==0` and `require.Equal(t, 1, ...)` would fail — validating the test is meaningful.

**Step 3: The fix was already applied in Task 1 — verify test passes**

```bash
go test ./sim/kv/... -run TestTieredKVCache_PartialReload -v
```

Expected: All PASS

**Step 4: Run full kv package tests**

```bash
go test ./sim/kv/... -v
```

Expected: All PASS

**Step 5: Run lint**

```bash
golangci-lint run ./sim/kv/...
```

Expected: zero new issues

**Step 6: Commit**

```bash
git add sim/kv/tiered_test.go
git commit -m "test(kv): add partial-reload block commitment tests (BC-3, BC-5)

- TestTieredKVCache_PartialReload_NewRequest_Revised: verifies new request
  partial reload commits prefix from block 0 with correct hash chain
- INV-4 conservation verified in both running and new-request tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 3: Run full test suite and verify no regressions

**Contracts Implemented:** All (regression verification)

**Files:** No changes

**Step 1: Run all kv tests**

```bash
go test ./sim/kv/... -count=1 -v 2>&1 | tail -20
```

Expected: All PASS, no failures

**Step 2: Run sim package tests (INV-4 integration)**

```bash
go test ./sim/... -count=1 2>&1 | tail -10
```

Expected: All PASS

**Step 3: Run full suite**

```bash
go test ./sim/... ./sim/kv/... ./sim/latency/... ./sim/cluster/... ./sim/workload/... -count=1
```

Expected: All PASS

**Step 4: Final lint**

```bash
golangci-lint run ./sim/kv/...
```

Expected: zero issues

**Step 5: No commit needed (no code changes)**

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|---|---|---|---|
| BC-1 eviction protection | Task 1 | Unit | `TestTieredKVCache_PartialReload_RunningRequest_BlocksCommitted`: verifies h8 preserved in HashToBlock, reloaded block in RequestMap with InUse=true |
| BC-2 hash chain integrity | Task 1 | Unit | Same test: `GetCachedBlocks` finds ≥3 blocks after partial reload |
| BC-3 KV conservation | Task 1, Task 2 | Invariant | Both tests verify `UsedBlocks + countFreeBlocks == TotalBlocks` |
| BC-4 no double-commit | Task 1 | Unit | Running-request test: `len(RequestMap["req1"])==3` (original 2 + 1 reloaded, not 4) |
| BC-5 new-request partial reload | Task 2 | Unit | `TestTieredKVCache_PartialReload_NewRequest_Revised` |
| BC-6 full-reload unchanged | Task 3 | Regression | Existing `TestTieredKVCache_*` tests in tiered_test.go |
| BC-7 no-improvement unchanged | Task 3 | Regression | Existing tests |

**Golden dataset:** No change to output format or metrics. Golden dataset regeneration not needed.

**Shared test infrastructure:** Uses `require` from `testify` — **must be added as a new import** (existing file uses `assert` only). Uses `hash` from `sim/internal/hash` — **must also be added**. No new helpers needed.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|---|---|---|---|---|
| Double-commit if `startBlock` computed incorrectly (floor instead of ceiling) | Medium | High | BC-4: running-request test verifies RequestMap has exactly 3 blocks (not 2 or 4); ceiling formula matches line 222 | Task 1 |
| `startBlock >= newStartBlock` guard skips commit when 0 blocks need committing | Low | Low | Range check before commit; safe no-op | Task 1 |
| New-request path behavior changes after `commitCachedBlocks` creates RequestMap entry | Low | Low | BC-5 test verifies correct blocks, hashes, and conservation | Task 2 |
| `gpu.AllocateKVBlocks` fails mid-allocation (after some `popFreeBlock` calls) after partial commit — `rollbackAllocation` deletes entire RequestMap including committed blocks | Medium | Medium | New risk (not pre-existing). Pre-check failure (no mid-allocation pops) leaves state stable. Mid-loop failure is a corner case requiring tight KV pressure. Follow-up issue filed. | — |

---

## PART 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — reuses `commitCachedBlocks` helper unchanged
- [x] No feature creep — fix is exactly the 5-line partial-improvement branch change
- [x] No unexercised flags or interfaces — all paths tested
- [x] No partial implementations — fix is complete
- [x] No breaking changes — BC-6, BC-7 (existing paths) unchanged
- [x] No hidden global state impact — only `KVCacheState` mutated, same as before
- [x] All new code passes golangci-lint — no new identifiers introduced
- [x] Shared test helpers used — `require` from testify, existing pattern
- [x] CLAUDE.md: no new files, no CLI flags, no architecture changes — no update needed
- [x] No stale references in CLAUDE.md
- [x] Documentation DRY: no canonical sources modified
- [x] Deviation log reviewed — one deviation (startBlock for new request), justified
- [x] Each task produces working, testable code
- [x] Task dependencies correct: Task 1 (fix + test) → Task 2 (invariant test) → Task 3 (regression)
- [x] All contracts mapped to tasks
- [x] Golden dataset regeneration not needed
- [x] Construction site audit: no new struct fields, no new construction sites
- [x] Not part of a macro plan — N/A

**Antipattern rules:**
- [x] R1: commitCachedBlocks call is not silent — all mutations tracked in KVCacheState
- [x] R4: No struct fields added
- [x] R5: No new resource-allocating loops
- [x] R22: Pre-check in `gpu.AllocateKVBlocks` accounts for committed blocks (UsedBlockCnt updated before pre-check in the delegated call) ← verify carefully: `commitCachedBlocks` increments `UsedBlockCnt`, so when `AllocateKVBlocks` runs `numNewBlocks > countFreeBlocks()`, it sees the reduced free count correctly
- [x] R23: Partial-improvement path now applies equivalent transformation to full-reload path (commit before allocate)

---

## APPENDIX: File-Level Implementation Details

### File: `sim/kv/tiered.go`

**Purpose:** Fix the partial-improvement branch in `AllocateKVBlocks` to commit reloaded blocks before fresh allocation.

**Current code (lines 232–233 — line 231 is `}` closing the full-reload branch; do not modify it):**
```go
			// More cache hits after reload — retry with reduced allocation range
			return t.gpu.AllocateKVBlocks(req, newStart, endIndex, newCached)
```

**Replacement:**
```go
			// Partial improvement: commit reloaded prefix blocks before allocating tail.
			// Without this, reloaded blocks sit on the GPU free list with RefCount=0 and
			// can be evicted by the subsequent popFreeBlock calls in AllocateKVBlocks,
			// destroying their hashes (R1 silent data loss). Also fixes hash chain: fresh
			// blocks' prevHash must chain from the last reloaded block, which only happens
			// if those blocks are in RequestMap[req.ID] before the fresh allocation loop.
			newStartBlock := newStart / t.gpu.BlockSize()
			if _, exists := t.gpu.RequestMap[req.ID]; exists {
				// Running request: skip blocks already in RequestMap (ceiling division
				// avoids double-committing the partially-filled last block, same as line 222).
				startBlock := (startIndex + t.gpu.BlockSize() - 1) / t.gpu.BlockSize()
				if startBlock < newStartBlock {
					t.gpu.commitCachedBlocks(req.ID, newCached[startBlock:newStartBlock])
				}
			} else {
				// New request: commit all reloaded blocks from block 0.
				t.gpu.commitCachedBlocks(req.ID, newCached[:newStartBlock])
			}
			return t.gpu.AllocateKVBlocks(req, newStart, endIndex, newCached)
```

**Key implementation notes:**
- `newStartBlock = newStart / t.gpu.BlockSize()` is exact (newStart is always a multiple of BlockSize — it comes from `int64(len(newCached)) * t.gpu.BlockSize()`)
- The `startBlock < newStartBlock` guard prevents a zero-length commit (no-op when 0 new blocks were actually reloaded for the range)
- After `commitCachedBlocks` for a new request creates `RequestMap[req.ID]`, `gpu.AllocateKVBlocks` sees `ok=true` and skips the cached-blocks loop. It then only allocates fresh blocks for `[newStart, endIndex)`. This is correct: the committed blocks already cover `[0, newStart)`.
- The variable name `newStartBlock` is camelCase (Go convention). Do not use underscore form (`newStart_blockIdx`) — golangci-lint will flag it.

### File: `sim/kv/tiered_test.go`

**Purpose:** Two new test functions covering running-request and new-request partial reload scenarios.

**Imports needed (both must be added — neither is in the existing file):**
- `"github.com/inference-sim/inference-sim/sim/internal/hash"` — **NOT imported; must be added**
- `"github.com/stretchr/testify/require"` — **NOT imported; existing file uses `testify/assert`; must be added**
- `"github.com/inference-sim/inference-sim/sim"` — already imported
- `"github.com/stretchr/testify/assert"` — already imported (existing tests use this, don't remove)

**Test design notes:**
- Both tests use `countFreeBlocks()` directly (unexported method, tests are in same package `kv`)
- Block IDs in tests are deterministic (NewKVCacheState assigns sequential IDs 0..N-1)
- The running-request test constrains GPU to exactly `(needed - 1)` free blocks, forcing the first allocation to fail and the reload to provide exactly 1 block, leaving exactly enough for the tail
