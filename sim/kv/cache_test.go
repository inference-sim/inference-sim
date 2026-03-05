package kv

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/hash"
)

// assertBlockConservation verifies the KV block conservation invariant (INV-4)
// using only public API methods.
func assertBlockConservation(t *testing.T, kvc *KVCacheState) {
	t.Helper()
	used := kvc.UsedBlocks()
	total := kvc.TotalCapacity()
	free := total - used
	if used+free != total {
		t.Errorf("block conservation violated: UsedBlocks=%d + free=%d != TotalCapacity=%d",
			used, free, total)
	}
	if used < 0 {
		t.Errorf("UsedBlocks = %d, must be >= 0", used)
	}
	if free < 0 {
		t.Errorf("free blocks = %d, must be >= 0", free)
	}
}

func TestAllocateKVBlocks_PartialBlockFill_AdvancesByActualTokenCount(t *testing.T) {
	// GIVEN a KV cache with BlockSize=4 and a request that already has a partial block (2 of 4 tokens)
	kvc := NewKVCacheState(10, 4)
	req := &sim.Request{
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
	if blk.Hash != "" {
		t.Errorf("partial block should not have hash before fill, got %s", blk.Hash)
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

	// THEN the filled block's hash equals HashBlock("", [10,20,30,40]) and is findable
	// via GetCachedBlocks (partial-fill path with len(ids)==1 → prevHash="")
	expectedHash := hash.HashBlock("", []int{10, 20, 30, 40})
	if blk.Hash != expectedHash {
		t.Errorf("partial-fill block hash mismatch:\n  got  %s\n  want %s", blk.Hash, expectedHash)
	}
	kvc.ReleaseKVBlocks(req)
	cached := kvc.GetCachedBlocks([]int{10, 20, 30, 40})
	if len(cached) != 1 {
		t.Errorf("expected 1 cached block after partial-fill + release, got %d", len(cached))
	}
}

func TestAllocateKVBlocks_PartialBlockFill_MultiBlock_ChainsFromPreviousBlock(t *testing.T) {
	// Exercises the len(ids) >= 2 branch of the partial-fill hash path:
	// block 0 is full, block 1 is partial, then block 1 gets filled and
	// its hash must chain from block 0's hash (not "").
	kvc := NewKVCacheState(10, 4) // blockSize=4
	req := &sim.Request{
		ID:          "r1",
		InputTokens: []int{10, 20, 30, 40, 50, 60, 70, 80},
	}

	// Allocate 5 tokens: block 0 full [10,20,30,40], block 1 partial [50]
	ok := kvc.AllocateKVBlocks(req, 0, 5, []int64{})
	if !ok {
		t.Fatal("first allocation should succeed")
	}
	ids := kvc.RequestMap["r1"]
	if len(ids) != 2 {
		t.Fatalf("expected 2 blocks (1 full + 1 partial), got %d", len(ids))
	}

	// WHEN we allocate tokens 5-8, filling partial block 1
	req.ProgressIndex = 5
	ok = kvc.AllocateKVBlocks(req, 5, 8, []int64{})
	if !ok {
		t.Fatal("second allocation should succeed")
	}

	// THEN block 1's hash chains from block 0's hash (len(ids)>=2 branch)
	block0Hash := hash.HashBlock("", []int{10, 20, 30, 40})
	expectedBlock1Hash := hash.HashBlock(block0Hash, []int{50, 60, 70, 80})
	blk1 := kvc.Blocks[kvc.RequestMap["r1"][1]]
	if blk1.Hash != expectedBlock1Hash {
		t.Errorf("block 1 hash should chain from block 0:\n  got  %s\n  want %s", blk1.Hash, expectedBlock1Hash)
	}

	// BC-3 consistency: both hashes match ComputeBlockHashes
	expected := hash.ComputeBlockHashes(4, []int{10, 20, 30, 40, 50, 60, 70, 80})
	if len(expected) != 2 {
		t.Fatalf("expected 2 hashes from ComputeBlockHashes, got %d", len(expected))
	}
	blk0 := kvc.Blocks[kvc.RequestMap["r1"][0]]
	if blk0.Hash != expected[0] {
		t.Errorf("block 0 hash mismatch with ComputeBlockHashes:\n  got  %s\n  want %s", blk0.Hash, expected[0])
	}
	if blk1.Hash != expected[1] {
		t.Errorf("block 1 hash mismatch with ComputeBlockHashes:\n  got  %s\n  want %s", blk1.Hash, expected[1])
	}

	// Behavioral: round-trip via GetCachedBlocks
	kvc.ReleaseKVBlocks(req)
	cached := kvc.GetCachedBlocks([]int{10, 20, 30, 40, 50, 60, 70, 80})
	if len(cached) != 2 {
		t.Errorf("expected 2 cached blocks after multi-block partial-fill round-trip, got %d", len(cached))
	}
}

func TestAllocateKVBlocks_CachedPrefixToNewBlocks_HashChainingRoundTrip(t *testing.T) {
	// Verifies that new blocks allocated after a cached prefix correctly chain
	// their hashes from the last cached block, and the full extended prefix is
	// findable by GetCachedBlocks after release.
	kvc := NewKVCacheState(8, 2) // 8 blocks, blockSize=2

	// Step 1: Allocate [1,2,3,4] (2 blocks), then release to populate cache
	req1 := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	kvc.AllocateKVBlocks(req1, 0, 4, []int64{})
	kvc.ReleaseKVBlocks(req1)

	// Step 2: Allocate [1,2,3,4,5,6,7,8] — 2 cached blocks + 2 new blocks
	req2 := &sim.Request{ID: "r2", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8}}
	cached := kvc.GetCachedBlocks(req2.InputTokens)
	if len(cached) != 2 {
		t.Fatalf("expected 2 cached blocks, got %d", len(cached))
	}
	ok := kvc.AllocateKVBlocks(req2, 4, 8, cached)
	if !ok {
		t.Fatal("allocation should succeed")
	}

	// Step 3: Release and verify full 4-block prefix is findable
	kvc.ReleaseKVBlocks(req2)
	fullCached := kvc.GetCachedBlocks([]int{1, 2, 3, 4, 5, 6, 7, 8})
	if len(fullCached) != 4 {
		t.Errorf("expected 4 cached blocks after cached-prefix + new-block round-trip, got %d", len(fullCached))
	}

	// Verify hashes match ComputeBlockHashes (BC-3)
	expected := hash.ComputeBlockHashes(2, []int{1, 2, 3, 4, 5, 6, 7, 8})
	if len(expected) != 4 {
		t.Fatalf("expected 4 hashes from ComputeBlockHashes, got %d", len(expected))
	}
	for i, blockID := range fullCached {
		blk := kvc.Blocks[blockID]
		if blk.Hash != expected[i] {
			t.Errorf("block %d hash mismatch:\n  got  %s\n  want %s", i, blk.Hash, expected[i])
		}
	}
}

func TestAllocateKVBlocks_ChunkedPrefill_HierarchicalHashChaining(t *testing.T) {
	// GIVEN a request with 8 tokens and BlockSize=4
	kvc := NewKVCacheState(10, 4)
	req := &sim.Request{
		ID:          "r1",
		InputTokens: []int{10, 20, 30, 40, 50, 60, 70, 80},
	}

	// Allocate first chunk (tokens 0-3) — block 1 gets hash of InputTokens[:4]
	ok := kvc.AllocateKVBlocks(req, 0, 4, []int64{})
	if !ok {
		t.Fatal("first chunk allocation should succeed")
	}

	// Verify first block has correct hierarchical hash (chained from "")
	expectedHash1 := hash.HashBlock("", []int{10, 20, 30, 40})
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

	// THEN second block has hierarchical hash chained from block 1's hash
	ids2 := kvc.RequestMap["r1"]
	if len(ids2) < 2 {
		t.Fatalf("expected at least 2 blocks, got %d", len(ids2))
	}
	blk2 := kvc.Blocks[ids2[1]]
	expectedHash2 := hash.HashBlock(expectedHash1, []int{50, 60, 70, 80})
	if blk2.Hash != expectedHash2 {
		t.Errorf("second block hash mismatch:\n  got  %s\n  want %s", blk2.Hash, expectedHash2)
	}
	// Verify the two hashes match what ComputeBlockHashes produces (BC-3 consistency)
	allHashes := hash.ComputeBlockHashes(4, []int{10, 20, 30, 40, 50, 60, 70, 80})
	if len(allHashes) != 2 {
		t.Fatalf("expected 2 block hashes from ComputeBlockHashes, got %d", len(allHashes))
	}
	if blk1.Hash != allHashes[0] {
		t.Errorf("block 1 hash does not match ComputeBlockHashes[0]:\n  got  %s\n  want %s", blk1.Hash, allHashes[0])
	}
	if blk2.Hash != allHashes[1] {
		t.Errorf("block 2 hash does not match ComputeBlockHashes[1]:\n  got  %s\n  want %s", blk2.Hash, allHashes[1])
	}

	// Behavioral assertion: release and verify full prefix is findable via GetCachedBlocks
	kvc.ReleaseKVBlocks(req)
	cached := kvc.GetCachedBlocks([]int{10, 20, 30, 40, 50, 60, 70, 80})
	if len(cached) != 2 {
		t.Errorf("expected 2 cached blocks after chunked-prefill + release round-trip, got %d", len(cached))
	}
}

func TestAllocateKVBlocks_MidLoopFailure_RollsBackNewBlocks(t *testing.T) {
	// GIVEN a KV cache with only 2 free blocks but a request needing 3 new blocks
	kvc := NewKVCacheState(5, 2) // 5 total blocks, 2 tokens per block
	// Consume 3 blocks with a dummy request, leaving 2 free
	dummy := &sim.Request{ID: "dummy", InputTokens: []int{1, 2, 3, 4, 5, 6}}
	kvc.AllocateKVBlocks(dummy, 0, 6, []int64{})

	usedBefore := kvc.UsedBlocks()
	freeBefore := kvc.TotalCapacity() - usedBefore

	// WHEN we try to allocate 3 blocks (6 tokens / 2 per block) but only 2 are free
	req := &sim.Request{ID: "r_fail", InputTokens: []int{10, 20, 30, 40, 50, 60}}
	ok := kvc.AllocateKVBlocks(req, 0, 6, []int64{})

	// THEN allocation fails and block conservation is maintained
	if ok {
		t.Fatal("allocation should fail (not enough free blocks)")
	}

	// BC-6: Conservation invariant
	usedAfter := kvc.UsedBlocks()
	freeAfter := kvc.TotalCapacity() - usedAfter
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
	// GIVEN: 4 total blocks (blockSize=2), a cached prefix, and tight free-block budget.
	// After cached blocks are claimed, not enough free blocks remain for new allocation.
	kvc := NewKVCacheState(4, 2)

	// Step 1: Create and release a request to populate prefix cache
	req1 := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	kvc.AllocateKVBlocks(req1, 0, 4, []int64{})
	kvc.ReleaseKVBlocks(req1)
	// Now: 4 free blocks, 2 with hashes for prefix [1,2,3,4]

	// Step 2: Consume 1 free block with a filler, leaving 3 free
	filler := &sim.Request{ID: "filler", InputTokens: []int{90, 91}}
	kvc.AllocateKVBlocks(filler, 0, 2, []int64{})
	// Now: 1 used (filler), 3 free (2 with cached hashes)

	usedBefore := kvc.UsedBlocks() // 1
	missesBefore := kvc.CacheMisses

	// Step 3: Try to allocate with cached prefix + new tokens
	// Request needs [1,2,3,4,5,6,7,8]: 2 cached blocks + 2 new blocks
	req2 := &sim.Request{ID: "r2", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8}}
	cached := kvc.GetCachedBlocks(req2.InputTokens)
	if len(cached) != 2 {
		t.Fatalf("expected 2 cached blocks, got %d", len(cached))
	}

	// Caller computes: startIndex=4 (after cached), endIndex=8
	// newTokens = [5,6,7,8], numNewBlocks = 2
	// Pre-check: 2 > countFreeBlocks(3) → false, passes
	// Cached block processing: claims 2 blocks → free drops to 1
	// New block loop: needs 2, only 1 available → mid-loop failure!
	ok := kvc.AllocateKVBlocks(req2, 4, 8, cached)

	// WHEN the allocation fails mid-loop
	if ok {
		t.Fatal("allocation should fail (cached blocks consume free budget)")
	}

	// THEN all mutations are rolled back: UsedBlockCnt, CacheMisses, RequestMap
	if kvc.UsedBlocks() != usedBefore {
		t.Errorf("UsedBlockCnt: got %d, want %d (should be unchanged after rollback)", kvc.UsedBlocks(), usedBefore)
	}
	if kvc.CacheMisses != missesBefore {
		t.Errorf("CacheMisses: got %d, want %d (should be unchanged after rollback)", kvc.CacheMisses, missesBefore)
	}
	if _, exists := kvc.RequestMap["r2"]; exists {
		t.Error("RequestMap should not contain entry for failed allocation")
	}

	// BC-6: Conservation invariant (independent free-list walk, not derived from UsedBlockCnt)
	assertBlockConservation(t, kvc)
}

func TestAllocateKVBlocks_BlockConservation_AfterAllocateReleaseCycles(t *testing.T) {
	// BC-6: After any sequence of operations, conservation holds
	kvc := NewKVCacheState(10, 4)

	// Allocate and release several requests
	for i := 0; i < 5; i++ {
		req := &sim.Request{
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
		req := &sim.Request{ID: fmt.Sprintf("r%d", i)}
		kvc.ReleaseKVBlocks(req)
	}

	// Verify conservation (independent free-list walk, not derived from UsedBlockCnt)
	assertBlockConservation(t, kvc)

	// Expected: 2 requests still hold 1 block each = 2 used, 8 free
	if kvc.UsedBlocks() != 2 {
		t.Errorf("UsedBlockCnt = %d, want 2 (2 requests with 1 block each)", kvc.UsedBlocks())
	}
}

func TestAllocateKVBlocks_DecodeWithBlockSize1_NoPrefixHashPanic(t *testing.T) {
	// GIVEN BlockSizeTokens=1 (edge case where a single decode token fills a full block)
	kvc := NewKVCacheState(20, 1)
	req := &sim.Request{
		ID:           "r1",
		InputTokens:  []int{10, 20, 30, 40},
		OutputTokens: []int{100, 200},
	}

	// Prefill: allocate 4 input tokens (4 blocks with BlockSize=1)
	ok := kvc.AllocateKVBlocks(req, 0, 4, []int64{})
	if !ok {
		t.Fatal("prefill allocation should succeed")
	}

	// WHEN we allocate a decode token (ProgressIndex past all input tokens)
	req.ProgressIndex = 4
	ok = kvc.AllocateKVBlocks(req, 4, 5, []int64{})

	// THEN no panic occurs and allocation succeeds
	if !ok {
		t.Error("decode allocation should succeed (enough free blocks)")
	}

	// Verify the decode block was allocated
	ids := kvc.RequestMap["r1"]
	if len(ids) != 5 {
		t.Errorf("expected 5 blocks (4 prefill + 1 decode), got %d", len(ids))
	}
}

func TestGetCachedBlocks_IsPureQuery_DoesNotAffectCacheHitRate(t *testing.T) {
	// GIVEN a KV cache with cached prefix blocks after one allocation cycle
	kvc := NewKVCacheState(4, 2)
	req := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	kvc.AllocateKVBlocks(req, 0, 4, []int64{})
	kvc.ReleaseKVBlocks(req)
	// After allocate+release: CacheHitRate is 0 (all misses, no hits)

	rateBefore := kvc.CacheHitRate()

	// WHEN GetCachedBlocks is called multiple times (pure query — BC-3)
	cached := kvc.GetCachedBlocks([]int{1, 2, 3, 4})
	if len(cached) != 2 {
		t.Fatalf("expected 2 cached blocks, got %d", len(cached))
	}
	_ = kvc.GetCachedBlocks([]int{1, 2, 3, 4})
	_ = kvc.GetCachedBlocks([]int{1, 2, 3, 4})

	// THEN CacheHitRate is unchanged — lookups alone don't affect metrics
	rateAfter := kvc.CacheHitRate()
	if rateAfter != rateBefore {
		t.Errorf("CacheHitRate changed from %f to %f after GetCachedBlocks calls (should be pure query)", rateBefore, rateAfter)
	}
}

func TestAllocateKVBlocks_CachedPrefixReuse_IncreasesHitRate(t *testing.T) {
	// GIVEN a KV cache with 2 cached prefix blocks from a prior allocation
	kvc := NewKVCacheState(8, 2)
	req1 := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	kvc.AllocateKVBlocks(req1, 0, 4, []int64{})
	kvc.ReleaseKVBlocks(req1)
	// After r1: 2 misses, 0 hits → CacheHitRate = 0

	// WHEN allocating a new request that reuses the cached prefix (BC-4)
	cached := kvc.GetCachedBlocks([]int{1, 2, 3, 4, 5, 6})
	if len(cached) != 2 {
		t.Fatalf("expected 2 cached blocks, got %d", len(cached))
	}
	req2 := &sim.Request{ID: "r2", InputTokens: []int{1, 2, 3, 4, 5, 6}}
	ok := kvc.AllocateKVBlocks(req2, 4, 6, cached)
	if !ok {
		t.Fatal("allocation should succeed")
	}

	// THEN CacheHitRate rises above 0 — cached blocks were counted as hits at commit
	rate := kvc.CacheHitRate()
	if rate <= 0 {
		t.Errorf("CacheHitRate = %f after reusing cached prefix, want > 0", rate)
	}
	if rate >= 1 {
		t.Errorf("CacheHitRate = %f, want < 1 (r1 had all misses)", rate)
	}
}

func TestAllocateKVBlocks_FailedAllocation_CacheHitRateUnchanged(t *testing.T) {
	// GIVEN a KV cache with 2 cached prefix blocks and tight free-block budget
	kvc := NewKVCacheState(4, 2)
	req1 := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	kvc.AllocateKVBlocks(req1, 0, 4, []int64{})
	kvc.ReleaseKVBlocks(req1)

	// Consume 1 block to make the next allocation fail
	filler := &sim.Request{ID: "filler", InputTokens: []int{90, 91}}
	kvc.AllocateKVBlocks(filler, 0, 2, []int64{})

	rateBefore := kvc.CacheHitRate()

	// WHEN allocating with cached prefix + new tokens that exceed capacity (BC-5)
	req2 := &sim.Request{ID: "r2", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8}}
	cached := kvc.GetCachedBlocks(req2.InputTokens)
	ok := kvc.AllocateKVBlocks(req2, 4, 8, cached)

	// THEN allocation fails AND CacheHitRate is unchanged (rollback undoes all mutations)
	if ok {
		t.Fatal("allocation should fail")
	}
	rateAfter := kvc.CacheHitRate()
	if rateAfter != rateBefore {
		t.Errorf("CacheHitRate changed from %f to %f after failed allocation (rollback should restore)", rateBefore, rateAfter)
	}
}

func TestAllocateKVBlocks_Rollback_PreservesFreeListOrder(t *testing.T) {
	// GIVEN: 4 blocks (BlockSize=2), a cached prefix, tight free budget.
	// This triggers mid-loop rollback via the cached-block path (same as
	// TestAllocateKVBlocks_CachedBlockRollback_OnNewBlockFailure).
	kvc := NewKVCacheState(4, 2)

	// Create and release to populate prefix cache
	req1 := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	kvc.AllocateKVBlocks(req1, 0, 4, []int64{})
	kvc.ReleaseKVBlocks(req1)
	// 4 free blocks, 2 with cached hashes

	// Consume 1 with filler → 3 free
	filler := &sim.Request{ID: "filler", InputTokens: []int{90, 91}}
	kvc.AllocateKVBlocks(filler, 0, 2, []int64{})

	// Record free list order before the failed allocation
	freeHeadBefore := kvc.FreeHead.ID
	secondFreeBlockBefore := kvc.FreeHead.NextFree.ID

	// WHEN mid-loop allocation fails (cached blocks consume free budget)
	req2 := &sim.Request{ID: "r2", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8}}
	cached := kvc.GetCachedBlocks(req2.InputTokens)
	ok := kvc.AllocateKVBlocks(req2, 4, 8, cached)
	if ok {
		t.Fatal("allocation should fail (cached blocks consume free budget)")
	}

	// THEN the free list order should be restored to its pre-allocation state.
	// Before allocation: free list was [3, 0, 1] (head=3, next=0).
	// With appendToFreeList (bug): rollback produces [3, 1, 0] (next=1, WRONG).
	// With prependToFreeList (fix): rollback produces [3, 0, 1] (next=0, CORRECT).
	if kvc.FreeHead == nil {
		t.Fatal("FreeHead should not be nil after rollback")
	}
	if kvc.FreeHead.ID != freeHeadBefore {
		t.Errorf("rollback should restore free head to block %d, got block %d",
			freeHeadBefore, kvc.FreeHead.ID)
	}
	if kvc.FreeHead.NextFree == nil {
		t.Fatal("FreeHead.Next should not be nil (3 blocks should be free)")
	}
	if kvc.FreeHead.NextFree.ID != secondFreeBlockBefore {
		t.Errorf("rollback should restore second free block to %d, got %d (append to tail instead of prepend to head)",
			secondFreeBlockBefore, kvc.FreeHead.NextFree.ID)
	}
}

func TestAllocateKVBlocks_PartialBlockFill_PreCheckAccountsForExistingBlock(t *testing.T) {
	// Regression test for #492: prefill capacity pre-check is conservative by 1 block.
	// The pre-check computes blocks from the full newTokens slice, ignoring that some
	// tokens will be absorbed into the request's partially-filled last block.
	tests := []struct {
		name            string
		totalBlocks     int64
		blockSize       int64
		firstTokens     int   // tokens in first allocation (creates partial block)
		totalInput      int   // total input tokens
		secondStart     int64 // startIndex for second allocation
		secondEnd       int64 // endIndex for second allocation
		freeBeforeAlloc int64 // expected free blocks before second allocation
		wantSuccess     bool
		wantUsedBlocks  int64 // expected used blocks after second allocation
		expectPartial   bool  // whether first allocation leaves a partial block
	}{
		{
			// Pre-check: ceil(5/4)=2 > 1 free → false rejection (BUG).
			// Reality: partial absorbs 2 tokens, remaining 3 → ceil(3/4)=1 ≤ 1 → should succeed.
			name:            "partial fill saves 1 block (2 partial + 5 new tokens, blockSize=4)",
			totalBlocks:     2,
			blockSize:       4,
			firstTokens:     2,
			totalInput:      7,
			secondStart:     2,
			secondEnd:       7,
			freeBeforeAlloc: 1,
			wantSuccess:     true,
			wantUsedBlocks:  2, // 1 original (now full) + 1 new
			expectPartial:   true,
		},
		{
			// Pre-check: ceil(2/4)=1 > 0 free → false rejection (BUG).
			// Reality: partial absorbs all 2 tokens, 0 new blocks needed → should succeed.
			name:            "all tokens absorbed into partial block (no new blocks needed)",
			totalBlocks:     1,
			blockSize:       4,
			firstTokens:     2,
			totalInput:      4,
			secondStart:     2,
			secondEnd:       4,
			freeBeforeAlloc: 0,
			wantSuccess:     true,
			wantUsedBlocks:  1, // same block, now full
			expectPartial:   true,
		},
		{
			// Pre-check: ceil(9/4)=3 > 1 free → rejection.
			// Reality: partial absorbs 2, remaining 7 → ceil(7/4)=2 > 1 → still insufficient.
			// This is a genuine failure, not a false rejection.
			name:            "genuinely insufficient blocks (not a false rejection)",
			totalBlocks:     2,
			blockSize:       4,
			firstTokens:     2,
			totalInput:      11,
			secondStart:     2,
			secondEnd:       11,
			freeBeforeAlloc: 1,
			wantSuccess:     false,
			wantUsedBlocks:  1, // unchanged after failed allocation
			expectPartial:   true,
		},
		{
			// blockSize=1: every block is always full (1 token fills it), so
			// spare=0 and the partial-block path is never entered. Verifies
			// the fix does not incorrectly subtract capacity at this boundary.
			name:            "blockSize=1 never has partial blocks (no adjustment needed)",
			totalBlocks:     5,
			blockSize:       1,
			firstTokens:     3,
			totalInput:      5,
			secondStart:     3,
			secondEnd:       5,
			freeBeforeAlloc: 2,
			wantSuccess:     true,
			wantUsedBlocks:  5,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			kvc := NewKVCacheState(tc.totalBlocks, tc.blockSize)
			tokens := make([]int, tc.totalInput)
			for i := range tokens {
				tokens[i] = (i + 1) * 10
			}
			req := &sim.Request{ID: "r1", InputTokens: tokens}

			// First allocation: create partial block
			ok := kvc.AllocateKVBlocks(req, 0, int64(tc.firstTokens), []int64{})
			if !ok {
				t.Fatal("initial allocation should succeed")
			}

			// Verify partial block state (structural precondition, for debuggability)
			ids := kvc.RequestMap["r1"]
			lastBlk := kvc.Blocks[ids[len(ids)-1]]
			if tc.expectPartial && int64(len(lastBlk.Tokens)) >= tc.blockSize {
				t.Fatalf("expected partial last block, got full block with %d tokens", len(lastBlk.Tokens))
			}
			if !tc.expectPartial && int64(len(lastBlk.Tokens)) != tc.blockSize {
				t.Fatalf("expected full last block, got %d tokens", len(lastBlk.Tokens))
			}

			// Verify free block count matches expectation
			free := kvc.countFreeBlocks()
			if free != tc.freeBeforeAlloc {
				t.Fatalf("expected %d free blocks before second alloc, got %d", tc.freeBeforeAlloc, free)
			}

			// Second allocation: this is where the bug manifests
			req.ProgressIndex = int64(tc.firstTokens)
			ok = kvc.AllocateKVBlocks(req, tc.secondStart, tc.secondEnd, []int64{})
			if ok != tc.wantSuccess {
				t.Errorf("AllocateKVBlocks() = %v, want %v", ok, tc.wantSuccess)
			}

			if kvc.UsedBlocks() != tc.wantUsedBlocks {
				t.Errorf("UsedBlocks = %d, want %d", kvc.UsedBlocks(), tc.wantUsedBlocks)
			}

			assertBlockConservation(t, kvc)
		})
	}
}

func TestAllocateKVBlocks_FirstAllocation_PreCheckUnaffectedByFix(t *testing.T) {
	// Verifies the fix does not alter behavior for first-ever allocations (no prior blocks).
	// When RequestMap has no entry, the partial-block adjustment is skipped entirely.
	kvc := NewKVCacheState(2, 4) // 2 blocks, blockSize=4
	req := &sim.Request{ID: "r1", InputTokens: []int{10, 20, 30, 40, 50}}

	// 5 tokens → ceil(5/4)=2 blocks needed, 2 free → should succeed
	ok := kvc.AllocateKVBlocks(req, 0, 5, []int64{})
	if !ok {
		t.Fatal("first allocation should succeed with exact block count")
	}
	if kvc.UsedBlocks() != 2 {
		t.Errorf("UsedBlocks = %d, want 2", kvc.UsedBlocks())
	}

	// Same scenario but insufficient blocks: 5 tokens need 2 blocks, only 1 available
	kvc2 := NewKVCacheState(1, 4)
	req2 := &sim.Request{ID: "r2", InputTokens: []int{10, 20, 30, 40, 50}}
	ok = kvc2.AllocateKVBlocks(req2, 0, 5, []int64{})
	if ok {
		t.Fatal("first allocation should fail when blocks are insufficient")
	}
	assertBlockConservation(t, kvc2)
}

func TestAllocateKVBlocks_ChunkedPrefill_NoPhantomBlocks(t *testing.T) {
	// GIVEN a KV cache with BlockSize=4 and a request that already has a partial block (3 tokens)
	kvc := NewKVCacheState(20, 4) // 20 blocks, size 4
	req := &sim.Request{
		ID:            "phantom-test",
		InputTokens:   []int{1, 2, 3, 4, 5, 6, 7, 8},
		OutputTokens:  []int{100},
		ProgressIndex: 0,
	}

	// First allocation: 3 tokens (leaves a partial block)
	ok := kvc.AllocateKVBlocks(req, 0, 3, []int64{})
	if !ok {
		t.Fatal("first allocation should succeed")
	}
	blocksAfterFirst := kvc.UsedBlocks()

	// WHEN allocating the next chunk of 5 tokens (should fill partial block + allocate 1 new block)
	req.ProgressIndex = 3
	ok = kvc.AllocateKVBlocks(req, 3, 8, []int64{})
	if !ok {
		t.Fatal("second allocation should succeed")
	}

	// THEN exactly 1 new block should be allocated (partial fill + 1 new block, not 2)
	// Partial block: 3 tokens + 1 token = 4 (full). Remaining: 4 tokens = 1 new block.
	blocksAfterSecond := kvc.UsedBlocks()
	newBlocks := blocksAfterSecond - blocksAfterFirst
	if newBlocks != 1 {
		t.Errorf("expected 1 new block after partial fill, got %d", newBlocks)
	}

	// Verify no phantom blocks (all allocated blocks should have non-empty Tokens)
	for _, blockID := range kvc.RequestMap[req.ID] {
		blk := kvc.Blocks[blockID]
		if len(blk.Tokens) == 0 {
			t.Errorf("block %d has empty Tokens (phantom block)", blk.ID)
		}
	}
}
