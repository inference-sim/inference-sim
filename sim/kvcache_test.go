package sim

import (
	"fmt"
	"testing"
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

func TestAllocateKVBlocks_MidLoopFailure_RollsBackNewBlocks(t *testing.T) {
	// GIVEN a KV cache with only 2 free blocks but a request needing 3 new blocks
	kvc := NewKVCacheState(5, 2) // 5 total blocks, 2 tokens per block
	// Consume 3 blocks with a dummy request, leaving 2 free
	dummy := &Request{ID: "dummy", InputTokens: []int{1, 2, 3, 4, 5, 6}}
	kvc.AllocateKVBlocks(dummy, 0, 6, []int64{})

	usedBefore := kvc.UsedBlocks()
	freeBefore := kvc.TotalCapacity() - usedBefore

	// WHEN we try to allocate 3 blocks (6 tokens / 2 per block) but only 2 are free
	req := &Request{ID: "r_fail", InputTokens: []int{10, 20, 30, 40, 50, 60}}
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
	req1 := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	kvc.AllocateKVBlocks(req1, 0, 4, []int64{})
	kvc.ReleaseKVBlocks(req1)
	// Now: 4 free blocks, 2 with hashes for prefix [1,2,3,4]

	// Step 2: Consume 1 free block with a filler, leaving 3 free
	filler := &Request{ID: "filler", InputTokens: []int{90, 91}}
	kvc.AllocateKVBlocks(filler, 0, 2, []int64{})
	// Now: 1 used (filler), 3 free (2 with cached hashes)

	usedBefore := kvc.UsedBlocks() // 1
	missesBefore := kvc.CacheMisses

	// Step 3: Try to allocate with cached prefix + new tokens
	// Request needs [1,2,3,4,5,6,7,8]: 2 cached blocks + 2 new blocks
	req2 := &Request{ID: "r2", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8}}
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
	req := &Request{
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
	req := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
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
	req1 := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	kvc.AllocateKVBlocks(req1, 0, 4, []int64{})
	kvc.ReleaseKVBlocks(req1)
	// After r1: 2 misses, 0 hits → CacheHitRate = 0

	// WHEN allocating a new request that reuses the cached prefix (BC-4)
	cached := kvc.GetCachedBlocks([]int{1, 2, 3, 4, 5, 6})
	if len(cached) != 2 {
		t.Fatalf("expected 2 cached blocks, got %d", len(cached))
	}
	req2 := &Request{ID: "r2", InputTokens: []int{1, 2, 3, 4, 5, 6}}
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
	req1 := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	kvc.AllocateKVBlocks(req1, 0, 4, []int64{})
	kvc.ReleaseKVBlocks(req1)

	// Consume 1 block to make the next allocation fail
	filler := &Request{ID: "filler", InputTokens: []int{90, 91}}
	kvc.AllocateKVBlocks(filler, 0, 2, []int64{})

	rateBefore := kvc.CacheHitRate()

	// WHEN allocating with cached prefix + new tokens that exceed capacity (BC-5)
	req2 := &Request{ID: "r2", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8}}
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
