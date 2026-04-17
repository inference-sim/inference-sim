package kv

import (
	"bytes"
	"fmt"
	"os"
	"testing"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/hash"
)

// assertBlockConservation verifies the KV block conservation invariant (INV-4)
// via an independent free-list walk and InUse scan (not derived from FreeBlockCnt).
func assertBlockConservation(t *testing.T, kvc *KVCacheState) {
	t.Helper()
	if err := kvc.verifyBlockConservation(); err != nil {
		t.Errorf("INV-4 block conservation violated: %v", err)
	}
	if used := kvc.UsedBlocks(); used < 0 {
		t.Errorf("UsedBlocks() = %d, must be >= 0", used)
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

	// Verify conservation via independent free-list walk
	assertBlockConservation(t, kvc)

	// Expected: 2 requests still hold 1 block each = 2 used, 8 free
	if kvc.UsedBlocks() != 2 {
		t.Errorf("UsedBlocks() = %d, want 2 (2 requests with 1 block each)", kvc.UsedBlocks())
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

func TestAllocateKVBlocks_PreCheckRejection_CacheHitRateUnchanged(t *testing.T) {
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

	// THEN allocation fails at pre-check AND CacheHitRate is unchanged (no mutations occurred)
	if ok {
		t.Fatal("allocation should fail")
	}
	rateAfter := kvc.CacheHitRate()
	if rateAfter != rateBefore {
		t.Errorf("CacheHitRate changed from %f to %f after pre-check rejection (should be unchanged)", rateBefore, rateAfter)
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

// BC-1 (#963): AllocateKVBlocks failure must not produce Warn-level output.
// vLLM proof: kv_cache_manager.py:334-336 returns None silently.
func TestAllocateKVBlocks_Failure_NoWarnOutput(t *testing.T) {
	// GIVEN a KV cache with 1 block (16 tokens) that is fully occupied
	kvc := NewKVCacheState(1, 16)
	filler := &sim.Request{
		ID:          "filler",
		InputTokens: make([]int, 16),
	}
	ok := kvc.AllocateKVBlocks(filler, 0, 16, nil)
	require.True(t, ok, "setup: filler allocation must succeed")

	// Capture log output at Warn level
	var buf bytes.Buffer
	logrus.SetOutput(&buf)
	logrus.SetLevel(logrus.WarnLevel)
	defer func() {
		logrus.SetOutput(os.Stderr)
		logrus.SetLevel(logrus.InfoLevel)
	}()

	// WHEN a second request tries to allocate and fails
	victim := &sim.Request{
		ID:          "victim",
		InputTokens: make([]int, 16),
	}
	ok = kvc.AllocateKVBlocks(victim, 0, 16, nil)

	// THEN allocation fails but no Warn-level output is produced
	assert.False(t, ok, "allocation must fail (cache full)")
	assert.Empty(t, buf.String(), "no Warn-level log output expected (BC-1: vLLM returns None silently)")
}

func TestFreeBlockCnt_DirectCounter_MatchesFreeListLength(t *testing.T) {
	kvc := NewKVCacheState(10, 4)

	req := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8}}
	ok := kvc.AllocateKVBlocks(req, 0, 8, []int64{})
	require.True(t, ok)

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

	kvc.ReleaseKVBlocks(req)

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

func TestAllocateKVBlocks_DecodeFailure_PreservesExistingBlocks(t *testing.T) {
	// GIVEN a request that completed prefill and has 2 blocks allocated
	kvc := NewKVCacheState(3, 4) // 3 blocks total
	req := &sim.Request{
		ID:           "r1",
		InputTokens:  []int{1, 2, 3, 4, 5, 6, 7, 8},
		OutputTokens: []int{100, 200},
	}
	// Allocate prefill (2 blocks for 8 tokens, blockSize=4)
	ok := kvc.AllocateKVBlocks(req, 0, 8, []int64{})
	req.ProgressIndex = 8 // now past prefill, in decode
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
		ID:           "r1",
		InputTokens:  []int{1, 2, 3}, // 3 tokens = 1 partial block (3/4)
		OutputTokens: []int{100},
	}
	ok := kvc.AllocateKVBlocks(req, 0, 3, []int64{})
	req.ProgressIndex = 3 // now past prefill, in decode
	require.True(t, ok)
	require.Equal(t, int64(1), kvc.countFreeBlocks())

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

func TestAllocateKVBlocks_PreCheck_CatchesCachedBlockBudgetExhaustion(t *testing.T) {
	// Previously: cached blocks consumed free budget mid-loop, triggering rollback.
	// Now: pre-check accounts for cached blocks and rejects upfront.
	kvc := NewKVCacheState(4, 2)
	req1 := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	kvc.AllocateKVBlocks(req1, 0, 4, []int64{})
	kvc.ReleaseKVBlocks(req1)

	filler := &sim.Request{ID: "filler", InputTokens: []int{90, 91}}
	kvc.AllocateKVBlocks(filler, 0, 2, []int64{})

	req2 := &sim.Request{ID: "r2", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8}}
	cached := kvc.GetCachedBlocks(req2.InputTokens)
	require.Len(t, cached, 2)

	freeBefore := kvc.countFreeBlocks()
	ok := kvc.AllocateKVBlocks(req2, 4, 8, cached)

	assert.False(t, ok)
	assert.Equal(t, freeBefore, kvc.countFreeBlocks(),
		"free block count must be unchanged after pre-check rejection")
	assert.Empty(t, kvc.RequestMap["r2"],
		"no RequestMap entry should exist after pre-check rejection")
	assertBlockConservation(t, kvc)
}

func TestPreemptForTokens_RetryPreservesRequestMap(t *testing.T) {
	// Reproduces #1061 leak path 1: decode failure -> eviction -> retry.
	// Under the old rollback, the retry would orphan blocks.
	kvc := NewKVCacheState(4, 4)

	r1 := &sim.Request{
		ID:           "r1",
		InputTokens:  []int{1, 2, 3, 4, 5, 6, 7, 8},
		OutputTokens: []int{100},
	}
	ok := kvc.AllocateKVBlocks(r1, 0, 8, []int64{})
	r1.ProgressIndex = 8 // now past prefill, in decode
	require.True(t, ok)
	require.Len(t, kvc.RequestMap["r1"], 2)

	r2 := &sim.Request{
		ID:          "r2",
		InputTokens: []int{10, 20, 30, 40, 50, 60, 70, 80},
	}
	ok = kvc.AllocateKVBlocks(r2, 0, 8, []int64{})
	require.True(t, ok)
	require.Equal(t, int64(0), kvc.countFreeBlocks())

	// Decode for r1 fails (last block full, 0 free)
	ok = kvc.AllocateKVBlocks(r1, 8, 9, []int64{})
	assert.False(t, ok)

	// RequestMap["r1"] still has 2 blocks (not deleted by rollback)
	assert.Len(t, kvc.RequestMap["r1"], 2,
		"decode failure must not delete RequestMap — this was the #1061 bug")

	// Simulate eviction: release r2
	kvc.ReleaseKVBlocks(r2)
	require.Equal(t, int64(2), kvc.countFreeBlocks())

	// Retry succeeds
	ok = kvc.AllocateKVBlocks(r1, 8, 9, []int64{})
	assert.True(t, ok, "retry after eviction should succeed")

	// r1 has 3 blocks: 2 original + 1 new decode block
	assert.Len(t, kvc.RequestMap["r1"], 3,
		"retry must see original 2 blocks + 1 new decode block")
	assertBlockConservation(t, kvc)
}

func TestVerifyBlockConservation_DetectsOrphanedBlocks(t *testing.T) {
	kvc := NewKVCacheState(10, 4)
	req := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3, 4}}
	kvc.AllocateKVBlocks(req, 0, 4, []int64{})

	err := kvc.verifyBlockConservation()
	assert.NoError(t, err)

	// Artificially orphan a block (simulate the old bug)
	kvc.FreeBlockCnt--

	err = kvc.verifyBlockConservation()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "drift")
}

func TestKVCacheState_SnapshotCachedBlocksFn_FrozenView(t *testing.T) {
	// GIVEN a KVCacheState with some cached blocks
	kvc := NewKVCacheState(100, 4)
	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8} // 2 blocks
	req := &sim.Request{ID: "r1", InputTokens: tokens}
	ok := kvc.AllocateKVBlocks(req, 0, 8, nil)
	require.True(t, ok)

	// WHEN we take a snapshot
	snapshotFn := kvc.SnapshotCachedBlocksFn()

	// AND then allocate more blocks (tokens 9-16 = 2 more blocks)
	tokens2 := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	req2 := &sim.Request{ID: "r2", InputTokens: tokens2}
	ok = kvc.AllocateKVBlocks(req2, 8, 16, kvc.GetCachedBlocks(tokens2))
	require.True(t, ok)

	// THEN the snapshot sees only the original 2 blocks
	assert.Equal(t, 2, snapshotFn(tokens2), "snapshot should see 2 blocks (frozen at snapshot time)")

	// AND the live query sees all 4 blocks
	assert.Equal(t, 4, len(kvc.GetCachedBlocks(tokens2)), "live query should see 4 blocks")
}

func TestAllocateKVBlocks_DecodeNoExistingBlocks_FailsWhenCacheFull(t *testing.T) {
	// GIVEN a request in decode with no RequestMap entry (simulates a preempted request
	// whose blocks were released but ProgressIndex was not reset). This exercises the
	// else-branch of the decode pre-check (cache.go:247-252).
	kvc := NewKVCacheState(2, 4)

	// Fill the cache completely
	filler := &sim.Request{ID: "filler", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8}}
	ok := kvc.AllocateKVBlocks(filler, 0, 8, []int64{})
	require.True(t, ok)
	require.Equal(t, int64(0), kvc.countFreeBlocks())

	// Create a request that appears to be in decode (ProgressIndex past input)
	// but has no RequestMap entry (blocks were released during preemption)
	req := &sim.Request{
		ID:            "preempted",
		InputTokens:   []int{10, 20, 30, 40},
		OutputTokens:  []int{100},
		ProgressIndex: 4, // past prefill
	}
	// Note: no AllocateKVBlocks call for prefill — RequestMap["preempted"] does not exist

	// WHEN decode allocation is attempted with 0 free blocks
	ok = kvc.AllocateKVBlocks(req, 4, 5, []int64{})

	// THEN fails without panic (pre-check catches it)
	assert.False(t, ok, "decode with no existing blocks and 0 free should fail cleanly")
	assert.Empty(t, kvc.RequestMap["preempted"],
		"no RequestMap entry should be created for failed allocation")
	assertBlockConservation(t, kvc)
}

func TestAllocateKVBlocks_DecodeNoExistingBlocks_SucceedsWhenFreeAvailable(t *testing.T) {
	// GIVEN a request in decode with no RequestMap entry but free blocks available.
	// This exercises the else-branch passing through to the allocation loop.
	kvc := NewKVCacheState(3, 4)

	// Fill 2 of 3 blocks
	filler := &sim.Request{ID: "filler", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8}}
	ok := kvc.AllocateKVBlocks(filler, 0, 8, []int64{})
	require.True(t, ok)
	require.Equal(t, int64(1), kvc.countFreeBlocks())

	req := &sim.Request{
		ID:            "preempted",
		InputTokens:   []int{10, 20, 30, 40},
		OutputTokens:  []int{100},
		ProgressIndex: 4,
	}

	// WHEN decode allocation is attempted with 1 free block available
	ok = kvc.AllocateKVBlocks(req, 4, 5, []int64{})

	// THEN succeeds — allocates 1 new block for the decode token
	assert.True(t, ok, "decode with no existing blocks but free available should succeed")
	assert.Len(t, kvc.RequestMap["preempted"], 1,
		"should have 1 block allocated for the decode token")
	assertBlockConservation(t, kvc)
}

func TestLazyHashDeletion_PreemptedRequestFindsCache(t *testing.T) {
	// Verifies issue #1057: preempted requests can find their cached prefix
	// blocks on readmission because popFreeBlock preserves hashes (lazy
	// deletion). Hashes are only cleared when blocks are filled with new
	// content in the allocation loop.
	//
	// Scenario:
	//   1. r1 allocates 1024 blocks (fills entire cache), then is preempted.
	//   2. r2 allocates 128 blocks, reusing 128 of r1's freed blocks.
	//      ReleaseKVBlocks adds blocks in reverse order (last block first),
	//      so popFreeBlock takes r1's LAST 128 blocks (blocks 896-1023).
	//      Lazy deletion clears those 128 old hashes when r2 fills them.
	//   3. GetCachedBlocks for r1's tokens finds the first 896 blocks
	//      (contiguous prefix with intact hash chain).
	const (
		totalBlocks = 1024
		blockSize   = 16
		prefixLen   = 1024 * 16 // 16384 tokens = 1024 blocks
		reuseBlocks = 128       // blocks reused by r2
	)
	kvc := NewKVCacheState(totalBlocks, blockSize)

	// Build input tokens for r1 (16384 tokens => 1024 full blocks)
	inputTokens := make([]int, prefixLen)
	for i := range inputTokens {
		inputTokens[i] = i + 1
	}
	r1 := &sim.Request{ID: "r1", InputTokens: inputTokens}
	ok := kvc.AllocateKVBlocks(r1, 0, int64(prefixLen), []int64{})
	require.True(t, ok, "r1 prefill allocation must succeed")
	require.Len(t, kvc.RequestMap["r1"], 1024, "r1 should own 1024 blocks")

	// WHEN r1 is preempted (blocks released to free list, hashes preserved)
	kvc.ReleaseKVBlocks(r1)
	require.Equal(t, int64(totalBlocks), kvc.countFreeBlocks(),
		"all blocks should be free after preemption")

	// Verify all 1024 prefix blocks are findable immediately after preemption
	// (no blocks reused yet, all hashes intact)
	allCached := kvc.GetCachedBlocks(inputTokens)
	require.Equal(t, 1024, len(allCached),
		"all 1024 prefix blocks should be findable immediately after preemption")

	// AND a second request r2 allocates 128 blocks (2048 tokens), consuming
	// the first 128 blocks popped from the free list. ReleaseKVBlocks added
	// r1's blocks in reverse order (block 1023 first -> head), so popFreeBlock
	// takes r1's blocks 1023, 1022, ..., 896. Lazy deletion clears those hashes
	// when r2 fills the blocks with new content.
	r2Tokens := make([]int, reuseBlocks*blockSize)
	for i := range r2Tokens {
		r2Tokens[i] = 90000 + i // distinct tokens so hashes differ from r1
	}
	r2 := &sim.Request{ID: "r2", InputTokens: r2Tokens}
	ok = kvc.AllocateKVBlocks(r2, 0, int64(len(r2Tokens)), []int64{})
	require.True(t, ok, "r2 allocation must succeed")

	// THEN r1's first 896 prefix blocks (0-895) are still findable via
	// GetCachedBlocks because their hashes were preserved by lazy deletion.
	// Blocks 896-1023 were reused by r2 (different tokens), so their old
	// hashes were cleared and the hierarchical hash chain breaks at block 896.
	cached := kvc.GetCachedBlocks(inputTokens)
	assert.Equal(t, 1024-reuseBlocks, len(cached),
		"preempted request should find 896 of 1024 prefix blocks cached (128 reused by r2)")

	// Verify the cached blocks form a contiguous prefix (all have intact hashes)
	for i, blockID := range cached {
		blk := kvc.Blocks[blockID]
		assert.NotEmpty(t, blk.Hash, "cached block %d should have a hash", i)
	}

	// INV-4: block conservation must hold throughout
	assertBlockConservation(t, kvc)
}

func TestLazyHashDeletion_HashClearedOnReuse(t *testing.T) {
	// GIVEN a KV cache with exactly 2 blocks (so B must reuse A's physical blocks)
	// and Request A allocates 2 blocks with specific tokens
	kvc := NewKVCacheState(2, 4) // blockSize=4, only 2 blocks total
	reqA := &sim.Request{
		ID:          "reqA",
		InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8}, // 2 blocks
	}

	// Allocate Request A
	ok := kvc.AllocateKVBlocks(reqA, 0, int64(len(reqA.InputTokens)), []int64{})
	require.True(t, ok, "Request A allocation should succeed")

	// Compute expected hashes for Request A's blocks
	h1_A := hash.HashBlock("", []int{1, 2, 3, 4})
	h2_A := hash.HashBlock(h1_A, []int{5, 6, 7, 8})

	// Verify Request A's hashes are in HashToBlock
	_, foundH1A := kvc.HashToBlock[h1_A]
	_, foundH2A := kvc.HashToBlock[h2_A]
	require.True(t, foundH1A, "Request A block 0 hash should be in HashToBlock")
	require.True(t, foundH2A, "Request A block 1 hash should be in HashToBlock")

	// Release Request A blocks
	kvc.ReleaseKVBlocks(reqA)

	// Hashes should still be in HashToBlock (lazy deletion - not yet reused)
	_, foundH1A_after := kvc.HashToBlock[h1_A]
	_, foundH2A_after := kvc.HashToBlock[h2_A]
	assert.True(t, foundH1A_after, "Request A hashes should survive in HashToBlock after release")
	assert.True(t, foundH2A_after, "Request A hashes should survive in HashToBlock after release")

	// WHEN Request B allocates same physical blocks with different tokens
	// (only 2 blocks exist, so B must reuse A's released blocks)
	reqB := &sim.Request{
		ID:          "reqB",
		InputTokens: []int{100, 200, 300, 400, 500, 600, 700, 800}, // 2 blocks, different tokens
	}
	ok = kvc.AllocateKVBlocks(reqB, 0, int64(len(reqB.InputTokens)), []int64{})
	require.True(t, ok, "Request B allocation should succeed")

	// Compute expected hashes for Request B's blocks
	h1_B := hash.HashBlock("", []int{100, 200, 300, 400})
	h2_B := hash.HashBlock(h1_B, []int{500, 600, 700, 800})

	// THEN Request A's hashes should be deleted (replaced by B's hashes)
	_, foundH1A_final := kvc.HashToBlock[h1_A]
	_, foundH2A_final := kvc.HashToBlock[h2_A]
	assert.False(t, foundH1A_final, "Request A hash h1 should be deleted when block is reused")
	assert.False(t, foundH2A_final, "Request A hash h2 should be deleted when block is reused")

	// AND Request B's hashes should exist
	_, foundH1B := kvc.HashToBlock[h1_B]
	_, foundH2B := kvc.HashToBlock[h2_B]
	assert.True(t, foundH1B, "Request B hash h1 should be in HashToBlock")
	assert.True(t, foundH2B, "Request B hash h2 should be in HashToBlock")

	assertBlockConservation(t, kvc)
}

func TestLazyHashDeletion_PartialEviction(t *testing.T) {
	// GIVEN a KV cache with 1024 blocks and Request A with 1024-block prefix
	kvc := NewKVCacheState(1024, 16)
	reqA := &sim.Request{
		ID:          "reqA",
		InputTokens: make([]int, 16384), // 1024 blocks * 16 tokens/block
	}
	for i := range reqA.InputTokens {
		reqA.InputTokens[i] = i + 1000
	}

	// Allocate and release Request A
	ok := kvc.AllocateKVBlocks(reqA, 0, int64(len(reqA.InputTokens)), []int64{})
	require.True(t, ok, "Request A allocation should succeed")
	kvc.ReleaseKVBlocks(reqA)
	require.Equal(t, int64(1024), kvc.countFreeBlocks(), "All blocks should be free")

	// WHEN Request B allocates 512 blocks with different content.
	// ReleaseKVBlocks adds A's blocks in reverse order (block 1023 at head),
	// so popFreeBlock gives B the tail half of A's prefix (blocks 1023..512).
	// After B allocates, blocks 0-511 still have A's hashes (contiguous prefix
	// from block 0), while blocks 512-1023 are overwritten with B's content.
	reqB := &sim.Request{
		ID:          "reqB",
		InputTokens: make([]int, 8192), // 512 blocks * 16 tokens/block
	}
	for i := range reqB.InputTokens {
		reqB.InputTokens[i] = i + 50000 // Different tokens
	}
	ok = kvc.AllocateKVBlocks(reqB, 0, int64(len(reqB.InputTokens)), []int64{})
	require.True(t, ok, "Request B allocation should succeed")
	require.Equal(t, int64(512), kvc.UsedBlocks(), "Request B uses 512 blocks")
	require.Equal(t, int64(512), kvc.countFreeBlocks(), "512 blocks remain free")

	// THEN Request C with original prefix should get 512 cache hits
	// (GetCachedBlocks finds contiguous prefix from block 0 through block 511;
	// chain breaks at block 512 which was overwritten by B).
	reqC := &sim.Request{
		ID:          "reqC",
		InputTokens: reqA.InputTokens, // Same prefix as A
	}
	cached := kvc.GetCachedBlocks(reqC.InputTokens)
	assert.Equal(t, 512, len(cached),
		"Should get 512 cache hits (first half of prefix intact, chain breaks at block 512)")

	// Verify vLLM parity: hierarchical hash chain breaks at the first evicted
	// block. The surviving prefix (blocks 0-511) forms a contiguous chain.
	assertBlockConservation(t, kvc)
}

// TestIncrementalCacheClaiming_RunningRequests verifies that running requests
// doing chunked prefill can incrementally claim cached blocks beyond their
// current allocation (Bug 1 fix for issue #1057).
func TestIncrementalCacheClaiming_RunningRequests(t *testing.T) {
	logrus.SetOutput(os.Stderr)

	// GIVEN a KV cache with a shared prefix
	totalBlocks := int64(200)
	blockSize := int64(16)
	kvc := NewKVCacheState(totalBlocks, blockSize)

	// Shared prefix: 128 blocks (2048 tokens)
	sharedPrefix := make([]int, 2048)
	for i := range sharedPrefix {
		sharedPrefix[i] = i + 10000
	}

	// WHEN request A fully allocates the shared prefix
	reqA := &sim.Request{
		ID:          "reqA",
		InputTokens: append([]int{}, sharedPrefix...),
	}
	ok := kvc.AllocateKVBlocks(reqA, 0, int64(len(reqA.InputTokens)), nil)
	require.True(t, ok, "Initial allocation should succeed")
	assert.Equal(t, 128, len(kvc.RequestMap[reqA.ID]), "Should have 128 blocks")

	// AND releases its blocks (simulating completion)
	kvc.ReleaseKVBlocks(reqA)
	initialFreeBlocks := kvc.countFreeBlocks()
	assertBlockConservation(t, kvc)

	// WHEN request B starts with the same prefix, allocates first chunk, gets preempted
	reqB := &sim.Request{
		ID:          "reqB",
		InputTokens: append([]int{}, sharedPrefix...),
	}

	// Step 1: B allocates first 512 tokens (32 blocks) and stops (simulating chunked prefill)
	cachedBlocks := kvc.GetCachedBlocks(reqB.InputTokens)
	require.Equal(t, 128, len(cachedBlocks), "Should find 128 cached blocks")

	ok = kvc.AllocateKVBlocks(reqB, 0, 512, cachedBlocks)
	require.True(t, ok, "First chunk allocation should succeed")
	// First-time request claims ALL cached blocks upfront (vLLM behavior)
	require.Equal(t, 128, len(kvc.RequestMap[reqB.ID]), "Should claim all 128 cached blocks")

	// Step 2: Simulate that B gets preempted and releases blocks
	kvc.ReleaseKVBlocks(reqB)
	assert.Equal(t, initialFreeBlocks, kvc.countFreeBlocks(), "Free blocks restored after release")

	// Step 3: B gets readmitted and tries to allocate next chunk (512-1024)
	// This is the Bug 1 scenario: running request trying to allocate within cached prefix
	// WITHOUT the fix, it would pass []int64{} and try to allocate fresh blocks
	// WITH the fix (batch_formation.go), it passes cachedBlocks
	cachedBlocksAgain := kvc.GetCachedBlocks(reqB.InputTokens)
	require.Equal(t, 128, len(cachedBlocksAgain), "Cache still has all 128 blocks")

	// Simulate readmission: B already processed first 512 tokens (32 blocks)
	ok = kvc.AllocateKVBlocks(reqB, 0, 512, cachedBlocksAgain)
	require.True(t, ok, "Readmission first chunk should succeed")

	// Bug 1 fix verification: Running request (RequestMap[reqB.ID] now exists)
	// continues to second chunk within cached prefix
	ok = kvc.AllocateKVBlocks(reqB, 512, 1024, cachedBlocksAgain)
	require.True(t, ok, "Second chunk should succeed with incremental cache claiming")

	// Verify cache hits counter increased (blocks were claimed from cache, not allocated fresh)
	assert.Greater(t, kvc.CacheHits, int64(128), "Should have cache hits from readmission")

	assertBlockConservation(t, kvc)
}
