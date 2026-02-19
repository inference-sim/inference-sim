package sim

import (
	"testing"
)

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
