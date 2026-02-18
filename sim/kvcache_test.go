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
