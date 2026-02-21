package sim

import (
	"testing"
)

// TestPreempt_EmptyBatch_ReturnsFalse verifies BC-1 (#293):
// preempt() must return false (not panic) when the batch is empty.
func TestPreempt_EmptyBatch_ReturnsFalse(t *testing.T) {
	// GIVEN a simulator with minimal KV cache (2 blocks, block size 16)
	config := SimConfig{
		TotalKVBlocks:     2,
		BlockSizeTokens:   16,
		MaxRunningReqs:    10,
		MaxScheduledTokens: 10000,
		Horizon:           1000000,
		BetaCoeffs:        []float64{100, 1, 1},
		AlphaCoeffs:       []float64{100, 1, 100},
	}
	s, err := NewSimulator(config)
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}

	// AND the running batch is empty
	s.RunningBatch = &Batch{Requests: []*Request{}}

	// AND a request that needs far more blocks than available
	req := &Request{
		ID:          "large-req",
		InputTokens: make([]int, 200), // needs ~13 blocks, only 2 available
	}

	// WHEN preempt is called
	// THEN it must return false (not panic)
	result := s.preempt(req, 0, 200)
	if result {
		t.Error("expected preempt to return false when batch is empty and allocation fails")
	}

	// AND KV cache conservation must hold (INV-4): no blocks leaked
	if s.KVCache.UsedBlocks() != 0 {
		t.Errorf("expected 0 used blocks after failed allocation on empty batch, got %d", s.KVCache.UsedBlocks())
	}
}

// TestPreempt_InsufficientBlocks_EvictsAllThenReturnsFalse verifies BC-2 (#297):
// preempt() must not loop forever when KV blocks are insufficient for any request.
func TestPreempt_InsufficientBlocks_EvictsAllThenReturnsFalse(t *testing.T) {
	// GIVEN a simulator with very small KV cache
	config := SimConfig{
		TotalKVBlocks:     2,
		BlockSizeTokens:   16,
		MaxRunningReqs:    10,
		MaxScheduledTokens: 10000,
		Horizon:           1000000,
		BetaCoeffs:        []float64{100, 1, 1},
		AlphaCoeffs:       []float64{100, 1, 100},
	}
	s, err := NewSimulator(config)
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}

	// AND one small request in the running batch
	existing := &Request{
		ID:          "existing",
		InputTokens: make([]int, 10),
		State:       StateRunning,
	}
	s.RunningBatch = &Batch{Requests: []*Request{existing}}
	// Allocate some blocks for the existing request
	if ok := s.KVCache.AllocateKVBlocks(existing, 0, 10, []int64{}); !ok {
		t.Fatal("setup: failed to allocate KV blocks for existing request")
	}

	// AND a new request that needs more blocks than total capacity
	req := &Request{
		ID:          "huge-req",
		InputTokens: make([]int, 200),
	}

	// WHEN preempt is called (should evict existing, then fail on empty batch)
	result := s.preempt(req, 0, 200)

	// THEN it must return false (not hang or panic)
	if result {
		t.Error("expected preempt to return false when blocks are insufficient for any request")
	}

	// AND KV cache conservation must hold (INV-4): all blocks free after eviction
	usedBlocks := s.KVCache.UsedBlocks()
	if usedBlocks != 0 {
		t.Errorf("expected 0 used blocks after all requests evicted, got %d", usedBlocks)
	}
}
