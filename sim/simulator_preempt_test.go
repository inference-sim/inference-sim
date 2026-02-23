package sim

import (
	"testing"
)

// TestPreempt_EmptyBatch_ReturnsFalse verifies BC-6 (#293):
// preemption with empty batch must not panic.
func TestPreempt_EmptyBatch_ReturnsFalse(t *testing.T) {
	// GIVEN a batch formation with minimal KV cache (2 blocks, block size 16)
	config := SimConfig{
		TotalKVBlocks:      2,
		BlockSizeTokens:    16,
		MaxRunningReqs:     10,
		MaxScheduledTokens: 10000,
		Horizon:            1000000,
		BetaCoeffs:         []float64{100, 1, 1},
		AlphaCoeffs:        []float64{100, 1, 100},
	}
	lm, err := NewLatencyModel(config)
	if err != nil {
		t.Fatalf("NewLatencyModel: %v", err)
	}
	bf := NewBatchFormation(config, lm)
	kvCache := NewKVStore(config)

	// AND the running batch is empty
	// AND a request that needs far more blocks than available, in the wait queue
	req := &Request{
		ID:           "large-req",
		InputTokens:  make([]int, 200), // needs ~13 blocks, only 2 available
		OutputTokens: make([]int, 1),
		State:        StateQueued,
	}
	wq := &WaitQueue{}
	wq.Enqueue(req)

	ctx := BatchContext{
		RunningBatch:          &Batch{Requests: []*Request{}},
		WaitQ:                 wq,
		KVCache:               kvCache,
		MaxScheduledTokens:    10000,
		MaxRunningReqs:        10,
		PrefillTokenThreshold: 0,
		Now:                   0,
		StepCount:             0,
		ComputedTokens:        make(map[string]int64),
	}

	// WHEN FormBatch is called
	// THEN it must not panic
	result := bf.FormBatch(ctx)

	// AND the large request must not be in the batch
	for _, r := range result.RunningBatch.Requests {
		if r.ID == "large-req" {
			t.Error("large request should not be in batch when KV blocks insufficient")
		}
	}

	// AND KV cache conservation must hold (INV-4): no blocks leaked
	if kvCache.UsedBlocks() != 0 {
		t.Errorf("expected 0 used blocks after failed allocation on empty batch, got %d", kvCache.UsedBlocks())
	}
}

// TestPreempt_InsufficientBlocks_EvictsAllThenReturnsFalse verifies BC-4 (#297):
// preemption evicts until empty, then stops without panic.
func TestPreempt_InsufficientBlocks_EvictsAllThenReturnsFalse(t *testing.T) {
	// GIVEN a batch formation with very small KV cache
	config := SimConfig{
		TotalKVBlocks:      2,
		BlockSizeTokens:    16,
		MaxRunningReqs:     10,
		MaxScheduledTokens: 10000,
		Horizon:            1000000,
		BetaCoeffs:         []float64{100, 1, 1},
		AlphaCoeffs:        []float64{100, 1, 100},
	}
	lm, err := NewLatencyModel(config)
	if err != nil {
		t.Fatalf("NewLatencyModel: %v", err)
	}
	bf := NewBatchFormation(config, lm)
	kvCache := NewKVStore(config)

	// AND one small request in the running batch with KV blocks allocated
	existing := &Request{
		ID:           "existing",
		InputTokens:  make([]int, 10),
		OutputTokens: make([]int, 1),
		State:        StateRunning,
	}
	if ok := kvCache.AllocateKVBlocks(existing, 0, 10, []int64{}); !ok {
		t.Fatal("setup: failed to allocate KV blocks for existing request")
	}
	existing.ProgressIndex = 10 // past prefill, in decode

	// AND a new request in the wait queue that needs more blocks than total capacity
	huge := &Request{
		ID:           "huge-req",
		InputTokens:  make([]int, 200),
		OutputTokens: make([]int, 1),
		State:        StateQueued,
	}
	wq := &WaitQueue{}
	wq.Enqueue(huge)

	computedTokens := map[string]int64{"existing": 10}
	ctx := BatchContext{
		RunningBatch:          &Batch{Requests: []*Request{existing}},
		WaitQ:                 wq,
		KVCache:               kvCache,
		MaxScheduledTokens:    10000,
		MaxRunningReqs:        10,
		PrefillTokenThreshold: 0,
		Now:                   0,
		StepCount:             0,
		ComputedTokens:        computedTokens,
	}

	// WHEN FormBatch is called
	result := bf.FormBatch(ctx)

	// THEN KV cache conservation must hold (INV-4)
	usedBlocks := kvCache.UsedBlocks()
	totalBlocks := kvCache.TotalCapacity()
	if usedBlocks < 0 || usedBlocks > totalBlocks {
		t.Errorf("KV conservation violated: used=%d total=%d", usedBlocks, totalBlocks)
	}

	// AND the result must not contain the huge request (insufficient blocks)
	for _, r := range result.RunningBatch.Requests {
		if r.ID == "huge-req" {
			t.Error("huge request should not be in batch")
		}
	}
}
