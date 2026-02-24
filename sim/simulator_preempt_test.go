package sim

import (
	"testing"
)

// TestPreempt_EmptyBatch_ReturnsFalse verifies BC-6 (#293):
// preemption with empty batch must not panic.
func TestPreempt_EmptyBatch_ReturnsFalse(t *testing.T) {
	// GIVEN a batch formation with minimal KV cache (2 blocks, block size 16)
	config := SimConfig{
		Horizon:             1000000,
		KVCacheConfig:       NewKVCacheConfig(2, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(10, 10000, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{100, 1, 1}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, false),
	}
	lm, err := MustNewLatencyModel(config.LatencyCoeffs, config.ModelHardwareConfig)
	if err != nil {
		t.Fatalf("MustNewLatencyModel: %v", err)
	}
	bf := NewBatchFormation(lm)
	kvCache := MustNewKVCacheState(config.TotalKVBlocks, config.BlockSizeTokens)

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
// preemption evicts until batch is empty, then circuit breaker fires.
func TestPreempt_InsufficientBlocks_EvictsAllThenReturnsFalse(t *testing.T) {
	// GIVEN a batch formation with very small KV cache (2 blocks * 16 = 32 tokens)
	config := SimConfig{
		Horizon:             1000000,
		KVCacheConfig:       NewKVCacheConfig(2, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(10, 10000, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{100, 1, 1}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, false),
	}
	lm, err := MustNewLatencyModel(config.LatencyCoeffs, config.ModelHardwareConfig)
	if err != nil {
		t.Fatalf("MustNewLatencyModel: %v", err)
	}
	bf := NewBatchFormation(lm)
	kvCache := MustNewKVCacheState(config.TotalKVBlocks, config.BlockSizeTokens)

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

	// AND a huge request also in the running batch at ProgressIndex=0 (needs prefill)
	// This forces Phase 1 to try allocating for huge, fail, and trigger preemption.
	huge := &Request{
		ID:           "huge-req",
		InputTokens:  make([]int, 200), // needs 13 blocks, only 2 total
		OutputTokens: make([]int, 1),
		State:        StateRunning,
	}

	// huge is first in batch (processed first in Phase 1), existing is at tail (evicted first)
	computedTokens := map[string]int64{"existing": 10, "huge-req": 0}
	ctx := BatchContext{
		RunningBatch:          &Batch{Requests: []*Request{huge, existing}},
		WaitQ:                 &WaitQueue{},
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

	// THEN preemption must have happened
	if !result.PreemptionHappened {
		t.Fatal("expected preemption to occur")
	}

	// AND KV cache conservation must hold (INV-4): after full eviction, all blocks freed
	usedBlocks := kvCache.UsedBlocks()
	if usedBlocks != 0 {
		t.Errorf("expected 0 used blocks after all requests evicted, got %d", usedBlocks)
	}

	// AND huge request should not be in the batch (insufficient blocks even after evicting existing)
	for _, r := range result.RunningBatch.Requests {
		if r.ID == "huge-req" {
			t.Error("huge request should not be in batch")
		}
	}
}
