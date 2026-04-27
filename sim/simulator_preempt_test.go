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
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "", "", 1, "roofline", 0),
	}
	bf := NewBatchFormation("", nil)
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
		MaxModelLen:           0,
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
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "", "", 1, "roofline", 0),
	}
	bf := NewBatchFormation("", nil)
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
		MaxModelLen:           0,
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

func TestNewSimulator_CustomSLOPriorityMap_AffectsPreemption(t *testing.T) {
	// BC-1: Custom slo_priorities override default preemption victim selection.
	// Default GAIE priorities: batch=-1, background=-3, critical=4.
	// Override: batch=0 (promoted from -1 to non-sheddable).
	// Setup: 10 blocks × 16 tokens = 160 capacity. 3 running × 48 tokens = 3 blocks each = 9/10 used, 1 free.
	// Phase 1: crit decode uses the 1 free block (10/10 used).
	// Phase 1: batch decode → full → preemption. With override: background(-3) < batch(0) → bg evicted.
	cfg := SimConfig{
		Horizon:             100_000_000,
		KVCacheConfig:       NewKVCacheConfig(10, 16, 0, 0.0, 0.0, 0.0),
		BatchConfig:         NewBatchConfig(10, 10000, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{0, 0, 0}, []float64{100, 1, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "", "", 1, "roofline", 0),
		PolicyConfig:        NewPolicyConfig("constant", "fcfs", "priority"),
		SLOPriorityOverrides: map[string]int{"batch": 0},
	}
	s := mustNewSimulator(t, cfg)

	// Inject three running requests to saturate cache (9/10 blocks used).
	// Distinct input tokens prevent prefix cache sharing across requests.
	critTokens := make([]int, 48)
	for i := range critTokens {
		critTokens[i] = i + 1
	}
	batchTokens := make([]int, 48)
	for i := range batchTokens {
		batchTokens[i] = i + 101
	}
	bgTokens := make([]int, 48)
	for i := range bgTokens {
		bgTokens[i] = i + 201
	}
	critReq := &Request{ID: "crit", SLOClass: "critical", ArrivalTime: 100, State: StateRunning,
		InputTokens: critTokens, OutputTokens: make([]int, 10)}
	batchReq := &Request{ID: "batch-req", SLOClass: "batch", ArrivalTime: 200, State: StateRunning,
		InputTokens: batchTokens, OutputTokens: make([]int, 10)}
	bgReq := &Request{ID: "bg-req", SLOClass: "background", ArrivalTime: 300, State: StateRunning,
		InputTokens: bgTokens, OutputTokens: make([]int, 10)}

	for _, req := range []*Request{critReq, batchReq, bgReq} {
		s.KVCache.AllocateKVBlocks(req, 0, 48, nil)
		req.ProgressIndex = 48
	}
	s.RunningBatch = &Batch{Requests: []*Request{critReq, batchReq, bgReq}}
	s.WaitQ.Enqueue(&Request{ID: "new", InputTokens: make([]int, 16), OutputTokens: make([]int, 1), State: StateQueued})

	result := s.batchFormation.FormBatch(BatchContext{
		RunningBatch:       s.RunningBatch,
		WaitQ:              s.WaitQ,
		KVCache:            s.KVCache,
		MaxScheduledTokens: 10000,
		MaxRunningReqs:     10,
		Now:                1000,
		ComputedTokens:     make(map[string]int64),
	})

	// With batch=0 override: background(-3) is least urgent → must be evicted.
	if len(result.Preempted) == 0 {
		t.Fatal("expected preemption but got none")
	}
	if result.Preempted[0].Request.ID != "bg-req" {
		t.Errorf("expected bg-req evicted (background=-3 < batch=0 with override), got %q",
			result.Preempted[0].Request.ID)
	}
	// batch-req must still be running (priority=0 with override > background=-3)
	runningIDs := make(map[string]bool)
	for _, r := range result.RunningBatch.Requests {
		runningIDs[r.ID] = true
	}
	if !runningIDs["batch-req"] {
		t.Error("batch-req should still be running (priority=0 with override, more urgent than background=-3)")
	}
}
