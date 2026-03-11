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
		BatchConfig:         NewBatchConfig(10, 10000, 0, 0, 0, false),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{100, 1, 1}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, "", 0),
	}
	bf := NewBatchFormation()
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
		BatchConfig:         NewBatchConfig(10, 10000, 0, 0, 0, false),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{100, 1, 1}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, "", 0),
	}
	bf := NewBatchFormation()
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

// TestSLOAwareKVEviction_EvictsLowestPriority verifies that when SLOAwareKVEviction
// is enabled, KV preemption targets the lowest-priority running request instead of
// the tail request.
func TestSLOAwareKVEviction_EvictsLowestPriority(t *testing.T) {
	// GIVEN a KV cache with 4 blocks of 16 tokens (64 tokens total)
	kvCache := MustNewKVCacheState(4, 16)
	bf := NewBatchFormation()

	// AND three running requests with different priorities:
	//   critical (pri=10.0) at index 0, input=16 tokens (1 block), in decode
	//   standard (pri=5.0)  at index 1, input=16 tokens (1 block), in decode
	//   sheddable (pri=1.0) at index 2, input=16 tokens (1 block), in decode
	// Total used: 3 blocks, 1 free
	critical := &Request{
		ID: "critical", InputTokens: make([]int, 16), OutputTokens: make([]int, 5),
		State: StateRunning, Priority: 10.0,
	}
	standard := &Request{
		ID: "standard", InputTokens: make([]int, 16), OutputTokens: make([]int, 5),
		State: StateRunning, Priority: 5.0,
	}
	sheddable := &Request{
		ID: "sheddable", InputTokens: make([]int, 16), OutputTokens: make([]int, 5),
		State: StateRunning, Priority: 1.0,
	}
	// Allocate KV blocks during prefill (ProgressIndex=0), then advance to decode
	for _, r := range []*Request{critical, standard, sheddable} {
		if ok := kvCache.AllocateKVBlocks(r, 0, 16, []int64{}); !ok {
			t.Fatalf("setup: failed to allocate KV blocks for %s", r.ID)
		}
		r.ProgressIndex = 16 // advance past prefill into decode
	}

	// AND a new request in the wait queue that needs 2 blocks (32 tokens)
	// (only 1 block free, so eviction is needed)
	newReq := &Request{
		ID: "new-req", InputTokens: make([]int, 32), OutputTokens: make([]int, 1),
		State: StateQueued, Priority: 7.0,
	}
	wq := &WaitQueue{}
	wq.Enqueue(newReq)

	// WHEN FormBatch is called with SLOAwareKVEviction=true
	ctx := BatchContext{
		RunningBatch:          &Batch{Requests: []*Request{critical, standard, sheddable}},
		WaitQ:                 wq,
		KVCache:               kvCache,
		MaxScheduledTokens:    10000,
		MaxRunningReqs:        10,
		PrefillTokenThreshold: 0,
		SLOAwareKVEviction:    true,
		Now:                   1000,
		StepCount:             5,
		ComputedTokens:        map[string]int64{"critical": 16, "standard": 16, "sheddable": 16},
	}
	result := bf.FormBatch(ctx)

	// THEN preemption must have happened
	if !result.PreemptionHappened {
		t.Fatal("expected preemption to occur")
	}

	// AND the sheddable request (lowest priority=1.0) must be evicted
	evictedIDs := map[string]bool{}
	for _, p := range result.Preempted {
		evictedIDs[p.Request.ID] = true
	}
	if !evictedIDs["sheddable"] {
		t.Errorf("expected sheddable (pri=1.0) to be evicted, got evicted: %v", evictedIDs)
	}

	// AND critical must still be in the running batch
	found := false
	for _, r := range result.RunningBatch.Requests {
		if r.ID == "critical" {
			found = true
		}
	}
	if !found {
		t.Error("critical request must remain in batch with SLO-aware eviction")
	}
}

// TestSLOAwareKVEviction_Disabled_EvictsTail verifies that when SLOAwareKVEviction
// is disabled (default), KV preemption evicts the tail request regardless of priority.
func TestSLOAwareKVEviction_Disabled_EvictsTail(t *testing.T) {
	// GIVEN same setup as above but SLOAwareKVEviction=false
	kvCache := MustNewKVCacheState(4, 16)
	bf := NewBatchFormation()

	// Three running requests: critical is at the TAIL (index 2)
	sheddable := &Request{
		ID: "sheddable", InputTokens: make([]int, 16), OutputTokens: make([]int, 5),
		State: StateRunning, Priority: 1.0,
	}
	standard := &Request{
		ID: "standard", InputTokens: make([]int, 16), OutputTokens: make([]int, 5),
		State: StateRunning, Priority: 5.0,
	}
	critical := &Request{
		ID: "critical", InputTokens: make([]int, 16), OutputTokens: make([]int, 5),
		State: StateRunning, Priority: 10.0,
	}
	// Allocate KV blocks during prefill (ProgressIndex=0), then advance to decode
	for _, r := range []*Request{sheddable, standard, critical} {
		if ok := kvCache.AllocateKVBlocks(r, 0, 16, []int64{}); !ok {
			t.Fatalf("setup: failed to allocate KV blocks for %s", r.ID)
		}
		r.ProgressIndex = 16 // advance past prefill into decode
	}

	// New request needing 2 blocks
	newReq := &Request{
		ID: "new-req", InputTokens: make([]int, 32), OutputTokens: make([]int, 1),
		State: StateQueued, Priority: 7.0,
	}
	wq := &WaitQueue{}
	wq.Enqueue(newReq)

	// WHEN FormBatch is called with SLOAwareKVEviction=false (default)
	ctx := BatchContext{
		RunningBatch:          &Batch{Requests: []*Request{sheddable, standard, critical}},
		WaitQ:                 wq,
		KVCache:               kvCache,
		MaxScheduledTokens:    10000,
		MaxRunningReqs:        10,
		PrefillTokenThreshold: 0,
		SLOAwareKVEviction:    false,
		Now:                   1000,
		StepCount:             5,
		ComputedTokens:        map[string]int64{"sheddable": 16, "standard": 16, "critical": 16},
	}
	result := bf.FormBatch(ctx)

	// THEN preemption must have happened
	if !result.PreemptionHappened {
		t.Fatal("expected preemption to occur")
	}

	// AND the TAIL request (critical, at index 2) must be evicted
	// This is the SLO-blind default behavior — it evicts critical because it's at the tail
	evictedIDs := map[string]bool{}
	for _, p := range result.Preempted {
		evictedIDs[p.Request.ID] = true
	}
	if !evictedIDs["critical"] {
		t.Errorf("expected critical (tail) to be evicted with SLO-blind eviction, got evicted: %v", evictedIDs)
	}

	// AND sheddable (lowest priority but NOT at tail) must still be running
	found := false
	for _, r := range result.RunningBatch.Requests {
		if r.ID == "sheddable" {
			found = true
		}
	}
	if !found {
		t.Error("sheddable (not at tail) must remain in batch with SLO-blind eviction")
	}
}
