package sim

import (
	"fmt"
	"testing"
)

// TestVLLMBatchFormation_ImplementsInterface verifies VLLMBatchFormation
// satisfies the BatchFormation interface (compile-time check via variable).
func TestVLLMBatchFormation_ImplementsInterface(t *testing.T) {
	// This is a compile-time check; if it compiles, the interface is satisfied.
	// We also verify the factory returns a working implementation.
	cfg := SimConfig{
		KVCacheConfig: KVCacheConfig{
			TotalKVBlocks:   100,
			BlockSizeTokens: 16,
		},
		BatchConfig: BatchConfig{
			MaxRunningReqs:     10,
			MaxScheduledTokens: 10000,
		},
		LatencyCoeffs: LatencyCoeffs{
			BetaCoeffs:  []float64{100, 1, 1},
			AlphaCoeffs: []float64{100, 1, 100},
		},
	}
	lm, err := NewLatencyModel(cfg)
	if err != nil {
		t.Fatalf("NewLatencyModel: %v", err)
	}
	bf := NewBatchFormation(cfg, lm)
	if bf == nil {
		t.Fatal("NewBatchFormation returned nil")
	}

	// Verify FormBatch works with empty context
	ctx := BatchContext{
		RunningBatch:          &Batch{},
		WaitQ:                 &WaitQueue{},
		KVCache:               NewKVStore(cfg),
		MaxScheduledTokens:    10000,
		MaxRunningReqs:        10,
		PrefillTokenThreshold: 0,
		Now:                   0,
		StepCount:             0,
		ComputedTokens:        make(map[string]int64),
	}
	result := bf.FormBatch(ctx)
	if result.RunningBatch == nil {
		t.Fatal("FormBatch returned nil RunningBatch")
	}
	if len(result.RunningBatch.Requests) != 0 {
		t.Errorf("expected 0 requests in batch from empty context, got %d", len(result.RunningBatch.Requests))
	}
}

// TestVLLMBatchFormation_TokenBudgetEnforced verifies BC-2:
// total new tokens in result batch must not exceed MaxScheduledTokens.
func TestVLLMBatchFormation_TokenBudgetEnforced(t *testing.T) {
	cfg := SimConfig{
		KVCacheConfig: KVCacheConfig{
			TotalKVBlocks:   100,
			BlockSizeTokens: 16,
		},
		BatchConfig: BatchConfig{
			MaxRunningReqs:     10,
			MaxScheduledTokens: 50, // tight token budget
		},
		LatencyCoeffs: LatencyCoeffs{
			BetaCoeffs:  []float64{100, 1, 1},
			AlphaCoeffs: []float64{100, 1, 100},
		},
	}
	lm, err := NewLatencyModel(cfg)
	if err != nil {
		t.Fatalf("NewLatencyModel: %v", err)
	}
	bf := NewBatchFormation(cfg, lm)
	kvCache := NewKVStore(cfg)

	// GIVEN 3 requests in the wait queue, each needing 30 tokens (total 90 > budget 50)
	wq := &WaitQueue{}
	for i := 0; i < 3; i++ {
		wq.Enqueue(&Request{
			ID:           fmt.Sprintf("req-%d", i),
			InputTokens:  make([]int, 30),
			OutputTokens: make([]int, 5),
			State:        StateQueued,
		})
	}

	ctx := BatchContext{
		RunningBatch:          &Batch{},
		WaitQ:                 wq,
		KVCache:               kvCache,
		MaxScheduledTokens:    50,
		MaxRunningReqs:        10,
		PrefillTokenThreshold: 0,
		Now:                   1000,
		StepCount:             1,
		ComputedTokens:        make(map[string]int64),
	}

	// WHEN FormBatch is called
	result := bf.FormBatch(ctx)

	// THEN total new tokens must not exceed budget
	var totalNewTokens int
	for _, req := range result.RunningBatch.Requests {
		totalNewTokens += req.NumNewTokens
	}
	if int64(totalNewTokens) > 50 {
		t.Errorf("token budget exceeded: total new tokens %d > budget 50", totalNewTokens)
	}

	// AND at least one request should be scheduled (budget allows first request's 30 tokens)
	if len(result.RunningBatch.Requests) == 0 {
		t.Error("expected at least one request scheduled")
	}
}

// TestVLLMBatchFormation_BatchSizeEnforced verifies BC-3:
// batch size must not exceed MaxRunningReqs.
func TestVLLMBatchFormation_BatchSizeEnforced(t *testing.T) {
	cfg := SimConfig{
		KVCacheConfig: KVCacheConfig{
			TotalKVBlocks:   200,
			BlockSizeTokens: 16,
		},
		BatchConfig: BatchConfig{
			MaxRunningReqs:     2, // tight batch size limit
			MaxScheduledTokens: 10000,
		},
		LatencyCoeffs: LatencyCoeffs{
			BetaCoeffs:  []float64{100, 1, 1},
			AlphaCoeffs: []float64{100, 1, 100},
		},
	}
	lm, err := NewLatencyModel(cfg)
	if err != nil {
		t.Fatalf("NewLatencyModel: %v", err)
	}
	bf := NewBatchFormation(cfg, lm)
	kvCache := NewKVStore(cfg)

	// GIVEN 5 requests in the wait queue
	wq := &WaitQueue{}
	for i := 0; i < 5; i++ {
		wq.Enqueue(&Request{
			ID:           fmt.Sprintf("req-%d", i),
			InputTokens:  make([]int, 10),
			OutputTokens: make([]int, 5),
			State:        StateQueued,
		})
	}

	ctx := BatchContext{
		RunningBatch:          &Batch{},
		WaitQ:                 wq,
		KVCache:               kvCache,
		MaxScheduledTokens:    10000,
		MaxRunningReqs:        2,
		PrefillTokenThreshold: 0,
		Now:                   1000,
		StepCount:             1,
		ComputedTokens:        make(map[string]int64),
	}

	// WHEN FormBatch is called
	result := bf.FormBatch(ctx)

	// THEN batch size must not exceed 2
	if len(result.RunningBatch.Requests) > 2 {
		t.Errorf("batch size exceeded: got %d > limit 2", len(result.RunningBatch.Requests))
	}

	// AND exactly 2 should be scheduled (enough tokens and KV blocks)
	if len(result.RunningBatch.Requests) != 2 {
		t.Errorf("expected 2 requests scheduled, got %d", len(result.RunningBatch.Requests))
	}

	// AND 3 should remain in wait queue
	if wq.Len() != 3 {
		t.Errorf("expected 3 remaining in wait queue, got %d", wq.Len())
	}
}

// TestVLLMBatchFormation_PreemptionReleasesKV verifies BC-4:
// preempted requests must have KV blocks released and appear in result.Preempted.
func TestVLLMBatchFormation_PreemptionReleasesKV(t *testing.T) {
	// 3 blocks * 16 tokens/block = 48 token capacity
	cfg := SimConfig{
		KVCacheConfig: KVCacheConfig{
			TotalKVBlocks:   3,
			BlockSizeTokens: 16,
		},
		BatchConfig: BatchConfig{
			MaxRunningReqs:     10,
			MaxScheduledTokens: 10000,
		},
		LatencyCoeffs: LatencyCoeffs{
			BetaCoeffs:  []float64{100, 1, 1},
			AlphaCoeffs: []float64{100, 1, 100},
		},
	}
	lm, err := NewLatencyModel(cfg)
	if err != nil {
		t.Fatalf("NewLatencyModel: %v", err)
	}
	bf := NewBatchFormation(cfg, lm)
	kvCache := NewKVStore(cfg)

	// GIVEN two running requests: victim occupies 2 blocks, needy needs 3 blocks for prefill
	victim := &Request{
		ID:           "victim",
		InputTokens:  make([]int, 20), // ceil(20/16) = 2 blocks
		OutputTokens: make([]int, 5),
		State:        StateRunning,
	}
	if ok := kvCache.AllocateKVBlocks(victim, 0, 20, []int64{}); !ok {
		t.Fatal("setup: failed to allocate KV blocks for victim")
	}
	victim.ProgressIndex = 20 // prefill complete, in decode phase

	needy := &Request{
		ID:           "needy",
		InputTokens:  make([]int, 40), // ceil(40/16) = 3 blocks, but only 1 free
		OutputTokens: make([]int, 5),
		State:        StateRunning,
	}
	// needy is in running batch with ProgressIndex=0, so Phase 1 prefill triggers preemptForTokens

	computedTokens := map[string]int64{"victim": 20, "needy": 0}
	ctx := BatchContext{
		// victim is at end (tail) — it will be evicted first during preemption
		RunningBatch:          &Batch{Requests: []*Request{needy, victim}},
		WaitQ:                 &WaitQueue{},
		KVCache:               kvCache,
		MaxScheduledTokens:    10000,
		MaxRunningReqs:        10,
		PrefillTokenThreshold: 0,
		Now:                   5000,
		StepCount:             5,
		ComputedTokens:        computedTokens,
	}

	// WHEN FormBatch is called
	result := bf.FormBatch(ctx)

	// THEN preemption must have happened (needy needed 3 blocks, only 1 free → evicts victim)
	if !result.PreemptionHappened {
		t.Fatal("expected preemption to occur: needy needs 3 blocks, only 1 free before eviction")
	}

	// AND preempted requests must appear in result.Preempted
	if len(result.Preempted) == 0 {
		t.Error("PreemptionHappened is true but Preempted slice is empty")
	}

	// AND KV usage must not exceed capacity (INV-4)
	usedAfter := kvCache.UsedBlocks()
	if usedAfter > kvCache.TotalCapacity() {
		t.Errorf("KV blocks exceed capacity after preemption: used=%d total=%d", usedAfter, kvCache.TotalCapacity())
	}
}

// TestVLLMBatchFormation_PreemptionStopsDequeue verifies BC-5:
// no new requests dequeued after preemption.
func TestVLLMBatchFormation_PreemptionStopsDequeue(t *testing.T) {
	cfg := SimConfig{
		KVCacheConfig: KVCacheConfig{
			TotalKVBlocks:   3, // very tight
			BlockSizeTokens: 16,
		},
		BatchConfig: BatchConfig{
			MaxRunningReqs:     10,
			MaxScheduledTokens: 10000,
		},
		LatencyCoeffs: LatencyCoeffs{
			BetaCoeffs:  []float64{100, 1, 1},
			AlphaCoeffs: []float64{100, 1, 100},
		},
	}
	lm, err := NewLatencyModel(cfg)
	if err != nil {
		t.Fatalf("NewLatencyModel: %v", err)
	}
	bf := NewBatchFormation(cfg, lm)
	kvCache := NewKVStore(cfg)

	// GIVEN two running requests where req2's prefill will trigger preemption
	req1 := &Request{ID: "r1", InputTokens: make([]int, 20), OutputTokens: make([]int, 5), State: StateRunning}
	req2 := &Request{ID: "r2", InputTokens: make([]int, 20), OutputTokens: make([]int, 5), State: StateRunning}

	// Allocate blocks for req1 (fills most of cache)
	if ok := kvCache.AllocateKVBlocks(req1, 0, 20, []int64{}); !ok {
		t.Fatal("setup: failed to allocate for r1")
	}
	req1.ProgressIndex = 20 // decode phase

	// req2 has ProgressIndex=0, so Phase 1 will try to allocate for its full prefill

	// AND a waiting request that should NOT be dequeued after preemption
	waitReq := &Request{ID: "wait", InputTokens: make([]int, 5), OutputTokens: make([]int, 2), State: StateQueued}
	wq := &WaitQueue{}
	wq.Enqueue(waitReq)

	computedTokens := map[string]int64{"r1": 20, "r2": 0}
	ctx := BatchContext{
		RunningBatch:          &Batch{Requests: []*Request{req1, req2}},
		WaitQ:                 wq,
		KVCache:               kvCache,
		MaxScheduledTokens:    10000,
		MaxRunningReqs:        10,
		PrefillTokenThreshold: 0,
		Now:                   5000,
		StepCount:             5,
		ComputedTokens:        computedTokens,
	}

	result := bf.FormBatch(ctx)

	// THEN preemption must have occurred (precondition for BC-5)
	if !result.PreemptionHappened {
		t.Fatal("expected preemption to occur — test precondition failed")
	}

	// AND no new requests should have been dequeued after preemption
	if len(result.NewlyScheduled) > 0 {
		t.Errorf("expected 0 newly scheduled after preemption, got %d", len(result.NewlyScheduled))
	}
}

// TestVLLMBatchFormation_CircuitBreaker verifies BC-6:
// empty batch + insufficient KV blocks must not panic.
func TestVLLMBatchFormation_CircuitBreaker(t *testing.T) {
	cfg := SimConfig{
		KVCacheConfig: KVCacheConfig{
			TotalKVBlocks:   2, // very small
			BlockSizeTokens: 16,
		},
		BatchConfig: BatchConfig{
			MaxRunningReqs:     10,
			MaxScheduledTokens: 10000,
		},
		LatencyCoeffs: LatencyCoeffs{
			BetaCoeffs:  []float64{100, 1, 1},
			AlphaCoeffs: []float64{100, 1, 100},
		},
	}
	lm, err := NewLatencyModel(cfg)
	if err != nil {
		t.Fatalf("NewLatencyModel: %v", err)
	}
	bf := NewBatchFormation(cfg, lm)
	kvCache := NewKVStore(cfg)

	// GIVEN a request needing more blocks than total capacity
	huge := &Request{ID: "huge", InputTokens: make([]int, 200), OutputTokens: make([]int, 5), State: StateQueued}
	wq := &WaitQueue{}
	wq.Enqueue(huge)

	ctx := BatchContext{
		RunningBatch:          &Batch{},
		WaitQ:                 wq,
		KVCache:               kvCache,
		MaxScheduledTokens:    10000,
		MaxRunningReqs:        10,
		PrefillTokenThreshold: 0,
		Now:                   0,
		StepCount:             0,
		ComputedTokens:        make(map[string]int64),
	}

	// WHEN FormBatch is called — must not panic
	result := bf.FormBatch(ctx)

	// THEN the huge request should not be in the batch
	for _, req := range result.RunningBatch.Requests {
		if req.ID == "huge" {
			t.Error("huge request should not be in batch when KV allocation fails")
		}
	}

	// AND KV conservation holds
	if kvCache.UsedBlocks() != 0 {
		t.Errorf("expected 0 used blocks, got %d", kvCache.UsedBlocks())
	}
}

// TestVLLMBatchFormation_KVAllocationFailure_StopsDequeue verifies BC-9:
// when KV allocation fails for a wait queue request, no further requests are dequeued.
func TestVLLMBatchFormation_KVAllocationFailure_StopsDequeue(t *testing.T) {
	cfg := SimConfig{
		KVCacheConfig: KVCacheConfig{
			TotalKVBlocks:   3, // limited KV blocks
			BlockSizeTokens: 16,
		},
		BatchConfig: BatchConfig{
			MaxRunningReqs:     10,
			MaxScheduledTokens: 10000,
		},
		LatencyCoeffs: LatencyCoeffs{
			BetaCoeffs:  []float64{100, 1, 1},
			AlphaCoeffs: []float64{100, 1, 100},
		},
	}
	lm, err := NewLatencyModel(cfg)
	if err != nil {
		t.Fatalf("NewLatencyModel: %v", err)
	}
	bf := NewBatchFormation(cfg, lm)
	kvCache := NewKVStore(cfg)

	// GIVEN: first request fits, second needs too many blocks, third is small but can't skip
	req1 := &Request{ID: "small", InputTokens: make([]int, 16), OutputTokens: make([]int, 2), State: StateQueued}
	req2 := &Request{ID: "big", InputTokens: make([]int, 100), OutputTokens: make([]int, 2), State: StateQueued}
	req3 := &Request{ID: "also-small", InputTokens: make([]int, 10), OutputTokens: make([]int, 2), State: StateQueued}

	wq := &WaitQueue{}
	wq.Enqueue(req1)
	wq.Enqueue(req2)
	wq.Enqueue(req3)

	ctx := BatchContext{
		RunningBatch:          &Batch{},
		WaitQ:                 wq,
		KVCache:               kvCache,
		MaxScheduledTokens:    10000,
		MaxRunningReqs:        10,
		PrefillTokenThreshold: 0,
		Now:                   1000,
		StepCount:             1,
		ComputedTokens:        make(map[string]int64),
	}

	// WHEN FormBatch is called
	result := bf.FormBatch(ctx)

	// THEN req1 should be scheduled (enough blocks)
	foundSmall := false
	for _, r := range result.RunningBatch.Requests {
		if r.ID == "small" {
			foundSmall = true
		}
	}
	if !foundSmall {
		t.Error("expected 'small' request to be scheduled")
	}

	// AND req2 should NOT be scheduled (allocation fails)
	for _, r := range result.RunningBatch.Requests {
		if r.ID == "big" {
			t.Error("'big' request should not be scheduled when KV allocation fails")
		}
	}

	// AND req3 should NOT be scheduled (FCFS: can't skip req2)
	for _, r := range result.RunningBatch.Requests {
		if r.ID == "also-small" {
			t.Error("'also-small' should not be scheduled — FCFS prevents skipping failed req2")
		}
	}
}

// TestPreemptForTokens_CleansUpComputedTokens verifies BC-6:
// preempted request's ComputedTokens entry is deleted.
func TestPreemptForTokens_CleansUpComputedTokens(t *testing.T) {
	// GIVEN a running request with a ComputedTokens entry
	kv := NewKVStore(SimConfig{TotalKVBlocks: 4, BlockSizeTokens: 4})
	victim := &Request{
		ID:           "victim",
		InputTokens:  []int{1, 2, 3, 4, 5, 6, 7, 8},
		OutputTokens: []int{100},
		State:        StateRunning,
	}
	kv.AllocateKVBlocks(victim, 0, 8, []int64{})
	victim.ProgressIndex = 8 // simulate completed prefill
	computedTokens := map[string]int64{victim.ID: 8}

	ctx := BatchContext{
		RunningBatch:   &Batch{Requests: []*Request{victim}},
		WaitQ:          &WaitQueue{},
		KVCache:        kv,
		ComputedTokens: computedTokens,
		Now:            1000,
	}
	result := BatchResult{RunningBatch: ctx.RunningBatch}

	// A new request that needs more tokens than available
	newReq := &Request{
		ID:           "newcomer",
		InputTokens:  []int{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160},
		OutputTokens: []int{100},
	}
	bf := &VLLMBatchFormation{latencyModel: &BlackboxLatencyModel{
		betaCoeffs:  []float64{100, 1, 1},
		alphaCoeffs: []float64{100, 1, 100},
	}}

	// WHEN preemption evicts victim to make room for newcomer
	bf.preemptForTokens(newReq, 16, &result, ctx)

	// THEN ComputedTokens should NOT contain the preempted request's entry
	if _, exists := computedTokens[victim.ID]; exists {
		t.Error("preempted request should be removed from ComputedTokens")
	}
}
