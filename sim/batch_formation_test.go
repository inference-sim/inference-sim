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
		KVCacheConfig:       NewKVCacheConfig(100, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(10, 10000, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{100, 1, 1}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "", "", 1, "roofline", 0),
	}
	bf := NewBatchFormation("", nil)
	if bf == nil {
		t.Fatal("NewBatchFormation returned nil")
	}

	// Verify FormBatch works with empty context
	ctx := BatchContext{
		RunningBatch:          &Batch{},
		WaitQ:                 &WaitQueue{},
		KVCache:               MustNewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens),
		MaxScheduledTokens:    10000,
		MaxRunningReqs:        10,
		PrefillTokenThreshold: 0,
		MaxModelLen:           0,
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
		KVCacheConfig:       NewKVCacheConfig(100, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(10, 50, 0), // tight token budget
		LatencyCoeffs:       NewLatencyCoeffs([]float64{100, 1, 1}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "", "", 1, "roofline", 0),
	}
	bf := NewBatchFormation("", nil)
	kvCache := MustNewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)

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
		MaxModelLen:           0,
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
		KVCacheConfig:       NewKVCacheConfig(200, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(2, 10000, 0), // tight batch size limit
		LatencyCoeffs:       NewLatencyCoeffs([]float64{100, 1, 1}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "", "", 1, "roofline", 0),
	}
	bf := NewBatchFormation("", nil)
	kvCache := MustNewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)

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
		MaxModelLen:           0,
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
		KVCacheConfig:       NewKVCacheConfig(3, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(10, 10000, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{100, 1, 1}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "", "", 1, "roofline", 0),
	}
	bf := NewBatchFormation("", nil)
	kvCache := MustNewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)

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
		MaxModelLen:           0,
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
		KVCacheConfig:       NewKVCacheConfig(3, 16, 0, 0, 0, 0), // very tight
		BatchConfig:         NewBatchConfig(10, 10000, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{100, 1, 1}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "", "", 1, "roofline", 0),
	}
	bf := NewBatchFormation("", nil)
	kvCache := MustNewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)

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
		MaxModelLen:           0,
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
		KVCacheConfig:       NewKVCacheConfig(2, 16, 0, 0, 0, 0), // very small
		BatchConfig:         NewBatchConfig(10, 10000, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{100, 1, 1}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "", "", 1, "roofline", 0),
	}
	bf := NewBatchFormation("", nil)
	kvCache := MustNewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)

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
		MaxModelLen:           0,
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
		KVCacheConfig:       NewKVCacheConfig(3, 16, 0, 0, 0, 0), // limited KV blocks
		BatchConfig:         NewBatchConfig(10, 10000, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{100, 1, 1}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "", "", 1, "roofline", 0),
	}
	bf := NewBatchFormation("", nil)
	kvCache := MustNewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)

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
		MaxModelLen:           0,
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
	kv := MustNewKVCacheState(4, 4)
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
		MaxModelLen:    0,
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
	bf := &VLLMBatchFormation{preemptionPolicy: PreemptionFCFS, sloMap: DefaultSLOPriorityMap()}

	// WHEN preemption evicts victim to make room for newcomer
	var budget int64 = 10000
	bf.preemptForTokens(newReq, 16, &result, ctx, &budget, 0)

	// THEN ComputedTokens should NOT contain the preempted request's entry
	if _, exists := computedTokens[victim.ID]; exists {
		t.Error("preempted request should be removed from ComputedTokens")
	}
}

// TestVLLMBatchFormation_Phase1_EvictedNotRevisited verifies FIX-1 and FIX-4:
// Phase 1 must not visit requests that were evicted by preemptForTokens.
// The old range-based loop continued iterating over evicted requests (ProgressIndex=0),
// causing cascading re-prefill allocations and state corruption.
func TestVLLMBatchFormation_Phase1_EvictedNotRevisited(t *testing.T) {
	// 6 blocks * 16 tokens = 96 token capacity
	cfg := SimConfig{
		KVCacheConfig:       NewKVCacheConfig(6, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(10, 10000, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{0, 0, 0}, []float64{100, 1, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "", "", 1, "roofline", 0),
	}
	bf := NewBatchFormation("", nil)
	kvCache := MustNewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)

	// GIVEN 3 running requests, all in decode phase with KV fully allocated:
	// r1 uses 3 blocks (48 tokens, exact multiple of 16 → last block full)
	// r2 uses 2 blocks (31 tokens, partial last block: 15/16 filled)
	// r3 uses 1 block (16 tokens, exact multiple → last block full)
	// Total: 6 blocks = full cache.
	// r1's decode needs a NEW block (last block full) → triggers preemption of r3 (tail).
	// r2's decode fills its partial block (15→16) → NO new block needed, r2 survives.
	r1 := &Request{ID: "r1", InputTokens: make([]int, 48), OutputTokens: make([]int, 100), State: StateRunning}
	r2 := &Request{ID: "r2", InputTokens: make([]int, 31), OutputTokens: make([]int, 100), State: StateRunning}
	r3 := &Request{ID: "r3", InputTokens: make([]int, 16), OutputTokens: make([]int, 100), State: StateRunning}

	if ok := kvCache.AllocateKVBlocks(r1, 0, 48, []int64{}); !ok {
		t.Fatal("setup: allocate r1")
	}
	r1.ProgressIndex = 48

	if ok := kvCache.AllocateKVBlocks(r2, 0, 31, []int64{}); !ok {
		t.Fatal("setup: allocate r2")
	}
	r2.ProgressIndex = 31

	if ok := kvCache.AllocateKVBlocks(r3, 0, 16, []int64{}); !ok {
		t.Fatal("setup: allocate r3")
	}
	r3.ProgressIndex = 16
	r3.NumNewTokens = 5 // Stale value from prior step — FIX-2 zeroing must clear this

	if kvCache.UsedBlocks() != 6 {
		t.Fatalf("setup: expected 6 used blocks, got %d", kvCache.UsedBlocks())
	}

	computedTokens := map[string]int64{"r1": 48, "r2": 31, "r3": 16}
	ctx := BatchContext{
		RunningBatch:          &Batch{Requests: []*Request{r1, r2, r3}},
		WaitQ:                 &WaitQueue{},
		KVCache:               kvCache,
		MaxScheduledTokens:    10000,
		MaxRunningReqs:        10,
		PrefillTokenThreshold: 0,
		MaxModelLen:           0,
		Now:                   5000,
		StepCount:             5,
		ComputedTokens:        computedTokens,
	}

	result := bf.FormBatch(ctx)

	// THEN r3 must be preempted (tail eviction to make room for r1's decode)
	if !result.PreemptionHappened {
		t.Fatal("expected preemption to occur")
	}

	// AND preemption count must be exactly 1 (only r3 evicted — no cascading)
	if len(result.Preempted) != 1 {
		t.Errorf("FIX-1: expected exactly 1 preemption (r3), got %d", len(result.Preempted))
	}

	// AND FIX-2: r3's stale NumNewTokens (5) must have been zeroed at FormBatch entry,
	// so preemption did NOT inflate the budget by 5.
	if len(result.Preempted) == 1 && result.Preempted[0].Request.NumNewTokens != 0 {
		t.Errorf("FIX-2: preempted r3 should have NumNewTokens=0 (zeroed at entry), got %d",
			result.Preempted[0].Request.NumNewTokens)
	}

	// AND r1 and r2 must still be in the running batch, r3 must not
	batchIDs := make(map[string]bool)
	for _, req := range result.RunningBatch.Requests {
		batchIDs[req.ID] = true
	}
	if !batchIDs["r1"] || !batchIDs["r2"] {
		t.Errorf("FIX-1: r1 and r2 must remain in batch, got %v", batchIDs)
	}
	if batchIDs["r3"] {
		t.Error("FIX-4: evicted request r3 must not be in running batch")
	}

	// AND KV conservation must hold (INV-4):
	// r1: 3 blocks (48 tokens) + 1 new block (decode at block boundary) = 4 blocks
	// r2: 2 blocks (31 tokens, decode fills partial block 15→16, no new block)
	// r3: freed (1 block released, used by r1's decode allocation)
	expectedUsed := int64(4 + 2) // r1=4, r2=2
	if kvCache.UsedBlocks() != expectedUsed {
		t.Errorf("INV-4: expected %d used blocks after preemption, got %d", expectedUsed, kvCache.UsedBlocks())
	}
}

// TestVLLMBatchFormation_LivelockResolution verifies FIX-3:
// The pathological workload from #349 (seed=7, 7463 blocks, output 3200-3596 tokens)
// must complete requests instead of livelocking with 100K+ preemptions.
func TestVLLMBatchFormation_LivelockResolution(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	// GIVEN parameters matching tmp/run.sh reproducer
	// Note: NewLatencyCoeffs takes (betaCoeffs, alphaCoeffs)
	cfg := SimConfig{
		Horizon: 120000000,
		Seed:    7,
		KVCacheConfig: NewKVCacheConfig(
			7463, // total blocks
			16,   // block size
			0, 0, 0, 0,
		),
		BatchConfig: NewBatchConfig(
			256,  // max running reqs
			2048, // max scheduled tokens
			0,    // long prefill threshold (disabled)
		),
		LatencyCoeffs: NewLatencyCoeffs(
			[]float64{5752.705191348184, 17.25086436834028, 5.999143920128404},   // beta
			[]float64{232.46191091038054, 1.752360364195244, 3357.4400353290152}, // alpha
		),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "", "", 1, "roofline", 0),
		PolicyConfig:        NewPolicyConfig("constant", "fcfs", ""),
	}

	sim := mustNewSimulator(t, cfg)

	// Inject 30 requests (subset of 300 for test speed) with the workload profile.
	// Uses its own RNG for workload generation (independent from simulator's RNG).
	rng := NewPartitionedRNG(NewSimulationKey(7))
	wlRng := rng.ForSubsystem(SubsystemWorkload)
	arrivalTime := int64(0)
	for i := 0; i < 30; i++ {
		inputLen := 200 + wlRng.Intn(201)  // 200-400
		outputLen := 3200 + wlRng.Intn(397) // 3200-3596
		req := &Request{
			ID:           fmt.Sprintf("req_%d", i),
			InputTokens:  make([]int, inputLen),
			OutputTokens: make([]int, outputLen),
			ArrivalTime:  arrivalTime,
			State:        StateQueued,
		}
		sim.InjectArrival(req)
		arrivalTime += 100000 // 100ms between arrivals (rate=10/s)
	}

	sim.Run()

	// THEN some requests must complete (livelock resolved)
	if sim.Metrics.CompletedRequests == 0 {
		t.Errorf("FIX-3: expected completed_requests > 0, got 0 (livelock not resolved)")
	}

	// AND preemption count must be dramatically reduced (not 100K+)
	if sim.Metrics.PreemptionCount > 1000 {
		t.Errorf("FIX-3: expected preemption_count < 1000, got %d (cascading preemption not resolved)",
			sim.Metrics.PreemptionCount)
	}

	// AND request conservation must hold (INV-1)
	total := sim.Metrics.CompletedRequests + sim.Metrics.StillQueued + sim.Metrics.StillRunning + sim.Metrics.DroppedUnservable
	if total != 30 {
		t.Errorf("INV-1: completed(%d) + queued(%d) + running(%d) + dropped(%d) = %d, expected 30",
			sim.Metrics.CompletedRequests, sim.Metrics.StillQueued, sim.Metrics.StillRunning,
			sim.Metrics.DroppedUnservable, total)
	}

	t.Logf("Results: completed=%d, queued=%d, running=%d, dropped=%d, preemptions=%d",
		sim.Metrics.CompletedRequests, sim.Metrics.StillQueued, sim.Metrics.StillRunning,
		sim.Metrics.DroppedUnservable, sim.Metrics.PreemptionCount)
}

// TestVLLMBatchFormation_MaxModelLen_ProactiveCap_Decode verifies BC-1:
// Running request near MaxModelLen boundary gets decode tokens clamped to 0.
func TestVLLMBatchFormation_MaxModelLen_ProactiveCap_Decode(t *testing.T) {
	kvStore := MustNewKVCacheState(1000, 16)
	bf := NewBatchFormation("", nil)

	// Request at ProgressIndex=99, MaxModelLen=100 → decode clamped (99+1 > 100-1)
	// No KV pre-allocation needed: FormBatch sets decodeTokens=0 at boundary,
	// so preemptForTokens is never called. The request stays in the batch with NumNewTokens=0.
	req := &Request{
		ID:            "near_cap",
		InputTokens:   make([]int, 50),
		OutputTokens:  make([]int, 200),
		State:         StateRunning,
		ProgressIndex: 99,
	}

	ctx := BatchContext{
		RunningBatch:       &Batch{Requests: []*Request{req}},
		WaitQ:              &WaitQueue{},
		KVCache:            kvStore,
		MaxScheduledTokens: 2048,
		MaxRunningReqs:     256,
		PrefillTokenThreshold: 0,
		MaxModelLen:        100,
		Now:                0,
		StepCount:          0,
		ComputedTokens:     make(map[string]int64),
	}
	bf.FormBatch(ctx)

	if req.NumNewTokens != 0 {
		t.Errorf("NumNewTokens = %d, want 0 (proactive cap: PI+1 > maxModelLen-1)", req.NumNewTokens)
	}
}

// TestVLLMBatchFormation_MaxModelLen_ProactiveCap_Phase2 verifies BC-2:
// New request prefill tokens clamped by MaxModelLen.
func TestVLLMBatchFormation_MaxModelLen_ProactiveCap_Phase2(t *testing.T) {
	kvStore := MustNewKVCacheState(1000, 16)
	bf := NewBatchFormation("", nil)

	// Defense-in-depth test: input=80 > MaxModelLen=50 would be rejected by
	// EnqueueRequest in production. Testing FormBatch cap in isolation.
	req := &Request{
		ID:           "new_req",
		InputTokens:  make([]int, 80),
		OutputTokens: make([]int, 50),
		State:        StateQueued,
	}
	wq := &WaitQueue{}
	wq.Enqueue(req)

	ctx := BatchContext{
		RunningBatch:       &Batch{},
		WaitQ:              wq,
		KVCache:            kvStore,
		MaxScheduledTokens: 2048,
		MaxRunningReqs:     256,
		PrefillTokenThreshold: 0,
		MaxModelLen:        50,
		Now:                0,
		StepCount:          0,
		ComputedTokens:     make(map[string]int64),
	}
	bf.FormBatch(ctx)

	// Proactive cap: min(80, 2048, max(0, 50-1-0)) = min(80, 2048, 49) = 49
	if req.NumNewTokens != 49 {
		t.Errorf("NumNewTokens = %d, want 49 (proactive cap: max(0, 50-1-0)=49)", req.NumNewTokens)
	}
}

// TestVLLMBatchFormation_MaxModelLen_Zero_NoClamp verifies BC-3.
func TestVLLMBatchFormation_MaxModelLen_Zero_NoClamp(t *testing.T) {
	kvStore := MustNewKVCacheState(10000, 16)
	bf := NewBatchFormation("", nil)

	req := &Request{
		ID:           "unlimited",
		InputTokens:  make([]int, 500),
		OutputTokens: make([]int, 500),
		State:        StateQueued,
	}
	wq := &WaitQueue{}
	wq.Enqueue(req)

	ctx := BatchContext{
		RunningBatch:       &Batch{},
		WaitQ:              wq,
		KVCache:            kvStore,
		MaxScheduledTokens: 10000,
		MaxRunningReqs:     256,
		PrefillTokenThreshold: 0,
		MaxModelLen:        0, // unlimited
		Now:                0,
		StepCount:          0,
		ComputedTokens:     make(map[string]int64),
	}
	bf.FormBatch(ctx)

	if req.NumNewTokens != 500 {
		t.Errorf("NumNewTokens = %d, want 500 (MaxModelLen=0 = no clamp)", req.NumNewTokens)
	}
}

// TestVLLMBatchFormation_ZeroInputRequest_SkipsDecodeOnlyPath is a regression
// test for the IsDecodeSubRequest flag introduced in FormBatch Phase 2. Before
// the flag, the decode-only fast-path was guarded by a ProgressIndex heuristic
// (ProgressIndex >= inputLen) that a zero-input non-PD request satisfied
// vacuously (0 >= 0), causing it to skip prefill KV allocation and produce a
// phantom output token in ComputedTokens. IsDecodeSubRequest is set only by
// KVTransferCompletedEvent, so non-PD requests can never reach this path.
func TestVLLMBatchFormation_ZeroInputRequest_SkipsDecodeOnlyPath(t *testing.T) {
	kvStore := MustNewKVCacheState(10000, 16)
	bf := NewBatchFormation("", nil)

	// A non-PD request: ProgressIndex stays 0 (never set by AllocateTransferredKV).
	req := &Request{
		ID:           "zero-input",
		InputTokens:  nil, // inputLen == 0
		OutputTokens: make([]int, 5),
		State:        StateQueued,
		// ProgressIndex defaults to 0
	}

	wq := &WaitQueue{}
	wq.Enqueue(req)

	computedTokens := make(map[string]int64)
	ctx := BatchContext{
		RunningBatch:          &Batch{},
		WaitQ:                 wq,
		KVCache:               kvStore,
		MaxScheduledTokens:    10000,
		MaxRunningReqs:        10,
		PrefillTokenThreshold: 0,
		MaxModelLen:           0,
		Now:                   0,
		StepCount:             0,
		ComputedTokens:        computedTokens,
	}
	bf.FormBatch(ctx)

	// Two assertions, each catching a distinct failure mode:
	//
	// 1. wq must be empty: guards against KV-allocation failure leaving the
	//    request stuck in the queue (unrelated to the decode-only path bug).
	// 2. ComputedTokens must be 0: catches the regression. The normal path
	//    sets numNewTokens=0 for zero-input requests (ComputedTokens=0).
	//    The decode-only fast-path would set it to ProgressIndex+1 = 1.
	if wq.Len() != 0 {
		t.Errorf("WaitQueue.Len() = %d, want 0: zero-input request was not dequeued", wq.Len())
	}
	if computedTokens[req.ID] != 0 {
		t.Errorf("ComputedTokens[%q] = %d, want 0: zero-input request must not take the decode-only fast-path (IsDecodeSubRequest guard violated)",
			req.ID, computedTokens[req.ID])
	}
}

func TestPreemption_FCFS_EvictsTail(t *testing.T) {
	// 10 blocks × 16 tokens/block = 160 token capacity.
	// 3 running requests × 3 blocks each = 9 blocks used, 1 block free.
	// Phase 1: "first" gets its decode block (uses the 1 free block).
	// Phase 1: "second" needs decode but cache is now full → preemption.
	// FCFS evicts tail = "third" (BC-1).
	kvCache := MustNewKVCacheState(10, 16)
	running := []*Request{
		{ID: "first", SLOClass: "critical", ArrivalTime: 100, State: StateRunning,
			InputTokens: make([]int, 48), OutputTokens: make([]int, 10)},
		{ID: "second", SLOClass: "standard", ArrivalTime: 200, State: StateRunning,
			InputTokens: make([]int, 48), OutputTokens: make([]int, 10)},
		{ID: "third", SLOClass: "background", ArrivalTime: 300, State: StateRunning,
			InputTokens: make([]int, 48), OutputTokens: make([]int, 10)},
	}
	for _, req := range running {
		// AllocateKVBlocks uses ProgressIndex < len(InputTokens) for prefill path.
		// Allocate with ProgressIndex=0 (prefill), then set to 48 (decode).
		kvCache.AllocateKVBlocks(req, 0, 48, nil)
		req.ProgressIndex = 48
	}
	newReq := &Request{ID: "new", InputTokens: make([]int, 16), OutputTokens: make([]int, 1), State: StateQueued}
	wq := &WaitQueue{}
	wq.Enqueue(newReq)

	bf := NewBatchFormation("fcfs", nil)
	ctx := BatchContext{
		RunningBatch:       &Batch{Requests: running},
		WaitQ:              wq,
		KVCache:            kvCache,
		MaxScheduledTokens: 10000,
		MaxRunningReqs:     10,
		Now:                1000,
		ComputedTokens:     make(map[string]int64),
	}

	result := bf.FormBatch(ctx)

	// THEN the tail request ("third") is evicted (FCFS = tail-of-batch, BC-1)
	if len(result.Preempted) == 0 {
		t.Fatal("expected preemption but got none")
	}
	if result.Preempted[0].Request.ID != "third" {
		t.Errorf("FCFS preemption: evicted %q, want \"third\" (tail)", result.Preempted[0].Request.ID)
	}
}

// makeRunningRequest creates a running request with prefill fully allocated.
// Allocates inputLen tokens (ceil(inputLen/blockSize) blocks) then sets ProgressIndex.
// blockSize matches the kvCache blockSize (16 for MustNewKVCacheState(N, 16)).
func makeRunningRequest(id, sloClass string, arrival int64, inputLen int, kvCache KVStore) *Request {
	req := &Request{
		ID:           id,
		SLOClass:     sloClass,
		ArrivalTime:  arrival,
		State:        StateRunning,
		InputTokens:  make([]int, inputLen),
		OutputTokens: make([]int, 10),
	}
	kvCache.AllocateKVBlocks(req, 0, int64(inputLen), nil)
	req.ProgressIndex = int64(inputLen)
	return req
}

func TestPreemption_Priority_EvictsLeastUrgent(t *testing.T) {
	// Table-driven: victim position varies to prove selection is priority-based, not positional.
	// All scenarios: bg (background=-3) must be evicted before crit (critical=4) and std (standard=3).
	tests := []struct {
		name    string
		order   []string // order of requests in batch: [sloClass, ...]
		wantID  string
	}{
		{"victim at head", []string{"background", "critical", "standard"}, "bg"},
		{"victim in middle", []string{"critical", "background", "standard"}, "bg"},
		{"victim at tail", []string{"critical", "standard", "background"}, "bg"},
	}

	ids := map[string]string{
		"background": "bg",
		"critical":   "crit",
		"standard":   "std",
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 10 blocks × 16 tokens. 3 running requests × 3 blocks each = 9 used, 1 free.
			// Phase 1 decode for first request uses the 1 free block.
			// Phase 1 decode for second request triggers preemption.
			kvCache := MustNewKVCacheState(10, 16)
			var running []*Request
			for i, slo := range tt.order {
				req := makeRunningRequest(ids[slo], slo, int64(100*(i+1)), 48, kvCache)
				running = append(running, req)
			}

			wq := &WaitQueue{}
			wq.Enqueue(&Request{ID: "new", InputTokens: make([]int, 16), OutputTokens: make([]int, 1), State: StateQueued})

			bf := NewBatchFormation("priority", nil)
			ctx := BatchContext{
				RunningBatch:       &Batch{Requests: running},
				WaitQ:              wq,
				KVCache:            kvCache,
				MaxScheduledTokens: 10000,
				MaxRunningReqs:     10,
				Now:                1000,
				ComputedTokens:     make(map[string]int64),
			}

			result := bf.FormBatch(ctx)

			if len(result.Preempted) == 0 {
				t.Fatal("expected preemption but got none")
			}
			// Priority mode must evict bg regardless of its position in the batch.
			if result.Preempted[0].Request.ID != tt.wantID {
				t.Errorf("priority preemption: evicted %q, want %q (background = least urgent, SLO=-3)",
					result.Preempted[0].Request.ID, tt.wantID)
			}
		})
	}
}

func TestPreemption_Priority_SelfPreemption(t *testing.T) {
	// NC-3: When the request being processed (Phase 1) is itself the least urgent,
	// selectPriorityVictim selects it as victim. The preemptedRequest==req guard fires,
	// returning (false, adj). Phase 1 breaks. The request goes to WaitQ.
	// Setup: all 10 blocks used (bg=4, crit=3, std=3). Cache fully saturated.
	// Phase 1 processes bg first (index 0). bg needs a decode block.
	// Cache is full → preemption: selectPriorityVictim returns bg (background=-3, least urgent).
	// victimIdx=0 == reqIndex=0 → self-eviction fires. bg goes to WaitQ.
	kvCache := MustNewKVCacheState(10, 16)
	bg := makeRunningRequest("bg", "background", 100, 64, kvCache)   // 4 blocks
	crit := makeRunningRequest("crit", "critical", 200, 48, kvCache)  // 3 blocks
	dummy := makeRunningRequest("dummy", "standard", 300, 48, kvCache) // 3 blocks → total 10

	wq := &WaitQueue{}
	wq.Enqueue(&Request{ID: "new", InputTokens: make([]int, 16), OutputTokens: make([]int, 1), State: StateQueued})

	bf := NewBatchFormation("priority", nil)
	ctx := BatchContext{
		RunningBatch:       &Batch{Requests: []*Request{bg, crit, dummy}},
		WaitQ:              wq,
		KVCache:            kvCache,
		MaxScheduledTokens: 10000,
		MaxRunningReqs:     10,
		Now:                1000,
		ComputedTokens:     make(map[string]int64),
	}

	result := bf.FormBatch(ctx)

	// bg must be preempted (least urgent; also the first to trigger decode allocation)
	preemptedIDs := make(map[string]bool)
	for _, p := range result.Preempted {
		preemptedIDs[p.Request.ID] = true
	}
	if !preemptedIDs["bg"] {
		t.Errorf("expected bg to be preempted; got preempted: %v", preemptedIDs)
	}
	// bg must be in WaitQ
	found := false
	for _, r := range ctx.WaitQ.Items() {
		if r.ID == "bg" {
			found = true
			break
		}
	}
	if !found {
		t.Error("bg was preempted but not found in WaitQ")
	}
}

func TestPreemption_Priority_TiebreakByLatestArrival(t *testing.T) {
	// BC-3: Among running requests with equal SLO tier priority, the most
	// recently arrived (highest ArrivalTime) is evicted first.
	// Matches vLLM scheduler.py:829: key=lambda r: (r.priority, r.arrival_time)
	// Three standard requests (SLO=3), arrival times 100/200/300.
	// "new" (arrival=300) should be evicted before "mid" (200) and "old" (100).
	kvCache := MustNewKVCacheState(10, 16)
	old := makeRunningRequest("old", "standard", 100, 48, kvCache) // 3 blocks
	mid := makeRunningRequest("mid", "standard", 200, 48, kvCache) // 3 blocks
	new_ := makeRunningRequest("new", "standard", 300, 48, kvCache) // 3 blocks → 9 used, 1 free

	wq := &WaitQueue{}
	wq.Enqueue(&Request{ID: "trigger", InputTokens: make([]int, 16), OutputTokens: make([]int, 1), State: StateQueued})

	bf := NewBatchFormation("priority", nil)
	ctx := BatchContext{
		RunningBatch:       &Batch{Requests: []*Request{old, mid, new_}},
		WaitQ:              wq,
		KVCache:            kvCache,
		MaxScheduledTokens: 10000,
		MaxRunningReqs:     10,
		Now:                1000,
		ComputedTokens:     make(map[string]int64),
	}

	result := bf.FormBatch(ctx)

	if len(result.Preempted) == 0 {
		t.Fatal("expected preemption but got none")
	}
	// "new" has the latest arrival (300) — must be evicted first among equal-priority requests.
	if result.Preempted[0].Request.ID != "new" {
		t.Errorf("tiebreak: evicted %q, want \"new\" (latest arrival ArrivalTime=300)", result.Preempted[0].Request.ID)
	}
}

func TestPreemption_Priority_KVConservation(t *testing.T) {
	// BC-7: KV blocks must not exceed total capacity after priority preemption.
	// Setup: bg=6 blocks, crit=3 blocks → 9 used, 1 free.
	// Phase 1: bg decode uses the 1 free block (now 10/10 used).
	// Phase 1: crit decode → cache full → preemption: evict bg (background=-3, least urgent).
	// bg loses 6+1 blocks (its prefill + the just-allocated decode block), crit gains 1.
	// usedAfter < usedBefore confirms blocks were freed.
	kvCache := MustNewKVCacheState(10, 16)
	bg := makeRunningRequest("bg", "background", 100, 96, kvCache)   // 6 blocks
	crit := makeRunningRequest("crit", "critical", 200, 48, kvCache)  // 3 blocks → 9 used, 1 free

	usedBefore := kvCache.UsedBlocks()

	wq := &WaitQueue{}
	wq.Enqueue(&Request{ID: "new", InputTokens: make([]int, 48), OutputTokens: make([]int, 1), State: StateQueued})

	bf := NewBatchFormation("priority", nil)
	ctx := BatchContext{
		RunningBatch:       &Batch{Requests: []*Request{bg, crit}},
		WaitQ:              wq,
		KVCache:            kvCache,
		MaxScheduledTokens: 10000,
		MaxRunningReqs:     10,
		Now:                1000,
		ComputedTokens:     make(map[string]int64),
	}

	bf.FormBatch(ctx)

	usedAfter := kvCache.UsedBlocks()

	// INV-4: used blocks must not exceed total capacity at any point
	if usedAfter > kvCache.TotalCapacity() {
		t.Errorf("INV-4 violated: used=%d > capacity=%d", usedAfter, kvCache.TotalCapacity())
	}
	// bg was evicted: its 3 blocks should be freed
	if usedAfter >= usedBefore {
		t.Errorf("expected blocks freed after priority preemption: before=%d, after=%d",
			usedBefore, usedAfter)
	}
}

func TestPreemption_Priority_EmptyBatch_NoPanic(t *testing.T) {
	// NC-2: priority mode with empty running batch must not panic.
	// Circuit breaker at batch_formation.go returns (false, 0) immediately.
	kvCache := MustNewKVCacheState(2, 16)
	wq := &WaitQueue{}
	wq.Enqueue(&Request{ID: "large", InputTokens: make([]int, 200), OutputTokens: make([]int, 1), State: StateQueued})

	bf := NewBatchFormation("priority", nil)
	ctx := BatchContext{
		RunningBatch:       &Batch{Requests: []*Request{}},
		WaitQ:              wq,
		KVCache:            kvCache,
		MaxScheduledTokens: 10000,
		MaxRunningReqs:     10,
		Now:                0,
		ComputedTokens:     make(map[string]int64),
	}

	result := bf.FormBatch(ctx) // must not panic
	for _, r := range result.RunningBatch.Requests {
		if r.ID == "large" {
			t.Error("large request should not be in batch with only 2 blocks available")
		}
	}
}
