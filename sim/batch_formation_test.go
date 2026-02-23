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
		TotalKVBlocks:      100,
		BlockSizeTokens:    16,
		MaxRunningReqs:     10,
		MaxScheduledTokens: 10000,
		BetaCoeffs:         []float64{100, 1, 1},
		AlphaCoeffs:        []float64{100, 1, 100},
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
		TotalKVBlocks:      100,
		BlockSizeTokens:    16,
		MaxRunningReqs:     10,
		MaxScheduledTokens: 50, // tight token budget
		BetaCoeffs:         []float64{100, 1, 1},
		AlphaCoeffs:        []float64{100, 1, 100},
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
		TotalKVBlocks:      200,
		BlockSizeTokens:    16,
		MaxRunningReqs:     2, // tight batch size limit
		MaxScheduledTokens: 10000,
		BetaCoeffs:         []float64{100, 1, 1},
		AlphaCoeffs:        []float64{100, 1, 100},
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
