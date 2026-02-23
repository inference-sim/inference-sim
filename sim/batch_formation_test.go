package sim

import "testing"

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
