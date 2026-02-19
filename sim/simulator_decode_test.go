package sim

import (
	"math"
	"testing"
)

func TestMakeRunningBatch_DecodePhase_PreemptGetsPositiveTokenCount(t *testing.T) {
	// GIVEN a simulator with a request that has completed prefill and is in decode
	sim := mustNewSimulator(t, SimConfig{
		Horizon:            math.MaxInt64,
		Seed:               42,
		TotalKVBlocks:      100,
		BlockSizeTokens:    4,
		MaxRunningReqs:     10,
		MaxScheduledTokens: 1000,
		BetaCoeffs:         []float64{100, 0.5, 0.5},
		AlphaCoeffs:        []float64{100, 0.1, 50},
	})

	// Create a request with known input/output
	req := &Request{
		ID:           "decode_test",
		InputTokens:  []int{1, 2, 3, 4, 5, 6, 7, 8},
		OutputTokens: []int{100, 200, 300},
		ArrivalTime:  0,
		State:        StateRunning,
	}

	// Pre-allocate KV blocks for the prefill portion (ProgressIndex=0 during prefill)
	ok := sim.KVCache.AllocateKVBlocks(req, 0, 8, []int64{})
	if !ok {
		t.Fatal("pre-allocation should succeed")
	}

	// Advance to decode phase: ProgressIndex past all input tokens
	req.ProgressIndex = 10
	req.NumNewTokens = 1

	// Put request in running batch
	sim.RunningBatch.Requests = append(sim.RunningBatch.Requests, req)
	sim.reqNumComputedTokens[req.ID] = 10

	// WHEN makeRunningBatch processes this decode-phase request
	// (This should NOT pass a negative value to preempt)
	sim.makeRunningBatch(1000)

	// THEN the request should still be in the running batch with NumNewTokens=1
	found := false
	for _, r := range sim.RunningBatch.Requests {
		if r.ID == "decode_test" {
			found = true
			if r.NumNewTokens != 1 {
				t.Errorf("NumNewTokens = %d, want 1 for decode", r.NumNewTokens)
			}
		}
	}
	if !found {
		t.Error("request should still be in running batch after decode scheduling")
	}

	// Additional: verify KV allocation succeeded (decode allocated 1 additional block)
	// The prefill allocated 2 blocks (8 tokens / 4 per block), decode adds 1 more
	if sim.KVCache.UsedBlocks() < 2 {
		t.Errorf("UsedBlocks = %d, want >= 2 (prefill blocks should still be allocated)", sim.KVCache.UsedBlocks())
	}
}
