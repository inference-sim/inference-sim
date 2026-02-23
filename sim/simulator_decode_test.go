package sim

import (
	"math"
	"testing"
)

func TestSimulator_DecodePhase_RequestCompletesSuccessfully(t *testing.T) {
	// BC-9: A request with known input/output tokens completes through
	// the full prefill->decode pipeline via normal simulation
	sim := mustNewSimulator(t, SimConfig{
		Horizon: math.MaxInt64,
		Seed:    42,
		KVCacheConfig: KVCacheConfig{
			TotalKVBlocks:   100,
			BlockSizeTokens: 4,
		},
		BatchConfig: BatchConfig{
			MaxRunningReqs:     10,
			MaxScheduledTokens: 1000,
		},
		LatencyCoeffs: LatencyCoeffs{
			BetaCoeffs:  []float64{100, 0.5, 0.5},
			AlphaCoeffs: []float64{100, 0.1, 50},
		},
	})

	// Create a request with known input/output that exercises decode phase
	req := &Request{
		ID:           "decode_test",
		InputTokens:  []int{1, 2, 3, 4, 5, 6, 7, 8},
		OutputTokens: []int{100, 200, 300},
		ArrivalTime:  0,
		State:        StateQueued,
	}

	sim.InjectArrival(req)
	sim.Run()

	// THEN the request completes successfully (exercised both prefill and decode)
	if sim.Metrics.CompletedRequests != 1 {
		t.Fatalf("CompletedRequests = %d, want 1", sim.Metrics.CompletedRequests)
	}

	// THEN request transitions to completed state
	if req.State != StateCompleted {
		t.Errorf("request State = %q, want %q", req.State, StateCompleted)
	}

	// THEN E2E latency is recorded (proves decode phase ran)
	e2e, ok := sim.Metrics.RequestE2Es["decode_test"]
	if !ok {
		t.Fatal("RequestE2Es missing entry for decode_test")
	}
	if e2e <= 0 {
		t.Errorf("E2E = %f, want > 0", e2e)
	}

	// THEN KV cache total capacity is unchanged (conservation)
	if sim.KVCache.TotalCapacity() != 100 {
		t.Errorf("TotalCapacity = %d, want 100 (unchanged)", sim.KVCache.TotalCapacity())
	}
}
