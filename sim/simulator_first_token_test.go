package sim

import "testing"

// OnFirstToken fires from the step loop at each first token, at that token's
// time, and again when a preemption re-prefills a request.
func TestSimulator_OnFirstTokenFiresPerPrefill(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(4, 16, 0, 0, 0, 0), // forces B's preemption, as in TestSimulator_TTFT_UpdatedAfterPreemption
		BatchConfig:         NewBatchConfig(256, 10_000, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{0, 1, 0}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "test", "H100", 1, 1, false, "", "roofline", 0),
	}
	s := mustNewSimulator(t, cfg)
	calls := map[string][]int64{}
	s.OnFirstToken = func(req *Request, tick int64) { calls[req.ID] = append(calls[req.ID], tick) }
	for _, r := range []*Request{
		{ID: "A", InputTokens: GenerateRandomTokenIDs(s.WorkloadRNG(), 16), OutputTokens: make([]TokenID, 5), State: StateQueued},
		{ID: "B", InputTokens: GenerateRandomTokenIDs(s.WorkloadRNG(), 32), OutputTokens: make([]TokenID, 5), State: StateQueued},
	} {
		s.InjectArrival(r)
	}
	for s.HasPendingEvents() {
		s.ProcessNextEvent()
	}
	if s.Metrics.PreemptionCount == 0 {
		t.Fatal("precondition: expected a preemption")
	}
	if got := calls["A"]; len(got) != 1 || got[0] != int64(s.Metrics.RequestTTFTs["A"]) {
		t.Fatalf("A: OnFirstToken calls %v, want one at its TTFT %v (arrival 0)", got, s.Metrics.RequestTTFTs["A"])
	}
	if got := calls["B"]; len(got) < 2 {
		t.Fatalf("B: OnFirstToken calls %v, want a repeat after its re-prefill", got)
	}
}
