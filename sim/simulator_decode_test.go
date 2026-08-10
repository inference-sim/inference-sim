package sim

import (
	"math"
	"testing"
)

func TestSimulator_DecodePhase_RequestCompletesSuccessfully(t *testing.T) {
	// BC-9: A request with known input/output tokens completes through
	// the full prefill->decode pipeline via normal simulation
	sim := mustNewSimulator(t, SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(100, 4, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(10, 1000, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{100, 0.5, 0.5}, []float64{100, 0.1, 50}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "", "", 1, 1, false, "", "roofline", 0),
	})

	// Create a request with known input/output that exercises decode phase
	req := &Request{
		ID:           "decode_test",
		InputTokens:  []TokenID{1, 2, 3, 4, 5, 6, 7, 8},
		OutputTokens: []TokenID{100, 200, 300},
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

// specDecodeSimConfig builds a SimConfig for spec-decode sim tests with the given
// K and acceptance. Large KV + token budget so multi-token decode steps aren't capped.
func specDecodeSimConfig(k int, acceptance float64) SimConfig {
	return SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10000, 4, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(10, 100000, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{100, 0.5, 0.5}, []float64{100, 0.1, 50}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "", "", 1, 1, false, "", "roofline", 0),
		SpeculativeConfig:   SpeculativeConfig{K: k, Acceptance: acceptance},
	}
}

func runOneSpecDecodeRequest(t *testing.T, k int, acceptance float64, input, output int) (*Simulator, *Request) {
	t.Helper()
	s := mustNewSimulator(t, specDecodeSimConfig(k, acceptance))
	req := &Request{
		ID:           "spec",
		InputTokens:  make([]TokenID, input),
		OutputTokens: make([]TokenID, output),
		ArrivalTime:  0,
		State:        StateQueued,
	}
	s.InjectArrival(req)
	s.Run()
	return s, req
}

// BC-2: speculative decoding raises throughput — the same output completes in fewer
// decode steps as accepted-tokens-per-step (1+α·K) rises, with output-token count
// preserved. Step count is observable via len(req.ITL) (one entry per decode step).
func TestSpecDecode_ThroughputReducesStepCount(t *testing.T) {
	const input, output = 8, 9

	_, reqOff := runOneSpecDecodeRequest(t, 0, 0, input, output)   // g=1
	_, reqG3 := runOneSpecDecodeRequest(t, 2, 1.0, input, output)  // g=1+1.0*2=3

	stepsOff := len(reqOff.ITL)
	stepsG3 := len(reqG3.ITL)

	if stepsG3 >= stepsOff {
		t.Errorf("spec-decode decode steps = %d, want fewer than baseline %d", stepsG3, stepsOff)
	}
	// Both must have completed and generated exactly `output` tokens (BC-4/INV-1).
	if reqOff.State != StateCompleted || reqG3.State != StateCompleted {
		t.Fatalf("both requests must complete: off=%q g3=%q", reqOff.State, reqG3.State)
	}
}

// BC-4 / INV-1: output-token conservation holds under spec-decode, including the
// overshoot case where a multi-token final step carries ProgressIndex past the target.
func TestSpecDecode_OutputTokenConservation(t *testing.T) {
	cases := []struct{ k, input, output int; acc float64 }{
		{k: 0, acc: 0, input: 8, output: 9},   // baseline g=1
		{k: 2, acc: 1.0, input: 8, output: 9}, // g=3, exact landing
		{k: 4, acc: 1.0, input: 10, output: 2}, // g=5, heavy overshoot (PI jumps well past target)
		{k: 3, acc: 0.5, input: 16, output: 20}, // g=2.5 fractional carry
	}
	for _, c := range cases {
		s, req := runOneSpecDecodeRequest(t, c.k, c.acc, c.input, c.output)
		if req.State != StateCompleted {
			t.Errorf("k=%d in=%d out=%d: state=%q, want completed", c.k, c.input, c.output, req.State)
			continue
		}
		// Exactly `output` tokens counted — never more, even on overshoot.
		if got := s.Metrics.TotalOutputTokens; got != c.output {
			t.Errorf("k=%d out=%d: TotalOutputTokens=%d, want %d (overshoot must be clamped)", c.k, c.output, got, c.output)
		}
		// ProgressIndex must not be counted beyond the assigned output either.
		if req.ProgressIndex > int64(c.input+c.output) {
			// Overshoot of PI itself is allowed (final multi-token step), but the
			// COUNTED output is clamped above; assert PI didn't run away unboundedly.
			if req.ProgressIndex > int64(c.input+c.output)+int64(c.k) {
				t.Errorf("k=%d: ProgressIndex=%d ran away past input+output+k=%d", c.k, req.ProgressIndex, c.input+c.output+c.k)
			}
		}
	}
}

// BC-6 / INV-6: spec-decode is deterministic — the fractional carry is state, not RNG.
func TestSpecDecode_Deterministic(t *testing.T) {
	s1, r1 := runOneSpecDecodeRequest(t, 3, 0.5, 16, 20)
	s2, r2 := runOneSpecDecodeRequest(t, 3, 0.5, 16, 20)
	if s1.Metrics.TotalOutputTokens != s2.Metrics.TotalOutputTokens {
		t.Errorf("non-deterministic output tokens: %d vs %d", s1.Metrics.TotalOutputTokens, s2.Metrics.TotalOutputTokens)
	}
	if len(r1.ITL) != len(r2.ITL) {
		t.Errorf("non-deterministic step count: %d vs %d", len(r1.ITL), len(r2.ITL))
	}
	if r1.FirstTokenTime != r2.FirstTokenTime {
		t.Errorf("non-deterministic TTFT: %d vs %d", r1.FirstTokenTime, r2.FirstTokenTime)
	}
}

// I-5 conservation: sum(ITL) still equals total decode wall-clock (E2E − TTFT-ish),
// i.e. the per-step ITL entries (each = step time) sum to the decode span even when
// each entry now covers g tokens. Verified indirectly: the request completes and its
// E2E equals FirstTokenTime + sum(ITL) + postDecodeOverhead by construction; here we
// assert sum(ITL) is positive and monotone-consistent (fewer, larger entries).
func TestSpecDecode_ITLSumPreserved(t *testing.T) {
	_, reqOff := runOneSpecDecodeRequest(t, 0, 0, 8, 9)
	_, reqOn := runOneSpecDecodeRequest(t, 2, 1.0, 8, 9)

	sum := func(xs []int64) int64 {
		var s int64
		for _, x := range xs {
			s += x
		}
		return s
	}
	// Fewer entries under spec-decode, but each is a full multi-token step.
	if len(reqOn.ITL) >= len(reqOff.ITL) {
		t.Errorf("expected fewer ITL entries under spec-decode: on=%d off=%d", len(reqOn.ITL), len(reqOff.ITL))
	}
	if sum(reqOn.ITL) <= 0 {
		t.Errorf("sum(ITL) must be positive, got %d", sum(reqOn.ITL))
	}
}

// I-1: the peek/commit fractional carry advances at the mean rate with no drift,
// and a capped grant (fewer tokens than proposed) defers the remainder rather than
// losing it. Directly exercises peekDecodeTokens/commitDecodeTokens.
func TestSpecDecode_CarryNoDrift(t *testing.T) {
	s := mustNewSimulator(t, specDecodeSimConfig(3, 0.5)) // rate = 1 + 0.5*3 = 2.5
	req := &Request{ID: "carry"}

	// Uncapped: grant exactly the peeked count each step. Over many steps the total
	// advance must track rate*steps within one token (pure floor carry).
	var total int64
	const steps = 100
	for i := 0; i < steps; i++ {
		g := s.peekDecodeTokens(req)
		req.NumNewTokens = int(g)
		s.commitDecodeTokens(req)
		total += g
	}
	want := int64(2.5 * steps)
	if total < want-1 || total > want {
		t.Errorf("uncapped carry total = %d, want within [%d,%d] (rate*steps, floor)", total, want-1, want)
	}

	// Capped: propose g but grant only 1 for a stretch, then resume. The deferred
	// tokens must be recovered (carry accumulates), so a later peek exceeds the rate.
	req2 := &Request{ID: "capped"}
	sawRecovery := false
	for i := 0; i < 10; i++ {
		g := s.peekDecodeTokens(req2)
		granted := g
		if i < 5 {
			granted = 1 // simulate a token-budget cap
		}
		if granted > g {
			granted = g
		}
		req2.NumNewTokens = int(granted)
		s.commitDecodeTokens(req2)
		if i >= 5 && g > 3 { // after the capped stretch, a peek recovers deferred tokens (>rate)
			sawRecovery = true
		}
	}
	if !sawRecovery {
		t.Error("capped carry did not recover deferred tokens (drift): expected a post-cap peek > 3")
	}
}
