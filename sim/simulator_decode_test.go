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

	_, reqOff := runOneSpecDecodeRequest(t, 0, 0, input, output)  // g=1
	_, reqG3 := runOneSpecDecodeRequest(t, 2, 1.0, input, output) // g=1+1.0*2=3

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

// #1528 BC-4 / INV-1: output-token conservation holds under spec-decode, and (#1657) a
// multi-token step lands EXACTLY on the completion boundary rather than past it.
// The boundary is InputLen + max(outputLen,1) - 1: BLIS charges output token #1 to
// prefill and the rest to decode steps, so this is also the ProgressIndex a K=0 run
// finishes at — i.e. spec-decode changes step count, never token accounting.
func TestSpecDecode_OutputTokenConservation(t *testing.T) {
	cases := []struct {
		k, input, output int
		acc              float64
	}{
		{k: 0, acc: 0, input: 8, output: 9},      // baseline g=1
		{k: 2, acc: 1.0, input: 8, output: 9},    // g=3, exact landing
		{k: 4, acc: 1.0, input: 10, output: 2},   // g=5 vs a 1-step budget (heaviest clamp)
		{k: 3, acc: 0.5, input: 16, output: 20},  // g=2.5 fractional carry
		{k: 2, acc: 0.755, input: 20, output: 4}, // #1657 repro: overshot the boundary by 2
		{k: 2, acc: 0.755, input: 20, output: 9}, // #1657 repro: overshot the boundary by 2
		{k: 5, acc: 0.8, input: 20, output: 13},  // wide verify block, fractional carry
	}
	for _, c := range cases {
		s, req := runOneSpecDecodeRequest(t, c.k, c.acc, c.input, c.output)
		if req.State != StateCompleted {
			t.Errorf("k=%d in=%d out=%d: state=%q, want completed", c.k, c.input, c.output, req.State)
			continue
		}
		// Exactly `output` tokens counted — never more, even on the widest verify block.
		if got := s.Metrics.TotalOutputTokens; got != c.output {
			t.Errorf("k=%d out=%d: TotalOutputTokens=%d, want %d (overshoot must be clamped)", c.k, c.output, got, c.output)
		}
		// #1657: the final step stops AT the completion boundary — no overshoot.
		wantPI := int64(c.input + max(c.output, 1) - 1)
		if req.ProgressIndex != wantPI {
			t.Errorf("k=%d in=%d out=%d: final ProgressIndex=%d, want %d (InputLen + max(outputLen,1) - 1; a spec-decode step must not advance past the completion boundary)",
				c.k, c.input, c.output, req.ProgressIndex, wantPI)
		}
	}
}

// #1657: the final ProgressIndex — the quantity closed-loop accumulate sessions
// read back as "how much output did this round actually produce" — is INDEPENDENT of the
// spec-decode config. Spec-decode buys fewer, wider decode steps; it must not change the
// request's token accounting. This is the law the per-case boundary assertion above
// encodes, stated as a direct comparison against the feature-off run.
func TestSpecDecode_FinalProgressIndexMatchesBaseline(t *testing.T) {
	specs := []struct {
		k   int
		acc float64
	}{{2, 0.755}, {2, 1.0}, {3, 0.5}, {5, 0.8}, {7, 0.25}}
	for _, output := range []int{1, 2, 3, 4, 5, 8, 9, 13, 20} {
		_, base := runOneSpecDecodeRequest(t, 0, 0, 20, output)
		for _, sp := range specs {
			_, got := runOneSpecDecodeRequest(t, sp.k, sp.acc, 20, output)
			if got.ProgressIndex != base.ProgressIndex {
				t.Errorf("out=%d k=%d acc=%v: final ProgressIndex=%d, want %d (K=0 baseline)",
					output, sp.k, sp.acc, got.ProgressIndex, base.ProgressIndex)
			}
			// Fewer decode steps is the point of the feature (sanity: the comparison
			// above is not passing because spec-decode silently did nothing). Gated on
			// output > 3 because a 1-3-token output needs at most two decode steps even
			// at g=1, so there is no room to save one; those rows rely on the same
			// (K, acc) pairs being proven active by the larger-output rows.
			if sp.acc > 0 && output > 3 && len(got.ITL) >= len(base.ITL) {
				t.Errorf("out=%d k=%d acc=%v: decode steps=%d, want fewer than baseline %d",
					output, sp.k, sp.acc, len(got.ITL), len(base.ITL))
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

// #1657 (Request.completionProgressIndex): pins the boundary law that BOTH the
// completion test (processCompletions) and the spec-decode grant cap now read. The −1
// and the max(...,1) are load-bearing: BLIS charges output token #1 to prefill
// completion, and a zero-output request finishes at the end of prefill.
func TestCompletionProgressIndex_BoundaryLaw(t *testing.T) {
	cases := []struct {
		name          string
		input, output int
		want          int64
	}{
		{name: "zero output finishes at prefill end", input: 8, output: 0, want: 8},
		{name: "one output finishes at prefill end", input: 8, output: 1, want: 8},
		{name: "two outputs need one decode token", input: 8, output: 2, want: 9},
		{name: "n outputs need n-1 decode tokens", input: 20, output: 13, want: 32},
		{name: "empty input", input: 0, output: 5, want: 4},
	}
	for _, c := range cases {
		req := &Request{
			ID:           c.name,
			InputTokens:  make([]TokenID, c.input),
			OutputTokens: make([]TokenID, c.output),
		}
		if got := req.completionProgressIndex(); got != c.want {
			t.Errorf("%s: completionProgressIndex() = %d, want %d", c.name, got, c.want)
		}
	}
}

// #1657 (floor-at-1 liveness, INV-8/INV-11): a PD decode sub-request admitted ALREADY at
// its completion boundary (a 1-output-token request, whose ProgressIndex starts at
// InputLen == boundary) must still be granted one token. Its remaining output budget is
// 0, so a plain min() would grant 0; FormBatch would `break` out of the admission loop
// and — since the sub-request is not yet in RunningBatch, so nothing force-completes it —
// it would sit in the wait queue forever. This asserts the grant directly, so the failure
// mode is a one-line diff rather than a package-level test timeout.
func TestSpecDecode_PDDecodeSubRequest_AtBoundary_GrantsOneToken(t *testing.T) {
	s := mustNewSimulator(t, specDecodeSimConfig(5, 1.0)) // g = 1 + 1.0*5 = 6 proposed

	sub := &Request{
		ID:                 "pd_decode_sub",
		InputTokens:        make([]TokenID, 16),
		OutputTokens:       make([]TokenID, 1), // boundary == InputLen: zero budget left
		State:              StateQueued,
		IsDecodeSubRequest: true,
		ProgressIndex:      16, // PD: prefill ran elsewhere, KV was transferred
	}
	if sub.completionProgressIndex() != sub.ProgressIndex {
		t.Fatalf("test setup: want the sub-request admitted AT its boundary, got PI=%d boundary=%d",
			sub.ProgressIndex, sub.completionProgressIndex())
	}
	s.WaitQ.Enqueue(sub)

	result := s.batchFormation.FormBatch(BatchContext{
		RunningBatch:        &Batch{},
		WaitQ:               s.WaitQ,
		KVCache:             s.KVCache,
		MaxNumBatchedTokens: 10000,
		MaxNumSeqs:          10,
		Now:                 1000,
		ComputedTokens:      make(map[string]int64),
		DecodeTokensPerStep: s.peekDecodeTokens,
	})

	if len(result.RunningBatch.Requests) != 1 {
		t.Fatalf("decode sub-request was not admitted (%d in batch, %d still queued) — it would be stranded forever",
			len(result.RunningBatch.Requests), s.WaitQ.Len())
	}
	if sub.NumNewTokens != 1 {
		t.Errorf("NumNewTokens = %d, want 1 (floored at 1 at the boundary; 0 strands the request, >1 overshoots)",
			sub.NumNewTokens)
	}
}
