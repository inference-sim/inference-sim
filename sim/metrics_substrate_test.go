// sim/metrics_substrate_test.go
//
// Metrics Substrate Verification Suite (H-M1 through H-M10)
//
// All tests in this file verify BEHAVIORAL CONTRACTS — observable relationships
// between metric outputs that must hold regardless of implementation details.
// Each test passes the refactor survival test: "Would this test still pass if
// the implementation were completely rewritten but the behavior preserved?"
//
// Behavioral contracts verified:
//   BC-MS-1: E2E = TTFT + total decode latency (E2E identity)
//   BC-MS-2: Mean ITL × (output_tokens - 1) = E2E - TTFT (mean ITL consistency)
//   BC-MS-3: TTFT recorded exactly once per request (chunked or not)
//   BC-MS-4: sum(AllITLs) = sum(E2E - TTFT) across completed requests
//   BC-MS-5: Scheduling delay monotonically increases with input length
//   BC-MS-6: Scheduling delay for an isolated request equals alpha overhead
//   BC-MS-7: CacheHitRate ∈ [0, 1] (range invariant)
//   BC-MS-8: Unit conversion is consistent: ticks/1000 = ticks/1e3
//   BC-MS-9: Zero-output requests have TTFT > 0 and E2E >= TTFT
//   BC-MS-10: PeakKVBlocksUsed ∈ (0, TotalKVBlocks] and 0 after completion
//   BC-MS-11: TTFT monotonically increases with input length (isolated requests)
//   BC-MS-12: Percentiles are monotonically ordered (p50 ≤ p90 ≤ p95 ≤ p99)
//   BC-MS-13: Chunked prefill preserves request conservation
//   BC-MS-14: E2E ≥ TTFT for every completed request (causality)
//
// Test coefficients:
//   alpha = [1000, 2, 500]  →  QueueingTime = 1000 + 2*inputLen
//   beta  = [5000, 10, 3]   →  StepTime = 5000 + 10*cacheMiss + 3*decode
//   Block size = 16 tokens.
package sim

import (
	"fmt"
	"math"
	"sort"
	"testing"
)

// msAlpha returns simple alpha coefficients for metrics substrate tests.
func msAlpha() []float64 { return []float64{1000, 2, 500} }

// msBeta returns simple beta coefficients for metrics substrate tests.
func msBeta() []float64 { return []float64{5000, 10, 3} }

// msConfig returns a SimConfig for metrics substrate tests.
// No workload — caller injects requests via InjectArrival.
func msConfig(horizon int64) SimConfig {
	return SimConfig{
		Horizon:             horizon,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 100000, 0),
		LatencyCoeffs:       NewLatencyCoeffs(msBeta(), msAlpha()),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-model", "test-gpu", 1, false),
		PolicyConfig:        NewPolicyConfig("constant", "fcfs"),
		WorkloadConfig:      NewWorkloadConfig(nil, ""),
	}
}

// msMakeTokens creates a token slice of the given length (values 1..n).
// Named with ms prefix to avoid collision with routing_prefix_scorer_test.go.
func msMakeTokens(n int) []int {
	tokens := make([]int, n)
	for i := range tokens {
		tokens[i] = i + 1
	}
	return tokens
}

// msInjectAndRun is a test helper that injects a single request and runs the simulator.
func msInjectAndRun(t *testing.T, cfg SimConfig, id string, inputLen, outputLen int, arrival int64) *Simulator {
	t.Helper()
	s := mustNewSimulator(t, cfg)
	req := &Request{
		ID:           id,
		InputTokens:  msMakeTokens(inputLen),
		OutputTokens: msMakeTokens(outputLen),
		ArrivalTime:  arrival,
		State:        StateQueued,
	}
	s.InjectArrival(req)
	s.Run()
	return s
}

// ═══════════════════════════════════════════════════════════════════════════════
// BC-MS-1 + BC-MS-2: E2E Identity and Mean ITL Consistency
//
// For every completed request:
//   E2E = TTFT + total decode latency       (BC-MS-1)
//   mean_ITL × max(output-1, 1) = E2E - TTFT  (BC-MS-2)
// ═══════════════════════════════════════════════════════════════════════════════

func TestMetrics_E2E_Identity_SingleRequest(t *testing.T) {
	s := msInjectAndRun(t, msConfig(math.MaxInt64), "e2e-1", 32, 5, 0)

	if s.Metrics.CompletedRequests != 1 {
		t.Fatalf("expected 1 completed, got %d", s.Metrics.CompletedRequests)
	}

	ttft := s.Metrics.RequestTTFTs["e2e-1"]
	e2e := s.Metrics.RequestE2Es["e2e-1"]

	// BC-MS-1: Reconstruct E2E from TTFT + sum(AllITLs)
	var itlSum int64
	for _, v := range s.Metrics.AllITLs {
		itlSum += v
	}
	reconstructed := ttft + float64(itlSum)
	if math.Abs(e2e-reconstructed) > 1.0 {
		t.Errorf("BC-MS-1 violated: E2E (%.1f) != TTFT + sum(ITL) (%.1f), diff=%.1f",
			e2e, reconstructed, e2e-reconstructed)
	}

	// BC-MS-2: Mean ITL consistency
	meanITL := s.Metrics.RequestITLs["e2e-1"]
	denom := float64(max(5-1, 1))
	totalFromMean := meanITL * denom
	totalFromE2E := e2e - ttft
	if math.Abs(totalFromMean-totalFromE2E) > 1.0 {
		t.Errorf("BC-MS-2 violated: meanITL*denom (%.1f) != E2E-TTFT (%.1f)",
			totalFromMean, totalFromE2E)
	}
}

func TestMetrics_E2E_Identity_MultipleRequests(t *testing.T) {
	cfg := msConfig(math.MaxInt64)
	s := mustNewSimulator(t, cfg)

	// Inject requests with varying sizes, spread out to avoid co-batching
	sizes := [][2]int{{16, 3}, {32, 5}, {48, 7}, {64, 2}, {80, 10}}
	for i, sz := range sizes {
		r := &Request{
			ID:           fmt.Sprintf("multi-%d", i),
			InputTokens:  msMakeTokens(sz[0]),
			OutputTokens: msMakeTokens(sz[1]),
			ArrivalTime:  int64(i) * 500000,
			State:        StateQueued,
		}
		s.InjectArrival(r)
	}
	s.Run()

	if s.Metrics.CompletedRequests != len(sizes) {
		t.Fatalf("expected %d completed, got %d", len(sizes), s.Metrics.CompletedRequests)
	}

	// BC-MS-2 for every request
	for i, sz := range sizes {
		id := fmt.Sprintf("multi-%d", i)
		e2e := s.Metrics.RequestE2Es[id]
		ttft := s.Metrics.RequestTTFTs[id]
		meanITL := s.Metrics.RequestITLs[id]
		outputTokens := sz[1]
		denom := float64(max(outputTokens-1, 1))

		totalFromMean := meanITL * denom
		totalFromE2E := e2e - ttft
		if math.Abs(totalFromMean-totalFromE2E) > 1.0 {
			t.Errorf("BC-MS-2 violated for %s: meanITL*denom (%.1f) != E2E-TTFT (%.1f)",
				id, totalFromMean, totalFromE2E)
		}

		// BC-MS-14: Causality
		if e2e < ttft {
			t.Errorf("BC-MS-14 violated for %s: E2E (%.1f) < TTFT (%.1f)", id, e2e, ttft)
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// BC-MS-3 + BC-MS-13: Chunked Prefill TTFT and Conservation
//
// Chunked prefill must not change the number of completed requests or
// cause TTFT to be recorded more/fewer than once.
// ═══════════════════════════════════════════════════════════════════════════════

func TestMetrics_ChunkedPrefill_TTFTRecordedOnce(t *testing.T) {
	cfg := msConfig(math.MaxInt64)
	cfg.LongPrefillTokenThreshold = 16 // 4 chunks for 64-token input
	s := msInjectAndRun(t, cfg, "chunk-1", 64, 3, 0)

	if s.Metrics.CompletedRequests != 1 {
		t.Fatalf("expected 1 completed, got %d", s.Metrics.CompletedRequests)
	}

	// BC-MS-3: TTFT exists, positive, and TTFTSum consistent
	ttft, ok := s.Metrics.RequestTTFTs["chunk-1"]
	if !ok {
		t.Fatal("BC-MS-3 violated: TTFT not recorded for chunked prefill request")
	}
	if ttft <= 0 {
		t.Errorf("BC-MS-3 violated: TTFT non-positive (%.1f)", ttft)
	}
	if float64(s.Metrics.TTFTSum) != ttft {
		t.Errorf("BC-MS-3 violated: TTFTSum (%d) != TTFT (%.1f) — TTFT recorded multiple times",
			s.Metrics.TTFTSum, ttft)
	}

	// BC-MS-14: Causality still holds
	e2e := s.Metrics.RequestE2Es["chunk-1"]
	if e2e < ttft {
		t.Errorf("BC-MS-14 violated: E2E (%.1f) < TTFT (%.1f)", e2e, ttft)
	}
}

func TestMetrics_ChunkedPrefill_PreservesConservation(t *testing.T) {
	// BC-MS-13: Chunked vs non-chunked produce same completed count
	for _, threshold := range []int64{0, 16, 32} {
		t.Run(fmt.Sprintf("threshold=%d", threshold), func(t *testing.T) {
			cfg := msConfig(math.MaxInt64)
			cfg.LongPrefillTokenThreshold = threshold
			s := mustNewSimulator(t, cfg)
			for i := 0; i < 5; i++ {
				r := &Request{
					ID:           fmt.Sprintf("cons-%d", i),
					InputTokens:  msMakeTokens(64),
					OutputTokens: msMakeTokens(3),
					ArrivalTime:  int64(i) * 200000,
					State:        StateQueued,
				}
				s.InjectArrival(r)
			}
			s.Run()

			// INV-1: All requests complete with infinite horizon
			if s.Metrics.CompletedRequests != 5 {
				t.Errorf("BC-MS-13: threshold=%d: completed=%d, expected=5",
					threshold, s.Metrics.CompletedRequests)
			}
		})
	}
}

func TestMetrics_ChunkedPrefill_TTFT_HigherThanNonChunked(t *testing.T) {
	// Behavioral: chunked prefill incurs overhead (more steps, each with beta0).
	// TTFT(chunked) >= TTFT(non-chunked) for the same request.
	sNC := msInjectAndRun(t, msConfig(math.MaxInt64), "nc", 64, 3, 0)

	cfgC := msConfig(math.MaxInt64)
	cfgC.LongPrefillTokenThreshold = 16
	sC := msInjectAndRun(t, cfgC, "c", 64, 3, 0)

	ttftNC := sNC.Metrics.RequestTTFTs["nc"]
	ttftC := sC.Metrics.RequestTTFTs["c"]

	if ttftNC <= 0 || ttftC <= 0 {
		t.Errorf("TTFT not positive: non-chunked=%.1f, chunked=%.1f", ttftNC, ttftC)
	}
	// Chunked should be >= non-chunked (3 extra beta0 costs for 4 chunks vs 1 step)
	if ttftC < ttftNC {
		t.Errorf("Chunked TTFT (%.1f) < non-chunked TTFT (%.1f) — expected >=", ttftC, ttftNC)
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// BC-MS-4: AllITLs Aggregate Consistency
//
// sum(AllITLs) = sum(E2E - TTFT) for all completed requests.
// This invariant holds regardless of how many ITL entries per request exist.
// ═══════════════════════════════════════════════════════════════════════════════

func TestMetrics_AllITLs_Sum_Equals_AggregateDecodeLatency(t *testing.T) {
	cfg := msConfig(math.MaxInt64)
	s := mustNewSimulator(t, cfg)

	for i := 0; i < 10; i++ {
		r := &Request{
			ID:           fmt.Sprintf("itl-%d", i),
			InputTokens:  msMakeTokens(16 + 16*i),
			OutputTokens: msMakeTokens(2 + i),
			ArrivalTime:  int64(i) * 100000,
			State:        StateQueued,
		}
		s.InjectArrival(r)
	}
	s.Run()

	if s.Metrics.CompletedRequests != 10 {
		t.Fatalf("expected 10 completed, got %d", s.Metrics.CompletedRequests)
	}

	var sumAllITLs float64
	for _, v := range s.Metrics.AllITLs {
		sumAllITLs += float64(v)
	}

	var sumE2EMinusTTFT float64
	for id, e2e := range s.Metrics.RequestE2Es {
		ttft, ok := s.Metrics.RequestTTFTs[id]
		if !ok {
			t.Errorf("request %s has E2E but no TTFT", id)
			continue
		}
		sumE2EMinusTTFT += e2e - ttft
	}

	if math.Abs(sumAllITLs-sumE2EMinusTTFT) > 10.0 {
		t.Errorf("BC-MS-4 violated: sum(AllITLs)=%.1f != sum(E2E-TTFT)=%.1f, diff=%.1f",
			sumAllITLs, sumE2EMinusTTFT, sumAllITLs-sumE2EMinusTTFT)
	}
}

func TestMetrics_AllITLs_Sum_WithZeroOutputRequests(t *testing.T) {
	// BC-MS-4 must hold even with zero-output requests in the mix.
	// This is the regression test for the phantom ITL fix.
	// Golden dataset unaffected: no zero-output requests in testdata/goldendataset.json (R12).
	cfg := msConfig(math.MaxInt64)
	s := mustNewSimulator(t, cfg)

	// Mix of zero-output and normal requests
	s.InjectArrival(&Request{
		ID: "z0", InputTokens: msMakeTokens(32), OutputTokens: []int{},
		ArrivalTime: 0, State: StateQueued,
	})
	s.InjectArrival(&Request{
		ID: "n1", InputTokens: msMakeTokens(32), OutputTokens: msMakeTokens(5),
		ArrivalTime: 100000, State: StateQueued,
	})
	s.InjectArrival(&Request{
		ID: "z2", InputTokens: msMakeTokens(48), OutputTokens: nil, // nil, not empty
		ArrivalTime: 200000, State: StateQueued,
	})
	s.InjectArrival(&Request{
		ID: "n3", InputTokens: msMakeTokens(16), OutputTokens: msMakeTokens(3),
		ArrivalTime: 300000, State: StateQueued,
	})
	s.Run()

	if s.Metrics.CompletedRequests != 4 {
		t.Fatalf("expected 4 completed, got %d", s.Metrics.CompletedRequests)
	}

	var sumAllITLs float64
	for _, v := range s.Metrics.AllITLs {
		sumAllITLs += float64(v)
	}
	var sumE2EMinusTTFT float64
	for id, e2e := range s.Metrics.RequestE2Es {
		ttft := s.Metrics.RequestTTFTs[id]
		sumE2EMinusTTFT += e2e - ttft
	}
	if math.Abs(sumAllITLs-sumE2EMinusTTFT) > 10.0 {
		t.Errorf("BC-MS-4 violated with zero-output mix: sum(AllITLs)=%.1f != sum(E2E-TTFT)=%.1f",
			sumAllITLs, sumE2EMinusTTFT)
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// BC-MS-5 + BC-MS-6: Scheduling Delay Properties
//
// For an isolated request (no queueing contention):
//   Scheduling delay = alpha overhead (QueueingTime)     (BC-MS-6)
// Scheduling delay monotonically increases with input length (BC-MS-5)
// ═══════════════════════════════════════════════════════════════════════════════

func TestMetrics_SchedulingDelay_EqualsAlpha_Isolated(t *testing.T) {
	// Regression anchor (one per method, per project convention): for an isolated
	// request with blackbox latency model and no queueing contention, the scheduling
	// delay equals QueueingTime = alpha0 + alpha1 * inputLen. This exact-value check
	// is intentionally model-specific — it would need updating if the latency model
	// changes, but it catches formula regressions that behavioral tests miss.
	inputLen := 32
	s := msInjectAndRun(t, msConfig(math.MaxInt64), "sd", inputLen, 3, 0)

	schedDelay := s.Metrics.RequestSchedulingDelays["sd"]
	expectedAlpha := int64(1000 + 2*inputLen) // alpha0 + alpha1 * inputLen

	if schedDelay != expectedAlpha {
		t.Errorf("BC-MS-6 regression: scheduling delay (%d) != alpha overhead (%d)",
			schedDelay, expectedAlpha)
	}
}

func TestMetrics_SchedulingDelay_Monotonic_WithInputLength(t *testing.T) {
	inputLens := []int{16, 32, 64, 128, 256}
	var delays []int64

	for _, inputLen := range inputLens {
		s := msInjectAndRun(t, msConfig(math.MaxInt64), "sd-mono", inputLen, 2, 0)
		delays = append(delays, s.Metrics.RequestSchedulingDelays["sd-mono"])
	}

	for i := 1; i < len(delays); i++ {
		if delays[i] <= delays[i-1] {
			t.Errorf("BC-MS-5 violated at input=%d: delay=%d <= prev=%d (delays=%v)",
				inputLens[i], delays[i], delays[i-1], delays)
			break
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// BC-MS-7: CacheHitRate Range Invariant
//
// CacheHitRate ∈ [0, 1] for any simulation. With identical sequential
// requests, hit rate should be non-negative.
// ═══════════════════════════════════════════════════════════════════════════════

func TestMetrics_CacheHitRate_BoundedZeroToOne(t *testing.T) {
	cfg := msConfig(math.MaxInt64)
	s := mustNewSimulator(t, cfg)

	// Two identical requests — may benefit from prefix caching depending
	// on block-level retention after release
	for i := 0; i < 2; i++ {
		r := &Request{
			ID:           fmt.Sprintf("cache-%d", i),
			InputTokens:  msMakeTokens(64),
			OutputTokens: msMakeTokens(2),
			ArrivalTime:  int64(i) * 200000,
			State:        StateQueued,
		}
		s.InjectArrival(r)
	}
	s.Run()

	hitRate := s.KVCache.CacheHitRate()
	if hitRate < 0 || hitRate > 1 {
		t.Errorf("BC-MS-7 violated: CacheHitRate (%.4f) outside [0, 1]", hitRate)
	}
	t.Logf("BC-MS-7: CacheHitRate = %.4f (2 identical requests)", hitRate)
}

// ═══════════════════════════════════════════════════════════════════════════════
// BC-MS-8: Aggregate and Per-Request Metrics Are Consistent
//
// The aggregate TTFT mean (from CalculateMean over RequestTTFTs values)
// must equal the per-request TTFT mean computed manually. This verifies
// the metric pipeline produces consistent values across output paths.
// ═══════════════════════════════════════════════════════════════════════════════

func TestMetrics_AggregateAndPerRequest_Consistent(t *testing.T) {
	cfg := msConfig(math.MaxInt64)
	s := mustNewSimulator(t, cfg)
	for i := 0; i < 5; i++ {
		r := &Request{
			ID:           fmt.Sprintf("cons-%d", i),
			InputTokens:  msMakeTokens(16 + 16*i),
			OutputTokens: msMakeTokens(3 + i),
			ArrivalTime:  int64(i) * 100000,
			State:        StateQueued,
		}
		s.InjectArrival(r)
	}
	s.Run()

	// Compute TTFT mean manually from per-request values
	var ttftSum float64
	for _, v := range s.Metrics.RequestTTFTs {
		ttftSum += v
	}
	manualMean := ttftSum / float64(len(s.Metrics.RequestTTFTs))

	// Compare with CalculateMean (the aggregate path)
	ttftVals := make([]float64, 0, len(s.Metrics.RequestTTFTs))
	for _, v := range s.Metrics.RequestTTFTs {
		ttftVals = append(ttftVals, v)
	}
	aggMean := CalculateMean(ttftVals) * 1000 // CalculateMean returns ms, convert back to ticks

	if math.Abs(manualMean-aggMean) > 1.0 {
		t.Errorf("BC-MS-8 violated: manual TTFT mean (%.1f) != CalculateMean (%.1f ticks)",
			manualMean, aggMean)
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// BC-MS-9: Zero-Output Requests
//
// A request with 0 output tokens should still have valid TTFT and E2E,
// with E2E >= TTFT, and should not contaminate ITL statistics for
// normal requests.
// ═══════════════════════════════════════════════════════════════════════════════

func TestMetrics_ZeroOutput_ValidTTFTAndE2E(t *testing.T) {
	// Test both empty slice and nil OutputTokens (Go zero-value).
	for _, tc := range []struct {
		name   string
		output []int
	}{
		{"empty-slice", []int{}},
		{"nil", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := msConfig(math.MaxInt64)
			s := mustNewSimulator(t, cfg)
			req := &Request{
				ID:           "zero-out",
				InputTokens:  msMakeTokens(32),
				OutputTokens: tc.output,
				ArrivalTime:  0,
				State:        StateQueued,
			}
			s.InjectArrival(req)
			s.Run()

			if s.Metrics.CompletedRequests != 1 {
				t.Fatalf("expected 1 completed, got %d", s.Metrics.CompletedRequests)
			}

			ttft := s.Metrics.RequestTTFTs["zero-out"]
			e2e := s.Metrics.RequestE2Es["zero-out"]

			if ttft <= 0 {
				t.Errorf("BC-MS-9 violated: TTFT (%.1f) not positive", ttft)
			}
			if e2e < ttft {
				t.Errorf("BC-MS-9/14 violated: E2E (%.1f) < TTFT (%.1f)", e2e, ttft)
			}

			reqITL := s.Metrics.RequestITLs["zero-out"]
			if reqITL != 0 {
				t.Errorf("BC-MS-9: RequestITLs = %.1f, expected 0", reqITL)
			}
		})
	}
}

func TestMetrics_ZeroOutput_DoesNotContaminateITL(t *testing.T) {
	// Run two simulations: one with a zero-output + normal request,
	// one with only the normal request. The normal request's ITL statistics
	// should not be affected by the zero-output request's presence.
	// (This is a behavioral isolation test.)
	cfgMix := msConfig(math.MaxInt64)
	sMix := mustNewSimulator(t, cfgMix)
	sMix.InjectArrival(&Request{
		ID:           "mix-zero",
		InputTokens:  msMakeTokens(32),
		OutputTokens: []int{},
		ArrivalTime:  0,
		State:        StateQueued,
	})
	sMix.InjectArrival(&Request{
		ID:           "mix-normal",
		InputTokens:  msMakeTokens(32),
		OutputTokens: msMakeTokens(5),
		ArrivalTime:  200000,
		State:        StateQueued,
	})
	sMix.Run()

	cfgOnly := msConfig(math.MaxInt64)
	sOnly := mustNewSimulator(t, cfgOnly)
	sOnly.InjectArrival(&Request{
		ID:           "only-normal",
		InputTokens:  msMakeTokens(32),
		OutputTokens: msMakeTokens(5),
		ArrivalTime:  200000,
		State:        StateQueued,
	})
	sOnly.Run()

	// The normal request's mean ITL should be identical in both scenarios
	mixITL := sMix.Metrics.RequestITLs["mix-normal"]
	onlyITL := sOnly.Metrics.RequestITLs["only-normal"]
	if math.Abs(mixITL-onlyITL) > 1.0 {
		t.Errorf("BC-MS-9: Zero-output contaminates normal request ITL: mix=%.1f, only=%.1f",
			mixITL, onlyITL)
	}

	// Assert zero-output requests add no phantom entries to AllITLs
	phantomCount := len(sMix.Metrics.AllITLs) - len(sOnly.Metrics.AllITLs)
	if phantomCount != 0 {
		t.Errorf("BC-MS-9 violated: zero-output request added %d phantom entries to AllITLs (expected 0)", phantomCount)
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// BC-MS-10: PeakKVBlocksUsed Properties
//
// Peak must be positive during simulation and bounded by total capacity.
// After all requests complete, UsedBlocks must be 0 (INV-4 corollary).
// ═══════════════════════════════════════════════════════════════════════════════

func TestMetrics_PeakKVBlocks_BoundedAndZeroAfterCompletion(t *testing.T) {
	cfg := msConfig(math.MaxInt64)
	cfg.TotalKVBlocks = 100
	s := mustNewSimulator(t, cfg)

	for i := 0; i < 3; i++ {
		r := &Request{
			ID:           fmt.Sprintf("peak-%d", i),
			InputTokens:  msMakeTokens(48),
			OutputTokens: msMakeTokens(10),
			ArrivalTime:  int64(i) * 1000,
			State:        StateQueued,
		}
		s.InjectArrival(r)
	}
	s.Run()

	peak := s.Metrics.PeakKVBlocksUsed
	if peak <= 0 {
		t.Errorf("BC-MS-10 violated: PeakKVBlocksUsed = %d, expected > 0", peak)
	}
	if peak > cfg.TotalKVBlocks {
		t.Errorf("BC-MS-10 violated: PeakKVBlocksUsed (%d) > TotalKVBlocks (%d)", peak, cfg.TotalKVBlocks)
	}
	if s.KVCache.UsedBlocks() != 0 {
		t.Errorf("BC-MS-10 violated: UsedBlocks = %d after completion, expected 0", s.KVCache.UsedBlocks())
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// BC-MS-11: TTFT Partial Monotonicity
//
// TTFT for an isolated request depends on two independent variables:
//
//   TTFT = QueueingTime(totalInputTokens) + StepTime(cacheMissTokens) + OutputTokenProcessingTime
//
// QueueingTime and StepTime are methods on the LatencyModel interface.
// Any conforming implementation must satisfy:
//   - QueueingTime non-decreasing in totalInputTokens
//   - StepTime non-decreasing in cacheMissTokens (holding batch composition fixed)
//
// This yields PARTIAL monotonicity:
//   (a) Holding cacheMissTokens fixed, TTFT ↑ when totalInputTokens ↑  (alpha side)
//   (b) Holding totalInputTokens fixed, TTFT ↑ when cacheMissTokens ↑  (step-time side)
//   (c) When totalInputTokens ↑ but cacheMissTokens ↓ (prefix caching),
//       the net direction depends on relative magnitudes — monotonicity NOT guaranteed.
//
// Without prefix caching: cacheMissTokens == totalInputTokens, so (a) and (b)
// collapse into simple monotonicity with input length.
//
// These properties hold for any LatencyModel (blackbox alpha/beta, roofline
// FLOPs/bandwidth, or future implementations) as long as the interface
// contract is satisfied. The tests below verify the properties, not any
// particular formula.
// ═══════════════════════════════════════════════════════════════════════════════

func TestMetrics_TTFT_Monotonic_NoCaching(t *testing.T) {
	// Without caching: cacheMissTokens == totalInputTokens.
	// Both components grow → TTFT is monotonically non-decreasing.
	inputLens := []int{16, 32, 64, 128, 256}
	var ttfts []float64

	for _, inputLen := range inputLens {
		// Each request in its own simulator → no prefix cache state
		s := msInjectAndRun(t, msConfig(math.MaxInt64), "mono", inputLen, 2, 0)
		ttfts = append(ttfts, s.Metrics.RequestTTFTs["mono"])
	}

	for i := 1; i < len(ttfts); i++ {
		if ttfts[i] < ttfts[i-1] {
			t.Errorf("BC-MS-11(a+b) violated at input=%d: TTFT=%.1f < prev=%.1f",
				inputLens[i], ttfts[i], ttfts[i-1])
		}
	}
}

func TestMetrics_TTFT_PartialMonotonicity_CacheMissFixed(t *testing.T) {
	// Property (a): Holding cacheMissTokens fixed, increasing totalInputTokens
	// increases TTFT (QueueingTime grows).
	//
	// We approximate this by using requests with the same number of NEW tokens
	// (same cache-miss count) but different total input lengths. Without shared
	// prefix state, cacheMissTokens == totalInputTokens, so we simulate the
	// "fixed cache-miss" condition by checking that QueueingTime is monotonic.
	//
	// Direct test: two isolated requests with same output but different input.
	// Since no caching, cacheMiss == input, and both QueueingTime and StepTime
	// grow. To truly test (a) alone we'd need caching, so this test is the
	// combined (a+b) case — documented for completeness.
	s16 := msInjectAndRun(t, msConfig(math.MaxInt64), "a16", 16, 2, 0)
	s64 := msInjectAndRun(t, msConfig(math.MaxInt64), "a64", 64, 2, 0)

	ttft16 := s16.Metrics.RequestTTFTs["a16"]
	ttft64 := s64.Metrics.RequestTTFTs["a64"]

	if ttft64 <= ttft16 {
		t.Errorf("BC-MS-11(a): 64-token TTFT (%.1f) <= 16-token TTFT (%.1f)", ttft64, ttft16)
	}
}

func TestMetrics_TTFT_NonMonotonic_WithPrefixCaching(t *testing.T) {
	// Property (c) counterexample: prefix caching can break total-input monotonicity.
	//
	// Three requests in one simulator:
	//   req-A (warmup): 128 tokens, no cache → establishes cached blocks
	//   req-B (cached): 128 tokens, same as A → cache hits reduce cacheMissTokens
	//   req-C (uncached): 64 different tokens → zero cache hits
	//
	// req-B has totalInputTokens=128 > req-C's 64, but fewer cache-miss tokens.
	// If the step-time reduction from caching exceeds the alpha increase from
	// longer input, req-B's TTFT < req-C's TTFT — breaking total-input monotonicity.
	cfg := msConfig(math.MaxInt64)
	s := mustNewSimulator(t, cfg)

	reqA := &Request{
		ID:           "warm",
		InputTokens:  msMakeTokens(128),
		OutputTokens: msMakeTokens(2),
		ArrivalTime:  0,
		State:        StateQueued,
	}
	reqB := &Request{
		ID:           "cached",
		InputTokens:  msMakeTokens(128),
		OutputTokens: msMakeTokens(2),
		ArrivalTime:  500000,
		State:        StateQueued,
	}
	reqCTokens := make([]int, 64)
	for i := range reqCTokens {
		reqCTokens[i] = 50000 + i // distinct tokens — no cache overlap
	}
	reqC := &Request{
		ID:           "uncached",
		InputTokens:  reqCTokens,
		OutputTokens: msMakeTokens(2),
		ArrivalTime:  1000000,
		State:        StateQueued,
	}

	s.InjectArrival(reqA)
	s.InjectArrival(reqB)
	s.InjectArrival(reqC)
	s.Run()

	if s.Metrics.CompletedRequests != 3 {
		t.Fatalf("expected 3 completed, got %d", s.Metrics.CompletedRequests)
	}

	ttftB := s.Metrics.RequestTTFTs["cached"]
	ttftC := s.Metrics.RequestTTFTs["uncached"]

	// Document — not a pass/fail, this records the observed behavior.
	t.Logf("128-token cached TTFT:   %.1f ticks", ttftB)
	t.Logf(" 64-token uncached TTFT: %.1f ticks", ttftC)
	if ttftB < ttftC {
		t.Logf("CONFIRMED: property (c) — prefix caching breaks total-input monotonicity")
	} else {
		t.Logf("NOTE: alpha dominates or no cache benefit in this scenario; " +
			"property (c) not triggered but the precondition remains necessary in general")
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// BC-MS-12: Percentile Monotonicity
//
// For any metric distribution: p50 ≤ p90 ≤ p95 ≤ p99.
// ═══════════════════════════════════════════════════════════════════════════════

func TestMetrics_Percentile_Monotonicity(t *testing.T) {
	cfg := msConfig(math.MaxInt64)
	s := mustNewSimulator(t, cfg)

	for i := 0; i < 50; i++ {
		r := &Request{
			ID:           fmt.Sprintf("pct-%d", i),
			InputTokens:  msMakeTokens(16 + i*2),
			OutputTokens: msMakeTokens(3 + (i % 5)),
			ArrivalTime:  int64(i) * 50000,
			State:        StateQueued,
		}
		s.InjectArrival(r)
	}
	s.Run()

	// Check TTFT percentile ordering
	ttftVals := make([]float64, 0, len(s.Metrics.RequestTTFTs))
	for _, v := range s.Metrics.RequestTTFTs {
		ttftVals = append(ttftVals, v)
	}
	sort.Float64s(ttftVals)

	ttftP50 := CalculatePercentile(ttftVals, 50)
	ttftP90 := CalculatePercentile(ttftVals, 90)
	ttftP95 := CalculatePercentile(ttftVals, 95)
	ttftP99 := CalculatePercentile(ttftVals, 99)

	if ttftP50 > ttftP90 || ttftP90 > ttftP95 || ttftP95 > ttftP99 {
		t.Errorf("BC-MS-12 violated for TTFT: p50=%.3f p90=%.3f p95=%.3f p99=%.3f",
			ttftP50, ttftP90, ttftP95, ttftP99)
	}

	// Check E2E percentile ordering
	e2eVals := make([]float64, 0, len(s.Metrics.RequestE2Es))
	for _, v := range s.Metrics.RequestE2Es {
		e2eVals = append(e2eVals, v)
	}
	sort.Float64s(e2eVals)

	e2eP50 := CalculatePercentile(e2eVals, 50)
	e2eP90 := CalculatePercentile(e2eVals, 90)
	e2eP95 := CalculatePercentile(e2eVals, 95)
	e2eP99 := CalculatePercentile(e2eVals, 99)

	if e2eP50 > e2eP90 || e2eP90 > e2eP95 || e2eP95 > e2eP99 {
		t.Errorf("BC-MS-12 violated for E2E: p50=%.3f p90=%.3f p95=%.3f p99=%.3f",
			e2eP50, e2eP90, e2eP95, e2eP99)
	}
}

// Design note: Cluster CacheHitRate uses unweighted instance average
// (sum(per_instance_rate) / N). With imbalanced load, this can underreport
// effective caching vs a block-weighted average. This is a design choice
// documented here rather than tested, since it exercises no simulator code.
