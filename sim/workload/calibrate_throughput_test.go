package workload

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

// matchedAll builds a matched-set map for every request ID present in records.
func matchedAll(records []TraceRecord) map[int]bool {
	m := make(map[int]bool, len(records))
	for _, r := range records {
		m[r.RequestID] = true
	}
	return m
}

// TestThroughputComparison_JSONOmitsOptionalPointers verifies BC-2/BC-4/BC-5: the
// per-GPU and verdict pointer fields are dropped from JSON when nil, so a report with
// no --num-gpus / --throughput-tolerance-pct keeps the minimal shape.
func TestThroughputComparison_JSONOmitsOptionalPointers(t *testing.T) {
	tc := &ThroughputComparison{
		MatchedRequests:        3,
		RealRuntimeSec:         2.0,
		SimRuntimeSec:          2.0,
		RealOutputTokens:       30,
		SimOutputTokens:        30,
		RealOutputTokensPerSec: 15,
		SimOutputTokensPerSec:  15,
	}
	data, err := json.Marshal(tc)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	s := string(data)
	for _, absent := range []string{"num_gpus", "per_gpu", "tolerance_pct", "\"within\""} {
		if strings.Contains(s, absent) {
			t.Errorf("JSON should omit %q when pointer is nil, got: %s", absent, s)
		}
	}
	// Round-trip preserves the scalar fields.
	var back ThroughputComparison
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.MatchedRequests != 3 || back.RealOutputTokensPerSec != 15 {
		t.Errorf("round-trip mismatch: %+v", back)
	}
}

// TestComputeThroughput_PerfectMatch verifies BC-1: with identical real/sim latencies
// and no network config, output-token and request throughput match on both sides.
func TestComputeThroughput_PerfectMatch(t *testing.T) {
	// 3 requests, 10 output tokens each, spanning send=0 to last-chunk=2_000_000 (2s).
	// Sim E2E chosen so send+E2E == LastChunkTimeUs for the tail request.
	records := []TraceRecord{
		{Status: "ok", RequestID: 0, SendTimeUs: 0, LastChunkTimeUs: 1_000_000, OutputTokens: 10},
		{Status: "ok", RequestID: 1, SendTimeUs: 500_000, LastChunkTimeUs: 1_500_000, OutputTokens: 10},
		{Status: "ok", RequestID: 2, SendTimeUs: 1_000_000, LastChunkTimeUs: 2_000_000, OutputTokens: 10},
	}
	sim := map[int]SimResult{
		0: {RequestID: 0, E2E: 1_000_000, OutputTokens: 10},
		1: {RequestID: 1, E2E: 1_000_000, OutputTokens: 10},
		2: {RequestID: 2, E2E: 1_000_000, OutputTokens: 10}, // send 1e6 + 1e6 = 2e6 == max real end
	}
	tc := ComputeThroughputComparison(records, sim, matchedAll(records), nil, 0, 0)
	if tc == nil {
		t.Fatal("expected non-nil throughput comparison")
	}
	if tc.MatchedRequests != 3 {
		t.Errorf("matched = %d, want 3", tc.MatchedRequests)
	}
	// makespan = 2e6 - 0 = 2s on both sides; tokens = 30 → 15 tok/s.
	if math.Abs(tc.RealRuntimeSec-2.0) > 1e-9 || math.Abs(tc.SimRuntimeSec-2.0) > 1e-9 {
		t.Errorf("runtime real=%v sim=%v, want 2.0 both", tc.RealRuntimeSec, tc.SimRuntimeSec)
	}
	if math.Abs(tc.RealOutputTokensPerSec-15) > 1e-9 || math.Abs(tc.SimOutputTokensPerSec-15) > 1e-9 {
		t.Errorf("output tok/s real=%v sim=%v, want 15", tc.RealOutputTokensPerSec, tc.SimOutputTokensPerSec)
	}
	if math.Abs(tc.OutputTokensPerSecPercentError) > 1e-9 {
		t.Errorf("percent error = %v, want ~0", tc.OutputTokensPerSecPercentError)
	}
	if math.Abs(tc.RealRequestsPerSec-1.5) > 1e-9 {
		t.Errorf("req/s = %v, want 1.5", tc.RealRequestsPerSec)
	}
	// Per-GPU + verdict pointers absent (BC-4/BC-5).
	if tc.NumGPUs != nil || tc.Within != nil {
		t.Errorf("optional pointers should be nil, got numGPUs=%v within=%v", tc.NumGPUs, tc.Within)
	}
}

// TestComputeThroughput_MetamorphicScaleE2E verifies BC-3: multiplying every sim E2E
// by k scales sim_runtime_sec by ~k and sim_output_tokens_per_sec by ~1/k, isolating
// the makespan (batching-contention) signal.
func TestComputeThroughput_MetamorphicScaleE2E(t *testing.T) {
	records := []TraceRecord{
		{Status: "ok", RequestID: 0, SendTimeUs: 0, LastChunkTimeUs: 1_000_000, OutputTokens: 10},
		{Status: "ok", RequestID: 1, SendTimeUs: 0, LastChunkTimeUs: 1_000_000, OutputTokens: 10},
	}
	base := map[int]SimResult{
		0: {RequestID: 0, E2E: 1_000_000, OutputTokens: 10},
		1: {RequestID: 1, E2E: 1_000_000, OutputTokens: 10},
	}
	scaled := map[int]SimResult{
		0: {RequestID: 0, E2E: 3_000_000, OutputTokens: 10},
		1: {RequestID: 1, E2E: 3_000_000, OutputTokens: 10},
	}
	tcBase := ComputeThroughputComparison(records, base, matchedAll(records), nil, 0, 0)
	tcScaled := ComputeThroughputComparison(records, scaled, matchedAll(records), nil, 0, 0)
	if tcBase == nil || tcScaled == nil {
		t.Fatal("expected non-nil comparisons")
	}
	// Sim runtime triples (3e6 vs 1e6).
	if math.Abs(tcScaled.SimRuntimeSec/tcBase.SimRuntimeSec-3.0) > 1e-6 {
		t.Errorf("sim runtime ratio = %v, want 3.0", tcScaled.SimRuntimeSec/tcBase.SimRuntimeSec)
	}
	// Sim output throughput drops to 1/3.
	if math.Abs(tcScaled.SimOutputTokensPerSec/tcBase.SimOutputTokensPerSec-(1.0/3.0)) > 1e-6 {
		t.Errorf("sim throughput ratio = %v, want 1/3", tcScaled.SimOutputTokensPerSec/tcBase.SimOutputTokensPerSec)
	}
	// Real side unchanged between the two.
	if tcBase.RealOutputTokensPerSec != tcScaled.RealOutputTokensPerSec {
		t.Errorf("real throughput must not change: %v vs %v", tcBase.RealOutputTokensPerSec, tcScaled.RealOutputTokensPerSec)
	}
}

// TestComputeThroughput_ClientFrameNetworkShift verifies I1: a positive NetworkRTTUs
// shifts the sim makespan tail outward (client frame), yielding a strictly larger sim
// runtime than the un-normalized (nil config) case for the same records.
func TestComputeThroughput_ClientFrameNetworkShift(t *testing.T) {
	records := []TraceRecord{
		{Status: "ok", RequestID: 0, SendTimeUs: 0, LastChunkTimeUs: 1_000_000, OutputTokens: 10},
	}
	sim := map[int]SimResult{
		0: {RequestID: 0, E2E: 1_000_000, InputTokens: 10, OutputTokens: 10},
	}
	noNet := ComputeThroughputComparison(records, sim, matchedAll(records), nil, 0, 0)
	withNet := ComputeThroughputComparison(records, sim, matchedAll(records),
		&CalibrationConfig{NetworkRTTUs: 500_000}, 0, 0)
	if noNet == nil || withNet == nil {
		t.Fatal("expected non-nil comparisons")
	}
	if !(withNet.SimRuntimeSec > noNet.SimRuntimeSec) {
		t.Errorf("network RTT should enlarge sim runtime: with=%v no=%v", withNet.SimRuntimeSec, noNet.SimRuntimeSec)
	}
	// Real side is client-side already — unaffected by the sim-side network shift.
	if noNet.RealRuntimeSec != withNet.RealRuntimeSec {
		t.Errorf("real runtime must not change with network config: %v vs %v", noNet.RealRuntimeSec, withNet.RealRuntimeSec)
	}
}

// TestComputeThroughput_PerGPU verifies BC-4: --num-gpus emits per-GPU pointers equal
// to raw ÷ N; numGPUs=0 leaves them nil.
func TestComputeThroughput_PerGPU(t *testing.T) {
	records := []TraceRecord{
		{Status: "ok", RequestID: 0, SendTimeUs: 0, LastChunkTimeUs: 2_000_000, OutputTokens: 40},
	}
	sim := map[int]SimResult{0: {RequestID: 0, E2E: 2_000_000, OutputTokens: 40}}
	tc := ComputeThroughputComparison(records, sim, matchedAll(records), nil, 4, 0)
	if tc == nil || tc.NumGPUs == nil || tc.RealOutputTokensPerSecPerGPU == nil {
		t.Fatal("expected per-GPU pointers set")
	}
	if *tc.NumGPUs != 4 {
		t.Errorf("num_gpus = %d, want 4", *tc.NumGPUs)
	}
	wantPerGPU := tc.RealOutputTokensPerSec / 4
	if math.Abs(*tc.RealOutputTokensPerSecPerGPU-wantPerGPU) > 1e-9 {
		t.Errorf("per-GPU = %v, want %v", *tc.RealOutputTokensPerSecPerGPU, wantPerGPU)
	}
	// numGPUs=0 → nil.
	tc0 := ComputeThroughputComparison(records, sim, matchedAll(records), nil, 0, 0)
	if tc0.NumGPUs != nil {
		t.Errorf("num_gpus should be nil when numGPUs=0, got %v", *tc0.NumGPUs)
	}
}

// TestComputeThroughput_ToleranceVerdict verifies BC-5: a tolerance emits within=true
// when percent error is under the band and false when over; tolerancePct=0 → nil.
func TestComputeThroughput_ToleranceVerdict(t *testing.T) {
	records := []TraceRecord{
		{Status: "ok", RequestID: 0, SendTimeUs: 0, LastChunkTimeUs: 1_000_000, OutputTokens: 10},
	}
	// Sim finishes in ~1.1s vs real 1.0s → ~9% throughput error (real 10/s, sim ~9.09/s).
	sim := map[int]SimResult{0: {RequestID: 0, E2E: 1_100_000, OutputTokens: 10}}
	within := ComputeThroughputComparison(records, sim, matchedAll(records), nil, 0, 15)
	if within == nil || within.Within == nil {
		t.Fatal("expected verdict set")
	}
	if !*within.Within {
		t.Errorf("~9%% error should be within 15%% band, got percentError=%v", within.OutputTokensPerSecPercentError)
	}
	if within.TolerancePct == nil || *within.TolerancePct != 15 {
		t.Errorf("tolerance_pct = %v, want 15", within.TolerancePct)
	}
	// Tighter band → exceeds.
	exceeds := ComputeThroughputComparison(records, sim, matchedAll(records), nil, 0, 5)
	if exceeds.Within == nil || *exceeds.Within {
		t.Errorf("~9%% error should exceed 5%% band")
	}
	// tolerancePct=0 → nil verdict.
	none := ComputeThroughputComparison(records, sim, matchedAll(records), nil, 0, 0)
	if none.Within != nil || none.TolerancePct != nil {
		t.Errorf("verdict pointers should be nil when tolerancePct=0")
	}
}

// TestComputeThroughput_ToleranceBoundaryInclusive verifies the band is inclusive at
// its exact edge (the boundaryEps guard): an exactly-P% error is WITHIN a P% band.
func TestComputeThroughput_ToleranceBoundaryInclusive(t *testing.T) {
	// Construct an EXACT 10% error so the boundaryEps is load-bearing (a value that IEEE-754
	// rounds to exactly 10.0 without the epsilon would fail a strict `< 10` band). Real: 100
	// output tokens over a 1s makespan → 100 tok/s. Sim: 90 output tokens over a 1s makespan →
	// 90 tok/s → |90-100|/100 = 0.1 exactly → 10.0%.
	records := []TraceRecord{
		{Status: "ok", RequestID: 0, SendTimeUs: 0, LastChunkTimeUs: 1_000_000, OutputTokens: 100},
	}
	sim := map[int]SimResult{0: {RequestID: 0, E2E: 1_000_000, OutputTokens: 90}}
	tc := ComputeThroughputComparison(records, sim, matchedAll(records), nil, 0, 10)
	if tc == nil || tc.Within == nil {
		t.Fatal("expected verdict set")
	}
	// An exactly-10% error against a 10% band must be inclusive (WITHIN), not a spurious
	// EXCEEDS — this is what the boundaryEps guards.
	if math.Abs(tc.OutputTokensPerSecPercentError-0.10) > 1e-12 {
		t.Fatalf("fixture should yield exactly 10%% error, got %v", tc.OutputTokensPerSecPercentError)
	}
	if !*tc.Within {
		t.Errorf("exactly-10%% error at 10%% band should be WITHIN (inclusive boundary)")
	}
}

// TestComputeThroughput_SkipsCorruptRecords verifies the per-record guard drops a
// negative-send / negative-sim-E2E record rather than letting it distort the window.
func TestComputeThroughput_SkipsCorruptRecords(t *testing.T) {
	records := []TraceRecord{
		{Status: "ok", RequestID: 0, SendTimeUs: 0, LastChunkTimeUs: 1_000_000, OutputTokens: 10},
		{Status: "ok", RequestID: 1, SendTimeUs: -5, LastChunkTimeUs: 2_000_000, OutputTokens: 10}, // negative send → skipped
		{Status: "ok", RequestID: 2, SendTimeUs: 500_000, LastChunkTimeUs: 1_500_000, OutputTokens: 10},
	}
	sim := map[int]SimResult{
		0: {RequestID: 0, E2E: 1_000_000, OutputTokens: 10},
		1: {RequestID: 1, E2E: 1_000_000, OutputTokens: 10},
		2: {RequestID: 2, E2E: -1, OutputTokens: 10}, // negative sim E2E → skipped
	}
	tc := ComputeThroughputComparison(records, sim, matchedAll(records), nil, 0, 0)
	if tc == nil {
		t.Fatal("expected non-nil (request 0 is valid)")
	}
	// Only request 0 survives → 1 matched, 10 output tokens.
	if tc.MatchedRequests != 1 || tc.RealOutputTokens != 10 {
		t.Errorf("corrupt records should be skipped: matched=%d tokens=%d, want 1/10", tc.MatchedRequests, tc.RealOutputTokens)
	}
}

// TestComputeThroughput_DegenerateReturnsNil verifies BC-2/I3: empty matched set, zero
// makespan, and zero output tokens all return nil (never Inf/NaN).
func TestComputeThroughput_DegenerateReturnsNil(t *testing.T) {
	// Empty matched set.
	if tc := ComputeThroughputComparison(nil, map[int]SimResult{}, map[int]bool{}, nil, 0, 0); tc != nil {
		t.Errorf("empty set should return nil, got %+v", tc)
	}
	// Single instant: send == last-chunk fails the LastChunkTimeUs > SendTimeUs guard → nil.
	instant := []TraceRecord{{Status: "ok", RequestID: 0, SendTimeUs: 1000, LastChunkTimeUs: 1000, OutputTokens: 10}}
	simInstant := map[int]SimResult{0: {RequestID: 0, E2E: 0, OutputTokens: 10}}
	if tc := ComputeThroughputComparison(instant, simInstant, matchedAll(instant), nil, 0, 0); tc != nil {
		t.Errorf("zero-makespan should return nil, got %+v", tc)
	}
	// Zero output tokens on the real side.
	zeroTok := []TraceRecord{{Status: "ok", RequestID: 0, SendTimeUs: 0, LastChunkTimeUs: 1_000_000, OutputTokens: 0}}
	simZeroTok := map[int]SimResult{0: {RequestID: 0, E2E: 1_000_000, OutputTokens: 0}}
	if tc := ComputeThroughputComparison(zeroTok, simZeroTok, matchedAll(zeroTok), nil, 0, 0); tc != nil {
		t.Errorf("zero output tokens should return nil, got %+v", tc)
	}
}
