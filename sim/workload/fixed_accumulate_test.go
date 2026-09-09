package workload

import (
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// buildAccumulateTrace writes a two-session accumulate corpus to a temp dir and
// loads it back, returning the parsed TraceV2. Records carry per-round DELTAS
// (the encoder's law), so the absolute inputs are reconstructed at replay.
func buildAccumulateTrace(t *testing.T, records []TraceRecord) *TraceV2 {
	t.Helper()
	header := &TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"}
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "header.yaml")
	dataPath := filepath.Join(dir, "data.csv")
	if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatal(err)
	}
	trace, err := LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatal(err)
	}
	return trace
}

// TestFixedAccumulate_ReconstructsAbsoluteInputs (BC-1): a session whose recorded
// absolute inputs grow across rounds must produce requests whose InputTokens count
// equals the recorded absolute per round, injected at the recorded arrival time.
func TestFixedAccumulate_ReconstructsAbsoluteInputs(t *testing.T) {
	// Session "s1": abs inputs 100, 100+50+30=180, 180+40+25=245.
	// Deltas: round0=100; round1 = 180-100-50 = 30; round2 = 245-180-40 = 25.
	records := []TraceRecord{
		{RequestID: 0, SessionID: "s1", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "s1", RoundIndex: 1, InputTokens: 30, OutputTokens: 40, ArrivalTimeUs: 1_000_000, Status: "ok"},
		{RequestID: 2, SessionID: "s1", RoundIndex: 2, InputTokens: 25, OutputTokens: 20, ArrivalTimeUs: 2_000_000, Status: "ok"},
	}
	trace := buildAccumulateTrace(t, records)

	reqs, err := LoadTraceV2FixedAccumulateRequests(trace, 42)
	if err != nil {
		t.Fatal(err)
	}
	if len(reqs) != 3 {
		t.Fatalf("expected 3 requests, got %d", len(reqs))
	}
	wantAbs := []int{100, 180, 245}
	wantArrival := []int64{0, 1_000_000, 2_000_000}
	for i, r := range reqs {
		if int(r.InputLen()) != wantAbs[i] {
			t.Errorf("round %d input len = %d, want absolute %d", i, r.InputLen(), wantAbs[i])
		}
		if r.ArrivalTime != wantArrival[i] {
			t.Errorf("round %d arrival = %d, want %d", i, r.ArrivalTime, wantArrival[i])
		}
		if r.RoundIndex != i {
			t.Errorf("round %d RoundIndex = %d, want %d", i, r.RoundIndex, i)
		}
		if r.SessionID != "s1" {
			t.Errorf("round %d SessionID = %q, want s1", i, r.SessionID)
		}
	}
}

// TestFixedAccumulate_GrowingPrefixConsistent (BC-2): round N's input token IDs are
// a strict prefix of round N+1's input token IDs (the growing conversation), so a
// prefix-cache probe across consecutive rounds hits.
func TestFixedAccumulate_GrowingPrefixConsistent(t *testing.T) {
	records := []TraceRecord{
		{RequestID: 0, SessionID: "s1", RoundIndex: 0, InputTokens: 60, OutputTokens: 10, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "s1", RoundIndex: 1, InputTokens: 15, OutputTokens: 8, ArrivalTimeUs: 500_000, Status: "ok"},
	}
	trace := buildAccumulateTrace(t, records)
	reqs, err := LoadTraceV2FixedAccumulateRequests(trace, 7)
	if err != nil {
		t.Fatal(err)
	}
	if len(reqs) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(reqs))
	}
	r0 := reqs[0].FullInputTokens()
	r1 := reqs[1].FullInputTokens()
	if len(r1) <= len(r0) {
		t.Fatalf("round 1 input (%d) should be longer than round 0 (%d)", len(r1), len(r0))
	}
	for i := range r0 {
		if r1[i] != r0[i] {
			t.Fatalf("round 1 input token %d = %d, want round-0 prefix token %d", i, r1[i], r0[i])
		}
	}
	// round 1 input must begin with round-0 input (60) then round-0 output (10).
	r0out := reqs[0].OutputTokens
	for i := range r0out {
		if r1[len(r0)+i] != r0out[i] {
			t.Fatalf("round 1 input token %d (after r0 input) = %d, want round-0 output token %d", len(r0)+i, r1[len(r0)+i], r0out[i])
		}
	}
}

// TestFixedAccumulate_CompactionReset (BC-3): a round carrying an input_tokens_reset
// marker re-seeds the buffer to the recorded absolute, not the over-counted delta.
func TestFixedAccumulate_CompactionReset(t *testing.T) {
	// round0 abs=200, out=100. round1 recorded abs=120 (< 200+100), so the encoder
	// emits delta=0 + input_tokens_reset=120. fixed-accumulate must produce input len 120.
	reset := int64(120)
	records := []TraceRecord{
		{RequestID: 0, SessionID: "s1", RoundIndex: 0, InputTokens: 200, OutputTokens: 100, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "s1", RoundIndex: 1, InputTokens: 0, InputTokensReset: &reset, OutputTokens: 30, ArrivalTimeUs: 1_000_000, Status: "ok"},
	}
	trace := buildAccumulateTrace(t, records)
	reqs, err := LoadTraceV2FixedAccumulateRequests(trace, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(reqs) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(reqs))
	}
	if int(reqs[1].InputLen()) != 120 {
		t.Errorf("round 1 (compaction) input len = %d, want reset absolute 120", reqs[1].InputLen())
	}
}

// TestFixedAccumulate_NonSessionPassThrough (BC-6): non-session single-shot records
// inject at their recorded arrival with their recorded absolute input.
func TestFixedAccumulate_NonSessionPassThrough(t *testing.T) {
	records := []TraceRecord{
		{RequestID: 0, SessionID: "s1", RoundIndex: 0, InputTokens: 50, OutputTokens: 10, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, InputTokens: 80, OutputTokens: 20, ArrivalTimeUs: 250_000, Status: "ok"}, // non-session
	}
	trace := buildAccumulateTrace(t, records)
	reqs, err := LoadTraceV2FixedAccumulateRequests(trace, 9)
	if err != nil {
		t.Fatal(err)
	}
	// find the non-session request
	var found *sim.Request
	for _, r := range reqs {
		if r.SessionID == "" {
			found = r
		}
	}
	if found == nil {
		t.Fatal("non-session request not found")
	}
	if int(found.InputLen()) != 80 {
		t.Errorf("non-session input len = %d, want 80", found.InputLen())
	}
	if found.ArrivalTime != 250_000 {
		t.Errorf("non-session arrival = %d, want 250000", found.ArrivalTime)
	}
}

// TestFixedAccumulate_NonConsecutiveRounds_Error (R1): a session with a round gap
// must error rather than silently misreconstruct.
func TestFixedAccumulate_NonConsecutiveRounds_Error(t *testing.T) {
	records := []TraceRecord{
		{RequestID: 0, SessionID: "s1", RoundIndex: 0, InputTokens: 10, OutputTokens: 5, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "s1", RoundIndex: 2, InputTokens: 5, OutputTokens: 5, ArrivalTimeUs: 1000, Status: "ok"},
	}
	trace := buildAccumulateTrace(t, records)
	if _, err := LoadTraceV2FixedAccumulateRequests(trace, 1); err == nil {
		t.Fatal("expected error for non-consecutive round indices, got nil")
	}
}

// TestFixedAccumulate_Deterministic (BC-4): same trace + seed → identical token IDs.
func TestFixedAccumulate_Deterministic(t *testing.T) {
	records := []TraceRecord{
		{RequestID: 0, SessionID: "s1", RoundIndex: 0, InputTokens: 40, OutputTokens: 10, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "s1", RoundIndex: 1, InputTokens: 5, OutputTokens: 8, ArrivalTimeUs: 1000, Status: "ok"},
	}
	trace := buildAccumulateTrace(t, records)
	a, err := LoadTraceV2FixedAccumulateRequests(trace, 123)
	if err != nil {
		t.Fatal(err)
	}
	b, err := LoadTraceV2FixedAccumulateRequests(trace, 123)
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != len(b) {
		t.Fatalf("length mismatch %d vs %d", len(a), len(b))
	}
	for i := range a {
		ai, bi := a[i].FullInputTokens(), b[i].FullInputTokens()
		if len(ai) != len(bi) {
			t.Fatalf("req %d input length mismatch", i)
		}
		for j := range ai {
			if ai[j] != bi[j] {
				t.Fatalf("req %d input token %d differs across runs", i, j)
			}
		}
	}
}
