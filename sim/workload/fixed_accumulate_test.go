package workload

import (
	"path/filepath"
	"reflect"
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

// TestFixedAccumulate_EncoderRoundTrip (BC-1, BC-3): pins the reconstruction to the REAL
// EncodeSessionToTraceRecords law rather than to hand-computed deltas. Every other test in
// this file hand-computes the delta (e.g. 180-100-50=30), so all would pass even if the
// loader and encoder disagreed. This drives known absolutes (including shrinking rounds
// that trigger compaction) through the actual encoder, then asserts the loader
// reconstructs each round's exact recorded absolute. Randomized over many trials — the RNG
// only varies the shape (seeded off the trial index; no Math.rand/time), keeping it
// deterministic and INV-6-safe.
func TestFixedAccumulate_EncoderRoundTrip(t *testing.T) {
	for trial := 0; trial < 200; trial++ {
		// Deterministic pseudo-random shape from the trial index (no wall-clock/global rng).
		rng := deterministicShapeRNG(int64(trial))
		nRounds := 1 + rng(8) // 1..8 rounds
		abs := make([]int, nRounds)
		out := make([]int, nRounds)
		prev := 20 + rng(200)
		for i := 0; i < nRounds; i++ {
			out[i] = rng(50)
			if i == 0 {
				abs[i] = prev
			} else if rng(3) == 0 {
				// ~1/3 compaction rounds: shrink below prev+out (forces InputTokensReset).
				abs[i] = 5 + rng(prev)
			} else {
				abs[i] = abs[i-1] + out[i-1] + rng(100) // monotone growth
			}
		}
		rounds := make([]NormalizedRound, nRounds)
		for i := range rounds {
			rounds[i] = NormalizedRound{InputTokensAbs: abs[i], OutputTokens: out[i], ArrivalUs: int64(i * 1000), Status: "ok"}
		}
		recs := EncodeSessionToTraceRecords("s1", rounds)
		for i := range recs {
			recs[i].RequestID = i
		}
		trace := buildAccumulateTrace(t, recs)
		reqs, err := LoadTraceV2FixedAccumulateRequests(trace, int64(trial))
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		if len(reqs) != nRounds {
			t.Fatalf("trial %d: got %d requests, want %d", trial, len(reqs), nRounds)
		}
		for i, r := range reqs {
			if int(r.InputLen()) != abs[i] {
				t.Fatalf("trial %d round %d: reconstructed input = %d, want recorded absolute %d (encoder/loader law drift)", trial, i, r.InputLen(), abs[i])
			}
		}
	}
}

// deterministicShapeRNG returns a closure yielding a pseudo-random int in [0, n) from a
// seeded LCG — deterministic (INV-6) and free of Math.rand/time so the round-trip test's
// shapes are reproducible across runs.
func deterministicShapeRNG(seed int64) func(n int) int {
	state := uint64(seed)*2862933555777941757 + 3037000493
	return func(n int) int {
		state = state*6364136223846793005 + 1442695040888963407
		if n <= 0 {
			return 0
		}
		return int((state >> 33) % uint64(n))
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

// TestFixedAccumulate_SingleRoundSession: a session with exactly one round is just its
// round-0 absolute input at its recorded arrival (no growth, no follow-up).
func TestFixedAccumulate_SingleRoundSession(t *testing.T) {
	records := []TraceRecord{
		{RequestID: 0, SessionID: "s1", RoundIndex: 0, InputTokens: 70, OutputTokens: 12, ArrivalTimeUs: 300_000, Status: "ok"},
	}
	trace := buildAccumulateTrace(t, records)
	reqs, err := LoadTraceV2FixedAccumulateRequests(trace, 11)
	if err != nil {
		t.Fatal(err)
	}
	if len(reqs) != 1 {
		t.Fatalf("expected 1 request, got %d", len(reqs))
	}
	if int(reqs[0].InputLen()) != 70 {
		t.Errorf("single-round input len = %d, want 70", reqs[0].InputLen())
	}
	if reqs[0].ArrivalTime != 300_000 {
		t.Errorf("single-round arrival = %d, want 300000", reqs[0].ArrivalTime)
	}
	if len(reqs[0].OutputTokens) != 12 {
		t.Errorf("single-round output len = %d, want 12", len(reqs[0].OutputTokens))
	}
}

// TestFixedAccumulate_CompactionReset_Content: after a reset round the input tokens are
// a FRESH segment, NOT a continuation of the pre-compaction buffer — the reset round's
// input must not share the pre-reset round-0 prefix (real compaction replaces history
// with a summary, mirroring SessionManager.OnComplete's Reset semantics, #1609).
func TestFixedAccumulate_CompactionReset_Content(t *testing.T) {
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
	r0 := reqs[0].FullInputTokens() // 200 tokens
	r1 := reqs[1].FullInputTokens() // 120 tokens (reset), NOT 200+100+delta
	if len(r1) != 120 {
		t.Fatalf("reset round input len = %d, want 120", len(r1))
	}
	// The reset segment is freshly generated: it must NOT be a prefix continuation of
	// round 0 (otherwise the buffer wasn't reset — it just kept growing).
	sharedPrefix := 0
	for sharedPrefix < len(r0) && sharedPrefix < len(r1) && r0[sharedPrefix] == r1[sharedPrefix] {
		sharedPrefix++
	}
	if sharedPrefix == len(r1) {
		t.Error("reset round input is a prefix of round 0 — buffer was not reset (compaction ignored)")
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

// TestFixedAccumulate_PrefixMetadataParity: round 0 carries PrefixGroup/PrefixLength
// (for cross-session prefix-affinity routing, parity with LoadTraceV2SessionBlueprints),
// but follow-up rounds do NOT (accumulate folds the prefix into the growing buffer).
func TestFixedAccumulate_PrefixMetadataParity(t *testing.T) {
	records := []TraceRecord{
		{RequestID: 0, SessionID: "s1", RoundIndex: 0, PrefixGroup: "sys", PrefixLength: 8, InputTokens: 20, OutputTokens: 5, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "s1", RoundIndex: 1, InputTokens: 4, OutputTokens: 3, ArrivalTimeUs: 1000, Status: "ok"},
	}
	trace := buildAccumulateTrace(t, records)
	reqs, err := LoadTraceV2FixedAccumulateRequests(trace, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(reqs) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(reqs))
	}
	if reqs[0].PrefixGroup != "sys" || reqs[0].PrefixLength != 8 {
		t.Errorf("round 0 prefix = %q/%d, want sys/8", reqs[0].PrefixGroup, reqs[0].PrefixLength)
	}
	// round 0 input = prefix(8) + suffix(20) = 28 tokens.
	if int(reqs[0].InputLen()) != 28 {
		t.Errorf("round 0 input len = %d, want 28 (prefix 8 + suffix 20)", reqs[0].InputLen())
	}
	if reqs[1].PrefixGroup != "" || reqs[1].PrefixLength != 0 {
		t.Errorf("follow-up prefix = %q/%d, want empty (prefix folded into buffer)", reqs[1].PrefixGroup, reqs[1].PrefixLength)
	}
}

// TestFixedAccumulate_ViewsStableAfterGrowthAndReset pins the shared-buffer view
// contract (the memory optimization from PR review F1): each round's InputTokens is a
// VIEW into the session buffer, not a copy, so it must remain content-correct after all
// later growth AND after a compaction reset reallocates the backing array. We snapshot
// every round's expected content up front, then re-read after the whole session is built.
func TestFixedAccumulate_ViewsStableAfterGrowthAndReset(t *testing.T) {
	// Growth rounds 0,1 then a compaction reset at round 2 (reset reallocates the buffer),
	// then growth again at round 3 — exercises both hazards the removed copy guarded against.
	reset := int64(50)
	records := []TraceRecord{
		{RequestID: 0, SessionID: "s1", RoundIndex: 0, InputTokens: 40, OutputTokens: 10, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "s1", RoundIndex: 1, InputTokens: 8, OutputTokens: 6, ArrivalTimeUs: 1000, Status: "ok"},
		{RequestID: 2, SessionID: "s1", RoundIndex: 2, InputTokens: 0, InputTokensReset: &reset, OutputTokens: 5, ArrivalTimeUs: 2000, Status: "ok"},
		{RequestID: 3, SessionID: "s1", RoundIndex: 3, InputTokens: 7, OutputTokens: 4, ArrivalTimeUs: 3000, Status: "ok"},
	}
	trace := buildAccumulateTrace(t, records)
	reqs, err := LoadTraceV2FixedAccumulateRequests(trace, 99)
	if err != nil {
		t.Fatal(err)
	}
	if len(reqs) != 4 {
		t.Fatalf("expected 4 requests, got %d", len(reqs))
	}
	// Expected absolute lengths: r0=40, r1=40+10+8=58, r2=reset 50, r3=50+5+7=62.
	wantLen := []int{40, 58, 50, 62}
	// Snapshot each round's content immediately (defensive copies) so we can detect
	// later mutation of a shared backing array.
	snaps := make([][]sim.TokenID, len(reqs))
	for i, r := range reqs {
		if int(r.InputLen()) != wantLen[i] {
			t.Fatalf("round %d input len = %d, want %d", i, r.InputLen(), wantLen[i])
		}
		snaps[i] = append([]sim.TokenID(nil), r.FullInputTokens()...)
	}
	// Re-read every round's live view AFTER the whole session is built: it must still
	// equal the snapshot (no shift from later Append, no orphan-corruption from Reset).
	for i, r := range reqs {
		live := r.FullInputTokens()
		if len(live) != len(snaps[i]) {
			t.Fatalf("round %d view length changed: %d != %d", i, len(live), len(snaps[i]))
		}
		for j := range live {
			if live[j] != snaps[i][j] {
				t.Fatalf("round %d view token %d mutated after session build (%d != %d)", i, j, live[j], snaps[i][j])
			}
		}
	}
}

// TestFixedAccumulate_EpochScopedCapacity pins PR-review F1: the backing array of each
// round's view is sized to its EPOCH's peak, not the session's GLOBAL peak. The failure
// mode: a tiny first epoch followed by a huge later epoch — sizing to the global peak
// would pin round 0's tiny view inside a giant array, retaining more than an eager copy.
func TestFixedAccumulate_EpochScopedCapacity(t *testing.T) {
	// Epoch 0: round 0 tiny (10 tokens). Round 1 compacts (reset to 5), opening epoch 1,
	// which then grows huge (delta 5000). Global peak ≈ 5000+; epoch-0 peak is 10.
	reset := int64(5)
	records := []TraceRecord{
		{RequestID: 0, SessionID: "s1", RoundIndex: 0, InputTokens: 10, OutputTokens: 2, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "s1", RoundIndex: 1, InputTokens: 0, InputTokensReset: &reset, OutputTokens: 3, ArrivalTimeUs: 1000, Status: "ok"},
		{RequestID: 2, SessionID: "s1", RoundIndex: 2, InputTokens: 5000, OutputTokens: 4, ArrivalTimeUs: 2000, Status: "ok"},
	}
	trace := buildAccumulateTrace(t, records)
	reqs, err := LoadTraceV2FixedAccumulateRequests(trace, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(reqs) != 3 {
		t.Fatalf("expected 3 requests, got %d", len(reqs))
	}
	// Round 0's view (epoch 0, peak 10) must NOT be backed by an array sized for the huge
	// later epoch. cap of the round-0 InputTokens slice reflects its backing array size.
	r0cap := cap(reqs[0].InputTokens)
	if int64(r0cap) > 100 { // generous slack over epoch-0 peak (10); the bug gives ~5008
		t.Errorf("round 0 view cap = %d, want ~10 (epoch-scoped); a global-peak alloc would pin ~5008 (F1 regression)", r0cap)
	}
	// Round 2 (epoch 1) is correctly the large one — sanity that reconstruction still works.
	if int(reqs[2].InputLen()) != 5008 { // reset 5 + out 3 + delta 5000
		t.Errorf("round 2 input len = %d, want 5008 (reset 5 + prev out 3 + delta 5000)", reqs[2].InputLen())
	}
}

// TestFixedAccumulate_MultiSessionArrivalOrder: the returned slice must be
// non-decreasing in ArrivalTime (the RequestSource contract) even when sessions
// interleave in time. Sessions are built session-major (all of s1, then s2), but a
// high-concurrency corpus interleaves arrivals — s2's round 0 lands between s1's rounds.
func TestFixedAccumulate_MultiSessionArrivalOrder(t *testing.T) {
	// s1 arrivals 0, 1_000_000, 2_000_000; s2 arrivals 500_000, 1_500_000 — interleaved.
	records := []TraceRecord{
		{RequestID: 0, SessionID: "s1", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, SessionID: "s1", RoundIndex: 1, InputTokens: 30, OutputTokens: 40, ArrivalTimeUs: 1_000_000, Status: "ok"},
		{RequestID: 2, SessionID: "s1", RoundIndex: 2, InputTokens: 25, OutputTokens: 20, ArrivalTimeUs: 2_000_000, Status: "ok"},
		{RequestID: 3, SessionID: "s2", RoundIndex: 0, InputTokens: 60, OutputTokens: 15, ArrivalTimeUs: 500_000, Status: "ok"},
		{RequestID: 4, SessionID: "s2", RoundIndex: 1, InputTokens: 20, OutputTokens: 10, ArrivalTimeUs: 1_500_000, Status: "ok"},
	}
	trace := buildAccumulateTrace(t, records)
	reqs, err := LoadTraceV2FixedAccumulateRequests(trace, 42)
	if err != nil {
		t.Fatal(err)
	}
	if len(reqs) != 5 {
		t.Fatalf("expected 5 requests, got %d", len(reqs))
	}
	for i := 1; i < len(reqs); i++ {
		if reqs[i].ArrivalTime < reqs[i-1].ArrivalTime {
			t.Errorf("request %d arrival %d < request %d arrival %d — RequestSource order violated",
				i, reqs[i].ArrivalTime, i-1, reqs[i-1].ArrivalTime)
		}
	}
	// Reconstruction must survive the sort: each session's rounds keep their absolute
	// inputs regardless of interleaving. Verify via a session_id → round → len map.
	got := map[string]map[int]int{}
	for _, r := range reqs {
		if got[r.SessionID] == nil {
			got[r.SessionID] = map[int]int{}
		}
		got[r.SessionID][r.RoundIndex] = int(r.InputLen())
	}
	if got["s1"][0] != 100 || got["s1"][1] != 180 || got["s1"][2] != 245 {
		t.Errorf("s1 absolute inputs = %v, want 100/180/245", got["s1"])
	}
	if got["s2"][0] != 60 || got["s2"][1] != 95 {
		t.Errorf("s2 absolute inputs = %v, want 60/95", got["s2"])
	}
}

// TestFixedAccumulate_FieldParityWithLoadTraceV2Requests (R4/R23): guards against field
// drift between the two TraceRecord → sim.Request construction sites. A fully-populated
// NON-session record must map to the SAME sim.Request through both LoadTraceV2Requests
// and the fixed-accumulate loader (both treat a non-session record as suffix-only input
// at recorded arrival). Token-slice CONTENTS differ by RNG, so we compare all fields
// except the token slices (compared by length) and ID. If a future field is added to one
// loader's struct literal but not the other, this test fails.
func TestFixedAccumulate_FieldParityWithLoadTraceV2Requests(t *testing.T) {
	rec := TraceRecord{
		RequestID: 7, ClientID: "c9", TenantID: "t3", SLOClass: "critical",
		PrefixGroup: "grp", PrefixLength: 4, Streaming: true,
		InputTokens: 30, OutputTokens: 12, TextTokens: 20, ImageTokens: 5,
		AudioTokens: 3, VideoTokens: 2, ReasonRatio: 0.25, Model: "m1",
		DeadlineUs: 900_000, SLOTargetUs: 120_000, Adapter: "ad1",
		ArrivalTimeUs: 111_000, Status: "ok",
	}
	// No session_id → non-session record. Both loaders take the same construction path.
	trace := buildAccumulateTrace(t, []TraceRecord{rec})
	viaFixedAcc, err := LoadTraceV2FixedAccumulateRequests(trace, 1)
	if err != nil {
		t.Fatal(err)
	}
	viaFixed, err := LoadTraceV2Requests(trace, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(viaFixedAcc) != 1 || len(viaFixed) != 1 {
		t.Fatalf("expected 1 request each, got %d / %d", len(viaFixedAcc), len(viaFixed))
	}
	a, b := viaFixedAcc[0], viaFixed[0]
	// Token slice lengths must match; contents are RNG-dependent so not compared here.
	if a.InputLen() != b.InputLen() || len(a.OutputTokens) != len(b.OutputTokens) {
		t.Fatalf("token lengths differ: in %d/%d out %d/%d", a.InputLen(), b.InputLen(), len(a.OutputTokens), len(b.OutputTokens))
	}
	// Zero the fields that legitimately differ or aren't structural (ID is identical here
	// anyway, but tokens carry RNG content), then require struct equality on the rest.
	clear := func(r *sim.Request) sim.Request {
		c := *r
		c.InputTokens = nil
		c.OutputTokens = nil
		return c
	}
	if !reflect.DeepEqual(clear(a), clear(b)) {
		t.Errorf("field drift between fixed-accumulate and LoadTraceV2Requests construction:\nfixed-acc: %+v\nfixed:     %+v", clear(a), clear(b))
	}
}

// TestFixedAccumulate_NormalizesEpochSendOrigin (#1606): a corpus whose send_time_us is
// epoch-scale while arrival_time_us is run-relative must inject on the arrival origin
// (not at epoch scale), with send-delta spacing preserved — the same normalization
// LoadTraceV2Requests applies, exercised on the fixed-accumulate loader.
func TestFixedAccumulate_NormalizesEpochSendOrigin(t *testing.T) {
	const epoch = int64(1_787_274_995_712_218)
	// One session, two rounds. Round 0 waited 100ms for a slot, round 1 waited 300ms,
	// so the SEND delta (200000) differs from the ARRIVAL delta (50000).
	records := []TraceRecord{
		{RequestID: 0, SessionID: "s1", RoundIndex: 0, InputTokens: 40, OutputTokens: 10,
			ArrivalTimeUs: 0, SendTimeUs: epoch + 100_000, DeadlineUs: 300_000_000, Status: "ok"},
		{RequestID: 1, SessionID: "s1", RoundIndex: 1, InputTokens: 5, OutputTokens: 8,
			ArrivalTimeUs: 50_000, SendTimeUs: epoch + 300_000, DeadlineUs: 350_000_000, Status: "ok"},
	}
	trace := buildAccumulateTrace(t, records)
	reqs, err := LoadTraceV2FixedAccumulateRequests(trace, 42)
	if err != nil {
		t.Fatal(err)
	}
	if len(reqs) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(reqs))
	}
	// Re-based onto the arrival origin: earliest injection is 0, not epoch-scale.
	if reqs[0].ArrivalTime != 0 {
		t.Errorf("round 0 ArrivalTime = %d, want 0 (arrival origin, not epoch)", reqs[0].ArrivalTime)
	}
	if reqs[0].ArrivalTime >= reqs[0].Deadline {
		t.Errorf("round 0 injected at %d >= deadline %d — would instant-timeout (#1606)", reqs[0].ArrivalTime, reqs[0].Deadline)
	}
	// Spacing follows the SEND delta (200000), not the arrival delta (50000).
	if gotDelta := reqs[1].ArrivalTime - reqs[0].ArrivalTime; gotDelta != 200_000 {
		t.Errorf("injection delta = %d, want 200000 (send delta)", gotDelta)
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
