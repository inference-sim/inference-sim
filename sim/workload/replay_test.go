package workload

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestLoadTraceV2Requests_CorrectTokenCounts(t *testing.T) {
	// GIVEN a trace with 2 requests
	header := &TraceHeader{Version: TraceVersion, TimeUnit: "microseconds", Mode: "generated"}
	records := []TraceRecord{
		{RequestID: 0, InputTokens: 100, OutputTokens: 50,
			ArrivalTimeUs: 0, TenantID: "t1", SLOClass: "batch", Status: "ok"},
		{RequestID: 1, InputTokens: 200, OutputTokens: 75,
			ArrivalTimeUs: 100000, TenantID: "t2", SLOClass: "critical", Status: "ok"},
	}

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

	requests, err := LoadTraceV2Requests(trace, 42)
	if err != nil {
		t.Fatal(err)
	}

	if len(requests) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(requests))
	}

	// Token counts should match (input + output)
	if len(requests[0].InputTokens) != 100 {
		t.Errorf("request 0 input tokens = %d, want 100", len(requests[0].InputTokens))
	}
	if len(requests[0].OutputTokens) != 50 {
		t.Errorf("request 0 output tokens = %d, want 50", len(requests[0].OutputTokens))
	}
	if requests[0].TenantID != "t1" {
		t.Errorf("request 0 tenant = %q, want t1", requests[0].TenantID)
	}
	if requests[1].ArrivalTime != 100000 {
		t.Errorf("request 1 arrival = %d, want 100000", requests[1].ArrivalTime)
	}

	// BC-6: MaxOutputLen = len(OutputTokens)
	if requests[0].MaxOutputLen != len(requests[0].OutputTokens) {
		t.Errorf("request 0 MaxOutputLen = %d, want %d", requests[0].MaxOutputLen, len(requests[0].OutputTokens))
	}
	if requests[1].MaxOutputLen != len(requests[1].OutputTokens) {
		t.Errorf("request 1 MaxOutputLen = %d, want %d", requests[1].MaxOutputLen, len(requests[1].OutputTokens))
	}
}

// TestRunReplayParity_Adapter_INV13 (#1470, T044) verifies the full run→replay
// adapter round-trip that INV-13 (per-adapter metric parity) rests on: a request's
// adapter id survives Request → RequestsToTraceRecords → Export/Load →
// LoadTraceV2Requests unchanged. Without adapter round-tripping through TraceV2, an
// adapter-blind replay of a LoRA trace would silently attribute those requests to
// the base model and per-adapter metrics would diverge.
func TestRunReplayParity_Adapter_INV13(t *testing.T) {
	// Build requests as `blis run` would produce them, carrying adapter ids.
	orig := []*sim.Request{
		{ID: "request_0", ArrivalTime: 0, InputTokens: make([]sim.TokenID, 100),
			OutputTokens: make([]sim.TokenID, 50), State: sim.StateCompleted, Adapter: "adapter_0"},
		{ID: "request_1", ArrivalTime: 1000, InputTokens: make([]sim.TokenID, 80),
			OutputTokens: make([]sim.TokenID, 40), State: sim.StateCompleted, Adapter: ""},
		{ID: "request_2", ArrivalTime: 2000, InputTokens: make([]sim.TokenID, 120),
			OutputTokens: make([]sim.TokenID, 60), State: sim.StateCompleted, Adapter: "adapter_7"},
	}
	// Mark TTFT so timing fields are emitted (mimics a completed run).
	for _, r := range orig {
		r.TTFTSet = true
		r.FirstTokenTime = 500
		r.ITL = []int64{10, 10}
	}

	records := RequestsToTraceRecords(orig)
	for i, want := range []string{"adapter_0", "", "adapter_7"} {
		if records[i].Adapter != want {
			t.Fatalf("record %d Adapter = %q, want %q (request→record mapping)", i, records[i].Adapter, want)
		}
	}

	dir := t.TempDir()
	headerPath := filepath.Join(dir, "h.yaml")
	dataPath := filepath.Join(dir, "d.csv")
	header := &TraceHeader{Version: TraceVersion, TimeUnit: "microseconds", Mode: "generated"}
	if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatal(err)
	}
	trace, err := LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := LoadTraceV2Requests(trace, 42)
	if err != nil {
		t.Fatal(err)
	}
	if len(replayed) != len(orig) {
		t.Fatalf("replayed %d requests, want %d", len(replayed), len(orig))
	}
	// INV-13 mechanism: adapter id survives the full cycle for every request.
	for i, r := range replayed {
		if r.Adapter != orig[i].Adapter {
			t.Errorf("request %d Adapter = %q after replay, want %q (INV-13 adapter round-trip)", i, r.Adapter, orig[i].Adapter)
		}
	}
}

func TestLoadTraceV2Requests_PrefixGroup_SharedTokens(t *testing.T) {
	header := &TraceHeader{Version: TraceVersion, TimeUnit: "microseconds", Mode: "generated"}
	records := []TraceRecord{
		{RequestID: 0, InputTokens: 100, OutputTokens: 50,
			PrefixGroup: "shared", PrefixLength: 128, ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, InputTokens: 100, OutputTokens: 50,
			PrefixGroup: "shared", PrefixLength: 128, ArrivalTimeUs: 100000, Status: "ok"},
	}

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

	requests, err := LoadTraceV2Requests(trace, 42)
	if err != nil {
		t.Fatal(err)
	}

	// BC-3: Both requests share identical first 128 tokens
	if len(requests[0].InputTokens) < 128 || len(requests[1].InputTokens) < 128 {
		t.Fatal("input tokens too short for prefix check")
	}
	for i := 0; i < 128; i++ {
		if requests[0].InputTokens[i] != requests[1].InputTokens[i] {
			t.Errorf("prefix token %d differs: %d vs %d", i,
				requests[0].InputTokens[i], requests[1].InputTokens[i])
			break
		}
	}
	// BC-6: Total input length = prefix(128) + suffix(100) = 228
	if len(requests[0].InputTokens) != 228 {
		t.Errorf("input length = %d, want 228 (128 prefix + 100 suffix)", len(requests[0].InputTokens))
	}
	// BC-3: PrefixGroup propagated to Request
	if requests[0].PrefixGroup != "shared" {
		t.Errorf("PrefixGroup = %q, want %q", requests[0].PrefixGroup, "shared")
	}
	// PrefixLength propagated to Request
	if requests[0].PrefixLength != 128 {
		t.Errorf("PrefixLength = %d, want 128", requests[0].PrefixLength)
	}
}

// --- LoadTraceV2SessionBlueprints tests (BC-5, BC-6) ---

func TestLoadTraceV2SessionBlueprints_GroupsBySession(t *testing.T) {
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0},
			{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 5000},
			{RequestID: 3, SessionID: "B", RoundIndex: 0, InputTokens: 150, OutputTokens: 60, ArrivalTimeUs: 1000},
		},
	}

	requests, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// BC-5: 2 blueprints (one per session)
	if len(blueprints) != 2 {
		t.Fatalf("BC-5: got %d blueprints, want 2", len(blueprints))
	}
	// BC-5: 2 round-0 requests injected
	if len(requests) != 2 {
		t.Fatalf("BC-5: got %d requests, want 2", len(requests))
	}

	var bpA *SessionBlueprint
	for i := range blueprints {
		if blueprints[i].SessionID == "A" {
			bpA = &blueprints[i]
			break
		}
	}
	if bpA == nil {
		t.Fatal("blueprint A not found")
	}
	if bpA.MaxRounds != 2 {
		t.Errorf("BC-5: session A MaxRounds = %d, want 2", bpA.MaxRounds)
	}

	// BC-6: input sampler replays round-1 token count (round 0 is injected directly)
	got1 := bpA.InputSampler.Sample(nil)
	if got1 != 200 {
		t.Errorf("BC-6: input sampler first value = %d, want 200 (round 1 token count)", got1)
	}
}

func TestLoadTraceV2SessionBlueprints_NonSessionPassThrough(t *testing.T) {
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0},
			{RequestID: 2, SessionID: "A", RoundIndex: 0, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 1000},
		},
	}

	requests, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// 1 non-session + 1 round-0 session request = 2 requests total
	if len(requests) != 2 {
		t.Fatalf("got %d requests, want 2 (1 non-session + 1 round-0 session)", len(requests))
	}
	if len(blueprints) != 1 {
		t.Errorf("got %d blueprints, want 1", len(blueprints))
	}
}

func TestLoadTraceV2SessionBlueprints_ThinkTimeFromTrace(t *testing.T) {
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0},
			{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 5000},
			{RequestID: 3, SessionID: "A", RoundIndex: 2, InputTokens: 300, OutputTokens: 90, ArrivalTimeUs: 12000},
		},
	}

	_, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	bp := blueprints[0]
	// Think times derived from inter-round arrival gaps: [5000, 7000]
	if bp.ThinkTimeSampler == nil {
		t.Fatal("expected ThinkTimeSampler to be set for multi-round session")
	}
	got1 := bp.ThinkTimeSampler.Sample(nil)
	got2 := bp.ThinkTimeSampler.Sample(nil)
	if got1 != 5000 || got2 != 7000 {
		t.Errorf("think times = [%d, %d], want [5000, 7000]", got1, got2)
	}
}

func TestLoadTraceV2SessionBlueprints_SingleRoundSession(t *testing.T) {
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0},
		},
	}

	requests, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) != 1 || len(blueprints) != 1 {
		t.Fatalf("got %d requests, %d blueprints; want 1, 1", len(requests), len(blueprints))
	}
	bp := blueprints[0]
	if bp.MaxRounds != 1 {
		t.Errorf("MaxRounds = %d, want 1", bp.MaxRounds)
	}
	if bp.ThinkTimeSampler != nil {
		t.Error("expected nil ThinkTimeSampler for single-round session")
	}
}

func TestLoadTraceV2SessionBlueprints_OverrideThinkTime(t *testing.T) {
	// GIVEN a 2-round session and a ConstantSampler providing 500ms think time
	// WHEN blueprints are built
	// THEN the session's ThinkTimeSampler returns 500_000 µs on every call
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0},
			{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 5000},
		},
	}

	sampler := &ConstantSampler{value: 500_000}
	_, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, sampler, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	bp := blueprints[0]
	if bp.ThinkTimeSampler == nil {
		t.Fatal("expected ThinkTimeSampler to be set when sampler provided")
	}
	got := bp.ThinkTimeSampler.Sample(nil)
	if got != 500_000 {
		t.Errorf("ThinkTimeSampler.Sample() = %d, want 500000 µs", got)
	}
}

func TestLoadTraceV2SessionBlueprints_PrefersRecordedThinkTime(t *testing.T) {
	// GIVEN a 3-round session whose recorded think_time_us (300, 400) DIFFERS from
	// its inter-round arrival gaps (5000, 7000)
	// WHEN blueprints are built with no caller sampler
	// THEN the think sampler yields the RECORDED think times, not the arrival gaps
	// (#1478: recorded per-round think time is pure client think, preferred over the
	// gap derivation which bundles service time).
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0, ThinkTimeUs: nil},
			{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 5000, ThinkTimeUs: i64p(300)},
			{RequestID: 3, SessionID: "A", RoundIndex: 2, InputTokens: 300, OutputTokens: 90, ArrivalTimeUs: 12000, ThinkTimeUs: i64p(400)},
		},
	}

	_, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	bp := blueprints[0]
	if bp.ThinkTimeSampler == nil {
		t.Fatal("expected ThinkTimeSampler to be set")
	}
	got1 := bp.ThinkTimeSampler.Sample(nil)
	got2 := bp.ThinkTimeSampler.Sample(nil)
	if got1 != 300 || got2 != 400 {
		t.Errorf("think times = [%d, %d], want recorded [300, 400] (not arrival gaps [5000, 7000])", got1, got2)
	}
}

func TestLoadTraceV2SessionBlueprints_RecordedThinkTime_CLIOverrides(t *testing.T) {
	// GIVEN a session with recorded think_time_us AND a caller-provided sampler
	// WHEN blueprints are built
	// THEN the caller sampler wins (#1478 precedence: CLI > recorded > gap).
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0, ThinkTimeUs: nil},
			{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 5000, ThinkTimeUs: i64p(300)},
		},
	}
	sampler := &ConstantSampler{value: 500_000}
	_, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, sampler, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := blueprints[0].ThinkTimeSampler.Sample(nil); got != 500_000 {
		t.Errorf("ThinkTimeSampler.Sample() = %d, want 500000 (CLI sampler overrides recorded think_time_us)", got)
	}
}

func TestLoadTraceV2SessionBlueprints_NonMonotoneGapClamped(t *testing.T) {
	// GIVEN a 2-round session where round-1 has an earlier arrival than round-0
	// (clock skew in observed trace), THEN ThinkTimeSampler returns 0 (not negative),
	// preserving INV-3 (clock monotonicity) in the follow-up arrival computation.
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 5000},
			{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 3000},
		},
	}

	_, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(blueprints) != 1 {
		t.Fatalf("expected 1 blueprint, got %d", len(blueprints))
	}
	bp := blueprints[0]
	if bp.ThinkTimeSampler == nil {
		t.Fatal("expected ThinkTimeSampler to be set for multi-round session")
	}
	got := bp.ThinkTimeSampler.Sample(nil)
	if got != 0 {
		t.Errorf("clamped think time = %d, want 0 (negative gap must be clamped to 0)", got)
	}
}

func TestLoadTraceV2SessionBlueprints_NonConsecutiveRoundIndex_Error(t *testing.T) {
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0},
			{RequestID: 2, SessionID: "A", RoundIndex: 2, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 5000},
		},
	}

	_, _, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err == nil {
		t.Fatal("expected error for non-consecutive round indices, got nil")
	}
}

// TestLoadTraceV2Requests_ModelAndDeadline verifies that Model, Deadline, empty Model,
// and zero Deadline are propagated from TraceRecord to sim.Request
// (2026-03-14-pr653-tracev2-schema-plan.md BC-3–6), and that the prompt is input_tokens long.
func TestLoadTraceV2Requests_ModelAndDeadline(t *testing.T) {
	header := &TraceHeader{Version: TraceVersion, TimeUnit: "microseconds", Mode: "converted"}
	records := []TraceRecord{
		{
			RequestID:     0,
			Model:         "meta-llama/Llama-3.1-8B-Instruct",
			DeadlineUs:    7500000,
			InputTokens:   300,
			OutputTokens:  50,
			ArrivalTimeUs: 0,
			Status:        "ok",
		},
		{
			RequestID:     1,
			Model:         "", // BC-6: empty = default model
			DeadlineUs:    0,  // BC-5: no timeout
			InputTokens:   50,
			OutputTokens:  25,
			ArrivalTimeUs: 1000,
			Status:        "ok",
		},
	}

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
	requests, err := LoadTraceV2Requests(trace, 42)
	if err != nil {
		t.Fatal(err)
	}
	if len(requests) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(requests))
	}

	// BC-3: Model propagated
	if requests[0].Model != "meta-llama/Llama-3.1-8B-Instruct" {
		t.Errorf("request 0 Model = %q, want %q", requests[0].Model, "meta-llama/Llama-3.1-8B-Instruct")
	}
	// BC-4: Deadline propagated
	if requests[0].Deadline != 7500000 {
		t.Errorf("request 0 Deadline = %d, want 7500000", requests[0].Deadline)
	}
	// BC-6: empty Model propagated as-is
	if requests[1].Model != "" {
		t.Errorf("request 1 Model = %q, want empty", requests[1].Model)
	}
	// BC-5: zero Deadline propagated as-is (no timeout)
	if requests[1].Deadline != 0 {
		t.Errorf("request 1 Deadline = %d, want 0", requests[1].Deadline)
	}
	// The synthetic prompt is input_tokens long.
	if len(requests[0].InputTokens) != 300 || len(requests[1].InputTokens) != 50 {
		t.Errorf("input token counts = %d, %d, want 300 and 50", len(requests[0].InputTokens), len(requests[1].InputTokens))
	}
}

// TestLoadTraceV2Requests_GeneratedTraceInjectionUnchanged verifies BC-3
// (INV-13/INV-6): a request is injected exactly at its arrival_time_us, also when the
// first arrival is after 0, as a blis run trace's is for a positive rate.
func TestLoadTraceV2Requests_GeneratedTraceInjectionUnchanged(t *testing.T) {
	// blis run --rate 10 produces the first arrival at 99999 (not 0).
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 0, ArrivalTimeUs: 99_999, InputTokens: 50, OutputTokens: 25, Status: "ok"},
			{RequestID: 1, ArrivalTimeUs: 199_998, InputTokens: 50, OutputTokens: 25, Status: "ok"},
			{RequestID: 2, ArrivalTimeUs: 299_997, InputTokens: 50, OutputTokens: 25, Status: "ok"},
		},
	}
	reqs, err := LoadTraceV2Requests(trace, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := []int64{99_999, 199_998, 299_997}
	for i, req := range reqs {
		if req.ArrivalTime != want[i] {
			t.Errorf("BC-3 (INV-13): request %d ArrivalTime = %d, want %d (the arrival time)", i, req.ArrivalTime, want[i])
		}
	}
}

// TestMaxNormalizedInjectionTimeUs verifies BC-6: the preliminary closed-loop horizon basis
// is the latest arrival among the records injected initially (round-0 and non-session).
func TestMaxNormalizedInjectionTimeUs(t *testing.T) {
	trace := &TraceV2{
		Records: []TraceRecord{
			// Session round-0 (injected initially).
			{RequestID: 0, SessionID: "A", RoundIndex: 0, ArrivalTimeUs: 0},
			// Session follow-up (NOT injected initially — must be skipped).
			{RequestID: 1, SessionID: "A", RoundIndex: 1, ArrivalTimeUs: 900_000},
			// Non-session (injected initially); latest arrival → sets the max.
			{RequestID: 2, ArrivalTimeUs: 700_000},
		},
	}
	if got := MaxNormalizedInjectionTimeUs(trace); got != 700_000 {
		t.Errorf("MaxNormalizedInjectionTimeUs = %d, want 700000 (max arrival over injected records)", got)
	}
	if got := MaxNormalizedInjectionTimeUs(nil); got != 0 {
		t.Errorf("MaxNormalizedInjectionTimeUs(nil) = %d, want 0", got)
	}
}

// TestLoadTraceV2SessionBlueprints_ConcurrencyModeInjection verifies that the round-0 and
// non-session requests are injected at their arrival_time_us, and that think time comes from
// arrival deltas.
func TestLoadTraceV2SessionBlueprints_ConcurrencyModeInjection(t *testing.T) {
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0, ArrivalTimeUs: 0, InputTokens: 100, OutputTokens: 50},
			{RequestID: 2, SessionID: "A", RoundIndex: 1, ArrivalTimeUs: 200000, InputTokens: 150, OutputTokens: 60},
			{RequestID: 3, SessionID: "", ArrivalTimeUs: 10000, InputTokens: 80, OutputTokens: 30},
		},
	}
	requests, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	got := map[string]int64{}
	for _, r := range requests {
		got[r.SessionID] = r.ArrivalTime
	}
	if got["A"] != 0 || got[""] != 10000 || len(got) != 2 {
		t.Errorf("injections %v, want session A round 0 at 0 and the non-session request at 10000", got)
	}
	if len(blueprints) != 1 || blueprints[0].ThinkTimeSampler == nil {
		t.Fatalf("expected one blueprint with a think-time sampler, got %+v", blueprints)
	}
	if think := blueprints[0].ThinkTimeSampler.Sample(nil); think != 200000 {
		t.Errorf("think time = %d, want 200000 (the arrival gap)", think)
	}
}

func TestLoadTraceV2SessionBlueprints_UnknownContextGrowth_Errors(t *testing.T) {
	// A typo'd session_context_growth (e.g. wrong case) must fail loudly rather than
	// silently falling through to the non-accumulate branch and disabling the feature.
	trace := &TraceV2{
		Header: TraceHeader{SessionContextGrowth: "Accumulate"}, // wrong case
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50},
			{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 60, OutputTokens: 40},
		},
	}
	_, _, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err == nil || !strings.Contains(err.Error(), "session_context_growth") {
		t.Errorf("expected error for unknown session_context_growth, got %v", err)
	}
}

func TestLoadTraceV2SessionBlueprints_AccumulateFromHeader(t *testing.T) {
	trace := &TraceV2{
		Header: TraceHeader{Version: TraceVersion, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"},
		Records: []TraceRecord{
			{RequestID: 0, SessionID: "s", RoundIndex: 0, InputTokens: 100, OutputTokens: 10, ArrivalTimeUs: 0},
			{RequestID: 1, SessionID: "s", RoundIndex: 1, InputTokens: 40, OutputTokens: 20, ArrivalTimeUs: 8_000_000},
		},
	}
	_, bps, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(bps) != 1 {
		t.Fatalf("got %d blueprints, want 1", len(bps))
	}
	if bps[0].ContextGrowth != "accumulate" {
		t.Errorf("ContextGrowth = %q, want accumulate", bps[0].ContextGrowth)
	}
}

func TestLoadTraceV2SessionBlueprints_DefaultNoAccumulate(t *testing.T) {
	trace := &TraceV2{
		Header: TraceHeader{Version: TraceVersion, TimeUnit: "microseconds", Mode: "generated"}, // no growth field
		Records: []TraceRecord{
			{RequestID: 0, SessionID: "s", RoundIndex: 0, InputTokens: 100, OutputTokens: 10, ArrivalTimeUs: 0},
			{RequestID: 1, SessionID: "s", RoundIndex: 1, InputTokens: 40, OutputTokens: 20, ArrivalTimeUs: 8_000_000},
		},
	}
	_, bps, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if bps[0].ContextGrowth != "" {
		t.Errorf("ContextGrowth = %q, want empty (unchanged default)", bps[0].ContextGrowth)
	}
}

func TestAccumulateReplay_StrictPrefixIdentity(t *testing.T) {
	trace := &TraceV2{
		Header: TraceHeader{Version: TraceVersion, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"},
		Records: []TraceRecord{
			{RequestID: 0, SessionID: "s", RoundIndex: 0, InputTokens: 100, OutputTokens: 10, ArrivalTimeUs: 0},
			{RequestID: 1, SessionID: "s", RoundIndex: 1, InputTokens: 40, OutputTokens: 20, ArrivalTimeUs: 8_000_000},
		},
	}
	r0, bps, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	sm := NewSessionManager(bps)

	// Round 0 input = 100 tokens (no prefix group).
	round0 := r0[0]
	if round0.InputLen() != 100 {
		t.Fatalf("round0 input len = %d, want 100", round0.InputLen())
	}
	// Simulate round 0 completing: mark completed with full output generated so
	// ProgressIndex - InputLen == actual output (accumulate uses actual output).
	round0.State = sim.StateCompleted
	round0.ProgressIndex = int64(round0.InputLen()) + 10 // 10 output tokens generated

	// Capture round 0's input token IDs before follow-up assembly.
	prefix := append([]sim.TokenID{}, round0.FullInputTokens()...)

	followUps := sm.OnComplete(round0, 8_000_000)
	if len(followUps) != 1 {
		t.Fatalf("got %d follow-ups, want 1", len(followUps))
	}
	round1 := followUps[0]
	// Round 1 total input = round0(100) + round0 output(10) + delta(40) = 150.
	if round1.InputLen() != 150 {
		t.Fatalf("round1 input len = %d, want 150", round1.InputLen())
	}
	// STRICT prefix identity: round1's first 100 tokens are byte-identical to round0's input.
	got := round1.FullInputTokens()
	for i := 0; i < 100; i++ {
		if got[i] != prefix[i] {
			t.Fatalf("round1 token[%d]=%d != round0 token[%d]=%d — prefix not strictly identical", i, got[i], i, prefix[i])
		}
	}
}

// TestLoadTraceV2SessionBlueprints_AllRecordedZeroThink_UsesRecordedZeros pins the
// non-lossy fix (#1608, was the F1 lossy-sentinel behavior of #1484): a session whose
// every non-round-0 round RECORDS think == 0 (non-nil &0, real for Weka's
// overlap-clamped rounds) now USES those recorded zeros (back-to-back rounds) —
// it no longer degrades to arrival-gap derivation, which bundled service time into
// the gap.
func TestLoadTraceV2SessionBlueprints_AllRecordedZeroThink_UsesRecordedZeros(t *testing.T) {
	trace := &TraceV2{Records: []TraceRecord{
		// Round 0 not recorded (nil); rounds 1..2 recorded &0 (overlapping turns).
		{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0, ThinkTimeUs: nil},
		{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 60, OutputTokens: 40, ArrivalTimeUs: 5000, ThinkTimeUs: i64p(0)},
		{RequestID: 3, SessionID: "A", RoundIndex: 2, InputTokens: 70, OutputTokens: 30, ArrivalTimeUs: 12000, ThinkTimeUs: i64p(0)},
	}}
	_, bps, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	s := bps[0].ThinkTimeSampler
	if s == nil {
		t.Fatal("expected a think sampler for a multi-round session")
	}
	// Uses the RECORDED zeros [0, 0], NOT the arrival gaps [5000, 7000].
	if g1, g2 := s.Sample(nil), s.Sample(nil); g1 != 0 || g2 != 0 {
		t.Errorf("all-recorded-zero think: sampler = [%d, %d], want recorded zeros [0, 0] (#1608 non-lossy), not arrival gaps [5000, 7000]", g1, g2)
	}
}

// TestLoadTraceV2SessionBlueprints_AbsentThink_FallsBackToGap pins the surviving
// fallback path (#1608): a session whose non-round-0 rounds are all NOT recorded
// (nil, real for OTel / generated traces) still derives think from arrival gaps.
func TestLoadTraceV2SessionBlueprints_AbsentThink_FallsBackToGap(t *testing.T) {
	trace := &TraceV2{Records: []TraceRecord{
		{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0, ThinkTimeUs: nil},
		{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 60, OutputTokens: 40, ArrivalTimeUs: 5000, ThinkTimeUs: nil},
		{RequestID: 3, SessionID: "A", RoundIndex: 2, InputTokens: 70, OutputTokens: 30, ArrivalTimeUs: 12000, ThinkTimeUs: nil},
	}}
	_, bps, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	s := bps[0].ThinkTimeSampler
	if s == nil {
		t.Fatal("expected a think sampler for a multi-round session")
	}
	// No recorded think → arrival gaps [5000, 7000].
	if g1, g2 := s.Sample(nil), s.Sample(nil); g1 != 5000 || g2 != 7000 {
		t.Errorf("absent think: sampler = [%d, %d], want arrival gaps [5000, 7000]", g1, g2)
	}
}

// TestLoadTraceV2SessionBlueprints_MixedInteriorThink_DerefDefensive exercises the
// defensive nil-interior path (#1608, blis-pr-review + qa-review G1 optional note): a
// single session with a nil interior round (round 1) among recorded rounds (round 2
// non-nil). sessionHasRecordedThinkTime is true (round 2 recorded), so the whole
// session uses the recorded path — and the nil interior round derefs to 0 think, NOT
// to its arrival gap. No shipped converter emits such a session; this pins the deref
// behavior for a hypothetical future sparse-think converter.
func TestLoadTraceV2SessionBlueprints_MixedInteriorThink_DerefDefensive(t *testing.T) {
	trace := &TraceV2{Records: []TraceRecord{
		// Arrival gaps would be [3000, 7000] if this fell back — it must NOT.
		{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0, ThinkTimeUs: nil},
		{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 60, OutputTokens: 40, ArrivalTimeUs: 3000, ThinkTimeUs: nil},         // interior nil
		{RequestID: 3, SessionID: "A", RoundIndex: 2, InputTokens: 70, OutputTokens: 30, ArrivalTimeUs: 10000, ThinkTimeUs: i64p(5000)}, // recorded
	}}
	_, bps, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	s := bps[0].ThinkTimeSampler
	if s == nil {
		t.Fatal("expected a think sampler for a multi-round session")
	}
	// Recorded path (round 2 non-nil) with nil interior derefing to 0: [0, 5000],
	// NOT the arrival gaps [3000, 7000].
	if g1, g2 := s.Sample(nil), s.Sample(nil); g1 != 0 || g2 != 5000 {
		t.Errorf("mixed-interior think: sampler = [%d, %d], want [0, 5000] (nil interior → 0 deref, round 2 recorded), not arrival gaps [3000, 7000]", g1, g2)
	}
}

// TestLoadTraceV2SessionBlueprints_MixedThinkSessions covers F4 (#1484 review),
// updated for #1608: think-time selection is per-session, so a trace mixing a
// recorded-think session with a not-recorded one selects each independently — A uses
// its recorded think, B (nil) falls back to arrival gaps. (Under the non-lossy
// encoding, a recorded &0 would NOT fall back — see AllRecordedZeroThink — so B must
// be genuinely absent (nil) to exercise the fallback.)
func TestLoadTraceV2SessionBlueprints_MixedThinkSessions(t *testing.T) {
	trace := &TraceV2{Records: []TraceRecord{
		// A: recorded think 300; arrival gap (9999) deliberately differs so the two are distinguishable.
		{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0, ThinkTimeUs: nil},
		{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 60, OutputTokens: 40, ArrivalTimeUs: 9999, ThinkTimeUs: i64p(300)},
		// B: NOT recorded (nil); arrival gap 5000.
		{RequestID: 4, SessionID: "B", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0, ThinkTimeUs: nil},
		{RequestID: 5, SessionID: "B", RoundIndex: 1, InputTokens: 60, OutputTokens: 40, ArrivalTimeUs: 5000, ThinkTimeUs: nil},
	}}
	_, bps, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	byID := map[string]*SessionBlueprint{}
	for i := range bps {
		byID[bps[i].SessionID] = &bps[i]
	}
	if got := byID["A"].ThinkTimeSampler.Sample(nil); got != 300 {
		t.Errorf("session A think = %d, want recorded 300", got)
	}
	if got := byID["B"].ThinkTimeSampler.Sample(nil); got != 5000 {
		t.Errorf("session B think = %d, want arrival gap 5000 (not-recorded fallback)", got)
	}
}

// TestAccumulateReplay_TransitivityAndLengthCap covers F3 (#1484 review): 3+ round
// transitivity — round 2's prefix is byte-identical to round 1's FULL input as the
// accumulate buffer keeps growing (which also verifies round 1's interior segment
// flows forward unchanged) — plus the length-capped path (actual output < recorded
// output shrinks the appended segment).
func TestAccumulateReplay_TransitivityAndLengthCap(t *testing.T) {
	mk := func() (*sim.Request, *SessionManager) {
		trace := &TraceV2{
			Header: TraceHeader{Version: TraceVersion, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"},
			Records: []TraceRecord{
				{RequestID: 0, SessionID: "s", RoundIndex: 0, InputTokens: 100, OutputTokens: 10, ArrivalTimeUs: 0},
				{RequestID: 1, SessionID: "s", RoundIndex: 1, InputTokens: 40, OutputTokens: 20, ArrivalTimeUs: 1_000_000},
				{RequestID: 2, SessionID: "s", RoundIndex: 2, InputTokens: 25, OutputTokens: 5, ArrivalTimeUs: 2_000_000},
			},
		}
		r0, bps, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		return r0[0], NewSessionManager(bps)
	}
	next := func(sm *SessionManager, req *sim.Request, tick int64) *sim.Request {
		fu := sm.OnComplete(req, tick)
		if len(fu) != 1 {
			t.Fatalf("expected 1 follow-up, got %d", len(fu))
		}
		return fu[0]
	}

	t.Run("full-output transitivity", func(t *testing.T) {
		round0, sm := mk()
		round0.State = sim.StateCompleted
		round0.ProgressIndex = int64(round0.InputLen()) + 10 // full 10 output tokens
		round1 := next(sm, round0, 1_000_000)
		if round1.InputLen() != 150 { // 100 + 10 output + 40 delta
			t.Fatalf("round1 len = %d, want 150", round1.InputLen())
		}
		round1In := append([]sim.TokenID{}, round1.FullInputTokens()...)
		round1.State = sim.StateCompleted
		round1.ProgressIndex = int64(round1.InputLen()) + 20
		round2 := next(sm, round1, 2_000_000)
		if round2.InputLen() != 195 { // 150 + 20 output + 25 delta
			t.Fatalf("round2 len = %d, want 195", round2.InputLen())
		}
		r2 := round2.FullInputTokens()
		for i := 0; i < 150; i++ {
			if r2[i] != round1In[i] {
				t.Fatalf("round2 token[%d] diverged from round1 full input — transitivity/interior broken", i)
			}
		}
	})

	t.Run("length-capped output", func(t *testing.T) {
		round0, sm := mk()
		round0.State = sim.StateCompleted
		round0.ProgressIndex = int64(round0.InputLen()) + 5 // only 5 of the 10 output tokens generated
		round1 := next(sm, round0, 1_000_000)
		// Appended segment tracks ACTUAL output, not the recorded count: 100 + 5 + 40 = 145.
		if round1.InputLen() != 145 {
			t.Errorf("length-capped round1 len = %d, want 145 (100 + actual-output 5 + delta 40)", round1.InputLen())
		}
	})
}

// TestAccumulateReplay_CompactionResetsBuffer covers #1609: a round whose trace
// carries an input_tokens_reset marker (context compaction) re-seeds the accumulate
// buffer to the recorded absolute — shrinking the reconstructed input to in_N exactly
// and intentionally breaking strict prefix identity across the boundary — while
// monotone rounds (before AND after the compaction) keep the growing-prefix behavior.
func TestAccumulateReplay_CompactionResetsBuffer(t *testing.T) {
	// Recorded absolutes: 100 → 150(mono) → 50(COMPACTION, 50 < 150+20) → 80(mono).
	// As delta-encoded records: round2 carries delta 0 + InputTokensReset=&50.
	trace := &TraceV2{
		Header: TraceHeader{Version: TraceVersion, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"},
		Records: []TraceRecord{
			{RequestID: 0, SessionID: "s", RoundIndex: 0, InputTokens: 100, OutputTokens: 10, ArrivalTimeUs: 0},
			{RequestID: 1, SessionID: "s", RoundIndex: 1, InputTokens: 40, OutputTokens: 20, ArrivalTimeUs: 1_000_000},
			{RequestID: 2, SessionID: "s", RoundIndex: 2, InputTokens: 0, OutputTokens: 5, ArrivalTimeUs: 2_000_000, InputTokensReset: i64p(50)},
			{RequestID: 3, SessionID: "s", RoundIndex: 3, InputTokens: 25, OutputTokens: 5, ArrivalTimeUs: 3_000_000},
		},
	}
	r0, bps, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	// The loader must build a reset sampler for a compaction-bearing session.
	if bps[0].InputResetSampler == nil {
		t.Fatal("InputResetSampler = nil, want a sampler (session carries a compaction marker)")
	}
	sm := NewSessionManager(bps)

	next := func(req *sim.Request, tick int64) *sim.Request {
		req.State = sim.StateCompleted
		req.ProgressIndex = int64(req.InputLen()) + int64(len(req.OutputTokens)) // full output generated
		fu := sm.OnComplete(req, tick)
		if len(fu) != 1 {
			t.Fatalf("expected 1 follow-up, got %d", len(fu))
		}
		return fu[0]
	}

	round1 := next(r0[0], 1_000_000)
	if round1.InputLen() != 150 { // 100 + 10 output + 40 delta — normal accumulate
		t.Fatalf("round1 len = %d, want 150 (monotone accumulate)", round1.InputLen())
	}
	round1In := append([]sim.TokenID{}, round1.FullInputTokens()...)

	round2 := next(round1, 2_000_000)
	// BC-2: buffer shrinks to the recorded absolute (50), NOT the over-counted
	// 150+20+0 = 170 the append-only buffer would have produced.
	if round2.InputLen() != 50 {
		t.Fatalf("compaction round2 len = %d, want 50 (re-seeded to recorded absolute, not 170)", round2.InputLen())
	}
	// BC-3: strict prefix identity is broken across the compaction boundary —
	// round2 is a fresh segment, not a token-prefix of round1.
	round2In := round2.FullInputTokens()
	prefixIdentical := true
	for i := 0; i < 50; i++ {
		if round2In[i] != round1In[i] {
			prefixIdentical = false
			break
		}
	}
	if prefixIdentical {
		t.Error("round2 is a strict prefix of round1 — compaction must break prefix identity (#1609)")
	}

	// BC-3 (other direction): a monotone round AFTER the compaction resumes the
	// growing-prefix behavior from the re-seeded segment. round3 = 50 + 5 out + 25 delta.
	round2FullIn := append([]sim.TokenID{}, round2In...)
	round3 := next(round2, 3_000_000)
	if round3.InputLen() != 80 {
		t.Fatalf("post-compaction round3 len = %d, want 80 (50 + 5 output + 25 delta)", round3.InputLen())
	}
	got := round3.FullInputTokens()
	for i := 0; i < 50; i++ {
		if got[i] != round2FullIn[i] {
			t.Fatalf("round3 token[%d] diverged from round2 — post-compaction prefix must be preserved", i)
		}
	}
}

// TestAccumulateReplay_ZeroReset_EmptyBuffer exercises the degenerate `&0` reset
// end-to-end through OnComplete (blis-pr-review coverage note): a compaction to an
// absolute of 0 (full context discarded) makes the buffer empty. The follow-up round
// then carries a 0-length input, and the buffer-continuity invariant still holds on the
// NEXT round (req.InputLen()==buf.Len()==0). No shipped converter emits a 0-token round
// (nil/0 in/out rounds are dropped at conversion), but the loader accepts &0, so this
// pins that the reset path handles it without panicking rather than assuming InputLen()>0.
func TestAccumulateReplay_ZeroReset_EmptyBuffer(t *testing.T) {
	trace := &TraceV2{
		Header: TraceHeader{Version: TraceVersion, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"},
		Records: []TraceRecord{
			{RequestID: 0, SessionID: "s", RoundIndex: 0, InputTokens: 100, OutputTokens: 10, ArrivalTimeUs: 0},
			// round 1 compacts to an absolute of 0 (full context discarded).
			{RequestID: 1, SessionID: "s", RoundIndex: 1, InputTokens: 0, OutputTokens: 5, ArrivalTimeUs: 1_000_000, InputTokensReset: i64p(0)},
			// round 2 grows again from the empty buffer.
			{RequestID: 2, SessionID: "s", RoundIndex: 2, InputTokens: 30, OutputTokens: 5, ArrivalTimeUs: 2_000_000},
		},
	}
	r0, bps, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	sm := NewSessionManager(bps)
	next := func(req *sim.Request, tick int64) *sim.Request {
		req.State = sim.StateCompleted
		req.ProgressIndex = int64(req.InputLen()) + int64(len(req.OutputTokens))
		fu := sm.OnComplete(req, tick)
		if len(fu) != 1 {
			t.Fatalf("expected 1 follow-up, got %d", len(fu))
		}
		return fu[0]
	}

	round1 := next(r0[0], 1_000_000)
	if round1.InputLen() != 0 {
		t.Fatalf("zero-reset round1 InputLen = %d, want 0 (buffer fully reset)", round1.InputLen())
	}
	// Continuity must still hold on the next round: growing from an empty buffer.
	// round2 = 0 (buffer) + 5 (round1 actual output) + 30 (delta) = 35.
	round2 := next(round1, 2_000_000)
	if round2.InputLen() != 35 {
		t.Fatalf("post-zero-reset round2 InputLen = %d, want 35 (0 + 5 output + 30 delta)", round2.InputLen())
	}
}

// TestAccumulateReplay_ConsecutiveCompactions covers back-to-back compaction rounds
// (qa-review G9): a session that compacts on two consecutive turns (abs 200 → 150 → 100)
// must re-seed the buffer to the recorded absolute EACH round, with the continuity
// invariant holding across successive resets (each reset installs a fresh buffer; the
// next completion validates its length before resetting again) and strict prefix identity
// broken at every compaction boundary.
func TestAccumulateReplay_ConsecutiveCompactions(t *testing.T) {
	// abs: 200, then 150 (150 < 200+10), then 100 (100 < 150+5) — two compactions in a row.
	trace := &TraceV2{
		Header: TraceHeader{Version: TraceVersion, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"},
		Records: []TraceRecord{
			{RequestID: 0, SessionID: "s", RoundIndex: 0, InputTokens: 200, OutputTokens: 10, ArrivalTimeUs: 0},
			{RequestID: 1, SessionID: "s", RoundIndex: 1, InputTokens: 0, OutputTokens: 5, ArrivalTimeUs: 1_000_000, InputTokensReset: i64p(150)},
			{RequestID: 2, SessionID: "s", RoundIndex: 2, InputTokens: 0, OutputTokens: 5, ArrivalTimeUs: 2_000_000, InputTokensReset: i64p(100)},
		},
	}
	r0, bps, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	sm := NewSessionManager(bps)
	next := func(req *sim.Request, tick int64) *sim.Request {
		req.State = sim.StateCompleted
		req.ProgressIndex = int64(req.InputLen()) + int64(len(req.OutputTokens))
		fu := sm.OnComplete(req, tick)
		if len(fu) != 1 {
			t.Fatalf("expected 1 follow-up, got %d", len(fu))
		}
		return fu[0]
	}

	round1 := next(r0[0], 1_000_000)
	round1In := append([]sim.TokenID{}, round1.FullInputTokens()...)
	if round1.InputLen() != 150 {
		t.Fatalf("compaction round1 len = %d, want 150 (re-seed to recorded absolute)", round1.InputLen())
	}
	// Second consecutive compaction: buffer must reset AGAIN to the new absolute (100),
	// not accumulate on top of the first reset (which would give 150+5+0 = 155).
	round2 := next(round1, 2_000_000)
	if round2.InputLen() != 100 {
		t.Fatalf("consecutive compaction round2 len = %d, want 100 (second re-seed, not 155)", round2.InputLen())
	}
	// Strict prefix identity broken at the second boundary too (fresh segment).
	round2In := round2.FullInputTokens()
	prefixIdentical := true
	for i := 0; i < 100; i++ {
		if round2In[i] != round1In[i] {
			prefixIdentical = false
			break
		}
	}
	if prefixIdentical {
		t.Error("round2 is a strict prefix of round1 — each consecutive compaction must break prefix identity")
	}
}

// TestAccumulateReplay_NoResetSampler_WhenMonotone pins BC-5: an accumulate trace
// with no compaction marker builds NO reset sampler, so replay stays on the
// pre-#1609 append-only path (byte-identical, INV-6).
func TestAccumulateReplay_NoResetSampler_WhenMonotone(t *testing.T) {
	trace := &TraceV2{
		Header: TraceHeader{Version: TraceVersion, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"},
		Records: []TraceRecord{
			{RequestID: 0, SessionID: "s", RoundIndex: 0, InputTokens: 100, OutputTokens: 10, ArrivalTimeUs: 0},
			{RequestID: 1, SessionID: "s", RoundIndex: 1, InputTokens: 40, OutputTokens: 20, ArrivalTimeUs: 1_000_000},
		},
	}
	_, bps, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if bps[0].InputResetSampler != nil {
		t.Error("InputResetSampler != nil for a monotone accumulate session — must be nil (INV-6)")
	}
}

// --- Issue #2064: follow-up deadline offsets built from the trace ---

// TestLoadTraceV2SessionBlueprints_DeadlineOffset_None verifies BC-2: a session whose
// rounds all record deadline_us=0 (no deadline) produces a DeadlineOffsetSampler that
// yields the -1 "no deadline" sentinel for every follow-up round.
func TestLoadTraceV2SessionBlueprints_DeadlineOffset_None(t *testing.T) {
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0, DeadlineUs: 0},
			{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 5000, DeadlineUs: 0},
			{RequestID: 3, SessionID: "A", RoundIndex: 2, InputTokens: 300, OutputTokens: 90, ArrivalTimeUs: 12000, DeadlineUs: 0},
		},
	}
	_, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	bp := blueprints[0]
	if bp.DeadlineOffsetSampler == nil {
		t.Fatal("expected DeadlineOffsetSampler to be set on the replay path")
	}
	// rounds 1..2 both carry no deadline => -1, -1
	if o1, o2 := bp.DeadlineOffsetSampler.Sample(nil), bp.DeadlineOffsetSampler.Sample(nil); o1 != -1 || o2 != -1 {
		t.Errorf("offsets = [%d, %d], want [-1, -1] (no deadline)", o1, o2)
	}
}

// TestLoadTraceV2SessionBlueprints_DeadlineOffset_Relative verifies BC-1: a recorded
// absolute deadline_us becomes a per-round offset of deadline_us - arrival_time_us,
// so it re-anchors to the follow-up's simulated arrival.
func TestLoadTraceV2SessionBlueprints_DeadlineOffset_Relative(t *testing.T) {
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0, DeadlineUs: 0},
			// round 1: arrival 5000, deadline 1_005_000 => offset 1_000_000
			{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 5000, DeadlineUs: 1_005_000},
			// round 2: arrival 12000, deadline 12000 + 2_000_000 => offset 2_000_000
			{RequestID: 3, SessionID: "A", RoundIndex: 2, InputTokens: 300, OutputTokens: 90, ArrivalTimeUs: 12000, DeadlineUs: 2_012_000},
		},
	}
	_, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	bp := blueprints[0]
	if bp.DeadlineOffsetSampler == nil {
		t.Fatal("expected DeadlineOffsetSampler to be set")
	}
	if o1, o2 := bp.DeadlineOffsetSampler.Sample(nil), bp.DeadlineOffsetSampler.Sample(nil); o1 != 1_000_000 || o2 != 2_000_000 {
		t.Errorf("offsets = [%d, %d], want [1000000, 2000000]", o1, o2)
	}
}

// TestReplay2064_ClosedLoopFollowUp_NoDeadline_Integration is the end-to-end
// reproduction of issue #2064 at the library boundary: a 2-round session with
// deadline_us=0 on every round, loaded by LoadTraceV2SessionBlueprints and driven
// by a SessionManager, must produce a follow-up request with NO deadline — not the
// 300s default that cancelled heavy sessions before the fix.
func TestReplay2064_ClosedLoopFollowUp_NoDeadline_Integration(t *testing.T) {
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 0, SessionID: "s:long", RoundIndex: 0, InputTokens: 1000, OutputTokens: 60000, ArrivalTimeUs: 0, DeadlineUs: 0},
			{RequestID: 1, SessionID: "s:long", RoundIndex: 1, InputTokens: 1000, OutputTokens: 60000, ArrivalTimeUs: 1_000_000, DeadlineUs: 0},
		},
	}
	r0Requests, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 1_000_000_000_000)
	if err != nil {
		t.Fatalf("LoadTraceV2SessionBlueprints: %v", err)
	}
	if len(r0Requests) != 1 || len(blueprints) != 1 {
		t.Fatalf("want 1 round-0 request + 1 blueprint, got %d, %d", len(r0Requests), len(blueprints))
	}
	// Round 0 carries no deadline (matches the trace).
	if r0Requests[0].Deadline != 0 {
		t.Errorf("round 0 deadline = %d, want 0 (trace deadline_us=0)", r0Requests[0].Deadline)
	}

	sm := NewSessionManager(blueprints)
	// Simulate round 0 completing well past 300s (a long reasoning step) — under the bug
	// the follow-up would be stamped with a 300s deadline relative to its arrival; here it
	// must get no deadline at all.
	req0 := r0Requests[0]
	req0.State = sim.StateCompleted
	req0.ProgressIndex = req0.InputLen() + int64(len(req0.OutputTokens))
	follow := sm.OnComplete(req0, 576_000_000) // round 0 finished at 576s
	if len(follow) != 1 {
		t.Fatalf("expected 1 follow-up round, got %d", len(follow))
	}
	if follow[0].Deadline != 0 {
		t.Errorf("issue #2064: follow-up round deadline = %d, want 0 (trace recorded no deadline); "+
			"a non-zero value is the 300s default bug", follow[0].Deadline)
	}
}
