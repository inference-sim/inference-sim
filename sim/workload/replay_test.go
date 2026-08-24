package workload

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestLoadTraceV2Requests_CorrectTokenCounts(t *testing.T) {
	// GIVEN a trace with 2 requests
	header := &TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "generated"}
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
	header := &TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "generated"}
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
	header := &TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "generated"}
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
			{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0, ThinkTimeUs: 0},
			{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 5000, ThinkTimeUs: 300},
			{RequestID: 3, SessionID: "A", RoundIndex: 2, InputTokens: 300, OutputTokens: 90, ArrivalTimeUs: 12000, ThinkTimeUs: 400},
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
			{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0, ThinkTimeUs: 0},
			{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 5000, ThinkTimeUs: 300},
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

// --- effectiveInputTokenCount unit tests ---

func TestEffectiveInputTokenCount(t *testing.T) {
	cases := []struct {
		name         string
		inputTokens  int
		serverTokens int
		prefixGroup  string
		want         int
	}{
		// Server > client, no prefix: use server (chat-template overhead case)
		{"server_overrides_client", 512, 530, "", 530},
		// Server < client, no prefix: use server (unusual but valid — server is authoritative)
		{"server_smaller_than_client", 512, 480, "", 480},
		// Server == client, no prefix: use server (no-op, same result)
		{"server_equals_client", 256, 256, "", 256},
		// Server > 0 but prefix group set: fall back to client (avoid double-counting)
		{"prefix_group_falls_back", 100, 246, "shared", 100},
		// Server == 0, no prefix: fall back to client (not recorded, e.g. generated trace)
		{"zero_server_falls_back", 256, 0, "", 256},
		// Server == 0 and prefix group: fall back to client
		{"zero_server_prefix_group", 100, 0, "shared", 100},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := effectiveInputTokenCount(tc.inputTokens, tc.serverTokens, tc.prefixGroup)
			if got != tc.want {
				t.Errorf("effectiveInputTokenCount(%d, %d, %q) = %d, want %d",
					tc.inputTokens, tc.serverTokens, tc.prefixGroup, got, tc.want)
			}
		})
	}
}

// --- ServerInputTokens tests (BC-1, BC-2) ---

func TestLoadTraceV2Requests_ServerInputTokens_UsedWhenPresent(t *testing.T) {
	// GIVEN a trace record where ServerInputTokens > InputTokens (chat template overhead)
	// and no PrefixGroup (the --api-format chat use case)
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 0, InputTokens: 512, ServerInputTokens: 530,
				OutputTokens: 64, ArrivalTimeUs: 0, Status: "ok"},
		},
	}
	requests, err := LoadTraceV2Requests(trace, 42)
	if err != nil {
		t.Fatal(err)
	}
	// BC-1: len(InputTokens) reflects server-reported count, not client-side count
	if len(requests[0].InputTokens) != 530 {
		t.Errorf("input token count = %d, want 530 (server-reported)", len(requests[0].InputTokens))
	}
}

func TestLoadTraceV2Requests_ServerInputTokens_Zero_FallsBackToInputTokens(t *testing.T) {
	// GIVEN a trace record with ServerInputTokens == 0 (generated trace, not observed)
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 0, InputTokens: 256, ServerInputTokens: 0,
				OutputTokens: 32, ArrivalTimeUs: 0, Status: "ok"},
		},
	}
	requests, err := LoadTraceV2Requests(trace, 42)
	if err != nil {
		t.Fatal(err)
	}
	// BC-2: fallback to InputTokens when ServerInputTokens not recorded
	if len(requests[0].InputTokens) != 256 {
		t.Errorf("input token count = %d, want 256 (fallback)", len(requests[0].InputTokens))
	}
}

func TestLoadTraceV2Requests_ServerInputTokens_PrefixGroup_FallsBackToInputTokens(t *testing.T) {
	// GIVEN a prefix-group record with ServerInputTokens > InputTokens.
	// ServerInputTokens includes the prefix length — applying it as suffix count would double-count.
	// WHEN LoadTraceV2Requests constructs the request
	// THEN the suffix uses InputTokens, not ServerInputTokens (prefix prepended separately)
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 0, InputTokens: 100, PrefixGroup: "shared", PrefixLength: 128,
				ServerInputTokens: 246, // = PrefixLength(128) + InputTokens(100) + overhead(18)
				OutputTokens:      32, ArrivalTimeUs: 0, Status: "ok"},
		},
	}
	requests, err := LoadTraceV2Requests(trace, 42)
	if err != nil {
		t.Fatal(err)
	}
	// BC-2: total = PrefixLength(128) + InputTokens(100) = 228, not 128+246=374
	if len(requests[0].InputTokens) != 228 {
		t.Errorf("input token count = %d, want 228 (prefix 128 + suffix 100, not ServerInputTokens 246)",
			len(requests[0].InputTokens))
	}
}

// --- ServerInputTokens session tests (BC-3, BC-4, BC-5) ---

func TestLoadTraceV2SessionBlueprints_ServerInputTokens_Round0(t *testing.T) {
	// GIVEN a 2-round session where round-0 has server overhead tokens
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0,
				InputTokens: 512, ServerInputTokens: 530,
				OutputTokens: 64, ArrivalTimeUs: 0},
			{RequestID: 2, SessionID: "A", RoundIndex: 1,
				InputTokens: 256, ServerInputTokens: 274,
				OutputTokens: 32, ArrivalTimeUs: 5000},
		},
	}
	requests, _, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(requests) != 1 {
		t.Fatalf("expected 1 round-0 request, got %d", len(requests))
	}
	// BC-3: round-0 token count uses ServerInputTokens
	if len(requests[0].InputTokens) != 530 {
		t.Errorf("round-0 input token count = %d, want 530 (server-reported)", len(requests[0].InputTokens))
	}
}

func TestLoadTraceV2SessionBlueprints_ServerInputTokens_Sampler(t *testing.T) {
	// GIVEN a 3-round session with ServerInputTokens on rounds 1 and 2
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0,
				InputTokens: 512, ServerInputTokens: 530, OutputTokens: 64, ArrivalTimeUs: 0},
			{RequestID: 2, SessionID: "A", RoundIndex: 1,
				InputTokens: 256, ServerInputTokens: 274, OutputTokens: 32, ArrivalTimeUs: 5000},
			{RequestID: 3, SessionID: "A", RoundIndex: 2,
				InputTokens: 128, ServerInputTokens: 0, OutputTokens: 16, ArrivalTimeUs: 10000},
		},
	}
	_, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	bp := blueprints[0]
	// Each successive Sample() call returns the next round's token count.
	// BC-4: round-1 sampler returns ServerInputTokens (274 > 256)
	got1 := bp.InputSampler.Sample(nil)
	if got1 != 274 {
		t.Errorf("round-1 sampler value = %d, want 274 (server-reported)", got1)
	}
	// BC-2: round-2 sampler falls back to InputTokens (ServerInputTokens == 0)
	got2 := bp.InputSampler.Sample(nil)
	if got2 != 128 {
		t.Errorf("round-2 sampler value = %d, want 128 (fallback)", got2)
	}
}

func TestLoadTraceV2SessionBlueprints_ServerInputTokens_NonSessionRecord(t *testing.T) {
	// GIVEN a non-session record with ServerInputTokens > InputTokens
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "", InputTokens: 512, ServerInputTokens: 530,
				OutputTokens: 64, ArrivalTimeUs: 0},
		},
	}
	requests, _, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(requests) != 1 {
		t.Fatalf("expected 1 request, got %d", len(requests))
	}
	// BC-5: non-session path uses ServerInputTokens
	if len(requests[0].InputTokens) != 530 {
		t.Errorf("non-session input token count = %d, want 530", len(requests[0].InputTokens))
	}
}

// BC-2 session guard: PrefixGroup records must fall back to InputTokens even when
// ServerInputTokens > 0, to avoid double-counting the prefix prepended by the session
// manager. These tests guard all three application sites in LoadTraceV2SessionBlueprints.

func TestLoadTraceV2SessionBlueprints_ServerInputTokens_Round0_PrefixGroup_FallsBack(t *testing.T) {
	// GIVEN a session with PrefixGroup on round-0 and ServerInputTokens set
	// ServerInputTokens(246) = PrefixLength(128) + InputTokens(100) + overhead(18, illustrative)
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0,
				InputTokens: 100, PrefixGroup: "shared", PrefixLength: 128,
				ServerInputTokens: 246, OutputTokens: 32, ArrivalTimeUs: 0},
			{RequestID: 2, SessionID: "A", RoundIndex: 1,
				InputTokens: 50, PrefixGroup: "shared", PrefixLength: 128,
				ServerInputTokens: 196, OutputTokens: 16, ArrivalTimeUs: 5000},
		},
	}
	requests, _, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(requests) != 1 {
		t.Fatalf("expected 1 round-0 request, got %d", len(requests))
	}
	// BC-2: total = PrefixLength(128) + InputTokens(100) = 228, not 128+246=374
	if len(requests[0].InputTokens) != 228 {
		t.Errorf("round-0 input token count = %d, want 228 (prefix 128 + suffix 100, not ServerInputTokens 246)",
			len(requests[0].InputTokens))
	}
}

func TestLoadTraceV2SessionBlueprints_ServerInputTokens_Sampler_PrefixGroup_FallsBack(t *testing.T) {
	// GIVEN a session where round-1 has PrefixGroup set and ServerInputTokens populated
	// THEN the InputSampler returns InputTokens (fallback), not ServerInputTokens
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0,
				InputTokens: 512, ServerInputTokens: 530, OutputTokens: 64, ArrivalTimeUs: 0},
			{RequestID: 2, SessionID: "A", RoundIndex: 1,
				InputTokens: 50, PrefixGroup: "shared", PrefixLength: 64,
				ServerInputTokens: 132, // = PrefixLength(64) + InputTokens(50) + overhead(18)
				OutputTokens:      16, ArrivalTimeUs: 5000},
		},
	}
	_, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	// first Sample() call returns inputSeq[1] (round-1 token count): prefix-group round → falls back to InputTokens(50)
	got := blueprints[0].InputSampler.Sample(nil)
	if got != 50 {
		t.Errorf("round-1 sampler value = %d, want 50 (fallback: PrefixGroup set, ServerInputTokens ignored)",
			got)
	}
}

func TestLoadTraceV2SessionBlueprints_ServerInputTokens_NonSession_PrefixGroup_FallsBack(t *testing.T) {
	// GIVEN a non-session record in LoadTraceV2SessionBlueprints with PrefixGroup set
	// and ServerInputTokens > InputTokens
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "", InputTokens: 100, PrefixGroup: "shared", PrefixLength: 128,
				ServerInputTokens: 246, OutputTokens: 32, ArrivalTimeUs: 0},
		},
	}
	requests, _, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(requests) != 1 {
		t.Fatalf("expected 1 request, got %d", len(requests))
	}
	// BC-2/BC-5: total = PrefixLength(128) + InputTokens(100) = 228, not 128+246=374
	if len(requests[0].InputTokens) != 228 {
		t.Errorf("non-session input token count = %d, want 228 (prefix 128 + suffix 100, not ServerInputTokens 246)",
			len(requests[0].InputTokens))
	}
}

// TestLoadTraceV2Requests_ModelAndDeadline verifies that Model, Deadline, empty Model,
// and zero Deadline are propagated from TraceRecord to sim.Request
// (2026-03-14-pr653-tracev2-schema-plan.md BC-3–6), and that ServerInputTokens is used
// as the token count when > 0 and PrefixGroup is empty (replay-server-tokens-plan.md BC-1).
func TestLoadTraceV2Requests_ModelAndDeadline(t *testing.T) {
	header := &TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "real"}
	records := []TraceRecord{
		{
			RequestID:         0,
			Model:             "meta-llama/Llama-3.1-8B-Instruct",
			DeadlineUs:        7500000,
			ServerInputTokens: 300, // used as token count for InputTokens generation (> InputTokens: 100)
			InputTokens:       100,
			OutputTokens:      50,
			ArrivalTimeUs:     0,
			Status:            "ok",
		},
		{
			RequestID:         1,
			Model:             "", // BC-6: empty = default model
			DeadlineUs:        0,  // BC-5: no timeout
			ServerInputTokens: 0,
			InputTokens:       50,
			OutputTokens:      25,
			ArrivalTimeUs:     1000,
			Status:            "ok",
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
	// BC-1: ServerInputTokens (300) used as token count for InputTokens generation, not InputTokens (100).
	// sim.Request has no ServerInputTokens field; the value is used only to size the synthetic token slice.
	if len(requests[0].InputTokens) != 300 {
		t.Errorf("request 0 input token count = %d, want 300 (server-reported)", len(requests[0].InputTokens))
	}
	// BC-2: ServerInputTokens == 0 → fallback to InputTokens (50)
	if len(requests[1].InputTokens) != 50 {
		t.Errorf("request 1 input token count = %d, want 50 (fallback)", len(requests[1].InputTokens))
	}
}

// TestInjectionTime_ChoiceLogic verifies the raw send-vs-arrival CHOICE in
// injectionTime (unchanged by #1606). The chosen basis is later re-based onto the
// arrival origin by injectionOriginShift; this test isolates the choice so a
// regression there is caught independently of the normalization.
func TestInjectionTime_ChoiceLogic(t *testing.T) {
	tests := []struct {
		name          string
		sendTimeUs    int64
		arrivalTimeUs int64
		wantBasis     int64
	}{
		{
			name:          "non-zero send_time is the injection basis (slot wait)",
			sendTimeUs:    50000,
			arrivalTimeUs: 0,
			wantBasis:     50000,
		},
		{
			name:          "send_time == arrival_time: basis unchanged",
			sendTimeUs:    100000,
			arrivalTimeUs: 100000,
			wantBasis:     100000,
		},
		{
			name:          "zero send_time falls back to arrival_time",
			sendTimeUs:    0,
			arrivalTimeUs: 200000,
			wantBasis:     200000,
		},
		{
			// Negative send_time (clock corruption) must NOT be used as the
			// basis — the > 0 guard falls back to arrival_time so no negative
			// DES timestamp reaches the sim (INV-3).
			name:          "negative send_time falls back to arrival_time",
			sendTimeUs:    -100,
			arrivalTimeUs: 300000,
			wantBasis:     300000,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := injectionTime(TraceRecord{ArrivalTimeUs: tc.arrivalTimeUs, SendTimeUs: tc.sendTimeUs})
			if got != tc.wantBasis {
				t.Errorf("injectionTime = %d, want %d", got, tc.wantBasis)
			}
		})
	}
}

// TestLoadTraceV2Requests_NormalizesEpochSendOrigin verifies BC-1 and BC-4: an
// epoch-scale send_time_us trace (a real blis observe trace) is injected on the
// arrival origin with send-delta spacing preserved. This is the #1606 fix.
func TestLoadTraceV2Requests_NormalizesEpochSendOrigin(t *testing.T) {
	// Two records. arrival_time_us is run-relative (starts ~0); send_time_us is
	// Unix-epoch µs. rec0 waited 100ms for a concurrency slot, rec1 waited 250ms,
	// so the SEND delta (200000) differs from the ARRIVAL delta (50000).
	const epoch = int64(1_787_274_995_712_218)
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 0, ArrivalTimeUs: 0, SendTimeUs: epoch + 100_000,
				DeadlineUs: 300_000_000, InputTokens: 50, OutputTokens: 25, Status: "ok"},
			{RequestID: 1, ArrivalTimeUs: 50_000, SendTimeUs: epoch + 300_000,
				DeadlineUs: 350_000_000, InputTokens: 50, OutputTokens: 25, Status: "ok"},
		},
	}
	reqs, err := LoadTraceV2Requests(trace, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqs) != 2 {
		t.Fatalf("got %d requests, want 2", len(reqs))
	}
	// BC-1: injection is re-based onto the arrival origin. The earliest injected
	// request lands at min(arrival) == 0, NOT at an epoch-scale tick.
	if reqs[0].ArrivalTime != 0 {
		t.Errorf("BC-1: request 0 ArrivalTime = %d, want 0 (arrival origin, not epoch)", reqs[0].ArrivalTime)
	}
	// BC-1: injection must be far below the deadline (no instant timeout).
	if reqs[0].ArrivalTime >= reqs[0].Deadline {
		t.Errorf("BC-1: request 0 injected at %d >= deadline %d — would instant-timeout (#1606)", reqs[0].ArrivalTime, reqs[0].Deadline)
	}
	// BC-4: spacing follows the SEND delta (200000), not the arrival delta (50000).
	gotDelta := reqs[1].ArrivalTime - reqs[0].ArrivalTime
	if gotDelta != 200_000 {
		t.Errorf("BC-4: injection delta = %d, want 200000 (send delta); arrival delta 50000 would be wrong", gotDelta)
	}
}

// TestLoadTraceV2Requests_GeneratedTraceInjectionUnchanged verifies BC-3
// (INV-13/INV-6): a generated blis run trace (send_time_us == arrival_time_us,
// arrival origin > 0 for a positive rate) injects EXACTLY at arrival_time_us —
// the origin shift is 0, so replay is byte-identical to pre-#1606.
func TestLoadTraceV2Requests_GeneratedTraceInjectionUnchanged(t *testing.T) {
	// blis run --rate 10 produces the first arrival at 99999 (not 0), with
	// send_time_us == arrival_time_us (tracev2.go). See #1606 deviation D1.
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 0, ArrivalTimeUs: 99_999, SendTimeUs: 99_999, InputTokens: 50, OutputTokens: 25, Status: "ok"},
			{RequestID: 1, ArrivalTimeUs: 199_998, SendTimeUs: 199_998, InputTokens: 50, OutputTokens: 25, Status: "ok"},
			{RequestID: 2, ArrivalTimeUs: 299_997, SendTimeUs: 299_997, InputTokens: 50, OutputTokens: 25, Status: "ok"},
		},
	}
	reqs, err := LoadTraceV2Requests(trace, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := []int64{99_999, 199_998, 299_997}
	for i, req := range reqs {
		if req.ArrivalTime != want[i] {
			t.Errorf("BC-3 (INV-13): request %d ArrivalTime = %d, want %d (shift must be 0 for generated traces)", i, req.ArrivalTime, want[i])
		}
	}
}

// TestInjectionOriginShift_AllFallback_IsZero verifies BC-5: a trace whose every
// record falls back to arrival_time_us (send_time_us <= 0) has a zero origin
// shift, so injection == arrival and no negative DES tick is produced (INV-3).
func TestInjectionOriginShift_AllFallback_IsZero(t *testing.T) {
	records := []TraceRecord{
		{RequestID: 0, ArrivalTimeUs: 10_000, SendTimeUs: 0},
		{RequestID: 1, ArrivalTimeUs: 20_000, SendTimeUs: -100},
	}
	if got := injectionOriginShift(records); got != 0 {
		t.Errorf("injectionOriginShift = %d, want 0 (all records fall back to arrival)", got)
	}
	// Empty record set → 0 (defensive).
	if got := injectionOriginShift(nil); got != 0 {
		t.Errorf("injectionOriginShift(nil) = %d, want 0", got)
	}
}

// TestMaxNormalizedInjectionTimeUs verifies BC-6: the preliminary closed-loop
// horizon basis is the max NORMALIZED injection over injected records (round-0 +
// non-session), on the arrival origin — not the raw arrival_time_us column.
func TestMaxNormalizedInjectionTimeUs(t *testing.T) {
	const epoch = int64(1_787_274_995_712_218)
	trace := &TraceV2{
		Records: []TraceRecord{
			// Session round-0 (injected initially).
			{RequestID: 0, SessionID: "A", RoundIndex: 0, ArrivalTimeUs: 0, SendTimeUs: epoch},
			// Session follow-up (NOT injected initially — must be skipped).
			{RequestID: 1, SessionID: "A", RoundIndex: 1, ArrivalTimeUs: 500_000, SendTimeUs: epoch + 9_000_000},
			// Non-session (injected initially); latest send → sets the max.
			{RequestID: 2, ArrivalTimeUs: 300_000, SendTimeUs: epoch + 700_000},
		},
	}
	// shift = min(injectionTime) - min(arrival) = epoch - 0 = epoch.
	// Normalized injected: r0 → 0, r2 → 700000. Follow-up r1 skipped.
	if got := MaxNormalizedInjectionTimeUs(trace); got != 700_000 {
		t.Errorf("MaxNormalizedInjectionTimeUs = %d, want 700000 (normalized max over injected records)", got)
	}
	if got := MaxNormalizedInjectionTimeUs(nil); got != 0 {
		t.Errorf("MaxNormalizedInjectionTimeUs(nil) = %d, want 0", got)
	}
}

// TestLoadTraceV2SessionBlueprints_ConcurrencyModeInjection verifies that the
// round-0 and non-session injection sites use the send-based basis re-based onto
// the arrival origin (#1606, BC-1/BC-4), and that think-time still comes from
// arrival deltas.
func TestLoadTraceV2SessionBlueprints_ConcurrencyModeInjection(t *testing.T) {
	// injectionTimes: A/r0 → 50000, A/r1 → 230000, non-session → 40000.
	// min(injectionTime) = 40000, min(arrival) = 0 → origin shift = 40000.
	// Normalized injection: A/r0 → 10000, non-session → 0.
	trace := &TraceV2{
		Records: []TraceRecord{
			// Session A: round-0 delayed by concurrency slot wait (50ms)
			{RequestID: 1, SessionID: "A", RoundIndex: 0,
				ArrivalTimeUs: 0, SendTimeUs: 50000,
				InputTokens: 100, OutputTokens: 50},
			// Session A: round-1 delayed by concurrency slot wait (30ms)
			{RequestID: 2, SessionID: "A", RoundIndex: 1,
				ArrivalTimeUs: 200000, SendTimeUs: 230000,
				InputTokens: 150, OutputTokens: 60},
			// Non-session record with send_time > arrival_time (earliest send → origin)
			{RequestID: 3, SessionID: "",
				ArrivalTimeUs: 10000, SendTimeUs: 40000,
				InputTokens: 80, OutputTokens: 30},
		},
	}

	requests, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Find the session round-0 request and the non-session request by ArrivalTime
	sessionArrival := int64(-1)
	nonSessionArrival := int64(-1)
	for _, r := range requests {
		switch r.SessionID {
		case "A":
			sessionArrival = r.ArrivalTime
		case "":
			nonSessionArrival = r.ArrivalTime
		}
	}
	if sessionArrival == -1 {
		t.Fatal("session round-0 request not found")
	}
	if nonSessionArrival == -1 {
		t.Fatal("non-session request not found")
	}

	// BC-1/BC-4: session round-0 injected at send basis re-based to arrival origin
	// (send 50000 - shift 40000 = 10000). The send-delta between the two injected
	// records (50000 - 40000 = 10000) is preserved, not the arrival delta.
	if sessionArrival != 10000 {
		t.Errorf("session round-0 ArrivalTime = %d, want 10000 (send 50000 - origin shift 40000)", sessionArrival)
	}

	// BC-4: earliest-sent (non-session) request lands at the arrival origin (0).
	if nonSessionArrival != 0 {
		t.Errorf("non-session ArrivalTime = %d, want 0 (earliest send at arrival origin)", nonSessionArrival)
	}

	// Think-time gap derived from ArrivalTimeUs differences (200000 - 0 = 200000),
	// origin-invariant and unaffected by the injection normalization.
	if len(blueprints) != 1 {
		t.Fatalf("expected 1 blueprint, got %d", len(blueprints))
	}
	bp := blueprints[0]
	if bp.ThinkTimeSampler == nil {
		t.Fatal("expected ThinkTimeSampler for multi-round session")
	}
	gotThinkTime := bp.ThinkTimeSampler.Sample(nil)
	// ArrivalTimeUs gap: 200000 - 0 = 200000.
	// SendTimeUs gap:    230000 - 50000 = 180000.
	// Asserting == 200000 AND != 180000 makes the law explicit: think-time MUST
	// come from ArrivalTimeUs deltas, not SendTimeUs deltas.
	if gotThinkTime != 200000 {
		t.Errorf("think time = %d, want 200000 (ArrivalTimeUs gap); if 180000, SendTimeUs gap was used instead", gotThinkTime)
	}
	if gotThinkTime == 180000 {
		t.Error("think time == 180000 (SendTimeUs gap) — must use ArrivalTimeUs gap instead")
	}
}

// TestLoadTraceV2SessionBlueprints_NegativeSendTime verifies that negative SendTimeUs
// (clock corruption) falls back to ArrivalTimeUs for both the session round-0 and
// non-session call sites in LoadTraceV2SessionBlueprints (INV-3 guard, mirroring the
// negative case in TestLoadTraceV2Requests_ConcurrencyModeUseSendTime).
func TestLoadTraceV2SessionBlueprints_NegativeSendTime(t *testing.T) {
	trace := &TraceV2{
		Records: []TraceRecord{
			// Session round-0: negative SendTimeUs must not become injection time.
			{RequestID: 1, SessionID: "A", RoundIndex: 0,
				ArrivalTimeUs: 10000, SendTimeUs: -500,
				InputTokens: 50, OutputTokens: 25},
			// Non-session: same guard on the independent call site.
			{RequestID: 2, SessionID: "",
				ArrivalTimeUs: 20000, SendTimeUs: -100,
				InputTokens: 30, OutputTokens: 15},
		},
	}

	requests, _, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	sessionArrival := int64(-1)
	nonSessionArrival := int64(-1)
	for _, r := range requests {
		switch r.SessionID {
		case "A":
			sessionArrival = r.ArrivalTime
		case "":
			nonSessionArrival = r.ArrivalTime
		}
	}

	// Session round-0: must use ArrivalTimeUs (10000), not SendTimeUs (-500).
	if sessionArrival != 10000 {
		t.Errorf("session round-0: ArrivalTime = %d, want 10000 (negative send_time must fall back)", sessionArrival)
	}
	// Non-session: must use ArrivalTimeUs (20000), not SendTimeUs (-100).
	if nonSessionArrival != 20000 {
		t.Errorf("non-session: ArrivalTime = %d, want 20000 (negative send_time must fall back)", nonSessionArrival)
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
		Header: TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"},
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
		Header: TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated"}, // no growth field
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
		Header: TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"},
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

// TestLoadTraceV2SessionBlueprints_AllZeroRecordedThink_FallsBackToGap pins the
// F1 lossy-sentinel behavior (#1484 review): a session whose every round records
// think_time_us == 0 (real for Weka's overlap-clamped rounds) is indistinguishable
// from an absent column, so it falls back to arrival-gap derivation.
func TestLoadTraceV2SessionBlueprints_AllZeroRecordedThink_FallsBackToGap(t *testing.T) {
	trace := &TraceV2{Records: []TraceRecord{
		{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0, ThinkTimeUs: 0},
		{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 60, OutputTokens: 40, ArrivalTimeUs: 5000, ThinkTimeUs: 0},
		{RequestID: 3, SessionID: "A", RoundIndex: 2, InputTokens: 70, OutputTokens: 30, ArrivalTimeUs: 12000, ThinkTimeUs: 0},
	}}
	_, bps, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	s := bps[0].ThinkTimeSampler
	if s == nil {
		t.Fatal("expected a think sampler for a multi-round session")
	}
	// Falls back to arrival gaps [5000, 7000], NOT the recorded zeros.
	if g1, g2 := s.Sample(nil), s.Sample(nil); g1 != 5000 || g2 != 7000 {
		t.Errorf("all-zero recorded think: sampler = [%d, %d], want arrival gaps [5000, 7000] (documented lossy-sentinel fallback)", g1, g2)
	}
}

// TestLoadTraceV2SessionBlueprints_MixedThinkSessions covers F4 (#1484 review): the
// export gate (includeThinkTime) is global, but think-time selection is per-session,
// so a trace mixing a recorded-think session with an all-zero one must select each
// independently — A uses its recorded think, B falls back to arrival gaps.
func TestLoadTraceV2SessionBlueprints_MixedThinkSessions(t *testing.T) {
	trace := &TraceV2{Records: []TraceRecord{
		// A: recorded think 300; arrival gap (9999) deliberately differs so the two are distinguishable.
		{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0, ThinkTimeUs: 0},
		{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 60, OutputTokens: 40, ArrivalTimeUs: 9999, ThinkTimeUs: 300},
		// B: all-zero recorded think; arrival gap 5000.
		{RequestID: 4, SessionID: "B", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0, ThinkTimeUs: 0},
		{RequestID: 5, SessionID: "B", RoundIndex: 1, InputTokens: 60, OutputTokens: 40, ArrivalTimeUs: 5000, ThinkTimeUs: 0},
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
		t.Errorf("session B think = %d, want arrival gap 5000 (all-zero fallback)", got)
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
			Header: TraceHeader{Version: 3, TimeUnit: "microseconds", Mode: "generated", SessionContextGrowth: "accumulate"},
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
