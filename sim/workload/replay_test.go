package workload

import (
	"path/filepath"
	"testing"
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
		name             string
		inputTokens      int
		serverTokens     int
		prefixGroup      string
		want             int
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
				OutputTokens: 32, ArrivalTimeUs: 0, Status: "ok"},
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
				OutputTokens: 16, ArrivalTimeUs: 5000},
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
			Model:             "",  // BC-6: empty = default model
			DeadlineUs:        0,   // BC-5: no timeout
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

// TestLoadTraceV2Requests_ConcurrencyModeUseSendTime verifies BC-1 and BC-2.
func TestLoadTraceV2Requests_ConcurrencyModeUseSendTime(t *testing.T) {
	tests := []struct {
		name          string
		sendTimeUs    int64
		arrivalTimeUs int64
		wantArrival   int64
	}{
		{
			name:          "non-zero send_time overrides arrival_time",
			sendTimeUs:    50000, // observed trace: slot wait caused send_time > arrival_time
			arrivalTimeUs: 0,
			wantArrival:   50000,
		},
		{
			name:          "send_time equals arrival_time: returns send_time unchanged",
			sendTimeUs:    100000,
			arrivalTimeUs: 100000,
			wantArrival:   100000,
		},
		{
			name:          "zero send_time falls back to arrival_time",
			sendTimeUs:    0,
			arrivalTimeUs: 200000,
			wantArrival:   200000,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			trace := &TraceV2{
				Records: []TraceRecord{
					{
						RequestID:     0,
						ArrivalTimeUs: tc.arrivalTimeUs,
						SendTimeUs:    tc.sendTimeUs,
						InputTokens:   50,
						OutputTokens:  25,
						Status:        "ok",
					},
				},
			}
			reqs, err := LoadTraceV2Requests(trace, 42)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(reqs) != 1 {
				t.Fatalf("got %d requests, want 1", len(reqs))
			}
			if reqs[0].ArrivalTime != tc.wantArrival {
				t.Errorf("ArrivalTime = %d, want %d", reqs[0].ArrivalTime, tc.wantArrival)
			}
		})
	}
}

// TestLoadTraceV2SessionBlueprints_ConcurrencyModeInjection verifies BC-3, BC-4, BC-5.
func TestLoadTraceV2SessionBlueprints_ConcurrencyModeInjection(t *testing.T) {
	// Session round-0 with send_time > arrival_time (BC-3)
	// Non-session record with send_time > arrival_time (BC-4)
	// Think-time gap still uses arrival_time_us deltas (BC-5)
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
			// Non-session record with send_time > arrival_time (BC-4)
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

	// BC-3: session round-0 uses send_time
	if sessionArrival != 50000 {
		t.Errorf("BC-3: session round-0 ArrivalTime = %d, want 50000 (send_time_us)", sessionArrival)
	}

	// BC-4: non-session request uses send_time
	if nonSessionArrival != 40000 {
		t.Errorf("BC-4: non-session ArrivalTime = %d, want 40000 (send_time_us)", nonSessionArrival)
	}

	// BC-5: think-time gap derived from ArrivalTimeUs differences (200000 - 0 = 200000)
	if len(blueprints) != 1 {
		t.Fatalf("expected 1 blueprint, got %d", len(blueprints))
	}
	bp := blueprints[0]
	if bp.ThinkTimeSampler == nil {
		t.Fatal("expected ThinkTimeSampler for multi-round session")
	}
	gotThinkTime := bp.ThinkTimeSampler.Sample(nil)
	if gotThinkTime != 200000 {
		t.Errorf("BC-5: think time = %d, want 200000 (from ArrivalTimeUs gap, not SendTimeUs gap)", gotThinkTime)
	}
}
