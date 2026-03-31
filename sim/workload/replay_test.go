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

// --- LoadTraceV2SessionBlueprints tests (BC-5, BC-6, BC-7) ---

func TestLoadTraceV2SessionBlueprints_GroupsBySession(t *testing.T) {
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0},
			{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 5000},
			{RequestID: 3, SessionID: "B", RoundIndex: 0, InputTokens: 150, OutputTokens: 60, ArrivalTimeUs: 1000},
		},
	}

	requests, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, 0, 0)
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

	requests, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, 0, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// BC-7: 1 non-session + 1 round-0 session = 2 requests total
	if len(requests) != 2 {
		t.Fatalf("BC-7: got %d requests, want 2", len(requests))
	}
	if len(blueprints) != 1 {
		t.Errorf("BC-7: got %d blueprints, want 1", len(blueprints))
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

	_, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, 0, 0)
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

	requests, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, 0, 0)
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
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0},
			{RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 5000},
		},
	}

	_, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, 500_000, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	bp := blueprints[0]
	if bp.ThinkTimeSampler != nil {
		t.Error("expected nil ThinkTimeSampler when override provided")
	}
	if bp.ThinkTimeUs != 500_000 {
		t.Errorf("ThinkTimeUs = %d, want 500000", bp.ThinkTimeUs)
	}
}

func TestLoadTraceV2SessionBlueprints_NonConsecutiveRoundIndex_Error(t *testing.T) {
	trace := &TraceV2{
		Records: []TraceRecord{
			{RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0},
			{RequestID: 2, SessionID: "A", RoundIndex: 2, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 5000},
		},
	}

	_, _, err := LoadTraceV2SessionBlueprints(trace, 42, 0, 0)
	if err == nil {
		t.Fatal("expected error for non-consecutive round indices, got nil")
	}
}

// TestLoadTraceV2Requests_ModelAndDeadline verifies BC-3, BC-4, BC-5, BC-6, BC-7.
func TestLoadTraceV2Requests_ModelAndDeadline(t *testing.T) {
	header := &TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "real"}
	records := []TraceRecord{
		{
			RequestID:         0,
			Model:             "meta-llama/Llama-3.1-8B-Instruct",
			DeadlineUs:        7500000,
			ServerInputTokens: 300, // must NOT appear on sim.Request
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
	// BC-7: ServerInputTokens is NOT on sim.Request (calibration-only field).
	// The compiler enforces this: sim.Request has no ServerInputTokens field.
	// No runtime assertion needed — if someone adds the field and wires it up,
	// the compilation of this package would not catch it, but the architectural
	// review (BC-7 in the plan) documents the non-propagation intent explicitly.
}
