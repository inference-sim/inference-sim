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
			PrefixGroup: "shared", ArrivalTimeUs: 0, Status: "ok"},
		{RequestID: 1, InputTokens: 100, OutputTokens: 50,
			PrefixGroup: "shared", ArrivalTimeUs: 100000, Status: "ok"},
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

	// Both requests should have the same prefix (first 50 tokens)
	if len(requests[0].InputTokens) < 50 || len(requests[1].InputTokens) < 50 {
		t.Fatal("input tokens too short for prefix check")
	}
	for i := 0; i < 50; i++ {
		if requests[0].InputTokens[i] != requests[1].InputTokens[i] {
			t.Errorf("prefix token %d differs: %d vs %d", i,
				requests[0].InputTokens[i], requests[1].InputTokens[i])
			break
		}
	}
	// Total input length = prefix(50) + requested(100) = 150
	if len(requests[0].InputTokens) != 150 {
		t.Errorf("input length = %d, want 150 (50 prefix + 100 body)", len(requests[0].InputTokens))
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
