package workload

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestTraceRecordsToRequests_BasicConversion(t *testing.T) {
	// GIVEN a TraceRecord with status="ok", complete timing, and multiple output tokens
	// WHEN converting to Request
	// THEN TTFTSet=true, State=StateCompleted, ITL length = OutputTokens-1, sum(ITL) <= total decode time
	records := []TraceRecord{
		{
			ArrivalTimeUs:    1000,
			FirstChunkTimeUs: 1050, // TTFT = 50 µs
			LastChunkTimeUs:  1250, // Total decode = 200 µs
			OutputTokens:     5,    // 5 tokens → 4 ITL intervals
			Status:           "ok",
		},
	}

	requests := TraceRecordsToRequests(records)

	if len(requests) != 1 {
		t.Fatalf("Expected 1 request, got %d", len(requests))
	}

	req := requests[0]

	// Verify basic fields
	if req.ArrivalTime != 1000 {
		t.Errorf("ArrivalTime=%d, want 1000", req.ArrivalTime)
	}
	if len(req.OutputTokens) != 1 || req.OutputTokens[0] != 5 {
		t.Errorf("OutputTokens=%v, want [5]", req.OutputTokens)
	}

	// Verify TTFT
	if !req.TTFTSet {
		t.Error("TTFTSet=false, want true")
	}
	if req.FirstTokenTime != 50 {
		t.Errorf("FirstTokenTime=%d, want 50", req.FirstTokenTime)
	}

	// Verify state
	if req.State != sim.StateCompleted {
		t.Errorf("State=%v, want StateCompleted", req.State)
	}

	// Verify ITL
	if len(req.ITL) != 4 {
		t.Errorf("len(ITL)=%d, want 4 (OutputTokens-1)", len(req.ITL))
	}

	// Sum of ITL should be <= total decode time (truncation allowed)
	totalDecodeUs := int64(200) // 1250 - 1050
	sumITL := int64(0)
	for _, itl := range req.ITL {
		sumITL += itl
	}
	if sumITL > totalDecodeUs {
		t.Errorf("sum(ITL)=%d > totalDecodeUs=%d", sumITL, totalDecodeUs)
	}

	// Verify ITL values are uniform approximation
	expectedAvgITL := totalDecodeUs / 4 // 200 / 4 = 50
	for i, itl := range req.ITL {
		if itl != expectedAvgITL {
			t.Errorf("ITL[%d]=%d, want %d (uniform approximation)", i, itl, expectedAvgITL)
		}
	}
}

func TestTraceRecordsToRequests_TimeoutStatus(t *testing.T) {
	// GIVEN a TraceRecord with status="timeout" and FirstChunkTimeUs=0
	// WHEN converting to Request
	// THEN TTFTSet=false, State=StateTimedOut
	records := []TraceRecord{
		{
			ArrivalTimeUs:    1000,
			FirstChunkTimeUs: 0, // Timed out before first token
			LastChunkTimeUs:  2000,
			OutputTokens:     0,
			Status:           "timeout",
		},
	}

	requests := TraceRecordsToRequests(records)

	if len(requests) != 1 {
		t.Fatalf("Expected 1 request, got %d", len(requests))
	}

	req := requests[0]

	if req.TTFTSet {
		t.Error("TTFTSet=true, want false (timed out before TTFT)")
	}
	if req.State != sim.StateTimedOut {
		t.Errorf("State=%v, want StateTimedOut", req.State)
	}
}

func TestTraceRecordsToRequests_SingleTokenResponse(t *testing.T) {
	// GIVEN a TraceRecord with status="ok" and OutputTokens=1
	// WHEN converting to Request
	// THEN ITL slice is empty (no inter-token intervals for single token)
	records := []TraceRecord{
		{
			ArrivalTimeUs:    1000,
			FirstChunkTimeUs: 1050,
			LastChunkTimeUs:  1050,
			OutputTokens:     1,
			Status:           "ok",
		},
	}

	requests := TraceRecordsToRequests(records)

	if len(requests) != 1 {
		t.Fatalf("Expected 1 request, got %d", len(requests))
	}

	req := requests[0]

	if req.TTFTSet != true {
		t.Error("TTFTSet=false, want true")
	}
	if len(req.ITL) != 0 {
		t.Errorf("len(ITL)=%d, want 0 (single token, no intervals)", len(req.ITL))
	}
}

func TestTraceRecordsToRequests_IncompleteStatus(t *testing.T) {
	// GIVEN a TraceRecord with status="incomplete"
	// WHEN converting to Request
	// THEN State=StateRunning
	records := []TraceRecord{
		{
			ArrivalTimeUs:    1000,
			FirstChunkTimeUs: 1050,
			LastChunkTimeUs:  1100,
			OutputTokens:     2,
			Status:           "incomplete",
		},
	}

	requests := TraceRecordsToRequests(records)

	if len(requests) != 1 {
		t.Fatalf("Expected 1 request, got %d", len(requests))
	}

	req := requests[0]

	if req.State != sim.StateRunning {
		t.Errorf("State=%v, want StateRunning", req.State)
	}
}

func TestTraceRecordsToRequests_EmptyInput(t *testing.T) {
	// GIVEN empty TraceRecord slice
	// WHEN converting to Request
	// THEN returns empty slice
	requests := TraceRecordsToRequests([]TraceRecord{})

	if len(requests) != 0 {
		t.Errorf("Expected empty slice, got %d requests", len(requests))
	}

	// Test nil input
	requests = TraceRecordsToRequests(nil)
	if len(requests) != 0 {
		t.Errorf("Expected empty slice for nil input, got %d requests", len(requests))
	}
}

func TestTraceRecordsToRequests_ErrorStatus(t *testing.T) {
	// GIVEN a TraceRecord with status="error"
	// WHEN converting to Request
	// THEN State=StateTimedOut (errors treated as timeouts for saturation analysis)
	records := []TraceRecord{
		{
			ArrivalTimeUs:    1000,
			FirstChunkTimeUs: 0,
			LastChunkTimeUs:  2000,
			OutputTokens:     0,
			Status:           "error",
		},
	}

	requests := TraceRecordsToRequests(records)

	if len(requests) != 1 {
		t.Fatalf("Expected 1 request, got %d", len(requests))
	}

	req := requests[0]

	if req.State != sim.StateTimedOut {
		t.Errorf("State=%v, want StateTimedOut", req.State)
	}
}
