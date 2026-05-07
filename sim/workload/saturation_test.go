package workload

import (
	"testing"
)

func TestComputeActiveRequests_SingleRequest(t *testing.T) {
	// GIVEN one request arriving at t=1000, completing at t=2000
	records := []TraceRecord{
		{
			RequestID:       0,
			ArrivalTimeUs:   1000,
			SendTimeUs:      1000,
			LastChunkTimeUs: 2000,
		},
	}

	// WHEN computing active requests at various timestamps
	samples := []int64{500, 1000, 1500, 2000, 2500}
	actives := computeActiveRequests(records, samples)

	// THEN active count is 1 only in the interval [1000, 2000)
	expected := []int{0, 1, 1, 0, 0}
	for i, sample := range samples {
		if actives[i] != expected[i] {
			t.Errorf("active_requests(%d) = %d, want %d", sample, actives[i], expected[i])
		}
	}
}

func TestComputeActiveRequests_OverlappingRequests(t *testing.T) {
	// GIVEN three overlapping requests
	records := []TraceRecord{
		{RequestID: 0, ArrivalTimeUs: 1000, SendTimeUs: 1000, LastChunkTimeUs: 3000}, // active [1000, 3000)
		{RequestID: 1, ArrivalTimeUs: 2000, SendTimeUs: 2000, LastChunkTimeUs: 4000}, // active [2000, 4000)
		{RequestID: 2, ArrivalTimeUs: 2500, SendTimeUs: 2500, LastChunkTimeUs: 3500}, // active [2500, 3500)
	}

	// WHEN sampling at t=2500 (all three active)
	samples := []int64{2500}
	actives := computeActiveRequests(records, samples)

	// THEN active count is 3
	if actives[0] != 3 {
		t.Errorf("active_requests(2500) = %d, want 3", actives[0])
	}
}

func TestComputeWindowMetrics_SingleWindow(t *testing.T) {
	// GIVEN 3 requests within a 60-second observation window
	// Observation spans from first arrival to last completion
	records := []TraceRecord{
		{RequestID: 0, ArrivalTimeUs: 5_000_000, SendTimeUs: 5_000_000, LastChunkTimeUs: 25_000_000},   // enter@5s, leave@25s
		{RequestID: 1, ArrivalTimeUs: 10_000_000, SendTimeUs: 10_000_000, LastChunkTimeUs: 40_000_000}, // enter@10s, leave@40s
		{RequestID: 2, ArrivalTimeUs: 15_000_000, SendTimeUs: 15_000_000, LastChunkTimeUs: 55_000_000}, // enter@15s, leave@55s
	}
	windowDurationUs := int64(60_000_000) // 60 seconds

	// WHEN computing window metrics
	windows := computeWindowMetrics(records, windowDurationUs)

	// THEN first (and only) window should show: 3 entered, 3 left, drain_ratio = 1.0
	if len(windows) != 1 {
		t.Fatalf("expected 1 window, got %d", len(windows))
	}
	w := windows[0]
	if w.NumEntered != 3 {
		t.Errorf("window.NumEntered = %d, want 3", w.NumEntered)
	}
	if w.NumLeft != 3 {
		t.Errorf("window.NumLeft = %d, want 3", w.NumLeft)
	}
	expectedRatio := 1.0
	if w.DrainRatio < expectedRatio-0.01 || w.DrainRatio > expectedRatio+0.01 {
		t.Errorf("window.DrainRatio = %.2f, want %.2f", w.DrainRatio, expectedRatio)
	}
}
