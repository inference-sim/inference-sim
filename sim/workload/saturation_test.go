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
