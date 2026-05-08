// sim/workload/saturation_test.go
package workload

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	sim "github.com/inference-sim/inference-sim/sim"
)

func TestBacklogDriftConfig_Validation_ZeroWindow(t *testing.T) {
	// GIVEN window size <= 0
	// WHEN constructing config
	// THEN panics with descriptive error
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for zero window size")
		} else if !strings.Contains(fmt.Sprint(r), "WindowSize must be > 0") {
			t.Fatalf("Wrong panic message: %v", r)
		}
	}()
	_ = NewBacklogDriftConfig(0, 5, 2.0, 0.95)
}

func TestBacklogDriftConfig_Validation_NegativeMinWindows(t *testing.T) {
	// GIVEN MinWindows <= 0
	// WHEN constructing config
	// THEN panics
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for negative MinWindows")
		}
	}()
	_ = NewBacklogDriftConfig(60*time.Second, 0, 2.0, 0.95)
}

func TestBacklogDriftConfig_Validation_NaNPeakRatio(t *testing.T) {
	// GIVEN PeakRatio is NaN
	// WHEN constructing config
	// THEN panics
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for NaN PeakRatio")
		}
	}()
	_ = NewBacklogDriftConfig(60*time.Second, 5, math.NaN(), 0.95)
}

func TestBacklogDriftConfig_Validation_CIOutOfRange(t *testing.T) {
	// GIVEN ConfidenceCI not in (0, 1)
	// WHEN constructing config
	// THEN panics
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for CI=1.5")
		}
	}()
	_ = NewBacklogDriftConfig(60*time.Second, 5, 2.0, 1.5)
}

func TestBacklogDriftConfig_Validation_ValidConfig(t *testing.T) {
	// GIVEN all parameters valid
	// WHEN constructing config
	// THEN succeeds without panic
	cfg := NewBacklogDriftConfig(60*time.Second, 5, 2.0, 0.95)
	if cfg.WindowSize != 60*time.Second {
		t.Errorf("WindowSize mismatch: got %v", cfg.WindowSize)
	}
}

func TestRequestsToIntervals_Eligibility_ThreeCases(t *testing.T) {
	// GIVEN requests with different completion states
	// WHEN building intervals
	// THEN timed-out excluded, horizon-truncated use simEnd, completed use computed time
	simEndUs := int64(1000000)
	requests := []*sim.Request{
		// Case 1: Completed (TTFTSet=true)
		{ArrivalTime: 100, FirstTokenTime: 200, ITL: []int64{50, 50}, TTFTSet: true, State: sim.StateCompleted},
		// Case 2: Timed out before TTFT (TTFTSet=false, State=Timeout) — EXCLUDED
		{ArrivalTime: 200, TTFTSet: false, State: sim.StateTimedOut},
		// Case 3: Horizon-truncated (TTFTSet=false, State=Running) — use simEndUs
		{ArrivalTime: 300, TTFTSet: false, State: sim.StateRunning},
	}

	intervals := RequestsToIntervals(requests, simEndUs)

	// Expected: 2 intervals (case 1 and case 3; case 2 excluded)
	if len(intervals) != 2 {
		t.Fatalf("Expected 2 intervals, got %d", len(intervals))
	}

	// Case 1: arrival=100, completion=100+200+50+50=400
	if intervals[0].ArrivalUs != 100 || intervals[0].CompletionUs != 400 {
		t.Errorf("Case 1 mismatch: got (%d, %d)", intervals[0].ArrivalUs, intervals[0].CompletionUs)
	}

	// Case 3: arrival=300, completion=simEndUs=1000000
	if intervals[1].ArrivalUs != 300 || intervals[1].CompletionUs != simEndUs {
		t.Errorf("Case 3 mismatch: got (%d, %d)", intervals[1].ArrivalUs, intervals[1].CompletionUs)
	}
}

func TestRequestsToIntervals_EmptyInput_ReturnsEmpty(t *testing.T) {
	// GIVEN empty request slice (BC-15)
	// WHEN building intervals
	// THEN returns empty slice without panic
	intervals := RequestsToIntervals(nil, 1000000)
	if len(intervals) != 0 {
		t.Errorf("Expected empty result for nil input, got %d", len(intervals))
	}

	intervals = RequestsToIntervals([]*sim.Request{}, 1000000)
	if len(intervals) != 0 {
		t.Errorf("Expected empty result for empty input, got %d", len(intervals))
	}
}
