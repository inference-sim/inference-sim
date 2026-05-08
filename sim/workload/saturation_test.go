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

func TestComputeWindowMetrics_Identity_DeltaBacklogEqualsEnterMinusLeft(t *testing.T) {
	// GIVEN intervals spanning multiple windows
	// WHEN computing per-window metrics
	// THEN delta_backlog = num_entered - num_left (BC-1 identity)
	intervals := []RequestInterval{
		{ArrivalUs: 10_000, CompletionUs: 70_000},   // Enters window 0, leaves window 1
		{ArrivalUs: 50_000, CompletionUs: 150_000},  // Enters window 0, leaves window 2
		{ArrivalUs: 90_000, CompletionUs: 130_000},  // Enters window 1, leaves window 2
	}
	windowSizeUs := int64(60_000_000) // 60 seconds in µs
	totalDurationUs := int64(200_000)

	windows := computeWindowMetrics(intervals, windowSizeUs, totalDurationUs)

	for i, w := range windows {
		identity := w.NumEntered - w.NumLeft
		if w.DeltaBacklog != identity {
			t.Errorf("Window %d: DeltaBacklog=%d, but NumEntered-NumLeft=%d (identity violation)",
				i, w.DeltaBacklog, identity)
		}
	}
}

func TestComputeWindowMetrics_ActiveCount_AtBoundaries(t *testing.T) {
	// GIVEN intervals with known active counts at boundaries
	// WHEN computing window metrics
	// THEN ActiveStart and ActiveEnd reflect interval containment
	intervals := []RequestInterval{
		{ArrivalUs: 10, CompletionUs: 50}, // Active during [10, 50)
		{ArrivalUs: 20, CompletionUs: 80}, // Active during [20, 80)
	}
	windowSizeUs := int64(30) // Window [0, 30), [30, 60), ...
	totalDurationUs := int64(100)

	windows := computeWindowMetrics(intervals, windowSizeUs, totalDurationUs)

	// Window 0: [0, 30)
	// ActiveStart(0): interval 1 starts at 10 (not yet), interval 2 starts at 20 (not yet) → 0
	// ActiveEnd(30): interval 1 [10, 50) contains 30 → yes, interval 2 [20, 80) contains 30 → yes → 2
	if windows[0].ActiveStart != 0 || windows[0].ActiveEnd != 2 {
		t.Errorf("Window 0: ActiveStart=%d, ActiveEnd=%d (expected 0, 2)", windows[0].ActiveStart, windows[0].ActiveEnd)
	}

	// Window 1: [30, 60)
	// ActiveStart(30): 2 active (from above)
	// ActiveEnd(60): interval 1 [10, 50) does NOT contain 60 → no, interval 2 [20, 80) contains 60 → yes → 1
	if windows[1].ActiveStart != 2 || windows[1].ActiveEnd != 1 {
		t.Errorf("Window 1: ActiveStart=%d, ActiveEnd=%d (expected 2, 1)", windows[1].ActiveStart, windows[1].ActiveEnd)
	}
}
