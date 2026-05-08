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

func TestFitSlopeRegression_PositiveSlope(t *testing.T) {
	// GIVEN active request counts increasing over time
	// WHEN fitting regression
	// THEN slope is positive
	samples := []struct {
		timeUs int64
		count  int
	}{
		{0, 10},
		{1000, 20},
		{2000, 30},
		{3000, 40},
	}

	slope, lower, upper := fitSlopeRegression(samples, 3000, 0.95)

	// Slope should be positive (requests increasing)
	if slope <= 0 {
		t.Errorf("Expected positive slope, got %f", slope)
	}

	// CI should exclude zero (statistically significant)
	if lower <= 0 {
		t.Errorf("Expected CI lower bound > 0, got %f", lower)
	}

	// Bounds should be ordered
	if !(lower < slope && slope < upper) {
		t.Errorf("Expected lower < slope < upper, got %f < %f < %f", lower, slope, upper)
	}
}

func TestFitSlopeRegression_NegativeSlope(t *testing.T) {
	// GIVEN active request counts decreasing over time
	// WHEN fitting regression
	// THEN slope is negative
	samples := []struct {
		timeUs int64
		count  int
	}{
		{0, 40},
		{1000, 30},
		{2000, 20},
		{3000, 10},
	}

	slope, lower, upper := fitSlopeRegression(samples, 3000, 0.95)

	// Slope should be negative (requests decreasing)
	if slope >= 0 {
		t.Errorf("Expected negative slope, got %f", slope)
	}

	// CI should exclude zero (statistically significant)
	if upper >= 0 {
		t.Errorf("Expected CI upper bound < 0, got %f", upper)
	}

	// Bounds should be ordered
	if !(lower < slope && slope < upper) {
		t.Errorf("Expected lower < slope < upper, got %f < %f < %f", lower, slope, upper)
	}
}

func TestFitSlopeRegression_FlatLine(t *testing.T) {
	// GIVEN constant active request count
	// WHEN fitting regression
	// THEN slope is near zero and CI includes zero
	samples := []struct {
		timeUs int64
		count  int
	}{
		{0, 25},
		{1000, 25},
		{2000, 25},
		{3000, 25},
	}

	slope, lower, upper := fitSlopeRegression(samples, 3000, 0.95)

	// Slope should be near zero
	if math.Abs(slope) > 1e-10 {
		t.Errorf("Expected slope near 0, got %f", slope)
	}

	// CI should include zero
	if lower > 0 || upper < 0 {
		t.Errorf("Expected CI to include zero, got [%f, %f]", lower, upper)
	}
}

func TestClassifyBacklogDrift_UNSATURATED(t *testing.T) {
	// GIVEN slope CI excludes positive values (upper < 0) or includes zero with low peak
	// WHEN classifying
	// THEN returns UNSATURATED
	cfg := NewBacklogDriftConfig(60*time.Second, 5, 2.0, 0.95)

	// Case 1: Negative slope (decreasing backlog)
	classification, note, recommendation := classifyBacklogDrift(-0.01, -0.02, 0.0, 10, 5, 10, 8.0, cfg)
	if classification != "UNSATURATED" {
		t.Errorf("Case 1: Expected UNSATURATED, got %s", classification)
	}
	if note == "" {
		t.Error("Case 1: Expected non-empty note")
	}
	if recommendation == "" {
		t.Error("Case 1: Expected non-empty recommendation")
	}

	// Case 2: Flat slope (CI includes zero) with low peak ratio
	classification, note, recommendation = classifyBacklogDrift(0.0, -0.01, 0.01, 10, 10, 15, 12.0, cfg)
	if classification != "UNSATURATED" {
		t.Errorf("Case 2: Expected UNSATURATED, got %s", classification)
	}
}

func TestClassifyBacklogDrift_TRANSIENT_BACKLOG(t *testing.T) {
	// GIVEN slope CI includes zero but peak > PeakRatio * mean
	// WHEN classifying
	// THEN returns TRANSIENT_BACKLOG
	cfg := NewBacklogDriftConfig(60*time.Second, 5, 2.0, 0.95)

	// Flat slope with high peak: peak=25, mean=10, ratio=2.5 > 2.0
	classification, note, recommendation := classifyBacklogDrift(0.0, -0.01, 0.01, 10, 10, 25, 10.0, cfg)
	if classification != "TRANSIENT_BACKLOG" {
		t.Errorf("Expected TRANSIENT_BACKLOG, got %s", classification)
	}
	if note == "" {
		t.Error("Expected non-empty note")
	}
	if recommendation == "" {
		t.Error("Expected non-empty recommendation")
	}
}

func TestClassifyBacklogDrift_PERSISTENTLY_SATURATED(t *testing.T) {
	// GIVEN slope CI excludes zero (lower > 0) — statistically significant positive drift
	// WHEN classifying
	// THEN returns PERSISTENTLY_SATURATED
	cfg := NewBacklogDriftConfig(60*time.Second, 5, 2.0, 0.95)

	// Positive slope with CI excluding zero
	classification, note, recommendation := classifyBacklogDrift(0.05, 0.02, 0.08, 10, 30, 35, 22.0, cfg)
	if classification != "PERSISTENTLY_SATURATED" {
		t.Errorf("Expected PERSISTENTLY_SATURATED, got %s", classification)
	}
	if note == "" {
		t.Error("Expected non-empty note")
	}
	if recommendation == "" {
		t.Error("Expected non-empty recommendation")
	}
}

func TestAnalyzeBacklogDrift_InsufficientData(t *testing.T) {
	// GIVEN observation with fewer than MinWindows complete windows (BC-7)
	// WHEN analyzing
	// THEN returns UNSATURATED with note explaining insufficient data
	cfg := NewBacklogDriftConfig(60*time.Second, 5, 2.0, 0.95)

	// Very short observation: only 2 windows (< MinWindows=5)
	requests := []*sim.Request{
		{ArrivalTime: 0, FirstTokenTime: 10, ITL: []int64{5, 5}, TTFTSet: true, State: sim.StateCompleted},
		{ArrivalTime: 50, FirstTokenTime: 10, ITL: []int64{5}, TTFTSet: true, State: sim.StateCompleted},
	}
	simEndUs := int64(120_000_000) // 120 seconds (only 2 complete 60s windows)

	report := AnalyzeBacklogDrift(requests, simEndUs, cfg)

	if report.Classification != "UNSATURATED" {
		t.Errorf("Expected UNSATURATED for insufficient data, got %s", report.Classification)
	}
	noteLower := strings.ToLower(report.Note)
	if !strings.Contains(noteLower, "observation too short") && !strings.Contains(noteLower, "fewer") && !strings.Contains(noteLower, "windows") {
		t.Errorf("Expected note about insufficient data, got: %s", report.Note)
	}
	if report.Recommendation == "" {
		t.Error("Expected non-empty recommendation")
	}
}

func TestAnalyzeBacklogDrift_EndToEnd_PERSISTENTLY_SATURATED(t *testing.T) {
	// GIVEN requests with growing backlog
	// WHEN analyzing with sufficient windows
	// THEN returns PERSISTENTLY_SATURATED classification
	cfg := NewBacklogDriftConfig(60*time.Second, 5, 2.0, 0.95)

	// Create requests that overlap to build backlog
	requests := []*sim.Request{
		// Window 0: 5 long requests start
		{ArrivalTime: 0, FirstTokenTime: 1000, ITL: []int64{500_000_000}, TTFTSet: true, State: sim.StateCompleted},
		{ArrivalTime: 10, FirstTokenTime: 1000, ITL: []int64{500_000_000}, TTFTSet: true, State: sim.StateCompleted},
		{ArrivalTime: 20, FirstTokenTime: 1000, ITL: []int64{500_000_000}, TTFTSet: true, State: sim.StateCompleted},
		{ArrivalTime: 30, FirstTokenTime: 1000, ITL: []int64{500_000_000}, TTFTSet: true, State: sim.StateCompleted},
		{ArrivalTime: 40, FirstTokenTime: 1000, ITL: []int64{500_000_000}, TTFTSet: true, State: sim.StateCompleted},
		// More requests in later windows to sustain backlog growth
		{ArrivalTime: 70_000_000, FirstTokenTime: 1000, ITL: []int64{500_000_000}, TTFTSet: true, State: sim.StateCompleted},
		{ArrivalTime: 130_000_000, FirstTokenTime: 1000, ITL: []int64{500_000_000}, TTFTSet: true, State: sim.StateCompleted},
		{ArrivalTime: 190_000_000, FirstTokenTime: 1000, ITL: []int64{500_000_000}, TTFTSet: true, State: sim.StateCompleted},
		{ArrivalTime: 250_000_000, FirstTokenTime: 1000, ITL: []int64{500_000_000}, TTFTSet: true, State: sim.StateCompleted},
	}
	simEndUs := int64(400_000_000) // 400 seconds (6+ windows)

	report := AnalyzeBacklogDrift(requests, simEndUs, cfg)

	if report.Classification != "PERSISTENTLY_SATURATED" {
		t.Errorf("Expected PERSISTENTLY_SATURATED, got %s (note: %s)", report.Classification, report.Note)
	}
	if report.Slope <= 0 {
		t.Errorf("Expected positive slope, got %f", report.Slope)
	}
	if report.FinalBacklog <= report.InitialBacklog {
		t.Errorf("Expected final backlog > initial backlog, got %d > %d", report.FinalBacklog, report.InitialBacklog)
	}
	if len(report.Windows) < cfg.MinWindows {
		t.Errorf("Expected at least %d windows, got %d", cfg.MinWindows, len(report.Windows))
	}
}

func TestAnalyzeBacklogDrift_EndToEnd_UNSATURATED(t *testing.T) {
	// GIVEN requests with stable backlog
	// WHEN analyzing
	// THEN returns UNSATURATED classification
	cfg := NewBacklogDriftConfig(60*time.Second, 5, 2.0, 0.95)

	// Create requests that complete quickly — no backlog buildup
	var requests []*sim.Request
	for i := int64(0); i < 50; i++ {
		requests = append(requests, &sim.Request{
			ArrivalTime:    i * 10_000_000, // Every 10 seconds
			FirstTokenTime: 1000,
			ITL:            []int64{100_000}, // 100ms completion
			TTFTSet:        true,
			State:          sim.StateCompleted,
		})
	}
	simEndUs := int64(500_000_000) // 500 seconds (8+ windows)

	report := AnalyzeBacklogDrift(requests, simEndUs, cfg)

	if report.Classification != "UNSATURATED" {
		t.Errorf("Expected UNSATURATED, got %s (note: %s)", report.Classification, report.Note)
	}
	if len(report.Windows) < cfg.MinWindows {
		t.Errorf("Expected at least %d windows, got %d", cfg.MinWindows, len(report.Windows))
	}
}
