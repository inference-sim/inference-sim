package workload

import (
	"encoding/json"
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

func TestClassifyBacklogTrend_Unsaturated(t *testing.T) {
	// GIVEN slope near zero, final backlog ≈ initial backlog
	slope := 0.05 // below threshold
	initialBacklog := 10
	finalBacklog := 11         // ratio = 1.1 (within [0.9, 1.1])
	hadTransientSpike := false

	// WHEN classifying
	verdict := classifyBacklogTrend(slope, initialBacklog, finalBacklog, hadTransientSpike)

	// THEN verdict is UNSATURATED
	if verdict != "UNSATURATED" {
		t.Errorf("verdict = %s, want UNSATURATED", verdict)
	}
}

func TestClassifyBacklogTrend_TransientBacklog(t *testing.T) {
	// GIVEN slope near zero, final ≈ initial, but some windows had drain_ratio < 1.0
	slope := 0.03
	initialBacklog := 5
	finalBacklog := 6          // ratio = 1.2 (close to 1.0)
	hadTransientSpike := true  // indicates temporary overload

	// WHEN classifying
	verdict := classifyBacklogTrend(slope, initialBacklog, finalBacklog, hadTransientSpike)

	// THEN verdict is TRANSIENT_BACKLOG
	if verdict != "TRANSIENT_BACKLOG" {
		t.Errorf("verdict = %s, want TRANSIENT_BACKLOG", verdict)
	}
}

func TestClassifyBacklogTrend_PersistentlySaturated(t *testing.T) {
	// GIVEN positive slope above threshold, final >> initial
	slope := 0.5 // significantly positive
	initialBacklog := 10
	finalBacklog := 20 // ratio = 2.0 (> 1.5)
	hadTransientSpike := false

	// WHEN classifying
	verdict := classifyBacklogTrend(slope, initialBacklog, finalBacklog, hadTransientSpike)

	// THEN verdict is PERSISTENTLY_SATURATED
	if verdict != "PERSISTENTLY_SATURATED" {
		t.Errorf("verdict = %s, want PERSISTENTLY_SATURATED", verdict)
	}
}

func TestAnalyzeSaturation_Unsaturated(t *testing.T) {
	// GIVEN trace with balanced arrivals/completions
	records := make([]TraceRecord, 100)
	for i := 0; i < 100; i++ {
		arrivalTime := int64(i * 1_000_000) // 1 request per second
		records[i] = TraceRecord{
			RequestID:       i,
			ArrivalTimeUs:   arrivalTime,
			SendTimeUs:      arrivalTime,
			LastChunkTimeUs: arrivalTime + 500_000, // 500ms latency
		}
	}
	trace := TraceV2{Records: records}

	// WHEN analyzing saturation with 10s windows
	verdict := AnalyzeSaturation(trace, 10.0)

	// THEN verdict should be UNSATURATED (system kept up)
	if verdict.Verdict != "UNSATURATED" {
		t.Errorf("verdict = %s, want UNSATURATED", verdict.Verdict)
	}
	if verdict.BacklogSlope > 0.05 {
		t.Errorf("backlog slope = %.3f, expected near zero", verdict.BacklogSlope)
	}
}

func TestAnalyzeSaturation_PersistentlySaturated(t *testing.T) {
	// GIVEN trace where requests accumulate rapidly (high arrival rate, long latencies)
	// Arrivals: 10 req/sec, Latency: constant 15 seconds
	// This means 10 new requests arrive per second, but each takes 15s to complete
	// Backlog grows by ~10 req/s (arrival rate) - ~0.67 req/s (completion rate) = 9.33 req/s
	records := make([]TraceRecord, 50)
	for i := 0; i < 50; i++ {
		arrivalTime := int64(i * 100_000)         // 10 req/s (100ms apart)
		latency := int64(15_000_000)              // constant 15s latency
		records[i] = TraceRecord{
			RequestID:       i,
			ArrivalTimeUs:   arrivalTime,
			SendTimeUs:      arrivalTime,
			LastChunkTimeUs: arrivalTime + latency,
		}
	}
	trace := TraceV2{Records: records}

	// WHEN analyzing with 1s windows
	verdict := AnalyzeSaturation(trace, 1.0)

	// THEN verdict should be PERSISTENTLY_SATURATED
	if verdict.Verdict != "PERSISTENTLY_SATURATED" {
		t.Errorf("verdict = %s, want PERSISTENTLY_SATURATED (slope=%.3f, ratio=%.2f)",
			verdict.Verdict, verdict.BacklogSlope, float64(verdict.FinalBacklog)/float64(verdict.InitialBacklog))
	}
	if verdict.BacklogSlope <= 0 {
		t.Errorf("backlog slope = %.3f, expected positive", verdict.BacklogSlope)
	}
	if verdict.FinalBacklog <= verdict.InitialBacklog {
		t.Errorf("final backlog %d should be > initial %d", verdict.FinalBacklog, verdict.InitialBacklog)
	}
}

func TestAnalyzeSaturation_InsufficientData(t *testing.T) {
	// GIVEN trace with only 5 requests (< 10 threshold)
	records := make([]TraceRecord, 5)
	for i := 0; i < 5; i++ {
		arrivalTime := int64(i * 1_000_000)
		records[i] = TraceRecord{
			RequestID:       i,
			ArrivalTimeUs:   arrivalTime,
			SendTimeUs:      arrivalTime,
			LastChunkTimeUs: arrivalTime + 500_000,
		}
	}
	trace := TraceV2{Records: records}

	// WHEN analyzing
	verdict := AnalyzeSaturation(trace, 10.0)

	// THEN verdict should be INSUFFICIENT_DATA
	if verdict.Verdict != "INSUFFICIENT_DATA" {
		t.Errorf("verdict = %s, want INSUFFICIENT_DATA", verdict.Verdict)
	}
}

func TestAnalyzeSaturation_EmptyTrace(t *testing.T) {
	// GIVEN empty trace (zero requests)
	trace := TraceV2{Records: []TraceRecord{}}

	// WHEN analyzing
	verdict := AnalyzeSaturation(trace, 10.0)

	// THEN verdict should be INSUFFICIENT_DATA
	if verdict.Verdict != "INSUFFICIENT_DATA" {
		t.Errorf("verdict = %s, want INSUFFICIENT_DATA", verdict.Verdict)
	}
	// THEN all metrics should be zero
	if verdict.WindowCount != 0 {
		t.Errorf("WindowCount = %d, want 0", verdict.WindowCount)
	}
	if verdict.BacklogSlope != 0.0 {
		t.Errorf("BacklogSlope = %.3f, want 0.0", verdict.BacklogSlope)
	}
	if verdict.ObservationDurationS != 0.0 {
		t.Errorf("ObservationDurationS = %.3f, want 0.0", verdict.ObservationDurationS)
	}
}

func TestSaturationVerdict_JSONRoundTrip(t *testing.T) {
	// GIVEN a saturation verdict
	original := SaturationVerdict{
		Verdict:              "PERSISTENTLY_SATURATED",
		WindowCount:          5,
		BacklogSlope:         0.25,
		InitialBacklog:       10,
		FinalBacklog:         20,
		ObservationDurationS: 120.5,
	}

	// WHEN marshaling to JSON and back
	jsonData, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var decoded SaturationVerdict
	if err := json.Unmarshal(jsonData, &decoded); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	// THEN all fields should match
	if decoded.Verdict != original.Verdict {
		t.Errorf("Verdict: got %s, want %s", decoded.Verdict, original.Verdict)
	}
	if decoded.WindowCount != original.WindowCount {
		t.Errorf("WindowCount: got %d, want %d", decoded.WindowCount, original.WindowCount)
	}
	if decoded.BacklogSlope != original.BacklogSlope {
		t.Errorf("BacklogSlope: got %.3f, want %.3f", decoded.BacklogSlope, original.BacklogSlope)
	}
	if decoded.InitialBacklog != original.InitialBacklog {
		t.Errorf("InitialBacklog: got %d, want %d", decoded.InitialBacklog, original.InitialBacklog)
	}
	if decoded.FinalBacklog != original.FinalBacklog {
		t.Errorf("FinalBacklog: got %d, want %d", decoded.FinalBacklog, original.FinalBacklog)
	}
	if decoded.ObservationDurationS != original.ObservationDurationS {
		t.Errorf("ObservationDurationS: got %.3f, want %.3f", decoded.ObservationDurationS, original.ObservationDurationS)
	}
}

func TestComputeActiveRequests_SimultaneousCompletions(t *testing.T) {
	// GIVEN three requests that all complete at exactly the same microsecond
	completionTime := int64(5_000_000) // 5 seconds
	records := []TraceRecord{
		{RequestID: 0, ArrivalTimeUs: 1_000_000, SendTimeUs: 1_000_000, LastChunkTimeUs: completionTime},
		{RequestID: 1, ArrivalTimeUs: 2_000_000, SendTimeUs: 2_000_000, LastChunkTimeUs: completionTime},
		{RequestID: 2, ArrivalTimeUs: 3_000_000, SendTimeUs: 3_000_000, LastChunkTimeUs: completionTime},
	}

	// WHEN sampling at and around the completion time
	samples := []int64{
		4_000_000,  // before all complete
		completionTime,  // exactly at completion
		6_000_000,  // after all complete
	}
	actives := computeActiveRequests(records, samples)

	// THEN all should be active before, none at completion (half-open interval)
	expected := []int{3, 0, 0}
	for i, sample := range samples {
		if actives[i] != expected[i] {
			t.Errorf("active_requests(%d) = %d, want %d", sample, actives[i], expected[i])
		}
	}
}

func TestAnalyzeSaturation_PreemptionScenario(t *testing.T) {
	// GIVEN trace where a request appears to have very long latency
	// (simulating preemption - request was evicted, re-queued, then completed)
	// CURRENT LIMITATION: Preemption is not captured in TraceV2, so appears as single long request
	records := []TraceRecord{
		// Normal requests: 1s latency
		{RequestID: 0, ArrivalTimeUs: 0, SendTimeUs: 0, LastChunkTimeUs: 1_000_000},
		{RequestID: 1, ArrivalTimeUs: 1_000_000, SendTimeUs: 1_000_000, LastChunkTimeUs: 2_000_000},
		// "Preempted" request: arrives at t=2s, completes at t=10s (8s latency)
		// Reality: might have been preempted at t=3s, idle until t=9s, re-executed
		// Trace shows: continuous execution from t=2s to t=10s
		{RequestID: 2, ArrivalTimeUs: 2_000_000, SendTimeUs: 2_000_000, LastChunkTimeUs: 10_000_000},
		// More normal requests
		{RequestID: 3, ArrivalTimeUs: 3_000_000, SendTimeUs: 3_000_000, LastChunkTimeUs: 4_000_000},
		{RequestID: 4, ArrivalTimeUs: 4_000_000, SendTimeUs: 4_000_000, LastChunkTimeUs: 5_000_000},
		{RequestID: 5, ArrivalTimeUs: 5_000_000, SendTimeUs: 5_000_000, LastChunkTimeUs: 6_000_000},
		{RequestID: 6, ArrivalTimeUs: 6_000_000, SendTimeUs: 6_000_000, LastChunkTimeUs: 7_000_000},
		{RequestID: 7, ArrivalTimeUs: 7_000_000, SendTimeUs: 7_000_000, LastChunkTimeUs: 8_000_000},
		{RequestID: 8, ArrivalTimeUs: 8_000_000, SendTimeUs: 8_000_000, LastChunkTimeUs: 9_000_000},
		{RequestID: 9, ArrivalTimeUs: 9_000_000, SendTimeUs: 9_000_000, LastChunkTimeUs: 10_000_000},
	}
	trace := TraceV2{Records: records}

	// WHEN analyzing with 2s windows
	verdict := AnalyzeSaturation(trace, 2.0)

	// THEN current behavior: treats preempted request as continuously active
	// This documents the KNOWN LIMITATION - preemption causes overestimation of backlog
	if verdict.Verdict == "" {
		t.Error("verdict should not be empty")
	}
	// The "preempted" request (RequestID: 2) will be counted as active for entire [2s, 10s) interval
	// Expected: NOT UNSATURATED because the long-running request creates artificial backlog
	if verdict.Verdict == "UNSATURATED" {
		t.Errorf("Verdict = UNSATURATED but expected TRANSIENT_BACKLOG or PERSISTENTLY_SATURATED due to preemption limitation")
	}
	t.Logf("Verdict: %s (documents current preemption-unaware behavior)", verdict.Verdict)
	t.Logf("BacklogSlope: %.3f (may be affected by preemption)", verdict.BacklogSlope)
}

func TestLinearTrend_FlatSeries(t *testing.T) {
	// GIVEN constant y values (flat series)
	xValues := []float64{0, 1, 2, 3, 4}
	yValues := []float64{5, 5, 5, 5, 5}

	// WHEN computing linear trend
	slope := linearTrend(xValues, yValues)

	// THEN slope should be zero
	if slope != 0.0 {
		t.Errorf("slope = %.6f, want 0.0 for flat series", slope)
	}
}

func TestLinearTrend_AscendingSeries(t *testing.T) {
	// GIVEN ascending series with slope = 2.0
	xValues := []float64{0, 1, 2, 3, 4}
	yValues := []float64{0, 2, 4, 6, 8}

	// WHEN computing linear trend
	slope := linearTrend(xValues, yValues)

	// THEN slope should be 2.0
	if slope < 1.99 || slope > 2.01 {
		t.Errorf("slope = %.6f, want 2.0 for ascending series", slope)
	}
}

func TestLinearTrend_DescendingSeries(t *testing.T) {
	// GIVEN descending series with slope = -1.5
	xValues := []float64{0, 1, 2, 3, 4}
	yValues := []float64{10, 8.5, 7, 5.5, 4}

	// WHEN computing linear trend
	slope := linearTrend(xValues, yValues)

	// THEN slope should be -1.5
	if slope < -1.51 || slope > -1.49 {
		t.Errorf("slope = %.6f, want -1.5 for descending series", slope)
	}
}

func TestLinearTrend_EmptyInput(t *testing.T) {
	// GIVEN empty input arrays
	xValues := []float64{}
	yValues := []float64{}

	// WHEN computing linear trend
	slope := linearTrend(xValues, yValues)

	// THEN should return 0.0 (no variance case)
	if slope != 0.0 {
		t.Errorf("slope = %.6f, want 0.0 for empty input", slope)
	}
}

func TestLinearTrend_MismatchedLengths(t *testing.T) {
	// GIVEN mismatched array lengths
	xValues := []float64{0, 1, 2}
	yValues := []float64{0, 1}

	// WHEN computing linear trend
	slope := linearTrend(xValues, yValues)

	// THEN should return 0.0 (guard case)
	if slope != 0.0 {
		t.Errorf("slope = %.6f, want 0.0 for mismatched lengths", slope)
	}
}

func TestLinearTrend_ConstantX(t *testing.T) {
	// GIVEN constant x values (no variance in independent variable)
	xValues := []float64{5, 5, 5, 5}
	yValues := []float64{1, 2, 3, 4}

	// WHEN computing linear trend
	slope := linearTrend(xValues, yValues)

	// THEN should return 0.0 (denominator near zero guard)
	if slope != 0.0 {
		t.Errorf("slope = %.6f, want 0.0 for constant x", slope)
	}
}

func TestComputeWindowMetrics_MultipleWindows(t *testing.T) {
	// GIVEN requests spanning multiple 10-second windows
	records := []TraceRecord{
		// Window 1 [0s, 10s): 2 enter, 1 leaves
		{RequestID: 0, ArrivalTimeUs: 1_000_000, SendTimeUs: 1_000_000, LastChunkTimeUs: 8_000_000},
		{RequestID: 1, ArrivalTimeUs: 5_000_000, SendTimeUs: 5_000_000, LastChunkTimeUs: 15_000_000}, // leaves in window 2
		// Window 2 [10s, 20s): 2 enter, 2 leave
		{RequestID: 2, ArrivalTimeUs: 12_000_000, SendTimeUs: 12_000_000, LastChunkTimeUs: 18_000_000},
		{RequestID: 3, ArrivalTimeUs: 14_000_000, SendTimeUs: 14_000_000, LastChunkTimeUs: 25_000_000}, // leaves in window 3
		// Window 3 [20s, 30s): 1 enters, 2 leave
		{RequestID: 4, ArrivalTimeUs: 22_000_000, SendTimeUs: 22_000_000, LastChunkTimeUs: 28_000_000},
	}
	windowDurationUs := int64(10_000_000) // 10 seconds

	// WHEN computing window metrics
	windows := computeWindowMetrics(records, windowDurationUs)

	// THEN should have 3 windows
	if len(windows) != 3 {
		t.Fatalf("expected 3 windows, got %d", len(windows))
	}

	// Window 1: 2 entered, 1 left
	if windows[0].NumEntered != 2 {
		t.Errorf("Window 0 NumEntered = %d, want 2", windows[0].NumEntered)
	}
	if windows[0].NumLeft != 1 {
		t.Errorf("Window 0 NumLeft = %d, want 1", windows[0].NumLeft)
	}

	// Window 2: 2 entered, 2 left
	if windows[1].NumEntered != 2 {
		t.Errorf("Window 1 NumEntered = %d, want 2", windows[1].NumEntered)
	}
	if windows[1].NumLeft != 2 {
		t.Errorf("Window 1 NumLeft = %d, want 2", windows[1].NumLeft)
	}

	// Window 3: 1 entered, 2 left
	if windows[2].NumEntered != 1 {
		t.Errorf("Window 2 NumEntered = %d, want 1", windows[2].NumEntered)
	}
	if windows[2].NumLeft != 2 {
		t.Errorf("Window 2 NumLeft = %d, want 2", windows[2].NumLeft)
	}
}

func TestComputeActiveRequests_EmptyRecords(t *testing.T) {
	// GIVEN empty records array
	records := []TraceRecord{}
	samples := []int64{1000, 2000, 3000}

	// WHEN computing active requests
	actives := computeActiveRequests(records, samples)

	// THEN all samples should have zero active requests
	for i, count := range actives {
		if count != 0 {
			t.Errorf("active_requests(%d) = %d, want 0", samples[i], count)
		}
	}
}

func TestComputeWindowMetrics_EmptyRecords(t *testing.T) {
	// GIVEN empty records array
	records := []TraceRecord{}
	windowDurationUs := int64(60_000_000)

	// WHEN computing window metrics
	windows := computeWindowMetrics(records, windowDurationUs)

	// THEN should return nil (no windows)
	if windows != nil {
		t.Errorf("expected nil windows for empty records, got %d windows", len(windows))
	}
}

func TestClassifyBacklogTrend_ZeroBacklogs(t *testing.T) {
	// GIVEN both initial and final backlog are zero (idle system)
	slope := 0.0
	initialBacklog := 0
	finalBacklog := 0
	hadTransientSpike := false

	// WHEN classifying
	verdict := classifyBacklogTrend(slope, initialBacklog, finalBacklog, hadTransientSpike)

	// THEN should classify as UNSATURATED (ratio is 1.0 by special case handling)
	if verdict != "UNSATURATED" {
		t.Errorf("verdict = %s, want UNSATURATED for idle system", verdict)
	}
}

func TestClassifyBacklogTrend_GrowthFromZero(t *testing.T) {
	// GIVEN backlog grows from zero to nonzero (initial warmup)
	slope := 0.5
	initialBacklog := 0
	finalBacklog := 10
	hadTransientSpike := false

	// WHEN classifying
	verdict := classifyBacklogTrend(slope, initialBacklog, finalBacklog, hadTransientSpike)

	// THEN should classify as PERSISTENTLY_SATURATED (ratio is infinity)
	if verdict != "PERSISTENTLY_SATURATED" {
		t.Errorf("verdict = %s, want PERSISTENTLY_SATURATED for growth from zero", verdict)
	}
}
