package workload

import (
	"math"
)

// SaturationVerdict represents the result of saturation analysis.
type SaturationVerdict struct {
	Verdict              string  `json:"verdict"` // "UNSATURATED", "TRANSIENT_BACKLOG", "PERSISTENTLY_SATURATED", "INSUFFICIENT_DATA"
	WindowCount          int     `json:"window_count"`
	BacklogSlope         float64 `json:"backlog_slope"`          // linear trend slope (requests/second)
	InitialBacklog       int     `json:"initial_backlog"`        // active requests at start
	FinalBacklog         int     `json:"final_backlog"`          // active requests at end
	ObservationDurationS float64 `json:"observation_duration_s"` // total observation time in seconds
}

// computeActiveRequests computes the number of active (in-flight) requests at each sample timestamp.
// A request is active at time t if arrival_time_us <= t < completion_time_us.
// Completion time is computed as SendTimeUs + (LastChunkTimeUs - SendTimeUs) = LastChunkTimeUs.
// Uses brute force: O(N*S) where N = num requests, S = num samples.
// Acceptable for typical workloads (100-1000 requests, 10-20 windows = ~10k iterations < 1ms).
func computeActiveRequests(records []TraceRecord, sampleTimestamps []int64) []int {
	result := make([]int, len(sampleTimestamps))

	for sampleIdx, sampleTime := range sampleTimestamps {
		count := 0
		for _, r := range records {
			// Active if: arrival <= sampleTime < completion
			if r.ArrivalTimeUs <= sampleTime && r.LastChunkTimeUs > sampleTime {
				count++
			}
		}
		result[sampleIdx] = count
	}

	return result
}

// windowMetrics captures per-window saturation indicators.
type windowMetrics struct {
	StartTimeUs  int64
	EndTimeUs    int64
	NumEntered   int     // requests with arrival in [start, end)
	NumLeft      int     // requests with completion in [start, end)
	ActiveStart  int     // active requests at start
	ActiveEnd    int     // active requests at end
	DeltaBacklog int     // ActiveEnd - ActiveStart
	DrainRatio   float64 // NumLeft / NumEntered (0.0 if NumEntered == 0)
}

// computeWindowMetrics partitions the observation into fixed-width windows and computes metrics per window.
func computeWindowMetrics(records []TraceRecord, windowDurationUs int64) []windowMetrics {
	if len(records) == 0 {
		return nil
	}

	// Find observation bounds
	minTime := records[0].ArrivalTimeUs
	maxTime := records[0].LastChunkTimeUs
	for _, r := range records {
		if r.ArrivalTimeUs < minTime {
			minTime = r.ArrivalTimeUs
		}
		if r.LastChunkTimeUs > maxTime {
			maxTime = r.LastChunkTimeUs
		}
	}

	// Generate window boundaries
	// Use ceiling division to avoid creating empty trailing windows when observation
	// duration is exact multiple of window duration (e.g., 60s trace with 60s windows).
	// Formula: ceil(duration / windowDuration) = (duration + windowDuration - 1) / windowDuration
	// Ensures at least 1 window even for traces shorter than windowDurationUs.
	duration := maxTime - minTime
	numWindows := int((duration+windowDurationUs-1)/windowDurationUs)
	if numWindows < 1 {
		numWindows = 1
	}
	windows := make([]windowMetrics, numWindows)
	for i := 0; i < numWindows; i++ {
		windows[i].StartTimeUs = minTime + int64(i)*windowDurationUs
		windows[i].EndTimeUs = windows[i].StartTimeUs + windowDurationUs
	}

	// Compute active requests at each window boundary
	boundaryTimes := make([]int64, 0, numWindows+1)
	for i := 0; i < numWindows; i++ {
		boundaryTimes = append(boundaryTimes, windows[i].StartTimeUs)
	}
	boundaryTimes = append(boundaryTimes, windows[numWindows-1].EndTimeUs)
	activeCounts := computeActiveRequests(records, boundaryTimes)

	for i := 0; i < numWindows; i++ {
		windows[i].ActiveStart = activeCounts[i]
		windows[i].ActiveEnd = activeCounts[i+1]
		windows[i].DeltaBacklog = windows[i].ActiveEnd - windows[i].ActiveStart
	}

	// Count arrivals and completions per window
	for _, r := range records {
		arrivalTime := r.ArrivalTimeUs
		completionTime := r.LastChunkTimeUs

		// Find arrival window (inclusive of minTime, exclusive of maxTime for intermediate points)
		if arrivalTime >= minTime {
			windowIdx := int((arrivalTime - minTime) / windowDurationUs)
			if windowIdx < len(windows) && arrivalTime < windows[windowIdx].EndTimeUs {
				windows[windowIdx].NumEntered++
			}
		}

		// Find completion window (inclusive of both minTime and maxTime since maxTime is defined by last completion)
		if completionTime >= minTime {
			windowIdx := int((completionTime - minTime) / windowDurationUs)
			// Handle edge case: completions at exactly maxTime can index beyond array.
			// Example: maxTime=120s, windowDuration=60s → windowIdx=2 but len(windows)=2
			// Clamp to last window since maxTime completion should count in final window.
			if windowIdx >= len(windows) {
				windowIdx = len(windows) - 1
			}
			if windowIdx >= 0 && windowIdx < len(windows) {
				windows[windowIdx].NumLeft++
			}
		}
	}

	// Compute drain ratio
	for i := 0; i < len(windows); i++ {
		if windows[i].NumEntered > 0 {
			windows[i].DrainRatio = float64(windows[i].NumLeft) / float64(windows[i].NumEntered)
		} else {
			windows[i].DrainRatio = 0.0 // No arrivals in window
		}
	}

	return windows
}

// linearTrend fits a simple linear regression y = a + b*x to the (x, y) data points.
// Returns slope b. Uses least squares method.
func linearTrend(xValues, yValues []float64) float64 {
	if len(xValues) != len(yValues) || len(xValues) == 0 {
		return 0.0
	}

	n := float64(len(xValues))
	var sumX, sumY, sumXY, sumX2 float64
	for i := 0; i < len(xValues); i++ {
		sumX += xValues[i]
		sumY += yValues[i]
		sumXY += xValues[i] * yValues[i]
		sumX2 += xValues[i] * xValues[i]
	}

	// Slope b = (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)
	denominator := n*sumX2 - sumX*sumX
	if math.Abs(denominator) < 1e-9 {
		return 0.0 // No variance in x (all same time)
	}
	slope := (n*sumXY - sumX*sumY) / denominator
	return slope
}

// Classification thresholds from discussion #1163 section 4.
// These empirically-derived values distinguish persistent saturation from transient spikes.
const (
	backlogSlopeThreshold      = 0.1  // requests per second - significant positive trend
	unsaturatedBacklogRatioMin = 0.9  // final/initial backlog ratio bounds for stable system
	unsaturatedBacklogRatioMax = 1.1
	saturatedBacklogRatioMin   = 1.5  // final/initial backlog ratio indicating growth
)

// classifyBacklogTrend determines saturation level from backlog trend analysis.
// Uses three-level classification from discussion #1163.
func classifyBacklogTrend(slope float64, initialBacklog, finalBacklog int, hadTransientSpike bool) string {
	// Guard against division by zero
	var backlogRatio float64
	if initialBacklog == 0 {
		if finalBacklog == 0 {
			backlogRatio = 1.0
		} else {
			backlogRatio = math.Inf(1) // Grew from zero to nonzero
		}
	} else {
		backlogRatio = float64(finalBacklog) / float64(initialBacklog)
	}

	// PERSISTENTLY_SATURATED: positive slope AND final >> initial
	if slope > backlogSlopeThreshold && backlogRatio > saturatedBacklogRatioMin {
		return "PERSISTENTLY_SATURATED"
	}

	// UNSATURATED: slope near zero AND final ≈ initial AND no transient spikes
	if math.Abs(slope) < backlogSlopeThreshold && backlogRatio >= unsaturatedBacklogRatioMin && backlogRatio <= unsaturatedBacklogRatioMax && !hadTransientSpike {
		return "UNSATURATED"
	}

	// TRANSIENT_BACKLOG: everything else (temporary overload but recovered)
	return "TRANSIENT_BACKLOG"
}

const minRequestsForAnalysis = 10

// AnalyzeSaturation performs backlog drift analysis on trace data and returns a saturation verdict.
// Uses the three-level classification from discussion #1163.
// Returns INSUFFICIENT_DATA verdict if trace has < 10 requests.
//
// LIMITATION: Does not account for request preemption. Preempted requests are treated as
// continuously active from arrival to final completion, which may overestimate backlog in
// systems with frequent preemption. This is a TraceV2 data limitation - preemption events
// are not currently captured in the trace format. See follow-up issue for TraceV2 extension.
func AnalyzeSaturation(trace TraceV2, windowDurationS float64) SaturationVerdict {
	// R3: Validate numeric parameters - prevent divide-by-zero in computeWindowMetrics
	if windowDurationS <= 0 {
		panic("AnalyzeSaturation: windowDurationS must be > 0")
	}

	records := trace.Records
	if len(records) < minRequestsForAnalysis {
		return SaturationVerdict{
			Verdict:              "INSUFFICIENT_DATA",
			WindowCount:          0,
			BacklogSlope:         0.0,
			InitialBacklog:       0,
			FinalBacklog:         0,
			ObservationDurationS: 0.0,
		}
	}

	// Convert window duration to microseconds
	windowDurationUs := int64(windowDurationS * 1_000_000)

	// Compute window metrics
	windows := computeWindowMetrics(records, windowDurationUs)
	if len(windows) == 0 {
		return SaturationVerdict{Verdict: "INSUFFICIENT_DATA"}
	}

	// Extract active request time series for trend fitting
	timePoints := make([]float64, len(windows))
	activePoints := make([]float64, len(windows))
	for i, w := range windows {
		timePoints[i] = float64(w.StartTimeUs) / 1_000_000.0 // convert to seconds
		activePoints[i] = float64(w.ActiveStart)
	}

	// Fit linear trend to active_requests(t)
	slope := linearTrend(timePoints, activePoints)

	// Determine if there were transient spikes (any window with drain_ratio < 1.0 AND delta > 0)
	hadTransientSpike := false
	for _, w := range windows {
		if w.DrainRatio < 1.0 && w.DeltaBacklog > 0 {
			hadTransientSpike = true
			break
		}
	}

	// Get initial and final backlog
	// Use ActiveStart of first and last windows (sampling at window starts, not ends).
	// Rationale: ActiveEnd of the last window would be 0 after all requests complete
	// (since maxTime is defined by the last completion), giving misleading final backlog.
	// Sampling at window start gives true backlog at that point in time.
	initialBacklog := windows[0].ActiveStart
	finalBacklog := windows[len(windows)-1].ActiveStart

	// Classify
	verdict := classifyBacklogTrend(slope, initialBacklog, finalBacklog, hadTransientSpike)

	// Compute observation duration
	minTime := records[0].ArrivalTimeUs
	maxTime := records[0].LastChunkTimeUs
	for _, r := range records {
		if r.ArrivalTimeUs < minTime {
			minTime = r.ArrivalTimeUs
		}
		if r.LastChunkTimeUs > maxTime {
			maxTime = r.LastChunkTimeUs
		}
	}
	observationDurationS := float64(maxTime-minTime) / 1_000_000.0

	return SaturationVerdict{
		Verdict:              verdict,
		WindowCount:          len(windows),
		BacklogSlope:         slope,
		InitialBacklog:       initialBacklog,
		FinalBacklog:         finalBacklog,
		ObservationDurationS: observationDurationS,
	}
}
