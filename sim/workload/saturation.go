package workload

import (
	"math"
	"sort"
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
// Uses sweep-line algorithm: O(N log N + S) where N = num requests, S = num samples.
func computeActiveRequests(records []TraceRecord, sampleTimestamps []int64) []int {
	type event struct {
		timestamp int64
		delta     int // +1 for arrival, -1 for completion
	}

	events := make([]event, 0, 2*len(records))
	for _, r := range records {
		arrivalTime := r.ArrivalTimeUs
		completionTime := r.LastChunkTimeUs // Already absolute timestamp
		events = append(events, event{timestamp: arrivalTime, delta: 1})
		events = append(events, event{timestamp: completionTime, delta: -1})
	}

	// Sort events by timestamp, with completions before arrivals at the same timestamp
	// (this ensures active_requests(t) does not include requests that complete exactly at t)
	sort.Slice(events, func(i, j int) bool {
		if events[i].timestamp != events[j].timestamp {
			return events[i].timestamp < events[j].timestamp
		}
		// At same timestamp: completion (-1) before arrival (+1)
		return events[i].delta < events[j].delta
	})

	// Sweep through sample timestamps and compute active count at each
	// A request is active at time t if arrival_time <= t < completion_time
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
	numWindows := int((maxTime-minTime)/windowDurationUs) + 1
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
			// Handle completions that fall exactly on the boundary of the last window
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
