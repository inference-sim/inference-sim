package workload

import (
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
