package workload

import (
	"fmt"
	"math"
	"time"

	"github.com/sirupsen/logrus"
	sim "github.com/inference-sim/inference-sim/sim"
)

// RequestInterval represents a request's active period [ArrivalUs, CompletionUs).
// Used as input to backlog-drift saturation analysis (BC-2).
type RequestInterval struct {
	ArrivalUs    int64
	CompletionUs int64
}

// BacklogDriftConfig configures the backlog-drift saturation analyzer.
type BacklogDriftConfig struct {
	WindowSize   time.Duration // Window width for sampling and per-window metrics (BC-1)
	MinWindows   int           // Minimum complete windows required for classification (BC-7)
	PeakRatio    float64       // Peak-to-mean threshold for TRANSIENT_BACKLOG detection (BC-6)
	ConfidenceCI float64       // Confidence level for slope significance test (BC-3)
}

// NewBacklogDriftConfig creates a BacklogDriftConfig with validation (BC-10, BC-14, R3).
// Panics if any parameter is invalid (NaN, Inf, out of range).
func NewBacklogDriftConfig(windowSize time.Duration, minWindows int, peakRatio, confidenceCI float64) BacklogDriftConfig {
	if windowSize <= 0 {
		panic(fmt.Sprintf("BacklogDriftConfig: WindowSize must be > 0, got %v", windowSize))
	}
	if minWindows <= 0 {
		panic(fmt.Sprintf("BacklogDriftConfig: MinWindows must be > 0, got %d", minWindows))
	}
	if peakRatio <= 0 || math.IsNaN(peakRatio) || math.IsInf(peakRatio, 0) {
		panic(fmt.Sprintf("BacklogDriftConfig: PeakRatio must be a finite value > 0, got %f", peakRatio))
	}
	if confidenceCI <= 0 || confidenceCI >= 1 || math.IsNaN(confidenceCI) || math.IsInf(confidenceCI, 0) {
		panic(fmt.Sprintf("BacklogDriftConfig: ConfidenceCI must be in (0, 1), got %f", confidenceCI))
	}
	return BacklogDriftConfig{
		WindowSize:   windowSize,
		MinWindows:   minWindows,
		PeakRatio:    peakRatio,
		ConfidenceCI: confidenceCI,
	}
}

// DefaultBacklogDriftConfig returns the default configuration per issue #1298.
func DefaultBacklogDriftConfig() BacklogDriftConfig {
	return BacklogDriftConfig{
		WindowSize:   60 * time.Second,
		MinWindows:   5,
		PeakRatio:    2.0,
		ConfidenceCI: 0.95,
	}
}

// WindowMetrics captures per-window saturation metrics (BC-1).
type WindowMetrics struct {
	StartUs      int64   // Window start timestamp (µs)
	EndUs        int64   // Window end timestamp (µs)
	NumEntered   int     // Requests with arrival in [start, end)
	NumLeft      int     // Requests with completion in [start, end)
	ActiveStart  int     // Active requests at window start
	ActiveEnd    int     // Active requests at window end
	DeltaBacklog int     // ActiveEnd - ActiveStart (must equal NumEntered - NumLeft, BC-1)
	DrainRatio   float64 // NumLeft / NumEntered (NaN if NumEntered==0)
}

// BacklogDriftReport contains saturation classification results (BC-4, BC-5, BC-6, BC-7).
type BacklogDriftReport struct {
	Classification string          `json:"classification"` // "UNSATURATED", "TRANSIENT_BACKLOG", "PERSISTENTLY_SATURATED"
	Slope          float64         `json:"slope"`          // Linear regression slope (req/µs)
	SlopeLower     float64         `json:"slope_lower"`    // Slope CI lower bound
	SlopeUpper     float64         `json:"slope_upper"`    // Slope CI upper bound
	InitialBacklog int             `json:"initial_backlog"`
	FinalBacklog   int             `json:"final_backlog"`
	PeakInFlight   int             `json:"peak_in_flight"`
	MeanInFlight   float64         `json:"mean_in_flight"`
	Windows        []WindowMetrics `json:"windows"`
	Note           string          `json:"note,omitempty"`           // Explanation (e.g., "observation too short")
	Recommendation string          `json:"recommendation,omitempty"` // User-facing guidance
}

// RequestsToIntervals converts sim.Request slices to RequestInterval slices for
// saturation analysis. Applies eligibility filter per BC-2:
//   - Exclude: TTFTSet==false AND State==StateTimedOut (timed out before TTFT)
//   - Include with simEndUs: TTFTSet==false AND State==StateRunning (horizon-truncated)
//   - Include with computed time: TTFTSet==true (completed)
// Logs included and excluded counts per BC-13 (R1: no silent discard).
func RequestsToIntervals(requests []*sim.Request, simEndUs int64) []RequestInterval {
	if len(requests) == 0 {
		return []RequestInterval{} // BC-15: degenerate input
	}

	intervals := make([]RequestInterval, 0, len(requests))
	excluded := 0

	for _, req := range requests {
		if !req.TTFTSet {
			// No valid TTFT — check if timed out or horizon-truncated
			if req.State == sim.StateTimedOut {
				// Case 1: Timed out before generating any output — exclude
				excluded++
				continue
			}
			// Case 2: Horizon-truncated (StateRunning without TTFT) — use simEndUs as completion
			intervals = append(intervals, RequestInterval{
				ArrivalUs:    req.ArrivalTime,
				CompletionUs: simEndUs,
			})
		} else {
			// Case 3: Completed (TTFTSet==true) — compute completion time
			// Completion = ArrivalTime + FirstTokenTime + Σ(inter-token latencies)
			completionUs := req.ArrivalTime + req.FirstTokenTime
			for _, itl := range req.ITL {
				completionUs += itl
			}
			intervals = append(intervals, RequestInterval{
				ArrivalUs:    req.ArrivalTime,
				CompletionUs: completionUs,
			})
		}
	}

	// BC-13: Log counts (R1 no silent discard)
	logrus.Infof("Saturation analysis: %d requests eligible, %d excluded (timed out before TTFT)", len(intervals), excluded)

	return intervals
}

// computeWindowMetrics computes per-window saturation metrics per BC-1.
// Returns one WindowMetrics entry per complete window of size windowSizeUs.
// Enforces identity: DeltaBacklog = NumEntered - NumLeft.
func computeWindowMetrics(intervals []RequestInterval, windowSizeUs, totalDurationUs int64) []WindowMetrics {
	if len(intervals) == 0 || totalDurationUs <= 0 || windowSizeUs <= 0 {
		return []WindowMetrics{}
	}

	numWindows := int((totalDurationUs + windowSizeUs - 1) / windowSizeUs) // Ceiling division
	windows := make([]WindowMetrics, numWindows)

	for i := 0; i < numWindows; i++ {
		startUs := int64(i) * windowSizeUs
		endUs := startUs + windowSizeUs
		if endUs > totalDurationUs {
			endUs = totalDurationUs
		}

		w := WindowMetrics{
			StartUs: startUs,
			EndUs:   endUs,
		}

		// Compute metrics by scanning all intervals
		for _, iv := range intervals {
			// NumEntered: arrival in [startUs, endUs)
			if iv.ArrivalUs >= startUs && iv.ArrivalUs < endUs {
				w.NumEntered++
			}
			// NumLeft: completion in [startUs, endUs)
			if iv.CompletionUs >= startUs && iv.CompletionUs < endUs {
				w.NumLeft++
			}
			// ActiveStart: interval contains startUs (arrival <= startUs < completion)
			if iv.ArrivalUs <= startUs && startUs < iv.CompletionUs {
				w.ActiveStart++
			}
			// ActiveEnd: interval contains endUs
			if iv.ArrivalUs <= endUs && endUs < iv.CompletionUs {
				w.ActiveEnd++
			}
		}

		// DeltaBacklog and DrainRatio
		w.DeltaBacklog = w.ActiveEnd - w.ActiveStart
		if w.NumEntered > 0 {
			w.DrainRatio = float64(w.NumLeft) / float64(w.NumEntered)
		} else {
			w.DrainRatio = math.NaN() // Undefined when no arrivals
		}

		windows[i] = w
	}

	return windows
}
