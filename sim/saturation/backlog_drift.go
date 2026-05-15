// sim/saturation/backlog_drift.go
package saturation

import (
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// BacklogDriftDetector wraps the workload.AnalyzeBacklogDrift logic
// as a post-hoc saturation detector (Issue #7).
// This detector is stateless - it performs regression analysis over
// completed request intervals during Classify().
type BacklogDriftDetector struct {
	config workload.BacklogDriftConfig
}

// NewBacklogDriftDetector creates a BacklogDriftDetector with default configuration.
func NewBacklogDriftDetector() Detector {
	return &BacklogDriftDetector{
		config: workload.DefaultBacklogDriftConfig(),
	}
}

func (b *BacklogDriftDetector) Name() string {
	return "backlog-drift"
}

// Observe is not used by backlog-drift detector (batch-only analysis).
func (b *BacklogDriftDetector) Observe(event Event) {
	// No-op: backlog-drift is a batch analyzer, doesn't use streaming events
}

// Detect is not used by backlog-drift detector (batch-only analysis).
func (b *BacklogDriftDetector) Detect() Result {
	// No-op: return stable with zero confidence
	return Result{
		Level:      Stable,
		Score:      0,
		Confidence: 0,
		Signals:    make(map[string]float64),
	}
}

// Classify performs backlog-drift saturation analysis on completed requests.
// Converts RequestMetrics to Request format, calls AnalyzeBacklogDrift,
// and maps the classification to Level enum.
//
// Classification mapping:
//   - "UNSATURATED" → Stable
//   - "TRANSIENT_BACKLOG" → Backlogged
//   - "PERSISTENTLY_SATURATED" → Overloaded
func (b *BacklogDriftDetector) Classify(requests []sim.RequestMetrics, totalArrivals int) interface{} {
	// Convert RequestMetrics to Request format for AnalyzeBacklogDrift
	// We need to construct requests with timing information
	reqs := make([]*sim.Request, len(requests))
	simEndUs := int64(0)

	for i, rm := range requests {
		// Compute completion time from arrival + E2E latency
		arrivalUs := int64(rm.ArrivedAt * 1e6) // Convert seconds to microseconds
		e2eUs := int64(rm.E2E * 1e3)           // Convert milliseconds to microseconds
		completionUs := arrivalUs + e2eUs

		if completionUs > simEndUs {
			simEndUs = completionUs
		}

		// Create a minimal Request with timing info
		// AnalyzeBacklogDrift only needs ArrivalTime, TTFTSet, and FirstTokenTime + ITL
		// For completed requests, set TTFTSet=true and put all latency in FirstTokenTime
		reqs[i] = &sim.Request{
			ID:             rm.ID,
			ArrivalTime:    arrivalUs,
			TTFTSet:        true,
			FirstTokenTime: e2eUs, // Put entire E2E latency in FirstTokenTime
			ITL:            []int64{},
			State:          sim.StateCompleted,
		}
	}

	// Run backlog-drift analysis
	report := workload.AnalyzeBacklogDrift(reqs, simEndUs, b.config)

	// Map classification to Level
	var level Level
	switch report.Classification {
	case "UNSATURATED":
		level = Stable
	case "TRANSIENT_BACKLOG":
		level = Backlogged
	case "PERSISTENTLY_SATURATED":
		level = Overloaded
	default:
		level = Stable // Conservative fallback
	}

	// Compute confidence based on number of windows analyzed
	// confidence = min(1.0, windows / MinWindows)
	confidence := 0.0
	if len(report.Windows) >= b.config.MinWindows {
		confidence = 1.0
	} else if b.config.MinWindows > 0 {
		confidence = float64(len(report.Windows)) / float64(b.config.MinWindows)
	}

	// Build signals map from report metrics
	signals := map[string]float64{
		"slope":           report.Slope,
		"slope_lower":     report.SlopeLower,
		"slope_upper":     report.SlopeUpper,
		"initial_backlog": float64(report.InitialBacklog),
		"final_backlog":   float64(report.FinalBacklog),
		"peak_in_flight":  float64(report.PeakInFlight),
		"mean_in_flight":  report.MeanInFlight,
		"num_windows":     float64(len(report.Windows)),
	}

	// Score: Use normalized slope magnitude (absolute value, capped at 1.0)
	// Positive slope → overload, negative slope → recovery, zero → stable
	score := 0.0
	if report.Slope > 0 {
		// Normalize slope to [0, 1] range
		// Use slopeUpper as reference for scaling (upper bound of CI)
		if report.SlopeUpper > 0 {
			score = report.Slope / report.SlopeUpper
			if score > 1.0 {
				score = 1.0
			}
		}
	}

	return Result{
		Level:      level,
		Score:      score,
		Confidence: confidence,
		Signals:    signals,
	}
}

// Reset clears accumulated state (no-op for stateless batch analyzer).
func (b *BacklogDriftDetector) Reset() {
	// No-op: backlog-drift is stateless
}
