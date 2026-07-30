// sim/saturation/backlog_drift.go
package saturation

import (
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// BacklogDriftDetector wraps the workload.AnalyzeBacklogDriftWithClassifier logic
// as a post-hoc saturation detector (Issue #7).
// This detector is stateless - it performs regression analysis over
// completed request intervals during Classify().
//
// The classifier is configurable (#1392): defaults to drain-ratio (matching the
// CLI default for --saturation-classifier) so that the post-hoc detector path
// stays consistent with the primary --saturation-report path. Pass an explicit
// classifier via NewBacklogDriftDetectorWithClassifier when slope-based behavior
// is preferred.
type BacklogDriftDetector struct {
	config     workload.BacklogDriftConfig
	classifier workload.BacklogClassifier

	// Live-detection buffers (LiveDetector). backlog-drift cannot stream a running
	// verdict — its classification is a windowed regression over the whole observed
	// history — so it buffers events and re-runs the batch analysis at each LabelAt.
	// arrivalUs maps request id → arrival time (µs); e2eUs maps id → end-to-end
	// latency (µs), present only once the request has completed.
	arrivalUs map[string]int64
	e2eUs     map[string]int64
}

// NewBacklogDriftDetector creates a BacklogDriftDetector with default configuration
// and the default classifier (drain-ratio, matching --saturation-classifier default).
func NewBacklogDriftDetector() Detector {
	return &BacklogDriftDetector{
		config:     workload.DefaultBacklogDriftConfig(),
		classifier: workload.NewBacklogClassifier(""), // empty string → drain-ratio default
		arrivalUs:  make(map[string]int64),
		e2eUs:      make(map[string]int64),
	}
}

// NewBacklogDriftDetectorWithClassifier creates a BacklogDriftDetector with an
// explicit classifier. Use this for callers that want to opt into slope-based
// or any future BacklogClassifier implementation.
func NewBacklogDriftDetectorWithClassifier(classifier workload.BacklogClassifier) Detector {
	if classifier == nil {
		classifier = workload.NewBacklogClassifier("")
	}
	return &BacklogDriftDetector{
		config:     workload.DefaultBacklogDriftConfig(),
		classifier: classifier,
		arrivalUs:  make(map[string]int64),
		e2eUs:      make(map[string]int64),
	}
}

// NewBacklogDriftDetectorWithConfig creates a BacklogDriftDetector with an explicit
// config and classifier. Used by the live/per-interval path so backlog-drift uses the
// SAME window size, MinWindows, and classifier as the CLI saturation flags configure
// for the end-of-run report (rather than library defaults). A nil classifier defaults
// to drain-ratio.
func NewBacklogDriftDetectorWithConfig(config workload.BacklogDriftConfig, classifier workload.BacklogClassifier) Detector {
	if classifier == nil {
		classifier = workload.NewBacklogClassifier("")
	}
	return &BacklogDriftDetector{
		config:     config,
		classifier: classifier,
		arrivalUs:  make(map[string]int64),
		e2eUs:      make(map[string]int64),
	}
}

func (b *BacklogDriftDetector) Name() string {
	return "backlog-drift"
}

// Observe buffers arrival/completion events for live (per-tick batch) detection.
// backlog-drift has no single-event streaming state — it re-runs its windowed
// regression over the buffered history at each LabelAt (see the type doc). The
// batch Classify path is unaffected; it ignores these buffers.
func (b *BacklogDriftDetector) Observe(event Event) {
	switch event.Type {
	case Arrival:
		b.arrivalUs[event.RequestID] = event.Timestamp
	case Completion:
		// LatencyMs → µs. FirstTokenTime carries the whole E2E in the reconstructed
		// request (matching the batch Classify convention below).
		b.e2eUs[event.RequestID] = int64(event.LatencyMs * 1000.0)
	}
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

	// Run backlog-drift analysis with the configured classifier (#1392).
	report := workload.AnalyzeBacklogDriftWithClassifier(reqs, simEndUs, b.config, b.classifier)
	return b.reportToResult(report)
}

// reportToResult maps a workload.BacklogDriftReport into a saturation.Result:
// classification → Level, window count → confidence, normalized slope → score.
// Shared by the batch Classify path and the live LabelAt path.
func (b *BacklogDriftDetector) reportToResult(report workload.BacklogDriftReport) Result {
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

	// Confidence based on number of windows analyzed: min(1.0, windows / MinWindows).
	confidence := 0.0
	if len(report.Windows) >= b.config.MinWindows {
		confidence = 1.0
	} else if b.config.MinWindows > 0 {
		confidence = float64(len(report.Windows)) / float64(b.config.MinWindows)
	}

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

	// Score: normalized positive-slope magnitude, capped at 1.0.
	score := 0.0
	if report.Slope > 0 && report.SlopeUpper > 0 {
		score = report.Slope / report.SlopeUpper
		if score > 1.0 {
			score = 1.0
		}
	}

	return Result{Level: level, Score: score, Confidence: confidence, Signals: signals}
}

// LabelAt implements LiveDetector. backlog-drift cannot stream a running verdict, so
// it reconstructs the completed/in-flight requests from its buffered events (those
// with arrival ≤ clockUs) and re-runs the SAME windowed regression it uses in batch
// mode, with simEndUs = clockUs. A request whose completion is known and ≤ clockUs is
// completed; one that arrived by clockUs but has no completion yet (or completes
// after clockUs) is in-flight — represented like the batch path's still-running case.
func (b *BacklogDriftDetector) LabelAt(clockUs int64, arrivals, completions int, cfg TimelineConfig) TimelinePoint {
	reqs := make([]*sim.Request, 0, len(b.arrivalUs))
	for id, arr := range b.arrivalUs {
		if arr > clockUs {
			continue // not yet arrived as of this boundary
		}
		e2e, done := b.e2eUs[id]
		if done && arr+e2e <= clockUs {
			// Completed by clockUs.
			reqs = append(reqs, &sim.Request{
				ID: id, ArrivalTime: arr, TTFTSet: true,
				FirstTokenTime: e2e, ITL: []int64{}, State: sim.StateCompleted,
			})
		} else {
			// Arrived but not yet completed as of clockUs — in-flight (backlog evidence).
			// TTFTSet=false + StateRunning mirrors the batch path's still-running case,
			// which RequestsToIntervals treats as active through simEndUs (#1389).
			reqs = append(reqs, &sim.Request{
				ID: id, ArrivalTime: arr, TTFTSet: false,
				ITL: []int64{}, State: sim.StateRunning,
			})
		}
	}

	report := workload.AnalyzeBacklogDriftWithClassifier(reqs, clockUs, b.config, b.classifier)
	res := b.reportToResult(report)
	return TimelinePoint{
		ClockUs:     clockUs,
		Label:       LabelFromResult(res, arrivals, cfg),
		Level:       res.Level,
		Score:       res.Score,
		Confidence:  res.Confidence,
		Arrivals:    arrivals,
		Completions: completions,
	}
}

// Reset clears accumulated live-detection buffers (batch Classify is stateless).
func (b *BacklogDriftDetector) Reset() {
	b.arrivalUs = make(map[string]int64)
	b.e2eUs = make(map[string]int64)
}
