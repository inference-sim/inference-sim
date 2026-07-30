package saturation

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

type DetectorOpts struct {
	ThresholdMs float64

	// BacklogConfig and BacklogClassifier configure the backlog-drift detector from
	// the CLI saturation flags (--saturation-window, --saturation-classifier, etc.),
	// so the live/per-interval backlog-drift path uses the SAME configuration as the
	// end-of-run report rather than library defaults. Zero value ⇒ defaults. Ignored
	// by composite/threshold.
	BacklogConfig     *workload.BacklogDriftConfig
	BacklogClassifier workload.BacklogClassifier
}

// ValidDetectorNames returns the set of recognized post-hoc detector names.
// Used by CLI validation in cmd/root.go, cmd/replay.go, and cmd/observe_cmd.go.
func ValidDetectorNames() map[string]bool {
	return map[string]bool{
		"none":          true,
		"composite":     true,
		"threshold":     true,
		"backlog-drift": true,
	}
}

func NewDetector(name string, opts DetectorOpts) Detector {
	switch name {
	case "composite":
		return NewCompositeDetector()
	case "threshold":
		threshold := opts.ThresholdMs
		if threshold == 0 {
			threshold = 5000.0
		}
		return NewThresholdDetector(threshold)
	case "backlog-drift":
		if opts.BacklogConfig != nil {
			return NewBacklogDriftDetectorWithConfig(*opts.BacklogConfig, opts.BacklogClassifier)
		}
		return NewBacklogDriftDetector()
	case "none":
		return &NoOpDetector{}
	}
	panic(fmt.Sprintf("unknown saturation detector %q", name))
}

// NewLiveDetector returns the LiveDetector for name, parallel to NewDetector. The
// concrete detectors (composite, threshold, backlog-drift, none) all satisfy both
// Detector and LiveDetector, so this is a typed view of the same constructors —
// callers wiring the live per-interval timeline use this. Panics on an unknown name
// (CLI validates against ValidDetectorNames first).
func NewLiveDetector(name string, opts DetectorOpts) LiveDetector {
	return NewDetector(name, opts).(LiveDetector)
}

type NoOpDetector struct{}

func (n *NoOpDetector) Name() string        { return "none" }
func (n *NoOpDetector) Observe(event Event) {}
func (n *NoOpDetector) Detect() Result {
	return Result{Level: Stable, Score: 0, Confidence: 0, Signals: make(map[string]float64)}
}
func (n *NoOpDetector) Classify(requests []sim.RequestMetrics, totalArrivals int) interface{} {
	return Result{Level: Stable, Score: 0, Confidence: 0, Signals: make(map[string]float64)}
}
func (n *NoOpDetector) Reset() {}

// LabelAt implements LiveDetector: the no-op detector always reports Unsaturated
// (Stable level) with zero confidence. It is never wired into the live timeline in
// practice (the CLI skips the timeline when --post-hoc-detector is "none").
func (n *NoOpDetector) LabelAt(clockUs int64, arrivals, completions int, cfg TimelineConfig) TimelinePoint {
	return TimelinePoint{
		ClockUs:     clockUs,
		Label:       LabelFromResult(Result{Level: Stable}, arrivals, cfg),
		Level:       Stable,
		Arrivals:    arrivals,
		Completions: completions,
	}
}
