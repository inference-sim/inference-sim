package saturation

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

type DetectorOpts struct {
	ThresholdMs float64
}

// ValidDetectorNames returns the set of recognized post-hoc detector names.
// Used by CLI validation in cmd/root.go, cmd/replay.go, and cmd/observe_cmd.go.
func ValidDetectorNames() map[string]bool {
	return map[string]bool{
		"none":           true,
		"composite":      true,
		"threshold":      true,
		"backlog-drift":  true,
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
		return NewBacklogDriftDetector()
	case "none":
		return &NoOpDetector{}
	}
	panic(fmt.Sprintf("unknown saturation detector %q", name))
}

type NoOpDetector struct{}

func (n *NoOpDetector) Name() string                { return "none" }
func (n *NoOpDetector) Observe(event Event)         {}
func (n *NoOpDetector) Detect() Result              { return Result{Level: Stable, Score: 0, Confidence: 0, Signals: make(map[string]float64)} }
func (n *NoOpDetector) Classify(requests []sim.RequestMetrics, totalArrivals int) interface{} {
	return Result{Level: Stable, Score: 0, Confidence: 0, Signals: make(map[string]float64)}
}
func (n *NoOpDetector) Reset() {}

