package saturation

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

type DetectorOpts struct {
	ThresholdMs float64
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
	case "none":
		return &NoOpDetector{}
	default:
		panic(fmt.Sprintf("unknown saturation detector %q", name))
	}
}

type NoOpDetector struct{}

func (n *NoOpDetector) Name() string                                    { return "none" }
func (n *NoOpDetector) Observe(event Event)                             {}
func (n *NoOpDetector) Detect() Result                                  { return Result{Level: Stable, Score: 0, Confidence: 0, Signals: make(map[string]float64)} }
func (n *NoOpDetector) Classify(requests []sim.RequestMetrics) Result { return Result{Level: Stable, Score: 0, Confidence: 0, Signals: make(map[string]float64)} }
func (n *NoOpDetector) Reset()                                          {}

func NewCompositeDetector() Detector {
	return &CompositeDetectorStub{}
}

func NewThresholdDetector(thresholdMs float64) Detector {
	return &ThresholdDetectorStub{thresholdMs: thresholdMs}
}

// CompositeDetectorStub is a temporary stub returning correct name
type CompositeDetectorStub struct{}

func (c *CompositeDetectorStub) Name() string                                    { return "composite" }
func (c *CompositeDetectorStub) Observe(event Event)                             {}
func (c *CompositeDetectorStub) Detect() Result                                  { return Result{Level: Stable, Score: 0, Confidence: 0, Signals: make(map[string]float64)} }
func (c *CompositeDetectorStub) Classify(requests []sim.RequestMetrics) Result { return Result{Level: Stable, Score: 0, Confidence: 0, Signals: make(map[string]float64)} }
func (c *CompositeDetectorStub) Reset()                                          {}

// ThresholdDetectorStub is a temporary stub returning correct name
type ThresholdDetectorStub struct {
	thresholdMs float64
}

func (t *ThresholdDetectorStub) Name() string                                    { return "threshold" }
func (t *ThresholdDetectorStub) Observe(event Event)                             {}
func (t *ThresholdDetectorStub) Detect() Result                                  { return Result{Level: Stable, Score: 0, Confidence: 0, Signals: make(map[string]float64)} }
func (t *ThresholdDetectorStub) Classify(requests []sim.RequestMetrics) Result { return Result{Level: Stable, Score: 0, Confidence: 0, Signals: make(map[string]float64)} }
func (t *ThresholdDetectorStub) Reset()                                          {}
