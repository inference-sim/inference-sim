// sim/saturation/composite.go
package saturation

import (
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// CompositeDetector combines rate deficit and latency trend signals.
// BC-1: STABLE when both signals below threshold (score < 0.5)
// BC-2: BACKLOGGED when one signal saturated (0.5 <= score < 0.75)
// BC-3: OVERLOADED when both signals saturated (score >= 0.75)
type CompositeDetector struct {
	arrivals    []Event
	completions []Event
}

// NewCompositeDetector creates a composite detector with zero parameters.
func NewCompositeDetector() Detector {
	return &CompositeDetector{
		arrivals:    make([]Event, 0),
		completions: make([]Event, 0),
	}
}

func (c *CompositeDetector) Name() string {
	return "composite"
}

// Observe records an arrival or completion event for streaming detection.
func (c *CompositeDetector) Observe(event Event) {
	if event.Type == Arrival {
		c.arrivals = append(c.arrivals, event)
	} else if event.Type == Completion {
		c.completions = append(c.completions, event)
	}
}

// Detect analyzes accumulated events for streaming detection.
func (c *CompositeDetector) Detect() Result {
	if len(c.arrivals) == 0 {
		return Result{Level: Stable, Score: 0, Confidence: 0, Signals: make(map[string]float64)}
	}

	// Calculate rate deficit: max(0, 1 - completions/arrivals)
	completionRate := float64(len(c.completions)) / float64(len(c.arrivals))
	rateDeficit := math.Max(0, 1.0-completionRate)

	// Calculate latency trend: (second_half_mean - first_half_mean) / first_half_mean
	latencyTrend := 0.0
	if len(c.completions) >= 2 {
		midpoint := len(c.completions) / 2
		firstHalf := c.completions[:midpoint]
		secondHalf := c.completions[midpoint:]

		firstMean := meanLatency(firstHalf)
		secondMean := meanLatency(secondHalf)

		if firstMean > 0 {
			latencyTrend = (secondMean - firstMean) / firstMean
		}
	}

	// Classify based on both signals
	score, level := classifyComposite(rateDeficit, latencyTrend)

	// Confidence inversely proportional to 1/sqrt(N) noise floor
	confidence := math.Min(1.0, math.Sqrt(float64(len(c.arrivals))))

	return Result{
		Level:      level,
		Score:      score,
		Confidence: confidence,
		Signals: map[string]float64{
			"rate_deficit":  rateDeficit,
			"latency_trend": latencyTrend,
		},
	}
}

// Classify performs batch post-hoc classification on completed requests.
func (c *CompositeDetector) Classify(requests []sim.RequestMetrics) Result {
	if len(requests) == 0 {
		return Result{Level: Stable, Score: 0, Confidence: 0, Signals: make(map[string]float64)}
	}

	// For post-hoc classification, we only have completed requests
	// Rate deficit: we assume arrivals == completions for batch analysis
	// (The issue says Classify signature needs totalArrivals, but current interface doesn't have it)
	// For now, we set rate deficit to 0 for batch mode
	rateDeficit := 0.0

	// Calculate latency trend: (second_half_mean - first_half_mean) / first_half_mean
	latencyTrend := 0.0
	if len(requests) >= 2 {
		midpoint := len(requests) / 2
		firstHalf := requests[:midpoint]
		secondHalf := requests[midpoint:]

		firstMean := meanE2E(firstHalf)
		secondMean := meanE2E(secondHalf)

		if firstMean > 0 {
			latencyTrend = (secondMean - firstMean) / firstMean
		}
	}

	// Classify based on both signals
	score, level := classifyComposite(rateDeficit, latencyTrend)

	// Confidence inversely proportional to 1/sqrt(N) noise floor
	confidence := math.Min(1.0, math.Sqrt(float64(len(requests))))

	return Result{
		Level:      level,
		Score:      score,
		Confidence: confidence,
		Signals: map[string]float64{
			"rate_deficit":  rateDeficit,
			"latency_trend": latencyTrend,
		},
	}
}

// Reset clears accumulated state for fresh detection.
func (c *CompositeDetector) Reset() {
	c.arrivals = make([]Event, 0)
	c.completions = make([]Event, 0)
}

// classifyComposite determines level and score from two signals.
// BC-1: STABLE when both signals below threshold (score < 0.5)
// BC-2: BACKLOGGED when one signal saturated (0.5 <= score < 0.75)
// BC-3: OVERLOADED when both signals saturated (score >= 0.75)
func classifyComposite(rateDeficit, latencyTrend float64) (float64, Level) {
	// Normalize signals to [0, 1] range
	// Rate deficit already in [0, 1]
	// Latency trend: normalize assuming trend > 0.5 is saturated
	normalizedLatencyTrend := math.Min(1.0, math.Max(0, latencyTrend/0.5))

	// Score is average of both signals
	score := (rateDeficit + normalizedLatencyTrend) / 2.0

	// Classify based on score thresholds
	if score < 0.5 {
		return score, Stable
	} else if score < 0.75 {
		return score, Backlogged
	} else {
		return score, Overloaded
	}
}

// meanLatency calculates mean latency from completion events.
func meanLatency(events []Event) float64 {
	if len(events) == 0 {
		return 0
	}
	sum := 0.0
	for _, e := range events {
		sum += e.LatencyMs
	}
	return sum / float64(len(events))
}

// meanE2E calculates mean E2E latency from request metrics.
func meanE2E(requests []sim.RequestMetrics) float64 {
	if len(requests) == 0 {
		return 0
	}
	sum := 0.0
	for _, r := range requests {
		sum += r.E2E
	}
	return sum / float64(len(requests))
}
