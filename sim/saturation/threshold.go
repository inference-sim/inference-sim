// sim/saturation/threshold.go
package saturation

import (
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// ThresholdDetector uses a simple mean E2E latency threshold.
// BC-4: STABLE when mean E2E < threshold
// BC-5: OVERLOADED when mean E2E > threshold
type ThresholdDetector struct {
	thresholdMs float64
	completions []Event
}

// NewThresholdDetector creates a threshold detector with the given threshold (ms).
// If thresholdMs is 0, uses default 5000ms.
func NewThresholdDetector(thresholdMs float64) Detector {
	if thresholdMs == 0 {
		thresholdMs = 5000.0
	}
	return &ThresholdDetector{
		thresholdMs: thresholdMs,
		completions: make([]Event, 0),
	}
}

func (t *ThresholdDetector) Name() string {
	return "threshold"
}

// Observe records completion events for streaming detection.
func (t *ThresholdDetector) Observe(event Event) {
	if event.Type == Completion {
		t.completions = append(t.completions, event)
	}
}

// Detect analyzes accumulated events for streaming detection.
func (t *ThresholdDetector) Detect() Result {
	if len(t.completions) == 0 {
		return Result{Level: Stable, Score: 0, Confidence: 0, Signals: make(map[string]float64)}
	}

	meanE2E := meanLatency(t.completions)
	score, level := classifyThreshold(meanE2E, t.thresholdMs)

	// Confidence formula: 1 - 1/sqrt(N+1), increases with sample size (C5 fix)
	confidence := 1.0 - 1.0/math.Sqrt(float64(len(t.completions))+1.0)

	return Result{
		Level:      level,
		Score:      score,
		Confidence: confidence,
		Signals: map[string]float64{
			"mean_e2e":  meanE2E,
			"threshold": t.thresholdMs,
		},
	}
}

// Classify performs batch post-hoc classification on completed requests.
func (t *ThresholdDetector) Classify(requests []sim.RequestMetrics) interface{} {
	if len(requests) == 0 {
		return Result{Level: Stable, Score: 0, Confidence: 0, Signals: make(map[string]float64)}
	}

	meanE2E := meanE2E(requests)
	score, level := classifyThreshold(meanE2E, t.thresholdMs)

	// Confidence formula: 1 - 1/sqrt(N+1), increases with sample size (C5 fix)
	confidence := 1.0 - 1.0/math.Sqrt(float64(len(requests))+1.0)

	return Result{
		Level:      level,
		Score:      score,
		Confidence: confidence,
		Signals: map[string]float64{
			"mean_e2e":  meanE2E,
			"threshold": t.thresholdMs,
		},
	}
}

// Reset clears accumulated state for fresh detection.
func (t *ThresholdDetector) Reset() {
	t.completions = make([]Event, 0)
}

// classifyThreshold determines level and score from mean E2E vs threshold.
// BC-4: STABLE when mean E2E < threshold (score < 0.5)
// BC-5: OVERLOADED when mean E2E > threshold (score >= 0.75)
func classifyThreshold(meanE2E, threshold float64) (float64, Level) {
	if meanE2E < threshold {
		// Stable: score in [0, 0.5) range, proportional to how close to threshold
		score := 0.5 * (meanE2E / threshold)
		return score, Stable
	} else {
		// Overloaded: score >= 0.75 for exceeding threshold
		score := 0.75 + math.Min(0.25, (meanE2E-threshold)/threshold/4.0)
		return score, Overloaded
	}
}
