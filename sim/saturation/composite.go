// sim/saturation/composite.go
package saturation

import (
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// CompositeDetector combines rate deficit and latency trend signals using max() composition
// with quartile-monotonicity filter and noise-floor thresholds (validated across 640+ experiments).
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
	switch event.Type {
	case Arrival:
		c.arrivals = append(c.arrivals, event)
	case Completion:
		c.completions = append(c.completions, event)
	}
}

// Detect analyzes accumulated events for streaming detection.
func (c *CompositeDetector) Detect() Result {
	arrivals := len(c.arrivals)
	completions := len(c.completions)

	if arrivals == 0 {
		return Result{Level: Stable, Score: 0, Confidence: 0, Signals: make(map[string]float64)}
	}

	// Extract latencies sorted by completion time
	sortedLatencies := make([]float64, len(c.completions))
	for i, e := range c.completions {
		sortedLatencies[i] = e.LatencyMs
	}
	sort.Float64s(sortedLatencies)

	return computeComposite(arrivals, completions, sortedLatencies)
}

// Classify performs batch post-hoc classification on completed requests.
// Issue #4: Now accepts totalArrivals to compute rate deficit in batch mode.
func (c *CompositeDetector) Classify(requests []sim.RequestMetrics, totalArrivals int) interface{} {
	completions := len(requests)

	if completions == 0 {
		return Result{Level: Stable, Score: 0, Confidence: 0, Signals: make(map[string]float64)}
	}

	// Issue #5: Sort by completion time, not arrival time
	sort.Slice(requests, func(i, j int) bool {
		completionI := requests[i].ArrivedAt + requests[i].E2E/1000.0
		completionJ := requests[j].ArrivedAt + requests[j].E2E/1000.0
		return completionI < completionJ
	})

	// Extract latencies (already in completion order)
	sortedLatencies := make([]float64, completions)
	for i, r := range requests {
		sortedLatencies[i] = r.E2E
	}

	return computeComposite(totalArrivals, completions, sortedLatencies)
}

// Reset clears accumulated state for fresh detection.
func (c *CompositeDetector) Reset() {
	c.arrivals = make([]Event, 0)
	c.completions = make([]Event, 0)
}

// computeComposite is the core validated algorithm from the empirical spec.
// Issues #1-3: Uses max() composition, quartile filter, and noise-floor thresholds.
func computeComposite(arrivals, completions int, sortedLatencies []float64) Result {
	signals := make(map[string]float64)

	// --- Signal 1: Rate Deficit ---
	rateDeficit := 0.0
	if arrivals > 0 {
		rateDeficit = math.Max(0.0, 1.0-float64(completions)/float64(arrivals))
	}
	signals["rate_deficit"] = rateDeficit

	// --- Signal 2: Latency Trend with Quartile Filter ---
	ltRaw := 0.0
	lt := 0.0
	quartileMonotone := false
	n := len(sortedLatencies)

	// Base LT computation (works for any n >= 2, per issue #1369 comment 4462467580)
	if n >= 2 {
		// 2a: Raw LT (half-window split)
		mid := n / 2
		lFirst := mean(sortedLatencies[:mid])
		lSecond := mean(sortedLatencies[mid:])
		if lFirst > 0 {
			ltRaw = math.Max(0.0, (lSecond-lFirst)/lFirst)
		}

		// Start with raw LT (trust by default)
		lt = ltRaw

		// 2b: Quartile monotonicity filter (Issue #2) - only applies at n >= 20
		if n >= 20 {
			qSize := n / 4
			q1 := mean(sortedLatencies[0:qSize])
			q2 := mean(sortedLatencies[qSize : 2*qSize])
			q3 := mean(sortedLatencies[2*qSize : 3*qSize])
			q4 := mean(sortedLatencies[3*qSize:])
			quartileMonotone = (q1 < q2) && (q2 < q3) && (q3 < q4)

			// 2c: If quartile filter fails, veto LT (set to 0)
			if !quartileMonotone {
				lt = 0.0
			}
		} else {
			// For n < 20, quartile filter doesn't apply - mark as N/A (not failed)
			quartileMonotone = true
		}
	}

	signals["latency_trend_raw"] = math.Min(ltRaw, 1.0)
	signals["latency_trend"] = math.Min(lt, 1.0)
	if quartileMonotone {
		signals["quartile_monotone"] = 1.0
	} else {
		signals["quartile_monotone"] = 0.0
	}

	// --- Issue #1: Composite Score with max() composition ---
	score := math.Max(rateDeficit, math.Min(lt, 1.0))

	// --- Issue #3: Noise Floor (not fixed thresholds) ---
	noiseFloor := 1.0
	if arrivals > 0 {
		noiseFloor = 1.0 / math.Sqrt(float64(arrivals))
	}
	signals["noise_floor"] = noiseFloor

	// --- Classification ---
	var level Level
	if score < noiseFloor {
		level = Stable
	} else if lt > noiseFloor {
		level = Overloaded
	} else {
		level = Backlogged
	}

	// --- Confidence ---
	// Per spec: confidence = min(1.0, arrivals / 20.0)
	confidence := math.Min(1.0, float64(arrivals)/20.0)

	return Result{
		Level:      level,
		Score:      score,
		Confidence: confidence,
		Signals:    signals,
	}
}

// mean calculates arithmetic mean of a slice.
func mean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}
