// sim/saturation/composite.go
package saturation

import (
	"math"
	"sort"
)

// defaultCompositeSensitivity is composite's noise-floor multiplier when no
// composite.sensitivity override is supplied. 1.0 means "use the historical
// 1/sqrt(arrivals) floor unchanged", so the knob is inert by default (INV-6).
const defaultCompositeSensitivity = 1.0

// CompositeDetector combines rate deficit and latency trend signals using max() composition
// with quartile-monotonicity filter and noise-floor thresholds (validated across 640+ experiments).
type CompositeDetector struct {
	arrivals    []Event
	completions []Event

	// sensitivity multiplies the noise floor, giving composite the ONE tunable
	// knob it needs to be calibrated to a target false-alarm rate. Detector
	// scores are only comparable when every detector sits at the same
	// false-alarm rate, and a detector with no knob cannot be moved onto that
	// rate -- it can only be disqualified (#1614).
	//
	// A LARGER value raises the bar, so the detector fires LESS. 1.0 reproduces
	// the pre-#1614 behaviour exactly: IEEE-754 guarantees 1.0*x == x for every
	// finite x, so an absent composite: block is byte-identical (INV-6).
	sensitivity float64
}

// NewCompositeDetector creates a composite detector at its default sensitivity
// (1.0), which is byte-identical to the pre-#1614 detector.
func NewCompositeDetector() Detector {
	return NewCompositeDetectorWithSensitivity(defaultCompositeSensitivity)
}

// NewCompositeDetectorWithSensitivity is the canonical constructor (R4): both
// exported constructors route through it so the struct literal lives in exactly
// one place.
//
// A non-positive or non-finite sensitivity cannot arrive from user config --
// buildDetector rejects it with an error naming the YAML field (R6) before
// construction. The clamp here is the in-process safety net for direct callers,
// so the detector can never divide its banding by a nonsense floor.
func NewCompositeDetectorWithSensitivity(sensitivity float64) Detector {
	if sensitivity <= 0 || math.IsNaN(sensitivity) || math.IsInf(sensitivity, 0) {
		sensitivity = defaultCompositeSensitivity
	}
	return &CompositeDetector{
		arrivals:    make([]Event, 0),
		completions: make([]Event, 0),
		sensitivity: sensitivity,
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

	// Sort completions by timestamp (completion time), not by latency value
	sorted := make([]Event, len(c.completions))
	copy(sorted, c.completions)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Timestamp < sorted[j].Timestamp
	})

	// Extract latencies in temporal order
	sortedLatencies := make([]float64, len(sorted))
	for i, e := range sorted {
		sortedLatencies[i] = e.LatencyMs
	}

	return computeComposite(arrivals, completions, sortedLatencies, c.sensitivity)
}

// Reset clears accumulated state for fresh detection.
func (c *CompositeDetector) Reset() {
	c.arrivals = make([]Event, 0)
	c.completions = make([]Event, 0)
}

// computeComposite is the core validated algorithm from the empirical spec.
// Issues #1-3: Uses max() composition, quartile filter, and noise-floor thresholds.
//
// sensitivity scales the noise floor (#1614). Pass 1.0 for the historical
// behaviour; larger values make the detector fire less.
func computeComposite(arrivals, completions int, sortedLatencies []float64, sensitivity float64) Result {
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
		// 2a: Raw LT — always compute for diagnostics (reported in signals)
		mid := n / 2
		lFirst := mean(sortedLatencies[:mid])
		lSecond := mean(sortedLatencies[mid:])
		if lFirst > 0 {
			ltRaw = math.Max(0.0, (lSecond-lFirst)/lFirst)
		}

		// 2b: LT only affects classification when quartile filter can validate it
		if n >= 20 {
			qSize := n / 4
			q1 := mean(sortedLatencies[0:qSize])
			q2 := mean(sortedLatencies[qSize : 2*qSize])
			q3 := mean(sortedLatencies[2*qSize : 3*qSize])
			q4 := mean(sortedLatencies[3*qSize:])
			quartileMonotone = (q1 < q2) && (q2 < q3) && (q3 < q4)

			if quartileMonotone {
				lt = ltRaw
			}
			// If !quartileMonotone: lt stays 0 (filter vetoed)
		}
		// If n < 20: lt stays 0 (insufficient data for reliable trend)
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
	// Scale BEFORE recording the signal, so the trace reports the floor actually
	// used by the banding below rather than an unscaled value that would explain
	// a verdict the detector did not reach.
	noiseFloor *= sensitivity
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
