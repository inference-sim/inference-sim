// sim/saturation/backlog_drift.go
package saturation

import (
	"math"
	"time"
)

// backlogDriftSlopeK is the DEFAULT "clearly rising" multiplier for the streaming
// band classifier (#1515): running_slope in (noiseFloor, K*noiseFloor] →
// BACKLOGGED, running_slope > K*noiseFloor → OVERLOADED.
//
// It remains a heuristic rather than an empirically calibrated value, which is
// exactly why it is now only a default: as of #1614 it is overridable via
// `backlog_drift.slope_k`, so an operator can calibrate the detector to a target
// false-alarm rate instead of inheriting this number. Read the effective value
// through BacklogDriftConfig.effectiveSlopeK(), never this constant directly.
const backlogDriftSlopeK = 3.0

// BacklogDriftDetector is a streaming saturation detector (#1515): Observe folds
// each event into an incremental in-flight estimate and Detect bands the online
// OLS slope of in-flight against a noise floor. The batch Classify path was
// removed in #1516 and the post-hoc batch classifier library in #1547; the
// detector now streams exclusively (the online band is self-contained; #1517).
type BacklogDriftDetector struct {
	config BacklogDriftConfig

	// Streaming state (#1515). Populated by Observe, read by Detect, cleared by
	// Reset. This is a causal computation: it consumes events in order and never
	// looks ahead, distinct from the removed non-causal batch analysis (#1547)
	// which needed the whole trace including the tail.
	arrivals    int64 // running count of Arrival events
	completions int64 // running count of Completion events

	// In-flight samples, one per WindowSize bucket spanned. buckets[i] holds the
	// in-flight value at the end of bucket i; empty intervening buckets are
	// forward-filled with the last value so the samples stay evenly spaced,
	// letting Detect use bucket position as the OLS x-axis (scale-stable,
	// independent of absolute timestamp magnitude).
	//
	// Memory is O(buckets spanned) = O(elapsed_span / WindowSize), i.e. it grows
	// with the observation horizon, not the event count — a saving over O(events)
	// when buckets are densely populated, but NOT a fixed bound (a sparse stream
	// over a long horizon forward-fills many empty buckets). #1516 may add a
	// trailing cap when it wires this to output; today the whole span is kept.
	buckets       []int64
	curBucketIdx  int64 // absolute index of the bucket currently being filled
	curBucketInit bool  // whether curBucketIdx has been established yet
	windowSizeUs  int64 // WindowSize in microseconds, cached from config
}

// NewBacklogDriftDetector creates a BacklogDriftDetector with default configuration.
func NewBacklogDriftDetector() Detector {
	return newBacklogDriftDetector(DefaultBacklogDriftConfig())
}

// NewBacklogDriftDetectorWithConfig creates a BacklogDriftDetector with an
// explicit config (#1515). The config's WindowSize governs the streaming bucket
// width; the default-config constructor hardwires the 60s production window, which
// is impractical for driving the streaming slope in a unit test. Callers (and
// #1516) pass a small WindowSize via NewBacklogDriftConfig so a handful of
// directly-fed events span enough buckets to exercise the online slope
// deterministically.
func NewBacklogDriftDetectorWithConfig(config BacklogDriftConfig) Detector {
	return newBacklogDriftDetector(config)
}

// newBacklogDriftDetector is the canonical constructor (R4): all exported
// constructors route through it so streaming state (windowSizeUs) is initialized
// in exactly one place.
func newBacklogDriftDetector(config BacklogDriftConfig) Detector {
	return &BacklogDriftDetector{
		config:       config,
		windowSizeUs: int64(config.WindowSize / time.Microsecond),
	}
}

func (b *BacklogDriftDetector) Name() string {
	return "backlog-drift"
}

// Observe records an arrival or completion event and folds it into the running
// in-flight count and the bucketed trailing-window samples (#1515). This is a
// causal, incremental computation — it accumulates only counts and per-bucket
// snapshots over the events it is fed, in order (no clock, no map iteration), so
// it is deterministic.
func (b *BacklogDriftDetector) Observe(event Event) {
	switch event.Type {
	case Arrival:
		b.arrivals++
	case Completion:
		b.completions++
	default:
		return // ignore unknown event types (no in-flight change)
	}

	// Map the event timestamp to its bucket index. Guard against a zero/negative
	// windowSizeUs (degenerate config) by falling back to a single bucket.
	bucketIdx := int64(0)
	if b.windowSizeUs > 0 {
		bucketIdx = event.Timestamp / b.windowSizeUs
	}

	inFlight := b.arrivals - b.completions

	if !b.curBucketInit {
		// First observed event establishes the first bucket.
		b.curBucketIdx = bucketIdx
		b.curBucketInit = true
		b.buckets = append(b.buckets, inFlight)
		return
	}

	if bucketIdx == b.curBucketIdx {
		// Same bucket: overwrite with the latest in-flight value (end-of-bucket).
		b.buckets[len(b.buckets)-1] = inFlight
		return
	}

	// Advanced to a later bucket. Carry the last value forward across any empty
	// intervening buckets so samples stay evenly spaced (one per bucket spanned).
	// Events must arrive in non-decreasing timestamp order; an out-of-order
	// earlier event folds into the current bucket rather than rewriting history.
	if bucketIdx > b.curBucketIdx {
		lastVal := b.buckets[len(b.buckets)-1]
		for gap := b.curBucketIdx + 1; gap < bucketIdx; gap++ {
			b.buckets = append(b.buckets, lastVal)
		}
		b.buckets = append(b.buckets, inFlight)
		b.curBucketIdx = bucketIdx
	} else {
		// Out-of-order (earlier) event: fold into the current bucket.
		b.buckets[len(b.buckets)-1] = inFlight
	}
}

// Detect computes an evolving per-event verdict from the streaming state (#1515):
// an online OLS slope of in-flight over the trailing window, banded against a
// noise floor. This is an online heuristic; the earlier batch drain-ratio/
// slope-based analysis it superseded (formerly in sim/workload) was removed in
// #1547 once the streaming detector had no live-path caller.
func (b *BacklogDriftDetector) Detect() Result {
	signals := make(map[string]float64)

	if b.arrivals == 0 {
		// No arrivals observed → nothing to say (R20: no panic on empty input).
		return Result{Level: Stable, Score: 0, Confidence: 0, Signals: signals}
	}

	inFlight := b.arrivals - b.completions

	// noise_floor mirrors composite: 1/√arrivals.
	noiseFloor := 1.0 / math.Sqrt(float64(b.arrivals))

	// running_slope: OLS slope of in-flight per window bucket over the trailing
	// samples, using bucket position (0,1,2,…) as the x-axis. Bucket-indexed (not
	// per-microsecond) so the value is scale-stable and independent of absolute
	// timestamp magnitude. Fewer than 2 samples ⇒ slope 0.
	runningSlope := onlineSlope(b.buckets)

	signals["in_flight"] = float64(inFlight)
	signals["arrivals"] = float64(b.arrivals)
	signals["completions"] = float64(b.completions)
	signals["running_slope"] = runningSlope
	signals["noise_floor"] = noiseFloor

	// The effective band multiplier, hoisted once so the band switch below and the
	// score denominator further down provably use the SAME value (#1614): if they
	// diverged, Score==1.0 would stop coinciding with the OVERLOADED boundary.
	slopeK := b.config.effectiveSlopeK()
	// Reported ONLY when the knob was explicitly configured. The Signals map is
	// serialized into --saturation-report, so emitting it unconditionally would make
	// a default-configured report differ from a pre-#1614 one -- breaking the
	// absent-config byte-identity this PR promises (INV-6) for the sake of a
	// diagnostic that just restates the documented default. When the operator HAS
	// tuned it, the trace must explain which multiplier produced the band.
	if b.config.SlopeK > 0 {
		signals["slope_k"] = slopeK
	}

	// Level bands mirror composite's two-threshold structure:
	//   slope <= noiseFloor            → STABLE
	//   noiseFloor < slope <= K·noise  → BACKLOGGED
	//   slope > K·noiseFloor           → OVERLOADED
	var level Level
	switch {
	case runningSlope <= noiseFloor:
		level = Stable
	case runningSlope <= slopeK*noiseFloor:
		level = Backlogged
	default:
		level = Overloaded
	}

	// Score: normalized slope magnitude in [0,1] — min(1, max(0, slope)/(K·noise)) —
	// so it crosses ~1.0 exactly as the level reaches OVERLOADED. Draining
	// (negative slope) ⇒ 0. Mirrors Classify's "normalized slope magnitude,
	// capped at 1.0" convention.
	//
	// Note the shared boundary: at exactly slope == K·noiseFloor the band switch
	// (<=) still reports BACKLOGGED while the score reaches 1.0, so Score==1.0 can
	// co-occur with Level==BACKLOGGED. This is a measure-zero float coincidence,
	// and both the band inequality and the score formula are pinned verbatim by
	// #1515 — kept as-is so callers get the contracted values rather than a
	// locally-nudged epsilon; Score is a magnitude, Level is the authoritative
	// band.
	// The denominator is the OVERLOADED boundary, so Score reaching its 1.0 cap
	// must coincide with Level crossing that boundary. A subnormal slope_k can
	// drive the product to exactly zero even though slope_k itself is positive and
	// finite, which would leave Score at 0 while Level is OVERLOADED -- Level and
	// Score decoupled. When the product underflows, the boundary is effectively
	// zero, so any positive slope is past it: report the cap.
	score := 0.0
	denom := slopeK * noiseFloor
	switch {
	case denom > 0:
		score = math.Min(1.0, math.Max(0.0, runningSlope)/denom)
	case runningSlope > 0:
		// Boundary underflowed to zero and the slope is rising: the OVERLOADED
		// band starts at zero, so the magnitude is saturated by construction.
		score = 1.0
	}

	// Confidence reuses composite's ramp so the streaming detectors agree.
	confidence := math.Min(1.0, float64(b.arrivals)/20.0)

	return Result{
		Level:      level,
		Score:      score,
		Confidence: confidence,
		Signals:    signals,
	}
}

// onlineSlope computes the ordinary-least-squares slope of the sample values
// against their evenly-spaced positions (x = 0,1,2,…). Returns 0 for fewer than
// 2 samples or when the x-variance is zero (R11: guarded denominator).
func onlineSlope(samples []int64) float64 {
	n := len(samples)
	if n < 2 {
		return 0
	}
	// x = 0..n-1. sumX and sumXX have closed forms but a loop keeps it obvious.
	var sumX, sumY, sumXY, sumXX float64
	for i, v := range samples {
		x := float64(i)
		y := float64(v)
		sumX += x
		sumY += y
		sumXY += x * y
		sumXX += x * x
	}
	fn := float64(n)
	denom := fn*sumXX - sumX*sumX
	if denom == 0 {
		return 0
	}
	return (fn*sumXY - sumX*sumY) / denom
}

// Reset clears the streaming state (#1515), returning the detector to its
// initial state: next Detect() on no events → STABLE, zero confidence.
func (b *BacklogDriftDetector) Reset() {
	b.arrivals = 0
	b.completions = 0
	b.buckets = nil
	b.curBucketIdx = 0
	b.curBucketInit = false
}
