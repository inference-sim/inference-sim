// sim/saturation/backlog_drift.go
package saturation

import (
	"math"
	"time"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// backlogDriftSlopeK is the "clearly rising" multiplier for the streaming band
// classifier (#1515): running_slope in (noiseFloor, K*noiseFloor] → BACKLOGGED,
// running_slope > K*noiseFloor → OVERLOADED. It is a tunable heuristic constant,
// NOT an empirically calibrated value. The streaming bands are an online
// heuristic and are explicitly NOT the drain-ratio classifier used by Classify;
// the two computations may legitimately disagree.
const backlogDriftSlopeK = 3.0

// BacklogDriftDetector is a streaming saturation detector (#1515): Observe folds
// each event into an incremental in-flight estimate and Detect bands the online
// OLS slope of in-flight against a noise floor. The batch Classify path was
// removed in #1516; the detector now streams exclusively.
//
// The classifier field is retained for the online band computation's config
// (#1392); it defaults to drain-ratio. Pass an explicit classifier via
// NewBacklogDriftDetectorWithClassifier when slope-based behavior is preferred.
type BacklogDriftDetector struct {
	config     workload.BacklogDriftConfig
	classifier workload.BacklogClassifier

	// Streaming state (#1515). Populated by Observe, read by Detect, cleared by
	// Reset. This is a separate, causal computation from the non-causal batch
	// Classify path (which needs the whole trace including the tail).
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

// NewBacklogDriftDetector creates a BacklogDriftDetector with default configuration
// and the default classifier (drain-ratio, matching --saturation-classifier default).
func NewBacklogDriftDetector() Detector {
	return newBacklogDriftDetector(workload.DefaultBacklogDriftConfig(), nil)
}

// NewBacklogDriftDetectorWithClassifier creates a BacklogDriftDetector with an
// explicit classifier. Use this for callers that want to opt into slope-based
// or any future BacklogClassifier implementation.
func NewBacklogDriftDetectorWithClassifier(classifier workload.BacklogClassifier) Detector {
	return newBacklogDriftDetector(workload.DefaultBacklogDriftConfig(), classifier)
}

// NewBacklogDriftDetectorWithConfig creates a BacklogDriftDetector with an
// explicit config and the default classifier (#1515). The config's WindowSize
// governs the streaming bucket width; the two default-config constructors
// hardwire the 60s production window, which is impractical for driving the
// streaming slope in a unit test. Callers (and #1516) pass a small WindowSize
// via workload.NewBacklogDriftConfig so a handful of directly-fed events span
// enough buckets to exercise the online slope deterministically.
func NewBacklogDriftDetectorWithConfig(config workload.BacklogDriftConfig) Detector {
	return newBacklogDriftDetector(config, nil)
}

// newBacklogDriftDetector is the canonical constructor (R4): all exported
// constructors route through it so streaming state (windowSizeUs) is initialized
// in exactly one place. A nil classifier defaults to drain-ratio.
func newBacklogDriftDetector(config workload.BacklogDriftConfig, classifier workload.BacklogClassifier) Detector {
	if classifier == nil {
		classifier = workload.NewBacklogClassifier("") // empty string → drain-ratio default
	}
	return &BacklogDriftDetector{
		config:       config,
		classifier:   classifier,
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
// noise floor. This is an online heuristic, explicitly NOT the drain-ratio
// classifier that Classify runs — the two may legitimately disagree.
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

	// Level bands mirror composite's two-threshold structure:
	//   slope <= noiseFloor            → STABLE
	//   noiseFloor < slope <= K·noise  → BACKLOGGED
	//   slope > K·noiseFloor           → OVERLOADED
	var level Level
	switch {
	case runningSlope <= noiseFloor:
		level = Stable
	case runningSlope <= backlogDriftSlopeK*noiseFloor:
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
	score := 0.0
	denom := backlogDriftSlopeK * noiseFloor
	if denom > 0 {
		score = math.Min(1.0, math.Max(0.0, runningSlope)/denom)
	}

	// Confidence reuses composite's ramp so the three streaming detectors agree.
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
