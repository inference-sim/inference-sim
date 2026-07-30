// sim/saturation/live_test.go
package saturation

import (
	"testing"
)

// Compile-time contract: every concrete detector satisfies LiveDetector, so the
// pluggable interface stays honest as detectors are added (Task: pluggable interface).
var (
	_ LiveDetector = (*CompositeDetector)(nil)
	_ LiveDetector = (*ThresholdDetector)(nil)
	_ LiveDetector = (*BacklogDriftDetector)(nil)
	_ LiveDetector = (*NoOpDetector)(nil)
)

// feed drives a LiveTimeline through a scripted event sequence, emitting a point at
// each boundary. arrivals/completions are (id, clockUs[, e2eMs]) triples.
func TestLiveTimeline_EmitsPointPerBoundary(t *testing.T) {
	cfg := TimelineConfig{IntervalUs: 1_000_000, MinRequests: 0, MinConfidence: 0}
	lt := NewLiveTimeline(NewCompositeDetector().(LiveDetector), cfg)

	// 3 arrivals + 3 completions within [0, 3s].
	lt.ObserveArrival("r1", 0)
	lt.ObserveCompletion("r1", 500_000, 100)
	lt.EmitPoint(1_000_000)
	lt.ObserveArrival("r2", 1_200_000)
	lt.ObserveCompletion("r2", 1_800_000, 100)
	lt.EmitPoint(2_000_000)
	lt.ObserveArrival("r3", 2_100_000)
	lt.ObserveCompletion("r3", 2_900_000, 100)
	lt.EmitPoint(3_000_000)

	pts := lt.Points()
	if len(pts) != 3 {
		t.Fatalf("expected 3 points, got %d", len(pts))
	}
	// Cumulative counts are non-decreasing and reflect the running totals.
	wantArr := []int{1, 2, 3}
	wantComp := []int{1, 2, 3}
	for i, p := range pts {
		if p.Arrivals != wantArr[i] || p.Completions != wantComp[i] {
			t.Errorf("point %d: arr/comp = %d/%d, want %d/%d", i, p.Arrivals, p.Completions, wantArr[i], wantComp[i])
		}
		if p.ClockUs != int64((i+1)*1_000_000) {
			t.Errorf("point %d: clock = %d", i, p.ClockUs)
		}
	}
}

// TestLiveTimeline_CompositeSaturationFlips verifies the live composite detector
// flips to SATURATED once latency trends sharply upward (composite maps a monotone
// rising latency trend above the noise floor to OVERLOADED → SATURATED; a pure rate
// deficit with flat latency is only BACKLOGGED → UNSATURATED under our mapping).
func TestLiveTimeline_CompositeSaturationFlips(t *testing.T) {
	cfg := TimelineConfig{IntervalUs: 1_000_000_000, MinRequests: 5, MinConfidence: 0.2}
	lt := NewLiveTimeline(NewCompositeDetector().(LiveDetector), cfg)

	// 40 completions with steadily rising latency (100ms → ~4s): a strong, quartile-
	// monotone latency trend, which composite classifies as OVERLOADED.
	for i := 0; i < 40; i++ {
		arr := int64(i) * 100_000
		lt.ObserveArrival(idOf(i), arr)
		lt.ObserveCompletion(idOf(i), arr+1000, float64(100+i*100)) // rising E2E ms
	}
	lt.EmitPoint(1_000_000_000)

	last := lt.Points()[len(lt.Points())-1]
	if last.Label != Saturated {
		t.Errorf("rising latency: label = %v (level %v, score %.2f), want SATURATED", last.Label, last.Level, last.Score)
	}
}

// TestLiveBacklogDrift_ReconstructsRequests verifies backlog-drift's live path
// buffers events and produces a verdict from the re-run batch analysis (it does not
// panic on in-flight requests and returns a valid label).
func TestLiveBacklogDrift_ReconstructsRequests(t *testing.T) {
	cfg := TimelineConfig{IntervalUs: 10_000_000, MinRequests: 0, MinConfidence: 0}
	lt := NewLiveTimeline(NewBacklogDriftDetector().(LiveDetector), cfg)

	// 40 requests arriving over 40s, each completing 1s later.
	for i := 0; i < 40; i++ {
		arr := int64(i) * 1_000_000
		lt.ObserveArrival(idOf(i), arr)
		lt.ObserveCompletion(idOf(i), arr+1_000_000, 1000)
	}
	// Emit at 10s, 20s, 40s.
	lt.EmitPoint(10_000_000)
	lt.EmitPoint(20_000_000)
	lt.EmitPoint(40_000_000)

	pts := lt.Points()
	if len(pts) != 3 {
		t.Fatalf("expected 3 points, got %d", len(pts))
	}
	// Completions counted cumulatively (all 40 observed by the last EmitPoint since we
	// fed them up front; the per-boundary reconstruction filters by clock internally).
	for i, p := range pts {
		if p.Label != Unsaturated && p.Label != Saturated && p.Label != Unsure {
			t.Errorf("point %d has invalid label %v", i, p.Label)
		}
	}
}

func idOf(i int) string {
	return "req_" + string(rune('a'+i%26)) + string(rune('0'+i/26))
}
