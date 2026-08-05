// sim/saturation/backlog_drift_test.go
package saturation

import (
	"strconv"
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// smallWindowConfig returns a BacklogDriftConfig with a 100ms window so a handful
// of directly-fed events span several buckets and the streaming slope is
// deterministically exercised (see #1515 "Testing note"). All other fields keep
// the production defaults.
func smallWindowConfig() workload.BacklogDriftConfig {
	return workload.NewBacklogDriftConfig(
		100*time.Millisecond, // WindowSize (small, for streaming tests)
		5,                    // MinWindows
		2.0,                  // PeakRatio
		0.2,                  // PeakRatioBand
		0.95,                 // ConfidenceCI
		2,                    // WarmupWindows
		1,                    // TailWindows
		0.95,                 // SaturatedDrainRatio
		0.98,                 // TransientDrainRatio
	)
}

// bucketUs is the 100ms window from smallWindowConfig expressed in microseconds.
const bucketUs = int64(100 * time.Millisecond / time.Microsecond)

// TestBacklogDriftDetector_Stable verifies STABLE via the streaming path when
// arrivals and completions stay balanced (flat in-flight → zero slope). This
// replaces the removed batch Classify path (#1516); the streaming signals are
// running_slope/in_flight, not the batch slope/num_windows.
func TestBacklogDriftDetector_Stable(t *testing.T) {
	det := NewBacklogDriftDetectorWithConfig(smallWindowConfig())

	// Each arrival is immediately followed by a completion in the same bucket, so
	// in-flight stays flat at ~0 across buckets → running_slope ≈ 0 → STABLE.
	for i := 0; i < 10; i++ {
		ts := int64(i) * bucketUs
		det.Observe(Event{Timestamp: ts, Type: Arrival, RequestID: "r" + strconv.Itoa(i)})
		det.Observe(Event{Timestamp: ts, Type: Completion, RequestID: "r" + strconv.Itoa(i), LatencyMs: 100})
	}
	result := det.Detect()

	if result.Level != Stable {
		t.Errorf("Expected Stable for flat in-flight, got %v (running_slope=%.4f)", result.Level, result.Signals["running_slope"])
	}
	// Verify streaming signals are present
	if _, ok := result.Signals["running_slope"]; !ok {
		t.Error("Missing running_slope signal")
	}
	if _, ok := result.Signals["in_flight"]; !ok {
		t.Error("Missing in_flight signal")
	}
}

// TestBacklogDriftDetector_Name verifies detector name
func TestBacklogDriftDetector_Name(t *testing.T) {
	det := NewBacklogDriftDetector()
	if det.Name() != "backlog-drift" {
		t.Errorf("Expected name 'backlog-drift', got %q", det.Name())
	}
}

// TestBacklogDriftDetector_Detect_NoEvents verifies the degenerate empty case:
// no events observed → STABLE, zero confidence, no panic (R20, #1515 contract).
func TestBacklogDriftDetector_Detect_NoEvents(t *testing.T) {
	det := NewBacklogDriftDetectorWithConfig(smallWindowConfig())

	result := det.Detect()
	if result.Level != Stable {
		t.Errorf("Expected Stable with no events, got %v", result.Level)
	}
	if result.Confidence != 0 {
		t.Errorf("Expected zero confidence with no events, got %.2f", result.Confidence)
	}
	if result.Score != 0 {
		t.Errorf("Expected zero score with no events, got %.2f", result.Score)
	}
}

// TestBacklogDriftDetector_Detect_RisingBacklog verifies that the verdict
// *evolves*: an initial Detect() before any buildup is STABLE, and feeding a
// rising-backlog sequence (arrivals outpacing completions across buckets) then
// makes in-flight and running_slope rise and drives the level off STABLE (#1515).
func TestBacklogDriftDetector_Detect_RisingBacklog(t *testing.T) {
	det := NewBacklogDriftDetectorWithConfig(smallWindowConfig())

	// One event → a single sample → slope 0 → STABLE. Pins the "before" of the
	// evolution so the off-STABLE assertion below is a genuine transition.
	det.Observe(Event{Type: Arrival, Timestamp: 0, RequestID: "seed"})
	if early := det.Detect(); early.Level != Stable {
		t.Fatalf("Expected STABLE before buildup (single sample), got %v", early.Level)
	}

	// 10 buckets: each bucket adds a growing number of arrivals with no
	// completions → in-flight climbs monotonically, one sample per bucket.
	arrivalID := 0
	for bucket := int64(0); bucket < 10; bucket++ {
		ts := bucket * bucketUs
		// Number of arrivals grows with the bucket → accelerating backlog.
		for k := 0; k <= int(bucket); k++ {
			arrivalID++
			det.Observe(Event{Type: Arrival, Timestamp: ts, RequestID: "a" + itoa(arrivalID)})
		}
	}

	result := det.Detect()

	if result.Signals["running_slope"] <= 0 {
		t.Errorf("Expected positive running_slope for rising backlog, got %.4f", result.Signals["running_slope"])
	}
	if result.Signals["in_flight"] <= 0 {
		t.Errorf("Expected positive in_flight for rising backlog, got %.1f", result.Signals["in_flight"])
	}
	if result.Level == Stable {
		t.Errorf("Expected level off STABLE for rising backlog, got %v (slope=%.4f, noise=%.4f)",
			result.Level, result.Signals["running_slope"], result.Signals["noise_floor"])
	}
}

// TestBacklogDriftDetector_Detect_Draining verifies that after a backlog builds,
// a draining sequence (completions catching up) trends the level back toward
// STABLE with score 0 (negative slope, #1515).
func TestBacklogDriftDetector_Detect_Draining(t *testing.T) {
	det := NewBacklogDriftDetectorWithConfig(smallWindowConfig())

	// Phase 1: build a backlog of 20 in-flight over the first 4 buckets.
	id := 0
	for bucket := int64(0); bucket < 4; bucket++ {
		ts := bucket * bucketUs
		for k := 0; k < 5; k++ {
			id++
			det.Observe(Event{Type: Arrival, Timestamp: ts, RequestID: "a" + itoa(id)})
		}
	}

	// Phase 2: drain — completions outpace new arrivals over the next 6 buckets,
	// so in-flight falls back toward zero.
	comp := 0
	for bucket := int64(4); bucket < 10; bucket++ {
		ts := bucket * bucketUs
		for k := 0; k < 3; k++ {
			comp++
			det.Observe(Event{Type: Completion, Timestamp: ts, RequestID: "a" + itoa(comp), LatencyMs: 100})
		}
	}

	result := det.Detect()

	if result.Signals["running_slope"] >= 0 {
		t.Errorf("Expected negative running_slope while draining, got %.4f", result.Signals["running_slope"])
	}
	if result.Level != Stable {
		t.Errorf("Expected STABLE while draining, got %v (slope=%.4f)", result.Level, result.Signals["running_slope"])
	}
	if result.Score != 0 {
		t.Errorf("Expected zero score while draining (negative slope), got %.4f", result.Score)
	}
}

// TestBacklogDriftDetector_Reset verifies Reset returns the detector to its
// initial state: next Detect() on no events → STABLE, zero confidence (#1515).
func TestBacklogDriftDetector_Reset(t *testing.T) {
	det := NewBacklogDriftDetectorWithConfig(smallWindowConfig())

	// Build some state.
	for bucket := int64(0); bucket < 5; bucket++ {
		det.Observe(Event{Type: Arrival, Timestamp: bucket * bucketUs, RequestID: "a" + itoa(int(bucket))})
	}
	if got := det.Detect(); got.Confidence == 0 {
		t.Fatalf("precondition: expected non-zero confidence after observing events")
	}

	det.Reset()

	result := det.Detect()
	if result.Level != Stable || result.Confidence != 0 || result.Score != 0 {
		t.Errorf("Expected initial state after Reset (STABLE, 0 confidence, 0 score), got %+v", result)
	}
	if result.Signals["in_flight"] != 0 {
		t.Errorf("Expected zero in_flight after Reset, got %.1f", result.Signals["in_flight"])
	}
}

// TestBacklogDriftDetector_Detect_Overloaded verifies the OVERLOADED band: a
// steep, sustained backlog climb drives running_slope past K·noise_floor and
// pushes score to ~1.0 (#1515).
func TestBacklogDriftDetector_Detect_Overloaded(t *testing.T) {
	det := NewBacklogDriftDetectorWithConfig(smallWindowConfig())

	// Steep climb: 50 arrivals per bucket, no completions, over 8 buckets. With
	// ~400 arrivals the noise floor is ~0.05, while the slope is ~50 → well past
	// K·noise, so the band must reach OVERLOADED and score saturates at 1.0.
	id := 0
	for bucket := int64(0); bucket < 8; bucket++ {
		ts := bucket * bucketUs
		for k := 0; k < 50; k++ {
			id++
			det.Observe(Event{Type: Arrival, Timestamp: ts, RequestID: "a" + itoa(id)})
		}
	}

	result := det.Detect()

	if result.Level != Overloaded {
		t.Errorf("Expected OVERLOADED for steep sustained climb, got %v (slope=%.4f, noise=%.4f)",
			result.Level, result.Signals["running_slope"], result.Signals["noise_floor"])
	}
	if result.Score < 0.99 {
		t.Errorf("Expected score ~1.0 when OVERLOADED, got %.4f", result.Score)
	}
}

// TestBacklogDriftDetector_Detect_Backlogged verifies the middle band is
// reachable: a gentle-but-above-noise climb yields noiseFloor < slope <=
// K·noiseFloor → BACKLOGGED, with score strictly in (0, 1) (#1515). Without this
// a band-boundary bug that collapsed BACKLOGGED into OVERLOADED would go
// unnoticed (rising/overloaded tests only pin the outer bands).
func TestBacklogDriftDetector_Detect_Backlogged(t *testing.T) {
	det := NewBacklogDriftDetectorWithConfig(smallWindowConfig())

	// Land the slope between noiseFloor and K·noiseFloor with margin on both
	// sides. Seed 37 in-flight in bucket 0, then add 1 net arrival at buckets 3,
	// 6, and 9. Forward-fill makes the per-bucket samples the ramp
	// [37,37,37,38,38,38,39,39,39,40], whose OLS slope is 27/82.5 ≈ 0.327/bucket.
	// With 40 total arrivals, noiseFloor = 1/√40 ≈ 0.158 and K·noiseFloor ≈ 0.474,
	// so 0.158 < 0.327 < 0.474 → BACKLOGGED (a gentle, above-noise climb), and
	// score = 0.327/0.474 ≈ 0.69, strictly inside (0,1).
	id := 0
	for k := 0; k < 37; k++ {
		id++
		det.Observe(Event{Type: Arrival, Timestamp: 0, RequestID: "a" + itoa(id)})
	}
	for _, bucket := range []int64{3, 6, 9} {
		id++
		det.Observe(Event{Type: Arrival, Timestamp: bucket * bucketUs, RequestID: "a" + itoa(id)})
	}

	result := det.Detect()

	noise := result.Signals["noise_floor"]
	slope := result.Signals["running_slope"]
	if result.Level != Backlogged {
		t.Errorf("Expected BACKLOGGED (noise=%.4f < slope=%.4f <= K·noise=%.4f), got %v",
			noise, slope, backlogDriftSlopeK*noise, result.Level)
	}
	if result.Score <= 0 || result.Score >= 1 {
		t.Errorf("Expected score strictly in (0,1) when BACKLOGGED, got %.4f", result.Score)
	}
}

// TestBacklogDriftDetector_Observe_ForwardFillEmptyBuckets verifies that skipped
// (empty) buckets are forward-filled with the last in-flight value so the OLS
// x-axis stays evenly spaced (#1515). A plateau held flat across a gap must
// produce a ~zero slope (STABLE), not a spuriously large slope from compressing
// the gap. Regression guard for the forward-fill loop.
func TestBacklogDriftDetector_Observe_ForwardFillEmptyBuckets(t *testing.T) {
	det := NewBacklogDriftDetectorWithConfig(smallWindowConfig())

	// 20 arrivals in bucket 0, then a single arrival far later in bucket 10.
	// In-flight goes 20 → (buckets 1..9 forward-filled at 20) → 21 at bucket 10:
	// an essentially flat plateau. If gaps were dropped instead of filled, the
	// two-sample regression would see a steep rise; forward-filling keeps it flat.
	id := 0
	for k := 0; k < 20; k++ {
		id++
		det.Observe(Event{Type: Arrival, Timestamp: 0, RequestID: "a" + itoa(id)})
	}
	id++
	det.Observe(Event{Type: Arrival, Timestamp: 10 * bucketUs, RequestID: "a" + itoa(id)})

	result := det.Detect()

	// Slope over an ~flat plateau must be tiny and below the noise floor → STABLE.
	if result.Signals["running_slope"] > result.Signals["noise_floor"] {
		t.Errorf("Expected forward-filled plateau to yield slope <= noise_floor, got slope=%.4f noise=%.4f",
			result.Signals["running_slope"], result.Signals["noise_floor"])
	}
	if result.Level != Stable {
		t.Errorf("Expected STABLE for a flat plateau across empty buckets, got %v", result.Level)
	}
	// In-flight must reflect all 21 arrivals (nothing dropped by the fill).
	if result.Signals["in_flight"] != 21 {
		t.Errorf("Expected in_flight=21 after forward-fill, got %.1f", result.Signals["in_flight"])
	}
}

// TestBacklogDriftDetector_Observe_OutOfOrderEvent verifies that an out-of-order
// (earlier-timestamp) event folds into the current bucket rather than rewriting
// history or panicking (#1515). Events are expected in non-decreasing time
// order; this pins the defensive branch (R20/R1: no drop, no crash).
func TestBacklogDriftDetector_Observe_OutOfOrderEvent(t *testing.T) {
	det := NewBacklogDriftDetectorWithConfig(smallWindowConfig())

	// Advance to bucket 5, then feed a stale event stamped back in bucket 1.
	det.Observe(Event{Type: Arrival, Timestamp: 0, RequestID: "a1"})
	det.Observe(Event{Type: Arrival, Timestamp: 5 * bucketUs, RequestID: "a2"})
	det.Observe(Event{Type: Arrival, Timestamp: 1 * bucketUs, RequestID: "a3"}) // out of order

	result := det.Detect()

	// The stale event must still be counted (in-flight = 3 arrivals, 0 completions).
	if result.Signals["in_flight"] != 3 {
		t.Errorf("Expected out-of-order event to still count (in_flight=3), got %.1f", result.Signals["in_flight"])
	}
	if result.Signals["arrivals"] != 3 {
		t.Errorf("Expected arrivals=3 after out-of-order event, got %.1f", result.Signals["arrivals"])
	}
}

// itoa aliases strconv.Itoa so test request IDs stay unique and readable.
func itoa(n int) string { return strconv.Itoa(n) }
