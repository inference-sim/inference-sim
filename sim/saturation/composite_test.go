// sim/saturation/composite_test.go
package saturation

import (
	"fmt"
	"math"
	"testing"
)

// streamComposite feeds a composite detector `arrivals` arrival events and one
// completion per latency (in latencies order), then returns the streamed
// verdict. It reconstructs, through the surviving Observe/Detect path, the same
// (arrivals, completions, latencies) inputs the removed Classify path took
// directly (#1516) — so the shared computeComposite algorithm stays covered.
// Completion timestamps are assigned in increasing order so the detector's
// completion-time sort preserves the given latency order.
func streamComposite(arrivals int, latencies []float64) Result {
	det := NewCompositeDetector()
	for i := 0; i < arrivals; i++ {
		det.Observe(Event{Timestamp: int64(i) * 1000, Type: Arrival, RequestID: fmt.Sprintf("r%d", i)})
	}
	for i, lat := range latencies {
		det.Observe(Event{
			Timestamp: int64(i+1) * 1_000_000, // strictly increasing completion order
			Type:      Completion,
			RequestID: fmt.Sprintf("r%d", i),
			LatencyMs: lat,
		})
	}
	return det.Detect()
}

// TestCompositeDetector_StableCase verifies BC-1: stable when completions match arrivals and latency stable
func TestCompositeDetector_StableCase(t *testing.T) {
	result := streamComposite(4, []float64{100, 100, 100, 100})

	// With n=4, LT is now computed (base spec has no n >= 20 requirement)
	// Stable latency (100ms constant) → LT = 0
	// RD = 1 - 4/4 = 0, score = max(0, 0) = 0
	// noise_floor = 1/sqrt(4) = 0.5
	// score < noise_floor → STABLE
	if result.Level != Stable {
		t.Errorf("Expected STABLE, got %v", result.Level)
	}
	if result.Score != 0 {
		t.Errorf("Expected score = 0 for stable latency, got %.2f", result.Score)
	}
	// Confidence = min(1.0, arrivals/20) = min(1.0, 4/20) = 0.2
	if result.Confidence != 0.2 {
		t.Errorf("Expected confidence = 0.2 (4 arrivals / 20), got %.2f", result.Confidence)
	}
	// Verify signals exist
	if _, ok := result.Signals["rate_deficit"]; !ok {
		t.Error("Missing rate_deficit signal")
	}
	if _, ok := result.Signals["latency_trend"]; !ok {
		t.Error("Missing latency_trend signal")
	}
	if _, ok := result.Signals["noise_floor"]; !ok {
		t.Error("Missing noise_floor signal")
	}
}

// TestCompositeDetector_BackloggedRateDeficit verifies BC-2: backlogged when moderate rate deficit
func TestCompositeDetector_BackloggedRateDeficit(t *testing.T) {
	latencies := []float64{100, 110, 120, 130}

	// 6 arrivals, 4 completions → RD = 1 - 4/6 = 0.33
	// noise_floor = 1/sqrt(6) = 0.408; score = 0.33 < noise_floor → STABLE
	result := streamComposite(6, latencies)
	if result.Level != Stable {
		t.Errorf("Expected STABLE with score < noise_floor, got %v (score=%.2f)", result.Level, result.Score)
	}

	// 10 arrivals, 4 completions → RD = 1 - 4/10 = 0.6, lt=0 (n=4 < 20)
	// noise_floor = 1/sqrt(10) = 0.316; score >= noise_floor, lt < noise_floor → BACKLOGGED
	result2 := streamComposite(10, latencies)
	if result2.Level != Backlogged {
		t.Errorf("Expected BACKLOGGED with strong RD, got %v (score=%.2f)", result2.Level, result2.Score)
	}
	expectedScore := 0.6
	if math.Abs(result2.Score-expectedScore) > 0.01 {
		t.Errorf("Expected score ≈ %.2f (RD), got %.2f", expectedScore, result2.Score)
	}
}

// TestCompositeDetector_StrongRateDeficit verifies: strong rate deficit detected
func TestCompositeDetector_StrongRateDeficit(t *testing.T) {
	// 4 arrivals, 1 completion → RD = 1 - 1/4 = 0.75
	// Single completion: no LT (n=1 < 2), lt = 0
	// score = max(0.75, 0) = 0.75; noise_floor = 1/sqrt(4) = 0.5
	// score >= noise_floor but lt == 0 → BACKLOGGED
	result := streamComposite(4, []float64{100})

	if result.Level != Backlogged {
		t.Errorf("Expected BACKLOGGED from strong RD (lt=0), got %v (score=%.2f)", result.Level, result.Score)
	}
	if result.Score < 0.75 {
		t.Errorf("Expected score >= 0.75 for strong deficit, got %.2f", result.Score)
	}
}

// TestCompositeDetector_SmallSampleLatencyTrend verifies LT=0 for n < 20 (spec compliance)
func TestCompositeDetector_SmallSampleLatencyTrend(t *testing.T) {
	// 4 completions with increasing latency: 100 → 300ms, all arrivals completed.
	// ltRaw = (250 - 125) / 125 = 1.0 (computed for diagnostics)
	// n=4 < 20, so lt=0 (quartile filter can't validate); RD=0; score=0
	// noise_floor = 1/sqrt(4) = 0.5; score < noise_floor → STABLE
	result := streamComposite(4, []float64{100, 150, 200, 300})

	if result.Signals["latency_trend_raw"] <= 0.5 {
		t.Errorf("Expected latency_trend_raw > 0.5 for 100→300ms increase, got %.2f", result.Signals["latency_trend_raw"])
	}
	if result.Signals["latency_trend"] != 0.0 {
		t.Errorf("Expected latency_trend = 0 for n < 20 (spec compliance), got %.2f", result.Signals["latency_trend"])
	}
	if result.Signals["quartile_monotone"] != 0.0 {
		t.Errorf("Expected quartile_monotone = 0 for n < 20, got %.2f", result.Signals["quartile_monotone"])
	}
	if result.Level != Stable {
		t.Errorf("Expected STABLE (n < 20, RD=0, lt=0), got %v (score=%.2f)", result.Level, result.Score)
	}
}

// TestCompositeDetector_LatencyTrendWith20Plus verifies BC-2: latency trend detection with 20+ requests
func TestCompositeDetector_LatencyTrendWith20Plus(t *testing.T) {
	// 20 completions with smoothly increasing latency (100, 105, ..., 195) to
	// satisfy the quartile filter; all arrivals completed (RD=0).
	latencies := make([]float64, 20)
	for i := 0; i < 20; i++ {
		latencies[i] = float64(100 + i*5)
	}
	result := streamComposite(20, latencies)

	if result.Signals["latency_trend"] <= 0 {
		t.Errorf("Expected latency_trend > 0 with 20+ monotonic requests, got %.2f", result.Signals["latency_trend"])
	}
	noiseFloor := result.Signals["noise_floor"]
	if result.Score < noiseFloor {
		t.Errorf("Expected score >= noise_floor for detectable trend, got score=%.2f, noise_floor=%.2f", result.Score, noiseFloor)
	}
	if result.Signals["quartile_monotone"] != 1.0 {
		t.Errorf("Expected quartile_monotone = 1 for smooth increase, got %.2f", result.Signals["quartile_monotone"])
	}
}

// TestCompositeDetector_ObserveDetectStable verifies I7: Stable via Observe/Detect (streaming mode)
func TestCompositeDetector_ObserveDetectStable(t *testing.T) {
	det := NewCompositeDetector()

	// 4 arrivals, all complete with stable latency
	det.Observe(Event{Timestamp: 0, Type: Arrival, RequestID: "r1"})
	det.Observe(Event{Timestamp: 100000, Type: Arrival, RequestID: "r2"})
	det.Observe(Event{Timestamp: 200000, Type: Arrival, RequestID: "r3"})
	det.Observe(Event{Timestamp: 300000, Type: Arrival, RequestID: "r4"})

	det.Observe(Event{Timestamp: 100000, Type: Completion, RequestID: "r1", LatencyMs: 100})
	det.Observe(Event{Timestamp: 200000, Type: Completion, RequestID: "r2", LatencyMs: 100})
	det.Observe(Event{Timestamp: 300000, Type: Completion, RequestID: "r3", LatencyMs: 100})
	det.Observe(Event{Timestamp: 400000, Type: Completion, RequestID: "r4", LatencyMs: 100})

	result := det.Detect()

	// n=4, RD=0, LT=0 (n<20), score=0, noise_floor=0.5 → STABLE
	if result.Level != Stable {
		t.Errorf("Expected STABLE, got %v (score=%.2f)", result.Level, result.Score)
	}
	if result.Score >= 0.5 {
		t.Errorf("Expected score < 0.5 for stable, got %.2f", result.Score)
	}
}

// TestCompositeDetector_ObserveDetectBacklogged verifies: Backlogged via Observe/Detect with rate deficit
func TestCompositeDetector_ObserveDetectBacklogged(t *testing.T) {
	det := NewCompositeDetector()

	// 10 arrivals, 4 completions → RD = 0.6 → BACKLOGGED
	for i := 0; i < 10; i++ {
		det.Observe(Event{Timestamp: int64(i * 100000), Type: Arrival, RequestID: fmt.Sprintf("r%d", i)})
	}
	for i := 0; i < 4; i++ {
		det.Observe(Event{Timestamp: int64((i + 1) * 100000), Type: Completion, RequestID: fmt.Sprintf("r%d", i), LatencyMs: 100})
	}

	result := det.Detect()

	// RD = 1 - 4/10 = 0.6, lt=0 (n=4 < 20), noise_floor = 0.316
	// score = 0.6 > noise_floor and lt < noise_floor → BACKLOGGED
	if result.Level != Backlogged {
		t.Errorf("Expected BACKLOGGED, got %v (score=%.2f)", result.Level, result.Score)
	}
	expectedScore := 0.6
	if math.Abs(result.Score-expectedScore) > 0.01 {
		t.Errorf("Expected score ≈ %.2f (RD), got %.2f", expectedScore, result.Score)
	}
}

// TestCompositeDetector_StrongRateDeficitStreaming verifies strong RD detection (streaming mode)
func TestCompositeDetector_StrongRateDeficitStreaming(t *testing.T) {
	det := NewCompositeDetector()

	// Observe 4 arrivals, only 1 completion → RD = 0.75
	det.Observe(Event{Timestamp: 0, Type: Arrival, RequestID: "r1"})
	det.Observe(Event{Timestamp: 100000, Type: Arrival, RequestID: "r2"})
	det.Observe(Event{Timestamp: 200000, Type: Arrival, RequestID: "r3"})
	det.Observe(Event{Timestamp: 300000, Type: Arrival, RequestID: "r4"})

	det.Observe(Event{Timestamp: 100000, Type: Completion, RequestID: "r1", LatencyMs: 100})

	result := det.Detect()

	// RD = 1 - 1/4 = 0.75, but n=1 < 20 so LT=0
	// Classification: score >= noise_floor but lt=0 → BACKLOGGED
	if result.Level != Backlogged {
		t.Errorf("Expected BACKLOGGED (strong RD, lt=0), got %v (score=%.2f)", result.Level, result.Score)
	}
	if result.Score < 0.75 {
		t.Errorf("Expected score >= 0.75 for strong deficit, got %.2f", result.Score)
	}
}

// TestCompositeDetector_NoiseFloor verifies noise floor and confidence formulas
func TestCompositeDetector_NoiseFloor(t *testing.T) {
	// 1 arrival, 1 completion → confidence = min(1.0, 1/20) = 0.05
	result1 := streamComposite(1, []float64{100})
	if result1.Confidence != 0.05 {
		t.Errorf("Expected confidence = 0.05 (1 arrival / 20), got %.2f", result1.Confidence)
	}

	// 100 arrivals, 100 completions → confidence = min(1, 100/20) = 1.0
	latencies := make([]float64, 100)
	for i := range latencies {
		latencies[i] = 100
	}
	result100 := streamComposite(100, latencies)
	if result100.Confidence != 1.0 {
		t.Errorf("Expected confidence = 1.0 (100 arrivals / 20), got %.2f", result100.Confidence)
	}
	// noise_floor formula: 1/sqrt(arrivals) = 1/sqrt(100) = 0.1
	if result100.Signals["noise_floor"] != 0.1 {
		t.Errorf("Expected noise_floor = 0.1 (1/sqrt(100)), got %.2f", result100.Signals["noise_floor"])
	}
}

// TestCompositeDetector_Reset verifies BC-9: Reset clears state (I8)
func TestCompositeDetector_Reset(t *testing.T) {
	det := NewCompositeDetector()

	// Observe events that produce non-stable state (rate deficit)
	det.Observe(Event{Timestamp: 0, Type: Arrival, RequestID: "r1"})
	det.Observe(Event{Timestamp: 0, Type: Arrival, RequestID: "r2"})
	det.Observe(Event{Timestamp: 100000, Type: Completion, RequestID: "r1", LatencyMs: 100})
	// Only 1 completion for 2 arrivals → rate deficit = 0.5

	// Verify detector has non-zero state before reset
	result1 := det.Detect()
	if result1.Level == Stable && result1.Score == 0 {
		t.Error("Expected non-zero state before reset")
	}

	// Reset
	det.Reset()

	// After reset, should be back to stable/zero
	result2 := det.Detect()
	if result2.Level != Stable {
		t.Errorf("After reset, expected STABLE, got %v", result2.Level)
	}
	if result2.Score != 0 {
		t.Errorf("After reset, expected score=0, got %.2f", result2.Score)
	}
}
