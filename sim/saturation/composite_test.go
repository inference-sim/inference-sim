// sim/saturation/composite_test.go
package saturation

import (
	"fmt"
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)


// TestCompositeDetector_StableCase verifies BC-1: stable when completions match arrivals and latency stable
func TestCompositeDetector_StableCase(t *testing.T) {
	requests := []sim.RequestMetrics{
		{E2E: 100, ArrivedAt: 0},
		{E2E: 100, ArrivedAt: 0.1},
		{E2E: 100, ArrivedAt: 0.2},
		{E2E: 100, ArrivedAt: 0.3},
	}

	det := NewCompositeDetector()
	rawResult := det.Classify(requests, len(requests)) // Issue #4: pass totalArrivals
	result := asResult(t, rawResult)

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
	// 6 arrivals, 4 completions → RD = 1 - 4/6 = 0.33
	// Latencies increasing (100→130ms): LT = (125-105)/105 = 0.19
	// score = max(0.33, 0.19) = 0.33
	// noise_floor = 1/sqrt(6) = 0.408
	// score < noise_floor → STABLE
	requests := []sim.RequestMetrics{
		{E2E: 100, ArrivedAt: 0},
		{E2E: 110, ArrivedAt: 0.1},
		{E2E: 120, ArrivedAt: 0.2},
		{E2E: 130, ArrivedAt: 0.3},
	}

	det := NewCompositeDetector()
	rawResult := det.Classify(requests, 6) // 6 arrivals, 4 completed
	result := asResult(t, rawResult)

	// Noise floor = 0.408, score = 0.33 < noise_floor → STABLE
	if result.Level != Stable {
		t.Errorf("Expected STABLE with score < noise_floor, got %v (score=%.2f)", result.Level, result.Score)
	}

	// Now test with stronger deficit: 10 arrivals, 4 completions
	// RD = 1 - 4/10 = 0.6, LT = 0.19, score = max(0.6, 0.19) = 0.6
	// noise_floor = 1/sqrt(10) = 0.316
	// score >= noise_floor but LT < noise_floor → BACKLOGGED
	rawResult2 := det.Classify(requests, 10) // 10 arrivals, 4 completed
	result2 := asResult(t, rawResult2)

	if result2.Level != Backlogged {
		t.Errorf("Expected BACKLOGGED with strong RD, got %v (score=%.2f)", result2.Level, result2.Score)
	}
	// Verify score is dominated by RD (0.6) since lt=0 for n=4 < 20
	expectedScore := 0.6
	if math.Abs(result2.Score-expectedScore) > 0.01 {
		t.Errorf("Expected score ≈ %.2f (RD), got %.2f", expectedScore, result2.Score)
	}
}

// TestCompositeDetector_StrongRateDeficit verifies: strong rate deficit detected
func TestCompositeDetector_StrongRateDeficit(t *testing.T) {
	// 4 arrivals, 1 completion → RD = 1 - 1/4 = 0.75
	// Single request: no LT (n=1 < 2), lt = 0
	// score = max(0.75, 0) = 0.75
	// noise_floor = 1/sqrt(4) = 0.5
	// Classification: score >= noise_floor but lt == 0 → BACKLOGGED
	// (OVERLOADED requires lt > noise_floor per validated algorithm)
	requests := []sim.RequestMetrics{
		{E2E: 100, ArrivedAt: 0},
	}

	det := NewCompositeDetector()
	rawResult := det.Classify(requests, 4) // 4 arrivals, 1 completed
	result := asResult(t, rawResult)

	// With n=1, no LT computed, classification: score=0.75 >= noise_floor and lt=0 → BACKLOGGED
	if result.Level != Backlogged {
		t.Errorf("Expected BACKLOGGED from strong RD (lt=0), got %v (score=%.2f)", result.Level, result.Score)
	}
	if result.Score < 0.75 {
		t.Errorf("Expected score >= 0.75 for strong deficit, got %.2f", result.Score)
	}
}

// TestCompositeDetector_SmallSampleLatencyTrend verifies LT=0 for n < 20 (spec compliance)
func TestCompositeDetector_SmallSampleLatencyTrend(t *testing.T) {
	// 4 requests with increasing latency: 100 → 300ms
	// ltRaw = (250 - 125) / 125 = 1.0 (computed for diagnostics)
	// But: n=4 < 20, so lt=0 (quartile filter can't validate, rely on RD only)
	// RD = 0 (all requests completed)
	// score = max(0, 0) = 0
	// noise_floor = 1/sqrt(4) = 0.5
	// Classification: score < noise_floor → STABLE
	requests := []sim.RequestMetrics{
		{E2E: 100, ArrivedAt: 0},
		{E2E: 150, ArrivedAt: 1},
		{E2E: 200, ArrivedAt: 2},
		{E2E: 300, ArrivedAt: 3},
	}

	det := NewCompositeDetector()
	rawResult := det.Classify(requests, len(requests))
	result := asResult(t, rawResult)

	// ltRaw should be computed (for diagnostics)
	if result.Signals["latency_trend_raw"] <= 0.5 {
		t.Errorf("Expected latency_trend_raw > 0.5 for 100→300ms increase, got %.2f", result.Signals["latency_trend_raw"])
	}

	// But lt (used for classification) should be 0 for n < 20
	if result.Signals["latency_trend"] != 0.0 {
		t.Errorf("Expected latency_trend = 0 for n < 20 (spec compliance), got %.2f", result.Signals["latency_trend"])
	}

	// With n=4 < 20, quartile_monotone is false (filter didn't run)
	if result.Signals["quartile_monotone"] != 0.0 {
		t.Errorf("Expected quartile_monotone = 0 for n < 20, got %.2f", result.Signals["quartile_monotone"])
	}

	// Should classify as STABLE (score=0 < noise_floor=0.5, RD=0, lt=0)
	if result.Level != Stable {
		t.Errorf("Expected STABLE (n < 20, RD=0, lt=0), got %v (score=%.2f)",
			result.Level, result.Score)
	}
}

// TestCompositeDetector_LatencyTrendWith20Plus verifies BC-2: latency trend detection with 20+ requests
func TestCompositeDetector_LatencyTrendWith20Plus(t *testing.T) {
	// Create 20 requests with smoothly increasing latency to satisfy quartile filter
	// Latencies: 100, 105, 110, ..., 195 (20 values, increasing by 5ms each)
	requests := make([]sim.RequestMetrics, 20)
	for i := 0; i < 20; i++ {
		requests[i] = sim.RequestMetrics{E2E: float64(100 + i*5), ArrivedAt: float64(i)}
	}

	det := NewCompositeDetector()
	rawResult := det.Classify(requests, len(requests))
	result := asResult(t, rawResult)

	// Should detect latency trend
	if result.Signals["latency_trend"] <= 0 {
		t.Errorf("Expected latency_trend > 0 with 20+ monotonic requests, got %.2f", result.Signals["latency_trend"])
	}

	// Should be above noise floor
	noiseFloor := result.Signals["noise_floor"]
	if result.Score < noiseFloor {
		t.Errorf("Expected score >= noise_floor for detectable trend, got score=%.2f, noise_floor=%.2f", result.Score, noiseFloor)
	}

	// Quartile filter should pass (smooth monotonic increase)
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
	// Verify score is dominated by RD (0.6) since lt=0 for n=4 < 20
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

	// Only 1 completion
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
	// Test 1: 1 arrival, 1 completion
	// Confidence = min(1.0, arrivals/20) = min(1.0, 1/20) = 0.05
	requests1 := []sim.RequestMetrics{{E2E: 100, ArrivedAt: 0}}
	det1 := NewCompositeDetector()
	result1 := asResult(t, det1.Classify(requests1, 1)) // 1 arrival

	if result1.Confidence != 0.05 {
		t.Errorf("Expected confidence = 0.05 (1 arrival / 20), got %.2f", result1.Confidence)
	}

	// Test 2: 100 arrivals, 100 completions
	// Confidence = min(1, 100/20) = 1.0
	requests100 := make([]sim.RequestMetrics, 100)
	for i := range requests100 {
		requests100[i] = sim.RequestMetrics{E2E: 100, ArrivedAt: float64(i)}
	}
	det100 := NewCompositeDetector()
	result100 := asResult(t, det100.Classify(requests100, 100)) // 100 arrivals

	if result100.Confidence != 1.0 {
		t.Errorf("Expected confidence = 1.0 (100 arrivals / 20), got %.2f", result100.Confidence)
	}

	// Verify noise floor formula: 1/sqrt(arrivals)
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
