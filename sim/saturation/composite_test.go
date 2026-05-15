// sim/saturation/composite_test.go
package saturation

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)


// TestCompositeDetector_StableCase verifies BC-1: stable when completions match arrivals
func TestCompositeDetector_StableCase(t *testing.T) {
	requests := []sim.RequestMetrics{
		{E2E: 100, ArrivedAt: 0},
		{E2E: 100, ArrivedAt: 0.1},
		{E2E: 100, ArrivedAt: 0.2},
		{E2E: 100, ArrivedAt: 0.3},
	}

	det := NewCompositeDetector()
	rawResult := det.Classify(requests)
	result := asResult(t, rawResult)

	if result.Level != Stable {
		t.Errorf("Expected STABLE, got %v", result.Level)
	}
	if result.Score >= 0.5 {
		t.Errorf("Expected score < 0.5 for stable, got %.2f", result.Score)
	}
	if result.Confidence <= 0 {
		t.Errorf("Expected confidence > 0, got %.2f", result.Confidence)
	}
	// Verify signals exist (C2: batch mode uses latency trend only)
	if _, ok := result.Signals["latency_trend"]; !ok {
		t.Error("Missing latency_trend signal")
	}
}

// TestCompositeDetector_BackloggedCase verifies BC-2: backlogged when latency trend indicates moderate degradation
func TestCompositeDetector_BackloggedCase(t *testing.T) {
	// Moderate latency increase: 100ms → 170ms (70% increase over first half mean)
	// First half: (100+110)/2 = 105, Second half: (160+170)/2 = 165
	// Trend: (165-105)/105 = 0.57 → normalized score = 0.57
	requests := []sim.RequestMetrics{
		{E2E: 100, ArrivedAt: 0},
		{E2E: 110, ArrivedAt: 0.1},
		{E2E: 160, ArrivedAt: 0.2},
		{E2E: 170, ArrivedAt: 0.3},
	}

	det := NewCompositeDetector()
	rawResult := det.Classify(requests)
	result := asResult(t, rawResult)

	if result.Level != Backlogged {
		t.Errorf("Expected BACKLOGGED, got %v (score=%.2f)", result.Level, result.Score)
	}
	if result.Score < 0.5 || result.Score >= 0.75 {
		t.Errorf("Expected score in [0.5, 0.75) for backlogged, got %.2f", result.Score)
	}
}

// TestCompositeDetector_ClassifyOverloaded verifies C2: Classify can reach Overloaded via latency trend
func TestCompositeDetector_ClassifyOverloaded(t *testing.T) {
	// Large latency increase: 100ms → 300ms (200% increase, normalized to 2.0, capped to 1.0)
	// Should produce score >= 0.75 → OVERLOADED
	requests := []sim.RequestMetrics{
		{E2E: 100, ArrivedAt: 0},
		{E2E: 120, ArrivedAt: 0.1},
		{E2E: 250, ArrivedAt: 0.2},
		{E2E: 300, ArrivedAt: 0.3},
	}

	det := NewCompositeDetector()
	rawResult := det.Classify(requests)
	result := asResult(t, rawResult)

	if result.Level != Overloaded {
		t.Errorf("Expected OVERLOADED from Classify, got %v (score=%.2f)", result.Level, result.Score)
	}
	if result.Score < 0.75 {
		t.Errorf("Expected score >= 0.75 for overloaded, got %.2f", result.Score)
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

	if result.Level != Stable {
		t.Errorf("Expected STABLE, got %v (score=%.2f)", result.Level, result.Score)
	}
	if result.Score >= 0.5 {
		t.Errorf("Expected score < 0.5 for stable, got %.2f", result.Score)
	}
}

// TestCompositeDetector_ObserveDetectBacklogged verifies I7: Backlogged via Observe/Detect (streaming mode)
func TestCompositeDetector_ObserveDetectBacklogged(t *testing.T) {
	det := NewCompositeDetector()

	// 4 arrivals, all complete but latency significantly increasing
	// First half: (100+110)/2 = 105, Second half: (160+170)/2 = 165
	// Trend: (165-105)/105 = 0.57 → should give Backlogged
	det.Observe(Event{Timestamp: 0, Type: Arrival, RequestID: "r1"})
	det.Observe(Event{Timestamp: 100000, Type: Arrival, RequestID: "r2"})
	det.Observe(Event{Timestamp: 200000, Type: Arrival, RequestID: "r3"})
	det.Observe(Event{Timestamp: 300000, Type: Arrival, RequestID: "r4"})

	det.Observe(Event{Timestamp: 100000, Type: Completion, RequestID: "r1", LatencyMs: 100})
	det.Observe(Event{Timestamp: 200000, Type: Completion, RequestID: "r2", LatencyMs: 110})
	det.Observe(Event{Timestamp: 300000, Type: Completion, RequestID: "r3", LatencyMs: 160})
	det.Observe(Event{Timestamp: 400000, Type: Completion, RequestID: "r4", LatencyMs: 170})

	result := det.Detect()

	if result.Level != Backlogged {
		t.Errorf("Expected BACKLOGGED, got %v (score=%.2f)", result.Level, result.Score)
	}
	if result.Score < 0.5 || result.Score >= 0.75 {
		t.Errorf("Expected score in [0.5, 0.75) for backlogged, got %.2f", result.Score)
	}
}

// TestCompositeDetector_OverloadedCase verifies BC-3: overloaded when both signals saturated (streaming mode)
func TestCompositeDetector_OverloadedCase(t *testing.T) {
	// Use Observe/Detect to test overload with both rate deficit and latency trend
	det := NewCompositeDetector()

	// Observe 4 arrivals
	det.Observe(Event{Timestamp: 0, Type: Arrival, RequestID: "r1"})
	det.Observe(Event{Timestamp: 100000, Type: Arrival, RequestID: "r2"})
	det.Observe(Event{Timestamp: 200000, Type: Arrival, RequestID: "r3"})
	det.Observe(Event{Timestamp: 300000, Type: Arrival, RequestID: "r4"})

	// Only 2 completions with increasing latency
	det.Observe(Event{Timestamp: 100000, Type: Completion, RequestID: "r1", LatencyMs: 100})
	det.Observe(Event{Timestamp: 400000, Type: Completion, RequestID: "r2", LatencyMs: 300})

	result := det.Detect()

	if result.Level != Overloaded {
		t.Errorf("Expected OVERLOADED, got %v", result.Level)
	}
	if result.Score < 0.75 {
		t.Errorf("Expected score >= 0.75 for overloaded, got %.2f", result.Score)
	}
}

// TestCompositeDetector_NoiseFloor verifies C5: confidence formula 1 - 1/sqrt(N+1)
func TestCompositeDetector_NoiseFloor(t *testing.T) {
	// N=1: confidence should be low (< 0.5)
	requests1 := []sim.RequestMetrics{{E2E: 100, ArrivedAt: 0}}
	det1 := NewCompositeDetector()
	result1 := asResult(t, det1.Classify(requests1))

	if result1.Confidence >= 0.5 {
		t.Errorf("Expected confidence < 0.5 for N=1, got %.2f", result1.Confidence)
	}

	// N=100: confidence should be high (> 0.9)
	requests100 := make([]sim.RequestMetrics, 100)
	for i := range requests100 {
		requests100[i] = sim.RequestMetrics{E2E: 100, ArrivedAt: float64(i)}
	}
	det100 := NewCompositeDetector()
	result100 := asResult(t, det100.Classify(requests100))

	if result100.Confidence <= 0.9 {
		t.Errorf("Expected confidence > 0.9 for N=100, got %.2f", result100.Confidence)
	}
}

// TestCompositeDetector_Reset verifies BC-9: Reset clears state (I8)
func TestCompositeDetector_Reset(t *testing.T) {
	det := NewCompositeDetector()

	// Observe events that produce non-stable state
	det.Observe(Event{Timestamp: 0, Type: Arrival, RequestID: "r1"})
	det.Observe(Event{Timestamp: 0, Type: Arrival, RequestID: "r2"})
	det.Observe(Event{Timestamp: 100000, Type: Completion, RequestID: "r1", LatencyMs: 100})
	// Only 1 completion for 2 arrivals → rate deficit > 0

	// Verify detector has non-empty state before reset
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

	// Verify result1 had some data (this test will need adjustment once Observe is implemented)
	_ = result1
}
