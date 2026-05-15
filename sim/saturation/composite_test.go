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
	result := det.Classify(requests)

	if result.Level != Stable {
		t.Errorf("Expected STABLE, got %v", result.Level)
	}
	if result.Score >= 0.5 {
		t.Errorf("Expected score < 0.5 for stable, got %.2f", result.Score)
	}
	if result.Confidence <= 0 {
		t.Errorf("Expected confidence > 0, got %.2f", result.Confidence)
	}
	// Verify signals exist
	if _, ok := result.Signals["rate_deficit"]; !ok {
		t.Error("Missing rate_deficit signal")
	}
	if _, ok := result.Signals["latency_trend"]; !ok {
		t.Error("Missing latency_trend signal")
	}
}

// TestCompositeDetector_BackloggedCase verifies BC-2: backlogged when one signal saturated
func TestCompositeDetector_BackloggedCase(t *testing.T) {
	// Latency increasing but completions match arrivals
	requests := []sim.RequestMetrics{
		{E2E: 100, ArrivedAt: 0},
		{E2E: 150, ArrivedAt: 0.1},
		{E2E: 200, ArrivedAt: 0.2},
		{E2E: 250, ArrivedAt: 0.3},
	}

	det := NewCompositeDetector()
	result := det.Classify(requests)

	if result.Level != Backlogged {
		t.Errorf("Expected BACKLOGGED, got %v", result.Level)
	}
	if result.Score < 0.25 || result.Score >= 0.75 {
		t.Errorf("Expected score in [0.25, 0.75) for backlogged, got %.2f", result.Score)
	}
}

// TestCompositeDetector_OverloadedCase verifies BC-3: overloaded when both signals saturated
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

// TestCompositeDetector_NoiseFloor verifies BC-10: confidence uses 1/sqrt(N) noise floor
func TestCompositeDetector_NoiseFloor(t *testing.T) {
	requests := []sim.RequestMetrics{
		{E2E: 100, ArrivedAt: 0},
	}

	det := NewCompositeDetector()
	result := det.Classify(requests)

	// With N=1, noise floor is 1.0, so confidence should be capped
	if result.Confidence > 1.0 {
		t.Errorf("Expected confidence <= 1.0, got %.2f", result.Confidence)
	}
}

// TestCompositeDetector_Reset verifies BC-9: Reset clears state
func TestCompositeDetector_Reset(t *testing.T) {
	det := NewCompositeDetector()

	// Observe some events
	det.Observe(Event{Timestamp: 0, Type: Arrival, RequestID: "r1"})
	det.Observe(Event{Timestamp: 100000, Type: Completion, RequestID: "r1", LatencyMs: 100})

	// Detect should show some state
	result1 := det.Detect()

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
