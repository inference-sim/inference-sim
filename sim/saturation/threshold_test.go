// sim/saturation/threshold_test.go
package saturation

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)


// TestThresholdDetector_BelowThreshold verifies BC-4: STABLE when mean E2E < threshold
func TestThresholdDetector_BelowThreshold(t *testing.T) {
	requests := []sim.RequestMetrics{
		{E2E: 3000},
		{E2E: 3500},
		{E2E: 4000},
		{E2E: 4500},
	}

	det := NewThresholdDetector(5000.0)
	rawResult := det.Classify(requests)
	result := asResult(t, rawResult)

	if result.Level != Stable {
		t.Errorf("Expected STABLE, got %v", result.Level)
	}
	if result.Score >= 0.5 {
		t.Errorf("Expected score < 0.5 for stable, got %.2f", result.Score)
	}
	if _, ok := result.Signals["mean_e2e"]; !ok {
		t.Error("Missing mean_e2e signal")
	}
	if _, ok := result.Signals["threshold"]; !ok {
		t.Error("Missing threshold signal")
	}
}

// TestThresholdDetector_AboveThreshold verifies BC-5: OVERLOADED when mean E2E > threshold
func TestThresholdDetector_AboveThreshold(t *testing.T) {
	requests := []sim.RequestMetrics{
		{E2E: 6000},
		{E2E: 7000},
		{E2E: 8000},
		{E2E: 9000},
	}

	det := NewThresholdDetector(5000.0)
	rawResult := det.Classify(requests)
	result := asResult(t, rawResult)

	if result.Level != Overloaded {
		t.Errorf("Expected OVERLOADED, got %v", result.Level)
	}
	if result.Score < 0.75 {
		t.Errorf("Expected score >= 0.75 for overloaded, got %.2f", result.Score)
	}
}

// TestThresholdDetector_DefaultThreshold verifies default 5000ms threshold
func TestThresholdDetector_DefaultThreshold(t *testing.T) {
	det := NewThresholdDetector(0) // 0 means use default

	requests := []sim.RequestMetrics{
		{E2E: 4500},
		{E2E: 4500},
	}

	rawResult := det.Classify(requests)
	result := asResult(t, rawResult)
	if result.Level != Stable {
		t.Errorf("Expected STABLE with default threshold, got %v", result.Level)
	}

	// Verify threshold signal shows 5000
	if threshold, ok := result.Signals["threshold"]; !ok || threshold != 5000.0 {
		t.Errorf("Expected threshold signal = 5000.0, got %.2f", threshold)
	}
}

// TestThresholdDetector_Reset verifies BC-9: Reset clears state
func TestThresholdDetector_Reset(t *testing.T) {
	det := NewThresholdDetector(5000.0)

	// Observe some events
	det.Observe(Event{Timestamp: 0, Type: Arrival, RequestID: "r1"})
	det.Observe(Event{Timestamp: 100000, Type: Completion, RequestID: "r1", LatencyMs: 6000})

	// Detect should show overload
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

	// Verify result1 showed overload
	if result1.Level != Overloaded {
		t.Errorf("Before reset, expected OVERLOADED, got %v", result1.Level)
	}
}
