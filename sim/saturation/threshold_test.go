// sim/saturation/threshold_test.go
package saturation

import (
	"fmt"
	"testing"
)

// streamThreshold feeds a threshold detector one completion per latency (via the
// surviving Observe/Detect path) and returns its verdict. It reconstructs, for
// the shared classifyThreshold algorithm, the same mean-E2E input the removed
// Classify path took directly (#1516). The threshold detector ignores arrivals,
// so only completions are fed.
func streamThreshold(thresholdMs float64, latencies []float64) Result {
	det := NewThresholdDetector(thresholdMs)
	for i, lat := range latencies {
		det.Observe(Event{
			Timestamp: int64(i+1) * 1_000_000,
			Type:      Completion,
			RequestID: fmt.Sprintf("r%d", i),
			LatencyMs: lat,
		})
	}
	return det.Detect()
}

// TestThresholdDetector_BelowThreshold verifies BC-4: STABLE when mean E2E < threshold
func TestThresholdDetector_BelowThreshold(t *testing.T) {
	result := streamThreshold(5000.0, []float64{3000, 3500, 4000, 4500})

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
	result := streamThreshold(5000.0, []float64{6000, 7000, 8000, 9000})

	if result.Level != Overloaded {
		t.Errorf("Expected OVERLOADED, got %v", result.Level)
	}
	if result.Score < 0.75 {
		t.Errorf("Expected score >= 0.75 for overloaded, got %.2f", result.Score)
	}
}

// TestThresholdDetector_DefaultThreshold verifies default 5000ms threshold
func TestThresholdDetector_DefaultThreshold(t *testing.T) {
	result := streamThreshold(0, []float64{4500, 4500}) // 0 means use default

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
