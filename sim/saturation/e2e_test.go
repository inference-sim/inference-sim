// sim/saturation/e2e_test.go
package saturation_test

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/saturation"
)

// TestE2E_CompositeDetector_WithMetrics verifies BC-6: end-to-end flow
// with CompositeDetector via SaveResults integration
func TestE2E_CompositeDetector_WithMetrics(t *testing.T) {
	// Create metrics with some completed requests
	m := &sim.Metrics{
		CompletedRequests: 3,
		SimEndedTime:      5000000, // 5 seconds
		Requests: map[string]sim.RequestMetrics{
			"r1": {ID: "r1", ArrivedAt: 0},
			"r2": {ID: "r2", ArrivedAt: 1},
			"r3": {ID: "r3", ArrivedAt: 2},
		},
		RequestE2Es: map[string]float64{
			"r1": 100000, // 100ms
			"r2": 150000, // 150ms
			"r3": 200000, // 200ms
		},
		RequestTTFTs:            map[string]float64{},
		RequestITLs:             map[string]float64{},
		RequestSchedulingDelays: map[string]int64{},
	}

	// Create detector
	det := saturation.NewDetector("composite", saturation.DetectorOpts{})

	// Call SaveResults with detector (should populate saturation field)
	if err := m.SaveResults("test", 5000000, 1000, "", det); err != nil {
		t.Fatalf("SaveResults failed: %v", err)
	}
	// Note: SaveResults outputs to stdout, actual verification would need
	// to capture output or use file path. This test just ensures no panic.
}

// TestE2E_ThresholdDetector_BelowThreshold verifies threshold detector
// correctly classifies stable workload
func TestE2E_ThresholdDetector_BelowThreshold(t *testing.T) {
	requests := []sim.RequestMetrics{
		{E2E: 3000, ArrivedAt: 0}, // All below 5000ms threshold
		{E2E: 4000, ArrivedAt: 1},
		{E2E: 3500, ArrivedAt: 2},
	}

	det := saturation.NewDetector("threshold", saturation.DetectorOpts{ThresholdMs: 5000})
	result := det.Classify(requests)

	if result.Level != saturation.Stable {
		t.Errorf("Expected STABLE for mean < threshold, got %v", result.Level)
	}
}

// TestE2E_ThresholdDetector_AboveThreshold verifies threshold detector
// correctly classifies overloaded workload
func TestE2E_ThresholdDetector_AboveThreshold(t *testing.T) {
	requests := []sim.RequestMetrics{
		{E2E: 6000, ArrivedAt: 0}, // All above 5000ms threshold
		{E2E: 7000, ArrivedAt: 1},
		{E2E: 8000, ArrivedAt: 2},
	}

	det := saturation.NewDetector("threshold", saturation.DetectorOpts{ThresholdMs: 5000})
	result := det.Classify(requests)

	if result.Level != saturation.Overloaded {
		t.Errorf("Expected OVERLOADED for mean > threshold, got %v", result.Level)
	}
	if result.Score < 0.75 {
		t.Errorf("Expected score >= 0.75 for overloaded, got %.2f", result.Score)
	}
}

// TestE2E_NoOpDetector verifies "none" detector always returns stable
func TestE2E_NoOpDetector(t *testing.T) {
	requests := []sim.RequestMetrics{
		{E2E: 10000, ArrivedAt: 0}, // High latencies
		{E2E: 10000, ArrivedAt: 1},
	}

	det := saturation.NewDetector("none", saturation.DetectorOpts{})
	result := det.Classify(requests)

	if result.Level != saturation.Stable {
		t.Errorf("NoOp detector should always return STABLE, got %v", result.Level)
	}
	if result.Score != 0 {
		t.Errorf("NoOp detector should return score=0, got %.2f", result.Score)
	}
}
