// sim/saturation/e2e_test.go
package saturation_test

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"blis/sim"
	"blis/sim/saturation"
)


// TestE2E_CompositeDetector_WithMetrics verifies BC-6: end-to-end flow
// with CompositeDetector via SaveResults integration (C6: with behavioral assertions)
func TestE2E_CompositeDetector_WithMetrics(t *testing.T) {
	// Create metrics with 24 completed requests showing smooth monotonic increase
	// This ensures quartile monotonicity: Q1 < Q2 < Q3 < Q4
	// Latencies: 100, 105, 110, ..., 215 (24 values, increasing by 5ms each)
	m := &sim.Metrics{
		CompletedRequests:       24,
		SimEndedTime:            5000000, // 5 seconds
		Requests:                make(map[string]sim.RequestMetrics),
		RequestE2Es:             make(map[string]float64),
		RequestTTFTs:            make(map[string]float64),
		RequestITLs:             make(map[string]float64),
		RequestSchedulingDelays: make(map[string]int64),
	}

	// 24 requests with smoothly increasing latency (100, 105, 110, ..., 215)
	for i := 0; i < 24; i++ {
		id := fmt.Sprintf("r%d", i)
		m.Requests[id] = sim.RequestMetrics{ID: id, ArrivedAt: float64(i)}
		m.RequestE2Es[id] = float64(100000 + i*5000) // 100ms + i*5ms in ticks
	}

	// Create detector
	det := saturation.NewDetector("composite", saturation.DetectorOpts{})

	// Write to temp file and verify JSON output (C6: behavioral assertion)
	tmpFile := t.TempDir() + "/metrics.json"
	if err := m.SaveResults("test", 5000000, 1000, tmpFile, det); err != nil {
		t.Fatalf("SaveResults failed: %v", err)
	}

	// Read and parse the output JSON
	data, err := os.ReadFile(tmpFile)
	if err != nil {
		t.Fatalf("Failed to read output file: %v", err)
	}

	var output struct {
		Saturation *saturation.Result `json:"saturation"`
	}
	if err := json.Unmarshal(data, &output); err != nil {
		t.Fatalf("Failed to unmarshal JSON: %v", err)
	}

	// Verify saturation field is populated
	if output.Saturation == nil {
		t.Fatal("Saturation field is nil in output JSON")
	}

	// With 24 requests (smoothly increasing 100→215ms), quartile monotonicity satisfied
	// Actual calculation will depend on Classify sorting by completion time (Issue #5)
	// Noise floor = 1/sqrt(24) = 0.204
	// Classification: score >= noise_floor and lt > noise_floor → OVERLOADED
	if output.Saturation.Level != saturation.Overloaded {
		t.Errorf("Expected Overloaded from latency trend, got %v (score=%.2f, lt=%.2f)",
			output.Saturation.Level, output.Saturation.Score, output.Saturation.Signals["latency_trend"])
	}

	// Verify score above noise floor
	noiseFloor := output.Saturation.Signals["noise_floor"]
	if output.Saturation.Score < noiseFloor {
		t.Errorf("Expected score >= noise_floor (%.2f) for detectable trend, got %.2f", noiseFloor, output.Saturation.Score)
	}

	// Verify latency trend was detected (quartile_monotone should be 1)
	if output.Saturation.Signals["quartile_monotone"] != 1.0 {
		t.Errorf("Expected quartile_monotone=1 for smooth increase, got %.2f", output.Saturation.Signals["quartile_monotone"])
	}

	// Verify latency trend above noise floor (needed for OVERLOADED classification)
	if output.Saturation.Signals["latency_trend"] <= noiseFloor {
		t.Errorf("Expected latency_trend > noise_floor for OVERLOADED, got lt=%.2f, noise=%.2f",
			output.Saturation.Signals["latency_trend"], noiseFloor)
	}
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
	result := det.Classify(requests, len(requests)).(saturation.Result) // Issue #4: pass totalArrivals

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
	result := det.Classify(requests, len(requests)).(saturation.Result) // Issue #4: pass totalArrivals

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
	result := det.Classify(requests, len(requests)).(saturation.Result) // Issue #4: pass totalArrivals

	if result.Level != saturation.Stable {
		t.Errorf("NoOp detector should always return STABLE, got %v", result.Level)
	}
	if result.Score != 0 {
		t.Errorf("NoOp detector should return score=0, got %.2f", result.Score)
	}
}
