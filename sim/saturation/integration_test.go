// sim/saturation/integration_test.go
package saturation_test

import (
	"encoding/json"
	"testing"

	"blis/sim"
	"blis/sim/saturation"
)

// TestMetricsOutput_SaturationField verifies BC-8: MetricsOutput.Saturation field exists and serializes
func TestMetricsOutput_SaturationField(t *testing.T) {
	// Create a MetricsOutput with saturation result
	output := sim.MetricsOutput{
		InstanceID:        "test",
		CompletedRequests: 100,
		Saturation: &saturation.Result{
			Level:      saturation.Stable,
			Score:      0.3,
			Confidence: 0.9,
			Signals: map[string]float64{
				"rate_deficit":  0.1,
				"latency_trend": 0.05,
			},
		},
	}

	// Serialize to JSON
	data, err := json.Marshal(output)
	if err != nil{
		t.Fatalf("Failed to marshal MetricsOutput: %v", err)
	}

	// Deserialize back
	var decoded sim.MetricsOutput
	err = json.Unmarshal(data, &decoded)
	if err != nil {
		t.Fatalf("Failed to unmarshal MetricsOutput: %v", err)
	}

	// Verify saturation field preserved
	if decoded.Saturation == nil {
		t.Fatal("Saturation field is nil after round-trip")
	}

	// Type assert from interface{} to map[string]interface{} (JSON unmarshaling default)
	satMap, ok := decoded.Saturation.(map[string]interface{})
	if !ok {
		t.Fatalf("Saturation field is not a map: %T", decoded.Saturation)
	}

	// Verify level (as string)
	level, ok := satMap["level"].(string)
	if !ok || level != "STABLE" {
		t.Errorf("Expected level=STABLE, got %v", satMap["level"])
	}

	// Verify score
	score, ok := satMap["score"].(float64)
	if !ok || score != 0.3 {
		t.Errorf("Expected score=0.3, got %v", satMap["score"])
	}

	// Verify confidence
	confidence, ok := satMap["confidence"].(float64)
	if !ok || confidence != 0.9 {
		t.Errorf("Expected confidence=0.9, got %v", satMap["confidence"])
	}

	// Verify JSON contains "saturation" key
	var raw map[string]interface{}
	err = json.Unmarshal(data, &raw)
	if err != nil {
		t.Fatalf("Failed to unmarshal to map: %v", err)
	}
	if _, ok := raw["saturation"]; !ok {
		t.Error("JSON output missing 'saturation' key")
	}
}

// TestMetricsOutput_SaturationNil verifies BC-8: Saturation field can be nil
func TestMetricsOutput_SaturationNil(t *testing.T) {
	// Create a MetricsOutput without saturation
	output := sim.MetricsOutput{
		InstanceID:        "test",
		CompletedRequests: 100,
		Saturation:        nil,
	}

	// Serialize to JSON
	data, err := json.Marshal(output)
	if err != nil {
		t.Fatalf("Failed to marshal MetricsOutput: %v", err)
	}

	// Deserialize back
	var decoded sim.MetricsOutput
	err = json.Unmarshal(data, &decoded)
	if err != nil {
		t.Fatalf("Failed to unmarshal MetricsOutput: %v", err)
	}

	// Verify saturation field is nil
	if decoded.Saturation != nil {
		t.Errorf("Expected Saturation=nil, got %v", decoded.Saturation)
	}
}
