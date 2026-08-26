// sim/saturation/integration_test.go
package saturation_test

import (
	"encoding/json"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/saturation"
)

// TestMetricsOutput_SaturationField verifies the #1517 stdout shape: the
// Saturation field carries a per-detector final-label map (map[string]Level),
// serializes each Level as its bare string via Level.MarshalJSON, and appears
// under the "saturation" key. This is the shape the production pipeline actually
// emits (ReduceAll's output spliced onto MetricsOutput by cmd) — NOT the retired
// single-detector *Result batch shape.
func TestMetricsOutput_SaturationField(t *testing.T) {
	// Create a MetricsOutput with the per-detector final-label map (#1517).
	output := sim.MetricsOutput{
		InstanceID:        "test",
		CompletedRequests: 100,
		Saturation: map[string]saturation.Level{
			"composite":     saturation.Stable,
			"threshold":     saturation.Overloaded,
			"backlog-drift": saturation.Backlogged,
		},
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

	// Verify saturation field preserved
	if decoded.Saturation == nil {
		t.Fatal("Saturation field is nil after round-trip")
	}

	// Type assert from interface{} to map[string]interface{} (JSON unmarshaling default).
	satMap, ok := decoded.Saturation.(map[string]interface{})
	if !ok {
		t.Fatalf("Saturation field is not a map: %T", decoded.Saturation)
	}

	// Each detector maps to its bare label string (Level.MarshalJSON).
	want := map[string]string{
		"composite":     "STABLE",
		"threshold":     "OVERLOADED",
		"backlog-drift": "BACKLOGGED",
	}
	if len(satMap) != len(want) {
		t.Errorf("Expected %d detector keys, got %d (%v)", len(want), len(satMap), satMap)
	}
	for det, wantLabel := range want {
		gotLabel, ok := satMap[det].(string)
		if !ok || gotLabel != wantLabel {
			t.Errorf("detector %q: expected %q, got %v", det, wantLabel, satMap[det])
		}
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
