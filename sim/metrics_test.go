package sim

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestSaveResults_InstanceID_InJSON verifies BC-8: JSON output includes instance_id field.
//
// Given: A Metrics struct with completed requests
// When: SaveResults is called with instanceID "test-instance"
// Then: The JSON file contains "instance_id": "test-instance"
func TestSaveResults_InstanceID_InJSON(t *testing.T) {
	// GIVEN a Metrics struct with at least one completed request
	m := NewMetrics()
	m.CompletedRequests = 1
	m.TotalInputTokens = 100
	m.TotalOutputTokens = 50
	m.SimEndedTime = 1000000 // 1 second in ticks
	m.RequestTTFTs = map[string]float64{"req1": 10000}
	m.RequestE2Es = map[string]float64{"req1": 50000}
	m.RequestITLs = map[string]float64{"req1": 1000}
	m.AllITLs = []int64{1000}
	m.RequestSchedulingDelays = map[string]int64{"req1": 500}
	m.Requests = map[string]RequestMetrics{
		"req1": {
			ID:               "req1",
			ArrivedAt:        0,
			NumPrefillTokens: 100,
			NumDecodeTokens:  50,
		},
	}

	tmpDir := t.TempDir()
	outputPath := filepath.Join(tmpDir, "test_output.json")

	// WHEN SaveResults is called with instanceID "test-instance"
	m.SaveResults("test-instance", 1000000, 1000, time.Now(), outputPath)

	// THEN the JSON file contains "instance_id": "test-instance"
	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("Failed to read output file: %v", err)
	}

	var output MetricsOutput
	if err := json.Unmarshal(data, &output); err != nil {
		t.Fatalf("Failed to unmarshal JSON: %v", err)
	}

	if output.InstanceID != "test-instance" {
		t.Errorf("InstanceID = %q, want %q", output.InstanceID, "test-instance")
	}
}

// TestSaveResults_InstanceID_Empty verifies BC-8: Empty instance_id is valid.
//
// Given: A Metrics struct with completed requests
// When: SaveResults is called with empty instanceID ""
// Then: The JSON file contains "instance_id": ""
func TestSaveResults_InstanceID_Empty(t *testing.T) {
	// GIVEN a Metrics struct with at least one completed request
	m := NewMetrics()
	m.CompletedRequests = 1
	m.TotalInputTokens = 100
	m.TotalOutputTokens = 50
	m.SimEndedTime = 1000000
	m.RequestTTFTs = map[string]float64{"req1": 10000}
	m.RequestE2Es = map[string]float64{"req1": 50000}
	m.RequestITLs = map[string]float64{"req1": 1000}
	m.AllITLs = []int64{1000}
	m.RequestSchedulingDelays = map[string]int64{"req1": 500}
	m.Requests = map[string]RequestMetrics{
		"req1": {
			ID:               "req1",
			ArrivedAt:        0,
			NumPrefillTokens: 100,
			NumDecodeTokens:  50,
		},
	}

	tmpDir := t.TempDir()
	outputPath := filepath.Join(tmpDir, "test_output.json")

	// WHEN SaveResults is called with empty instanceID
	m.SaveResults("", 1000000, 1000, time.Now(), outputPath)

	// THEN the JSON file contains "instance_id": ""
	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("Failed to read output file: %v", err)
	}

	var output MetricsOutput
	if err := json.Unmarshal(data, &output); err != nil {
		t.Fatalf("Failed to unmarshal JSON: %v", err)
	}

	if output.InstanceID != "" {
		t.Errorf("InstanceID = %q, want empty string", output.InstanceID)
	}
}

// TestSaveResults_InstanceID_Default verifies BC-8: Default "default" instance_id works.
//
// Given: A Metrics struct with completed requests
// When: SaveResults is called with instanceID "default" (CLI default)
// Then: The JSON file contains "instance_id": "default"
func TestSaveResults_InstanceID_Default(t *testing.T) {
	// GIVEN a Metrics struct with at least one completed request
	m := NewMetrics()
	m.CompletedRequests = 1
	m.TotalInputTokens = 100
	m.TotalOutputTokens = 50
	m.SimEndedTime = 1000000
	m.RequestTTFTs = map[string]float64{"req1": 10000}
	m.RequestE2Es = map[string]float64{"req1": 50000}
	m.RequestITLs = map[string]float64{"req1": 1000}
	m.AllITLs = []int64{1000}
	m.RequestSchedulingDelays = map[string]int64{"req1": 500}
	m.Requests = map[string]RequestMetrics{
		"req1": {
			ID:               "req1",
			ArrivedAt:        0,
			NumPrefillTokens: 100,
			NumDecodeTokens:  50,
		},
	}

	tmpDir := t.TempDir()
	outputPath := filepath.Join(tmpDir, "test_output.json")

	// WHEN SaveResults is called with instanceID "default"
	m.SaveResults("default", 1000000, 1000, time.Now(), outputPath)

	// THEN the JSON file contains "instance_id": "default"
	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("Failed to read output file: %v", err)
	}

	var output MetricsOutput
	if err := json.Unmarshal(data, &output); err != nil {
		t.Fatalf("Failed to unmarshal JSON: %v", err)
	}

	if output.InstanceID != "default" {
		t.Errorf("InstanceID = %q, want %q", output.InstanceID, "default")
	}
}
