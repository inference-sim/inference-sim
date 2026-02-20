package sim

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
	m.SaveResults("test-instance", 1000000, 1000, outputPath)

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
	m.SaveResults("", 1000000, 1000, outputPath)

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
	m.SaveResults("default", 1000000, 1000, outputPath)

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

func TestSaveResults_IncludesIncompleteRequests(t *testing.T) {
	// GIVEN metrics where 2 of 3 requests completed prefill
	m := NewMetrics()
	// All 3 registered
	m.Requests["r1"] = RequestMetrics{ID: "r1", ArrivedAt: 1.0, NumPrefillTokens: 10, NumDecodeTokens: 5}
	m.Requests["r2"] = RequestMetrics{ID: "r2", ArrivedAt: 2.0, NumPrefillTokens: 20, NumDecodeTokens: 10}
	m.Requests["r3"] = RequestMetrics{ID: "r3", ArrivedAt: 3.0, NumPrefillTokens: 30, NumDecodeTokens: 0} // incomplete

	// Only r1 and r2 completed prefill
	m.RequestTTFTs["r1"] = 100.0
	m.RequestTTFTs["r2"] = 200.0
	m.RequestE2Es["r1"] = 500.0
	m.RequestE2Es["r2"] = 1000.0
	m.RequestITLs["r1"] = 50.0
	m.RequestITLs["r2"] = 100.0
	m.RequestSchedulingDelays["r1"] = 10
	m.RequestSchedulingDelays["r2"] = 20

	m.CompletedRequests = 2
	m.TotalOutputTokens = 15
	m.SimEndedTime = 1_000_000
	m.AllITLs = []int64{50, 100} // required to avoid CalculatePercentile empty-input panic (Phase 5, 5c)

	// WHEN SaveResults writes to a temp file
	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "results.json")
	m.SaveResults("test-instance", 10_000_000, 100, outPath)

	// THEN the output file contains all 3 requests
	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("failed to read output: %v", err)
	}
	var output MetricsOutput
	if err := json.Unmarshal(data, &output); err != nil {
		t.Fatalf("failed to parse output: %v", err)
	}

	if len(output.Requests) != 3 {
		t.Errorf("output.Requests count = %d, want 3 (all registered, including incomplete)", len(output.Requests))
	}

	// Verify incomplete request r3 has zero-valued metrics
	for _, req := range output.Requests {
		if req.ID == "r3" {
			if req.TTFT != 0 || req.E2E != 0 || req.ITL != 0 {
				t.Errorf("incomplete request r3 should have zero metrics, got TTFT=%f E2E=%f ITL=%f",
					req.TTFT, req.E2E, req.ITL)
			}
			return
		}
	}
	t.Error("incomplete request r3 not found in output")
}

func TestSaveResults_NoWallClockFields(t *testing.T) {
	// GIVEN a Metrics struct with completed requests
	m := NewMetrics()
	m.CompletedRequests = 1
	m.SimEndedTime = 1_000_000
	m.TotalInputTokens = 100
	m.TotalOutputTokens = 100
	m.RequestTTFTs["req1"] = 10.0
	m.RequestE2Es["req1"] = 100.0
	m.AllITLs = []int64{10}
	m.RequestSchedulingDelays["req1"] = 5

	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "results.json")

	// WHEN SaveResults writes output
	m.SaveResults("test", 1_000_000, 1000, outPath)

	// THEN the JSON must not contain wall-clock fields
	data, err := os.ReadFile(outPath)
	require.NoError(t, err)
	jsonStr := string(data)
	assert.NotContains(t, jsonStr, "simulation_duration_s")
	assert.NotContains(t, jsonStr, "sim_start_timestamp")
	assert.NotContains(t, jsonStr, "sim_end_timestamp")
	// But it must still contain simulation-derived fields
	assert.Contains(t, jsonStr, "vllm_estimated_duration_s")
	assert.Contains(t, jsonStr, "completed_requests")
}
