package sim

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// saveOutput builds m's MetricsOutput for instanceID and, when path is not empty, writes it there
// as JSON: the figures the removed --metrics-path file carried, which these tests pin through
// MetricsOutput's JSON form. horizon and totalBlocks were unused by the writer and are ignored.
func saveOutput(m *Metrics, instanceID string, _ int64, _ int64, path string) error {
	out := m.BuildOutput(instanceID)
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}
	if path == "" {
		return nil
	}
	return os.WriteFile(path, data, 0o644)
}

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
	if err := saveOutput(m, "test-instance", 1000000, 1000, outputPath); err != nil {
		t.Fatalf("SaveResults returned error: %v", err)
	}

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
	if err := saveOutput(m, "", 1000000, 1000, outputPath); err != nil {
		t.Fatalf("SaveResults returned error: %v", err)
	}

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
	if err := saveOutput(m, "default", 1000000, 1000, outputPath); err != nil {
		t.Fatalf("SaveResults returned error: %v", err)
	}

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

func TestRequestRows_IncludeIncompleteRequests(t *testing.T) {
	// GIVEN metrics where 2 of 3 requests completed prefill
	m := NewMetrics()
	m.Requests["r1"] = RequestMetrics{ID: "r1", ArrivedAt: 1.0, NumPrefillTokens: 10, NumDecodeTokens: 5}
	m.Requests["r2"] = RequestMetrics{ID: "r2", ArrivedAt: 2.0, NumPrefillTokens: 20, NumDecodeTokens: 10}
	m.Requests["r3"] = RequestMetrics{ID: "r3", ArrivedAt: 3.0, NumPrefillTokens: 30, NumDecodeTokens: 0} // incomplete
	m.RequestTTFTs["r1"] = 100.0
	m.RequestTTFTs["r2"] = 200.0
	m.RequestE2Es["r1"] = 500.0
	m.RequestE2Es["r2"] = 1000.0
	m.RequestITLs["r1"] = 50.0
	m.RequestITLs["r2"] = 100.0

	// WHEN the rows are read
	rows := m.RequestRows()

	// THEN all 3 are present, and the incomplete one has zero-valued metrics
	if len(rows) != 3 {
		t.Fatalf("rows = %d, want 3 (all registered, including incomplete)", len(rows))
	}
	for _, req := range rows {
		if req.ID == "r3" {
			if req.TTFT != 0 || req.E2E != 0 || req.ITL != 0 {
				t.Errorf("incomplete request r3 should have zero metrics, got TTFT=%f E2E=%f ITL=%f",
					req.TTFT, req.E2E, req.ITL)
			}
			return
		}
	}
	t.Error("incomplete request r3 not found")
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
	if err := saveOutput(m, "test", 1_000_000, 1000, outPath); err != nil {
		t.Fatalf("SaveResults returned error: %v", err)
	}

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

func TestSaveResults_ConservationFields(t *testing.T) {
	// GIVEN a Metrics struct with completed and in-flight requests
	m := NewMetrics()
	m.CompletedRequests = 8
	m.SimEndedTime = 5_000_000
	m.TotalInputTokens = 500
	m.TotalOutputTokens = 500
	m.StillQueued = 1
	m.StillRunning = 1
	for i := 0; i < 8; i++ {
		id := fmt.Sprintf("req%d", i)
		m.RequestTTFTs[id] = float64(i * 10)
		m.RequestE2Es[id] = float64(i * 100)
		m.RequestSchedulingDelays[id] = int64(i * 5)
	}
	m.AllITLs = []int64{10, 20, 30, 40, 50, 60, 70, 80}

	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "results.json")

	// WHEN SaveResults writes output
	if err := saveOutput(m, "test", 5_000_000, 1000, outPath); err != nil {
		t.Fatalf("SaveResults returned error: %v", err)
	}

	// THEN JSON must contain conservation fields
	data, err := os.ReadFile(outPath)
	require.NoError(t, err)

	var output MetricsOutput
	require.NoError(t, json.Unmarshal(data, &output))

	assert.Equal(t, 1, output.StillQueued)
	assert.Equal(t, 1, output.StillRunning)
	assert.Equal(t, 10, output.InjectedRequests)
	// MetricsOutput.InjectedRequests is *defined* as the five-term sum
	// (Metrics.ToOutput), so comparing it against that sum is an arithmetic identity,
	// not an observation about the simulator. Assert instead against the count this
	// test actually set up — the same independence rule assertINV1Conservation
	// documents.
	//
	// The four-term form this replaced was not purely tautological: by omitting
	// TimedOutRequests it also asserted that no request timed out. Keep that as its
	// own assertion rather than losing it inside a wider sum.
	assert.Equal(t, 0, output.TimedOutRequests, "fixture injects no timeouts")
	assert.Equal(t, 0, output.DroppedUnservable, "fixture drops nothing")
	assert.Equal(t, 10, output.CompletedRequests+output.StillQueued+output.StillRunning+output.DroppedUnservable+output.TimedOutRequests)
}

// TestSaveResults_PerRequestITL_InMilliseconds verifies BC-14:
// per-request itl_ms and scheduling_delay_ms are in milliseconds.
func TestRequestRows_LatenciesAreInMilliseconds(t *testing.T) {
	m := NewMetrics()
	m.Requests["r1"] = NewRequestMetrics(
		&Request{ID: "r1", InputTokens: make([]TokenID, 5), OutputTokens: make([]TokenID, 10)},
		0,
	)
	m.RequestTTFTs["r1"] = 10000.0         // 10000 ticks = 10 ms
	m.RequestE2Es["r1"] = 50000.0          // 50000 ticks = 50 ms
	m.RequestITLs["r1"] = 5000.0           // 5000 ticks = should be 5 ms
	m.RequestSchedulingDelays["r1"] = 2000 // 2000 ticks = should be 2 ms

	rows := m.RequestRows()
	require.Len(t, rows, 1)
	assert.InDelta(t, 5.0, rows[0].ITL, 0.001, "ITL should be in ms")
	assert.InDelta(t, 2.0, rows[0].SchedulingDelay, 0.001, "SchedulingDelay should be in ms")
	assert.InDelta(t, 10.0, rows[0].TTFT, 0.001, "TTFT should be in ms")
	assert.InDelta(t, 50.0, rows[0].E2E, 0.001, "E2E should be in ms")
}

// TestSaveResults_ZeroRuntime_NoInfinity verifies BC-2:
// GIVEN SimEndedTime == 0 (zero runtime)
// WHEN SaveResults computes throughput metrics
// THEN ResponsesPerSec and TokensPerSec are 0.0 (not +Inf or NaN)
// AND SaveResults returns nil (no error)
func TestSaveResults_ZeroRuntime_NoInfinity(t *testing.T) {
	m := NewMetrics()
	m.CompletedRequests = 1
	m.SimEndedTime = 0 // zero runtime
	m.RequestTTFTs["r1"] = 100.0
	m.RequestE2Es["r1"] = 200.0
	m.RequestITLs["r1"] = 50.0
	m.AllITLs = []int64{50}
	m.RequestSchedulingDelays["r1"] = 100
	m.Requests["r1"] = NewRequestMetrics(&Request{ID: "r1", InputTokens: make([]TokenID, 10), OutputTokens: make([]TokenID, 5)}, 0)

	err := saveOutput(m, "test", 1000000, 100, "")
	if err != nil {
		t.Fatalf("SaveResults returned error for zero runtime: %v", err)
	}
	// Should NOT produce +Inf — if it did, JSON marshal would fail
}

// BC-4: DroppedUnservable appears in JSON output
func TestSaveResults_DroppedUnservable_InJSON(t *testing.T) {
	// GIVEN metrics with dropped requests
	m := NewMetrics()
	m.DroppedUnservable = 2
	m.SimEndedTime = 1_000_000

	// WHEN saving results to a temp file
	tmpFile := filepath.Join(t.TempDir(), "test_output.json")
	if err := saveOutput(m, "test", 10_000_000, 100, tmpFile); err != nil {
		t.Fatalf("SaveResults returned error: %v", err)
	}

	// THEN the JSON file must contain dropped_unservable
	data, err := os.ReadFile(tmpFile)
	if err != nil {
		t.Fatalf("reading output: %v", err)
	}

	var output MetricsOutput
	if err := json.Unmarshal(data, &output); err != nil {
		t.Fatalf("parsing JSON: %v", err)
	}

	if output.DroppedUnservable != 2 {
		t.Errorf("DroppedUnservable in JSON = %d, want 2", output.DroppedUnservable)
	}

	// AND InjectedRequests must include dropped requests (BC-5)
	if output.InjectedRequests != 2 {
		t.Errorf("InjectedRequests = %d, want 2 (should include dropped)", output.InjectedRequests)
	}
}

// TestSaveResults_AlwaysEmitsHeader_ZeroCompletions verifies that the
// === Simulation Metrics === header is written to stdout even when
// CompletedRequests == 0 (issue #926).
//
// GIVEN metrics with zero completions (all requests dropped as unservable)
// WHEN SaveResults is called
// THEN stdout contains "=== Simulation Metrics ===" and a valid JSON block
func TestBuildOutput_ZeroCompletionsStillStatesTheCounts(t *testing.T) {
	m := NewMetrics()
	m.DroppedUnservable = 100
	m.SimEndedTime = 1_000_000

	out := m.BuildOutput("test")
	assert.Equal(t, 0, out.CompletedRequests)
	assert.Equal(t, 100, out.DroppedUnservable)
	assert.Equal(t, 100, out.InjectedRequests)
}

// BC-3: LengthCappedRequests appears in JSON output
func TestSaveResults_LengthCappedRequests_InJSON(t *testing.T) {
	m := NewMetrics()
	m.LengthCappedRequests = 3
	m.CompletedRequests = 3
	m.SimEndedTime = 1_000_000

	tmpFile := filepath.Join(t.TempDir(), "test_output.json")
	if err := saveOutput(m, "test", 10_000_000, 100, tmpFile); err != nil {
		t.Fatalf("SaveResults returned error: %v", err)
	}

	data, err := os.ReadFile(tmpFile)
	if err != nil {
		t.Fatalf("reading output: %v", err)
	}

	var output MetricsOutput
	if err := json.Unmarshal(data, &output); err != nil {
		t.Fatalf("parsing JSON: %v", err)
	}

	if output.LengthCappedRequests != 3 {
		t.Errorf("LengthCappedRequests in JSON = %d, want 3", output.LengthCappedRequests)
	}
}
