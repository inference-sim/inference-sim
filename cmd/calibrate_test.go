package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// saveRestoreCalibrateFlags saves all calibrate flag vars and returns a restore func.
// Use as: defer saveRestoreCalibrateFlags()()
func saveRestoreCalibrateFlags() func() {
	origHeader := calibrateTraceHeaderPath
	origData := calibrateTraceDataPath
	origSim := calibrateSimResultsPath
	origReport := calibrateReportPath
	origWarmUp := calibrateWarmUpRequests
	origRTT := calibrateNetworkRTTUs
	origBW := calibrateNetworkBandwidthMbps
	return func() {
		calibrateTraceHeaderPath = origHeader
		calibrateTraceDataPath = origData
		calibrateSimResultsPath = origSim
		calibrateReportPath = origReport
		calibrateWarmUpRequests = origWarmUp
		calibrateNetworkRTTUs = origRTT
		calibrateNetworkBandwidthMbps = origBW
	}
}

func TestCalibrateCmd_BasicReport_WritesMatchedPairs(t *testing.T) {
	// GIVEN a valid TraceV2 with 3 requests and matching SimResults
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "trace.yaml")
	dataPath := filepath.Join(dir, "trace.csv")
	simPath := filepath.Join(dir, "results.json")
	reportPath := filepath.Join(dir, "report.json")

	header := `trace_version: 2
time_unit: microseconds
mode: real
warm_up_requests: 0
`
	if err := os.WriteFile(headerPath, []byte(header), 0644); err != nil {
		t.Fatal(err)
	}
	csvData := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message\n" +
		"0,c1,t1,standard,s1,0,,true,10,5,10,0,0,0,0.0,,0,10,0,1000,5000,10000,5,ok,\n" +
		"1,c1,t1,standard,s1,0,,true,10,5,10,0,0,0,0.0,,0,10,100000,101000,105000,110000,5,ok,\n" +
		"2,c1,t1,standard,s1,0,,true,10,5,10,0,0,0,0.0,,0,10,200000,201000,205000,210000,5,ok,\n"
	if err := os.WriteFile(dataPath, []byte(csvData), 0644); err != nil {
		t.Fatal(err)
	}
	simResults := []workload.SimResult{
		{RequestID: 0, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5},
		{RequestID: 1, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5},
		{RequestID: 2, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5},
	}
	simData, _ := json.Marshal(simResults)
	if err := os.WriteFile(simPath, simData, 0644); err != nil {
		t.Fatal(err)
	}

	defer saveRestoreCalibrateFlags()()
	calibrateTraceHeaderPath = headerPath
	calibrateTraceDataPath = dataPath
	calibrateSimResultsPath = simPath
	calibrateReportPath = reportPath
	calibrateWarmUpRequests = -1
	calibrateNetworkRTTUs = -1
	calibrateNetworkBandwidthMbps = 0

	// WHEN we invoke the command Run function directly
	calibrateCmd.Run(calibrateCmd, []string{})

	// THEN the report file is written
	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("report not written: %v", err)
	}

	// THEN it parses as valid JSON with matched_pairs == 3 (BC-1)
	var report workload.CalibrationReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("report is not valid JSON: %v", err)
	}
	if report.TraceInfo.MatchedPairs != 3 {
		t.Errorf("matched_pairs = %d, want 3", report.TraceInfo.MatchedPairs)
	}
	if _, ok := report.Metrics["ttft"]; !ok {
		t.Error("report missing metrics.ttft")
	}
	if _, ok := report.Metrics["e2e"]; !ok {
		t.Error("report missing metrics.e2e")
	}
	if len(report.KnownLimitations) == 0 {
		t.Error("report.known_limitations should be non-empty")
	}
}

func TestCalibrateCmd_Flags_Registered(t *testing.T) {
	// GIVEN the calibrate command
	// WHEN we inspect its registered flags
	// THEN all 7 flags must be present
	flags := []string{
		"trace-header",
		"trace-data",
		"sim-results",
		"report",
		"warmup-requests",
		"network-rtt-us",
		"network-bandwidth-mbps",
	}
	for _, name := range flags {
		f := calibrateCmd.Flags().Lookup(name)
		if f == nil {
			t.Errorf("calibrateCmd missing flag --%s", name)
		}
	}
}
