package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/sirupsen/logrus"
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
	origITL := calibrateITLDataPath
	return func() {
		calibrateTraceHeaderPath = origHeader
		calibrateTraceDataPath = origData
		calibrateSimResultsPath = origSim
		calibrateReportPath = origReport
		calibrateWarmUpRequests = origWarmUp
		calibrateNetworkRTTUs = origRTT
		calibrateNetworkBandwidthMbps = origBW
		calibrateITLDataPath = origITL
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
	csvData := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason\n" +
		"0,c1,t1,standard,s1,0,,0,true,10,5,10,0,0,0,0.0,,0,10,0,1000,5000,10000,5,ok,,stop\n" +
		"1,c1,t1,standard,s1,0,,0,true,10,5,10,0,0,0,0.0,,0,10,100000,101000,105000,110000,5,ok,,stop\n" +
		"2,c1,t1,standard,s1,0,,0,true,10,5,10,0,0,0,0.0,,0,10,200000,201000,205000,210000,5,ok,,stop\n"
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
	// THEN all 8 flags must be present
	flags := []string{
		"trace-header",
		"trace-data",
		"sim-results",
		"report",
		"warmup-requests",
		"network-rtt-us",
		"network-bandwidth-mbps",
		"itl-data",
	}
	for _, name := range flags {
		f := calibrateCmd.Flags().Lookup(name)
		if f == nil {
			t.Errorf("calibrateCmd missing flag --%s", name)
		}
	}
}

// writeTempTrace writes a TraceV2 header YAML and data CSV to dir.
// requests is a list of (requestID, sendUs, firstChunkUs, lastChunkUs).
func writeTempTrace(t *testing.T, dir, headerYAML string, rows [][4]int64) (string, string) {
	t.Helper()
	headerPath := filepath.Join(dir, "trace.yaml")
	dataPath := filepath.Join(dir, "trace.csv")
	if err := os.WriteFile(headerPath, []byte(headerYAML), 0644); err != nil {
		t.Fatal(err)
	}
	lines := make([]string, 0, len(rows)+1)
	lines = append(lines, "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason")
	for _, r := range rows {
		lines = append(lines, fmt.Sprintf("%d,c1,t1,standard,s1,0,,0,true,10,5,10,0,0,0,0.0,,0,10,%d,%d,%d,%d,5,ok,,",
			r[0], r[0]*100000, r[1], r[2], r[3]))
	}
	if err := os.WriteFile(dataPath, []byte(strings.Join(lines, "\n")+"\n"), 0644); err != nil {
		t.Fatal(err)
	}
	return headerPath, dataPath
}

func TestCalibrateCmd_WarmUpFromHeader_ExcludesFirstN(t *testing.T) {
	// GIVEN a trace header with warm_up_requests=3 and 10 requests (IDs 0-9)
	// AND --warmup-requests not set (sentinel -1)
	// WHEN blis calibrate is run
	// THEN report shows warm_up_excluded=3 and matched_pairs=7 (BC-2)
	dir := t.TempDir()
	var rows [10][4]int64
	for i := 0; i < 10; i++ {
		rows[i] = [4]int64{int64(i), int64(i)*100000 + 1000, int64(i)*100000 + 5000, int64(i)*100000 + 10000}
	}
	headerPath, dataPath := writeTempTrace(t, dir, `trace_version: 2
time_unit: microseconds
mode: real
warm_up_requests: 3
`, [][4]int64(rows[:]))

	simPath := filepath.Join(dir, "results.json")
	simResults := make([]workload.SimResult, 10)
	for i := 0; i < 10; i++ {
		simResults[i] = workload.SimResult{RequestID: i, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5}
	}
	simData, _ := json.Marshal(simResults)
	if err := os.WriteFile(simPath, simData, 0644); err != nil {
		t.Fatal(err)
	}

	reportPath := filepath.Join(dir, "report.json")
	defer saveRestoreCalibrateFlags()()
	calibrateTraceHeaderPath = headerPath
	calibrateTraceDataPath = dataPath
	calibrateSimResultsPath = simPath
	calibrateReportPath = reportPath
	calibrateWarmUpRequests = -1 // sentinel: use header
	calibrateNetworkRTTUs = -1
	calibrateNetworkBandwidthMbps = 0

	calibrateCmd.Run(calibrateCmd, []string{})

	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("report not written: %v", err)
	}
	var report workload.CalibrationReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("report is not valid JSON: %v", err)
	}
	if report.TraceInfo.WarmUpExcluded != 3 {
		t.Errorf("warm_up_excluded = %d, want 3 (from header)", report.TraceInfo.WarmUpExcluded)
	}
	if report.TraceInfo.MatchedPairs != 7 {
		t.Errorf("matched_pairs = %d, want 7", report.TraceInfo.MatchedPairs)
	}
}

func TestCalibrateCmd_WarmUpFlagOverridesHeader(t *testing.T) {
	// GIVEN a trace header with warm_up_requests=3 and 10 requests
	// AND --warmup-requests 0 explicitly set
	// WHEN blis calibrate is run
	// THEN report shows warm_up_excluded=0 and matched_pairs=10 (BC-3)
	dir := t.TempDir()
	var rows [10][4]int64
	for i := 0; i < 10; i++ {
		rows[i] = [4]int64{int64(i), int64(i)*100000 + 1000, int64(i)*100000 + 5000, int64(i)*100000 + 10000}
	}
	headerPath, dataPath := writeTempTrace(t, dir, `trace_version: 2
time_unit: microseconds
mode: real
warm_up_requests: 3
`, [][4]int64(rows[:]))

	simPath := filepath.Join(dir, "results.json")
	simResults := make([]workload.SimResult, 10)
	for i := 0; i < 10; i++ {
		simResults[i] = workload.SimResult{RequestID: i, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5}
	}
	simData, _ := json.Marshal(simResults)
	if err := os.WriteFile(simPath, simData, 0644); err != nil {
		t.Fatal(err)
	}

	reportPath := filepath.Join(dir, "report.json")
	defer saveRestoreCalibrateFlags()()
	calibrateTraceHeaderPath = headerPath
	calibrateTraceDataPath = dataPath
	calibrateSimResultsPath = simPath
	calibrateReportPath = reportPath
	calibrateWarmUpRequests = 0 // explicit override: include all
	calibrateNetworkRTTUs = -1
	calibrateNetworkBandwidthMbps = 0

	calibrateCmd.Run(calibrateCmd, []string{})

	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("report not written: %v", err)
	}
	var report workload.CalibrationReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("report is not valid JSON: %v", err)
	}
	if report.TraceInfo.WarmUpExcluded != 0 {
		t.Errorf("warm_up_excluded = %d, want 0 (flag override)", report.TraceInfo.WarmUpExcluded)
	}
	if report.TraceInfo.MatchedPairs != 10 {
		t.Errorf("matched_pairs = %d, want 10", report.TraceInfo.MatchedPairs)
	}
}

func TestCalibrateCmd_UnmatchedRequests_ReportSucceeds(t *testing.T) {
	// GIVEN a trace with IDs [0,1,2] and sim results with IDs [0,1,3]
	// WHEN blis calibrate is run
	// THEN it succeeds and the report shows 2 matched pairs (BC-5)
	dir := t.TempDir()
	headerPath, dataPath := writeTempTrace(t, dir, `trace_version: 2
time_unit: microseconds
mode: real
warm_up_requests: 0
`, [][4]int64{
		{0, 1000, 5000, 10000},
		{1, 101000, 105000, 110000},
		{2, 201000, 205000, 210000},
	})

	simPath := filepath.Join(dir, "results.json")
	// Sim has IDs 0, 1, 3 — ID 2 missing, extra ID 3
	simResults := []workload.SimResult{
		{RequestID: 0, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5},
		{RequestID: 1, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5},
		{RequestID: 3, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5},
	}
	simData, _ := json.Marshal(simResults)
	if err := os.WriteFile(simPath, simData, 0644); err != nil {
		t.Fatal(err)
	}

	reportPath := filepath.Join(dir, "report.json")
	defer saveRestoreCalibrateFlags()()
	calibrateTraceHeaderPath = headerPath
	calibrateTraceDataPath = dataPath
	calibrateSimResultsPath = simPath
	calibrateReportPath = reportPath
	calibrateWarmUpRequests = -1
	calibrateNetworkRTTUs = -1
	calibrateNetworkBandwidthMbps = 0

	calibrateCmd.Run(calibrateCmd, []string{})

	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("report not written: %v", err)
	}
	var report workload.CalibrationReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("report is not valid JSON: %v", err)
	}
	if report.TraceInfo.MatchedPairs != 2 {
		t.Errorf("matched_pairs = %d, want 2", report.TraceInfo.MatchedPairs)
	}
}

func TestCalibrateCmd_RTTFromHeader_AppliesCorrectly(t *testing.T) {
	// GIVEN a trace header with network.measured_rtt_ms=2.0 and --network-rtt-us not set
	// AND real TTFT = simTTFT + 2000µs (exactly what RTT=2ms should add)
	// WHEN blis calibrate is run
	// THEN TTFT MAPE ≈ 0.0 (sim+RTT = real), verifying the header was read (BC-4)
	dir := t.TempDir()
	// Use 5 requests with varying simTTFT so Pearson r is computable
	simTTFTs := []int64{3000, 4000, 5000, 6000, 7000}
	rows := make([][4]int64, 5)
	for i, st := range simTTFTs {
		send := int64(i)*100000 + 1000
		firstChunk := send + st + 2000 // realTTFT = simTTFT + RTT(2000µs)
		lastChunk := firstChunk + 5000
		rows[i] = [4]int64{int64(i), send, firstChunk, lastChunk}
	}
	headerPath, dataPath := writeTempTrace(t, dir, `trace_version: 2
time_unit: microseconds
mode: real
warm_up_requests: 0
network:
  measured_rtt_ms: 2.0
`, rows)

	simPath := filepath.Join(dir, "results.json")
	simResults := []workload.SimResult{
		{RequestID: 0, TTFT: 3000, E2E: 8000, InputTokens: 10, OutputTokens: 5},
		{RequestID: 1, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5},
		{RequestID: 2, TTFT: 5000, E2E: 10000, InputTokens: 10, OutputTokens: 5},
		{RequestID: 3, TTFT: 6000, E2E: 11000, InputTokens: 10, OutputTokens: 5},
		{RequestID: 4, TTFT: 7000, E2E: 12000, InputTokens: 10, OutputTokens: 5},
	}
	simData, _ := json.Marshal(simResults)
	if err := os.WriteFile(simPath, simData, 0644); err != nil {
		t.Fatal(err)
	}

	reportPath := filepath.Join(dir, "report.json")
	defer saveRestoreCalibrateFlags()()
	calibrateTraceHeaderPath = headerPath
	calibrateTraceDataPath = dataPath
	calibrateSimResultsPath = simPath
	calibrateReportPath = reportPath
	calibrateWarmUpRequests = -1
	calibrateNetworkRTTUs = -1 // sentinel: use header (2ms = 2000µs)
	calibrateNetworkBandwidthMbps = 0

	calibrateCmd.Run(calibrateCmd, []string{})

	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("report not written: %v", err)
	}
	var report workload.CalibrationReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("report is not valid JSON: %v", err)
	}
	ttftMetric, ok := report.Metrics["ttft"]
	if !ok {
		t.Fatal("report missing metrics.ttft")
	}
	// RTT applied → sim+RTT = real → MAPE ≈ 0
	if ttftMetric.RequestLevel.MAPE > 0.001 {
		t.Errorf("TTFT RequestLevel.MAPE = %.4f, want ~0.0 (RTT from header not applied correctly)", ttftMetric.RequestLevel.MAPE)
	}
}

func TestCalibrateCmd_ITLDataFlag_Defined(t *testing.T) {
	// GIVEN the calibrate command
	// WHEN checking for --itl-data flag
	// THEN flag is defined and optional (BC-6, BC-11)
	cmd := calibrateCmd

	flag := cmd.Flags().Lookup("itl-data")
	if flag == nil {
		t.Fatal("--itl-data flag not defined")
	}
	if flag.DefValue != "" {
		t.Errorf("--itl-data default should be empty (optional), got %q", flag.DefValue)
	}
}

func TestCalibrateCmd_WithITLData_IncludesITLMetric(t *testing.T) {
	// GIVEN a valid trace, sim results, and ITL CSV
	// WHEN calibrate is run with --itl-data
	// THEN report includes ITL metric with Count > 0
	dir := t.TempDir()

	// Write trace (2 requests)
	rows := [][4]int64{
		{0, 1000, 5000, 10000},
		{1, 101000, 105000, 110000},
	}
	headerPath, dataPath := writeTempTrace(t, dir, `trace_version: 2
time_unit: microseconds
mode: real
warm_up_requests: 0
`, rows)

	// Write sim results
	simPath := filepath.Join(dir, "results.json")
	simResults := []workload.SimResult{
		{RequestID: 0, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5},
		{RequestID: 1, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5},
	}
	simData, _ := json.Marshal(simResults)
	if err := os.WriteFile(simPath, simData, 0644); err != nil {
		t.Fatal(err)
	}

	// Write ITL CSV (3 chunks per request)
	itlPath := filepath.Join(dir, "trace.itl.csv")
	itlCSV := "request_id,chunk_index,timestamp_us\n" +
		"0,0,5000\n0,1,5020\n0,2,5040\n" +
		"1,0,105000\n1,1,105020\n1,2,105040\n"
	if err := os.WriteFile(itlPath, []byte(itlCSV), 0644); err != nil {
		t.Fatal(err)
	}

	reportPath := filepath.Join(dir, "report.json")
	defer saveRestoreCalibrateFlags()()
	calibrateTraceHeaderPath = headerPath
	calibrateTraceDataPath = dataPath
	calibrateSimResultsPath = simPath
	calibrateReportPath = reportPath
	calibrateITLDataPath = itlPath
	calibrateWarmUpRequests = -1
	calibrateNetworkRTTUs = -1
	calibrateNetworkBandwidthMbps = 0

	calibrateCmd.Run(calibrateCmd, []string{})

	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("report not written: %v", err)
	}
	var report workload.CalibrationReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("report is not valid JSON: %v", err)
	}

	// Assert ITL metric is present
	itlMetric, ok := report.Metrics["itl"]
	if !ok {
		t.Fatal("report.Metrics[\"itl\"] not found")
	}
	if itlMetric.Count == 0 {
		t.Error("ITL metric Count should be > 0")
	}
}

func TestCalibrateCmd_LogsIncludeMeanError(t *testing.T) {
	// GIVEN a calibration run with known real and sim latencies
	// WHEN blis calibrate is run
	// THEN CLI output includes MeanError summary (BC-8)
	// AND JSON report includes real_mean, sim_mean, mean_error fields

	dir := t.TempDir()
	// 3 requests: real TTFT=4000, sim TTFT=4200 → mean error = +200µs
	rows := [][4]int64{
		{0, 1000, 5000, 10000},   // realTTFT = 5000-1000 = 4000
		{1, 101000, 105000, 110000}, // realTTFT = 105000-101000 = 4000
		{2, 201000, 205000, 210000}, // realTTFT = 205000-201000 = 4000
	}
	headerPath, dataPath := writeTempTrace(t, dir, `trace_version: 2
time_unit: microseconds
mode: real
warm_up_requests: 0
`, rows)

	simPath := filepath.Join(dir, "results.json")
	simResults := []workload.SimResult{
		{RequestID: 0, TTFT: 4200, E2E: 9200, InputTokens: 10, OutputTokens: 5},
		{RequestID: 1, TTFT: 4200, E2E: 9200, InputTokens: 10, OutputTokens: 5},
		{RequestID: 2, TTFT: 4200, E2E: 9200, InputTokens: 10, OutputTokens: 5},
	}
	simData, _ := json.Marshal(simResults)
	if err := os.WriteFile(simPath, simData, 0644); err != nil {
		t.Fatal(err)
	}

	reportPath := filepath.Join(dir, "report.json")
	defer saveRestoreCalibrateFlags()()
	calibrateTraceHeaderPath = headerPath
	calibrateTraceDataPath = dataPath
	calibrateSimResultsPath = simPath
	calibrateReportPath = reportPath
	calibrateWarmUpRequests = -1
	calibrateNetworkRTTUs = -1
	calibrateNetworkBandwidthMbps = 0
	calibrateITLDataPath = ""

	// Capture logrus output to check CLI format
	var logBuf bytes.Buffer
	logrus.SetOutput(&logBuf)
	origLevel := logrus.GetLevel()
	logrus.SetLevel(logrus.InfoLevel)
	defer func() {
		logrus.SetOutput(os.Stderr)
		logrus.SetLevel(origLevel)
	}()

	calibrateCmd.Run(calibrateCmd, []string{})

	// THEN CLI output includes Workload-level section with mean error
	logOutput := logBuf.String()
	if !strings.Contains(logOutput, "Workload-level aggregate metrics") {
		t.Errorf("CLI output missing 'Workload-level aggregate metrics' section:\n%s", logOutput)
	}
	if !strings.Contains(logOutput, "Error=+200") {
		t.Errorf("CLI output missing expected Error=+200 in workload-level section:\n%s", logOutput)
	}
	// AND includes Request-level prediction quality section
	if !strings.Contains(logOutput, "Request-level prediction quality") {
		t.Errorf("CLI output missing 'Request-level prediction quality' section:\n%s", logOutput)
	}

	// THEN JSON report includes new aggregate fields
	reportData, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatal(err)
	}
	reportStr := string(reportData)
	if !strings.Contains(reportStr, "\"real_mean\":") {
		t.Errorf("JSON report missing real_mean field:\n%s", reportStr)
	}
	if !strings.Contains(reportStr, "\"mean_error\":") {
		t.Errorf("JSON report missing mean_error field:\n%s", reportStr)
	}
	if !strings.Contains(reportStr, "\"sim_mean\":") {
		t.Errorf("JSON report missing sim_mean field:\n%s", reportStr)
	}
}
