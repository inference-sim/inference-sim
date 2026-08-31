package cmd

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/sirupsen/logrus"
)

// writeThroughputFixture writes a 3-request TraceV2 + matching SimResults where the sim
// finishes each request faster than the real server, so real and sim throughput differ.
func writeThroughputFixture(t *testing.T, dir string) (headerPath, dataPath, simPath string) {
	t.Helper()
	headerPath = filepath.Join(dir, "trace.yaml")
	dataPath = filepath.Join(dir, "trace.csv")
	simPath = filepath.Join(dir, "results.json")

	header := "trace_version: 2\ntime_unit: microseconds\nmode: real\nwarm_up_requests: 0\n"
	if err := os.WriteFile(headerPath, []byte(header), 0644); err != nil {
		t.Fatal(err)
	}
	// send=0/1e6/2e6, last-chunk each +1e6 → real makespan 3e6 (3s), 30 output tokens → 10 tok/s.
	csv := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason\n" +
		"0,c1,t1,standard,s1,0,,0,true,10,10,10,0,0,0,0.0,,0,10,0,0,500000,1000000,10,ok,,stop\n" +
		"1,c1,t1,standard,s1,0,,0,true,10,10,10,0,0,0,0.0,,0,10,1000000,1000000,1500000,2000000,10,ok,,stop\n" +
		"2,c1,t1,standard,s1,0,,0,true,10,10,10,0,0,0,0.0,,0,10,2000000,2000000,2500000,3000000,10,ok,,stop\n"
	if err := os.WriteFile(dataPath, []byte(csv), 0644); err != nil {
		t.Fatal(err)
	}
	// Sim E2E = 0.5e6 each → sim makespan = 2e6 + 0.5e6 - 0 = 2.5e6 (2.5s) → 12 tok/s.
	simResults := []workload.SimResult{
		{RequestID: 0, TTFT: 250000, E2E: 500000, InputTokens: 10, OutputTokens: 10},
		{RequestID: 1, TTFT: 250000, E2E: 500000, InputTokens: 10, OutputTokens: 10},
		{RequestID: 2, TTFT: 250000, E2E: 500000, InputTokens: 10, OutputTokens: 10},
	}
	simData, _ := json.Marshal(simResults)
	if err := os.WriteFile(simPath, simData, 0644); err != nil {
		t.Fatal(err)
	}
	return headerPath, dataPath, simPath
}

// TestCalibrateCmd_Throughput_Present verifies BC-1: the report gains a throughput block
// with real & sim output-token and request throughput from the required inputs.
func TestCalibrateCmd_Throughput_Present(t *testing.T) {
	dir := t.TempDir()
	headerPath, dataPath, simPath := writeThroughputFixture(t, dir)
	reportPath := filepath.Join(dir, "report.json")

	defer saveRestoreCalibrateFlags()()
	calibrateTraceHeaderPath = headerPath
	calibrateTraceDataPath = dataPath
	calibrateSimResultsPath = simPath
	calibrateReportPath = reportPath
	calibrateWarmUpRequests = -1
	calibrateNetworkRTTUs = -1
	calibrateNetworkBandwidthMbps = 0
	calibrateNumGPUs = 0
	calibrateThroughputTolerancePct = 0

	calibrateCmd.Run(calibrateCmd, []string{})

	var report workload.CalibrationReport
	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("report not written: %v", err)
	}
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	tp := report.Throughput
	if tp == nil {
		t.Fatal("report.throughput should be populated from required inputs")
	}
	if tp.MatchedRequests != 3 {
		t.Errorf("matched = %d, want 3", tp.MatchedRequests)
	}
	if !almostEq(tp.RealOutputTokensPerSec, 10, 0.01) {
		t.Errorf("real output tok/s = %v, want ~10", tp.RealOutputTokensPerSec)
	}
	if !almostEq(tp.SimOutputTokensPerSec, 12, 0.01) {
		t.Errorf("sim output tok/s = %v, want ~12", tp.SimOutputTokensPerSec)
	}
	// Sim faster → sim throughput higher → positive error.
	if tp.OutputTokensPerSecError <= 0 {
		t.Errorf("expected positive error (sim faster), got %v", tp.OutputTokensPerSecError)
	}
	// Optional pointers absent without flags (BC-4/BC-5).
	if tp.NumGPUs != nil || tp.Within != nil {
		t.Errorf("optional pointers should be nil, got numGPUs=%v within=%v", tp.NumGPUs, tp.Within)
	}
}

// TestCalibrateCmd_Throughput_PerGPUAndVerdict verifies BC-4 and BC-5: --num-gpus adds
// per-GPU fields; --throughput-tolerance-pct adds the within verdict.
func TestCalibrateCmd_Throughput_PerGPUAndVerdict(t *testing.T) {
	dir := t.TempDir()
	headerPath, dataPath, simPath := writeThroughputFixture(t, dir)
	reportPath := filepath.Join(dir, "report.json")

	defer saveRestoreCalibrateFlags()()
	calibrateTraceHeaderPath = headerPath
	calibrateTraceDataPath = dataPath
	calibrateSimResultsPath = simPath
	calibrateReportPath = reportPath
	calibrateWarmUpRequests = -1
	calibrateNetworkRTTUs = -1
	calibrateNetworkBandwidthMbps = 0
	calibrateNumGPUs = 4
	calibrateThroughputTolerancePct = 15

	calibrateCmd.Run(calibrateCmd, []string{})

	var report workload.CalibrationReport
	data, _ := os.ReadFile(reportPath)
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	tp := report.Throughput
	if tp == nil || tp.NumGPUs == nil || tp.RealOutputTokensPerSecPerGPU == nil {
		t.Fatal("expected per-GPU fields with --num-gpus 4")
	}
	if *tp.NumGPUs != 4 {
		t.Errorf("num_gpus = %d, want 4", *tp.NumGPUs)
	}
	if !almostEq(*tp.RealOutputTokensPerSecPerGPU, tp.RealOutputTokensPerSec/4, 1e-6) {
		t.Errorf("per-GPU = %v, want raw/4", *tp.RealOutputTokensPerSecPerGPU)
	}
	// ~20% throughput error (10 vs 12) exceeds a 15% band.
	if tp.Within == nil {
		t.Fatal("expected within verdict with --throughput-tolerance-pct 15")
	}
	if *tp.Within {
		t.Errorf("~20%% error should EXCEED 15%% band, percentError=%v", tp.OutputTokensPerSecPercentError)
	}
	if tp.TolerancePct == nil || *tp.TolerancePct != 15 {
		t.Errorf("tolerance_pct = %v, want 15", tp.TolerancePct)
	}
}

func almostEq(a, b, tol float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}

// TestCalibrateCmd_Throughput_FlagValidation verifies BC-6: a negative --num-gpus and a
// negative/NaN/Inf --throughput-tolerance-pct are rejected loudly (logrus.Fatalf → exit),
// never silently accepted. Intercepts the fatal exit via logrus ExitFunc.
func TestCalibrateCmd_Throughput_FlagValidation(t *testing.T) {
	dir := t.TempDir()
	headerPath, dataPath, simPath := writeThroughputFixture(t, dir)
	reportPath := filepath.Join(dir, "report.json")

	cases := []struct {
		name    string
		numGPUs int
		tolPct  float64
	}{
		{"negative num-gpus", -1, 0},
		{"negative tolerance", 0, -5},
		{"NaN tolerance", 0, math.NaN()},
		{"Inf tolerance", 0, math.Inf(1)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer saveRestoreCalibrateFlags()()
			calibrateTraceHeaderPath = headerPath
			calibrateTraceDataPath = dataPath
			calibrateSimResultsPath = simPath
			calibrateReportPath = reportPath
			calibrateWarmUpRequests = -1
			calibrateNetworkRTTUs = -1
			calibrateNetworkBandwidthMbps = 0
			calibrateNumGPUs = tc.numGPUs
			calibrateThroughputTolerancePct = tc.tolPct

			// Intercept logrus.Fatalf's exit so the test observes the guard firing
			// instead of terminating the process.
			exited := false
			logger := logrus.StandardLogger()
			origExit := logger.ExitFunc
			logger.ExitFunc = func(int) { exited = true; panic("fatal") }
			defer func() {
				logger.ExitFunc = origExit
				if r := recover(); r != "fatal" {
					t.Fatalf("expected fatal guard to fire for %s, recover=%v", tc.name, r)
				}
				if !exited {
					t.Errorf("expected ExitFunc to be invoked for %s", tc.name)
				}
			}()
			calibrateCmd.Run(calibrateCmd, []string{})
			t.Fatalf("expected %s to trigger a fatal exit, but Run returned", tc.name)
		})
	}
}
