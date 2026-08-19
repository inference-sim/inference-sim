package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// traceWithObservedHitRate writes a minimal 2-request TraceV2 whose header carries an
// observed KV hit-rate, plus a matching SimResult file. Returns the paths.
func traceWithObservedHitRate(t *testing.T, dir string, source string, hitRate float64) (headerPath, dataPath, simResultsPath string) {
	t.Helper()
	headerPath = filepath.Join(dir, "trace.yaml")
	dataPath = filepath.Join(dir, "trace.csv")
	simResultsPath = filepath.Join(dir, "results.json")

	hdr := &workload.TraceHeader{
		Version: 3, TimeUnit: "microseconds", Mode: "real",
		ObservedKVMetrics: &workload.TraceObservedKVMetrics{
			Source: source, HitRate: hitRate, BlockHits: int64(hitRate * 1000), BlockQueries: 1000,
		},
	}
	records := []workload.TraceRecord{
		{RequestID: 0, ClientID: "c1", SLOClass: "standard", Streaming: true, InputTokens: 10, OutputTokens: 5,
			ArrivalTimeUs: 0, SendTimeUs: 1000, FirstChunkTimeUs: 5000, LastChunkTimeUs: 10000, NumChunks: 5, Status: "ok"},
		{RequestID: 1, ClientID: "c1", SLOClass: "standard", Streaming: true, InputTokens: 10, OutputTokens: 5,
			ArrivalTimeUs: 100000, SendTimeUs: 101000, FirstChunkTimeUs: 105000, LastChunkTimeUs: 110000, NumChunks: 5, Status: "ok"},
	}
	if err := workload.ExportTraceV2(hdr, records, headerPath, dataPath); err != nil {
		t.Fatal(err)
	}
	simResults := []workload.SimResult{
		{RequestID: 0, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5},
		{RequestID: 1, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5},
	}
	simData, _ := json.Marshal(simResults)
	if err := os.WriteFile(simResultsPath, simData, 0644); err != nil {
		t.Fatal(err)
	}
	return headerPath, dataPath, simResultsPath
}

func writeSimMetricsFile(t *testing.T, dir string, cacheHitRate *float64) string {
	t.Helper()
	p := filepath.Join(dir, "simmetrics.json")
	m := sim.MetricsOutput{InstanceID: "cluster", CacheHitRate: cacheHitRate}
	data, _ := json.Marshal(m)
	if err := os.WriteFile(p, data, 0644); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestCalibrateCmd_HitRate_Present verifies BC-6: with an observed hit-rate in the
// header AND --sim-metrics carrying cache_hit_rate, the report gains a hit_rate block.
func TestCalibrateCmd_HitRate_Present(t *testing.T) {
	dir := t.TempDir()
	headerPath, dataPath, simResultsPath := traceWithObservedHitRate(t, dir, workload.ObservedKVSourceTiered, 0.70)
	simHR := 0.73
	simMetricsPath := writeSimMetricsFile(t, dir, &simHR)
	reportPath := filepath.Join(dir, "report.json")

	defer saveRestoreCalibrateFlags()()
	calibrateTraceHeaderPath = headerPath
	calibrateTraceDataPath = dataPath
	calibrateSimResultsPath = simResultsPath
	calibrateReportPath = reportPath
	calibrateWarmUpRequests = -1
	calibrateNetworkRTTUs = -1
	calibrateNetworkBandwidthMbps = 0
	calibrateSimMetrics = simMetricsPath
	calibrateHitRateTolerancePP = 5.0
	calibrateTTFTMapeThreshold = 0.15

	calibrateCmd.Run(calibrateCmd, []string{})

	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("report not written: %v", err)
	}
	var report workload.CalibrationReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatal(err)
	}
	if report.HitRate == nil {
		t.Fatal("report.hit_rate should be populated")
	}
	if report.HitRate.RealHitRate != 0.70 || report.HitRate.SimHitRate != 0.73 {
		t.Errorf("hit_rate real/sim = %v/%v, want 0.70/0.73", report.HitRate.RealHitRate, report.HitRate.SimHitRate)
	}
	if !report.HitRate.Within {
		t.Errorf("3pp error should be within 5pp band, got abs_error_pp=%v", report.HitRate.AbsErrorPP)
	}
	if report.HitRate.Source != workload.ObservedKVSourceTiered {
		t.Errorf("source = %q, want tiered", report.HitRate.Source)
	}
}

// TestCalibrateCmd_HitRate_SkipWhenNoSimMetrics verifies BC-12: an observed hit-rate
// with no --sim-metrics skips the hit_rate block (TTFT/E2E still computed).
func TestCalibrateCmd_HitRate_SkipWhenNoSimMetrics(t *testing.T) {
	dir := t.TempDir()
	headerPath, dataPath, simResultsPath := traceWithObservedHitRate(t, dir, workload.ObservedKVSourceTiered, 0.70)
	reportPath := filepath.Join(dir, "report.json")

	defer saveRestoreCalibrateFlags()()
	calibrateTraceHeaderPath = headerPath
	calibrateTraceDataPath = dataPath
	calibrateSimResultsPath = simResultsPath
	calibrateReportPath = reportPath
	calibrateWarmUpRequests = -1
	calibrateNetworkRTTUs = -1
	calibrateNetworkBandwidthMbps = 0
	calibrateSimMetrics = "" // no sim metrics → skip
	calibrateHitRateTolerancePP = 5.0
	calibrateTTFTMapeThreshold = 0.15

	calibrateCmd.Run(calibrateCmd, []string{})

	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("report not written: %v", err)
	}
	var report workload.CalibrationReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatal(err)
	}
	if report.HitRate != nil {
		t.Errorf("hit_rate must be skipped without --sim-metrics, got %+v", report.HitRate)
	}
	if _, ok := report.Metrics["ttft"]; !ok {
		t.Error("TTFT metric must still be computed when hit-rate is skipped")
	}
}

// TestCalibrateCmd_HitRate_SkipWhenSimMetricsLacksField verifies BC-12: --sim-metrics
// present but without cache_hit_rate (e.g. produced without --metrics-path) skips.
func TestCalibrateCmd_HitRate_SkipWhenSimMetricsLacksField(t *testing.T) {
	dir := t.TempDir()
	headerPath, dataPath, simResultsPath := traceWithObservedHitRate(t, dir, workload.ObservedKVSourceGPUCache, 0.60)
	simMetricsPath := writeSimMetricsFile(t, dir, nil) // no cache_hit_rate
	reportPath := filepath.Join(dir, "report.json")

	defer saveRestoreCalibrateFlags()()
	calibrateTraceHeaderPath = headerPath
	calibrateTraceDataPath = dataPath
	calibrateSimResultsPath = simResultsPath
	calibrateReportPath = reportPath
	calibrateWarmUpRequests = -1
	calibrateNetworkRTTUs = -1
	calibrateNetworkBandwidthMbps = 0
	calibrateSimMetrics = simMetricsPath
	calibrateHitRateTolerancePP = 5.0
	calibrateTTFTMapeThreshold = 0.15

	calibrateCmd.Run(calibrateCmd, []string{})

	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("report not written: %v", err)
	}
	var report workload.CalibrationReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatal(err)
	}
	if report.HitRate != nil {
		t.Errorf("hit_rate must be skipped when --sim-metrics lacks cache_hit_rate, got %+v", report.HitRate)
	}
}
