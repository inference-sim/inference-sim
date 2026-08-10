package cmd

import (
	"path/filepath"
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/latency"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// buildSpecDecodeParityConfig builds a PD DeploymentConfig with speculative decoding
// enabled (#1528). PD (prefill+decode instances) is chosen deliberately so the test
// exercises the PD decode-sub-request spec-decode branch (batch_formation.go), which
// has distinct control flow from the Phase-1 decode branch.
func buildSpecDecodeParityConfig(t *testing.T, seed int64, k int, acc float64) cluster.DeploymentConfig {
	t.Helper()
	mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)
	hfConfig, err := latency.ParseHFConfig(filepath.Join(mcFolder, "config.json"))
	if err != nil {
		t.Fatalf("ParseHFConfig: %v", err)
	}
	mc, err := latency.GetModelConfigFromHF(hfConfig)
	if err != nil {
		t.Fatalf("GetModelConfigFromHF: %v", err)
	}
	hwCfg, err := latency.GetHWConfig(hwPath, "H100")
	if err != nil {
		t.Fatalf("GetHWConfig: %v", err)
	}
	beta := []float64{0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0}
	alpha := []float64{100.0, 1.0, 100.0}

	return cluster.DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             10_000_000,
			Seed:                seed,
			KVCacheConfig:       sim.NewKVCacheConfig(1000, 16, 0, 0.9, 100.0, 0),
			BatchConfig:         sim.NewBatchConfig(64, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs(beta, alpha),
			ModelHardwareConfig: sim.NewModelHardwareConfig(*mc, hwCfg, "test-model", "H100", 1, 1, false, "", "trained-physics", 4096),
			PolicyConfig:        sim.NewPolicyConfig("fcfs", ""),
			// Speculative decoding / MTP enabled — the config under test.
			SpeculativeConfig: sim.SpeculativeConfig{K: k, Acceptance: acc, Method: "mtp"},
		},
		NumInstances:            2,
		AdmissionPolicy:         "always-admit",
		RoutingPolicy:           "round-robin",
		PrefillInstances:        1,
		DecodeInstances:         1,
		PDDecider:               "always",
		PDTransferBandwidthGBps: 25.0,
		PDTransferBaseLatencyMs: 0.05,
	}
}

// TestINV13_RunReplayParity_SpecDecode pins INV-13 / BC-7 for speculative decoding:
// a direct run and a replay of that run's exported trace — both with the SAME
// spec-decode config — produce byte-identical per-request TTFT/E2E. This is the true
// run-vs-replay law (the run leg's metrics are captured from cs1, not a second
// replay). Using a PD deployment also exercises the PD decode-sub-request spec-decode
// branch, whose break-on-<1 control flow differs from the Phase-1 decode branch.
//
// The spec-decode config is model-level and identical on both legs (not persisted in
// the trace) — the same mechanism as --tp / --max-model-len — so INV-13 must hold.
func TestINV13_RunReplayParity_SpecDecode(t *testing.T) {
	const seed int64 = 99
	const k = 3
	const acc = 0.6

	cfg := buildSpecDecodeParityConfig(t, seed, k, acc)
	requests := makeMinimalPDRequests(t)
	dir := t.TempDir()

	// Direct run leg — capture ITS per-request metrics (not a replay).
	cs1 := cluster.NewClusterSimulator(cfg, cluster.NewSliceRequestSource(requests), nil)
	if err := cs1.Run(); err != nil {
		t.Fatalf("direct run failed: %v", err)
	}
	runTTFTs := cs1.AggregatedMetrics().RequestTTFTs
	runE2Es := cs1.AggregatedMetrics().RequestE2Es
	if len(runTTFTs) == 0 {
		t.Fatal("INV-13 spec-decode: direct run produced no completed requests — vacuous")
	}

	// Export → reload → replay leg with the SAME config.
	traceRecords := workload.RequestsToTraceRecords(requests)
	traceHdr := &workload.TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "generated"}
	hdrFile := filepath.Join(dir, "trace.yaml")
	dataFile := filepath.Join(dir, "trace.csv")
	if err := workload.ExportTraceV2(traceHdr, traceRecords, hdrFile, dataFile); err != nil {
		t.Fatalf("ExportTraceV2: %v", err)
	}
	traceData, err := workload.LoadTraceV2(hdrFile, dataFile)
	if err != nil {
		t.Fatalf("LoadTraceV2: %v", err)
	}
	replayReqs, err := workload.LoadTraceV2Requests(traceData, seed)
	if err != nil {
		t.Fatalf("LoadTraceV2Requests: %v", err)
	}
	cs2 := cluster.NewClusterSimulator(cfg, cluster.NewSliceRequestSource(replayReqs), nil)
	if err := cs2.Run(); err != nil {
		t.Fatalf("replay run failed: %v", err)
	}
	replayTTFTs := cs2.AggregatedMetrics().RequestTTFTs
	replayE2Es := cs2.AggregatedMetrics().RequestE2Es

	// Per-request metrics must be identical run-vs-replay (INV-13).
	if len(runTTFTs) != len(replayTTFTs) {
		t.Errorf("INV-13 spec-decode: TTFT map size run=%d replay=%d", len(runTTFTs), len(replayTTFTs))
	}
	for id, ttft := range runTTFTs {
		if got, ok := replayTTFTs[id]; !ok {
			t.Errorf("INV-13 spec-decode: request %s in run, missing from replay TTFTs", id)
		} else if got != ttft {
			t.Errorf("INV-13 spec-decode: request %s TTFT run=%f replay=%f", id, ttft, got)
		}
	}
	for id, e2e := range runE2Es {
		if got, ok := replayE2Es[id]; !ok {
			t.Errorf("INV-13 spec-decode: request %s in run, missing from replay E2Es", id)
		} else if got != e2e {
			t.Errorf("INV-13 spec-decode: request %s E2E run=%f replay=%f", id, e2e, got)
		}
	}
}

// TestSpecDecode_PDDecodeSubRequest_Completes exercises the PD decode-sub-request
// spec-decode branch directly: with spec-decode enabled and a PD deployment, all
// requests must still complete (INV-11) and conserve output tokens (INV-1). This
// covers the branch's break-on-<1 / floor-at-1 control flow that differs from the
// Phase-1 decode path.
func TestSpecDecode_PDDecodeSubRequest_Completes(t *testing.T) {
	cfg := buildSpecDecodeParityConfig(t, 7, 4, 0.75)
	requests := makeMinimalPDRequests(t)

	cs := cluster.NewClusterSimulator(cfg, cluster.NewSliceRequestSource(requests), nil)
	if err := cs.Run(); err != nil {
		t.Fatalf("PD spec-decode run failed: %v", err)
	}
	m := cs.AggregatedMetrics()
	if got := len(m.RequestE2Es); got != len(requests) {
		t.Errorf("PD spec-decode: completed=%d, want %d (INV-11: every request reaches a terminal state)", got, len(requests))
	}
	// Output tokens conserved: each request had 5 output tokens (makeMinimalPDRequests).
	if m.TotalOutputTokens != 5*len(requests) {
		t.Errorf("PD spec-decode: TotalOutputTokens=%d, want %d (INV-1 conservation)", m.TotalOutputTokens, 5*len(requests))
	}
}
