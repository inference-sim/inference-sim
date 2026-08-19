package cmd

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/latency"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// jitterRampOffloadCfg builds the parity DeploymentConfig with a KV-offload chain
// whose single fs tier has BOTH a queue-depth bandwidth ramp AND latency jitter
// enabled (#1581). The jitter draws from the seed-derived kv-offload RNG partition,
// so run and replay under the same seed must draw identically (BC-D6/INV-13).
func jitterRampOffloadCfg(t *testing.T, seed int64) cluster.DeploymentConfig {
	t.Helper()
	mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)

	defaultsContent := `trained_physics_coefficients:
  alpha_coeffs: [100.0, 1.0, 100.0]
  beta_coeffs: [0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
`
	defaultsPath := filepath.Join(filepath.Dir(hwPath), "defaults.yaml")
	if err := os.WriteFile(defaultsPath, []byte(defaultsContent), 0644); err != nil {
		t.Fatalf("write defaults.yaml: %v", err)
	}

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
	perTok, err := latency.KVBytesPerToken(*mc, 1)
	if err != nil {
		t.Fatalf("KVBytesPerToken: %v", err)
	}

	offload := sim.KVOffloadConfig{
		Enabled: true, CPUBytesToUse: 1 << 30, PerBlockBytes: int64(perTok * 16),
		BlockSize: 16, BlocksPerChunk: 1, TokensPerHash: 16,
		EvictionPolicy: "lru", OffloadPromptOnly: true,
		Tiers: []sim.KVOffloadTier{{
			Type: "fs", RootDir: "/mnt", NReadThreads: 16, NWriteThreads: 16,
			DirectIO: true, ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80,
			SaturationQueueDepth: 8, SingleTransferFraction: 0.4, // active ramp (BC-D1)
			LatencyJitterStddev: 0.2, // active jitter (BC-D5)
		}},
	}
	if offload.PerBlockBytes <= 0 {
		t.Fatalf("derived PerBlockBytes must be > 0, got %d", offload.PerBlockBytes)
	}

	betaCfg := []float64{0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0}
	alphaCfg := []float64{100.0, 1.0, 100.0}
	return cluster.DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             10_000_000,
			Seed:                seed,
			KVCacheConfig:       sim.NewKVCacheConfig(1000, 16, 0, 0.9, 100.0, 0, sim.WithKVOffload(offload)),
			BatchConfig:         sim.NewBatchConfig(64, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs(betaCfg, alphaCfg),
			ModelHardwareConfig: sim.NewModelHardwareConfig(*mc, hwCfg, "test-model", "H100", 1, 1, false, "", "trained-physics", 4096),
			PolicyConfig:        sim.NewPolicyConfig("fcfs", ""),
		},
		NumInstances:    1,
		AdmissionPolicy: "always-admit",
		RoutingPolicy:   "round-robin",
	}
}

// BC-D6/INV-13: with the device model's ramp AND jitter ACTIVE, a run and a replay
// of the same requests under the same seed produce identical per-request TTFT/E2E.
// The jitter draws come from the seed-derived kv-offload RNG partition, which both
// run and replay build from cfg.Seed; the Submit order is the same deterministic
// request-processing sequence — so the draws match exactly.
func TestINV13_RunReplayParity_Offload_Jitter(t *testing.T) {
	const fixedSeed int64 = 99
	requests := makeSharedPrefixRequests()
	cfg := jitterRampOffloadCfg(t, fixedSeed)
	dir := t.TempDir()

	cs1 := cluster.NewClusterSimulator(cfg, cluster.NewSliceRequestSource(requests), nil)
	if err := cs1.Run(); err != nil {
		t.Fatalf("direct run failed: %v", err)
	}
	runTTFTs := cs1.AggregatedMetrics().RequestTTFTs
	runE2Es := cs1.AggregatedMetrics().RequestE2Es
	if len(runTTFTs) == 0 {
		t.Fatal("INV-13: direct offload+jitter run produced no completed requests")
	}

	traceRecords := workload.RequestsToTraceRecords(requests)
	traceHdr := &workload.TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "generated"}
	traceHeaderFile := filepath.Join(dir, "trace.yaml")
	traceDataFile := filepath.Join(dir, "trace.csv")
	if err := workload.ExportTraceV2(traceHdr, traceRecords, traceHeaderFile, traceDataFile); err != nil {
		t.Fatalf("ExportTraceV2: %v", err)
	}
	traceData, err := workload.LoadTraceV2(traceHeaderFile, traceDataFile)
	if err != nil {
		t.Fatalf("LoadTraceV2: %v", err)
	}
	replayReqs, err := workload.LoadTraceV2Requests(traceData, fixedSeed)
	if err != nil {
		t.Fatalf("LoadTraceV2Requests: %v", err)
	}

	cs2 := cluster.NewClusterSimulator(cfg, cluster.NewSliceRequestSource(replayReqs), nil)
	if err := cs2.Run(); err != nil {
		t.Fatalf("replay run failed: %v", err)
	}
	replayTTFTs := cs2.AggregatedMetrics().RequestTTFTs
	replayE2Es := cs2.AggregatedMetrics().RequestE2Es

	if len(runTTFTs) != len(replayTTFTs) {
		t.Fatalf("INV-13: TTFT map size mismatch: run=%d replay=%d", len(runTTFTs), len(replayTTFTs))
	}
	for id, ttft := range runTTFTs {
		if got, ok := replayTTFTs[id]; !ok || got != ttft {
			t.Errorf("INV-13: request %s TTFT mismatch: run=%f replay=%f ok=%v", id, ttft, got, ok)
		}
	}
	for id, e2e := range runE2Es {
		if got, ok := replayE2Es[id]; !ok || got != e2e {
			t.Errorf("INV-13: request %s E2E mismatch: run=%f replay=%f ok=%v", id, e2e, got, ok)
		}
	}
}

// BC-D6/INV-6: two direct runs of the identical offload+jitter config at the same
// seed produce byte-identical per-request metrics (the jitter is seeded, not
// wall-clock).
func TestINV6_OffloadJitter_RunTwiceIdentical(t *testing.T) {
	const fixedSeed int64 = 99
	cfg := jitterRampOffloadCfg(t, fixedSeed)

	// Fresh requests per run: the simulator mutates Request objects, so reusing the
	// same pointers would confound the determinism check.
	runOnce := func() (map[string]float64, map[string]float64) {
		cs := cluster.NewClusterSimulator(cfg, cluster.NewSliceRequestSource(makeSharedPrefixRequests()), nil)
		if err := cs.Run(); err != nil {
			t.Fatalf("run failed: %v", err)
		}
		return cs.AggregatedMetrics().RequestTTFTs, cs.AggregatedMetrics().RequestE2Es
	}
	ttfts1, e2es1 := runOnce()
	if len(ttfts1) == 0 {
		t.Fatal("INV-6: offload+jitter run produced no completed requests")
	}
	ttfts2, e2es2 := runOnce()

	for id, v := range ttfts1 {
		if got := ttfts2[id]; got != v {
			t.Errorf("INV-6: request %s TTFT differs across runs: %f vs %f", id, v, got)
		}
	}
	for id, v := range e2es1 {
		if got := e2es2[id]; got != v {
			t.Errorf("INV-6: request %s E2E differs across runs: %f vs %f", id, v, got)
		}
	}
}
