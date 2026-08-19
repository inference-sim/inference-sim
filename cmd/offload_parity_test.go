package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/latency"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// makeSharedPrefixRequests builds requests that share a long common prefix, so the
// KV-offload chain actually exercises mirror + cascade + reload paths.
func makeSharedPrefixRequests() []*sim.Request {
	shared := make([]sim.TokenID, 48) // 3 shared prompt blocks (blockSize 16)
	for j := range shared {
		shared[j] = sim.TokenID(1000 + j)
	}
	reqs := make([]*sim.Request, 4)
	for i := range reqs {
		in := append([]sim.TokenID{}, shared...)
		in = append(in, sim.TokenID(9000+i)) // per-request tail
		out := make([]sim.TokenID, 4)
		for j := range out {
			out[j] = sim.TokenID(200 + j)
		}
		reqs[i] = &sim.Request{
			ID:           fmt.Sprintf("request_%d", i),
			ArrivalTime:  int64(i) * 50_000,
			InputTokens:  in,
			OutputTokens: out,
			MaxOutputLen: 100,
		}
	}
	return reqs
}

// INV-13 (BC-C13): with the KV-offload chain ACTIVE, a run and a replay of the same
// requests under the same resolved config produce identical per-request TTFT/E2E.
// The offload mechanism is deterministic (station has no RNG/wall-clock, tier ops
// are slice/sorted-ordered), so parity holds; combined with the config round-trip
// test (TestKVOffload_EndToEnd_RunReplayRoundTrip) this covers INV-13 for offload.
func TestINV13_RunReplayParity_Offload(t *testing.T) {
	const fixedSeed int64 = 99
	requests := makeSharedPrefixRequests()

	mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)
	dir := t.TempDir()

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
		}},
	}
	if offload.PerBlockBytes <= 0 {
		t.Fatalf("derived PerBlockBytes must be > 0, got %d", offload.PerBlockBytes)
	}

	betaCfg := []float64{0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0}
	alphaCfg := []float64{100.0, 1.0, 100.0}
	cfg := cluster.DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             10_000_000,
			Seed:                fixedSeed,
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

	// Direct run.
	cs1 := cluster.NewClusterSimulator(cfg, cluster.NewSliceRequestSource(requests), nil)
	if err := cs1.Run(); err != nil {
		t.Fatalf("direct run failed: %v", err)
	}
	runTTFTs := cs1.AggregatedMetrics().RequestTTFTs
	runE2Es := cs1.AggregatedMetrics().RequestE2Es
	if len(runTTFTs) == 0 {
		t.Fatal("INV-13: direct offload run produced no completed requests")
	}

	// Export requests -> reload -> replay with the same config.
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
