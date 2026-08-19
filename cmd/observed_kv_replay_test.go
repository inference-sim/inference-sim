package cmd

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/latency"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// TestValidateObservedKVReplayable enforces BC-10 (#1583): a tiered observed hit-rate
// requires a reproducible offload config on replay; all other cases pass.
func TestValidateObservedKVReplayable(t *testing.T) {
	tiered := &workload.TraceObservedKVMetrics{Source: workload.ObservedKVSourceTiered, HitRate: 0.5}
	gpu := &workload.TraceObservedKVMetrics{Source: workload.ObservedKVSourceGPUCache, HitRate: 0.5}

	if err := validateObservedKVReplayable(tiered, false); err == nil {
		t.Error("tiered observation with offload DISABLED must error (BC-10)")
	}
	if err := validateObservedKVReplayable(tiered, true); err != nil {
		t.Errorf("tiered observation with offload ENABLED must pass, got %v", err)
	}
	if err := validateObservedKVReplayable(gpu, false); err != nil {
		t.Errorf("gpu-fallback observation must pass regardless of offload, got %v", err)
	}
	if err := validateObservedKVReplayable(nil, false); err != nil {
		t.Errorf("nil observation must pass, got %v", err)
	}
}

// TestValidateObservedKVReplayable_ErrorMentionsConfig verifies the BC-10 error is
// actionable (names the missing config), not a bare failure.
func TestValidateObservedKVReplayable_ErrorMentionsConfig(t *testing.T) {
	tiered := &workload.TraceObservedKVMetrics{Source: workload.ObservedKVSourceTiered}
	err := validateObservedKVReplayable(tiered, false)
	if err == nil || !strings.Contains(err.Error(), "kv_offload") {
		t.Errorf("BC-10 error must mention kv_offload config, got %v", err)
	}
}

// makeSharedPrefixGroupRequests builds requests that carry PrefixGroup + PrefixLength
// so the shared prefix ROUND-TRIPS through the trace: RequestsToTraceRecords records
// the group/length, and LoadTraceV2Requests reconstructs the same group-shared
// structure on replay (token values differ but the sharing structure — and thus the
// hit/miss counts — is identical). Distinct from makeSharedPrefixRequests, whose
// explicit shared token IDs do NOT round-trip (no PrefixGroup ⇒ replay synthesizes
// independent per-request tokens).
func makeSharedPrefixGroupRequests() []*sim.Request {
	const prefixLen = 48 // 3 shared prompt blocks (blockSize 16)
	shared := make([]sim.TokenID, prefixLen)
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
			PrefixGroup:  "g",
			PrefixLength: prefixLen,
		}
	}
	return reqs
}

// TestINV13_RunReplayParity_CacheHitRate verifies BC-5: with the KV-offload chain
// active, a run and a replay of the same shared-prefix requests produce identical
// aggregate cache_hit_rate — the value calibrate reads from --metrics-path (INV-13).
func TestINV13_RunReplayParity_CacheHitRate(t *testing.T) {
	const fixedSeed int64 = 99
	requests := makeSharedPrefixGroupRequests()

	mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)
	dir := t.TempDir()

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

	cs1 := cluster.NewClusterSimulator(cfg, cluster.NewSliceRequestSource(requests), nil)
	if err := cs1.Run(); err != nil {
		t.Fatalf("direct run failed: %v", err)
	}
	runHitRate := cs1.AggregatedMetrics().CacheHitRate

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
	replayHitRate := cs2.AggregatedMetrics().CacheHitRate

	if runHitRate != replayHitRate {
		t.Errorf("INV-13: cache_hit_rate mismatch run=%v replay=%v", runHitRate, replayHitRate)
	}
	if runHitRate <= 0 {
		t.Errorf("shared-prefix offload run should register a positive hit rate, got %v (test would be vacuous)", runHitRate)
	}
}
