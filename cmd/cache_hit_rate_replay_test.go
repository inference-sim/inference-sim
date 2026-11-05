package cmd

import (
	"fmt"
	"path/filepath"
	"testing"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

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

// TestResolveReplayKVOffload_UnrecordedDeploymentAllowsFlagAdd verifies that a trace
// recording no deployment (allowFlagAdd=true: a converted trace, #2067, or an agentic
// replay) with no header offload config accepts a scenario- or flag-supplied config, and
// a trace blis run wrote (allowFlagAdd=false) still rejects it (INV-13).
func TestResolveReplayKVOffload_UnrecordedDeploymentAllowsFlagAdd(t *testing.T) {
	flagCfg := kvOffloadConfig{
		Enabled: true, CPUBytesToUse: 1 << 30, BlockSize: 16, BlocksPerChunk: 1,
		TokensPerHash: 16, EvictionPolicy: "lru", OffloadPromptOnly: true,
	}
	// No recorded deployment: no header offload, flag supplied, add ALLOWED.
	got, err := resolveReplayKVOffload(nil, true, flagCfg, true)
	if err != nil {
		t.Fatalf("a trace recording no deployment must accept --kv-offload-config, got %v", err)
	}
	if !got.IsEnabled() {
		t.Fatal("returned config should be the flag-supplied offload config")
	}
	// Sim-generated trace: identical inputs, add NOT allowed → error (INV-13).
	if _, err := resolveReplayKVOffload(nil, true, flagCfg, false); err == nil {
		t.Fatal("sim-generated trace must still reject a flag-added offload config (INV-13)")
	}
}

// TestINV13_RunReplayParity_CacheHitRate verifies BC-5: with the KV-offload chain
// active, a run and a replay of the same shared-prefix requests produce identical
// aggregate cache_hit_rate, the value --metrics-path writes (INV-13).
func TestINV13_RunReplayParity_CacheHitRate(t *testing.T) {
	const fixedSeed int64 = 99
	requests := makeSharedPrefixGroupRequests()

	dir := t.TempDir()
	d := newKernelDeployment(t, fixedSeed, nil)
	offload := kvOffloadConfig{
		Enabled: true, CPUBytesToUse: 1 << 30, PerBlockBytes: d.BlockBytes,
		BlockSize: d.BlockSize, BlocksPerChunk: 1, TokensPerHash: d.BlockSize,
		EvictionPolicy: "lru", OffloadPromptOnly: true,
		Tiers: []kvOffloadTier{{
			Type: "fs", RootDir: "/mnt", NReadThreads: 16, NWriteThreads: 16,
			DirectIO: true, ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80,
		}},
	}
	cfg := d.Config
	cfg.KVCacheConfig = cengine.NewKVCacheConfig(4096, d.BlockSize, 0, 0.9, 0, 0, cengine.WithKVOffload(offload))

	cs1 := newRoutedCluster(cfg, requests)
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
	cs2 := newRoutedCluster(cfg, replayReqs)
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
