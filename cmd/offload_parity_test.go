package cmd

import (
	"fmt"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// makeSharedPrefixRequests builds requests that share a long common prefix (as a prefix
// group, so the sharing survives the trace round trip), so the KV-offload chain actually
// exercises mirror + cascade + reload paths. outputLen sets
// the number of decode tokens per request: a value >= blockSize (16) forms full decode
// blocks, which the offload_prompt_only=false path mirrors (decode-KV offload).
func makeSharedPrefixRequests(outputLen int) []*sim.Request {
	shared := make([]sim.TokenID, 48) // 3 shared prompt blocks (blockSize 16)
	for j := range shared {
		shared[j] = sim.TokenID(1000 + j)
	}
	reqs := make([]*sim.Request, 4)
	for i := range reqs {
		in := append([]sim.TokenID{}, shared...)
		in = append(in, sim.TokenID(9000+i)) // per-request tail
		out := make([]sim.TokenID, outputLen)
		for j := range out {
			out[j] = sim.TokenID(200 + j)
		}
		reqs[i] = &sim.Request{
			ID:           fmt.Sprintf("request_%d", i),
			ArrivalTime:  int64(i) * 50_000,
			InputTokens:  in,
			OutputTokens: out,
			MaxOutputLen: 100,
			// The group makes the sharing structure round-trip through the trace: replay
			// re-synthesizes the shared prefix, so hit/miss counts -- and with them the
			// prefill work the kernel prices -- are the run's.
			PrefixGroup:  "g",
			PrefixLength: len(shared),
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
	// Default policy (prompt-only): short outputs, no full decode blocks.
	assertOffloadRunReplayParity(t, true, 4)
}

// INV-13 for the decode-KV offload path (offload_prompt_only=false, BC-6): longer outputs
// form full decode blocks that the false policy mirrors to the tiers. Run and replay are
// driven by the same in-process resolved config through the sim/kv kernel, so the
// decode-offload path is deterministic across run/replay exactly as prompt-only is. (The
// kv_offload trace-header round-trip itself — including OffloadPromptOnly — is covered
// separately by TestKVOffload_EndToEnd_RunReplayRoundTrip.)
func TestINV13_RunReplayParity_Offload_DecodeOffload(t *testing.T) {
	assertOffloadRunReplayParity(t, false, 32) // 32 decode tokens = 2 full decode blocks
}

func assertOffloadRunReplayParity(t *testing.T, offloadPromptOnly bool, outputLen int) {
	t.Helper()
	const fixedSeed int64 = 99
	requests := makeSharedPrefixRequests(outputLen)

	dir := t.TempDir()
	d := newKernelDeployment(t, fixedSeed, nil)
	offload := sim.KVOffloadConfig{
		Enabled: true, CPUBytesToUse: 1 << 30, PerBlockBytes: d.BlockBytes,
		BlockSize: d.BlockSize, BlocksPerChunk: 1, TokensPerHash: d.BlockSize,
		EvictionPolicy: "lru", OffloadPromptOnly: offloadPromptOnly,
		Tiers: []sim.KVOffloadTier{{
			Type: "fs", RootDir: "/mnt", NReadThreads: 16, NWriteThreads: 16,
			DirectIO: true, ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80,
		}},
	}
	if offload.PerBlockBytes <= 0 {
		t.Fatalf("the kernel's per-block bytes must be > 0, got %d", offload.PerBlockBytes)
	}
	cfg := d.Config
	// A small GPU pool, so the offload chain sees eviction pressure.
	cfg.KVCacheConfig = sim.NewKVCacheConfig(4096, d.BlockSize, 0, 0.9, 0, 0, sim.WithKVOffload(offload))

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
