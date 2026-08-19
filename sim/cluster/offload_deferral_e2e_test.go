package cluster

import (
	"fmt"
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

// offloadE2ECfg builds a single-instance SimConfig with the KV-offload chain
// enabled: a small GPU + small CPU tier and one fast "fs" secondary tier. A churny
// shared-prefix workload (below) evicts re-used prefixes down to the secondary
// tier, so later requests hit it and take the H3 step-boundary deferral path.
func offloadE2ECfg(seed int64) sim.SimConfig {
	off := sim.KVOffloadConfig{
		Enabled:           true,
		CPUBytesToUse:     24 * 4096, // 24 CPU blocks
		PerBlockBytes:     4096,
		BlockSize:         16,
		BlocksPerChunk:    1,
		TokensPerHash:     16,
		EvictionPolicy:    "lru",
		OffloadPromptOnly: true,
		Tiers: []sim.KVOffloadTier{{
			Type: "fs", RootDir: "/mnt", NReadThreads: 4, NWriteThreads: 4,
			DirectIO: true, ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80,
		}},
	}
	return sim.SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                seed,
		KVCacheConfig:       sim.NewKVCacheConfig(64, 16, 0, 0, 0, 0, sim.WithKVOffload(off)), // small GPU
		BatchConfig:         sim.NewBatchConfig(8, 512, 0),
		LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(testRooflineModelConfig(), testRooflineHWCalib(), "test", "H100", 1, 1, false, "", "roofline", 0),
	}
}

// cyclingPrefixWorkload builds requests that CYCLE through nPrefix distinct
// 3-block prefixes for nCycles rounds. Each cycle touches every prefix, and one
// cycle's worth of distinct prefixes exceeds both the small GPU and CPU capacities,
// so by the time a prefix recurs in the next cycle it has been evicted from GPU AND
// CPU and survives only on the secondary tier (populated by the earlier request's
// mirror+cascade). Reusing it then takes the H3 deferral path. This deterministic
// churn reliably exercises the deferral, unlike a single hot shared prefix (which
// stays GPU-resident forever).
func cyclingPrefixWorkload(seed int64, nPrefix, nCycles int) []*sim.Request {
	rng := sim.NewPartitionedRNG(sim.NewSimulationKey(seed)).ForSubsystem(sim.SubsystemWorkload)
	prefixes := make([][]sim.TokenID, nPrefix)
	for i := range prefixes {
		prefixes[i] = sim.GenerateRandomTokenIDs(rng, 48) // 3 blocks @ 16 tokens
	}
	var reqs []*sim.Request
	t := int64(0)
	idx := 0
	for c := 0; c < nCycles; c++ {
		for p := 0; p < nPrefix; p++ {
			suffix := sim.GenerateRandomTokenIDs(rng, 32) // distinct tail per request
			input := append(append([]sim.TokenID{}, prefixes[p]...), suffix...)
			reqs = append(reqs, &sim.Request{
				ID:           fmt.Sprintf("req_%d", idx),
				ArrivalTime:  t,
				InputTokens:  input,
				OutputTokens: sim.GenerateRandomTokenIDs(rng, 6),
				State:        sim.StateQueued,
			})
			t += 500 // 500µs apart: leaves time for mirror+cascade before reuse
			idx++
		}
	}
	return reqs
}

func runOffloadE2E(seed int64) *InstanceSimulator {
	inst := NewInstanceSimulator(InstanceID("offload-e2e"), offloadE2ECfg(seed))
	for _, r := range cyclingPrefixWorkload(seed, 12, 4) { // 12 prefixes × 4 cycles = 48 requests
		inst.InjectRequest(r)
	}
	inst.Run()
	return inst
}

// End-to-end: an offload run with a churny shared-prefix workload must (a) drain
// completely — every injected request completes, so no deferred request is
// stranded (INV-8 work-conserving, BC-T5, and INV-1 conservation), and (b) be
// deterministic — two same-seed runs produce byte-identical aggregate timing, so
// the step-boundary deferral introduces no nondeterminism (INV-6).
func TestInstanceSimulator_Offload_EndToEnd_DrainsAndDeterministic(t *testing.T) {
	const injected = 48 // 12 prefixes × 4 cycles

	a := runOffloadE2E(7)
	if got := a.Metrics().CompletedRequests; got != injected {
		t.Fatalf("every request must complete (INV-8 no stranded deferral, INV-1): completed=%d want %d", got, injected)
	}

	// Non-vacuous: the workload must actually exercise the deferral path (otherwise
	// this would only test a plain offload run).
	oc, ok := a.sim.KVCache.(*kv.OffloadCache)
	if !ok {
		t.Fatalf("offload run must use the OffloadCache, got %T", a.sim.KVCache)
	}
	if oc.DeferralsStarted() == 0 {
		t.Fatalf("the shared-prefix churn workload must trigger at least one secondary-tier deferral")
	}

	// Determinism: a second identical run must match aggregate timing exactly.
	b := runOffloadE2E(7)
	if a.Metrics().SimEndedTime != b.Metrics().SimEndedTime {
		t.Fatalf("offload run must be deterministic (INV-6): SimEndedTime %d vs %d", a.Metrics().SimEndedTime, b.Metrics().SimEndedTime)
	}
	if a.Metrics().TTFTSum != b.Metrics().TTFTSum {
		t.Fatalf("offload run TTFT must be deterministic (INV-6): TTFTSum %d vs %d", a.Metrics().TTFTSum, b.Metrics().TTFTSum)
	}
	if a.Metrics().TotalOutputTokens != b.Metrics().TotalOutputTokens {
		t.Fatalf("offload run must be deterministic (INV-6): TotalOutputTokens %d vs %d", a.Metrics().TotalOutputTokens, b.Metrics().TotalOutputTokens)
	}
}
