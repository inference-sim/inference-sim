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

// cpuOnlyOffloadCfg builds an offload config with NO secondary tiers (CPU-only), the
// #1699 repro shape. cpuBlocks sizes the CPU staging tier: a large tier retains evicted
// prefixes for reload; a tiny tier cannot, so requests recompute.
func cpuOnlyOffloadCfg(seed, cpuBlocks int64) sim.SimConfig {
	off := sim.KVOffloadConfig{
		Enabled:           true,
		CPUBytesToUse:     cpuBlocks * 4096,
		PerBlockBytes:     4096,
		BlockSize:         16,
		BlocksPerChunk:    1,
		TokensPerHash:     16,
		EvictionPolicy:    "lru",
		OffloadPromptOnly: true,
		// No Tiers: CPU-only offload (the config in issue #1699).
	}
	return sim.SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                seed,
		KVCacheConfig:       sim.NewKVCacheConfig(64, 16, 0, 0, 0, 0, sim.WithKVOffload(off)),
		BatchConfig:         sim.NewBatchConfig(8, 512, 0),
		LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(testRooflineModelConfig(), testRooflineHWCalib(), "test", "H100", 1, 1, false, "", "roofline", 0),
	}
}

func runCPUOnlyOffloadE2E(seed, cpuBlocks int64) *InstanceSimulator {
	inst := NewInstanceSimulator(InstanceID("offload-cpu-e2e"), cpuOnlyOffloadCfg(seed, cpuBlocks))
	for _, r := range cyclingPrefixWorkload(seed, 12, 4) {
		inst.InjectRequest(r)
	}
	inst.Run()
	return inst
}

// #1699 regression (T3): with CPU-only offload, a LARGER CPU tier retains more evicted
// prefixes for reload, so more prompt tokens are served from cache instead of
// recomputed — which MUST lower aggregate TTFT. Before the fix, a CPU reload moved
// cache_hit_rate but left NumNewTokens (and therefore TTFT) unchanged, so this
// comparison was byte-identical. The run must also drain fully and be deterministic.
func TestInstanceSimulator_Offload_CPUHitReducesTTFT(t *testing.T) {
	const injected = 48

	big := runCPUOnlyOffloadE2E(7, 512) // ample CPU tier: prefixes survive for reload
	small := runCPUOnlyOffloadE2E(7, 1) // 1-block CPU tier: no useful retention (≈ offload off)

	if got := big.Metrics().CompletedRequests; got != injected {
		t.Fatalf("big-CPU run must drain (INV-1/INV-8): completed=%d want %d", got, injected)
	}
	if got := small.Metrics().CompletedRequests; got != injected {
		t.Fatalf("small-CPU run must drain (INV-1/INV-8): completed=%d want %d", got, injected)
	}

	// Non-vacuity: the big-CPU run must actually reload prefixes from CPU→GPU
	// (otherwise there is no hit for the fix to bill cheaply), and the tiny-CPU run
	// must not. ReloadsPerformed is the diagnostic behind the prefill-shrink.
	bigOC, ok := big.sim.KVCache.(*kv.OffloadCache)
	if !ok {
		t.Fatalf("offload run must use OffloadCache, got %T", big.sim.KVCache)
	}
	smallOC := small.sim.KVCache.(*kv.OffloadCache)
	if bigOC.ReloadsPerformed() == 0 {
		t.Fatalf("big CPU tier must perform CPU→GPU reloads (the hits the fix bills cheaply)")
	}
	if smallOC.ReloadsPerformed() != 0 {
		t.Fatalf("a 1-block CPU tier cannot retain prefixes, so it must perform 0 reloads, got %d", smallOC.ReloadsPerformed())
	}

	// The core #1699 assertion: CPU reloads (billed as hits, not recompute) must reduce
	// aggregate TTFT. Before the fix these reloads left NumNewTokens unchanged, so
	// big.TTFTSum == small.TTFTSum exactly (the bug). Now more reloads ⇒ lower TTFT.
	if big.Metrics().TTFTSum >= small.Metrics().TTFTSum {
		t.Fatalf("CPU reloads must reduce TTFT (#1699): big TTFTSum=%d (%d reloads) small TTFTSum=%d (%d reloads)",
			big.Metrics().TTFTSum, bigOC.ReloadsPerformed(), small.Metrics().TTFTSum, smallOC.ReloadsPerformed())
	}

	// Determinism (INV-6 / INV-13): a second identical big-CPU run matches exactly.
	big2 := runCPUOnlyOffloadE2E(7, 512)
	if big.Metrics().TTFTSum != big2.Metrics().TTFTSum {
		t.Fatalf("CPU-offload TTFT must be deterministic: %d vs %d", big.Metrics().TTFTSum, big2.Metrics().TTFTSum)
	}
}

// #1699 (T4): the SAME prefill-shrink must hold on the secondary-tier path. A run whose
// churny workload lands secondary→CPU→GPU reloads must show lower aggregate TTFT than a
// GPU-only run of the identical workload (the reloaded prefixes are billed as hits, not
// recomputes). This guards the resolved-deferral branch, which funnels through the same
// allocateThroughChain reporting point.
func TestInstanceSimulator_Offload_SecondaryHitReducesTTFT(t *testing.T) {
	// GPU-only baseline: same GPU/workload, no offload tier.
	gpuOnlyCfg := func(seed int64) sim.SimConfig {
		return sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                seed,
			KVCacheConfig:       sim.NewKVCacheConfig(64, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(8, 512, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(testRooflineModelConfig(), testRooflineHWCalib(), "test", "H100", 1, 1, false, "", "roofline", 0),
		}
	}
	runGPUOnly := func(seed int64) *InstanceSimulator {
		inst := NewInstanceSimulator(InstanceID("gpu-only"), gpuOnlyCfg(seed))
		for _, r := range cyclingPrefixWorkload(seed, 12, 4) {
			inst.InjectRequest(r)
		}
		inst.Run()
		return inst
	}

	offload := runOffloadE2E(7) // has a secondary tier; workload exercises deferral
	gpuOnly := runGPUOnly(7)

	// Non-vacuity: the offload run must actually take the secondary-tier deferral path.
	oc, ok := offload.sim.KVCache.(*kv.OffloadCache)
	if !ok || oc.DeferralsStarted() == 0 {
		t.Fatalf("secondary-tier reload path must be exercised (DeferralsStarted>0)")
	}

	if offload.Metrics().TTFTSum >= gpuOnly.Metrics().TTFTSum {
		t.Fatalf("secondary-tier reloads must reduce TTFT vs GPU-only recompute (#1699): offload TTFTSum=%d gpu-only TTFTSum=%d",
			offload.Metrics().TTFTSum, gpuOnly.Metrics().TTFTSum)
	}
}
