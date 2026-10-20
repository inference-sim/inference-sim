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
func offloadE2ECfg(seed int64) sim.SimConfig { return offloadE2ECfgReadBW(seed, 7000) }

// offloadE2ECfgReadBW is offloadE2ECfg with the secondary tier's read bandwidth set
// (bytes/tick), so a test can make promotions far slower than a step.
func offloadE2ECfgReadBW(seed int64, readBW float64) sim.SimConfig {
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
			DirectIO: true, ReadBandwidth: readBW, WriteBandwidth: 5000, BaseLatency: 80,
		}},
	}
	return sim.SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                seed,
		KVCacheConfig:       sim.NewKVCacheConfig(64, 16, 0, 0, 0, 0, sim.WithKVOffload(off)), // small GPU
		BatchConfig:         sim.NewBatchConfig(8, 512, 0),
		LatencyModel:        testFakeLatency(),
		ModelHardwareConfig: sim.NewModelHardwareConfig(testModelConfig(), "test", "H100", 1, 1, false, 0),
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
	return cyclingPrefixWorkloadSpaced(seed, nPrefix, nCycles, 500)
}

// cyclingPrefixWorkloadSpaced is cyclingPrefixWorkload with the arrival gap (µs) set.
// The default 500 µs packs every arrival into the first few tens of ms, which is well
// inside one slow-tier promotion — so the queue is complete long before any idle wait
// begins and no arrival ever supersedes an idle wake. A gap wider than a promotion
// interleaves the two, which is what the cluster clock-restore path needs.
func cyclingPrefixWorkloadSpaced(seed int64, nPrefix, nCycles int, gapUs int64) []*sim.Request {
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
			t += gapUs // default 500µs: leaves time for mirror+cascade before reuse
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

// A secondary tier far slower than a step (a 3-block promotion takes ~123 ms at 0.1
// bytes/tick) leaves the instance idle with every waiting request deferred. The run
// must still drain (INV-1, INV-8) and stay deterministic (INV-6), and superseded idle
// wakes must not stretch the run past its last completion (SimEndedTime).
func TestInstanceSimulator_Offload_SlowTierIdleWaitDrains(t *testing.T) {
	const injected = 48
	run := func() *InstanceSimulator {
		inst := NewInstanceSimulator(InstanceID("offload-slow"), offloadE2ECfgReadBW(7, 0.1))
		for _, r := range cyclingPrefixWorkload(7, 12, 4) {
			inst.InjectRequest(r)
		}
		inst.Run()
		return inst
	}

	a := run()
	m := a.Metrics()
	if m.CompletedRequests != injected {
		t.Fatalf("every request must complete (INV-1, INV-8): completed=%d want %d", m.CompletedRequests, injected)
	}
	if oc := a.sim.KVCache.(*kv.OffloadCache); oc.DeferralsStarted() == 0 {
		t.Fatal("the slow tier must actually defer requests, otherwise this tests nothing")
	}
	lastDone := 0.0
	for _, done := range m.RequestCompletionTimes {
		lastDone = math.Max(lastDone, done)
	}
	if float64(m.SimEndedTime) > lastDone {
		t.Fatalf("SimEndedTime %d is past the last completion %.0f: a superseded wake advanced the clock", m.SimEndedTime, lastDone)
	}

	b := run()
	if m.SimEndedTime != b.Metrics().SimEndedTime || m.TTFTSum != b.Metrics().TTFTSum ||
		m.CompletedRequests != b.Metrics().CompletedRequests {
		t.Fatalf("slow-tier run must be deterministic (INV-6): SimEndedTime %d/%d TTFTSum %d/%d completed %d/%d",
			m.SimEndedTime, b.Metrics().SimEndedTime, m.TTFTSum, b.Metrics().TTFTSum,
			m.CompletedRequests, b.Metrics().CompletedRequests)
	}
}

// clockSampler records the cluster clock as reported by each progress snapshot, which is
// the only outside view of it: ClusterSimulator.Clock() can be read after Run, but the
// snapshot is delivered at the end of every loop iteration, so it also samples the
// iterations that pop a no-op event.
type clockSampler struct{ clocks []int64 }

func (s *clockSampler) OnProgress(snap sim.ProgressSnapshot) {
	s.clocks = append(s.clocks, snap.Clock)
}

// firstDecrease returns the first sample that is lower than its predecessor, that
// predecessor, and its index — or index -1 when the sequence never decreases.
func (s *clockSampler) firstDecrease() (int64, int64, int) {
	for i := 1; i < len(s.clocks); i++ {
		if s.clocks[i] < s.clocks[i-1] {
			return s.clocks[i], s.clocks[i-1], i
		}
	}
	return 0, 0, -1
}

var _ sim.ProgressHook = (*clockSampler)(nil)

// The cluster loop advances c.clock to an instance's next event time before knowing the
// event is real, and restores it when ProcessNextEvent reports a superseded idle wake
// (cluster.go, the INV-3 carve-out). This test drives that branch through
// ClusterSimulator.Run; the single-instance tests above cannot, because they call
// InstanceSimulator.Run directly and never enter the cluster loop.
//
// Two things to know before changing it.
//
// It needs arrivals spaced WIDER than a promotion. With the default 500 µs gap every
// request is queued within the first few tens of ms, the idle waits all happen later with
// the queue complete, and no arrival is left to supersede a wake — measured zero
// supersessions, so the branch never ran. Hence cyclingPrefixWorkloadSpaced and the
// IdleWakesSuperseded gate below: if the workload drifts back to that shape the test fails
// loudly instead of quietly covering nothing.
//
// It cannot assert the restored VALUE. The restore is invisible to every observable:
// aggregated SimEndedTime is a max over per-instance metrics (cluster.go), each taken from
// that instance's OWN clock (sim/simulator.go), and c.clock is overwritten at the top of
// every loop iteration, so by end of run the cluster clock, max instance clock and
// SimEndedTime are identical with the restore and without it (verified by deleting it).
// Its only reader is a progress snapshot delivered in the same iteration, which is why the
// clock is sampled through one here: that gives an INV-3 monotonicity check over a sequence
// that includes every cancelled-wake iteration, and it is the value the restore protects.
// So the test pins that the branch executes on concurrent instances and that the clock laws
// hold across it — not the restored number itself.
func TestClusterSimulator_SupersededIdleWake_DoesNotAdvanceClusterClock(t *testing.T) {
	// 24 prefixes × 4 cycles: twice the single-instance workload, so each of the two
	// instances still sees more distinct prefixes per cycle than its GPU and CPU tiers
	// hold and a recurring prefix survives only on the secondary tier.
	const injected = 96

	run := func() (*ClusterSimulator, *clockSampler) {
		cfg := DeploymentConfig{
			SimConfig:        offloadE2ECfgReadBW(7, 0.1), // ~123 ms per 3-block promotion
			NumInstances:     2,
			CacheSignalDelay: DefaultCacheSignalDelay,
		}
		cs := NewClusterSimulator(cfg, NewSliceRequestSource(cyclingPrefixWorkloadSpaced(7, 24, 4, 40_000)), nil)
		// Interval 1 so a snapshot lands in essentially every loop iteration, including
		// the ones that pop a cancelled wake — that iteration is the only place the
		// cluster clock is readable from outside (cluster.go delivers at iteration end).
		sampler := &clockSampler{}
		cs.SetProgressHook(sampler, 1)
		mustRun(t, cs)
		return cs, sampler
	}

	cs, sampler := run()
	m := cs.AggregatedMetrics()
	if m.CompletedRequests != injected {
		t.Fatalf("every request must complete (INV-1, INV-8): completed=%d want %d", m.CompletedRequests, injected)
	}

	// INV-3 at cluster altitude: the clock the cluster reports never goes backwards, over
	// a sample that includes every cancelled-wake iteration.
	if n, prev, at := sampler.firstDecrease(); at >= 0 {
		t.Errorf("cluster clock went backwards at snapshot %d: %d after %d (INV-3)", at, n, prev)
	}
	if len(sampler.clocks) < 100 {
		t.Fatalf("only %d snapshots sampled: too few to cover the cancelled-wake iterations", len(sampler.clocks))
	}

	deferrals, superseded, instancesWithSupersession := int64(0), int64(0), 0
	for _, inst := range cs.Instances() {
		if oc, ok := inst.sim.KVCache.(*kv.OffloadCache); ok {
			deferrals += oc.DeferralsStarted()
		}
		if n := inst.sim.IdleWakesSuperseded(); n > 0 {
			instancesWithSupersession++
			superseded += n
		}
	}
	if deferrals == 0 {
		t.Fatal("the slow tier must actually defer requests, otherwise this tests nothing")
	}
	if superseded == 0 {
		t.Fatal("no idle wake was superseded, so the cluster clock-restore branch never ran: " +
			"space arrivals wider than a promotion")
	}
	// Concurrent instances, not just one: the branch must fire while more than one
	// instance is driving the same cluster clock.
	if instancesWithSupersession < 2 {
		t.Errorf("only %d of %d instances superseded a wake; the multi-instance case is what this test is for",
			instancesWithSupersession, len(cs.Instances()))
	}

	lastDone := 0.0
	for _, done := range m.RequestCompletionTimes {
		lastDone = math.Max(lastDone, done)
	}
	if float64(cs.Clock()) > lastDone {
		t.Errorf("cluster clock %d is past the last completion %.0f: a superseded idle wake advanced it",
			cs.Clock(), lastDone)
	}

	b, _ := run()
	if m.SimEndedTime != b.AggregatedMetrics().SimEndedTime || m.TTFTSum != b.AggregatedMetrics().TTFTSum {
		t.Errorf("cluster slow-tier run must be deterministic (INV-6): SimEndedTime %d/%d TTFTSum %d/%d",
			m.SimEndedTime, b.AggregatedMetrics().SimEndedTime, m.TTFTSum, b.AggregatedMetrics().TTFTSum)
	}
}

// A run cut off by --horizon while a deferral is still pending must end AT the horizon,
// the way the pre-#1886 one-tick crawl did. The cluster loop tests the horizon before
// executing an instance event, so an unclamped wake past the horizon is never processed:
// the instance clock stops at the last real event and SimEndedTime — the throughput
// denominator in stdout — comes out short. Single instance is enough; this is about the
// horizon, not about instance interleaving.
//
// The state has to be built deliberately: the churn workload drains, then ONE request
// arrives two seconds later reusing an early prefix that by then survives only on the
// secondary tier. It defers, nothing else is queued or running, and the horizon falls
// 30 ms into its ~123 ms promotion — so the pending wake is the only remaining instance
// event and it sits well past the horizon. (Picking a horizon inside the churn instead
// does not test this: those runs are cut off mid-batch by an ordinary step landing past
// the horizon, which ends below it with or without the clamp.)
func TestClusterSimulator_IdleWakeClampedToHorizon(t *testing.T) {
	reqs := cyclingPrefixWorkload(7, 12, 4)
	lateArrival := reqs[len(reqs)-1].ArrivalTime + 2_000_000
	reqs = append(reqs, &sim.Request{
		ID:           "late-reuse",
		ArrivalTime:  lateArrival,
		InputTokens:  reqs[0].InputTokens, // same prefix, long since evicted to the secondary tier
		OutputTokens: sim.GenerateRandomTokenIDs(sim.NewPartitionedRNG(sim.NewSimulationKey(7)).ForSubsystem(sim.SubsystemWorkload), 6),
		State:        sim.StateQueued,
	})
	horizon := lateArrival + 30_000 // inside the promotion, so the wake is beyond it

	cfg := offloadE2ECfgReadBW(7, 0.1)
	cfg.Horizon = horizon
	cs := NewClusterSimulator(
		DeploymentConfig{SimConfig: cfg, NumInstances: 1, CacheSignalDelay: DefaultCacheSignalDelay},
		NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	if m.CompletedRequests != len(reqs)-1 || m.StillQueued != 1 {
		t.Fatalf("the horizon must fall with the queue holding only the deferred late request, "+
			"got completed=%d stillQueued=%d (want %d and 1)", m.CompletedRequests, m.StillQueued, len(reqs)-1)
	}
	if m.SimEndedTime != horizon {
		t.Errorf("SimEndedTime = %d, want the horizon %d: an idle wake past the horizon was left unclamped, "+
			"so the run reports a shorter elapsed time than the pre-#1886 one-tick crawl did",
			m.SimEndedTime, horizon)
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
		LatencyModel:        testFakeLatency(),
		ModelHardwareConfig: sim.NewModelHardwareConfig(testModelConfig(), "test", "H100", 1, 1, false, 0),
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

// Note: the secondary-tier prefill-shrink is guarded discriminatingly at the sim/kv
// level by TestDeferral_ResolvedSecondaryBecomesReloadable (offload_deferral_test.go),
// which drives the deferral to resolution and asserts the now-CPU-resident prefix is
// reported by the pure ReloadablePrefixEnd query — the exact signal this fix adds, which
// fails on base. An aggregate e2e TTFT comparison against a GPU-only baseline is NOT a
// valid guard here: it conflates the H3 deferral penalty (which raises secondary-path
// TTFT) with the prefill shrink, so it holds on base too.

// loadedPrefixWorkload is a 40-block prompt, three distinct 40-block fillers that evict it
// from a 64-block GPU, and then the prompt again with one more block: its prefix is on the
// CPU tier only. Requests arrive a second apart, so each finds the instance idle.
func loadedPrefixWorkload(seed int64) []*sim.Request {
	rng := sim.NewPartitionedRNG(sim.NewSimulationKey(seed)).ForSubsystem(sim.SubsystemWorkload)
	prompt := sim.GenerateRandomTokenIDs(rng, 640)
	inputs := [][]sim.TokenID{prompt}
	for i := 0; i < 3; i++ {
		inputs = append(inputs, sim.GenerateRandomTokenIDs(rng, 640))
	}
	inputs = append(inputs, append(append([]sim.TokenID(nil), prompt...), sim.GenerateRandomTokenIDs(rng, 16)...))
	reqs := make([]*sim.Request, len(inputs))
	for i, in := range inputs {
		reqs[i] = &sim.Request{ID: fmt.Sprintf("req_%d", i), ArrivalTime: int64(i) * 1_000_000,
			InputTokens: in, OutputTokens: sim.GenerateRandomTokenIDs(rng, 4), State: sim.StateQueued}
	}
	return reqs
}

// runLoadedPrefix runs loadedPrefixWorkload on a 64-block GPU whose sequences hold a
// sliding-window-like state (a block per block in flight, plus one), with or without a CPU
// tier, to a 100 s horizon.
func runLoadedPrefix(t *testing.T, tier bool) *InstanceSimulator {
	t.Helper()
	opts := []sim.KVCacheOption{sim.WithSequenceState(sim.KVSequenceState{
		Hold: func(_, inFlight int64) int64 { return (inFlight+15)/16 + 1 },
		Gate: func(int64) int64 { return 1 },
	})}
	if tier {
		off := cpuOnlyOffloadCfg(7, 512).Offload
		off.CPULoadTime = func(bytes int64) int64 { return bytes / 4096 }
		opts = append(opts, sim.WithKVOffload(off))
	}
	cfg := cpuOnlyOffloadCfg(7, 512)
	cfg.Horizon = 100_000_000
	cfg.KVCacheConfig = sim.NewKVCacheConfig(64, 16, 0, 0, 0, 0, opts...)
	cfg.BatchConfig = sim.NewBatchConfig(8, 128, 0)
	inst := NewInstanceSimulator(InstanceID("loaded-prefix"), cfg)
	for _, r := range loadedPrefixWorkload(7) {
		inst.InjectRequest(r)
	}
	inst.Run()
	return inst
}

// #1955 invariant: no request waits forever. A request whose prefix the CPU tier loads, and
// whose sequence fits the instance's pool, is scheduled within its load and a step of the
// pool freeing -- here it arrives at an idle instance -- and the run drains. Its loaded
// prefix used to be charged as in flight, sizing its state at the whole prompt, which the
// pool cannot hold: it was refused at every step of the run.
func TestInstanceSimulator_Offload_ALoadedRequestIsScheduledOnceThePoolFrees(t *testing.T) {
	inst := runLoadedPrefix(t, true)
	m := inst.Metrics()
	if m.StillQueued != 0 || m.CompletedRequests != 5 {
		t.Fatalf("completed %d of 5, %d still queued at the horizon", m.CompletedRequests, m.StillQueued)
	}
	oc := inst.sim.KVCache.(*kv.OffloadCache)
	if oc.CPULoadsStarted() == 0 {
		t.Fatal("the reused prompt was not loaded from the CPU tier")
	}
	if d := m.RequestSchedulingDelays["req_4"]; d > 50_000 {
		t.Errorf("the loaded request waited %d ticks at an idle instance; its load takes %d", d, 40*4096/4096)
	}
}

// #1955 metamorphic: adding a CPU tier never leaves unstarted a request that starts without
// it. A tier that only shortens prefills must not lift interactivity by holding requests
// that never get a first token.
func TestInstanceSimulator_Offload_TheTierStartsEveryRequestTheGPUAloneStarts(t *testing.T) {
	without, with := runLoadedPrefix(t, false).Metrics(), runLoadedPrefix(t, true).Metrics()
	if len(without.RequestTTFTs) != 5 {
		t.Fatalf("without the tier %d of 5 requests started", len(without.RequestTTFTs))
	}
	for id := range without.RequestTTFTs {
		if _, ok := with.RequestTTFTs[id]; !ok {
			t.Errorf("%s starts without the CPU tier and never starts with it", id)
		}
	}
	if with.StillQueued > without.StillQueued {
		t.Errorf("the tier leaves %d requests queued at the horizon, against %d without it", with.StillQueued, without.StillQueued)
	}
}
