package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/kvkey"
	"github.com/inference-sim/inference-sim/sim/kvtransfer"
)

// OffloadCache must satisfy both the base KVStore seam and the optional H3
// deferral capability the scheduler type-asserts.
var (
	_ sim.KVStore           = (*OffloadCache)(nil)
	_ sim.DeferrableKVStore = (*OffloadCache)(nil)
)

// deferOC builds an offload chain with one secondary tier and a fast disk
// (service ≪ step), the regime in which step-quantization dominates. base/bw let a
// test make the service time bandwidth-dominated for the BC-T2 discrimination.
func deferOC(base, readBW float64) *OffloadCache {
	gpu := NewKVCacheState(256, 2)
	cfg := enabledOffloadCfg(1<<20, 4096, 1)
	cfg.Tiers[0].BaseLatency = base
	cfg.Tiers[0].ReadBandwidth = readBW
	return NewOffloadCache(gpu, cfg)
}

// roundsToAdmit drives the request through the step loop exactly as batch
// formation does — SetClock (applies completions) then PollDeferred (advances the
// state machine; a still-deferred request is skipped) then, if not skipped, an
// admission attempt — and returns the round on which it is admitted (or -1). step
// is the per-round clock advance; the request needs endBlocks blocks.
func roundsToAdmit(oc *OffloadCache, req *sim.Request, endIndex, step int64) int {
	for r := 1; r <= 64; r++ {
		clock := int64(r) * step
		oc.SetClock(clock)
		still := oc.PollDeferred(clock)
		skipped := false
		for _, id := range still {
			if id == req.ID {
				skipped = true
				break
			}
		}
		if skipped {
			continue
		}
		if oc.AllocateKVBlocks(req, 0, endIndex, nil) {
			return r
		}
	}
	return -1
}

// seedSecondary stores a 2-block prefix in secondary tier 0 and returns its keys.
func seedSecondary(oc *OffloadCache, tokens []sim.TokenID) []kvkey.BlockKey {
	keys := blockKeysFor(tokens, 2)
	oc.secondary[0].store(keys[0])
	oc.secondary[0].store(keys[1])
	return keys
}

// BC-T4: a COLD secondary hit costs k>=3 rounds (RETRY, promote, resolve) and a
// WARM hit k>=2 (existence cached, skip RETRY); cold - warm == 1. This distinguishes
// the two paths — a warm-only benchmark would make a wrong (k=1/k=2) model look right.
func TestDeferral_ColdVersusWarmRoundCount(t *testing.T) {
	const step = int64(1000) // ≫ service (~81 ticks) so the disk fits in one step
	tokens := []sim.TokenID{1, 2, 3, 4}

	// COLD: fresh keys, existence not yet known.
	ocCold := deferOC(80, 7000)
	seedSecondary(ocCold, tokens)
	cold := roundsToAdmit(ocCold, &sim.Request{ID: "cold", InputTokens: tokens}, 4, step)

	// WARM: same disk, but the keys' secondary existence is already resolved.
	ocWarm := deferOC(80, 7000)
	keys := seedSecondary(ocWarm, tokens)
	ocWarm.markKnown(keys) // existence cache warmed by a prior (elided) fetch
	warm := roundsToAdmit(ocWarm, &sim.Request{ID: "warm", InputTokens: tokens}, 4, step)

	if cold < 3 {
		t.Fatalf("COLD secondary hit must cost >=3 rounds (RETRY, promote, resolve), got %d", cold)
	}
	if warm < 2 {
		t.Fatalf("WARM secondary hit must cost >=2 rounds (promote, resolve), got %d", warm)
	}
	if cold-warm != 1 {
		t.Fatalf("cold - warm must be exactly one RETRY round: cold=%d warm=%d", cold, warm)
	}
}

// BC-T4 detail: the cold sequence hits the exact phases in order — RETRY (no
// promotion) → promote (HIT_PENDING, one Read job) → resolve (HIT) → admit — and
// the promotion is submitted only on round 2 (not round 1).
func TestDeferral_ColdPhaseSequence(t *testing.T) {
	const step = int64(1000)
	tokens := []sim.TokenID{1, 2, 3, 4}
	oc := deferOC(80, 7000)
	keys := seedSecondary(oc, tokens)
	req := &sim.Request{ID: "r", InputTokens: tokens}

	// Round 1: RETRY. Deferred, no promotion yet.
	oc.SetClock(step)
	oc.PollDeferred(step)
	if oc.AllocateKVBlocks(req, 0, 4, nil) {
		t.Fatalf("round 1: a cold secondary hit must defer, not admit")
	}
	if oc.promotionsFired != 0 {
		t.Fatalf("round 1 (RETRY): no promotion must fire yet, got %d", oc.promotionsFired)
	}
	if st := oc.deferred[req.ID]; st == nil || st.phase != deferRetry {
		t.Fatalf("round 1: request must be tracked in deferRetry")
	}

	// Round 2: promote. Existence resolved → one Read job, block HIT_PENDING.
	oc.SetClock(2 * step)
	still := oc.PollDeferred(2 * step)
	if len(still) != 1 || still[0] != req.ID {
		t.Fatalf("round 2: request must still be deferred (promoting), got %v", still)
	}
	if oc.promotionsFired != 1 {
		t.Fatalf("round 2 (promote): exactly one promotion must fire, got %d", oc.promotionsFired)
	}
	if oc.cpu.lookup(keys[0]) != cpuHitPending {
		t.Fatalf("round 2: the promoted block must be HIT_PENDING")
	}
	if n := oc.station.ActiveJobs(0, kvtransfer.Read); n != 1 {
		t.Fatalf("round 2: one Read job must be in service, got %d", n)
	}

	// Round 3: resolve + admit. The Read has landed (service ≪ step).
	oc.SetClock(3 * step)
	still = oc.PollDeferred(3 * step)
	if len(still) != 0 {
		t.Fatalf("round 3: promotion completed, request must no longer be deferred, got %v", still)
	}
	if !oc.AllocateKVBlocks(req, 0, 4, nil) {
		t.Fatalf("round 3: the request must be admitted after its promotion lands")
	}
	if _, tracked := oc.deferred[req.ID]; tracked {
		t.Fatalf("round 3: the deferral episode must be cleared on admission")
	}
	if len(gpu2GetCached(oc, tokens)) != 2 {
		t.Fatalf("round 3: the promoted prefix must be reloaded to GPU on admission")
	}
}

func gpu2GetCached(oc *OffloadCache, tokens []sim.TokenID) []int64 {
	return oc.gpu.GetCachedBlocks(tokens)
}

// BC-T2 (metamorphic + discrimination): the offload-attributable delay = (k-1)*step.
// (a) 3x longer steps ⇒ ~3x delay (slope≈1); (b) halving disk bandwidth ⇒ delay
// unchanged (≪2x) as long as service ≪ step. A bandwidth-only surrogate (delay =
// service) fails BOTH: it is flat under step-scaling and ~2x under bandwidth-halving.
func TestDeferral_Metamorphic_StepVsBandwidth(t *testing.T) {
	tokens := []sim.TokenID{1, 2, 3, 4}
	// Bandwidth-dominated service (tiny base) so the bandwidth surrogate has bite,
	// but still ≪ the step sizes below.
	const base, bw = 1.0, 100.0

	delay := func(step, readBW float64) int64 {
		oc := deferOC(base, readBW)
		seedSecondary(oc, tokens)
		r := roundsToAdmit(oc, &sim.Request{ID: "r", InputTokens: tokens}, 4, int64(step))
		if r < 0 {
			t.Fatalf("request never admitted (step=%v bw=%v)", step, readBW)
		}
		return int64(r-1) * int64(step) // offload-attributable delay: admitted on round r
	}
	// Bandwidth surrogate: "delay is set by the disk" — service time of the run.
	surrogate := func(readBW float64) int64 {
		oc := deferOC(base, readBW)
		return oc.station.ServiceTicks(0, kvtransfer.Read, 2*oc.perBlockBytes)
	}

	dStep1 := delay(1000, 100)
	dStep3 := delay(3000, 100) // 3x longer steps, same disk
	dBwHalf := delay(1000, 50) // same steps, half the disk bandwidth

	// (a) Our model scales ~linearly with step (slope ≈ 1): 3x step ⇒ ~3x delay.
	if dStep3 != 3*dStep1 {
		t.Fatalf("step-scaling: 3x step must give 3x delay, got %d vs %d", dStep3, dStep1)
	}
	// (b) Our model is insensitive to disk bandwidth (delay unchanged).
	if dBwHalf != dStep1 {
		t.Fatalf("bandwidth-insensitivity: halving disk bandwidth must not change delay, got %d vs %d", dBwHalf, dStep1)
	}

	// Discrimination: the bandwidth surrogate fails BOTH relations.
	sFull, sHalf := surrogate(100), surrogate(50)
	if sFull == 0 {
		t.Fatalf("surrogate service must be non-zero to discriminate")
	}
	// (a') Surrogate is flat under step-scaling (it ignores step) — unlike ours (3x).
	if dStep3 <= dStep1 {
		t.Fatalf("our model must grow with step while the surrogate would not")
	}
	// (b') Surrogate ~doubles under bandwidth-halving — unlike ours (unchanged).
	if sHalf < 2*sFull-2 || dBwHalf != dStep1 {
		t.Fatalf("surrogate must ~double under bw-halving (%d->%d) while our delay stays flat (%d)", sFull, sHalf, dBwHalf)
	}
}

// BC-T3 (property / no livelock): when the CPU tier cannot make room for the fetch
// (all blocks pinned by in-flight writes), the promotion is refused and the request
// is admitted by recompute in a bounded number of rounds — never deferred forever.
func TestDeferral_BoundedThenRecompute(t *testing.T) {
	const step = int64(1000)
	tokens := []sim.TokenID{1, 2, 3, 4}
	gpu := NewKVCacheState(256, 2)
	oc := NewOffloadCache(gpu, enabledOffloadCfg(8192, 4096, 1)) // CPU capacity 2
	// Fill + pin both CPU slots so the evictable pool is empty (locked tier).
	filler := blockKeysFor([]sim.TokenID{90, 91, 92, 93}, 2)
	oc.cpu.store(filler[0])
	oc.cpu.store(filler[1])
	oc.cpu.pin(filler[0])
	oc.cpu.pin(filler[1])
	keys := seedSecondary(oc, tokens)
	oc.markKnown(keys) // warm so the promotion is attempted on round 1 (gate then refuses)

	r := roundsToAdmit(oc, &sim.Request{ID: "r", InputTokens: tokens}, 4, step)
	if r < 0 || r > 3 {
		t.Fatalf("a locked-tier fetch must be recomputed in a few bounded rounds, got %d", r)
	}
	if oc.promotionsFailed < 1 {
		t.Fatalf("the evictable gate must refuse the promotion (BC-C5), promotionsFailed=%d", oc.promotionsFailed)
	}
}

// BC-T1 & BC-T6 (differential): re-poll happens at the step boundary, not at
// transfer completion — a promotion submitted on a step is NOT visible within that
// same step, only on the next SetClock (completions polled before lookups).
func TestDeferral_RePollAtStepBoundaryNotCompletion(t *testing.T) {
	const step = int64(1000)
	tokens := []sim.TokenID{1, 2, 3, 4}
	oc := deferOC(80, 7000)
	keys := seedSecondary(oc, tokens)
	oc.markKnown(keys) // warm: promotion submitted on round 1
	req := &sim.Request{ID: "r", InputTokens: tokens}

	// Round 1: submit the promotion (SubmitTick = round-1 clock).
	oc.SetClock(step)
	oc.PollDeferred(step)
	if oc.AllocateKVBlocks(req, 0, 4, nil) {
		t.Fatalf("round 1: warm hit must defer while its promotion is in flight")
	}
	if oc.promotionsFired != 1 {
		t.Fatalf("round 1: promotion must be submitted, got %d", oc.promotionsFired)
	}
	// Same step: the block is still HIT_PENDING — the just-submitted promotion is NOT
	// visible within the step it was submitted (visible only next SetClock).
	if oc.cpu.lookup(keys[0]) != cpuHitPending {
		t.Fatalf("a promotion submitted this step must not complete within the same step")
	}

	// Round 2: SetClock (completions polled) THEN PollDeferred (lookups) — the
	// promotion that landed since last step is a HIT here, so the request resolves.
	oc.SetClock(2 * step)
	still := oc.PollDeferred(2 * step)
	if len(still) != 0 {
		t.Fatalf("round 2: the completed promotion must resolve the deferral (re-poll at step boundary)")
	}
	if oc.cpu.lookup(keys[0]) != cpuHit {
		t.Fatalf("round 2: SetClock must have applied the completion before the lookup (BC-T6)")
	}
}

// INV-6 determinism: with several requests deferred concurrently against one disk,
// PollDeferred's sorted-ID side-effect order makes station JobID assignment (and
// therefore completion order and admission rounds) identical across runs.
func TestDeferral_DeterministicUnderConcurrentDeferrals(t *testing.T) {
	const step = int64(1000)
	run := func() []int {
		oc := deferOC(80, 7000)
		// Five distinct 2-block prefixes, all secondary-resident, all cold.
		reqs := make([]*sim.Request, 5)
		for i := range reqs {
			base := sim.TokenID(i*10 + 1)
			toks := []sim.TokenID{base, base + 1, base + 2, base + 3}
			seedSecondary(oc, toks)
			reqs[i] = &sim.Request{ID: string(rune('a' + i)), InputTokens: toks}
		}
		admit := make([]int, len(reqs))
		for i := range admit {
			admit[i] = -1
		}
		for r := 1; r <= 32; r++ {
			clock := int64(r) * step
			oc.SetClock(clock)
			still := map[string]bool{}
			for _, id := range oc.PollDeferred(clock) {
				still[id] = true
			}
			for i, req := range reqs {
				if admit[i] >= 0 || still[req.ID] {
					continue
				}
				if oc.AllocateKVBlocks(req, 0, 4, nil) {
					admit[i] = r
				}
			}
		}
		return admit
	}
	a, b := run(), run()
	for i := range a {
		if a[i] != b[i] || a[i] < 0 {
			t.Fatalf("concurrent deferrals must admit deterministically: run1=%v run2=%v", a, b)
		}
	}
}

// Two concurrent COLD requests for the SAME secondary prefix must share ONE
// promotion (convoy dedup, BC-C2): the first to reach the promote round fires it;
// the second rides the in-flight promotion instead of recomputing. Both are
// admitted, exactly one promotion fires, and nothing is force-recomputed.
func TestDeferral_ConcurrentColdSamePrefixShareOnePromotion(t *testing.T) {
	const step = int64(1000)
	tokens := []sim.TokenID{1, 2, 3, 4}
	oc := deferOC(80, 7000)
	seedSecondary(oc, tokens)
	A := &sim.Request{ID: "A", InputTokens: tokens}
	B := &sim.Request{ID: "B", InputTokens: tokens}

	aAdm, bAdm := false, false
	for r := 1; r <= 10 && (!aAdm || !bAdm); r++ {
		clock := int64(r) * step
		oc.SetClock(clock)
		still := map[string]bool{}
		for _, id := range oc.PollDeferred(clock) {
			still[id] = true
		}
		if !aAdm && !still["A"] {
			aAdm = oc.AllocateKVBlocks(A, 0, 4, nil)
		}
		if !bAdm && !still["B"] {
			bAdm = oc.AllocateKVBlocks(B, 0, 4, nil)
		}
	}
	if !aAdm || !bAdm {
		t.Fatalf("both concurrent cold requests must be admitted, A=%v B=%v", aAdm, bAdm)
	}
	if oc.promotionsFired != 1 {
		t.Fatalf("concurrent cold requests for one prefix must share ONE promotion (convoy), fired=%d", oc.promotionsFired)
	}
	if oc.promotionsFailed != 0 {
		t.Fatalf("neither request should be force-recomputed (both ride the one promotion), failed=%d", oc.promotionsFailed)
	}
}

// A PD decode sub-request (KV pre-reservation via ReserveTransferredKV, which calls
// AllocateKVBlocks(req, 0, inputLen, nil) with IsDecodeSubRequest set) must NEVER be
// deferred even when its prompt prefix is secondary-resident — deferring would make
// the reservation return false and the request be dropped. It keeps the H1 path
// (pre-#1591 behavior): the reservation succeeds and a background promotion fires.
func TestDeferral_DecodeSubRequestDoesNotDefer(t *testing.T) {
	tokens := []sim.TokenID{1, 2, 3, 4}
	oc := deferOC(80, 7000)
	seedSecondary(oc, tokens) // prompt prefix resides only on the secondary tier
	req := &sim.Request{ID: "d", InputTokens: tokens, IsDecodeSubRequest: true}

	oc.SetClock(1000)
	ok := oc.AllocateKVBlocks(req, 0, 4, nil) // the ReserveTransferredKV call shape
	if oc.IsDeferred(req.ID) {
		t.Fatalf("a PD decode sub-request must never be deferred (would be dropped)")
	}
	if !ok {
		t.Fatalf("the KV reservation must succeed via the H1 path (GPU has capacity)")
	}
	if _, tracked := oc.deferred[req.ID]; tracked {
		t.Fatalf("a decode sub-request must not register a deferral episode")
	}
}

// ClearDeferred forgets a request that leaves the WaitQ by a non-admit path, so the
// deferred map does not leak (the P benchmark stays O(live deferrals)).
func TestDeferral_ClearDeferredForgets(t *testing.T) {
	tokens := []sim.TokenID{1, 2, 3, 4}
	oc := deferOC(80, 7000)
	seedSecondary(oc, tokens)
	req := &sim.Request{ID: "r", InputTokens: tokens}

	oc.SetClock(1000)
	oc.PollDeferred(1000)
	oc.AllocateKVBlocks(req, 0, 4, nil) // defers
	if !oc.IsDeferred(req.ID) {
		t.Fatalf("request must be deferred after hitting a secondary tier")
	}
	oc.ClearDeferred(req.ID)
	if oc.IsDeferred(req.ID) {
		t.Fatalf("ClearDeferred must forget the request")
	}
	if _, tracked := oc.deferred[req.ID]; tracked {
		t.Fatalf("ClearDeferred must remove the map entry")
	}
	oc.ClearDeferred(req.ID) // idempotent, must not panic
}

// A running-request continuation (already GPU-resident) must NOT defer — the defer
// branch is gated to NEW admissions (C1). The SAME request defers when new but not
// when running, so a false return on the running path is GPU pressure, not "skip".
func TestDeferral_RunningRequestDoesNotDefer(t *testing.T) {
	tokens := []sim.TokenID{1, 2, 3, 4}
	oc := deferOC(80, 7000)
	seedSecondary(oc, tokens) // full prefix secondary-resident
	req := &sim.Request{ID: "r", InputTokens: tokens}

	// Baseline: as a NEW request, hitting the secondary prefix defers.
	oc.SetClock(1000)
	oc.PollDeferred(1000)
	oc.AllocateKVBlocks(req, 0, 4, nil)
	if !oc.IsDeferred(req.ID) {
		t.Fatalf("baseline: a new request hitting a secondary prefix must defer")
	}

	// Same request, now marked RUNNING (owns a GPU block): the defer branch is
	// skipped, so AllocateKVBlocks never registers it as deferred.
	oc.ClearDeferred(req.ID)
	blk := oc.gpu.popFreeBlock()
	oc.gpu.RequestMap[req.ID] = []int64{blk.ID}
	oc.AllocateKVBlocks(req, 0, 4, nil)
	if oc.IsDeferred(req.ID) {
		t.Fatalf("a running-request continuation must not be registered as deferred (C1 gating)")
	}
}
