package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// BC-C5 metamorphic (end-to-end): slower cascade writes keep CPU blocks pinned
// longer, starving the evictable pool, so a promotion that needs more blocks than
// are free-plus-evictable is REFUSED and the prefix is recomputed. A faster write
// bandwidth releases the pins in time and the same promotion SUCCEEDS.
//
// The scenario deliberately keeps free > 0 (one free CPU slot) while the promotion
// needs 2 blocks: the correct EVICTABLE gate refuses when evictable==0 (writes in
// flight), whereas a free-count gate would grant it (free covers the shortfall) —
// so this relation ALSO fails under the mandatory mutant, and the promotions-fired
// guard prevents a vacuous (zero-promotion) pass. The direct correct-vs-mutant
// boundary lives in TestOffloadCPUTier_PrepareStoreEvictableGate.
func TestOffload_LockSaturationRecompute(t *testing.T) {
	// CPU capacity 3 blocks (12288 / 4096). One secondary tier.
	run := func(writeBandwidth float64) (fired, failed int64) {
		gpu := NewKVCacheState(64, 2)
		cfg := enabledOffloadCfg(12288, 4096, 1)
		cfg.Tiers[0].WriteBandwidth = writeBandwidth // vary write speed only
		oc := NewOffloadCache(gpu, cfg)

		// Fill 2 of 3 CPU slots via mirror+cascade at t=10 (each block pinned by its
		// in-flight write). free = 1, evictable = 0 immediately after.
		oc.SetClock(10)
		for _, toks := range [][]sim.TokenID{{3, 4}, {5, 6}} {
			r := &sim.Request{ID: string(rune('Y')) + string(rune(toks[0])), InputTokens: toks}
			oc.AllocateKVBlocks(r, 0, 2, nil)
			oc.MirrorToCPU([]*sim.Request{r})
		}

		// A 2-block prefix X lives only in the secondary tier (warm).
		xt := []sim.TokenID{1, 2, 7, 8}
		xk := blockKeysFor(xt, 2)
		oc.secondary[0].store(xk[0])
		oc.secondary[0].store(xk[1])

		// Advance a fixed 100 ticks: a FAST write (service 80) has completed (pins
		// released, evictable 2); a SLOW write is still in flight (evictable 0).
		oc.SetClock(110)
		oc.consultAndReload(xt, 0, "") // attempt to promote the 2-block prefix X
		return oc.promotionsFired, oc.promotionsFailed
	}

	firedFast, failedFast := run(5000) // write service ~80 ticks -> done by t=110
	firedSlow, failedSlow := run(1)    // write service ~4176 ticks -> still in flight at t=110

	if firedFast < 1 {
		t.Fatalf("fast writes must release the pins so the promotion SUCCEEDS (gate exercised), fired=%d", firedFast)
	}
	if failedFast != 0 {
		t.Fatalf("fast writes must not force a recompute, failed=%d", failedFast)
	}
	if failedSlow < 1 {
		t.Fatalf("slow writes must keep blocks pinned so the promotion is REFUSED -> recompute (BC-C5), failed=%d", failedSlow)
	}
	if firedSlow != 0 {
		t.Fatalf("slow writes must not admit the promotion, fired=%d", firedSlow)
	}
}

// promotedPending returns the sorted set of keys the tier chain marked HIT_PENDING
// (i.e. promoted) for prefix xt, plus how many promotions fired — the observable
// promotion outcome, independent of per-tier attribution.
func promotedPending(oc *OffloadCache, xt []sim.TokenID) (pending int, fired int64) {
	oc.SetClock(10)
	oc.consultAndReload(xt, 0, "")
	xk := blockKeysFor(xt, 2)
	for _, k := range xk {
		if oc.cpu.lookup(k) == cpuHitPending {
			pending++
		}
	}
	return pending, oc.promotionsFired
}

// BC-C7b (order invariance): permuting secondary tiers with disjoint contents does
// not change WHICH blocks are promoted or the promotion count — only per-tier
// attribution differs (lookups are ordered but the found content is the same).
func TestOffload_TierOrderInvariant(t *testing.T) {
	xt := []sim.TokenID{1, 2, 7, 8} // 2-block prefix X
	xk := blockKeysFor(xt, 2)
	other := blockKeysFor([]sim.TokenID{40, 41}, 2)[0] // disjoint content

	// Config A: tier0 holds X (both blocks), tier1 holds the disjoint block.
	gpuA := NewKVCacheState(64, 2)
	ocA := NewOffloadCache(gpuA, enabledOffloadCfg(1<<20, 4096, 2))
	ocA.secondary[0].store(xk[0])
	ocA.secondary[0].store(xk[1])
	ocA.secondary[1].store(other)

	// Config B: the SAME contents, tiers permuted (X in tier1).
	gpuB := NewKVCacheState(64, 2)
	ocB := NewOffloadCache(gpuB, enabledOffloadCfg(1<<20, 4096, 2))
	ocB.secondary[1].store(xk[0])
	ocB.secondary[1].store(xk[1])
	ocB.secondary[0].store(other)

	pA, fA := promotedPending(ocA, xt)
	pB, fB := promotedPending(ocB, xt)
	if pA != pB || fA != fB {
		t.Fatalf("tier-order permutation must not change the promotion outcome: A(pending=%d fired=%d) B(pending=%d fired=%d)", pA, fA, pB, fB)
	}
	if pA != 2 {
		t.Fatalf("both X blocks should be promoted, got %d", pA)
	}
}

// BC-C7b (no spurious coupling): adding a secondary tier that is never hit does not
// change the promotion outcome for a prefix served by another tier.
func TestOffload_UnusedTierNoCoupling(t *testing.T) {
	xt := []sim.TokenID{1, 2, 7, 8}
	xk := blockKeysFor(xt, 2)

	// One tier holding X.
	gpu1 := NewKVCacheState(64, 2)
	oc1 := NewOffloadCache(gpu1, enabledOffloadCfg(1<<20, 4096, 1))
	oc1.secondary[0].store(xk[0])
	oc1.secondary[0].store(xk[1])

	// Same, plus a second tier that holds nothing (never hit).
	gpu2 := NewKVCacheState(64, 2)
	oc2 := NewOffloadCache(gpu2, enabledOffloadCfg(1<<20, 4096, 2))
	oc2.secondary[0].store(xk[0])
	oc2.secondary[0].store(xk[1])
	// secondary[1] left empty.

	p1, f1 := promotedPending(oc1, xt)
	p2, f2 := promotedPending(oc2, xt)
	if p1 != p2 || f1 != f2 {
		t.Fatalf("a never-hit tier must not change the promotion outcome: one-tier(pending=%d fired=%d) two-tier(pending=%d fired=%d)", p1, f1, p2, f2)
	}
}
