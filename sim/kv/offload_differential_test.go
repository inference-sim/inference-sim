package kv

import (
	"math/rand"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// BC-C1 (differential vs vLLM tiering/manager.py staging): a block resident only in
// a secondary tier reaches the GPU strictly via two hops — secondary->CPU
// (prepare_read/promotion) THEN CPU->GPU (prepare_load/reload) — and is NEVER
// transferred secondary->GPU directly. This asserts the residency ordering at each
// stage, matching vLLM's read/load split.
func TestOffload_Differential_TwoHopStaging(t *testing.T) {
	gpu := NewKVCacheState(64, 2)
	oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 1))
	tokens := []sim.TokenID{1, 2, 3, 4}
	keys := blockKeysFor(tokens, 2)
	k0 := string(keys[0])

	// Stage 0: block lives ONLY in the secondary tier.
	oc.secondary[0].store(keys[0])
	oc.secondary[0].store(keys[1])
	if _, onGPU := gpu.HashToBlock[k0]; onGPU {
		t.Fatalf("stage 0: block must not start on GPU")
	}
	if oc.cpu.lookup(keys[0]) != cpuMiss {
		t.Fatalf("stage 0: block must not start in CPU")
	}

	// Stage 1: consult initiates the promotion (secondary->CPU). The block is now in
	// CPU as HIT_PENDING and STILL NOT on GPU (no secondary->GPU shortcut).
	oc.SetClock(100)
	oc.consultAndReload(tokens, 0, "")
	if oc.cpu.lookup(keys[0]) != cpuHitPending {
		t.Fatalf("stage 1: promotion must place the block in CPU as HIT_PENDING (hop 1 in flight)")
	}
	if _, onGPU := gpu.HashToBlock[k0]; onGPU {
		t.Fatalf("stage 1: block must NOT be on GPU while only the secondary->CPU hop has started (no direct secondary->GPU, BC-C1)")
	}
	if oc.reloadCount != 0 {
		t.Fatalf("stage 1: no CPU->GPU reload yet, reloadCount=%d", oc.reloadCount)
	}

	// Stage 2: the promotion completes (hop 1 done). The block is readable in CPU but
	// STILL not on GPU — CPU->GPU is a separate, later hop.
	oc.SetClock(300)
	if oc.cpu.lookup(keys[0]) != cpuHit {
		t.Fatalf("stage 2: promotion must complete to a CPU HIT (hop 1 done)")
	}
	if _, onGPU := gpu.HashToBlock[k0]; onGPU {
		t.Fatalf("stage 2: a completed promotion lands in CPU, not GPU (BC-C1)")
	}

	// Stage 3: a request reloads CPU->GPU (hop 2). Only now is the block on GPU.
	reqB := &sim.Request{ID: "b", InputTokens: tokens}
	oc.AllocateKVBlocks(reqB, 0, 4, nil)
	if _, onGPU := gpu.HashToBlock[k0]; !onGPU {
		t.Fatalf("stage 3: after CPU->GPU reload the block must be GPU-resident (hop 2 done)")
	}
	if oc.reloadCount < 1 {
		t.Fatalf("stage 3: the second hop must be a CPU->GPU reload, reloadCount=%d", oc.reloadCount)
	}
}

// INV-4 (multi-tier conservation) + C4 (delayed finalization): over a mixed
// workload the CPU tier's used+free==capacity holds at every observation point,
// and a request that COMPLETES while its blocks are mid-cascade does not drop the
// CPU write-pin (no leak): the content-addressed CPU tier is decoupled from the
// request lifecycle, so ReleaseKVBlocks (GPU-only) never frees a pinned CPU block;
// the pin releases only when the cascade Write completes.
func TestOffload_INV4_ConservationAndC4(t *testing.T) {
	gpu := NewKVCacheState(256, 2)
	oc := NewOffloadCache(gpu, enabledOffloadCfg(64*4096, 4096, 2)) // CPU capacity 64, 2 tiers
	rng := rand.New(rand.NewSource(7))
	clock := int64(0)

	assertConservation := func(where string) {
		if oc.cpu.usedCount()+oc.cpu.freeCount() != oc.cpu.capacity {
			t.Fatalf("%s: CPU conservation broken used=%d free=%d cap=%d", where, oc.cpu.usedCount(), oc.cpu.freeCount(), oc.cpu.capacity)
		}
		if err := gpu.verifyBlockConservation(); err != nil {
			t.Fatalf("%s: GPU conservation broken: %v", where, err)
		}
	}

	for round := 0; round < 40; round++ {
		clock += 25
		oc.SetClock(clock)
		assertConservation("after SetClock")

		// A fresh request with a partly-shared prefix.
		base := (round % 5) * 8
		toks := []sim.TokenID{sim.TokenID(base + 1), sim.TokenID(base + 2), sim.TokenID(base + 3), sim.TokenID(base + 4), sim.TokenID(1000 + round)}
		req := &sim.Request{ID: "r" + string(rune('a'+round%26)) + string(rune('0'+round/26)), InputTokens: toks}
		oc.AllocateKVBlocks(req, 0, int64(len(toks)), nil)
		assertConservation("after AllocateKVBlocks")

		oc.MirrorToCPU([]*sim.Request{req})
		assertConservation("after MirrorToCPU")

		// C4: with 50% chance, release the request IMMEDIATELY (mid-cascade), before
		// its Write jobs complete. The GPU blocks free; the CPU blocks stay pinned.
		if rng.Intn(2) == 0 {
			pinnedBefore := oc.cpu.usedCount() - oc.cpu.evictableCount()
			oc.ReleaseKVBlocks(req)
			assertConservation("after mid-cascade ReleaseKVBlocks")
			if oc.cpu.usedCount()-oc.cpu.evictableCount() != pinnedBefore {
				t.Fatalf("round %d: mid-cascade release must NOT drop CPU write-pins (C4)", round)
			}
		}
	}

	// Drain all in-flight cascade writes far in the future: every pin must release,
	// so the whole CPU tier becomes evictable (no leaked pin).
	oc.SetClock(clock + 10_000_000)
	assertConservation("after drain")
	if oc.cpu.usedCount() != oc.cpu.evictableCount() {
		t.Fatalf("after draining all cascade writes, every CPU block must be unpinned (no leak): used=%d evictable=%d", oc.cpu.usedCount(), oc.cpu.evictableCount())
	}
}
