package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// BC-C1 stage 2 + BC-C3: a promotion (secondary→CPU Read) completes on the
// SetClock→Poll that reaches its service time — the block transitions
// HIT_PENDING(-1) → HIT(0). A LATER request then reloads it CPU→GPU (the full
// two-hop realization).
func TestOffload_PromotionCompletesOnPoll(t *testing.T) {
	gpu := NewKVCacheState(64, 2)
	oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 1))
	tokens := []sim.TokenID{1, 2, 3, 4}
	keys := blockKeysFor(tokens, 2)
	oc.secondary[0].store(keys[0])
	oc.secondary[0].store(keys[1])

	oc.SetClock(100)
	oc.consultAndReload(tokens, 0, "") // initiate promotion at t=100 (service = base 80)
	if oc.cpu.lookup(keys[0]) != cpuHitPending {
		t.Fatalf("promotion must be HIT_PENDING before completion")
	}

	oc.SetClock(300) // past 100+80 -> Read completes
	if oc.cpu.lookup(keys[0]) != cpuHit || oc.cpu.lookup(keys[1]) != cpuHit {
		t.Fatalf("promotion must complete to a readable HIT after the service window")
	}

	// Stage 2: a later request reloads the now-CPU-resident prefix onto the GPU.
	reqB := &sim.Request{ID: "b", InputTokens: tokens}
	oc.AllocateKVBlocks(reqB, 0, 4, nil)
	if oc.reloadCount != 2 {
		t.Fatalf("a later request must reload both promoted blocks CPU->GPU, reloadCount=%d", oc.reloadCount)
	}
	if got := len(gpu.GetCachedBlocks(tokens)); got != 2 {
		t.Fatalf("the promoted+reloaded prefix must now be GPU-resident, got %d", got)
	}
}

// BC-C7a completion + BC-C3 unpin: a cascade Write completes → the secondary tier
// records the block and the CPU write-lock is released (block re-evictable).
func TestOffload_CascadeCompletesAndUnpins(t *testing.T) {
	gpu := NewKVCacheState(64, 2)
	oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 2)) // 2 tiers
	tokens := []sim.TokenID{1, 2, 3, 4}
	keys := blockKeysFor(tokens, 2)

	req := &sim.Request{ID: "r", InputTokens: tokens}
	oc.SetClock(100)
	oc.AllocateKVBlocks(req, 0, 4, nil)
	oc.MirrorToCPU([]*sim.Request{req}) // stores + cascades (pins) at t=100
	if oc.cpu.evictableCount() != 0 {
		t.Fatalf("blocks must be pinned by in-flight cascade writes, evictable=%d", oc.cpu.evictableCount())
	}

	oc.SetClock(300) // Write jobs (base 80) complete
	for tier := 0; tier < 2; tier++ {
		if !oc.secondary[tier].has(keys[0]) || !oc.secondary[tier].has(keys[1]) {
			t.Fatalf("both blocks must be recorded in secondary tier %d after cascade completion", tier)
		}
	}
	if oc.cpu.evictableCount() != 2 {
		t.Fatalf("both blocks must be unpinned (re-evictable) after all cascade writes complete, evictable=%d", oc.cpu.evictableCount())
	}
}

// INV-3: a backward SetClock is a safe no-op (never panics, never regresses state).
func TestOffload_SetClockBackwardNoOp(t *testing.T) {
	gpu := NewKVCacheState(64, 2)
	oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 1))
	oc.SetClock(500)
	oc.SetClock(300) // backward: must not panic
	oc.SetClock(500)
}
