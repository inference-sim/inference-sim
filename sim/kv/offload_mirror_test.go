package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kvtransfer"
)

// BC-C7a + BC-C3(pin): MirrorToCPU stores each new full block and cascades it to
// EVERY secondary tier, pinning the CPU block once per in-flight write so it is
// non-evictable until the writes complete.
func TestOffload_CascadePinsAllTiers(t *testing.T) {
	gpu := NewKVCacheState(64, 2)
	oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 2)) // 2 secondary tiers
	tokens := []sim.TokenID{1, 2, 3, 4}                            // 2 prompt blocks
	keys := blockKeysFor(tokens, 2)

	req := &sim.Request{ID: "r", InputTokens: tokens}
	oc.SetClock(100)
	oc.AllocateKVBlocks(req, 0, 4, nil) // 2 GPU blocks with hashes
	oc.MirrorToCPU([]*sim.Request{req})

	// Both blocks landed in CPU and are pinned by 2 in-flight writes each -> evictable 0.
	if oc.cpu.lookup(keys[0]) != cpuHit || oc.cpu.lookup(keys[1]) != cpuHit {
		t.Fatalf("mirrored blocks must be CPU-resident HITs")
	}
	if oc.cpu.evictableCount() != 0 {
		t.Fatalf("cascade writes must pin every mirrored block (evictable 0), got %d", oc.cpu.evictableCount())
	}
	// 2 blocks × 2 tiers = 4 Write jobs; per tier, 2 write jobs in service.
	if w0, w1 := oc.station.ActiveJobs(0, kvtransfer.Write), oc.station.ActiveJobs(1, kvtransfer.Write); w0 != 2 || w1 != 2 {
		t.Fatalf("each tier must have 2 cascade Write jobs, got tier0=%d tier1=%d", w0, w1)
	}
}

// BC-C4 + I3: when the CPU tier is full and every block is pinned, MirrorToCPU
// SKIPS the store (counted) rather than force-evicting a locked block.
func TestOffload_MirrorSkipsWhenFullPinned(t *testing.T) {
	gpu := NewKVCacheState(64, 2)
	oc := NewOffloadCache(gpu, enabledOffloadCfg(8192, 4096, 1)) // CPU capacity 2
	// Pre-fill + pin both CPU slots.
	filler := blockKeysFor([]sim.TokenID{90, 91, 92, 93}, 2)
	oc.cpu.store(filler[0])
	oc.cpu.store(filler[1])
	oc.cpu.pin(filler[0])
	oc.cpu.pin(filler[1])

	req := &sim.Request{ID: "r", InputTokens: []sim.TokenID{1, 2, 3, 4}}
	oc.SetClock(10)
	oc.AllocateKVBlocks(req, 0, 4, nil)
	oc.MirrorToCPU([]*sim.Request{req})

	if oc.mirrorSkipped != 2 {
		t.Fatalf("both mirror stores must be skipped when CPU is full-and-pinned, mirrorSkipped=%d", oc.mirrorSkipped)
	}
	// The pinned fillers survive; nothing new was force-stored.
	if oc.cpu.usedCount() != 2 {
		t.Fatalf("CPU must still hold exactly the 2 pinned fillers, used=%d", oc.cpu.usedCount())
	}
}

// OffloadPromptOnly (vLLM default TRUE): decode-generated blocks are NOT offloaded.
// The gate is driven by InputLen()/blockSize; a full, hashed decode block (block
// index >= promptBlocks) is offloaded only when OffloadPromptOnly is false.
func TestOffload_PromptOnlySkipsDecode(t *testing.T) {
	tokens := []sim.TokenID{1, 2, 3, 4} // InputLen 4 -> 2 prompt blocks

	build := func(promptOnly bool) *OffloadCache {
		gpu := NewKVCacheState(64, 2)
		cfg := enabledOffloadCfg(1<<20, 4096, 1)
		cfg.OffloadPromptOnly = promptOnly
		oc := NewOffloadCache(gpu, cfg)
		req := &sim.Request{ID: "r", InputTokens: tokens}
		oc.SetClock(1)
		oc.AllocateKVBlocks(req, 0, 4, nil) // 2 prompt blocks (full, hashed)

		// Append a FULL, HASHED "decode" block (index 2) beyond the prompt so the
		// only thing gating its mirror is OffloadPromptOnly (not the Hash/full guard).
		db := gpu.popFreeBlock()
		db.Tokens = []sim.TokenID{5, 6}
		db.Hash = "decodeblockhash000000000000000000000000000000000000000000000000"
		db.RefCount = 1
		db.InUse = true
		gpu.HashToBlock[db.Hash] = db.ID
		gpu.RequestMap["r"] = append(gpu.RequestMap["r"], db.ID)

		oc.MirrorToCPU([]*sim.Request{req})
		return oc
	}

	if oc := build(true); oc.cpu.usedCount() != 2 {
		t.Fatalf("OffloadPromptOnly=true must mirror only the 2 prompt blocks, got %d", oc.cpu.usedCount())
	}
	if oc := build(false); oc.cpu.usedCount() != 3 {
		t.Fatalf("OffloadPromptOnly=false must mirror the decode block too (3), got %d", oc.cpu.usedCount())
	}
}
