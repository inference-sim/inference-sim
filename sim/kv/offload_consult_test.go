package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/kvkey"
	"github.com/inference-sim/inference-sim/sim/kvtransfer"
)

// blockKeysFor derives the block-stride keys for tokens (== the GPU block hashes,
// BC-K1), so a test can pre-populate the CPU / secondary tiers by content.
func blockKeysFor(tokens []sim.TokenID, blockSize int64) []kvkey.BlockKey {
	return kvkey.DeriveChunkKeys("", tokens, int(blockSize))
}

// BC-C1 stage: a CPU-resident prefix is RELOADED onto the GPU (CPU->GPU), so the
// request's blocks become GPU cache entries. Reload count reflects the CPU->GPU
// hops.
func TestOffload_ReloadsCPUResidentPrefix(t *testing.T) {
	gpu := NewKVCacheState(64, 2)
	oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 1))
	tokens := []sim.TokenID{1, 2, 3, 4} // 2 blocks
	keys := blockKeysFor(tokens, 2)

	// Pre-seed both prefix blocks into the CPU tier (ready), absent from GPU.
	oc.cpu.store(keys[0])
	oc.cpu.store(keys[1])

	req := &sim.Request{ID: "r", InputTokens: tokens}
	if ok := oc.AllocateKVBlocks(req, 0, 4, nil); !ok {
		t.Fatalf("allocation should succeed")
	}
	if oc.reloadCount != 2 {
		t.Fatalf("both CPU-resident prefix blocks must be reloaded to GPU (CPU->GPU), reloadCount=%d", oc.reloadCount)
	}
	if got := len(gpu.GetCachedBlocks(tokens)); got != 2 {
		t.Fatalf("after reload the prefix must be GPU-resident, GetCachedBlocks=%d", got)
	}
}

// BC-C1 (no direct secondary->GPU): a secondary-resident-only prefix triggers an
// async PROMOTION into CPU (a Read job + HIT_PENDING), NOT a secondary->GPU
// transfer. The request itself does not get the promoted block (reloadCount==0);
// the promotion benefits a later request.
func TestOffload_SecondaryHitInitiatesPromotionNotDirectToGPU(t *testing.T) {
	gpu := NewKVCacheState(64, 2)
	oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 1))
	tokens := []sim.TokenID{1, 2, 3, 4}
	keys := blockKeysFor(tokens, 2)

	// Blocks live ONLY in secondary tier 0 (absent from CPU and GPU).
	oc.secondary[0].store(keys[0])
	oc.secondary[0].store(keys[1])

	oc.SetClock(100)
	req := &sim.Request{ID: "r", InputTokens: tokens}
	oc.AllocateKVBlocks(req, 0, 4, nil)

	if oc.promotionsFired != 1 {
		t.Fatalf("a secondary hit must initiate exactly one promotion, got %d", oc.promotionsFired)
	}
	if oc.reloadCount != 0 {
		t.Fatalf("no CPU->GPU reload: a secondary block is promoted to CPU, never transferred straight to GPU (BC-C1), reloadCount=%d", oc.reloadCount)
	}
	if oc.cpu.lookup(keys[0]) != cpuHitPending {
		t.Fatalf("the promoted block must be HIT_PENDING (-1) in CPU, got %v", oc.cpu.lookup(keys[0]))
	}
	if n := oc.station.ActiveJobs(0, kvtransfer.Read); n != 1 {
		t.Fatalf("exactly one Read (promotion) job must be in service on tier 0, got %d", n)
	}
}

// BC-C5 at the chain level: when the CPU evictable pool is too small for the
// promotion run, the promotion is REFUSED (recompute), not partially forced.
func TestOffload_PromotionRefusedWhenEvictableShort(t *testing.T) {
	gpu := NewKVCacheState(64, 2)
	// CPU capacity 2 blocks (8192 bytes / 4096).
	oc := NewOffloadCache(gpu, enabledOffloadCfg(8192, 4096, 1))
	// Fill + pin both CPU blocks -> free=0, evictable=0.
	filler := blockKeysFor([]sim.TokenID{90, 91, 92, 93}, 2)
	oc.cpu.store(filler[0])
	oc.cpu.store(filler[1])
	oc.cpu.pin(filler[0])
	oc.cpu.pin(filler[1])

	tokens := []sim.TokenID{1, 2, 3, 4}
	keys := blockKeysFor(tokens, 2)
	oc.secondary[0].store(keys[0])
	oc.secondary[0].store(keys[1])

	oc.SetClock(10)
	oc.consultAndReload(tokens, 0, "")

	if oc.promotionsFailed != 1 {
		t.Fatalf("promotion into a locked CPU tier must be refused (BC-C5), promotionsFailed=%d", oc.promotionsFailed)
	}
	if oc.promotionsFired != 0 {
		t.Fatalf("no promotion should fire when the evictable gate refuses it, promotionsFired=%d", oc.promotionsFired)
	}
	if oc.cpu.lookup(keys[0]) != cpuMiss {
		t.Fatalf("a refused promotion must not allocate a CPU slot, got %v", oc.cpu.lookup(keys[0]))
	}
}

// C2 (same-step dedup): once a promotion is in flight (HIT_PENDING), a second
// consult of the same prefix in the same step does NOT start a duplicate
// promotion — the -1 sentinel is the convoy latch.
func TestOffload_SameStepDedup(t *testing.T) {
	gpu := NewKVCacheState(64, 2)
	oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 1))
	tokens := []sim.TokenID{1, 2, 3, 4}
	keys := blockKeysFor(tokens, 2)
	oc.secondary[0].store(keys[0])
	oc.secondary[0].store(keys[1])

	oc.SetClock(50)
	oc.consultAndReload(tokens, 0, "") // request A: initiates promotion
	oc.consultAndReload(tokens, 0, "") // request B (same step): must see HIT_PENDING, no new promotion

	if oc.promotionsFired != 1 {
		t.Fatalf("two same-step consults of the same prefix must promote ONCE (C2 convoy), got %d", oc.promotionsFired)
	}
	if n := oc.station.ActiveJobs(0, kvtransfer.Read); n != 1 {
		t.Fatalf("only one Read job must exist after the dedup'd second consult, got %d", n)
	}
}
