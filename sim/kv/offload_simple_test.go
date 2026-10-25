package kv

import (
	"testing"

	sim "github.com/inference-sim/inference-sim/sim/engine"
)

// vLLM v0.31.0's SimpleCPUOffloadConnector store side (simple_kv_offload/manager.py).

func simpleOC(lazy bool, storeTicks int64) *OffloadCache {
	cfg := enabledOffloadCfg(1<<20, 4096, 0)
	cfg.Simple, cfg.LazyOffload, cfg.OffloadPromptOnly = true, lazy, false
	cfg.LazyTargetFree = 4
	if storeTicks > 0 {
		cfg.CPUStoreTime = func(int64) int64 { return storeTicks }
	}
	return NewOffloadCache(NewKVCacheState(64, 2), cfg)
}

// Eager stores every full block of a scheduled request, decode blocks too (no prompt-only
// clamp, manager.py:641-951), and a stored block is readable only once its copy lands
// (manager.py:1008-1165), the copies landing one after another (one store stream).
func TestSimpleOffload_EagerStoresEveryFullBlockOnceItsCopyLands(t *testing.T) {
	oc := simpleOC(false, 100)
	req := &sim.Request{ID: "r", InputTokens: []sim.TokenID{1, 2}, OutputTokens: []sim.TokenID{3, 4}}
	oc.SetClock(1)
	if !oc.AllocateKVBlocks(req, 0, 2, nil) {
		t.Fatal("prefill not admitted")
	}
	for pi := int64(2); pi < 4; pi++ { // two decode steps fill block 1 with output tokens
		req.ProgressIndex = pi
		if !oc.AllocateKVBlocks(req, pi, pi+1, nil) {
			t.Fatal("decode not allocated")
		}
	}
	oc.MirrorToCPU([]*sim.Request{req})
	keys := blockKeysFor([]sim.TokenID{1, 2, 3, 4}, 2)
	for i, k := range keys {
		if oc.cpu.lookup(k) != cpuHitPending {
			t.Fatalf("block %d: a store in flight is not readable yet", i)
		}
	}
	oc.SetClock(101)
	if oc.cpu.lookup(keys[0]) != cpuHit || oc.cpu.lookup(keys[1]) != cpuHitPending {
		t.Fatal("at 100 ticks the first copy has landed and the second, queued behind it, has not")
	}
	oc.SetClock(201)
	if oc.cpu.lookup(keys[1]) != cpuHit {
		t.Fatal("the decode block must be stored too: the eager store has no prompt-only clamp")
	}
}

// Lazy stores only blocks at the head of the GPU free queue -- cached blocks no request holds,
// the next to be evicted -- never a running request's (manager.py:585-639).
func TestSimpleOffload_LazyStoresOnlyFreedBlocks(t *testing.T) {
	oc := simpleOC(true, 0)
	held := &sim.Request{ID: "h", InputTokens: []sim.TokenID{1, 2}}
	done := &sim.Request{ID: "d", InputTokens: []sim.TokenID{5, 6}, State: sim.StateCompleted}
	oc.SetClock(1)
	if !oc.AllocateKVBlocks(held, 0, 2, nil) || !oc.AllocateKVBlocks(done, 0, 2, nil) {
		t.Fatal("not admitted")
	}
	// The free queue's head holds the never-used blocks first; push them behind the freed one
	// by taking them, so the freed cached block is within the lazy window.
	for blk := oc.gpu.FreeHead; blk != nil && blk.Hash == ""; blk = oc.gpu.FreeHead {
		oc.gpu.popFreeBlock()
	}
	oc.ReleaseKVBlocks(done)
	oc.MirrorToCPU([]*sim.Request{held})
	if oc.cpu.lookup(blockKeysFor([]sim.TokenID{5, 6}, 2)[0]) != cpuHit {
		t.Fatal("a freed cached block at the head of the free queue must be stored")
	}
	if oc.cpu.lookup(blockKeysFor([]sim.TokenID{1, 2}, 2)[0]) != cpuMiss {
		t.Fatal("a running request's block must not be stored by the lazy mode")
	}
}
