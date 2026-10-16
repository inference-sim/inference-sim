package kv

import (
	"testing"

	"pgregory.net/rapid"

	"github.com/inference-sim/inference-sim/sim"
)

// loadCache is a CPU-tier store with a priced load: bs-token blocks, one block a chunk.
func loadCache(pool, bs int64) *OffloadCache {
	oc := chunkCache(pool, bs, 1, 64, 0)
	oc.cpuLoadTime = bandwidthLoad(64)
	return oc
}

// vLLM v0.31.0 allocates an async load's blocks when the load starts (allocate_slots with the
// external tokens and delay_cache_blocks, sched/scheduler.py:1177-1220) and caches them only
// when it lands (_update_waiting_for_remote_kv, :3052-3085). So from its start the load holds
// a block for every token it brings, no other request hits on them until it lands, and once
// admitted the request takes only the blocks past the loaded prefix.
func TestLoadStart_TheLoadHoldsItsBlocksFromItsStart(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		const bs = 2
		n := rapid.Int64Range(1, 12).Draw(rt, "loadedBlocks")
		tail := rapid.Int64Range(1, 6).Draw(rt, "tailTokens")
		oc := loadCache(64, bs)
		tokens := seqTokens(1, int(n*bs+tail))
		seedCPU(oc, tokens[:n*bs])
		req := &sim.Request{ID: "r", InputTokens: tokens}

		oc.SetClock(0)
		oc.PollDeferred(0)
		if oc.AllocateKVBlocks(req, 0, req.InputLen(), nil) || !oc.IsDeferred("r") || !oc.HoldsKV("r") {
			rt.Fatal("a CPU-tier hit did not start a load holding its blocks")
		}
		if used := oc.UsedBlocks(); used != n {
			rt.Fatalf("the load of %d blocks holds %d blocks at its start", n, used)
		}
		if hit := len(oc.GetCachedBlocks(tokens)); hit != 0 {
			rt.Fatalf("another request hits %d blocks a load is still filling", hit)
		}
		at := oc.loadBusyUntil
		oc.SetClock(at)
		oc.PollDeferred(at)
		computed, gpuHit, ok := oc.LoadedPrefix("r")
		if !ok || computed != n*bs || gpuHit != 0 {
			rt.Fatalf("landed load reports computed %d, GPU hit %d, ok %v; want %d, 0, true", computed, gpuHit, ok, n*bs)
		}
		if hit := int64(len(oc.GetCachedBlocks(tokens))); hit != n {
			rt.Fatalf("a landed load is cached for %d of its %d blocks", hit, n)
		}
		if !oc.AllocateKVBlocks(req, computed, req.InputLen(), nil) || oc.HoldsKV("r") {
			rt.Fatal("the landed request was not admitted")
		}
		if used, want := oc.UsedBlocks(), (req.InputLen()+bs-1)/bs; used != want {
			rt.Fatalf("admitted holding %d blocks, want %d for its prompt", used, want)
		}
		if err := oc.gpu.verifyBlockConservation(); err != nil {
			rt.Fatal(err)
		}
	})
}

// An async load must leave free what the in-flight prefills still need to hold their whole
// sequence (reserved_blocks, sched/scheduler.py:1194-1205, 3003-3036 at v0.31.0), so it cannot
// take blocks a running prefill relies on. A load that fits only into that reserve is refused,
// takes nothing and is not set aside (the admission loop stops there); with the reserve free it
// starts.
func TestLoadStart_ALoadLeavesTheInflightPrefillsRoom(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		const bs = 2
		n := rapid.Int64Range(1, 8).Draw(rt, "loadedBlocks")
		held := rapid.Int64Range(1, 4).Draw(rt, "prefillHeld")
		rest := rapid.Int64Range(1, 6).Draw(rt, "prefillRemaining")
		room := rapid.Bool().Draw(rt, "roomForBoth")
		pool := held + n + rest
		if !room {
			pool--
		}
		oc := loadCache(pool, bs)
		running := promptRequest("p", int((held+rest)*bs))
		if !oc.AllocateKVBlocks(running, 0, held*bs, nil) {
			rt.Fatal("the running prefill's first chunk was refused")
		}
		running.ProgressIndex = held * bs
		oc.SetInflightPrefills([]sim.InflightPrefill{{Req: running, Computed: held * bs}})

		tokens := seqTokens(1, int(n*bs+1))
		seedCPU(oc, tokens[:n*bs])
		req := &sim.Request{ID: "r", InputTokens: tokens}
		used := oc.UsedBlocks()
		oc.AllocateKVBlocks(req, 0, req.InputLen(), nil)
		if started := oc.HoldsKV("r"); started != room {
			rt.Fatalf("pool %d, prefill holding %d of %d, load of %d: started=%v, want %v",
				pool, held, held+rest, n, started, room)
		}
		if !room && (oc.UsedBlocks() != used || oc.IsDeferred("r")) {
			rt.Fatalf("a refused load took %d blocks or was set aside", oc.UsedBlocks()-used)
		}
	})
}

// A request that leaves while its load holds blocks (timeout, eviction) frees them, as vLLM
// frees a WAITING_FOR_REMOTE_KVS request's blocks.
func TestLoadStart_AHolderThatLeavesFreesItsBlocks(t *testing.T) {
	oc := loadCache(32, 2)
	tokens := seqTokens(1, 13)
	seedCPU(oc, tokens[:12])
	req := &sim.Request{ID: "r", InputTokens: tokens}
	oc.AllocateKVBlocks(req, 0, req.InputLen(), nil)
	if !oc.HoldsKV("r") || oc.UsedBlocks() == 0 {
		t.Fatal("precondition: the load holds blocks")
	}
	oc.ClearDeferred("r")
	if oc.UsedBlocks() != 0 || oc.HoldsKV("r") {
		t.Fatalf("%d blocks still held after the request left", oc.UsedBlocks())
	}
	if err := oc.gpu.verifyBlockConservation(); err != nil {
		t.Fatal(err)
	}
}

// Requests holding blocks are drained ahead of the waiting queue (kv_holding_waiting,
// sched/scheduler.py:887 at v0.31.0): a landed load is admitted although a request ahead of it
// in the queue does not fit, instead of waiting behind it.
func TestLoadStart_ALandedLoadIsScheduledAheadOfTheWaitingQueue(t *testing.T) {
	oc := loadCache(8, 2)
	tokens := seqTokens(1, 10)
	seedCPU(oc, tokens[:8])
	loaded := &sim.Request{ID: "l", InputTokens: tokens, OutputTokens: make([]sim.TokenID, 2), State: sim.StateQueued}
	wq := &sim.WaitQueue{}
	wq.Enqueue(loaded)
	bf := sim.NewBatchFormation("fcfs")
	ctx := func(now int64, rb *sim.Batch) sim.BatchContext {
		return sim.BatchContext{RunningBatch: rb, WaitQ: wq, KVCache: oc, MaxNumBatchedTokens: 64,
			MaxNumSeqs: 8, Now: now, ComputedTokens: map[string]int64{}}
	}
	r := bf.FormBatch(ctx(0, &sim.Batch{}))
	if len(r.RunningBatch.Requests) != 0 || !oc.HoldsKV("l") {
		t.Fatal("precondition: the load started and holds its blocks")
	}
	blocked := promptRequest("x", 10) // five blocks; four are free
	blocked.OutputTokens = make([]sim.TokenID, 2)
	wq.PrependFront(blocked)
	r = bf.FormBatch(ctx(oc.loadBusyUntil, r.RunningBatch))
	if len(r.RunningBatch.Requests) != 1 || r.RunningBatch.Requests[0] != loaded {
		t.Fatalf("after its load landed, running %v; want the loaded request ahead of the one that does not fit",
			r.RunningBatch.Requests)
	}
}
