package kv

import (
	"testing"

	sim "github.com/inference-sim/inference-sim/sim/engine"
	"pgregory.net/rapid"
)

// bandwidthLoad prices a CPU->GPU load at bytesPerTick, rounded up, as the kernel's TierTime
// does for a host tier with no base latency.
func bandwidthLoad(bytesPerTick int64) func(int64) int64 {
	return func(bytes int64) int64 { return (bytes + bytesPerTick - 1) / bytesPerTick }
}

// admitTick drives req through the step loop as batch formation does (SetClock, PollDeferred,
// then an admission attempt unless still set aside) from clock start in steps of step ticks,
// and returns the clock it was admitted at, or -1.
func admitTick(oc *OffloadCache, req *sim.Request, start, step int64) int64 {
	for clock := start; clock < start+1_000_000*step; clock += step {
		oc.SetClock(clock)
		skipped := false
		for _, id := range oc.PollDeferred(clock) {
			skipped = skipped || id == req.ID
		}
		if !skipped && oc.AllocateKVBlocks(req, 0, req.InputLen(), nil) {
			return clock
		}
	}
	return -1
}

// seedCPU stores every whole chunk of tokens in the CPU tier, as a finished request's mirror
// would after the GPU evicted it.
func seedCPU(oc *OffloadCache, tokens []sim.TokenID) {
	bs, bpc := oc.gpu.BlockSize(), oc.blocksPerChunk
	keys := blockKeysFor(tokens, bs)
	for c := int64(1); c*bpc <= int64(len(keys)); c++ {
		oc.cpu.store(keys[c*bpc-1])
	}
}

// INV-K7: a new request whose prefix is CPU-resident is admitted no sooner than the load of
// its bytes over the host link takes -- reload time >= bytes / host-device bandwidth -- and,
// once admitted, holds the loaded prefix on the GPU. A request with nothing on the CPU tier
// is admitted on its first attempt. Holds at every chunk width, step length and bandwidth,
// and behind earlier loads that still occupy the link.
func TestCPULoad_AHitWaitsAtLeastItsTransfer(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		const bs, perBlock = 2, 4096
		bpc := rapid.Int64Range(1, 4).Draw(rt, "bpc")
		oc := chunkCache(512, bs, bpc, 64, 0)
		bw := rapid.Int64Range(1, 20000).Draw(rt, "bytesPerTick")
		oc.cpuLoadTime = bandwidthLoad(bw)
		step := rapid.Int64Range(1, 5000).Draw(rt, "step")
		start := rapid.Int64Range(0, 1000).Draw(rt, "start")

		tokens := seqTokens(1, rapid.IntRange(1, 64).Draw(rt, "len"))
		seedCPU(oc, tokens)
		loaded := (int64(len(tokens)) / bs / bpc) * bpc
		if busy := rapid.Int64Range(0, 3000).Draw(rt, "linkBusy"); busy > 0 {
			oc.loadBusyUntil = start + busy
		}
		at := admitTick(oc, &sim.Request{ID: "r", InputTokens: tokens}, start, step)
		if at < 0 {
			rt.Fatal("the request was never admitted")
		}
		if loaded == 0 {
			if at != start || oc.CPULoadsStarted() != 0 {
				rt.Fatalf("no CPU-resident chunk, yet admitted at %d (first attempt %d), %d loads",
					at, start, oc.CPULoadsStarted())
			}
			return
		}
		if floor := loaded * perBlock / bw; at-start < floor {
			rt.Fatalf("loaded %d blocks (%d B) at %d B/tick in %d ticks; the transfer takes at least %d",
				loaded, loaded*perBlock, bw, at-start, floor)
		}
		if got := int64(len(oc.GetCachedBlocks(tokens))); got < loaded {
			rt.Fatalf("admitted holding %d cached blocks of the %d it loaded", got, loaded)
		}
		if oc.cpu.evictableCount() != oc.cpu.usedCount() {
			rt.Fatalf("a finished load left %d of %d CPU chunks pinned",
				oc.cpu.usedCount()-oc.cpu.evictableCount(), oc.cpu.usedCount())
		}
	})
}

// Halving the host bandwidth strictly delays a CPU-tier hit's admission and leaves a request
// with no CPU-tier hit admitted at once, whichever goes first.
func TestCPULoad_HalvingBandwidthDelaysOnlyTheHit(t *testing.T) {
	const bs, step = 2, 10
	hit, miss := seqTokens(1, 64), seqTokens(1000, 64)
	admit := func(bw int64) (hitAt, missAt int64) {
		oc := chunkCache(512, bs, 1, 64, 0)
		oc.cpuLoadTime = bandwidthLoad(bw)
		seedCPU(oc, hit)
		return admitTick(oc, &sim.Request{ID: "hit", InputTokens: hit}, 0, step),
			admitTick(oc, &sim.Request{ID: "miss", InputTokens: miss}, 0, step)
	}
	hitFull, missFull := admit(4096)
	hitHalf, missHalf := admit(2048)
	if hitHalf <= hitFull {
		t.Errorf("halving the bandwidth admitted the CPU-tier hit at %d, against %d at full bandwidth", hitHalf, hitFull)
	}
	if missFull != 0 || missHalf != 0 {
		t.Errorf("a request with no CPU-tier hit waited: admitted at %d and %d, want 0", missFull, missHalf)
	}
}

// Leaving the queue mid-load releases the load's pins, so its chunks stay evictable.
func TestCPULoad_ClearDeferredReleasesThePins(t *testing.T) {
	oc := chunkCache(512, 2, 1, 64, 0)
	oc.cpuLoadTime = bandwidthLoad(1)
	tokens := seqTokens(1, 16)
	seedCPU(oc, tokens)
	req := &sim.Request{ID: "r", InputTokens: tokens}
	if oc.AllocateKVBlocks(req, 0, req.InputLen(), nil) || !oc.IsDeferred("r") {
		t.Fatal("a CPU-tier hit was admitted without waiting for its load")
	}
	if oc.cpu.evictableCount() == oc.cpu.usedCount() {
		t.Fatal("a load in flight pinned none of its chunks")
	}
	oc.ClearDeferred("r")
	if oc.cpu.evictableCount() != oc.cpu.usedCount() {
		t.Errorf("ClearDeferred left %d chunks pinned", oc.cpu.usedCount()-oc.cpu.evictableCount())
	}
}
