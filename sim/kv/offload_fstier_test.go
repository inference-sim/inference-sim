package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kvtransfer"
)

// vLLM v0.31.0's fs tier semantics, each against the source it ports. Fixtures: 2-token GPU
// blocks, one chunk per block, a 4096-byte block.

// fsOC is an offload chain with one fs tier priced by price (nil keeps the tier's physics).
func fsOC(price func(write bool, bytes int64, q int) int64) *OffloadCache {
	cfg := enabledOffloadCfg(1<<20, 4096, 1)
	cfg.Tiers[0].ServiceTime = price
	return NewOffloadCache(NewKVCacheState(64, 2), cfg)
}

// A CPU miss waits one scheduler round even when no secondary tier holds the chunk: the fs
// tier answers RETRY for a chunk whose existence it has not checked
// (tiering/fs/manager.py:215-220, tiering/async_lookup.py:140-161), which defers the request
// (offloading/scheduler.py:697-703, sched/scheduler.py:956-961). The next round it is a
// MISS and the request is admitted to recompute.
func TestFSTier_AColdMissWaitsOneRoundForTheExistenceCheck(t *testing.T) {
	oc := fsOC(nil)
	req := &sim.Request{ID: "r", InputTokens: []sim.TokenID{1, 2, 3, 4}}
	oc.SetClock(10)
	oc.PollDeferred(10)
	if oc.AllocateKVBlocks(req, 0, 4, nil) || !oc.IsDeferred(req.ID) {
		t.Fatal("round 1: a cold CPU miss must defer for the fs existence check")
	}
	oc.SetClock(20)
	oc.PollDeferred(20)
	if !oc.AllocateKVBlocks(req, 0, 4, nil) {
		t.Fatal("round 2: the probe found nothing; the request must be admitted to recompute")
	}
	if oc.promotionsFired != 0 {
		t.Fatalf("a miss promotes nothing, fired %d", oc.promotionsFired)
	}
}

// An existence verdict belongs to the requests that asked: a second request asking while the
// first is live reuses it (no round); once the asker has finished the verdict is gone and the
// next request probes again (async_lookup.py:226-236, fs/manager.py:338-339). Preemption is
// not finishing: the verdict survives it.
func TestFSTier_AnExistenceVerdictLivesWithTheRequestsThatAsked(t *testing.T) {
	oc := fsOC(nil)
	tokens := []sim.TokenID{1, 2, 3, 4}
	a := &sim.Request{ID: "a", InputTokens: tokens}
	if !admitOffload(oc, a, 0, 4, nil) {
		t.Fatal("a was not admitted")
	}
	oc.ReleaseKVBlocks(&sim.Request{ID: "a", State: sim.StateQueued}) // preempted: keeps it
	b := &sim.Request{ID: "b", InputTokens: tokens}
	oc.gpu.HashToBlock = map[string]int64{} // nothing left on GPU: b is a CPU miss again
	if !oc.AllocateKVBlocks(b, 0, 4, nil) {
		t.Fatal("b asked while a's verdict was live; it must not wait a round")
	}
	oc.ReleaseKVBlocks(&sim.Request{ID: "a", State: sim.StateCompleted})
	oc.ReleaseKVBlocks(&sim.Request{ID: "b", State: sim.StateCompleted})
	oc.gpu.HashToBlock = map[string]int64{}
	c := &sim.Request{ID: "c", InputTokens: tokens}
	if oc.AllocateKVBlocks(c, 0, 4, nil) || !oc.IsDeferred(c.ID) {
		t.Fatal("every asker finished; c must probe again and wait a round")
	}
}

// A request's newly stored chunks go to a tier as one store job, written one file after
// another, so the job costs each file's own transfer (tiering/manager.py:737-747,
// fs/manager.py:222-236, fs/io.py:166-187); a chunk the tier already holds is a file that
// exists, which the write skips at no cost (fs/io.py:98-100).
func TestFSTier_AStoreJobWritesEachNewChunkAsItsOwnFile(t *testing.T) {
	var priced []int64
	oc := fsOC(func(write bool, bytes int64, _ int) int64 {
		if write {
			priced = append(priced, bytes)
		}
		return 100
	})
	tokens := []sim.TokenID{1, 2, 3, 4, 5, 6}
	keys := blockKeysFor(tokens, 2)
	oc.secondary[0].store(keys[1]) // already on the tier
	req := &sim.Request{ID: "r", InputTokens: tokens}
	oc.SetClock(1)
	if !admitOffload(oc, req, 0, 6, nil) {
		t.Fatal("not admitted")
	}
	oc.MirrorToCPU([]*sim.Request{req})
	if got := oc.secondary[0].activeJobs(kvtransfer.Write); got != 1 {
		t.Fatalf("%d store jobs, want one for the request", got)
	}
	for _, p := range priced {
		if p != 4096 {
			t.Fatalf("priced a %d-byte write; each file is one 4096-byte chunk", p)
		}
	}
	if oc.cpu.evictableCount() != 1 {
		t.Fatalf("only the 2 chunks being written stay pinned; evictable %d, want 1", oc.cpu.evictableCount())
	}
	oc.SetClock(1 + 150)
	if oc.secondary[0].activeJobs(kvtransfer.Write) != 1 {
		t.Fatal("the 2-file job is still writing its second file at 150 ticks")
	}
	oc.SetClock(1 + 200)
	if oc.secondary[0].activeJobs(kvtransfer.Write) != 0 {
		t.Fatal("the 2-file job must take 2 x 100 ticks")
	}
}

// The fs tier's read and write threads do their I/O on one device: a load priced while a
// store is in service is priced at depth 2.
func TestFSTier_ReadsAndWritesShareTheDeviceDepth(t *testing.T) {
	var readDepth int
	oc := fsOC(func(write bool, _ int64, q int) int64 {
		if !write {
			readDepth = q
		}
		return 1000
	})
	src := []sim.TokenID{7, 8}
	req := &sim.Request{ID: "w", InputTokens: src}
	oc.SetClock(1)
	if !admitOffload(oc, req, 0, 2, nil) {
		t.Fatal("not admitted")
	}
	oc.MirrorToCPU([]*sim.Request{req}) // a store in service
	tokens := []sim.TokenID{1, 2}
	oc.secondary[0].store(blockKeysFor(tokens, 2)[0])
	oc.consultAndReload(tokens, 0, "")
	if readDepth != 2 {
		t.Fatalf("the load was priced at depth %d; with a store in service on the device it is 2", readDepth)
	}
}

// Engines on one node with one root_dir share the tier (file_mapper.py:21-144: no DP rank in
// the path): a chunk one engine wrote is a hit for another, and both engines' threads serve
// the one device.
func TestFSTier_EnginesOfANodeShareOneTier(t *testing.T) {
	cfg := enabledOffloadCfg(1<<20, 4096, 1)
	cfg.Sharing = &sim.OffloadTierSharing{NodeOf: func(id string) int {
		if id == "other" {
			return 1
		}
		return 0
	}}
	a := NewOffloadCache(NewKVCacheState(64, 2), cfg.ForInstance("a"))
	b := NewOffloadCache(NewKVCacheState(64, 2), cfg.ForInstance("b"))
	c := NewOffloadCache(NewKVCacheState(64, 2), cfg.ForInstance("other"))
	tokens := []sim.TokenID{1, 2, 3, 4}
	req := &sim.Request{ID: "r", InputTokens: tokens}
	a.SetClock(1)
	if !admitOffload(a, req, 0, 4, nil) {
		t.Fatal("not admitted")
	}
	a.MirrorToCPU([]*sim.Request{req})
	for _, oc := range []*OffloadCache{a, b, c} {
		oc.SetClock(1_000_000)
	}
	key := blockKeysFor(tokens, 2)[1]
	if !b.secondary[0].has(key) {
		t.Fatal("an engine on the same node must see the chunk another wrote")
	}
	if c.secondary[0].has(key) {
		t.Fatal("an engine on another node has its own disk")
	}
	if a.cpu.evictableCount() != 2 {
		t.Fatalf("the writer's pins must be released when its store lands, evictable %d", a.cpu.evictableCount())
	}
	if got := a.secondary[0].station; got != b.secondary[0].station {
		t.Fatal("engines of a node must share one device station")
	}
}

// The tier has no eviction (the fs tier deletes nothing): once full it stores nothing more,
// as a write to a full filesystem fails.
func TestFSTier_AFullTierStoresNothingMore(t *testing.T) {
	cfg := enabledOffloadCfg(1<<20, 4096, 1)
	cfg.Tiers[0].CapacityBytes = 4096
	oc := NewOffloadCache(NewKVCacheState(64, 2), cfg)
	tokens := []sim.TokenID{1, 2, 3, 4}
	keys := blockKeysFor(tokens, 2)
	oc.secondary[0].store(keys[0])
	oc.secondary[0].store(keys[1])
	if !oc.secondary[0].has(keys[0]) || oc.secondary[0].has(keys[1]) {
		t.Fatal("a one-chunk tier keeps its first chunk and refuses the second")
	}
}

// One lookup promotes every fs-resident chunk up to the first miss, past chunks the CPU tier
// already holds: vLLM's prefix walk goes on through hits and pending promotions and stops only
// at a miss (offloading/scheduler.py:671-703, tiering/manager.py:383-465).
func TestFSTier_OneLookupPromotesEveryFSChunkUpToTheFirstMiss(t *testing.T) {
	oc := fsOC(nil)
	tokens := []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8}
	keys := blockKeysFor(tokens, 2)
	oc.secondary[0].store(keys[0])
	oc.cpu.store(keys[1]) // in the CPU tier: a hit, the walk goes on
	oc.secondary[0].store(keys[2])
	oc.markKnown(keys)
	req := &sim.Request{ID: "r", InputTokens: tokens}
	oc.SetClock(1)
	if oc.AllocateKVBlocks(req, 0, 8, nil) {
		t.Fatal("an fs hit must defer for its promotion")
	}
	if oc.cpu.lookup(keys[0]) != cpuHitPending || oc.cpu.lookup(keys[2]) != cpuHitPending {
		t.Fatal("both fs-resident chunks must be promoted in the one round")
	}
}
