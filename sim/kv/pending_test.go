package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func countEvents(kvc *KVCacheState) (stored, removed int) {
	for _, e := range kvc.TakeKVEvents() {
		if e.Removed {
			removed++
		} else {
			stored++
		}
	}
	return stored, removed
}

// TestPendingKV_NotAHitSourceAndAbortLeavesTheCache (review 6218929443): a valid cached prefix A;
// a transfer B reserves the same prompt (AllocateRemoteKV: its KV has not arrived); a request C with that
// prefix is admitted before B completes and hits A's blocks, never B's; aborting B leaves A cached
// and publishes nothing.
func TestPendingKV_NotAHitSourceAndAbortLeavesTheCache(t *testing.T) {
	tokens := []sim.TokenID{11, 12, 13, 14}
	kvc := NewKVCacheState(8, 2)
	kvc.EnableKVEvents()
	a := &sim.Request{ID: "a", InputTokens: tokens}
	if !kvc.AllocateKVBlocks(a, 0, int64(len(tokens)), nil) {
		t.Fatal("allocate a")
	}
	aBlocks := append([]int64(nil), kvc.RequestMap["a"]...)
	kvc.ReleaseKVBlocks(a)
	before := kvc.CachedBlockHashes()
	_ = kvc.TakeKVEvents()

	b := &sim.Request{ID: "b", InputTokens: tokens}
	if !kvc.AllocateRemoteKV(b, 0, b.InputLen(), nil) {
		t.Fatal("reserve b")
	}
	bBlocks := map[int64]bool{}
	for _, id := range kvc.RequestMap["b"] {
		bBlocks[id] = true
		if h := kvc.Blocks[id].Hash; h != "" && kvc.HashToBlock[h] == id {
			t.Errorf("pending block %d is registered (hash %q) before its KV arrived", id, h)
		}
	}

	c := &sim.Request{ID: "c", InputTokens: append(append([]sim.TokenID{}, tokens...), 15)}
	hits := kvc.GetCachedBlocks(c.InputTokens)
	if len(hits) != len(aBlocks) {
		t.Fatalf("c hits %v; want a's %d blocks", hits, len(aBlocks))
	}
	for i, id := range hits {
		if bBlocks[id] || id != aBlocks[i] {
			t.Fatalf("c hits block %d; want a's %v, never pending b's", id, aBlocks)
		}
	}
	if !kvc.AllocateKVBlocks(c, int64(len(hits))*2, int64(len(c.InputTokens)), hits) {
		t.Fatal("admit c")
	}

	kvc.ReleaseKVBlocks(b) // the transfer is aborted
	if after := kvc.CachedBlockHashes(); len(after) != len(before) || after[0] != before[0] || after[1] != before[1] {
		t.Errorf("cached hashes %v -> %v; aborting b must leave a's prefix cached", before, after)
	}
	if stored, removed := countEvents(kvc); stored != 0 || removed != 0 {
		t.Errorf("%d stored, %d removed events; want none (the cached content never changed)", stored, removed)
	}
}

// TestPendingKV_CommitCaches: once a transfer's KV arrives (CacheRemoteKV), its blocks are cached in prompt order
// and published; a new prefix is announced, and a prefix already cached keeps its holder.
func TestPendingKV_CommitCaches(t *testing.T) {
	tokens := []sim.TokenID{21, 22, 23, 24}
	kvc := NewKVCacheState(8, 2)
	kvc.EnableKVEvents()
	b := &sim.Request{ID: "b", InputTokens: tokens}
	if !kvc.AllocateRemoteKV(b, 0, b.InputLen(), nil) {
		t.Fatal("reserve")
	}
	if n := len(kvc.CachedBlockHashes()); n != 0 {
		t.Fatalf("%d hashes cached before the KV arrived", n)
	}
	kvc.CacheRemoteKV(b.ID)
	if n := len(kvc.CachedBlockHashes()); n != 2 {
		t.Fatalf("%d hashes cached after the KV arrived, want 2", n)
	}
	events := kvc.TakeKVEvents()
	if len(events) != 2 || events[0].Parent != "" || events[1].Parent != events[0].Hash {
		t.Fatalf("events %+v; want the two blocks stored root first", events)
	}
	if hits := kvc.GetCachedBlocks(tokens); len(hits) != 2 {
		t.Errorf("%d hits on the arrived prompt, want 2", len(hits))
	}
}

// TestUncacheBlock_KeepsContentAnotherBlockHolds: when two blocks hold the same content and the
// one the hash maps to is reused, the hash moves to the other, with no event; it is forgotten with
// its last holder.
func TestUncacheBlock_KeepsContentAnotherBlockHolds(t *testing.T) {
	tokens := []sim.TokenID{31, 32}
	kvc := NewKVCacheState(4, 2)
	kvc.EnableKVEvents()
	for _, id := range []string{"x", "y"} {
		r := &sim.Request{ID: id, InputTokens: tokens}
		if !kvc.AllocateRemoteKV(r, 0, r.InputLen(), nil) {
			t.Fatal("reserve")
		}
		kvc.CacheRemoteKV(r.ID) // two blocks, same content
	}
	h := kvc.CachedBlockHashes()[0]
	owner := kvc.HashToBlock[h]
	other := kvc.RequestMap["x"][0]
	if other == owner {
		other = kvc.RequestMap["y"][0]
	}
	_ = kvc.TakeKVEvents()
	kvc.uncacheBlock(kvc.Blocks[owner])
	if kvc.HashToBlock[h] != other {
		t.Fatalf("hash maps to %d after its owner %d was reused; want the other holder %d", kvc.HashToBlock[h], owner, other)
	}
	if stored, removed := countEvents(kvc); stored+removed != 0 {
		t.Errorf("%d events; want none", stored+removed)
	}
	kvc.uncacheBlock(kvc.Blocks[other])
	if _, ok := kvc.HashToBlock[h]; ok {
		t.Fatal("hash still cached after its last holder was reused")
	}
	if _, removed := countEvents(kvc); removed != 1 {
		t.Errorf("%d removed events; want 1 with the last holder", removed)
	}
}
