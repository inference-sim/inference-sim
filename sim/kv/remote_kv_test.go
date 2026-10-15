package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func remoteKVReq(id string, toks ...sim.TokenID) *sim.Request {
	return &sim.Request{ID: id, InputTokens: toks, State: sim.StateQueued}
}

// A remote reservation's blocks are no prefix hit until CacheRemoteKV, and reusing such an
// unregistered block -- whose content hash another block owns -- leaves that block's entry
// in place.
func TestAllocateRemoteKV_HashesRegisterOnlyOnceCached(t *testing.T) {
	kvc := NewKVCacheState(4, 2)
	owner := remoteKVReq("owner", 1, 2)
	if !kvc.AllocateKVBlocks(owner, 0, 2, nil) {
		t.Fatal("owner allocation failed")
	}
	pulled := remoteKVReq("pulled", 7, 8)
	if !kvc.AllocateRemoteKV(pulled, 0, 2, nil) {
		t.Fatal("remote allocation failed")
	}
	if got := kvc.GetCachedBlocks([]sim.TokenID{7, 8}); len(got) != 0 {
		t.Fatalf("blocks still being pulled hit the prefix cache: %v", got)
	}
	kvc.CacheRemoteKV("pulled")
	if got := kvc.GetCachedBlocks([]sim.TokenID{7, 8}); len(got) != 1 {
		t.Fatalf("after CacheRemoteKV the pulled block must hit, got %v", got)
	}

	// A second remote reservation of the owner's content, released uncached, then reused.
	dup := remoteKVReq("dup", 1, 2)
	if !kvc.AllocateRemoteKV(dup, 0, 2, nil) {
		t.Fatal("dup allocation failed")
	}
	kvc.ReleaseKVBlocks(dup)
	filler := remoteKVReq("filler", 20, 21, 22, 23)
	if !kvc.AllocateKVBlocks(filler, 0, 4, nil) { // reuses the freed dup block
		t.Fatal("filler allocation failed")
	}
	if got := kvc.GetCachedBlocks([]sim.TokenID{1, 2}); len(got) != 1 || got[0] != kvc.RequestMap["owner"][0] {
		t.Fatalf("reusing an unregistered duplicate dropped the owner's prefix entry: %v", got)
	}
	if err := kvc.verifyBlockConservation(); err != nil {
		t.Fatal(err)
	}
}
