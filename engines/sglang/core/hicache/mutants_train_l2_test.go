// Original to blis-sglang.

package hicache

import (
	"slices"
	"testing"

	"github.com/inference-sim/inference-sim/engines/sglang/core/cache"
)

// Behavioural tests of the host tier (the mutation sweep of the M3.1 train): each fails on one
// generated mutant of l2.go that the golden traces do not reach. They run a real radix cache over a
// real token allocator and host pool, and assert what the cache then holds and answers.

// tierAlloc is the token allocator with the done-event hook the controller sets.
type tierAlloc struct{ *cache.TokenAllocator }

func (tierAlloc) SetHicacheTransferDoneEvent(string, *Event) {}

type tierRig struct {
	t     *testing.T
	rc    *cache.RadixCache
	l2    *L2
	alloc *tierAlloc
	host  *HostPool
	clock *Clock
}

// newTierRig is a cache of pages of one token over deviceSlots device slots and hostSlots host
// slots, under the given write policy.
func newTierRig(t *testing.T, deviceSlots, hostSlots int, policy string) *tierRig {
	t.Helper()
	r := &tierRig{t: t, clock: &Clock{}}
	r.alloc = &tierAlloc{cache.NewTokenAllocator(deviceSlots, nil, false)}
	rc, err := cache.NewRadixCache(cache.RadixConfig{PageSize: 1}, r.alloc, cache.NewRequestPool(8, 64))
	if err != nil {
		t.Fatal(err)
	}
	r.rc = rc
	tokens := hostSlots
	if r.host, err = NewHostPool(HostSizing{PageSize: 1, HostTokenCapacity: &tokens}); err != nil {
		t.Fatal(err)
	}
	ctrl, err := NewController(ControllerConfig{WritePolicy: policy, IOBackend: "direct", Layout: "layer_first", PageSize: 1, TransferLayerIDMax: 2},
		r.clock, r.host, r.host.Size(), r.alloc, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r.l2, err = NewL2(rc, L2Config{WritePolicy: policy, IOBackend: "direct", Layout: "layer_first"}, ctrl, r.host); err != nil {
		t.Fatal(err)
	}
	return r
}

func key(toks ...int64) cache.Key { return cache.Key{Tokens: toks} }

// insert puts the tokens in the tree on freshly allocated device slots and returns the node that
// ends them.
func (r *tierRig) insert(toks ...int64) cache.NodeID {
	r.t.Helper()
	slots, ok := r.alloc.Alloc(len(toks))
	if !ok {
		r.t.Fatalf("no device room for %d tokens", len(toks))
	}
	r.rc.Insert(cache.InsertParams{Key: key(toks...), Value: slots})
	return r.rc.MatchPrefix(cache.MatchPrefixParams{Key: key(toks...)}).LastDeviceNode
}

func (r *tierRig) match(toks ...int64) cache.MatchResult {
	return r.rc.MatchPrefix(cache.MatchPrefixParams{Key: key(toks...)})
}

// A load-back of exactly load_back_threshold tokens is worth the copy; one token fewer is not.
func TestLoadBackLoadsAPathOfExactlyTheThreshold(t *testing.T) {
	for _, tc := range []struct {
		threshold int
		want      bool
	}{{4, true}, {5, false}, {1, true}} {
		r := newTierRig(t, 16, 16, "write_through")
		r.insert(1, 2, 3, 4)
		r.l2.WritingCheck(true)
		r.rc.Evict(cache.EvictParams{NumTokens: 4})
		m := r.match(1, 2, 3, 4)
		if m.HostHitLength != 4 || m.DevicePrefixLen != 0 {
			t.Fatalf("setup: host hit %d, device prefix %d, want a host-only path of 4", m.HostHitLength, m.DevicePrefixLen)
		}
		r.l2.loadBackThreshold = tc.threshold
		if got := r.l2.LoadBack(m.BestMatchNode, nil); got != tc.want {
			t.Errorf("threshold %d, path of 4 tokens: LoadBack = %v, want %v", tc.threshold, got, tc.want)
		}
		if tc.want && r.alloc.AvailableSize() != 12 {
			t.Errorf("threshold %d: %d device slots free after the load, want 12", tc.threshold, r.alloc.AvailableSize())
		}
		if !tc.want && r.alloc.AvailableSize() != 16 {
			t.Errorf("threshold %d: %d device slots free after a refused load, want all 16", tc.threshold, r.alloc.AvailableSize())
		}
	}
}

// The memory quota counts the tokens loaded plus the device ancestors the lock newly protects (the
// receipt's delta): a quota equal to the sum admits the load, one token less refuses it, and a
// refusal leaves nothing locked.
func TestLoadBackCountsTheProtectedAncestorsAgainstTheQuota(t *testing.T) {
	setup := func() (*tierRig, cache.NodeID) {
		r := newTierRig(t, 16, 16, "write_through")
		r.insert(1, 2)
		r.insert(1, 2, 3, 4)
		r.l2.WritingCheck(true)
		r.rc.Evict(cache.EvictParams{NumTokens: 2}) // the leaf's 2 tokens leave the device; the parent's 2 stay
		m := r.match(1, 2, 3, 4)
		if m.HostHitLength != 2 || m.DevicePrefixLen != 2 {
			t.Fatalf("setup: host hit %d, device prefix %d, want 2 and 2", m.HostHitLength, m.DevicePrefixLen)
		}
		r.l2.loadBackThreshold = 2
		return r, m.BestMatchNode
	}
	for _, tc := range []struct {
		quota int
		want  bool
	}{{4, true}, {3, false}, {2, false}, {100, true}} {
		r, node := setup()
		before := r.rc.EvictableSize()
		q := tc.quota
		if got := r.l2.LoadBack(node, &q); got != tc.want {
			t.Errorf("quota %d (2 loaded + 2 protected): LoadBack = %v, want %v", tc.quota, got, tc.want)
		}
		if !tc.want && (r.rc.EvictableSize() != before || r.rc.ProtectedSize() != 0) {
			t.Errorf("quota %d refused: evictable %d (was %d), protected %d; want the locks released", tc.quota, r.rc.EvictableSize(), before, r.rc.ProtectedSize())
		}
	}
	// No quota admits it whatever it would protect.
	r, node := setup()
	if !r.l2.LoadBack(node, nil) {
		t.Error("no quota: LoadBack refused")
	}
}

// When the device is short, the load evicts what it can; the load goes through when the eviction
// frees exactly enough, and is refused, with its locks released, when it frees less.
func TestLoadBackTakesExactlyTheRoomEvictionFrees(t *testing.T) {
	setup := func(lockC bool) (*tierRig, cache.NodeID) {
		r := newTierRig(t, 10, 16, "write_through")
		r.insert(1, 2, 3, 4) // A: to be loaded back
		r.l2.WritingCheck(true)
		r.rc.Evict(cache.EvictParams{NumTokens: 4}) // A is host-only; 10 device slots free
		b := r.insert(10, 11, 12, 13, 14, 15)       // B: 6 tokens, locked below so eviction cannot take it
		c := r.insert(20, 21)                       // C: 2 tokens, the only evictable leaf unless locked
		r.l2.WritingCheck(true)
		r.rc.IncLockRef(b)
		if lockC {
			r.rc.IncLockRef(c)
		}
		if got := r.alloc.AvailableSize(); got != 2 {
			t.Fatalf("setup: %d device slots free, want 2", got)
		}
		r.l2.loadBackThreshold = 4
		return r, r.match(1, 2, 3, 4).BestMatchNode
	}
	r, a := setup(false)
	if !r.l2.LoadBack(a, nil) {
		t.Error("evicting C frees the 2 slots the 4-token load is short of: LoadBack refused")
	}
	if got := r.alloc.AvailableSize(); got != 0 {
		t.Errorf("after the load %d device slots are free, want 0 (A's 4 took what C's 2 and the 2 free gave)", got)
	}
	if m := r.match(20, 21); m.DevicePrefixLen != 0 {
		t.Errorf("C is still on the device after the load needed its room: prefix %d", m.DevicePrefixLen)
	}
	r, a = setup(true)
	if r.l2.LoadBack(a, nil) {
		t.Error("nothing is evictable and 2 slots are free: LoadBack admitted a load of 4")
	}
	if got := r.alloc.AvailableSize(); got != 2 {
		t.Errorf("after a refused load %d device slots are free, want 2", got)
	}
}

// A backup that finds exactly the host room it needs evicts nothing; one that finds less evicts
// host leaves until it has the room, and goes through if they free exactly what it was short.
func TestExecuteKVBackupEvictsHostLeavesOnlyWhenShortAndAcceptsExactRoom(t *testing.T) {
	// Exact room: 8 host slots, a host-only leaf X of 4, a new node Y of 4 needing 4 of the 4 free.
	r := newTierRig(t, 16, 8, "write_back")
	x := r.insert(1, 2, 3, 4)
	if got := r.l2.ExecuteKVBackup([]cache.NodeID{x}, true); got != 4 {
		t.Fatalf("backup of X: %d host slots, want 4", got)
	}
	r.l2.WritingCheck(true)
	r.rc.Evict(cache.EvictParams{NumTokens: 4})
	if m := r.match(1, 2, 3, 4); m.HostHitLength != 4 {
		t.Fatalf("setup: X host hit %d, want 4", m.HostHitLength)
	}
	y := r.insert(5, 6, 7, 8)
	if got := r.host.AvailableSize(); got != 4 {
		t.Fatalf("setup: %d host slots free, want 4", got)
	}
	if got := r.l2.ExecuteKVBackup([]cache.NodeID{y}, true); got != 4 {
		t.Errorf("backup of Y into exactly the 4 free host slots: %d, want 4", got)
	}
	if m := r.match(1, 2, 3, 4); m.HostHitLength != 4 {
		t.Errorf("X's host copy was evicted although Y fitted: host hit %d, want 4", m.HostHitLength)
	}

	// Short by exactly what one host leaf frees: 4 host slots, X holds them, Y needs 4.
	r = newTierRig(t, 16, 4, "write_back")
	x = r.insert(1, 2, 3, 4)
	r.l2.ExecuteKVBackup([]cache.NodeID{x}, true)
	r.l2.WritingCheck(true)
	r.rc.Evict(cache.EvictParams{NumTokens: 4})
	y = r.insert(5, 6, 7, 8)
	if got := r.l2.ExecuteKVBackup([]cache.NodeID{y}, true); got != 4 {
		t.Errorf("backup of Y when evicting X's host copy frees exactly the 4 it is short of: %d, want 4", got)
	}
	if m := r.match(1, 2, 3, 4); m.HostHitLength != 0 {
		t.Errorf("X's host copy survived the eviction that made room: host hit %d, want 0", m.HostHitLength)
	}

	// Short by more than the host leaves can free: nothing is backed up.
	r = newTierRig(t, 16, 4, "write_back")
	x = r.insert(1, 2)
	r.l2.ExecuteKVBackup([]cache.NodeID{x}, true)
	r.l2.WritingCheck(true)
	r.rc.Evict(cache.EvictParams{NumTokens: 2})
	y = r.insert(5, 6, 7, 8, 9)
	if got := r.l2.ExecuteKVBackup([]cache.NodeID{y}, true); got != 0 {
		t.Errorf("backup of 5 tokens into a host of 4: %d, want 0", got)
	}
}

// A split node's pending backup is published by both halves: the old node's place in the
// publish list is taken by its replacements, in order, and the others keep theirs.
func TestReplacePendingWriteThroughNodeSwapsOnlyTheOldNode(t *testing.T) {
	r := newTierRig(t, 16, 16, "write_through")
	r.l2.ongoingWT[9] = &ongoingWriteThrough{node: 9, publish: []cache.NodeID{4, 5, 6}}
	r.l2.ReplacePendingWriteThroughNode(9, 5, []cache.NodeID{7, 8})
	if got := r.l2.ongoingWT[9].publish; !slices.Equal(got, []cache.NodeID{4, 7, 8, 6}) {
		t.Errorf("publish after replacing 5 by 7, 8: %v, want [4 7 8 6]", got)
	}
	// A node the ack does not publish changes nothing.
	r.l2.ReplacePendingWriteThroughNode(9, 99, []cache.NodeID{1})
	if got := r.l2.ongoingWT[9].publish; !slices.Equal(got, []cache.NodeID{4, 7, 8, 6}) {
		t.Errorf("publish after replacing an unlisted node: %v, want it unchanged", got)
	}
	// An ack with no pending backup is ignored, not dereferenced.
	func() {
		defer func() {
			if p := recover(); p != nil {
				t.Errorf("an unknown ack panicked: %v", p)
			}
		}()
		r.l2.ReplacePendingWriteThroughNode(77, 5, []cache.NodeID{1})
	}()
	if _, ok := r.l2.ongoingWT[77]; ok {
		t.Error("an unknown ack gained a pending backup")
	}
}

// init_load_back loads when the best match is device-evicted or the host hit is positive, and
// answers with the tokens loaded and the new anchor; otherwise nothing and the request's own anchor.
func TestInitLoadBackLoadsAnEvictedBestMatchOrAHostHit(t *testing.T) {
	evicted := func() (*tierRig, cache.MatchResult, *cache.Req) {
		r := newTierRig(t, 16, 16, "write_through")
		r.insert(1, 2, 3, 4)
		r.l2.WritingCheck(true)
		r.rc.Evict(cache.EvictParams{NumTokens: 4})
		r.l2.loadBackThreshold = 4
		m := r.match(1, 2, 3, 4)
		anchor := m.LastDeviceNode
		return r, m, &cache.Req{LastNode: &anchor}
	}
	// A host hit on an evicted node: loaded.
	r, m, req := evicted()
	n, node := r.l2.InitLoadBack(InitLoadBackParams{BestMatchNode: m.BestMatchNode, HostHitLength: m.HostHitLength, Req: req})
	if n != 4 || node != m.BestMatchNode {
		t.Errorf("host hit of 4 on an evicted node: loaded %d under %d, want 4 under %d", n, node, m.BestMatchNode)
	}
	// The node is device-evicted whatever the stated host hit: still loaded.
	r, m, req = evicted()
	if n, node = r.l2.InitLoadBack(InitLoadBackParams{BestMatchNode: m.BestMatchNode, HostHitLength: 0, Req: req}); n != 4 || node != m.BestMatchNode {
		t.Errorf("an evicted node with a host hit of 0: loaded %d under %d, want 4 under %d", n, node, m.BestMatchNode)
	}
	// A device-resident node with no host hit: nothing is loaded, the anchor stays, and no lock is left behind.
	r = newTierRig(t, 16, 16, "write_through")
	id := r.insert(1, 2, 3, 4)
	r.l2.WritingCheck(true)
	r.l2.loadBackThreshold = 1
	anchor := cache.NodeID(0)
	req = &cache.Req{LastNode: &anchor}
	before := [2]int{r.rc.EvictableSize(), r.rc.ProtectedSize()}
	if n, node = r.l2.InitLoadBack(InitLoadBackParams{BestMatchNode: id, HostHitLength: 0, Req: req}); n != 0 || node != anchor {
		t.Errorf("a resident node, no host hit: loaded %d under %d, want 0 under %d", n, node, anchor)
	}
	if after := [2]int{r.rc.EvictableSize(), r.rc.ProtectedSize()}; after != before {
		t.Errorf("evictable and protected %v after, %v before: a lock was left behind", after, before)
	}
}

// A host hit of zero on a device-resident node asks for no load: it takes no host lock, so the
// next host lock anyone takes is the first, as in a cache that was never asked.
func TestInitLoadBackWithNoHostHitTakesNoHostLock(t *testing.T) {
	build := func() (*tierRig, cache.NodeID, cache.NodeID) {
		r := newTierRig(t, 16, 16, "write_through")
		a := r.insert(1, 2, 3, 4)
		b := r.insert(9, 8)
		r.l2.WritingCheck(true) // both are backed up on host and resident on the device
		r.l2.loadBackThreshold = 1
		return r, a, b
	}
	control, _, cb := build()
	want := control.l2.IncHostLockRef(cb)
	if want.HostUUID == nil {
		t.Fatal("setup: a backed-up node's host lock carries no boundary uuid")
	}
	r, a, b := build()
	anchor := cache.NodeID(0)
	if n, node := r.l2.InitLoadBack(InitLoadBackParams{BestMatchNode: a, HostHitLength: 0, Req: &cache.Req{LastNode: &anchor}}); n != 0 || node != anchor {
		t.Fatalf("a resident node, no host hit: loaded %d under %d, want nothing under %d", n, node, anchor)
	}
	got := r.l2.IncHostLockRef(b)
	if got.HostUUID == nil || *got.HostUUID != *want.HostUUID {
		t.Errorf("the first host lock after the call has boundary uuid %v, want %d as in a cache never asked", got.HostUUID, *want.HostUUID)
	}
}
