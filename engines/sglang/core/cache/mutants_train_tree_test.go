// Original to blis-sglang.

package cache

import (
	"fmt"
	"slices"
	"testing"
)

// Behavioural tests of the tree core's host-tier paths and the radix cache's calls to its host tier
// (the mutation sweep of the M3.1 train): each fails on one generated mutant of treecore.go, radix.go
// or events.go that the golden traces do not reach. The package's own tests have no host tier, so
// tcTier is a synchronous one: it takes host slots from a counter, commits them to the tree, and
// completes every backup when the test settles it, as core/hicache's L2 does with a controller.

type tcPending struct {
	publish []NodeID
	lock    *LockReceipt
	node    NodeID
}

type tcReplace struct {
	ack, old NodeID
	repl     []NodeID
}

type tcTier struct {
	rc       *RadixCache
	nextHost int64
	pending  map[NodeID]*tcPending
	// forced, when set, is what ExecuteKVBackup answers without backing anything up.
	forced   *int
	backups  [][]NodeID
	checks   []bool
	freed    [][]int64
	replaced []tcReplace
}

func (h *tcTier) ExecuteKVBackup(nodes []NodeID, writeBack bool) int {
	h.backups = append(h.backups, slices.Clone(nodes))
	if h.forced != nil {
		return *h.forced
	}
	core := h.rc.Core()
	written := 0
	for _, id := range nodes {
		device := core.BuildBackupSpec(id)
		if len(device) == 0 {
			continue
		}
		host := make([]int64, len(device))
		for i := range host {
			h.nextHost++
			host[i] = h.nextHost
		}
		core.CommitBackup(id, host)
		var lock *LockReceipt
		if !writeBack {
			r := h.rc.IncLockRef(id)
			r.Delta = nil
			lock = &r
		}
		publish := core.MarkWriteThroughPending([]NodeID{id}, id)
		h.pending[id] = &tcPending{publish: publish, lock: lock, node: id}
		written = len(host)
	}
	return written
}

func (h *tcTier) WritingCheck(writeBack bool) {
	h.checks = append(h.checks, writeBack)
	h.settle()
}

// settle completes every backup in flight: the nodes are published and their locks released.
func (h *tcTier) settle() {
	ids := make([]NodeID, 0, len(h.pending))
	for id := range h.pending {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	for _, id := range ids {
		p := h.pending[id]
		delete(h.pending, id)
		h.rc.Core().FinishWriteThrough(p.publish, id)
		if p.lock != nil {
			h.rc.DecLockRef(p.node, *p.lock)
		}
	}
}

func (h *tcTier) FreeHostValues(values [][]int64) { h.freed = append(h.freed, values...) }

func (h *tcTier) ReplacePendingWriteThroughNode(ack, old NodeID, replacement []NodeID) {
	h.replaced = append(h.replaced, tcReplace{ack, old, slices.Clone(replacement)})
}

type tcRig struct {
	t     *testing.T
	rc    *RadixCache
	alloc *TokenAllocator
	tier  *tcTier
}

// newTCRig is a cache of pages of one token over deviceSlots device slots. With a host tier it is
// attached under write-through (threshold 1) or, with writeBack, write-back (threshold 2).
func newTCRig(t *testing.T, deviceSlots int, cfg RadixConfig, withTier, writeBack bool, threshold int) *tcRig {
	t.Helper()
	cfg.PageSize = 1
	r := &tcRig{t: t, alloc: NewTokenAllocator(deviceSlots, nil, false)}
	var err error
	if r.rc, err = NewRadixCache(cfg, r.alloc, NewRequestPool(8, 64)); err != nil {
		t.Fatal(err)
	}
	if withTier {
		r.tier = &tcTier{rc: r.rc, pending: map[NodeID]*tcPending{}}
		r.rc.AttachHostTier(r.tier, writeBack, threshold)
	}
	return r
}

func tkey(toks ...int64) Key { return Key{Tokens: toks} }

// insert puts the tokens in the tree on freshly allocated device slots and returns the node that
// ends them.
func (r *tcRig) insert(toks ...int64) NodeID {
	r.t.Helper()
	slots, ok := r.alloc.Alloc(len(toks))
	if !ok {
		r.t.Fatalf("no device room for %d tokens", len(toks))
	}
	return r.rc.Insert(InsertParams{Key: tkey(toks...), Value: slots}).LastDeviceNode
}

func (r *tcRig) match(toks ...int64) MatchResult {
	return r.rc.MatchPrefix(MatchPrefixParams{Key: tkey(toks...)})
}

// hostChain builds a path of host-only nodes under write-through: [1 2], [1 2 3 4], ... each added
// by insert, backed up, settled, and its device slots evicted. It returns the node ids, root side first.
func (r *tcRig) hostOnlyChain(lens ...int) []NodeID {
	r.t.Helper()
	var ids []NodeID
	var toks []int64
	next := int64(1)
	for _, l := range lens {
		for i := 0; i < l; i++ {
			toks = append(toks, next)
			next++
		}
		ids = append(ids, r.insert(toks...))
	}
	r.tier.settle()
	total := len(toks)
	if got := r.rc.Evict(EvictParams{NumTokens: total}).NumTokensEvicted; got != total {
		r.t.Fatalf("setup: %d tokens left the device, want %d", got, total)
	}
	return ids
}

// The match of a path that ends on host reports the host slots between the best match and the
// device anchor, and keeps the device anchor on the last node that holds device slots.
func TestMatchPrefixCountsOnlyTheHostSlotsBelowTheDeviceAnchor(t *testing.T) {
	r := newTCRig(t, 16, RadixConfig{}, true, false, 1)
	parent := r.insert(1, 2)
	child := r.insert(1, 2, 3, 4)
	r.tier.settle()
	// Both nodes are backed up; the child's device slots leave, the parent's stay.
	if got := r.rc.Evict(EvictParams{NumTokens: 2}).NumTokensEvicted; got != 2 {
		t.Fatalf("evicted %d tokens, want the child's 2", got)
	}
	m := r.match(1, 2, 3, 4)
	if m.DevicePrefixLen != 2 || m.LastDeviceNode != parent || m.BestMatchNode != child || m.LastHostNode != child || m.HostHitLength != 2 {
		t.Errorf("device prefix %d, device anchor %d, best %d, host anchor %d, host hit %d; want 2, %d, %d, %d, 2",
			m.DevicePrefixLen, m.LastDeviceNode, m.BestMatchNode, m.LastHostNode, m.HostHitLength, parent, child, child)
	}
	// A path that is host-only from its top down sums every node's host slots and anchors at the root.
	r = newTCRig(t, 16, RadixConfig{}, true, false, 1)
	ids := r.hostOnlyChain(2, 2, 2)
	m = r.match(1, 2, 3, 4, 5, 6)
	root := r.rc.Core().RootNodeHandle(nil)
	if m.DevicePrefixLen != 0 || m.LastDeviceNode != root || m.BestMatchNode != ids[2] || m.HostHitLength != 6 {
		t.Errorf("a host-only path of 6: device prefix %d, device anchor %d, best %d, host hit %d; want 0, root %d, %d, 6",
			m.DevicePrefixLen, m.LastDeviceNode, m.BestMatchNode, m.HostHitLength, root, ids[2])
	}
}

// Evicting host leaves one after another frees a chain from its end: each time a leaf goes, its
// parent, now childless, becomes the next host leaf. A host copy that is still there is kept, and
// the root is never examined.
func TestHostEvictionWalksUpAChainAndStopsAtTheRoot(t *testing.T) {
	r := newTCRig(t, 16, RadixConfig{}, true, false, 1)
	ids := r.hostOnlyChain(2, 2)
	core := r.rc.Core()
	step := core.DriveHostEviction(FullComponentType, 4)
	r.rc.ApplyEvictionStep(step)
	if got := step.Tracker[FullComponentType]; got != 4 {
		t.Errorf("host eviction freed %d slots from a chain of two nodes of 2, want 4", got)
	}
	if m := r.match(1, 2, 3, 4); m.BestMatchNode != core.RootNodeHandle(nil) || m.HostHitLength != 0 {
		t.Errorf("after the chain is gone the match is node %d with host hit %d, want the root with 0", m.BestMatchNode, m.HostHitLength)
	}
	_ = ids
	// Asking for fewer slots than the chain's end frees only that: the parent keeps its host copy.
	r = newTCRig(t, 16, RadixConfig{}, true, false, 1)
	ids = r.hostOnlyChain(2, 2)
	step = r.rc.Core().DriveHostEviction(FullComponentType, 2)
	r.rc.ApplyEvictionStep(step)
	if got := step.Tracker[FullComponentType]; got != 2 {
		t.Errorf("host eviction of 2 freed %d", got)
	}
	if m := r.match(1, 2, 3, 4); m.BestMatchNode != ids[0] || m.HostHitLength != 2 {
		t.Errorf("after the leaf is gone the match is node %d with host hit %d, want its parent %d with 2", m.BestMatchNode, m.HostHitLength, ids[0])
	}
	// The only node, under the root, goes without the root being examined for deletion.
	r = newTCRig(t, 16, RadixConfig{}, true, false, 1)
	r.hostOnlyChain(2)
	func() {
		defer func() {
			if p := recover(); p != nil {
				t.Errorf("evicting the only host node panicked: %v", p)
			}
		}()
		step = r.rc.Core().DriveHostEviction(FullComponentType, 2)
		r.rc.ApplyEvictionStep(step)
	}()
	if got := step.Tracker[FullComponentType]; got != 2 {
		t.Errorf("host eviction of the only node freed %d, want 2", got)
	}
	// A parent that holds a host lock is kept when its child goes, and stays a host leaf only once released.
	r = newTCRig(t, 16, RadixConfig{}, true, false, 1)
	ids = r.hostOnlyChain(2, 2)
	lock := r.rc.Core().IncHostLockRef(ids[0])
	step = r.rc.Core().DriveHostEviction(FullComponentType, 4)
	r.rc.ApplyEvictionStep(step)
	if got := step.Tracker[FullComponentType]; got != 2 {
		t.Errorf("with the parent host-locked, host eviction freed %d, want only the leaf's 2", got)
	}
	r.rc.Core().DecHostLockRef(ids[0], lock)
	step = r.rc.Core().DriveHostEviction(FullComponentType, 2)
	r.rc.ApplyEvictionStep(step)
	if got := step.Tracker[FullComponentType]; got != 2 {
		t.Errorf("after the host lock is released, host eviction freed %d, want the parent's 2", got)
	}
}

// ResumeInsert is for an insert that is in flight: with none it refuses, by name.
func TestResumeInsertRefusesWhenNoInsertIsInFlight(t *testing.T) {
	r := newTCRig(t, 8, RadixConfig{}, false, false, 0)
	func() {
		defer func() {
			if p := recover(); p != "cache: no in-flight insert" {
				t.Errorf("ResumeInsert with no insert in flight panicked with %v, want the named refusal", p)
			}
		}()
		r.rc.Core().ResumeInsert()
	}()
	// A walk that suspends at a backup is resumed, and finishes.
	r = newTCRig(t, 8, RadixConfig{}, true, false, 1)
	slots, _ := r.alloc.Alloc(2)
	core := r.rc.Core()
	step := core.BeginInsert(InsertParams{Key: tkey(1, 2), Value: slots})
	if step.Result == nil {
		if !core.HasOngoingInsert() {
			t.Fatal("an insert that returned no result is not in flight")
		}
		r.rc.ApplyActions(step.Actions)
		step = core.ResumeInsert()
	}
	if step.Result == nil {
		t.Fatal("the resumed insert returned no result")
	}
	r.rc.ApplyActions(step.Actions)
	r.rc.ApplyActions(core.EndInsert())
	if core.HasOngoingInsert() {
		t.Error("an insert is still in flight after its end")
	}
}

// A hit counts toward a write-through backup only on a resident node, only under write-through, and
// the backup is asked for when the count reaches the policy's threshold, not before and not again
// once the node is backed up.
func TestHitCountsDriveTheWriteThroughBackupAsThePolicySays(t *testing.T) {
	// Threshold 1: a new node is backed up on the insert that makes it.
	r := newTCRig(t, 16, RadixConfig{}, true, false, 1)
	r.insert(1, 2, 3, 4)
	if len(r.tier.backups) != 1 {
		t.Errorf("threshold 1: %d backups asked for after the first insert, want 1", len(r.tier.backups))
	}
	// A node that is already backed up is not asked for again, however many times it is hit.
	r.tier.settle()
	r.insert(1, 2, 3, 4)
	r.insert(1, 2, 3, 4)
	if len(r.tier.backups) != 1 {
		t.Errorf("a backed-up node was asked for again: %d backups, want 1", len(r.tier.backups))
	}

	// Threshold 2: the first hit only counts; the second asks.
	r = newTCRig(t, 16, RadixConfig{}, true, false, 2)
	r.insert(1, 2, 3, 4)
	if len(r.tier.backups) != 0 {
		t.Errorf("threshold 2: %d backups after the first insert, want none", len(r.tier.backups))
	}
	r.insert(1, 2, 3, 4)
	if len(r.tier.backups) != 1 {
		t.Errorf("threshold 2: %d backups after the second hit, want 1", len(r.tier.backups))
	}

	// Write-back counts nothing: hits never ask for a backup (eviction does).
	r = newTCRig(t, 16, RadixConfig{}, true, true, 2)
	for i := 0; i < 4; i++ {
		r.insert(1, 2, 3, 4)
	}
	if len(r.tier.backups) != 0 {
		t.Errorf("write-back: %d backups from hits, want none", len(r.tier.backups))
	}

	// A cache with no host tier never asks for a backup, whatever the hits.
	r = newTCRig(t, 16, RadixConfig{}, false, false, 0)
	func() {
		defer func() {
			if p := recover(); p != nil {
				t.Errorf("a device-only cache asked for a backup on a repeated insert: %v", p)
			}
		}()
		r.insert(1, 2, 3, 4)
		r.insert(1, 2, 3, 4)
		r.insert(1, 2, 3, 4)
	}()
}

// The device tokens an eviction frees count as unbacked only when the node had no host copy.
func TestEvictingAnUnbackedLeafReportsItsTokensAsUnbacked(t *testing.T) {
	r := newTCRig(t, 16, RadixConfig{}, true, false, 5) // threshold 5: nothing is backed up
	id := r.insert(1, 2, 3)
	core := r.rc.Core()
	core.EvictDeviceStart(FullComponentType, 3)
	defer core.EvictDeviceEnd(FullComponentType)
	step := core.EvictDeviceNextNode(FullComponentType, Tracker{FullComponentType: 0})
	if step.Node == nil || *step.Node != id {
		t.Fatalf("the walk chose %v, want node %d", step.Node, id)
	}
	leaf := core.EvictDeviceLeaf(id, false)
	if leaf.UnbackedTokens != 3 || leaf.Tracker[FullComponentType] != 3 {
		t.Errorf("an unbacked leaf of 3 evicted: unbacked %d, freed %d; want 3 and 3", leaf.UnbackedTokens, leaf.Tracker[FullComponentType])
	}
}

// ---- the radix cache's calls to its host tier ----

func TestApplyActionsRoutesASplitToTheTierOnlyWhenThereIsAnAckAndATier(t *testing.T) {
	ack := NodeID(9)
	split := CacheAction{Kind: UpdatePendingBackupOnNodeSplitAction, AckID: &ack, OldNode: 5, NewNode: 6, NewChild: 7}
	r := newTCRig(t, 8, RadixConfig{}, true, false, 1)
	r.rc.ApplyActions([]CacheAction{split})
	if len(r.tier.replaced) != 1 || r.tier.replaced[0].ack != 9 || r.tier.replaced[0].old != 5 || !slices.Equal(r.tier.replaced[0].repl, []NodeID{6, 7}) {
		t.Errorf("a split with an ack: tier told %+v, want ack 9 replacing 5 by [6 7]", r.tier.replaced)
	}
	// No ack: nothing to tell, and no dereference.
	noAck := split
	noAck.AckID = nil
	func() {
		defer func() {
			if p := recover(); p != nil {
				t.Errorf("a split with no ack panicked: %v", p)
			}
		}()
		r.rc.ApplyActions([]CacheAction{noAck})
	}()
	if len(r.tier.replaced) != 1 {
		t.Errorf("a split with no ack told the tier: %+v", r.tier.replaced)
	}
	// No tier: the split is dropped without a panic.
	d := newTCRig(t, 8, RadixConfig{}, false, false, 0)
	func() {
		defer func() {
			if p := recover(); p != nil {
				t.Errorf("a split with an ack and no tier panicked: %v", p)
			}
		}()
		d.rc.ApplyActions([]CacheAction{split})
	}()
}

func TestApplyActionsBacksUpThroughTheTierAndRefusesWithoutOne(t *testing.T) {
	act := CacheAction{Kind: BackupKVAction, NodeIDs: []NodeID{3, 4}}
	r := newTCRig(t, 8, RadixConfig{}, true, false, 1)
	forced := 0
	r.tier.forced = &forced
	r.rc.ApplyActions([]CacheAction{act})
	if len(r.tier.backups) != 1 || !slices.Equal(r.tier.backups[0], []NodeID{3, 4}) {
		t.Errorf("a backup action: tier asked for %v, want [[3 4]]", r.tier.backups)
	}
	d := newTCRig(t, 8, RadixConfig{}, false, false, 0)
	func() {
		defer func() {
			if p := recover(); p != "cache: a backup action with no host tier" {
				t.Errorf("a backup action with no tier panicked with %v, want the named refusal", p)
			}
		}()
		d.rc.ApplyActions([]CacheAction{act})
	}()
}

func TestApplyEvictionStepFreesHostSlotsThroughTheTierAndRefusesWithoutOne(t *testing.T) {
	step := newStep()
	step.HostFrees[FullComponentType] = [][]int64{{1, 2}, {3}}
	r := newTCRig(t, 8, RadixConfig{}, true, false, 1)
	r.rc.ApplyEvictionStep(step)
	if fmt.Sprint(r.tier.freed) != "[[1 2] [3]]" {
		t.Errorf("host frees: tier freed %v, want [[1 2] [3]]", r.tier.freed)
	}
	d := newTCRig(t, 8, RadixConfig{}, false, false, 0)
	func() {
		defer func() {
			if p := recover(); p != "cache: host slots freed with no host tier" {
				t.Errorf("host frees with no tier panicked with %v, want the named refusal", p)
			}
		}()
		d.rc.ApplyEvictionStep(step)
	}()
	// A step that frees no host slots needs no tier.
	empty := newStep()
	d.rc.ApplyEvictionStep(empty)
}

// An eviction ends with the blocking write check only under write-back with a host tier.
func TestEvictionRunsTheBlockingWriteCheckOnlyUnderWriteBackWithATier(t *testing.T) {
	for _, tc := range []struct {
		name      string
		writeBack bool
		want      int
	}{{"write-through", false, 0}, {"write-back", true, 1}} {
		r := newTCRig(t, 8, RadixConfig{}, true, tc.writeBack, 2)
		r.rc.Evict(EvictParams{NumTokens: 1})
		if len(r.tier.checks) != tc.want || (tc.want == 1 && !r.tier.checks[0]) {
			t.Errorf("%s: blocking write checks %v, want %d (blocking)", tc.name, r.tier.checks, tc.want)
		}
	}
	// A tree configured for write-back with no tier attached evicts without calling one.
	r := newTCRig(t, 8, RadixConfig{}, false, false, 0)
	r.rc.tree.ConfigureWritePolicy(true, 2)
	func() {
		defer func() {
			if p := recover(); p != nil {
				t.Errorf("write-back with no tier panicked in eviction: %v", p)
			}
		}()
		r.rc.Evict(EvictParams{NumTokens: 1})
	}()
}

// Under write-back, a leaf evicted before it was backed up is backed up first; when the backup
// writes nothing, the leaf is dropped instead of demoted.
func TestWriteBackEvictionDropsALeafWhoseBackupWroteNothing(t *testing.T) {
	r := newTCRig(t, 8, RadixConfig{}, true, true, 2)
	leaf := r.insert(1, 2, 3, 4)
	zero := 0
	r.tier.forced = &zero
	got := r.rc.Evict(EvictParams{NumTokens: 4}).NumTokensEvicted
	if got != 4 {
		t.Errorf("evicted %d tokens, want the leaf's 4", got)
	}
	if len(r.tier.backups) != 1 || !slices.Equal(r.tier.backups[0], []NodeID{leaf}) {
		t.Errorf("backups asked for: %v, want [[%d]]", r.tier.backups, leaf)
	}
	if m := r.match(1, 2, 3, 4); m.BestMatchNode != r.rc.Core().RootNodeHandle(nil) || m.HostHitLength != 0 || m.DevicePrefixLen != 0 {
		t.Errorf("the dropped leaf still matches: best %d, host hit %d, device prefix %d", m.BestMatchNode, m.HostHitLength, m.DevicePrefixLen)
	}
	if r.alloc.AvailableSize() != 8 {
		t.Errorf("%d device slots free after the drop, want all 8", r.alloc.AvailableSize())
	}
	// The same leaf with a backup that works is demoted: it stays matchable on host.
	r = newTCRig(t, 8, RadixConfig{}, true, true, 2)
	id := r.insert(1, 2, 3, 4)
	r.rc.Evict(EvictParams{NumTokens: 4})
	if m := r.match(1, 2, 3, 4); m.BestMatchNode != id || m.HostHitLength != 4 {
		t.Errorf("a leaf backed up by its eviction: best %d, host hit %d; want %d and 4", m.BestMatchNode, m.HostHitLength, id)
	}
}
