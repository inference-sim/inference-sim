// Original to blis-sglang.

package cache

import (
	"slices"
	"testing"
)

// These tests pin the host tier (HiCache) of the tree core through its public methods: what a
// lock receipt holds, which nodes a backup, a load-back or an eviction touches, and in what order.
// Each asserts a result the tree reports; the mutation sweep (oracle/mutate.py) checks that a
// one-token break of host.go changes one of them.

// hostRig is a tree core with HiCache on, built node by node.
type hostRig struct {
	t    *testing.T
	tc   *treeCore
	slot int64 // next device slot
	hslt int64 // next host slot
	ack  NodeID
}

func newHostRig(t *testing.T, writeBack bool) *hostRig {
	tc := newTreeCore(1, false, false, false)
	tc.SetHicacheEnabled()
	tc.ConfigureWritePolicy(writeBack, 256)
	return &hostRig{t: t, tc: tc, slot: 1, hslt: 1_000_000, ack: 100}
}

// add inserts the path from the root and returns the node holding its last tokens.
func (r *hostRig) add(path ...int64) NodeID {
	val := make([]int64, len(path))
	for i := range val {
		val[i] = r.slot
		r.slot++
	}
	s := r.tc.BeginInsert(InsertParams{Key: Key{Tokens: path}, Value: val})
	for s.Result == nil {
		s = r.tc.ResumeInsert()
	}
	return s.Result.LastDeviceNode
}

// backup copies a node to host and returns its host slots.
func (r *hostRig) backup(id NodeID) []int64 {
	n := r.tc.node(id)
	h := make([]int64, len(n.value))
	for i := range h {
		h[i] = r.hslt
		r.hslt++
	}
	r.tc.CommitBackup(id, h)
	return h
}

// both leaves the node on both tiers and settled: a duplicate.
func (r *hostRig) both(id NodeID) []int64 {
	h := r.backup(id)
	r.ack++
	r.tc.MarkWriteThroughPending([]NodeID{id}, r.ack)
	r.tc.FinishWriteThrough([]NodeID{id}, r.ack)
	return h
}

// hostOnly leaves the node on host alone.
func (r *hostRig) hostOnly(id NodeID) []int64 {
	h := r.backup(id)
	r.tc.Demote(id)
	return h
}

func expectPanic(t *testing.T, what string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: no panic", what)
		}
	}()
	f()
}

func expectNoPanic(t *testing.T, what string, f func()) {
	t.Helper()
	defer func() {
		if e := recover(); e != nil {
			t.Errorf("%s: panicked: %v", what, e)
		}
	}()
	f()
}

func freed(s EvictionStep) [][]int64 { return s.HostFrees[FullComponentType] }

// --- host lock ---

func TestHostLockNeedsAHostCopyUnlessWriteBack(t *testing.T) {
	for _, c := range []struct {
		name      string
		writeBack bool
		backed    bool
		locks     bool
	}{
		{"write-through, no host copy", false, false, false},
		{"write-through, host copy", false, true, true},
		{"write-back, no host copy", true, false, true},
		{"write-back, host copy", true, true, true},
	} {
		r := newHostRig(t, c.writeBack)
		id := r.add(1, 2)
		if c.backed {
			r.backup(id)
		}
		rc := r.tc.IncHostLockRef(id)
		if (rc.HostUUID != nil) != c.locks || (r.tc.node(id).hostLockRef == 1) != c.locks {
			t.Errorf("%s: receipt uuid %v, host_lock_ref %d; want a lock: %v", c.name, rc.HostUUID, r.tc.node(id).hostLockRef, c.locks)
		}
		if rc.Node != id {
			t.Errorf("%s: receipt anchored on %d, want %d", c.name, rc.Node, id)
		}
	}
}

func TestHostLockUUIDsAreFullsRangeAndStayWithTheNode(t *testing.T) {
	r := newHostRig(t, false)
	a, b := r.add(1, 2), r.add(3, 4)
	r.backup(a)
	r.backup(b)
	ra := r.tc.IncHostLockRef(a)
	if *ra.HostUUID != fullUUIDBase+1 {
		t.Errorf("first uuid %d, want %d", *ra.HostUUID, fullUUIDBase+1)
	}
	again := r.tc.IncHostLockRef(a)
	if *again.HostUUID != *ra.HostUUID || r.tc.node(a).hostLockRef != 2 {
		t.Errorf("a second lock on the node: uuid %d (was %d), ref %d", *again.HostUUID, *ra.HostUUID, r.tc.node(a).hostLockRef)
	}
	rb := r.tc.IncHostLockRef(b)
	if *rb.HostUUID != fullUUIDBase+2 {
		t.Errorf("the next node's uuid %d, want %d (the counter moved once)", *rb.HostUUID, fullUUIDBase+2)
	}
}

func TestResetKeepsTheProcessWideUUIDCounterAndPolicy(t *testing.T) {
	r := newHostRig(t, false)
	a := r.add(1, 2)
	r.backup(a)
	first := *r.tc.IncHostLockRef(a).HostUUID
	r.tc.Reset()
	b := r.add(1, 2)
	r.backup(b)
	if got := *r.tc.IncHostLockRef(b).HostUUID; got != first+1 {
		t.Errorf("uuid after reset %d, want %d: SGLang's counter is the process's", got, first+1)
	}
	// A configured threshold survives a reset; a fresh tree starts at 256.
	r2 := newHostRig(t, false)
	r2.tc.ConfigureWritePolicy(false, 5)
	r2.tc.Reset()
	if n := insertsUntilBackup(r2); n != 5 {
		t.Errorf("backup due after %d inserts of one key following a reset, want the configured 5", n)
	}
}

func TestFreshTreeBacksUpAfter256Hits(t *testing.T) {
	r := newHostRig(t, false)
	r.tc.ConfigureWritePolicy(false, 0) // what a tree is before init_hicache sets the policy
	r.tc.Reset()
	if n := insertsUntilBackup(r); n != 256 {
		t.Errorf("backup due after %d inserts, want the default threshold 256", n)
	}
	r2 := newTreeCore(1, false, false, false)
	r2.SetHicacheEnabled()
	rr := &hostRig{t: t, tc: r2, slot: 1}
	if n := insertsUntilBackup(rr); n != 256 {
		t.Errorf("a new tree: backup due after %d inserts, want 256", n)
	}
	r3 := newTreeCore(1, false, false, false)
	if got := *func() *int64 {
		r3.SetHicacheEnabled()
		r3.ConfigureWritePolicy(false, 256)
		id := r3.BeginInsert(InsertParams{Key: Key{Tokens: []int64{1, 2}}, Value: []int64{1, 2}}).Result.LastDeviceNode
		r3.CommitBackup(id, []int64{9, 9})
		return r3.IncHostLockRef(id).HostUUID
	}(); got != fullUUIDBase+1 {
		t.Errorf("a new tree's first host-lock uuid %d, want %d", got, fullUUIDBase+1)
	}
}

// insertsUntilBackup inserts one key until the tree asks for it to be backed up.
func insertsUntilBackup(r *hostRig) int {
	for i := 1; i <= 1000; i++ {
		val := []int64{r.slot, r.slot + 1}
		r.slot += 2
		s := r.tc.BeginInsert(InsertParams{Key: Key{Tokens: []int64{7, 8}}, Value: val})
		for {
			for _, a := range s.Actions {
				if a.Kind == BackupKVAction {
					if s.Result == nil {
						r.tc.EndInsert()
					}
					return i
				}
			}
			if s.Result != nil {
				break
			}
			s = r.tc.ResumeInsert()
		}
	}
	return -1
}

func TestHostLockReleaseWalksToTheReceiptsBoundary(t *testing.T) {
	r := newHostRig(t, false)
	p := r.add(1, 2)
	c := r.add(1, 2, 3, 4)
	r.backup(p)
	r.backup(c)
	rp := r.tc.IncHostLockRef(p)
	rc := r.tc.IncHostLockRef(c)
	// A receipt that names the parent's boundary releases the child and then the parent.
	r.tc.DecHostLockRef(c, LockReceipt{Node: c, HostUUID: rp.HostUUID})
	if r.tc.node(c).hostLockRef != 0 || r.tc.node(p).hostLockRef != 0 {
		t.Errorf("host_lock_ref child %d, parent %d, want both released", r.tc.node(c).hostLockRef, r.tc.node(p).hostLockRef)
	}
	// Its own receipt releases only the child.
	r.tc.IncHostLockRef(p)
	r.tc.IncHostLockRef(c)
	r.tc.DecHostLockRef(c, LockReceipt{Node: c, HostUUID: rc.HostUUID})
	if r.tc.node(c).hostLockRef != 0 || r.tc.node(p).hostLockRef != 1 {
		t.Errorf("host_lock_ref child %d, parent %d, want 0 and 1", r.tc.node(c).hostLockRef, r.tc.node(p).hostLockRef)
	}
}

func TestHostLockReleaseWithoutAUUIDChangesNothing(t *testing.T) {
	r := newHostRig(t, false)
	id := r.add(1, 2)
	r.backup(id)
	r.tc.IncHostLockRef(id)
	r.tc.DecHostLockRef(id, LockReceipt{Node: id})
	if r.tc.node(id).hostLockRef != 1 {
		t.Errorf("host_lock_ref %d after a release that took no lock, want 1", r.tc.node(id).hostLockRef)
	}
}

func TestHostLockReleaseRefusals(t *testing.T) {
	r := newHostRig(t, false)
	a, b := r.add(1, 2), r.add(3, 4)
	r.backup(a)
	r.backup(b)
	ra := r.tc.IncHostLockRef(a)
	expectPanic(t, "a receipt released on another node", func() { r.tc.DecHostLockRef(b, ra) })
	r.tc.DecHostLockRef(a, ra)
	expectPanic(t, "a release below zero", func() { r.tc.DecHostLockRef(a, ra) })
	// A boundary no ancestor holds is refused at the root of the walk.
	w := newHostRig(t, true)
	x := w.add(1, 2)
	w.tc.IncHostLockRef(x)
	w.tc.IncHostLockRef(w.tc.root.id)
	bogus := int64(5)
	expectPanic(t, "a boundary that is not an ancestor", func() { w.tc.DecHostLockRef(x, LockReceipt{Node: x, HostUUID: &bogus}) })
}

func TestHostLockReleaseStopsAtANodeWithoutHostCopy(t *testing.T) {
	r := newHostRig(t, false)
	p := r.add(1, 2) // never backed up
	c := r.add(1, 2, 3, 4)
	r.backup(c)
	rc := r.tc.IncHostLockRef(c)
	other := int64(1)
	expectNoPanic(t, "a walk that meets a node with no host copy", func() {
		r.tc.DecHostLockRef(c, LockReceipt{Node: c, HostUUID: &other})
	})
	if r.tc.node(c).hostLockRef != 0 || r.tc.node(p).hostLockRef != 0 {
		t.Errorf("host_lock_ref child %d parent %d, want 0 and 0", r.tc.node(c).hostLockRef, r.tc.node(p).hostLockRef)
	}
	_ = rc
	// Under write-back the walk goes past a node without a host copy.
	w := newHostRig(t, true)
	wp := w.add(1, 2)
	wc := w.add(1, 2, 3, 4)
	rp := w.tc.IncHostLockRef(wp)
	w.tc.IncHostLockRef(wc)
	w.tc.DecHostLockRef(wc, LockReceipt{Node: wc, HostUUID: rp.HostUUID})
	if w.tc.node(wc).hostLockRef != 0 || w.tc.node(wp).hostLockRef != 0 {
		t.Errorf("write-back: host_lock_ref child %d parent %d, want both released", w.tc.node(wc).hostLockRef, w.tc.node(wp).hostLockRef)
	}
}

// --- backup ---

func TestBackupChainRunsFromTheShallowestUnbackedAncestor(t *testing.T) {
	r := newHostRig(t, false)
	a := r.add(1, 2)
	b := r.add(1, 2, 3, 4)
	c := r.add(1, 2, 3, 4, 5, 6)
	if got := r.tc.backupKVAction(r.tc.node(c), false).NodeIDs; !slices.Equal(got, []NodeID{a, b, c}) {
		t.Errorf("chain %v, want the unbacked path %v, ancestors first", got, []NodeID{a, b, c})
	}
	r.backup(a)
	if got := r.tc.backupKVAction(r.tc.node(c), false).NodeIDs; !slices.Equal(got, []NodeID{b, c}) {
		t.Errorf("chain %v, want %v: a backed-up ancestor ends it", got, []NodeID{b, c})
	}
	if got := r.tc.backupKVAction(r.tc.node(c), true).NodeIDs; !slices.Equal(got, []NodeID{c}) {
		t.Errorf("write-back chain %v, want the node alone", got)
	}
	if got := r.tc.backupKVAction(r.tc.node(a), false).NodeIDs; !slices.Equal(got, []NodeID{a}) {
		t.Errorf("chain of a child of the root %v, want the node alone", got)
	}
}

func TestBackupSpecAndCommit(t *testing.T) {
	r := newHostRig(t, false)
	id := r.add(1, 2, 3)
	spec := r.tc.BuildBackupSpec(id)
	if !slices.Equal(spec, r.tc.node(id).value) || len(spec) != 3 {
		t.Errorf("spec %v, want the node's device slots %v", spec, r.tc.node(id).value)
	}
	r.tc.CommitBackup(id, []int64{})
	if r.tc.node(id).backuped() {
		t.Errorf("a commit of no slots made the node backed up")
	}
	h := r.backup(id)
	if got := r.tc.node(id).hostValue; !slices.Equal(got, h) {
		t.Errorf("host slots %v, want %v", got, h)
	}
	if spec := r.tc.BuildBackupSpec(id); spec == nil || len(spec) != 0 {
		t.Errorf("spec of a node already on host %v, want empty and non-nil", spec)
	}
	r.tc.Demote(id)
	expectPanic(t, "a backup of a node with no device value", func() { r.tc.BuildBackupSpec(id) })
}

func TestWriteThroughPendingOrderAndAck(t *testing.T) {
	r := newHostRig(t, false)
	a := r.add(1, 2)
	b := r.add(1, 2, 3, 4)
	c := r.add(1, 2, 3, 4, 5, 6)
	d := r.add(9, 9)
	e := r.add(8, 8)
	// Shallowest first; equal depths by id.
	got := r.tc.MarkWriteThroughPending([]NodeID{c, e, b, d, a}, 7)
	want := []NodeID{a, d, e, b, c}
	if !slices.Equal(got, want) {
		t.Errorf("order %v, want %v", got, want)
	}
	for _, id := range want {
		if n := r.tc.node(id); n.writeThroughAck == nil || *n.writeThroughAck != 7 {
			t.Errorf("node %d ack %v, want 7", id, n.writeThroughAck)
		}
	}
	expectNoPanic(t, "the same ack twice", func() { r.tc.MarkWriteThroughPending([]NodeID{a}, 7) })
	expectPanic(t, "another ack on a pending node", func() { r.tc.MarkWriteThroughPending([]NodeID{a}, 8) })
}

func TestFinishWriteThroughClearsOnlyItsOwnAck(t *testing.T) {
	r := newHostRig(t, false)
	a, b := r.add(1, 2), r.add(3, 4)
	r.backup(a)
	r.backup(b)
	r.tc.MarkWriteThroughPending([]NodeID{a}, 7)
	r.tc.MarkWriteThroughPending([]NodeID{b}, 8)
	r.tc.FinishWriteThrough([]NodeID{a, b}, 7)
	if r.tc.node(a).writeThroughAck != nil {
		t.Errorf("node a still pending under its finished ack")
	}
	if n := r.tc.node(b); n.writeThroughAck == nil || *n.writeThroughAck != 8 {
		t.Errorf("node b ack %v, want 8: another ack's finish must not clear it", n.writeThroughAck)
	}
	// A finished node is a duplicate (reclaimable under write-back); one still pending is not.
	r.tc.ConfigureWritePolicy(true, 256)
	got := freed(r.tc.DriveHostEviction(FullComponentType, 100))
	if len(got) != 1 || !slices.Equal(got[0], r.tc.node(a).hostValue) && r.tc.node(a).hostValue != nil {
		t.Errorf("freed %v, want only node a's host copy", got)
	}
	if r.tc.node(b).hostValue == nil {
		t.Errorf("node b, still pending, lost its host copy")
	}
}

// --- load back ---

// evictedPath returns a chain a -> b -> c, all on host only, and their host slots.
func evictedPath(r *hostRig) (a, b, c NodeID, ha, hb, hc []int64) {
	a = r.add(1, 2)
	b = r.add(1, 2, 3, 4)
	c = r.add(1, 2, 3, 4, 5, 6)
	hc, hb, ha = r.hostOnly(c), r.hostOnly(b), r.hostOnly(a)
	return
}

func TestLoadBackSpecIsTheEvictedPathRootFirst(t *testing.T) {
	r := newHostRig(t, false)
	a, b, c, ha, hb, hc := evictedPath(r)
	spec := r.tc.BuildLoadBackSpec(c)
	if !slices.Equal(spec.Nodes, []NodeID{a, b, c}) || !slices.Equal(spec.Host, slices.Concat(ha, hb, hc)) {
		t.Errorf("spec %+v, want nodes %v and host %v", spec, []NodeID{a, b, c}, slices.Concat(ha, hb, hc))
	}
	spec = r.tc.BuildLoadBackSpec(b)
	if !slices.Equal(spec.Nodes, []NodeID{a, b}) {
		t.Errorf("spec nodes %v, want %v", spec.Nodes, []NodeID{a, b})
	}
	r.tc.node(b).hostValue = nil
	expectPanic(t, "a path node with no host value", func() { r.tc.BuildLoadBackSpec(c) })
}

func TestLoadBackSpecYieldsToAnotherLoadBack(t *testing.T) {
	r := newHostRig(t, true)
	_, b, c, _, _, _ := evictedPath(r)
	spec := r.tc.BuildLoadBackSpec(b)
	r.tc.CommitLoadBack(b, nil, spec) // pins a and b under write-back
	if got := r.tc.BuildLoadBackSpec(c); len(got.Nodes) != 0 || len(got.Host) != 0 || got.Host == nil || got.Nodes == nil {
		t.Errorf("spec while another load-back holds the path: %+v, want empty and non-nil", got)
	}
	if got := r.tc.BuildLoadBackSpec(b); len(got.Nodes) != 2 {
		t.Errorf("spec of the pinning load-back itself: %+v, want its own two nodes", got)
	}
	r.tc.FinishLoadBack(b)
	if got := r.tc.BuildLoadBackSpec(c); len(got.Nodes) != 3 {
		t.Errorf("spec after the pin is cleared: %+v, want three nodes", got)
	}
}

func TestLoadBackPinsOnlyUnderWriteBack(t *testing.T) {
	for _, writeBack := range []bool{false, true} {
		r := newHostRig(t, writeBack)
		_, b, c, _, _, _ := evictedPath(r)
		r.tc.CommitLoadBack(b, nil, r.tc.BuildLoadBackSpec(b))
		pinned := len(r.tc.BuildLoadBackSpec(c).Nodes) == 0
		if pinned != writeBack {
			t.Errorf("write-back %v: path pinned %v", writeBack, pinned)
		}
	}
	r := newHostRig(t, true)
	_, b, c, _, _, _ := evictedPath(r)
	specB := r.tc.BuildLoadBackSpec(b)
	r.tc.CommitLoadBack(c, nil, r.tc.BuildLoadBackSpec(c))
	expectPanic(t, "a second load-back over a pinned node", func() { r.tc.CommitLoadBack(b, nil, specB) })
	expectNoPanic(t, "the same load-back again", func() { r.tc.CommitLoadBack(c, nil, LoadBackSpec{Nodes: []NodeID{c}}) })
}

func TestCommitLoadBackFillsDeviceSlots(t *testing.T) {
	r := newHostRig(t, false)
	a, b, _, ha, hb, _ := evictedPath(r)
	evBefore, prBefore := r.tc.EvictableSize(), r.tc.ProtectedSize()
	spec := r.tc.BuildLoadBackSpec(b)
	dev := []int64{901, 902, 903, 904}
	r.tc.CommitLoadBack(b, dev, spec)
	if !slices.Equal(r.tc.node(a).value, []int64{901, 902}) || !slices.Equal(r.tc.node(b).value, []int64{903, 904}) {
		t.Errorf("device slots a %v b %v", r.tc.node(a).value, r.tc.node(b).value)
	}
	if got := r.tc.EvictableSize() - evBefore; got != len(ha)+len(hb) {
		t.Errorf("evictable grew by %d, want %d", got, len(ha)+len(hb))
	}
	if r.tc.ProtectedSize() != prBefore {
		t.Errorf("protected moved to %d from %d", r.tc.ProtectedSize(), prBefore)
	}
	// A device-less commit changes no slots.
	r2 := newHostRig(t, false)
	_, b2, _, _, _, _ := evictedPath(r2)
	r2.tc.CommitLoadBack(b2, nil, r2.tc.BuildLoadBackSpec(b2))
	if r2.tc.node(b2).value != nil {
		t.Errorf("a commit with no device slots filled %v", r2.tc.node(b2).value)
	}
}

func TestCommitLoadBackOfALockedNodeIsProtected(t *testing.T) {
	r := newHostRig(t, false)
	_, b, c, _, _, _ := evictedPath(r)
	r.tc.IncLockRef(b) // a request holds the evicted path to b
	spec := r.tc.BuildLoadBackSpec(b)
	ev, pr := r.tc.EvictableSize(), r.tc.ProtectedSize()
	r.tc.CommitLoadBack(b, []int64{901, 902, 903, 904}, spec)
	if r.tc.ProtectedSize()-pr != 4 || r.tc.EvictableSize() != ev {
		t.Errorf("protected +%d evictable +%d, want the locked path protected", r.tc.ProtectedSize()-pr, r.tc.EvictableSize()-ev)
	}
	_ = c
	// An unlocked node that follows joins the evictable.
	r2 := newHostRig(t, false)
	_, b2, _, _, _, _ := evictedPath(r2)
	ev, pr = r2.tc.EvictableSize(), r2.tc.ProtectedSize()
	r2.tc.CommitLoadBack(b2, []int64{901, 902, 903, 904}, r2.tc.BuildLoadBackSpec(b2))
	if r2.tc.EvictableSize()-ev != 4 || r2.tc.ProtectedSize() != pr {
		t.Errorf("unlocked: evictable +%d protected +%d", r2.tc.EvictableSize()-ev, r2.tc.ProtectedSize()-pr)
	}
}

func TestFinishLoadBackClearsItsPinsAndTracksDuplicates(t *testing.T) {
	r := newHostRig(t, true)
	a, b, c, _, _, _ := evictedPath(r)
	other := r.tc.BuildLoadBackSpec(c)
	r.tc.CommitLoadBack(c, []int64{901, 902, 903, 904, 905, 906}, other)
	// Another anchor's pin on a path node is left alone.
	foreign := NodeID(999)
	r.tc.node(a).loadBackAnchor = &foreign
	r.tc.FinishLoadBack(c)
	if r.tc.node(b).loadBackAnchor != nil || r.tc.node(c).loadBackAnchor != nil {
		t.Errorf("pins left on b %v c %v", r.tc.node(b).loadBackAnchor, r.tc.node(c).loadBackAnchor)
	}
	if r.tc.node(a).loadBackAnchor == nil {
		t.Errorf("a pin of another load-back was cleared")
	}
	// b and c are now on both tiers and settled: reclaimable duplicates. a is not: it is pinned.
	got := freed(r.tc.DriveHostEviction(FullComponentType, 1000))
	if len(got) != 2 {
		t.Errorf("reclaimed %d host copies, want b and c: %v", len(got), got)
	}
	if r.tc.node(a).hostValue == nil {
		t.Errorf("pinned node a lost its host copy")
	}
}

func TestFinishLoadBackUnderWriteThroughTracksWithoutClearing(t *testing.T) {
	r := newHostRig(t, false)
	_, b, c, _, _, _ := evictedPath(r)
	r.tc.CommitLoadBack(c, []int64{901, 902, 903, 904, 905, 906}, r.tc.BuildLoadBackSpec(c))
	pin := b
	r.tc.node(b).loadBackAnchor = &pin
	r.tc.FinishLoadBack(c)
	if r.tc.node(b).loadBackAnchor == nil {
		t.Errorf("write-through cleared a pin")
	}
	r.tc.ConfigureWritePolicy(true, 256)
	freedNow := freed(r.tc.DriveHostEviction(FullComponentType, 1000))
	hosts := 0
	for _, f := range freedNow {
		hosts += len(f)
	}
	if hosts != 4 { // a and c; b is pinned
		t.Errorf("reclaimed %d host slots, want 4 (a and c, not the pinned b)", hosts)
	}
}

// --- demote ---

func TestDemoteNeedsBothTiers(t *testing.T) {
	r := newHostRig(t, false)
	unbacked := r.add(1, 2)
	expectPanic(t, "demote of a node with no host copy", func() { r.tc.Demote(unbacked) })
	id := r.add(3, 4)
	h := r.backup(id)
	dev := slices.Clone(r.tc.node(id).value)
	step := r.tc.Demote(id)
	if r.tc.node(id).value != nil || !slices.Equal(r.tc.node(id).hostValue, h) {
		t.Errorf("after demote: device %v host %v", r.tc.node(id).value, r.tc.node(id).hostValue)
	}
	if got := step.DeviceFrees[FullComponentType]; len(got) != 1 || !slices.Equal(got[0], dev) {
		t.Errorf("freed device slots %v, want %v", got, dev)
	}
	expectPanic(t, "demote of a node already off device", func() { r.tc.Demote(id) })
}

// --- drop subtree ---

// dropRig builds n -> c -> g with c and g on host only, under write-back.
func dropRig(t *testing.T) (r *hostRig, n, c, g NodeID) {
	r = newHostRig(t, true)
	n = r.add(1, 2)
	c = r.add(1, 2, 3, 4)
	g = r.add(1, 2, 3, 4, 5, 6)
	r.hostOnly(g)
	r.hostOnly(c)
	return
}

func TestDropSubtreeDeletesTheHostOnlySubtree(t *testing.T) {
	r, n, c, g := dropRig(t)
	hc, hg := slices.Clone(r.tc.node(c).hostValue), slices.Clone(r.tc.node(g).hostValue)
	step := r.tc.DropSubtreeNoHost(n)
	if !step.IsDropped {
		t.Fatalf("subtree not dropped")
	}
	for _, id := range []NodeID{n, c, g} {
		if _, ok := r.tc.nodes[id]; ok {
			t.Errorf("node %d still in the tree", id)
		}
	}
	if got := freed(step); len(got) != 2 || !slices.Equal(got[0], hg) || !slices.Equal(got[1], hc) {
		t.Errorf("host frees %v, want the deepest first: %v then %v", got, hg, hc)
	}
	if got := step.DeviceFrees[FullComponentType]; len(got) != 1 {
		t.Errorf("device frees %v, want the leaf's slots", got)
	}
}

func TestDropSubtreeOfASingleChild(t *testing.T) {
	r := newHostRig(t, true)
	n := r.add(1, 2)
	c := r.add(1, 2, 3, 4)
	r.hostOnly(c)
	if step := r.tc.DropSubtreeNoHost(n); !step.IsDropped {
		t.Fatalf("not dropped")
	}
	if _, ok := r.tc.nodes[c]; ok {
		t.Errorf("the only child survived the drop")
	}
}

func TestDropSubtreeRefusesBusyNodes(t *testing.T) {
	type setup func(r *hostRig, n, c, g NodeID)
	for name, f := range map[string]setup{
		"leaf load-back":     func(r *hostRig, n, c, g NodeID) { a := n; r.tc.node(n).loadBackAnchor = &a },
		"leaf host lock":     func(r *hostRig, n, c, g NodeID) { r.tc.IncHostLockRef(n) },
		"child pending":      func(r *hostRig, n, c, g NodeID) { r.tc.MarkWriteThroughPending([]NodeID{c}, 3) },
		"child load-back":    func(r *hostRig, n, c, g NodeID) { a := c; r.tc.node(c).loadBackAnchor = &a },
		"child host lock":    func(r *hostRig, n, c, g NodeID) { r.tc.IncHostLockRef(c) },
		"child locked":       func(r *hostRig, n, c, g NodeID) { r.tc.node(c).lockRef = 1 },
		"grandchild pending": func(r *hostRig, n, c, g NodeID) { r.tc.MarkWriteThroughPending([]NodeID{g}, 3) },
		"grandchild load":    func(r *hostRig, n, c, g NodeID) { a := g; r.tc.node(g).loadBackAnchor = &a },
		"grandchild host":    func(r *hostRig, n, c, g NodeID) { r.tc.IncHostLockRef(g) },
		"grandchild locked":  func(r *hostRig, n, c, g NodeID) { r.tc.node(g).lockRef = 1 },
	} {
		r, n, c, g := dropRig(t)
		f(r, n, c, g)
		step := r.tc.DropSubtreeNoHost(n)
		if step.IsDropped || len(freed(step)) != 0 {
			t.Errorf("%s: dropped (%+v)", name, step)
		}
		for _, id := range []NodeID{n, c, g} {
			if _, ok := r.tc.nodes[id]; !ok {
				t.Errorf("%s: node %d deleted", name, id)
			}
		}
	}
}

func TestDropSubtreeRefusesALockedOrNonLeaf(t *testing.T) {
	r, n, _, _ := dropRig(t)
	r.tc.IncLockRef(n)
	expectPanic(t, "a locked node", func() { r.tc.DropSubtreeNoHost(n) })
	r2, _, c, _ := dropRig(t)
	_ = c
	r2.tc.root.lockRef = 1
	n2 := r2.add(5, 6)
	m := r2.add(5, 6, 7, 8) // a device child: n2 is no leaf
	_ = m
	expectPanic(t, "a node with a device child", func() { r2.tc.DropSubtreeNoHost(n2) })
}

// --- host eviction ---

func TestHostEvictionWalksUpAsParentsBecomeLeaves(t *testing.T) {
	build := func() (*hostRig, NodeID, NodeID, NodeID) {
		r := newHostRig(t, false)
		a := r.add(1, 2)
		b := r.add(1, 2, 3, 4)
		c := r.add(1, 2, 3, 4, 5, 6)
		r.hostOnly(c)
		r.hostOnly(b)
		r.hostOnly(a)
		return r, a, b, c
	}
	for _, c := range []struct {
		num  int
		gone []NodeID // indexes into a, b, c order: 0 a, 1 b, 2 c
		want int
	}{{1, []NodeID{2}, 2}, {2, []NodeID{2}, 2}, {3, []NodeID{2, 1}, 4}, {4, []NodeID{2, 1}, 4}, {5, []NodeID{2, 1, 0}, 6}, {100, []NodeID{2, 1, 0}, 6}} {
		r, a, b, cc := build()
		ids := []NodeID{a, b, cc}
		step := r.tc.DriveHostEviction(FullComponentType, c.num)
		if step.Tracker[FullComponentType] != c.want {
			t.Errorf("num %d: tracker %d, want %d", c.num, step.Tracker[FullComponentType], c.want)
		}
		for i, id := range ids {
			_, present := r.tc.nodes[id]
			if gone := slices.Contains(c.gone, NodeID(i)); present == gone {
				t.Errorf("num %d: node index %d present %v", c.num, i, present)
			}
		}
	}
}

func TestHostEvictionOfAnotherComponentDoesNothing(t *testing.T) {
	r := newHostRig(t, true)
	id := r.add(1, 2)
	r.hostOnly(id)
	step := r.tc.DriveHostEviction(ComponentType("MAMBA"), 100)
	if len(freed(step)) != 0 || step.Tracker[FullComponentType] != 0 {
		t.Errorf("a non-FULL component freed %v", step)
	}
	if _, ok := r.tc.nodes[id]; !ok {
		t.Errorf("a non-FULL eviction deleted the node")
	}
}

func TestHostLeafNeedsEveryConditionAlone(t *testing.T) {
	check := func(name string, prep func(r *hostRig, id NodeID), evicts bool) {
		r := newHostRig(t, false)
		id := r.add(1, 2)
		prep(r, id)
		r.tc.updateLeaves(r.tc.node(id))
		step := r.tc.DriveHostEviction(FullComponentType, 100)
		if got := len(freed(step)) == 1; got != evicts {
			t.Errorf("%s: evicted %v, want %v", name, got, evicts)
		}
	}
	check("host only", func(r *hostRig, id NodeID) { r.hostOnly(id) }, true)
	check("still on device", func(r *hostRig, id NodeID) { r.backup(id) }, false)
	check("not on host", func(r *hostRig, id NodeID) {
		r.hostOnly(id)
		r.tc.node(id).hostValue = nil
	}, false)
	check("host locked", func(r *hostRig, id NodeID) { r.hostOnly(id); r.tc.IncHostLockRef(id) }, false)
	check("locked", func(r *hostRig, id NodeID) { r.hostOnly(id); r.tc.IncLockRef(id) }, false)
	check("has a child", func(r *hostRig, id NodeID) {
		r.hostOnly(id)
		c := r.add(1, 2, 3, 4)
		r.backup(c)
	}, false)
	// The lock released, the node is a leaf again.
	r := newHostRig(t, false)
	id := r.add(1, 2)
	r.hostOnly(id)
	rc := r.tc.IncHostLockRef(id)
	r.tc.DecHostLockRef(id, rc)
	if got := len(freed(r.tc.DriveHostEviction(FullComponentType, 100))); got != 1 {
		t.Errorf("after the host lock is released: evicted %d, want 1", got)
	}
}

// --- write-back reclaim of host duplicates ---

// dupRig builds device leaf D joined first, then P1 and P2, each with a device child.
func dupRig(t *testing.T) (r *hostRig, d, p1, p2 NodeID) {
	r = newHostRig(t, true)
	d = r.add(9, 10)
	p1 = r.add(1, 2)
	r.add(1, 2, 3, 4)
	p2 = r.add(5, 6)
	r.add(5, 6, 7, 8)
	r.both(d)
	r.both(p1)
	r.both(p2)
	return
}

func hostLen(r *hostRig, ids ...NodeID) (n int) {
	for _, id := range ids {
		n += len(r.tc.node(id).hostValue)
	}
	return
}

func TestReclaimSparesDeviceLeavesInTheFirstPass(t *testing.T) {
	r, d, p1, p2 := dupRig(t)
	hp1 := slices.Clone(r.tc.node(p1).hostValue)
	step := r.tc.DriveHostEviction(FullComponentType, 2)
	if got := freed(step); len(got) != 1 || !slices.Equal(got[0], hp1) {
		t.Errorf("freed %v, want only the first non-leaf's %v although the leaf joined first", got, hp1)
	}
	if hostLen(r, d, p2) != 4 {
		t.Errorf("the leaf or the second node lost its host copy")
	}
	if step.Tracker[FullComponentType] != 2 {
		t.Errorf("tracker %d, want 2: reclaim stops once enough is freed", step.Tracker[FullComponentType])
	}
}

func TestReclaimTakesBothNonLeavesThenTheLeaf(t *testing.T) {
	r, d, p1, p2 := dupRig(t)
	step := r.tc.DriveHostEviction(FullComponentType, 4)
	if hostLen(r, p1, p2) != 0 || hostLen(r, d) != 2 {
		t.Errorf("after 4 tokens: p1+p2 host %d, d host %d", hostLen(r, p1, p2), hostLen(r, d))
	}
	if step.Tracker[FullComponentType] != 4 {
		t.Errorf("tracker %d, want 4", step.Tracker[FullComponentType])
	}
	r2, d2, p1b, p2b := dupRig(t)
	step = r2.tc.DriveHostEviction(FullComponentType, 5)
	if hostLen(r2, d2, p1b, p2b) != 0 || step.Tracker[FullComponentType] != 6 || len(freed(step)) != 3 {
		t.Errorf("after 5 tokens: host %d tracker %d frees %d, want all three reclaimed in the second pass", hostLen(r2, d2, p1b, p2b), step.Tracker[FullComponentType], len(freed(step)))
	}
	// The reclaimed nodes are no longer duplicates: a later call frees nothing more of them.
	if again := r2.tc.DriveHostEviction(FullComponentType, 5); len(freed(again)) != 0 {
		t.Errorf("a second reclaim freed %v", freed(again))
	}
}

func TestReclaimJoinOrderIsKept(t *testing.T) {
	r, _, p1, p2 := dupRig(t)
	// p1 leaves the duplicates and rejoins: it keeps no earlier place, so p2 is now first.
	r.tc.node(p1).writeThroughAck = nil
	r.ack++
	r.tc.MarkWriteThroughPending([]NodeID{p1}, r.ack)
	r.tc.host.trackDuplicate(r.tc, r.tc.node(p1)) // not settled: leaves
	r.tc.FinishWriteThrough([]NodeID{p1}, r.ack)  // settled again: rejoins last
	hp2 := slices.Clone(r.tc.node(p2).hostValue)
	if got := freed(r.tc.DriveHostEviction(FullComponentType, 2)); len(got) != 1 || !slices.Equal(got[0], hp2) {
		t.Errorf("freed %v, want p2's %v first", got, hp2)
	}
	// Tracking a node that is already a duplicate does not move it.
	r2, _, q1, q2 := dupRig(t)
	r2.tc.host.trackDuplicate(r2.tc, r2.tc.node(q1))
	hq1 := slices.Clone(r2.tc.node(q1).hostValue)
	if got := freed(r2.tc.DriveHostEviction(FullComponentType, 2)); len(got) != 1 || !slices.Equal(got[0], hq1) {
		t.Errorf("freed %v, want q1's %v: re-tracking must not reorder", got, hq1)
	}
	_ = q2
}

func TestReclaimSkipsPinnedAndLockedDuplicates(t *testing.T) {
	for name, f := range map[string]func(r *hostRig, p1 NodeID){
		"backup in flight": func(r *hostRig, p1 NodeID) { r.tc.MarkWriteThroughPending([]NodeID{p1}, 3) },
		"load-back":        func(r *hostRig, p1 NodeID) { a := p1; r.tc.node(p1).loadBackAnchor = &a },
		"host lock":        func(r *hostRig, p1 NodeID) { r.tc.IncHostLockRef(p1) },
	} {
		r, _, p1, p2 := dupRig(t)
		f(r, p1)
		hp2 := slices.Clone(r.tc.node(p2).hostValue)
		got := freed(r.tc.DriveHostEviction(FullComponentType, 2))
		if len(got) != 1 || !slices.Equal(got[0], hp2) {
			t.Errorf("%s: freed %v, want only p2's %v", name, got, hp2)
		}
		if r.tc.node(p1).hostValue == nil {
			t.Errorf("%s: p1 lost its host copy", name)
		}
	}
}

func TestReclaimSweepsADuplicateThatLeftTheDevice(t *testing.T) {
	r, _, p1, p2 := dupRig(t)
	r.tc.Demote(p1) // p1 has a device child, so it is no host leaf; its host copy stays
	hp1 := slices.Clone(r.tc.node(p1).hostValue)
	hp2 := slices.Clone(r.tc.node(p2).hostValue)
	got := freed(r.tc.DriveHostEviction(FullComponentType, 2))
	if len(got) != 1 || !slices.Equal(got[0], hp2) {
		t.Errorf("freed %v, want only p2's %v, not the demoted p1's", got, hp2)
	}
	if !slices.Equal(r.tc.node(p1).hostValue, hp1) {
		t.Errorf("the demoted p1's host copy changed")
	}
	// A duplicate that lost its host copy is swept too.
	r2, _, q1, q2 := dupRig(t)
	r2.tc.node(q1).hostValue = nil
	hq2 := slices.Clone(r2.tc.node(q2).hostValue)
	if got := freed(r2.tc.DriveHostEviction(FullComponentType, 2)); len(got) != 1 || !slices.Equal(got[0], hq2) {
		t.Errorf("freed %v, want only q2's %v", got, hq2)
	}
}

func TestWriteThroughDoesNotReclaimDuplicates(t *testing.T) {
	r, _, _, _ := dupRig(t)
	r.tc.ConfigureWritePolicy(false, 256)
	if got := freed(r.tc.DriveHostEviction(FullComponentType, 100)); len(got) != 0 {
		t.Errorf("write-through freed %v", got)
	}
}

// --- which nodes are duplicates ---

func TestOnlySettledNodesOnBothTiersAreDuplicates(t *testing.T) {
	r := newHostRig(t, true)
	mk := func(a, b int64) NodeID {
		id := r.add(a, b)
		r.backup(id)
		return id
	}
	settled := mk(1, 2)
	pending := mk(3, 4)
	loading := mk(5, 6)
	noHost := r.add(7, 8)
	for _, id := range []NodeID{settled, pending, loading, noHost} {
		r.tc.host.trackDuplicate(r.tc, r.tc.node(id))
	}
	// pending and loading are made so after joining: tracking them again drops them.
	r.tc.node(pending).writeThroughAck = func() *NodeID { a := NodeID(3); return &a }()
	r.tc.node(loading).loadBackAnchor = func() *NodeID { a := loading; return &a }()
	r.tc.host.trackDuplicate(r.tc, r.tc.node(pending))
	r.tc.host.trackDuplicate(r.tc, r.tc.node(loading))
	// The root is never one.
	r.tc.host.trackDuplicate(r.tc, r.tc.root)
	// A node with only host copy is not one.
	hostOnly := r.add(11, 12)
	r.hostOnly(hostOnly)
	r.tc.host.trackDuplicate(r.tc, r.tc.node(hostOnly))
	// Clear the markers so the reclaim pass could take them, if they were tracked.
	r.tc.node(pending).writeThroughAck = nil
	r.tc.node(loading).loadBackAnchor = nil
	step := r.tc.DriveHostEviction(FullComponentType, 1000)
	got := 0
	for _, f := range freed(step) {
		got += len(f)
	}
	// settled is a device leaf, reclaimed in the second pass; the host-only node is a host leaf.
	if got != 4 {
		t.Errorf("freed %d host slots, want 4: the settled duplicate and the host leaf", got)
	}
	if r.tc.node(settled).hostValue != nil {
		t.Errorf("the settled duplicate kept its host copy")
	}
	for _, id := range []NodeID{pending, loading} {
		if r.tc.node(id).hostValue == nil {
			t.Errorf("node %d, not a duplicate when last tracked, was reclaimed", id)
		}
	}
	if _, ok := r.tc.nodes[r.tc.root.id]; !ok || r.tc.root.hostValue != nil {
		t.Errorf("the root was reclaimed")
	}
}

func TestDuplicateLeavesTheListWhenItsNodeIsDeleted(t *testing.T) {
	r := newHostRig(t, true)
	x := r.add(1, 2)
	y := r.add(3, 4)
	z := r.add(5, 6)
	r.both(x)
	r.both(y)
	r.both(z)
	// y leaves: its device copy goes, then the node.
	r.tc.node(y).hostValue = nil
	r.tc.host.trackDuplicate(r.tc, r.tc.node(y))
	hx, hz := slices.Clone(r.tc.node(x).hostValue), slices.Clone(r.tc.node(z).hostValue)
	got := freed(r.tc.DriveHostEviction(FullComponentType, 1000))
	if len(got) != 2 || !slices.Equal(got[0], hx) || !slices.Equal(got[1], hz) {
		t.Errorf("freed %v, want x's %v then z's %v", got, hx, hz)
	}
}
