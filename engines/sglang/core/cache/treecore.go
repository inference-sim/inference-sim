// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).
// Also derived from, at the same release: python/sglang/srt/mem_cache/radix_cache.py (RadixKey).

package cache

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// node is a UnifiedTreeNode (unified_tree_core.py:112-189@297e54fd) holding the FULL component
// only: its key, its device slots and its lock count.
type node struct {
	id       NodeID
	parent   *node
	children map[string]*node
	// order is the children's keys in the order they entered the children dict, which Python
	// keeps: a key assigned again keeps its place (a split's new node takes the old child's), a
	// popped key leaves it. _dfs_weight_order breaks ties between children in this order.
	order []string
	key   Key
	// value is the FULL device slots; nil once evicted. The root holds an empty, non-nil value.
	value      []int64
	lockRef    int
	lastAccess float64
	creation   float64
	hitCount   int
	priority   int
	// hashValue and eventHashValue are the node's page hashes, computed when an event or a
	// storage backend first needs them; nil until then.
	hashValue      []string
	eventHashValue []string

	// The host tier (unified_tree_core.py:112-158@297e54fd): FULL's host slots (nil when not
	// backed up), its host lock count and the boundary uuid of a host lock taken on it, the ack
	// of a backup in flight that covers it, and the anchor of a load-back reading its host slots.
	hostValue       []int64
	hostLockRef     int
	hostUUID        *int64
	writeThroughAck *NodeID
	loadBackAnchor  *NodeID
}

func (n *node) evicted() bool { return n.parent != nil && n.value == nil }

// backuped is UnifiedTreeNode.backuped: FULL KV present on host.
func (n *node) backuped() bool { return n.hostValue != nil }

// setChild is children[k] = c.
func (n *node) setChild(k string, c *node) {
	if _, ok := n.children[k]; !ok {
		n.order = append(n.order, k)
	}
	n.children[k] = c
}

// popChild is children.pop(k).
func (n *node) popChild(k string) {
	if _, ok := n.children[k]; !ok {
		return
	}
	delete(n.children, k)
	for i, x := range n.order {
		if x == k {
			n.order = append(n.order[:i:i], n.order[i+1:]...)
			break
		}
	}
}

// orderedChildren is list(children.values()).
func (n *node) orderedChildren() []*node {
	out := make([]*node, len(n.order))
	for i, k := range n.order {
		out[i] = n.children[k]
	}
	return out
}

// treeCore is UnifiedTreeCore (unified_tree_core.py:419-3469@297e54fd) with the FULL component
// alone and no host tier: the tree, its access clock, its evictable leaves and its sizes.
type treeCore struct {
	pageSize int
	isEagle  bool
	full     *fullComponent
	events   eventRecorder

	// clock is _LAST_ACCESS_TIME_COUNTER_FLOAT (components/base.py:158-165@297e54fd): a counter,
	// not a clock, that orders accesses.
	clock                float64
	nextID               NodeID
	root                 *node
	nodes                map[NodeID]*node
	leaves               map[*node]bool // evictable_device_leaves
	evictable, protected int

	walk *insertWalk

	host hostState
}

func newTreeCore(pageSize int, isEagle, events, dynamo bool) *treeCore {
	t := &treeCore{pageSize: pageSize, isEagle: isEagle,
		events: eventRecorder{enabled: events, pageSize: pageSize, dynamoFormat: dynamo}}
	t.full = &fullComponent{tree: t}
	t.reset()
	return t
}

func (t *treeCore) tick() float64 {
	v := t.clock
	t.clock++
	return v
}

func (t *treeCore) newNode(priority int) *node {
	// UnifiedTreeNode.__init__ reads the counter twice: last access, then creation.
	n := &node{id: t.nextID, children: map[string]*node{}, priority: priority}
	n.lastAccess = t.tick()
	n.creation = t.tick()
	t.nextID++
	t.nodes[n.id] = n
	return n
}

// reset is UnifiedTreeCore.reset (:482-544).
func (t *treeCore) reset() {
	if t.full.walking {
		t.full.evictDeviceEnd()
	}
	t.nodes = map[NodeID]*node{}
	t.walk = nil
	t.root = t.newNode(0)
	t.root.priority = math.MinInt + 1 // -sys.maxsize
	t.root.value = []int64{}
	t.root.hashValue = []string{}
	t.root.lockRef = 1
	t.evictable, t.protected = 0, 0
	t.leaves = map[*node]bool{}
	t.host.resetTree()
}

func (t *treeCore) node(id NodeID) *node {
	n, ok := t.nodes[id]
	if !ok {
		panic(fmt.Sprintf("cache: no node %d in the tree", id))
	}
	return n
}

// Reset is reset (:482-544), as the cache calls it.
func (t *treeCore) Reset() { t.reset() }

var _ TreeCore = (*treeCore)(nil)

// --- RadixKey ---

func (k Key) slice(start, stop int) Key {
	out := Key{ExtraKey: k.ExtraKey, CacheSalt: k.CacheSalt, Bigram: k.Bigram}
	if k.Bigram {
		if stop > start {
			out.Tokens = k.Tokens[start : stop+1]
		}
		return out
	}
	out.Tokens = k.Tokens[start:stop]
	return out
}

func (k Key) pageAligned(page int) Key {
	if page == 1 {
		return k
	}
	return k.slice(0, k.Len()/page*page)
}

// childKey is RadixKey.child_key_at (radix_cache.py:221-246@297e54fd) as a string: the first
// page of units at offset, namespaced by extra_key and cache_salt.
func (k Key) childKey(offset, page int) string {
	var b strings.Builder
	if k.CacheSalt != nil {
		b.WriteString("s:" + strconv.Quote(*k.CacheSalt) + ":")
	}
	if k.ExtraKey != nil {
		b.WriteString("e:" + strconv.Quote(*k.ExtraKey) + ":")
	}
	for j := offset; j < offset+page; j++ {
		b.WriteString(strconv.FormatInt(k.Tokens[j], 10))
		if k.Bigram {
			b.WriteString("." + strconv.FormatInt(k.Tokens[j+1], 10))
		}
		b.WriteByte(',')
	}
	return b.String()
}

// matchAt is RadixKey.match_at (radix_cache.py:158-219@297e54fd): the units k shares with
// other from offset, rounded down to whole pages.
func (k Key) matchAt(other Key, offset, page int) int {
	if !sameNamespace(k, other) || k.Bigram != other.Bigram {
		panic("cache: RadixKey operations require matching namespaces and bigram modes")
	}
	raw0, raw1 := k.Tokens, other.Tokens
	n := min(len(raw0), len(raw1)-offset)
	matched := n
	for i := 0; i < n; i++ {
		if raw0[i] != raw1[offset+i] {
			matched = i
			break
		}
	}
	if k.Bigram {
		m := max(0, min(matched-1, k.Len(), other.Len()-offset))
		if page > 1 {
			return m / page * page
		}
		return m
	}
	matched = min(matched, k.Len(), other.Len()-offset)
	return matched / page * page
}

// --- matching ---

// MatchPrefix is UnifiedTreeCore.match_prefix (:886-895), _match_prefix_helper (:905-984) and
// _match_post_processor (:1012-1068). With HiCache on, a node backed up on host is a valid match
// boundary too, and the device anchor is tracked apart from it; an evicted node with no host copy
// ends the walk.
func (t *treeCore) MatchPrefix(p MatchPrefixParams) MatchResult {
	key := p.Key
	if t.isEagle {
		key.Bigram = true
	}
	key = key.pageAligned(t.pageSize)
	if key.Len() == 0 {
		return t.emptyMatch()
	}
	sep := t.host.enabled
	cur := t.root
	offset := 0
	var chunks [][]int64
	best, bestDev, bestDevLen := t.root, t.root, 0
	full := 0
	var actions []CacheAction
	// FULL's validators (components/full.py:110-121): device only, or device or host.
	valid := func(n *node) bool { return n.value != nil || (sep && n.backuped()) }
	update := func(n *node) {
		matched := valid(n)
		if matched {
			best = n
		}
		if !sep {
			if matched {
				bestDev, bestDevLen = n, len(chunks)
			}
			return
		}
		if n.value != nil {
			bestDev, bestDevLen = n, len(chunks)
		}
	}
	for offset < key.Len() {
		child, ok := cur.children[key.childKey(offset, t.pageSize)]
		if !ok || (child.evicted() && !child.backuped()) {
			break
		}
		pl := child.key.matchAt(key, offset, t.pageSize)
		full += pl
		if pl < child.key.Len() {
			n, a := t.splitNode(child, pl)
			if a != nil {
				actions = append(actions, *a)
			}
			if !n.evicted() {
				chunks = append(chunks, n.value)
			}
			update(n)
			cur = n
			break
		}
		if !child.evicted() {
			chunks = append(chunks, child.value)
		}
		cur = child
		update(cur)
		offset += pl
	}
	now := t.tick()
	for n := best; n != nil; n = n.parent {
		n.lastAccess = now
		now -= 0.00001
	}
	devLen := 0
	for _, c := range chunks[:bestDevLen] {
		devLen += len(c)
	}
	lastHost := bestDev
	if sep {
		lastHost = best
	}
	r := MatchResult{DevicePrefixLen: devLen, LastDeviceNode: bestDev.id, LastHostNode: lastHost.id, BestMatchNode: best.id,
		FullKVHitLength: full, Actions: actions}
	// FullComponent.finalize_match_result_in_tree_core (components/full.py:123-145): the host
	// slots between the best match and the device anchor.
	hit := 0
	for n := best; n != bestDev && n != t.root; n = n.parent {
		hit += len(n.hostValue)
	}
	r.HostHitLength = hit
	if cur != t.root {
		id := cur.id
		r.LastVisitedNode = &id
	}
	return r
}

func (t *treeCore) emptyMatch() MatchResult {
	return MatchResult{LastDeviceNode: t.root.id, LastHostNode: t.root.id, BestMatchNode: t.root.id}
}

// CollectFullDeviceIndices is collect_full_device_indices (:1078-1094).
func (t *treeCore) CollectFullDeviceIndices(from, until NodeID) []int64 {
	stop := t.node(until)
	var chunks [][]int64
	for n := t.node(from); n != stop; n = n.parent {
		if n.value == nil {
			panic("cache: an evicted node on a device path")
		}
		chunks = append(chunks, n.value)
	}
	out := []int64{}
	for i := len(chunks) - 1; i >= 0; i-- {
		out = append(out, chunks[i]...)
	}
	return out
}

// splitNode is _split_node (:1467-1530): child's first splitLen units become a new parent. When
// a backup of child is in flight, the new parent joins its ack and the cache is told.
func (t *treeCore) splitNode(child *node, splitLen int) (*node, *CacheAction) {
	nn := t.newNode(child.priority)
	nn.children = map[string]*node{}
	nn.setChild(child.key.childKey(splitLen, t.pageSize), child)
	nn.parent = child.parent
	nn.key = child.key.slice(0, splitLen)
	nn.hitCount = child.hitCount
	nn.creation = child.creation
	nn.loadBackAnchor = child.loadBackAnchor
	oldKey := child.key
	child.parent = nn
	child.key = child.key.slice(splitLen, child.key.Len())
	nn.hashValue, child.hashValue = splitHash(child.hashValue, splitLen, t.pageSize)
	nn.eventHashValue, child.eventHashValue = splitHash(child.eventHashValue, splitLen, t.pageSize)
	t.full.redistributeOnNodeSplit(nn, child)
	nn.parent.setChild(oldKey.childKey(0, t.pageSize), nn)
	var action *CacheAction
	if ack := child.writeThroughAck; ack != nil {
		a := *ack
		nn.writeThroughAck = &a
		action = &CacheAction{Kind: UpdatePendingBackupOnNodeSplitAction, AckID: &a, OldNode: child.id, NewNode: nn.id,
			NewChild: child.id}
	}
	child.lastAccess = t.tick()
	t.updateLeaves(nn)
	t.updateLeaves(child)
	t.host.trackDuplicate(t, nn)
	return nn, action
}

// splitHash is split_node_hash_value (mem_cache/utils.py:217-241@297e54fd).
func splitHash(h []string, splitLen, page int) ([]string, []string) {
	if h == nil {
		return nil, nil
	}
	pages := splitLen / page
	return append([]string{}, h[:pages]...), append([]string{}, h[pages:]...)
}

// --- inserting ---

type insertPhase int

const (
	phaseWalk insertPhase = iota
	phaseCommit
	phaseTail
)

// insertWalk is _InsertWalkState (unified_tree_core.py:400-416@297e54fd): the in-flight
// resumable insert.
type insertWalk struct {
	phase   insertPhase
	node    *node
	key     Key
	value   []int64
	p       InsertParams
	total   int
	pending []CacheAction
	result  InsertResult
	target  *node
	newLeaf bool
}

func (t *treeCore) touch(n *node) { n.lastAccess = t.tick() }

// BeginInsert is begin_insert (:1129-1184): the walk runs to its first barrier, a backup, or to
// its end.
func (t *treeCore) BeginInsert(p InsertParams) InsertStep {
	if t.walk != nil {
		panic("cache: concurrent insert walks")
	}
	key, value := p.Key, p.Value
	if t.isEagle && !key.Bigram {
		key.Bigram = true
	}
	key = key.pageAligned(t.pageSize)
	value = value[:key.Len()]
	t.touch(t.root)
	t.root.priority = max(t.root.priority, p.Priority)
	if key.Len() == 0 {
		return InsertStep{Result: &InsertResult{LastDeviceNode: t.root.id}}
	}
	t.walk = &insertWalk{node: t.root, key: key, value: value, p: p}
	return t.advanceInsert()
}

// advanceInsert is _advance_insert (:1248-1273): it suspends when a step emits an action that
// cannot wait for the next barrier (a BackupKV), handing over everything pending.
func (t *treeCore) advanceInsert() InsertStep {
	w := t.walk
	for {
		flushed := len(w.pending)
		switch w.phase {
		case phaseWalk:
			t.insertWalkStep(w)
		case phaseCommit:
			t.insertCommitStep(w)
		case phaseTail:
			t.insertTailStep(w)
			t.walk = nil
			r := w.result
			return InsertStep{Actions: w.pending, Result: &r}
		}
		for _, a := range w.pending[flushed:] {
			if a.Kind == BackupKVAction {
				out := w.pending
				w.pending = nil
				return InsertStep{Actions: out}
			}
		}
	}
}

// insertWalkStep is _insert_walk_step and _insert_overlap_step (:1282-1366).
func (t *treeCore) insertWalkStep(w *insertWalk) {
	if w.key.Len() == 0 {
		w.phase = phaseCommit
		return
	}
	child, ok := w.node.children[w.key.childKey(0, t.pageSize)]
	if !ok {
		w.phase = phaseCommit
		return
	}
	t.touch(child)
	pl := child.key.matchAt(w.key, 0, t.pageSize)
	n := child
	if pl < child.key.Len() {
		var a *CacheAction
		n, a = t.splitNode(child, pl)
		if a != nil {
			w.pending = append(w.pending, *a)
		}
	}
	n.priority = max(n.priority, w.p.Priority)
	if n.evicted() {
		t.uneevict(n, w.value[:pl], w.p.SessionID)
	} else {
		// The tree keeps its slots; the inserted copies past the request's own protected prefix
		// are duplicates to free.
		dupStart := max(0, w.p.PrevPrefixLen-w.total)
		if dupStart < pl {
			w.pending = append(w.pending, CacheAction{Kind: FreeDeviceKVAction,
				Indices: [][]int64{append([]int64{}, w.value[dupStart:pl]...)}})
		}
	}
	// _finish_insert_walk_node (:1368-1381).
	if w.total+pl > w.p.InsertedLen && t.incHitCount(n) {
		w.pending = append(w.pending, t.backupKVAction(n, false))
	}
	w.node = n
	w.total += pl
	w.key = w.key.slice(pl, w.key.Len())
	w.value = w.value[pl:]
}

// insertCommitStep is _insert_commit_step (:1383-1421).
func (t *treeCore) insertCommitStep(w *insertWalk) {
	w.target = w.node
	if w.key.Len() > 0 {
		w.target = t.addNewNode(w.node, w.key, w.value, w.p.Priority, w.p.SessionID)
		w.newLeaf = true
	}
	w.result = InsertResult{PrefixLen: w.total, LastDeviceNode: w.target.id}
	w.phase = phaseTail
}

// insertTailStep is _insert_tail_step and _should_backup_after_insert (:1430-1465): a new leaf
// past what the request inserted before counts a hit, and is backed up once it has enough.
func (t *treeCore) insertTailStep(w *insertWalk) {
	if t.host.writeBack || !w.newLeaf {
		return
	}
	if w.total+w.target.key.Len() > w.p.InsertedLen && t.incHitCount(w.target) {
		w.pending = append(w.pending, t.backupKVAction(w.target, t.host.writeBack))
	}
}

// ResumeInsert is resume_insert (:1232-1236).
func (t *treeCore) ResumeInsert() InsertStep {
	if t.walk == nil {
		panic("cache: no in-flight insert")
	}
	return t.advanceInsert()
}

// HasOngoingInsert is has_ongoing_insert (:1238-1240).
func (t *treeCore) HasOngoingInsert() bool { return t.walk != nil }

// EndInsert is end_insert (:1242-1246): the walk's still-pending actions.
func (t *treeCore) EndInsert() []CacheAction {
	w := t.walk
	t.walk = nil
	if w == nil {
		return nil
	}
	return w.pending
}

// incHitCount is _inc_hit_count_and_check (:1104-1123): the count, and whether the node is due
// a write-through backup. Under write-back nothing counts.
func (t *treeCore) incHitCount(n *node) bool {
	if n.evicted() || t.host.writeBack {
		return false
	}
	n.hitCount++
	return t.host.enabled && !n.backuped() && n.hitCount >= t.host.writeThroughThreshold
}

// addNewNode is _add_new_node (:1532-1562).
func (t *treeCore) addNewNode(parent *node, key Key, value []int64, priority int, sessionID *string) *node {
	n := t.newNode(priority)
	n.parent = parent
	n.key = key
	n.value = append([]int64{}, value...)
	parent.setChild(key.childKey(0, t.pageSize), n)
	t.evictable += len(n.value)
	t.updateLeaves(n)
	t.updateLeaves(parent)
	t.recordStore(n, sessionID)
	return n
}

// uneevict is _unevict_node_on_insert (:1564-1590).
func (t *treeCore) uneevict(n *node, fresh []int64, sessionID *string) {
	n.value = append([]int64{}, fresh...)
	if n.lockRef > 0 {
		t.protected += len(n.value)
	} else {
		t.evictable += len(n.value)
	}
	t.updateLeaves(n)
	t.host.trackDuplicate(t, n)
	if n.parent != nil {
		t.updateLeaves(n.parent)
	}
	t.recordStore(n, sessionID)
}

// isDeviceLeaf is _is_device_leaf (:2348-2364).
func (t *treeCore) isDeviceLeaf(n *node) bool {
	if n == t.root || n.evicted() || n.lockRef > 0 {
		return false
	}
	for _, c := range n.children {
		if c.value != nil {
			return false
		}
	}
	return true
}

// updateLeaves is _update_evictable_leaf_sets (:1592-1602): both the device and the host set.
func (t *treeCore) updateLeaves(n *node) {
	if t.isDeviceLeaf(n) {
		t.leaves[n] = true
	} else {
		delete(t.leaves, n)
	}
	if t.isHostLeaf(n) {
		t.host.leaves[n] = true
	} else {
		delete(t.host.leaves, n)
	}
}

// --- locking ---

// IncLockRef is inc_lock_ref (:730-744).
func (t *treeCore) IncLockRef(id NodeID) LockReceipt {
	n := t.node(id)
	r := LockReceipt{Node: n.id}
	d := t.full.acquireLock(n)
	r.Delta = &d
	t.updateLeaves(n)
	return r
}

// DecLockRef is dec_lock_ref (:783-796).
func (t *treeCore) DecLockRef(id NodeID, r LockReceipt) {
	n := t.node(id)
	if r.Node != n.id {
		panic(fmt.Sprintf("cache: lock receipt anchored on node %d released on node %d", r.Node, n.id))
	}
	t.full.releaseLock(n)
	t.updateLeaves(n)
}

// --- evicting ---

// EvictDeviceStart, EvictDeviceNextNode and EvictDeviceEnd are the FULL walk (:1644-1786).
func (t *treeCore) EvictDeviceStart(c ComponentType, requestCnt int) {
	t.full.evictDeviceStart(requestCnt)
}
func (t *treeCore) EvictDeviceEnd(c ComponentType) { t.full.evictDeviceEnd() }
func (t *treeCore) EvictDeviceNextNode(c ComponentType, tr Tracker) EvictionStep {
	n := t.full.evictDeviceNextNode(tr)
	if n == nil {
		return EvictionStep{}
	}
	id := n.id
	return EvictionStep{Node: &id, MadeProgress: true}
}

// EvictDeviceLeaf is evict_device_leaf (:1788-1823): a leaf backed up on host is demoted; an
// unbacked one is deleted under write-through, and under write-back the step carries the backup
// the cache must run before it demotes the leaf.
func (t *treeCore) EvictDeviceLeaf(id NodeID, isWriteBack bool) EvictionStep {
	n := t.node(id)
	if !t.isDeviceLeaf(n) {
		panic(fmt.Sprintf("cache: node %d is not a device leaf", id))
	}
	step := newStep()
	switch {
	case !n.backuped() && isWriteBack:
		step.BackupKV = t.backupKVAction(n, true).NodeIDs
	case !n.backuped():
		t.deleteUnbackedDeviceLeaf(n, &step)
	default:
		t.demote(n, &step)
	}
	return step
}

func newStep() EvictionStep {
	return EvictionStep{DeviceFrees: map[ComponentType][][]int64{}, HostFrees: map[ComponentType][][]int64{}, Tracker: Tracker{}}
}

// deleteUnbackedDeviceLeaf is _delete_unbacked_device_leaf (:1903-1929) and
// _release_all_component_layers (:1878-1901).
func (t *treeCore) deleteUnbackedDeviceLeaf(n *node, step *EvictionStep) {
	t.releaseAllLayers(n, "GPU", step)
	parent := n.parent
	t.removeLeafFromParent(n)
	t.updateLeaves(parent)
	t.deleteTombstoneAncestors(parent, step)
}

// releaseAllLayers frees both of FULL's layers on n (EvictLayer.ALL: the tracker counts the
// device tokens) and takes n out of both leaf sets.
func (t *treeCore) releaseAllLayers(n *node, medium string, step *EvictionStep) {
	t.recordRemoveMedium(n, medium)
	if n.value != nil {
		step.DeviceFrees[FullComponentType] = append(step.DeviceFrees[FullComponentType], n.value)
		step.Tracker[FullComponentType] += len(n.value)
		t.evictable -= len(n.value)
		if n.hostValue == nil {
			step.UnbackedTokens += len(n.value)
		}
	}
	if n.hostValue != nil {
		step.HostFrees[FullComponentType] = append(step.HostFrees[FullComponentType], n.hostValue)
		n.hostValue = nil
	}
	delete(t.leaves, n)
	delete(t.host.leaves, n)
}

// removeLeafFromParent is _remove_leaf_from_parent (:2212-2221).
func (t *treeCore) removeLeafFromParent(n *node) {
	n.parent.popChild(n.key.childKey(0, t.pageSize))
	t.host.dropDuplicate(n.id)
	delete(t.nodes, n.id)
}

// deleteTombstoneAncestors is _iteratively_delete_tombstone_ancestors (:2289-2346): childless
// ancestors with FULL on neither tier are deleted, up to a live or locked node.
func (t *treeCore) deleteTombstoneAncestors(cur *node, step *EvictionStep) {
	for cur != t.root && len(cur.children) == 0 {
		if cur.lockRef > 0 || cur.hostLockRef > 0 {
			break
		}
		if cur.value != nil || cur.hostValue != nil {
			t.updateLeaves(cur)
			break
		}
		delete(t.host.leaves, cur)
		t.removeLeafFromParent(cur)
		parent := cur.parent
		t.updateLeaves(parent)
		cur = parent
	}
}

func (t *treeCore) EvictableSize() int  { return t.evictable }
func (t *treeCore) ProtectedSize() int  { return t.protected }
func (t *treeCore) TakeEvents() []Event { return t.events.take() }

// RootNodeHandle is root_node_handle (:680-682): one root serves every namespace.
func (t *treeCore) RootNodeHandle(extraKey *string) NodeID { return t.root.id }
func (t *treeCore) SupportsFastMatchPrefix() bool          { return false }

// totalSize is total_size (:3368-3385).
func (t *treeCore) totalSize() int {
	total := 0
	stack := []*node{t.root}
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		total += len(n.value)
		for _, c := range n.children {
			stack = append(stack, c)
		}
	}
	return total
}

// leafHeap orders leaves as FullComponent's heap does (components/full.py:209-244@297e54fd): by
// the eviction strategy's priority, then by the node's own order, its last access time
// (UnifiedTreeNode.__lt__, :175-176).
type leafHeap struct {
	nodes    []*node
	priority func(*node) [2]float64
}

func (h leafHeap) Len() int { return len(h.nodes) }
func (h leafHeap) Less(i, j int) bool {
	a, b := h.nodes[i], h.nodes[j]
	pa, pb := h.priority(a), h.priority(b)
	if pa != pb {
		if pa[0] != pb[0] {
			return pa[0] < pb[0]
		}
		return pa[1] < pb[1]
	}
	if a.lastAccess != b.lastAccess {
		return a.lastAccess < b.lastAccess
	}
	return a.id < b.id
}
func (h leafHeap) Swap(i, j int) { h.nodes[i], h.nodes[j] = h.nodes[j], h.nodes[i] }
func (h *leafHeap) Push(x any)   { h.nodes = append(h.nodes, x.(*node)) }
func (h *leafHeap) Pop() any {
	old := h.nodes
	x := old[len(old)-1]
	h.nodes = old[:len(old)-1]
	return x
}

// RefusalIssue is the open issue the radix cache's refusals cite (X28).
const RefusalIssue = "https://github.com/inference-sim/inference-sim/issues/2452"

// evictionPriority is get_eviction_strategy (mem_cache/utils.py:72-84@297e54fd) with each
// strategy's get_priority (evict_policy.py@297e54fd) as a pair: the smaller is evicted first.
func evictionPriority(policy string) (func(*node) [2]float64, error) {
	switch strings.ToLower(policy) {
	case "", "lru":
		return func(n *node) [2]float64 { return [2]float64{n.lastAccess} }, nil
	case "lfu":
		return func(n *node) [2]float64 { return [2]float64{float64(n.hitCount), n.lastAccess} }, nil
	case "fifo":
		return func(n *node) [2]float64 { return [2]float64{n.creation} }, nil
	case "mru":
		return func(n *node) [2]float64 { return [2]float64{-n.lastAccess} }, nil
	case "filo":
		return func(n *node) [2]float64 { return [2]float64{-n.creation} }, nil
	case "priority":
		return func(n *node) [2]float64 { return [2]float64{float64(n.priority), n.lastAccess} }, nil
	case "slru":
		// SLRUStrategy's default protected_threshold of 2 (:84-100).
		return func(n *node) [2]float64 {
			protected := 0.0
			if n.hitCount >= 2 {
				protected = 1
			}
			return [2]float64{protected, n.lastAccess}
		}, nil
	case "tlru":
		// SGLang accepts tlru on the UnifiedRadixCache this cache is (registry.py:261-271@297e54fd
		// refuses it only on another tree cache); the port does not price T-LRU's per-node tail ordering.
		return nil, fmt.Errorf("cache: the tlru eviction policy is not modelled (%s)", RefusalIssue)
	}
	return nil, fmt.Errorf("cache: unknown eviction policy %q (SGLang raises ValueError, mem_cache/utils.py:77-83)", policy)
}

// FullEvictableSize and FullProtectedSize are the FULL ledgers (:3350-3354): with FULL alone,
// the tree's.
func (t *treeCore) FullEvictableSize() int { return t.evictable }
func (t *treeCore) FullProtectedSize() int { return t.protected }

// DFSWeightOrder is dfs_weight_order (:684-685) and _dfs_weight_order
// (base_prefix_cache.py:347-398@297e54fd): the indices of handles in a depth-first walk that
// visits the heavier subtree first, a node's own handles after its subtrees'. A node's weight is
// the handles that end on it and on its children, as the walk accumulates them; children of equal
// weight keep their order in the children dict (a stable sort).
func (t *treeCore) DFSWeightOrder(handles []NodeID) []int {
	at := map[*node][]int{}
	for i, h := range handles {
		n := t.node(h)
		at[n] = append(at[n], i)
	}
	weight := map[*node]int{}
	for n, ix := range at {
		weight[n] = len(ix)
	}
	type frame struct {
		n       *node
		visited bool
	}
	stack := []frame{{t.root, false}}
	for len(stack) > 0 {
		f := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if f.visited {
			w := weight[f.n]
			for _, c := range f.n.orderedChildren() {
				w += weight[c]
			}
			weight[f.n] = w
			continue
		}
		stack = append(stack, frame{f.n, true})
		cs := f.n.orderedChildren()
		for i := len(cs) - 1; i >= 0; i-- {
			stack = append(stack, frame{cs[i], false})
		}
	}
	var order []int
	stack = []frame{{t.root, false}}
	for len(stack) > 0 {
		f := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if f.visited {
			order = append(order, at[f.n]...)
			continue
		}
		cs := f.n.orderedChildren()
		sort.SliceStable(cs, func(i, j int) bool { return weight[cs[i]] > weight[cs[j]] })
		stack = append(stack, frame{f.n, true})
		for i := len(cs) - 1; i >= 0; i-- {
			stack = append(stack, frame{cs[i], false})
		}
	}
	return order
}
