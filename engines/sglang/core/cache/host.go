// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).
// Also derived from, at the same release: python/sglang/srt/mem_cache/unified_cache/components/full.py
// (the host lock, host eviction and HiCache transfer hooks).

package cache

import (
	"container/heap"
	"fmt"
	"slices"
)

// hostState is the tree core's host tier (unified_tree_core.py:419-545@297e54fd): whether
// HiCache is on, the write policy as the tree reads it, the evictable host leaves, and the nodes
// whose FULL KV is on both tiers (full_host_duplicates), in the order they joined.
type hostState struct {
	enabled               bool
	writeBack             bool
	writeThroughThreshold int
	leaves                map[*node]bool
	dupOrder              []NodeID
	dups                  map[NodeID]*node
	// uuid is the FULL host-lock uuid counter (components/base.py:54-170): FULL's range starts at
	// 2e14. SGLang's counter is the process's; the port's is the tree's.
	uuid int64
}

// fullUUIDBase is FULL's uuid range start, (range index 1 + 1) * 1e14 (components/base.py:54-65).
const fullUUIDBase int64 = 200_000_000_000_000

func (h *hostState) resetTree() {
	if h.writeThroughThreshold == 0 {
		// UnifiedTreeCore.__init__ (:437); init_hicache sets the policy's threshold.
		h.writeThroughThreshold = 256
	}
	if h.uuid == 0 {
		h.uuid = fullUUIDBase
	}
	h.leaves = map[*node]bool{}
	h.dupOrder = nil
	h.dups = map[NodeID]*node{}
}

// trackDuplicate is _update_duplicate_tracking (:1604-1622): a node joins when its FULL KV is
// settled on both tiers, keeping the place it first joined at; it leaves otherwise.
func (h *hostState) trackDuplicate(t *treeCore, n *node) {
	settled := n != t.root && n.value != nil && n.hostValue != nil && n.writeThroughAck == nil && n.loadBackAnchor == nil
	if !settled {
		h.dropDuplicate(n.id)
		return
	}
	if _, ok := h.dups[n.id]; !ok {
		h.dups[n.id] = n
		h.dupOrder = append(h.dupOrder, n.id)
	}
}

func (h *hostState) dropDuplicate(id NodeID) {
	if _, ok := h.dups[id]; !ok {
		return
	}
	delete(h.dups, id)
	h.dupOrder = slices.DeleteFunc(h.dupOrder, func(x NodeID) bool { return x == id })
}

// ConfigureWritePolicy sets what the tree reads of the write policy: UnifiedRadixCache's
// is_write_back and write_through_threshold properties (unified_radix_cache.py:557-565, 3917-3931
// @297e54fd), which init_hicache sets on the tree core.
func (t *treeCore) ConfigureWritePolicy(writeBack bool, writeThroughThreshold int) {
	t.host.writeBack = writeBack
	t.host.writeThroughThreshold = writeThroughThreshold
}

// SetHicacheEnabled is set_hicache_enabled (:2389-2390).
func (t *treeCore) SetHicacheEnabled() { t.host.enabled = true }

// isHostLeaf is _is_host_leaf (:2366-2387).
func (t *treeCore) isHostLeaf(n *node) bool {
	return n != t.root && n.evicted() && n.backuped() && n.hostLockRef == 0 && n.lockRef == 0 && len(n.children) == 0
}

// IsFullDeviceEvicted is is_full_device_evicted (:1074-1076).
func (t *treeCore) IsFullDeviceEvicted(id NodeID) bool { return t.node(id).evicted() }

// RotationPhaseOf is rotation_phase_of (:584-595): logical-page KV sharding is not modelled, so
// no node has a rotation base.
func (t *treeCore) RotationPhaseOf(id NodeID) *int {
	t.node(id)
	return nil
}

// IncHostLockRef is inc_host_lock_ref (:867-875) with FULL's acquire_component_lock(lock_host)
// (components/full.py:282-299): only the anchor is locked, and only when it has a host copy (or
// under write-back); the receipt names the segment's boundary uuid.
func (t *treeCore) IncHostLockRef(id NodeID) LockReceipt {
	n := t.node(id)
	r := LockReceipt{Node: n.id}
	if n.hostValue != nil || t.host.writeBack {
		if n.hostUUID == nil {
			t.host.uuid++
			u := t.host.uuid
			n.hostUUID = &u
		}
		u := *n.hostUUID
		r.HostUUID = &u
		n.hostLockRef++
		t.updateLeaves(n)
	}
	t.updateLeaves(n)
	return r
}

// DecHostLockRef is dec_host_lock_ref (:877-884) with FULL's release_component_lock(lock_host)
// (components/full.py:337-357): the walk releases each node from the anchor up to the one that
// holds the receipt's boundary uuid.
func (t *treeCore) DecHostLockRef(id NodeID, r LockReceipt) {
	anchor := t.node(id)
	if r.Node != anchor.id {
		panic(fmt.Sprintf("cache: lock receipt anchored on node %d released on node %d", r.Node, anchor.id))
	}
	if r.HostUUID != nil {
		for n := anchor; ; n = n.parent {
			if n.hostValue == nil && !t.host.writeBack {
				break
			}
			if n.hostLockRef <= 0 {
				panic(fmt.Sprintf("cache: Full host segment release hit host_lock_ref=0 on node %d", n.id))
			}
			n.hostLockRef--
			t.updateLeaves(n)
			if n.hostUUID != nil && *n.hostUUID == *r.HostUUID {
				break
			}
			if n.parent == nil {
				panic(fmt.Sprintf("cache: Full host lock boundary %d is not an ancestor of receipt anchor %d", *r.HostUUID, r.Node))
			}
		}
	}
	t.updateLeaves(anchor)
}

// backupKVAction is _build_backup_kv_action (:2622-2644): the node, and under write-through its
// unbacked ancestors, ancestors first.
func (t *treeCore) backupKVAction(n *node, writeBack bool) CacheAction {
	chain := []NodeID{n.id}
	if !writeBack {
		for a := n.parent; a != nil && a != t.root && !a.backuped(); a = a.parent {
			chain = append(chain, a.id)
		}
		slices.Reverse(chain)
	}
	return CacheAction{Kind: BackupKVAction, NodeIDs: chain}
}

// BuildBackupSpec is build_backup_spec (:2491-2526): the device slots a node's backup copies,
// none when it is already on host.
func (t *treeCore) BuildBackupSpec(id NodeID) []int64 {
	n := t.node(id)
	if n.value == nil {
		panic(fmt.Sprintf("cache: a backup of node %d, which has no device value", id))
	}
	if n.backuped() {
		return []int64{}
	}
	return n.value
}

// CommitBackup is commit_backup (:2668-2692) with FULL's BACKUP_HOST commit
// (components/full.py:456-458): the node's host slots.
func (t *treeCore) CommitBackup(id NodeID, host []int64) {
	n := t.node(id)
	if len(host) > 0 {
		n.hostValue = append([]int64{}, host...)
	}
}

// MarkWriteThroughPending is mark_write_through_pending (:2753-2768): it stamps the ack on every
// node and returns them shallowest first (ties by id).
func (t *treeCore) MarkWriteThroughPending(ids []NodeID, ack NodeID) []NodeID {
	type marked struct {
		depth int
		id    NodeID
	}
	var ms []marked
	for _, id := range ids {
		n := t.node(id)
		if n.writeThroughAck != nil && *n.writeThroughAck != ack {
			panic(fmt.Sprintf("cache: node %d is already pending under a different write-through ack", id))
		}
		a := ack
		n.writeThroughAck = &a
		d := 0
		for x := n; x.parent != nil; x = x.parent {
			d++
		}
		ms = append(ms, marked{d, id})
	}
	slices.SortFunc(ms, func(a, b marked) int {
		if a.depth != b.depth {
			return a.depth - b.depth
		}
		return int(a.id - b.id)
	})
	out := make([]NodeID, len(ms))
	for i, m := range ms {
		out[i] = m.id
	}
	return out
}

// FinishWriteThrough is finish_write_through (:2840-2849).
func (t *treeCore) FinishWriteThrough(ids []NodeID, ack NodeID) {
	for _, id := range ids {
		n := t.node(id)
		if n.writeThroughAck != nil && *n.writeThroughAck == ack {
			n.writeThroughAck = nil
			t.host.trackDuplicate(t, n)
		}
		t.recordStoreMedium(n, nil, "CPU_PINNED")
	}
}

// BuildLoadBackSpec is build_load_back_spec (:2578-2611) with FULL's LOAD_BACK transfer
// (components/full.py:416-445): the host slots of the evicted path from the node up to its first
// device ancestor. A path another load-back has pinned yields an empty spec.
func (t *treeCore) BuildLoadBackSpec(id NodeID) LoadBackSpec {
	n := t.node(id)
	var nodes []*node
	for cur := n; cur.evicted(); cur = cur.parent {
		if cur.hostValue == nil {
			panic(fmt.Sprintf("cache: node %d on a load-back path has no host value", cur.id))
		}
		nodes = append(nodes, cur)
	}
	slices.Reverse(nodes)
	spec := LoadBackSpec{Host: []int64{}, Nodes: []NodeID{}}
	for _, x := range nodes {
		if x.loadBackAnchor != nil && *x.loadBackAnchor != n.id {
			return LoadBackSpec{Host: []int64{}, Nodes: []NodeID{}}
		}
		spec.Host = append(spec.Host, x.hostValue...)
		spec.Nodes = append(spec.Nodes, x.id)
	}
	return spec
}

// CommitLoadBack is commit_load_back (:2694-2733) with FULL's LOAD_BACK commit
// (components/full.py:460-485): each node on the path takes its device slots, protected when it
// is locked.
func (t *treeCore) CommitLoadBack(id NodeID, device []int64, spec LoadBackSpec) []CacheAction {
	n := t.node(id)
	if t.host.writeBack {
		for _, nid := range spec.Nodes {
			p := t.node(nid)
			if p.loadBackAnchor != nil && *p.loadBackAnchor != id {
				panic(fmt.Sprintf("cache: node %d pinned by load-back %d, new anchor %d", nid, *p.loadBackAnchor, id))
			}
			a := id
			p.loadBackAnchor = &a
		}
	}
	if device != nil {
		off := 0
		for _, nid := range spec.Nodes {
			x := t.node(nid)
			k := len(x.hostValue)
			x.value = append([]int64{}, device[off:off+k]...)
			off += k
			if x.lockRef > 0 {
				t.protected += k
			} else {
				t.evictable += k
			}
			t.updateLeaves(x)
		}
	}
	t.updateLeaves(n)
	for _, nid := range spec.Nodes {
		t.recordStoreMedium(t.node(nid), nil, "GPU")
	}
	t.updateLeaves(n)
	return nil
}

// FinishLoadBack is finish_load_back (:2735-2751): along the anchor's path, write-back's pins
// are cleared and duplicate tracking catches up with the loaded copies.
func (t *treeCore) FinishLoadBack(anchor NodeID) {
	for n := t.node(anchor); n != nil && n != t.root; n = n.parent {
		if t.host.writeBack {
			if n.loadBackAnchor == nil || *n.loadBackAnchor != anchor {
				continue
			}
			n.loadBackAnchor = nil
		}
		t.host.trackDuplicate(t, n)
	}
}

// Demote is demote (:2077-2088).
func (t *treeCore) Demote(id NodeID) EvictionStep {
	step := newStep()
	t.demote(t.node(id), &step)
	return step
}

// demote is _demote (:2090-2123): the node's device slots are freed and it stays in the tree on
// host only.
func (t *treeCore) demote(n *node, step *EvictionStep) {
	if n.evicted() || !n.backuped() {
		panic(fmt.Sprintf("cache: demote of node %d, which is not on both tiers", n.id))
	}
	step.DeviceFrees[FullComponentType] = append(step.DeviceFrees[FullComponentType], n.value)
	step.Tracker[FullComponentType] += len(n.value)
	t.evictable -= len(n.value)
	n.value = nil
	t.updateLeaves(n)
	t.recordRemoveMedium(n, "GPU")
	t.updateLeaves(n.parent)
}

// DropSubtreeNoHost is drop_subtree_no_host (:1825-1876): write-back's fallback when a leaf's
// backup failed for want of host room. The leaf and its host-only subtree are deleted, unless a
// node of it is locked or in flight.
func (t *treeCore) DropSubtreeNoHost(id NodeID) EvictionStep {
	step := newStep()
	n := t.node(id)
	if !t.isDeviceLeaf(n) {
		panic(fmt.Sprintf("cache: node %d is not a device leaf", id))
	}
	busy := func(x *node) bool {
		return x.writeThroughAck != nil || x.loadBackAnchor != nil || x.lockRef > 0 || x.hostLockRef > 0
	}
	if n.loadBackAnchor != nil || n.lockRef > 0 || n.hostLockRef > 0 {
		return step
	}
	var desc []*node
	stack := n.orderedChildren()
	for len(stack) > 0 {
		cur := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if busy(cur) {
			return step
		}
		desc = append(desc, cur)
		stack = append(stack, cur.orderedChildren()...)
	}
	for i := len(desc) - 1; i >= 0; i-- {
		t.releaseAllLayers(desc[i], "CPU_PINNED", &step)
		t.removeLeafFromParent(desc[i])
	}
	t.deleteUnbackedDeviceLeaf(n, &step)
	step.IsDropped = true
	return step
}

// DriveHostEviction is drive_host_eviction (:1931-1959): under write-back FULL first reclaims
// the host copies of nodes also on device, then FULL evicts host leaves in eviction order
// (components/full.py:246-280) until numTokens host slots are free.
func (t *treeCore) DriveHostEviction(c ComponentType, numTokens int) EvictionStep {
	step := newStep()
	if c != FullComponentType {
		return step
	}
	if t.host.writeBack {
		t.reclaimHostDuplicates(numTokens, &step)
	}
	h := leafHeap{priority: t.full.heap.priority}
	for n := range t.host.leaves {
		h.nodes = append(h.nodes, n)
	}
	heap.Init(&h)
	for step.Tracker[FullComponentType] < numTokens && h.Len() > 0 {
		x := heap.Pop(&h).(*node)
		if !t.host.leaves[x] {
			continue
		}
		t.evictHostLeaf(x, &step)
		if x.parent != nil && t.host.leaves[x.parent] {
			heap.Push(&h, x.parent)
		}
	}
	return step
}

// evictHostLeaf is _evict_host_leaf (:2046-2075).
func (t *treeCore) evictHostLeaf(n *node, step *EvictionStep) {
	if !t.isHostLeaf(n) {
		panic(fmt.Sprintf("cache: node %d is not a host leaf", n.id))
	}
	t.recordRemoveMedium(n, "CPU_PINNED")
	step.HostFrees[FullComponentType] = append(step.HostFrees[FullComponentType], n.hostValue)
	step.Tracker[FullComponentType] += len(n.hostValue)
	n.hostValue = nil
	delete(t.host.leaves, n)
	t.removeLeafFromParent(n)
	t.deleteTombstoneAncestors(n.parent, step)
}

// reclaimHostDuplicates is _reclaim_full_host_duplicates (:1975-2006): two passes over the
// duplicates in the order they joined, the first sparing device leaves (a demote is imminent).
func (t *treeCore) reclaimHostDuplicates(numTokens int, step *EvictionStep) {
	var swept []NodeID
	for _, spare := range []bool{true, false} {
		if step.Tracker[FullComponentType] >= numTokens {
			break
		}
		for _, id := range slices.Clone(t.host.dupOrder) {
			if step.Tracker[FullComponentType] >= numTokens {
				break
			}
			n := t.host.dups[id]
			if n == nil {
				continue
			}
			if n.value == nil || n.hostValue == nil {
				swept = append(swept, id)
				continue
			}
			if spare && t.leaves[n] {
				continue
			}
			if n == t.root || n.writeThroughAck != nil || n.loadBackAnchor != nil || n.hostLockRef != 0 {
				continue
			}
			t.recordRemoveMedium(n, "CPU_PINNED")
			step.HostFrees[FullComponentType] = append(step.HostFrees[FullComponentType], n.hostValue)
			step.Tracker[FullComponentType] += len(n.hostValue)
			n.hostValue = nil
			swept = append(swept, id)
		}
	}
	for _, id := range swept {
		t.host.dropDuplicate(id)
	}
}
