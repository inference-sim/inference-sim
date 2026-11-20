// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/unified_cache/components/full.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package cache

import (
	"container/heap"
	"fmt"
)

// fullComponent is FullComponent (components/full.py:48-552@297e54fd) on the device: its lock
// walk, its ledger moves and its LRU walk over the evictable leaves.
type fullComponent struct {
	tree *treeCore

	walking  bool
	request  int
	lastNode *node
	heap     leafHeap
}

// acquireLock is acquire_component_lock (:282-329) on the device: lock every node from n to
// the root, moving the value of each node that was unlocked from evictable to protected.
func (f *fullComponent) acquireLock(n *node) int {
	t := f.tree
	cur := n
	for cur != t.root && cur.value == nil {
		cur.lockRef++
		cur = cur.parent
	}
	delta := 0
	for cur != t.root {
		if cur.value == nil {
			panic(fmt.Sprintf("cache: FULL invariant broken: evicted ancestor %d above device-on segment", cur.id))
		}
		if cur.lockRef == 0 {
			t.evictable -= len(cur.value)
			t.protected += len(cur.value)
			delta += len(cur.value)
		}
		cur.lockRef++
		delete(t.leaves, cur)
		cur = cur.parent
	}
	return delta
}

// releaseLock is release_component_lock (:331-378) on the device.
func (f *fullComponent) releaseLock(n *node) {
	t := f.tree
	for cur := n; cur != t.root; cur = cur.parent {
		if cur.lockRef <= 0 {
			panic(fmt.Sprintf("cache: FULL segment release hit lock_ref=0 on node %d", cur.id))
		}
		if cur.lockRef == 1 && cur.value != nil {
			t.evictable += len(cur.value)
			t.protected -= len(cur.value)
		}
		cur.lockRef--
		if cur.lockRef == 0 {
			t.updateLeaves(cur)
		}
	}
}

// redistributeOnNodeSplit is redistribute_on_node_split (:147-170) on the device.
func (f *fullComponent) redistributeOnNodeSplit(parent, child *node) {
	parent.lockRef = child.lockRef
	// A host lock's segment extends through the new prefix: its boundary moves to the prefix.
	parent.hostLockRef = child.hostLockRef
	parent.hostUUID, child.hostUUID = child.hostUUID, nil
	k := parent.key.Len()
	if child.value != nil {
		parent.value = append([]int64{}, child.value[:k]...)
		child.value = append([]int64{}, child.value[k:]...)
	}
	if child.hostValue != nil {
		parent.hostValue = append([]int64{}, child.hostValue[:k]...)
		child.hostValue = append([]int64{}, child.hostValue[k:]...)
	}
}

// evictDeviceStart is _evict_device_start (:209-217): a heap of the evictable leaves.
func (f *fullComponent) evictDeviceStart(request int) {
	if f.walking {
		panic("cache: FULL device eviction already in progress")
	}
	f.walking = true
	f.request = request
	f.lastNode = nil
	f.heap.nodes = f.heap.nodes[:0]
	for n := range f.tree.leaves {
		f.heap.nodes = append(f.heap.nodes, n)
	}
	heap.Init(&f.heap)
}

// evictDeviceNextNode is _evict_device_next_node (:219-244): the parent of the leaf last
// evicted joins the heap once it is a leaf itself; the least recently used leaf comes next,
// while the walk has not evicted its request.
func (f *fullComponent) evictDeviceNextNode(tr Tracker) *node {
	t := f.tree
	if lv := f.lastNode; lv != nil && lv.parent != nil && t.leaves[lv.parent] {
		heap.Push(&f.heap, lv.parent)
	}
	f.lastNode = nil
	for tr[FullComponentType] < f.request && f.heap.Len() > 0 {
		x := heap.Pop(&f.heap).(*node)
		if !t.leaves[x] {
			continue
		}
		f.lastNode = x
		return x
	}
	return nil
}

func (f *fullComponent) evictDeviceEnd() {
	if !f.walking {
		panic("cache: FULL device eviction not started")
	}
	f.heap.nodes = f.heap.nodes[:0]
	f.lastNode = nil
	f.walking = false
}
