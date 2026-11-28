// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/unified_cache/components/mamba.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).
// Derived from python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).

package cache

import "fmt"

// MambaTreeConfig is what the tree's state component reads of its configuration: the grid its
// checkpoints land on (the checkpoint interval, aligned to the page: math.lcm, mamba.py:101-103) and
// the cap of states a path keeps (mamba_path_state_cap: 1 below checkpoint level 3, else -1 for none;
// arg_groups/overrides.py mamba_path_state_cap_of).
type MambaTreeConfig struct {
	// CheckpointInterval is the slot's checkpoint_interval.
	CheckpointInterval int
	// PathStateCap is the cap; negative for none.
	PathStateCap int
	// CheckpointsLevel is --hybrid-cache-checkpoints-level (0 for its default, 2): below 3 a running
	// request's checkpoint is held back unless it sits at the branching point.
	CheckpointsLevel int
	// Pool is the hybrid request table whose state slot allocator the tree's states are slots of,
	// and whose lifecycle (ping-pong tracking slots, working slots) the checkpoints of a request
	// drive.
	Pool *HybridRequestPool
}

// mambaTree is MambaComponent on the device: one state slot per node that holds one, an LRU list of
// those nodes, its ledgers and its eviction walk. A state sits on a node only where a request
// donated it (the leaf of an insert), never on the prefix of a split.
type mambaTree struct {
	tree *treeCore
	cfg  MambaTreeConfig
	grid int

	// head and tail are the LRU list's sentinels: the most recent state is after head, the least
	// recent before tail (UnifiedLRUList, unified_tree_core.py:270-451).
	head, tail *node
	inList     map[*node]bool

	evictable, protected int

	walking bool
	request int
	cursor  *node
}

func newMambaTree(t *treeCore, cfg MambaTreeConfig) *mambaTree {
	m := &mambaTree{tree: t, cfg: cfg}
	page := t.pageSize
	m.grid = cfg.CheckpointInterval / gcd(cfg.CheckpointInterval, page) * page
	return m
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func (m *mambaTree) reset() {
	m.head, m.tail = &node{}, &node{}
	m.head.mNext, m.tail.mPrev = m.tail, m.head
	m.inList = map[*node]bool{}
	m.evictable, m.protected = 0, 0
	m.walking, m.cursor = false, nil
}

func (m *mambaTree) addAfter(prev, n *node) {
	n.mPrev, n.mNext = prev, prev.mNext
	prev.mNext.mPrev = n
	prev.mNext = n
}

func (m *mambaTree) unlink(n *node) {
	n.mPrev.mNext, n.mNext.mPrev = n.mNext, n.mPrev
	n.mPrev, n.mNext = nil, nil
}

// insertMRU is UnifiedLRUList.insert_mru.
func (m *mambaTree) insertMRU(n *node) {
	if m.inList[n] {
		panic(fmt.Sprintf("cache: node %d is already in the state LRU", n.id))
	}
	m.inList[n] = true
	m.addAfter(m.head, n)
}

// insertAfter is insert_after: the node joins the list right after prev.
func (m *mambaTree) insertAfter(prev, n *node) {
	m.inList[n] = true
	m.addAfter(prev, n)
}

// remove is remove_node.
func (m *mambaTree) remove(n *node) {
	if !m.inList[n] {
		panic(fmt.Sprintf("cache: node %d is not in the state LRU", n.id))
	}
	delete(m.inList, n)
	m.unlink(n)
}

// resetMRU is reset_node_mru.
func (m *mambaTree) resetMRU(n *node) {
	if !m.inList[n] {
		panic(fmt.Sprintf("cache: node %d is not in the state LRU", n.id))
	}
	m.unlink(n)
	m.addAfter(m.head, n)
}

// prevNoLock is get_prev_no_lock: the next node toward the head whose state is not locked.
func (m *mambaTree) prevNoLock(n *node) *node {
	x := n.mPrev
	for x != m.head && x.mambaLock > 0 {
		x = x.mPrev
	}
	if x == m.head {
		return nil
	}
	return x
}

// lruNoLock is get_lru_no_lock: the least recent state that is not locked.
func (m *mambaTree) lruNoLock() *node { return m.prevNoLock(m.tail) }

// acquireLock is acquire_component_lock (components/mamba.py:778-804): the node's own state only,
// not the path.
func (m *mambaTree) acquireLock(n *node) {
	if n == m.tree.root {
		return
	}
	if n.mambaLock == 0 && n.mamba != nil {
		v := len(n.mamba)
		m.evictable -= v
		m.protected += v
	}
	n.mambaLock++
}

// releaseLock is release_component_lock (:806-835).
func (m *mambaTree) releaseLock(n *node) {
	if n == m.tree.root {
		return
	}
	if n.mambaLock <= 0 {
		panic(fmt.Sprintf("cache: Mamba release hit lock_ref=0 on node %d", n.id))
	}
	if n.mambaLock == 1 && n.mamba != nil {
		v := len(n.mamba)
		m.evictable += v
		m.protected -= v
	}
	n.mambaLock--
	if n.mambaLock == 0 {
		m.tree.updateLeaves(n)
	}
}

// refreshMatchEnd is refresh_lru at MATCH_END (:155-181): a match consumes only the best match's
// state, so only that one is touched. The walk down and the end of an insert refresh nothing.
func (m *mambaTree) refreshMatchEnd(n *node) {
	if n.mamba != nil {
		m.resetMRU(n)
	}
}

// branchingDepth is branching_checkpoint_depth (:68-74): the last grid multiple within the full-KV
// hit when it lies past the state boundary.
func branchingDepth(fullKVHit, boundary, grid int) *int {
	aligned := fullKVHit / grid * grid
	if aligned > boundary {
		return &aligned
	}
	return nil
}

// commitInsert is commit_insert_component_data (:391-430): the donated state goes on the insert's
// target node, and a path beyond its cap is trimmed once the insert's other actions have run.
func (m *mambaTree) commitInsert(n *node, isNewLeaf bool, p InsertParams, res *InsertResult, pending *[]CacheAction) {
	if p.MambaValue == nil && p.AllowMissingMamba {
		res.MambaExist = true
		return
	}
	if p.MambaValue == nil {
		panic("cache: an insert with no state for a tree that keeps one")
	}
	switch {
	case isNewLeaf:
		n.mamba = append([]int64{}, p.MambaValue...)
		m.insertMRU(n)
		m.evictable += len(n.mamba)
		m.emitExcess(n, pending)
	case n.mamba == nil:
		// set_component_device_value (unified_tree_core.py:3331-3351): a state materialized under a
		// lock is protected, and the last release makes it evictable.
		n.mamba = append([]int64{}, p.MambaValue...)
		m.insertMRU(n)
		if n.mambaLock > 0 {
			m.protected += len(n.mamba)
		} else {
			m.evictable += len(n.mamba)
		}
		n.lastAccess = m.tree.tick()
		m.emitExcess(n, pending)
	default:
		m.resetMRU(n)
		n.lastAccess = m.tree.tick()
		res.MambaExist = true
	}
}

func (m *mambaTree) emitExcess(tail *node, pending *[]CacheAction) {
	if m.cfg.PathStateCap < 0 {
		return
	}
	*pending = append(*pending, CacheAction{Kind: MambaEvictExcessPathStatesAction, TailNode: tail.id})
}

// evictState is evict_component on the device (:519-550) and the detach of _evict_component_and_detach_lru
// (unified_tree_core.py:2528-2575): the state is freed to the caller and leaves the LRU list.
func (m *mambaTree) evictState(n *node, step *EvictionStep) {
	if n.mamba != nil {
		step.DeviceFrees[MambaComponentType] = append(step.DeviceFrees[MambaComponentType], n.mamba)
		freed := len(n.mamba)
		m.evictable -= freed
		n.mamba = nil
		step.Tracker[MambaComponentType] += freed
	}
	if m.inList[n] {
		m.remove(n)
	}
}

// evictExcessPathStates is _evict_excess_path_states (:432-504): of the states on the path to the
// tail beyond the cap, the shallowest evictable ones go (the tail, forks, locked nodes and device
// leaves stay), so the cap is a soft limit.
func (m *mambaTree) evictExcessPathStates(tail *node, step *EvictionStep) {
	cap := m.cfg.PathStateCap
	if cap < 0 {
		return
	}
	var holders []*node
	for n := tail; n != nil && n != m.tree.root; n = n.parent {
		if n.mamba != nil {
			holders = append(holders, n)
		}
	}
	excess := len(holders) - cap
	if excess <= 0 {
		return
	}
	for i := len(holders) - 1; i >= 0; i-- {
		n := holders[i]
		if excess <= 0 || n == tail {
			break
		}
		if n.mambaLock > 0 || len(n.children) != 1 || m.tree.leaves[n] {
			continue
		}
		m.evictState(n, step)
		m.tree.updateLeaves(n)
		excess--
	}
}

// evictStart, evictNext and evictEnd are the device walk (:540-590, 612-669): from the least
// recent unlocked state toward the most recent; a state on a leaf is returned for the cache to
// evict as a leaf, and a state on an interior node is a tombstone made in this step.
func (m *mambaTree) evictStart(request int) {
	if m.walking {
		panic("cache: the state eviction walk is already in progress")
	}
	m.walking, m.request = true, request
	m.cursor = m.lruNoLock()
}

func (m *mambaTree) evictNext(tr Tracker) (EvictionStep, *node) {
	step := newStep()
	if m.cursor != nil && !m.inList[m.cursor] {
		m.cursor = m.lruNoLock()
	}
	if tr[MambaComponentType] >= m.request || m.cursor == nil || !m.inList[m.cursor] {
		return step, nil
	}
	x := m.cursor
	if x.mamba == nil {
		panic("cache: a node in the state LRU holds no state")
	}
	m.cursor = m.prevNoLock(x)
	if m.tree.leaves[x] {
		return step, x
	}
	// An interior state is tombstoned now; nothing cascades (the state ranks lowest on an interior
	// node, :637-690 of components/base.py).
	m.evictState(x, &step)
	m.tree.updateLeaves(x)
	return step, nil
}

func (m *mambaTree) evictEnd() {
	if !m.walking {
		panic("cache: the state eviction walk was not started")
	}
	m.walking, m.cursor = false, nil
}
