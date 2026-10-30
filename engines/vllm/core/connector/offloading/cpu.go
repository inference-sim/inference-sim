// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/cpu/manager.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/kv_offload/cpu/policies/lru.py (LRUCachePolicy), cpu/policies/base.py
// (ChunkStatus, order_request_keys) and cpu/spec.py (num_chunks from cpu_bytes_to_use).

package offloading

import (
	"fmt"
	"sort"
)

// Key is an OffloadKey: a block hash with its KV cache group index appended as 4 big-endian
// bytes (make_offload_key, vllm/v1/kv_offload/base.py:29-31@5df63a72), as a string.
type Key string

func makeKey(hash []byte, group int) Key {
	b := append(append(make([]byte, 0, len(hash)+4), hash...), byte(group>>24), byte(group>>16), byte(group>>8), byte(group))
	return Key(b)
}

// group is get_offload_group_idx (base.py:39-41).
func (k Key) group() int {
	n := len(k)
	return int(k[n-4])<<24 | int(k[n-3])<<16 | int(k[n-2])<<8 | int(k[n-1])
}

// lookup is LookupResult (base.py:120-126) for a CPU tier, which never defers: MISS, HIT or
// HIT_PENDING (a chunk whose store is still in flight).
type lookup int

const (
	miss lookup = iota
	hit
	hitPending
)

// chunk is ChunkStatus (cpu/policies/base.py:29-49): ref_cnt is -1 until the chunk's store
// completes, then the number of loads reading it.
type chunk struct {
	refCnt int
	id     int
}

func (c *chunk) ready() bool { return c.refCnt >= 0 }

// NumChunks is CPUOffloadingSpec's num_chunks (vllm/v1/kv_offload/cpu/spec.py:96-127@5df63a72):
// cpu_bytes_to_use over one chunk's bytes, a worker's KV bytes per block times the workers
// times blocks_per_chunk, rounded up to the page size (the host's mmap.PAGESIZE).
func NumChunks(cpuBytes, workerBytesPerBlock int64, worldSize, blocksPerChunk int, pageSize int64) int {
	if workerBytesPerBlock <= 0 || worldSize <= 0 {
		return 0
	}
	perChunk := workerBytesPerBlock * int64(worldSize) * int64(blocksPerChunk)
	perChunk = (perChunk + pageSize - 1) / pageSize * pageSize
	return int(cpuBytes / perChunk)
}

// cpuManager is CPUOffloadingManager (vllm/v1/kv_offload/cpu/manager.py:47-483@5df63a72) with the
// LRU cache policy (cpu/policies/lru.py:16-173), store_threshold below 2 (no admission filter)
// and no events. Request-scoped access state is reqAccess.
type cpuManager struct {
	numChunks, numAllocated int
	freeList                []int
	numEvictable            int
	lru                     *lruPolicy
}

func newCPUManager(numChunks int) *cpuManager {
	return &cpuManager{numChunks: numChunks, lru: newLRU()}
}

// reqAccess is _RequestCacheAccess (manager.py:33-44): the keys a request observed, grouped by
// KV cache group in observation order, with their token end positions (ReqContext's
// _offload_key_positions, base.py:103-117).
type reqAccess struct {
	groups    map[int][]Key
	seen      map[Key]bool
	positions map[Key]int
	finished  bool
}

func newReqAccess() *reqAccess {
	return &reqAccess{groups: map[int][]Key{}, seen: map[Key]bool{}, positions: map[Key]int{}}
}

func (m *cpuManager) numFree() int { return len(m.freeList) + m.numChunks - m.numAllocated }

// allocate is _allocate_chunks (manager.py:97-111): fresh chunks first, then reused ones off the
// free list's tail.
func (m *cpuManager) allocate(n int) []*chunk {
	fresh := min(n, m.numChunks-m.numAllocated)
	out := make([]*chunk, 0, n)
	for i := 0; i < fresh; i++ {
		out = append(out, &chunk{refCnt: -1, id: m.numAllocated})
		m.numAllocated++
	}
	for i := fresh; i < n; i++ {
		id := m.freeList[len(m.freeList)-1]
		m.freeList = m.freeList[:len(m.freeList)-1]
		out = append(out, &chunk{refCnt: -1, id: id})
	}
	return out
}

// record is _record_request_cache_access (manager.py:157-179) for the keys' order.
func (a *reqAccess) record(keys []Key) {
	for _, k := range keys {
		if a.seen[k] {
			continue
		}
		if a.finished {
			panic("offloading: new cache keys observed after request finalization")
		}
		a.groups[k.group()] = append(a.groups[k.group()], k)
		a.seen[k] = true
	}
}

// lookup is lookup (manager.py:188-195).
func (m *cpuManager) lookup(k Key) lookup {
	c := m.lru.chunks[k]
	switch {
	case c == nil:
		return miss
	case !c.ready():
		return hitPending
	}
	return hit
}

// prepareLoad is prepare_load (manager.py:203-231): it pins each chunk and returns their ids.
func (m *cpuManager) prepareLoad(keys []Key, a *reqAccess) []int {
	ids := make([]int, len(keys))
	for i, k := range keys {
		c := m.lru.chunks[k]
		if c == nil || !c.ready() {
			panic(fmt.Sprintf("offloading: chunk %x is not ready for reading", []byte(k)))
		}
		if c.refCnt == 0 {
			m.lru.markNonEvictable(k)
			m.numEvictable--
		}
		c.refCnt++
		ids[i] = c.id
	}
	a.record(keys)
	return ids
}

// completeLoad is complete_load (manager.py:237-248).
func (m *cpuManager) completeLoad(keys []Key) {
	for _, k := range keys {
		c := m.lru.chunks[k]
		c.refCnt--
		if c.refCnt == 0 {
			m.numEvictable++
			m.lru.markEvictable(k)
		}
	}
}

// prepareStore is prepare_store (manager.py:250-362) without a store threshold: the keys not yet
// in the cache are stored, evicting LRU chunks for room; ok is false when that fails (the
// caller's "cannot store chunks", scheduler.py:1735-1740). Chunk ids are the store's
// destinations.
func (m *cpuManager) prepareStore(keys []Key, a *reqAccess) (stored []Key, chunkIDs []int, ok bool) {
	var ready []Key
	for _, k := range keys {
		c := m.lru.chunks[k]
		if c == nil {
			stored = append(stored, k)
		} else if c.ready() {
			ready = append(ready, k)
		}
	}
	// on_store_miss (manager.py:283-291) is the base policy's touch (cpu/policies/base.py:94-103)
	// of keys not in the cache, which the LRU skips (_assign_new_rank, lru.py:520-525): a no-op.
	recorded := map[Key]bool{}
	for _, k := range stored {
		recorded[k] = true
	}
	for _, k := range ready {
		recorded[k] = true
	}
	var rec []Key
	for _, k := range keys {
		if recorded[k] {
			rec = append(rec, k)
		}
	}
	a.record(rec)
	if len(stored) == 0 {
		return nil, nil, true
	}
	if need := len(stored) - m.numFree(); need > 0 {
		if need > m.numEvictable {
			return nil, nil, false
		}
		protected := map[Key]bool{}
		for _, k := range keys {
			protected[k] = true
		}
		evicted := m.lru.evict(need, protected)
		if evicted == nil {
			return nil, nil, false
		}
		m.numEvictable -= len(evicted)
		for _, c := range evicted {
			m.freeList = append(m.freeList, c.id)
		}
	}
	chunks := m.allocate(len(stored))
	chunkIDs = make([]int, len(chunks))
	for i, k := range stored {
		m.lru.insert(k, chunks[i])
		chunkIDs[i] = chunks[i].id
	}
	return stored, chunkIDs, true
}

// completeStore is complete_store (manager.py:364-397) for a successful store.
func (m *cpuManager) completeStore(keys []Key) {
	for _, k := range keys {
		if c := m.lru.chunks[k]; c != nil && !c.ready() {
			c.refCnt = 0
			m.numEvictable++
			m.lru.markEvictable(k)
		}
	}
}

// requestFinished is on_request_finished (manager.py:399-426) and the LRU's
// (lru.py:578-587): the request's keys, each group's in position order, then across groups in
// head-to-tail order (order_request_keys, cpu/policies/base.py:12-24), become most recent, the
// head last.
func (m *cpuManager) requestFinished(a *reqAccess) {
	if a.finished {
		return
	}
	a.finished = true
	gs := make([]int, 0, len(a.groups))
	for g := range a.groups {
		gs = append(gs, g)
	}
	sort.Ints(gs)
	var keys []Key
	for _, g := range gs {
		ks := append([]Key(nil), a.groups[g]...)
		if a.allPositioned(ks) {
			sort.SliceStable(ks, func(i, j int) bool { return a.positions[ks[i]] < a.positions[ks[j]] })
		}
		keys = append(keys, ks...)
	}
	if a.allPositioned(keys) {
		sort.SliceStable(keys, func(i, j int) bool { return a.positions[keys[i]] < a.positions[keys[j]] })
	}
	rev := make([]Key, len(keys))
	for i, k := range keys {
		rev[len(keys)-1-i] = k
	}
	m.lru.touch(rev)
}

func (a *reqAccess) allPositioned(keys []Key) bool {
	for _, k := range keys {
		if _, ok := a.positions[k]; !ok {
			return false
		}
	}
	return true
}

// lruPolicy is LRUCachePolicy (vllm/v1/kv_offload/cpu/policies/lru.py:16-173@5df63a72): every
// chunk has a recency rank, assigned on insert and on a request's access; eviction takes the
// evictable chunks of lowest rank. vLLM keeps a lazy heap; the order it evicts in is the ranks',
// which this keeps directly.
type lruPolicy struct {
	chunks    map[Key]*chunk
	ranks     map[Key]int
	evictable map[Key]bool
	next      int
}

func newLRU() *lruPolicy {
	return &lruPolicy{chunks: map[Key]*chunk{}, ranks: map[Key]int{}, evictable: map[Key]bool{}}
}

// insert is insert (lru.py:559-566).
func (p *lruPolicy) insert(k Key, c *chunk) {
	p.chunks[k] = c
	p.next++
	p.ranks[k] = p.next
	if c.refCnt == 0 {
		p.evictable[k] = true
	}
}

// touch is _update_recency (lru.py:543-553): each present key, in order, gets the next rank.
func (p *lruPolicy) touch(keys []Key) {
	for _, k := range keys {
		if _, ok := p.chunks[k]; ok {
			p.next++
			p.ranks[k] = p.next
		}
	}
}

// evict is evict (lru.py:597-642): the n evictable, unprotected chunks of lowest rank, or nil
// (and nothing changed) if there are fewer.
func (p *lruPolicy) evict(n int, protected map[Key]bool) []*chunk {
	if n == 0 {
		return []*chunk{}
	}
	cands := make([]Key, 0, len(p.evictable))
	for k := range p.evictable {
		if !protected[k] {
			cands = append(cands, k)
		}
	}
	if len(cands) < n {
		return nil
	}
	sort.Slice(cands, func(i, j int) bool { return p.ranks[cands[i]] < p.ranks[cands[j]] })
	out := make([]*chunk, n)
	for i, k := range cands[:n] {
		out[i] = p.chunks[k]
		delete(p.evictable, k)
		delete(p.chunks, k)
		delete(p.ranks, k)
	}
	return out
}

func (p *lruPolicy) markEvictable(k Key)    { p.evictable[k] = true }
func (p *lruPolicy) markNonEvictable(k Key) { delete(p.evictable, k) }

// inRankOrder lists the cached chunks, least recent first: a trace's view of the CPU tier.
func (p *lruPolicy) inRankOrder() []Key {
	out := make([]Key, 0, len(p.chunks))
	for k := range p.chunks {
		out = append(out, k)
	}
	sort.Slice(out, func(i, j int) bool { return p.ranks[out[i]] < p.ranks[out[j]] })
	return out
}
