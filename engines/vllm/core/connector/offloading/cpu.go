// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/cpu/manager.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).
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
// and no events. Request-scoped access state is reqAccess. Each method named after an
// OffloadingManager method (vllm/v1/kv_offload/base.py:235-421) is that method, and
// fid5_test.go replays vLLM's recorded and directly driven calls at the interface against them.
type cpuManager struct {
	numChunks, numAllocated int
	freeList                []int
	numEvictable            int
	lru                     *lruPolicy
	// writePending counts chunks whose store is in flight (_num_write_pending_chunks), and
	// allocations each prepare_store's chunk count since the last get_stats
	// (allocation_sizes_in_current_batch, manager.py:85-92).
	writePending int
	allocations  []int
	// generation is _cache_generation: reset_cache starts a new one, and a request's access
	// state from an earlier one is discarded (manager.py:144-155).
	generation int
}

func newCPUManager(numChunks int) *cpuManager {
	return &cpuManager{numChunks: numChunks, lru: newLRU()}
}

// current is _get_request_cache_access (manager.py:144-155): a request's state from before the
// last reset_cache is started afresh.
func (m *cpuManager) current(a *reqAccess) *reqAccess {
	if a.generation != m.generation {
		positions := a.positions
		*a = *newReqAccess()
		a.positions, a.generation = positions, m.generation
	}
	return a
}

// onNewRequest is on_new_request (manager.py:184-187): the request's access state, offloaded at
// chunk level (RequestOffloadingContext's default, base.py:139-141).
func (m *cpuManager) onNewRequest(a *reqAccess) string {
	m.current(a)
	return "CHUNK_LEVEL"
}

// loadSource is get_load_source (manager.py:197-202): a CPU tier's hits are the host's.
func (m *cpuManager) loadSource() string { return "HOST" }

// touch is touch (manager.py:234-236), the policy's.
func (m *cpuManager) touch(keys []Key) { m.lru.touchReq(keys) }

// hasPendingWork is has_pending_work (base.py:403-409): a CPU tier keeps no work of its own.
func (m *cpuManager) hasPendingWork() bool { return false }

// resetCache is reset_cache (manager.py:429-442).
func (m *cpuManager) resetCache() {
	m.lru.clear()
	m.generation++
	m.numEvictable, m.writePending = 0, 0
	m.freeList = m.freeList[:0]
	m.numAllocated = 0
}

// cpuStats are get_stats's CPU gauges and allocation sizes (manager.py:449-483), after which the
// allocation sizes start again.
type cpuStats struct {
	usage, write, read float64
	allocations        []int
}

func (m *cpuManager) stats() cpuStats {
	var st cpuStats
	used := m.numAllocated - len(m.freeList) - m.numEvictable
	if m.numChunks > 0 {
		st.usage = float64(used) / float64(m.numChunks)
		st.write = float64(m.writePending) / float64(m.numChunks)
	}
	st.read = max(st.usage-st.write, 0)
	st.allocations, m.allocations = m.allocations, nil
	return st
}

// reqAccess is _RequestCacheAccess (manager.py:33-44): the keys a request observed, grouped by
// KV cache group in observation order, with their token end positions (ReqContext's
// _offload_key_positions, base.py:103-117).
type reqAccess struct {
	groups    map[int][]Key
	seen      map[Key]bool
	positions map[Key]int
	finished  bool
	// storeMisses are the keys this request already reported to on_store_miss (store_miss_keys).
	storeMisses map[Key]bool
	generation  int
	// lruRank is the request's provisional LRU rank and the cache generation it was taken in
	// (_RequestRank, vllm/v1/kv_offload/cpu/policies/lru.py:20-23@4ba787ec); 0 is none yet.
	lruRank, lruGeneration int
}

// positionOf is ReqContext.get_offload_key_position (vllm/v1/kv_offload/base.py@4ba787ec).
func (a *reqAccess) positionOf(k Key) (int, bool) {
	if a == nil {
		return 0, false
	}
	p, ok := a.positions[k]
	return p, ok
}

func newReqAccess() *reqAccess {
	return &reqAccess{groups: map[int][]Key{}, seen: map[Key]bool{}, positions: map[Key]int{}, storeMisses: map[Key]bool{}}
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

// record is _record_request_cache_access (vllm/v1/kv_offload/cpu/manager.py:163-189@4ba787ec)
// for the keys' order. It returns the keys the request observes for the first time, which the
// manager reports to the policy (on_request_access, :182-183).
func (a *reqAccess) record(keys []Key) []Key {
	var fresh []Key
	for _, k := range keys {
		if a.seen[k] {
			continue
		}
		if a.finished {
			panic("offloading: new cache keys observed after request finalization")
		}
		a.groups[k.group()] = append(a.groups[k.group()], k)
		a.seen[k] = true
		fresh = append(fresh, k)
	}
	return fresh
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
	a = m.current(a)
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
	m.lru.onRequestAccess(a.record(keys), a)
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
// destinations; evicted are the keys evicted for room.
func (m *cpuManager) prepareStore(keys []Key, a *reqAccess) (stored []Key, chunkIDs []int, evicted []Key, ok bool) {
	a = m.current(a)
	var ready []Key
	for _, k := range keys {
		c := m.lru.chunks[k]
		if c == nil {
			stored = append(stored, k)
		} else if c.ready() {
			ready = append(ready, k)
		}
	}
	// on_store_miss (manager.py:283-291), once per request and key.
	var misses []Key
	for _, k := range stored {
		if !a.storeMisses[k] {
			misses = append(misses, k)
		}
	}
	if len(misses) > 0 {
		m.lru.onStoreMiss(misses)
		for _, k := range misses {
			a.storeMisses[k] = true
		}
	}
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
	m.lru.onRequestAccess(a.record(rec), a)
	if len(stored) == 0 {
		return nil, nil, nil, true
	}
	m.allocations = append(m.allocations, len(stored))
	if need := len(stored) - m.numFree(); need > 0 {
		if need > m.numEvictable {
			return nil, nil, nil, false
		}
		protected := map[Key]bool{}
		for _, k := range keys {
			protected[k] = true
		}
		out := m.lru.evict(need, protected)
		if out == nil {
			return nil, nil, nil, false
		}
		m.numEvictable -= len(out)
		for _, e := range out {
			m.freeList = append(m.freeList, e.chunk.id)
			evicted = append(evicted, e.key)
		}
	}
	chunks := m.allocate(len(stored))
	chunkIDs = make([]int, len(chunks))
	for i, k := range stored {
		m.lru.insert(k, chunks[i], a)
		chunkIDs[i] = chunks[i].id
	}
	m.writePending += len(stored)
	return stored, chunkIDs, evicted, true
}

// completeStore is complete_store (manager.py:364-397): a stored chunk becomes readable, and a
// failed store's chunk is dropped and its slot freed.
func (m *cpuManager) completeStore(keys []Key, success bool) {
	for _, k := range keys {
		c := m.lru.get(k)
		if c == nil || c.ready() {
			continue
		}
		m.writePending--
		if success {
			c.refCnt = 0
			m.numEvictable++
			m.lru.markEvictable(k)
		} else {
			m.lru.remove(k)
			m.freeList = append(m.freeList, c.id)
		}
	}
}

// requestFinished is on_request_finished (manager.py:399-426) and the LRU's
// (lru.py:578-587): the request's keys, each group's in position order, then across groups in
// head-to-tail order (order_request_keys, cpu/policies/base.py:12-24), become most recent, the
// head last.
func (m *cpuManager) requestFinished(a *reqAccess) {
	if a.finished || a.generation != m.generation {
		return
	}
	a.finished = true
	gs := make([]int, 0, len(a.groups))
	for g := range a.groups {
		gs = append(gs, g)
	}
	sort.Ints(gs)
	groups := make([][]Key, 0, len(gs))
	for _, g := range gs {
		ks := append([]Key(nil), a.groups[g]...)
		if allPositioned(ks, a.positions) {
			sort.SliceStable(ks, func(i, j int) bool { return a.positions[ks[i]] < a.positions[ks[j]] })
		}
		groups = append(groups, ks)
	}
	m.lru.onRequestFinished(groups, a.positions)
}

func allPositioned(keys []Key, positions map[Key]int) bool {
	for _, k := range keys {
		if _, ok := positions[k]; !ok {
			return false
		}
	}
	return true
}

// rank is an LRU recency rank (lru.py:16, _HeapEntry@4ba787ec): a request's keys share its
// provisional rank, ordered by minus their token position so its tails precede its heads; every
// other rank is (n, 0). Ranks order lexicographically.
type rank struct{ major, minor int }

func (r rank) less(o rank) bool {
	return r.major < o.major || (r.major == o.major && r.minor < o.minor)
}

// lruPolicy is LRUCachePolicy (vllm/v1/kv_offload/cpu/policies/lru.py:26-230@4ba787ec): every
// chunk has a recency rank, assigned on insert and on a request's access; eviction takes the
// evictable chunks of lowest rank. At v0.33.0 an active request's keys share one provisional
// rank across store batches, so a running request's prefix is not evicted between its batches;
// finalization commits its recency. vLLM keeps a lazy heap; the order it evicts in is the ranks',
// which this keeps directly.
type lruPolicy struct {
	chunks    map[Key]*chunk
	ranks     map[Key]rank
	evictable map[Key]bool
	next      int
	// generation is _cache_generation, which clear advances so no request keeps a rank taken
	// before the reset.
	generation int
}

func newLRU() *lruPolicy {
	return &lruPolicy{chunks: map[Key]*chunk{}, ranks: map[Key]rank{}, evictable: map[Key]bool{}}
}

// get is get (lru.py:104-105).
func (p *lruPolicy) get(k Key) *chunk { return p.chunks[k] }

// remove is remove, after a failed store.
func (p *lruPolicy) remove(k Key) {
	delete(p.chunks, k)
	delete(p.ranks, k)
	delete(p.evictable, k)
}

// clear is clear (lru.py:165-172): it also advances the cache generation.
func (p *lruPolicy) clear() {
	p.chunks, p.ranks, p.evictable, p.next = map[Key]*chunk{}, map[Key]rank{}, map[Key]bool{}, 0
	p.generation++
}

// requestRank is _get_request_rank (lru.py:51-61): the request's provisional rank, taken once
// per cache generation.
func (p *lruPolicy) requestRank(a *reqAccess) int {
	if a == nil {
		p.next++
		return p.next
	}
	if a.lruRank == 0 || a.lruGeneration != p.generation {
		p.next++
		a.lruRank, a.lruGeneration = p.next, p.generation
	}
	return a.lruRank
}

// touchReq is touch (lru.py:128-130): the keys, last first, become most recent.
func (p *lruPolicy) touchReq(keys []Key) {
	rev := make([]Key, len(keys))
	for i, k := range keys {
		rev[len(keys)-1-i] = k
	}
	p.touch(rev)
}

// onStoreMiss is the base policy's on_store_miss, a touch. The keys are not in the cache yet,
// so for the LRU it changes nothing.
func (p *lruPolicy) onStoreMiss(keys []Key) { p.touchReq(keys) }

// onRequestAccess is on_request_access (lru.py:132-152): each newly observed key with a token
// position takes the request's shared rank at minus that position, unless it already ranks at or
// above it.
func (p *lruPolicy) onRequestAccess(keys []Key, a *reqAccess) {
	if len(keys) == 0 || a == nil {
		return
	}
	rr := p.requestRank(a)
	for _, k := range keys {
		pos, ok := a.positions[k]
		if _, cached := p.chunks[k]; !ok || !cached {
			continue
		}
		nr := rank{rr, -pos}
		if !p.ranks[k].less(nr) {
			continue
		}
		p.ranks[k] = nr
	}
}

// onRequestFinished is on_request_finished (lru.py:154-163): the request's keys in head-to-tail
// order across groups (order_request_keys), the head made most recent.
func (p *lruPolicy) onRequestFinished(groups [][]Key, positions map[Key]int) {
	var keys []Key
	for _, g := range groups {
		keys = append(keys, g...)
	}
	if allPositioned(keys, positions) {
		sort.SliceStable(keys, func(i, j int) bool { return positions[keys[i]] < positions[keys[j]] })
	}
	p.touchReq(keys)
}

// insert is insert (lru.py:107-121): a key the request positions takes the request's shared
// rank at minus its position; any other the next rank.
func (p *lruPolicy) insert(k Key, c *chunk, a *reqAccess) {
	p.chunks[k] = c
	if pos, ok := a.positionOf(k); ok {
		p.ranks[k] = rank{p.requestRank(a), -pos}
	} else {
		p.next++
		p.ranks[k] = rank{p.next, 0}
	}
	if c.refCnt == 0 {
		p.evictable[k] = true
	}
}

// touch is _update_recency (lru.py:86-88, _assign_new_rank :63-68): each present key, in order,
// gets the next rank.
func (p *lruPolicy) touch(keys []Key) {
	for _, k := range keys {
		if _, ok := p.chunks[k]; ok {
			p.next++
			p.ranks[k] = rank{p.next, 0}
		}
	}
}

// before orders two keys as vLLM's heap does its (rank, key) entries: by rank, then by the key's
// bytes. A request's keys in two KV cache groups at one position share a rank, so the key decides.
func (p *lruPolicy) before(a, b Key) bool {
	ra, rb := p.ranks[a], p.ranks[b]
	if ra != rb {
		return ra.less(rb)
	}
	return a < b
}

// eviction is one evicted chunk with its key.
type eviction struct {
	key   Key
	chunk *chunk
}

// evict is evict (lru.py:115-160): the n evictable, unprotected chunks of lowest rank, least
// recent first, or nil (and nothing changed) if there are fewer.
func (p *lruPolicy) evict(n int, protected map[Key]bool) []eviction {
	if n == 0 {
		return []eviction{}
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
	sort.Slice(cands, func(i, j int) bool { return p.before(cands[i], cands[j]) })
	out := make([]eviction, n)
	for i, k := range cands[:n] {
		out[i] = eviction{k, p.chunks[k]}
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
	sort.Slice(out, func(i, j int) bool { return p.before(out[i], out[j]) })
	return out
}
