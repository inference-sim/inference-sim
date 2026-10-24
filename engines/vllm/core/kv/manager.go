// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/kv_cache_manager.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/core/single_type_kv_cache_manager.py (SingleTypeKVCacheManager,
// FullAttentionManager), for one full-attention KV cache group.

package kv

// Manager is vLLM's KVCacheManager for a single full-attention group: per-request block
// tables over a Pool, prefix-cache lookup, slot allocation and freeing.
type Manager struct {
	Pool        *Pool
	blockSize   int
	maxModelLen int
	caching     bool
	watermark   int // blocks kept free when admitting a waiting request
	reqBlocks   map[string][]*Block
	numCached   map[string]int // full blocks of the request already cached
}

// NewManager builds a manager over a fresh pool of numBlocks blocks. watermarkBlocks is the
// free blocks admission must leave: int(watermark * numBlocks) in vLLM.
func NewManager(numBlocks, blockSize, maxModelLen int, enableCaching bool, watermarkBlocks int) *Manager {
	return &Manager{
		Pool: NewPool(numBlocks, enableCaching), blockSize: blockSize, maxModelLen: maxModelLen,
		caching: enableCaching, watermark: watermarkBlocks,
		reqBlocks: map[string][]*Block{}, numCached: map[string]int{},
	}
}

// BlockSize is tokens per block.
func (m *Manager) BlockSize() int { return m.blockSize }

// Blocks returns the request's block table. The slice is the manager's own.
func (m *Manager) Blocks(id string) []*Block { return m.reqBlocks[id] }

// ComputedBlocks is get_computed_blocks (kv_cache_manager.py:265-310): the longest run of
// cached blocks matching the request's block hashes, capped so the last token is always
// recomputed (max_cache_hit_length = num_tokens - 1), and the tokens they cover.
func (m *Manager) ComputedBlocks(hashes []BlockHash, numTokens int) ([]*Block, int) {
	if !m.caching {
		return nil, 0
	}
	maxBlocks := (numTokens - 1) / m.blockSize
	var hit []*Block
	for i := 0; i < len(hashes) && i < maxBlocks; i++ {
		b := m.Pool.Cached(hashes[i])
		if b == nil {
			break
		}
		hit = append(hit, b)
	}
	return hit, len(hit) * m.blockSize
}

// AllocateSlots is allocate_slots (kv_cache_manager.py:353-611) without a connector,
// lookahead or encoder: it adds the prefix-hit blocks (if any) and new blocks so the
// request has slots for computed+newComputed+numNew tokens, then caches its full blocks.
// It returns false, changing nothing, when the blocks do not fit beside the watermark.
//
// computed is the request's num_computed_tokens before this call; waiting is whether the
// request is WAITING or PREEMPTED (the watermark applies only to those, and only when
// others are scheduled, hasScheduled); hashes are the request's block hashes so far;
// numTokens is its token count (cache only up to it). fullSequenceMustFit is the
// scheduler's scheduler_reserve_full_isl admission gate (vllm/config/scheduler.py:190,
// default True; kv_cache_manager.py:514-541): a new request is admitted only if its whole
// sequence fits, not just its first chunk.
func (m *Manager) AllocateSlots(id string, computed, numNew int, newComputed []*Block, hashes []BlockHash,
	numTokens int, waiting, hasScheduled, fullSequenceMustFit bool) bool {
	if numNew == 0 && len(newComputed) == 0 {
		panic("kv: allocate_slots needs new tokens or computed tokens to adopt")
	}
	numNewComputed := len(newComputed) * m.blockSize
	totalComputed := min(computed+numNewComputed, m.maxModelLen)
	needSlots := min(totalComputed+numNew, m.maxModelLen)

	watermark := 0
	if hasScheduled && waiting {
		watermark = m.watermark
	}
	if fullSequenceMustFit {
		full := min(numTokens, m.maxModelLen)
		if m.toAllocate(id, full, newComputed)+watermark > m.Pool.NumFree() {
			return false
		}
	}
	if m.toAllocate(id, needSlots, newComputed)+watermark > m.Pool.NumFree() {
		return false
	}
	if len(newComputed) > 0 {
		if m.caching {
			m.Pool.Touch(newComputed)
		}
		m.reqBlocks[id] = append(m.reqBlocks[id], newComputed...)
		m.numCached[id] = len(m.reqBlocks[id])
	}
	if need := cdiv(needSlots, m.blockSize) - len(m.reqBlocks[id]); need > 0 {
		m.reqBlocks[id] = append(m.reqBlocks[id], m.Pool.New(need)...)
	}
	if m.caching {
		m.cache(id, hashes, min(totalComputed+numNew, numTokens))
	}
	return true
}

// toAllocate is get_num_blocks_to_allocate (single_type_kv_cache_manager.py:141-205) for
// full attention: new blocks, plus prefix-hit blocks that are free (a touch takes them
// out of the free queue).
func (m *Manager) toAllocate(id string, numTokens int, newComputed []*Block) int {
	required := cdiv(numTokens, m.blockSize)
	have := len(m.reqBlocks[id])
	if _, ok := m.numCached[id]; ok {
		return max(required-have, 0)
	}
	newBlocks := max(required-(have+len(newComputed)), 0)
	evictable := 0
	for _, b := range newComputed {
		if b.RefCnt == 0 && !b.Null {
			evictable++
		}
	}
	return newBlocks + evictable
}

// cache records the request's full blocks up to numTokens as prefix-cached
// (cache_blocks, single_type_kv_cache_manager.py:468-516).
func (m *Manager) cache(id string, hashes []BlockHash, numTokens int) {
	numCached := m.numCached[id]
	numFull := numTokens / m.blockSize
	if numCached >= numFull {
		return
	}
	numFull = min(numFull, len(hashes))
	if numCached >= numFull {
		return
	}
	m.Pool.CacheFull(m.reqBlocks[id], hashes, numCached, numFull)
	m.numCached[id] = numFull
}

// Free releases the request's blocks, tail first, so tail blocks are evicted first
// (free, single_type_kv_cache_manager.py:581-589).
func (m *Manager) Free(id string) {
	bs := m.reqBlocks[id]
	rev := make([]*Block, len(bs))
	for i, b := range bs {
		rev[len(bs)-1-i] = b
	}
	m.Pool.Free(rev)
	delete(m.reqBlocks, id)
	delete(m.numCached, id)
}

func cdiv(a, b int) int { return (a + b - 1) / b }
