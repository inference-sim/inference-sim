// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/kv_cache_manager.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/core/single_type_kv_cache_manager.py (SingleTypeKVCacheManager,
// FullAttentionManager) and vllm/v1/core/kv_cache_coordinator.py (UnitaryKVCacheCoordinator),
// for one full-attention KV cache group.

package kv

// Manager is vLLM's KVCacheManager for a single full-attention group: per-request block
// tables over a Pool, prefix-cache lookup, slot allocation and freeing.
type Manager struct {
	Pool        *Pool
	blockSize   int
	maxModelLen int
	caching     bool
	watermark   int // blocks kept free when admitting a waiting request
	// eagle is use_eagle_block_drop: an EAGLE-family drafter's group drops the last matched
	// block of a prefix hit (single_type_kv_cache_manager.py:819-820) and caches one block past
	// the aligned boundary (kv_cache_coordinator.py:779-794).
	eagle bool
	// reprefillable is num_reprefillable_tokens, max(0, num_prefill_lookahead - 1): the trailing
	// computed tokens a multi-module MTP drafter may re-prefill, never cached
	// (kv_cache_coordinator.py:117, 346-352).
	reprefillable int
	reqBlocks     map[string][]*Block
	numCached     map[string]int // full blocks of the request already cached
}

// Options are the parts of the KV manager's configuration a drafter changes.
type Options struct {
	// EagleBlockDrop is the scheduler's use_eagle_block_drop
	// (vllm/v1/core/sched/scheduler.py:315-316, 333@5df63a72).
	EagleBlockDrop bool
	// PrefillLookahead is num_prefill_lookahead_tokens (vllm/config/vllm.py:644-665@5df63a72).
	PrefillLookahead int
}

// NewManager builds a manager over a fresh pool of numBlocks blocks. watermarkBlocks is the
// free blocks admission must leave: int(watermark * numBlocks) in vLLM.
func NewManager(numBlocks, blockSize, maxModelLen int, enableCaching bool, watermarkBlocks int) *Manager {
	return NewManagerWith(numBlocks, blockSize, maxModelLen, enableCaching, watermarkBlocks, Options{})
}

// NewManagerWith is NewManager with a drafter's options.
func NewManagerWith(numBlocks, blockSize, maxModelLen int, enableCaching bool, watermarkBlocks int, o Options) *Manager {
	return &Manager{
		Pool: NewPool(numBlocks, enableCaching), blockSize: blockSize, maxModelLen: maxModelLen,
		caching: enableCaching, watermark: watermarkBlocks, eagle: o.EagleBlockDrop,
		reprefillable: max(0, o.PrefillLookahead-1),
		reqBlocks:     map[string][]*Block{}, numCached: map[string]int{},
	}
}

// BlockSize is tokens per block.
func (m *Manager) BlockSize() int { return m.blockSize }

// Blocks returns the request's block table. The slice is the manager's own.
func (m *Manager) Blocks(id string) []*Block { return m.reqBlocks[id] }

// ComputedBlocks is get_computed_blocks (kv_cache_manager.py:265-310): the longest run of
// cached blocks matching the request's block hashes, capped so the last token is always
// recomputed (max_cache_hit_length = num_tokens - 1), and the tokens they cover. An EAGLE
// group drops the last matched block (find_longest_cache_hit,
// single_type_kv_cache_manager.py:799-827): the drafter needs the tokens before the
// generation point recomputed.
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
	if m.eagle && len(hit) > 0 {
		hit = hit[:len(hit)-1]
	}
	return hit, len(hit) * m.blockSize
}

// Alloc is one allocate_slots call (kv_cache_manager.py:353-617).
type Alloc struct {
	// Computed is the request's num_computed_tokens before the call.
	Computed int
	// New is num_new_tokens: the tokens to compute this step, drafts included.
	New int
	// Lookahead is num_lookahead_tokens: slots past the scheduled tokens a drafter writes
	// (vllm/config/vllm.py:615-641@5df63a72).
	Lookahead int
	// NewComputed are the prefix-hit blocks to adopt.
	NewComputed []*Block
	// Hashes are the request's block hashes so far.
	Hashes []BlockHash
	// NumTokens is the request's num_tokens (prompt plus output, drafts excluded): caching
	// stops there, so an unverified draft is never cached.
	NumTokens int
	// Run is what a BlockStored event reports (its tokens and adapter); used only with
	// events on.
	Run StoredRun
	// Waiting is whether the request is WAITING or PREEMPTED, which with HasScheduled
	// applies the watermark.
	Waiting, HasScheduled bool
	// FullSequenceMustFit is scheduler_reserve_full_isl (vllm/config/scheduler.py:190,
	// default True; kv_cache_manager.py:514-541): admit a new request only if its whole
	// sequence fits.
	FullSequenceMustFit bool
}

// AllocateSlots is allocate_slots (kv_cache_manager.py:353-611) without a connector or
// encoder: it adds the prefix-hit blocks (if any) and new blocks so the request has slots
// for computed+newComputed+numNew tokens, then caches its full blocks. It returns false,
// changing nothing, when the blocks do not fit beside the watermark.
func (m *Manager) AllocateSlots(id string, computed, numNew int, newComputed []*Block, hashes []BlockHash,
	numTokens int, waiting, hasScheduled, fullSequenceMustFit bool) bool {
	return m.Allocate(id, Alloc{Computed: computed, New: numNew, NewComputed: newComputed, Hashes: hashes,
		NumTokens: numTokens, Waiting: waiting, HasScheduled: hasScheduled, FullSequenceMustFit: fullSequenceMustFit})
}

// Allocate is allocate_slots for one Alloc.
func (m *Manager) Allocate(id string, a Alloc) bool {
	if a.New == 0 && len(a.NewComputed) == 0 {
		panic("kv: allocate_slots needs new tokens or computed tokens to adopt")
	}
	numNewComputed := len(a.NewComputed) * m.blockSize
	totalComputed := min(a.Computed+numNewComputed, m.maxModelLen)

	watermark := 0
	if a.HasScheduled && a.Waiting {
		watermark = m.watermark
	}
	if a.FullSequenceMustFit {
		full := min(a.NumTokens, m.maxModelLen)
		if m.toAllocate(id, full, a.NewComputed)+watermark > m.Pool.NumFree() {
			return false
		}
	}
	mainModel := totalComputed + a.New
	needSlots := min(mainModel+a.Lookahead, m.maxModelLen)
	if m.toAllocate(id, needSlots, a.NewComputed)+watermark > m.Pool.NumFree() {
		return false
	}
	if len(a.NewComputed) > 0 {
		if m.caching {
			m.Pool.Touch(a.NewComputed)
		}
		m.reqBlocks[id] = append(m.reqBlocks[id], a.NewComputed...)
		m.numCached[id] = len(m.reqBlocks[id])
	}
	if need := cdiv(needSlots, m.blockSize) - len(m.reqBlocks[id]); need > 0 {
		m.reqBlocks[id] = append(m.reqBlocks[id], m.Pool.New(need)...)
	}
	if m.caching {
		m.cache(id, a.Hashes, m.cacheable(min(mainModel, a.NumTokens)), a.Run)
	}
	return true
}

// cacheable is get_num_cacheable_tokens for the single full-attention group
// (UnitaryKVCacheCoordinator inherits KVCacheCoordinator's, kv_cache_coordinator.py:346-352):
// the computed tokens less the ones a multi-module MTP drafter may re-prefill.
func (m *Manager) cacheable(numTokens int) int {
	return max(0, numTokens-m.reprefillable)
}

// toAllocate is get_num_blocks_to_allocate (single_type_kv_cache_manager.py:174-251) for
// full attention: new blocks, plus prefix-hit blocks that are free (a touch takes them
// out of the free queue). A running request may need fewer blocks than it holds, after
// rejected drafts (:220-223); it then needs none.
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
func (m *Manager) cache(id string, hashes []BlockHash, numTokens int, run StoredRun) {
	numCached := m.numCached[id]
	numFull := numTokens / m.blockSize
	if numCached >= numFull {
		return
	}
	numFull = min(numFull, len(hashes))
	if numCached >= numFull {
		return
	}
	m.Pool.CacheFull(m.reqBlocks[id], hashes, numCached, numFull, run, m.blockSize)
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
