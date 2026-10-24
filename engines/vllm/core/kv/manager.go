// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/kv_cache_manager.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/core/single_type_kv_cache_manager.py (SingleTypeKVCacheManager,
// FullAttentionManager, SlidingWindowManager) and vllm/v1/core/kv_cache_coordinator.py
// (UnitaryKVCacheCoordinator, HybridKVCacheCoordinator) for full-attention and
// sliding-window KV cache groups over one block pool.

package kv

// GroupKind is a KV cache group's attention type (vllm/v1/kv_cache_interface.py@5df63a72).
type GroupKind int

const (
	// FullAttention is FullAttentionSpec.
	FullAttention GroupKind = iota
	// SlidingWindow is SlidingWindowSpec.
	SlidingWindow
)

// Group is one KV cache group: its attention type and, for a sliding window, the window.
type Group struct {
	Kind GroupKind
	// Window is SlidingWindowSpec.sliding_window, in tokens.
	Window int
}

// ResolveGroups is the grouping of a model's layer specs that the scheduler sees
// (get_kv_cache_groups, vllm/v1/core/kv_cache_utils.py:2159-2243@5df63a72), for layers that
// are full attention or sliding windows of one block size: one group per distinct spec in
// the order first seen, and with disableHybrid (disable_hybrid_kv_cache_manager,
// vllm/config/scheduler.py:182) every sliding window promoted to full attention when full
// attention is present (unify_hybrid_kv_cache_specs, :1799-1822), which leaves one group.
func ResolveGroups(layers []Group, disableHybrid bool) []Group {
	if len(layers) == 0 {
		return nil
	}
	hasFull := false
	for _, l := range layers {
		if l.Kind == FullAttention {
			hasFull = true
		}
	}
	if disableHybrid && hasFull {
		return []Group{{Kind: FullAttention}}
	}
	return layers
}

// Manager is vLLM's KVCacheManager over its KV cache groups: per-request block tables per
// group over one Pool, prefix-cache lookup, slot allocation and freeing. One full-attention
// group is vLLM's UnitaryKVCacheCoordinator; more groups are its HybridKVCacheCoordinator.
type Manager struct {
	Pool        *Pool
	blockSize   int
	maxModelLen int
	caching     bool
	watermark   int // blocks kept free when admitting a waiting request
	// eagle is use_eagle_block_drop: an EAGLE-family drafter's groups drop the last matched
	// block of a prefix hit (single_type_kv_cache_manager.py:819-820) and cache one block past
	// the aligned boundary (kv_cache_coordinator.py:779-794).
	eagle bool
	// reprefillable is num_reprefillable_tokens, max(0, num_prefill_lookahead - 1): the trailing
	// computed tokens a multi-module MTP drafter may re-prefill, never cached
	// (kv_cache_coordinator.py:117, 346-352).
	reprefillable int
	groups        []*group
	// maxInFlight is max_in_flight_tokens, which bounds a sliding window's admission
	// reservation (vllm/v1/kv_cache_interface.py:859-885@5df63a72).
	maxInFlight int
}

// group is one SingleTypeKVCacheManager.
type group struct {
	id        uint32
	spec      Group
	reqBlocks map[string][]*Block
	numCached map[string]int // full blocks of the request already cached
}

// Options are the parts of the KV manager's configuration a drafter or a hybrid model
// changes.
type Options struct {
	// EagleBlockDrop is the scheduler's use_eagle_block_drop
	// (vllm/v1/core/sched/scheduler.py:315-316, 333@5df63a72).
	EagleBlockDrop bool
	// PrefillLookahead is num_prefill_lookahead_tokens (vllm/config/vllm.py:644-665@5df63a72).
	PrefillLookahead int
	// Groups are the KV cache groups; none means one full-attention group.
	Groups []Group
	// MaxInFlightTokens is max_in_flight_tokens (vllm/config/vllm.py:589-598@5df63a72):
	// max_concurrent_batches times max_num_batched_tokens.
	MaxInFlightTokens int
}

// NewManager builds a manager over a fresh pool of numBlocks blocks. watermarkBlocks is the
// free blocks admission must leave: int(watermark * numBlocks) in vLLM.
func NewManager(numBlocks, blockSize, maxModelLen int, enableCaching bool, watermarkBlocks int) *Manager {
	return NewManagerWith(numBlocks, blockSize, maxModelLen, enableCaching, watermarkBlocks, Options{})
}

// NewManagerWith is NewManager with a drafter's or a hybrid model's options.
func NewManagerWith(numBlocks, blockSize, maxModelLen int, enableCaching bool, watermarkBlocks int, o Options) *Manager {
	specs := o.Groups
	if len(specs) == 0 {
		specs = []Group{{Kind: FullAttention}}
	}
	m := &Manager{
		Pool: NewPool(numBlocks, enableCaching), blockSize: blockSize, maxModelLen: maxModelLen,
		caching: enableCaching, watermark: watermarkBlocks, eagle: o.EagleBlockDrop,
		reprefillable: max(0, o.PrefillLookahead-1), maxInFlight: o.MaxInFlightTokens,
	}
	for i, g := range specs {
		m.groups = append(m.groups, &group{id: uint32(i), spec: g, reqBlocks: map[string][]*Block{}, numCached: map[string]int{}})
	}
	return m
}

// BlockSize is tokens per block.
func (m *Manager) BlockSize() int { return m.blockSize }

// NumGroups is the number of KV cache groups.
func (m *Manager) NumGroups() int { return len(m.groups) }

// Blocks returns the request's block table in group 0. The slice is the manager's own.
func (m *Manager) Blocks(id string) []*Block { return m.groups[0].reqBlocks[id] }

// GroupBlocks returns the request's block table in group g. The slice is the manager's own.
func (m *Manager) GroupBlocks(g int, id string) []*Block { return m.groups[g].reqBlocks[id] }

// ComputedBlocks is get_computed_blocks (kv_cache_manager.py:265-319): the longest prefix
// every group can serve from cached blocks, capped so the last token is always recomputed
// (max_cache_hit_length = num_tokens - 1), each group's blocks for it, and the tokens they
// cover. With one group it is the full-attention scan (UnitaryKVCacheCoordinator); with
// several it is HybridKVCacheCoordinator's fixed point (kv_cache_coordinator.py:796-931).
func (m *Manager) ComputedBlocks(hashes []BlockHash, numTokens int) ([][]*Block, int) {
	if !m.caching {
		return make([][]*Block, len(m.groups)), 0
	}
	maxLen := numTokens - 1
	if len(m.groups) == 1 {
		hit := m.fullHit(hashes, maxLen, []int{0}, m.eagle)
		return hit, len(hit[0]) * m.blockSize
	}
	return m.hybridHit(hashes, maxLen)
}

// specGroup is the coordinator's SpecGroup (kv_cache_coordinator.py:594-607): groups with
// one spec, looked up together.
type specGroup struct {
	spec Group
	ids  []int
}

// specGroups is verify_and_split_kv_cache_groups (kv_cache_coordinator.py:713-765): groups
// with equal specs merged, full attention first.
func (m *Manager) specGroups() []specGroup {
	var out []specGroup
	for i, g := range m.groups {
		found := false
		for j := range out {
			if out[j].spec == g.spec {
				out[j].ids = append(out[j].ids, i)
				found = true
				break
			}
		}
		if !found {
			out = append(out, specGroup{spec: g.spec, ids: []int{i}})
		}
	}
	// A stable sort with full attention first (:745-749).
	var fulls, others []specGroup
	for _, sg := range out {
		if sg.spec.Kind == FullAttention {
			fulls = append(fulls, sg)
		} else {
			others = append(others, sg)
		}
	}
	return append(fulls, others...)
}

// hybridHit is HybridKVCacheCoordinator.find_longest_cache_hit (kv_cache_coordinator.py:
// 796-931): each spec group accepts the candidate length or shortens it, and when one
// shortens it every group is checked again, until none does. Full attention is looked up
// once and then only trimmed; an EAGLE group matches one block past the candidate before its
// drop, and its drop is redone after any group shortens the length (issue #32802).
func (m *Manager) hybridHit(hashes []BlockHash, maxLen int) ([][]*Block, int) {
	sgs := m.specGroups()
	n := len(m.groups)
	hitLength := maxLen
	blocks := make([][]*Block, n)
	have := make([]bool, n)
	lengths := make([]int, n)
	simple := len(sgs) == 2 && sgs[0].spec.Kind == FullAttention
	eagleVerified := map[int]bool{}
	for {
		curr := hitLength
		for idx, sg := range sgs {
			first := sg.ids[0]
			if sg.spec.Kind == FullAttention && have[first] {
				curr = min(curr, lengths[first])
				continue
			}
			drop := m.eagle && !eagleVerified[idx]
			maxScan := curr
			if drop {
				// eagle_proof_margin: one block (kv_cache_utils.py:792-801).
				maxScan = min(curr+m.blockSize, len(hashes)*m.blockSize)
			}
			var hit [][]*Block
			var got int
			if sg.spec.Kind == FullAttention {
				hit = m.fullHit(hashes, maxScan, sg.ids, drop)
				got = len(hit[0]) * m.blockSize
			} else {
				hit, got = m.windowHit(hashes, maxScan, sg, drop)
			}
			if drop {
				eagleVerified[idx] = true
			} else if got < curr {
				eagleVerified = map[int]bool{}
			}
			curr = got
			for j, gi := range sg.ids {
				blocks[gi], have[gi], lengths[gi] = hit[j], true, got
			}
		}
		if curr >= hitLength {
			break
		}
		hitLength = curr
		if simple {
			break
		}
	}
	// Full attention is cut to the final length (:906-919).
	for _, sg := range sgs {
		if sg.spec.Kind != FullAttention {
			continue
		}
		for _, gi := range sg.ids {
			if len(blocks[gi]) > cdiv(hitLength, m.blockSize) {
				blocks[gi] = blocks[gi][:cdiv(hitLength, m.blockSize)]
			}
		}
	}
	return blocks, hitLength
}

// fullHit is FullAttentionManager.find_longest_cache_hit (single_type_kv_cache_manager.py:
// 733-827) over full-attention groups at once: a block matches only if every group caches
// its hash (get_cached_block, block_pool.py:198-224); then, with drop, the EAGLE drop.
func (m *Manager) fullHit(hashes []BlockHash, maxLen int, gis []int, drop bool) [][]*Block {
	out := make([][]*Block, len(gis))
	for i := 0; i < len(hashes) && i < maxLen/m.blockSize; i++ {
		row := make([]*Block, 0, len(gis))
		for _, gi := range gis {
			b := m.Pool.CachedIn(hashes[i], m.groups[gi].id)
			if b == nil {
				row = nil
				break
			}
			row = append(row, b)
		}
		if row == nil {
			break
		}
		for j := range gis {
			out[j] = append(out[j], row[j])
		}
	}
	if drop && len(out[0]) > 0 {
		for j := range out {
			out[j] = out[j][:len(out[j])-1]
		}
	}
	return out
}

// windowHit is SlidingWindowManager.find_longest_cache_hit (single_type_kv_cache_manager.py:
// 955-1051) with block_size equal to the alignment, over the spec group's groups at once:
// scanning right to left for the window's contiguous cached blocks, with null blocks below
// the window; with drop, one block more is needed and then dropped.
func (m *Manager) windowHit(hashes []BlockHash, maxLen int, sg specGroup, drop bool) ([][]*Block, int) {
	need := cdiv(sg.spec.Window-1, m.blockSize) // _contiguous_blocks_for_hit, :943-955
	if drop {
		need++
	}
	maxBlocks := min(maxLen/m.blockSize, len(hashes))
	computed := make([][]*Block, len(sg.ids))
	for j := range computed {
		computed[j] = make([]*Block, maxBlocks)
		for i := range computed[j] {
			computed[j][i] = m.Pool.null
		}
	}
	contiguous, found := 0, false
	for i := maxBlocks - 1; i >= 0; i-- {
		row := make([]*Block, 0, len(sg.ids))
		for _, gi := range sg.ids {
			b := m.Pool.CachedIn(hashes[i], m.groups[gi].id)
			if b == nil {
				row = nil
				break
			}
			row = append(row, b)
		}
		if row == nil {
			contiguous = 0
			continue
		}
		for j := range sg.ids {
			computed[j][i] = row[j]
		}
		contiguous++
		if contiguous >= need {
			for j := range computed {
				computed[j] = computed[j][:i+contiguous]
			}
			found = true
			break
		}
	}
	if !found {
		for j := range computed {
			computed[j] = computed[j][:contiguous]
		}
	}
	if drop && len(computed[0]) > 0 {
		for j := range computed {
			computed[j] = computed[j][:len(computed[j])-1]
		}
	}
	return computed, len(computed[0]) * m.blockSize
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
	// NewComputed are the prefix-hit blocks to adopt, one list per group.
	NewComputed [][]*Block
	// NumNewComputed is the tokens the prefix hit covers (num_new_computed_tokens).
	NumNewComputed int
	// Hashes are the request's block hashes so far.
	Hashes []BlockHash
	// NumTokens is the request's num_tokens (prompt plus output, drafts excluded): caching
	// stops there, so an unverified draft is never cached.
	NumTokens int
	// InFlight is the request's num_in_flight_tokens: blocks a step still reads are freed
	// only once it settles (kv_cache_manager.py:545-557).
	InFlight int
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

// AllocateSlots is allocate_slots (kv_cache_manager.py:353-611) for one full-attention
// group.
func (m *Manager) AllocateSlots(id string, computed, numNew int, newComputed []*Block, hashes []BlockHash,
	numTokens int, waiting, hasScheduled, fullSequenceMustFit bool) bool {
	return m.Allocate(id, Alloc{Computed: computed, New: numNew, NewComputed: [][]*Block{newComputed},
		NumNewComputed: len(newComputed) * m.blockSize, Hashes: hashes,
		NumTokens: numTokens, Waiting: waiting, HasScheduled: hasScheduled, FullSequenceMustFit: fullSequenceMustFit})
}

// Allocate is allocate_slots for one Alloc.
func (m *Manager) Allocate(id string, a Alloc) bool {
	newComputed := make([][]*Block, len(m.groups))
	copy(newComputed, a.NewComputed)
	adopting := 0
	for _, bs := range newComputed {
		adopting += len(bs)
	}
	if a.New == 0 && adopting == 0 {
		panic("kv: allocate_slots needs new tokens or computed tokens to adopt")
	}
	localComputed := a.Computed + a.NumNewComputed
	totalComputed := min(localComputed, m.maxModelLen)

	watermark := 0
	if a.HasScheduled && a.Waiting {
		watermark = m.watermark
	}
	if a.FullSequenceMustFit {
		full := min(a.NumTokens, m.maxModelLen)
		if m.toAllocate(id, full, newComputed, totalComputed, true)+watermark > m.Pool.NumFree() {
			return false
		}
	}
	mainModel := totalComputed + a.New
	needSlots := min(mainModel+a.Lookahead, m.maxModelLen)
	// Out-of-window blocks are freed first, even if the allocation then fails, on the
	// processed-token basis (:545-557).
	m.removeSkipped(id, max(0, totalComputed-a.InFlight))
	if m.toAllocate(id, needSlots, newComputed, localComputed, false)+watermark > m.Pool.NumFree() {
		return false
	}
	// Adopt the prefix hit in every group before allocating any new block (:577-590,
	// kv_cache_coordinator.py:261-302); a running request has none to adopt.
	if adopting > 0 && !m.isRunning(id) {
		for i, g := range m.groups {
			bs := newComputed[i]
			skipped := g.skippedTokens(localComputed) / m.blockSize
			if skipped > 0 {
				bs = bs[min(skipped, len(bs)):]
			}
			if m.caching {
				m.Pool.Touch(bs)
			}
			blocks := make([]*Block, 0, skipped+len(bs))
			for j := 0; j < skipped; j++ {
				blocks = append(blocks, m.Pool.null)
			}
			g.reqBlocks[id] = append(blocks, bs...)
			g.numCached[id] = len(g.reqBlocks[id])
		}
	}
	for _, g := range m.groups {
		if need := cdiv(needSlots, m.blockSize) - len(g.reqBlocks[id]); need > 0 {
			g.reqBlocks[id] = append(g.reqBlocks[id], m.Pool.New(need)...)
		}
	}
	if m.caching {
		for _, g := range m.groups {
			m.cache(g, id, a.Hashes, m.cacheable(min(mainModel, a.NumTokens)), a.Run)
		}
	}
	return true
}

func (m *Manager) isRunning(id string) bool {
	for _, g := range m.groups {
		if _, ok := g.numCached[id]; ok {
			return true
		}
	}
	return false
}

// cacheable is get_num_cacheable_tokens (kv_cache_coordinator.py:346-352 for one group,
// :779-794 for several): the computed tokens less the ones a multi-module MTP drafter may
// re-prefill; a hybrid coordinator also rounds down to the block, and an EAGLE group keeps
// one block past it.
func (m *Manager) cacheable(numTokens int) int {
	if len(m.groups) == 1 {
		return max(0, numTokens-m.reprefillable)
	}
	aligned := numTokens / m.blockSize * m.blockSize
	if m.eagle && aligned > 0 {
		finalized := max(0, numTokens-m.reprefillable)
		return min(finalized, finalized/m.blockSize*m.blockSize+m.blockSize)
	}
	return aligned
}

// skippedTokens is get_num_skipped_tokens (single_type_kv_cache_manager.py:712-723 and
// :1135-1171): the tokens below a sliding window, none for full attention.
func (g *group) skippedTokens(computed int) int {
	if g.spec.Kind != SlidingWindow {
		return 0
	}
	return max(0, computed-g.spec.Window+1)
}

// removeSkipped is remove_skipped_blocks (single_type_kv_cache_manager.py:672-710): each
// sliding-window group frees the blocks wholly below its window, tail first, into null slots
// (_remove_blocks_in_range, :645-670).
func (m *Manager) removeSkipped(id string, processed int) {
	for _, g := range m.groups {
		n := g.skippedTokens(processed)
		if n <= 0 {
			continue
		}
		bs, ok := g.reqBlocks[id]
		if !ok {
			continue
		}
		last := min(n/m.blockSize, len(bs))
		var freed []*Block
		for i := last - 1; i >= 0; i-- {
			if bs[i] == m.Pool.null {
				break
			}
			freed = append(freed, bs[i])
			bs[i] = m.Pool.null
		}
		if len(freed) > 0 {
			m.Pool.Free(freed)
		}
	}
}

// toAllocate is the coordinator's get_num_blocks_to_allocate (kv_cache_coordinator.py:
// 186-259) summed over groups, each group's single_type_kv_cache_manager.py:174-251: new
// blocks, plus prefix-hit blocks that are free (a touch takes them out of the free queue),
// not counting what a sliding window skips. admission applies a sliding window's
// recycling-aware cap, as the full-sequence gate does. A running request may need fewer
// blocks than it holds, after rejected drafts (:220-223); it then needs none.
func (m *Manager) toAllocate(id string, numTokens int, newComputed [][]*Block, totalComputed int, admission bool) int {
	total := 0
	for i, g := range m.groups {
		required := cdiv(numTokens, m.blockSize)
		if admission && g.spec.Kind == SlidingWindow {
			// max_admission_blocks_per_request (kv_cache_interface.py:859-885).
			capTokens := min(g.spec.Window-1+m.maxInFlight, m.maxModelLen)
			required = min(required, cdiv(capTokens, m.blockSize)+1)
		}
		have := len(g.reqBlocks[id])
		if _, ok := g.numCached[id]; ok {
			total += max(required-have, 0)
			continue
		}
		hit := newComputed[i]
		skippedBlocks := g.skippedTokens(totalComputed) / m.blockSize
		local := len(hit) + have
		newBlocks := max(required-max(skippedBlocks, local), 0)
		evictable := 0
		for _, b := range hit[min(max(0, skippedBlocks-have), len(hit)):] {
			if b.RefCnt == 0 && !b.Null {
				evictable++
			}
		}
		total += newBlocks + evictable
	}
	return total
}

// cache records the request's full blocks up to numTokens as prefix-cached in one group
// (cache_blocks, single_type_kv_cache_manager.py:468-516).
func (m *Manager) cache(g *group, id string, hashes []BlockHash, numTokens int, run StoredRun) {
	numCached := g.numCached[id]
	numFull := numTokens / m.blockSize
	if numCached >= numFull {
		return
	}
	numFull = min(numFull, len(hashes))
	if numCached >= numFull {
		return
	}
	m.Pool.CacheFullIn(g.id, g.reqBlocks[id], hashes, numCached, numFull, run, m.blockSize)
	g.numCached[id] = numFull
}

// Free releases the request's blocks, each group's tail first, so tail blocks are evicted
// first (free, single_type_kv_cache_manager.py:581-589; kv_cache_coordinator.py:400-408).
func (m *Manager) Free(id string) {
	for _, g := range m.groups {
		bs := g.reqBlocks[id]
		rev := make([]*Block, len(bs))
		for i, b := range bs {
			rev[len(bs)-1-i] = b
		}
		m.Pool.Free(rev)
		delete(g.reqBlocks, id)
		delete(g.numCached, id)
	}
}

func cdiv(a, b int) int { return (a + b - 1) / b }
