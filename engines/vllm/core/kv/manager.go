// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/kv_cache_manager.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/core/single_type_kv_cache_manager.py (SingleTypeKVCacheManager,
// FullAttentionManager, SlidingWindowManager) and vllm/v1/core/kv_cache_coordinator.py
// (UnitaryKVCacheCoordinator, HybridKVCacheCoordinator) for full-attention and
// sliding-window KV cache groups over one block pool, and CircularBufferManager
// (single_type_kv_cache_manager.py:1173-1260) and groups of their own block sizes
// (resolve_kv_cache_block_sizes, vllm/v1/core/kv_cache_utils.py:804-904; BlockHashListWithBlockSize,
// :2690-2763) for DeepSeek V4's packed groups.

package kv

import "fmt"

// GroupKind is a KV cache group's attention type (vllm/v1/kv_cache_interface.py@5df63a72).
type GroupKind int

const (
	// FullAttention is FullAttentionSpec.
	FullAttention GroupKind = iota
	// SlidingWindow is SlidingWindowSpec.
	SlidingWindow
	// SlidingWindowMLA is SlidingWindowMLASpec, DeepSeek V4's sliding-window caches; vLLM
	// manages it with the SlidingWindowManager (single_type_kv_cache_manager.py:2663-2667).
	SlidingWindowMLA
	// CircularBuffer is CircularBufferSpec, one ring block per request (CircularBufferManager,
	// single_type_kv_cache_manager.py:1173-1260).
	CircularBuffer
)

// Group is one KV cache group: its attention type and, for a sliding window, the window.
type Group struct {
	Kind GroupKind
	// Window is SlidingWindowSpec.sliding_window, in tokens.
	Window int
	// BlockSize is the group's own block size, in tokens, when it differs from the engine's
	// (_get_packed_kv_cache_groups' groups, vllm/v1/core/kv_cache_utils.py:1860-2014@5df63a72);
	// 0 is the engine's.
	BlockSize int
	// BoundedReplay is a sliding-window MLA group kept out of prefix caching
	// (SlidingWindowMLASpec.prefix_cacheable, vllm/v1/kv_cache_interface.py:1003-1009@5df63a72).
	BoundedReplay bool
	// Eagle is KVCacheGroupSpec.is_eagle_group: the group holds a drafter's layers
	// (_annotate_eagle_groups, vllm/v1/core/kv_cache_utils.py:2051-2109@5df63a72). With EAGLE block
	// drop on, only such groups drop the block; when none is, every group does
	// (kv_cache_coordinator.py:127-133).
	Eagle bool
}

// ResolveGroups is the grouping of a model's layer specs that the scheduler sees
// (get_kv_cache_groups, vllm/v1/core/kv_cache_utils.py:2159-2243@5df63a72), for layers that
// are full attention or sliding windows of one block size. A model whose layers all share
// one spec is one group (_get_kv_cache_groups_uniform_spec, :2185-2189); with disableHybrid
// (disable_hybrid_kv_cache_manager, vllm/config/scheduler.py:182) every sliding window is
// first promoted to full attention when full attention is present
// (unify_hybrid_kv_cache_specs, :1799-1822), which leaves one group. A hybrid model's layers
// are taken as its groups, one each, in order: vLLM's split of a hybrid model into
// equal-size groups (_get_kv_cache_groups_uniform_page_size, :1362-1502) gives that for one
// layer per attention type, which is what the scenarios state.
func ResolveGroups(layers []Group, disableHybrid bool) []Group {
	if len(layers) == 0 {
		return nil
	}
	hasFull, uniform := false, true
	for _, l := range layers {
		if l.Kind == FullAttention {
			hasFull = true
		}
		if l != layers[0] {
			uniform = false
		}
	}
	if disableHybrid && hasFull {
		return []Group{{Kind: FullAttention}}
	}
	if uniform {
		return []Group{layers[0]}
	}
	return layers
}

// Manager is vLLM's KVCacheManager over its KV cache groups: per-request block tables per
// group over one Pool, prefix-cache lookup, slot allocation and freeing. One full-attention
// group is vLLM's UnitaryKVCacheCoordinator; more groups are its HybridKVCacheCoordinator.
type Manager struct {
	Pool *Pool
	// blockSize is the scheduler block size: the engine's block size for one group, the LCM of
	// the groups' block sizes for several (resolve_kv_cache_block_sizes, kv_cache_utils.py:
	// 804-861@5df63a72). Hits and the hybrid coordinator's caching are aligned to it.
	blockSize int
	// hashBlockSize is the block size request block hashes are computed at: the GCD of the
	// prefix-cacheable groups' block sizes (:863-890); the scheduler block size for one group.
	hashBlockSize int
	maxModelLen   int
	caching       bool
	watermark     int // blocks kept free when admitting a waiting request
	// eagle is use_eagle_block_drop: an EAGLE-family drafter's groups drop the last matched
	// block of a prefix hit (single_type_kv_cache_manager.py:819-820) and cache one block past
	// the aligned boundary (kv_cache_coordinator.py:779-794).
	eagle bool
	// reprefillable is num_reprefillable_tokens, max(0, num_prefill_lookahead - 1): the trailing
	// computed tokens a multi-module MTP drafter may re-prefill, never cached
	// (kv_cache_coordinator.py:117, 346-352). The same count is every sliding window's
	// extra_retained_tokens (get_kv_cache_configs, kv_cache_utils.py:2582-2589): tokens kept
	// below the window so the drafter's re-prefill can still read them.
	reprefillable int
	groups        []*group
	// maxInFlight is max_in_flight_tokens, which bounds a sliding window's admission
	// reservation (vllm/v1/kv_cache_interface.py:859-885@5df63a72).
	maxInFlight int
}

// group is one SingleTypeKVCacheManager.
type group struct {
	id   uint32
	spec Group
	// blockSize is the group's own block size (SingleTypeKVCacheManager.block_size,
	// single_type_kv_cache_manager.py:108@5df63a72).
	blockSize int
	reqBlocks map[string][]*Block
	numCached map[string]int // full blocks of the request already cached
	// extraRetained is a sliding window's extra_retained_tokens
	// (vllm/v1/kv_cache_interface.py:852-857@5df63a72), 0 for full attention.
	extraRetained int
	// eagle is the manager's use_eagle: the group drops the EAGLE block
	// (kv_cache_coordinator.py:127-133, 572, 760-765@5df63a72).
	eagle bool
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
		bs := g.BlockSize
		if bs == 0 {
			bs = blockSize
		}
		gr := &group{id: uint32(i), spec: g, blockSize: bs, reqBlocks: map[string][]*Block{}, numCached: map[string]int{}}
		if g.Kind == SlidingWindow || g.Kind == SlidingWindowMLA {
			gr.extraRetained = m.reprefillable
		}
		m.groups = append(m.groups, gr)
	}
	m.blockSize, m.hashBlockSize = blockSizes(specs, blockSize, enableCaching)
	// eagle_group_ids: the groups flagged as the drafter's, or every group when none is
	// (kv_cache_coordinator.py:127-133@5df63a72).
	flagged := false
	for _, g := range specs {
		flagged = flagged || g.Eagle
	}
	for _, g := range m.groups {
		g.eagle = m.eagle && (g.spec.Eagle || !flagged)
	}
	// The coordinator then propagates the bit to every member of a spec group holding a
	// drafter's group (verify_and_split_kv_cache_groups, kv_cache_coordinator.py:726-736 and
	// 761-765@5df63a72): a target group of the drafter group's spec drops the EAGLE block too.
	for _, sg := range m.specGroups() {
		if sg.eagle {
			for _, i := range sg.ids {
				m.groups[i].eagle = true
			}
		}
	}
	return m
}

// blockSizes is resolve_kv_cache_block_sizes (vllm/v1/core/kv_cache_utils.py:804-904@5df63a72)
// without DCP or a connector: one group runs at the engine's block size; several at the LCM of
// their block sizes, with hashes at the GCD of the prefix-cacheable ones' when prefix caching
// is on, at the LCM when it is off (:856-861).
func blockSizes(groups []Group, engine int, caching bool) (scheduler, hash int) {
	if len(groups) <= 1 {
		return engine, engine
	}
	scheduler, hash = 1, 0
	for _, g := range groups {
		bs := g.BlockSize
		if bs == 0 {
			bs = engine
		}
		scheduler = scheduler / gcd(scheduler, bs) * bs
		if g.cacheable() {
			hash = gcd(hash, bs)
		}
	}
	if !caching || hash == 0 {
		return scheduler, scheduler
	}
	return scheduler, hash
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// cacheable is the group spec's prefix_cacheable: a ring never is (CircularBufferSpec,
// vllm/v1/kv_cache_interface.py:939-941), nor a bounded-replay window (:1003-1005@5df63a72).
func (g Group) cacheable() bool {
	return g.Kind != CircularBuffer && !g.BoundedReplay
}

// sameSpec is whether two groups have one spec, the drafter flag aside.
func (g Group) sameSpec(o Group) bool {
	g.Eagle, o.Eagle = false, false
	return g == o
}

// window reports whether a group is a sliding window, of either spec, managed by the
// SlidingWindowManager (single_type_kv_cache_manager.py:2655-2667@5df63a72).
func (g Group) window() bool { return g.Kind == SlidingWindow || g.Kind == SlidingWindowMLA }

// BlockSize is the scheduler block size, in tokens: the engine's block size for one group.
func (m *Manager) BlockSize() int { return m.blockSize }

// HashBlockSize is the block size request block hashes are computed at.
func (m *Manager) HashBlockSize() int { return m.hashBlockSize }

// SpecDecodeStepBlocks is _spec_decode_step_blocks (vllm/v1/core/sched/scheduler.py:346-357
// @5df63a72): the blocks one spec-decode step's tokens take in every group, at each group's
// block size.
func (m *Manager) SpecDecodeStepBlocks(tokens int) int {
	n := 0
	for _, g := range m.groups {
		n += cdiv(tokens, g.blockSize)
	}
	return n
}

// NumGroups is the number of KV cache groups.
func (m *Manager) NumGroups() int { return len(m.groups) }

// PrefixReplayTokens is the scheduler's prefix_replay_tokens (vllm/v1/core/sched/scheduler.py:
// 290-305@5df63a72): the window a bounded-replay group rebuilds after a prefix hit by recomputing
// its trailing tokens (SlidingWindowMLASpec.prefix_replay_tokens, vllm/v1/kv_cache_interface.py:
// 972-974@5df63a72), 0 when no group replays. Every such group states one window.
func (m *Manager) PrefixReplayTokens() int {
	window := 0
	for _, g := range m.groups {
		if g.spec.Kind == SlidingWindowMLA && g.spec.BoundedReplay {
			if window != 0 && g.spec.Window != window {
				panic(fmt.Sprintf("kv: prefix replay windows should agree: %d and %d", window, g.spec.Window))
			}
			window = g.spec.Window
		}
	}
	return window
}

// Blocks returns the request's block table in group 0. The slice is the manager's own.
func (m *Manager) Blocks(id string) []*Block { return m.groups[0].reqBlocks[id] }

// GroupBlocks returns the request's block table in group g. The slice is the manager's own.
func (m *Manager) GroupBlocks(g int, id string) []*Block { return m.groups[g].reqBlocks[id] }

// ComputedBlocks is get_computed_blocks (kv_cache_manager.py:265-319): the longest prefix
// every group can serve from cached blocks, capped so the last token is always recomputed
// (max_cache_hit_length = num_tokens - 1), each group's blocks for it, and the tokens they
// cover. With one group it is that group's own lookup (UnitaryKVCacheCoordinator,
// kv_cache_coordinator.py:574-591): the full-attention scan, or for a model whose every layer
// is one sliding window, the window's; with several it is HybridKVCacheCoordinator's fixed
// point (kv_cache_coordinator.py:796-931).
func (m *Manager) ComputedBlocks(hashes []BlockHash, numTokens int) ([][]*Block, int) {
	if !m.caching {
		return make([][]*Block, len(m.groups)), 0
	}
	maxLen := numTokens - 1
	if len(m.groups) == 1 {
		if g := m.groups[0]; g.spec.window() {
			return m.windowHit(hashes, maxLen, specGroup{spec: g.spec, ids: []int{0}}, g.eagle)
		}
		hit := m.fullHit(hashes, maxLen, []int{0}, m.groups[0].eagle)
		return hit, len(hit[0]) * m.blockSize
	}
	return m.hybridHit(hashes, maxLen)
}

// groupHashes is the request's block hashes at a group's block size: the hash at the last
// hash_block_size boundary of each of the group's blocks, which chains over its whole prefix
// (BlockHashListWithBlockSize, kv_cache_utils.py:2690-2763; resolve_block_hashes,
// :2766-2796@5df63a72).
func (m *Manager) groupHashes(hashes []BlockHash, blockSize int) []BlockHash {
	scale := blockSize / m.hashBlockSize
	if scale <= 1 {
		return hashes
	}
	out := make([]BlockHash, len(hashes)/scale)
	for i := range out {
		out[i] = hashes[(i+1)*scale-1]
	}
	return out
}

// specGroup is the coordinator's SpecGroup (kv_cache_coordinator.py:594-607): groups with
// one spec, looked up together.
type specGroup struct {
	spec  Group
	ids   []int
	eagle bool
}

// specGroups is verify_and_split_kv_cache_groups (kv_cache_coordinator.py:713-765): groups
// with equal specs merged, full attention first.
func (m *Manager) specGroups() []specGroup {
	var out []specGroup
	for i, g := range m.groups {
		// A group that opts out of prefix caching (a ring) takes no part in the lookup
		// (:719-725@5df63a72); its slot of the hit stays empty.
		if !g.spec.cacheable() {
			continue
		}
		found := false
		for j := range out {
			// Specs compare without the drafter flag, which is the group's, not its spec's
			// (SpecGroup's spec, :726-736); the merged group drops if any member does.
			if out[j].spec.sameSpec(g.spec) {
				out[j].ids = append(out[j].ids, i)
				out[j].eagle = out[j].eagle || g.eagle
				found = true
				break
			}
		}
		if !found {
			out = append(out, specGroup{spec: g.spec, ids: []int{i}, eagle: g.eagle})
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
			drop := sg.eagle && !eagleVerified[idx]
			maxScan := curr
			if drop {
				// eagle_proof_margin: one of the group's blocks (kv_cache_utils.py:792-801).
				maxScan = min(curr+m.groups[first].blockSize, len(hashes)*m.hashBlockSize)
			}
			var hit [][]*Block
			var got int
			if sg.spec.Kind == FullAttention {
				hit = m.fullHit(hashes, maxScan, sg.ids, drop)
				got = len(hit[0]) * m.groups[first].blockSize
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
			if n := cdiv(hitLength, m.groups[gi].blockSize); len(blocks[gi]) > n {
				blocks[gi] = blocks[gi][:n]
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
	bs := m.groups[gis[0]].blockSize
	hashes = m.groupHashes(hashes, bs)
	for i := 0; i < len(hashes) && i < maxLen/bs; i++ {
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
	// The hit is rounded down to the alignment, the scheduler block size (:821-826).
	if n := len(out[0]) * bs / m.blockSize * m.blockSize / bs; n < len(out[0]) {
		for j := range out {
			out[j] = out[j][:n]
		}
	}
	return out
}

// windowHit is SlidingWindowManager.find_longest_cache_hit (single_type_kv_cache_manager.py:
// 955-1051) with block_size equal to the alignment, over the spec group's groups at once:
// scanning right to left for the window's contiguous cached blocks, with null blocks below
// the window; with drop, one block more is needed and then dropped.
func (m *Manager) windowHit(hashes []BlockHash, maxLen int, sg specGroup, drop bool) ([][]*Block, int) {
	bs := m.groups[sg.ids[0]].blockSize
	hashes = m.groupHashes(hashes, bs)
	// aligned is whether i blocks end on the alignment, the scheduler block size (:1006-1012).
	aligned := func(i int) bool { return i*bs%m.blockSize == 0 }
	need := cdiv(sg.spec.Window-1, bs) // _contiguous_blocks_for_hit, :943-955
	if drop {
		need++
	}
	maxBlocks := min(maxLen/bs, len(hashes))
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
		// A run starts only where the hit, after the EAGLE drop, ends on the alignment.
		if contiguous == 0 && bs != m.blockSize {
			end := i + 1
			if drop {
				end = i
			}
			if !aligned(end) {
				continue
			}
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
	trimToAlignment := func() {
		for bs != m.blockSize && !aligned(len(computed[0])) {
			for j := range computed {
				computed[j] = computed[j][:len(computed[j])-1]
			}
		}
	}
	if !found {
		for j := range computed {
			computed[j] = computed[j][:contiguous]
		}
		trimToAlignment()
	}
	if drop && len(computed[0]) > 0 {
		for j := range computed {
			computed[j] = computed[j][:len(computed[j])-1]
		}
		trimToAlignment()
	}
	return computed, len(computed[0]) * bs
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
	// External is num_external_computed_tokens: tokens a KV connector supplies past the
	// local hit, whose blocks are allocated here and filled by the connector's load.
	External int
	// Reserved is reserved_blocks: free blocks the call must leave for other in-flight
	// sequences (an asynchronous load's admission, kv_cache_manager.py:577-584).
	Reserved int
	// DelayCache is delay_cache_blocks: the blocks are not cached yet, because an
	// asynchronous load fills them in a later step (:601-604).
	DelayCache bool
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
	if a.New == 0 && adopting == 0 && a.External == 0 {
		panic("kv: allocate_slots needs new tokens or computed tokens to adopt")
	}
	localComputed := a.Computed + a.NumNewComputed
	totalComputed := min(localComputed+a.External, m.maxModelLen)

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
	if m.toAllocate(id, needSlots, newComputed, localComputed+a.External, false)+watermark > m.Pool.NumFree()-a.Reserved {
		return false
	}
	// Adopt the prefix hit in every group before allocating any new block, then allocate
	// the blocks a connector's external tokens fill (:577-590, kv_cache_coordinator.py:
	// 261-310); a running request has neither.
	if (adopting > 0 || a.External > 0) && !m.isRunning(id) {
		for i, g := range m.groups {
			if g.spec.Kind == CircularBuffer {
				// add_local_computed_blocks is a no-op for a ring (:1240-1247); it claims its
				// block with the new blocks below.
				continue
			}
			bs := newComputed[i]
			skipped := g.skippedTokens(localComputed+a.External) / g.blockSize
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
		if a.External > 0 {
			// allocate_external_computed_blocks (single_type_kv_cache_manager.py:328-367).
			total := localComputed + a.External
			for _, g := range m.groups {
				if g.spec.Kind == CircularBuffer {
					// The ring's one block (allocate_external_computed_blocks, :1209-1216).
					m.claimRing(g, id)
					continue
				}
				if skipped := g.skippedTokens(total); skipped > 0 && total-skipped <= 0 {
					continue
				}
				if need := cdiv(total, g.blockSize) - len(g.reqBlocks[id]); need > 0 {
					g.reqBlocks[id] = append(g.reqBlocks[id], m.Pool.New(need)...)
				}
			}
		}
	}
	for _, g := range m.groups {
		if g.spec.Kind == CircularBuffer {
			m.claimRing(g, id)
			continue
		}
		if need := cdiv(needSlots, g.blockSize) - len(g.reqBlocks[id]); need > 0 {
			g.reqBlocks[id] = append(g.reqBlocks[id], m.Pool.New(need)...)
		}
	}
	if m.caching && !a.DelayCache {
		for _, g := range m.groups {
			m.cache(g, id, a.Hashes, m.cacheable(g, min(mainModel, a.NumTokens)), a.Run)
		}
	}
	return true
}

// claimRing is CircularBufferManager._claim_ring_block (single_type_kv_cache_manager.py:
// 1178-1188@5df63a72): a request's ring is one block for its whole life, taken once.
func (m *Manager) claimRing(g *group, id string) {
	if len(g.reqBlocks[id]) > 0 {
		return
	}
	g.reqBlocks[id] = append(g.reqBlocks[id], m.Pool.New(1)...)
	if _, ok := g.numCached[id]; !ok {
		// The ring never caches, but a request holding it is running: a later step must not
		// adopt a hit for it (allocate_new_computed_blocks' running check, :280-286).
		g.numCached[id] = 0
	}
}

// RemainingBlocks is the scheduler's _request_remaining_blocks (scheduler.py:3060-3076): the
// blocks a request still needs to hold numTokens, given numComputed, as the full-sequence
// admission gate counts them.
func (m *Manager) RemainingBlocks(id string, numTokens, numComputed int) int {
	return m.toAllocate(id, numTokens, make([][]*Block, len(m.groups)), numComputed, true)
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
func (m *Manager) cacheable(g *group, numTokens int) int {
	if len(m.groups) == 1 {
		return max(0, numTokens-m.reprefillable)
	}
	aligned := numTokens / m.blockSize * m.blockSize
	if g.eagle && aligned > 0 {
		finalized := max(0, numTokens-m.reprefillable)
		return min(finalized, finalized/m.blockSize*m.blockSize+g.blockSize)
	}
	return aligned
}

// skippedTokens is get_num_skipped_tokens (single_type_kv_cache_manager.py:712-723 and
// :1135-1171): the tokens below a sliding window less the extra ones it retains, none for
// full attention.
func (g *group) skippedTokens(computed int) int {
	if !g.spec.window() {
		return 0
	}
	return max(0, computed-g.spec.Window+1-g.extraRetained)
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
		last := min(n/g.blockSize, len(bs))
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
		have := len(g.reqBlocks[id])
		if g.spec.Kind == CircularBuffer {
			// CircularBufferManager.get_num_blocks_to_allocate (:1190-1201): the ring's one
			// block, until the request holds it.
			if have == 0 {
				total++
			}
			continue
		}
		required := cdiv(numTokens, g.blockSize)
		if admission && g.spec.window() {
			// max_admission_blocks_per_request (kv_cache_interface.py:859-885).
			capTokens := min(g.spec.Window-1+g.extraRetained+m.maxInFlight, m.maxModelLen)
			required = min(required, cdiv(capTokens, g.blockSize)+1)
		}
		if _, ok := g.numCached[id]; ok {
			total += max(required-have, 0)
			continue
		}
		hit := newComputed[i]
		skippedBlocks := g.skippedTokens(totalComputed) / g.blockSize
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

// CacheUpTo is cache_blocks (kv_cache_manager.py:782-792; kv_cache_coordinator.py:353-376):
// it records the request's full blocks up to numTokens as prefix-cached in every group, less
// what a drafter may re-prefill, for a request that holds blocks. The AsyncScheduler calls it
// when a sampled token replaces its placeholder (async_scheduler.py:74-78).
func (m *Manager) CacheUpTo(id string, hashes []BlockHash, numTokens int, run StoredRun) {
	if !m.caching {
		return
	}
	// A running request holds blocks in every group (Allocate gives each group at least
	// one); one that holds none has nothing to cache. numCached is no test: it is unset
	// until a request's first block is full.
	if _, holds := m.groups[0].reqBlocks[id]; !holds {
		return
	}
	for _, g := range m.groups {
		m.cache(g, id, hashes, m.cacheable(g, numTokens), run)
	}
}

// CacheBlocks is cache_blocks (kv_cache_manager.py:782-792; kv_cache_coordinator.py:353-376)
// as a landed asynchronous load calls it (scheduler.py:3123-3160): it records the request's
// full blocks up to numComputed as prefix-cached in every group, less what a drafter may
// re-prefill.
func (m *Manager) CacheBlocks(id string, hashes []BlockHash, numComputed int, run StoredRun) {
	if !m.caching {
		return
	}
	for _, g := range m.groups {
		m.cache(g, id, hashes, m.cacheable(g, numComputed), run)
	}
}

// cache records the request's full blocks up to numTokens as prefix-cached in one group
// (cache_blocks, single_type_kv_cache_manager.py:468-516).
// A group that opts out of prefix caching caches nothing (:498-499, the ring's no-op
// :1229-1237); a group of its own block size caches its blocks under the hashes at that size.
// A sliding window whose block size is below the scheduler block size caches only the blocks a
// hit at an aligned boundary can use (reachableWindowBlocks).
func (m *Manager) cache(g *group, id string, hashes []BlockHash, numTokens int, run StoredRun) {
	if !g.spec.cacheable() {
		return
	}
	numCached := g.numCached[id]
	numFull := numTokens / g.blockSize
	if numCached >= numFull {
		return
	}
	hashes = m.groupHashes(hashes, g.blockSize)
	numFull = min(numFull, len(hashes))
	if numCached >= numFull {
		return
	}
	mask := m.reachableWindowBlocks(g, numCached, numFull)
	m.Pool.CacheFullMaskedIn(g.id, g.reqBlocks[id], hashes, numCached, numFull, run, g.blockSize, mask)
	g.numCached[id] = numFull
}

// reachableWindowBlocks is SlidingWindowManager.reachable_block_mask
// (single_type_kv_cache_manager.py:1054-1133@5df63a72) for blocks [from, to) of a window
// group, with hits aligned to the scheduler block size and dense retention (a retention
// interval of None, which the oracle's KV cache config carries): a hit needs the window's
// contiguous blocks ending on an aligned boundary, so of each aligned segment only its last
// `need` blocks (one further along with the EAGLE peek) are kept. nil caches every block: full
// attention, and a window whose run covers its whole segment, as one at the scheduler block
// size always does.
func (m *Manager) reachableWindowBlocks(g *group, from, to int) []bool {
	if !g.spec.window() || m.blockSize%g.blockSize != 0 {
		return nil
	}
	perSegment := m.blockSize / g.blockSize
	need := cdiv(g.spec.Window-1, g.blockSize) // _contiguous_blocks_for_hit, :943-955
	shift := 0
	if g.eagle {
		need, shift = need+1, 1
	}
	if need >= perSegment {
		return nil
	}
	mask := make([]bool, to-from)
	for i := from; i < to; i++ {
		mask[i-from] = i >= shift && (i-shift)%perSegment >= perSegment-need
	}
	return mask
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
