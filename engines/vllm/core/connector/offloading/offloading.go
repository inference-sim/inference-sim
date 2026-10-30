// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py (the scheduler-role
// dispatch), offloading/common.py (TransferJob, OffloadingWorkerMetadata), offloading/config.py
// (blocks_per_chunk) and vllm/v1/core/single_type_kv_cache_manager.py (the sliding window's
// reachable_block_mask the store path calls).

// Package offloading is the scheduler side of vLLM's OffloadingConnector
// (vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:53@5df63a72) over the CPU
// tier: OffloadingConnectorScheduler (offloading/scheduler.py:563-2101) with
// CPUOffloadingManager and its LRU policy (vllm/v1/kv_offload/cpu/manager.py:47,
// cpu/policies/lru.py:16). It decides which prompt chunks a request finds offloaded (a hit
// loaded asynchronously into the request's blocks), which computed chunks are stored to the CPU
// tier each step, which CPU chunks are evicted for them, and which in-flight jobs the worker
// must flush first; it keeps the engine stepping while jobs are in flight. The worker, which
// moves the bytes, is the caller's: each step's Meta lists the jobs it must run, and Completed
// reports the ones it finished (OffloadingWorkerMetadata.completed_jobs).
//
// Modelled and diff-tested: KV cache groups of full attention and sliding windows of one block
// size, any blocks_per_chunk, one worker, dense prefix-cache retention, LRU eviction,
// offload_prompt_only, synchronous scheduling without a drafter. The rest is refused, by Check
// (configuration) or by the scheduler's SetConnector (Capabilities): see Check.
package offloading

import (
	"errors"
	"fmt"
	"sort"

	"github.com/inference-sim/blis-vllm/core/sched"
)

// Issue tracks what this package does not model yet.
const Issue = "https://github.com/inference-sim/blis-vllm/issues/14"

// Group is one KV cache group as the connector sees it (GroupOffloadConfig,
// offloading/scheduler.py:88-126): its block size, and for a sliding window the window and its
// extra retained tokens.
type Group struct {
	BlockSize     int
	Window        int // 0 is full attention
	ExtraRetained int
}

// Config is the connector's configuration (SchedulerOffloadConfig.from_spec,
// offloading/scheduler.py:176-344, over CPUOffloadingSpec).
type Config struct {
	Groups []Group
	// BlocksPerChunk is blocks_per_chunk (offloading/config.py:98-134); 0 means 1.
	BlocksPerChunk int
	// TokensPerHash is the scheduler's hash block size.
	TokensPerHash int
	// NumChunks is the CPU tier's capacity in chunks (NumChunks).
	NumChunks int
	// OffloadPromptOnly is offload_prompt_only (vllm/v1/kv_offload/base.py:617-623, default true).
	OffloadPromptOnly bool
	// PrefixCaching is enable_prefix_caching: with it, a chunk being loaded is tracked so a
	// second request does not load it again (scheduler.py:619-621).
	PrefixCaching bool
	// Workers is num_workers, the engine's world size (scheduler.py:335); 0 means 1. A job is
	// pending on every worker until each reports it (TransferJobStatus.pending_count, :76-78).
	Workers int
}

// groupCfg is GroupOffloadConfig with its derived sizes.
type groupCfg struct {
	Group
	tokensPerChunk int
	hashesPerChunk int
	// windowChunks is sliding_window_size_in_chunks; 0 for full attention.
	windowChunks int
}

func (g groupCfg) sliding() bool { return g.Window > 0 }

// loadWindowChunks is load_window_size_in_chunks (scheduler.py:109-126).
func (g groupCfg) loadWindowChunks(numTokens int) int {
	if !g.sliding() {
		return 0
	}
	right := mod(-numTokens, g.tokensPerChunk)
	return max(g.windowChunks, cdiv(g.Window-1+g.ExtraRetained+right, g.tokensPerChunk))
}

// Job is a TransferJob (offloading/common.py:53-63): one load or store between the GPU blocks
// and the CPU chunks.
type Job struct {
	ID    int
	Req   string
	Store bool
	// CPU are the CPU chunk ids (CPULoadStoreSpec).
	CPU []int
	// GPU are the GPU block ids, GroupSizes how many belong to each group in order, and
	// BlockIndices each group's first block's index in the request (GPULoadStoreSpec,
	// vllm/v1/kv_offload/base.py:438-469).
	GPU, GroupSizes, BlockIndices []int
}

// Meta is OffloadingConnectorMetadata (offloading/common.py:66-72): the step's jobs, each
// ordered by job id, and the in-flight jobs the worker must wait for before the step runs.
type Meta struct {
	Loads, Stores []Job
	Flush         []int
}

// jobStatus is TransferJobStatus (scheduler.py:71-85).
type jobStatus struct {
	req      string
	pending  int
	keys     []Key
	store    bool
	deferred []int // deferred_fence_block_ids
	fenced   []int // fenced_block_ids
}

// groupState is RequestGroupState (scheduler.py:347-355).
type groupState struct {
	keys       []Key
	blockIDs   []int
	nextStored int
}

// reqState is RequestOffloadState (scheduler.py:358-501).
type reqState struct {
	req      *sched.Request
	access   *reqAccess
	groups   []*groupState
	localHit int // num_locally_computed_tokens
	jobs     map[int]bool
	signaled bool
}

// Scheduler is OffloadingConnectorScheduler for one engine.
type Scheduler struct {
	cfg        Config
	groups     []groupCfg
	full, swa  []int // _full_attention_groups, _sliding_window_groups (largest window first)
	alignment  int   // alignment_tokens
	manager    *cpuManager
	reqs       map[string]*reqState
	loadJobs   []Job
	flush      map[int]bool
	allocated  map[int]bool // _current_batch_allocated_block_ids
	loading    map[Key]bool // _chunks_being_loaded; nil without prefix caching
	jobCounter int
	jobs       map[int]*jobStatus
	pendingOf  map[int]map[int]bool // _block_id_to_pending_jobs
}

// New builds the connector's scheduler side.
func New(cfg Config) (*Scheduler, error) {
	if len(cfg.Groups) == 0 {
		return nil, errors.New("offloading: no KV cache groups")
	}
	if cfg.BlocksPerChunk == 0 {
		cfg.BlocksPerChunk = 1
	}
	if cfg.Workers == 0 {
		cfg.Workers = 1
	}
	s := &Scheduler{cfg: cfg, manager: newCPUManager(cfg.NumChunks), reqs: map[string]*reqState{},
		flush: map[int]bool{}, allocated: map[int]bool{}, jobs: map[int]*jobStatus{}, pendingOf: map[int]map[int]bool{}}
	if cfg.PrefixCaching {
		s.loading = map[Key]bool{}
	}
	fullChunk := map[int]bool{}
	for _, g := range cfg.Groups {
		if g.BlockSize != cfg.Groups[0].BlockSize {
			// Groups of several block sizes reach the partial-tail and mamba-alignment paths
			// (scheduler.py:155-173, 295-315) the oracle has not run.
			return nil, fmt.Errorf("offloading: KV cache groups of several block sizes are not modelled (%s)", Issue)
		}
		if g.BlockSize%cfg.TokensPerHash != 0 {
			return nil, fmt.Errorf("offloading: block size %d is not a multiple of the hash block size %d", g.BlockSize, cfg.TokensPerHash)
		}
		gc := groupCfg{Group: g, tokensPerChunk: g.BlockSize * cfg.BlocksPerChunk}
		gc.hashesPerChunk = gc.tokensPerChunk / cfg.TokensPerHash
		if g.Window > 0 {
			// get_sliding_window_size_in_chunks (scheduler.py:129-140).
			gc.windowChunks = max(cdiv(g.Window, gc.tokensPerChunk), cdiv(g.Window-1+g.ExtraRetained, gc.tokensPerChunk))
		} else {
			fullChunk[gc.tokensPerChunk] = true
		}
		s.groups = append(s.groups, gc)
	}
	// alignment_tokens (scheduler.py:194-222).
	if len(fullChunk) == 1 {
		for t := range fullChunk {
			s.alignment = t
		}
	} else if len(fullChunk) == 0 {
		for _, g := range s.groups {
			s.alignment = max(s.alignment, g.tokensPerChunk)
		}
	}
	for i, g := range s.groups {
		if g.sliding() {
			s.swa = append(s.swa, i)
		} else {
			s.full = append(s.full, i)
		}
	}
	sort.SliceStable(s.swa, func(a, b int) bool { return s.groups[s.swa[a]].windowChunks > s.groups[s.swa[b]].windowChunks })
	return s, nil
}

// Options are the kv_connector_extra_config keys and engine settings Check refuses beside the
// OffloadingConnector, with the vLLM code each would reach.
type Options struct {
	// SpecName is spec_name (vllm/v1/kv_offload/factory.py:31, default CPUOffloadingSpec).
	SpecName string
	// SecondaryTiers is len(secondary_tiers) (vllm/v1/kv_offload/tiering/spec.py:285).
	SecondaryTiers int
	// EvictionPolicy is eviction_policy (cpu/spec.py:131, default lru).
	EvictionPolicy string
	// StoreThreshold is store_threshold (cpu/spec.py:139, default 0).
	StoreThreshold int
	// RetentionInterval is prefix_cache_retention_interval; nil is dense retention.
	RetentionInterval *int
}

// Check refuses a configuration this package does not model, naming the vLLM code it would
// reach. TieringOffloadingSpec with no secondary tiers is the CPU tier: its manager forwards every
// call the connector's scheduler side makes to its CPU primary tier, a CPUOffloadingManager with
// the same policy (vllm/v1/kv_offload/tiering/manager.py:396-417, 569-661, 736-781, 843-873,
// 912-921@5df63a72), which is what this package models.
func Check(o Options) error {
	switch {
	case o.SpecName != "" && o.SpecName != "CPUOffloadingSpec" && o.SpecName != "TieringOffloadingSpec":
		return fmt.Errorf("offloading: spec_name %q is not modelled; CPUOffloadingSpec and TieringOffloadingSpec are (%s)", o.SpecName, Issue)
	case o.SecondaryTiers > 0:
		// Tier cascades, promotions and their backpressure clocks (tiering/manager.py:269-360,
		// backpressure.py) are https://github.com/inference-sim/blis-vllm/issues/13.
		return fmt.Errorf("offloading: TieringOffloadingSpec with %d secondary tier(s) is not modelled, only its CPU "+
			"primary tier (https://github.com/inference-sim/blis-vllm/issues/13)", o.SecondaryTiers)
	case o.EvictionPolicy != "" && o.EvictionPolicy != "lru":
		// ARC (cpu/policies/arc.py:16) is https://github.com/inference-sim/blis-vllm/issues/1.
		return fmt.Errorf("offloading: eviction_policy %q is not modelled, only lru "+
			"(https://github.com/inference-sim/blis-vllm/issues/1)", o.EvictionPolicy)
	case o.StoreThreshold >= 2:
		// The admission tracker (cpu/manager.py:123-142, 272-281); TieringOffloadingSpec refuses
		// it itself (tiering/spec.py:340-343).
		return fmt.Errorf("offloading: store_threshold %d is not modelled (%s)", o.StoreThreshold, Issue)
	case o.RetentionInterval != nil:
		// Sparse retention (reachable_block_mask with retention_interval, single_type_kv_cache_manager.
		// py:1093-1131; scheduler.py:1608-1612) is https://github.com/inference-sim/blis-vllm/issues/7.
		return fmt.Errorf("offloading: prefix_cache_retention_interval %d is not modelled, only dense retention "+
			"(https://github.com/inference-sim/blis-vllm/issues/7)", *o.RetentionInterval)
	}
	return nil
}

// ConnectorModels states what the connector is diff-tested beside (sched.Capabilities): several
// KV cache groups (it is SupportsHMA, offloading_connector.py:53).
func (s *Scheduler) ConnectorModels() sched.Capabilities {
	return sched.Capabilities{HybridGroups: true}
}

// RequiresKVDelivery is requires_kv_delivery (offloading_connector.py:72-76): a best-effort
// cache, false.
func (s *Scheduler) RequiresKVDelivery() bool { return false }

func (s *Scheduler) newJob() int {
	id := s.jobCounter
	s.jobCounter++
	return id
}

// updateKeys is update_offload_keys (scheduler.py:420-437): each group's keys for every whole
// chunk of the request's hashes, with the key's end position.
func (s *Scheduler) updateKeys(st *reqState) {
	hs := st.req.BlockHashes()
	for gi, g := range s.groups {
		gs := st.groups[gi]
		for i := g.hashesPerChunk*len(gs.keys) + g.hashesPerChunk - 1; i < len(hs); i += g.hashesPerChunk {
			k := makeKey(hs[i], gi)
			gs.keys = append(gs.keys, k)
			st.access.positions[k] = len(gs.keys) * g.tokensPerChunk
		}
	}
}

// OnNewRequest is on_new_request (scheduler.py:1027-1037).
func (s *Scheduler) OnNewRequest(r *sched.Request) {
	st := &reqState{req: r, access: newReqAccess(), jobs: map[int]bool{}}
	for range s.groups {
		st.groups = append(st.groups, &groupState{})
	}
	s.reqs[r.ID] = st
}

// MatchedTokens is get_num_new_matched_tokens without a deferral (sched.Connector).
func (s *Scheduler) MatchedTokens(r *sched.Request, local int) (int, bool) {
	n, async, _ := s.MatchedTokensOrDefer(r, local)
	return n, async
}

// MatchedTokensOrDefer is get_num_new_matched_tokens (scheduler.py:1039-1097): the offloaded
// tokens beyond the local hit, loaded asynchronously; deferred while the request has jobs in
// flight or a hit chunk is still being loaded for another request.
func (s *Scheduler) MatchedTokensOrDefer(r *sched.Request, local int) (int, bool, bool) {
	st := s.reqs[r.ID]
	for _, gs := range st.groups {
		gs.blockIDs = gs.blockIDs[:0]
	}
	if len(st.jobs) > 0 {
		return 0, false, true
	}
	s.updateKeys(st)
	st.localHit = local
	n, deferred := s.lookupComplete(st)
	if deferred {
		return 0, false, true
	}
	return n, n > 0, false
}

// prefixLookup is _maximal_prefix_lookup (scheduler.py:674-706) for the CPU tier: the leading
// run of hits; deferred if one is still being stored (HIT_PENDING).
func (s *Scheduler) prefixLookup(keys []Key) (int, bool) {
	n, pending := 0, false
	for _, k := range keys {
		switch s.manager.lookup(k) {
		case hit:
			n++
		case hitPending:
			pending = true
			n++
		case miss:
			return n, pending
		}
	}
	return n, pending
}

// windowLookup is _sliding_window_lookup (scheduler.py:708-750): scanning from the end, the end
// index of the last run of window consecutive hits, the first run allowed a larger window;
// deferred when a pending chunk is in the run found.
func (s *Scheduler) windowLookup(keys []Key, window, initial int) (int, bool) {
	pending := false
	run := 0
	need := window
	if initial > 0 {
		need = initial
	}
	for i := len(keys) - 1; i >= 0; i-- {
		switch s.manager.lookup(keys[i]) {
		case hit:
			run++
		case hitPending:
			pending = true
			run++
		case miss:
			run = 0
			pending = false
			need = window
		}
		if run == need {
			return i + need, pending
		}
	}
	return run, pending
}

// lookupComplete is _lookup_complete_chunks (scheduler.py:752-952) without max_load_tokens, an
// external bound or EAGLE groups.
func (s *Scheduler) lookupComplete(st *reqState) (int, bool) {
	local := st.localHit
	maxHit := st.req.NumTokens()
	if len(s.swa) > 0 {
		maxHit--
	}
	hitTokens := 0
	deferred := false
	order := append(append([]int(nil), s.full...), s.swa...)
	for len(order) > 0 {
		lookedUpSWA := false
		groups := order
		order = nil
		for _, gi := range groups {
			g := s.groups[gi]
			keys := st.groups[gi].keys
			maxHit = min(maxHit, len(keys)*g.tokensPerChunk)
			if maxHit-local < g.tokensPerChunk {
				return 0, false
			}
			numChunks := min(cdiv(maxHit, g.tokensPerChunk), len(keys))
			start := local / g.tokensPerChunk
			ks := keys[start:numChunks]
			var n int
			var pend bool
			if !g.sliding() {
				n, pend = s.prefixLookup(ks)
			} else {
				end := min(maxHit, numChunks*g.tokensPerChunk)
				n, pend = s.windowLookup(ks, g.windowChunks, g.loadWindowChunks(end))
			}
			if n == 0 && !pend {
				return 0, false
			}
			if pend {
				deferred = true
			} else {
				maxHit = min(maxHit, g.tokensPerChunk*(start+n))
			}
			newHit := maxHit - local
			if newHit < g.tokensPerChunk {
				return 0, false
			}
			if newHit < hitTokens {
				if deferred {
					deferred = false
					order = append(append([]int(nil), s.full...), s.swa...)
				} else if lookedUpSWA && len(order) == 0 {
					order = append([]int(nil), s.swa...)
				}
			}
			lookedUpSWA = lookedUpSWA || g.sliding()
			hitTokens = newHit
		}
	}
	if deferred {
		return 0, true
	}
	if len(s.loading) > 0 {
		for gi, g := range s.groups {
			keys := st.groups[gi].keys
			end := cdiv(local+hitTokens, g.tokensPerChunk)
			ks := keys[local/g.tokensPerChunk : end]
			if w := g.loadWindowChunks(local + hitTokens); w > 0 && len(ks) > w {
				ks = ks[len(ks)-w:]
			}
			for _, k := range ks {
				if s.loading[k] {
					return 0, true
				}
			}
		}
	}
	return hitTokens, false
}

// AfterAlloc is update_state_after_alloc (scheduler.py:1146-1279): the load job for the hit's
// chunks into the request's fresh blocks.
func (s *Scheduler) AfterAlloc(r *sched.Request, blocks sched.AllocatedBlocks, external int) {
	if external == 0 {
		return
	}
	st := s.reqs[r.ID]
	local := st.localHit
	cached := local + external
	var keys []Key
	job := Job{Req: r.ID}
	bpc := s.cfg.BlocksPerChunk
	for gi, g := range s.groups {
		gb := blocks.Groups[gi]
		for _, id := range gb {
			if id != 0 {
				s.allocated[id] = true
			}
		}
		numGPU := cdiv(cached, g.BlockSize)
		first := cdiv(local, g.BlockSize)
		start := numGPU
		// The first block past the local hit that is neither null nor cached (:1186-1194). The
		// local hit is block-aligned (:1196), so every block from first on is one allocate_slots
		// took fresh off the free queue, which drops a cached hash (get_new_blocks,
		// block_pool.py:666-700), or a sliding window's null slot.
		for i := first; i < numGPU; i++ {
			if gb[i] != 0 {
				start = i
				break
			}
		}
		pendingBlocks := numGPU - start
		numChunks := cdiv(cached, g.tokensPerChunk)
		if pendingBlocks > 0 {
			keys = append(keys, st.groups[gi].keys[start/bpc:numChunks]...)
		}
		job.GPU = append(job.GPU, gb[start:numGPU]...)
		job.GroupSizes = append(job.GroupSizes, pendingBlocks)
		job.BlockIndices = append(job.BlockIndices, start)
		// CHUNK_LEVEL: the hit's chunks are not stored again (:1250-1254).
		st.groups[gi].nextStored = numChunks
	}
	job.CPU = s.manager.prepareLoad(keys, st.access)
	job.ID = s.newJob()
	s.loadJobs = append(s.loadJobs, job)
	st.jobs[job.ID] = true
	s.jobs[job.ID] = &jobStatus{req: r.ID, pending: s.cfg.Workers, keys: keys}
	if s.loading != nil {
		for _, k := range keys {
			s.loading[k] = true
		}
	}
}

// Finished is request_finished for one group (sched.Connector).
func (s *Scheduler) Finished(r *sched.Request, _ []int) bool { return s.finished(r) }

// FinishedAllGroups is request_finished_all_groups (offloading_connector.py:236-243).
func (s *Scheduler) FinishedAllGroups(r *sched.Request, _ [][]int) bool { return s.finished(r) }

// finished is request_finished (scheduler.py:2005-2044): the blocks are never held; the
// request's last chunks are stored at the next step, and the blocks its in-flight stores read
// are watched for reuse from now on.
func (s *Scheduler) finished(r *sched.Request) bool {
	st := s.reqs[r.ID]
	if st == nil {
		return false
	}
	s.updateKeys(st)
	for id := range st.jobs {
		for _, b := range s.jobs[id].deferred {
			s.watch(b, id)
		}
	}
	return false
}

func (s *Scheduler) watch(block, job int) {
	if s.pendingOf[block] == nil {
		s.pendingOf[block] = map[int]bool{}
	}
	s.pendingOf[block][job] = true
}

func (s *Scheduler) unwatch(job int, blocks []int) {
	for _, b := range blocks {
		delete(s.pendingOf[b], job)
		if len(s.pendingOf[b]) == 0 {
			delete(s.pendingOf, b)
		}
	}
}

// updateReqStates is _update_req_states (scheduler.py:1281-1331).
func (s *Scheduler) updateReqStates(step sched.StepOutput) {
	ends := map[string][]int{}
	for _, rq := range step.Reqs {
		st := s.reqs[rq.ID]
		s.updateKeys(st)
		if rq.Resumed {
			for _, gs := range st.groups {
				gs.blockIDs = gs.blockIDs[:0]
			}
		}
		if rq.NewBlocks == nil {
			continue
		}
		if len(s.swa) > 0 {
			e := make([]int, len(s.swa))
			for i, gi := range s.swa {
				e[i] = len(st.groups[gi].blockIDs)
			}
			ends[rq.ID] = e
		}
		for gi := range s.groups {
			st.groups[gi].blockIDs = append(st.groups[gi].blockIDs, rq.NewBlocks[gi]...)
			for _, b := range rq.NewBlocks[gi] {
				if b != 0 {
					s.allocated[b] = true
				}
			}
		}
	}
	if len(s.swa) == 0 || len(s.allocated) == 0 {
		return
	}
	bpc := s.cfg.BlocksPerChunk
	for _, id := range s.sortedReqIDs() {
		st := s.reqs[id]
		e := ends[id]
		for i, gi := range s.swa {
			gs := st.groups[gi]
			end := len(gs.blockIDs)
			if e != nil {
				end = e[i]
			}
			for j := gs.nextStored * bpc; j < end; j++ {
				if s.allocated[gs.blockIDs[j]] {
					gs.blockIDs[j] = 0
				}
			}
		}
	}
}

func (s *Scheduler) sortedReqIDs() []string {
	out := make([]string, 0, len(s.reqs))
	for id := range s.reqs {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// offloadable is _calc_num_offloadable_tokens (scheduler.py:663-672).
func (s *Scheduler) offloadable(r *sched.Request, computed int) int {
	n := min(computed, r.NumTokens())
	if s.cfg.OffloadPromptOnly {
		n = min(n, r.NumPromptTokens())
	}
	return n
}

// storable is storable_chunks (scheduler.py:450-481) for a group without EAGLE.
func (s *Scheduler) storable(g groupCfg, gs *groupState, tokens int) int {
	return min(tokens/g.tokensPerChunk, len(gs.blockIDs)/s.cfg.BlocksPerChunk, len(gs.keys))
}

func (s *Scheduler) advanceStored(st *reqState, tokens int) {
	for gi, g := range s.groups {
		gs := st.groups[gi]
		gs.nextStored = max(gs.nextStored, s.storable(g, gs, tokens))
	}
}

// reachable is _reachable_store_block_mask (scheduler.py:1517-1554) over the managers'
// reachable_block_mask with dense retention (retention_interval None): nil (every block) for full
// attention (single_type_kv_cache_manager.py:535-560); for a sliding window, each alignment
// segment's last blocks a hit can need, plus the final segment's tail
// (SlidingWindowManager.reachable_block_mask, :1053-1133).
func (s *Scheduler) reachable(g groupCfg, start, end, finalEnd int) []bool {
	if !g.sliding() || s.alignment == 0 || s.alignment%g.BlockSize != 0 {
		return nil
	}
	bpc := s.cfg.BlocksPerChunk
	window := g.Window + g.ExtraRetained
	need := cdiv(window-1, g.BlockSize)
	per := s.alignment / g.BlockSize
	if need >= per {
		return nil
	}
	sb, eb := start*bpc, end*bpc
	mask := make([]bool, eb-sb)
	for i := sb; i < eb; i++ {
		if i%per >= per-need {
			mask[i-sb] = true
		}
	}
	if finalEnd >= 0 {
		fe := finalEnd * bpc
		if size := fe % per; size > 0 {
			from := max(fe-size, fe-need)
			for i := max(sb, from); i < min(eb, fe); i++ {
				mask[i-sb] = true
			}
		}
	}
	return mask
}

// storeRange is one group's [start, end) chunk range of a store.
type storeRange struct{ start, end int }

// buildStores is _build_store_jobs (scheduler.py:1570-1844) with retention None and no EAGLE
// groups.
func (s *Scheduler) buildStores(step sched.StepOutput) []Job {
	bpc := s.cfg.BlocksPerChunk
	var out []Job
	ids := append(append([]string(nil), step.Order...), step.Finished...)
	for _, id := range ids {
		st := s.reqs[id]
		if st == nil {
			continue
		}
		r := st.req
		var after int
		switch {
		case r.Status == sched.FinishedAborted:
			after = r.NumComputed
		case r.IsFinished():
			after = max(r.NumPromptTokens(), r.NumTokens()-1)
		default:
			after = r.NumComputed + step.Scheduled[id]
		}
		tokens := s.offloadable(r, after)
		prompt := s.offloadable(r, r.NumPromptTokens())
		var newKeys []Key
		ranges := make([]storeRange, len(s.groups))
		for gi, g := range s.groups {
			gs := st.groups[gi]
			n := s.storable(g, gs, tokens)
			start := gs.nextStored
			horizon := prompt / g.tokensPerChunk
			finalAlign := 0
			if g.sliding() && s.alignment > 0 && s.alignment%g.BlockSize == 0 {
				finalAlign = s.alignment / g.BlockSize
			}
			reconsider := r.IsFinished() && n != horizon && finalAlign > 0
			finalEnd := -1
			switch {
			case r.IsFinished():
				finalEnd = n
			case n <= horizon:
				finalEnd = horizon
			}
			if reconsider {
				hb := n * bpc
				start = min(start, (hb-hb%finalAlign)/bpc)
			}
			ranges[gi] = storeRange{start, n}
			if n <= start {
				continue
			}
			mask := s.reachable(g, start, n, finalEnd)
			var promptMask []bool
			if reconsider {
				promptMask = s.reachable(g, start, n, horizon)
			}
			for k := start; k < n; k++ {
				if gs.blockIDs[k*bpc+bpc-1] == 0 {
					continue
				}
				idx := k - start
				if reconsider && k < gs.nextStored && (promptMask == nil || anyIn(promptMask, idx*bpc, bpc)) {
					continue
				}
				if mask != nil && !anyIn(mask, idx*bpc, bpc) {
					continue
				}
				newKeys = append(newKeys, gs.keys[k])
			}
		}
		if len(newKeys) == 0 {
			s.advanceStored(st, tokens)
			continue
		}
		stored, chunkIDs, ok := s.manager.prepareStore(newKeys, st.access)
		if !ok {
			// "cannot store chunks" (:1735-1740): nothing advances; the chunks are offered again.
			continue
		}
		if len(stored) == 0 {
			s.advanceStored(st, tokens)
			continue
		}
		storing := map[Key]bool{}
		for _, k := range stored {
			storing[k] = true
		}
		job := Job{Req: id, Store: true, CPU: chunkIDs}
		var fenced, deferred []int
		for gi, g := range s.groups {
			gs := st.groups[gi]
			rg := ranges[gi]
			cnt, first := 0, -1
			for k := rg.start; k < rg.end; k++ {
				if !storing[gs.keys[k]] {
					continue
				}
				for b := 0; b < bpc; b++ {
					bid := gs.blockIDs[k*bpc+b]
					if bid == 0 {
						continue
					}
					if first < 0 {
						first = k*bpc + b
					}
					job.GPU = append(job.GPU, bid)
					cnt++
					if g.sliding() {
						fenced = append(fenced, bid)
					} else {
						deferred = append(deferred, bid)
					}
				}
			}
			job.GroupSizes = append(job.GroupSizes, cnt)
			job.BlockIndices = append(job.BlockIndices, max(first, 0))
			gs.nextStored = max(gs.nextStored, rg.end)
		}
		job.ID = s.newJob()
		st.jobs[job.ID] = true
		for _, b := range fenced {
			s.watch(b, job.ID)
		}
		s.jobs[job.ID] = &jobStatus{req: id, pending: s.cfg.Workers, keys: stored, store: true, deferred: deferred, fenced: fenced}
		out = append(out, job)
		if r.IsFinished() {
			for _, b := range deferred {
				s.watch(b, job.ID)
				if s.allocated[b] {
					s.flush[job.ID] = true
				}
			}
		}
	}
	return out
}

func anyIn(mask []bool, from, n int) bool {
	for i := from; i < from+n; i++ {
		if mask[i] {
			return true
		}
	}
	return false
}

// BuildMeta is build_connector_meta (scheduler.py:1846-1900).
func (s *Scheduler) BuildMeta(step sched.StepOutput) any {
	s.updateReqStates(step)
	for _, id := range step.Preempted {
		if st := s.reqs[id]; st != nil {
			for j := range st.jobs {
				s.flush[j] = true
			}
		}
	}
	for b := range s.allocated {
		for j := range s.pendingOf[b] {
			s.flush[j] = true
		}
	}
	m := Meta{Loads: s.loadJobs, Stores: s.buildStores(step)}
	for j := range s.flush {
		m.Flush = append(m.Flush, j)
	}
	sort.Ints(m.Flush)
	for _, id := range step.Finished {
		st := s.reqs[id]
		if st == nil {
			continue
		}
		st.signaled = true
		s.manager.requestFinished(st.access)
		if len(st.jobs) == 0 {
			delete(s.reqs, id)
		}
	}
	s.loadJobs = nil
	s.flush = map[int]bool{}
	s.allocated = map[int]bool{}
	return m
}

// HasPendingPushWork is has_pending_push_work (scheduler.py:1902-1908).
func (s *Scheduler) HasPendingPushWork() bool { return len(s.jobs) > 0 }

// Completion is one job's count in OffloadingWorkerMetadata.completed_jobs: how many workers
// reported it (offloading/common.py:76-104, merged across workers as KVOutputAggregator merges
// them, vllm/distributed/kv_transfer/kv_connector/utils.py:144-150@5df63a72).
type Completion struct {
	Job, Workers int
}

// Completed is update_connector_output's worker metadata (scheduler.py:1952-1988): each job the
// workers reported, in job-id order (the transport's order, oracle/mocks.json). A job completes
// once every worker has.
func (s *Scheduler) Completed(done []Completion) {
	for _, c := range done {
		j := c.Job
		js := s.jobs[j]
		if js == nil || c.Workers < 1 || c.Workers > js.pending {
			panic(fmt.Sprintf("offloading: job %d reported by %d worker(s), %d pending", j, c.Workers, s.pendingOfJob(j)))
		}
		js.pending -= c.Workers
		if js.pending > 0 {
			continue
		}
		st := s.reqs[js.req]
		if js.store {
			s.manager.completeStore(js.keys)
		} else {
			s.manager.completeLoad(js.keys)
			for _, k := range js.keys {
				delete(s.loading, k)
			}
		}
		if len(s.pendingOf) > 0 {
			s.unwatch(j, js.fenced)
			if st.req.IsFinished() {
				s.unwatch(j, js.deferred)
			}
		}
		delete(s.jobs, j)
		delete(st.jobs, j)
		if st.signaled && len(st.jobs) == 0 {
			delete(s.reqs, js.req)
		}
	}
}

// AfterOutput is update_connector_output (scheduler.py:1910-1988): the worker metadata is the
// jobs the workers completed this step ([]Completion, in job-id order).
func (s *Scheduler) AfterOutput(o sched.TransferOutcome) {
	if done, ok := o.WorkerMeta.([]Completion); ok {
		s.Completed(done)
	}
}

func (s *Scheduler) pendingOfJob(j int) int {
	if js := s.jobs[j]; js != nil {
		return js.pending
	}
	return 0
}

// Workers is num_workers.
func (s *Scheduler) Workers() int { return s.cfg.Workers }

// Jobs lists the jobs in flight (_jobs), by id.
func (s *Scheduler) Jobs() []int {
	out := make([]int, 0, len(s.jobs))
	for j := range s.jobs {
		out = append(out, j)
	}
	sort.Ints(out)
	return out
}

// LoadReq is the request a job loads for, and whether it is a load.
func (s *Scheduler) LoadReq(j int) (string, bool) {
	js := s.jobs[j]
	if js == nil || js.store {
		return "", false
	}
	return js.req, true
}

// CPUChunk is one chunk of the CPU tier: its key's hash prefix and group, its chunk id and its
// reference count (-1 while its store is in flight).
type CPUChunk struct {
	Key    Key
	ID     int
	RefCnt int
}

// CPU is the CPU tier, least recent first, with its free and evictable chunk counts.
func (s *Scheduler) CPU() (chunks []CPUChunk, free, evictable int) {
	for _, k := range s.manager.lru.inRankOrder() {
		c := s.manager.lru.chunks[k]
		chunks = append(chunks, CPUChunk{Key: k, ID: c.id, RefCnt: c.refCnt})
	}
	return chunks, s.manager.numFree(), s.manager.numEvictable
}

// Group returns the key's group.
func (k Key) Group() int { return k.group() }

// HashPrefix is the key's hash, its first 8 bytes.
func (k Key) HashPrefix() []byte { return []byte(k)[:min(8, len(k)-4)] }

var _ sched.Connector = (*Scheduler)(nil)
var _ sched.SupportsHMA = (*Scheduler)(nil)
var _ sched.MetaBuilder = (*Scheduler)(nil)
var _ sched.PushWork = (*Scheduler)(nil)
var _ sched.Deferrer = (*Scheduler)(nil)

func cdiv(a, b int) int { return (a + b - 1) / b }

func mod(a, b int) int { return ((a % b) + b) % b }
