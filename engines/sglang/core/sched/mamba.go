// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/managers/schedule_batch.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).
// Derived from python/sglang/srt/managers/scheduler.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).
// Derived from python/sglang/srt/managers/scheduler_components/batch_result_processor.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).
// Derived from python/sglang/srt/mem_cache/allocation.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).

package sched

import (
	"fmt"
	"slices"

	"github.com/inference-sim/inference-sim/engines/sglang/core/cache"
)

// The scheduler's handling of a hybrid (linear-attention) model's recurrent state
// (inference-sim/inference-sim#2366, #2442). Each request holds a working state slot and, with the
// extra buffer, tracking slots: the places a forward pass leaves a snapshot of its state where the
// prefix tree can take it over. The batch carries, per pass, which requests leave a snapshot, at
// which length and into which slot; the result of a decode pass moves the tracking position on. The
// model runner that writes the snapshots is not modelled, only the slots and lengths it is told.

// MambaConfig is what the scheduler reads of the hybrid model's state settings: the extra buffer
// strategy (get_exec().mamba), the pools' sizes and the chunk sizes the checkpoints land on.
type MambaConfig struct {
	ExtraBuffer      bool `json:"extra_buffer"`
	Lazy             bool `json:"lazy"`
	PathStateCap     int  `json:"path_state_cap"`
	CheckpointsLevel int  `json:"checkpoints_level"`
	MambaSize        int  `json:"mamba_size"`
	PingPongSize     int  `json:"ping_pong_size"`
	// InitialTrackingSlots is the tracking slots a request takes at its first row.
	InitialTrackingSlots int `json:"initial_tracking_slots"`
	// TrackInterval is mamba_track_interval: a decode checkpoint lands on a multiple of it.
	TrackInterval int `json:"track_interval"`
	// StateChunkSize is the model's mamba_chunk_size, which the state kernel works in.
	StateChunkSize int `json:"state_chunk_size"`
	// TreeSlotInterval is mamba_cache_chunk_size: max(StateChunkSize, page size).
	TreeSlotInterval int `json:"tree_slot_interval"`
}

// mambaTrack is the batch's mamba fields (ScheduleBatch.mamba_*, schedule_batch.py:2105-2125). A nil
// slice is a field that is None.
type mambaTrack struct {
	Indices        []int64
	BufferIndices  []int
	Mask           []bool
	Seqlens        []int
	SeqlensCPU     []int
	PrefillMaskCPU []bool
	MaskCPU        []bool
	MaskNextCPU    []bool
	DecodeBatchIdx []int
	CowDst         []int64
	ClearIdx       []int64
}

// dropTracks is what merge_batch does to the fields (schedule_batch.py:3498-3507): the tracking
// arrays go, the deferred copy and clear lists stay.
func (m *mambaTrack) dropTracks() {
	*m = mambaTrack{CowDst: m.CowDst, ClearIdx: m.ClearIdx}
}

// MambaBatch is the batch's tracking arrays as the oracle's forward record names them
// (oracle/run_sched_mamba.py BATCH_FIELDS): the fields that are not None.
type MambaBatch struct {
	PrefillTrackMaskCPU []bool  `json:"mamba_prefill_track_mask_cpu,omitempty"`
	TrackMaskCPU        []bool  `json:"mamba_track_mask_cpu,omitempty"`
	TrackSeqlensCPU     []int   `json:"mamba_track_seqlens_cpu,omitempty"`
	TrackBufferIndices  []int   `json:"mamba_track_buffer_indices,omitempty"`
	TrackIndices        []int64 `json:"mamba_track_indices,omitempty"`
	TrackMask           []bool  `json:"mamba_track_mask,omitempty"`
	TrackSeqlens        []int   `json:"mamba_track_seqlens,omitempty"`
	TrackMaskNextCPU    []bool  `json:"mamba_track_mask_next_cpu,omitempty"`
	CowDstIndices       []int64 `json:"mamba_cow_dst_indices,omitempty"`
}

func (m mambaTrack) record() *MambaBatch {
	return &MambaBatch{PrefillTrackMaskCPU: slices.Clone(m.PrefillMaskCPU), TrackMaskCPU: slices.Clone(m.MaskCPU),
		TrackSeqlensCPU: slices.Clone(m.SeqlensCPU), TrackBufferIndices: slices.Clone(m.BufferIndices),
		TrackIndices: slices.Clone(m.Indices), TrackMask: slices.Clone(m.Mask), TrackSeqlens: slices.Clone(m.Seqlens),
		TrackMaskNextCPU: slices.Clone(m.MaskNextCPU), CowDstIndices: slices.Clone(m.CowDst)}
}

// ReqMamba is a request's recurrent-state fields as the oracle records them (req_state).
type ReqMamba struct {
	RID             string  `json:"rid"`
	PoolIdx         *int64  `json:"mamba_pool_idx"`
	PingPongBuffer  []int64 `json:"mamba_ping_pong_track_buffer"`
	PingPongEmpty   []bool  `json:"mamba_ping_pong_empty"`
	NextTrackIdx    *int    `json:"mamba_next_track_idx"`
	LastTrackIdx    *int    `json:"mamba_last_track_idx"`
	LastTrackSeqlen *int    `json:"mamba_last_track_seqlen"`
	PrevTrackSeqlen *int    `json:"mamba_prev_track_seqlen"`
	CowSrcIndex     *int64  `json:"mamba_cow_src_index"`
	NeedsClear      bool    `json:"mamba_needs_clear"`
	Branching       *int    `json:"branching"`
}

// MambaState is the recurrent state the scheduler holds between two iterations: the free state
// slots, the tree's state ledgers and each live request's slots and tracked lengths.
type MambaState struct {
	Available     int        `json:"mamba_available"`
	TreeEvictable int        `json:"tree_mamba_evictable"`
	TreeProtected int        `json:"tree_mamba_protected"`
	RunningMamba  []ReqMamba `json:"running_mamba"`
	WaitingMamba  []ReqMamba `json:"waiting_mamba"`
	ChunkedMamba  []ReqMamba `json:"chunked_mamba"`
}

func cp[T any](p *T) *T {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}

func reqMamba(r *Req) ReqMamba {
	out := ReqMamba{RID: r.RID(), Branching: r.C.MambaBranchingSeqlen}
	if m := r.C.KV.Mamba; m != nil {
		// The record is a snapshot: the request's own fields move on.
		out.PoolIdx, out.PingPongBuffer, out.PingPongEmpty = cp(m.PoolIdx), slices.Clone(m.PingPongBuffer), slices.Clone(m.PingPongEmpty)
		out.NextTrackIdx, out.LastTrackIdx = cp(m.NextTrackIdx), cp(m.LastTrackIdx)
		out.LastTrackSeqlen, out.PrevTrackSeqlen = cp(m.LastTrackSeqlen), cp(m.PrevTrackSeqlen)
		out.CowSrcIndex, out.NeedsClear = cp(m.CowSrcIndex), m.NeedsClear
	}
	out.Branching = cp(out.Branching)
	return out
}

// mambaLedger is the tree's state ledgers.
type mambaLedger interface {
	MambaEvictableSize() int
	MambaProtectedSize() int
}

func (s *Scheduler) mambaState() *MambaState {
	l := s.tree.(mambaLedger)
	st := &MambaState{Available: s.hybrid.Mamba.AvailableSize(), TreeEvictable: l.MambaEvictableSize(),
		TreeProtected: l.MambaProtectedSize(), RunningMamba: []ReqMamba{}, WaitingMamba: []ReqMamba{}, ChunkedMamba: []ReqMamba{}}
	for _, r := range s.running.Reqs {
		st.RunningMamba = append(st.RunningMamba, reqMamba(r))
	}
	for _, r := range s.waiting {
		st.WaitingMamba = append(st.WaitingMamba, reqMamba(r))
	}
	if s.chunkedReq != nil {
		st.ChunkedMamba = append(st.ChunkedMamba, reqMamba(s.chunkedReq))
	}
	return st
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func lcm(a, b int) int { return a / gcd(a, b) * b }

// checkpointGrid is mamba_checkpoint_grid (runtime_context.py:1998): the granularity a donated
// checkpoint's depth lands on.
func (s *Scheduler) checkpointGrid() int { return lcm(s.cfg.Mamba.TreeSlotInterval, s.cfg.PageSize) }

// trackGrid is mamba_track_grid (runtime_context.py:2006): the same for a decode checkpoint, which
// also lands on mamba_track_interval.
func (s *Scheduler) trackGrid() int { return lcm(s.checkpointGrid(), s.cfg.Mamba.TrackInterval) }

func (s *Scheduler) advance(r *Req, u cache.TrackingAdvance) { s.hybrid.AdvanceTracking(&r.C, u) }

func posp(v int) *int { return &v }

// ensureMambaCapacity is ensure_mamba_capacity (allocation.py:257-291@c4af57ff) for a static pool:
// the state slots the requests will take, made free by evicting cached states if they are not.
func (s *Scheduler) ensureMambaCapacity(reqs []*Req) {
	if s.hybrid == nil {
		return
	}
	needed := 0
	for _, r := range reqs {
		if r.C.KV.Mamba == nil || !r.C.KV.Mamba.HoldsMamba() {
			needed++
		}
		if s.hybrid.ExtraBuffer() && (r.C.KV.Mamba == nil || r.C.KV.Mamba.PingPongBuffer == nil) {
			needed += s.hybrid.InitialTrackingSlots()
		}
	}
	if s.hybrid.Mamba.AvailableSize() >= needed {
		return
	}
	if s.tree.SupportsMamba() {
		s.tree.EvictForAlloc(cache.KVDemand{StateSlots: needed})
	}
}

// allocTrackingSlots is alloc_tracking_slots (allocation.py:294-312@c4af57ff): a tracking slot for
// each candidate position, after one eviction of cached states if the free slots fall short; false
// for a candidate that still gets none.
func (s *Scheduler) allocTrackingSlots(reqs []*Req, positions []int) []bool {
	if len(reqs) > 0 && s.hybrid.Mamba.AvailableSize() < len(reqs) && s.tree.SupportsMamba() {
		s.tree.EvictForAlloc(cache.KVDemand{StateSlots: len(reqs)})
	}
	out := make([]bool, len(reqs))
	for i, r := range reqs {
		out[i] = s.hybrid.TakeTrackingSlot(&r.C, positions[i])
	}
	return out
}

// dropUnadmittedMatch is _drop_unadmitted_match (scheduler.py:4601-4609): what the prefix match
// staged for a request this pass does not admit goes back.
func (s *Scheduler) dropUnadmittedMatch(r *Req) {
	if s.hybrid == nil || r.C.KV.Mamba == nil {
		return
	}
	r.C.KV.Mamba.CowSrcIndex = nil
	r.C.KV.Mamba.NeedsClear = false
	if r.C.KV.Mamba.HoldsMamba() {
		s.hybrid.ReleaseStateSlot(&r.C)
	}
}

// prepareMambaForExtend is the mamba part of prepare_for_extend (schedule_batch.py:2557-2562,
// 2677-2698, and _mamba_radix_cache_v2_req_prepare_for_extend, 2708-2810): for each request whether
// this pass leaves a state snapshot, into which slot and at which length, then the deferred copies.
func (b *Batch) prepareMambaForExtend(s *Scheduler) {
	if s.hybrid == nil {
		return
	}
	if s.cfg.Mamba.ExtraBuffer {
		n := len(b.Reqs)
		m := &b.Mamba
		m.PrefillMaskCPU, m.SeqlensCPU = make([]bool, n), make([]int, n)
		positions := make([]int, n)
		for i, r := range b.Reqs {
			m.PrefillMaskCPU[i], positions[i], m.SeqlensCPU[i] = s.prepareTrackForExtend(r)
		}
		m.Indices = make([]int64, n)
		for i := range b.Reqs {
			m.Indices[i] = s.hybrid.TrackSlot(b.ReqPoolIndices[i], positions[i])
		}
		m.Mask = slices.Clone(m.PrefillMaskCPU)
		m.Seqlens = slices.Clone(m.SeqlensCPU)
	}
	b.collectDeferredMamba()
}

// prepareTrackForExtend is _mamba_radix_cache_v2_req_prepare_for_extend without data-parallel
// context: whether the extend tracks (a boundary on the checkpoint grid, or the branching point),
// the ping-pong position it writes, and the length it reports.
func (s *Scheduler) prepareTrackForExtend(r *Req) (mask bool, position, seqlen int) {
	cacheChunk, stateChunk, grid := s.cfg.Mamba.TreeSlotInterval, s.cfg.Mamba.StateChunkSize, s.checkpointGrid()
	prefix, extend, end := r.C.PrefixLen, r.ExtendLen(), r.ExtendEnd
	forceTrackH := func(i int) int {
		if (i-prefix)%cacheChunk != 0 {
			panic(fmt.Sprintf("the force track calculation only handles last-position or unaligned seqlens, so it needs a chunk-aligned offset to start from: i=%d prefix_len=%d chunk_size=%d", i, prefix, cacheChunk))
		}
		return i + 1
	}
	// prefill_checkpoint_boundary (schedule_batch.py:1976): the last whole grid multiple past the prefix.
	mask = extend >= grid
	aligned := prefix
	if mask {
		aligned = prefix + extend/grid*grid
	}
	m := r.C.KV.Mamba
	position = -1
	if m.NextTrackIdx != nil {
		position = *m.NextTrackIdx
	}
	seqlen = -1
	if mask {
		seqlen = end
		if prefix+extend/stateChunk*stateChunk != aligned {
			seqlen = forceTrackH(aligned)
		}
		// branch_checkpoint (schedule_batch.py:1987): the branching point, if it lies inside this extend.
		if br := r.C.MambaBranchingSeqlen; br != nil && prefix < *br && *br < seqlen && (*br-prefix)%cacheChunk == 0 {
			seqlen = forceTrackH(*br)
			aligned = *br
		}
		next := s.hybrid.PingPongOtherIdx(position)
		if s.cfg.Mamba.Lazy {
			next = position
		}
		s.advance(r, cache.TrackingAdvance{HasBoundary: true, Boundary: aligned, LastPosition: posp(position), NextPosition: posp(next)})
	}
	return mask, position, seqlen
}

// collectDeferredMamba is _collect_deferred_mamba_cow_and_clear (schedule_batch.py:2812-2832): the
// copies of a matched state into a request's own slot, and the slots to zero, that the forward stream
// performs.
func (b *Batch) collectDeferredMamba() {
	holds := false
	for _, r := range b.Reqs {
		holds = holds || r.C.KV.Mamba.HoldsMamba()
	}
	if !holds {
		return
	}
	b.Mamba.CowDst, b.Mamba.ClearIdx = nil, nil
	for _, r := range b.Reqs {
		m := r.C.KV.Mamba
		switch {
		case m.CowSrcIndex != nil:
			b.Mamba.CowDst = append(b.Mamba.CowDst, *m.PoolIdx)
			m.CowSrcIndex, m.NeedsClear = nil, false
		case m.NeedsClear:
			b.Mamba.ClearIdx = append(b.Mamba.ClearIdx, *m.PoolIdx)
			m.NeedsClear = false
		}
	}
}

// prepareMambaForDecode is the mamba part of prepare_for_decode (schedule_batch.py:3285-3296,
// 3343-3376): a decode pass tracks a request whose sequence reaches a multiple of the track grid.
func (b *Batch) prepareMambaForDecode(s *Scheduler) {
	if s.hybrid == nil {
		return
	}
	m := &b.Mamba
	m.SeqlensCPU, m.PrefillMaskCPU, m.CowDst, m.ClearIdx = nil, nil, nil, nil
	if !s.cfg.Mamba.ExtraBuffer {
		return
	}
	interval := s.trackGrid()
	n := len(b.Reqs)
	mask := make([]bool, n)
	for i := range b.Reqs {
		mask[i] = b.SeqLens[i]%interval == 0
	}
	if n == 0 {
		m.Indices, m.BufferIndices = []int64{}, []int{}
	} else {
		if s.cfg.Mamba.Lazy {
			mask = b.lazyPreallocAtBoundary(s, mask)
		}
		b.setTrackIndicesFromReqs(s)
	}
	m.MaskCPU = slices.Clone(mask)
	m.MaskNextCPU = nil
	if s.cfg.EnableOverlap {
		m.MaskNextCPU = make([]bool, n)
		for i := range b.Reqs {
			m.MaskNextCPU[i] = b.SeqLens[i]%interval == interval-1
		}
	}
	m.DecodeBatchIdx = make([]int, n)
	for i, r := range b.Reqs {
		m.DecodeBatchIdx[i] = r.DecodeBatchIdx
	}
	m.Mask = slices.Clone(mask)
}

// setTrackIndicesFromReqs is set_mamba_track_indices_from_reqs (schedule_batch.py:1805-1830): the
// slot each request's next tracking position names.
func (b *Batch) setTrackIndicesFromReqs(s *Scheduler) {
	n := len(b.Reqs)
	b.Mamba.BufferIndices, b.Mamba.Indices = make([]int, n), make([]int64, n)
	for i, r := range b.Reqs {
		pos := -1
		if nx := r.C.KV.Mamba.NextTrackIdx; nx != nil {
			pos = *nx
			b.Mamba.BufferIndices[i] = *nx
		}
		b.Mamba.Indices[i] = s.hybrid.TrackSlot(b.ReqPoolIndices[i], pos)
	}
}

// lazyPreallocAtBoundary is mamba_lazy_prealloc_at_boundary (schedule_batch.py:3203-3232): a request
// at a boundary takes the other tracking slot, so the forward writes its snapshot there and never over
// the kept one; a request that gets none has its bit cleared.
func (b *Batch) lazyPreallocAtBoundary(s *Scheduler, mask []bool) []bool {
	mask = slices.Clone(mask)
	var idx, positions []int
	var reqs []*Req
	for i, r := range b.Reqs {
		m := r.C.KV.Mamba
		if m.PingPongBuffer == nil {
			panic("lazy tracking needs a ping-pong buffer")
		}
		other := 1 - *m.NextTrackIdx
		switch {
		case mask[i] && m.PingPongEmpty[other]:
			idx, reqs, positions = append(idx, i), append(reqs, r), append(positions, other)
		case mask[i]:
			// A held other slot means the previous boundary's post-processing, which frees it, has not run yet under overlap.
			mask[i] = false
		}
	}
	took := s.allocTrackingSlots(reqs, positions)
	for k, ok := range took {
		if ok {
			s.advance(reqs[k], cache.TrackingAdvance{NextPosition: posp(positions[k])})
		} else {
			mask[idx[k]] = false
		}
	}
	return mask
}

// lazyUndoPreallocAtBoundary is mamba_lazy_undo_prealloc_at_boundary (schedule_batch.py:3234-3250): the
// slots a decode batch took at a boundary go back when it rides a mixed pass, which writes none.
func (b *Batch) lazyUndoPreallocAtBoundary(s *Scheduler) {
	if b.Mamba.MaskCPU == nil {
		return
	}
	for i, r := range b.Reqs {
		if !b.Mamba.MaskCPU[i] {
			continue
		}
		next := *r.C.KV.Mamba.NextTrackIdx
		s.advance(r, cache.TrackingAdvance{NextPosition: posp(1 - next), ReleasePositions: []int{next}})
	}
}

// mixMamba is the mamba part of mix_with_running (schedule_batch.py:2888-2919): the prefill rows keep
// their tracking through the merge, the decode rows track nothing (slot 0, length -1).
func (b *Batch) mixMamba(s *Scheduler, running *Batch, merge func()) {
	if s.hybrid == nil || b.Mamba.Indices == nil {
		merge()
		return
	}
	if s.cfg.Mamba.Lazy {
		running.lazyUndoPreallocAtBoundary(s)
	}
	t := b.Mamba
	merge()
	n := len(running.Reqs)
	b.Mamba.Indices = append(slices.Clone(t.Indices), make([]int64, n)...)
	b.Mamba.Mask = append(slices.Clone(t.Mask), make([]bool, n)...)
	b.Mamba.Seqlens = slices.Clone(t.Seqlens)
	b.Mamba.SeqlensCPU = slices.Clone(t.SeqlensCPU)
	b.Mamba.PrefillMaskCPU = slices.Clone(t.PrefillMaskCPU)
	for range n {
		b.Mamba.Seqlens = append(b.Mamba.Seqlens, -1)
		b.Mamba.SeqlensCPU = append(b.Mamba.SeqlensCPU, -1)
		b.Mamba.PrefillMaskCPU = append(b.Mamba.PrefillMaskCPU, false)
	}
}

// mambaDecodeResult is the mamba part of _handle_finish_state_updated_req (batch_result_processor.py:
// 1582-1620) and _mamba_prefix_cache_update (1745-1812): the result of a decode pass moves each
// request's tracked length and ping-pong position on.
func (s *Scheduler) mambaDecodeResult(b *Batch, i int, r *Req) {
	if s.hybrid == nil || !s.cfg.Mamba.ExtraBuffer {
		return
	}
	lazy := s.cfg.Mamba.Lazy
	var known, completed *bool
	lookahead := 0
	if b.Mamba.MaskCPU != nil {
		c := b.Mamba.MaskCPU[i]
		completed = &c
		lookahead = r.DecodeBatchIdx - b.Mamba.DecodeBatchIdx[i]
		if lookahead != 0 && lookahead != 1 {
			panic(fmt.Sprintf("mamba result lookahead=%d for req %s; overlap advanced more than one decode batch", lookahead, r.RID()))
		}
		k := b.Mamba.MaskCPU[i]
		if lookahead != 0 {
			k = b.Mamba.MaskNextCPU[i]
		}
		known = &k
		if c && !lazy {
			s.advance(r, cache.TrackingAdvance{HasBoundary: true, Boundary: r.C.KV.KVCommittedLen - lookahead,
				LastPosition: posp(b.Mamba.BufferIndices[i]), PreservePrevious: true})
		}
	}
	update := known
	if lazy {
		update = completed
	}
	if update == nil || *update {
		s.mambaPrefixCacheUpdate(b, i, r, !lazy && known != nil && *known)
	}
}

func (s *Scheduler) mambaPrefixCacheUpdate(b *Batch, i int, r *Req, knownBoundary bool) {
	m := r.C.KV.Mamba
	if m == nil || m.PingPongBuffer == nil {
		return
	}
	lazy := s.cfg.Mamba.Lazy
	var trackSeqlen int
	if knownBoundary {
		// _mamba_assert_committed_len_lookahead
		if len(r.C.OutputIDs) == 0 {
			panic("mamba track boundary reached before a decode token was appended")
		}
		if lead := r.C.KV.KVCommittedLen - (len(r.C.OriginInputIDs) + len(r.C.OutputIDs) - 1); lead != 0 && lead != 1 {
			panic(fmt.Sprintf("mamba track boundary: kv_committed_len leads seq_len by %d (req %s); overlap lookahead wider than assumed", lead, r.RID()))
		}
		trackSeqlen = r.C.KV.KVCommittedLen
		if trackSeqlen%s.trackGrid() != 0 {
			panic("mamba track boundary off the track grid")
		}
	} else {
		// _mamba_check_track_boundary without speculation: the forward's mask, reproduced from the
		// committed length less the overlap lookahead.
		committed := r.C.KV.KVCommittedLen - (r.DecodeBatchIdx - b.Mamba.DecodeBatchIdx[i])
		if committed%s.trackGrid() != 0 {
			return
		}
		trackSeqlen = committed
	}
	trackIdx := *m.NextTrackIdx
	if !knownBoundary && b.Mamba.BufferIndices != nil {
		trackIdx = b.Mamba.BufferIndices[i]
	}
	if lazy {
		if !knownBoundary {
			s.advance(r, cache.TrackingAdvance{HasBoundary: true, Boundary: trackSeqlen})
		}
		// mamba_lazy_post_decode_at_boundary: commit the boundary and free its old slot.
		s.advance(r, cache.TrackingAdvance{LastPosition: posp(trackIdx), NextPosition: posp(trackIdx), ReleasePositions: []int{1 - trackIdx}})
		return
	}
	if !knownBoundary {
		s.advance(r, cache.TrackingAdvance{HasBoundary: true, Boundary: trackSeqlen, LastPosition: posp(trackIdx), PreservePrevious: true})
	}
	s.advance(r, cache.TrackingAdvance{NextPosition: posp(s.hybrid.PingPongOtherIdx(trackIdx))})
}
