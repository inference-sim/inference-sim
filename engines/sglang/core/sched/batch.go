// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/managers/schedule_batch.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package sched

import (
	"fmt"
	"slices"
	"sort"

	"github.com/inference-sim/blis-sglang/core/cache"
)

// Mode is a ForwardMode (model_executor/forward_batch_info.py@297e54fd) the normal loop launches.
type Mode string

const (
	Extend Mode = "EXTEND"
	Decode Mode = "DECODE"
	Mixed  Mode = "MIXED"
)

// Batch is ScheduleBatch (schedule_batch.py:2456-4130@297e54fd): the requests of one forward pass
// with their sequence lengths and request rows. The tensors are the slices of the same name.
type Batch struct {
	Reqs           []*Req
	Mode           Mode
	ReqPoolIndices []int
	SeqLens        []int
	PrefixLens     []int
	ExtendLens     []int
	// ExtendNumTokens is the tokens the pass computes.
	ExtendNumTokens int
	BatchIsFull     bool
	ChunkedReq      *Req
	// DecodingReqs are the running requests a mixed pass decodes.
	DecodingReqs             []*Req
	ContainsLastPrefillChunk bool
}

func (b *Batch) Size() int     { return len(b.Reqs) }
func (b *Batch) IsEmpty() bool { return len(b.Reqs) == 0 }

// FilterBatch is filter_batch (:3834-3920): keep the requests that are not finished and not
// excluded, in order. A batch left empty drops its rows.
func (b *Batch) FilterBatch(exclude map[*Req]bool) {
	var keep []int
	for i, r := range b.Reqs {
		if !r.Finished() && !exclude[r] {
			keep = append(keep, i)
		}
	}
	b.keepIndices(keep)
}

func (b *Batch) keepIndices(keep []int) {
	if len(keep) == len(b.Reqs) {
		return
	}
	pick := func(xs []int) []int {
		if len(xs) == 0 {
			return xs
		}
		out := make([]int, 0, len(keep))
		for _, i := range keep {
			out = append(out, xs[i])
		}
		return out
	}
	reqs := make([]*Req, 0, len(keep))
	for _, i := range keep {
		reqs = append(reqs, b.Reqs[i])
	}
	b.Reqs = reqs
	if len(keep) == 0 {
		return
	}
	b.ReqPoolIndices = pick(b.ReqPoolIndices)
	b.SeqLens = pick(b.SeqLens)
}

// MergeBatch is merge_batch (:3922-3990): other's requests follow this batch's.
func (b *Batch) MergeBatch(other *Batch) {
	b.ReqPoolIndices = append(b.ReqPoolIndices, other.ReqPoolIndices...)
	b.SeqLens = append(b.SeqLens, other.SeqLens...)
	b.Reqs = append(b.Reqs, other.Reqs...)
}

// allocator-facing helpers (mem_cache/allocation.py@297e54fd) -----------------------------------

func (s *Scheduler) evictFromTreeCache(numTokens int) {
	if s.tree.SupportsPrefixSharing() {
		s.alloc.EvictToFreeTokens(s.tree, numTokens)
	}
}

// allocTokenSlots is alloc_token_slots (:152-172).
func (s *Scheduler) allocTokenSlots(n int) []int64 {
	s.evictFromTreeCache(n)
	out, ok := s.alloc.Alloc(n)
	if !ok {
		panic(fmt.Sprintf("Out of memory. Try to lower your batch size. Try to allocate %d tokens.", n))
	}
	return out
}

// prefixKVIndices is _prefix_kv_indices (:363-371): a request holding its row reads its prefix
// there; a new one reads its match off the tree.
func (s *Scheduler) prefixKVIndices(r *Req) []int64 {
	if r.C.KV.ReqPoolIdx != nil {
		return slices.Clone(s.pool.Row(*r.C.KV.ReqPoolIdx)[:r.C.PrefixLen])
	}
	if r.C.PrefixLen == 0 {
		return nil
	}
	return s.tree.PrefixDeviceIndices(&r.C)
}

// allocForExtend is alloc_for_extend (:374-545) for the token and paged allocators: request rows
// first, then KV for the extend, written into the rows.
func (s *Scheduler) allocForExtend(b *Batch) {
	prefix := make([][]int64, len(b.Reqs))
	for i, r := range b.Reqs {
		prefix[i] = s.prefixKVIndices(r)
	}
	creqs := make([]*cache.Req, len(b.Reqs))
	for i, r := range b.Reqs {
		creqs[i] = &r.C
	}
	rows, ok := s.pool.Alloc(creqs)
	if !ok {
		panic("alloc_req_slots runs out of memory. Please set a smaller number for `--max-running-requests`.")
	}
	b.ReqPoolIndices = rows
	var out []int64
	if s.cfg.PageSize == 1 {
		out = s.allocTokenSlots(b.ExtendNumTokens)
	} else {
		lastLoc := make([]int64, len(b.Reqs))
		for i := range b.Reqs {
			lastLoc[i] = -1
			if n := len(prefix[i]); n > 0 {
				lastLoc[i] = prefix[i][n-1]
			}
		}
		conservative := b.ExtendNumTokens + len(b.Reqs)*s.cfg.PageSize
		s.evictFromTreeCache(s.alloc.GetExtendAllocationDemand(b.PrefixLens, b.SeqLens, conservative, 1))
		var ok bool
		out, ok = s.alloc.AllocExtend(b.PrefixLens, b.SeqLens, lastLoc, b.ExtendNumTokens)
		if !ok {
			panic(fmt.Sprintf("Prefill out of memory. Try to lower your batch size. Try to allocate %d tokens.", b.ExtendNumTokens))
		}
	}
	pt := 0
	for i, r := range b.Reqs {
		row := rows[i]
		s.pool.Write(row, 0, prefix[i])
		s.pool.Write(row, b.PrefixLens[i], out[pt:pt+b.ExtendLens[i]])
		pt += b.ExtendLens[i]
		r.C.KV.KVAllocatedLen = b.SeqLens[i]
		r.C.KV.KVCommittedLen = b.SeqLens[i]
	}
}

// PrepareForExtend is prepare_for_extend (:2847-3135) without logprobs, embeddings or mamba.
func (b *Batch) PrepareForExtend(s *Scheduler) {
	b.Mode = Extend
	b.PrefixLens, b.ExtendLens, b.SeqLens = nil, nil, nil
	b.ExtendNumTokens = 0
	for _, r := range b.Reqs {
		n := r.ExtendEnd - r.C.PrefixLen
		b.ExtendNumTokens += n
		b.SeqLens = append(b.SeqLens, r.ExtendEnd)
		b.PrefixLens = append(b.PrefixLens, r.C.PrefixLen)
		b.ExtendLens = append(b.ExtendLens, r.ExtendLen())
	}
	s.allocForExtend(b)
	for i, r := range b.Reqs {
		// Only calculate cached_tokens once: once retracted, the stain stays (:2964-2995).
		if !r.RetractedStain {
			r.CachedTokens += b.PrefixLens[i] - r.AlreadyComputed
			r.AlreadyComputed = b.SeqLens[i]
		}
		r.IsRetracted = false
	}
}

// newTokensRequiredNextDecode is new_tokens_required_next_decode (:3403-3419) without
// speculation: a page for each request about to start one.
func (b *Batch) newTokensRequiredNextDecode(s *Scheduler, idx []int) int {
	pages := 0
	for _, i := range idx {
		if b.Reqs[i].C.KV.KVCommittedLen%s.cfg.PageSize == 0 {
			pages++
		}
	}
	return pages * s.cfg.PageSize
}

// CheckDecodeMem is check_decode_mem (:3432-3449): whether the next decode step of the selected
// requests fits; the allocator evicts for it.
func (b *Batch) CheckDecodeMem(s *Scheduler, idx []int) bool {
	if idx == nil {
		idx = make([]int, len(b.Reqs))
		for i := range idx {
			idx[i] = i
		}
	}
	return s.alloc.CheckDecodeCapacity(b.newTokensRequiredNextDecode(s, idx), s.tree)
}

// PrepareForDecode is prepare_for_decode (:3736-3832) without speculation, beams or mamba: KV for
// each request's next token, written into its row, and every sequence one token longer.
func (b *Batch) PrepareForDecode(s *Scheduler) {
	b.Mode = Decode
	bs := len(b.Reqs)
	var out []int64
	if s.cfg.PageSize == 1 {
		out = s.allocTokenSlots(bs)
	} else {
		last := make([]int64, bs)
		next := make([]int, bs)
		for i := range b.Reqs {
			last[i] = s.pool.Row(b.ReqPoolIndices[i])[b.SeqLens[i]-1]
			next[i] = b.SeqLens[i] + 1
		}
		s.evictFromTreeCache(bs * s.cfg.PageSize)
		var ok bool
		out, ok = s.alloc.AllocDecode(next, last)
		if !ok {
			panic("Decode out of memory. Try to lower your batch size.")
		}
	}
	for i, r := range b.Reqs {
		s.pool.Write(b.ReqPoolIndices[i], b.SeqLens[i], out[i:i+1])
		r.C.KV.KVAllocatedLen++
		r.C.KV.KVCommittedLen++
		b.SeqLens[i]++
	}
}

// MixWithRunning is mix_with_running (:3267-3365): the running requests decode in the prefill
// pass, each as an extend of one token after the sequence it has.
func (b *Batch) MixWithRunning(running *Batch) {
	b.Mode = Mixed
	prefixes := make([]int, len(running.Reqs))
	for i, r := range running.Reqs {
		prefixes[i] = running.SeqLens[i] - 1
		r.RefreshFillIDs()
		r.ExtendEnd = prefixes[i] + 1
	}
	n := len(running.Reqs)
	b.MergeBatch(running)
	b.PrefixLens = append(b.PrefixLens, prefixes...)
	for range n {
		b.ExtendLens = append(b.ExtendLens, 1)
	}
	b.ExtendNumTokens += n
}

// retractionOrder is _get_decode_retraction_order (:3542-3580) for the default policy: indices
// from the request to keep first to the one to retract first; the most output first, and among
// equals the shortest prompt first.
func retractionOrder(reqs []*Req) []int {
	idx := make([]int, len(reqs))
	for i := range idx {
		idx[i] = i
	}
	key := func(r *Req) (int, int) { return len(r.C.OutputIDs), -len(r.C.OriginInputIDs) }
	sort.SliceStable(idx, func(a, b int) bool {
		ao, al := key(reqs[idx[a]])
		bo, bl := key(reqs[idx[b]])
		if ao != bo {
			return ao > bo
		}
		return al > bl
	})
	return idx
}

// releaseKVCache is release_kv_cache (mem_cache/common.py:400-444) for a request in no session.
func (s *Scheduler) releaseKVCache(r *Req, checkpoint bool) {
	if r.C.KV.ReqPoolIdx == nil {
		return
	}
	owned := r.OwnedKVLen()
	if checkpoint {
		// The fill-id array lags output_ids until the next prepare_for_decode.
		r.RefreshFillIDs()
		s.tree.Checkpoint(&r.C, owned)
	}
	// The protected prefix is not this request's to free.
	s.tree.FreeKVRow(&r.C, [][2]int{{r.C.KV.CacheProtectedLen, owned}})
	s.tree.Unlock(r.C.Lock)
	r.C.Lock = nil
	s.releaseOverallocated(r, owned, r.C.KV.KVAllocatedLen)
	s.tree.OnRelease(&r.C, checkpoint)
	s.pool.Free(&r.C)
	r.C.KV.KVAllocatedLen = 0
}

// releaseOverallocated is _release_overallocated_kv_indices (common.py:447-476) with no
// speculation: nothing is over-allocated unless a stop landed before the last committed token.
func (s *Scheduler) releaseOverallocated(r *Req, start, end int) {
	if ps := s.cfg.PageSize; ps > 1 {
		start = ceilTo(start, ps)
	}
	if start < end {
		s.tree.FreeKVRow(&r.C, [][2]int{{start, end}})
	}
}

// releaseReq is release_req (:2360-2395): free the request's KV without inserting it in the tree
// and make room at once, then reset it to start over.
func (s *Scheduler) releaseReq(r *Req, remaining int) {
	s.releaseKVCache(r, false)
	// We should use the newly evictable memory instantly.
	s.evictFromTreeCache(remaining * s.cfg.RetractDecodeSteps)
	r.ResetForRetract()
}

// RetractDecode is retract_decode (:3451-3540): requests leave the batch, the one the order puts
// last first, until the next decode step fits; at least one stays. It returns the retracted
// requests and the new_token_ratio estimate.
func (b *Batch) RetractDecode(s *Scheduler) (retracted []*Req, ratio float64) {
	sorted := retractionOrder(b.Reqs)
	first := true
	for first || !b.CheckDecodeMem(s, sorted) {
		if len(sorted) == 1 {
			// Always keep at least one request.
			break
		}
		first = false
		idx := sorted[len(sorted)-1]
		sorted = sorted[:len(sorted)-1]
		r := b.Reqs[idx]
		// Release memory and don't insert into the tree because we need the space instantly.
		s.releaseReq(r, len(sorted))
		retracted = append(retracted, r)
	}
	if len(sorted) <= 1 && !b.CheckDecodeMem(s, sorted) {
		panic(refusedAt("aborting the last request when it cannot fit after retraction", RetractionRefusal))
	}
	b.keepIndices(sorted)
	return retracted, estimateNewTokenRatioAfterRetract(b.Reqs, s.cfg.RetractDecodeSteps)
}

// estimateNewTokenRatioAfterRetract is NewTokenRatioTracker.estimate_new_token_ratio_after_retract
// (scheduler_components/new_token_ratio_tracker.py@297e54fd).
func estimateNewTokenRatioAfterRetract(reqs []*Req, retractSteps int) float64 {
	decoded, maxNew := 0, 0
	for _, r := range reqs {
		decoded += len(r.C.OutputIDs)
		maxNew += r.MaxNewTokens
	}
	return min(1.0, float64(decoded+retractSteps*len(reqs))/float64(maxNew+1))
}
