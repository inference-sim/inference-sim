// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/managers/schedule_batch.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

// Package sched is SGLang v0.5.23's scheduling loops, normal and overlap (managers/scheduler.py,
// schedule_policy.py, schedule_batch.py, scheduler_components/batch_result_processor.py at
// 297e54fd) over the cache framework of core/cache. A step is one iteration of
// Scheduler.event_loop_normal or Scheduler.event_loop_overlap. Nothing here reads a clock or starts a goroutine: the loop's time
// is the step count.
package sched

import (
	"fmt"
	"slices"

	"github.com/inference-sim/inference-sim/engines/sglang/core/cache"
)

// Req is Req (schedule_batch.py:1033-2250@297e54fd) as the normal loop reads it. C is the cache
// framework's view of the same request: its tokens, its KV record, the node it matched and the lock
// it holds there.
type Req struct {
	C cache.Req
	// MaxNewTokens is sampling_params.max_new_tokens after init_req_max_new_tokens clipped it.
	MaxNewTokens int
	IgnoreEOS    bool
	// Script is the tokens the model will sample for this request, in order, when the caller knows
	// them (the engine's Request.Output); empty means the scheduler's one sampled token.
	Script []int64
	// Priority and RoutingKey are the request's own (schedule_batch.py:1033@297e54fd). Priority is
	// meaningful only once _set_or_validate_priority has run; HasPriority says the client set it.
	Priority    int
	HasPriority bool
	RoutingKey  string

	// ExtendEnd is the position the request's current extend stops at (extend_end).
	ExtendEnd int
	// NumMatchedPrefixTokens is the prefix the last match found, capped to leave one token to
	// compute (num_matched_prefix_tokens).
	NumMatchedPrefixTokens int
	// InflightMiddleChunks counts chunks launched whose result is not processed (:2049).
	InflightMiddleChunks int
	IsRetracted          bool
	// RetractedStain stays true once the request was retracted (:2049).
	RetractedStain bool
	// FinishedReason is "" until the request finishes: "length", "matched_token" or "abort".
	FinishedReason string
	FinishedLen    *int
	// ToFinish is a finish reason set for the next update_finish_state to apply.
	ToFinish string

	// ArrivalProcessedTokens is the prefill-token counter when the request entered the waiting
	// queue, which hrrn ages from (_add_request_to_queue, scheduler.py:3495@297e54fd).
	ArrivalProcessedTokens int
	// WaitQueueEntryStep is the step the request entered the waiting queue.
	WaitQueueEntryStep int
	AlreadyComputed    int
	CachedTokens       int
	RetractionCount    int
}

// Finished is Req.finished (schedule_batch.py:1657).
func (r *Req) Finished() bool { return r.FinishedReason != "" }

// ExtendLen is Req.extend_len (:1666).
func (r *Req) ExtendLen() int { return r.ExtendEnd - r.C.PrefixLen }

// RID is the request id.
func (r *Req) RID() string { return r.C.ID }

// InputLen is len(full_untruncated_fill_ids).
func (r *Req) InputLen() int { return len(r.C.FillIDs) }

// RefreshFillIDs is refresh_fill_ids (:1672-1690): the fill ids are the prompt then the output.
func (r *Req) RefreshFillIDs() {
	r.C.FillIDs = append(slices.Clone(r.C.OriginInputIDs), r.C.OutputIDs...)
}

// ComputeMaxPrefixLen is _compute_max_prefix_len (:1766-1790) without logprobs or diffusion
// models: a match leaves at least one token to compute.
func (r *Req) ComputeMaxPrefixLen(inputLen int) int { return max(inputLen-1, 0) }

// NextOutputFinishesByLength is next_output_finishes_by_length (:1664).
func (r *Req) NextOutputFinishesByLength() bool {
	return len(r.C.OutputIDs)+1 >= r.MaxNewTokens
}

// UpdateFinishState is update_finish_state (:2007-2047) for one accepted token: a pending finish
// (an abort) first, then the model's end-of-sequence token unless the request ignores it, then
// the length cap. Out-of-vocabulary tokens, stop strings and grammars are refused at
// construction.
func (r *Req) UpdateFinishState(eos []int64) {
	if r.Finished() {
		return
	}
	if r.ToFinish != "" {
		r.FinishedReason, r.ToFinish = r.ToFinish, ""
		r.C.Finished = true
		return
	}
	last := r.C.OutputIDs[len(r.C.OutputIDs)-1]
	if !r.IgnoreEOS && slices.Contains(eos, last) {
		r.FinishedReason = "matched_token"
		n := len(r.C.OutputIDs)
		r.FinishedLen = &n
		r.C.Finished = true
		return
	}
	if len(r.C.OutputIDs) >= r.MaxNewTokens {
		r.FinishedReason = "length"
		n := r.MaxNewTokens
		r.FinishedLen = &n
		r.C.Finished = true
	}
}

// OwnedKVLen is owned_kv_len (:1552-1570): the committed KV, less any output past a stop.
func (r *Req) OwnedKVLen() int {
	if r.FinishedLen == nil {
		return r.C.KV.KVCommittedLen
	}
	return min(r.C.KV.KVCommittedLen, len(r.C.OriginInputIDs)+min(len(r.C.OutputIDs), *r.FinishedLen))
}

// ResetForRetract is reset_for_retract (:2049-2100): the request starts over from its prompt and
// the output it has, its KV already released.
func (r *Req) ResetForRetract() {
	r.RetractionCount++
	r.C.PrefixLen = 0
	r.C.LastNode = nil
	r.C.KV.CacheProtectedLen = 0
	r.C.KV.CacheInsertedLen = 0
	r.NumMatchedPrefixTokens = 0
	r.C.Lock = nil
	r.ExtendEnd = 0
	r.IsRetracted = true
	r.RetractedStain = true
	r.InflightMiddleChunks = 0
	r.AlreadyComputed = 0
	if r.C.KV.ReqPoolIdx != nil {
		panic("expect it is already released")
	}
	r.C.KV.KVCommittedLen = 0
}

// Arrival is a request as the tokenizer hands it to the scheduler.
type Arrival struct {
	RID          string
	Prompt       []int64
	MaxNewTokens int
	IgnoreEOS    bool
	// Script is the request's assigned output (see Req.Script); MaxNewTokens must not exceed its
	// length.
	Script []int64
	// CacheSalt isolates the request's prefix-cache entries (empty: none).
	CacheSalt string
	// Priority is nil when the client sent none. RoutingKey is empty when it sent none.
	Priority   *int
	RoutingKey string
}

func (a Arrival) String() string { return fmt.Sprintf("%s(%d tokens)", a.RID, len(a.Prompt)) }

// ceilTo rounds n up to a multiple of m (Python's -(-n // m) * m; Go's division truncates).
func ceilTo(n, m int) int { return (n + m - 1) / m * m }
