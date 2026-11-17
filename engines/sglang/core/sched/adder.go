// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/managers/schedule_policy.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package sched

import "github.com/inference-sim/blis-sglang/core/cache"

// AddResult is AddReqResult (schedule_policy.py:571-575@297e54fd).
type AddResult int

const (
	Continue AddResult = iota // keep adding requests
	NoToken                   // no token left
	Other                     // another reason to stop adding requests
)

// admission is _PrefillAdmission (:578-583).
type admission struct {
	prefixLen, extendLen, maxNewTokens int
	isChunked                          bool
}

// adder is PrefillAdder (:607-1778) for one pool, one page-sized allocator and no host tier,
// priority preemption, mamba, sharding or diffusion model.
type adder struct {
	s              *Scheduler
	pageSize       int
	running        *Batch
	newTokenRatio  float64
	remInputTokens int
	// remChunkTokens is nil when chunked prefill is off.
	remChunkTokens  *int
	chunkedReqLimit *int
	budget          *budget

	canRunList    []*Req
	newChunkedReq *Req
}

// newAdder is PrefillAdder.__init__ (:608-743).
func (s *Scheduler) newAdder(running *Batch, numMixedDecodeTokens int) *adder {
	a := &adder{s: s, pageSize: s.cfg.PageSize, running: running, newTokenRatio: s.newTokenRatio.current,
		remInputTokens: s.cfg.MaxPrefillTokens - numMixedDecodeTokens}
	if s.cfg.ChunkedPrefillSize != nil {
		v := *s.cfg.ChunkedPrefillSize - numMixedDecodeTokens
		a.remChunkTokens = &v
	}
	a.budget = newBudget(s.alloc, s.tree, numMixedDecodeTokens)
	if running != nil {
		for _, r := range running.Reqs {
			a.budget.totalOffset += a.runningRequestTotalTokenOffset(r)
		}
	}
	return a
}

// runningRequestTotalTokenOffset is _get_running_request_total_token_offset (:785-795): the decode
// tokens a running request is still expected to take; a fixed-length request takes all of them.
func (a *adder) runningRequestTotalTokenOffset(r *Req) float64 {
	ratio := a.newTokenRatio
	if r.IgnoreEOS {
		ratio = 1.0
	}
	return float64(min(r.MaxNewTokens-len(r.C.OutputIDs), a.s.cfg.ClipMaxNewTokens)) * ratio
}

func (a *adder) ceilPaged(n int) int { return ceilTo(n, a.pageSize) }

// budgetState is budget_state (:874-893).
func (a *adder) budgetState() AddResult {
	if !a.budget.hasCapacity() {
		return NoToken
	}
	if a.remInputTokens <= 0 {
		return Other
	}
	if a.remChunkTokens != nil && *a.remChunkTokens <= 0 {
		return Other
	}
	return Continue
}

// updatePrefillBudget is _update_prefill_budget (:895-952): the compute budgets take the extend
// rounded to pages, the KV budget what the extend and the expected decode need.
func (a *adder) updatePrefillBudget(extendInputLen int, maxNewTokens int) {
	extend := a.ceilPaged(extendInputLen)
	// reserve() already charges one page; the per-request overhead is one page (no sharding).
	a.budget.reserve(extend, float64(maxNewTokens), 0)
	a.remInputTokens -= extend
	if a.remChunkTokens != nil {
		*a.remChunkTokens -= extend
	}
}

// addChunkedReq is add_chunked_req (:1140-1223): the next chunk of the request being prefilled in
// pieces. It returns the request while chunks remain, nil when this chunk finishes it.
func (a *adder) addChunkedReq(r *Req) *Req {
	rem := a.budget.availableChunkTokens(*a.remChunkTokens)
	if a.chunkedReqLimit != nil {
		if *a.chunkedReqLimit <= 0 {
			panic("chunked_req_limit must be positive")
		}
		rem = min(rem, *a.chunkedReqLimit)
	}
	cand := r.InputLen() - r.C.PrefixLen
	truncated := cand > rem
	newLen := min(cand, rem)
	r.ExtendEnd = r.C.PrefixLen + newLen
	a.canRunList = append(a.canRunList, r)
	maxNew := 0
	if !truncated {
		maxNew = min(r.MaxNewTokens, a.s.cfg.ClipMaxNewTokens)
	}
	a.updatePrefillBudget(r.ExtendLen(), maxNew)
	if truncated {
		return r
	}
	return nil
}

// addOneReq is add_one_req (:1395-1590) for a request with no host hit: select a shape under a
// temporary lock on its prefix, then commit it.
func (a *adder) addOneReq(r *Req, hasChunkedReq bool) AddResult {
	if a.s.cfg.Disable && r.IgnoreEOS {
		return a.addOneReqIgnoreEOS(r)
	}
	maxNew := min(max(r.MaxNewTokens-len(r.C.OutputIDs), 0), a.s.cfg.ClipMaxNewTokens)
	cand := r.InputLen() - r.C.PrefixLen
	total := cand + maxNew + a.pageSize

	// The temporary pin excludes this prefix from the evictable budget (_lock_node, :1225-1239).
	pin := a.s.tree.Lock(*r.C.LastNode)
	adm, verdict, ok := a.selectAdmission(r, total, hasChunkedReq)
	if ok {
		a.commit(r, adm)
	}
	a.s.tree.Unlock(pin)
	if !ok {
		return verdict
	}
	// This verdict controls the next candidate, not the committed request.
	return a.budgetState()
}

// selectAdmission is _select_prefill_admission (:1592-1683): the prefill shape without allocating
// or publishing KV.
func (a *adder) selectAdmission(r *Req, totalTokens int, hasChunkedReq bool) (admission, AddResult, bool) {
	prefix := r.C.PrefixLen
	extend := r.InputLen() - prefix
	inputTokens := a.ceilPaged(extend)
	canAdmit, chunkTokensLimit := a.budget.checkPrefill(totalTokens, a.remChunkTokens)
	if !canAdmit {
		return admission{}, NoToken, false
	}
	// Without chunking, allow the first request even above the input cap.
	if a.remChunkTokens == nil && len(a.canRunList) > 0 && inputTokens >= a.remInputTokens {
		return admission{}, Other, false
	}
	isChunked := false
	maxNew := min(r.MaxNewTokens, a.s.cfg.ClipMaxNewTokens)
	if chunkTokensLimit != nil && inputTokens > *chunkTokensLimit {
		if hasChunkedReq && a.s.policy == ShortestPrefillFirst {
			// Only one unfinished chunked request can be tracked.
			return admission{}, Other, false
		}
		extend = *chunkTokensLimit / a.pageSize * a.pageSize
		end := (prefix + extend) / a.pageSize * a.pageSize
		extend = end - prefix
		if extend <= 0 {
			return admission{}, Other, false
		}
		isChunked = true
		maxNew = 0
	}
	return admission{prefix, extend, maxNew, isChunked}, Continue, true
}

// commit is _commit_prefill_admission (:1685-1705).
func (a *adder) commit(r *Req, adm admission) {
	r.ExtendEnd = adm.prefixLen + adm.extendLen
	lock := a.s.tree.Lock(*r.C.LastNode)
	r.C.Lock = lock
	a.canRunList = append(a.canRunList, r)
	if adm.isChunked {
		a.newChunkedReq = r
	}
	a.updatePrefillBudget(adm.extendLen, adm.maxNewTokens)
}

// addOneReqIgnoreEOS is add_one_req_ignore_eos (:1241-1383) for a fixed-length request with the
// radix cache off: it reserves the request's whole generation, and admits whole or not at all
// (chunking of such a request is not modelled).
func (a *adder) addOneReqIgnoreEOS(r *Req) AddResult {
	panic(refused("add_one_req_ignore_eos: a fixed-length request with the radix cache off"))
}

var _ = cache.FullComponentType
