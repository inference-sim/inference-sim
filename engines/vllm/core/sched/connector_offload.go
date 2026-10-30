// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/sched/scheduler.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/core/sched/output.py (SchedulerOutput, CachedRequestData),
// vllm/distributed/kv_transfer/kv_connector/utils.py (yield_req_data) and
// vllm/distributed/kv_transfer/kv_connector/v1/base.py (build_connector_meta,
// has_pending_push_work).

package sched

import (
	"sort"

	"github.com/inference-sim/blis-vllm/core/kv"
)

// The hooks a connector that plans its own transfers each step uses: vLLM's Scheduler hands it
// the step's SchedulerOutput in build_connector_meta (scheduler.py:1527-1530), keeps the engine
// stepping while it reports pending work (has_pending_push_work, :2823-2838), lets it defer a
// request whose hit it cannot size yet (get_num_new_matched_tokens returning None, :983-988),
// and hands it the worker's report (KVConnectorOutput.kv_connector_worker_meta,
// vllm/v1/outputs.py:221). A connector implements whichever it needs; NIXL needs none.

// MetaBuilder is a connector whose build_connector_meta reads the step's scheduler output.
type MetaBuilder interface {
	// BuildMeta is build_connector_meta (scheduler.py:1527-1530), called after the step is
	// scheduled and before _update_after_schedule advances the requests' computed tokens. Its
	// result is the step's Output.KVConnectorMetadata.
	BuildMeta(step StepOutput) any
}

// PushWork is a connector whose has_pending_push_work keeps the engine stepping
// (scheduler.py:2823-2838): HasRequests is true while it reports work.
type PushWork interface {
	HasPendingPushWork() bool
}

// Deferrer is a connector whose get_num_new_matched_tokens may return None: the connector cannot
// size the hit yet, and the scheduler skips the request this step (scheduler.py:983-988).
type Deferrer interface {
	// MatchedTokensOrDefer is MatchedTokens with defer: true is None.
	MatchedTokensOrDefer(r *Request, localComputed int) (tokens int, async, deferred bool)
}

// StepOutput is the part of a step's SchedulerOutput a connector reads (vllm/v1/core/sched/
// output.py:229-292@5df63a72).
type StepOutput struct {
	// Scheduled is num_scheduled_tokens; Order its insertion order (Output.Order).
	Scheduled map[string]int
	Order     []string
	// Preempted is preempted_req_ids.
	Preempted []string
	// Finished is finished_req_ids: the requests finished since the last step, in request-id
	// order. vLLM's is a set of strings, whose iteration order depends on the process's hash
	// seed; the oracle fixes it to sorted order (oracle/mocks.json, "finished_req_ids iteration
	// order"), and this matches it.
	Finished []string
	// Reqs are the scheduled requests as yield_req_data yields them
	// (vllm/distributed/kv_transfer/kv_connector/utils.py:295-312): scheduled_new_reqs, then
	// scheduled_cached_reqs (running, then resumed).
	Reqs []StepRequest
}

// StepRequest is one request of yield_req_data: its new block ids per KV cache group (nil when
// none, get_block_ids(allow_none=True), kv_cache_manager.py:81-107) and whether it resumed from
// preemption (resumed_req_ids).
type StepRequest struct {
	ID        string
	NewBlocks [][]int
	Resumed   bool
}

// BlockHashes is the request's block_hashes (Request.block_hashes), its full blocks' hashes at the
// hash block size. The slice is the request's own.
func (r *Request) BlockHashes() []kv.BlockHash { return r.hashes }

// NumPromptTokens is num_prompt_tokens.
func (r *Request) NumPromptTokens() int { return len(r.Prompt) }

// IsFinished is is_finished().
func (r *Request) IsFinished() bool { return r.Status >= FinishedStopped }

// matchedTokens is get_num_new_matched_tokens (scheduler.py:976-988) for any connector: deferred
// is the connector's None.
func (s *Scheduler) matchedTokens(r *Request, local int) (tokens int, async, deferred bool) {
	if d, ok := s.connector.(Deferrer); ok {
		return d.MatchedTokensOrDefer(r, local)
	}
	tokens, async = s.connector.MatchedTokens(r, local)
	return tokens, async, false
}

// hasPushWork is the connector side of has_requests (scheduler.py:2833).
func (s *Scheduler) hasPushWork() bool {
	p, ok := s.connector.(PushWork)
	return ok && p.HasPendingPushWork()
}

// noteTable records a running request's block table lengths before this step allocates for it,
// so the step's new block ids are the blocks appended after them (allocate_slots' new blocks,
// kv_cache_manager.py:594-617).
func (s *Scheduler) noteTable(r *Request) {
	if _, ok := s.connector.(MetaBuilder); !ok {
		return
	}
	if s.tableLens == nil {
		s.tableLens = map[string][]int{}
	}
	n := make([]int, s.KV.NumGroups())
	for g := range n {
		n[g] = len(s.KV.GroupBlocks(g, r.ID))
	}
	s.tableLens[r.ID] = n
}

// noteFinished records a finished request for the next step's finished_req_ids
// (scheduler.py:2736).
func (s *Scheduler) noteFinished(r *Request) {
	if _, ok := s.connector.(MetaBuilder); ok {
		s.finishedIDs = append(s.finishedIDs, r.ID)
	}
}

// buildConnectorMeta is the connector part of schedule() (scheduler.py:1413-1530): the step's
// SchedulerOutput as yield_req_data reads it, handed to build_connector_meta.
func (s *Scheduler) buildConnectorMeta(out *Output, running, admitted []*Request) {
	mb, ok := s.connector.(MetaBuilder)
	if !ok {
		return
	}
	step := StepOutput{Scheduled: out.Scheduled, Order: out.Order, Preempted: out.Preempted}
	step.Finished = append([]string(nil), s.finishedIDs...)
	sort.Strings(step.Finished)
	s.finishedIDs = nil
	full := func(r *Request) [][]int {
		ids := make([][]int, s.KV.NumGroups())
		for g := range ids {
			for _, b := range s.KV.GroupBlocks(g, r.ID) {
				ids[g] = append(ids[g], b.ID)
			}
			if ids[g] == nil {
				ids[g] = []int{}
			}
		}
		return ids
	}
	var resumed []*Request
	// scheduled_new_reqs: requests admitted from WAITING (:1352-1357); with the V2 model
	// runner, resumed ones join them (:1418-1420).
	for _, r := range admitted {
		if r.NumPreemptions > 0 && !s.cfg.UseV2ModelRunner {
			resumed = append(resumed, r)
			continue
		}
		step.Reqs = append(step.Reqs, StepRequest{ID: r.ID, NewBlocks: full(r)})
	}
	// scheduled_cached_reqs: running requests with the blocks this step appended, then resumed
	// ones with their whole table (_make_cached_request_data, :1693-1750).
	for _, r := range running {
		lens := s.tableLens[r.ID]
		ids := make([][]int, s.KV.NumGroups())
		any := false
		for g := range ids {
			bs := s.KV.GroupBlocks(g, r.ID)
			from := 0
			if g < len(lens) {
				from = lens[g]
			}
			ids[g] = []int{}
			for _, b := range bs[min(from, len(bs)):] {
				ids[g] = append(ids[g], b.ID)
				any = true
			}
		}
		if !any {
			ids = nil
		}
		step.Reqs = append(step.Reqs, StepRequest{ID: r.ID, NewBlocks: ids})
	}
	for _, r := range resumed {
		step.Reqs = append(step.Reqs, StepRequest{ID: r.ID, NewBlocks: full(r), Resumed: true})
	}
	s.tableLens = nil
	out.KVConnectorMetadata = mb.BuildMeta(step)
}
