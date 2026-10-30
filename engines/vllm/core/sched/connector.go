// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/sched/scheduler.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/distributed/kv_transfer/kv_connector/v1/base.py (the scheduler-side hooks a
// connector implements) and vllm/v1/outputs.py (KVConnectorOutput).

package sched

import "sort"

// Connector is the scheduler side of a KV connector (KVConnectorBase_V1's scheduler-role
// methods, vllm/distributed/kv_transfer/kv_connector/v1/base.py@5df63a72), called at exactly
// the points vLLM's Scheduler calls them. The worker side, which moves the bytes, is not
// modelled: its outcomes arrive in Update's TransferOutcome, as a KVConnectorOutput.
type Connector interface {
	// OnNewRequest is on_new_request, from add_request (scheduler.py:2638-2639).
	OnNewRequest(r *Request)
	// MatchedTokens is get_num_new_matched_tokens (scheduler.py:976-980): the tokens the
	// connector can supply beyond the block-aligned local hit, and whether they load
	// asynchronously, between steps.
	MatchedTokens(r *Request, localComputed int) (tokens int, async bool)
	// AfterAlloc is update_state_after_alloc (scheduler.py:1288-1293): the request's blocks
	// after admission, and how many external tokens they hold.
	AfterAlloc(r *Request, blocks AllocatedBlocks, external int)
	// Finished is request_finished (scheduler.py:3043-3058, through _connector_finished):
	// whether the request's blocks must outlive it until a transfer releases them.
	Finished(r *Request, computedBlockIDs []int) (delayFree bool)
	// AfterOutput is update_connector_output (scheduler.py:3219-3220).
	AfterOutput(o TransferOutcome)
}

// SupportsHMA is a connector that supports the hybrid memory allocator (SupportsHMA,
// vllm/distributed/kv_transfer/kv_connector/v1/base.py:100-125@5df63a72): the scheduler hands it
// the finished request's block ids of every KV cache group (request_finished_all_groups,
// scheduler.py:3045-3058), so it may run beside several KV cache groups. A connector without
// it is called with group 0's ids through Finished, and only with one group (:3046-3053).
type SupportsHMA interface {
	// FinishedAllGroups is request_finished_all_groups: blockIDs are the request's block ids
	// covering its computed tokens, one list per KV cache group in group order
	// (get_block_ids_for_computed_tokens, vllm/v1/core/kv_cache_manager.py:731-749@5df63a72), a
	// sliding window's freed slots as the null block's id.
	FinishedAllGroups(r *Request, blockIDs [][]int) (delayFree bool)
}

// TransferParams are a request's kv_transfer_params (Request.kv_transfer_params) as the
// scheduler and the NIXL connector read and update them: what a P/D proxy hands each side
// (vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py@5df63a72).
type TransferParams struct {
	// DoRemotePrefill is do_remote_prefill: this engine decodes, and pulls the prompt's KV
	// from a remote prefiller. The connector clears it once the pull is planned.
	DoRemotePrefill bool
	// DoRemoteDecode is do_remote_decode: this engine prefills, and a remote decoder reads
	// the KV after the request finishes.
	DoRemoteDecode bool
	// RemoteBlocks is len(remote_block_ids).
	RemoteBlocks int
	// HasRemote is whether remote_engine_id, remote_request_id, remote_host and remote_port
	// are all present.
	HasRemote bool
	// RemoteNumTokens is remote_num_tokens (a bidirectional pull back to the prefiller).
	RemoteNumTokens int
	// RemoteBlocksProcessed is _remote_blocks_processed.
	RemoteBlocksProcessed bool
	// PSideTruncated is _p_side_truncated: the prefiller has dropped the prompt's trailing
	// prefill-backoff tokens (nixl/base_scheduler.py:405-438@5df63a72).
	PSideTruncated bool
}

// Capabilities are the configurations beside which a connector's scheduler side is modelled
// and diff-tested against vLLM; SetConnector refuses the rest (https://github.com/inference-sim/blis-vllm/issues/34).
type Capabilities struct {
	// HybridGroups: several KV cache groups (full attention and sliding windows); the
	// connector must also be SupportsHMA.
	HybridGroups bool
	// Drafters: speculative decoding.
	Drafters bool
	// ConcurrentBatches: async scheduling or max_concurrent_batches > 1.
	ConcurrentBatches bool
}

// ConnectorCapabilities is a connector that states its Capabilities; one that does not is
// modelled for one full-attention group, synchronous scheduling and no drafter.
type ConnectorCapabilities interface {
	ConnectorModels() Capabilities
}

// AllocatedBlocks is what update_state_after_alloc sees of a request's blocks
// (kv_cache_manager.get_blocks, scheduler.py:1288-1293@5df63a72).
type AllocatedBlocks struct {
	// IDs and Hashed are group 0's blocks: each block's id, and whether it is prefix-cached
	// (has a hash) and not the null block.
	IDs    []int
	Hashed []bool
	// Groups are the block ids of every KV cache group, in group order: KVCacheBlocks.
	// get_block_ids() (vllm/v1/core/kv_cache_manager.py:81-107@5df63a72). A sliding window's
	// freed slot is the null block's id, 0.
	Groups [][]int
}

// TransferOutcome is the worker side's report for one step (KVConnectorOutput,
// vllm/v1/outputs.py:215-233@5df63a72).
type TransferOutcome struct {
	// FinishedRecving are requests whose asynchronous KV load completed.
	FinishedRecving []string
	// FinishedSending are finished requests whose KV the remote side has read.
	FinishedSending []string
	// FailedRecving are requests whose load failed as a whole.
	FailedRecving []string
	// InvalidBlocks are block IDs whose externally computed KV failed to load.
	InvalidBlocks []int
}

func (o TransferOutcome) empty() bool {
	return len(o.FinishedRecving) == 0 && len(o.FinishedSending) == 0 && len(o.FailedRecving) == 0 &&
		len(o.InvalidBlocks) == 0
}

// FetchStage is a request's KV-fetch stage (KV_FETCH_STAGES, vllm/v1/metrics/stats.py:212-219).
type FetchStage int

const (
	fetchNone FetchStage = iota
	FetchWaitingToStart
	FetchInProgress
	FetchCompletedWaiting
)

// ConnectorConfig is the scheduler's slice of KVTransferConfig.
type ConnectorConfig struct {
	// RecomputeLoadFailures is kv_load_failure_policy == "recompute"
	// (vllm/config/kv_transfer.py:88@5df63a72; scheduler.py:181-182).
	RecomputeLoadFailures bool
}

// setFetchStage is _set_kv_fetch_stage (scheduler.py:3078-3086).
func (s *Scheduler) setFetchStage(r *Request, st FetchStage) {
	if prev := r.fetchStage; prev != fetchNone {
		s.fetchCounts[prev]--
	}
	r.fetchStage = st
	if st != fetchNone {
		s.fetchCounts[st]++
	}
}

// FetchCounts are the requests at each KV-fetch stage (num_kv_fetch_reqs_by_stage):
// waiting to start, in progress, completed and waiting.
func (s *Scheduler) FetchCounts() [3]int {
	return [3]int{s.fetchCounts[FetchWaitingToStart], s.fetchCounts[FetchInProgress], s.fetchCounts[FetchCompletedWaiting]}
}

// WaitingForRemoteKV lists the requests waiting for an asynchronous KV load, in queue order.
func (s *Scheduler) WaitingForRemoteKV() []string {
	var out []string
	for _, r := range s.holding.items {
		if r.Status == WaitingForRemoteKVs {
			out = append(out, r.ID)
		}
	}
	return out
}

// RecvLanded lists the requests whose load has landed and that are not yet promoted
// (finished_recving_kv_req_ids).
func (s *Scheduler) RecvLanded() []string { return sortedSet(s.recvDone) }

// Holding lists the waiting requests that hold KV blocks (kv_holding_waiting), in order.
func (s *Scheduler) Holding() []string {
	out := make([]string, 0, len(s.holding.items))
	for _, r := range s.holding.items {
		out = append(out, r.ID)
	}
	return out
}

// DelayFreed lists the finished requests whose blocks the connector still holds.
func (s *Scheduler) DelayFreed() []string {
	var out []string
	for id, r := range s.requests {
		if r.Status >= FinishedStopped {
			out = append(out, id)
		}
	}
	sort.Strings(out)
	return out
}

// NumRequests is len(self.requests): live requests plus finished ones a connector holds.
func (s *Scheduler) NumRequests() int { return len(s.requests) }

// holdsKVBlocks is _holds_kv_blocks (scheduler.py:2444-2455).
func holdsKVBlocks(r *Request) bool {
	return r.NumComputed > 0 || r.Status == WaitingForRemoteKVs
}

// updateWaitingForRemoteKV is _update_waiting_for_remote_kv (scheduler.py:3123-3173) for one
// full-attention group with no replay window.
func (s *Scheduler) updateWaitingForRemoteKV(r *Request) {
	if s.recvFailed[r.ID] {
		if r.NumComputed > 0 {
			s.KV.CacheBlocks(r.ID, r.hashes, r.NumComputed, s.run(r))
		} else {
			s.KV.Free(r.ID)
		}
		delete(s.recvFailed, r.ID)
	} else {
		s.KV.CacheBlocks(r.ID, r.hashes, r.NumComputed, s.run(r))
	}
	if r.NumComputed == r.NumTokens() {
		r.NumComputed = r.NumTokens() - 1
	}
	delete(s.recvDone, r.ID)
}

// handleBlockedWaiting is _handle_blocked_waiting_request (scheduler.py:3175-3207) for
// WAITING_FOR_REMOTE_KVS: whether the request can be scheduled this step.
func (s *Scheduler) handleBlockedWaiting(r *Request) bool {
	if r.Status != WaitingForRemoteKVs {
		return true
	}
	if !s.recvDone[r.ID] {
		return false
	}
	s.updateWaitingForRemoteKV(r)
	if r.NumPreemptions > 0 {
		r.Status = Preempted
	} else {
		r.Status = Waiting
	}
	return true
}

// freeRequest is _free_request (scheduler.py:2714-2743) for a request that finished: the
// connector decides whether its blocks outlive it.
func (s *Scheduler) freeRequest(r *Request, delayFree bool) {
	delete(s.inflight, r)
	s.setFetchStage(r, fetchNone)
	s.finishedPending++
	if s.connector != nil && s.connectorFinished(r) {
		delayFree = true
	}
	if !delayFree {
		s.freeBlocks(r)
	}
}

// connectorFinished is _connector_finished (scheduler.py:3005-3058): each sliding window first
// frees the blocks wholly below its window, on the processed-token basis (:3024-3031), then the
// connector sees the block ids covering the computed tokens, per group
// (get_block_ids_for_computed_tokens, vllm/v1/core/kv_cache_manager.py:731-749@5df63a72). A
// SupportsHMA connector gets every group's (request_finished_all_groups); any other runs only
// beside one group (SetConnector) and gets group 0's (:3045-3053).
func (s *Scheduler) connectorFinished(r *Request) bool {
	s.KV.RemoveSkipped(r.ID, max(0, r.NumComputed-r.numInFlight))
	ids := make([][]int, s.KV.NumGroups())
	for g := range ids {
		blocks := s.KV.GroupBlocks(g, r.ID)
		n := min(cdiv(r.NumComputed, s.KV.GroupBlockSize(g)), len(blocks))
		ids[g] = make([]int, n)
		for i := 0; i < n; i++ {
			ids[g][i] = blocks[i].ID
		}
	}
	if h, ok := s.connector.(SupportsHMA); ok {
		return h.FinishedAllGroups(r, ids)
	}
	return s.connector.Finished(r, ids[0])
}

// freeBlocks is _free_blocks (scheduler.py:2745-2748).
func (s *Scheduler) freeBlocks(r *Request) {
	s.KV.Free(r.ID)
	delete(s.requests, r.ID)
}

// failedLoads is update_from_output's handling of failed KV loads, before the requests are
// visited (scheduler.py:2058-2071): it returns the requests the main loop must skip, and,
// under kv_load_failure_policy "fail", those finished as errors after it (:2306-2324).
func (s *Scheduler) failedLoads(o TransferOutcome, scheduled map[string]int) (skip map[string]bool, fail []string) {
	skip = map[string]bool{}
	if len(o.FailedRecving) > 0 {
		for _, id := range s.handleFailedRecving(o.FailedRecving) {
			skip[id] = true
		}
	}
	if len(o.InvalidBlocks) > 0 {
		for _, id := range s.handleInvalidBlocks(o.InvalidBlocks, scheduled) {
			skip[id] = true
		}
	}
	if !s.conf.RecomputeLoadFailures {
		for id := range skip {
			fail = append(fail, id)
		}
		sort.Strings(fail)
	}
	return skip, fail
}

// handleFailedRecving is _handle_failed_recving (scheduler.py:3341-3354).
func (s *Scheduler) handleFailedRecving(ids []string) []string {
	var affected []string
	for _, id := range sortedUnique(ids) {
		r := s.requests[id]
		if r != nil && r.Status == WaitingForRemoteKVs {
			affected = append(affected, id)
			if s.conf.RecomputeLoadFailures {
				r.NumComputed = 0
				s.recvFailed[id] = true
			}
		}
	}
	if s.conf.RecomputeLoadFailures {
		return nil
	}
	return affected
}

// handleInvalidBlocks is _handle_invalid_blocks (scheduler.py:3356-3435) for one KV cache
// group: async loads first (not yet cached), then the running requests (sync loads).
func (s *Scheduler) handleInvalidBlocks(invalid []int, scheduled map[string]int) []string {
	bad := map[int]bool{}
	for _, b := range invalid {
		bad[b] = true
	}
	var async []*Request
	for _, r := range s.holding.items {
		if r.Status == WaitingForRemoteKVs {
			async = append(async, r)
		}
	}
	asyncFailed, n1 := s.markInvalid(async, bad, scheduled)
	syncFailed, n2 := s.markInvalid(s.running, bad, scheduled)
	if n1+n2 == 0 {
		return nil
	}
	if !s.conf.RecomputeLoadFailures {
		return append(asyncFailed, syncFailed...)
	}
	for _, id := range asyncFailed {
		s.recvFailed[id] = true
	}
	return syncFailed
}

// markInvalid is _update_requests_with_invalid_blocks (scheduler.py:3237-3339): each
// request's computed tokens are cut at its first invalid block, which only the first request
// holding it recomputes.
func (s *Scheduler) markInvalid(rs []*Request, bad map[int]bool, scheduled map[string]int) ([]string, int) {
	var affected []string
	marked := map[int]bool{}
	for _, r := range rs {
		isAffected, markedOne := false, false
		computed := r.NumComputed - scheduled[r.ID]
		blocks := s.KV.Blocks(r.ID)
		for idx := 0; idx < cdiv(computed, s.cfg.BlockSize) && idx < len(blocks); idx++ {
			id := blocks[idx].ID
			if !bad[id] {
				continue
			}
			isAffected = true
			if marked[id] {
				continue
			}
			marked[id] = true
			if markedOne {
				continue
			}
			markedOne = true
			r.NumComputed = idx * s.cfg.BlockSize
		}
		if isAffected {
			if !markedOne {
				r.NumComputed = computed
			}
			affected = append(affected, r.ID)
		}
	}
	return affected, len(affected)
}

// transfersFinished is _update_from_kv_xfer_finished (scheduler.py:3209-3235).
func (s *Scheduler) transfersFinished(o TransferOutcome) {
	if s.connector != nil {
		s.connector.AfterOutput(o)
	}
	for _, id := range o.FinishedRecving {
		r := s.requests[id]
		if r == nil {
			continue
		}
		if r.Status == WaitingForRemoteKVs {
			s.recvDone[id] = true
			s.setFetchStage(r, FetchCompletedWaiting)
		} else {
			s.freeBlocks(r)
		}
	}
	for _, id := range o.FinishedSending {
		if r := s.requests[id]; r != nil {
			s.freeBlocks(r)
		}
	}
}

// finishRequests is finish_requests (scheduler.py:2649-2712) for the error path of a load
// failure under kv_load_failure_policy "fail". The ids are a set (:2667-2668): a repeated id
// finishes its request once. Counted twice, the second copy would see the first's finished
// status and free blocks an in-flight load still writes.
func (s *Scheduler) finishRequests(ids []string, status Status) []*Request {
	var done []*Request
	seen := make(map[string]bool, len(ids))
	for _, id := range ids {
		if seen[id] {
			continue
		}
		seen[id] = true
		r := s.requests[id]
		if r == nil || r.Status >= FinishedStopped {
			continue
		}
		done = append(done, r)
		if r.Status == Running {
			s.removeRunning(r)
		} else {
			s.waiting.remove(r)
			s.holding.remove(r)
			delete(s.deferred, r.ID)
		}
	}
	for _, r := range done {
		delayFree := false
		if r.Status == WaitingForRemoteKVs {
			delayFree = !s.recvDone[r.ID]
			delete(s.recvDone, r.ID)
			delete(s.recvFailed, r.ID)
		}
		r.Status = status
		s.freeRequest(r, delayFree)
	}
	return done
}

// Abort is finish_requests(ids, FINISHED_ABORTED) (scheduler.py:2649-2712), as EngineCore
// applies a client's aborts (vllm/v1/engine/core.py:554-559@5df63a72).
func (s *Scheduler) Abort(ids []string) {
	s.finishRequests(ids, FinishedAborted)
}

func (s *Scheduler) removeRunning(r *Request) {
	if i := indexOf(s.running, r); i >= 0 {
		s.running = append(s.running[:i], s.running[i+1:]...)
	}
}

func sortedSet(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func sortedUnique(in []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	sort.Strings(out)
	return out
}
