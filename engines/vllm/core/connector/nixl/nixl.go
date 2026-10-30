// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py
// (NixlBaseConnectorScheduler) and nixl/connector.py (the scheduler-role dispatch).

// Package nixl is the scheduler side of vLLM's NIXL connector in its pull mode
// (NixlConnector = NixlPullConnector, vllm/distributed/kv_transfer/kv_connector/factory.py:
// 177-189@5df63a72): what the engine decides about a P/D request. On the decoder
// (kv_consumer) a request with do_remote_prefill waits in WAITING_FOR_REMOTE_KVS while its
// prompt's KV is pulled; on the prefiller (kv_producer) a request with do_remote_decode keeps
// its blocks after it finishes until the decoder has read them. The worker side, which moves
// the bytes, is the transport: its outcomes reach the scheduler as a sched.TransferOutcome.
//
// The NIXL worker's notifications, lease timers and heartbeats run on the worker; the
// scheduler's decisions depend only on the outcome they report, which is the input here.
package nixl

import (
	"fmt"

	"github.com/inference-sim/blis-vllm/core/sched"
)

// Config is the connector's slice of KVTransferConfig.kv_connector_extra_config.
type Config struct {
	// RecomputeThreshold is kv_recompute_threshold (base_scheduler.py:160-167, default 64):
	// a bidirectional pull of fewer remote tokens is skipped and recomputed.
	RecomputeThreshold int
	// Bidirectional is bidirectional_kv_xfer (base_scheduler.py:169-175, default false).
	Bidirectional bool
	// PrefillBackoff is _prefill_backoff (base_scheduler.py:379-394): trailing prompt
	// tokens the prefiller does not compute and the decoder recomputes. It is
	// max(1 if the model has mamba layers or a replay window else 0,
	// num_prefill_lookahead_tokens - 1); 0 for a full-attention model without a
	// multi-module MTP drafter. Backoff derives it.
	PrefillBackoff int
}

// Default is the defaults kv_connector_extra_config leaves (base_scheduler.py:160-175).
func Default() Config { return Config{RecomputeThreshold: 64} }

// Backoff is _prefill_backoff (base_scheduler.py:379-394) for a model without Mamba layers or
// a bounded-replay window: num_prefill_lookahead_tokens - 1, so a multi-module MTP drafter's
// lookahead window stays off the prefiller, which would otherwise embed unverified drafts in
// the MTP layers' KV (base_scheduler.py:391-394). Mamba layers and DeepSeek V4.1's bounded
// replay add a floor of one token, which SetConnector refuses (no such group is modelled here).
func Backoff(spec sched.Spec) int { return max(0, spec.PrefillLookahead-1) }

// CheckWorker refuses what vLLM's NIXL worker refuses at start-up for the parallel layout
// (NixlBaseConnectorWorker.__init__, nixl/base_worker.py:757-779@5df63a72): with
// pipeline_parallel_size > 1, the pull worker refuses a hybrid KV cache layout that needs the
// hybrid memory allocator (_is_hma_required, :560-566: some layer is not full attention and
// disable_hybrid_kv_cache_manager is off; _supports_pp_hma is False for the pull worker, :132,
// :761-765) and any decoder (kv_consumer, :775-779). The scheduler side never sees PP, so these
// are the only parallel refusals.
func CheckWorker(pp int, hmaRequired, kvConsumer bool) error {
	if pp <= 1 {
		return nil
	}
	if hmaRequired {
		return fmt.Errorf("nixl: NixlConnector (pull) does not support pipeline_parallel_size %d with a hybrid "+
			"KV cache layout (HMA); vLLM's worker refuses it (vllm/distributed/kv_transfer/kv_connector/v1/nixl/"+
			"base_worker.py:760-765@5df63a72)", pp)
	}
	if kvConsumer {
		return fmt.Errorf("nixl: a NIXL decoder (kv_consumer) does not support pipeline_parallel_size %d; vLLM's "+
			"worker refuses it (vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:775-779@5df63a72)", pp)
	}
	return nil
}

// Scheduler is NixlPullConnectorScheduler for one engine.
type Scheduler struct {
	cfg Config
	// recv are the requests whose pull the next metadata asks the worker to start
	// (_reqs_need_recv); send are the finished requests whose blocks wait to be read
	// (_reqs_need_send); notProcessed are requests aborted on the prefiller
	// (_reqs_not_processed). The scheduler's decisions read none of them; they are kept
	// so a test can see what the worker would be told.
	recv         map[string]bool
	send         map[string]bool
	inBatch      map[string]bool
	notProcessed map[string]bool
}

// New builds the connector's scheduler side.
func New(cfg Config) *Scheduler {
	return &Scheduler{cfg: cfg, recv: map[string]bool{}, send: map[string]bool{}, inBatch: map[string]bool{},
		notProcessed: map[string]bool{}}
}

// remoteTokenCount is _get_remote_prefill_token_count (base_scheduler.py:396-403).
func (s *Scheduler) remoteTokenCount(promptTokens int) int {
	if b := s.cfg.PrefillBackoff; b > 0 && promptTokens > b {
		return promptTokens - b
	}
	return promptTokens
}

// OnNewRequest is on_new_request (base_scheduler.py:199-238): a remote-decode request has its
// prompt truncated by the backoff (_truncate_request_for_prefill, :405-438); heartbeats are
// worker-side bookkeeping.
func (s *Scheduler) OnNewRequest(r *sched.Request) {
	if p := r.Transfer; p != nil && p.DoRemoteDecode {
		s.truncateForPrefill(r)
	}
}

// truncateForPrefill is _truncate_request_for_prefill (base_scheduler.py:405-438): the
// prefiller drops the trailing backoff prompt tokens, which the decoder recomputes, and samples
// one token, once per request. The request's block hashes were computed when it was built
// (Request.__init__, vllm/v1/request.py:225-230@5df63a72) and are not recomputed, as in vLLM.
func (s *Scheduler) truncateForPrefill(r *sched.Request) {
	b := s.cfg.PrefillBackoff
	if b == 0 || r.Transfer.PSideTruncated || len(r.Prompt) <= b {
		return
	}
	r.Prompt = r.Prompt[:len(r.Prompt)-b]
	r.MaxTokens = 1
	r.Transfer.PSideTruncated = true
}

// ConnectorModels implements sched.ConnectorCapabilities: what this model of NIXL's scheduler
// side is diff-tested beside, against vLLM v0.32.0 (oracle/run_connector.py).
func (s *Scheduler) ConnectorModels() sched.Capabilities {
	return sched.Capabilities{HybridGroups: true, Drafters: true, ConcurrentBatches: true}
}

// MatchedTokens is get_num_new_matched_tokens (pull_scheduler.py:34-106).
func (s *Scheduler) MatchedTokens(r *sched.Request, localComputed int) (int, bool) {
	p := r.Transfer
	if p == nil {
		return 0, false
	}
	if p.DoRemotePrefill {
		// Remote prefill: pull every prompt block not computed locally.
		if n := s.remoteTokenCount(len(r.Prompt)) - localComputed; n > 0 {
			return n, true
		}
	}
	if p.DoRemoteDecode && p.RemoteBlocks > 0 && p.HasRemote {
		// A prefiller holding part of the prompt on a decoder (bidirectional): pull it back
		// unless it is below the recompute threshold.
		n := min(p.RemoteNumTokens, len(r.Prompt)) - localComputed
		if n > 0 {
			if s.cfg.RecomputeThreshold > 0 && n < s.cfg.RecomputeThreshold {
				return 0, false
			}
			return n, true
		}
	}
	return 0, false
}

// AfterAlloc is update_state_after_alloc (pull_scheduler.py:108-189).
func (s *Scheduler) AfterAlloc(r *sched.Request, blocks sched.AllocatedBlocks, external int) {
	p := r.Transfer
	if p == nil {
		return
	}
	if p.DoRemoteDecode || (p.DoRemotePrefill && s.cfg.Bidirectional) {
		s.inBatch[r.ID] = true
	}
	if p.DoRemotePrefill || (p.DoRemoteDecode && s.cfg.Bidirectional && !p.RemoteBlocksProcessed) {
		if p.RemoteBlocks > 0 && p.HasRemote {
			s.recv[r.ID] = true
		}
		// Only one transfer per request.
		p.DoRemotePrefill = false
		p.RemoteBlocksProcessed = true
	}
}

// Finished is request_finished (pull_scheduler.py:191-292) as the scheduler calls it for one
// KV cache group (NixlBaseConnector.request_finished, nixl/connector.py:203-210).
func (s *Scheduler) Finished(r *sched.Request, computedBlockIDs []int) bool {
	return s.FinishedAllGroups(r, [][]int{computedBlockIDs})
}

// FinishedAllGroups is request_finished_all_groups (nixl/connector.py:212-218): NIXL is
// SupportsHMA (:78), so with several KV cache groups it gets every group's block ids, and the
// request's blocks outlive it if any group holds one (pull_scheduler.py:248).
func (s *Scheduler) FinishedAllGroups(r *sched.Request, blockIDs [][]int) bool {
	p := r.Transfer
	if p == nil {
		return false
	}
	isPrefiller := p.DoRemoteDecode
	if p.DoRemotePrefill {
		// Finished before its pull was planned (aborted while waiting): tell the worker to
		// release the prefiller's blocks, and free this engine's at once.
		s.recv[r.ID] = true
		p.DoRemotePrefill = false
		return false
	}
	if !isPrefiller && !s.cfg.Bidirectional {
		return false
	}
	if r.Status != sched.FinishedLength && r.Status != sched.FinishedStopped {
		s.notProcessed[r.ID] = true
		return false
	}
	delay := false
	for _, g := range blockIDs {
		delay = delay || len(g) > 0
	}
	if delay {
		s.send[r.ID] = true
	}
	return delay
}

// AfterOutput is update_connector_output (base_scheduler.py:522-525): it stops heartbeats,
// which no scheduler decision reads.
func (s *Scheduler) AfterOutput(o sched.TransferOutcome) {
	for _, id := range o.FinishedSending {
		delete(s.send, id)
	}
}

// BuildMeta is build_connector_meta (base_scheduler.py:474-520) as far as a scheduler
// decision depends on it: it clears the per-step sets the worker is handed.
func (s *Scheduler) BuildMeta() {
	s.recv = map[string]bool{}
	s.inBatch = map[string]bool{}
	s.notProcessed = map[string]bool{}
}
