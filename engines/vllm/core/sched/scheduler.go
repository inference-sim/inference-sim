// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/sched/scheduler.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/core/sched/request_queue.py, vllm/v1/core/sched/utils.py (check_stop) and
// vllm/v1/core/sched/async_scheduler.py (AsyncScheduler). SWA bounded replay: scheduler.py:290-305,
// 1047-1066, 1103-1124, 1253-1256 and 3103-3121.

// Package sched is vLLM's v1 Scheduler: schedule() and update_from_output() for a
// full-attention model, with chunked prefill, the long-prefill cap, FCFS and priority
// policies, recompute preemption, prefix caching, speculative decoding (acceptance supplied)
// and the max_loras batch limit, and the AsyncScheduler's output placeholders and
// pipeline-parallel decode spacing (Config.Async). Connectors are separate; the batch queue
// that keeps several batches in flight is core/engine's.
package sched

import (
	"fmt"
	"sort"

	"github.com/inference-sim/blis-vllm/core/kv"
)

// Policy is SchedulerConfig.policy (vllm/config/scheduler.py:21,159).
type Policy string

const (
	FCFS     Policy = "fcfs"
	Priority Policy = "priority"
)

// Config is the scheduler's slice of vLLM's configuration.
type Config struct {
	MaxNumSeqs          int // max_num_seqs
	MaxNumActiveSeqs    int // max_num_active_seqs; 0 = MaxNumSeqs
	MaxNumBatchedTokens int // max_num_batched_tokens
	// MaxNumScheduledTokens is max_num_scheduled_tokens; 0 = MaxNumBatchedTokens.
	MaxNumScheduledTokens     int
	LongPrefillTokenThreshold int  // long_prefill_token_threshold
	LongPrefillAdaptive       bool // long_prefill_token_threshold_adaptive
	EnableChunkedPrefill      bool
	Policy                    Policy
	MaxModelLen               int
	BlockSize                 int
	NumBlocks                 int // the fixed KV capacity
	EnablePrefixCaching       bool
	Watermark                 float64
	// ReserveFullSequence is scheduler_reserve_full_isl (vllm/config/scheduler.py:190,
	// default true): admit a new request only if its whole sequence fits.
	ReserveFullSequence bool
	// Spec is speculative decoding; its zero value is off. ResolveSpec derives it.
	Spec Spec
	// MaxLoRAs is LoRAConfig.max_loras (vllm/config/lora.py:35@5df63a72); 0 = LoRA off.
	MaxLoRAs int
	// KVEvents is kv_events_config.enable_kv_cache_events (vllm/config/kv_events.py:14@5df63a72):
	// the KV cache records BlockStored and BlockRemoved events, taken each step by
	// TakeKVEvents.
	KVEvents bool
	// KVGroups are a hybrid model's KV cache groups (full attention, sliding windows); none
	// means one full-attention group.
	KVGroups []kv.Group
	// MaxConcurrentBatches is max_concurrent_batches (vllm/config/vllm.py:577-588@5df63a72);
	// 0 means 1, this scheduler's synchronous case.
	MaxConcurrentBatches int
	// Async selects vLLM's AsyncScheduler (vllm/v1/core/sched/async_scheduler.py:14), which
	// scheduler_config.async_scheduling picks (vllm/config/scheduler.py:228-236): a request
	// scheduled to the end of its known tokens counts the token it will sample as an output
	// placeholder, so the next step can schedule it again before that token returns.
	Async bool
	// DecodeStagger is the AsyncScheduler's decode_stagger, pipeline_parallel_size
	// (async_scheduler.py:19-23); with UseV2ModelRunner a request's next decode waits that
	// many steps (next_decode_eligible_step, :52-58; scheduler.py:680-684).
	DecodeStagger    int
	UseV2ModelRunner bool
}

// Status is RequestStatus (vllm/v1/request.py:370).
type Status int

const (
	Waiting Status = iota
	Running
	Preempted
	// WaitingForRemoteKVs is a request whose prompt KV a connector is loading
	// asynchronously (vllm/v1/request.py:375@5df63a72).
	WaitingForRemoteKVs
	FinishedStopped
	FinishedLength
	FinishedAborted
	// FinishedError is a request the engine failed, such as a KV load under
	// kv_load_failure_policy "fail".
	FinishedError
)

// Request is vLLM's Request (vllm/v1/request.py:61) for a token-only prompt.
type Request struct {
	ID          string
	Priority    int
	ArrivalTime float64
	Prompt      []int64
	// MaxTokens is sampling_params.max_tokens.
	MaxTokens int
	// EOS is the token that stops the request (sampling_params.eos_token_id).
	EOS  int64
	Keys kv.BlockKeys
	// LoRA is the adapter's lora_int_id, 0 for the base model.
	LoRA int
	// LoRAName is the adapter's lora_name, reported in KV events.
	LoRAName string

	Status         Status
	Output         []int64
	NumComputed    int
	NumPreemptions int
	IsPrefillChunk bool
	// SpecTokens are spec_token_ids: the drafts proposed for the next step.
	SpecTokens []int64
	// Transfer is the request's KV-transfer parameters, which a connector reads and updates
	// (Request.kv_transfer_params); nil for a request no connector moves.
	Transfer    *TransferParams
	fetchStage  FetchStage
	hashes      []kv.BlockHash
	numInFlight int
	seq         int // FIFO tie-break within a priority queue
	// Placeholders is num_output_placeholders: tokens scheduled to be sampled whose output
	// has not returned yet (async scheduling only).
	Placeholders int
	// numStale is num_stale_output_tokens: the in-flight tokens a preemption made stale,
	// whose outputs still return but must not move the reset counters (scheduler.py:1596-1607).
	numStale int
	// nextDecodeStep is next_decode_eligible_step.
	nextDecodeStep int
	// ReplayStart is replay_start (vllm/v1/request.py:209@5df63a72): where a bounded-replay
	// prefix hit's recomputed window begins, 0 when nothing is replayed (markPrefixReplay).
	ReplayStart int
}

// NumTokens is len(all_token_ids): prompt plus output.
func (r *Request) NumTokens() int { return len(r.Prompt) + len(r.Output) }

// numTokensWithSpec is num_tokens_with_spec (vllm/v1/request.py:298-300).
func (r *Request) numTokensWithSpec() int { return r.NumTokens() + len(r.SpecTokens) }

func (r *Request) allTokens() []int64 {
	return append(append(make([]int64, 0, r.NumTokens()), r.Prompt...), r.Output...)
}

// updateHashes appends the hashes of newly full blocks (update_block_hashes), with each
// block's extra keys from kv.BlockKeys.ExtraFor, as kv.BlockHashes keys them. It reads only
// the new blocks' tokens, so a decode step costs one block, not the whole context.
func (r *Request) updateHashes(blockSize int) {
	for (len(r.hashes)+1)*blockSize <= r.NumTokens() {
		i := len(r.hashes)
		var parent kv.BlockHash
		if i > 0 {
			parent = r.hashes[i-1]
		}
		r.hashes = append(r.hashes, kv.HashBlock(parent, r.tokenRange(i*blockSize, (i+1)*blockSize), r.Keys.ExtraFor(i)))
	}
}

// tokenRange is all_token_ids[from:to] without copying the whole sequence: the prompt's
// slice when the range lies in it, else a copy of just the range.
func (r *Request) tokenRange(from, to int) []int64 {
	if to <= len(r.Prompt) {
		return r.Prompt[from:to]
	}
	if from >= len(r.Prompt) {
		return r.Output[from-len(r.Prompt) : to-len(r.Prompt)]
	}
	return append(append(make([]int64, 0, to-from), r.Prompt[from:]...), r.Output[:to-len(r.Prompt)]...)
}

// Output is SchedulerOutput's decision fields.
type Output struct {
	// Scheduled is num_scheduled_tokens.
	Scheduled map[string]int
	// Order lists the scheduled requests: running ones in running order, then new ones.
	Order []string
	// Preempted is preempted_req_ids, in preemption order.
	Preempted []string
	// NewIDs are requests admitted from the waiting queue this step.
	NewIDs []string
	// PrefixHit is the locally cached tokens of each newly admitted request.
	PrefixHit map[string]int
	// SpecDecode is scheduled_spec_decode_tokens: the drafts each request verifies this step
	// (-1 for first-decode padding).
	SpecDecode map[string][]int64
}

// Finished is one request update_from_output finished.
type Finished struct {
	ID     string
	Status Status
	Output int
}

// Scheduler is vLLM's Scheduler.
type Scheduler struct {
	cfg      Config
	KV       *kv.Manager
	requests map[string]*Request
	running  []*Request
	waiting  queue
	// holding is kv_holding_waiting: waiting requests that hold KV blocks, always drained
	// before waiting (scheduler.py:223-227, 913-917).
	holding queue
	seq     int
	// The KV connector and its state (scheduler.py:163-192, 247-248, 430-441); connector is
	// nil without one.
	connector   Connector
	conf        ConnectorConfig
	recvDone    map[string]bool // finished_recving_kv_req_ids
	recvFailed  map[string]bool // failed_recving_kv_req_ids
	fetchCounts [4]int          // _kv_fetch_counts, indexed by FetchStage
	// inflight is _inflight_prefills: requests still prefilling, chunks and asynchronous
	// loads, whose remaining blocks an asynchronous load must leave free.
	inflight map[*Request]bool
	// currentStep is schedule()'s current_step (scheduler.py:366, :592).
	currentStep int
	// finishedPending is finished_req_ids: requests finished since the last schedule(),
	// which keep has_requests() true until schedule() hands them to the runner
	// (interface.py:199-202; scheduler.py:1649-1652, 2813-2815).
	finishedPending int
	// deferred is deferred_waiting: requests skipped in a waiting pass (scheduler.py:230,
	// 906, 1344).
	deferred map[string]bool
	// capacityBound is prefill_capacity_bound (scheduler.py:370, :1397-1401).
	capacityBound bool
}

// New builds a scheduler.
func New(cfg Config) (*Scheduler, error) {
	if cfg.MaxNumSeqs < 1 || cfg.MaxNumBatchedTokens < 1 || cfg.BlockSize < 1 || cfg.NumBlocks < 2 || cfg.MaxModelLen < 1 {
		return nil, fmt.Errorf("sched: invalid config %+v", cfg)
	}
	if cfg.Policy == "" {
		cfg.Policy = FCFS
	}
	if cfg.Policy != FCFS && cfg.Policy != Priority {
		return nil, fmt.Errorf("sched: unknown policy %q; vLLM accepts fcfs and priority", cfg.Policy)
	}
	// vllm/config/vllm.py:2285-2290@5df63a72 refuses a budget no larger than the draft slots.
	if cfg.MaxNumBatchedTokens <= cfg.Spec.DraftSlots {
		return nil, fmt.Errorf("sched: max_num_batched_tokens %d leaves no room beside %d draft slots",
			cfg.MaxNumBatchedTokens, cfg.Spec.DraftSlots)
	}
	wm := int(cfg.Watermark * float64(cfg.NumBlocks))
	s := &Scheduler{
		cfg: cfg,
		KV: kv.NewManagerWith(cfg.NumBlocks, cfg.BlockSize, cfg.MaxModelLen, cfg.EnablePrefixCaching, wm,
			kv.Options{EagleBlockDrop: cfg.Spec.EagleBlockDrop, PrefillLookahead: cfg.Spec.PrefillLookahead,
				Groups: cfg.KVGroups, MaxInFlightTokens: max(cfg.MaxConcurrentBatches, 1) * cfg.MaxNumBatchedTokens}),
		requests: map[string]*Request{},
		waiting:  queue{priority: cfg.Policy == Priority},
		holding:  queue{priority: cfg.Policy == Priority},
		deferred: map[string]bool{},
		recvDone: map[string]bool{}, recvFailed: map[string]bool{}, inflight: map[*Request]bool{},
	}
	if cfg.KVEvents {
		s.KV.Pool.EnableEvents()
	}
	return s, nil
}

// TakeKVEvents returns the KV cache events of the last step, in the order vLLM queues them
// (kv_cache_manager.take_events, scheduler.py:2371@5df63a72).
func (s *Scheduler) TakeKVEvents() []kv.Event { return s.KV.Pool.TakeEvents() }

// Add is add_request.
func (s *Scheduler) Add(r *Request) error {
	if _, dup := s.requests[r.ID]; dup {
		return fmt.Errorf("sched: request %s added twice", r.ID)
	}
	r.Status = Waiting
	s.seq++
	r.seq = s.seq
	r.updateHashes(s.KV.HashBlockSize())
	// A new request holds no blocks, so _enqueue_waiting_request queues it in waiting
	// (scheduler.py:2457-2462, 2633).
	s.waiting.add(r)
	s.requests[r.ID] = r
	if s.connector != nil {
		s.connector.OnNewRequest(r)
		// A P/D request counts from arrival: connector matching is skipped while all run
		// slots are taken (scheduler.py:2640-2645).
		if r.Transfer != nil && r.Transfer.DoRemotePrefill {
			s.setFetchStage(r, FetchWaitingToStart)
		}
	}
	return nil
}

// SetConnector installs a KV connector (KVConnectorFactory.create_connector,
// scheduler.py:170-182). Call before any request is added. A connector is modelled for one
// full-attention KV cache group, synchronous scheduling and no drafter; the rest is refused
// until connectorRefusalIssue lands (docs/feature-matrix.md, "KV connectors and offload").
func (s *Scheduler) SetConnector(c Connector, conf ConnectorConfig) error {
	switch {
	case s.KV.NumGroups() > 1:
		return fmt.Errorf("sched: a KV connector with %d KV cache groups is not modelled (%s)", s.KV.NumGroups(),
			connectorRefusalIssue)
	case s.cfg.Async || s.cfg.MaxConcurrentBatches > 1:
		// defer_block_free and requires_kv_delivery (scheduler.py:184-194) apply there.
		return fmt.Errorf("sched: a KV connector with async scheduling or concurrent batches is not modelled (%s)",
			connectorRefusalIssue)
	case s.cfg.Spec.NumTokens > 0:
		return fmt.Errorf("sched: a KV connector with speculative decoding is not modelled (%s)", connectorRefusalIssue)
	}
	s.connector, s.conf = c, conf
	return nil
}

// connectorRefusalIssue tracks the connector combinations SetConnector refuses: several KV
// cache groups, the batch queue and drafters.
const connectorRefusalIssue = "https://github.com/inference-sim/blis-vllm/issues/34"

// Counts is get_request_counts: (running, waiting) (scheduler.py:2608-2610).
func (s *Scheduler) Counts() (int, int) { return len(s.running), s.waiting.len() + s.holding.len() }

// HasUnfinished is has_unfinished_requests.
func (s *Scheduler) HasUnfinished() bool { return len(s.running)+s.waiting.len()+s.holding.len() > 0 }

// HasRequests is has_requests (scheduler.py:2823-2838): unfinished requests, finished ones
// the next schedule() still has to report, or, with a connector, finished ones whose blocks
// it still holds (has_finished_requests, :2813-2821).
func (s *Scheduler) HasRequests() bool {
	if s.HasUnfinished() || s.finishedPending > 0 {
		return true
	}
	return s.connector != nil && len(s.requests) > len(s.running)+s.waiting.len()+s.holding.len()
}

// Deferred is how many waiting requests the last waiting pass skipped (deferred_waiting).
func (s *Scheduler) Deferred() int { return len(s.deferred) }

// Running returns the running requests' IDs in running order.
func (s *Scheduler) Running() []string {
	out := make([]string, len(s.running))
	for i, r := range s.running {
		out[i] = r.ID
	}
	return out
}

// Request returns a request by ID.
func (s *Scheduler) Request(id string) *Request { return s.requests[id] }

// MaxModelLen is the scheduler's max_model_len.
func (s *Scheduler) MaxModelLen() int { return s.cfg.MaxModelLen }

// BlockSize is the scheduler block size, in tokens: the KV cache's block size, or with groups of
// several block sizes their LCM (resolve_kv_cache_block_sizes, vllm/v1/core/kv_cache_utils.py:
// 804-861@5df63a72), which vLLM's EngineCore hands the Scheduler as block_size.
func (s *Scheduler) BlockSize() int { return s.KV.BlockSize() }

// PrefixCaching reports whether prefix-cache lookups run (enable_prefix_caching).
func (s *Scheduler) PrefixCaching() bool { return s.cfg.EnablePrefixCaching }

func (s *Scheduler) maxScheduled() int {
	if s.cfg.MaxNumScheduledTokens > 0 {
		return s.cfg.MaxNumScheduledTokens
	}
	return s.cfg.MaxNumBatchedTokens
}

func (s *Scheduler) maxActive() int {
	if s.cfg.MaxNumActiveSeqs > 0 {
		return s.cfg.MaxNumActiveSeqs
	}
	return s.cfg.MaxNumSeqs
}

// reservePrefillLookahead is _reserve_prefill_lookahead (scheduler.py:570-589): never end a
// prefill chunk within the drafter's prefill lookahead of the prefill end.
func (s *Scheduler) reservePrefillLookahead(r *Request, computed, numNew int) int {
	remaining := r.NumTokens() - computed - numNew
	if la := s.cfg.Spec.PrefillLookahead; 0 < remaining && remaining < la {
		numNew -= la - remaining
	}
	return max(numNew, 0)
}

// markPrefixReplay is _mark_prefix_replay (scheduler.py:3103-3121) without a KV load: it records
// where a prefix hit's replay starts and returns the hit tokens to recompute, the hit's last
// prefix_replay_tokens, whose sliding-window KV the worker rebuilds while their cached KV stays.
func (s *Scheduler) markPrefixReplay(r *Request, hitTokens int) int {
	window := s.KV.PrefixReplayTokens()
	if window == 0 {
		return 0
	}
	n := min(window, hitTokens)
	r.ReplayStart = hitTokens - n
	return n
}

// Schedule is schedule() (scheduler.py:591-1558) for this configuration.
func (s *Scheduler) Schedule() Output { return s.ScheduleThrottled(false) }

// ScheduleThrottled is schedule(throttle_prefills): on a throttled step of DP prefill
// balancing, every prefill (a new admission or a running chunk) is deferred unless the last
// release step was capacity-bound, while decodes still run (scheduler.py:634-638, 686-690,
// 1113-1116).
func (s *Scheduler) ScheduleThrottled(throttle bool) Output {
	s.currentStep++
	// finished_req_ids are handed to the runner with this output (scheduler.py:1649-1652).
	s.finishedPending = 0
	deferPrefills := false
	if throttle && !s.capacityBound {
		for _, r := range s.running {
			if !r.IsPrefillChunk {
				deferPrefills = true
				break
			}
		}
	}
	out := Output{Scheduled: map[string]int{}, PrefixHit: map[string]int{}, SpecDecode: map[string][]int64{}}
	tokenBudget := s.maxScheduled()
	inputBudget := s.cfg.MaxNumBatchedTokens
	draftSlots := s.cfg.Spec.DraftSlots
	k := s.cfg.Spec.NumTokens
	prefillScheduled := false

	// The long-prefill cap is waived when one request is all there is (:640-655).
	nr, nw := s.Counts()
	eligible := nr + nw
	threshold := 0
	if eligible > 1 {
		threshold = s.cfg.LongPrefillTokenThreshold
	}
	if threshold > 0 && s.cfg.LongPrefillAdaptive {
		threshold = max(threshold, inputBudget/eligible)
	}

	// Running requests (:657-886).
	var scheduledRunning []*Request
	var preempted []*Request
	reqIndex := 0
	for reqIndex < len(s.running) && tokenBudget > 0 {
		r := s.running[reqIndex]
		if inputBudget <= draftSlots {
			break
		}
		if r.Placeholders > 0 && r.NumComputed+2-r.Placeholders >= len(r.Prompt)+r.MaxTokens {
			// Async: the step in flight already reaches max_tokens (:664-677).
			reqIndex++
			continue
		}
		if s.currentStep < r.nextDecodeStep {
			// V2 + PP + async: pp_size steps between a request's decodes (:680-684).
			reqIndex++
			continue
		}
		if deferPrefills && r.IsPrefillChunk {
			// DP prefill balancing defers a running prefill chunk (:686-690).
			reqIndex++
			continue
		}
		numNew := r.numTokensWithSpec() + r.Placeholders - r.NumComputed
		if 0 < threshold && threshold < numNew {
			numNew = threshold
		}
		numNew = min(numNew, tokenBudget, inputBudget-draftSlots)
		// The input position must stay within max_model_len (:713-720): one slot is the
		// token this step samples.
		numNew = min(numNew, s.cfg.MaxModelLen-r.NumComputed-1)
		numNew = s.reservePrefillLookahead(r, r.NumComputed, numNew)
		if numNew <= 0 {
			// Skipped, not a stop: a later request may still fit (:755-773).
			reqIndex++
			continue
		}
		ok := false
		for {
			if s.KV.Allocate(r.ID, kv.Alloc{Computed: r.NumComputed, New: numNew, Lookahead: s.cfg.Spec.Lookahead,
				Hashes: r.hashes, NumTokens: r.NumTokens(), InFlight: r.numInFlight, HasScheduled: true, Run: s.run(r)}) {
				ok = true
				break
			}
			var victim *Request
			if s.cfg.Policy == Priority {
				victim = maxRunning(s.running)
				vi := indexOf(s.running, victim)
				s.running = append(s.running[:vi], s.running[vi+1:]...)
				if vi < reqIndex {
					reqIndex--
				}
				if j := indexOf(scheduledRunning, victim); j >= 0 {
					scheduledRunning = append(scheduledRunning[:j], scheduledRunning[j+1:]...)
					restored := out.Scheduled[victim.ID]
					delete(out.Scheduled, victim.ID)
					delete(out.SpecDecode, victim.ID)
					tokenBudget += restored
					inputBudget += restored + draftSlots
				}
			} else {
				victim = s.running[len(s.running)-1]
				s.running = s.running[:len(s.running)-1]
			}
			s.preempt(victim)
			preempted = append(preempted, victim)
			if victim == r {
				break
			}
		}
		if !ok {
			break
		}
		scheduledRunning = append(scheduledRunning, r)
		prefillScheduled = prefillScheduled || r.IsPrefillChunk
		out.Scheduled[r.ID] = numNew
		tokenBudget -= numNew
		inputBudget -= numNew + draftSlots
		reqIndex++
		// Speculative decode (:860-876): the drafts this step verifies are those beyond the
		// request's known tokens; the rest wait for update_draft_token_ids.
		if len(r.SpecTokens) > 0 {
			if n := numNew + r.NumComputed - r.NumTokens() - r.Placeholders; n > 0 {
				drafts := r.SpecTokens
				if len(drafts) > n {
					drafts = drafts[:n]
				}
				out.SpecDecode[r.ID] = append([]int64(nil), drafts...)
			}
			r.SpecTokens = nil
		}
	}

	// The LoRAs the scheduled running requests hold (:888-897).
	scheduledLoRAs := map[int]bool{}
	if s.cfg.MaxLoRAs > 0 {
		for _, r := range scheduledRunning {
			if r.LoRA > 0 {
				scheduledLoRAs[r.LoRA] = true
			}
		}
	}

	// Waiting requests, only when nothing was preempted (:899-1395). Requests holding KV blocks
	// (kv_holding_waiting) are drained before the rest (:913-917). A request skipped this step
	// is set aside and put back at the front of its queue afterwards, in its order (:1391-1395).
	var admitted []*Request
	if len(preempted) == 0 {
		var skipped, skippedHolding []*Request
		skip := func(q *queue) {
			r := q.pop()
			s.deferred[r.ID] = true
			if holdsKVBlocks(r) {
				skippedHolding = append(skippedHolding, r)
			} else {
				skipped = append(skipped, r)
			}
		}
		for tokenBudget > 0 {
			q := &s.holding
			if q.len() == 0 {
				q = &s.waiting
			}
			if q.len() == 0 || inputBudget <= draftSlots {
				break
			}
			if len(s.running) >= s.maxActive() {
				break
			}
			r := q.peek()
			if !s.handleBlockedWaiting(r) {
				skip(q)
				continue
			}
			if r.numStale > 0 {
				// Deliverable stale output still in flight: resuming now could resample a
				// position it later delivers (:934-938).
				skip(q)
				continue
			}
			// max_loras (:939-951): a request whose adapter would make one too many in this
			// batch is skipped this step, and those behind it still run.
			if s.cfg.MaxLoRAs > 0 && r.LoRA > 0 && len(scheduledLoRAs) == s.cfg.MaxLoRAs && !scheduledLoRAs[r.LoRA] {
				skip(q)
				continue
			}
			var hit [][]*kv.Block
			numComputed := r.NumComputed
			// newLocal is num_new_local_computed_tokens; external is the connector's tokens
			// beyond it, and async whether they load between steps.
			newLocal, external, async := 0, 0, false
			lookedUp := false
			if r.NumComputed == 0 {
				lookedUp = true
				hit, newLocal = s.KV.ComputedBlocks(r.hashes, r.NumTokens())
				if s.connector != nil {
					// The connector sees the block-aligned local hit (:968-980). A hit here is
					// always whole blocks, so the partial-tail cases (:1000-1027) never arise.
					external, async = s.connector.MatchedTokens(r, newLocal-newLocal%s.KV.BlockSize())
				}
				numComputed = newLocal + external
				// SWA bounded replay (:1047-1066): a hit no longer than the replayed window would be
				// recomputed in full and save nothing, so it is dropped. No connector runs beside
				// several KV cache groups (SetConnector), so no load restores the window here.
				if replay := s.KV.PrefixReplayTokens(); 0 < numComputed && numComputed <= replay {
					hit, newLocal, external, async, numComputed = nil, 0, 0, false, 0
				}
				if async {
					s.setFetchStage(r, FetchWaitingToStart)
				} else {
					s.setFetchStage(r, fetchNone)
				}
			}
			numNew, pad, replayed := 0, false, 0
			if !async {
				if deferPrefills && numComputed < r.NumTokens()-1 {
					// DP prefill balancing defers this step's prefill (:1113-1116): the pass
					// stops.
					break
				}
				room := min(tokenBudget, inputBudget-draftSlots)
				if lookedUp {
					// A fresh admission recomputes its hit's last window (:1117-1124).
					replayed = s.markPrefixReplay(r, numComputed)
					numComputed -= replayed
				}
				numNew = r.NumTokens() - numComputed
				// First-decode padding (:1130-1152): a request whose first step is a single
				// decode token runs as 1 + K, so the batch keeps the speculative decodes' shape.
				if k > 0 && numNew == 1 && !prefillScheduled && (len(scheduledRunning) > 0 || numComputed > 0) &&
					numComputed+1+k+1 <= s.cfg.MaxModelLen {
					if 1+k > room {
						break
					}
					numNew, pad = 1+k, true
				}
				if 0 < threshold && threshold < numNew {
					numNew = threshold
				}
				if !s.cfg.EnableChunkedPrefill && numNew > room {
					break
				}
				numNew = min(numNew, room)
				numNew = s.reservePrefillLookahead(r, numComputed, numNew)
				if numNew <= 0 {
					break
				}
			}
			// An asynchronous load computes nothing this step, allocates no lookahead slots
			// yet, and must leave the blocks the in-flight prefills still need (:1219-1251).
			lookahead, reserved := s.cfg.Spec.Lookahead, 0
			if async {
				lookahead, reserved = 0, s.inflightReserved()
			}
			// The replayed tokens are already counted in the adopted hit; a chunk that ends inside
			// them needs no new slots (:1253-1256).
			if !s.KV.Allocate(r.ID, kv.Alloc{Computed: r.NumComputed, New: max(numNew-replayed, 0), Lookahead: lookahead, NewComputed: hit,
				NumNewComputed: newLocal, External: external, Reserved: reserved, DelayCache: async,
				Hashes: r.hashes, NumTokens: r.NumTokens(), InFlight: r.numInFlight, Waiting: true,
				HasScheduled: len(s.running) > 0, FullSequenceMustFit: s.cfg.ReserveFullSequence, Run: s.run(r)}) {
				break
			}
			if s.connector != nil {
				s.connector.AfterAlloc(r, s.allocatedBlocks(r.ID), external)
			}
			if async {
				// WAITING_FOR_REMOTE_KVS (:1320-1341): the blocks are allocated, the load runs
				// between steps, and the request waits in kv_holding_waiting.
				r.Status = WaitingForRemoteKVs
				r.NumComputed = numComputed
				s.inflight[r] = true
				s.setFetchStage(r, FetchInProgress)
				skip(q)
				continue
			}
			q.pop()
			delete(s.deferred, r.ID)
			s.running = append(s.running, r)
			admitted = append(admitted, r)
			if s.cfg.MaxLoRAs > 0 && r.LoRA > 0 {
				scheduledLoRAs[r.LoRA] = true
			}
			out.Scheduled[r.ID] = numNew
			if lookedUp {
				out.PrefixHit[r.ID] = newLocal
			}
			tokenBudget -= numNew
			inputBudget -= numNew + draftSlots
			r.Status = Running
			r.NumComputed = numComputed
			s.setFetchStage(r, fetchNone)
			if pad {
				drafts := make([]int64, k)
				for i := range drafts {
					drafts[i] = -1
				}
				out.SpecDecode[r.ID] = drafts
			}
			// Only requests still prefilling after this chunk stay in flight (:1376-1378).
			if numComputed+numNew < r.NumTokens() {
				s.inflight[r] = true
			}
		}
		// prepend_requests (:1391-1395): extendleft on a deque reverses, which restores the
		// skipped requests' order at the front; a priority queue re-sorts.
		for i := len(skippedHolding) - 1; i >= 0; i-- {
			s.holding.prepend(skippedHolding[i])
		}
		for i := len(skipped) - 1; i >= 0; i-- {
			s.waiting.prepend(skipped[i])
		}
		if !deferPrefills {
			// On a step that released prefills, whether it was capacity-bound (:1397-1401).
			_, nw := s.Counts()
			s.capacityBound = nw > len(s.deferred)
		}
	}

	for _, r := range scheduledRunning {
		out.Order = append(out.Order, r.ID)
	}
	for _, r := range admitted {
		out.Order = append(out.Order, r.ID)
		out.NewIDs = append(out.NewIDs, r.ID)
	}
	for _, r := range preempted {
		out.Preempted = append(out.Preempted, r.ID)
	}
	// _update_after_schedule (:1616-1648): advance computed tokens now, so a prefill can
	// be scheduled again next step.
	for id, n := range out.Scheduled {
		r := s.requests[id]
		r.NumComputed += n
		r.numInFlight += n
		r.IsPrefillChunk = r.NumComputed < r.NumTokens()+r.Placeholders
		if s.cfg.Async && !r.IsPrefillChunk {
			// AsyncScheduler._update_after_schedule (async_scheduler.py:25-58): the token this
			// step samples, and the drafts it verifies, are placeholders until the output
			// returns.
			r.Placeholders += 1 + len(out.SpecDecode[id])
			// The next step's drafts are num_spec_tokens_to_schedule placeholders (:28-31,
			// :48-50): the worker proposes the real ones itself, and EngineCore.post_step
			// never calls update_draft_token_ids under async scheduling
			// (vllm/v1/engine/core.py:689-696@5df63a72). num_spec_tokens_to_schedule is
			// num_spec_tokens without dynamic speculative decoding (scheduler.py:1486-1490),
			// which no deployment this engine builds turns on.
			if k := s.cfg.Spec.NumTokens; k > 0 {
				r.SpecTokens = placeholderDrafts(k)
			}
			if s.cfg.UseV2ModelRunner {
				r.nextDecodeStep = s.currentStep + max(s.cfg.DecodeStagger, 1)
			}
		}
		if !r.IsPrefillChunk {
			delete(s.inflight, r)
		}
	}
	return out
}

// inflightReserved is _inflight_prefill_reserved_blocks plus _spec_decode_step_blocks
// (scheduler.py:3088-3092, 346-357, _request_remaining_blocks :3060-3076): the blocks the
// in-flight prefills still need to hold their whole sequences, which an asynchronous load
// must leave free.
func (s *Scheduler) inflightReserved() int {
	extra := 0
	if k := s.cfg.Spec.NumTokens; k > 0 {
		extra = 1 + k + s.cfg.Spec.Lookahead
	}
	n := s.KV.SpecDecodeStepBlocks(extra)
	for r := range s.inflight {
		n += s.KV.RemainingBlocks(r.ID, min(r.NumTokens(), s.cfg.MaxModelLen)+extra, r.NumComputed)
	}
	return n
}

// allocatedBlocks is the request's block table as update_state_after_alloc sees it.
func (s *Scheduler) allocatedBlocks(id string) AllocatedBlocks {
	bs := s.KV.Blocks(id)
	out := AllocatedBlocks{IDs: make([]int, len(bs)), Hashed: make([]bool, len(bs))}
	for i, b := range bs {
		out.IDs[i] = b.ID
		out.Hashed[i] = b.Hash != nil && !b.Null
	}
	return out
}

func cdiv(a, b int) int { return (a + b - 1) / b }

// run is what a KV event reports about r's blocks; nothing is built with events off.
func (s *Scheduler) run(r *Request) kv.StoredRun {
	if !s.cfg.KVEvents {
		return kv.StoredRun{}
	}
	return kv.StoredRun{Tokens: r.allTokens(), LoRA: r.LoRAName}
}

// preempt is _preempt_request (:1571-1614) for recompute preemption. Under async
// scheduling the request's in-flight output becomes stale: it still returns, but must not
// move the counters the preemption reset (:1596-1607).
func (s *Scheduler) preempt(r *Request) {
	s.KV.Free(r.ID)
	r.Status = Preempted
	r.NumComputed = 0
	r.SpecTokens = nil
	r.numStale = r.numInFlight
	r.Placeholders = 0
	r.NumPreemptions++
	delete(s.inflight, r)
	s.waiting.prepend(r)
}

// Update is update_from_output (:2026-2425): each scheduled request that completed its
// known tokens samples the tokens in sampled (none means it sampled nothing): one, plus
// the drafts the target accepted. Rejected drafts roll num_computed_tokens back (:2110-2130).
// A request stops on EOS, at max_tokens, or at max_model_len (check_stop,
// sched/utils.py:98-140), checked after each token, the rest discarded
// (_update_request_with_output, :2493-2510).
//
// Requests are visited in num_scheduled_tokens' insertion order (:2075), which is out.Order:
// the scheduled running requests in running order, then the admitted ones. The order is not
// cosmetic: a request that stops frees its blocks there and then (_free_request), so it
// decides which freed prefix blocks sit nearer the free queue's head and are evicted first.
func (s *Scheduler) Update(out Output, sampled map[string][]int64) []Finished {
	return s.UpdateWith(out, sampled, TransferOutcome{})
}

// UpdateWith is Update with the worker side's report of the step's KV transfers
// (ModelRunnerOutput.kv_connector_output). Failed loads are handled before the requests are
// visited (scheduler.py:2058-2071), the requests a "fail" policy fails are finished as errors
// after them (:2306-2324), and finished transfers are applied last (:2342-2343).
func (s *Scheduler) UpdateWith(out Output, sampled map[string][]int64, o TransferOutcome) []Finished {
	var finished []Finished
	var stopped []*Request
	var skip map[string]bool
	var fail []string
	if !o.empty() {
		skip, fail = s.failedLoads(o, out.Scheduled)
	}
	for _, id := range out.Order {
		r := s.requests[id]
		stale := false
		if r != nil {
			r.numInFlight -= out.Scheduled[id]
			if r.numStale > 0 {
				stale = true
				r.numStale -= out.Scheduled[id]
			}
		}
		if skip[id] {
			// A failed or rescheduled KV load (:2087-2089).
			continue
		}
		if r == nil || r.Status >= FinishedStopped {
			continue
		}
		toks := sampled[id]
		if drafts := out.SpecDecode[id]; len(drafts) > 0 && len(toks) > 0 && !stale {
			// A stale rejection count predates the preemption's rollback (:2118-2128).
			rejected := len(drafts) - max(len(toks)-1, 0)
			if r.NumComputed > 0 {
				r.NumComputed -= rejected
			}
			if r.Placeholders > 0 {
				r.Placeholders -= rejected
			}
		}
		if len(toks) == 0 {
			continue
		}
		running := r.Status == Running
		var status Status
		stop := false
		emitted := 0
		for _, tok := range toks {
			r.Output = append(r.Output, tok)
			r.updateHashes(s.KV.HashBlockSize())
			emitted++
			switch {
			case tok == r.EOS:
				status, stop = FinishedStopped, true
			case r.NumTokens() >= s.cfg.MaxModelLen || len(r.Output) >= r.MaxTokens:
				status, stop = FinishedLength, true
			}
			if stop {
				break
			}
		}
		if s.cfg.Async {
			// AsyncScheduler._update_request_with_output (async_scheduler.py:60-78): the
			// sampled tokens replace their placeholders, unless the output is stale, and a
			// running request caches the blocks its now-known tokens fill.
			if !stale {
				r.Placeholders -= emitted
			}
			if running {
				s.KV.CacheUpTo(r.ID, r.hashes, r.NumComputed-r.Placeholders, s.run(r))
			}
		}
		if !stop {
			continue
		}
		r.Status = status
		r.SpecTokens = nil
		stopped = append(stopped, r)
		finished = append(finished, Finished{ID: r.ID, Status: status, Output: len(r.Output)})
		s.freeRequest(r, false)
	}
	if len(stopped) > 0 {
		keep := s.running[:0]
		for _, r := range s.running {
			if indexOf(stopped, r) < 0 {
				keep = append(keep, r)
			}
		}
		s.running = keep
		for _, r := range stopped {
			// A request stopped by stale output while preempted leaves the waiting queue
			// (:2296-2300).
			s.waiting.remove(r)
			s.holding.remove(r)
			delete(s.deferred, r.ID)
		}
	}
	for _, r := range s.finishRequests(fail, FinishedError) {
		finished = append(finished, Finished{ID: r.ID, Status: FinishedError, Output: len(r.Output)})
	}
	if !o.empty() {
		s.transfersFinished(o)
	}
	return finished
}

// placeholderDrafts are k draft placeholders, -1 each: the AsyncScheduler's
// _spec_token_placeholders (async_scheduler.py:17, 29-31@5df63a72). The rejection sampler always
// rejects one, as a real draft it never verified (vllm/v1/sample/rejection_sampler.py:762-763).
func placeholderDrafts(k int) []int64 {
	d := make([]int64, k)
	for i := range d {
		d[i] = -1
	}
	return d
}

// UpdateDrafts is update_draft_token_ids (:2550-2569): the drafter's proposals after a step,
// for each request it ran. A request still prefilling takes none.
func (s *Scheduler) UpdateDrafts(drafts map[string][]int64) {
	for id, d := range drafts {
		r := s.requests[id]
		if r == nil || r.Status >= FinishedStopped {
			continue
		}
		if r.IsPrefillChunk {
			r.SpecTokens = nil
			continue
		}
		r.SpecTokens = append([]int64(nil), d...)
	}
}

func maxRunning(rs []*Request) *Request {
	best := rs[0]
	for _, r := range rs[1:] {
		if r.Priority > best.Priority || (r.Priority == best.Priority && r.ArrivalTime > best.ArrivalTime) {
			best = r
		}
	}
	return best
}

func indexOf(rs []*Request, r *Request) int {
	for i, x := range rs {
		if x == r {
			return i
		}
	}
	return -1
}

// queue is FCFSRequestQueue or PriorityRequestQueue (request_queue.py:75-207).
type queue struct {
	priority bool
	items    []*Request
}

func (q *queue) len() int { return len(q.items) }

func (q *queue) add(r *Request) {
	q.items = append(q.items, r)
	if q.priority {
		q.sort()
	}
}

// prepend is prepend_request: the front for FCFS; for priority, ordering decides.
func (q *queue) prepend(r *Request) {
	if q.priority {
		q.add(r)
		return
	}
	q.items = append([]*Request{r}, q.items...)
}

func (q *queue) peek() *Request { return q.items[0] }

// remove is remove_requests for one request; absent is a no-op.
func (q *queue) remove(r *Request) {
	for i, x := range q.items {
		if x == r {
			q.items = append(q.items[:i], q.items[i+1:]...)
			return
		}
	}
}

func (q *queue) pop() *Request {
	r := q.items[0]
	q.items = q.items[1:]
	return r
}

// sort orders by Request.__lt__ (vllm/v1/request.py:357-367): priority, arrival time,
// request ID.
func (q *queue) sort() {
	sort.SliceStable(q.items, func(i, j int) bool {
		a, b := q.items[i], q.items[j]
		if a.Priority != b.Priority {
			return a.Priority < b.Priority
		}
		if a.ArrivalTime != b.ArrivalTime {
			return a.ArrivalTime < b.ArrivalTime
		}
		return a.ID < b.ID
	})
}
