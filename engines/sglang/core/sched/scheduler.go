// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/managers/scheduler.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package sched

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/inference-sim/blis-sglang/core/cache"
)

// NewTokenRatioConfig is NewTokenRatioTracker.from_config's result (scheduler_components/
// new_token_ratio_tracker.py@297e54fd).
type NewTokenRatioConfig struct {
	Init  float64 `json:"init"`
	Min   float64 `json:"min"`
	Decay float64 `json:"decay"`
}

// Config is what the scheduler reads of SGLang's resolved settings. T1 (config/) resolves them;
// the T2 replay takes them from the oracle's own resolution.
type Config struct {
	PageSize           int    `json:"page_size"`
	Policy             string `json:"policy"`
	MaxTotalNumTokens  int    `json:"max_total_num_tokens"`
	MaxPrefillTokens   int    `json:"max_prefill_tokens"`
	ChunkedPrefillSize *int   `json:"chunked_prefill_size"`
	MaxRunningRequests int    `json:"max_running_requests"`
	ReqPoolSize        int    `json:"req_pool_size"`
	MaxContextLen      int    `json:"max_context_len"`
	// PPMaxMicroBatchSize bounds the requests running and admitted together: max_running_requests
	// divided by the pipeline size (scheduler.py:1237-1244@297e54fd). PPSize above 1 is refused.
	PPMaxMicroBatchSize int                 `json:"pp_max_micro_batch_size"`
	PPSize              int                 `json:"pp_size"`
	MaxReqLen           int                 `json:"max_req_len"`
	MaxReqInputLen      int                 `json:"max_req_input_len"`
	IsMixedChunk        bool                `json:"is_mixed_chunk"`
	Disable             bool                `json:"radix_cache_disabled"`
	NeedSort            bool                `json:"need_sort"`
	EOSTokenIDs         []int64             `json:"eos_token_ids"`
	VocabSize           int                 `json:"vocab_size"`
	NewTokenRatio       NewTokenRatioConfig `json:"new_token_ratio"`
	RetractDecodeSteps  int                 `json:"retract_decode_steps"`
	ClipMaxNewTokens    int                 `json:"clip_max_new_tokens"`
	// InBatchPrefixCheckThreshold and InBatchPrefixDeprioritizeThreshold are the module constants
	// of schedule_policy.py:106-117@297e54fd.
	InBatchPrefixCheckThreshold        int    `json:"in_batch_prefix_check_threshold"`
	InBatchPrefixDeprioritizeThreshold int    `json:"in_batch_prefix_deprioritize_threshold"`
	EnablePriorityScheduling           bool   `json:"enable_priority_scheduling"`
	PrefillDecodeInterval              int    `json:"prefill_decode_interval"`
	Allocator                          string `json:"allocator"`
	TreeCache                          string `json:"tree_cache"`
	DisaggregationMode                 string `json:"disaggregation_mode"`
	EnableOverlap                      bool   `json:"enable_overlap"`
	// DisableConsecutivePrefillOverlap is SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP
	// (environ.py:693@297e54fd, default off): two extend batches in a row do not overlap.
	DisableConsecutivePrefillOverlap bool `json:"disable_consecutive_prefill_overlap"`
	// EnableMetrics and DecodeLogInterval are --enable-metrics and --decode-log-interval: the
	// scheduler makes its Reports only when metrics are on (SchedulerMetricsReporter.
	// current_scheduler_metrics_enabled), and a decode batch logs every interval-th time.
	EnableMetrics bool `json:"enable_metrics"`
	// AllowAutoTruncate is --allow-auto-truncate (get_serving().allow_auto_truncate).
	AllowAutoTruncate bool `json:"allow_auto_truncate"`
	// EnableKVEvents is --kv-events-config naming a publisher on this rank: the scheduler takes the
	// tree cache's events at the points SchedulerKvEventsPublisher.publish_kv_events is called.
	EnableKVEvents bool `json:"enable_kv_events"`
	// KVEventsDynamo is the dynamo event format (KVEventsConfig.format): block hashes carry the
	// request's extra_key and cache_salt.
	KVEventsDynamo bool `json:"kv_events_dynamo"`
	// ContextLen is the model's context length (model_config.context_len), which the context_len
	// gauge reports; MaxContextLen is the request pool's row length.
	ContextLen        int `json:"context_len"`
	DecodeLogInterval int `json:"decode_log_interval"`
	// RadixEvictionPolicy is the tree's eviction policy; empty is lru, SGLang's default.
	RadixEvictionPolicy string `json:"radix_eviction_policy"`
	// RetractionPolicy is --retraction-policy; empty is "length", the default the port models.
	RetractionPolicy string `json:"retraction_policy"`
	// RandomSeed seeds the random policy's stream (server_args.random_seed).
	RandomSeed int `json:"random_seed"`
	// ScheduleLowPriorityValuesFirst, PriorityPreemptionThreshold and EnablePriorityPreemption are
	// the priority-scheduling settings (scheduler.py:486-494, 1469-1472@297e54fd).
	ScheduleLowPriorityValuesFirst bool `json:"schedule_low_priority_values_first"`
	PriorityPreemptionThreshold    int  `json:"priority_scheduling_preemption_threshold"`
	EnablePriorityPreemption       bool `json:"enable_priority_preemption"`
	AbortOnPriorityWhenDisabled    bool `json:"abort_on_priority_when_disabled"`
	// MaxQueuedRequests bounds the waiting queue; nil is unbounded.
	MaxQueuedRequests *int `json:"max_queued_requests"`
	// PrefillMaxRequests bounds the requests one prefill batch admits; nil is unbounded.
	PrefillMaxRequests *int `json:"prefill_max_requests"`
}

func (c Config) radixEvictionPolicy() string {
	if c.RadixEvictionPolicy == "" {
		return "lru"
	}
	return c.RadixEvictionPolicy
}

type refusal struct{ what, issue string }

func refused(what string) refusal { return refusal{what, Refusal} }

// RetractionRefusal is the open issue the priority retraction policy and the abort of the last
// request cite (X28).
const RetractionRefusal = "inference-sim/blis-sglang#77"

func refusedAt(what, issue string) refusal { return refusal{what, issue} }

func (r refusal) Error() string {
	return fmt.Sprintf("sched: %s is not modelled (%s)", r.what, r.issue)
}

// ErrRefused is wrapped by every error that names a feature the port refuses.
var ErrRefused = errors.New("refused")

// Scheduler is the state of Scheduler (managers/scheduler.py@297e54fd) the normal loop keeps
// between iterations.
type Scheduler struct {
	cfg    Config
	policy Policy
	tree   cache.PrefixCache
	alloc  cache.Allocator
	pool   cache.ReqToTokenPool

	waiting       []*Req
	running       *Batch
	last          *Batch
	chunkedReq    *Req
	newTokenRatio struct{ current float64 }
	waitTree      simTree
	// Shuffle orders the waiting queue for the random policy: it receives the request ids and returns
	// them in the order to schedule. The default draws from the stream named
	// "sched/random-policy/waiting-queue"; the replay test supplies SGLang's order.
	Shuffle func(rids []string) []string
	stream  *Stream

	waitingPrefixRefreshCalls int

	// resultQueue is Scheduler.result_queue (scheduler.py:2064@297e54fd): the launched batches
	// whose results the overlap loop has not processed. At scheduling time it holds at most one.
	resultQueue []*Batch

	// reports are the Reports made this step; numRetracted and forwardCtDecode are the
	// reporter's counters (metrics_reporter.py: num_retracted_reqs, forward_ct_decode).
	reports []Report
	// abortedSinceTake are requests finished at ingest, for TakeAborted.
	abortedSinceTake []string
	// preemptedSinceTake are the requests retracted or preempted since the last TakeReports.
	preemptedSinceTake []string
	numRetracted       int
	forwardCtDecode    int

	processedTokens  int
	forwardCt        int
	step             int
	sampledToken     int64
	finishedThisStep []string
}

// New builds the scheduler over a tree cache, an allocator and a request pool (Scheduler.__init__,
// scheduler.py:470-735@297e54fd, with the cache pieces made by the caller). It refuses, by name,
// what the normal loop does not model.
func New(cfg Config, tree cache.PrefixCache, alloc cache.Allocator, pool cache.ReqToTokenPool, sampledToken int64) (*Scheduler, error) {
	switch {
	case cfg.RetractionPolicy != "" && cfg.RetractionPolicy != "length" && cfg.RetractionPolicy != "priority":
		return nil, fmt.Errorf("sched: the %s retraction policy does not exist in SGLang v0.5.23 (choices: length, priority)", cfg.RetractionPolicy)
	case cfg.DisaggregationMode != "null" && cfg.DisaggregationMode != "":
		return nil, fmt.Errorf("sched: disaggregation mode %q belongs to S25 (%s)", cfg.DisaggregationMode, Refusal)
	case cfg.PPSize > 1:
		return nil, fmt.Errorf("sched: pipeline parallelism belongs to S37 (%s)", Refusal)
	case cfg.PrefillDecodeInterval != 0:
		return nil, refused("a prefill-decode interval")
	case cfg.TreeCache != "UnifiedRadixCache":
		return nil, refused("tree cache " + cfg.TreeCache)
	case cfg.Allocator != "TokenToKVPoolAllocator" && cfg.Allocator != "PagedTokenToKVPoolAllocator":
		return nil, refused("allocator " + cfg.Allocator)
	case cfg.ChunkedPrefillSize != nil && *cfg.ChunkedPrefillSize <= 0:
		return nil, fmt.Errorf("sched: a non-positive chunked prefill size is chunked prefill off, which SGLang spells None")
	}
	p, err := validatePolicy(cfg.Policy, cfg.Disable)
	if err != nil {
		return nil, err
	}
	s := &Scheduler{cfg: cfg, policy: p, tree: tree, alloc: alloc, pool: pool, sampledToken: sampledToken,
		running: &Batch{}}
	s.newTokenRatio.current = cfg.NewTokenRatio.Init
	s.stream = NewStream(int64(cfg.RandomSeed), "sched/random-policy/waiting-queue")
	return s, nil
}

// matchKVCache is match_kv_cache (mem_cache/common.py:221-268): match tokenIDs, up to maxPrefix,
// and adopt the hit as the request's prefix.
func (s *Scheduler) matchKVCache(r *Req, tokenIDs []int64, maxPrefix *int) cache.MatchResult {
	toks := tokenIDs
	if maxPrefix != nil && *maxPrefix < len(toks) {
		toks = toks[:*maxPrefix]
	}
	res := s.tree.MatchPrefix(cache.MatchPrefixParams{
		Key: cache.Key{Tokens: slices.Clone(toks), ExtraKey: r.C.ExtraKey, CacheSalt: r.C.CacheSalt}, Req: &r.C})
	r.C.PrefixLen = res.DevicePrefixLen
	last := res.LastDeviceNode
	r.C.LastNode = &last
	if res.HostHitLength != 0 {
		panic(refused("a host-tier hit"))
	}
	r.NumMatchedPrefixTokens = min(r.C.PrefixLen, r.ComputeMaxPrefixLen(len(tokenIDs)))
	if res.CacheProtectedLen != nil {
		r.C.KV.CacheProtectedLen = *res.CacheProtectedLen
	}
	return res
}

// initNextRoundInput is init_next_round_input (:1692-1764): refresh the fill ids and, given a
// tree, match them for a new round.
func (s *Scheduler) initNextRoundInput(r *Req, withTree bool) {
	r.RefreshFillIDs()
	if withTree {
		limit := r.ComputeMaxPrefixLen(r.InputLen())
		res := s.matchKVCache(r, r.C.FillIDs, &limit)
		if res.CacheProtectedLen == nil {
			r.C.KV.CacheProtectedLen = r.C.PrefixLen
		}
	}
}

// handleGenerateRequest is _handle_generate_request (scheduler.py:2959-3300) for a text-only
// request with no session, grammar, logprobs or multimodal input. A request SGLang rejects at
// ingest is not refused: it is aborted the way SGLang aborts it (an input too long, a prompt past
// the KV budget, a priority sent to a server that has none, a full queue).
func (s *Scheduler) handleGenerateRequest(a Arrival) {
	for _, t := range a.Prompt {
		if t < 0 || int(t) >= s.cfg.VocabSize {
			// SGLang does not abort it; the embedding lookup of a real model fails.
			panic(refused("a prompt token outside the vocabulary"))
		}
	}
	r := &Req{MaxNewTokens: a.MaxNewTokens, IgnoreEOS: a.IgnoreEOS, RoutingKey: a.RoutingKey, Script: a.Script}
	if a.Priority != nil {
		r.Priority = *a.Priority
		r.HasPriority = true
	}
	r.C = cache.Req{ID: a.RID, OriginInputIDs: slices.Clone(a.Prompt)}
	if a.CacheSalt != "" {
		salt := a.CacheSalt
		r.C.CacheSalt = &salt
	}
	s.initReqMaxNewTokens(r)
	// validate_input_length (managers/utils.py:259-289@297e54fd): an input at or past the limit is cut
	// to it under --allow-auto-truncate, and aborted otherwise.
	if len(r.C.OriginInputIDs) >= s.cfg.MaxReqInputLen {
		if s.cfg.AllowAutoTruncate {
			r.C.OriginInputIDs = r.C.OriginInputIDs[:s.cfg.MaxReqInputLen]
		} else {
			r.setFinishWithAbort()
		}
	}
	s.addRequestToQueue(r)
}

// setFinishWithAbort is set_finish_with_abort (schedule_batch.py:2221-2238): the request shrinks to
// one token, so its prefill is as cheap as possible, and finishes with an abort when that token is
// processed.
func (r *Req) setFinishWithAbort() {
	r.C.OriginInputIDs = []int64{0}
	r.ToFinish = "abort"
}

// initReqMaxNewTokens is init_req_max_new_tokens (:2693-2730).
func (s *Scheduler) initReqMaxNewTokens(r *Req) {
	inputLen := len(r.C.OriginInputIDs)
	maxNew := max(0, min(r.MaxNewTokens, s.cfg.MaxReqLen-inputLen-1))
	capacity := s.cfg.MaxTotalNumTokens
	m, ok := s.alloc.MaxNewTokensForMemory(inputLen, maxNew, capacity, nil, s.cfg.ChunkedPrefillSize)
	if !ok {
		r.setFinishWithAbort()
		m = 0
	}
	r.MaxNewTokens = max(0, m)
}

// setOrValidatePriority is _set_or_validate_priority (:3534-3555): false when the request is
// answered with an abort instead of being queued.
func (s *Scheduler) setOrValidatePriority(r *Req) bool {
	if s.cfg.EnablePriorityScheduling && !r.HasPriority {
		if s.cfg.ScheduleLowPriorityValuesFirst {
			r.Priority = math.MaxInt64
		} else {
			r.Priority = math.MinInt64
		}
		r.HasPriority = true
	} else if !s.cfg.EnablePriorityScheduling && r.HasPriority && s.cfg.AbortOnPriorityWhenDisabled {
		s.noteFinished(r)
		return false
	}
	return true
}

// abortOnQueuedLimit is _abort_on_queued_limit (:3552-3596): when the queue is full, the incoming
// request is aborted, or under priority scheduling the least preferred waiting request if the
// incoming one is strictly better. It reports whether the incoming request was aborted.
func (s *Scheduler) abortOnQueuedLimit(in *Req) bool {
	if s.cfg.MaxQueuedRequests == nil || len(s.waiting)+1 <= *s.cfg.MaxQueuedRequests {
		return false
	}
	victim := in
	if s.cfg.EnablePriorityScheduling {
		// The least preferred waiting request: the largest (priority*direction, queue time), the
		// first of equals.
		idx := 0
		for i, c := range s.waiting {
			w := s.waiting[idx]
			if s.priorityBefore(w.Priority, c.Priority) || (w.Priority == c.Priority && c.WaitQueueEntryStep > w.WaitQueueEntryStep) {
				idx = i
			}
		}
		if cand := s.waiting[idx]; s.priorityBefore(in.Priority, cand.Priority) {
			s.waiting = slices.Delete(s.waiting, idx, idx+1)
			victim = cand
		}
	}
	s.noteFinished(victim)
	return victim == in
}

// addRequestToQueue is _add_request_to_queue (:3480-3502) outside disaggregation.
func (s *Scheduler) addRequestToQueue(r *Req) {
	if !s.setOrValidatePriority(r) {
		return
	}
	if s.abortOnQueuedLimit(r) {
		return
	}
	// The tree records the request's priority on the nodes it inserts (unified_radix_cache.py:1186@297e54fd).
	r.C.Priority = r.Priority
	s.waiting = append(s.waiting, r)
	r.WaitQueueEntryStep = s.step
	r.ArrivalProcessedTokens = s.processedTokens
}

// isFullyIdle is is_fully_idle (:5337-5365) outside pipeline parallelism and disaggregation.
func (s *Scheduler) isFullyIdle() bool {
	return s.running.IsEmpty() && s.chunkedReq == nil && (s.last == nil || s.last.IsEmpty()) &&
		(!s.cfg.EnableOverlap || len(s.resultQueue) == 0) && len(s.waiting) == 0
}

// getNextBatchToRun is get_next_batch_to_run (:3894-4050).
func (s *Scheduler) getNextBatchToRun() *Batch {
	running, last := s.running, s.last
	exclude := map[*Req]bool{}
	if s.chunkedReq != nil {
		// Move the chunked request out of the batch so that we can merge only finished requests
		// to running_batch.
		exclude[s.chunkedReq] = true
		if s.chunkedReq.ExtendEnd > s.chunkedReq.C.PrefixLen {
			s.stashChunkedRequest(s.chunkedReq)
		}
	}
	if last != nil && (last.Mode == Extend || last.Mode == Mixed) {
		if last.ChunkedReq != nil {
			exclude[last.ChunkedReq] = true
		}
		lastBS := last.Size()
		last.FilterBatch(exclude)
		if last.Size() < lastBS {
			running.BatchIsFull = false
		}
		// Merge the new batch into the running batch.
		if !last.IsEmpty() && last != running {
			if running.IsEmpty() {
				running = last
			} else {
				running.MergeBatch(last)
			}
		}
	}
	// Every step, not only after a merge.
	runningBS := running.Size()
	running.FilterBatch(nil)
	if running.Size() < runningBS {
		running.BatchIsFull = false
	}
	newBatch, running := s.getNewBatchPrefill(running)
	s.running = running
	if newBatch != nil {
		// Run prefill first if possible.
		return newBatch
	}
	if !running.IsEmpty() {
		if d := s.updateRunningBatch(running); d != nil && !d.IsEmpty() {
			return d
		}
	}
	return nil
}

// stashChunkedRequest is stash_chunked_request (:3772-3776): publish the chunk's KV.
func (s *Scheduler) stashChunkedRequest(r *Req) { s.checkpointKVCache(r) }

// checkpointKVCache is checkpoint_kv_cache (common.py:271-280).
func (s *Scheduler) checkpointKVCache(r *Req) {
	if r.Finished() {
		panic("checkpointing finished request " + r.RID())
	}
	s.tree.Checkpoint(&r.C, r.ExtendEnd)
	// The next extend resumes after this one.
	r.C.PrefixLen = r.ExtendEnd
}

// allocatableReqs is get_num_allocatable_reqs (:4025-4049) with no beams: the micro-batch's free
// seats, or the free request rows plus those the counted requests already hold, whichever is less.
func (s *Scheduler) allocatableReqs(runningBS, heldRows int) int {
	ppBudget := s.cfg.PPMaxMicroBatchSize - runningBS
	rowBudget := max(s.pool.AvailableSize()+heldRows, 0)
	return min(ppBudget, rowBudget)
}

// getNewBatchPrefill is get_new_batch_prefill and _get_new_batch_prefill_raw (:4052-4457).
func (s *Scheduler) getNewBatchPrefill(running *Batch) (*Batch, *Batch) {
	if s.cfg.EnablePriorityPreemption || running.IsEmpty() {
		// Preemption can make room, so a full batch is not final; an empty batch has no request
		// left to leave, and admission would stay shut.
		running.BatchIsFull = false
	}
	if (running.BatchIsFull || len(s.waiting) == 0) && s.chunkedReq == nil {
		return nil, running
	}
	runningBS := len(running.Reqs)
	// A chunked request skips this check: it keeps its row across chunks.
	if s.allocatableReqs(runningBS, 0) <= 0 && s.chunkedReq == nil && !s.cfg.EnablePriorityPreemption {
		running.BatchIsFull = true
		return nil, running
	}
	s.calcPriority(s.waiting, running)

	mixedDecode := 0
	if s.cfg.IsMixedChunk {
		mixedDecode = runningBS
	}
	a := s.newAdder(running, mixedDecode)
	if s.chunkedReq != nil {
		s.initNextRoundInput(s.chunkedReq, false)
		var rem int
		if a.remChunkTokens != nil {
			rem = *a.remChunkTokens
		}
		a.chunkedReqLimit = s.shortestPrefillChunkLimit(s.chunkedReq, s.waiting, rem)
		s.chunkedReq = a.addChunkedReq(s.chunkedReq)
	}
	// Get requests from the waiting queue to a new prefill batch.
	for _, r := range s.waiting {
		// A spent pass budget admits no one, even before the first candidate.
		if a.budgetState() == Other {
			break
		}
		runningBS = len(running.Reqs)
		held := 0
		for _, c := range a.canRunList {
			if c.C.KV.ReqPoolIdx != nil {
				held++
			}
		}
		if len(a.canRunList) >= s.allocatableReqs(runningBS, held) {
			running.BatchIsFull = true
		}
		if running.BatchIsFull && (!s.cfg.EnablePriorityPreemption || !a.preemptToSchedule(r)) {
			break
		}
		s.initNextRoundInput(r, true)
		res := a.addOneReq(r, s.chunkedReq != nil)
		if res != Continue {
			if res == NoToken {
				// Set batch_is_full after making sure there are requests that can be served.
				running.BatchIsFull = len(a.canRunList) > 0 || !running.IsEmpty()
			}
			break
		}
	}
	// Update waiting queue.
	if len(a.canRunList) == 0 {
		for _, r := range a.preemptList {
			s.preemptedSinceTake = append(s.preemptedSinceTake, r.RID())
			s.addRequestToQueue(r)
		}
		return nil, running
	}
	admitted := map[*Req]bool{}
	for _, r := range a.canRunList {
		admitted[r] = true
	}
	var rest []*Req
	for _, r := range s.waiting {
		if !admitted[r] {
			rest = append(rest, r)
		}
	}
	s.waiting = rest
	for _, r := range a.preemptList {
		s.preemptedSinceTake = append(s.preemptedSinceTake, r.RID())
		s.addRequestToQueue(r)
	}
	if a.newChunkedReq != nil {
		if s.chunkedReq != nil {
			panic("a second chunked request")
		}
		s.chunkedReq = a.newChunkedReq
	}
	if s.chunkedReq != nil {
		s.chunkedReq.InflightMiddleChunks++
	}
	nb := &Batch{Reqs: a.canRunList, ChunkedReq: s.chunkedReq}
	nb.ContainsLastPrefillChunk = s.chunkedReq == nil || len(a.canRunList) != 1
	nb.PrepareForExtend(s)
	nb.Prefill = s.prefillStats(a, running)

	// Mixed-style chunked prefill.
	if s.cfg.IsMixedChunk && !running.IsEmpty() {
		finishing := s.reqsFinishingInflight(running)
		running.FilterBatch(finishing)
		if len(finishing) > 0 {
			running.BatchIsFull = false
		}
		// The prefill admitted above may leave no KV room for the decode rows. They then sit out
		// this step and decode on the next decode step, which fits them.
		if !running.IsEmpty() && running.CheckDecodeMem(s, nil) {
			running.PrepareForDecode(s)
			nb.MixWithRunning(running)
			nb.DecodingReqs = slices.Clone(running.Reqs)
			running = &Batch{BatchIsFull: running.BatchIsFull}
		}
	}
	return nb, running
}

// updateRunningBatch is update_running_batch (:4459-4560): retract if the decode step does not
// fit, then prepare it.
func (s *Scheduler) updateRunningBatch(b *Batch) *Batch {
	initial := b.Size()
	finishing := s.reqsFinishingInflight(b)
	b.FilterBatch(finishing)
	if b.IsEmpty() {
		b.BatchIsFull = false
		return b
	}
	retract := !b.CheckDecodeMem(s, nil)
	if retract && len(finishing) > 0 {
		// The queued result frees the finishing requests' KV at the end of this step; decode the
		// rest next step instead of retracting them (:4472-4477).
		b.BatchIsFull = false
		return nil
	}
	if retract {
		retracted, ratio := b.RetractDecode(s)
		s.numRetracted = len(retracted)
		for _, r := range retracted {
			s.preemptedSinceTake = append(s.preemptedSinceTake, r.RID())
		}
		s.newTokenRatio.current = ratio
		for _, r := range retracted {
			s.addRequestToQueue(r)
		}
	} else {
		s.newTokenRatio.current = max(s.newTokenRatio.current-s.cfg.NewTokenRatio.Decay, s.cfg.NewTokenRatio.Min)
	}
	if b.Size() < initial {
		b.BatchIsFull = false
	}
	if b.IsEmpty() {
		return b
	}
	b.PrepareForDecode(s)
	return b
}

// Forward is what one forward pass was asked to run (the oracle's batch_record).
type Forward struct {
	Mode            Mode     `json:"mode"`
	BatchSize       int      `json:"batch_size"`
	SeqLens         []int    `json:"seq_lens"`
	ExtendSeqLens   []int    `json:"extend_seq_lens,omitempty"`
	ExtendPrefixLen []int    `json:"extend_prefix_lens,omitempty"`
	RIDs            []string `json:"rids"`
	ReqPoolIndices  []int    `json:"req_pool_indices"`
	Chunked         *string  `json:"chunked"`
	Decoding        []string `json:"decoding"`
	AvailableTokens int      `json:"available_tokens"`
}

// RunningReq is a running request as the state records it.
type RunningReq struct {
	RID         string `json:"rid"`
	OutputLen   int    `json:"output_len"`
	KVCommitted int    `json:"kv_committed"`
	KVAllocated int    `json:"kv_allocated"`
	PrefixLen   int    `json:"prefix_len"`
}

// State is what the scheduler holds between two loop iterations.
type State struct {
	Waiting         []string     `json:"waiting"`
	Running         []RunningReq `json:"running"`
	BatchIsFull     bool         `json:"batch_is_full"`
	Chunked         *string      `json:"chunked"`
	AvailableTokens int          `json:"available_tokens"`
	Evictable       int          `json:"evictable"`
	Protected       int          `json:"protected"`
	ReqPoolAvail    int          `json:"req_pool_available"`
	NewTokenRatio   float64      `json:"new_token_ratio"`
	ForwardCt       int          `json:"forward_ct"`
	ProcessedTokens int          `json:"processed_tokens"`
}

// StepRecord is one loop iteration.
type StepRecord struct {
	Forwards []Forward `json:"forwards"`
	Finished []string  `json:"finished"`
	After    State     `json:"after"`
	// Reports are the calls the metrics reporter received this step, in order (EnableMetrics).
	Reports []Report `json:"reports,omitempty"`
}

func ridOrNil(r *Req) *string {
	if r == nil {
		return nil
	}
	id := r.RID()
	return &id
}

// runBatch is run_batch (:4614-4900) with the model replaced: it counts the pass, and every
// request's next token is the scenario's.
func (s *Scheduler) runBatch(b *Batch) Forward {
	s.forwardCt++
	if b.ExtendNumTokens > 0 {
		s.processedTokens += b.ExtendNumTokens
	}
	f := Forward{Mode: b.Mode, BatchSize: b.Size(), SeqLens: slices.Clone(b.SeqLens),
		ReqPoolIndices: slices.Clone(b.ReqPoolIndices), Chunked: ridOrNil(b.ChunkedReq),
		AvailableTokens: s.alloc.AvailableSize()}
	for _, r := range b.Reqs {
		f.RIDs = append(f.RIDs, r.RID())
	}
	if b.Mode == Extend || b.Mode == Mixed {
		f.ExtendSeqLens = slices.Clone(b.ExtendLens)
		f.ExtendPrefixLen = slices.Clone(b.PrefixLens)
	}
	if b.DecodingReqs != nil {
		f.Decoding = []string{}
		for _, r := range b.DecodingReqs {
			f.Decoding = append(f.Decoding, r.RID())
		}
	}
	return f
}

// processBatchResult is process_batch_result (:5074-5110): a decode pass or an extend pass.
func (s *Scheduler) processBatchResult(b *Batch) {
	if b.Mode == Decode {
		s.processDecode(b)
	} else {
		s.processPrefill(b)
	}
}

func (s *Scheduler) noteFinished(r *Req) { s.finishedThisStep = append(s.finishedThisStep, r.RID()) }

// processPrefill is process_batch_result_prefill (batch_result_processor.py:258-515).
func (s *Scheduler) processPrefill(b *Batch) {
	s.alloc.FreeGroupBegin()
	for _, r := range b.Reqs {
		if (r.Finished() && r.InflightMiddleChunks <= 0) || r.IsRetracted {
			continue
		}
		if r.InflightMiddleChunks <= 0 {
			r.C.OutputIDs = append(r.C.OutputIDs, s.tokenFor(r))
			r.UpdateFinishState(s.cfg.EOSTokenIDs)
			if r.Finished() {
				s.noteFinished(r)
				s.releaseKVCache(r, true)
			} else if b.DecodingReqs == nil || !slices.Contains(b.DecodingReqs, r) {
				s.checkpointKVCache(r)
			}
		} else {
			r.InflightMiddleChunks--
		}
	}
	s.alloc.FreeGroupEnd()
	if s.cfg.EnableMetrics || s.cfg.EnableKVEvents {
		s.reportPrefill(b)
	}
}

// processDecode is process_batch_result_decode (batch_result_processor.py:952-1109).
func (s *Scheduler) processDecode(b *Batch) {
	s.alloc.FreeGroupBegin()
	for _, r := range b.Reqs {
		if r.Finished() || r.IsRetracted {
			// A row that finished (or was retracted) while this pass ran reappears for one tick;
			// release_kv_cache already freed its over-allocated tokens (:1022-1027). Without
			// overlap only an external end (Finish) can make a row of the pass finished.
			continue
		}
		r.C.OutputIDs = append(r.C.OutputIDs, s.tokenFor(r))
		r.UpdateFinishState(s.cfg.EOSTokenIDs)
		if r.Finished() {
			s.noteFinished(r)
			s.releaseKVCache(r, true)
		}
	}
	s.alloc.FreeGroupEnd()
	if s.cfg.EnableMetrics || s.cfg.EnableKVEvents {
		s.reportDecode(b)
	}
}

// onIdle is on_idle (:5241-5335) when the scheduler is fully idle: the events are published and
// new_token_ratio resets.
func (s *Scheduler) onIdle() {
	s.publishKVEvents(nil)
	s.newTokenRatio.current = s.cfg.NewTokenRatio.Init
}

// Step runs one iteration of event_loop_normal (scheduler.py:2026-2059), or of event_loop_overlap
// (:2061-2134) when overlap scheduling is on, with the given arrivals.
// A refused feature is returned as an error wrapping ErrRefused.
func (s *Scheduler) Step(arrivals []Arrival) (rec StepRecord, err error) {
	defer func() {
		if p := recover(); p != nil {
			if rf, ok := p.(refusal); ok {
				err = fmt.Errorf("%w: %s", ErrRefused, rf.Error())
				return
			}
			panic(p)
		}
	}()
	s.finishedThisStep = nil
	s.reports = nil
	for _, a := range arrivals {
		s.handleGenerateRequest(a)
	}
	var batch *Batch
	if s.cfg.EnableOverlap {
		batch = s.overlapIteration(&rec)
	} else {
		batch = s.getNextBatchToRun()
		if batch != nil {
			rec.Forwards = append(rec.Forwards, s.runBatch(batch))
			s.processBatchResult(batch)
		} else if s.isFullyIdle() {
			s.onIdle()
		}
	}
	s.last = batch
	rec.Finished = s.finishedThisStep
	rec.Reports = s.reports
	rec.After = s.State()
	s.step++
	return rec, nil
}

// State reads what the scheduler holds now.
func (s *Scheduler) State() State {
	st := State{BatchIsFull: s.running.BatchIsFull, Chunked: ridOrNil(s.chunkedReq),
		AvailableTokens: s.alloc.AvailableSize(), Evictable: s.tree.EvictableSize(),
		Protected: s.tree.ProtectedSize(), ReqPoolAvail: s.pool.AvailableSize(),
		NewTokenRatio: s.newTokenRatio.current, ForwardCt: s.forwardCt, ProcessedTokens: s.processedTokens,
		Waiting: []string{}, Running: []RunningReq{}}
	for _, r := range s.waiting {
		st.Waiting = append(st.Waiting, r.RID())
	}
	for _, r := range s.running.Reqs {
		st.Running = append(st.Running, RunningReq{r.RID(), len(r.C.OutputIDs), r.C.KV.KVCommittedLen,
			r.C.KV.KVAllocatedLen, r.C.PrefixLen})
	}
	return st
}
