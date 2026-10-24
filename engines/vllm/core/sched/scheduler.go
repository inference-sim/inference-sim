// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/sched/scheduler.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/core/sched/request_queue.py and vllm/v1/core/sched/utils.py (check_stop).

// Package sched is vLLM's v1 Scheduler: schedule() and update_from_output() for a
// full-attention model, with chunked prefill, the long-prefill cap, FCFS and priority
// policies, recompute preemption, prefix caching, speculative decoding (acceptance supplied)
// and the max_loras batch limit. Synchronous scheduling (one batch in flight); connectors
// and async scheduling are separate.
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
}

// Status is RequestStatus (vllm/v1/request.py:370).
type Status int

const (
	Waiting Status = iota
	Running
	Preempted
	FinishedStopped
	FinishedLength
	FinishedAborted
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

	Status         Status
	Output         []int64
	NumComputed    int
	NumPreemptions int
	IsPrefillChunk bool
	// SpecTokens are spec_token_ids: the drafts proposed for the next step.
	SpecTokens  []int64
	hashes      []kv.BlockHash
	numInFlight int
	seq         int // FIFO tie-break within a priority queue
}

// NumTokens is len(all_token_ids): prompt plus output.
func (r *Request) NumTokens() int { return len(r.Prompt) + len(r.Output) }

// numTokensWithSpec is num_tokens_with_spec (vllm/v1/request.py:298-300).
func (r *Request) numTokensWithSpec() int { return r.NumTokens() + len(r.SpecTokens) }

func (r *Request) allTokens() []int64 {
	return append(append(make([]int64, 0, r.NumTokens()), r.Prompt...), r.Output...)
}

// updateHashes appends the hashes of newly full blocks (update_block_hashes).
func (r *Request) updateHashes(blockSize int) {
	all := r.allTokens()
	for (len(r.hashes)+1)*blockSize <= len(all) {
		i := len(r.hashes)
		var parent kv.BlockHash
		if i > 0 {
			parent = r.hashes[i-1]
		}
		var extra []kv.ExtraKey
		extra = append(extra, r.Keys.Extra...)
		if i == 0 && r.Keys.CacheSalt != "" {
			extra = append(extra, kv.ExtraKey{"cache_salt", r.Keys.CacheSalt})
		}
		r.hashes = append(r.hashes, kv.HashBlock(parent, all[i*blockSize:(i+1)*blockSize], extra))
	}
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
	seq      int
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
	return &Scheduler{
		cfg: cfg,
		KV: kv.NewManagerWith(cfg.NumBlocks, cfg.BlockSize, cfg.MaxModelLen, cfg.EnablePrefixCaching, wm,
			kv.Options{EagleBlockDrop: cfg.Spec.EagleBlockDrop, PrefillLookahead: cfg.Spec.PrefillLookahead}),
		requests: map[string]*Request{},
		waiting:  queue{priority: cfg.Policy == Priority},
	}, nil
}

// Add is add_request.
func (s *Scheduler) Add(r *Request) error {
	if _, dup := s.requests[r.ID]; dup {
		return fmt.Errorf("sched: request %s added twice", r.ID)
	}
	r.Status = Waiting
	s.seq++
	r.seq = s.seq
	r.updateHashes(s.cfg.BlockSize)
	s.requests[r.ID] = r
	s.waiting.add(r)
	return nil
}

// Counts is get_request_counts: (running, waiting).
func (s *Scheduler) Counts() (int, int) { return len(s.running), s.waiting.len() }

// HasUnfinished is has_unfinished_requests.
func (s *Scheduler) HasUnfinished() bool { return len(s.running)+s.waiting.len() > 0 }

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

// Schedule is schedule() (scheduler.py:591-1558) for this configuration.
func (s *Scheduler) Schedule() Output {
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
		numNew := r.numTokensWithSpec() - r.NumComputed
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
				Hashes: r.hashes, NumTokens: r.NumTokens(), HasScheduled: true}) {
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
			if n := numNew + r.NumComputed - r.NumTokens(); n > 0 {
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

	// Waiting requests, only when nothing was preempted (:899-1395). A request skipped this
	// step is set aside and put back at the front afterwards, in its order.
	var admitted []*Request
	if len(preempted) == 0 {
		var skipped []*Request
		for tokenBudget > 0 {
			if s.waiting.len() == 0 || inputBudget <= draftSlots {
				break
			}
			if len(s.running) >= s.maxActive() {
				break
			}
			r := s.waiting.peek()
			// max_loras (:939-951): a request whose adapter would make one too many in this
			// batch is skipped this step, and those behind it still run.
			if s.cfg.MaxLoRAs > 0 && r.LoRA > 0 && len(scheduledLoRAs) == s.cfg.MaxLoRAs && !scheduledLoRAs[r.LoRA] {
				skipped = append(skipped, s.waiting.pop())
				continue
			}
			var hit []*kv.Block
			numComputed := r.NumComputed
			lookedUp := false
			if r.NumComputed == 0 {
				lookedUp = true
				hit, numComputed = s.KV.ComputedBlocks(r.hashes, r.NumTokens())
			}
			room := min(tokenBudget, inputBudget-draftSlots)
			numNew := r.NumTokens() - numComputed
			// First-decode padding (:1130-1152): a request whose first step is a single decode
			// token runs as 1 + K, so the batch keeps the speculative decodes' shape.
			pad := false
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
			if !s.KV.Allocate(r.ID, kv.Alloc{New: numNew, Lookahead: s.cfg.Spec.Lookahead, NewComputed: hit,
				Hashes: r.hashes, NumTokens: r.NumTokens(), Waiting: true, HasScheduled: len(s.running) > 0,
				FullSequenceMustFit: s.cfg.ReserveFullSequence}) {
				break
			}
			s.waiting.pop()
			s.running = append(s.running, r)
			admitted = append(admitted, r)
			if s.cfg.MaxLoRAs > 0 && r.LoRA > 0 {
				scheduledLoRAs[r.LoRA] = true
			}
			out.Scheduled[r.ID] = numNew
			if lookedUp {
				out.PrefixHit[r.ID] = numComputed
			}
			tokenBudget -= numNew
			inputBudget -= numNew + draftSlots
			r.Status = Running
			r.NumComputed = numComputed
			if pad {
				drafts := make([]int64, k)
				for i := range drafts {
					drafts[i] = -1
				}
				out.SpecDecode[r.ID] = drafts
			}
		}
		// prepend_requests (:1391-1395): extendleft on a deque reverses, which restores the
		// skipped requests' order at the front; a priority queue re-sorts.
		for i := len(skipped) - 1; i >= 0; i-- {
			s.waiting.prepend(skipped[i])
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
		r.IsPrefillChunk = r.NumComputed < r.NumTokens()
	}
	return out
}

// preempt is _preempt_request (:1571-1614) for recompute preemption.
func (s *Scheduler) preempt(r *Request) {
	s.KV.Free(r.ID)
	r.Status = Preempted
	r.NumComputed = 0
	r.SpecTokens = nil
	r.NumPreemptions++
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
	var finished []Finished
	var stopped []*Request
	for _, id := range out.Order {
		r := s.requests[id]
		if r == nil || r.Status > Preempted {
			continue
		}
		r.numInFlight -= out.Scheduled[id]
		toks := sampled[id]
		if drafts := out.SpecDecode[id]; len(drafts) > 0 && len(toks) > 0 {
			accepted := max(len(toks)-1, 0)
			if r.NumComputed > 0 {
				r.NumComputed -= len(drafts) - accepted
			}
		}
		if len(toks) == 0 {
			continue
		}
		var status Status
		stop := false
		for _, tok := range toks {
			r.Output = append(r.Output, tok)
			r.updateHashes(s.cfg.BlockSize)
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
		if !stop {
			continue
		}
		r.Status = status
		r.SpecTokens = nil
		s.KV.Free(r.ID)
		stopped = append(stopped, r)
		finished = append(finished, Finished{ID: r.ID, Status: status, Output: len(r.Output)})
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
			delete(s.requests, r.ID)
		}
	}
	return finished
}

// UpdateDrafts is update_draft_token_ids (:2550-2569): the drafter's proposals after a step,
// for each request it ran. A request still prefilling takes none.
func (s *Scheduler) UpdateDrafts(drafts map[string][]int64) {
	for id, d := range drafts {
		r := s.requests[id]
		if r == nil || r.Status > Preempted {
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
