// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/sched/scheduler.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/core/sched/request_queue.py and vllm/v1/core/sched/utils.py (check_stop).

// Package sched is vLLM's v1 Scheduler: schedule() and update_from_output() for a
// full-attention model, with chunked prefill, the long-prefill cap, FCFS and priority
// policies, recompute preemption and prefix caching. Synchronous scheduling (one batch in
// flight); speculative decoding, connectors and async scheduling are separate.
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

	Status         Status
	Output         []int64
	NumComputed    int
	NumPreemptions int
	IsPrefillChunk bool
	hashes         []kv.BlockHash
	numInFlight    int
	seq            int // FIFO tie-break within a priority queue
}

// NumTokens is len(all_token_ids): prompt plus output.
func (r *Request) NumTokens() int { return len(r.Prompt) + len(r.Output) }

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
	wm := int(cfg.Watermark * float64(cfg.NumBlocks))
	return &Scheduler{
		cfg:      cfg,
		KV:       kv.NewManager(cfg.NumBlocks, cfg.BlockSize, cfg.MaxModelLen, cfg.EnablePrefixCaching, wm),
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

// Schedule is schedule() (scheduler.py:591-1558) for this configuration.
func (s *Scheduler) Schedule() Output {
	out := Output{Scheduled: map[string]int{}, PrefixHit: map[string]int{}}
	tokenBudget := s.maxScheduled()
	inputBudget := s.cfg.MaxNumBatchedTokens

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
		if inputBudget <= 0 {
			break
		}
		numNew := r.NumTokens() - r.NumComputed
		if 0 < threshold && threshold < numNew {
			numNew = threshold
		}
		numNew = min(numNew, tokenBudget, inputBudget)
		// The input position must stay within max_model_len (:713-718).
		numNew = min(numNew, s.cfg.MaxModelLen-r.NumComputed-1)
		if numNew <= 0 {
			reqIndex++
			continue
		}
		ok := false
		for {
			if s.KV.AllocateSlots(r.ID, r.NumComputed, numNew, nil, r.hashes, r.NumTokens(), false, true, false) {
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
					tokenBudget += restored
					inputBudget += restored
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
		out.Scheduled[r.ID] = numNew
		tokenBudget -= numNew
		inputBudget -= numNew
		reqIndex++
	}

	// Waiting requests, only when nothing was preempted (:899-1386).
	var admitted []*Request
	if len(preempted) == 0 {
		for tokenBudget > 0 && s.waiting.len() > 0 && inputBudget > 0 {
			if len(s.running) >= s.maxActive() {
				break
			}
			r := s.waiting.peek()
			var hit []*kv.Block
			numComputed := r.NumComputed
			lookedUp := false
			if r.NumComputed == 0 {
				lookedUp = true
				hit, numComputed = s.KV.ComputedBlocks(r.hashes, r.NumTokens())
			}
			numNew := r.NumTokens() - numComputed
			if 0 < threshold && threshold < numNew {
				numNew = threshold
			}
			room := min(tokenBudget, inputBudget)
			if !s.cfg.EnableChunkedPrefill && numNew > room {
				break
			}
			numNew = min(numNew, room)
			if numNew <= 0 {
				break
			}
			if !s.KV.AllocateSlots(r.ID, 0, numNew, hit, r.hashes, r.NumTokens(), true, len(s.running) > 0, s.cfg.ReserveFullSequence) {
				break
			}
			s.waiting.pop()
			s.running = append(s.running, r)
			admitted = append(admitted, r)
			out.Scheduled[r.ID] = numNew
			if lookedUp {
				out.PrefixHit[r.ID] = numComputed
			}
			tokenBudget -= numNew
			inputBudget -= numNew
			r.Status = Running
			r.NumComputed = numComputed
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
	r.NumPreemptions++
	s.waiting.prepend(r)
}

// Update is update_from_output (:2026-2425): each scheduled request that completed its
// known tokens samples one token from sampled (nil means it sampled nothing); a request
// stops on EOS, at max_tokens, or at max_model_len (check_stop, sched/utils.py:98-140).
//
// Requests are visited in num_scheduled_tokens' insertion order (:2075), which is out.Order:
// the scheduled running requests in running order, then the admitted ones. The order is not
// cosmetic: a request that stops frees its blocks there and then (_free_request), so it
// decides which freed prefix blocks sit nearer the free queue's head and are evicted first.
func (s *Scheduler) Update(out Output, sampled map[string]int64) []Finished {
	var finished []Finished
	var stopped []*Request
	for _, id := range out.Order {
		r := s.requests[id]
		if r == nil || r.Status > Preempted {
			continue
		}
		r.numInFlight -= out.Scheduled[id]
		tok, ok := sampled[id]
		if !ok {
			continue
		}
		r.Output = append(r.Output, tok)
		r.updateHashes(s.cfg.BlockSize)
		var status Status
		switch {
		case tok == r.EOS:
			status = FinishedStopped
		case r.NumTokens() >= s.cfg.MaxModelLen || len(r.Output) >= r.MaxTokens:
			status = FinishedLength
		default:
			continue
		}
		r.Status = status
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
