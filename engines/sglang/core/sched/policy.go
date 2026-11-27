// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/managers/schedule_policy.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).

package sched

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/inference-sim/inference-sim/engines/sglang/core/cache"
)

// Refusal is the open issue every refused scheduling feature cites (X28).
const Refusal = "inference-sim/inference-sim#2346"

// Policy is a built-in schedule policy (schedule_policy.py:172-188@297e54fd).
type Policy string

const (
	FCFS                 Policy = "fcfs"
	LPM                  Policy = "lpm"
	HRRN                 Policy = "hrrn"
	ShortestPrefillFirst Policy = "shortest-prefill-first"
	LOF                  Policy = "lof"
	Random               Policy = "random"
	RoutingKey           Policy = "routing-key"
	DFSWeight            Policy = "dfs-weight"
)

// waitingQueuePrefixMatchMax bounds the waiting queue lpm and hrrn match (:127).
const waitingQueuePrefixMatchMax = 128

// validatePolicy is _validate_and_adjust_policy (:324-344): a disabled tree cache turns every
// cache-aware policy into fcfs, except that shortest-prefill-first is refused there. A name
// SGLang does not know is refused as SGLang refuses it. The ValueError that
// shortest-prefill-first raises without a prefix cache (:332-336) is caught by the except at :340,
// and the retry as a cache-agnostic policy fails, so the refusal a user sees is the one at :344
// (ValueError: Unknown schedule_policy: policy='shortest-prefill-first').
func validatePolicy(name string, treeDisabled bool) (Policy, error) {
	switch Policy(name) {
	case LPM, HRRN, DFSWeight:
		if treeDisabled {
			return FCFS, nil
		}
		return Policy(name), nil
	case ShortestPrefillFirst:
		if treeDisabled {
			return "", unknownPolicy(name)
		}
		return ShortestPrefillFirst, nil
	case FCFS, LOF, Random, RoutingKey:
		return Policy(name), nil
	}
	return "", unknownPolicy(name)
}

// unknownPolicy is the ValueError at schedule_policy.py:344@297e54fd, `f"Unknown schedule_policy:
// {policy=}"`, with Python's repr of the name.
func unknownPolicy(name string) error {
	quote := "'"
	if strings.Contains(name, "'") && !strings.Contains(name, `"`) {
		quote = `"`
	}
	return fmt.Errorf("sched: SGLang refuses this configuration: Unknown schedule_policy: policy=%s%s%s", quote, name, quote)
}

func (p Policy) cacheAware() bool {
	return p == LPM || p == HRRN || p == ShortestPrefillFirst || p == DFSWeight
}

// simTree is the waiting-queue radix tree (RadixCache.create_simulated, radix_cache.py:320-339,
// page size 1): a match is the longest prefix some inserted key shares with the lookup.
type simTree struct{ keys [][]int64 }

func (t *simTree) reset() { t.keys = nil }

func (t *simTree) match(ids []int64) int {
	best := 0
	for _, k := range t.keys {
		n := 0
		for n < len(k) && n < len(ids) && k[n] == ids[n] {
			n++
		}
		best = max(best, n)
	}
	return best
}

func (t *simTree) insert(ids []int64) { t.keys = append(t.keys, ids) }

// calcPriority is SchedulePolicy.calc_priority (:213-280): it matches the queue against the
// tree and sorts it in place (a stable sort, as Python's). running is the running batch, which the
// routing-key policy counts keys in.
func (s *Scheduler) calcPriority(queue []*Req, running *Batch) {
	if s.calcBoundedPriority(queue) {
		return
	}
	policy := s.policy
	if policy == LPM || policy == HRRN {
		if len(queue) > waitingQueuePrefixMatchMax {
			// Turn off the expensive prefix matching and sorting when the queue is large.
			policy = FCFS
		}
	}
	if !policy.cacheAware() && s.tree.SupportsFastMatchPrefix() {
		for _, r := range queue {
			s.matchKVCache(r, r.allIDs(), nil)
		}
	}
	switch {
	case s.policy == FCFS:
		// The test is on the configured policy, not the active one: a queue too large for lpm runs
		// as fcfs without the priority sort.
		if s.cfg.EnablePriorityScheduling {
			s.sortByPriorityAndFCFS(queue)
		}
	case policy.cacheAware():
		deprioritized := s.computePrefixMatches(queue)
		switch policy {
		case LPM:
			sort.SliceStable(queue, func(i, j int) bool {
				return lpmKey(queue[i], deprioritized) < lpmKey(queue[j], deprioritized)
			})
		case DFSWeight:
			// _sort_by_dfs_weight (:493-499): the tree orders the requests by the nodes they matched.
			handles := make([]cache.NodeID, len(queue))
			for i, r := range queue {
				handles[i] = *r.C.LastNode
			}
			order := s.tree.DFSWeightOrder(handles)
			sorted := make([]*Req, len(order))
			for i, idx := range order {
				sorted[i] = queue[idx]
			}
			copy(queue, sorted)
		case HRRN:
			processed := s.processedTokens
			sort.SliceStable(queue, func(i, j int) bool {
				ki, kj := hrrnKey(queue[i], deprioritized, processed), hrrnKey(queue[j], deprioritized, processed)
				if ki.rank != kj.rank || ki.ratio != kj.ratio {
					return ki.rank < kj.rank || (ki.rank == kj.rank && ki.ratio < kj.ratio)
				}
				return queue[i].RID() < queue[j].RID()
			})
		case ShortestPrefillFirst:
			sort.SliceStable(queue, func(i, j int) bool {
				a, b := queue[i], queue[j]
				da, db := deprioritized[a.RID()], deprioritized[b.RID()]
				if da != db {
					return !da
				}
				wa, wb := shortestPrefillWork(a), shortestPrefillWork(b)
				if wa != wb {
					return wa < wb
				}
				return a.WaitQueueEntryStep < b.WaitQueueEntryStep
			})
		}
	case policy == LOF:
		sort.SliceStable(queue, func(i, j int) bool {
			a, b := queue[i], queue[j]
			if s.cfg.EnablePriorityScheduling && a.Priority != b.Priority {
				return s.priorityBefore(a.Priority, b.Priority)
			}
			return a.MaxNewTokens > b.MaxNewTokens
		})
	case policy == Random:
		s.shuffleQueue(queue)
	case policy == RoutingKey:
		if running != nil {
			sortByRoutingKey(queue, running)
		}
	}
	s.touchWaitingPrefixes(policy, queue)
}

// waitingPrefixRefreshInterval is WAITING_PREFIX_REFRESH_INTERVAL (:130-132).
const waitingPrefixRefreshInterval = 32

// touchWaitingPrefixes is _touch_waiting_prefixes (:282-306): a policy that does not match the
// queue keeps the waiting requests' prefixes from being evicted first, by touching them in the
// tree on every 32nd pass, the head of the queue last so least-recently-used eviction goes in
// reverse admission order. SGLANG_ENABLE_WAITING_PREFIX_REFRESH is on by default and the oracle
// leaves it so.
func (s *Scheduler) touchWaitingPrefixes(policy Policy, queue []*Req) {
	if policy.cacheAware() || len(queue) == 0 || !strings.EqualFold(s.cfg.radixEvictionPolicy(), "lru") ||
		s.tree.SupportsFastMatchPrefix() || !s.tree.SupportsPrefixSharing() {
		return
	}
	refresh := s.waitingPrefixRefreshCalls%waitingPrefixRefreshInterval == 0
	s.waitingPrefixRefreshCalls++
	if !refresh {
		return
	}
	head := queue[:min(len(queue), waitingQueuePrefixMatchMax)]
	for i := len(head) - 1; i >= 0; i-- {
		r := head[i]
		s.tree.TouchPrefix(cache.Key{Tokens: r.allIDs(), ExtraKey: r.C.ExtraKey, CacheSalt: r.C.CacheSalt})
	}
}

// prioritySign is priority_sign (:204): 1 when low values go first, else -1.
func (s *Scheduler) prioritySign() int {
	if s.cfg.ScheduleLowPriorityValuesFirst {
		return 1
	}
	return -1
}

// priorityBefore says whether priority a sorts before b, that is whether a*priority_sign < b*priority_sign,
// without the overflow the sentinel priorities (the extremes of a 64-bit int) would cause.
func (s *Scheduler) priorityBefore(a, b int) bool {
	if s.prioritySign() > 0 {
		return a < b
	}
	return a > b
}

// sortByPriorityAndFCFS is _sort_by_priority_and_fcfs (:538-547): priority, then arrival.
func (s *Scheduler) sortByPriorityAndFCFS(queue []*Req) {
	sort.SliceStable(queue, func(i, j int) bool {
		a, b := queue[i], queue[j]
		if a.Priority != b.Priority {
			return s.priorityBefore(a.Priority, b.Priority)
		}
		return a.WaitQueueEntryStep < b.WaitQueueEntryStep
	})
}

// shuffleQueue is _sort_randomly (:534-536), the order drawn from the scheduler's named stream
// unless a Shuffle function supplies it.
func (s *Scheduler) shuffleQueue(queue []*Req) {
	if s.Shuffle == nil {
		s.stream.Shuffle(len(queue), func(i, j int) { queue[i], queue[j] = queue[j], queue[i] })
		return
	}
	rids := make([]string, len(queue))
	byRID := make(map[string]*Req, len(queue))
	for i, r := range queue {
		rids[i] = r.RID()
		byRID[r.RID()] = r
	}
	order := s.Shuffle(rids)
	if len(order) != len(queue) {
		panic(fmt.Sprintf("sched: Shuffle returned %d requests for a queue of %d", len(order), len(queue)))
	}
	seen := map[string]bool{}
	for i, rid := range order {
		r, ok := byRID[rid]
		if !ok || seen[rid] {
			panic(fmt.Sprintf("sched: Shuffle returned %q, which is not a distinct member of the queue", rid))
		}
		seen[rid] = true
		queue[i] = r
	}
}

// sortByRoutingKey is _sort_by_routing_key (:549-583): requests whose routing key a running request
// holds go first, the most frequent key first and keys alphabetically among equals; the rest follow
// in key order. With no keyed request running the queue is left as it is.
func sortByRoutingKey(queue []*Req, running *Batch) {
	counts := map[string]int{}
	for _, r := range running.Reqs {
		if r.RoutingKey != "" {
			counts[r.RoutingKey]++
		}
	}
	if len(counts) == 0 {
		return
	}
	sort.SliceStable(queue, func(i, j int) bool {
		a, b := queue[i], queue[j]
		ca, ina := counts[a.RoutingKey]
		cb, inb := counts[b.RoutingKey]
		ina, inb = ina && a.RoutingKey != "", inb && b.RoutingKey != ""
		switch {
		case ina != inb:
			return ina
		case ina:
			if ca != cb {
				return ca > cb
			}
		}
		return a.RoutingKey < b.RoutingKey
	})
}

func (r *Req) allIDs() []int64 {
	return append(append([]int64{}, r.C.OriginInputIDs...), r.C.OutputIDs...)
}

// lpmKey is _sort_by_longest_prefix's key (:441-451): the longest match first, a request
// deprioritized by the in-batch prefix check last.
func lpmKey(r *Req, deprioritized map[string]bool) float64 {
	if deprioritized[r.RID()] {
		return math.Inf(1)
	}
	return -float64(r.NumMatchedPrefixTokens)
}

type hrrn struct {
	rank  int // -1: no prefill work, 0: ratio, 1: deprioritized
	ratio float64
}

// hrrnKey is _sort_by_hrrn's key (:460-494): the largest waited-over-uncached ratio first.
func hrrnKey(r *Req, deprioritized map[string]bool, processed int) hrrn {
	if deprioritized[r.RID()] {
		return hrrn{rank: 1}
	}
	uncached := max(0, len(r.C.OriginInputIDs)-r.NumMatchedPrefixTokens)
	if uncached <= 0 {
		return hrrn{rank: -1}
	}
	waited := max(0, processed-r.ArrivalProcessedTokens)
	return hrrn{rank: 0, ratio: -float64(waited) / float64(uncached)}
}

// shortestPrefillWork is _shortest_prefill_work (:399-404).
func shortestPrefillWork(r *Req) int {
	return max(1, len(r.C.OriginInputIDs)+len(r.C.OutputIDs)-r.NumMatchedPrefixTokens)
}

// shortestPrefillChunkLimit is shortest_prefill_chunk_limit (:418-439): the active chunk is capped
// so shorter waiting requests keep some of the budget.
func (s *Scheduler) shortestPrefillChunkLimit(chunked *Req, queue []*Req, budget int) *int {
	ps := s.cfg.PageSize
	if s.policy != ShortestPrefillFirst || budget < 2*ps {
		return nil
	}
	remaining := chunked.InputLen() - chunked.C.PrefixLen
	reserved := 0
	for _, r := range queue {
		work := shortestPrefillWork(r)
		charge := ceilTo(work, ps)
		if work >= remaining || reserved+charge > budget-ps {
			break
		}
		reserved += charge
	}
	if reserved == 0 {
		return nil
	}
	v := (budget - reserved) / ps * ps
	return &v
}

// computePrefixMatches is _compute_prefix_matches (:589-651@c4af57ff): the match against the tree
// for each request, and the in-batch prefix check that defers a request whose prefix a queued
// one shares. Of the requests with almost nothing cached that share a prefix only one runs, so
// the shared prefix is computed once: under shortest-prefill-first the checked requests are
// visited shortest uncached work first, so the leader that runs is the shortest of its group and
// not the first of it in queue order. Only the first InBatchPrefixDeprioritizeThreshold tokens
// can decide a match, so the check keys on them.
func (s *Scheduler) computePrefixMatches(queue []*Req) map[string]bool {
	deprioritized := map[string]bool{}
	s.waitTree.reset()
	type candidate struct {
		r   *Req
		key []int64
	}
	var candidates []candidate
	for _, r := range queue {
		ids := r.allIDs()
		s.matchKVCache(r, ids, nil)
		if r.C.PrefixLen <= s.cfg.InBatchPrefixCheckThreshold {
			candidates = append(candidates, candidate{r, ids[:pySliceEnd(len(ids), s.cfg.InBatchPrefixDeprioritizeThreshold)]})
		}
	}
	if s.policy == ShortestPrefillFirst {
		sort.SliceStable(candidates, func(i, j int) bool {
			return shortestPrefillWork(candidates[i].r) < shortestPrefillWork(candidates[j].r)
		})
	}
	for _, c := range candidates {
		if s.waitTree.match(c.key) >= s.cfg.InBatchPrefixDeprioritizeThreshold {
			deprioritized[c.r.RID()] = true
		} else {
			s.waitTree.insert(c.key)
		}
	}
	return deprioritized
}

// pySliceEnd is the end index of Python's seq[:n] for a sequence of length size.
func pySliceEnd(size, n int) int {
	if n < 0 {
		return max(size+n, 0)
	}
	return min(size, n)
}

var _ = cache.FullComponentType

// ResolvePolicy is the policy SGLang runs for a configured name once it knows whether the tree
// cache is disabled (_validate_and_adjust_policy, schedule_policy.py:324-344@297e54fd), refused
// as SGLang refuses it. The engine's factory calls it so the settings it reports are the ones the
// scheduler runs.
func ResolvePolicy(name string, treeDisabled bool) (string, error) {
	p, err := validatePolicy(name, treeDisabled)
	return string(p), err
}
