// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/managers/schedule_policy.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package sched

import (
	"fmt"
	"math"
	"sort"

	"github.com/inference-sim/blis-sglang/core/cache"
)

// Refusal is the open issue every refused scheduling feature cites (X28).
const Refusal = "inference-sim/blis-sglang#56"

// Policy is a built-in schedule policy (schedule_policy.py:172-188@297e54fd).
type Policy string

const (
	FCFS                 Policy = "fcfs"
	LPM                  Policy = "lpm"
	HRRN                 Policy = "hrrn"
	ShortestPrefillFirst Policy = "shortest-prefill-first"
	LOF                  Policy = "lof"
)

// waitingQueuePrefixMatchMax bounds the waiting queue lpm and hrrn match (:127).
const waitingQueuePrefixMatchMax = 128

// validatePolicy is _validate_and_adjust_policy (:324-344): a disabled tree cache turns every
// cache-aware policy into fcfs, except that shortest-prefill-first is refused there. The
// policies SGLang has and the port does not are refused by name.
func validatePolicy(name string, treeDisabled bool) (Policy, error) {
	switch Policy(name) {
	case LPM, HRRN:
		if treeDisabled {
			return FCFS, nil
		}
		return Policy(name), nil
	case ShortestPrefillFirst:
		if treeDisabled {
			return "", fmt.Errorf("sched: SGLang refuses shortest-prefill-first without prefix caching (schedule_policy.py:332-336@297e54fd)")
		}
		return ShortestPrefillFirst, nil
	case FCFS, LOF:
		return Policy(name), nil
	case "dfs-weight", "random", "routing-key":
		return "", fmt.Errorf("sched: the %s schedule policy is not modelled (%s)", name, Refusal)
	}
	return "", fmt.Errorf("sched: unknown schedule_policy %q", name)
}

func (p Policy) cacheAware() bool {
	return p == LPM || p == HRRN || p == ShortestPrefillFirst
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
// tree and sorts it in place (a stable sort, as Python's).
func (s *Scheduler) calcPriority(queue []*Req) {
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
	case policy == FCFS:
	case policy.cacheAware():
		deprioritized := s.computePrefixMatches(queue)
		switch policy {
		case LPM:
			sort.SliceStable(queue, func(i, j int) bool {
				return lpmKey(queue[i], deprioritized) < lpmKey(queue[j], deprioritized)
			})
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
		sort.SliceStable(queue, func(i, j int) bool { return queue[i].MaxNewTokens > queue[j].MaxNewTokens })
	}
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

// computePrefixMatches is _compute_prefix_matches (:346-397): the match against the tree for each
// request, and the in-batch prefix check that defers a request whose prefix a queued one shares.
func (s *Scheduler) computePrefixMatches(queue []*Req) map[string]bool {
	deprioritized := map[string]bool{}
	s.waitTree.reset()
	for _, r := range queue {
		ids := r.allIDs()
		s.matchKVCache(r, ids, nil)
		if r.C.PrefixLen <= s.cfg.InBatchPrefixCheckThreshold {
			if s.waitTree.match(ids) >= s.cfg.InBatchPrefixDeprioritizeThreshold {
				deprioritized[r.RID()] = true
			} else {
				s.waitTree.insert(ids)
			}
		}
	}
	return deprioritized
}

var _ = cache.FullComponentType
