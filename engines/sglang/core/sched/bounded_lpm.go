// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/managers/schedule_policy.py and python/sglang/srt/managers/lpm_opportunity.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).

package sched

import (
	"slices"
	"sort"
)

// Bounded longest-prefix-match. Above waitingQueuePrefixMatchMax waiting requests, lpm used to fall
// back to fcfs. It now may instead run fixed batches (schedule_policy.py:277-413@c4af57ff): on
// every boundedLPMRecheckPasses-th pass with a long queue, the first boundedLPMWindowSize requests
// become a batch when queue-only evidence says that fcfs order would evict a long shared prefix
// before its next reader arrives (competitionOpportunity). The batch is sorted by longest prefix
// and admitted before the rest of the queue, and is finished even if the queue drops below the
// limit. Everything else keeps the arrival order. There is no server argument: it is on for every
// run with lpm (bounded_lpm_enabled defaults to True, scheduler.py:1570).
const (
	// boundedLPMWindowSize is BOUNDED_LPM_WINDOW_SIZE (schedule_policy.py:179).
	boundedLPMWindowSize = 64
	// boundedLPMRecheckPasses is RECHECK_PASSES (lpm_opportunity.py:6): the gate is evaluated
	// on a pass count, never on time, so ranks sharing a queue agree.
	boundedLPMRecheckPasses = 32
	// boundedLPMPrefixTokens is PREFIX_TOKENS (lpm_opportunity.py:5): requests share a prefix for
	// this gate only when their first this many tokens do.
	boundedLPMPrefixTokens = 4096
)

// boundedLPM is the state of the bounded batches (SchedulePolicy._bounded_*).
type boundedLPM struct {
	gateCalls int
	active    bool
	members   []*Req
}

// shouldStartBoundedBatch is _should_start_bounded_batch (:415-449@c4af57ff). Cache modes the
// observation study does not cover (a disabled cache, sliding-window or Mamba state) fail closed;
// the others the port does not model are refused before a scheduler exists.
func (s *Scheduler) shouldStartBoundedBatch(queue []*Req) bool {
	s.bounded.gateCalls++
	if (s.bounded.gateCalls-1)%boundedLPMRecheckPasses != 0 {
		return false
	}
	if s.cfg.Disable || s.tree.SupportsMamba() {
		return false
	}
	capacity := s.alloc.Size()
	if capacity <= 0 {
		return false
	}
	return competitionOpportunity(queue[:min(len(queue), boundedLPMWindowSize)], capacity)
}

// competitionOpportunity is competition_opportunity (lpm_opportunity.py:26-101@c4af57ff), without
// the per-pair report and the elapsed time: whether, among the requests, long-prefix reuse is
// threatened by what arrives between the readers. Count the pairs of consecutive requests that share
// their first boundedLPMPrefixTokens tokens (in one namespace) whose intervening unique input,
// with the shared prefix, exceeds the KV capacity: the prefix would be evicted before its next
// reader. The queue is worth reordering when 2 * (tokens at risk) >= 3 * capacity.
func competitionOpportunity(requests []*Req, capacity int) bool {
	if capacity <= 0 || len(requests) > boundedLPMWindowSize {
		return false
	}
	seqs := make([][]int64, len(requests))
	ns := make([]string, len(requests))
	for i, r := range requests {
		seqs[i] = r.allIDs()
		ns[i] = namespaceOf(r)
	}
	type groupKey struct {
		ns     string
		prefix string
	}
	var order []groupKey
	groups := map[groupKey][]int{}
	for i, t := range seqs {
		if len(t) >= boundedLPMPrefixTokens {
			k := groupKey{ns[i], tokensKey(t[:boundedLPMPrefixTokens])}
			if _, ok := groups[k]; !ok {
				order = append(order, k)
			}
			groups[k] = append(groups[k], i)
		}
	}
	// Cached within this evaluation only: no stale result across queue changes.
	lcpCache := map[[2]int]int{}
	common := func(left, right, known int) int {
		key := [2]int{min(left, right), max(left, right)}
		if v, ok := lcpCache[key]; ok {
			return v
		}
		a, b := seqs[left], seqs[right]
		limit := min(len(a), len(b))
		n := known
		for n < limit && a[n] == b[n] {
			n++
		}
		lcpCache[key] = n
		return n
	}
	risk := 0
	for _, k := range order {
		positions := groups[k]
		for p := 0; p+1 < len(positions); p++ {
			left, right := positions[p], positions[p+1]
			prefix := common(left, right, boundedLPMPrefixTokens)
			intervening := map[string][]int{}
			var namespaces []string
			for i := left + 1; i < right; i++ {
				if _, ok := intervening[ns[i]]; !ok {
					namespaces = append(namespaces, ns[i])
				}
				intervening[ns[i]] = append(intervening[ns[i]], i)
			}
			footprint := 0
			for _, name := range namespaces {
				idx := slices.Clone(intervening[name])
				sort.SliceStable(idx, func(x, y int) bool { return slices.Compare(seqs[idx[x]], seqs[idx[y]]) < 0 })
				footprint += len(seqs[idx[0]])
				for q := 0; q+1 < len(idx); q++ {
					footprint += len(seqs[idx[q+1]]) - common(idx[q], idx[q+1], 0)
				}
			}
			overlap := 0
			for _, v := range intervening[ns[left]] {
				overlap = max(overlap, min(prefix, common(left, v, 0)))
			}
			if footprint+prefix-overlap > capacity {
				risk += prefix
			}
		}
	}
	return 2*risk >= 3*capacity
}

// namespaceOf is a request's (extra_key, cache_salt) as one comparable value; two requests share a
// cache namespace only when both are equal, a missing one differing from an empty one.
func namespaceOf(r *Req) string {
	f := func(p *string) string {
		if p == nil {
			return "-"
		}
		return "+" + *p
	}
	return f(r.C.ExtraKey) + "\x00" + f(r.C.CacheSalt)
}

func tokensKey(t []int64) string {
	b := make([]byte, 0, len(t)*8)
	for _, v := range t {
		b = append(b, byte(v), byte(v>>8), byte(v>>16), byte(v>>24), byte(v>>32), byte(v>>40), byte(v>>48), byte(v>>56))
	}
	return string(b)
}

// calcBoundedPriority is the bounded branch of calc_priority (:289-343@c4af57ff). It reports whether
// the bounded batch is active; when it is, queue is reordered to the batch (longest prefix first),
// then the rest in arrival order. The batch runs until all of its members have left the queue; it
// is retired by liveness at the next pass (the upstream also retires on departure, which cannot be
// seen before the next pass reads the queue).
func (s *Scheduler) calcBoundedPriority(queue []*Req) bool {
	b := &s.bounded
	if !(b.active || (len(queue) > waitingQueuePrefixMatchMax && s.policy == LPM)) {
		return false
	}
	if b.active {
		live := map[*Req]bool{}
		for _, r := range queue {
			live[r] = true
		}
		b.members = slices.DeleteFunc(slices.Clone(b.members), func(r *Req) bool { return !live[r] })
	}
	if len(b.members) == 0 && len(queue) > waitingQueuePrefixMatchMax && s.shouldStartBoundedBatch(queue) {
		b.members = slices.Clone(queue[:boundedLPMWindowSize])
	}
	// A committed batch is finished even after the queue drops below the limit.
	b.active = len(b.members) > 0
	if !b.active {
		return false
	}
	member := map[*Req]bool{}
	for _, r := range b.members {
		member[r] = true
	}
	var tail []*Req
	for _, r := range queue {
		if !member[r] {
			tail = append(tail, r)
		}
	}
	// The tail keeps its arrival order; it is matched as the fcfs path matches it, and refreshed
	// so that matching the batch makes its prefixes the more recent.
	if s.tree.SupportsFastMatchPrefix() {
		for _, r := range tail {
			s.initNextRoundInput(r, true)
		}
	}
	s.touchWaitingPrefixes(FCFS, tail)
	deprioritized := s.computePrefixMatches(b.members)
	sort.SliceStable(b.members, func(i, j int) bool {
		return lpmKey(b.members[i], deprioritized) < lpmKey(b.members[j], deprioritized)
	})
	copy(queue, append(slices.Clone(b.members), tail...))
	return true
}
