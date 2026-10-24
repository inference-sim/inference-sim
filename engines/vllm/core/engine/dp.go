// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/engine/core.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/engine/core_client.py (DPLBAsyncMPClient.get_core_engine_for_request) and
// vllm/v1/engine/coordinator.py (wave start).

package engine

import "math"

// Wave is one DP rank's wave state in DPEngineCoreProc (core.py:2160-2176): the step counter
// the sync cadence and the prefill throttle read, the wave number, and whether the rank's
// busy loop is stepping.
type Wave struct {
	// PrefillScheduleInterval is scheduler_config.prefill_schedule_interval
	// (vllm/config/scheduler.py:203@5df63a72); 1 never throttles.
	PrefillScheduleInterval int
	// SyncInterval is parallel_config.dp_sync_interval (vllm/config/parallel.py:162, default 16).
	SyncInterval int

	StepCounter int
	Current     int
	Running     bool
}

// Throttle is _should_throttle_prefills (core.py:2318-2325): defer new prefills to steps the
// counter, identical across ranks, aligns to the interval; a fresh wave's counter is 0.
func (w *Wave) Throttle() bool {
	return w.PrefillScheduleInterval > 1 && w.StepCounter%w.PrefillScheduleInterval != 0
}

// Sync is _has_global_unfinished_reqs (core.py:2407-2428) without a pause: the step counter
// advances; the ranks' all-reduce runs at the wave's first step and then every
// SyncInterval steps; between syncs the wave continues. global is the all-reduce's answer
// (any rank has unfinished requests).
func (w *Wave) Sync(global func() bool) bool {
	w.StepCounter++
	if w.StepCounter != 1 && w.StepCounter%max(w.SyncInterval, 1) != 0 {
		return true
	}
	return global()
}

// DP is a data-parallel engine group stepping in lockstep, each rank's iteration being one
// pass of DPEngineCoreProc.run_busy_loop (core.py:2328-2405).
type DP struct {
	Ranks []*Core
	Waves []*Wave
	LB    *LoadBalancer
}

// RankStep is what one rank did in one lockstep iteration.
type RankStep struct {
	// Step is the rank's engine step; empty when it had no requests.
	Step StepResult
	// Throttled is whether this step deferred prefills.
	Throttled bool
	// Dummy is whether the rank ran a dummy batch: nothing it scheduled ran, but it has
	// unfinished requests or its wave is still running (core.py:2357-2368).
	Dummy bool
}

// Add routes a request with the internal load balancer and hands it to the rank it picks.
// A request arriving while its rank's wave is idle starts the next wave on every rank
// (FIRST_REQ, core_client.py:1590-1593; START_DP_WAVE, coordinator.py:363-374 and
// core.py:2281-2298).
func (d *DP) Add(id string) int {
	rank := d.LB.Choose(id)
	if !d.Waves[rank].Running {
		for _, w := range d.Waves {
			w.Running = true
		}
	}
	return rank
}

// Step runs every rank's busy-loop iteration, then the wave's all-reduce on each.
func (d *DP) Step() []RankStep {
	out := make([]RankStep, len(d.Ranks))
	local := make([]bool, len(d.Ranks))
	for i, c := range d.Ranks {
		w := d.Waves[i]
		if c.Sched.HasRequests() {
			throttle := w.Throttle()
			c.Throttle = func() bool { return throttle }
			out[i].Throttled = throttle
			out[i].Step = c.Step()
		}
		unfinished := c.Sched.HasUnfinished()
		if !out[i].Step.Executed && (unfinished || w.Running) {
			out[i].Dummy = true
		}
		local[i] = unfinished
	}
	any := false
	for _, u := range local {
		any = any || u
	}
	for i, w := range d.Waves {
		if !w.Running && !local[i] {
			continue
		}
		w.Running = w.Sync(func() bool { return any })
		if !w.Running {
			// The wave is complete: the next starts with its counter at 0 (core.py:2378-2394).
			w.Current++
			w.StepCounter = 0
		}
	}
	return out
}

// LoadBalancer is DPLBAsyncMPClient's routing state for one client (core_client.py:1607-1640):
// the coordinator's latest [waiting, running, kv_cache_usage] per engine, the client's exact
// in-flight count per engine, and the rotating scan start.
type LoadBalancer struct {
	Counts     [][3]float64
	inflight   []int
	reqs       map[string]int
	startIndex int
}

// NewLoadBalancer is a client for n engines (client_count 1, client_index 0).
func NewLoadBalancer(n int) *LoadBalancer {
	return &LoadBalancer{Counts: make([][3]float64, n), inflight: make([]int, n), reqs: map[string]int{}}
}

// Publish is the coordinator's stats update, which rebinds the counts (core_client.py:1562-1575).
func (lb *LoadBalancer) Publish(counts [][3]float64) {
	lb.Counts = append([][3]float64(nil), counts...)
}

// Choose is get_core_engine_for_request (core_client.py:1642-1697) for a request with no
// explicit data_parallel_rank: the engine whose load score is lowest, scanning from the
// rotating start; the score is the greater of its waiting + running snapshot and this
// client's in-flight count, plus a penalty for waiting requests on a KV-bound engine.
func (lb *LoadBalancer) Choose(id string) int {
	if e, ok := lb.reqs[id]; ok {
		// Streaming-input session affinity.
		return e
	}
	n := len(lb.Counts)
	minScore := math.MaxFloat64
	choice := 0
	for i := 0; i < n; i++ {
		idx := (lb.startIndex + i) % n
		waiting, running, kv := lb.Counts[idx][0], lb.Counts[idx][1], lb.Counts[idx][2]
		score := max(float64(lb.inflight[idx]), waiting+running)
		if waiting != 0 {
			score += waiting * 6.0 * max(0.0, kv-0.5)
		}
		if score < minScore {
			minScore = score
			choice = idx
		}
	}
	lb.Counts[choice][0]++
	lb.startIndex = (lb.startIndex + 1) % n
	lb.reqs[id] = choice
	lb.inflight[choice]++
	return choice
}

// Finished is process_engine_outputs (core_client.py:1703-1710): a finished request leaves
// its engine's in-flight count.
func (lb *LoadBalancer) Finished(id string) {
	if e, ok := lb.reqs[id]; ok {
		delete(lb.reqs, id)
		lb.inflight[e]--
	}
}
