// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/engine/core.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).
// Also vllm/v1/engine/core_client.py (DPLBAsyncMPClient.get_core_engine_for_request) and
// vllm/v1/engine/coordinator.py (wave start).

package engine

import (
	"fmt"
	"math"
)

// Wave is one DP rank's wave state in DPEngineCoreProc (core.py:2298-2314): the step counter
// the sync cadence and the prefill throttle read, the wave number, and whether the rank's
// busy loop is stepping.
type Wave struct {
	// PrefillScheduleInterval is scheduler_config.prefill_schedule_interval
	// (vllm/config/scheduler.py:203@4ba787ec); 1 never throttles.
	PrefillScheduleInterval int
	// SyncInterval is parallel_config.dp_sync_interval (vllm/config/parallel.py:163, default 16).
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
	// Block is whether the rank's step blocked on its oldest queued batch's output: the batch
	// it scheduled, with a queue of one (step(), core.py:657-687), or with a batch queue the
	// one step_with_batch_queue waits on (core.py:754-766). Core.StepEnd applies it.
	Block bool
}

// Add routes a request with the internal load balancer and hands it to the rank it picks.
// A request arriving while its rank's wave is idle starts the next wave on every rank
// (FIRST_REQ, core_client.py:1590-1593; START_DP_WAVE, coordinator.py:363-374 and
// core.py:2281-2298).
func (d *DP) Add(id string) int {
	rank := d.LB.Choose(id)
	d.startWave(rank)
	return rank
}

// AddVia routes a request through the given load-balancing client, as Add does through the next
// client in turn.
func (d *DP) AddVia(client int, id string) int {
	rank := d.LB.ChooseVia(client, id)
	d.startWave(rank)
	return rank
}

// AddTo hands a request to the rank the client named (EngineCoreRequest.data_parallel_rank):
// get_core_engine_for_request obeys it without consulting the load counts (core_client.py:
// 1647-1650), so the counts and the rotating scan start stay as they were, and the request
// counts against the rank's in-flight total. The wave starts as Add's does. A P/D proxy that
// picks the prefill or decode rank itself sends its requests this way.
func (d *DP) AddTo(id string, rank int) int {
	rank = d.LB.Pin(id, rank)
	d.startWave(rank)
	return rank
}

func (d *DP) startWave(rank int) {
	if !d.Waves[rank].Running {
		for _, w := range d.Waves {
			w.Running = true
		}
	}
}

// Step runs every rank's busy-loop iteration, then the wave's all-reduce on each. It is
// StepBegin, every rank's output applied (StepEnd), then EndStep: the split a simulator driving
// the group through blis-schemas' engine.Engine uses, where the forward pass runs on its clock
// between the two halves.
func (d *DP) Step() []RankStep {
	out := d.StepBegin()
	for i, c := range d.Ranks {
		if out[i].Block {
			out[i].Step = c.StepEnd(out[i].Step)
		}
	}
	d.EndStep(out)
	return out
}

// StepBegin is the first half of every rank's busy-loop iteration: each rank's engine step
// (step(), or step_with_batch_queue with a batch queue) up to where it blocks on a GPU result.
// A rank with requests schedules, throttled as _should_throttle_prefills says
// (core.py:2318-2325), and queues the batch its forward pass then runs; a rank with no requests
// but batches still queued blocks on the oldest (core.py:760-766). A rank with neither steps
// nothing: its loop waits for input (_process_input_queue, core.py:1573-1594). The caller
// applies the output each blocked rank waits on (Core.StepEnd) and then calls EndStep.
func (d *DP) StepBegin() []RankStep {
	out := make([]RankStep, len(d.Ranks))
	for i, c := range d.Ranks {
		w := d.Waves[i]
		if c.Sched.HasRequests() {
			throttle := w.Throttle()
			c.Throttle = func() bool { return throttle }
			out[i].Throttled = throttle
		}
		if c.Sched.HasRequests() || c.InFlight() > 0 {
			out[i].Step, out[i].Block = c.StepBegin()
		}
	}
	return out
}

// EndStep is the rest of the iteration once every rank's output is applied: a rank that ran
// nothing but has unfinished requests, or whose wave is still running, ran a dummy batch
// (core.py:2357-2368); then every rank's wave takes the all-reduce of the ranks' unfinished
// flags (_has_global_unfinished_reqs, core.py:2370-2394).
func (d *DP) EndStep(out []RankStep) {
	local := make([]bool, len(d.Ranks))
	for i, c := range d.Ranks {
		unfinished := c.Sched.HasUnfinished()
		if !out[i].Step.Executed && (unfinished || d.Waves[i].Running) {
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
}

// LoadBalancer is the routing state of a data-parallel group's load-balancing clients: one
// DPLBAsyncMPClient per API-server process, api_server_count of them (vllm/entrypoints/cli/serve.py:
// 107-137), each with its own copy of the coordinator's latest [waiting, running, kv_cache_usage]
// per engine, its own exact in-flight count per engine (only the requests it routed) and its own
// rotating scan start (core_client.py:1607-1640). The score scales a client's in-flight count by
// the client count and the optimistic increment is the client count, because each client sees only
// its share of the traffic (:1786, :1800@4ba787ec).
type LoadBalancer struct {
	clients []*lbClient
	// reqs is the client and engine each unfinished request was routed to.
	reqs map[string]lbRoute
	// arrivals counts the requests spread so far, which picks the next client (see Choose).
	arrivals int
	// Counts is the latest published snapshot, kept for the group to read.
	Counts [][3]float64
}

type lbClient struct {
	counts     [][3]float64
	inflight   []int
	startIndex int
}

type lbRoute struct{ client, engine int }

// NewLoadBalancer is one client for n engines (client_count 1, client_index 0).
func NewLoadBalancer(n int) *LoadBalancer { return NewLoadBalancers(n, 1) }

// NewLoadBalancers is clientCount clients for n engines. Client i starts its scan at
// n*i/clientCount (eng_start_index, core_client.py:1623-1625); a count below 1 is 1.
func NewLoadBalancers(n, clientCount int) *LoadBalancer {
	clientCount = max(clientCount, 1)
	lb := &LoadBalancer{reqs: map[string]lbRoute{}, Counts: make([][3]float64, n)}
	for i := 0; i < clientCount; i++ {
		lb.clients = append(lb.clients, &lbClient{counts: make([][3]float64, n), inflight: make([]int, n), startIndex: n * i / clientCount})
	}
	return lb
}

// SetClientCount replaces the clients with k of them, before any request is routed.
func (lb *LoadBalancer) SetClientCount(k int) error {
	if k == len(lb.clients) {
		return nil
	}
	if lb.arrivals > 0 || len(lb.reqs) > 0 {
		return fmt.Errorf("engine: the client count is set before the first request, %d routed", lb.arrivals)
	}
	*lb = *NewLoadBalancers(len(lb.Counts), k)
	return nil
}

// ClientCount is the number of clients.
func (lb *LoadBalancer) ClientCount() int { return len(lb.clients) }

// Publish is the coordinator's stats update, which rebinds every client's counts
// (core_client.py:1562-1575).
func (lb *LoadBalancer) Publish(counts [][3]float64) {
	lb.Counts = append([][3]float64(nil), counts...)
	for _, c := range lb.clients {
		c.counts = append([][3]float64(nil), counts...)
	}
}

// Choose routes a request with no explicit data_parallel_rank through the next client in turn.
// vLLM's API-server processes share one listening socket (SO_REUSEPORT, serve.py:298), and the
// kernel picks the process for each connection by a hash of the connection's addresses, which
// spreads connections evenly in expectation and in no fixed order. BLIS stands the even spread in
// for that hash: request k goes to client k mod client_count (a named substitution). ChooseVia
// routes through a named client.
func (lb *LoadBalancer) Choose(id string) int {
	if r, ok := lb.reqs[id]; ok {
		return r.engine
	}
	client := lb.arrivals % len(lb.clients)
	lb.arrivals++
	return lb.ChooseVia(client, id)
}

// ChooseVia is get_core_engine_for_request (core_client.py:1642-1697) on the given client, for a
// request with no explicit data_parallel_rank: the engine whose load score is lowest, scanning
// from the client's rotating start; the score is the greater of its waiting + running snapshot and
// client_count times this client's in-flight count, plus a penalty for waiting requests on a
// KV-bound engine.
func (lb *LoadBalancer) ChooseVia(client int, id string) int {
	if r, ok := lb.reqs[id]; ok {
		// Streaming-input session affinity.
		return r.engine
	}
	c, count := lb.clients[client], len(lb.clients)
	n := len(c.counts)
	minScore := math.MaxFloat64
	choice := 0
	for i := 0; i < n; i++ {
		idx := (c.startIndex + i) % n
		waiting, running, kv := c.counts[idx][0], c.counts[idx][1], c.counts[idx][2]
		score := max(float64(count*c.inflight[idx]), waiting+running)
		if waiting != 0 {
			score += waiting * 6.0 * max(0.0, kv-0.5)
		}
		if score < minScore {
			minScore = score
			choice = idx
		}
	}
	c.counts[choice][0] += float64(count)
	c.startIndex = (c.startIndex + 1) % n
	lb.reqs[id] = lbRoute{client, choice}
	c.inflight[choice]++
	return choice
}

// Pin is get_core_engine_for_request for a request that names its rank (core_client.py:1647-1650,
// 1692-1696): the rank is the request's, and only the in-flight count and the request's route
// change. A request already routed keeps its engine (session affinity, :1643-1645). The request
// reaches the next client in turn, as Choose spreads.
func (lb *LoadBalancer) Pin(id string, rank int) int {
	if r, ok := lb.reqs[id]; ok {
		return r.engine
	}
	client := lb.arrivals % len(lb.clients)
	lb.arrivals++
	lb.reqs[id] = lbRoute{client, rank}
	lb.clients[client].inflight[rank]++
	return rank
}

// Finished is process_engine_outputs (core_client.py:1703-1710): a finished request leaves its
// engine's in-flight count on the client that routed it.
func (lb *LoadBalancer) Finished(id string) {
	if r, ok := lb.reqs[id]; ok {
		delete(lb.reqs, id)
		lb.clients[r.client].inflight[r.engine]--
	}
}
