// Moved from inference-sim/sim/cluster/dp_replica.go at 6e9ef2fb1a370717bcdc34e13998ea7a5a83cd46.

// Package dplb is the compatibility engine's model of vLLM's internal data-parallel load
// balancer, the front-end rule that sends a request reaching a data-parallel replica's endpoint
// to one of its engine ranks. It moved here from inference-sim's cluster (#2021 E4), which keeps
// the replica grouping, the lockstep group and the in-flight bookkeeping, and asks this package
// which rank serves a request.
package dplb

import "math"

// StatsIntervalUs is how often vLLM's DP coordinator publishes the engines' request counts to
// the front-end's load balancer: min_stats_update_interval_ms = 100
// (vllm/v1/engine/coordinator.py:154-162, :267-269 at v0.31.0).
const StatsIntervalUs int64 = 100_000

// clientCount is the number of front-ends one balancer stands for. vLLM starts
// api_server_count front-ends, dp by default under the internal balancer
// (vllm/entrypoints/cli/serve.py:105-127), each scaling its own in-flight count by the client
// count to estimate the total. BLIS models one front-end that sees every in-flight request,
// which is the quantity that scaling estimates, so the count is 1.
const clientCount = 1

// Counts are one rank's published load: waiting and running requests, and KV cache usage.
type Counts struct {
	Waiting, Running int
	KVUsage          float64
}

// Rank is what the balancer reads from one rank when the coordinator publishes (Counts) and
// at every pick (InFlight, the front-end's exact count of its unfinished requests on the rank,
// engine_inflight, vllm/v1/engine/core_client.py:1569-1570, :1639, :1655-1660).
type Rank interface {
	Counts() Counts
	InFlight() int
}

// Balancer is vLLM's internal DP load balancer state for one replica (DPLBAsyncMPClient,
// vllm/v1/engine/core_client.py:1550-1640 at v0.31.0).
type Balancer struct {
	// counts is the coordinator's latest [waiting, running, kv_cache_usage] per rank, which
	// the front-end bumps locally on each assignment until the next publish overwrites it
	// (:1626-1628).
	counts []Counts
	// epoch is the StatsIntervalUs period counts was published in; -1 before the first.
	epoch int64
	// start is the balancer's rotating scan start (eng_start_index, :1629-1634).
	start int
}

// New is the balancer of a replica of n ranks, before any publish.
func New(n int) *Balancer { return &Balancer{epoch: -1, counts: make([]Counts, n)} }

// Pick assigns a request that reached the replica's endpoint at now to a rank, as vLLM
// v0.31.0's internal DP load balancer does (DPLBAsyncMPClient.get_core_engine_for_request,
// vllm/v1/engine/core_client.py:1589-1640):
//
//	score = max(client_count x in_flight, waiting + running)
//	        + waiting x 6 x max(0, kv_cache_usage - 0.5)      (only when waiting > 0)
//
// over the coordinator's last published [waiting, running, kv_cache_usage], scanning from a
// rotating start and keeping the first minimum. The chosen rank's waiting count is bumped by
// client_count until the next publish.
//
// The coordinator publishes at most every StatsIntervalUs; BLIS samples the ranks at the
// first assignment of each period rather than at the publish instant.
func (b *Balancer) Pick(ranks []Rank, now int64) int {
	if epoch := now / StatsIntervalUs; epoch != b.epoch {
		b.epoch = epoch
		for k, r := range ranks {
			b.counts[k] = r.Counts()
		}
	}
	n := len(ranks)
	best, bestScore := 0, math.Inf(1)
	for i := 0; i < n; i++ {
		k := (b.start + i) % n
		c := b.counts[k]
		score := float64(max(clientCount*ranks[k].InFlight(), c.Waiting+c.Running))
		if c.Waiting > 0 {
			score += float64(c.Waiting) * 6.0 * math.Max(0, c.KVUsage-0.5)
		}
		if score < bestScore {
			best, bestScore = k, score
		}
	}
	b.counts[best].Waiting += clientCount
	b.start = (b.start + 1) % n
	return best
}

// SetPublished sets the counts as last published in the given period, for tests that start
// the balancer from a known state.
func (b *Balancer) SetPublished(epoch int64, start int, counts []Counts) {
	b.epoch, b.start, b.counts = epoch, start, append([]Counts(nil), counts...)
}
