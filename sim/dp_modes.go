// Moved from blis-vllm/compat/dplb/modes.go at fd036d5, the host's own: the mode is the run's
// serve flag, and the host builds the endpoints it names (phase 2c). The rank pick behind an
// internal or hybrid balancer is the engine's (its PickRank query).

package sim

// How a data-parallel replica's requests reach its ranks, as vLLM's serve flags choose
// (vllm/engine/arg_utils.py:536-538, :1225-1242; EngineCoreClient.make_async_mp_client,
// vllm/v1/engine/core_client.py:155-166 at v0.31.0). The cluster asks these rules and keeps the
// endpoints, the replica grouping and the routing (#2021 B2, verification pass 4, gap 3).
//   - DPLoadBalancingInternal ("" or "internal", vLLM's default) and DPLoadBalancingHybrid
//     (--data-parallel-hybrid-lb): one endpoint per replica, whose front-end balances requests
//     over its ranks (DPLBAsyncMPClient). Hybrid balances over one node's ranks, which is the
//     whole replica when it fits a node.
//   - DPLoadBalancingExternal (--data-parallel-external-lb, or
//     --data-parallel-multi-port-external-lb): every rank serves its own port (DPAsyncMPClient,
//     one client per rank), so the router sees and picks ranks.
const (
	DPLoadBalancingInternal = "internal"
	DPLoadBalancingHybrid   = "hybrid"
	DPLoadBalancingExternal = "external"
)

// ValidDPLoadBalancing reports whether mode names a DP load-balancing mode ("" is internal).
func ValidDPLoadBalancing(mode string) bool {
	switch mode {
	case "", DPLoadBalancingInternal, DPLoadBalancingHybrid, DPLoadBalancingExternal:
		return true
	}
	return false
}

// DPEachRankIsAnEndpoint reports whether, under mode, every rank of a replica serves its own
// endpoint (external balancing), rather than the replica serving one endpoint whose front-end
// picks the rank.
func DPEachRankIsAnEndpoint(mode string) bool { return mode == DPLoadBalancingExternal }
