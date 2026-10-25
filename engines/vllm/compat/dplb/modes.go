// Moved from inference-sim/sim/cluster/dp_replica.go at b631140d48ba37fedf834874d7c083bab3e8b116.

package dplb

// How a data-parallel replica's requests reach its ranks, as vLLM's serve flags choose
// (vllm/engine/arg_utils.py:536-538, :1225-1242; EngineCoreClient.make_async_mp_client,
// vllm/v1/engine/core_client.py:155-166 at v0.31.0). The cluster asks these rules and keeps the
// endpoints, the replica grouping and the routing (#2021 B2, verification pass 4, gap 3).
//   - Internal ("" or "internal", vLLM's default) and Hybrid (--data-parallel-hybrid-lb): one
//     endpoint per replica, whose front-end balances requests over its ranks (DPLBAsyncMPClient).
//     Hybrid balances over one node's ranks, which is the whole replica when it fits a node.
//   - External (--data-parallel-external-lb, or --data-parallel-multi-port-external-lb): every
//     rank serves its own port (DPAsyncMPClient, one client per rank), so the router sees and
//     picks ranks.
const (
	Internal = "internal"
	Hybrid   = "hybrid"
	External = "external"
)

// ValidMode reports whether mode names a DP load-balancing mode ("" is internal).
func ValidMode(mode string) bool {
	switch mode {
	case "", Internal, Hybrid, External:
		return true
	}
	return false
}

// EachRankIsAnEndpoint reports whether, under mode, every rank of a replica serves its own
// endpoint (external balancing), rather than the replica serving one endpoint whose front-end
// picks the rank.
func EachRankIsAnEndpoint(mode string) bool { return mode == External }
