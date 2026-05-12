package sim

// RouterState provides cluster-wide state to policy interfaces.
// Built by ClusterSimulator before each policy invocation.
// This is a bridge type in sim/ (not sim/cluster/) to avoid import cycles —
// same pattern as RoutingSnapshot.
//
// USAGE BOUNDARY: In production, only constructed by ClusterSimulator's event
// handlers (via buildRouterState). Tests may construct directly.
// Single-instance Simulator does not use RouterState — instance-level policies
// (InstanceScheduler) receive parameters directly.
// This prevents import cycles: sim/cluster/ imports sim/, not the reverse.
type RouterState struct {
	Snapshots        []RoutingSnapshot // One per routable instance (Active + WarmingUp)
	// One per Loading instance. Populated fields: ID, Model, GPUType, TPDegree, CostPerHour,
	// TotalKvCapacityTokens. QueueDepth, BatchSize, KVUtilization, FreeKVBlocks, CacheHitRate,
	// InFlightRequests, KvTokensInUse remain zero.
	LoadingSnapshots []RoutingSnapshot
	Clock            int64 // Current simulation clock in microseconds
	// SelectedInstance is the instance ID pre-selected by an upstream caller
	// (typically the decode-routing policy) before invoking a
	// DisaggregationDecider. Empty for contexts where no prior selection has
	// been made (e.g., RoutingPolicy.Route itself receives RouterState with
	// SelectedInstance == ""). DisaggregationDecider implementations may use
	// this ID to locate the selected pod's state (e.g., in a per-pod cache-
	// query map). Membership in Snapshots is not structurally guaranteed —
	// implementations must tolerate both a zero value ("no selection known")
	// and an ID that has no corresponding entry in Snapshots (e.g., the pod
	// was removed after routing) and guard accordingly.
	SelectedInstance string
	// ActiveTransfers is the live count of in-flight KV transfers at the
	// moment RouterState is built, as observed by ClusterSimulator. It is
	// populated only when --pd-transfer-contention is enabled; otherwise
	// it is zero. DisaggregationDecider implementations may read it for
	// backpressure-aware decisions (G2). No built-in decider currently
	// reads it — adoption by an external policy in llm-d would require a
	// cross-repo metric-emission PR; see issue #1342.
	ActiveTransfers int
}
