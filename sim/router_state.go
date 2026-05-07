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
	// SelectedInstance == ""). DisaggregationDecider implementations may read
	// this to identify which of Snapshots was chosen upstream; a zero value
	// must be tolerated (treat as "no selection known").
	SelectedInstance string
}
