package sim

// RouterState is a cluster-wide view of live instance state, read by the model autoscaler.
// Despite the name it feeds no routing decision: routing is the embedded llm-d router's
// (sim/llmd), which reads the engines' published metrics, not this struct. The rename is
// tracked in #1951.
//
// USAGE BOUNDARY: In production, only constructed by buildRouterState
// (sim/cluster/cluster_event.go), called once per autoscaler tick; its consumers are the
// autoscaler Collector (DefaultCollector) and, through LoadingSnapshots, the saturation
// analyzer's pending-supply terms. Tests may construct it directly. The single-instance
// Simulator does not use it. It lives in sim/ (not sim/cluster/) so the Collector interface
// can name it without an import cycle.
type RouterState struct {
	Snapshots []RoutingSnapshot // One per routable instance (Active + WarmingUp)
	// One per Loading instance. Populated fields: ID, Model, GPUType, TPDegree, CostPerHour,
	// TotalKvCapacityTokens. QueueDepth, BatchSize, KVUtilization, FreeKVBlocks, CacheHitRate,
	// InFlightRequests, KvTokensInUse remain zero.
	LoadingSnapshots []RoutingSnapshot
	Clock            int64 // Current simulation clock in microseconds
	// SelectedInstance is never set: buildRouterState leaves it empty and nothing reads it.
	// Its only writer and reader (BLIS's native P/D decider) were removed with BLIS's own
	// routing (#2000).
	SelectedInstance string
}
