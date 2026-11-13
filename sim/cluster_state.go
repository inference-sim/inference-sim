package sim

// ClusterState is a cluster-wide view of live instance state, read by the model autoscaler. It
// feeds no routing decision: routing is the embedded llm-d router's (sim/llmd), which reads the
// engines' published metrics, not this struct.
//
// USAGE BOUNDARY: In production, only constructed by buildClusterState
// (sim/cluster/cluster_event.go), called once per autoscaler tick; its consumers are the
// autoscaler Collector (DefaultCollector) and, through LoadingSnapshots, the saturation
// analyzer's pending-supply terms. Tests may construct it directly. The single-instance
// Simulator does not use it. It lives in sim/ (not sim/cluster/) so the Collector interface
// can name it without an import cycle.
type ClusterState struct {
	Snapshots []InstanceStatus // One per routable instance (Active + WarmingUp)
	// One per Loading instance. Populated fields: ID, Model, GPUType, TPDegree, CostPerHour,
	// TotalKvCapacityTokens. QueueDepth, BatchSize, KVUtilization, FreeKVBlocks, CacheHitRate,
	// InFlightRequests, KvTokensInUse remain zero.
	LoadingSnapshots []InstanceStatus
	Clock            int64 // Current simulation clock in microseconds
	// SelectedInstance is never set: buildClusterState leaves it empty and nothing reads it.
	// Its only writer and reader (BLIS's native P/D decider) were removed with BLIS's own
	// routing (#2000).
	SelectedInstance string
}
