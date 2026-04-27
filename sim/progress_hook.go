package sim

// ProgressHook receives periodic state snapshots during simulation execution.
// Implementations must treat snapshots as read-only and must not enqueue new
// simulation events or modify request state. A read-only, synchronous callback
// with no side-effects on simulation state cannot affect event ordering,
// therefore it cannot affect stdout (INV-6).
//
// This hook applies to both blis run and blis replay (both drive simulation
// through ClusterSimulator.Run). It does not apply to blis observe, which
// communicates with a real server.
type ProgressHook interface {
	OnProgress(snapshot ProgressSnapshot)
}

// ProgressSnapshot captures simulation state at a point in time.
// Value type — fully copied, safe to hold indefinitely.
type ProgressSnapshot struct {
	Clock             int64
	TotalCompleted    int
	TotalTimedOut     int
	TotalDropped      int
	TotalInputTokens  int
	TotalOutputTokens int
	TotalPreemptions  int64
	InstanceSnapshots []InstanceSnapshot
	RejectedRequests  int
	RoutingRejections int
	GatewayQueueDepth int
	GatewayQueueShed  int
	ActivePDTransfers int
	ActiveInstances   int
	TotalInstances    int
	IsFinal           bool
}

// InstanceSnapshot captures per-instance state at a point in time.
// Value type — fully copied, safe to hold indefinitely.
// In single-instance mode, ID is "instance-0" and InFlightRequests is 0
// (the concept only applies in cluster mode).
type InstanceSnapshot struct {
	ID                string
	QueueDepth        int
	BatchSize         int
	KVUtilization     float64
	KVFreeBlocks      int64
	KVTotalBlocks     int64
	CacheHitRate      float64
	PreemptionCount   int64
	CompletedRequests int
	InFlightRequests  int
	TimedOutRequests  int
	State             string
	Model             string
}
