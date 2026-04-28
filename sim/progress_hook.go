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
// Scalar fields are independently copied on assignment. InstanceSnapshots is a
// slice — do not mutate elements after receipt. Each OnProgress callback receives
// a freshly allocated slice (BC-7).
type ProgressSnapshot struct {
	Clock int64

	// TotalCompleted is the raw sum of CompletedRequests across all instances.
	// In PD (prefill/decode disaggregation) mode, this value is inflated for
	// non-final snapshots because the PD correction (subtracting prefill-only
	// completions) is applied only during post-simulation aggregation.
	TotalCompleted int

	TotalTimedOut   int
	TotalDropped    int
	TotalInputTokens int

	// TotalOutputTokens is incremented at request completion, not per decode step.
	// During long decode runs it stays flat until completion, then jumps. This is
	// intentional to avoid double-counting under preemption.
	TotalOutputTokens int

	TotalPreemptions  int64
	InstanceSnapshots []InstanceSnapshot

	// Cluster-mode only; always 0 in single-instance mode.
	RejectedRequests  int
	RoutingRejections int
	GatewayQueueDepth int
	GatewayQueueShed  int
	ActivePDTransfers int

	// ActiveInstances counts instances in Active or WarmingUp state.
	// Loading and Draining instances appear in InstanceSnapshots but are excluded
	// from this count.
	ActiveInstances int

	// TotalInstances includes all instances (including Terminated).
	// len(InstanceSnapshots) excludes Terminated instances and may be smaller.
	TotalInstances int

	IsFinal bool
}

// InstanceSnapshot captures per-instance state at a point in time.
// All fields are value types — safe to hold indefinitely.
type InstanceSnapshot struct {
	ID        string
	QueueDepth int
	BatchSize  int
	KVUtilization float64
	KVFreeBlocks  int64
	KVTotalBlocks int64
	CacheHitRate  float64
	PreemptionCount   int64
	CompletedRequests int

	// InFlightRequests is always 0 in single-instance mode (no dispatch layer);
	// use QueueDepth + BatchSize for an equivalent in-flight count.
	InFlightRequests int

	TimedOutRequests int

	// State matches InstanceState string constants defined in sim/cluster
	// (e.g. "Active", "Loading", "Draining"). Always "Active" in single-instance mode.
	State string

	Model string
}
