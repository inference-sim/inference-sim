package sim

// RouterState provides cluster-wide state to policy interfaces.
// Built by ClusterSimulator before each policy invocation.
// This is a bridge type in sim/ (not sim/cluster/) to avoid import cycles —
// same pattern as RoutingSnapshot.
//
// USAGE BOUNDARY: In production, only constructed by ClusterSimulator's event
// handlers (via buildRouterState). Tests may construct directly.
// Single-instance Simulator does not use RouterState — instance-level policies
// (PriorityPolicy, InstanceScheduler) receive parameters directly.
// This prevents import cycles: sim/cluster/ imports sim/, not the reverse.
type RouterState struct {
	Snapshots []RoutingSnapshot // One per instance, same order as ClusterSimulator.instances
	Clock     int64             // Current simulation clock in microseconds
}
