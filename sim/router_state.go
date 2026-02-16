package sim

// RouterState provides cluster-wide state to policy interfaces.
// Built by ClusterSimulator before each policy invocation.
// This is a bridge type in sim/ (not sim/cluster/) to avoid import cycles —
// same pattern as RoutingSnapshot.
//
// USAGE BOUNDARY: Only constructed by ClusterSimulator's event handlers.
// Single-instance Simulator does not use RouterState — instance-level policies
// (PriorityPolicy, InstanceScheduler) receive parameters directly.
type RouterState struct {
	Snapshots []RoutingSnapshot // One per instance, same order as ClusterSimulator.instances
	Clock     int64             // Current simulation clock in microseconds
}
