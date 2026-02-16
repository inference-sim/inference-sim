package sim

import "fmt"

// RoutingSnapshot is a lightweight view of instance state for routing decisions.
// Populated by ClusterSimulator from cluster.InstanceSnapshot at routing time.
// Timestamp is intentionally excluded: snapshot freshness is managed by
// CachedSnapshotProvider and is not a routing concern.
type RoutingSnapshot struct {
	ID            string
	QueueDepth    int
	BatchSize     int
	KVUtilization float64
	FreeKVBlocks  int64
}

// RoutingDecision encapsulates the routing decision for a request.
type RoutingDecision struct {
	TargetInstance string             // Instance ID to route to (must match a snapshot ID)
	Reason         string             // Human-readable explanation
	Scores         map[string]float64 // Instance ID → composite score (nil for policies without scoring)
}

// RoutingPolicy decides which instance should handle a request.
// Implementations receive request, instance snapshots, and current clock.
// This is a transitional interface for PR 6; PR 8 will extend with RouterState parameter.
type RoutingPolicy interface {
	Route(req *Request, snapshots []RoutingSnapshot, clock int64) RoutingDecision
}

// RoundRobin routes requests in round-robin order across instances.
type RoundRobin struct {
	counter int
}

// Route implements RoutingPolicy for RoundRobin.
func (rr *RoundRobin) Route(req *Request, snapshots []RoutingSnapshot, clock int64) RoutingDecision {
	if len(snapshots) == 0 {
		panic("RoundRobin.Route: empty snapshots")
	}
	target := snapshots[rr.counter%len(snapshots)]
	rr.counter++
	return RoutingDecision{
		TargetInstance: target.ID,
		Reason:         fmt.Sprintf("round-robin[%d]", rr.counter-1),
	}
}

// LeastLoaded routes requests to the instance with minimum (QueueDepth + BatchSize).
// Ties are broken by first occurrence in snapshot order (lowest index).
type LeastLoaded struct{}

// Route implements RoutingPolicy for LeastLoaded.
func (ll *LeastLoaded) Route(req *Request, snapshots []RoutingSnapshot, clock int64) RoutingDecision {
	if len(snapshots) == 0 {
		panic("LeastLoaded.Route: empty snapshots")
	}

	minLoad := snapshots[0].QueueDepth + snapshots[0].BatchSize
	target := snapshots[0]

	for i := 1; i < len(snapshots); i++ {
		load := snapshots[i].QueueDepth + snapshots[i].BatchSize
		if load < minLoad {
			minLoad = load
			target = snapshots[i]
		}
	}

	return RoutingDecision{
		TargetInstance: target.ID,
		Reason:         fmt.Sprintf("least-loaded (load=%d)", minLoad),
	}
}

// NewRoutingPolicy creates a routing policy by name.
// Valid names: "", "round-robin", "least-loaded", "weighted", "prefix-affinity".
// Empty string defaults to round-robin.
// For weighted scoring, cacheWeight and loadWeight configure the composite score.
// Panics on unrecognized names.
func NewRoutingPolicy(name string, cacheWeight, loadWeight float64) RoutingPolicy {
	switch name {
	case "", "round-robin":
		return &RoundRobin{}
	case "least-loaded":
		return &LeastLoaded{}
	default:
		panic(fmt.Sprintf("unknown routing policy %q", name))
	}
}
