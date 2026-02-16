package sim

import "fmt"

// RoutingSnapshot is a lightweight view of instance state for policy decisions.
// Populated by ClusterSimulator from cluster.InstanceSnapshot when building RouterState
// (used by both AdmissionPolicy and RoutingPolicy).
// Timestamp is intentionally excluded: snapshot freshness is managed by
// CachedSnapshotProvider and is not a policy concern.
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
	// Priority is a one-shot cluster-level priority hint applied before instance injection.
	// Zero (default) means defer to instance-level PriorityPolicy entirely.
	// Non-zero value sets req.Priority for initial queue ordering only — the instance-level
	// PriorityPolicy recomputes priority each step, so this hint affects first-step scheduling
	// but does not persist. This is intentional: it allows priority to evolve over time
	// (e.g., SLOBasedPriority ages requests) while giving routing a way to influence initial placement.
	Priority float64
}

// RoutingPolicy decides which instance should handle a request.
// Implementations receive request and cluster-wide state via *RouterState.
type RoutingPolicy interface {
	Route(req *Request, state *RouterState) RoutingDecision
}

// RoundRobin routes requests in round-robin order across instances.
type RoundRobin struct {
	counter int
}

// Route implements RoutingPolicy for RoundRobin.
func (rr *RoundRobin) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
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
func (ll *LeastLoaded) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
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

// WeightedScoring routes requests using a weighted combination of cache affinity and load balance.
// Score = (1 - KVUtilization) * cacheWeight + (1 - normalizedLoad) * loadWeight.
// Higher scores are preferred. Ties broken by first occurrence in snapshot order.
type WeightedScoring struct {
	cacheWeight float64
	loadWeight  float64
}

// Route implements RoutingPolicy for WeightedScoring.
func (ws *WeightedScoring) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("WeightedScoring.Route: empty snapshots")
	}

	// Compute loads and find max for normalization
	loads := make([]int, len(snapshots))
	maxLoad := 0
	for i, snap := range snapshots {
		loads[i] = snap.QueueDepth + snap.BatchSize
		if loads[i] > maxLoad {
			maxLoad = loads[i]
		}
	}

	// Compute scores
	scores := make(map[string]float64, len(snapshots))
	bestScore := -1.0
	bestIdx := 0

	for i, snap := range snapshots {
		// Normalize load to [0,1] (handle uniform load: all zero → normalizedLoad = 0)
		normalizedLoad := 0.0
		if maxLoad > 0 {
			normalizedLoad = float64(loads[i]) / float64(maxLoad)
		}

		// Composite score
		cacheScore := (1.0 - snap.KVUtilization) * ws.cacheWeight
		loadScore := (1.0 - normalizedLoad) * ws.loadWeight
		score := cacheScore + loadScore
		scores[snap.ID] = score

		// Select argmax; first occurrence wins on tie (strict >)
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	return RoutingDecision{
		TargetInstance: snapshots[bestIdx].ID,
		Reason:         fmt.Sprintf("weighted-scoring (score=%.3f)", bestScore),
		Scores:         scores,
	}
}

// PrefixAffinity routes requests with matching prefixes to the same instance (cache-aware).
// On cache miss, falls back to LeastLoaded. Maintains prefix-to-instance mapping.
// Note: prefixMap grows with unique prefix count and is not evicted. This is acceptable
// for finite-duration simulations; large-cardinality workloads will consume proportional memory.
type PrefixAffinity struct {
	prefixMap map[string]string // prefix hash → instance ID (unbounded; grows with unique prefix count)
}

// Route implements RoutingPolicy for PrefixAffinity.
func (pa *PrefixAffinity) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("PrefixAffinity.Route: empty snapshots")
	}

	// Compute prefix hash using KVCache's hashTokens (pipe-delimited decimal strings)
	prefixHash := hashTokens(req.InputTokens)

	// Check cache for existing mapping
	if targetID, found := pa.prefixMap[prefixHash]; found {
		// Verify target still in snapshots (instance may have been removed)
		for _, snap := range snapshots {
			if snap.ID == targetID {
				return RoutingDecision{
					TargetInstance: targetID,
					Reason:         "prefix-affinity (cache-hit)",
				}
			}
		}
	}

	// Cache miss or stale entry: fallback to LeastLoaded, passing state through
	ll := &LeastLoaded{}
	decision := ll.Route(req, state)

	// Update cache with new mapping
	pa.prefixMap[prefixHash] = decision.TargetInstance

	return RoutingDecision{
		TargetInstance: decision.TargetInstance,
		Reason:         "prefix-affinity (cache-miss, fallback to least-loaded)",
	}
}

// NewRoutingPolicy creates a routing policy by name.
// Valid names: "", "round-robin", "least-loaded", "weighted", "prefix-affinity".
// Empty string defaults to round-robin.
// For weighted scoring, cacheWeight and loadWeight configure the composite score.
// Panics on unrecognized names.
func NewRoutingPolicy(name string, cacheWeight, loadWeight float64) RoutingPolicy {
	if !ValidRoutingPolicies[name] {
		panic(fmt.Sprintf("unknown routing policy %q", name))
	}
	switch name {
	case "", "round-robin":
		return &RoundRobin{}
	case "least-loaded":
		return &LeastLoaded{}
	case "weighted":
		return &WeightedScoring{cacheWeight: cacheWeight, loadWeight: loadWeight}
	case "prefix-affinity":
		return &PrefixAffinity{prefixMap: make(map[string]string)}
	default:
		panic(fmt.Sprintf("unhandled routing policy %q", name))
	}
}
