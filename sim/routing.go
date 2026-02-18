package sim

import "fmt"

// RoutingSnapshot is a lightweight view of instance state for policy decisions.
// Populated by ClusterSimulator from cluster.InstanceSnapshot when building RouterState
// (used by both AdmissionPolicy and RoutingPolicy).
// Timestamp is intentionally excluded: snapshot freshness is managed by
// CachedSnapshotProvider and is not a policy concern.
type RoutingSnapshot struct {
	ID              string
	QueueDepth      int
	BatchSize       int
	KVUtilization   float64
	FreeKVBlocks    int64
	CacheHitRate    float64
	PendingRequests int // Requests routed to this instance but not yet in queue
}

// EffectiveLoad returns the total effective load on this instance:
// QueueDepth + BatchSize + PendingRequests.
// Used by routing policies and counterfactual scoring for consistent load calculations.
func (s RoutingSnapshot) EffectiveLoad() int {
	return s.QueueDepth + s.BatchSize + s.PendingRequests
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

// LeastLoaded routes requests to the instance with minimum (QueueDepth + BatchSize + PendingRequests).
// PendingRequests prevents pile-on at high request rates where multiple routing decisions
// occur at the same timestamp before instance events process (#175).
// Ties are broken by first occurrence in snapshot order (lowest index).
type LeastLoaded struct{}

// Route implements RoutingPolicy for LeastLoaded.
func (ll *LeastLoaded) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("LeastLoaded.Route: empty snapshots")
	}

	minLoad := snapshots[0].EffectiveLoad()
	target := snapshots[0]

	for i := 1; i < len(snapshots); i++ {
		load := snapshots[i].EffectiveLoad()
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

// WeightedScoring routes requests using a weighted combination of cache availability and load balance.
//
// Two scoring dimensions:
//   - Cache: FreeKVBlocks / maxFreeKVBlocks — measures memory availability
//   - Load:  1 / (1 + effectiveLoad) — measures load balance without max-normalization
//
// Where effectiveLoad = QueueDepth + BatchSize + PendingRequests.
//
// PendingRequests counts requests routed to an instance but not yet in its queue (#170).
// This gives routing visibility into its own recent decisions, breaking the symmetric
// equilibrium that otherwise makes weight changes unobservable (#169).
// Load increases immediately (via PendingRequests) while FreeKVBlocks stays unchanged
// (no KV blocks allocated yet for pending requests), creating signal disagreement
// where weight changes produce different routing decisions.
//
// The load dimension uses 1/(1+load) instead of max-normalization to preserve
// absolute differences between instances. Max-normalization collapses small
// differences, making weights ineffective in balanced clusters.
//
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

	// Find max FreeKVBlocks for cache normalization
	maxFreeKV := int64(0)
	for _, snap := range snapshots {
		if snap.FreeKVBlocks > maxFreeKV {
			maxFreeKV = snap.FreeKVBlocks
		}
	}

	// Compute scores using independent signals
	scores := make(map[string]float64, len(snapshots))
	bestScore := -1.0
	bestIdx := 0

	for i, snap := range snapshots {
		// Cache dimension: FreeKVBlocks normalized by cluster max
		cacheScore := 0.0
		if maxFreeKV > 0 {
			cacheScore = float64(snap.FreeKVBlocks) / float64(maxFreeKV)
		}

		// Load dimension: inverse of effective load (no max-normalization)
		effectiveLoad := snap.EffectiveLoad()
		loadScore := 1.0 / (1.0 + float64(effectiveLoad))

		score := cacheScore*ws.cacheWeight + loadScore*ws.loadWeight
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

// AlwaysBusiest routes requests to the instance with maximum (QueueDepth + BatchSize + PendingRequests).
// Pathological template for testing load imbalance detection.
// Ties broken by first occurrence in snapshot order (lowest index).
type AlwaysBusiest struct{}

// Route implements RoutingPolicy for AlwaysBusiest.
func (ab *AlwaysBusiest) Route(_ *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("AlwaysBusiest.Route: empty snapshots")
	}

	maxLoad := snapshots[0].EffectiveLoad()
	target := snapshots[0]

	for i := 1; i < len(snapshots); i++ {
		load := snapshots[i].EffectiveLoad()
		if load > maxLoad {
			maxLoad = load
			target = snapshots[i]
		}
	}

	return RoutingDecision{
		TargetInstance: target.ID,
		Reason:         fmt.Sprintf("always-busiest (load=%d)", maxLoad),
	}
}

// NewRoutingPolicy creates a routing policy by name.
// Valid names are defined in validRoutingPolicies (bundle.go).
// Empty string defaults to round-robin.
// For weighted scoring, cacheWeight and loadWeight configure the composite score.
// Panics on unrecognized names.
func NewRoutingPolicy(name string, cacheWeight, loadWeight float64) RoutingPolicy {
	if !IsValidRoutingPolicy(name) {
		panic(fmt.Sprintf("unknown routing policy %q", name))
	}
	switch name {
	case "", "round-robin":
		return &RoundRobin{}
	case "least-loaded":
		return &LeastLoaded{}
	case "weighted":
		// Normalize weights so they sum to 1.0 (preserves ratio).
		// This ensures (0.6, 0.2) behaves identically to (0.75, 0.25).
		// Panics on non-positive sum (CLI validates before reaching here).
		sum := cacheWeight + loadWeight
		if sum <= 0 {
			panic(fmt.Sprintf("WeightedScoring requires positive weight sum, got cacheWeight=%f + loadWeight=%f = %f", cacheWeight, loadWeight, sum))
		}
		cacheWeight = cacheWeight / sum
		loadWeight = loadWeight / sum
		return &WeightedScoring{cacheWeight: cacheWeight, loadWeight: loadWeight}
	case "prefix-affinity":
		return &PrefixAffinity{prefixMap: make(map[string]string)}
	case "always-busiest":
		return &AlwaysBusiest{}
	default:
		panic(fmt.Sprintf("unhandled routing policy %q", name))
	}
}
