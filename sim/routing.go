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

// WeightedScoring routes requests using a composable scorer pipeline.
//
// Each scorer evaluates all instances on a [0,1] scale. Scores are combined
// with configurable weights: composite = Σ clamp(s_i) × w_i, then argmax.
//
// Available scorers: queue-depth (min-max normalization of EffectiveLoad),
// kv-utilization (1 - KVUtilization), load-balance (1/(1 + EffectiveLoad)).
// See sim/routing_scorers.go for implementations.
//
// Higher scores are preferred. Ties broken by first occurrence in snapshot order.
type WeightedScoring struct {
	scorers []scorerFunc
	weights []float64 // normalized to sum to 1.0
	names   []string  // for debugging/logging
}

// Route implements RoutingPolicy for WeightedScoring.
func (ws *WeightedScoring) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("WeightedScoring.Route: empty snapshots")
	}

	// Compute composite scores from all scorers
	scores := make(map[string]float64, len(snapshots))
	for i, scorer := range ws.scorers {
		dimScores := scorer(snapshots)
		for _, snap := range snapshots {
			s := dimScores[snap.ID]
			// Clamp to [0,1] per INV-1
			if s < 0 {
				s = 0
			}
			if s > 1 {
				s = 1
			}
			scores[snap.ID] += s * ws.weights[i]
		}
	}

	// Argmax: select instance with highest composite score.
	// Ties broken by first occurrence in snapshot order (strict >).
	bestScore := -1.0
	bestIdx := 0
	for i, snap := range snapshots {
		if scores[snap.ID] > bestScore {
			bestScore = scores[snap.ID]
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
// For weighted scoring, scorerConfigs configures the scorer pipeline.
// If scorerConfigs is nil/empty for "weighted", DefaultScorerConfigs() is used.
// Non-weighted policies ignore scorerConfigs.
// Panics on unrecognized names.
func NewRoutingPolicy(name string, scorerConfigs []ScorerConfig) RoutingPolicy {
	if !IsValidRoutingPolicy(name) {
		panic(fmt.Sprintf("unknown routing policy %q", name))
	}
	switch name {
	case "", "round-robin":
		return &RoundRobin{}
	case "least-loaded":
		return &LeastLoaded{}
	case "weighted":
		if len(scorerConfigs) == 0 {
			scorerConfigs = DefaultScorerConfigs()
		}
		scorers := make([]scorerFunc, len(scorerConfigs))
		names := make([]string, len(scorerConfigs))
		for i, cfg := range scorerConfigs {
			scorers[i] = newScorer(cfg.Name)
			names[i] = cfg.Name
		}
		weights := normalizeScorerWeights(scorerConfigs)
		return &WeightedScoring{scorers: scorers, weights: weights, names: names}
	case "prefix-affinity":
		return &PrefixAffinity{prefixMap: make(map[string]string)}
	case "always-busiest":
		return &AlwaysBusiest{}
	default:
		panic(fmt.Sprintf("unhandled routing policy %q", name))
	}
}
