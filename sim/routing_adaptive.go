package sim

import (
	"fmt"
	"math"
)

// SLOProfile defines a scorer weight profile for a specific SLO class.
// Each SLO tier gets its own balance of cache-affinity vs load-balancing,
// reflecting different latency tolerances.
type SLOProfile struct {
	Scorers []ScorerConfig
	// MaxLoadHeadroom is the maximum load difference (vs least-loaded instance)
	// that this SLO class tolerates for cache-affinity routing. Critical requests
	// get 0 (only exploit if cached instance IS least-loaded), batch requests
	// get a large value (tolerate significant queuing for cache benefit).
	MaxLoadHeadroom int
}

// DefaultSLOProfiles returns the default per-SLO-class weight profiles.
//
// Design rationale (derived from analytical model with beta3=0.004):
//   - critical: TTFT budget ~50ms. Even 1 extra queued request adds ~173ms delay,
//     which exceeds the cache saving of 53ms (prefix=2048). Never sacrifice load
//     balance for cache affinity.
//   - standard: Balanced approach. The default pa:3,qd:2,kv:2 scorer's natural
//     per-request adaptation handles this well.
//   - sheddable: Similar to standard but slightly more cache-tolerant.
//   - batch: TTFT budget ~5000ms. Queue delay of 5-10 extra requests is acceptable.
//     Maximize cache affinity to save compute.
//   - background: No TTFT constraint. Pure cache affinity to maximize throughput.
func DefaultSLOProfiles() map[string]SLOProfile {
	return map[string]SLOProfile{
		"critical": {
			// Critical: the cost-benefit scorer naturally suppresses cache affinity
			// when queue delay is high, and slo-headroom penalizes instances that
			// would blow the tight TTFT budget. No load headroom override needed —
			// the scorers handle it continuously.
			Scorers: []ScorerConfig{
				{Name: "cost-benefit", Weight: 3.0},
				{Name: "slo-headroom", Weight: 4.0},
				{Name: "queue-depth", Weight: 2.0},
			},
			MaxLoadHeadroom: math.MaxInt, // let the scorers decide
		},
		"standard": {
			// Standard: cost-benefit handles the cache-vs-load tradeoff,
			// supplemented by direct QD for load-balancing.
			Scorers: []ScorerConfig{
				{Name: "cost-benefit", Weight: 3.0},
				{Name: "slo-headroom", Weight: 2.0},
				{Name: "queue-depth", Weight: 2.0},
			},
			MaxLoadHeadroom: math.MaxInt,
		},
		"sheddable": {
			Scorers: []ScorerConfig{
				{Name: "cost-benefit", Weight: 3.0},
				{Name: "slo-headroom", Weight: 1.0},
				{Name: "queue-depth", Weight: 2.0},
			},
			MaxLoadHeadroom: math.MaxInt,
		},
		"batch": {
			// Batch: cost-benefit with large budget means CB ≈ 1.0 for any cache hit
			// (budget is so large that queue delay is negligible relative to saving).
			// Heavy PA weight exploits cache aggressively.
			Scorers: []ScorerConfig{
				{Name: "cost-benefit", Weight: 4.0},
				{Name: "prefix-affinity", Weight: 3.0},
				{Name: "queue-depth", Weight: 1.0},
			},
			MaxLoadHeadroom: math.MaxInt,
		},
		"background": {
			// Background: pure cache affinity + cost-benefit. No SLO constraint.
			Scorers: []ScorerConfig{
				{Name: "cost-benefit", Weight: 3.0},
				{Name: "prefix-affinity", Weight: 5.0},
			},
			MaxLoadHeadroom: math.MaxInt,
		},
	}
}

// AdaptiveConfig configures the SLO-aware adaptive routing behavior.
type AdaptiveConfig struct {
	// ExploitThreshold is the minimum prefix match ratio (matched_blocks/total_blocks)
	// to consider cache-affinity routing. Below this, the scorer pipeline handles
	// distribution naturally (PA returns 0, QD/KV take over).
	// Range: (0, 1]. Default: 0.3.
	ExploitThreshold float64

	// SLOProfiles maps SLO class names to scorer weight profiles.
	// If nil, DefaultSLOProfiles() is used.
	SLOProfiles map[string]SLOProfile

	// BetaCoeffs are the latency model beta coefficients used by cost-benefit
	// and slo-headroom scorers. If nil, uses default llama-3.1-8b coefficients.
	BetaCoeffs []float64
}

// DefaultAdaptiveConfig returns sensible defaults.
func DefaultAdaptiveConfig() AdaptiveConfig {
	return AdaptiveConfig{
		ExploitThreshold: 0.3,
	}
}

// ValidateAdaptiveConfig returns an error if the config is invalid.
func ValidateAdaptiveConfig(cfg AdaptiveConfig) error {
	if cfg.ExploitThreshold <= 0 || cfg.ExploitThreshold > 1 ||
		math.IsNaN(cfg.ExploitThreshold) || math.IsInf(cfg.ExploitThreshold, 0) {
		return fmt.Errorf("ExploitThreshold must be in (0, 1], got %v", cfg.ExploitThreshold)
	}
	return nil
}

// sloScorerPipeline holds the pre-built scorer pipeline for one SLO class.
type sloScorerPipeline struct {
	scorers         []scorerFunc
	weights         []float64
	maxLoadHeadroom int
}

// AdaptiveWeightedScoring routes requests using SLO-aware per-request weight profiles.
//
// Each SLO tier gets a different balance of cache-affinity vs load-balancing:
//   - critical: pure load-balancing (pa:0, qd:5, kv:2) — never sacrifice latency for cache
//   - standard: balanced (pa:3, qd:2, kv:2) — the static default, handles both naturally
//   - batch: cache-heavy (pa:5, qd:1, kv:1) — tolerate queuing for compute savings
//   - background: maximum cache (pa:5, qd:0.5, kv:0.5) — throughput over latency
//
// The static composable scorer's natural per-request adaptation (PA=0 on cache miss,
// PA>0 on cache hit) works within each profile. The SLO-aware layer adds a second
// adaptation axis: different latency tolerances get different weight balances.
//
// This design was motivated by the finding that a single weight profile cannot be
// optimal for both latency-sensitive and throughput-sensitive requests in the same
// stream. A critical request should never queue behind others for a cache hit,
// while a batch request should almost always take a cache hit even with queuing.
type AdaptiveWeightedScoring struct {
	// Per-SLO-class scorer pipelines (pre-built at construction)
	pipelines map[string]*sloScorerPipeline

	// Default pipeline for requests with unrecognized or empty SLO class
	defaultPipeline *sloScorerPipeline

	// Shared observers (called regardless of SLO class)
	observers []observerFunc

	config    AdaptiveConfig
	prefixIdx *PrefixCacheIndex
}

// Route implements RoutingPolicy for AdaptiveWeightedScoring.
func (aws *AdaptiveWeightedScoring) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("AdaptiveWeightedScoring.Route: empty snapshots")
	}

	// Select scorer pipeline based on request SLO class
	sloClass := ""
	if req != nil {
		sloClass = req.SLOClass
	}
	pipeline := aws.pipelines[sloClass]
	if pipeline == nil {
		pipeline = aws.defaultPipeline
	}

	// Compute composite scores with the SLO-specific weights
	scores := make(map[string]float64, len(snapshots))
	for i, scorer := range pipeline.scorers {
		dimScores := scorer(req, snapshots)
		for _, snap := range snapshots {
			s := dimScores[snap.ID]
			if s < 0 {
				s = 0
			}
			if s > 1 {
				s = 1
			}
			scores[snap.ID] += s * pipeline.weights[i]
		}
	}

	// Argmax: select instance with highest composite score
	bestScore := -1.0
	bestIdx := 0
	for i, snap := range snapshots {
		if scores[snap.ID] > bestScore {
			bestScore = scores[snap.ID]
			bestIdx = i
		}
	}

	// Load headroom check: if the selected instance exceeds the SLO's load
	// tolerance relative to the least-loaded, fall back to least-loaded.
	if pipeline.maxLoadHeadroom < math.MaxInt {
		minLoad := math.MaxInt
		minIdx := 0
		for i, snap := range snapshots {
			if snap.EffectiveLoad() < minLoad {
				minLoad = snap.EffectiveLoad()
				minIdx = i
			}
		}
		chosenLoad := snapshots[bestIdx].EffectiveLoad()
		if chosenLoad-minLoad > pipeline.maxLoadHeadroom {
			bestIdx = minIdx
		}
	}

	// Offloading pressure check: avoid instances with pending CPU→GPU reloads
	if snapshots[bestIdx].PendingTransferLatency > 0 || snapshots[bestIdx].KVThrashingRate > 0.1 {
		// Find the least-loaded non-thrashing instance
		minLoad := math.MaxInt
		for i, snap := range snapshots {
			if snap.PendingTransferLatency == 0 && snap.KVThrashingRate <= 0.1 {
				if snap.EffectiveLoad() < minLoad {
					minLoad = snap.EffectiveLoad()
					bestIdx = i
				}
			}
		}
	}

	// Notify observers of routing decision
	for _, obs := range aws.observers {
		obs(req, snapshots[bestIdx].ID)
	}

	return NewRoutingDecisionWithScores(
		snapshots[bestIdx].ID,
		fmt.Sprintf("adaptive-slo[%s] (score=%.3f)", sloClass, bestScore),
		scores,
	)
}
