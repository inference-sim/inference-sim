package sim

import (
	"fmt"
	"math"
)

// AdaptiveConfig configures the exploit/explore adaptive routing behavior.
type AdaptiveConfig struct {
	// ExploitThreshold is the minimum prefix match ratio (matched_blocks/total_blocks)
	// to trigger exploit mode. Below this, the request is routed via load-balanced
	// explore mode. Range: (0, 1]. Default: 0.3 (30% prefix match triggers exploit).
	ExploitThreshold float64

	// LoadHeadroom is the maximum load difference (in effective load units) between
	// the best-cached instance and the least-loaded instance that still permits
	// exploit mode. If the cached instance exceeds the least-loaded by more than
	// this margin, fall back to explore mode to prevent queue concentration.
	// Range: [0, inf). Default: 2.
	LoadHeadroom int

	// ExploitWeights are the scorer weights used in exploit mode (cache-heavy).
	// If nil, defaults to prefix-affinity:5,queue-depth:1,kv-utilization:1.
	ExploitWeights []ScorerConfig
}

// DefaultAdaptiveConfig returns sensible defaults for exploit/explore routing.
func DefaultAdaptiveConfig() AdaptiveConfig {
	return AdaptiveConfig{
		ExploitThreshold: 0.3,
		LoadHeadroom:     5,
	}
}

// ValidateAdaptiveConfig returns an error if the config is invalid.
func ValidateAdaptiveConfig(cfg AdaptiveConfig) error {
	if cfg.ExploitThreshold <= 0 || cfg.ExploitThreshold > 1 ||
		math.IsNaN(cfg.ExploitThreshold) || math.IsInf(cfg.ExploitThreshold, 0) {
		return fmt.Errorf("ExploitThreshold must be in (0, 1], got %v", cfg.ExploitThreshold)
	}
	if cfg.LoadHeadroom < 0 {
		return fmt.Errorf("LoadHeadroom must be non-negative, got %d", cfg.LoadHeadroom)
	}
	return nil
}

// defaultExploitWeights returns cache-heavy scorer config for exploit mode.
func defaultExploitWeights() []ScorerConfig {
	return []ScorerConfig{
		{Name: "prefix-affinity", Weight: 5.0},
		{Name: "queue-depth", Weight: 1.0},
		{Name: "kv-utilization", Weight: 1.0},
	}
}

// AdaptiveWeightedScoring routes requests using an exploit/explore strategy with
// temporal workload tracking.
//
// The design addresses three adaptation axes:
//
//  1. Per-request: each request is classified as exploit or explore based on its
//     prefix cache match and the target instance's load.
//  2. Over time: an EMA (exponential moving average) of recent cache hit rates
//     tracks whether the current workload phase benefits from cache-aware routing.
//     When the EMA is low (independent workload), exploit decisions are suppressed.
//  3. Over workloads: the EMA automatically adapts to workload transitions —
//     when traffic shifts from prefix-heavy to independent (or vice versa),
//     the EMA tracks the change within ~50 requests (alpha=0.02).
//
// The exploit/explore decision:
//   - EXPLOIT: Strong prefix cache hit + instance not overloaded + recent workload
//     shows cache benefit (EMA above threshold). Routes directly to cached instance.
//   - EXPLORE: Otherwise, uses round-robin for perfect distribution uniformity.
//
// The LoadHeadroom parameter prevents the degenerate cascade (H21): even with
// a cache hit, if the target instance has substantially more load than the lightest
// instance, the policy falls back to explore mode.
type AdaptiveWeightedScoring struct {
	// Exploit mode: cache-aware scorers for scoring when a cache hit qualifies
	exploitScorers []scorerFunc
	exploitWeights []float64

	// Explore mode: round-robin counter for uniform distribution
	rrCounter int

	// Temporal tracking: EMA of recent cache hit ratios.
	// Tracks whether the current workload phase benefits from cache-aware routing.
	// Updated after every request. When low (<ExploitThreshold), exploit is suppressed
	// even for individual requests with cache hits, because the workload is in an
	// independent/non-prefix phase where concentration hurts.
	cacheHitEMA float64
	emaAlpha    float64 // EMA smoothing factor (default: 0.02 ≈ 50-request window)

	// Shared observers (called regardless of mode)
	observers []observerFunc

	config    AdaptiveConfig
	prefixIdx *PrefixCacheIndex // shared with prefix-affinity scorer, for cache probing
}

// Route implements RoutingPolicy for AdaptiveWeightedScoring.
func (aws *AdaptiveWeightedScoring) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("AdaptiveWeightedScoring.Route: empty snapshots")
	}

	// Determine exploit vs explore mode (per-request + temporal)
	mode, cacheRatio, bestCacheInst := aws.classifyRequest(req, snapshots)

	// Update temporal EMA with this request's cache opportunity
	aws.cacheHitEMA = aws.emaAlpha*cacheRatio + (1-aws.emaAlpha)*aws.cacheHitEMA

	var bestIdx int
	var scores map[string]float64

	if mode == "exploit" {
		// Exploit: route directly to the best-cached instance.
		// No scoring needed — classifyRequest already verified the cache hit is strong
		// and the instance isn't overloaded (within LoadHeadroom of least-loaded).
		for i, snap := range snapshots {
			if snap.ID == bestCacheInst {
				bestIdx = i
				break
			}
		}
	} else {
		// Explore: round-robin for perfect distribution uniformity
		bestIdx = aws.rrCounter % len(snapshots)
		aws.rrCounter++
	}

	// Notify observers of routing decision
	for _, obs := range aws.observers {
		obs(req, snapshots[bestIdx].ID)
	}

	return NewRoutingDecisionWithScores(
		snapshots[bestIdx].ID,
		fmt.Sprintf("adaptive-%s (cache=%.2f, cached=%s)",
			mode, cacheRatio, bestCacheInst),
		scores,
	)
}

// classifyRequest determines whether to exploit (cache-affinity) or explore (load-balance).
// Returns the mode, best cache match ratio, and best-cached instance ID.
func (aws *AdaptiveWeightedScoring) classifyRequest(
	req *Request, snapshots []RoutingSnapshot,
) (string, float64, string) {
	if aws.prefixIdx == nil || req == nil || len(req.InputTokens) == 0 {
		return "explore", 0.0, ""
	}

	hashes := aws.prefixIdx.ComputeBlockHashes(req.InputTokens)
	totalBlocks := len(hashes)
	if totalBlocks == 0 {
		return "explore", 0.0, ""
	}

	// Find the instance with the best prefix match
	bestMatch := 0
	bestInst := ""
	for _, snap := range snapshots {
		matched := aws.prefixIdx.MatchLength(hashes, snap.ID)
		if matched > bestMatch {
			bestMatch = matched
			bestInst = snap.ID
		}
	}

	cacheRatio := float64(bestMatch) / float64(totalBlocks)

	// Check threshold: is the cache hit strong enough to exploit?
	if cacheRatio < aws.config.ExploitThreshold {
		return "explore", cacheRatio, bestInst
	}

	// Temporal check: suppress exploit if recent workload doesn't benefit from caching.
	// This adapts to workload shifts — when traffic transitions from prefix-heavy
	// to independent, the EMA drops and exploit is suppressed, avoiding the
	// concentration overhead of cache-affinity routing during non-prefix phases.
	if aws.cacheHitEMA < aws.config.ExploitThreshold {
		return "explore", cacheRatio, bestInst
	}

	// Check load headroom: is the cached instance too overloaded?
	// Find the least-loaded instance for comparison
	minLoad := math.MaxInt
	cachedLoad := 0
	for _, snap := range snapshots {
		load := snap.EffectiveLoad()
		if load < minLoad {
			minLoad = load
		}
		if snap.ID == bestInst {
			cachedLoad = load
		}
	}

	if cachedLoad-minLoad > aws.config.LoadHeadroom {
		return "explore", cacheRatio, bestInst
	}

	// Check offloading pressure: avoid instances with high pending transfer latency
	// or KV thrashing. When a tiered KV cache is reloading blocks from CPU→GPU,
	// routing more requests there amplifies the reload penalty. Fall back to explore
	// if the cached instance has significant pending transfers.
	for _, snap := range snapshots {
		if snap.ID == bestInst {
			if snap.PendingTransferLatency > 0 || snap.KVThrashingRate > 0.1 {
				return "explore", cacheRatio, bestInst
			}
			break
		}
	}

	return "exploit", cacheRatio, bestInst
}
