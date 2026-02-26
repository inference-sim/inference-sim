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
	// Range: [0, inf). Default: 5.
	LoadHeadroom int

	// ExploreWeights are the scorer weights used in explore mode (load-balanced).
	// If nil, defaults to queue-depth:3,kv-utilization:2 (no prefix-affinity).
	ExploreWeights []ScorerConfig

	// ExploitWeights are the scorer weights used in exploit mode (cache-heavy).
	// If nil, defaults to prefix-affinity:5,queue-depth:1,kv-utilization:1.
	ExploitWeights []ScorerConfig
}

// DefaultAdaptiveConfig returns sensible defaults for exploit/explore routing.
func DefaultAdaptiveConfig() AdaptiveConfig {
	return AdaptiveConfig{
		ExploitThreshold: 0.3,
		LoadHeadroom:     0,
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

// AdaptiveWeightedScoring routes requests using an exploit/explore strategy.
//
// The core insight (from E2/CacheGen and BLIS experiment evidence H17, H23, H-Cross-Model):
// optimal routing weights depend on whether the current request can benefit from
// prefix caching. Rather than continuously adjusting weights, this policy makes
// a binary per-request decision:
//
//   - EXPLOIT: When a strong prefix cache hit exists on a specific instance AND
//     that instance isn't overloaded, route directly to that instance.
//   - EXPLORE: When no good cache hit exists or the cached instance is overloaded,
//     use round-robin distribution for perfect load uniformity.
//
// The key experimental finding motivating the explore=round-robin choice:
// round-robin consistently beats all score-based policies at moderate-to-high
// load because perfect distribution uniformity avoids the queue concentration
// that score-based routing introduces. Score-based explore (queue-depth, kv-util)
// introduces asymmetry that worsens TTFT.
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

	// Determine exploit vs explore mode
	mode, cacheRatio, bestCacheInst := aws.classifyRequest(req, snapshots)

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

	return "exploit", cacheRatio, bestInst
}
