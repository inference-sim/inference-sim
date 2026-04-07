package sim

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// ScorerConfig describes a named scorer with a weight for weighted routing.
type ScorerConfig struct {
	Name   string  `yaml:"name"`
	Weight float64 `yaml:"weight"`
}

// scorerFunc computes per-instance scores in [0,1] for a scoring dimension.
// Some scorers use a sub-range by design (e.g., load-aware scores in [0, 0.5]
// per llm-d semantics); weighted combination normalizes the effective contribution.
// The req parameter provides request metadata (e.g., InputTokens for prefix matching).
// Stateless scorers may ignore it.
type scorerFunc func(req *Request, snapshots []RoutingSnapshot) map[string]float64

// cacheQueryFn maps instance IDs to functions that return the count of
// consecutive cached prefix blocks for given tokens. Used by precise
// prefix cache scoring. Nil for sim-level tests without cluster instances.
type cacheQueryFn map[string]func([]int) int

// validScorerNames maps scorer names to validity. Unexported to prevent mutation (antipattern rule 8).
var validScorerNames = map[string]bool{
	"prefix-affinity":      true,
	"precise-prefix-cache": true,
	"no-hit-lru":           true,
	"queue-depth":          true,
	"kv-utilization":       true,
	"load-balance":         true,
	"active-requests":      true,
	"running-requests":     true,
	"load-aware":           true,
}

// IsValidScorer returns true if name is a recognized scorer.
func IsValidScorer(name string) bool { return validScorerNames[name] }

// ValidScorerNames returns sorted valid scorer names.
func ValidScorerNames() []string { return validNamesList(validScorerNames) }

// DefaultScorerConfigs returns the default scorer configuration for weighted routing.
// Default profile: precise-prefix-cache:2, queue-depth:1, kv-utilization:1 (llm-d parity).
func DefaultScorerConfigs() []ScorerConfig {
	return []ScorerConfig{
		{Name: "precise-prefix-cache", Weight: 2.0},
		{Name: "queue-depth", Weight: 1.0},
		{Name: "kv-utilization", Weight: 1.0},
	}
}

// ParseScorerConfigs parses a comma-separated string of "name:weight" pairs.
// Returns nil for empty input. Returns error for invalid names, non-positive weights,
// NaN, Inf, or malformed input.
func ParseScorerConfigs(s string) ([]ScorerConfig, error) {
	if s == "" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	configs := make([]ScorerConfig, 0, len(parts))
	seen := make(map[string]bool, len(parts))
	for _, part := range parts {
		kv := strings.SplitN(strings.TrimSpace(part), ":", 2)
		if len(kv) != 2 {
			return nil, fmt.Errorf("invalid scorer config %q (expected name:weight)", strings.TrimSpace(part))
		}
		name := strings.TrimSpace(kv[0])
		if !IsValidScorer(name) {
			return nil, fmt.Errorf("unknown scorer %q; valid: %s", name, strings.Join(ValidScorerNames(), ", "))
		}
		if seen[name] {
			return nil, fmt.Errorf("duplicate scorer %q; each scorer may appear at most once", name)
		}
		seen[name] = true
		weight, err := strconv.ParseFloat(strings.TrimSpace(kv[1]), 64)
		if err != nil {
			return nil, fmt.Errorf("invalid weight for scorer %q: %w", name, err)
		}
		if weight <= 0 || math.IsNaN(weight) || math.IsInf(weight, 0) {
			return nil, fmt.Errorf("scorer %q weight must be a finite positive number, got %v", name, weight)
		}
		configs = append(configs, ScorerConfig{Name: name, Weight: weight})
	}
	return configs, nil
}

// normalizeScorerWeights returns weights normalized to sum to 1.0.
// Panics if total weight is zero (should be prevented by validation).
func normalizeScorerWeights(configs []ScorerConfig) []float64 {
	total := 0.0
	for _, c := range configs {
		total += c.Weight
	}
	if total <= 0 {
		panic(fmt.Sprintf("scorer weights sum to %f; must be positive", total))
	}
	weights := make([]float64, len(configs))
	for i, c := range configs {
		weights[i] = c.Weight / total
	}
	return weights
}

// newScorerWithObserver creates a scorer function and optional observer for a named scorer.
// Returns (scorer, observer) where observer is nil for stateless scorers.
// blockSize is used by stateful scorers (e.g., prefix-affinity) for block hash computation.
// Panics on unknown name (validation should catch this before reaching here).
func newScorerWithObserver(name string, blockSize int, cacheFn cacheQueryFn) (scorerFunc, observerFunc) {
	switch name {
	case "prefix-affinity":
		return newPrefixAffinityScorer(blockSize)
	case "precise-prefix-cache":
		return newPrecisePrefixCacheScorer(cacheFn)
	case "no-hit-lru":
		return newNoHitLRUScorer(cacheFn)
	case "queue-depth":
		return scoreQueueDepth, nil
	case "kv-utilization":
		return scoreKVUtilization, nil
	case "load-balance":
		return scoreLoadBalance, nil
	case "active-requests":
		return scoreActiveRequests, nil
	case "running-requests":
		return scoreRunningRequests, nil
	case "load-aware":
		return scoreLoadAware, nil
	default:
		panic(fmt.Sprintf("unknown scorer %q", name))
	}
}

// scoreQueueDepth computes per-instance queue depth scores using min-max normalization.
// Lower effective load → higher score. All-equal loads → all score 1.0.
// Matches llm-d's queue-scorer semantics.
//
// Signal freshness (R17, INV-7):
//
//	Reads: EffectiveLoad() = QueueDepth (Periodic when interval>0, else Immediate) +
//	       BatchSize (same) + InFlightRequests (synchronous).
//	The synchronous InFlightRequests term compensates for Periodic staleness.
func scoreQueueDepth(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	minLoad, maxLoad := math.MaxInt, 0
	for _, snap := range snapshots {
		load := snap.EffectiveLoad()
		if load < minLoad {
			minLoad = load
		}
		if load > maxLoad {
			maxLoad = load
		}
	}
	for _, snap := range snapshots {
		if maxLoad == minLoad {
			scores[snap.ID] = 1.0
		} else {
			load := snap.EffectiveLoad()
			scores[snap.ID] = float64(maxLoad-load) / float64(maxLoad-minLoad)
		}
	}
	return scores
}

// scoreKVUtilization computes per-instance KV utilization scores.
// Lower utilization → higher score: score = 1 - KVUtilization.
// Matches llm-d's kv-cache-utilization-scorer semantics.
//
// Signal freshness (R17, INV-7):
//
//	Reads: KVUtilization (Periodic when interval>0, else Immediate).
//	WARNING: At high request rates with large intervals, this signal can be significantly stale.
//	Pair with a load-aware scorer (e.g., queue-depth) for robust routing.
//	See H3 experiment: 200x worse distribution uniformity at rate=5000.
func scoreKVUtilization(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		scores[snap.ID] = 1.0 - snap.KVUtilization
	}
	return scores
}

// scoreLoadBalance computes per-instance load balance scores using inverse transform.
// Lower effective load → higher score: score = 1/(1 + effectiveLoad).
// BLIS-native formula preserving absolute load differences (alternative to min-max).
//
// Signal freshness (R17, INV-7):
//
//	Reads: EffectiveLoad() — same as scoreQueueDepth (synchronous + Periodic composite).
func scoreLoadBalance(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		scores[snap.ID] = 1.0 / (1.0 + float64(snap.EffectiveLoad()))
	}
	return scores
}

// scoreActiveRequests computes per-instance scores based on in-flight request count.
// Instances with zero in-flight always score 1.0. Non-zero instances use max-only
// normalization: (maxCount - count) / maxCount. When all instances have the same
// non-zero count, all score 0.0 (no differentiation) — contrast with running-requests
// which uses min-max normalization and scores 1.0 for all-equal. This asymmetry is
// intentional: it matches llm-d's active-request-scorer (active_request.go:193-230).
//
// Signal freshness (R17, INV-7):
//
//	Reads: InFlightRequests (synchronous — updated on dispatch/completion events).
func scoreActiveRequests(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	maxCount := 0
	for _, snap := range snapshots {
		if snap.InFlightRequests > maxCount {
			maxCount = snap.InFlightRequests
		}
	}
	for _, snap := range snapshots {
		if snap.InFlightRequests == 0 || maxCount == 0 {
			scores[snap.ID] = 1.0
		} else {
			scores[snap.ID] = float64(maxCount-snap.InFlightRequests) / float64(maxCount)
		}
	}
	return scores
}

// scoreRunningRequests computes per-instance scores based on running (in-batch) request count.
// Uses min-max normalization: (maxBatch - batch) / (maxBatch - minBatch). All equal = 1.0.
// Matches GIE's running-requests-size-scorer semantics (runningrequest.go:99).
//
// Signal freshness (R17, INV-7):
//
//	Reads: BatchSize (Periodic when interval>0, else Immediate).
func scoreRunningRequests(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	minBatch, maxBatch := math.MaxInt, 0
	for _, snap := range snapshots {
		if snap.BatchSize < minBatch {
			minBatch = snap.BatchSize
		}
		if snap.BatchSize > maxBatch {
			maxBatch = snap.BatchSize
		}
	}
	for _, snap := range snapshots {
		if maxBatch == minBatch {
			scores[snap.ID] = 1.0
		} else {
			scores[snap.ID] = float64(maxBatch-snap.BatchSize) / float64(maxBatch-minBatch)
		}
	}
	return scores
}

// loadAwareQueueThreshold is the default queue depth threshold for the load-aware scorer.
// Matches llm-d's QueueThresholdDefault (load_aware.go:42). Queue depths at or above
// this value score 0.0.
const loadAwareQueueThreshold = 128

// scoreLoadAware computes per-instance scores based on waiting queue depth with a
// linear threshold-capped formula. Score range: [0, 0.5].
// Empty queue scores 0.5 (maximum). Non-zero queue: 0.5 * (1 - queue/threshold),
// where queue depth is clamped to loadAwareQueueThreshold. At-or-above threshold = 0.0.
// Matches llm-d's load-aware-scorer semantics (load_aware.go:83-99).
//
// Signal freshness (R17, INV-7):
//
//	Reads: QueueDepth (Periodic when interval>0, else Immediate).
func scoreLoadAware(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		if snap.QueueDepth == 0 {
			scores[snap.ID] = 0.5
		} else {
			clamped := snap.QueueDepth
			if clamped > loadAwareQueueThreshold {
				clamped = loadAwareQueueThreshold
			}
			scores[snap.ID] = 0.5 * (1.0 - float64(clamped)/float64(loadAwareQueueThreshold))
		}
	}
	return scores
}
