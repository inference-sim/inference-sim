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
// The req parameter provides request metadata (e.g., InputTokens for prefix matching).
// Stateless scorers may ignore it.
type scorerFunc func(req *Request, snapshots []RoutingSnapshot) map[string]float64

// defaultBlockSize is the default block size for the prefix cache index.
// Matches the most common KV cache block size. Used when constructing
// the prefix-affinity scorer without explicit configuration.
const defaultBlockSize = 16

// validScorerNames maps scorer names to validity. Unexported to prevent mutation (antipattern rule 8).
var validScorerNames = map[string]bool{
	"prefix-affinity": true,
	"queue-depth":     true,
	"kv-utilization":  true,
	"load-balance":    true,
}

// IsValidScorer returns true if name is a recognized scorer.
func IsValidScorer(name string) bool { return validScorerNames[name] }

// ValidScorerNames returns sorted valid scorer names.
func ValidScorerNames() []string { return validNamesList(validScorerNames) }

// DefaultScorerConfigs returns the default scorer configuration for weighted routing.
// Default profile: prefix-affinity:3, queue-depth:2, kv-utilization:2 (llm-d parity).
func DefaultScorerConfigs() []ScorerConfig {
	return []ScorerConfig{
		{Name: "prefix-affinity", Weight: 3.0},
		{Name: "queue-depth", Weight: 2.0},
		{Name: "kv-utilization", Weight: 2.0},
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
// blockSize is used by stateful scorers (prefix-affinity) for block hash computation.
// Panics on unknown name (validation should catch this before reaching here).
func newScorerWithObserver(name string, blockSize int) (scorerFunc, observerFunc) {
	switch name {
	case "prefix-affinity":
		return newPrefixAffinityScorer(blockSize)
	case "queue-depth":
		return scoreQueueDepth, nil
	case "kv-utilization":
		return scoreKVUtilization, nil
	case "load-balance":
		return scoreLoadBalance, nil
	default:
		panic(fmt.Sprintf("unknown scorer %q", name))
	}
}

// scoreQueueDepth computes per-instance queue depth scores using min-max normalization.
// Lower effective load → higher score. All-equal loads → all score 1.0.
// Matches llm-d's queue-scorer semantics.
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
func scoreLoadBalance(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		scores[snap.ID] = 1.0 / (1.0 + float64(snap.EffectiveLoad()))
	}
	return scores
}
