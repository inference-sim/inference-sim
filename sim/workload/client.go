package workload

import (
	"math/rand"

	"github.com/inference-sim/inference-sim/sim"
)

// DefaultPrefixLength is the number of shared prefix tokens for prefix groups.

const defaultPrefixLength = 50

// normalizeRateFractions normalizes client rate fractions to sum to 1.0.
// Returns per-client rates in requests/microsecond.
func normalizeRateFractions(clients []ClientSpec, aggregateRate float64) []float64 {
	totalFraction := 0.0
	for i := range clients {
		totalFraction += clients[i].RateFraction
	}
	if totalFraction == 0 {
		return make([]float64, len(clients))
	}
	rates := make([]float64, len(clients))
	for i := range clients {
		normalizedFraction := clients[i].RateFraction / totalFraction
		rates[i] = aggregateRate * normalizedFraction / 1e6 // convert req/sec to req/µs
	}
	return rates
}

// generatePrefixTokens creates shared prefix token sequences per prefix group.
// Clients in the same group get the same prefix tokens. The length is determined
// by the first client in the group that specifies prefix_length; others in the
// same group inherit it. If no client specifies prefix_length, defaultPrefixLength is used.
func generatePrefixTokens(clients []ClientSpec, rng *rand.Rand) map[string][]int {
	prefixes := make(map[string][]int)
	for i := range clients {
		group := clients[i].PrefixGroup
		if group == "" {
			continue
		}
		if _, exists := prefixes[group]; !exists {
			length := defaultPrefixLength
			if clients[i].PrefixLength > 0 {
				length = clients[i].PrefixLength
			}
			prefixes[group] = sim.GenerateRandomTokenIDs(rng, length)
		}
	}
	return prefixes
}
