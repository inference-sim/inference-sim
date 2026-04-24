package workload

import (
	"math/rand"

	"github.com/inference-sim/inference-sim/sim"
)

// defaultPrefixLength is the number of shared prefix tokens for prefix groups.

const defaultPrefixLength = 50

// normalizeRateFractions normalizes client rate fractions and returns per-client
// rates in requests/microsecond.
//
// When no client has lifecycle windows, fractions are normalized globally (sum to 1.0).
// When lifecycle windows are present, fractions are normalized per-phase: each
// client's fraction is divided by the sum of fractions of all co-active clients
// (those whose lifecycle windows overlap). This ensures aggregate_rate is achieved
// during every active phase, not diluted by clients in non-overlapping phases.
//
// Clients without lifecycle windows are "always-on" and overlap with everything.
//
// Limitation: always-on clients compute a single rate using co-active sums across
// ALL phases they overlap with. When combined with multiple non-overlapping phased
// clients, the always-on client's rate is diluted by the union of all phases, so
// per-phase totals may be less than aggregate_rate. A fully phase-aware algorithm
// (computing separate rates per discrete time interval) would fix this but is
// significantly more complex.
func normalizeRateFractions(clients []ClientSpec, aggregateRate float64) []float64 {
	// Fast path: no lifecycle windows → global normalization (original behavior).
	hasLifecycle := false
	for i := range clients {
		if clients[i].Lifecycle != nil && len(clients[i].Lifecycle.Windows) > 0 {
			hasLifecycle = true
			break
		}
	}

	if !hasLifecycle {
		totalFraction := 0.0
		for i := range clients {
			totalFraction += clients[i].RateFraction
		}
		if totalFraction == 0 {
			return make([]float64, len(clients))
		}
		rates := make([]float64, len(clients))
		for i := range clients {
			rates[i] = aggregateRate * (clients[i].RateFraction / totalFraction) / 1e6
		}
		return rates
	}

	// Lifecycle-aware path: normalize each client's fraction by the sum of
	// fractions of all co-active clients (clients whose windows overlap).
	rates := make([]float64, len(clients))
	for i := range clients {
		if clients[i].RateFraction <= 0 {
			continue
		}
		coActiveSum := 0.0
		for j := range clients {
			if clients[j].RateFraction <= 0 {
				continue
			}
			if windowsOverlap(clients[i].Lifecycle, clients[j].Lifecycle) {
				coActiveSum += clients[j].RateFraction
			}
		}
		if coActiveSum == 0 {
			continue
		}
		rates[i] = aggregateRate * (clients[i].RateFraction / coActiveSum) / 1e6
	}
	return rates
}

// windowsOverlap reports whether two clients' lifecycle windows overlap.
// A nil or empty lifecycle means "always on" — overlaps with everything.
func windowsOverlap(a, b *LifecycleSpec) bool {
	aOn := a == nil || len(a.Windows) == 0
	bOn := b == nil || len(b.Windows) == 0
	if aOn || bOn {
		return true
	}
	for _, wa := range a.Windows {
		for _, wb := range b.Windows {
			if wa.StartUs < wb.EndUs && wb.StartUs < wa.EndUs {
				return true
			}
		}
	}
	return false
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
