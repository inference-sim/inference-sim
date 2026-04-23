package workload

import (
	"math"
	"math/rand"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// defaultPrefixLength is the number of shared prefix tokens for prefix groups.

const defaultPrefixLength = 50

// normalizeRateFractions normalizes client rate fractions so that, at any
// point in simulated time, the sum of rates of the clients active at that
// moment equals aggregateRate. The normalization divisor is the maximum
// RateFraction sum across all moments — i.e. across the peak-concurrency
// phase — not the unconditional sum across every client.
//
// This matters when clients declare non-overlapping lifecycle windows. For a
// two-phase workload where phase 1 uses fractions 0.7+0.3 and phase 2 uses
// 1.0, the unconditional sum is 2.0 and halves every rate, producing ~half
// the intended aggregateRate during each phase (issue #1144). Using the
// max-overlap sum (1.0) keeps each phase at aggregateRate.
//
// Returns per-client rates in requests/microsecond.
func normalizeRateFractions(clients []ClientSpec, aggregateRate float64) []float64 {
	divisor := maxOverlappingFractionSum(clients)
	if divisor == 0 {
		return make([]float64, len(clients))
	}
	rates := make([]float64, len(clients))
	for i := range clients {
		rates[i] = aggregateRate * (clients[i].RateFraction / divisor) / 1e6
	}
	return rates
}

// maxOverlappingFractionSum sweeps the client lifecycle timeline and returns
// the largest RateFraction sum over any sub-interval. Clients without a
// Lifecycle spec are treated as active from t=0 to t=+∞. If no client is
// ever active (or all fractions are zero) the function returns 0 — callers
// are expected to short-circuit in that case.
func maxOverlappingFractionSum(clients []ClientSpec) float64 {
	type event struct {
		t     int64
		delta float64
	}
	events := make([]event, 0, 2*len(clients))
	hasUnbounded := false
	unboundedSum := 0.0
	for i := range clients {
		frac := clients[i].RateFraction
		if frac == 0 {
			continue
		}
		if clients[i].Lifecycle == nil || len(clients[i].Lifecycle.Windows) == 0 {
			hasUnbounded = true
			unboundedSum += frac
			continue
		}
		for _, w := range clients[i].Lifecycle.Windows {
			if w.EndUs <= w.StartUs {
				continue
			}
			events = append(events, event{t: w.StartUs, delta: frac})
			events = append(events, event{t: w.EndUs, delta: -frac})
		}
	}
	if len(events) == 0 {
		return unboundedSum
	}
	sort.Slice(events, func(i, j int) bool {
		if events[i].t != events[j].t {
			return events[i].t < events[j].t
		}
		// Windows are half-open [start, end). At a tied instant where
		// one window ends and another begins, process the removal first
		// so back-to-back phases do not falsely register a moment of
		// concurrent overlap.
		return events[i].delta < events[j].delta
	})
	active := unboundedSum
	peak := unboundedSum
	if hasUnbounded && peak == 0 {
		peak = 0
	}
	for _, e := range events {
		active += e.delta
		if active > peak {
			peak = active
		}
	}
	if math.IsNaN(peak) || peak < 0 {
		return 0
	}
	return peak
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
