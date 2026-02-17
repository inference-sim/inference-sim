package cluster

import (
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// Distribution captures statistical summary of a metric.
type Distribution struct {
	Mean  float64
	P50   float64
	P95   float64
	P99   float64
	Min   float64
	Max   float64
	Count int
}

// NewDistribution computes a Distribution from raw values.
// Returns zero-value Distribution for empty input.
func NewDistribution(values []float64) Distribution {
	if len(values) == 0 {
		return Distribution{}
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	sum := 0.0
	for _, v := range sorted {
		sum += v
	}

	return Distribution{
		Mean:  sum / float64(len(sorted)),
		P50:   percentile(sorted, 50),
		P95:   percentile(sorted, 95),
		P99:   percentile(sorted, 99),
		Min:   sorted[0],
		Max:   sorted[len(sorted)-1],
		Count: len(sorted),
	}
}

// percentile computes the p-th percentile using linear interpolation.
// Input must be sorted. Returns raw value (not converted to ms).
func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if len(sorted) == 1 {
		return sorted[0]
	}
	rank := p / 100.0 * float64(len(sorted)-1)
	lower := int(math.Floor(rank))
	upper := int(math.Ceil(rank))
	if lower == upper {
		return sorted[lower]
	}
	frac := rank - float64(lower)
	return sorted[lower] + frac*(sorted[upper]-sorted[lower])
}

// RawMetrics holds cluster-level metrics aggregated after simulation.
type RawMetrics struct {
	// Latency distributions (in ticks)
	TTFT Distribution
	E2E  Distribution

	// Throughput
	RequestsPerSec float64
	TokensPerSec   float64

	// Anomaly counters
	PriorityInversions int
	HOLBlockingEvents  int
	RejectedRequests   int
}

// CollectRawMetrics builds RawMetrics from aggregated and per-instance metrics.
// perInstance is optional (may be nil for anomaly-free collection).
func CollectRawMetrics(aggregated *sim.Metrics, perInstance []*sim.Metrics, rejectedRequests int) *RawMetrics {
	raw := &RawMetrics{
		RejectedRequests: rejectedRequests,
	}

	// Latency distributions
	ttftValues := mapValues(aggregated.RequestTTFTs)
	raw.TTFT = NewDistribution(ttftValues)

	e2eValues := mapValues(aggregated.RequestE2Es)
	raw.E2E = NewDistribution(e2eValues)

	// Throughput
	if aggregated.SimEndedTime > 0 && aggregated.CompletedRequests > 0 {
		durationSec := float64(aggregated.SimEndedTime) / 1e6
		raw.RequestsPerSec = float64(aggregated.CompletedRequests) / durationSec
		raw.TokensPerSec = float64(aggregated.TotalOutputTokens) / durationSec
	}

	return raw
}

// mapValues extracts values from a map into a slice.
func mapValues(m map[string]float64) []float64 {
	values := make([]float64, 0, len(m))
	for _, v := range m {
		values = append(values, v)
	}
	return values
}
