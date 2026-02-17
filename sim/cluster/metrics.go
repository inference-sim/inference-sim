package cluster

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
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

	// Anomaly detection
	if perInstance != nil {
		raw.PriorityInversions = detectPriorityInversions(perInstance)
		raw.HOLBlockingEvents = detectHOLBlocking(perInstance)
	}

	return raw
}

// detectPriorityInversions counts priority inversion events from per-instance metrics.
// Simplified PR9 heuristic: counts pairs where an earlier-arriving request has
// worse E2E than a later-arriving request (with 2× threshold).
// Full decision-trace-based detection deferred to PR13.
func detectPriorityInversions(perInstance []*sim.Metrics) int {
	count := 0
	for _, m := range perInstance {
		if len(m.Requests) < 2 {
			continue
		}
		type reqInfo struct {
			arrived float64
			e2e     float64
		}
		var reqs []reqInfo
		for id, rm := range m.Requests {
			if e2e, ok := m.RequestE2Es[id]; ok {
				reqs = append(reqs, reqInfo{arrived: rm.ArrivedAt, e2e: e2e})
			}
		}
		sort.Slice(reqs, func(i, j int) bool {
			return reqs[i].arrived < reqs[j].arrived
		})
		for i := 0; i < len(reqs)-1; i++ {
			for j := i + 1; j < len(reqs); j++ {
				if reqs[i].e2e > reqs[j].e2e*2.0 {
					count++
				}
			}
		}
	}
	return count
}

// detectHOLBlocking counts head-of-line blocking events from per-instance metrics.
// HOL blocking is detected when any instance's average queue depth exceeds
// 2× the mean average queue depth across all instances.
func detectHOLBlocking(perInstance []*sim.Metrics) int {
	if len(perInstance) < 2 {
		return 0
	}

	// Only include instances with queue depth samples in the mean calculation
	var avgDepths []float64
	totalAvg := 0.0
	for _, m := range perInstance {
		if len(m.NumWaitQRequests) == 0 {
			continue
		}
		sum := 0
		for _, d := range m.NumWaitQRequests {
			sum += d
		}
		avg := float64(sum) / float64(len(m.NumWaitQRequests))
		avgDepths = append(avgDepths, avg)
		totalAvg += avg
	}

	if len(avgDepths) < 2 {
		return 0 // need at least 2 instances with samples
	}
	meanAvg := totalAvg / float64(len(avgDepths))

	count := 0
	if meanAvg > 0 {
		for _, avg := range avgDepths {
			if avg > 2.0*meanAvg {
				count++
			}
		}
	}
	return count
}

// mapValues extracts values from a map into a slice.
func mapValues(m map[string]float64) []float64 {
	vals := make([]float64, 0, len(m))
	for _, v := range m {
		vals = append(vals, v)
	}
	return vals
}

// Reference scales for normalizing metrics to [0,1] range.
// Without reference scales, throughput (raw value ~100) dominates latency (1/(1+5000) ≈ 0.0002)
// by 500,000×, making multi-objective optimization impossible.
const (
	referenceRPS   = 100.0  // 100 requests/sec as reference throughput
	referenceTicks = 1000.0 // 1ms (1000 ticks) as reference latency
)

// FitnessResult holds the computed fitness score and per-component breakdown.
type FitnessResult struct {
	Score      float64            // Weighted sum of normalized metric components
	Components map[string]float64 // Per-component normalized scores before weighting
}

// ComputeFitness computes a weighted fitness score from RawMetrics.
// All metrics are normalized to [0,1] range before weighting:
// - Throughput: value / (value + referenceRPS) — higher is better, saturates at 1.0
// - Latency: 1.0 / (1.0 + value/referenceTicks) — lower is better, 1ms → 0.5
// Unknown weight keys are logged as warnings and ignored (EC-1).
func ComputeFitness(metrics *RawMetrics, weights map[string]float64) *FitnessResult {
	result := &FitnessResult{
		Components: make(map[string]float64, len(weights)),
	}

	for key, weight := range weights {
		value, ok := extractMetric(metrics, key)
		if !ok {
			logrus.Warnf("ComputeFitness: unknown metric key %q, ignoring", key)
			continue
		}
		result.Components[key] = value
		result.Score += value * weight
	}

	return result
}

// extractMetric returns a normalized [0,1] metric value for the given key.
// Throughput: value / (value + reference). Latency: 1 / (1 + value/reference).
// Both formulas are safe for zero values: 0/(0+ref)=0, 1/(1+0/ref)=1.
// Returns (value, true) on success, (0, false) for unknown keys.
func extractMetric(m *RawMetrics, key string) (float64, bool) {
	switch key {
	// Higher is better — normalized via value / (value + reference)
	case "throughput":
		return m.RequestsPerSec / (m.RequestsPerSec + referenceRPS), true
	case "tokens_per_sec":
		return m.TokensPerSec / (m.TokensPerSec + referenceRPS), true
	// Lower is better — normalized via 1 / (1 + value/reference)
	case "p99_ttft":
		return 1.0 / (1.0 + m.TTFT.P99/referenceTicks), true
	case "p50_ttft":
		return 1.0 / (1.0 + m.TTFT.P50/referenceTicks), true
	case "mean_ttft":
		return 1.0 / (1.0 + m.TTFT.Mean/referenceTicks), true
	case "p99_e2e":
		return 1.0 / (1.0 + m.E2E.P99/referenceTicks), true
	case "p50_e2e":
		return 1.0 / (1.0 + m.E2E.P50/referenceTicks), true
	case "mean_e2e":
		return 1.0 / (1.0 + m.E2E.Mean/referenceTicks), true
	default:
		return 0, false
	}
}

// ParseFitnessWeights parses a "key:value,key:value" string into a weight map.
// Returns empty map for empty input (EC-2). Returns error for malformed entries.
func ParseFitnessWeights(s string) (map[string]float64, error) {
	if s == "" {
		return map[string]float64{}, nil
	}
	weights := make(map[string]float64)
	for _, pair := range strings.Split(s, ",") {
		pair = strings.TrimSpace(pair)
		parts := strings.SplitN(pair, ":", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid fitness weight %q: expected key:value", pair)
		}
		key := strings.TrimSpace(parts[0])
		val, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		if err != nil {
			return nil, fmt.Errorf("invalid fitness weight value for %q: %w", key, err)
		}
		weights[key] = val
	}
	return weights, nil
}
