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

// sortedKeys returns the keys of a map[string]float64 in sorted order.
// Used to ensure deterministic float accumulation across map iterations.
func sortedKeys(m map[string]float64) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

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

// SLOMetrics holds per-SLO-class latency distributions.
type SLOMetrics struct {
	TTFT Distribution
	E2E  Distribution
}

// RawMetrics holds cluster-level metrics aggregated after simulation.
type RawMetrics struct {
	// Latency distributions (in ticks)
	TTFT Distribution
	E2E  Distribution

	// Per-SLO-class distributions (PR10: keyed by SLOClass string)
	PerSLOClass map[string]*SLOMetrics

	// Throughput
	RequestsPerSec float64
	TokensPerSec   float64

	// Anomaly counters
	PriorityInversions int
	HOLBlockingEvents  int
	RejectedRequests   int

	// KV cache metrics (PR12)
	CacheHitRate    float64
	PreemptionRate  float64
	KVThrashingRate float64
}

// CollectRawMetrics builds RawMetrics from aggregated and per-instance metrics.
// perInstance is optional (may be nil for anomaly-free collection).
// priorityPolicy controls whether priority inversion detection runs:
// when "constant" or "" (both map to ConstantPriority), inversions are
// suppressed (always 0) since all requests share the same priority and
// E2E differences reflect workload variance, not unfairness.
func CollectRawMetrics(aggregated *sim.Metrics, perInstance []*sim.Metrics, rejectedRequests int, priorityPolicy string) *RawMetrics {
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
		raw.PriorityInversions = detectPriorityInversions(perInstance, priorityPolicy)
		raw.HOLBlockingEvents = detectHOLBlocking(perInstance)

		// KV cache metrics (PR12)
		totalPreemptions := int64(0)
		cacheHitSum := 0.0
		thrashingSum := 0.0
		count := 0
		for _, m := range perInstance {
			totalPreemptions += m.PreemptionCount
			cacheHitSum += m.CacheHitRate
			thrashingSum += m.KVThrashingRate
			count++
		}
		if aggregated.CompletedRequests > 0 {
			raw.PreemptionRate = float64(totalPreemptions) / float64(aggregated.CompletedRequests)
		}
		if count > 0 {
			raw.CacheHitRate = cacheHitSum / float64(count)
			raw.KVThrashingRate = thrashingSum / float64(count)
		}
	}

	return raw
}

// detectPriorityInversions counts priority inversion events from per-instance metrics.
// PR9 heuristic: counts pairs where an earlier-arriving request has
// worse E2E than a later-arriving request (with 2× threshold).
// Requests are grouped by SLO class before comparison — cross-class differences
// reflect workload size heterogeneity, not scheduling unfairness. (#292, R20)
// When priorityPolicy is "constant" or "" (both map to ConstantPriority),
// returns 0 — all requests share the same priority.
func detectPriorityInversions(perInstance []*sim.Metrics, priorityPolicy string) int {
	if priorityPolicy == "constant" || priorityPolicy == "" {
		return 0
	}
	count := 0
	for _, m := range perInstance {
		if len(m.Requests) < 2 {
			continue
		}
		type reqInfo struct {
			arrived float64
			e2e     float64
		}
		// Group requests by SLO class to avoid cross-class false positives (#292, R20).
		// Requests in different SLO classes have naturally different E2E due to
		// workload size differences, not scheduling unfairness.
		groups := make(map[string][]reqInfo)
		skippedCount := 0
		for id, rm := range m.Requests {
			if e2e, ok := m.RequestE2Es[id]; ok {
				sloClass := rm.SLOClass
				if sloClass == "" {
					sloClass = "default"
				}
				groups[sloClass] = append(groups[sloClass], reqInfo{arrived: rm.ArrivedAt, e2e: e2e})
			} else {
				skippedCount++
			}
		}
		if skippedCount > 0 {
			logrus.Warnf("detectPriorityInversions: %d requests missing E2E data, skipped", skippedCount)
		}
		// Check inversions within each SLO class
		for _, reqs := range groups {
			if len(reqs) < 2 {
				continue
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
	}
	return count
}

// detectHOLBlocking counts head-of-line blocking events from per-instance metrics.
// HOL blocking is detected when any instance's average queue depth exceeds
// 2× the mean average queue depth across all instances.
// Instances with no traffic are included with avg=0.0 — an instance receiving
// 0 requests while a sibling receives 500 IS HOL blocking. (#291, R20)
func detectHOLBlocking(perInstance []*sim.Metrics) int {
	if len(perInstance) < 2 {
		return 0
	}

	// Include ALL instances in the comparison. Instances with no traffic
	// contribute avg=0.0, which correctly lowers the mean and makes
	// concentrated instances stand out. (#291, R20)
	avgDepths := make([]float64, 0, len(perInstance))
	totalAvg := 0.0
	for _, m := range perInstance {
		avg := 0.0
		if len(m.NumWaitQRequests) > 0 {
			sum := 0
			for _, d := range m.NumWaitQRequests {
				sum += d
			}
			avg = float64(sum) / float64(len(m.NumWaitQRequests))
		}
		avgDepths = append(avgDepths, avg)
		totalAvg += avg
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

// ComputePerSLODistributions builds per-SLO-class latency distributions.
// Filters requests by their SLOClass field and computes distributions per class.
func ComputePerSLODistributions(aggregated *sim.Metrics) map[string]*SLOMetrics {
	// Group TTFT and E2E values by SLO class
	ttftByClass := make(map[string][]float64)
	e2eByClass := make(map[string][]float64)

	droppedTTFT := 0
	for reqID, ttft := range aggregated.RequestTTFTs {
		req, ok := aggregated.Requests[reqID]
		if !ok {
			droppedTTFT++
			continue
		}
		sloClass := req.SLOClass
		if sloClass == "" {
			sloClass = "default"
		}
		ttftByClass[sloClass] = append(ttftByClass[sloClass], ttft)
	}
	if droppedTTFT > 0 {
		logrus.Warnf("ComputePerSLODistributions: %d requests in RequestTTFTs missing from Requests map", droppedTTFT)
	}
	droppedE2E := 0
	for reqID, e2e := range aggregated.RequestE2Es {
		req, ok := aggregated.Requests[reqID]
		if !ok {
			droppedE2E++
			continue
		}
		sloClass := req.SLOClass
		if sloClass == "" {
			sloClass = "default"
		}
		e2eByClass[sloClass] = append(e2eByClass[sloClass], e2e)
	}
	if droppedE2E > 0 {
		logrus.Warnf("ComputePerSLODistributions: %d requests in RequestE2Es missing from Requests map", droppedE2E)
	}

	result := make(map[string]*SLOMetrics)
	// Collect all classes from both maps
	allClasses := make(map[string]bool)
	for k := range ttftByClass {
		allClasses[k] = true
	}
	for k := range e2eByClass {
		allClasses[k] = true
	}
	for cls := range allClasses {
		result[cls] = &SLOMetrics{
			TTFT: NewDistribution(ttftByClass[cls]),
			E2E:  NewDistribution(e2eByClass[cls]),
		}
	}
	return result
}

// SLOAttainment computes the fraction of requests meeting their SLO target.
// targets maps SLO class to max acceptable E2E latency (in ticks).
// Returns a value in [0.0, 1.0].
// Requests in RequestE2Es that are missing from Requests map are counted
// as SLO violations (conservative: missing data = violation).
func SLOAttainment(aggregated *sim.Metrics, targets map[string]float64) float64 {
	if len(aggregated.RequestE2Es) == 0 {
		return 0
	}
	met := 0
	total := 0
	droppedCount := 0
	for reqID, e2e := range aggregated.RequestE2Es {
		total++
		req, ok := aggregated.Requests[reqID]
		if !ok {
			droppedCount++
			continue // counted in total but not in met (= violation)
		}
		sloClass := req.SLOClass
		if target, ok := targets[sloClass]; ok {
			if e2e <= target {
				met++
			}
		} else {
			// No target for this class = always meets SLO
			met++
		}
	}
	if droppedCount > 0 {
		logrus.Warnf("SLOAttainment: %d requests in RequestE2Es missing from Requests map (counted as violations)", droppedCount)
	}
	if total == 0 {
		return 0
	}
	return float64(met) / float64(total)
}

// JainFairnessIndex computes the Jain's fairness index across tenant throughputs.
// Formula: (Σxi)² / (N * Σxi²) where xi = per-tenant throughput.
// Returns a value in [1/N, 1.0] where 1.0 means perfect fairness.
func JainFairnessIndex(throughputs map[string]float64) float64 {
	n := float64(len(throughputs))
	if n == 0 {
		return 0
	}
	sumX := 0.0
	sumX2 := 0.0
	for _, k := range sortedKeys(throughputs) {
		x := throughputs[k]
		sumX += x
		sumX2 += x * x
	}
	if sumX2 == 0 {
		return 0
	}
	return (sumX * sumX) / (n * sumX2)
}

// mapValues extracts values from a map into a slice.
func mapValues(m map[string]float64) []float64 {
	vals := make([]float64, 0, len(m))
	for _, k := range sortedKeys(m) {
		vals = append(vals, m[k])
	}
	return vals
}

// validFitnessKeysList returns the metric keys accepted by ComputeFitness/extractMetric.
// Returns a fresh slice each call to prevent mutation of shared state.
func validFitnessKeysList() []string {
	return []string{
		"throughput", "tokens_per_sec",
		"p99_ttft", "p50_ttft", "mean_ttft",
		"p99_e2e", "p50_e2e", "mean_e2e",
	}
}

// Reference scales for normalizing metrics to [0,1] range.
// Without reference scales, throughput (raw value ~100) dominates latency (1/(1+5000) ≈ 0.0002)
// by 500,000×, making multi-objective optimization impossible.
const (
	referenceRPS   = 100.0   // 100 requests/sec as reference throughput
	referenceTPS   = 10000.0 // 10,000 tokens/sec as reference token throughput
	referenceTicks = 1000.0  // 1ms (1000 ticks) as reference latency
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
// Returns error for unknown weight keys (BC-7).
func ComputeFitness(metrics *RawMetrics, weights map[string]float64) (FitnessResult, error) {
	// Validate all keys before computing
	for _, key := range sortedKeys(weights) {
		if _, ok := extractMetric(metrics, key); !ok {
			return FitnessResult{}, fmt.Errorf("unknown fitness metric key %q; valid keys: %s", key,
				strings.Join(validFitnessKeysList(), ", "))
		}
	}

	result := FitnessResult{
		Components: make(map[string]float64, len(weights)),
	}

	for _, key := range sortedKeys(weights) {
		weight := weights[key]
		value, _ := extractMetric(metrics, key) // already validated
		result.Components[key] = value
		result.Score += value * weight
	}

	return result, nil
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
		return m.TokensPerSec / (m.TokensPerSec + referenceTPS), true
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
		if math.IsNaN(val) || math.IsInf(val, 0) || val < 0 {
			return nil, fmt.Errorf("invalid weight for %q: must be a finite non-negative number, got %v", key, val)
		}
		weights[key] = val
	}
	return weights, nil
}
