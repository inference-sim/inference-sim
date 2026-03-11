package cluster

import (
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// PDMetrics captures disaggregation-specific metrics collected after simulation ends.
// All fields are computed from post-simulation state; nil when disaggregation is inactive.
type PDMetrics struct {
	// DisaggregatedCount is the number of requests that completed KV transfer
	// (i.e., TransferCompleteTime > 0).
	DisaggregatedCount int

	// ParentTTFT is the distribution of parent-level TTFT (microseconds).
	// Parent TTFT = decode sub-request TTFT from aggregated.RequestTTFTs[DecodeSubReqID],
	// which equals (first_token_time - original_arrival_time) because the decode
	// sub-request is created with ArrivalTime = original request ArrivalTime.
	// Values of 0.0 (missing key) are excluded (BC-11).
	ParentTTFT Distribution

	// TransferDuration is the distribution of KV transfer latencies (microseconds),
	// computed as TransferCompleteTime - TransferStartTime per parent request.
	// Only requests with TransferStartTime > 0 AND TransferCompleteTime >= TransferStartTime
	// are included.
	TransferDuration Distribution

	// PrefillThroughput is the completed sub-request rate for the prefill pool
	// (sub-req/s = sum(prefill_instance.CompletedRequests) / simEndedTime).
	PrefillThroughput float64

	// DecodeThroughput is the completed sub-request rate for the decode pool
	// (sub-req/s = sum(decode_instance.CompletedRequests) / simEndedTime).
	DecodeThroughput float64

	// LoadImbalanceRatio is max(PrefillRPS, DecodeRPS) / min(PrefillRPS, DecodeRPS).
	// Sentinels (BC-10):
	//   - 1.0 when both pools have 0 completions (no-data)
	//   - math.MaxFloat64 when one pool is idle (extreme imbalance)
	LoadImbalanceRatio float64
}

// CollectPDMetrics computes disaggregation-aware metrics from post-simulation state.
// Returns nil when parents is empty (BC-7). Pure function — no mutation of inputs.
//
// Parameters:
//   - parents: slice of ParentRequest records (disaggregated request lifecycles)
//   - aggregated: cluster-aggregated sim.Metrics (provides RequestTTFTs map)
//   - poolMembership: map of instance ID → PoolRole (from ClusterSimulator.PoolMembership)
//   - metricsByID: map of instance ID → *sim.Metrics (from ClusterSimulator.PerInstanceMetricsByID)
func CollectPDMetrics(
	parents []*ParentRequest,
	aggregated *sim.Metrics,
	poolMembership map[string]PoolRole,
	metricsByID map[string]*sim.Metrics,
) *PDMetrics {
	if len(parents) == 0 {
		return nil
	}

	// Sort by ID for deterministic processing (R2).
	sorted := make([]*ParentRequest, len(parents))
	copy(sorted, parents)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].ID < sorted[j].ID })

	var ttftValues, transferValues []float64
	var disaggCount int

	for _, p := range sorted {
		// BC-6: count requests that completed KV transfer.
		if p.TransferCompleteTime > 0 {
			disaggCount++
		}

		// BC-1: parent TTFT from decode sub-request TTFT.
		// Missing key returns 0.0 in Go maps; exclude 0.0 values (BC-11).
		if ttft := aggregated.RequestTTFTs[p.DecodeSubReqID]; ttft > 0 {
			ttftValues = append(ttftValues, ttft)
		}

		// BC-2: transfer duration = complete - start (microseconds).
		// Guard: require both timestamps set and completion >= start.
		if p.TransferStartTime > 0 && p.TransferCompleteTime >= p.TransferStartTime {
			transferValues = append(transferValues, float64(p.TransferCompleteTime-p.TransferStartTime))
		}
	}

	pd := &PDMetrics{
		DisaggregatedCount: disaggCount,
		ParentTTFT:         NewDistribution(ttftValues),
		TransferDuration:   NewDistribution(transferValues),
		LoadImbalanceRatio: 1.0,
	}

	pd.PrefillThroughput, pd.DecodeThroughput, pd.LoadImbalanceRatio =
		collectPoolThroughput(poolMembership, metricsByID, aggregated.SimEndedTime)

	return pd
}

// collectPoolThroughput computes per-pool throughput and load imbalance ratio.
// Returns defaults (0.0, 0.0, 1.0) when inputs are insufficient for calculation.
func collectPoolThroughput(
	poolMembership map[string]PoolRole,
	metricsByID map[string]*sim.Metrics,
	simEndedTimeUs int64,
) (prefillRPS, decodeRPS, imbalanceRatio float64) {
	imbalanceRatio = 1.0

	// R11: guard against zero denominator; return defaults when no data.
	if len(poolMembership) == 0 || len(metricsByID) == 0 || simEndedTimeUs <= 0 {
		return
	}

	var prefillCompleted, decodeCompleted int

	// Sort instance IDs for deterministic accumulation (R2).
	ids := make([]string, 0, len(metricsByID))
	for id := range metricsByID {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	for _, id := range ids {
		m := metricsByID[id]
		switch poolMembership[id] {
		case PoolRolePrefill:
			prefillCompleted += m.CompletedRequests
		case PoolRoleDecode:
			decodeCompleted += m.CompletedRequests
		// default: instance not in pool membership (no-op, BC-10 partial membership safe)
		}
	}

	durationSec := float64(simEndedTimeUs) / 1e6
	prefillRPS = float64(prefillCompleted) / durationSec
	decodeRPS = float64(decodeCompleted) / durationSec

	// Load imbalance = max(RPS) / min(RPS) with sentinels (BC-10, R11).
	minRPS := prefillRPS
	maxRPS := decodeRPS
	if decodeRPS < prefillRPS {
		minRPS = decodeRPS
		maxRPS = prefillRPS
	}

	switch {
	case maxRPS == 0:
		// Both pools have zero completions — no-data sentinel.
		imbalanceRatio = 1.0
	case minRPS <= 0:
		// One pool completely idle — extreme imbalance sentinel.
		imbalanceRatio = math.MaxFloat64
	default:
		// R11: minRPS > 0 guaranteed by the case above.
		imbalanceRatio = maxRPS / minRPS
	}

	return
}
