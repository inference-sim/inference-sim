package cluster

import (
	"fmt"
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
	// Parent TTFT = prefill sub-request TTFT from aggregated.RequestTTFTs[parent.ID].
	// projectPDMetrics() rekeys the entry from PrefillSubReqID to the parent ID before
	// CollectPDMetrics runs, so the lookup uses parent.ID (see BC-1 comment at the call site).
	// The prefill sub-request records TTFT when ProgressIndex == inputLen (prefill completion
	// = first token generated). Decode sub-requests start with ProgressIndex = inputLen
	// (pre-allocated) and never trigger TTFT recording.
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

	// DroppedAtDecodeKV is the number of requests that completed KV transfer but
	// were dropped because the decode instance had insufficient KV capacity.
	// Detection predicate: TransferCompleteTime > 0 && DecodeInstanceID == "".
	DroppedAtDecodeKV int

	// Transfers is the number of handoffs whose pull completed. TransferP90 is the 90th
	// percentile of their durations (microseconds, numpy-linear like the rest).
	Transfers   int
	TransferP90 float64
	// TransferQueueDelay is each completed handoff's duration beyond its uncontended price
	// (TransferAloneTicks): the time contended links added (INV-P2-2). All zero with
	// contention off. TransferQueueDelayP90 is its 90th percentile.
	TransferQueueDelay    Distribution
	TransferQueueDelayP90 float64
	// LinkPeakConcurrent is, per fabric link, the most handoffs in flight on it together; nil
	// with contention off.
	LinkPeakConcurrent map[string]int
	// PrefillKVHoldPeakTokens is the most KV any one prefill instance held at once for pulls
	// not yet complete (INV-PD-7); zero when the deployment does not hold.
	PrefillKVHoldPeakTokens int64
	// CPUTierHitRate is, per pool, the share of looked-up prompt tokens the pool's CPU
	// offload tier served (reloaded tokens over looked-up tokens, summed over the pool's
	// instances), keyed "prefill" and "decode"; absent for a pool with no lookups.
	CPUTierHitRate map[string]float64
}

// PDMetrics collects the run's disaggregation metrics: CollectPDMetrics over its parents,
// plus what only the simulator knows -- the per-link handoff concurrency and the prefill KV
// hold. Nil when disaggregation is inactive.
func (c *ClusterSimulator) PDMetrics() *PDMetrics {
	pd := CollectPDMetrics(c.ParentRequests(), c.AggregatedMetrics(), c.PoolMembership(), c.PerInstanceMetricsByID())
	if pd == nil {
		return nil
	}
	pd.LinkPeakConcurrent = c.PDLinkPeakConcurrent()
	pd.PrefillKVHoldPeakTokens = c.PrefillKVHoldPeakTokens()
	return pd
}

// CollectPDMetrics computes disaggregation-aware metrics from post-simulation state.
// Returns nil when parents is empty (BC-7). Pure function -- no mutation of inputs.
//
// Parameters:
//   - parents: slice of ParentRequest records (disaggregated request lifecycles)
//   - aggregated: cluster-aggregated sim.Metrics (provides RequestTTFTs map)
//   - poolMembership: map of instance ID -> PoolRole (from ClusterSimulator.PoolMembership)
//   - metricsByID: map of instance ID -> *sim.Metrics (from ClusterSimulator.PerInstanceMetricsByID)
func CollectPDMetrics(
	parents []*ParentRequest,
	aggregated *sim.Metrics,
	poolMembership map[string]PoolRole,
	metricsByID map[string]*sim.Metrics,
) *PDMetrics {
	if len(parents) == 0 {
		return nil
	}
	if aggregated == nil {
		// R1: aggregated is required for TTFT lookup and SimEndedTime access.
		// In production, aggregated comes from cs.AggregatedMetrics() which is always
		// non-nil after Run(). A nil aggregated with non-empty parents is a programming
		// error in the caller -- panic rather than silently dropping all parent data.
		panic("CollectPDMetrics: aggregated is nil with non-empty parents (programming error)")
	}

	// Sort by ID for deterministic processing (R2).
	sorted := make([]*ParentRequest, len(parents))
	copy(sorted, parents)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].ID < sorted[j].ID })

	var ttftValues, transferValues, delayValues []float64
	var disaggCount, droppedAtDecodeKV int

	for _, p := range sorted {
		// BC-6: count requests that completed KV transfer.
		if p.TransferCompleteTime > 0 {
			disaggCount++
		}

		// Count requests dropped at decode KV allocation: completed transfer but
		// never assigned to a decode instance (DecodeInstanceID remains empty).
		if p.TransferCompleteTime > 0 && p.DecodeInstanceID == "" {
			droppedAtDecodeKV++
		}

		// BC-1: parent TTFT from the aggregated TTFT map, keyed by parent ID.
		// After projectPDMetrics(), the user-visible TTFT is stored under the parent ID.
		// Value = decodeSchedulingDelay + first decode step — the full arrival →
		// first-token-emitted-by-decode span, carrying exactly one OutputTokenProcessingTime
		// (issue #1510, correcting the earlier prefillTTFT+transfer+decode composition of
		// #930 which double-counted OTPT and omitted the decode-queue wait).
		// Missing key returns 0.0 in Go maps; exclude 0.0 values (BC-11).
		if ttft := aggregated.RequestTTFTs[p.ID]; ttft > 0 {
			ttftValues = append(ttftValues, ttft)
		}

		// BC-2: transfer duration = complete - start (microseconds).
		// Guard: require both timestamps set and completion >= start.
		if p.TransferStartTime > 0 && p.TransferCompleteTime >= p.TransferStartTime {
			d := p.TransferCompleteTime - p.TransferStartTime
			transferValues = append(transferValues, float64(d))
			delayValues = append(delayValues, float64(max(0, d-p.TransferAloneTicks)))
		}
	}

	pd := &PDMetrics{
		DisaggregatedCount:    disaggCount,
		ParentTTFT:            NewDistribution(ttftValues),
		TransferDuration:      NewDistribution(transferValues),
		LoadImbalanceRatio:    1.0,
		DroppedAtDecodeKV:     droppedAtDecodeKV,
		Transfers:             len(transferValues),
		TransferP90:           percentileOf(transferValues, 90),
		TransferQueueDelay:    NewDistribution(delayValues),
		TransferQueueDelayP90: percentileOf(delayValues, 90),
		CPUTierHitRate:        collectPoolCPUTierHitRate(poolMembership, metricsByID),
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
		if m == nil {
			// R1: nil metrics pointer is a programming error in the caller.
			// In production, PerInstanceMetricsByID() always returns non-nil pointers from inst.Metrics().
			// A nil entry from a test helper signals incorrect test setup -- panic to make the error visible.
			panic(fmt.Sprintf("collectPoolThroughput: nil *sim.Metrics for instance %q (programming error)", id))
		}
		// Set-membership (.Has) so a shared-role pod (PoolRolePrefillDecode)
		// contributes to both pool counters — issue #1276 BC-6. Instances
		// not in pool membership (PoolRole(0)) contribute to neither
		// (BC-10 partial-membership safe).
		role := poolMembership[id]
		if role.Has(PoolRolePrefill) {
			prefillCompleted += m.CompletedRequests
		}
		if role.Has(PoolRoleDecode) {
			decodeCompleted += m.CompletedRequests
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
		// Both pools have zero completions -- no-data sentinel.
		imbalanceRatio = 1.0
	case minRPS <= 0:
		// One pool completely idle -- extreme imbalance sentinel.
		imbalanceRatio = math.MaxFloat64
	default:
		// R11: minRPS > 0 guaranteed by the case above.
		imbalanceRatio = maxRPS / minRPS
	}

	return
}

// percentileOf is the p-th percentile of values (linear interpolation, as Distribution), 0 for
// none.
func percentileOf(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	return percentile(sorted, p)
}

// collectPoolCPUTierHitRate is each pool's CPU-tier hit rate: prompt tokens reloaded from the
// CPU tier over prompt tokens looked up, summed over the pool's instances (the counts behind
// cache_hit_rate, #1904). A shared-role instance counts toward both pools.
func collectPoolCPUTierHitRate(poolMembership map[string]PoolRole, metricsByID map[string]*sim.Metrics) map[string]float64 {
	hits, lookups := map[string]int64{}, map[string]int64{}
	for id, m := range metricsByID {
		role := poolMembership[id]
		for _, pool := range []struct {
			name string
			role PoolRole
		}{{"prefill", PoolRolePrefill}, {"decode", PoolRoleDecode}} {
			if m != nil && role.Has(pool.role) {
				hits[pool.name] += m.CPUTierHitTokens
				lookups[pool.name] += m.PrefixCacheLookupTokens
			}
		}
	}
	out := map[string]float64{}
	for pool, n := range lookups {
		if n > 0 {
			out[pool] = float64(hits[pool]) / float64(n)
		}
	}
	return out
}

// Output is the results file's P/D block for these metrics.
func (pd *PDMetrics) Output() *sim.PDOutput {
	if pd == nil {
		return nil
	}
	return &sim.PDOutput{
		Transfers:                pd.Transfers,
		TransferTimeP50Ms:        pd.TransferDuration.P50 / 1e3,
		TransferTimeP90Ms:        pd.TransferP90 / 1e3,
		TransferQueueDelayMeanMs: pd.TransferQueueDelay.Mean / 1e3,
		TransferQueueDelayP90Ms:  pd.TransferQueueDelayP90 / 1e3,
		PrefillKVHoldPeakTokens:  pd.PrefillKVHoldPeakTokens,
		LinkPeakConcurrent:       pd.LinkPeakConcurrent,
		CPUTierHitRate:           pd.CPUTierHitRate,
	}
}
