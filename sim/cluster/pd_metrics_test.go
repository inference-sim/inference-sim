package cluster

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// buildAggregatedWithTTFTs creates a sim.Metrics stub with given TTFT map and SimEndedTime.
func buildAggregatedWithTTFTs(ttfts map[string]float64, simEndedTime int64) *sim.Metrics {
	m := sim.NewMetrics()
	for id, v := range ttfts {
		m.RequestTTFTs[id] = v
	}
	m.SimEndedTime = simEndedTime
	return m
}

// buildParentRequest creates a minimal ParentRequest for testing.
func buildParentRequest(id string, decodeSubReqID string, transferStart, transferComplete int64) *ParentRequest {
	return &ParentRequest{
		ID:                   id,
		DecodeSubReqID:       decodeSubReqID,
		TransferStartTime:    transferStart,
		TransferCompleteTime: transferComplete,
	}
}

// ---- Task 1: Unit tests (BC-1, BC-2, BC-6, BC-7, BC-11) ----

// BC-7: nil return when no parents.
func TestCollectPDMetrics_NilWhenNoParents(t *testing.T) {
	agg := buildAggregatedWithTTFTs(nil, 1_000_000)
	pd := CollectPDMetrics(nil, agg, nil, nil)
	if pd != nil {
		t.Errorf("expected nil for empty parents, got non-nil")
	}
	pd2 := CollectPDMetrics([]*ParentRequest{}, agg, nil, nil)
	if pd2 != nil {
		t.Errorf("expected nil for zero-length parents slice, got non-nil")
	}
}

// BC-1: ParentTTFT populated from aggregated.RequestTTFTs[DecodeSubReqID].
func TestCollectPDMetrics_ParentTTFT(t *testing.T) {
	parents := []*ParentRequest{
		buildParentRequest("req-1", "req-1_decode", 100, 200),
		buildParentRequest("req-2", "req-2_decode", 100, 200),
	}
	agg := buildAggregatedWithTTFTs(map[string]float64{
		"req-1_decode": 5000.0,
		"req-2_decode": 3000.0,
	}, 1_000_000)

	pd := CollectPDMetrics(parents, agg, nil, nil)
	if pd == nil {
		t.Fatal("expected non-nil PDMetrics")
	}
	if pd.ParentTTFT.Count != 2 {
		t.Errorf("expected ParentTTFT.Count=2, got %d", pd.ParentTTFT.Count)
	}
	expectedMean := (5000.0 + 3000.0) / 2.0
	if math.Abs(pd.ParentTTFT.Mean-expectedMean) > 1e-9 {
		t.Errorf("expected ParentTTFT.Mean=%.1f, got %.1f", expectedMean, pd.ParentTTFT.Mean)
	}
}

// BC-11: TTFT value 0.0 (missing key in map) excluded from distribution.
func TestCollectPDMetrics_TTFTExcludesMissing(t *testing.T) {
	parents := []*ParentRequest{
		buildParentRequest("req-1", "req-1_decode", 100, 200),
		buildParentRequest("req-2", "req-2_decode", 100, 200), // no TTFT in map
	}
	agg := buildAggregatedWithTTFTs(map[string]float64{
		"req-1_decode": 5000.0,
		// req-2_decode missing → 0.0 default → excluded
	}, 1_000_000)

	pd := CollectPDMetrics(parents, agg, nil, nil)
	if pd == nil {
		t.Fatal("expected non-nil PDMetrics")
	}
	if pd.ParentTTFT.Count != 1 {
		t.Errorf("expected ParentTTFT.Count=1 (missing excluded), got %d", pd.ParentTTFT.Count)
	}
	if math.Abs(pd.ParentTTFT.Mean-5000.0) > 1e-9 {
		t.Errorf("expected ParentTTFT.Mean=5000.0, got %.1f", pd.ParentTTFT.Mean)
	}
}

// BC-2: TransferDuration = TransferCompleteTime - TransferStartTime (microseconds).
func TestCollectPDMetrics_TransferDuration(t *testing.T) {
	parents := []*ParentRequest{
		buildParentRequest("req-1", "req-1_decode", 1000, 1500), // duration = 500
		buildParentRequest("req-2", "req-2_decode", 2000, 2300), // duration = 300
	}
	agg := buildAggregatedWithTTFTs(map[string]float64{
		"req-1_decode": 5000.0,
		"req-2_decode": 3000.0,
	}, 1_000_000)

	pd := CollectPDMetrics(parents, agg, nil, nil)
	if pd == nil {
		t.Fatal("expected non-nil PDMetrics")
	}
	if pd.TransferDuration.Count != 2 {
		t.Errorf("expected TransferDuration.Count=2, got %d", pd.TransferDuration.Count)
	}
	expectedMean := (500.0 + 300.0) / 2.0
	if math.Abs(pd.TransferDuration.Mean-expectedMean) > 1e-9 {
		t.Errorf("expected TransferDuration.Mean=%.1f, got %.1f", expectedMean, pd.TransferDuration.Mean)
	}
}

// BC-6: DisaggregatedCount counts parents with TransferCompleteTime > 0.
func TestCollectPDMetrics_DisaggregatedCount(t *testing.T) {
	parents := []*ParentRequest{
		buildParentRequest("req-1", "req-1_decode", 100, 200),  // completed transfer
		buildParentRequest("req-2", "req-2_decode", 0, 0),       // no transfer (local)
		buildParentRequest("req-3", "req-3_decode", 100, 300),  // completed transfer
	}
	agg := buildAggregatedWithTTFTs(map[string]float64{
		"req-1_decode": 5000.0,
		"req-3_decode": 3000.0,
	}, 1_000_000)

	pd := CollectPDMetrics(parents, agg, nil, nil)
	if pd == nil {
		t.Fatal("expected non-nil PDMetrics")
	}
	if pd.DisaggregatedCount != 2 {
		t.Errorf("expected DisaggregatedCount=2, got %d", pd.DisaggregatedCount)
	}
}

// ---- Task 2: Integration tests (BC-4, BC-5, BC-10) ----

// BC-4: PrefillThroughput = prefill_completions / simEndedTime (in seconds).
// BC-5: DecodeThroughput = decode_completions / simEndedTime (in seconds).
func TestCollectPDMetrics_PerPoolThroughput(t *testing.T) {
	parents := []*ParentRequest{
		buildParentRequest("req-1", "req-1_decode", 100, 200),
	}
	// 2s sim, prefill has 10 completions, decode has 5 completions
	simEndedUs := int64(2_000_000)
	agg := buildAggregatedWithTTFTs(map[string]float64{"req-1_decode": 5000.0}, simEndedUs)
	agg.CompletedRequests = 15

	poolMembership := map[string]PoolRole{
		"instance_0": PoolRolePrefill,
		"instance_1": PoolRoleDecode,
	}
	m0 := sim.NewMetrics()
	m0.CompletedRequests = 10
	m1 := sim.NewMetrics()
	m1.CompletedRequests = 5
	metricsByID := map[string]*sim.Metrics{
		"instance_0": m0,
		"instance_1": m1,
	}

	pd := CollectPDMetrics(parents, agg, poolMembership, metricsByID)
	if pd == nil {
		t.Fatal("expected non-nil PDMetrics")
	}

	expectedPrefill := 10.0 / 2.0 // = 5.0
	expectedDecode := 5.0 / 2.0   // = 2.5

	if math.Abs(pd.PrefillThroughput-expectedPrefill) > 1e-9 {
		t.Errorf("expected PrefillThroughput=%.4f, got %.4f", expectedPrefill, pd.PrefillThroughput)
	}
	if math.Abs(pd.DecodeThroughput-expectedDecode) > 1e-9 {
		t.Errorf("expected DecodeThroughput=%.4f, got %.4f", expectedDecode, pd.DecodeThroughput)
	}
}

// BC-10 case 1: balanced pools → LoadImbalanceRatio close to 1.0.
func TestCollectPDMetrics_LoadImbalanceRatio_Balanced(t *testing.T) {
	parents := []*ParentRequest{buildParentRequest("req-1", "req-1_decode", 100, 200)}
	simEndedUs := int64(1_000_000)
	agg := buildAggregatedWithTTFTs(map[string]float64{"req-1_decode": 5000.0}, simEndedUs)

	poolMembership := map[string]PoolRole{
		"instance_0": PoolRolePrefill,
		"instance_1": PoolRoleDecode,
	}
	m0 := sim.NewMetrics()
	m0.CompletedRequests = 10
	m1 := sim.NewMetrics()
	m1.CompletedRequests = 10
	metricsByID := map[string]*sim.Metrics{
		"instance_0": m0,
		"instance_1": m1,
	}

	pd := CollectPDMetrics(parents, agg, poolMembership, metricsByID)
	if pd == nil {
		t.Fatal("expected non-nil PDMetrics")
	}
	if math.Abs(pd.LoadImbalanceRatio-1.0) > 1e-9 {
		t.Errorf("expected LoadImbalanceRatio=1.0 (balanced), got %.4f", pd.LoadImbalanceRatio)
	}
}

// BC-10 case 2: imbalanced pools → LoadImbalanceRatio = max/min > 1.0.
func TestCollectPDMetrics_LoadImbalanceRatio_Imbalanced(t *testing.T) {
	parents := []*ParentRequest{buildParentRequest("req-1", "req-1_decode", 100, 200)}
	simEndedUs := int64(1_000_000)
	agg := buildAggregatedWithTTFTs(map[string]float64{"req-1_decode": 5000.0}, simEndedUs)

	poolMembership := map[string]PoolRole{
		"instance_0": PoolRolePrefill,
		"instance_1": PoolRoleDecode,
	}
	m0 := sim.NewMetrics()
	m0.CompletedRequests = 20
	m1 := sim.NewMetrics()
	m1.CompletedRequests = 10
	metricsByID := map[string]*sim.Metrics{
		"instance_0": m0,
		"instance_1": m1,
	}

	pd := CollectPDMetrics(parents, agg, poolMembership, metricsByID)
	if pd == nil {
		t.Fatal("expected non-nil PDMetrics")
	}
	// prefill=20/s, decode=10/s → ratio = 20/10 = 2.0
	expected := 2.0
	if math.Abs(pd.LoadImbalanceRatio-expected) > 1e-9 {
		t.Errorf("expected LoadImbalanceRatio=%.4f, got %.4f", expected, pd.LoadImbalanceRatio)
	}
}

// BC-10 case 3: one pool idle → LoadImbalanceRatio = math.MaxFloat64 (sentinel).
func TestCollectPDMetrics_LoadImbalanceRatio_ZeroMinGuard(t *testing.T) {
	parents := []*ParentRequest{buildParentRequest("req-1", "req-1_decode", 100, 200)}
	simEndedUs := int64(1_000_000)
	agg := buildAggregatedWithTTFTs(map[string]float64{"req-1_decode": 5000.0}, simEndedUs)

	poolMembership := map[string]PoolRole{
		"instance_0": PoolRolePrefill,
		"instance_1": PoolRoleDecode,
	}
	m0 := sim.NewMetrics()
	m0.CompletedRequests = 10
	m1 := sim.NewMetrics()
	m1.CompletedRequests = 0 // decode pool idle
	metricsByID := map[string]*sim.Metrics{
		"instance_0": m0,
		"instance_1": m1,
	}

	pd := CollectPDMetrics(parents, agg, poolMembership, metricsByID)
	if pd == nil {
		t.Fatal("expected non-nil PDMetrics")
	}
	if pd.LoadImbalanceRatio < math.MaxFloat64/2 {
		t.Errorf("expected LoadImbalanceRatio=MaxFloat64 (sentinel), got %.4f", pd.LoadImbalanceRatio)
	}
}

// BC-10 case 4: both pools idle → LoadImbalanceRatio = 1.0 (no-data sentinel).
func TestCollectPDMetrics_LoadImbalanceRatio_BothZeroGuard(t *testing.T) {
	parents := []*ParentRequest{buildParentRequest("req-1", "req-1_decode", 100, 200)}
	simEndedUs := int64(1_000_000)
	agg := buildAggregatedWithTTFTs(map[string]float64{"req-1_decode": 5000.0}, simEndedUs)

	poolMembership := map[string]PoolRole{
		"instance_0": PoolRolePrefill,
		"instance_1": PoolRoleDecode,
	}
	m0 := sim.NewMetrics()
	m0.CompletedRequests = 0
	m1 := sim.NewMetrics()
	m1.CompletedRequests = 0
	metricsByID := map[string]*sim.Metrics{
		"instance_0": m0,
		"instance_1": m1,
	}

	pd := CollectPDMetrics(parents, agg, poolMembership, metricsByID)
	if pd == nil {
		t.Fatal("expected non-nil PDMetrics")
	}
	if math.Abs(pd.LoadImbalanceRatio-1.0) > 1e-9 {
		t.Errorf("expected LoadImbalanceRatio=1.0 (both idle sentinel), got %.4f", pd.LoadImbalanceRatio)
	}
}
