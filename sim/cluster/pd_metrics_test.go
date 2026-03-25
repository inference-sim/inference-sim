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
// prefillSubReqID is the ID used to look up TTFT in RequestTTFTs.
func buildParentRequest(id string, prefillSubReqID string, transferStart, transferComplete int64) *ParentRequest {
	return &ParentRequest{
		ID:                   id,
		PrefillSubReqID:      prefillSubReqID,
		DecodeSubReqID:       id + "_decode",
		TransferStartTime:    transferStart,
		TransferCompleteTime: transferComplete,
	}
}

// ---- Panic-path tests ----

// CollectPDMetrics panics when aggregated is nil with non-empty parents (programming error guard).
func TestCollectPDMetrics_PanicsOnNilAggregated(t *testing.T) {
	parents := []*ParentRequest{
		buildParentRequest("req-1", "req-1_prefill", 100, 200),
	}
	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("expected panic for nil aggregated with non-empty parents, got none")
		}
	}()
	CollectPDMetrics(parents, nil, nil, nil)
}

// collectPoolThroughput panics when metricsByID contains a nil *sim.Metrics entry (programming error guard).
func TestCollectPoolThroughput_PanicsOnNilMetricsEntry(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("expected panic for nil *sim.Metrics entry in metricsByID, got none")
		}
	}()
	poolMembership := map[string]PoolRole{"instance_0": PoolRolePrefill}
	metricsByID := map[string]*sim.Metrics{"instance_0": nil} // nil pointer — programming error
	collectPoolThroughput(poolMembership, metricsByID, 1_000_000)
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

// BC-1: ParentTTFT populated from aggregated.RequestTTFTs[PrefillSubReqID].
func TestCollectPDMetrics_ParentTTFT(t *testing.T) {
	parents := []*ParentRequest{
		buildParentRequest("req-1", "req-1_prefill", 100, 200),
		buildParentRequest("req-2", "req-2_prefill", 100, 200),
	}
	agg := buildAggregatedWithTTFTs(map[string]float64{
		"req-1_prefill": 5000.0,
		"req-2_prefill": 3000.0,
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
		buildParentRequest("req-1", "req-1_prefill", 100, 200),
		buildParentRequest("req-2", "req-2_prefill", 100, 200), // no TTFT in map
	}
	agg := buildAggregatedWithTTFTs(map[string]float64{
		"req-1_prefill": 5000.0,
		// req-2_prefill missing -> 0.0 default -> excluded
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
		buildParentRequest("req-1", "req-1_prefill", 1000, 1500), // duration = 500
		buildParentRequest("req-2", "req-2_prefill", 2000, 2300), // duration = 300
	}
	agg := buildAggregatedWithTTFTs(map[string]float64{
		"req-1_prefill": 5000.0,
		"req-2_prefill": 3000.0,
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
		buildParentRequest("req-1", "req-1_prefill", 100, 200),  // completed transfer
		buildParentRequest("req-2", "req-2_prefill", 0, 0),       // no transfer (local)
		buildParentRequest("req-3", "req-3_prefill", 100, 300),  // completed transfer
	}
	agg := buildAggregatedWithTTFTs(map[string]float64{
		"req-1_prefill": 5000.0,
		"req-3_prefill": 3000.0,
	}, 1_000_000)

	pd := CollectPDMetrics(parents, agg, nil, nil)
	if pd == nil {
		t.Fatal("expected non-nil PDMetrics")
	}
	if pd.DisaggregatedCount != 2 {
		t.Errorf("expected DisaggregatedCount=2, got %d", pd.DisaggregatedCount)
	}
}

// TestCollectPDMetrics_DroppedAtDecodeKV verifies that DroppedAtDecodeKV counts parent
// requests where the KV transfer completed (TransferCompleteTime > 0) but the decode
// sub-request was never assigned to an instance (DecodeInstanceID == "").
func TestCollectPDMetrics_DroppedAtDecodeKV(t *testing.T) {
	tests := []struct {
		name     string
		parents  []*ParentRequest
		wantDrop int
	}{
		{
			name: "none dropped",
			parents: []*ParentRequest{
				{ID: "req-1", TransferCompleteTime: 200, DecodeInstanceID: "instance_2"},
			},
			wantDrop: 0,
		},
		{
			name: "all dropped",
			parents: []*ParentRequest{
				{ID: "req-1", TransferCompleteTime: 200, DecodeInstanceID: ""},
				{ID: "req-2", TransferCompleteTime: 300, DecodeInstanceID: ""},
			},
			wantDrop: 2,
		},
		{
			name: "mixed: transfer-then-drop, transfer-then-assigned, no-transfer",
			parents: []*ParentRequest{
				{ID: "req-1", TransferCompleteTime: 200, DecodeInstanceID: "instance_2"}, // assigned
				{ID: "req-2", TransferCompleteTime: 300, DecodeInstanceID: ""},           // dropped
				{ID: "req-3", TransferCompleteTime: 0, DecodeInstanceID: ""},             // no transfer -> not dropped
			},
			wantDrop: 1,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			agg := buildAggregatedWithTTFTs(nil, 1_000_000)
			pd := CollectPDMetrics(tc.parents, agg, nil, nil)
			if pd == nil {
				t.Fatal("expected non-nil PDMetrics")
			}
			if pd.DroppedAtDecodeKV != tc.wantDrop {
				t.Errorf("DroppedAtDecodeKV = %d, want %d", pd.DroppedAtDecodeKV, tc.wantDrop)
			}
		})
	}
}

// ---- Task 2: Integration tests (BC-4, BC-5, BC-10) ----

// BC-4: PrefillThroughput = prefill_completions / simEndedTime (in seconds).
// BC-5: DecodeThroughput = decode_completions / simEndedTime (in seconds).
func TestCollectPDMetrics_PerPoolThroughput(t *testing.T) {
	parents := []*ParentRequest{
		buildParentRequest("req-1", "req-1_prefill", 100, 200),
	}
	// 2s sim, prefill has 10 completions, decode has 5 completions
	simEndedUs := int64(2_000_000)
	agg := buildAggregatedWithTTFTs(map[string]float64{"req-1_prefill": 5000.0}, simEndedUs)
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

// BC-10 case 1: balanced pools -> LoadImbalanceRatio close to 1.0.
func TestCollectPDMetrics_LoadImbalanceRatio_Balanced(t *testing.T) {
	parents := []*ParentRequest{buildParentRequest("req-1", "req-1_prefill", 100, 200)}
	simEndedUs := int64(1_000_000)
	agg := buildAggregatedWithTTFTs(map[string]float64{"req-1_prefill": 5000.0}, simEndedUs)

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

// BC-10 case 2: imbalanced pools -> LoadImbalanceRatio = max/min > 1.0.
func TestCollectPDMetrics_LoadImbalanceRatio_Imbalanced(t *testing.T) {
	parents := []*ParentRequest{buildParentRequest("req-1", "req-1_prefill", 100, 200)}
	simEndedUs := int64(1_000_000)
	agg := buildAggregatedWithTTFTs(map[string]float64{"req-1_prefill": 5000.0}, simEndedUs)

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
	// prefill=20/s, decode=10/s -> ratio = 20/10 = 2.0
	expected := 2.0
	if math.Abs(pd.LoadImbalanceRatio-expected) > 1e-9 {
		t.Errorf("expected LoadImbalanceRatio=%.4f, got %.4f", expected, pd.LoadImbalanceRatio)
	}
}

// BC-10 case 3: one pool idle -> LoadImbalanceRatio = math.MaxFloat64 (sentinel).
func TestCollectPDMetrics_LoadImbalanceRatio_ZeroMinGuard(t *testing.T) {
	parents := []*ParentRequest{buildParentRequest("req-1", "req-1_prefill", 100, 200)}
	simEndedUs := int64(1_000_000)
	agg := buildAggregatedWithTTFTs(map[string]float64{"req-1_prefill": 5000.0}, simEndedUs)

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
	if pd.LoadImbalanceRatio != math.MaxFloat64 {
		t.Errorf("expected LoadImbalanceRatio=math.MaxFloat64 (one-pool-idle sentinel), got %v", pd.LoadImbalanceRatio)
	}
}

// TestCollectPDMetrics_Invariant_LoadImbalanceGeq1 verifies the law:
// LoadImbalanceRatio is always >= 1.0 (max/min >= 1 by definition) for non-sentinel values.
func TestCollectPDMetrics_Invariant_LoadImbalanceGeq1(t *testing.T) {
	cases := []struct{ prefillCompletions, decodeCompletions int }{
		{10, 10}, // balanced
		{10, 5},  // prefill faster
		{5, 10},  // decode faster
		{1, 100}, // extreme imbalance
	}
	for _, tc := range cases {
		parents := []*ParentRequest{buildParentRequest("req-1", "req-1_prefill", 100, 200)}
		simEndedUs := int64(1_000_000)
		agg := buildAggregatedWithTTFTs(map[string]float64{"req-1_prefill": 5000.0}, simEndedUs)
		poolMembership := map[string]PoolRole{
			"instance_0": PoolRolePrefill,
			"instance_1": PoolRoleDecode,
		}
		m0 := sim.NewMetrics()
		m0.CompletedRequests = tc.prefillCompletions
		m1 := sim.NewMetrics()
		m1.CompletedRequests = tc.decodeCompletions
		metricsByID := map[string]*sim.Metrics{"instance_0": m0, "instance_1": m1}

		pd := CollectPDMetrics(parents, agg, poolMembership, metricsByID)
		if pd == nil {
			t.Errorf("expected non-nil PDMetrics for prefill=%d decode=%d", tc.prefillCompletions, tc.decodeCompletions)
			continue
		}
		if pd.LoadImbalanceRatio != math.MaxFloat64 && pd.LoadImbalanceRatio < 1.0 {
			t.Errorf("law violated: LoadImbalanceRatio=%.4f < 1.0 for prefill=%d decode=%d",
				pd.LoadImbalanceRatio, tc.prefillCompletions, tc.decodeCompletions)
		}
	}
}

// BC-10 case 4: both pools idle -> LoadImbalanceRatio = 1.0 (no-data sentinel).
func TestCollectPDMetrics_LoadImbalanceRatio_BothZeroGuard(t *testing.T) {
	parents := []*ParentRequest{buildParentRequest("req-1", "req-1_prefill", 100, 200)}
	simEndedUs := int64(1_000_000)
	agg := buildAggregatedWithTTFTs(map[string]float64{"req-1_prefill": 5000.0}, simEndedUs)

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

// GAP-3: collectPoolThroughput returns defaults when simEndedTime <= 0.
// Guards against +Inf throughput from division by zero.
func TestCollectPoolThroughput_ZeroSimEndedTime(t *testing.T) {
	poolMembership := map[string]PoolRole{
		"instance_0": PoolRolePrefill,
		"instance_1": PoolRoleDecode,
	}
	m0 := sim.NewMetrics()
	m0.CompletedRequests = 10
	m1 := sim.NewMetrics()
	m1.CompletedRequests = 5
	metricsByID := map[string]*sim.Metrics{"instance_0": m0, "instance_1": m1}

	prefill, decode, imbalance := collectPoolThroughput(poolMembership, metricsByID, 0)
	if prefill != 0.0 || decode != 0.0 || math.Abs(imbalance-1.0) > 1e-9 {
		t.Errorf("expected defaults (0.0, 0.0, 1.0) for simEndedTime=0, got (%.4f, %.4f, %.4f)",
			prefill, decode, imbalance)
	}
}

// GAP-4: TransferDuration excludes entries where TransferStartTime > TransferCompleteTime
// (inverted timestamps are excluded; only well-formed intervals contribute to the distribution).
func TestCollectPDMetrics_TransferDuration_InvertedTimestampExcluded(t *testing.T) {
	parents := []*ParentRequest{
		buildParentRequest("req-1", "req-1_prefill", 1000, 1500), // valid: duration = 500
		buildParentRequest("req-2", "req-2_prefill", 2000, 1800), // invalid: start > complete
		buildParentRequest("req-3", "req-3_prefill", 0, 500),     // invalid: start == 0
	}
	agg := buildAggregatedWithTTFTs(nil, 1_000_000)

	pd := CollectPDMetrics(parents, agg, nil, nil)
	if pd == nil {
		t.Fatal("expected non-nil PDMetrics")
	}
	if pd.TransferDuration.Count != 1 {
		t.Errorf("expected TransferDuration.Count=1 (only valid interval counted), got %d", pd.TransferDuration.Count)
	}
	if math.Abs(pd.TransferDuration.Mean-500.0) > 1e-9 {
		t.Errorf("expected TransferDuration.Mean=500.0, got %.1f", pd.TransferDuration.Mean)
	}
}
