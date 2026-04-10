package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/trace"
)

// TestPDTrace_NonDisaggMode_NoDisaggRecords verifies BC-PD-18:
// when disaggregation is not configured, no PD-specific trace records are emitted.
func TestPDTrace_NonDisaggMode_NoDisaggRecords(t *testing.T) {
	// GIVEN non-disaggregated simulation with trace enabled
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             10000000,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(10, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 1, "blackbox", 0),
		},
		NumInstances: 4,
		TraceLevel:   "decisions",
	}
	requests := newTestRequests(5)
	cs := NewClusterSimulator(config, requests, nil)

	// WHEN run
	mustRun(t, cs)

	// THEN no disaggregation-specific trace records
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace with trace-level decisions")
	}
	if len(tr.Disaggregations) != 0 {
		t.Errorf("expected 0 disaggregation records in non-disagg mode, got %d", len(tr.Disaggregations))
	}
	if len(tr.PrefillRoutings) != 0 {
		t.Errorf("expected 0 prefill routing records in non-disagg mode, got %d", len(tr.PrefillRoutings))
	}
	if len(tr.DecodeRoutings) != 0 {
		t.Errorf("expected 0 decode routing records in non-disagg mode, got %d", len(tr.DecodeRoutings))
	}
	if len(tr.KVTransfers) != 0 {
		t.Errorf("expected 0 KV transfer records in non-disagg mode, got %d", len(tr.KVTransfers))
	}
	// Existing admission/routing records still present (BC-TRACE-COMPAT)
	if len(tr.Admissions) != 5 {
		t.Errorf("expected 5 admission records, got %d", len(tr.Admissions))
	}
	if len(tr.Routings) != 5 {
		t.Errorf("expected 5 routing records, got %d", len(tr.Routings))
	}
}

// TestPDTrace_DisaggMode_AllRecordTypesPresent verifies BC-PD-17:
// all 4 PD trace record types are emitted for each disaggregated request.
func TestPDTrace_DisaggMode_AllRecordTypesPresent(t *testing.T) {
	const numRequests = 5
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.TraceLevel = "decisions"
	requests := newTestRequests(numRequests)
	cs := NewClusterSimulator(config, requests, nil)

	mustRun(t, cs)

	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}

	// BC-PD-17: one record of each type per disaggregated request
	if len(tr.Disaggregations) != numRequests {
		t.Errorf("Disaggregations: expected %d, got %d", numRequests, len(tr.Disaggregations))
	}
	if len(tr.PrefillRoutings) != numRequests {
		t.Errorf("PrefillRoutings: expected %d, got %d", numRequests, len(tr.PrefillRoutings))
	}
	if len(tr.KVTransfers) != numRequests {
		t.Errorf("KVTransfers: expected %d, got %d", numRequests, len(tr.KVTransfers))
	}
	if len(tr.DecodeRoutings) != numRequests {
		t.Errorf("DecodeRoutings: expected %d, got %d", numRequests, len(tr.DecodeRoutings))
	}

	// Verify KVTransfer records have both instance IDs and non-zero duration
	for i, kv := range tr.KVTransfers {
		if kv.PrefillInstanceID == "" {
			t.Errorf("KVTransfers[%d]: PrefillInstanceID empty", i)
		}
		if kv.DecodeInstanceID == "" {
			t.Errorf("KVTransfers[%d]: DecodeInstanceID empty", i)
		}
		if kv.TransferStartTime <= 0 {
			t.Errorf("KVTransfers[%d]: TransferStartTime=%d, want > 0", i, kv.TransferStartTime)
		}
		if kv.TransferDuration <= 0 {
			t.Errorf("KVTransfers[%d]: TransferDuration=%d, want > 0", i, kv.TransferDuration)
		}
		if kv.NumKVBlocks <= 0 {
			t.Errorf("KVTransfers[%d]: NumKVBlocks=%d, want > 0", i, kv.NumKVBlocks)
		}
	}

	// Verify per-pool routing records have non-empty chosen instances
	for i, r := range tr.PrefillRoutings {
		if r.ChosenInstance == "" {
			t.Errorf("PrefillRoutings[%d]: ChosenInstance empty", i)
		}
		if r.ParentRequestID == "" {
			t.Errorf("PrefillRoutings[%d]: ParentRequestID empty", i)
		}
	}
	for i, r := range tr.DecodeRoutings {
		if r.ChosenInstance == "" {
			t.Errorf("DecodeRoutings[%d]: ChosenInstance empty", i)
		}
		if r.ParentRequestID == "" {
			t.Errorf("DecodeRoutings[%d]: ParentRequestID empty", i)
		}
	}

	// Verify cross-record ID linkage: every downstream ParentRequestID must appear
	// in Disaggregations[*].RequestID (the contract stated in DisaggregationRecord doc).
	disaggIDs := make(map[string]struct{}, len(tr.Disaggregations))
	for _, d := range tr.Disaggregations {
		disaggIDs[d.RequestID] = struct{}{}
	}
	for i, r := range tr.PrefillRoutings {
		if _, ok := disaggIDs[r.ParentRequestID]; !ok {
			t.Errorf("PrefillRoutings[%d]: ParentRequestID %q not found in Disaggregations", i, r.ParentRequestID)
		}
	}
	for i, kv := range tr.KVTransfers {
		if _, ok := disaggIDs[kv.ParentRequestID]; !ok {
			t.Errorf("KVTransfers[%d]: ParentRequestID %q not found in Disaggregations", i, kv.ParentRequestID)
		}
	}
	for i, r := range tr.DecodeRoutings {
		if _, ok := disaggIDs[r.ParentRequestID]; !ok {
			t.Errorf("DecodeRoutings[%d]: ParentRequestID %q not found in Disaggregations", i, r.ParentRequestID)
		}
	}
}

// TestPDTrace_DisaggMode_Counterfactual verifies BC-PD-19:
// per-pool routing records have counterfactual candidates when k > 0.
func TestPDTrace_DisaggMode_Counterfactual(t *testing.T) {
	// GIVEN disaggregated simulation with k=2, 2 prefill + 2 decode instances
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.TraceLevel = "decisions"
	config.CounterfactualK = 2
	config.RoutingPolicy = "weighted"
	config.RoutingScorerConfigs = sim.DefaultScorerConfigs()
	requests := newTestRequests(3)
	cs := NewClusterSimulator(config, requests, nil)

	mustRun(t, cs)

	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}

	// THEN prefill routing records have candidates (BC-PD-19)
	for i, r := range tr.PrefillRoutings {
		if len(r.Candidates) == 0 {
			t.Errorf("PrefillRoutings[%d]: expected candidates with k=2, got none", i)
		}
		if len(r.Candidates) > 2 {
			t.Errorf("PrefillRoutings[%d]: expected ≤2 candidates, got %d", i, len(r.Candidates))
		}
		if r.Regret < 0 {
			t.Errorf("PrefillRoutings[%d]: Regret=%f, want ≥0", i, r.Regret)
		}
	}

	// THEN decode routing records have candidates (BC-PD-19)
	for i, r := range tr.DecodeRoutings {
		if len(r.Candidates) == 0 {
			t.Errorf("DecodeRoutings[%d]: expected candidates with k=2, got none", i)
		}
		if len(r.Candidates) > 2 {
			t.Errorf("DecodeRoutings[%d]: expected ≤2 candidates, got %d", i, len(r.Candidates))
		}
		if r.Regret < 0 {
			t.Errorf("DecodeRoutings[%d]: Regret=%f, want ≥0", i, r.Regret)
		}
	}
}

// TestPDTrace_DisaggMode_Cardinality verifies PD trace cardinality with AlwaysDisaggregate.
//
// Decode-first cardinality invariants:
//   - len(DecodeRoutings) == len(Disaggregations): decode routing fires once per PD request,
//     always before the disaggregation decision, regardless of skip or disagg path.
//   - len(PrefillRoutings) == len(KVTransfers) == DisaggregatedCount: prefill routing and KV
//     transfer records are emitted only on the disagg path.
//
// This test uses AlwaysDisaggregate, so all paths are disagg and all four counts are equal.
// For a mixed skip/disagg scenario: len(DecodeRoutings) == len(Disaggregations) == N, but
// len(PrefillRoutings) == len(KVTransfers) == DisaggregatedCount < N.
func TestPDTrace_DisaggMode_Cardinality(t *testing.T) {
	// GIVEN disaggregated simulation with trace enabled
	const numRequests = 5
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.TraceLevel = "decisions"
	requests := newTestRequests(numRequests)
	cs := NewClusterSimulator(config, requests, nil)

	// WHEN run
	mustRun(t, cs)

	// THEN trace record counts satisfy the cardinality conservation law.
	// With AlwaysDisaggregate: DisaggregatedCount == len(Disaggregations) == len(PrefillRoutings) == len(KVTransfers) == len(DecodeRoutings)
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}
	summary := trace.Summarize(tr)
	disaggCount := summary.DisaggregatedCount
	np := len(tr.PrefillRoutings)
	nk := len(tr.KVTransfers)
	nd2 := len(tr.DecodeRoutings)
	// With AlwaysDisaggregate, DisaggregatedCount == len(Disaggregations)
	if disaggCount != len(tr.Disaggregations) {
		t.Errorf("cardinality violation: DisaggregatedCount=%d != len(Disaggregations)=%d (AlwaysDisaggregate expects all true)", disaggCount, len(tr.Disaggregations))
	}
	// With AlwaysDisaggregate, all paths are disagg so all four counts are equal.
	if np != disaggCount {
		t.Errorf("cardinality violation: PrefillRoutings=%d != DisaggregatedCount=%d", np, disaggCount)
	}
	if nk != np {
		t.Errorf("cardinality violation: KVTransfers=%d != PrefillRoutings=%d", nk, np)
	}
	if nd2 != nk {
		t.Errorf("cardinality violation: DecodeRoutings=%d != KVTransfers=%d", nd2, nk)
	}
}

// TestPDTrace_DisaggMode_DisaggDecisionRecorded verifies disaggregation decisions are recorded.
func TestPDTrace_DisaggMode_DisaggDecisionRecorded(t *testing.T) {
	// GIVEN disaggregated simulation with trace enabled
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.TraceLevel = "decisions"
	requests := newTestRequests(3)
	cs := NewClusterSimulator(config, requests, nil)

	// WHEN run
	mustRun(t, cs)

	// THEN exactly 3 disaggregation records (one per request), all Disaggregate=true
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}
	if len(tr.Disaggregations) != 3 {
		t.Errorf("expected 3 disaggregation records (one per request), got %d", len(tr.Disaggregations))
	}
	for i, r := range tr.Disaggregations {
		if !r.Disaggregate {
			t.Errorf("disaggregation[%d]: Disaggregate=false, want true (AlwaysDisaggregate)", i)
		}
		if r.RequestID == "" {
			t.Errorf("disaggregation[%d]: RequestID empty", i)
		}
	}
}

// TestPDTrace_DroppedAtDecodeKV_NoOrphanRecords verifies that when AllocateTransferredKV
// fails (decode instance KV OOM), no KVTransferRecord or DecodeRoutingRecord is emitted
// for the dropped request. DisaggregationRecord and PrefillRoutingRecord are still present.
// This exercises the "placement after AllocateTransferredKV" invariant for trace records.
//
// Note: drop scenario case 1 (no routable prefill pool instances) is not tested here because
// ValidatePoolTopology requires PrefillInstances > 0, and in this test all prefill instances
// remain Active throughout the run. The PrefillRoutingEvent.Execute guard for empty
// filteredSnapshots is a live defensive path: buildPoolFilteredSnapshots filters by IsRoutable(),
// so all-Draining or all-Terminated prefill pool instances would trigger it. Cases 2 and 3
// (decode-side KV drops via AllocateTransferredKV) are exercised here.
func TestPDTrace_DroppedAtDecodeKV_NoOrphanRecords(t *testing.T) {
	// GIVEN tight KV capacity on the decode instance (same setup as TestDisaggregation_DroppedAtDecodeKV)
	// 2 prefill, 1 decode instance with only 3 blocks (48 tokens).
	// newShortRequests produces requests needing ~2 blocks each; the decode instance fills up
	// and subsequent AllocateTransferredKV calls fail → droppedAtDecodeKV > 0.
	config := newTestDisaggDeploymentConfig(3, 2, 1)
	config.KVCacheConfig = sim.NewKVCacheConfig(3, 16, 0, 0, 0, 0)
	config.TraceLevel = "decisions"
	requests := newShortRequests(4)
	cs := NewClusterSimulator(config, requests, nil)

	// WHEN run
	mustRun(t, cs)

	// THEN at least one request was dropped at decode KV allocation
	if cs.AggregatedMetrics().DroppedUnservable == 0 {
		t.Fatal("expected DroppedUnservable > 0 with 3-block decode KV capacity and 4 requests, but no drops occurred — check test setup")
	}

	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace with trace-level decisions")
	}

	// In the decode-first flow, DecodeRoutingRecord is written at priority 3 (before
	// disaggregation decision). KVTransferRecord is written only after a successful
	// AllocateTransferredKV in DecodeEnqueueEvent. Under decode KV drops:
	//   DecodeRoutings == Disaggregations (all disagg requests have both)
	//   KVTransfers < DecodeRoutings (drops reduce KV count, not decode routing count)
	if len(tr.DecodeRoutings) != len(tr.Disaggregations) {
		t.Errorf("DecodeRoutings=%d != Disaggregations=%d under decode KV drops (decode routing is decode-first; all disagg requests get a record)",
			len(tr.DecodeRoutings), len(tr.Disaggregations))
	}
	if len(tr.KVTransfers) >= len(tr.Disaggregations) {
		t.Errorf("KVTransfers=%d >= Disaggregations=%d under drops (drops must reduce KV record count)", len(tr.KVTransfers), len(tr.Disaggregations))
	}
	// For decode-side drops (cases 2/3), PrefillRoutings == Disaggregations because
	// prefill routing succeeds before the drop. This equality does NOT hold for case 1
	// (no routable prefill instances), where PrefillRoutings < Disaggregations.
	// This test exercises cases 2/3 exclusively (decode KV OOM via AllocateTransferredKV).
	if len(tr.PrefillRoutings) != len(tr.Disaggregations) {
		t.Errorf("PrefillRoutings=%d != Disaggregations=%d (prefill routing unaffected by decode KV drop)", len(tr.PrefillRoutings), len(tr.Disaggregations))
	}
}

// TestPDTrace_NeverDecider_WithPools_OnlyDisaggRecords verifies that when PDDecider="never"
// is configured alongside pool topology, DisaggregationDecisionEvent still fires and records
// decisions, but all Disaggregate=false so no PrefillRouting, KVTransfer, or DecodeRouting
// records are emitted.
func TestPDTrace_NeverDecider_WithPools_OnlyDisaggRecords(t *testing.T) {
	// GIVEN pools configured but disaggregation decider set to "never"
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.PDDecider = "never"
	config.TraceLevel = "decisions"
	const numRequests = 4
	requests := newTestRequests(numRequests)
	cs := NewClusterSimulator(config, requests, nil)

	// WHEN run
	mustRun(t, cs)

	// THEN DisaggregationDecisionEvent fires for every request
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace with trace-level decisions")
	}
	if len(tr.Disaggregations) != numRequests {
		t.Errorf("expected %d disaggregation records, got %d", numRequests, len(tr.Disaggregations))
	}
	// All decisions are false
	for i, r := range tr.Disaggregations {
		if r.Disaggregate {
			t.Errorf("Disaggregations[%d]: Disaggregate=true, want false (NeverDisaggregate)", i)
		}
		if r.RequestID == "" {
			t.Errorf("Disaggregations[%d]: RequestID empty", i)
		}
	}
	// No prefill/KV records emitted (all requests are skip-path)
	if len(tr.PrefillRoutings) != 0 {
		t.Errorf("expected 0 PrefillRoutings with NeverDisaggregate, got %d", len(tr.PrefillRoutings))
	}
	if len(tr.KVTransfers) != 0 {
		t.Errorf("expected 0 KVTransfers with NeverDisaggregate, got %d", len(tr.KVTransfers))
	}
	// In the decode-first flow, DecodeRoutingRecord is always written for PD requests
	// (decode routing happens before the disaggregation decision).
	if len(tr.DecodeRoutings) != numRequests {
		t.Errorf("expected %d DecodeRoutings with NeverDisaggregate (decode routing always fires in PD mode), got %d",
			numRequests, len(tr.DecodeRoutings))
	}
	// Skip-path requests go through DecodeEnqueueEvent, not RoutingDecisionEvent.
	// Standard routing records are only written by RoutingDecisionEvent (non-PD path).
	if len(tr.Routings) != 0 {
		t.Errorf("expected 0 standard routing records in PD mode (skip-path uses DecodeEnqueueEvent), got %d", len(tr.Routings))
	}
}

// TestPDTrace_SkipPath_TraceRecords verifies trace record presence for skip-path requests.
// With NeverDisaggregate, all PD requests take the skip path (cache hit: served locally
// on the selected decode instance). The decode-first invariant means DecodeRoutingRecord
// and DisaggregationRecord (Disaggregate=false) are always emitted, while PrefillRoutingRecord
// and KVTransferRecord are absent (no prefill routing, no KV transfer on skip path).
func TestPDTrace_SkipPath_TraceRecords(t *testing.T) {
	// GIVEN pools configured with NeverDisaggregate (all requests → skip path)
	const numRequests = 4
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.PDDecider = "never"
	config.TraceLevel = "decisions"
	requests := newTestRequests(numRequests)
	cs := NewClusterSimulator(config, requests, nil)

	// WHEN run
	mustRun(t, cs)

	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}

	// THEN DecodeRoutingRecord present for every request (decode-first always records)
	if len(tr.DecodeRoutings) != numRequests {
		t.Errorf("DecodeRoutings=%d, want %d (decode-first: always written per PD request)", len(tr.DecodeRoutings), numRequests)
	}

	// THEN DisaggregationRecord present for every request, all Disaggregate=false
	if len(tr.Disaggregations) != numRequests {
		t.Errorf("Disaggregations=%d, want %d", len(tr.Disaggregations), numRequests)
	}
	for i, r := range tr.Disaggregations {
		if r.Disaggregate {
			t.Errorf("Disaggregations[%d]: Disaggregate=true, want false (NeverDisaggregate)", i)
		}
		if r.RequestID == "" {
			t.Errorf("Disaggregations[%d]: RequestID empty", i)
		}
		if r.DecodeInstanceID == "" {
			t.Errorf("Disaggregations[%d]: DecodeInstanceID empty (decode-first: always set before disagg decision)", i)
		}
	}

	// THEN no PrefillRoutingRecord (skip path: no prefill routing)
	if len(tr.PrefillRoutings) != 0 {
		t.Errorf("PrefillRoutings=%d, want 0 (skip path)", len(tr.PrefillRoutings))
	}

	// THEN no KVTransferRecord (skip path: no KV transfer)
	if len(tr.KVTransfers) != 0 {
		t.Errorf("KVTransfers=%d, want 0 (skip path)", len(tr.KVTransfers))
	}

	// THEN no standard RoutingRecord (skip path uses DecodeEnqueueEvent, not RoutingDecisionEvent)
	if len(tr.Routings) != 0 {
		t.Errorf("Routings=%d, want 0 (PD skip path does not emit standard routing records)", len(tr.Routings))
	}
}

// TestPDTrace_NormalMode_NoDroppedUnservable verifies R1 invariant:
// under normal conditions (sufficient KV capacity) no decode sub-requests are
// dropped due to KV OOM. Dropped decode KV allocations are folded into
// DroppedUnservable in the aggregated metrics.
func TestPDTrace_NormalMode_NoDroppedUnservable(t *testing.T) {
	// GIVEN disaggregated simulation with ample KV capacity (100 blocks) and trace enabled
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.TraceLevel = "decisions"
	requests := newTestRequests(5)
	cs := NewClusterSimulator(config, requests, nil)

	// WHEN run
	mustRun(t, cs)

	// THEN no requests were dropped due to KV OOM (R1: counter not silently hidden)
	if cs.AggregatedMetrics().DroppedUnservable != 0 {
		t.Errorf("expected 0 DroppedUnservable under ample capacity, got %d", cs.AggregatedMetrics().DroppedUnservable)
	}

	// THEN trace machinery fired: all 5 requests have disaggregation + all downstream records
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace with trace-level decisions")
	}
	if len(tr.Disaggregations) != 5 {
		t.Errorf("expected 5 disaggregation records, got %d", len(tr.Disaggregations))
	}
	if len(tr.KVTransfers) != 5 {
		t.Errorf("expected 5 KV transfer records under ample capacity, got %d", len(tr.KVTransfers))
	}
}

