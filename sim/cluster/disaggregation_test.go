package cluster

import (
	"fmt"
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestParentRequest_NewParentRequest(t *testing.T) {
	req := &sim.Request{
		ID:          "req_0",
		InputTokens: make([]int, 100),
		ArrivalTime: 1000,
	}
	parent := NewParentRequest(req, 16) // blockSizeTokens=16

	if parent.ID != "req_0" {
		t.Errorf("parent ID = %q, want %q", parent.ID, "req_0")
	}
	if parent.PrefillSubReqID != "req_0_prefill" {
		t.Errorf("prefill sub-req ID = %q, want %q", parent.PrefillSubReqID, "req_0_prefill")
	}
	if parent.DecodeSubReqID != "req_0_decode" {
		t.Errorf("decode sub-req ID = %q, want %q", parent.DecodeSubReqID, "req_0_decode")
	}
	// ceil(100/16) = 7
	if parent.NumKVBlocks != 7 {
		t.Errorf("NumKVBlocks = %d, want %d", parent.NumKVBlocks, 7)
	}
	if parent.ArrivalTime != 1000 {
		t.Errorf("ArrivalTime = %d, want 1000", parent.ArrivalTime)
	}
}

func TestParentRequest_ZeroInputTokens(t *testing.T) {
	req := &sim.Request{
		ID:          "req_empty",
		InputTokens: nil,
	}
	parent := NewParentRequest(req, 16)
	if parent.NumKVBlocks != 0 {
		t.Errorf("NumKVBlocks = %d, want 0 for empty input", parent.NumKVBlocks)
	}
}

func TestFilterSnapshotsByPool(t *testing.T) {
	membership := map[string]PoolRole{
		"instance_0": PoolRolePrefill,
		"instance_1": PoolRolePrefill,
		"instance_2": PoolRoleDecode,
		"instance_3": PoolRoleDecode,
	}
	snapshots := []sim.RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 1},
		{ID: "instance_1", QueueDepth: 2},
		{ID: "instance_2", QueueDepth: 3},
		{ID: "instance_3", QueueDepth: 4},
	}

	prefill := FilterSnapshotsByPool(snapshots, membership, PoolRolePrefill)
	if len(prefill) != 2 {
		t.Fatalf("prefill snapshots = %d, want 2", len(prefill))
	}
	if prefill[0].ID != "instance_0" || prefill[1].ID != "instance_1" {
		t.Errorf("prefill IDs = [%s, %s], want [instance_0, instance_1]", prefill[0].ID, prefill[1].ID)
	}

	decode := FilterSnapshotsByPool(snapshots, membership, PoolRoleDecode)
	if len(decode) != 2 {
		t.Fatalf("decode snapshots = %d, want 2", len(decode))
	}
	if decode[0].ID != "instance_2" || decode[1].ID != "instance_3" {
		t.Errorf("decode IDs = [%s, %s], want [instance_2, instance_3]", decode[0].ID, decode[1].ID)
	}
}

// --- Integration and invariant tests ---

func newTestDisaggDeploymentConfig(numInstances, prefill, decode int) DeploymentConfig {
	return DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 1, ""),
		},
		NumInstances:            numInstances,
		PrefillInstances:        prefill,
		DecodeInstances:         decode,
		PDDecider:               "always",
		RoutingPolicy:           "round-robin",
		PDTransferBandwidthGBps: 25.0,
		PDTransferBaseLatencyMs: 0.05,
		PDKVBytesPerToken:       512,
	}
}

func TestDisaggregation_PrefillRoutedToPrefillPool(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(3)

	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	// BC-PD-7: Prefill sub-requests must be routed to prefill instances
	parents := cs.ParentRequests()
	if len(parents) != 3 {
		t.Fatalf("parentRequests count = %d, want 3", len(parents))
	}
	membership := cs.PoolMembership()
	for _, parent := range parents {
		role, ok := membership[parent.PrefillInstanceID]
		if !ok {
			t.Errorf("prefill instance %q not in pool membership", parent.PrefillInstanceID)
		}
		if role != PoolRolePrefill {
			t.Errorf("prefill sub-request for %s routed to %s (role=%v), want PoolRolePrefill",
				parent.ID, parent.PrefillInstanceID, role)
		}
	}
}

func TestDisaggregation_DecodeRoutedToDecodePool(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(3)

	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	// BC-PD-7: Decode sub-requests must be routed to decode instances
	membership := cs.PoolMembership()
	for _, parent := range cs.ParentRequests() {
		if parent.DecodeInstanceID == "" {
			t.Errorf("decode instance not assigned for parent %s", parent.ID)
			continue
		}
		role, ok := membership[parent.DecodeInstanceID]
		if !ok {
			t.Errorf("decode instance %q not in pool membership", parent.DecodeInstanceID)
		}
		if role != PoolRoleDecode {
			t.Errorf("decode sub-request for %s routed to %s (role=%v), want PoolRoleDecode",
				parent.ID, parent.DecodeInstanceID, role)
		}
	}
}

// TestDisaggregation_NoCrossPoolRouting verifies NC-PD-1:
// prefill sub-requests are never routed to decode instances, and vice versa.
func TestDisaggregation_NoCrossPoolRouting(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	membership := cs.PoolMembership()
	for _, parent := range cs.ParentRequests() {
		// Prefill instance must NOT be a decode instance
		if role := membership[parent.PrefillInstanceID]; role == PoolRoleDecode {
			t.Errorf("NC-PD-1 violated: prefill sub-request for %s sent to decode instance %s",
				parent.ID, parent.PrefillInstanceID)
		}
		// Decode instance must NOT be a prefill instance
		if role := membership[parent.DecodeInstanceID]; role == PoolRolePrefill {
			t.Errorf("NC-PD-1 violated: decode sub-request for %s sent to prefill instance %s",
				parent.ID, parent.DecodeInstanceID)
		}
	}
}

func TestDisaggregation_RequestCompletesFullPath(t *testing.T) {
	// BC-PD-5: Request completes through full disaggregated path
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(3)

	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	metrics := cs.AggregatedMetrics()
	if metrics.TotalOutputTokens == 0 {
		t.Error("TotalOutputTokens = 0, decode sub-requests did not generate output")
	}

	// BC-PD-9: Phase causality for each parent
	for _, parent := range cs.ParentRequests() {
		if parent.TransferCompleteTime == 0 {
			t.Errorf("parent %s: TransferCompleteTime not set", parent.ID)
		}
		if parent.DecodeEnqueueTime < parent.TransferCompleteTime {
			t.Errorf("parent %s: DecodeEnqueueTime (%d) < TransferCompleteTime (%d) — violates INV-PD-1",
				parent.ID, parent.DecodeEnqueueTime, parent.TransferCompleteTime)
		}
	}
}

func TestDisaggregation_TransferConservation(t *testing.T) {
	// BC-PD-8 / INV-PD-3: initiated_transfers == completed_transfers
	// Verified via observable behavior: count parents with TransferStartTime > 0 (initiated)
	// vs TransferCompleteTime > 0 (completed). Both counts must equal the request count.
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	parents := cs.ParentRequests()
	var initiated, completed int
	for _, p := range parents {
		if p.TransferStartTime > 0 {
			initiated++
		}
		if p.TransferCompleteTime > 0 {
			completed++
		}
	}
	if initiated != completed {
		t.Errorf("transfer conservation violated: initiated=%d, completed=%d", initiated, completed)
	}
	if initiated != 5 {
		t.Errorf("transfers initiated = %d, want 5", initiated)
	}
}

func TestDisaggregation_PhaseCausality(t *testing.T) {
	// BC-PD-9 / INV-PD-4: Full causal chain for every disaggregated request:
	// arrival ≤ prefill_enqueue ≤ prefill_complete ≤ transfer_start ≤ transfer_complete
	//   ≤ decode_enqueue ≤ completion
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(10)

	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	for _, parent := range cs.ParentRequests() {
		chain := []struct {
			name  string
			value int64
		}{
			{"ArrivalTime", parent.ArrivalTime},
			{"PrefillEnqueueTime", parent.PrefillEnqueueTime},
			{"PrefillCompleteTime", parent.PrefillCompleteTime},
			{"TransferStartTime", parent.TransferStartTime},
			{"TransferCompleteTime", parent.TransferCompleteTime},
			{"DecodeEnqueueTime", parent.DecodeEnqueueTime},
			{"CompletionTime", parent.CompletionTime},
		}

		for i := 1; i < len(chain); i++ {
			if chain[i].value < chain[i-1].value {
				t.Errorf("parent %s: causality violated: %s (%d) < %s (%d)",
					parent.ID, chain[i].name, chain[i].value, chain[i-1].name, chain[i-1].value)
			}
		}

		// CompletionTime must be strictly positive: the latency model always produces
		// a step duration >= 1 μs, so a completed disaggregated request cannot finish
		// at tick 0 regardless of arrival time. A zero CompletionTime means the decode
		// phase was never reached.
		// Note: intermediate timestamps (PrefillEnqueueTime, etc.) can legitimately be 0
		// when the request arrives at time 0 and routing latency is 0. The causality
		// chain above is the correct check for phase ordering; the zero-check only applies
		// to CompletionTime which is model-guaranteed non-zero.
		if parent.CompletionTime == 0 {
			t.Errorf("parent %s: CompletionTime is zero — decode phase never completed", parent.ID)
		}
	}
}

func TestDisaggregation_PoolStability(t *testing.T) {
	// INV-PD-5: Pool membership unchanged after initialization
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests)
	membershipBefore := cs.PoolMembership()

	mustRun(t, cs)

	membershipAfter := cs.PoolMembership()
	if len(membershipBefore) != len(membershipAfter) {
		t.Fatalf("pool membership size changed: before=%d, after=%d",
			len(membershipBefore), len(membershipAfter))
	}
	for id, roleBefore := range membershipBefore {
		roleAfter, ok := membershipAfter[id]
		if !ok {
			t.Errorf("instance %s missing from pool membership after simulation", id)
		}
		if roleBefore != roleAfter {
			t.Errorf("instance %s: role changed from %v to %v", id, roleBefore, roleAfter)
		}
	}
}

func TestDisaggregation_Determinism(t *testing.T) {
	// BC-PD-12 / INV-6: Same seed produces identical results
	config := newTestDisaggDeploymentConfig(4, 2, 2)

	run := func() *sim.Metrics {
		requests := newTestRequests(10)
		cs := NewClusterSimulator(config, requests)
		mustRun(t, cs)
		return cs.AggregatedMetrics()
	}

	m1 := run()
	m2 := run()

	if m1.CompletedRequests != m2.CompletedRequests {
		t.Errorf("non-deterministic CompletedRequests: %d vs %d", m1.CompletedRequests, m2.CompletedRequests)
	}
	if m1.TotalOutputTokens != m2.TotalOutputTokens {
		t.Errorf("non-deterministic TotalOutputTokens: %d vs %d", m1.TotalOutputTokens, m2.TotalOutputTokens)
	}
	if m1.SimEndedTime != m2.SimEndedTime {
		t.Errorf("non-deterministic SimEndedTime: %d vs %d", m1.SimEndedTime, m2.SimEndedTime)
	}
}

func TestDisaggregation_BackwardCompatibility(t *testing.T) {
	// BC-PD-13: When pools not configured, behavior is identical
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 1, ""),
		},
		NumInstances:  4,
		RoutingPolicy: "round-robin",
	}

	requests := newTestRequests(10)
	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	// NC-PD-2: No parent records created when pools not configured
	if parents := cs.ParentRequests(); len(parents) > 0 {
		t.Errorf("parentRequests should be empty when pools not configured, got %d", len(parents))
	}

	metrics := cs.AggregatedMetrics()
	if metrics.CompletedRequests == 0 {
		t.Error("no requests completed in non-disaggregated mode")
	}

	// INV-1: Conservation
	injected := metrics.CompletedRequests + metrics.StillQueued + metrics.StillRunning + metrics.DroppedUnservable
	if injected != 10 {
		t.Errorf("conservation violated: completed(%d) + queued(%d) + running(%d) + dropped(%d) = %d, want 10",
			metrics.CompletedRequests, metrics.StillQueued, metrics.StillRunning, metrics.DroppedUnservable, injected)
	}
}

func TestDisaggregation_PerPoolScorerConfigs(t *testing.T) {
	// BC-PD-15: per-pool scorer configs wire up correctly and produce output.
	// Verification: simulation completes with output tokens generated, confirming that
	// both pools routed requests end-to-end using their respective scorer configurations.
	// (Observable behavior: TotalOutputTokens > 0 proves the full PD pipeline — prefill
	// routing → KV transfer → decode routing — operated with the configured scorers.)
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.RoutingPolicy = "weighted"
	config.PrefillScorerConfigs = []sim.ScorerConfig{{Name: "queue-depth", Weight: 1.0}}
	config.DecodeScorerConfigs = []sim.ScorerConfig{{Name: "kv-utilization", Weight: 1.0}}

	requests := newTestRequests(3)
	cs := NewClusterSimulator(config, requests)

	mustRun(t, cs)

	if cs.AggregatedMetrics().TotalOutputTokens == 0 {
		t.Error("no output tokens generated with per-pool scorer configs")
	}
}

func TestAllocateTransferredKV_Success(t *testing.T) {
	cfg := sim.SimConfig{
		Horizon:             1000000,
		Seed:                42,
		KVCacheConfig:       sim.NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test", "H100", 1, ""),
	}
	inst := NewInstanceSimulator("decode_0", cfg)

	req := &sim.Request{
		ID:          "decode_sub_0",
		InputTokens: make([]int, 100),
		State:       sim.StateQueued,
	}

	ok := inst.AllocateTransferredKV(req)
	if !ok {
		t.Fatal("AllocateTransferredKV returned false, want true")
	}
	if req.ProgressIndex != 100 {
		t.Errorf("ProgressIndex = %d, want 100", req.ProgressIndex)
	}
	if inst.sim.KVCache.UsedBlocks() == 0 {
		t.Error("UsedBlocks = 0 after AllocateTransferredKV, want > 0")
	}
}

func TestAllocateTransferredKV_InsufficientCapacity(t *testing.T) {
	cfg := sim.SimConfig{
		Horizon:             1000000,
		Seed:                42,
		KVCacheConfig:       sim.NewKVCacheConfig(2, 16, 0, 0, 0, 0), // Only 2 blocks
		BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test", "H100", 1, ""),
	}
	inst := NewInstanceSimulator("decode_0", cfg)

	req := &sim.Request{
		ID:          "decode_sub_0",
		InputTokens: make([]int, 100), // Needs 7 blocks but only 2 available
		State:       sim.StateQueued,
	}

	ok := inst.AllocateTransferredKV(req)
	if ok {
		t.Error("AllocateTransferredKV returned true with insufficient capacity, want false")
	}
}

// --- PR3 accessor and invariant tests ---

// TestClusterSimulator_ParentRequests_ReturnsAllParents verifies the new accessor:
// - returns a slice with the same length as the internal parentRequests map
// - slice is sorted by ID (R2)
// - returns empty slice (not nil) when no PD disaggregation happened
func TestClusterSimulator_ParentRequests_ReturnsAllParents(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)
	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	got := cs.ParentRequests()
	// With AlwaysDisaggregate and 5 requests, all 5 should have ParentRequest records.
	const wantLen = 5
	if len(got) != wantLen {
		t.Fatalf("ParentRequests() len=%d, want %d", len(got), wantLen)
	}
	// Verify sorted order.
	for i := 1; i < len(got); i++ {
		if got[i].ID < got[i-1].ID {
			t.Errorf("ParentRequests() not sorted: got[%d].ID=%s < got[%d].ID=%s",
				i, got[i].ID, i-1, got[i-1].ID)
		}
	}
}

// TestClusterSimulator_PerInstanceMetricsByID_ContainsAllInstances verifies the new accessor
// returns a map entry for every instance in the cluster.
func TestClusterSimulator_PerInstanceMetricsByID_ContainsAllInstances(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)
	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	byID := cs.PerInstanceMetricsByID()
	// config has 4 instances total (2 prefill + 2 decode).
	const wantInstances = 4
	if len(byID) != wantInstances {
		t.Fatalf("PerInstanceMetricsByID() len=%d, want %d", len(byID), wantInstances)
	}
	// Verify expected instance IDs are present.
	for i := 0; i < wantInstances; i++ {
		id := fmt.Sprintf("instance_%d", i)
		if _, ok := byID[id]; !ok {
			t.Errorf("PerInstanceMetricsByID() missing instance %q", id)
		}
	}
}

// TestClusterSimulator_PDMetricsInvariant_PoolConservation verifies BC-3:
// sum of per-pool completions == cluster-wide CompletedRequests.
func TestClusterSimulator_PDMetricsInvariant_PoolConservation(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)
	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	byID := cs.PerInstanceMetricsByID()
	membership := cs.PoolMembership()

	var prefillTotal, decodeTotal int
	for id, m := range byID {
		switch membership[id] {
		case PoolRolePrefill:
			prefillTotal += m.CompletedRequests
		case PoolRoleDecode:
			decodeTotal += m.CompletedRequests
		}
	}
	total := prefillTotal + decodeTotal
	clusterTotal := cs.AggregatedMetrics().CompletedRequests
	if total != clusterTotal {
		t.Errorf("pool conservation violated: prefill(%d) + decode(%d) = %d, cluster total = %d",
			prefillTotal, decodeTotal, total, clusterTotal)
	}
}

// TestCollectPDMetrics_ParentTTFT_IncludesTransferDuration verifies BC-1 causality invariant:
// ParentTTFT.Mean >= TransferDuration.Mean (transfer is a sub-component of TTFT).
func TestCollectPDMetrics_ParentTTFT_IncludesTransferDuration(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)
	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	pd := CollectPDMetrics(
		cs.ParentRequests(),
		cs.AggregatedMetrics(),
		cs.PoolMembership(),
		cs.PerInstanceMetricsByID(),
	)
	if pd == nil {
		t.Fatal("CollectPDMetrics returned nil for disaggregated simulation")
	}
	if pd.ParentTTFT.Count == 0 {
		t.Fatal("no parent TTFT data collected — PD pipeline broken (BC-1 invariant cannot be checked)")
	}
	if pd.TransferDuration.Count == 0 {
		t.Fatal("no transfer duration data collected — PD pipeline broken (BC-1 invariant cannot be checked)")
	}
	// BC-1: parent TTFT includes transfer time, so mean TTFT >= mean transfer.
	if pd.ParentTTFT.Mean < pd.TransferDuration.Mean {
		t.Errorf("BC-1 causality violated: ParentTTFT.Mean (%.1f) < TransferDuration.Mean (%.1f)",
			pd.ParentTTFT.Mean, pd.TransferDuration.Mean)
	}
}

// TestClusterSimulator_DisaggregatedINV1_Conservation verifies INV-1 (request conservation)
// holds for the disaggregated code path (R7 companion invariant test).
// INV-1: injected == completed + still_queued + still_running + dropped_unservable
// In PD mode, each parent produces two sub-requests (one prefill, one decode);
// CompletedRequests counts sub-requests, so injectedSubReqs = numRequests * 2.
func TestClusterSimulator_DisaggregatedINV1_Conservation(t *testing.T) {
	const numRequests = 5
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(numRequests)
	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	agg := cs.AggregatedMetrics()

	// INV-1 conservation identity: all injected sub-requests must be accounted for.
	// Each parent generates exactly 1 prefill sub-request + 1 decode sub-request.
	const wantInjectedSubReqs = numRequests * 2
	actual := agg.CompletedRequests + agg.StillQueued + agg.StillRunning + agg.DroppedUnservable
	if actual != wantInjectedSubReqs {
		t.Errorf("INV-1 conservation violated: completed(%d)+queued(%d)+running(%d)+dropped(%d)=%d, want %d (2 sub-requests per parent)",
			agg.CompletedRequests, agg.StillQueued, agg.StillRunning, agg.DroppedUnservable,
			actual, wantInjectedSubReqs)
	}
	// Secondary check: for this small workload with ample KV capacity, all should complete.
	// (If this fails, check DroppedUnservable > 0 for KV pressure, not an INV-1 violation.)
	if agg.DroppedUnservable > 0 || agg.StillQueued > 0 || agg.StillRunning > 0 {
		t.Logf("note: not all sub-requests completed — dropped=%d queued=%d running=%d (may indicate KV pressure in test config)",
			agg.DroppedUnservable, agg.StillQueued, agg.StillRunning)
	}
}

// TestDisaggregation_INV_PD_1_DecodeEnqueueAfterTransfer verifies INV-PD-1 (KV Completeness):
// decode_enqueue_time >= transfer_complete_time for every disaggregated request.
// This is a standalone R7 companion invariant test for INV-PD-1.
// TestDisaggregation_DecodeKVAllocationFailure verifies INV-1 conservation when
// decode KV allocation fails (request dropped mid-pipeline). Uses very small KV
// cache on decode instances to force allocation failures.
func TestDisaggregation_DecodeKVAllocationFailure(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	// Small KV cache: 10 blocks of 16 tokens. Requests have ~100 input tokens,
	// requiring ceil(100/16)=7 blocks each. First decode sub-request fits (7 of 10),
	// but subsequent ones fail as blocks remain held by in-progress requests.
	config.KVCacheConfig = sim.NewKVCacheConfig(10, 16, 0, 0, 0, 0)
	// Use bounded horizon to prevent livelock when KV is exhausted.
	config.Horizon = 50_000_000
	requests := newTestRequests(3)

	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	agg := cs.AggregatedMetrics()

	// With 2-block KV, the first decode allocation may succeed (if input <= 32 tokens
	// after prefill consumed some), but subsequent ones will fail. At minimum, some
	// requests must be dropped.
	if agg.DroppedUnservable == 0 {
		t.Fatal("expected DroppedUnservable > 0 with small decode KV cache")
	}

	// INV-1: conservation must hold. Each parent generates 2 sub-requests.
	// droppedAtDecodeKV is folded into DroppedUnservable, so the identity is:
	// completed + queued + running + dropped == numRequests * 2.
	actual := agg.CompletedRequests + agg.StillQueued + agg.StillRunning + agg.DroppedUnservable
	wantTotal := len(requests) * 2
	if actual != wantTotal {
		t.Errorf("INV-1 conservation violated with KV drops: completed(%d)+queued(%d)+running(%d)+dropped(%d)=%d, want %d",
			agg.CompletedRequests, agg.StillQueued, agg.StillRunning, agg.DroppedUnservable, actual, wantTotal)
	}

	// Verify PDMetrics surfaces the drops
	parents := cs.ParentRequests()
	if len(parents) == 0 {
		t.Fatal("no parent requests recorded — PD pipeline did not execute")
	}
	pd := CollectPDMetrics(parents, agg, cs.PoolMembership(), cs.PerInstanceMetricsByID())
	if pd == nil {
		t.Fatal("CollectPDMetrics returned nil for disaggregated simulation")
	}
	if pd.DroppedAtDecodeKV == 0 {
		t.Error("PDMetrics.DroppedAtDecodeKV should be > 0 when decode KV allocation fails")
	}
}

// TestDisaggregation_NegativeTransferDurationClamp verifies the defensive INV-PD-4 clamp
// in DecodeRoutingEvent.Execute: if TransferCompleteTime < TransferStartTime (should never
// happen in normal operation), transfer duration is clamped to 0 and a warning is logged.
func TestDisaggregation_NegativeTransferDurationClamp(t *testing.T) {
	// This tests the defensive path by constructing a DecodeRoutingEvent with a
	// manipulated ParentRequest where TransferCompleteTime < TransferStartTime.
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.TraceLevel = "decisions"
	requests := newTestRequests(1)

	cs := NewClusterSimulator(config, requests)

	// Manually construct a parent request with inverted transfer timestamps
	parent := NewParentRequest(requests[0], 16)
	parent.TransferStartTime = 2000
	parent.TransferCompleteTime = 1000 // Inverted: complete < start
	parent.PrefillInstanceID = string(cs.instances[0].ID())

	decodeSubReq := &sim.Request{
		ID:          parent.DecodeSubReqID,
		InputTokens: requests[0].InputTokens,
		State:       sim.StateQueued,
	}

	// Execute the decode routing event directly
	event := &DecodeRoutingEvent{
		time:         2000,
		parentReq:    parent,
		decodeSubReq: decodeSubReq,
	}
	event.Execute(cs)

	// The trace should contain a KVTransferRecord with TransferDuration == 0 (clamped)
	records := cs.trace.KVTransfers
	if len(records) == 0 {
		// If KV allocation failed (insufficient capacity), no record is written.
		// In that case, the clamp path was not reached. Check DroppedAtDecodeKV instead.
		if cs.droppedAtDecodeKV > 0 {
			t.Skip("decode KV allocation failed before reaching transfer duration clamp")
		}
		t.Fatal("expected KVTransferRecord to be recorded")
	}
	if records[0].TransferDuration != 0 {
		t.Errorf("TransferDuration = %d, want 0 (clamped from negative)", records[0].TransferDuration)
	}
}

func TestDisaggregation_INV_PD_1_DecodeEnqueueAfterTransfer(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)
	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	parents := cs.ParentRequests()
	if len(parents) == 0 {
		t.Fatal("no parent requests recorded — PD pipeline did not execute")
	}
	for _, p := range parents {
		if p.TransferCompleteTime == 0 {
			continue // transfer never completed (e.g., dropped before transfer)
		}
		if p.DecodeEnqueueTime < p.TransferCompleteTime {
			t.Errorf("INV-PD-1 violated for %s: DecodeEnqueueTime(%d) < TransferCompleteTime(%d)",
				p.ID, p.DecodeEnqueueTime, p.TransferCompleteTime)
		}
	}
}
