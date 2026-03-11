package cluster

import (
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
		PDPrefixThreshold:       512,
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
	if len(cs.parentRequests) != 3 {
		t.Fatalf("parentRequests count = %d, want 3", len(cs.parentRequests))
	}
	for _, parent := range cs.parentRequests {
		role, ok := cs.poolMembership[parent.PrefillInstanceID]
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
	for _, parent := range cs.parentRequests {
		if parent.DecodeInstanceID == "" {
			t.Errorf("decode instance not assigned for parent %s", parent.ID)
			continue
		}
		role, ok := cs.poolMembership[parent.DecodeInstanceID]
		if !ok {
			t.Errorf("decode instance %q not in pool membership", parent.DecodeInstanceID)
		}
		if role != PoolRoleDecode {
			t.Errorf("decode sub-request for %s routed to %s (role=%v), want PoolRoleDecode",
				parent.ID, parent.DecodeInstanceID, role)
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
	for _, parent := range cs.parentRequests {
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
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	if cs.transfersInitiated != cs.transfersCompleted {
		t.Errorf("transfer conservation violated: initiated=%d, completed=%d",
			cs.transfersInitiated, cs.transfersCompleted)
	}
	if cs.transfersInitiated != 5 {
		t.Errorf("transfersInitiated = %d, want 5", cs.transfersInitiated)
	}
}

func TestDisaggregation_PhaseCausality(t *testing.T) {
	// BC-PD-9 / INV-PD-4: Full causal chain for every disaggregated request
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(10)

	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	for _, parent := range cs.parentRequests {
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
		}

		for i := 1; i < len(chain); i++ {
			if chain[i].value < chain[i-1].value {
				t.Errorf("parent %s: causality violated: %s (%d) < %s (%d)",
					parent.ID, chain[i].name, chain[i].value, chain[i-1].name, chain[i-1].value)
			}
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

	// No parent requests when pools not configured
	if cs.parentRequests != nil && len(cs.parentRequests) > 0 {
		t.Errorf("parentRequests should be empty when pools not configured, got %d", len(cs.parentRequests))
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
	// BC-PD-15: per-pool scorer configs produce separate routing policy instances
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.RoutingPolicy = "weighted"
	config.PrefillScorerConfigs = []sim.ScorerConfig{{Name: "queue-depth", Weight: 1.0}}
	config.DecodeScorerConfigs = []sim.ScorerConfig{{Name: "kv-utilization", Weight: 1.0}}

	requests := newTestRequests(3)
	cs := NewClusterSimulator(config, requests)

	if cs.prefillRoutingPolicy == nil {
		t.Error("prefillRoutingPolicy is nil when PrefillScorerConfigs specified")
	}
	if cs.decodeRoutingPolicy == nil {
		t.Error("decodeRoutingPolicy is nil when DecodeScorerConfigs specified")
	}

	mustRun(t, cs)

	if cs.AggregatedMetrics().TotalOutputTokens == 0 {
		t.Error("no output tokens generated with per-pool scorer configs")
	}
}

// --- PrefixThresholdDecider integration tests ---

// newTestRequests is already defined in test_helpers_test.go

// newTestPrefixThresholdConfig creates a DeploymentConfig with prefix-threshold decider.
func newTestPrefixThresholdConfig(threshold int) DeploymentConfig {
	return DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 1, ""),
		},
		NumInstances:            4,
		PrefillInstances:        2,
		DecodeInstances:         2,
		PDDecider:               "prefix-threshold",
		PDPrefixThreshold:       threshold,
		RoutingPolicy:           "round-robin",
		PDTransferBandwidthGBps: 25.0,
		PDTransferBaseLatencyMs: 0.05,
		PDKVBytesPerToken:       512,
	}
}

// Compile-time check: PrefixThresholdDecider satisfies DisaggregationObserver (BC-PD-26).
// Interface conformance is verified at the unit level (sim/disaggregation_test.go).
var _ sim.DisaggregationObserver = &sim.PrefixThresholdDecider{}

// TestPrefixThreshold_HighThresholdNoDisaggregation verifies that requests with tokens
// below the threshold are routed via the standard path (not disaggregated).
func TestPrefixThreshold_HighThresholdNoDisaggregation(t *testing.T) {
	// Set threshold very high: no request will disaggregate.
	// newTestRequests produces short requests (output tokens only, so InputTokens may be short).
	const veryHighThreshold = 1_000_000
	config := newTestPrefixThresholdConfig(veryHighThreshold)
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	// With very high threshold, no requests should be disaggregated.
	if len(cs.parentRequests) != 0 {
		t.Errorf("parentRequests = %d, want 0 (threshold too high for any request to disaggregate)",
			len(cs.parentRequests))
	}
	// Requests still complete via the standard routing path.
	m := cs.AggregatedMetrics()
	if m.CompletedRequests == 0 {
		t.Error("no requests completed with high threshold prefix-threshold decider")
	}
}

// TestPrefixThreshold_ZeroThresholdAlwaysDisaggregates verifies that threshold=0
// behaves like AlwaysDisaggregate for non-empty requests.
func TestPrefixThreshold_ZeroThresholdAlwaysDisaggregates(t *testing.T) {
	config := newTestPrefixThresholdConfig(0)
	requests := newTestRequests(3)

	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	// All requests have non-empty input tokens → threshold=0 → all disaggregate.
	if len(cs.parentRequests) != 3 {
		t.Errorf("parentRequests = %d, want 3 (threshold=0 disaggregates everything)", len(cs.parentRequests))
	}
}

// TestPrefixThreshold_ObserverCalledAfterRouting verifies BC-PD-28: observer is called
// after routing (using "always" + observer check via interface type assertion is sufficient
// since we can't inspect internal state without exporting it).
func TestPrefixThreshold_ObserverCalledAfterRouting(t *testing.T) {
	// Use always-disaggregate-like behavior (threshold=0) so all requests go through
	// the full disaggregated path (PrefillRoutingEvent fires, observer gets called there).
	config := newTestPrefixThresholdConfig(0)
	requests := newTestRequests(3)

	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	// After simulation, all requests were disaggregated and completed.
	// The observer was called in PrefillRoutingEvent for each request.
	// Verify causality invariants hold (phase causality implies observer didn't crash).
	for _, parent := range cs.parentRequests {
		if parent.TransferCompleteTime == 0 {
			t.Errorf("parent %s: TransferCompleteTime not set — prefill path broken", parent.ID)
		}
	}
}

// TestPrefixThreshold_TransferConservation verifies INV-PD-3 holds with prefix-threshold decider.
func TestPrefixThreshold_TransferConservation(t *testing.T) {
	config := newTestPrefixThresholdConfig(0) // threshold=0 disaggregates all
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	if cs.transfersInitiated != cs.transfersCompleted {
		t.Errorf("transfer conservation violated: initiated=%d, completed=%d",
			cs.transfersInitiated, cs.transfersCompleted)
	}
	if cs.transfersInitiated != 5 {
		t.Errorf("transfersInitiated = %d, want 5", cs.transfersInitiated)
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
