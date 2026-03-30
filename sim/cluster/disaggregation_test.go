package cluster

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
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

// --- Integration and invariant tests ---

// newTestDisaggDeploymentConfigWithOverhead creates a 4-instance (2 prefill, 2 decode)
// disaggregated DeploymentConfig using trained-roofline with the given post-decode
// overhead (µs). alpha[1] = overhead, so PostDecodeFixedOverhead() == overhead.
// Used to test that detectDecodeCompletions stamps parent.CompletionTime correctly
// when overhead > 0 (issue #846).
func newTestDisaggDeploymentConfigWithOverhead(overhead float64) DeploymentConfig {
	// Minimal trained-roofline model: 2-layer, 4-head, 64-dim with positive HW numbers.
	// beta[4] = 100 µs/layer gives finite step times; remaining betas zero.
	// NumKVHeads=0 triggers MHA fallback (uses NumHeads), divisible by TP=1.
	modelCfg := sim.ModelConfig{
		NumLayers:       2,
		NumHeads:        4,
		HiddenDim:       64,
		IntermediateDim: 128,
		BytesPerParam:   2.0,
	}
	hwCfg := sim.HardwareCalib{TFlopsPeak: 1.0, BwPeakTBs: 0.001}
	betas := []float64{0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0} // β₅ = 100 µs/layer
	alphas := []float64{0.0, overhead, 0.0}                   // α₁ = overhead (PostDecodeFixedOverhead)
	return DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs(betas, alphas),
			ModelHardwareConfig: sim.NewModelHardwareConfig(modelCfg, hwCfg, "test-model", "H100", 1, "trained-roofline", 0),
		},
		NumInstances:            4,
		PrefillInstances:        2,
		DecodeInstances:         2,
		PDDecider:               "always",
		RoutingPolicy:           "round-robin",
		PDTransferBandwidthGBps: 25.0,
		PDTransferBaseLatencyMs: 0.05,
		PDKVBytesPerToken:       512,
	}
}

func newTestDisaggDeploymentConfig(numInstances, prefill, decode int) DeploymentConfig {
	return DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 1, "blackbox", 0),
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

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	// BC-PD-7: Prefill sub-requests must be routed to prefill instances
	if len(cs.parentRequests) != 3 {
		t.Fatalf("parentRequests count = %d, want 3", len(cs.parentRequests))
	}
	for _, parent := range cs.parentRequests {
		role, ok := cs.poolMembership[string(parent.PrefillInstanceID)]
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

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	// BC-PD-7: Decode sub-requests must be routed to decode instances
	for _, parent := range cs.parentRequests {
		if parent.DecodeInstanceID == "" {
			t.Errorf("decode instance not assigned for parent %s", parent.ID)
			continue
		}
		role, ok := cs.poolMembership[string(parent.DecodeInstanceID)]
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

	cs := NewClusterSimulator(config, requests, nil)
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

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	if cs.transfersInitiated != cs.transfersCompleted {
		t.Errorf("transfer conservation violated: initiated=%d, completed=%d",
			cs.transfersInitiated, cs.transfersCompleted)
	}
	if cs.transfersInitiated != len(requests) {
		t.Errorf("transfersInitiated = %d, want %d", cs.transfersInitiated, len(requests))
	}
}

// assertINV1Conservation checks the full INV-1 conservation equation including TimedOutRequests.
func assertINV1Conservation(t *testing.T, metrics *sim.Metrics, expected int, label string) {
	t.Helper()
	sum := metrics.CompletedRequests + metrics.StillQueued + metrics.StillRunning +
		metrics.DroppedUnservable + metrics.TimedOutRequests
	if sum != expected {
		t.Errorf("INV-1 conservation violated (%s): completed(%d) + queued(%d) + running(%d) + dropped(%d) + timedOut(%d) = %d, want %d",
			label, metrics.CompletedRequests, metrics.StillQueued, metrics.StillRunning,
			metrics.DroppedUnservable, metrics.TimedOutRequests, sum, expected)
	}
}

func TestDisaggregation_INV1Conservation(t *testing.T) {
	// INV-1: CompletedRequests + StillQueued + StillRunning + DroppedUnservable + TimedOutRequests == N
	// in disaggregated mode (must not double-count prefill + decode sub-requests)
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	metrics := cs.AggregatedMetrics()
	if metrics.CompletedRequests != 5 {
		t.Errorf("INV-1: CompletedRequests = %d, want 5 (possible double-counting of sub-requests)",
			metrics.CompletedRequests)
	}
	assertINV1Conservation(t, metrics, 5, "disaggregated mode")
}

func TestDisaggregation_INV1Conservation_BoundedHorizon(t *testing.T) {
	// INV-1 at bounded horizon: requests with completed prefills but in-flight KV
	// transfers must be accounted for (counted in StillRunning, not lost).
	// Use a horizon long enough for all requests to arrive and enter PD pipeline,
	// but verify that pdInFlight accounting prevents conservation gaps.
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.Horizon = 5000000 // 5 seconds — all requests arrive, most but maybe not all complete
	requests := newTestRequests(10)

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	metrics := cs.AggregatedMetrics()
	// All 10 requests should have arrived within the horizon (last arrives at ~900000 μs).
	// The pdInTransfer correction ensures requests mid-transfer are counted in StillRunning.
	assertINV1Conservation(t, metrics, 10, "bounded horizon")
	// Verify pdInTransfer accounting is non-negative (no over-subtraction)
	pdInTransfer := cs.pdPrefillCompletedCount - cs.pdDecodeCompletedCount - cs.droppedAtDecodeKV - len(cs.pendingDecodeCompletions)
	if pdInTransfer < 0 {
		t.Errorf("pdInTransfer = %d, must be >= 0 (prefillCompleted=%d, decodeCompleted=%d, droppedAtDecodeKV=%d, pendingDecode=%d)",
			pdInTransfer, cs.pdPrefillCompletedCount, cs.pdDecodeCompletedCount, cs.droppedAtDecodeKV, len(cs.pendingDecodeCompletions))
	}
}

func TestDisaggregation_DecodeOnlyBatchKVPressure(t *testing.T) {
	// Verify that the decode-only batch path handles KV pressure correctly:
	// when KV cache is nearly full, the decode-only path breaks (does not crash)
	// and the request stays in the wait queue.
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.KVCacheConfig = sim.NewKVCacheConfig(50, 16, 0, 0, 0, 0) // small KV cache
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	metrics := cs.AggregatedMetrics()
	// Under tight KV pressure, some requests may be dropped — conservation must hold
	assertINV1Conservation(t, metrics, 5, "KV pressure")
}

func newShortRequests(n int) []*sim.Request {
	// Create requests with short input (20 tokens = 2 blocks at blockSize=16) and
	// short output (5 tokens) to complete quickly. Spaced 1000μs apart so multiple
	// prefills complete and transfers land on the decode instance concurrently.
	requests := make([]*sim.Request, n)
	for i := 0; i < n; i++ {
		requests[i] = &sim.Request{
			ID:          fmt.Sprintf("request_%d", i),
			InputTokens: make([]int, 20), // 2 blocks at blockSize=16
			OutputTokens: make([]int, 5),
			State:       sim.StateQueued,
			ArrivalTime: int64(i * 1000), // 1000μs apart
		}
	}
	return requests
}

func TestDisaggregation_DroppedAtDecodeKV(t *testing.T) {
	// Verify that droppedAtDecodeKV is triggered and counted in DroppedUnservable
	// when decode instances have insufficient KV capacity for transferred input.
	// Strategy: 1 decode instance with only 3 blocks (48 tokens). Each request needs
	// 2 blocks (20 tokens). First request fills 2/3 blocks, second request tries to
	// allocate 2 more but only 1 free → AllocateTransferredKV fails.
	config := newTestDisaggDeploymentConfig(3, 2, 1) // 2 prefill, 1 decode
	config.KVCacheConfig = sim.NewKVCacheConfig(3, 16, 0, 0, 0, 0) // 3 blocks = 48 tokens

	requests := newShortRequests(4)
	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	if cs.droppedAtDecodeKV == 0 {
		t.Error("droppedAtDecodeKV = 0, expected > 0 with 1 decode instance and tight KV")
	}

	metrics := cs.AggregatedMetrics()
	// INV-1 conservation must hold even when decode drops occur
	assertINV1Conservation(t, metrics, 4, "decode KV drops")
}

func TestDisaggregation_PhaseCausality(t *testing.T) {
	// BC-PD-9 / INV-PD-4: Full causal chain for every disaggregated request
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(10)

	cs := NewClusterSimulator(config, requests, nil)
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
		// Note: CompletionTime is not included in the chain because it is set by
		// detectDecodeCompletions using c.clock at detection time, which may differ
		// from the actual decode completion instant. A dedicated CompletionTime test
		// would need to use instance-level RequestCompletionTimes directly.

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

	cs := NewClusterSimulator(config, requests, nil)
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
		cs := NewClusterSimulator(config, requests, nil)
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
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 1, "blackbox", 0),
		},
		NumInstances:  4,
		RoutingPolicy: "round-robin",
	}

	requests := newTestRequests(10)
	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	// No parent requests when pools not configured
	if len(cs.parentRequests) > 0 {
		t.Errorf("parentRequests should be empty when pools not configured, got %d", len(cs.parentRequests))
	}

	metrics := cs.AggregatedMetrics()
	if metrics.CompletedRequests == 0 {
		t.Error("no requests completed in non-disaggregated mode")
	}

	// INV-1: Conservation
	assertINV1Conservation(t, metrics, 10, "non-disaggregated backward compat")
}

func TestDisaggregation_PerPoolScorerConfigs(t *testing.T) {
	// BC-PD-15: per-pool scorer configs produce separate routing policy instances
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.RoutingPolicy = "weighted"
	config.PrefillScorerConfigs = []sim.ScorerConfig{{Name: "queue-depth", Weight: 1.0}}
	config.DecodeScorerConfigs = []sim.ScorerConfig{{Name: "kv-utilization", Weight: 1.0}}

	requests := newTestRequests(3)
	cs := NewClusterSimulator(config, requests, nil)

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

func TestAllocateTransferredKV_Success(t *testing.T) {
	cfg := sim.SimConfig{
		Horizon:             1000000,
		Seed:                42,
		KVCacheConfig:       sim.NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test", "H100", 1, "blackbox", 0),
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
		ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test", "H100", 1, "blackbox", 0),
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

// TestPDDisagg_OneOutputToken_CompletesWith1Token is a regression test for two edge-case
// bugs discovered during PD disaggregation development:
//
//  1. processCompletions used == instead of >= for the completion check. In PD
//     mode, a 1-output-token decode sub-request enters with ProgressIndex ==
//     inputLen; after one decode step ProgressIndex becomes inputLen+1, which
//     missed the == threshold, triggered a second decode step, and called
//     AllocateKVBlocks with an out-of-bounds index, producing a phantom token.
//     Fixed by changing == to >= and adding a ProgressIndex < inputLen+outputLen
//     bounds guard on the AllocateKVBlocks call.
//
//  2. FormBatch Phase 2 used a ProgressIndex >= inputLen heuristic to detect PD
//     decode sub-requests. A zero-input non-PD request satisfied this vacuously
//     (0 >= 0) and incorrectly took the decode-only fast-path. Fixed by
//     replacing the heuristic with an explicit IsDecodeSubRequest flag set only
//     by KVTransferCompletedEvent.
func TestPDDisagg_OneOutputToken_CompletesWith1Token(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)

	requests := []*sim.Request{
		{
			ID:           "req-1output",
			ArrivalTime:  0,
			InputTokens:  make([]int, 20),
			OutputTokens: []int{42}, // exactly 1 output token
			State:        sim.StateQueued,
		},
	}

	cs := NewClusterSimulator(config, requests, nil)
	if err := cs.Run(); err != nil {
		t.Fatalf("ClusterSimulator.Run: %v", err)
	}

	metrics := cs.AggregatedMetrics()
	// INV-1: all requests accounted for.
	assertINV1Conservation(t, metrics, 1, "1-output PD request")
	// The request must produce exactly 1 output token — not 0 (hung) or 2+ (phantom decode step).
	if metrics.TotalOutputTokens != 1 {
		t.Errorf("TotalOutputTokens = %d, want 1 (off-by-one would produce 2)", metrics.TotalOutputTokens)
	}
}

// --- PrefixThresholdDecider cluster integration tests ---

// newTestPrefixThresholdConfig returns a DeploymentConfig with PDDecider = "prefix-threshold"
// and the specified threshold, reusing the standard 4-instance (2 prefill, 2 decode) topology.
func newTestPrefixThresholdConfig(threshold int) DeploymentConfig {
	cfg := newTestDisaggDeploymentConfig(4, 2, 2)
	cfg.PDDecider = "prefix-threshold"
	cfg.PDPrefixThreshold = threshold
	return cfg
}

// TestPrefixThreshold_BelowThresholdNotDisaggregated verifies BC-PD-21 at the cluster level:
// requests with non-cached token counts well below the threshold must not be disaggregated
// (absent from parentRequests). Tests the full NewClusterSimulator → PrefixThresholdDecider
// constructor path and the DisaggregationDecisionEvent bifurcation.
func TestPrefixThreshold_BelowThresholdNotDisaggregated(t *testing.T) {
	const threshold = 200
	config := newTestPrefixThresholdConfig(threshold)

	// Requests with 20 unique tokens: nonCached = 20, 20 <= 200 → should NOT disaggregate.
	requests := make([]*sim.Request, 3)
	for i := range requests {
		tokens := make([]int, 20)
		for j := range tokens {
			tokens[j] = j + i*1000 + 1 // unique across requests, no prefix cache hit
		}
		requests[i] = &sim.Request{
			ID:           fmt.Sprintf("short_%d", i),
			InputTokens:  tokens,
			OutputTokens: make([]int, 5),
			State:        sim.StateQueued,
			ArrivalTime:  int64(i * 100000),
		}
	}

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	if len(cs.parentRequests) != 0 {
		t.Errorf("parentRequests = %d, want 0: short requests (20 tokens <= %d threshold) should not be disaggregated",
			len(cs.parentRequests), threshold)
	}
	// INV-1: below-threshold requests route through RoutingDecisionEvent; verify all complete.
	assertINV1Conservation(t, cs.AggregatedMetrics(), len(requests), "below-threshold")
}

// TestPrefixThreshold_AboveThresholdDisaggregated verifies BC-PD-21 at the cluster level:
// requests with non-cached token counts above the threshold must be disaggregated
// (present in parentRequests).
func TestPrefixThreshold_AboveThresholdDisaggregated(t *testing.T) {
	const threshold = 200
	config := newTestPrefixThresholdConfig(threshold)

	// Requests with 400 unique tokens: nonCached = 400, 400 > 200 → must disaggregate.
	requests := make([]*sim.Request, 3)
	for i := range requests {
		tokens := make([]int, 400)
		for j := range tokens {
			tokens[j] = j + i*10000 + 1 // unique across requests, no prefix cache hit
		}
		requests[i] = &sim.Request{
			ID:           fmt.Sprintf("long_%d", i),
			InputTokens:  tokens,
			OutputTokens: make([]int, 5),
			State:        sim.StateQueued,
			ArrivalTime:  int64(i * 500000),
		}
	}

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	if len(cs.parentRequests) != 3 {
		t.Errorf("parentRequests = %d, want 3: long requests (400 tokens > %d threshold) should all be disaggregated",
			len(cs.parentRequests), threshold)
	}
}

// TestPrefixThreshold_ObserverWarmsCache verifies BC-PD-24 at the cluster level:
// req1 (disaggregated) warms the prefix cache via notifyDisaggregationObserver in
// PrefillRoutingEvent.Execute (pd_events.go); req2 (non-disaggregated) verifies that
// the warmed cache is consulted in DisaggregationDecisionEvent, reducing non-cached
// token count below the threshold. Tests the full end-to-end wiring path.
func TestPrefixThreshold_ObserverWarmsCache(t *testing.T) {
	const threshold = 300
	const blockSize = 16
	config := newTestPrefixThresholdConfig(threshold)

	// req1: 400 tokens (25 complete blocks), no prior cache.
	// nonCached = 400 > 300 → disaggregated; ObserveRouting warms 25 blocks in cache.
	prefix := make([]int, 400)
	for i := range prefix {
		prefix[i] = i + 1
	}
	req1 := &sim.Request{
		ID:           "req-warm",
		InputTokens:  append([]int{}, prefix...),
		OutputTokens: make([]int, 5),
		State:        sim.StateQueued,
		ArrivalTime:  0,
	}

	// req2: same 400-token prefix + 50 new tokens = 450 total.
	// After req1 warms the cache: nonCached = 450 - 25*16 = 450 - 400 = 50 <= 300 → NOT disaggregated.
	// req2 arrives 2s after req1, well after req1's PrefillRoutingEvent fires and warms the cache.
	extended := make([]int, len(prefix)+50)
	copy(extended, prefix)
	for i := len(prefix); i < len(extended); i++ {
		extended[i] = 10000 + i
	}
	req2 := &sim.Request{
		ID:           "req-follow",
		InputTokens:  extended,
		OutputTokens: make([]int, 5),
		State:        sim.StateQueued,
		ArrivalTime:  2000000, // req1's PrefillRoutingEvent fires at t=0+routingLatency=0; req2 arrives at t=2,000,000;
		// ordering is guaranteed by event timestamps alone (t=0 < t=2,000,000), not the gap magnitude
	}
	_ = blockSize // documents the block arithmetic above

	cs := NewClusterSimulator(config, []*sim.Request{req1, req2}, nil)
	mustRun(t, cs)

	// req1 must be disaggregated (400 non-cached tokens > 300 threshold).
	var req1Disaggregated bool
	for _, pr := range cs.parentRequests {
		if pr.ID == "req-warm" {
			req1Disaggregated = true
		}
	}
	if !req1Disaggregated {
		t.Error("req-warm (400 uncached tokens > 300 threshold) must be disaggregated; " +
			"check PrefixThresholdDecider constructor wiring in NewClusterSimulator")
	}

	// req2 must NOT be disaggregated: prefix is cached after req1's PrefillRoutingEvent
	// fires ObserveRouting, so only 50 tokens are non-cached (50 <= 300 threshold).
	for _, pr := range cs.parentRequests {
		if pr.ID == "req-follow" {
			t.Error("req-follow (50 non-cached tokens <= 300 threshold after cache warming) must NOT be disaggregated; " +
				"check notifyDisaggregationObserver wiring in PrefillRoutingEvent.Execute (pd_events.go)")
		}
	}
}

func newTestDirectToDecodeConfig(threshold int) DeploymentConfig {
	cfg := newTestDisaggDeploymentConfig(4, 2, 2)
	cfg.PDDecider = "direct-to-decode"
	cfg.PDDirectDecodeThreshold = threshold
	return cfg
}

// TestDirectToDecode_BelowThresholdNotDisaggregated verifies the cluster-level wiring for
// direct-to-decode: requests with fewer tokens than the threshold must not be disaggregated.
// Tests the full NewClusterSimulator → NewDirectToDecodeDecider constructor path and
// the DisaggregationDecisionEvent bifurcation.
func TestDirectToDecode_BelowThresholdNotDisaggregated(t *testing.T) {
	const threshold = 200
	config := newTestDirectToDecodeConfig(threshold)

	// Requests with 20 input tokens: 20 < 200 threshold → must NOT disaggregate.
	requests := make([]*sim.Request, 3)
	for i := range requests {
		requests[i] = &sim.Request{
			ID:           fmt.Sprintf("short_%d", i),
			InputTokens:  make([]int, 20),
			OutputTokens: make([]int, 5),
			State:        sim.StateQueued,
			ArrivalTime:  int64(i * 100000),
		}
	}

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	if len(cs.parentRequests) != 0 {
		t.Errorf("parentRequests = %d, want 0: short requests (20 tokens < %d threshold) should not be disaggregated",
			len(cs.parentRequests), threshold)
	}
	assertINV1Conservation(t, cs.AggregatedMetrics(), len(requests), "direct-to-decode below threshold")
}

// TestDirectToDecode_AboveThresholdDisaggregated verifies the cluster-level wiring for
// direct-to-decode: requests with at least as many tokens as the threshold must be disaggregated.
func TestDirectToDecode_AboveThresholdDisaggregated(t *testing.T) {
	const threshold = 200
	config := newTestDirectToDecodeConfig(threshold)

	// Requests with 400 input tokens: 400 >= 200 threshold → must disaggregate.
	requests := make([]*sim.Request, 3)
	for i := range requests {
		requests[i] = &sim.Request{
			ID:           fmt.Sprintf("long_%d", i),
			InputTokens:  make([]int, 400),
			OutputTokens: make([]int, 5),
			State:        sim.StateQueued,
			ArrivalTime:  int64(i * 500000),
		}
	}

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	if len(cs.parentRequests) != 3 {
		t.Errorf("parentRequests = %d, want 3: long requests (400 tokens >= %d threshold) should all be disaggregated",
			len(cs.parentRequests), threshold)
	}
	// INV-1: disaggregated requests complete via the full prefill → KV-transfer → decode path;
	// verify no sub-request is double-counted or silently dropped.
	assertINV1Conservation(t, cs.AggregatedMetrics(), len(requests), "direct-to-decode above threshold")
}

// --- Metric projection tests (INV-PD-6, Issue #821) ---

// hasSubRequestSuffix returns true if key ends with "_prefill" or "_decode".
func hasSubRequestSuffix(key string) bool {
	return (len(key) >= 8 && key[len(key)-8:] == "_prefill") ||
		(len(key) >= 7 && key[len(key)-7:] == "_decode")
}

// TestDisaggregation_MetricProjection_NoOp verifies that projectPDMetrics is a
// no-op when disaggregation is not active (parentRequests is empty).
// GIVEN a non-disaggregated cluster (all instances in the default pool)
// WHEN  Run() completes
// THEN  per-request maps contain the original request keys, unmodified by projection.
func TestDisaggregation_MetricProjection_NoOp(t *testing.T) {
	config := newTestDeploymentConfig(2) // standard cluster, no PD roles
	requests := newTestRequests(3)

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	if len(cs.parentRequests) != 0 {
		t.Fatalf("expected no parentRequests in non-disaggregated cluster, got %d", len(cs.parentRequests))
	}

	m := cs.AggregatedMetrics()
	// All injected request IDs should appear in RequestE2Es (no suffix mangling).
	for _, req := range requests {
		if _, ok := m.RequestE2Es[req.ID]; !ok {
			// Some requests may not complete (e.g., horizon), but no key should have a suffix.
			continue
		}
		if hasSubRequestSuffix(req.ID) {
			t.Errorf("non-disaggregated cluster: original request key %q has a sub-request suffix", req.ID)
		}
	}
	// None of the map keys should carry a sub-request suffix.
	for mapName, keys := range map[string][]string{
		"RequestE2Es":             mapKeys(m.RequestE2Es),
		"RequestTTFTs":            mapKeys(m.RequestTTFTs),
		"RequestITLs":             mapKeys(m.RequestITLs),
		"RequestCompletionTimes":  mapKeys(m.RequestCompletionTimes),
		"RequestSchedulingDelays": mapKeysInt64(m.RequestSchedulingDelays),
		"Requests":                mapKeysRM(m.Requests),
	} {
		for _, key := range keys {
			if hasSubRequestSuffix(key) {
				t.Errorf("non-disaggregated cluster: %s contains sub-request key %q", mapName, key)
			}
		}
	}
}

// TestDisaggregation_MetricProjection_NoSubRequestKeys verifies INV-PD-6:
// after Run(), per-request metric maps contain only parent-level IDs.
// GIVEN a PD disaggregation simulation with N requests
// WHEN  all requests complete through the full disaggregated path
// THEN  no per-request map entry has a "_prefill" or "_decode" suffix.
func TestDisaggregation_MetricProjection_NoSubRequestKeys(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	maps := map[string][]string{
		"RequestE2Es":             mapKeys(m.RequestE2Es),
		"RequestTTFTs":            mapKeys(m.RequestTTFTs),
		"RequestITLs":             mapKeys(m.RequestITLs),
		"RequestCompletionTimes":  mapKeys(m.RequestCompletionTimes),
		"RequestSchedulingDelays": mapKeysInt64(m.RequestSchedulingDelays),
		"Requests":                mapKeysRM(m.Requests),
	}

	for mapName, keys := range maps {
		for _, key := range keys {
			if hasSubRequestSuffix(key) {
				t.Errorf("INV-PD-6 violated: %s contains sub-request key %q", mapName, key)
			}
		}
	}
}

// TestDisaggregation_MetricProjection_E2ECount verifies that the E2E
// distribution has exactly N entries (not 2N sub-request entries).
func TestDisaggregation_MetricProjection_E2ECount(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	if m.CompletedRequests != 5 {
		t.Fatalf("CompletedRequests = %d, want 5", m.CompletedRequests)
	}
	if len(m.RequestE2Es) != m.CompletedRequests {
		t.Errorf("len(RequestE2Es) = %d, want %d (CompletedRequests)", len(m.RequestE2Es), m.CompletedRequests)
	}
}

// TestDisaggregation_MetricProjection_E2ECorrectness verifies that each
// parent E2E = CompletionTime - ArrivalTime, and E2E > TTFT.
func TestDisaggregation_MetricProjection_E2ECorrectness(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	for _, parent := range cs.parentRequests {
		if parent.CompletionTime == 0 || parent.DecodeInstanceID == "" {
			continue // skip incomplete/dropped
		}
		pid := parent.ID
		expectedE2E := float64(parent.CompletionTime - parent.ArrivalTime)

		e2e, ok := m.RequestE2Es[pid]
		if !ok {
			t.Errorf("parent %s: missing RequestE2Es entry", pid)
			continue
		}
		if math.Abs(e2e-expectedE2E) > 1e-9 {
			t.Errorf("parent %s: E2E = %.0f, want %.0f (CompletionTime-ArrivalTime)",
				pid, e2e, expectedE2E)
		}

		ttft, hasTTFT := m.RequestTTFTs[pid]
		if hasTTFT && e2e <= ttft {
			t.Errorf("parent %s: E2E (%.0f) <= TTFT (%.0f), E2E must exceed TTFT (includes decode)",
				pid, e2e, ttft)
		}
	}
}

// TestDisaggregation_MetricProjection_SchedulingDelay verifies that the
// scheduling delay is the prefill sub-request delay (not inflated by decode pipeline).
func TestDisaggregation_MetricProjection_SchedulingDelay(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	for _, parent := range cs.parentRequests {
		if parent.CompletionTime == 0 || parent.DecodeInstanceID == "" {
			continue
		}
		pid := parent.ID
		delay, ok := m.RequestSchedulingDelays[pid]
		if !ok {
			t.Errorf("parent %s: missing RequestSchedulingDelays entry", pid)
			continue
		}
		// Scheduling delay must be less than E2E (it's just the queuing portion).
		e2e := m.RequestE2Es[pid]
		if float64(delay) >= e2e {
			t.Errorf("parent %s: scheduling delay (%d) >= E2E (%.0f), delay should be queuing time only",
				pid, delay, e2e)
		}
	}
}

// TestDisaggregation_MetricProjection_CompletionTimes verifies that the
// projected completion time matches the parent's CompletionTime.
func TestDisaggregation_MetricProjection_CompletionTimes(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	for _, parent := range cs.parentRequests {
		if parent.CompletionTime == 0 || parent.DecodeInstanceID == "" {
			continue
		}
		pid := parent.ID
		ct, ok := m.RequestCompletionTimes[pid]
		if !ok {
			t.Errorf("parent %s: missing RequestCompletionTimes entry", pid)
			continue
		}
		expected := float64(parent.CompletionTime)
		if math.Abs(ct-expected) > 1e-9 {
			t.Errorf("parent %s: RequestCompletionTimes = %.0f, want %.0f",
				pid, ct, expected)
		}
	}
}

// TestDisaggregation_MetricProjection_RequestsMap verifies that the Requests
// map contains parent IDs (not sub-request IDs) after projection.
func TestDisaggregation_MetricProjection_RequestsMap(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	for _, parent := range cs.parentRequests {
		if parent.CompletionTime == 0 || parent.DecodeInstanceID == "" {
			continue
		}
		pid := parent.ID
		rm, ok := m.Requests[pid]
		if !ok {
			t.Errorf("parent %s: missing Requests entry", pid)
			continue
		}
		if rm.ID != pid {
			t.Errorf("parent %s: Requests[%s].ID = %q, want %q", pid, pid, rm.ID, pid)
		}
		wantHandledBy := string(parent.DecodeInstanceID)
		if rm.HandledBy != wantHandledBy {
			t.Errorf("parent %s: Requests[%s].HandledBy = %q, want decode instance %q",
				pid, pid, rm.HandledBy, wantHandledBy)
		}
	}
}

// TestDisaggregation_MetricProjection_DroppedParent_NoSubRequestKeys verifies
// INV-PD-6 for the dropped-parent path: when decode KV allocation fails,
// no sub-request key must remain in any per-request metric map.
// GIVEN a cluster with tight decode KV causing some parents to be dropped
// WHEN  Run() completes
// THEN  no map entry has a "_prefill" or "_decode" suffix (dropped or completed).
func TestDisaggregation_MetricProjection_DroppedParent_NoSubRequestKeys(t *testing.T) {
	// 1 decode instance with 3 blocks (48 tokens); each short request needs 2 blocks.
	// First request fills 2/3, second request tries 2 more with only 1 free → dropped.
	config := newTestDisaggDeploymentConfig(3, 2, 1)
	config.KVCacheConfig = sim.NewKVCacheConfig(3, 16, 0, 0, 0, 0)

	requests := newShortRequests(4)
	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	if cs.droppedAtDecodeKV == 0 {
		t.Skip("no decode drops in this scenario; test precondition not met")
	}

	m := cs.AggregatedMetrics()
	maps := map[string][]string{
		"RequestE2Es":             mapKeys(m.RequestE2Es),
		"RequestTTFTs":            mapKeys(m.RequestTTFTs),
		"RequestITLs":             mapKeys(m.RequestITLs),
		"RequestCompletionTimes":  mapKeys(m.RequestCompletionTimes),
		"RequestSchedulingDelays": mapKeysInt64(m.RequestSchedulingDelays),
		"Requests":                mapKeysRM(m.Requests),
	}
	for mapName, keys := range maps {
		for _, key := range keys {
			if hasSubRequestSuffix(key) {
				t.Errorf("INV-PD-6 violated: %s contains sub-request key %q (dropped parent not cleaned up)",
					mapName, key)
			}
		}
	}
}

// TestDisaggregation_MetricProjection_ITL verifies that after projection,
// ITL entries are keyed by parent ID and carry a positive value (from the
// decode sub-request, not zero noise from the prefill sub-request).
// GIVEN a PD disaggregation simulation with multi-token output requests
// WHEN  Run() completes
// THEN  no sub-request ITL keys remain, and any present parent ITL is > 0.
func TestDisaggregation_MetricProjection_ITL(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(10)

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	completedCount := 0
	for _, parent := range cs.parentRequests {
		if parent.CompletionTime == 0 || parent.DecodeInstanceID == "" {
			continue
		}
		completedCount++
		pid := parent.ID

		// Sub-request ITL keys must be absent after projection.
		if _, ok := m.RequestITLs[parent.PrefillSubReqID]; ok {
			t.Errorf("parent %s: prefill sub-req key %q in RequestITLs after projection",
				pid, parent.PrefillSubReqID)
		}
		if _, ok := m.RequestITLs[parent.DecodeSubReqID]; ok {
			t.Errorf("parent %s: decode sub-req key %q in RequestITLs after projection",
				pid, parent.DecodeSubReqID)
		}

		// When ITL is present for the parent, it must be positive (decode generates real ITL;
		// prefill ITL is zero noise that projectPDMetrics discards).
		if itl, ok := m.RequestITLs[pid]; ok && itl <= 0 {
			t.Errorf("parent %s: ITL = %.4f, expected > 0 (from decode sub-request)", pid, itl)
		}
	}
	if completedCount == 0 {
		t.Fatal("no completed parents: test inconclusive")
	}
}

// helper: extract keys from map[string]float64.
func mapKeys(m map[string]float64) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// helper: extract keys from map[string]int64.
func mapKeysInt64(m map[string]int64) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// helper: extract keys from map[string]RequestMetrics.
func mapKeysRM(m map[string]sim.RequestMetrics) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// BC-3b: detectDecodeCompletions adds PostDecodeFixedOverhead to parent.CompletionTime.
// Law: for matching completed parents across two runs (zero vs non-zero overhead),
// E2E_with_overhead - E2E_without_overhead == wantOverhead exactly.
// This is the only cluster-level test that exercises overhead > 0, directly
// verifying the line `parent.CompletionTime = c.clock + inst.PostDecodeFixedOverhead()`
// that was the bug site fixed in issue #846.
func TestDisaggregation_CompletionTime_IncludesNonZeroOverhead(t *testing.T) {
	const wantOverheadUs = int64(1000) // 1ms overhead, chosen to be clearly distinguishable

	requests := newTestRequests(3)
	// Run 1: overhead = 0 (baseline)
	cs0 := NewClusterSimulator(newTestDisaggDeploymentConfigWithOverhead(0), requests, nil)
	mustRun(t, cs0)
	m0 := cs0.AggregatedMetrics()

	// Run 2: overhead = wantOverheadUs
	cs1 := NewClusterSimulator(newTestDisaggDeploymentConfigWithOverhead(float64(wantOverheadUs)), requests, nil)
	mustRun(t, cs1)
	m1 := cs1.AggregatedMetrics()

	// Both runs must complete at least one parent for the test to be meaningful.
	completed := 0
	for _, parent := range cs0.parentRequests {
		if parent.CompletionTime == 0 || parent.DecodeInstanceID == "" {
			continue // dropped or horizon-interrupted
		}
		e2e0, ok0 := m0.RequestE2Es[parent.ID]
		e2e1, ok1 := m1.RequestE2Es[parent.ID]
		if !ok0 || !ok1 {
			t.Errorf("parent %s: missing E2E in one or both runs (ok0=%v ok1=%v)", parent.ID, ok0, ok1)
			continue
		}
		// Law: E2E with overhead = E2E without overhead + wantOverheadUs (exactly, int64 arithmetic)
		gotDiff := e2e1 - e2e0
		if gotDiff != float64(wantOverheadUs) {
			t.Errorf("parent %s: E2E diff = %.0f µs, want %d µs (overhead not stamped into CompletionTime)",
				parent.ID, gotDiff, wantOverheadUs)
		}
		completed++
	}
	if completed == 0 {
		t.Fatal("no completed parents in baseline run — test is vacuously passing, check config")
	}
}

// BC-3: parent.CompletionTime is >= all prior phase timestamps.
// Law: CompletionTime >= DecodeEnqueueTime >= TransferCompleteTime (phase causality).
// For blackbox (overhead=0): CompletionTime == cluster clock at decode completion tick.
func TestDisaggregation_CompletionTime_GeqAllPriorPhaseTimestamps(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(3)
	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	for _, parent := range cs.parentRequests {
		if parent.CompletionTime == 0 {
			continue // incomplete (horizon-interrupted)
		}
		if parent.CompletionTime < parent.DecodeEnqueueTime {
			t.Errorf("parent %s: CompletionTime (%d) < DecodeEnqueueTime (%d) — causality violated",
				parent.ID, parent.CompletionTime, parent.DecodeEnqueueTime)
		}
		if parent.CompletionTime < parent.TransferCompleteTime {
			t.Errorf("parent %s: CompletionTime (%d) < TransferCompleteTime (%d) — causality violated",
				parent.ID, parent.CompletionTime, parent.TransferCompleteTime)
		}
	}
}

// BC-4 regression: With blackbox (overhead=0), RequestE2Es[parentID] equals
// parent.CompletionTime - parent.ArrivalTime, and E2E >= TTFT (causality law).
func TestDisaggregation_E2E_IncludesOverhead_ZeroOverheadRegression(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(3)
	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	for _, parent := range cs.parentRequests {
		if parent.CompletionTime == 0 || parent.DecodeInstanceID == "" {
			continue // dropped or incomplete
		}
		e2e, ok := m.RequestE2Es[parent.ID]
		if !ok {
			t.Errorf("parent %s: no RequestE2Es entry after projectPDMetrics", parent.ID)
			continue
		}
		wantE2E := float64(parent.CompletionTime - parent.ArrivalTime)
		if e2e != wantE2E {
			t.Errorf("parent %s: RequestE2Es = %.0f, want %.0f (CompletionTime-ArrivalTime)",
				parent.ID, e2e, wantE2E)
		}
		// Law: E2E >= TTFT (first token precedes full decode completion)
		ttft, hasTTFT := m.RequestTTFTs[parent.ID]
		if hasTTFT && e2e < ttft {
			t.Errorf("parent %s: E2E (%.0f) < TTFT (%.0f) — causality violated", parent.ID, e2e, ttft)
		}
	}
}

// --- Session follow-up tests (issue #884) ---

// sessionCallbackCapture records every invocation of the onRequestDone callback.
// No mutex needed: the DES is single-threaded (see session.go).
type sessionCallbackCapture struct {
	calls []sessionCallbackCall
}

type sessionCallbackCall struct {
	req  *sim.Request
	tick int64
}

// newTestRequestsWithSession creates n test requests with SessionID and MaxOutputLen set.
func newTestRequestsWithSession(n int, sessionID string) []*sim.Request {
	reqs := newTestRequests(n)
	for _, r := range reqs {
		r.SessionID = sessionID
		r.MaxOutputLen = len(r.OutputTokens)
	}
	return reqs
}

// TestDisaggregation_SessionFollowUp_CallsOnRequestDone verifies that
// detectDecodeCompletions triggers the session callback with the original
// request (which carries SessionID), not the decode sub-request.
//
// GIVEN: a PD cluster (2P + 2D) with onRequestDone callback
// AND: requests have SessionID set
// WHEN: simulation runs to completion
// THEN: callback is invoked with a request where SessionID is preserved,
//
//	State == StateCompleted, and ProgressIndex == len(Input) + len(Output)
func TestDisaggregation_SessionFollowUp_CallsOnRequestDone(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	reqs := newTestRequestsWithSession(3, "sess_0")

	var capture sessionCallbackCapture
	callback := func(req *sim.Request, tick int64) []*sim.Request {
		capture.calls = append(capture.calls, sessionCallbackCall{req: req, tick: tick})
		return nil // no follow-ups — just capture
	}

	cs := NewClusterSimulator(config, reqs, callback)
	mustRun(t, cs)

	// Filter calls with non-empty SessionID (sub-request callbacks have empty SessionID)
	var sessionCalls []sessionCallbackCall
	for _, c := range capture.calls {
		if c.req.SessionID != "" {
			sessionCalls = append(sessionCalls, c)
		}
	}

	if len(sessionCalls) != 3 {
		t.Fatalf("expected 3 session callback calls (one per request), got %d", len(sessionCalls))
	}

	for i, sc := range sessionCalls {
		if sc.req.SessionID != "sess_0" {
			t.Errorf("call %d: SessionID = %q, want %q", i, sc.req.SessionID, "sess_0")
		}
		if sc.req.State != sim.StateCompleted {
			t.Errorf("call %d: State = %q, want %q", i, sc.req.State, sim.StateCompleted)
		}
		// ProgressIndex comes from the decode sub-request's actual final position
		// (len(Input) + len(Output) - 1), matching non-PD behavior.
		wantProgress := int64(len(sc.req.InputTokens) + len(sc.req.OutputTokens) - 1)
		if sc.req.ProgressIndex != wantProgress {
			t.Errorf("call %d: ProgressIndex = %d, want %d (len(Input)=%d + len(Output)=%d - 1)",
				i, sc.req.ProgressIndex, wantProgress, len(sc.req.InputTokens), len(sc.req.OutputTokens))
		}
		if sc.tick < sc.req.ArrivalTime {
			t.Errorf("call %d: tick (%d) < ArrivalTime (%d) — violates causality",
				i, sc.tick, sc.req.ArrivalTime)
		}
	}
}

// TestDisaggregation_SessionFollowUp_InjectsFollowUp verifies that follow-up
// requests returned by the session callback are injected into the cluster
// pipeline and complete through the PD disaggregation path.
//
// GIVEN: PD cluster with onRequestDone that returns 1 follow-up per call (single extra round)
// AND: 2 initial requests with SessionID
// WHEN: simulation runs
// THEN: follow-up requests complete through PD pipeline (more parentRequests than initial)
func TestDisaggregation_SessionFollowUp_InjectsFollowUp(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	reqs := newTestRequestsWithSession(2, "sess_inject")

	followUpCount := 0
	callback := func(req *sim.Request, tick int64) []*sim.Request {
		if req.SessionID == "" {
			return nil // sub-request callback, ignore
		}
		// Generate exactly one follow-up per original request (cap at 2 total follow-ups)
		if followUpCount >= 2 {
			return nil
		}
		followUpCount++
		return []*sim.Request{{
			ID:           fmt.Sprintf("followup_%d", followUpCount),
			ArrivalTime:  tick + 1000, // 1ms think time
			InputTokens:  make([]int, 50),
			OutputTokens: make([]int, 20),
			MaxOutputLen: 20,
			State:        sim.StateQueued,
			SessionID:    req.SessionID,
			RoundIndex:   1,
		}}
	}

	cs := NewClusterSimulator(config, reqs, callback)
	mustRun(t, cs)

	// Follow-ups should have been disaggregated too — more parentRequests than initial
	parents := cs.ParentRequests()
	if len(parents) <= 2 {
		t.Errorf("ParentRequests() = %d, want > 2 (follow-ups should have been disaggregated)",
			len(parents))
	}

	// All parent requests should have CompletionTime > 0
	for _, parent := range parents {
		if parent.CompletionTime == 0 {
			t.Errorf("parent %s: CompletionTime = 0, follow-up may not have completed", parent.ID)
		}
	}

	metrics := cs.AggregatedMetrics()
	if metrics.CompletedRequests < 4 {
		t.Errorf("CompletedRequests = %d, want >= 4 (2 initial + 2 follow-ups)", metrics.CompletedRequests)
	}
}

// TestDisaggregation_AggregateMode_Unaffected verifies that aggregate (non-PD)
// clusters still trigger session callbacks correctly — regression guard.
//
// GIVEN: non-PD cluster with onRequestDone callback
// AND: requests with SessionID
// WHEN: simulation runs
// THEN: callback fires for each completed request with SessionID preserved
func TestDisaggregation_AggregateMode_Unaffected(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 0, 0) // no PD
	config.PDDecider = ""                             // disable decider
	reqs := newTestRequestsWithSession(3, "sess_agg")

	var capture sessionCallbackCapture
	callback := func(req *sim.Request, tick int64) []*sim.Request {
		capture.calls = append(capture.calls, sessionCallbackCall{req: req, tick: tick})
		return nil
	}

	cs := NewClusterSimulator(config, reqs, callback)
	mustRun(t, cs)

	// In aggregate mode, ALL completed requests should trigger callback with SessionID
	var sessionCalls []sessionCallbackCall
	for _, c := range capture.calls {
		if c.req.SessionID == "sess_agg" {
			sessionCalls = append(sessionCalls, c)
		}
	}

	if len(sessionCalls) != 3 {
		t.Fatalf("aggregate mode: expected 3 session callback calls, got %d", len(sessionCalls))
	}
	for i, sc := range sessionCalls {
		if sc.req.State != sim.StateCompleted {
			t.Errorf("aggregate call %d: State = %q, want %q", i, sc.req.State, sim.StateCompleted)
		}
	}
}

// TestDisaggregation_PD_SessionManager_GeneratesFollowUps exercises the real
// SessionManager (not a mock callback) through the PD pipeline. This verifies
// that State and ProgressIndex are correctly threaded through OnComplete, and
// that follow-up requests complete through the full disaggregation path.
//
// GIVEN: PD cluster (2P + 2D) with real SessionManager (MaxRounds=2)
// AND: 2 initial session requests
// WHEN: simulation runs to completion
// THEN: SessionManager generates follow-ups that complete through PD,
//
//	and total completed requests == 4 (2 rounds x 2 sessions)
func TestDisaggregation_PD_SessionManager_GeneratesFollowUps(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)

	rng := rand.New(rand.NewSource(99))
	inputSampler, err := workload.NewLengthSampler(workload.DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 50},
	})
	if err != nil {
		t.Fatalf("NewLengthSampler (input): %v", err)
	}
	outputSampler, err := workload.NewLengthSampler(workload.DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 20},
	})
	if err != nil {
		t.Fatalf("NewLengthSampler (output): %v", err)
	}

	blueprints := []workload.SessionBlueprint{
		{
			SessionID:     "pd_sess_0",
			MaxRounds:     2,
			ThinkTimeUs:   1000,
			Horizon:       math.MaxInt64,
			InputSampler:  inputSampler,
			OutputSampler: outputSampler,
			RNG:           rand.New(rand.NewSource(rng.Int63())),
		},
		{
			SessionID:     "pd_sess_1",
			MaxRounds:     2,
			ThinkTimeUs:   1000,
			Horizon:       math.MaxInt64,
			InputSampler:  inputSampler,
			OutputSampler: outputSampler,
			RNG:           rand.New(rand.NewSource(rng.Int63())),
		},
	}

	sm := workload.NewSessionManager(blueprints)
	callback := sm.OnComplete

	// Initial round-0 requests (one per session)
	reqs := make([]*sim.Request, 2)
	for i := 0; i < 2; i++ {
		reqs[i] = &sim.Request{
			ID:           fmt.Sprintf("pd_sess_%d_r0", i),
			ArrivalTime:  int64(i * 1000),
			InputTokens:  make([]int, 50),
			OutputTokens: make([]int, 20),
			MaxOutputLen: 20,
			State:        sim.StateQueued,
			SessionID:    fmt.Sprintf("pd_sess_%d", i),
			RoundIndex:   0,
		}
	}

	cs := NewClusterSimulator(config, reqs, callback)
	mustRun(t, cs)

	metrics := cs.AggregatedMetrics()
	// 2 sessions x 2 rounds = 4 completed requests
	if metrics.CompletedRequests != 4 {
		t.Errorf("CompletedRequests = %d, want 4 (2 sessions x 2 rounds)", metrics.CompletedRequests)
	}

	// All parent requests should have completed through the PD pipeline
	parents := cs.ParentRequests()
	if len(parents) != 4 {
		t.Errorf("ParentRequests() = %d, want 4", len(parents))
	}
	for _, parent := range parents {
		if parent.CompletionTime == 0 {
			t.Errorf("parent %s: CompletionTime = 0", parent.ID)
		}
	}

	// INV-1 conservation
	assertINV1Conservation(t, metrics, 4, "PD SessionManager follow-ups")
}

// TestDisaggregation_PD_SessionManager_ContextAccumulation verifies that
// ProgressIndex flows correctly through the session manager's context
// accumulation logic (BC-8) when using the PD pipeline.
//
// GIVEN: PD cluster (2P + 2D) with real SessionManager (MaxRounds=2, ContextGrowth="accumulate")
// AND: 1 initial session request with input=50, output=20
// WHEN: simulation runs to completion
// THEN: round-1 follow-up has InputTokens longer than round-0 (context accumulated),
//
//	and total completed requests == 2 (1 session x 2 rounds)
func TestDisaggregation_PD_SessionManager_ContextAccumulation(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)

	rng := rand.New(rand.NewSource(77))
	inputSampler, err := workload.NewLengthSampler(workload.DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 50},
	})
	if err != nil {
		t.Fatalf("NewLengthSampler (input): %v", err)
	}
	outputSampler, err := workload.NewLengthSampler(workload.DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 20},
	})
	if err != nil {
		t.Fatalf("NewLengthSampler (output): %v", err)
	}

	blueprints := []workload.SessionBlueprint{
		{
			SessionID:     "acc_sess_0",
			MaxRounds:     2,
			ThinkTimeUs:   1000,
			Horizon:       math.MaxInt64,
			ContextGrowth: "accumulate",
			InputSampler:  inputSampler,
			OutputSampler: outputSampler,
			RNG:           rand.New(rand.NewSource(rng.Int63())),
		},
	}

	sm := workload.NewSessionManager(blueprints)
	callback := sm.OnComplete

	// Round-0 request: 50 input tokens, 20 output tokens
	round0InputLen := 50
	round0OutputLen := 20
	reqs := []*sim.Request{
		{
			ID:           "acc_sess_0_r0",
			ArrivalTime:  0,
			InputTokens:  make([]int, round0InputLen),
			OutputTokens: make([]int, round0OutputLen),
			MaxOutputLen: round0OutputLen,
			State:        sim.StateQueued,
			SessionID:    "acc_sess_0",
			RoundIndex:   0,
		},
	}

	cs := NewClusterSimulator(config, reqs, callback)
	mustRun(t, cs)

	metrics := cs.AggregatedMetrics()
	// 1 session x 2 rounds = 2 completed requests
	if metrics.CompletedRequests != 2 {
		t.Errorf("CompletedRequests = %d, want 2 (1 session x 2 rounds)", metrics.CompletedRequests)
	}

	// Find the round-1 follow-up among parent requests
	parents := cs.ParentRequests()
	if len(parents) != 2 {
		t.Fatalf("ParentRequests() = %d, want 2", len(parents))
	}

	var round1Parent *ParentRequest
	for _, p := range parents {
		if p.OriginalRequest.RoundIndex == 1 {
			round1Parent = p
			break
		}
	}
	if round1Parent == nil {
		t.Fatal("no round-1 parent request found")
	}

	// Context accumulation: round-1 input should include accumulated context from round-0.
	// Round 0: input=50, actual output = PI - len(Input) = (50+20-1) - 50 = 19 tokens
	// Accumulated context: 50 (round-0 input) + 19 (round-0 actual output) = 69
	// Round 1 new input: 50 (from constant sampler)
	// Round 1 total input: 69 (context) + 50 (new) = 119
	round1InputLen := len(round1Parent.OriginalRequest.InputTokens)
	if round1InputLen <= round0InputLen {
		t.Errorf("round-1 InputTokens length = %d, want > %d (context should have accumulated)",
			round1InputLen, round0InputLen)
	}

	// Verify the exact accumulated length: context(69) + new_input(50) = 119
	wantRound1InputLen := (round0InputLen + (round0OutputLen - 1)) + 50 // context + new_input
	if round1InputLen != wantRound1InputLen {
		t.Errorf("round-1 InputTokens length = %d, want %d (context=%d + new_input=50)",
			round1InputLen, wantRound1InputLen, round0InputLen+(round0OutputLen-1))
	}

	// All parents should have completed
	for _, p := range parents {
		if p.CompletionTime == 0 {
			t.Errorf("parent %s: CompletionTime = 0", p.ID)
		}
	}

	// INV-1 conservation
	assertINV1Conservation(t, metrics, 2, "PD SessionManager context accumulation")
}
