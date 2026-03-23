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

// --- Integration and invariant tests ---

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

// TestPDDisagg_OneOutputToken_NoPanic is a regression test for the off-by-one
// bug in processCompletions that caused an index-out-of-range panic when a
// disaggregated request had exactly 1 output token.
//
// Root cause: completion check used == instead of >=. In PD mode, the decode
// sub-request enters with ProgressIndex == inputLen; after one decode step
// ProgressIndex becomes inputLen+1, which failed the == check and allowed a
// second decode step to call AllocateKVBlocks with out-of-bounds index.
func TestPDDisagg_OneOutputToken_NoPanic(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)

	input := make([]int, 20)
	for i := range input {
		input[i] = i + 1
	}
	output := []int{42} // exactly 1 output token

	requests := []*sim.Request{
		{
			ID:           "req-1output",
			ArrivalTime:  0,
			InputTokens:  input,
			OutputTokens: output,
			State:        sim.StateQueued,
		},
	}

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs) // must not panic

	metrics := cs.AggregatedMetrics()
	// The request must complete — not hang or get dropped.
	if metrics.CompletedRequests == 0 {
		t.Errorf("expected completed requests > 0, got %d (request did not complete)", metrics.CompletedRequests)
	}
}
