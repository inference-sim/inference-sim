package cluster

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// testModelHardwareConfig returns a ModelHardwareConfig that produces 512 KV bytes/token/GPU
// at TP=1: 2 layers × 2 (K+V) × 16 headDim × 4 numKVHeads × 2.0 BytesPerParam = 512.
func testModelHardwareConfig() sim.ModelHardwareConfig {
	mc := sim.ModelConfig{
		NumLayers:       2,
		NumHeads:        4,
		HiddenDim:       64,
		IntermediateDim: 128,
		BytesPerParam:   2.0,
		// NumKVHeads=0: MHA fallback, uses NumHeads=4
	}
	return sim.NewModelHardwareConfig(mc, sim.HardwareCalib{}, "test-model", "H100", 1, "blackbox", 0)
}

// newContentionConfig creates a PD deployment config with transfer contention enabled.
// Uses high bandwidth and zero base latency for predictable duration calculations.
func newContentionConfig(numInstances, prefill, decode int, bandwidthGBps float64) DeploymentConfig {
	config := newTestDisaggDeploymentConfig(numInstances, prefill, decode)
	config.PDTransferContention = true
	config.PDTransferBandwidthGBps = bandwidthGBps
	config.PDTransferBaseLatencyMs = 0 // zero base latency for clean duration math
	return config
}

// TestTransferContention_INVP22_FairShareBandwidth verifies INV-P2-2:
// effective_bandwidth = total_bandwidth / max(1, active_transfers).
// With contention enabled and concurrent transfers, each gets a fair share.
func TestTransferContention_INVP22_FairShareBandwidth(t *testing.T) {
	// Setup: 4 instances (2 prefill, 2 decode), high bandwidth so transfers complete
	// on known timelines. All requests arrive at time 0 so prefills overlap.
	config := newContentionConfig(4, 2, 2, 25.0)
	config.PDTransferBaseLatencyMs = 0

	requests := newTestRequests(4)
	// Force all arrivals to time 0 to maximize concurrent prefills → concurrent transfers
	for _, r := range requests {
		r.ArrivalTime = 0
	}

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	parents := cs.ParentRequests()
	if len(parents) == 0 {
		t.Fatal("no parent requests found — disaggregation did not activate")
	}

	// INV-PD-3 still holds (BC-P2-7): initiated == completed
	if cs.transfersInitiated != cs.transfersCompleted {
		t.Errorf("INV-PD-3 violated: initiated=%d != completed=%d",
			cs.transfersInitiated, cs.transfersCompleted)
	}

	// At least one transfer must have started (peak >= 1 required).
	// With 4 simultaneous arrivals on 2 prefill instances, multiple concurrent transfers
	// are likely; TestTransferContention_BCP26_FairShareDivision covers strict > 1 concurrency.
	if cs.PeakConcurrentTransfers() < 1 {
		t.Errorf("PeakConcurrentTransfers = %d, want >= 1", cs.PeakConcurrentTransfers())
	}
}

// TestTransferContention_BCP25_SingleTransferIdentical verifies BC-P2-5:
// with 1 concurrent transfer, duration is identical to Phase 1 (non-contention mode).
func TestTransferContention_BCP25_SingleTransferIdentical(t *testing.T) {
	bandwidthGBps := 25.0

	// Create two identical configs: one with contention, one without.
	// Use 1 request so only 1 transfer is ever in flight.
	configNoContention := newTestDisaggDeploymentConfig(4, 2, 2)
	configNoContention.PDTransferBandwidthGBps = bandwidthGBps
	configNoContention.PDTransferBaseLatencyMs = 0.05

	configContention := newTestDisaggDeploymentConfig(4, 2, 2)
	configContention.PDTransferContention = true
	configContention.PDTransferBandwidthGBps = bandwidthGBps
	configContention.PDTransferBaseLatencyMs = 0.05

	requests1 := newTestRequests(1)
	requests2 := make([]*sim.Request, len(requests1))
	for i, r := range requests1 {
		cp := *r
		cp.InputTokens = make([]int, len(r.InputTokens))
		copy(cp.InputTokens, r.InputTokens)
		cp.OutputTokens = make([]int, len(r.OutputTokens))
		copy(cp.OutputTokens, r.OutputTokens)
		requests2[i] = &cp
	}

	cs1 := NewClusterSimulator(configNoContention, requests1, nil)
	mustRun(t, cs1)
	cs2 := NewClusterSimulator(configContention, requests2, nil)
	mustRun(t, cs2)

	parents1 := cs1.ParentRequests()
	parents2 := cs2.ParentRequests()
	if len(parents1) != 1 || len(parents2) != 1 {
		t.Fatalf("expected 1 parent each, got %d and %d", len(parents1), len(parents2))
	}

	dur1 := parents1[0].TransferCompleteTime - parents1[0].TransferStartTime
	dur2 := parents2[0].TransferCompleteTime - parents2[0].TransferStartTime

	if dur1 != dur2 {
		t.Errorf("BC-P2-5 violated: non-contention duration=%d, contention duration=%d — want identical for single transfer",
			dur1, dur2)
	}
}

// TestTransferContention_BCP26_FairShareDivision verifies BC-P2-6:
// with N concurrent transfers, each gets bandwidth/N.
// Uses a synthetic approach: compute expected duration for a known payload
// at full vs shared bandwidth, and verify contention transfers take longer.
func TestTransferContention_BCP26_FairShareDivision(t *testing.T) {
	// Use a large number of simultaneous requests to force concurrent transfers.
	config := newContentionConfig(4, 2, 2, 25.0)
	config.PDTransferBaseLatencyMs = 0

	// Create many requests arriving at the same time to force overlapping transfers
	requests := newTestRequests(8)
	for _, r := range requests {
		r.ArrivalTime = 0
	}

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	parents := cs.ParentRequests()
	if len(parents) == 0 {
		t.Fatal("no parent requests found")
	}

	// With contention enabled and concurrent transfers, the peak should be > 1.
	// At least some transfers should experience bandwidth sharing.
	if cs.PeakConcurrentTransfers() <= 1 {
		t.Fatalf("PeakConcurrentTransfers = %d — expected concurrent overlap with 8 simultaneous requests; if this fails, increase workload or verify prefill scheduling",
			cs.PeakConcurrentTransfers())
	}

	// Now compare with non-contention mode.
	configNoContention := newTestDisaggDeploymentConfig(4, 2, 2)
	configNoContention.PDTransferBandwidthGBps = 25.0
	configNoContention.PDTransferBaseLatencyMs = 0

	requestsCopy := newTestRequests(8)
	for _, r := range requestsCopy {
		r.ArrivalTime = 0
	}

	csNoContention := NewClusterSimulator(configNoContention, requestsCopy, nil)
	mustRun(t, csNoContention)

	parentsNC := csNoContention.ParentRequests()

	// Compute mean transfer duration for both modes
	var sumContention, sumNoContention float64
	var countContention, countNoContention int
	for _, p := range parents {
		if p.TransferStartTime > 0 && p.TransferCompleteTime > p.TransferStartTime {
			sumContention += float64(p.TransferCompleteTime - p.TransferStartTime)
			countContention++
		}
	}
	for _, p := range parentsNC {
		if p.TransferStartTime > 0 && p.TransferCompleteTime > p.TransferStartTime {
			sumNoContention += float64(p.TransferCompleteTime - p.TransferStartTime)
			countNoContention++
		}
	}

	if countContention == 0 || countNoContention == 0 {
		t.Fatal("no completed transfers in one or both modes")
	}

	meanContention := sumContention / float64(countContention)
	meanNoContention := sumNoContention / float64(countNoContention)

	// With PeakConcurrentTransfers > 1 confirmed, at least one transfer received reduced
	// bandwidth, so the mean duration under contention must strictly exceed the
	// mean without contention. Equality would mean the contention branch was dead code.
	if meanContention <= meanNoContention {
		t.Errorf("BC-P2-6 violated: contention mean=%.1f <= non-contention mean=%.1f — bandwidth sharing should strictly increase mean duration when peak concurrent > 1",
			meanContention, meanNoContention)
	}
}

// TestTransferContention_BCP27_INVPD3_Holds verifies BC-P2-7:
// INV-PD-3 (transfer conservation) still holds with contention enabled.
func TestTransferContention_BCP27_INVPD3_Holds(t *testing.T) {
	config := newContentionConfig(4, 2, 2, 25.0)
	requests := newTestRequests(10)
	for _, r := range requests {
		r.ArrivalTime = 0
	}

	cs := NewClusterSimulator(config, requests, nil)
	// Run() returns non-nil only if contention bookkeeping was corrupted; err == nil
	// rules out that failure path but does not by itself prove INV-PD-3.
	if err := cs.Run(); err != nil {
		t.Fatalf("INV-PD-3 violated with contention: %v", err)
	}

	// Direct conservation check: INV-PD-3 requires initiated == completed when all
	// transfers finish before the simulation horizon.
	if cs.transfersInitiated != cs.transfersCompleted {
		t.Errorf("INV-PD-3: initiated=%d != completed=%d",
			cs.transfersInitiated, cs.transfersCompleted)
	}
}

// TestTransferContention_BCP28_MetricsAvailable verifies BC-P2-8:
// contention metrics are available in PDMetrics when feature is enabled.
func TestTransferContention_BCP28_MetricsAvailable(t *testing.T) {
	config := newContentionConfig(4, 2, 2, 25.0)
	requests := newTestRequests(6)
	for _, r := range requests {
		r.ArrivalTime = 0
	}

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	pd := CollectPDMetrics(
		cs.ParentRequests(),
		cs.AggregatedMetrics(),
		cs.PoolMembership(),
		cs.PerInstanceMetricsByID(),
	)
	if pd == nil {
		t.Fatal("CollectPDMetrics returned nil for disaggregated simulation with contention")
	}

	// Attach contention metrics as cmd/root.go does
	pd.PeakConcurrentTransfers = cs.PeakConcurrentTransfers()
	pd.MeanTransferQueueDepth = cs.MeanTransferQueueDepth()

	if pd.PeakConcurrentTransfers < 1 {
		t.Errorf("PeakConcurrentTransfers = %d, want >= 1", pd.PeakConcurrentTransfers)
	}
	if pd.MeanTransferQueueDepth <= 0 {
		t.Errorf("MeanTransferQueueDepth = %f, want > 0", pd.MeanTransferQueueDepth)
	}
}

// TestTransferContention_DisabledByDefault verifies backward compatibility:
// when --pd-transfer-contention is not set, contention state remains zero.
func TestTransferContention_DisabledByDefault(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	// PDTransferContention defaults to false
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	if cs.PeakConcurrentTransfers() != 0 {
		t.Errorf("PeakConcurrentTransfers = %d, want 0 when contention disabled",
			cs.PeakConcurrentTransfers())
	}
	if cs.MeanTransferQueueDepth() != 0 {
		t.Errorf("MeanTransferQueueDepth = %f, want 0 when contention disabled",
			cs.MeanTransferQueueDepth())
	}
}

// TestTransferContention_ActiveTransfersZeroAtEnd verifies that activeTransfers
// returns to zero after simulation completes (all transfers finish).
func TestTransferContention_ActiveTransfersZeroAtEnd(t *testing.T) {
	config := newContentionConfig(4, 2, 2, 25.0)
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	// After simulation completes, activeTransfers should be back to 0
	// since INV-PD-3 guarantees initiated == completed.
	if cs.activeTransfers != 0 {
		t.Errorf("activeTransfers = %d after simulation, want 0 (every start must have a matching completion)",
			cs.activeTransfers)
	}
}

// TestTransferContention_HorizonCutoff_RunReturnsNil verifies that when the simulation
// horizon terminates with KV transfers still in flight (activeTransfers != 0 at end),
// Run() emits a warning but still returns nil — not an error.
// This is the complement of TestTransferContention_CorruptionFlagCausesRunError:
//   - Corruption (activeTransfers went negative) → Run() returns error
//   - Horizon cutoff (transfers still in flight) → Run() returns nil + Warnf
//
// The pre-set activeTransfers simulates a horizon that terminated mid-transfer.
func TestTransferContention_HorizonCutoff_RunReturnsNil(t *testing.T) {
	config := newContentionConfig(4, 2, 2, 25.0)
	// Empty request set: simulation finishes immediately without any events,
	// leaving activeTransfers untouched at whatever we set before Run().
	cs := NewClusterSimulator(config, newTestRequests(0), nil)
	// Pre-set to simulate a horizon that cut off one in-flight transfer.
	cs.activeTransfers = 1

	err := cs.Run()
	if err != nil {
		t.Fatalf("Run() = %v, want nil (horizon-cutoff with activeTransfers=1 should warn, not error)", err)
	}
	// Verify the residual count was not silently reset by Run().
	if cs.activeTransfers != 1 {
		t.Errorf("activeTransfers = %d after Run(), want 1 (horizon-cutoff residual must be preserved)", cs.activeTransfers)
	}
}

// TestTransferContention_INVP22_EffectiveBandwidthFormula verifies the fair-share formula
// by running the actual ClusterSimulator with controlled parameters and measuring the
// observed transfer duration produced by KVTransferStartedEvent.Execute().
//
// Uses 160 input tokens → 10 KV blocks (blockSize=16), bandwidth=10 GB/s, zero base latency.
// Expected N=1 duration: 10 blocks × 16 tok/block × 512 B/tok = 81920 B; 10 GB/s = 10000 B/μs
// → ceil(81920 / 10000) = ceil(8.192) = 9 μs.
// This verifies production code, not a reimplementation of the formula.
// N>1 behavioral coverage is provided by TestTransferContention_BCP26_FairShareDivision.
func TestTransferContention_INVP22_EffectiveBandwidthFormula(t *testing.T) {
	// 160 tokens → ceil(160/16) = 10 KV blocks
	inputTokens := make([]int, 160)
	for i := range inputTokens {
		inputTokens[i] = i + 1
	}
	outputTokens := make([]int, 10)
	for i := range outputTokens {
		outputTokens[i] = i + 1
	}
	req := &sim.Request{
		ID:          "request_0",
		ArrivalTime: 0,
		InputTokens: inputTokens,
		OutputTokens: outputTokens,
		State:       sim.StateQueued,
	}

	config := newContentionConfig(4, 2, 2, 10.0) // 10 GB/s, zero base latency
	cs := NewClusterSimulator(config, []*sim.Request{req}, nil)
	mustRun(t, cs)

	parents := cs.ParentRequests()
	if len(parents) != 1 {
		t.Fatalf("expected 1 parent request, got %d", len(parents))
	}
	if parents[0].TransferStartTime == 0 {
		t.Fatal("transfer never started — disaggregation did not fire")
	}

	dur := parents[0].TransferCompleteTime - parents[0].TransferStartTime
	// 10 blocks × 16 tok/block × 512 B/tok = 81920 B; 10 GB/s = 10000 B/μs; base=0
	// duration = ceil(81920 / 10000) = ceil(8.192) = 9 μs
	const wantDur = int64(9)
	if dur != wantDur {
		t.Errorf("N=1 transfer duration = %d μs, want %d μs (10 blocks × 16 tok × 512 B / 10 GB/s)",
			dur, wantDur)
	}
}

// TestTransferContention_MeanQueueDepthCalculation verifies the mean calculation
// is correct for known inputs.
//
// NOTE: This test constructs a ClusterSimulator directly with only the two
// accumulator fields set (all others zero). This is an in-package arithmetic
// test for MeanTransferQueueDepth() and is intentional — standard Go practice
// allows same-package test access to unexported fields. The other fields are
// not relevant to the method under test.
func TestTransferContention_MeanQueueDepthCalculation(t *testing.T) {
	cs := &ClusterSimulator{
		transferDepthSum:   10, // e.g., depths were 1+2+3+4
		transferStartCount: 4,
	}
	got := cs.MeanTransferQueueDepth()
	want := 2.5
	if got != want {
		t.Errorf("MeanTransferQueueDepth = %f, want %f", got, want)
	}
}

// TestTransferContention_MeanQueueDepthZeroTransfers verifies zero-division safety
// when no transfers have occurred. Uses a production simulation path with 0 requests
// so no transfers are ever started.
func TestTransferContention_MeanQueueDepthZeroTransfers(t *testing.T) {
	config := newContentionConfig(4, 2, 2, 25.0)
	cs := NewClusterSimulator(config, []*sim.Request{}, nil)
	mustRun(t, cs)
	got := cs.MeanTransferQueueDepth()
	if got != 0 {
		t.Errorf("MeanTransferQueueDepth = %f, want 0 for zero transfers", got)
	}
}

// TestTransferContention_INVP22_N2FormulaExact verifies the fair-share formula
// for the N=2 case by calling KVTransferStartedEvent.Execute() directly with
// activeTransfers pre-set to 1 (so the increment makes it 2).
//
// Parameters: 10 KV blocks, bandwidth=10 GB/s, zero base latency.
// N=2: effectiveBW = 10/2 = 5 GB/s = 5000 B/µs
// Expected duration: ceil(10 × 16 × 512 / 5000) = ceil(81920/5000) = ceil(16.384) = 17 µs.
//
// Companion to TestTransferContention_INVP22_EffectiveBandwidthFormula (N=1 case, 9 µs).
// Together they confirm the divisor is active_transfers at the moment of Execute(), not
// some fixed or post-decrement value — covering the ordering invariant from the comment
// at the top of KVTransferStartedEvent.Execute().
func TestTransferContention_INVP22_N2FormulaExact(t *testing.T) {
	cs := &ClusterSimulator{
		config: DeploymentConfig{
			PDTransferContention:    true,
			PDTransferBandwidthGBps: 10.0,
			PDTransferBaseLatencyMs: 0,
			SimConfig: sim.SimConfig{
				KVCacheConfig: sim.KVCacheConfig{
					BlockSizeTokens: 16,
				},
				ModelHardwareConfig: testModelHardwareConfig(),
			},
		},
		activeTransfers: 1, // pre-existing transfer; Execute() increments to 2
		clusterEvents:   make(ClusterEventQueue, 0),
	}
	parentReq := &ParentRequest{
		ID:          "test-parent",
		NumKVBlocks: 10,
	}
	event := &KVTransferStartedEvent{time: 0, parentReq: parentReq}
	event.Execute(cs)

	if len(cs.clusterEvents) != 1 {
		t.Fatalf("expected 1 scheduled completion event, got %d", len(cs.clusterEvents))
	}
	completedAt := cs.clusterEvents[0].event.Timestamp()
	duration := completedAt - event.time

	// N=2: 10 blocks × 16 tok/block × 512 B/tok = 81920 B; BW = 10/2 = 5 GB/s = 5000 B/µs
	// duration = ceil(81920 / 5000) = ceil(16.384) = 17 µs
	const wantDur = int64(17)
	if duration != wantDur {
		t.Errorf("N=2 transfer duration = %d µs, want %d µs (10 blocks × 16 tok × 512 B / 5 GB/s with N=2 divisor)",
			duration, wantDur)
	}
	// Confirm activeTransfers is now 2 (both the pre-existing and this one)
	if cs.activeTransfers != 2 {
		t.Errorf("activeTransfers = %d after N=2 start, want 2", cs.activeTransfers)
	}
}

// TestTransferContention_PrefillOverridesTP_AffectsTransferDuration verifies that
// KVTransferStartedEvent.Execute uses the prefill pool's TP (PrefillOverrides.TP)
// rather than the global TP when computing KV bytes per token for transfer sizing.
func TestTransferContention_PrefillOverridesTP_AffectsTransferDuration(t *testing.T) {
	// testModelHardwareConfig uses TP=1 → 512 bytes/token/GPU.
	// With PrefillOverrides.TP=4, KVBytesPerToken should produce 128 bytes/token/GPU.
	prefillTP := 4
	mhc := testModelHardwareConfig() // global TP=1
	cs := &ClusterSimulator{
		config: DeploymentConfig{
			PDTransferContention:    true,
			PDTransferBandwidthGBps: 10.0,
			PDTransferBaseLatencyMs: 0,
			PrefillOverrides:        PoolOverrides{TP: &prefillTP},
			SimConfig: sim.SimConfig{
				KVCacheConfig: sim.KVCacheConfig{
					BlockSizeTokens: 16,
				},
				ModelHardwareConfig: mhc,
			},
		},
		activeTransfers: 0,
		clusterEvents:   make(ClusterEventQueue, 0),
	}
	parentReq := &ParentRequest{ID: "test-tp-override", NumKVBlocks: 10}
	event := &KVTransferStartedEvent{time: 0, parentReq: parentReq}
	event.Execute(cs)

	if len(cs.clusterEvents) != 1 {
		t.Fatalf("expected 1 scheduled completion event, got %d", len(cs.clusterEvents))
	}
	duration := cs.clusterEvents[0].event.Timestamp() - event.time

	// With prefill TP=4: 10 blocks × 16 tok/block × 128 B/tok = 20480 B
	// BW = 10 GB/s = 10000 B/µs; duration = ceil(20480/10000) = 3 µs
	const wantDur = int64(3)
	if duration != wantDur {
		t.Errorf("prefill TP=4 transfer duration = %d µs, want %d µs", duration, wantDur)
	}

	// Verify this differs from global TP=1: 10 × 16 × 512 = 81920 B → ceil(81920/10000) = 9 µs
	// If the override was ignored, duration would be 9 instead of 3.
	if duration == 9 {
		t.Errorf("transfer duration matches global TP=1 (9 µs) — PrefillOverrides.TP was ignored")
	}
}

// TestTransferContention_CorruptionFlagCausesRunError verifies end-to-end that when
// contentionBookkeepingCorrupted is true after the event loop, Run() returns a non-nil
// error rather than delivering silently invalid contention metrics.
//
// NOTE: This test sets the corruption flag directly via in-package access after
// constructing a valid ClusterSimulator. This is intentional — the negative guard
// that sets the flag in production is tested separately in
// TestTransferContention_NegativeGuard_SetsCorruptionFlag.
func TestTransferContention_CorruptionFlagCausesRunError(t *testing.T) {
	config := newContentionConfig(4, 2, 2, 25.0)
	// Use 1 request so the simulation completes quickly and INV-PD-3 holds.
	requests := newTestRequests(1)

	cs := NewClusterSimulator(config, requests, nil)
	// Inject corruption flag before Run(). The event loop will execute normally,
	// but the post-loop check should detect the corruption and return an error.
	cs.contentionBookkeepingCorrupted = true

	err := cs.Run()
	if err == nil {
		t.Fatal("Run() returned nil error when contentionBookkeepingCorrupted=true, want non-nil")
	}
	if !strings.Contains(err.Error(), "contention bookkeeping corrupted") {
		t.Errorf("Run() error = %q, want substring %q", err.Error(), "contention bookkeeping corrupted")
	}
}

// TestTransferContention_DurationFloor_ZeroBlocks verifies that when NumKVBlocks is 0,
// the minimum duration floor (1 µs) is applied rather than scheduling a 0-duration event.
//
// Parameters: 0 KV blocks, bandwidth=10 GB/s, zero base latency.
// transferBytes = 0 × 16 × 512 = 0; duration = ceil(0/10000) = 0 → clamped to 1 µs.
func TestTransferContention_DurationFloor_ZeroBlocks(t *testing.T) {
	cs := &ClusterSimulator{
		config: DeploymentConfig{
			PDTransferContention:    true,
			PDTransferBandwidthGBps: 10.0,
			PDTransferBaseLatencyMs: 0,
			SimConfig: sim.SimConfig{
				KVCacheConfig: sim.KVCacheConfig{
					BlockSizeTokens: 16,
				},
				ModelHardwareConfig: testModelHardwareConfig(),
			},
		},
		activeTransfers: 0,
		clusterEvents:   make(ClusterEventQueue, 0),
	}
	parentReq := &ParentRequest{
		ID:          "test-parent",
		NumKVBlocks: 0, // zero blocks → zero transfer bytes
	}
	event := &KVTransferStartedEvent{time: 100, parentReq: parentReq}
	event.Execute(cs)

	if len(cs.clusterEvents) != 1 {
		t.Fatalf("expected 1 scheduled completion event, got %d", len(cs.clusterEvents))
	}
	completedAt := cs.clusterEvents[0].event.Timestamp()
	duration := completedAt - event.time

	// 0 blocks → 0 bytes → ceil(0) = 0 → clamped to 1 µs minimum
	const wantDur = int64(1)
	if duration != wantDur {
		t.Errorf("zero-block transfer duration = %d µs, want %d µs (minimum floor)", duration, wantDur)
	}
}

// TestTransferContention_F1_RequiresPDEnabled verifies that PDTransferContention
// panics at construction time when PD disaggregation is not active (constructor
// prerequisite check — cross-field guard, not a numeric range rule).
func TestTransferContention_F1_RequiresPDEnabled(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic when PDTransferContention=true without PD, got nil")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("expected string panic, got %T: %v", r, r)
		}
		if !strings.Contains(msg, "PDTransferContention requires PD disaggregation") {
			t.Errorf("panic message = %q, want substring about PD disaggregation required", msg)
		}
	}()

	// Use standard test config with NO prefill/decode instances.
	config := newTestDeploymentConfig(2)
	config.PDTransferContention = true
	// This must panic because PD is not active.
	NewClusterSimulator(config, []*sim.Request{}, nil)
}

// TestTransferContention_NegativeGuard_SetsCorruptionFlag verifies that when
// KVTransferCompletedEvent.Execute() would decrement activeTransfers below zero,
// the negative guard fires, resets to 0, and marks contentionBookkeepingCorrupted=true
// so Run() can return an error rather than delivering invalid metrics.
//
// NOTE: This test manipulates ClusterSimulator state directly using in-package access
// to construct the failure scenario (activeTransfers=0 at the point of decrement).
// Standard Go practice allows same-package test access to unexported fields.
func TestTransferContention_NegativeGuard_SetsCorruptionFlag(t *testing.T) {
	cs := &ClusterSimulator{
		config: DeploymentConfig{
			PDTransferContention: true,
		},
		activeTransfers: 0, // decrement in Execute() will go to -1 → triggers guard
		clusterEvents:   make(ClusterEventQueue, 0),
	}
	parentReq := &ParentRequest{
		ID: "test-parent",
		OriginalRequest: &sim.Request{
			ID:           "test-req",
			InputTokens:  []int{1, 2, 3},
			OutputTokens: []int{1},
		},
	}
	event := &KVTransferCompletedEvent{time: 100, parentReq: parentReq}
	event.Execute(cs)

	// THEN: corruption flag is set
	if !cs.contentionBookkeepingCorrupted {
		t.Error("contentionBookkeepingCorrupted = false after negative-guard correction, want true")
	}
	// THEN: activeTransfers is reset to 0 (not left at -1)
	if cs.activeTransfers != 0 {
		t.Errorf("activeTransfers = %d after guard reset, want 0", cs.activeTransfers)
	}
}

// TestTransferContention_INVP22_DivisorLaw verifies the invariant companion for the
// golden formula tests: duration(N) / duration(1) ≈ N for identical payloads.
// This is the R7 companion for TestTransferContention_INVP22_EffectiveBandwidthFormula
// and TestTransferContention_INVP22_N2FormulaExact — it tests the divisor *property*
// (monotonic scaling with active transfers) rather than exact microsecond values.
//
// The law: for a fixed payload and bandwidth, fair-share contention causes
// transfer duration to scale linearly with the number of concurrent transfers.
// Specifically: duration(N) = ceil(bytes / (bw/N)) ≈ N × duration(1), with
// ceiling arithmetic introducing at most ±1 μs deviation per step.
func TestTransferContention_INVP22_DivisorLaw(t *testing.T) {
	makeCS := func(preExisting int) *ClusterSimulator {
		return &ClusterSimulator{
			config: DeploymentConfig{
				PDTransferContention:    true,
				PDTransferBandwidthGBps: 10.0,
				PDTransferBaseLatencyMs: 0,
				SimConfig: sim.SimConfig{
					KVCacheConfig: sim.KVCacheConfig{
						BlockSizeTokens: 16,
					},
					ModelHardwareConfig: testModelHardwareConfig(),
				},
			},
			activeTransfers: preExisting,
			clusterEvents:   make(ClusterEventQueue, 0),
		}
	}

	getDuration := func(preExisting int) int64 {
		cs := makeCS(preExisting)
		parentReq := &ParentRequest{ID: "test", NumKVBlocks: 10}
		event := &KVTransferStartedEvent{time: 0, parentReq: parentReq}
		event.Execute(cs)
		if len(cs.clusterEvents) != 1 {
			t.Fatalf("N=%d: expected 1 completion event, got %d", preExisting+1, len(cs.clusterEvents))
		}
		return cs.clusterEvents[0].event.Timestamp()
	}

	// Table of N values (active transfers including the one being started).
	// Pre-existing = N-1 because Execute() increments before calculating.
	cases := []struct {
		n int // total active transfers
	}{
		{1}, {2}, {3}, {4}, {5}, {8}, {10},
	}

	dur1 := getDuration(0) // N=1 baseline

	for _, tc := range cases {
		durN := getDuration(tc.n - 1)

		// Law: duration(N) / duration(1) should be approximately N.
		// Ceiling arithmetic can introduce deviation, so allow ±1 μs per N step.
		ratio := float64(durN) / float64(dur1)
		expectedRatio := float64(tc.n)
		tolerance := float64(tc.n) // ±1 μs per active transfer from ceiling

		if ratio < expectedRatio-tolerance/float64(dur1) || ratio > expectedRatio+tolerance/float64(dur1) {
			t.Errorf("N=%d: duration=%d, dur1=%d, ratio=%.3f, want ≈%.1f (divisor law violated)",
				tc.n, durN, dur1, ratio, expectedRatio)
		}

		// Monotonicity: duration must increase with N.
		if tc.n > 1 {
			durPrev := getDuration(tc.n - 2)
			if durN < durPrev {
				t.Errorf("N=%d: duration=%d < N=%d duration=%d (monotonicity violated)",
					tc.n, durN, tc.n-1, durPrev)
			}
		}
	}
}

// TestTransferContention_DurationFloor_ZeroBlocks_Invariant is the R7 companion
// for TestTransferContention_DurationFloor_ZeroBlocks. Instead of testing the exact
// 1 μs value, it verifies the floor *property*: duration is always >= 1 μs
// regardless of payload size, including zero.
func TestTransferContention_DurationFloor_ZeroBlocks_Invariant(t *testing.T) {
	blockCounts := []int64{0, 1, 5, 100}

	for _, blocks := range blockCounts {
		cs := &ClusterSimulator{
			config: DeploymentConfig{
				PDTransferContention:    true,
				PDTransferBandwidthGBps: 10.0,
				PDTransferBaseLatencyMs: 0,
				SimConfig: sim.SimConfig{
					KVCacheConfig: sim.KVCacheConfig{
						BlockSizeTokens: 16,
					},
					ModelHardwareConfig: testModelHardwareConfig(),
				},
			},
			activeTransfers: 0,
			clusterEvents:   make(ClusterEventQueue, 0),
		}
		parentReq := &ParentRequest{ID: "test", NumKVBlocks: blocks}
		event := &KVTransferStartedEvent{time: 100, parentReq: parentReq}
		event.Execute(cs)

		if len(cs.clusterEvents) != 1 {
			t.Fatalf("blocks=%d: expected 1 completion event, got %d", blocks, len(cs.clusterEvents))
		}
		duration := cs.clusterEvents[0].event.Timestamp() - event.time

		// Floor invariant: duration >= 1 μs always.
		if duration < 1 {
			t.Errorf("blocks=%d: duration=%d μs, want >= 1 μs (floor invariant violated)", blocks, duration)
		}

		// Monotonicity: more blocks → longer (or equal) duration.
		if blocks > 0 {
			csPrev := &ClusterSimulator{
				config:        cs.config,
				clusterEvents: make(ClusterEventQueue, 0),
			}
			prevReq := &ParentRequest{ID: "test-prev", NumKVBlocks: blocks - 1}
			prevEvent := &KVTransferStartedEvent{time: 100, parentReq: prevReq}
			prevEvent.Execute(csPrev)
			prevDur := csPrev.clusterEvents[0].event.Timestamp() - prevEvent.time
			if duration < prevDur {
				t.Errorf("blocks=%d: duration=%d < blocks=%d duration=%d (monotonicity violated)",
					blocks, duration, blocks-1, prevDur)
			}
		}
	}
}

// TestTransferContention_ZeroBandwidth_FallsBackToBaseLatency verifies that when
// PDTransferBandwidthGBps is 0 and contention is enabled, the duration falls back to
// the base latency floor (i.e., the contention fair-share divisor does not produce NaN
// or divide-by-zero when bandwidth is zero).
//
// With bandwidth=0, bandwidthBytesPerUs is 0 before the fair-share check. Since only
// one transfer is in flight (activeTransfers == 1 after increment), the fair-share branch
// (activeTransfers > 1) is skipped. The duration formula therefore uses only baseLatUs,
// rounded up and subject to the 1 µs floor.
// This tests the boundary where bandwidthBytesPerUs <= 0 even with contention active.
func TestTransferContention_ZeroBandwidth_FallsBackToBaseLatency(t *testing.T) {
	// 10 ms base latency = 10000 µs; bandwidth = 0 (degenerate config)
	cs := &ClusterSimulator{
		config: DeploymentConfig{
			PDTransferContention:    true,
			PDTransferBandwidthGBps: 0, // zero bandwidth — triggers else branch
			PDTransferBaseLatencyMs: 10.0,
			SimConfig: sim.SimConfig{
				KVCacheConfig: sim.KVCacheConfig{
					BlockSizeTokens: 16,
				},
				ModelHardwareConfig: testModelHardwareConfig(),
			},
		},
		activeTransfers: 0,
		clusterEvents:   make(ClusterEventQueue, 0),
	}
	parentReq := &ParentRequest{ID: "test-zero-bw", NumKVBlocks: 100}
	event := &KVTransferStartedEvent{time: 0, parentReq: parentReq}
	event.Execute(cs)

	if len(cs.clusterEvents) != 1 {
		t.Fatalf("expected 1 scheduled completion event, got %d", len(cs.clusterEvents))
	}
	duration := cs.clusterEvents[0].event.Timestamp() - event.time

	// With zero bandwidth: duration = ceil(10.0 * 1000) = 10000 µs
	const wantDur = int64(10000)
	if duration != wantDur {
		t.Errorf("zero-bandwidth contention duration = %d µs, want %d µs (base latency only)", duration, wantDur)
	}

	// Invariant: contention tracking still incremented (even with zero bandwidth)
	if cs.activeTransfers != 1 {
		t.Errorf("activeTransfers = %d after zero-bw start, want 1 (contention tracking must work regardless of bandwidth)", cs.activeTransfers)
	}
	if cs.peakConcurrentTransfers != 1 {
		t.Errorf("peakConcurrentTransfers = %d, want 1", cs.peakConcurrentTransfers)
	}
}

// TestTransferContention_ZeroBandwidth_PayloadIndependence verifies that when bandwidth is zero,
// duration is independent of NumKVBlocks (block count does not affect the result).
// This is the invariant companion to TestTransferContention_ZeroBandwidth_FallsBackToBaseLatency:
// the zero-bandwidth formula is duration = ceil(baseLatMs*1000) regardless of payload size.
func TestTransferContention_ZeroBandwidth_PayloadIndependence(t *testing.T) {
	const baseLatMs = 5.0
	const wantDur = int64(5000) // ceil(5.0 * 1000) µs

	blockCounts := []int{0, 1, 100, 10_000}
	for _, blocks := range blockCounts {
		cs := &ClusterSimulator{
			config: DeploymentConfig{
				PDTransferContention:    true,
				PDTransferBandwidthGBps: 0,
				PDTransferBaseLatencyMs: baseLatMs,

				SimConfig: sim.SimConfig{
					KVCacheConfig: sim.KVCacheConfig{
						BlockSizeTokens: 16,
					},
					ModelHardwareConfig: testModelHardwareConfig(),
				},
			},
			activeTransfers: 0,
			clusterEvents:   make(ClusterEventQueue, 0),
		}
		parentReq := &ParentRequest{ID: "test-payload-indep", NumKVBlocks: int64(blocks)}
		event := &KVTransferStartedEvent{time: 0, parentReq: parentReq}
		event.Execute(cs)

		if len(cs.clusterEvents) != 1 {
			t.Fatalf("blocks=%d: expected 1 scheduled completion event, got %d", blocks, len(cs.clusterEvents))
		}
		duration := cs.clusterEvents[0].event.Timestamp() - event.time
		if duration != wantDur {
			t.Errorf("blocks=%d: duration=%d µs, want %d µs (zero bandwidth — duration must be independent of block count)",
				blocks, duration, wantDur)
		}
	}
}

// TestTransferContention_INV6_Determinism verifies INV-6 (determinism) for the contention
// subsystem: given the same seed, PeakConcurrentTransfers and MeanTransferQueueDepth must be
// identical across two independent runs of the same simulation.
//
// This catches event-ordering regressions (e.g., tie-break instability in the cluster event
// heap for same-timestamp KVTransferStartedEvents) that would cause the contention accumulators
// to diverge between runs while producing the same completed-request metrics.
func TestTransferContention_INV6_Determinism(t *testing.T) {
	run := func() (peak int, mean float64, initiated, completed int) {
		config := newContentionConfig(4, 2, 2, 25.0)
		// 8 requests at ArrivalTime=0 maximizes the chance of concurrent transfers.
		requests := newTestRequests(8)
		for _, r := range requests {
			r.ArrivalTime = 0
		}
		cs := NewClusterSimulator(config, requests, nil)
		mustRun(t, cs)
		return cs.PeakConcurrentTransfers(), cs.MeanTransferQueueDepth(),
			cs.transfersInitiated, cs.transfersCompleted
	}

	peak1, mean1, init1, done1 := run()
	peak2, mean2, init2, done2 := run()

	if peak1 != peak2 {
		t.Errorf("INV-6: PeakConcurrentTransfers = %d vs %d across identical runs", peak1, peak2)
	}
	if mean1 != mean2 {
		t.Errorf("INV-6: MeanTransferQueueDepth = %v vs %v across identical runs", mean1, mean2)
	}
	if init1 != init2 || done1 != done2 {
		t.Errorf("INV-6: transfer counts diverged: initiated %d vs %d, completed %d vs %d",
			init1, init2, done1, done2)
	}
}
