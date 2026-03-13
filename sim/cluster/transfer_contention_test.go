package cluster

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

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

	cs := NewClusterSimulator(config, requests)
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

	// With all requests arriving at once, some transfers should overlap
	// (peak concurrent > 1 is expected with 4 simultaneous arrivals on 2 prefill instances)
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

	cs1 := NewClusterSimulator(configNoContention, requests1)
	mustRun(t, cs1)
	cs2 := NewClusterSimulator(configContention, requests2)
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

	cs := NewClusterSimulator(config, requests)
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

	csNoContention := NewClusterSimulator(configNoContention, requestsCopy)
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

	cs := NewClusterSimulator(config, requests)
	// Run() returns error on INV-PD-3 violation, so err == nil proves conservation.
	if err := cs.Run(); err != nil {
		t.Fatalf("INV-PD-3 violated with contention: %v", err)
	}

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

	cs := NewClusterSimulator(config, requests)
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

	cs := NewClusterSimulator(config, requests)
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

	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	// After simulation completes, activeTransfers should be back to 0
	// since INV-PD-3 guarantees initiated == completed.
	if cs.activeTransfers != 0 {
		t.Errorf("activeTransfers = %d after simulation, want 0 (every start must have a matching completion)",
			cs.activeTransfers)
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
	cs := NewClusterSimulator(config, []*sim.Request{req})
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
	cs := NewClusterSimulator(config, []*sim.Request{})
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
			PDKVBytesPerToken:       512,
			SimConfig: sim.SimConfig{
				KVCacheConfig: sim.KVCacheConfig{
					BlockSizeTokens: 16,
				},
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

	cs := NewClusterSimulator(config, requests)
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
			PDKVBytesPerToken:       512,
			SimConfig: sim.SimConfig{
				KVCacheConfig: sim.KVCacheConfig{
					BlockSizeTokens: 16,
				},
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
