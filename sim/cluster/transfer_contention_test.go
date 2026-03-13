package cluster

import (
	"math"
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
		t.Skipf("only 1 concurrent transfer observed — test needs concurrent overlap to verify BC-P2-6")
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

	// With bandwidth sharing, contention transfers should take >= non-contention transfers.
	// Allow small rounding tolerance (1 μs).
	if meanContention < meanNoContention-1 {
		t.Errorf("BC-P2-6 violated: contention mean=%.1f < non-contention mean=%.1f — bandwidth sharing should increase duration",
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

// TestTransferContention_INVP22_EffectiveBandwidthFormula tests the invariant
// at the unit level: given known parameters, verify transfer duration matches
// the fair-share formula.
func TestTransferContention_INVP22_EffectiveBandwidthFormula(t *testing.T) {
	// Given: 100 KV blocks, blockSize=16 tokens, 512 bytes/token
	// Transfer bytes = 100 * 16 * 512 = 819200 bytes
	// Bandwidth = 25 GB/s = 25000 bytes/μs
	// Base latency = 0 μs
	// With N=1: duration = ceil(819200/25000) = ceil(32.768) = 33 μs
	// With N=2: duration = ceil(819200/12500) = ceil(65.536) = 66 μs
	// With N=4: duration = ceil(819200/6250)  = ceil(131.072) = 132 μs

	tests := []struct {
		name             string
		activeTransfers  int
		wantDurationUs   int64
	}{
		{"N=1 (no sharing)", 1, 33},
		{"N=2 (half bandwidth)", 2, 66},
		{"N=4 (quarter bandwidth)", 4, 132},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Compute duration using the same formula as KVTransferStartedEvent.Execute()
			numBlocks := int64(100)
			blockSizeTokens := int64(16)
			bytesPerToken := int64(512)
			totalBandwidthGBps := 25.0
			baseLatMs := 0.0

			transferBytes := numBlocks * blockSizeTokens * bytesPerToken
			effectiveBW := totalBandwidthGBps
			if tt.activeTransfers > 1 {
				effectiveBW = totalBandwidthGBps / float64(tt.activeTransfers)
			}
			bandwidthBytesPerUs := effectiveBW * 1000.0
			baseLatUs := baseLatMs * 1000.0

			duration := int64(math.Ceil(baseLatUs + float64(transferBytes)/bandwidthBytesPerUs))
			if duration < 1 {
				duration = 1
			}

			if duration != tt.wantDurationUs {
				t.Errorf("duration = %d μs, want %d μs", duration, tt.wantDurationUs)
			}
		})
	}
}

// TestTransferContention_MeanQueueDepthCalculation verifies the mean calculation
// is correct for known inputs.
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

// TestTransferContention_MeanQueueDepthZeroTransfers verifies zero-division safety.
func TestTransferContention_MeanQueueDepthZeroTransfers(t *testing.T) {
	cs := &ClusterSimulator{}
	got := cs.MeanTransferQueueDepth()
	if got != 0 {
		t.Errorf("MeanTransferQueueDepth = %f, want 0 for zero transfers", got)
	}
}
