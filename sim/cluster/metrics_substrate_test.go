// sim/cluster/metrics_substrate_test.go
//
// Cluster-level metrics substrate verification.
// Behavioral contracts verified at the cluster aggregation boundary:
//   BC-MS-1c: E2E identity holds in aggregated cluster metrics
//   BC-MS-4c: sum(AllITLs) == sum(E2E-TTFT) in aggregated cluster metrics
//   BC-MS-14c: E2E >= TTFT for every request in aggregated cluster metrics
//   BC-MS-7c: CacheHitRate ∈ [0, 1] in aggregated cluster metrics
//   BC-MS-10c: PeakKVBlocksUsed bounded by total capacity per instance
package cluster

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// msClusterConfig returns a DeploymentConfig for metrics substrate cluster tests.
func msClusterConfig(numInstances int) DeploymentConfig {
	return DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 100000, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{5000, 10, 3}, []float64{1000, 2, 500}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "test-gpu", 1, false),
			PolicyConfig:        sim.NewPolicyConfig("constant", "fcfs"),
			WorkloadConfig:      sim.NewWorkloadConfig(nil, ""),
		},
		NumInstances:    numInstances,
		RoutingPolicy:   "round-robin",
		AdmissionPolicy: "always-admit",
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// BC-MS-1c + BC-MS-14c: E2E Identity and Causality in Cluster Aggregation
// ═══════════════════════════════════════════════════════════════════════════════

func TestClusterMetrics_E2E_Identity_AcrossInstances(t *testing.T) {
	cfg := msClusterConfig(4)
	workload := sim.NewGuideLLMConfig(
		0.0001, // rate (ticks): slow enough for all to complete
		20,     // numRequests
		0,      // prefixTokens
		64,     // promptTokens
		0, 64, 64, // stddev, min, max
		5,      // outputTokens
		0, 5, 5, // stddev, min, max
	)
	cs := NewClusterSimulator(cfg, workload, "")
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}

	agg := cs.AggregatedMetrics()
	if agg.CompletedRequests == 0 {
		t.Fatal("no completed requests")
	}

	// Verify requests were distributed across multiple instances
	instancesWithRequests := 0
	for _, inst := range cs.Instances() {
		if inst.Metrics().CompletedRequests > 0 {
			instancesWithRequests++
		}
	}
	if instancesWithRequests < 2 {
		t.Errorf("expected requests on >= 2 instances, got %d", instancesWithRequests)
	}

	// BC-MS-14c: E2E >= TTFT for all requests
	for id, e2e := range agg.RequestE2Es {
		ttft, ok := agg.RequestTTFTs[id]
		if !ok {
			t.Errorf("request %s has E2E but no TTFT", id)
			continue
		}
		if e2e < ttft {
			t.Errorf("BC-MS-14c violated for %s: E2E (%.1f) < TTFT (%.1f)", id, e2e, ttft)
		}
	}

	// BC-MS-2c: Mean ITL consistency for all requests
	for id, e2e := range agg.RequestE2Es {
		ttft := agg.RequestTTFTs[id]
		meanITL := agg.RequestITLs[id]
		rm := agg.Requests[id]
		denom := float64(max(rm.NumDecodeTokens-1, 1))
		totalFromMean := meanITL * denom
		totalFromE2E := e2e - ttft
		if math.Abs(totalFromMean-totalFromE2E) > 1.0 {
			t.Errorf("BC-MS-2c violated for %s: meanITL*denom (%.1f) != E2E-TTFT (%.1f)",
				id, totalFromMean, totalFromE2E)
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// BC-MS-4c: AllITLs Aggregate Consistency Across Cluster
// ═══════════════════════════════════════════════════════════════════════════════

func TestClusterMetrics_AllITLs_Sum_Consistency(t *testing.T) {
	cfg := msClusterConfig(2)
	workload := sim.NewGuideLLMConfig(
		0.0001, 15, 0, 32, 0, 32, 32, 4, 0, 4, 4,
	)
	cs := NewClusterSimulator(cfg, workload, "")
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}

	agg := cs.AggregatedMetrics()
	if agg.CompletedRequests == 0 {
		t.Fatal("no completed requests")
	}

	// BC-MS-4c: sum(AllITLs) == sum(E2E - TTFT)
	var sumAllITLs float64
	for _, v := range agg.AllITLs {
		sumAllITLs += float64(v)
	}
	var sumE2EMinusTTFT float64
	for id, e2e := range agg.RequestE2Es {
		ttft := agg.RequestTTFTs[id]
		sumE2EMinusTTFT += e2e - ttft
	}
	if math.Abs(sumAllITLs-sumE2EMinusTTFT) > 10.0 {
		t.Errorf("BC-MS-4c violated: sum(AllITLs)=%.1f != sum(E2E-TTFT)=%.1f, diff=%.1f",
			sumAllITLs, sumE2EMinusTTFT, sumAllITLs-sumE2EMinusTTFT)
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// BC-MS-7c + BC-MS-10c: Cache and KV Metrics Bounded
// ═══════════════════════════════════════════════════════════════════════════════

func TestClusterMetrics_CacheHitRate_Bounded(t *testing.T) {
	cfg := msClusterConfig(2)
	workload := sim.NewGuideLLMConfig(
		0.0001, 10, 0, 32, 0, 32, 32, 3, 0, 3, 3,
	)
	cs := NewClusterSimulator(cfg, workload, "")
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}

	agg := cs.AggregatedMetrics()
	if agg.CacheHitRate < 0 || agg.CacheHitRate > 1 {
		t.Errorf("BC-MS-7c violated: aggregated CacheHitRate (%.4f) outside [0, 1]",
			agg.CacheHitRate)
	}

	// BC-MS-10c: per-instance peak bounded
	for i, inst := range cs.Instances() {
		m := inst.Metrics()
		if m.PeakKVBlocksUsed < 0 || m.PeakKVBlocksUsed > cfg.TotalKVBlocks {
			t.Errorf("BC-MS-10c violated for instance %d: PeakKVBlocksUsed=%d, capacity=%d",
				i, m.PeakKVBlocksUsed, cfg.TotalKVBlocks)
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// BC-MS-9c: Zero-Output Requests in Cluster Mode
//
// The phantom ITL fix must propagate correctly through cluster aggregation.
// ═══════════════════════════════════════════════════════════════════════════════

func TestClusterMetrics_ZeroOutput_NoPollution(t *testing.T) {
	cfg := msClusterConfig(2)
	// Dummy workload config (NumRequests=0) — real requests injected via SetPreGeneratedRequests
	dummyWorkload := sim.NewGuideLLMConfig(
		0.0001, 0, 0, 32, 0, 32, 32, 3, 0, 3, 3,
	)
	cs := NewClusterSimulator(cfg, dummyWorkload, "")

	// Inject a mix of zero-output and normal requests directly
	makeTokens := func(n int) []int {
		t := make([]int, n)
		for i := range t {
			t[i] = i + 1
		}
		return t
	}
	reqs := []*sim.Request{
		{ID: "z0", InputTokens: makeTokens(32), OutputTokens: []int{}, ArrivalTime: 0, State: sim.StateQueued},
		{ID: "n1", InputTokens: makeTokens(32), OutputTokens: makeTokens(4), ArrivalTime: 100000, State: sim.StateQueued},
		{ID: "z2", InputTokens: makeTokens(48), OutputTokens: nil, ArrivalTime: 200000, State: sim.StateQueued},
		{ID: "n3", InputTokens: makeTokens(16), OutputTokens: makeTokens(3), ArrivalTime: 300000, State: sim.StateQueued},
		{ID: "n4", InputTokens: makeTokens(32), OutputTokens: makeTokens(5), ArrivalTime: 400000, State: sim.StateQueued},
		{ID: "z5", InputTokens: makeTokens(16), OutputTokens: []int{}, ArrivalTime: 500000, State: sim.StateQueued},
	}
	cs.SetPreGeneratedRequests(reqs)

	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}

	agg := cs.AggregatedMetrics()
	if agg.CompletedRequests != 6 {
		t.Fatalf("expected 6 completed, got %d", agg.CompletedRequests)
	}

	// BC-MS-4c must hold with zero-output requests in the mix
	var sumAllITLs float64
	for _, v := range agg.AllITLs {
		sumAllITLs += float64(v)
	}
	var sumE2EMinusTTFT float64
	for id, e2e := range agg.RequestE2Es {
		ttft := agg.RequestTTFTs[id]
		sumE2EMinusTTFT += e2e - ttft
	}
	if math.Abs(sumAllITLs-sumE2EMinusTTFT) > 10.0 {
		t.Errorf("BC-MS-4c/9c violated: sum(AllITLs)=%.1f != sum(E2E-TTFT)=%.1f",
			sumAllITLs, sumE2EMinusTTFT)
	}

	// Zero-output requests should have E2E == TTFT (no decode phase)
	for _, zeroID := range []string{"z0", "z2", "z5"} {
		e2e := agg.RequestE2Es[zeroID]
		ttft := agg.RequestTTFTs[zeroID]
		if math.Abs(e2e-ttft) > 1.0 {
			t.Errorf("BC-MS-9c: zero-output %s has E2E (%.1f) != TTFT (%.1f)", zeroID, e2e, ttft)
		}
	}
}
