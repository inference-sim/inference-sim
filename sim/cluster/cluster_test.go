package cluster

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// newTestDeploymentConfig creates a DeploymentConfig suitable for testing.
func newTestDeploymentConfig(numInstances int) DeploymentConfig {
	return DeploymentConfig{
		NumInstances:              numInstances,
		Horizon:                   math.MaxInt64,
		Seed:                      42,
		TotalKVBlocks:             10000,
		BlockSizeTokens:           16,
		MaxRunningReqs:            256,
		MaxScheduledTokens:        2048,
		LongPrefillTokenThreshold: 0,
		BetaCoeffs:                []float64{1000, 10, 5},
		AlphaCoeffs:               []float64{100, 1, 100},
		ModelConfig:               sim.ModelConfig{},
		HWConfig:                  sim.HardwareCalib{},
		Model:                     "test-model",
		GPU:                       "H100",
		TP:                        1,
		Roofline:                  false,
	}
}

// newTestWorkload creates a GuideLLMConfig suitable for testing.
func newTestWorkload(maxPrompts int) *sim.GuideLLMConfig {
	return &sim.GuideLLMConfig{
		Rate:               10.0 / 1e6,
		MaxPrompts:         maxPrompts,
		PrefixTokens:       0,
		PromptTokens:       100,
		PromptTokensStdDev: 20,
		PromptTokensMin:    10,
		PromptTokensMax:    200,
		OutputTokens:       50,
		OutputTokensStdDev: 10,
		OutputTokensMin:    10,
		OutputTokensMax:    100,
	}
}

// TestClusterSimulator_SingleInstance_GoldenEquivalence verifies BC-7, BC-9:
// GIVEN each golden dataset test case configured as NumInstances=1 via ClusterSimulator
// WHEN Run() called
// THEN CompletedRequests, TotalInputTokens, TotalOutputTokens match golden values exactly.
func TestClusterSimulator_SingleInstance_GoldenEquivalence(t *testing.T) {
	dataset := loadGoldenDataset(t)

	if len(dataset.Tests) == 0 {
		t.Fatal("Golden dataset contains no test cases")
	}

	for _, tc := range dataset.Tests {
		t.Run(tc.Model, func(t *testing.T) {
			config := DeploymentConfig{
				NumInstances:              1,
				Horizon:                   math.MaxInt64,
				Seed:                      tc.Seed,
				TotalKVBlocks:             tc.TotalKVBlocks,
				BlockSizeTokens:           tc.BlockSizeInTokens,
				MaxRunningReqs:            tc.MaxNumRunningReqs,
				MaxScheduledTokens:        tc.MaxNumScheduledTokens,
				LongPrefillTokenThreshold: tc.LongPrefillTokenThreshold,
				BetaCoeffs:                tc.BetaCoeffs,
				AlphaCoeffs:               tc.AlphaCoeffs,
				ModelConfig:               sim.ModelConfig{},
				HWConfig:                  sim.HardwareCalib{},
				Model:                     tc.Model,
				GPU:                       tc.Hardware,
				TP:                        tc.TP,
				Roofline:                  false,
			}

			workload := &sim.GuideLLMConfig{
				Rate:               tc.Rate / 1e6,
				MaxPrompts:         tc.MaxPrompts,
				PrefixTokens:       tc.PrefixTokens,
				PromptTokens:       tc.PromptTokens,
				PromptTokensStdDev: tc.PromptTokensStdev,
				PromptTokensMin:    tc.PromptTokensMin,
				PromptTokensMax:    tc.PromptTokensMax,
				OutputTokens:       tc.OutputTokens,
				OutputTokensStdDev: tc.OutputTokensStdev,
				OutputTokensMin:    tc.OutputTokensMin,
				OutputTokensMax:    tc.OutputTokensMax,
			}

			cs := NewClusterSimulator(config, workload, "")
			cs.Run()

			m := cs.AggregatedMetrics()
			if m.CompletedRequests != tc.Metrics.CompletedRequests {
				t.Errorf("completed_requests: got %d, want %d",
					m.CompletedRequests, tc.Metrics.CompletedRequests)
			}
			if m.TotalInputTokens != tc.Metrics.TotalInputTokens {
				t.Errorf("total_input_tokens: got %d, want %d",
					m.TotalInputTokens, tc.Metrics.TotalInputTokens)
			}
			if m.TotalOutputTokens != tc.Metrics.TotalOutputTokens {
				t.Errorf("total_output_tokens: got %d, want %d",
					m.TotalOutputTokens, tc.Metrics.TotalOutputTokens)
			}
			// Verify timing: SimEndedTime must match golden vllm_estimated_duration_s
			vllmRuntime := float64(m.SimEndedTime) / 1e6
			assertFloat64Equal(t, "vllm_estimated_duration_s",
				tc.Metrics.VllmEstimatedDurationS, vllmRuntime, 1e-9)
		})
	}
}

// TestClusterSimulator_MultiInstance_Determinism verifies BC-2:
// GIVEN N=4, seed=42, 100 requests
// WHEN run twice
// THEN per-instance and aggregated CompletedRequests are identical.
func TestClusterSimulator_MultiInstance_Determinism(t *testing.T) {
	config := newTestDeploymentConfig(4)
	workload := newTestWorkload(100)

	cs1 := NewClusterSimulator(config, workload, "")
	cs1.Run()

	cs2 := NewClusterSimulator(config, workload, "")
	cs2.Run()

	// Check aggregated
	if cs1.AggregatedMetrics().CompletedRequests != cs2.AggregatedMetrics().CompletedRequests {
		t.Errorf("aggregated CompletedRequests differ: %d vs %d",
			cs1.AggregatedMetrics().CompletedRequests, cs2.AggregatedMetrics().CompletedRequests)
	}

	// Check aggregated token counts and SimEndedTime
	a1, a2 := cs1.AggregatedMetrics(), cs2.AggregatedMetrics()
	if a1.TotalInputTokens != a2.TotalInputTokens {
		t.Errorf("aggregated TotalInputTokens differ: %d vs %d",
			a1.TotalInputTokens, a2.TotalInputTokens)
	}
	if a1.TotalOutputTokens != a2.TotalOutputTokens {
		t.Errorf("aggregated TotalOutputTokens differ: %d vs %d",
			a1.TotalOutputTokens, a2.TotalOutputTokens)
	}
	if a1.SimEndedTime != a2.SimEndedTime {
		t.Errorf("aggregated SimEndedTime differ: %d vs %d",
			a1.SimEndedTime, a2.SimEndedTime)
	}

	// Check per-instance (counts and timing)
	for i := 0; i < 4; i++ {
		m1, m2 := cs1.Instances()[i].Metrics(), cs2.Instances()[i].Metrics()
		if m1.CompletedRequests != m2.CompletedRequests {
			t.Errorf("instance %d CompletedRequests differ: %d vs %d", i, m1.CompletedRequests, m2.CompletedRequests)
		}
		if m1.TotalInputTokens != m2.TotalInputTokens {
			t.Errorf("instance %d TotalInputTokens differ: %d vs %d", i, m1.TotalInputTokens, m2.TotalInputTokens)
		}
		if m1.TotalOutputTokens != m2.TotalOutputTokens {
			t.Errorf("instance %d TotalOutputTokens differ: %d vs %d", i, m1.TotalOutputTokens, m2.TotalOutputTokens)
		}
		if m1.TTFTSum != m2.TTFTSum {
			t.Errorf("instance %d TTFTSum differ: %d vs %d", i, m1.TTFTSum, m2.TTFTSum)
		}
		if m1.SimEndedTime != m2.SimEndedTime {
			t.Errorf("instance %d SimEndedTime differ: %d vs %d", i, m1.SimEndedTime, m2.SimEndedTime)
		}
	}
}

// TestClusterSimulator_MultiInstance_AllComplete verifies BC-3, BC-5:
// GIVEN N=4, 100 requests
// WHEN run
// THEN aggregated CompletedRequests == 100 AND each instance completed > 0.
func TestClusterSimulator_MultiInstance_AllComplete(t *testing.T) {
	config := newTestDeploymentConfig(4)
	workload := newTestWorkload(100)

	cs := NewClusterSimulator(config, workload, "")
	cs.Run()

	m := cs.AggregatedMetrics()
	if m.CompletedRequests != 100 {
		t.Errorf("aggregated CompletedRequests: got %d, want 100", m.CompletedRequests)
	}

	for i, inst := range cs.Instances() {
		if inst.Metrics().CompletedRequests == 0 {
			t.Errorf("instance %d CompletedRequests == 0, want > 0", i)
		}
	}
}

// TestClusterSimulator_RoundRobin_EvenDistribution verifies BC-3:
// GIVEN N=3, 9 requests
// WHEN run
// THEN each instance has CompletedRequests == 3.
func TestClusterSimulator_RoundRobin_EvenDistribution(t *testing.T) {
	config := newTestDeploymentConfig(3)
	workload := newTestWorkload(9)

	cs := NewClusterSimulator(config, workload, "")
	cs.Run()

	for i, inst := range cs.Instances() {
		if inst.Metrics().CompletedRequests != 3 {
			t.Errorf("instance %d CompletedRequests: got %d, want 3",
				i, inst.Metrics().CompletedRequests)
		}
	}
}

// TestClusterSimulator_RoundRobin_UnevenDistribution verifies BC-3:
// GIVEN N=3, 10 requests
// WHEN run
// THEN instance 0 has 4, instances 1,2 have 3.
func TestClusterSimulator_RoundRobin_UnevenDistribution(t *testing.T) {
	config := newTestDeploymentConfig(3)
	workload := newTestWorkload(10)

	cs := NewClusterSimulator(config, workload, "")
	cs.Run()

	expected := []int{4, 3, 3}
	for i, inst := range cs.Instances() {
		if inst.Metrics().CompletedRequests != expected[i] {
			t.Errorf("instance %d CompletedRequests: got %d, want %d",
				i, inst.Metrics().CompletedRequests, expected[i])
		}
	}
}

// TestClusterSimulator_ZeroRequestInstances verifies C.4:
// GIVEN N=4, 2 requests
// WHEN run
// THEN instances 0,1 have CompletedRequests == 1, instances 2,3 have 0, no panic.
func TestClusterSimulator_ZeroRequestInstances(t *testing.T) {
	config := newTestDeploymentConfig(4)
	workload := newTestWorkload(2)

	cs := NewClusterSimulator(config, workload, "")
	cs.Run()

	expected := []int{1, 1, 0, 0}
	for i, inst := range cs.Instances() {
		if inst.Metrics().CompletedRequests != expected[i] {
			t.Errorf("instance %d CompletedRequests: got %d, want %d",
				i, inst.Metrics().CompletedRequests, expected[i])
		}
	}

	if cs.AggregatedMetrics().CompletedRequests != 2 {
		t.Errorf("aggregated CompletedRequests: got %d, want 2",
			cs.AggregatedMetrics().CompletedRequests)
	}
}

// TestClusterSimulator_AggregatedMetrics_Correctness verifies BC-7:
// GIVEN N=2
// WHEN run
// THEN aggregated == sum(per-instance) for counts, max for SimEndedTime.
func TestClusterSimulator_AggregatedMetrics_Correctness(t *testing.T) {
	config := newTestDeploymentConfig(2)
	workload := newTestWorkload(50)

	cs := NewClusterSimulator(config, workload, "")
	cs.Run()

	var sumCompleted, sumInput, sumOutput int
	var maxSimEnded int64
	for _, inst := range cs.Instances() {
		m := inst.Metrics()
		sumCompleted += m.CompletedRequests
		sumInput += m.TotalInputTokens
		sumOutput += m.TotalOutputTokens
		if m.SimEndedTime > maxSimEnded {
			maxSimEnded = m.SimEndedTime
		}
	}

	agg := cs.AggregatedMetrics()
	if agg.CompletedRequests != sumCompleted {
		t.Errorf("aggregated CompletedRequests: got %d, want %d (sum)", agg.CompletedRequests, sumCompleted)
	}
	if agg.TotalInputTokens != sumInput {
		t.Errorf("aggregated TotalInputTokens: got %d, want %d (sum)", agg.TotalInputTokens, sumInput)
	}
	if agg.TotalOutputTokens != sumOutput {
		t.Errorf("aggregated TotalOutputTokens: got %d, want %d (sum)", agg.TotalOutputTokens, sumOutput)
	}
	if agg.SimEndedTime != maxSimEnded {
		t.Errorf("aggregated SimEndedTime: got %d, want %d (max)", agg.SimEndedTime, maxSimEnded)
	}

	// Verify per-request map merging
	var sumRequests, sumTTFTs, sumE2Es, sumITLs, sumAllITLs int
	var sumTTFTSum int64
	for _, inst := range cs.Instances() {
		m := inst.Metrics()
		sumRequests += len(m.Requests)
		sumTTFTs += len(m.RequestTTFTs)
		sumE2Es += len(m.RequestE2Es)
		sumITLs += len(m.RequestITLs)
		sumAllITLs += len(m.AllITLs)
		sumTTFTSum += m.TTFTSum
	}
	if len(agg.Requests) != sumRequests {
		t.Errorf("aggregated len(Requests): got %d, want %d (sum)", len(agg.Requests), sumRequests)
	}
	if len(agg.RequestTTFTs) != sumTTFTs {
		t.Errorf("aggregated len(RequestTTFTs): got %d, want %d (sum)", len(agg.RequestTTFTs), sumTTFTs)
	}
	if len(agg.RequestE2Es) != sumE2Es {
		t.Errorf("aggregated len(RequestE2Es): got %d, want %d (sum)", len(agg.RequestE2Es), sumE2Es)
	}
	if len(agg.AllITLs) != sumAllITLs {
		t.Errorf("aggregated len(AllITLs): got %d, want %d (sum)", len(agg.AllITLs), sumAllITLs)
	}
	if agg.TTFTSum != sumTTFTSum {
		t.Errorf("aggregated TTFTSum: got %d, want %d (sum)", agg.TTFTSum, sumTTFTSum)
	}
	if agg.RequestRate != workload.Rate {
		t.Errorf("aggregated RequestRate: got %v, want %v", agg.RequestRate, workload.Rate)
	}
}

// TestClusterSimulator_SharedClock_MonotonicGlobal verifies BC-6:
// GIVEN N=2
// WHEN run
// THEN cluster.Clock() >= every instance's Clock().
func TestClusterSimulator_SharedClock_MonotonicGlobal(t *testing.T) {
	config := newTestDeploymentConfig(2)
	workload := newTestWorkload(50)

	cs := NewClusterSimulator(config, workload, "")
	cs.Run()

	for i, inst := range cs.Instances() {
		if cs.Clock() < inst.Clock() {
			t.Errorf("cluster clock %d < instance %d clock %d",
				cs.Clock(), i, inst.Clock())
		}
	}
}

// TestClusterSimulator_RunOnce_Panics verifies C.3:
// GIVEN cluster has Run()
// WHEN Run() called again
// THEN panic.
func TestClusterSimulator_RunOnce_Panics(t *testing.T) {
	config := newTestDeploymentConfig(2)
	workload := newTestWorkload(10)

	cs := NewClusterSimulator(config, workload, "")
	cs.Run()

	defer func() {
		r := recover()
		if r == nil {
			t.Error("expected panic on second Run(), got none")
		}
		expected := "ClusterSimulator.Run() called more than once"
		if r != expected {
			t.Errorf("panic message = %q, want %q", r, expected)
		}
	}()
	cs.Run()
}

// TestNewClusterSimulator_ZeroInstances_Panics verifies C.4:
// GIVEN NumInstances=0
// WHEN NewClusterSimulator()
// THEN panic.
func TestNewClusterSimulator_ZeroInstances_Panics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Error("expected panic for NumInstances=0, got none")
		}
		expected := "ClusterSimulator: NumInstances must be >= 1"
		if r != expected {
			t.Errorf("panic message = %q, want %q", r, expected)
		}
	}()

	config := newTestDeploymentConfig(0)
	NewClusterSimulator(config, newTestWorkload(10), "")
}

// TestInstanceSimulator_InjectAfterRun_Panics verifies C.3:
// GIVEN instance has Run()
// WHEN InjectRequest() called
// THEN panic.
func TestInstanceSimulator_InjectAfterRun_Panics(t *testing.T) {
	inst := NewInstanceSimulatorWithoutWorkload(
		"test", math.MaxInt64, 42, 10000, 16, 256, 2048, 0,
		[]float64{1000, 10, 5}, []float64{100, 1, 100},
		sim.ModelConfig{}, sim.HardwareCalib{}, "test", "H100", 1, false,
	)
	inst.Run()

	defer func() {
		r := recover()
		if r == nil {
			t.Error("expected panic on InjectRequest after Run(), got none")
		}
		expected := "InstanceSimulator.InjectRequest() called after Run()"
		if r != expected {
			t.Errorf("panic message = %q, want %q", r, expected)
		}
	}()
	inst.InjectRequest(&sim.Request{
		ID: "req", ArrivalTime: 0, InputTokens: make([]int, 5),
		OutputTokens: make([]int, 3), State: "queued",
	})
}

// TestClusterSimulator_GloballyUniqueRequestIDs verifies BC-4:
// GIVEN N=4, 20 requests
// WHEN run
// THEN len(AggregatedMetrics().Requests) == AggregatedMetrics().CompletedRequests
// AND all request IDs across instances are distinct.
func TestClusterSimulator_GloballyUniqueRequestIDs(t *testing.T) {
	config := newTestDeploymentConfig(4)
	workload := newTestWorkload(20)

	cs := NewClusterSimulator(config, workload, "")
	cs.Run()

	agg := cs.AggregatedMetrics()
	if len(agg.Requests) != agg.CompletedRequests {
		t.Errorf("len(Requests)=%d != CompletedRequests=%d — possible ID collision",
			len(agg.Requests), agg.CompletedRequests)
	}

	// Verify all IDs across instances are distinct
	seen := make(map[string]int) // request ID -> instance index
	for i, inst := range cs.Instances() {
		for id := range inst.Metrics().Requests {
			if prev, exists := seen[id]; exists {
				t.Errorf("duplicate request ID %q: instance %d and instance %d", id, prev, i)
			}
			seen[id] = i
		}
	}
}

// TestClusterSimulator_HorizonEnforcement verifies BC-8:
// GIVEN a finite horizon and enough requests to exceed it
// WHEN run
// THEN some requests are not completed AND cluster clock does not far exceed horizon.
func TestClusterSimulator_HorizonEnforcement(t *testing.T) {
	config := newTestDeploymentConfig(2)
	config.Horizon = 500000 // finite horizon
	workload := newTestWorkload(100)

	cs := NewClusterSimulator(config, workload, "")
	cs.Run()

	agg := cs.AggregatedMetrics()

	// With a tight horizon, not all requests should complete
	if agg.CompletedRequests >= 100 {
		t.Errorf("expected fewer than 100 completed requests with tight horizon, got %d",
			agg.CompletedRequests)
	}

	// SimEndedTime should be capped at horizon
	if agg.SimEndedTime > config.Horizon {
		t.Errorf("SimEndedTime %d exceeds horizon %d", agg.SimEndedTime, config.Horizon)
	}
}

// TestClusterSimulator_NilWorkload_Panics verifies the nil workload guard.
func TestClusterSimulator_NilWorkload_Panics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Error("expected panic for nil workload and empty traces path, got none")
		}
		expected := "ClusterSimulator: workload config is nil and no traces path provided"
		if r != expected {
			t.Errorf("panic message = %q, want %q", r, expected)
		}
	}()

	config := newTestDeploymentConfig(2)
	NewClusterSimulator(config, nil, "")
}

// TestClusterSimulator_AggregatedMetrics_BeforeRun_Panics verifies the hasRun guard.
func TestClusterSimulator_AggregatedMetrics_BeforeRun_Panics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Error("expected panic for AggregatedMetrics() before Run(), got none")
		}
		expected := "ClusterSimulator.AggregatedMetrics() called before Run()"
		if r != expected {
			t.Errorf("panic message = %q, want %q", r, expected)
		}
	}()

	config := newTestDeploymentConfig(2)
	cs := NewClusterSimulator(config, newTestWorkload(10), "")
	cs.AggregatedMetrics()
}

// === Benchmarks ===

func BenchmarkClusterSimulator_1K_1Instance(b *testing.B) {
	config := newTestDeploymentConfig(1)
	workload := newTestWorkload(1000)
	for i := 0; i < b.N; i++ {
		cs := NewClusterSimulator(config, workload, "")
		cs.Run()
	}
}

func BenchmarkClusterSimulator_10K_4Instances(b *testing.B) {
	config := newTestDeploymentConfig(4)
	workload := newTestWorkload(10000)
	for i := 0; i < b.N; i++ {
		cs := NewClusterSimulator(config, workload, "")
		cs.Run()
	}
}

func BenchmarkClusterSimulator_1K_10Instances(b *testing.B) {
	config := newTestDeploymentConfig(10)
	workload := newTestWorkload(1000)
	for i := 0; i < b.N; i++ {
		cs := NewClusterSimulator(config, workload, "")
		cs.Run()
	}
}

// === Workload Parity Tests (F.4) ===

// TestClusterWorkloadGen_MatchesSimulator verifies D.6:
// GIVEN same seed and workload config
// WHEN sim.NewSimulator generates workload AND ClusterSimulator generates requests
// THEN count matches and for each request: ArrivalTime, len(InputTokens), len(OutputTokens) match.
func TestClusterWorkloadGen_MatchesSimulator(t *testing.T) {
	dataset := loadGoldenDataset(t)
	if len(dataset.Tests) == 0 {
		t.Fatal("Golden dataset contains no test cases")
	}

	for _, tc := range dataset.Tests {
		t.Run(tc.Model, func(t *testing.T) {
			guideLLMConfig := &sim.GuideLLMConfig{
				Rate:               tc.Rate / 1e6,
				MaxPrompts:         tc.MaxPrompts,
				PrefixTokens:       tc.PrefixTokens,
				PromptTokens:       tc.PromptTokens,
				PromptTokensStdDev: tc.PromptTokensStdev,
				PromptTokensMin:    tc.PromptTokensMin,
				PromptTokensMax:    tc.PromptTokensMax,
				OutputTokens:       tc.OutputTokens,
				OutputTokensStdDev: tc.OutputTokensStdev,
				OutputTokensMin:    tc.OutputTokensMin,
				OutputTokensMax:    tc.OutputTokensMax,
			}

			// Reference: sim.NewSimulator generates workload internally
			refSim := sim.NewSimulator(
				math.MaxInt64, tc.Seed, tc.TotalKVBlocks, tc.BlockSizeInTokens,
				tc.MaxNumRunningReqs, tc.MaxNumScheduledTokens,
				tc.LongPrefillTokenThreshold, tc.BetaCoeffs, tc.AlphaCoeffs,
				guideLLMConfig, sim.ModelConfig{}, sim.HardwareCalib{},
				tc.Model, tc.Hardware, tc.TP, false, "",
			)

			// Cluster workload generation
			config := DeploymentConfig{
				NumInstances:              1,
				Horizon:                   math.MaxInt64,
				Seed:                      tc.Seed,
				TotalKVBlocks:             tc.TotalKVBlocks,
				BlockSizeTokens:           tc.BlockSizeInTokens,
				MaxRunningReqs:            tc.MaxNumRunningReqs,
				MaxScheduledTokens:        tc.MaxNumScheduledTokens,
				LongPrefillTokenThreshold: tc.LongPrefillTokenThreshold,
				BetaCoeffs:                tc.BetaCoeffs,
				AlphaCoeffs:               tc.AlphaCoeffs,
				Model:                     tc.Model,
				GPU:                       tc.Hardware,
				TP:                        tc.TP,
			}
			cs := NewClusterSimulator(config, guideLLMConfig, "")
			requests := cs.generateRequestsFromDistribution()

			// Compare counts
			refCount := len(refSim.Metrics.Requests)
			if len(requests) != refCount {
				t.Fatalf("request count mismatch: cluster=%d, simulator=%d", len(requests), refCount)
			}

			// Compare each request
			for _, req := range requests {
				refMetric, ok := refSim.Metrics.Requests[req.ID]
				if !ok {
					t.Errorf("request %s not found in simulator Metrics.Requests", req.ID)
					continue
				}
				// ArrivalTime: compare via ArrivedAt (float64 seconds)
				gotArrived := float64(req.ArrivalTime) / 1e6
				if gotArrived != refMetric.ArrivedAt {
					t.Errorf("request %s ArrivalTime: cluster=%v, simulator=%v",
						req.ID, gotArrived, refMetric.ArrivedAt)
				}
				if len(req.InputTokens) != refMetric.NumPrefillTokens {
					t.Errorf("request %s InputTokens len: cluster=%d, simulator=%d",
						req.ID, len(req.InputTokens), refMetric.NumPrefillTokens)
				}
				if len(req.OutputTokens) != refMetric.NumDecodeTokens {
					t.Errorf("request %s OutputTokens len: cluster=%d, simulator=%d",
						req.ID, len(req.OutputTokens), refMetric.NumDecodeTokens)
				}
			}
		})
	}
}

// TestClusterWorkloadGen_Determinism verifies BC-2:
// GIVEN same seed
// WHEN called twice
// THEN request lists are identical.
func TestClusterWorkloadGen_Determinism(t *testing.T) {
	config := newTestDeploymentConfig(1)
	workload := newTestWorkload(50)

	cs1 := NewClusterSimulator(config, workload, "")
	reqs1 := cs1.generateRequestsFromDistribution()

	cs2 := NewClusterSimulator(config, workload, "")
	reqs2 := cs2.generateRequestsFromDistribution()

	if len(reqs1) != len(reqs2) {
		t.Fatalf("request count mismatch: %d vs %d", len(reqs1), len(reqs2))
	}

	for i := range reqs1 {
		if reqs1[i].ID != reqs2[i].ID {
			t.Errorf("request %d ID mismatch: %s vs %s", i, reqs1[i].ID, reqs2[i].ID)
		}
		if reqs1[i].ArrivalTime != reqs2[i].ArrivalTime {
			t.Errorf("request %d ArrivalTime mismatch: %d vs %d",
				i, reqs1[i].ArrivalTime, reqs2[i].ArrivalTime)
		}
		if len(reqs1[i].InputTokens) != len(reqs2[i].InputTokens) {
			t.Errorf("request %d InputTokens len mismatch: %d vs %d",
				i, len(reqs1[i].InputTokens), len(reqs2[i].InputTokens))
		}
		if len(reqs1[i].OutputTokens) != len(reqs2[i].OutputTokens) {
			t.Errorf("request %d OutputTokens len mismatch: %d vs %d",
				i, len(reqs1[i].OutputTokens), len(reqs2[i].OutputTokens))
		}
	}
}
