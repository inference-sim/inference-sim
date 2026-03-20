// multi_model_test.go — BDD/TDD tests for Phase 1A multi-model routing (US5).
package cluster

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// newTestDeploymentConfigWithModel creates a test DeploymentConfig using the specified model.
func newTestDeploymentConfigWithModel(numInstances int, model string) DeploymentConfig {
	cfg := newTestDeploymentConfig(numInstances)
	cfg.SimConfig = sim.SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                42,
		KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, model, "H100", 1, "blackbox", 0),
	}
	return cfg
}

// newModelRequest creates a minimal test request tagged with the given model.
func newModelRequest(id, model string, arrivalTime int64) *sim.Request {
	return &sim.Request{
		ID:          id,
		Model:       model,
		ArrivalTime: arrivalTime,
		InputTokens: []int{1, 2, 3, 4, 5},
		OutputTokens: []int{1, 2, 3},
		State:       sim.StateQueued,
	}
}

// ─── US5: Route Requests to Active Instances of the Correct Model ────────────

// T045: llama request routes only to llama instances — qwen absent from RouterState.
func TestMultiModel_BuildRouterState_FiltersbyModel(t *testing.T) {
	// Create cluster with 2 "llama" instances and 2 "qwen" instances by
	// creating a cluster where instances have Model set.
	llamaCfg := newTestDeploymentConfigWithModel(2, "llama")
	cs := NewClusterSimulator(llamaCfg, nil, nil)

	// Manually set model on instances to simulate multi-model setup
	for _, inst := range cs.instances {
		inst.Model = "llama"
	}
	// Add 2 "qwen" instances
	qwenCfg := newTestDeploymentConfigWithModel(2, "qwen")
	cs2 := NewClusterSimulator(qwenCfg, nil, nil)
	for _, inst := range cs2.instances {
		inst.Model = "qwen"
	}
	// Merge qwen instances into cs
	cs.instances = append(cs.instances, cs2.instances...)
	// Rebuild inFlightRequests map
	for _, inst := range cs2.instances {
		cs.inFlightRequests[string(inst.ID())] = 0
	}

	llamaReq := newModelRequest("req-llama", "llama", 0)
	state := buildRouterState(cs, llamaReq)

	t.Run("RouterState contains only llama snapshots", func(t *testing.T) {
		if len(state.Snapshots) != 2 {
			t.Errorf("got %d snapshots for llama request, want 2 (only llama instances)", len(state.Snapshots))
		}
		for _, snap := range state.Snapshots {
			if snap.Model != "llama" {
				t.Errorf("snapshot model = %q, want llama", snap.Model)
			}
		}
	})
}

// T046: All instances of model M non-Active → empty RouterState.
func TestMultiModel_BuildRouterState_EmptyWhenNonActive(t *testing.T) {
	cfg := newTestDeploymentConfigWithModel(2, "model-m")
	cs := NewClusterSimulator(cfg, nil, nil)

	// Mark all instances as Draining (not routable)
	for _, inst := range cs.instances {
		inst.Model = "model-m"
		inst.TransitionTo(InstanceStateDraining)
	}

	req := newModelRequest("req-1", "model-m", 0)
	state := buildRouterState(cs, req)

	t.Run("empty RouterState when no Active instances", func(t *testing.T) {
		if len(state.Snapshots) != 0 {
			t.Errorf("got %d snapshots, want 0 (all instances non-Active)", len(state.Snapshots))
		}
	})
}

// T046b: Request with empty Model includes all routable instances (backward-compat).
func TestMultiModel_BuildRouterState_EmptyModelIncludesAll(t *testing.T) {
	cfg := newTestDeploymentConfig(3)
	cs := NewClusterSimulator(cfg, nil, nil)

	// No model set — all instances treated as routable (backward-compat)
	noModelReq := newModelRequest("req-nomodel", "", 0)
	state := buildRouterState(cs, noModelReq)

	if len(state.Snapshots) != 3 {
		t.Errorf("got %d snapshots for empty-model request, want 3 (all instances)", len(state.Snapshots))
	}
}

// ─── T047: Per-model metrics in CollectRawMetrics output ─────────────────────

// T047: ComputePerModelMetrics partitions requests by model correctly.
func TestMultiModel_ComputePerModelMetrics(t *testing.T) {
	// Build a Metrics struct with requests from two models
	m := sim.NewMetrics()

	// Add llama requests
	for i := 0; i < 5; i++ {
		id := "llama-req-" + string(rune('0'+i))
		m.Requests[id] = sim.RequestMetrics{Model: "llama", ID: id}
		m.RequestTTFTs[id] = float64(i+1) * 100.0 // 100, 200, 300, 400, 500 μs
		m.RequestE2Es[id] = float64(i+1) * 500.0
	}

	// Add qwen requests
	for i := 0; i < 3; i++ {
		id := "qwen-req-" + string(rune('0'+i))
		m.Requests[id] = sim.RequestMetrics{Model: "qwen", ID: id}
		m.RequestTTFTs[id] = float64(i+1) * 200.0
		m.RequestE2Es[id] = float64(i+1) * 800.0
	}

	m.CompletedRequests = 8
	m.SimEndedTime = int64(10e6) // 10 seconds

	result := ComputePerModelMetrics(m)

	t.Run("two model keys present", func(t *testing.T) {
		if len(result) != 2 {
			t.Errorf("PerModelMetrics has %d keys, want 2 (llama + qwen)", len(result))
		}
	})

	t.Run("llama total requests", func(t *testing.T) {
		mm, ok := result["llama"]
		if !ok {
			t.Fatal("no llama entry in PerModelMetrics")
		}
		if mm.TotalRequests != 5 {
			t.Errorf("llama TotalRequests = %d, want 5", mm.TotalRequests)
		}
	})

	t.Run("qwen total requests", func(t *testing.T) {
		mm, ok := result["qwen"]
		if !ok {
			t.Fatal("no qwen entry in PerModelMetrics")
		}
		if mm.TotalRequests != 3 {
			t.Errorf("qwen TotalRequests = %d, want 3", mm.TotalRequests)
		}
	})

	t.Run("both have non-nil TTFT distributions", func(t *testing.T) {
		for model, mm := range result {
			if mm.TTFT.Count == 0 {
				t.Errorf("model %q TTFT distribution is empty", model)
			}
			if mm.E2E.Count == 0 {
				t.Errorf("model %q E2E distribution is empty", model)
			}
		}
	})
}

// T047b: ComputePerModelMetrics returns nil when no requests have Model set.
func TestMultiModel_ComputePerModelMetrics_EmptyWhenNoModel(t *testing.T) {
	m := sim.NewMetrics()
	m.Requests["req-1"] = sim.RequestMetrics{ID: "req-1", Model: ""} // no model
	m.RequestE2Es["req-1"] = 1000.0

	result := ComputePerModelMetrics(m)
	if result != nil {
		t.Errorf("ComputePerModelMetrics with no Model fields should return nil, got %v", result)
	}
}

// ─── T042: Warm-up TTFT factor applied in aggregateMetrics ───────────────────

// TestWarmUpTTFTFactor verifies that aggregateMetrics multiplies warm-up request TTFTs
// by WarmUpTTFTFactor (Gap 1 fix: factor wired into aggregateMetrics).
func TestWarmUpTTFTFactor_AppliedInAggregateMetrics(t *testing.T) {
	cfg := newTestDeploymentConfig(1)
	cfg.InstanceLifecycle = InstanceLifecycleConfig{
		WarmUpRequestCount: 2,
		WarmUpTTFTFactor:   2.0,
	}
	requests := newTestRequests(3)
	cs := NewClusterSimulator(cfg, requests, nil)

	// Manually set instance to WarmingUp and record two warm-up requests
	inst := cs.instances[0]
	inst.State = InstanceStateWarmingUp
	inst.warmUpRemaining = 2
	inst.RecordWarmUpRequest(requests[0].ID)
	inst.RecordWarmUpRequest(requests[1].ID)
	// Consume warm-up slots (simulating completions)
	inst.ConsumeWarmUpRequest()
	inst.ConsumeWarmUpRequest()

	// Inject fake TTFT values directly into the instance metrics
	inst.Metrics().RequestTTFTs[requests[0].ID] = 100.0
	inst.Metrics().RequestTTFTs[requests[1].ID] = 200.0
	inst.Metrics().RequestTTFTs[requests[2].ID] = 150.0 // not a warm-up request

	// aggregateMetrics should multiply warm-up TTFTs by 2.0
	merged := cs.aggregateMetrics()

	t.Run("warm-up request 0 TTFT multiplied by factor", func(t *testing.T) {
		got := merged.RequestTTFTs[requests[0].ID]
		want := 200.0 // 100 * 2.0
		if got != want {
			t.Errorf("TTFT[req0] = %.1f, want %.1f", got, want)
		}
	})

	t.Run("warm-up request 1 TTFT multiplied by factor", func(t *testing.T) {
		got := merged.RequestTTFTs[requests[1].ID]
		want := 400.0 // 200 * 2.0
		if got != want {
			t.Errorf("TTFT[req1] = %.1f, want %.1f", got, want)
		}
	})

	t.Run("non-warm-up request TTFT unchanged", func(t *testing.T) {
		got := merged.RequestTTFTs[requests[2].ID]
		want := 150.0
		if got != want {
			t.Errorf("TTFT[req2] = %.1f, want %.1f (should not be multiplied)", got, want)
		}
	})
}

// TestEvaluationResult_WithPerModelMetrics verifies Gap 2 fix: fluent builder.
func TestEvaluationResult_WithPerModelMetrics(t *testing.T) {
	perModel := map[string]*ModelMetrics{
		"test-model": {Model: "test-model", TotalRequests: 10},
	}
	result := NewEvaluationResult(nil, nil, nil, nil, 0, 0).WithPerModelMetrics(perModel)

	if result.PerModelMetrics == nil {
		t.Fatal("PerModelMetrics should not be nil after WithPerModelMetrics")
	}
	if mm, ok := result.PerModelMetrics["test-model"]; !ok || mm.TotalRequests != 10 {
		t.Errorf("PerModelMetrics[test-model].TotalRequests = %v, want 10", result.PerModelMetrics["test-model"])
	}
}
