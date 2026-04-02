// autoscaler_test.go — US1 tests for Phase 1C autoscaler pipeline wiring.
// Tests in this file MUST be written before implementing T011–T015.
// T009: TestScalingTickScheduling verifies tick interval and actuation delay semantics.
// T010: TestNoOpPipelineDeterminism verifies INV-6 — stub autoscaler must not change output.
package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// ---------------------------------------------------------------------------
// Stub implementations for autoscaler pipeline testing
// ---------------------------------------------------------------------------

// countingCollector counts how many times Collect() is called.
// Each call represents one pipeline tick firing.
type countingCollector struct {
	calls int
}

func (c *countingCollector) Collect(_ *sim.RouterState) []ModelSignals {
	c.calls++
	return nil
}

// nopAnalyzer is a no-op Analyzer that returns an empty AnalyzerResult.
type nopAnalyzer struct{}

func (n *nopAnalyzer) Name() string                        { return "nop" }
func (n *nopAnalyzer) Analyze(_ ModelSignals) AnalyzerResult { return AnalyzerResult{} }

// nopEngine is a no-op Engine that never emits ScaleDecisions.
type nopEngine struct{}

func (n *nopEngine) Optimize(_ []AnalyzerResult, _ GPUInventory) []ScaleDecision { return nil }

// nopActuator is a no-op Actuator.
type nopActuator struct{}

func (n *nopActuator) Apply(_ []ScaleDecision) {}

// newAutoscalerTestConfig returns a DeploymentConfig wired for autoscaler tests.
// All requests are nil (no-load run). Horizon is 200s so we can count tick firings.
func newAutoscalerTestConfig(intervalUs float64) DeploymentConfig {
	cfg := newTestDeploymentConfig(1)
	cfg.SimConfig.Horizon = 200_000_000 // 200s in microseconds
	cfg.ModelAutoscalerIntervalUs = intervalUs
	return cfg
}

// wireAutoscaler attaches a countingCollector + nop pipeline to cs.autoscaler.
// Returns the collector so the test can read its call count.
func wireAutoscaler(cs *ClusterSimulator) *countingCollector {
	collector := &countingCollector{}
	cs.autoscaler = &autoscalerPipeline{
		collector:       collector,
		analyzer:        &nopAnalyzer{},
		engine:          &nopEngine{},
		actuator:        &nopActuator{},
		lastScaleUpAt:   make(map[string]int64),
		lastScaleDownAt: make(map[string]int64),
	}
	return collector
}

// ---------------------------------------------------------------------------
// T009: TestScalingTickScheduling
// ---------------------------------------------------------------------------

// TestScalingTickScheduling verifies autoscaler tick firing behavior.
// Sub-tests:
//   (a) ModelAutoscalerIntervalUs=0 → no ScalingTickEvent fires (autoscaler disabled).
//   (b) interval=60s, horizon=200s → ticks fire at t=0, 60s, 120s, 180s (4 total).
//   (c) ActuationDelayUs={Mean:0} → ScaleActuationEvent.At == ScalingTickEvent.At.
//   (d) ActuationDelayUs={Mean:30s} → ScaleActuationEvent.At == tick.At + 30_000_000.
//
// Tests (a) and (b) fail before T015 (first tick scheduling) is implemented.
// Tests (c) and (d) fail before T013 (ScaleActuationEvent scheduling) is implemented.
func TestScalingTickScheduling(t *testing.T) {
	t.Run("a_zero_interval_no_tick", func(t *testing.T) {
		cfg := newAutoscalerTestConfig(0)
		cs := NewClusterSimulator(cfg, nil, nil)
		collector := wireAutoscaler(cs)
		if err := cs.Run(); err != nil {
			t.Fatalf("Run: %v", err)
		}
		if collector.calls != 0 {
			t.Errorf("interval=0: expected 0 ticks, got %d", collector.calls)
		}
	})

	t.Run("b_60s_interval_200s_horizon_fires_4_ticks", func(t *testing.T) {
		const intervalUs = 60_000_000.0 // 60s
		cfg := newAutoscalerTestConfig(intervalUs)
		cs := NewClusterSimulator(cfg, nil, nil)
		collector := wireAutoscaler(cs)
		if err := cs.Run(); err != nil {
			t.Fatalf("Run: %v", err)
		}
		// Horizon=200s, interval=60s → ticks fire at t=0, 60s, 120s, 180s → 4 ticks.
		// The tick at 240s is scheduled but exceeds horizon and is not executed.
		wantTicks := 4
		if collector.calls != wantTicks {
			t.Errorf("interval=60s, horizon=200s: expected %d ticks, got %d", wantTicks, collector.calls)
		}
	})

	t.Run("c_zero_actuation_delay_actuation_same_tick", func(t *testing.T) {
		const intervalUs = 60_000_000.0
		cfg := newAutoscalerTestConfig(intervalUs)
		cfg.ActuationDelayUs = DelaySpec{Mean: 0, Stddev: 0} // zero delay
		cs := NewClusterSimulator(cfg, nil, nil)
		wireAutoscaler(cs)

		// Manually inject a ScalingTickEvent at t=0 to test actuation scheduling.
		tickAt := int64(0)
		ev := &ScalingTickEvent{At: tickAt}
		ev.Execute(cs)

		// With zero delay, there should be a ScaleActuationEvent.At == tickAt.
		actuationAt, found := findFirstActuationEventAt(cs)
		if !found {
			t.Fatal("expected ScaleActuationEvent in queue after zero-delay tick, got none")
		}
		if actuationAt != tickAt {
			t.Errorf("zero delay: actuation.At=%d, want %d (== tick.At)", actuationAt, tickAt)
		}
	})

	t.Run("d_30s_actuation_delay_shifts_actuation", func(t *testing.T) {
		const intervalUs = 60_000_000.0
		cfg := newAutoscalerTestConfig(intervalUs)
		cfg.ActuationDelayUs = DelaySpec{Mean: 30, Stddev: 0} // 30s deterministic delay
		cs := NewClusterSimulator(cfg, nil, nil)
		wireAutoscaler(cs)

		tickAt := int64(0)
		ev := &ScalingTickEvent{At: tickAt}
		ev.Execute(cs)

		actuationAt, found := findFirstActuationEventAt(cs)
		if !found {
			t.Fatal("expected ScaleActuationEvent in queue after 30s-delay tick, got none")
		}
		wantActuationAt := tickAt + 30_000_000 // tick.At + 30s in μs
		if actuationAt != wantActuationAt {
			t.Errorf("30s delay: actuation.At=%d, want %d", actuationAt, wantActuationAt)
		}
	})
}

// findFirstActuationEventAt scans the cluster event queue for a ScaleActuationEvent.
// Returns its At timestamp and true if found, or 0 and false if not.
func findFirstActuationEventAt(cs *ClusterSimulator) (int64, bool) {
	for _, entry := range cs.clusterEvents {
		if ev, ok := entry.event.(*ScaleActuationEvent); ok {
			return ev.At, true
		}
	}
	return 0, false
}

// ---------------------------------------------------------------------------
// T010: TestNoOpPipelineDeterminism
// ---------------------------------------------------------------------------

// TestNoOpPipelineDeterminism verifies INV-6: a no-op autoscaler pipeline
// produces byte-identical simulation metrics compared to no autoscaler.
// Uses newTestDeploymentConfig (math.MaxInt64 horizon) with requests to get real completions.
// This is a regression guard — it must keep passing after T011–T015 are wired.
func TestNoOpPipelineDeterminism(t *testing.T) {
	const intervalUs = 60_000_000.0 // 60s
	// Use a bounded horizon so tick scheduling terminates.
	// 20 requests at rate 10/s arrive within the first ~2s; horizon=200s ensures all complete.
	const horizonUs = 200_000_000 // 200s

	// Each run needs its own request slice — sim.Request is mutated during simulation.
	// Run A: with stub autoscaler wired
	cfgA := newTestDeploymentConfig(1)
	cfgA.ModelAutoscalerIntervalUs = intervalUs
	cfgA.SimConfig.Horizon = horizonUs
	csA := NewClusterSimulator(cfgA, newTestRequests(20), nil)
	wireAutoscaler(csA)
	if err := csA.Run(); err != nil {
		t.Fatalf("Run A (with autoscaler): %v", err)
	}

	// Run B: no autoscaler (interval=0 → disabled), same horizon for fair comparison
	cfgB := newTestDeploymentConfig(1)
	cfgB.ModelAutoscalerIntervalUs = 0
	cfgB.SimConfig.Horizon = horizonUs
	csB := NewClusterSimulator(cfgB, newTestRequests(20), nil)
	if err := csB.Run(); err != nil {
		t.Fatalf("Run B (no autoscaler): %v", err)
	}

	// Both runs must produce identical aggregated metrics.
	mA := csA.AggregatedMetrics()
	mB := csB.AggregatedMetrics()

	if mA.CompletedRequests != mB.CompletedRequests {
		t.Errorf("CompletedRequests: with=%d, without=%d", mA.CompletedRequests, mB.CompletedRequests)
	}
	if mA.TotalInputTokens != mB.TotalInputTokens {
		t.Errorf("TotalInputTokens: with=%d, without=%d", mA.TotalInputTokens, mB.TotalInputTokens)
	}
	if mA.TotalOutputTokens != mB.TotalOutputTokens {
		t.Errorf("TotalOutputTokens: with=%d, without=%d", mA.TotalOutputTokens, mB.TotalOutputTokens)
	}
	if mA.SimEndedTime != mB.SimEndedTime {
		t.Errorf("SimEndedTime: with=%d, without=%d", mA.SimEndedTime, mB.SimEndedTime)
	}
}
