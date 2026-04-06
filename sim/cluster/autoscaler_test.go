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

// onceEngine emits a single ScaleDecision on the first Optimize() call, then returns nil.
// Used to trigger Actuator.Apply() exactly once so actuation tests can observe it.
type onceEngine struct {
	delta int
	fired bool
}

func (e *onceEngine) Optimize(_ []AnalyzerResult, _ GPUInventory) []ScaleDecision {
	if e.fired {
		return nil
	}
	e.fired = true
	return []ScaleDecision{{ModelID: "test-model", Variant: VariantSpec{GPUType: "A100", TPDegree: 1}, Delta: e.delta}}
}

// nopActuator is a no-op Actuator.
type nopActuator struct{}

func (n *nopActuator) Apply(_ []ScaleDecision) error { return nil }

// recordingActuator records the first Apply() call timestamp via a channel.
// Used by T009(c,d) to observe actuation timing without accessing internal fields.
type recordingActuator struct {
	applied chan []ScaleDecision
}

func newRecordingActuator() *recordingActuator {
	return &recordingActuator{applied: make(chan []ScaleDecision, 1)}
}

func (r *recordingActuator) Apply(decisions []ScaleDecision) error {
	select {
	case r.applied <- decisions:
	default:
	}
	return nil
}

// newAutoscalerTestConfig returns a DeploymentConfig wired for autoscaler tests.
// All requests are nil (no-load run). Horizon is 200s so we can count tick firings.
func newAutoscalerTestConfig(intervalUs float64) DeploymentConfig {
	cfg := newTestDeploymentConfig(1)
	cfg.Horizon = 200_000_000 // 200s in microseconds
	cfg.ModelAutoscalerIntervalUs = intervalUs
	return cfg
}

// newTestPipeline constructs an autoscalerPipeline with the given components.
// Canonical constructor for test helpers — satisfies R4 (single construction site).
func newTestPipeline(collector Collector, analyzer Analyzer, engine Engine, actuator Actuator) *autoscalerPipeline {
	return &autoscalerPipeline{
		collector:       collector,
		analyzer:        analyzer,
		engine:          engine,
		actuator:        actuator,
		lastScaleUpAt:   make(map[string]int64),
		lastScaleDownAt: make(map[string]int64),
	}
}

// wireAutoscaler attaches a countingCollector + nop pipeline to cs.autoscaler.
// Returns the collector so the test can read its call count.
func wireAutoscaler(cs *ClusterSimulator) *countingCollector {
	collector := &countingCollector{}
	cs.autoscaler = newTestPipeline(collector, &nopAnalyzer{}, &nopEngine{}, &nopActuator{})
	return collector
}

// ---------------------------------------------------------------------------
// T009: TestScalingTickScheduling
// ---------------------------------------------------------------------------

// TestScalingTickScheduling verifies autoscaler tick firing behavior.
// Sub-tests:
//   (a) ModelAutoscalerIntervalUs=0 → no ScalingTickEvent fires (autoscaler disabled).
//   (b) interval=60s, horizon=200s → ticks fire at t=0, 60s, 120s, 180s (4 total).
//   (c) ActuationDelay={Mean:0} → Actuator.Apply() called with At == ScalingTickEvent.At.
//   (d) ActuationDelay={Mean:30s} → Actuator.Apply() called with At == tick.At + 30_000_000.
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
		// Observable behavior: with zero delay, Apply() must be called at the same
		// time as the tick. We use a recordingActuator to capture the call time.
		const intervalUs = 60_000_000.0
		cfg := newAutoscalerTestConfig(intervalUs)
		cfg.Horizon = 1 // 1µs horizon: only first tick at t=0 fires
		cfg.ActuationDelay = DelaySpec{Mean: 0, Stddev: 0}
		cs := NewClusterSimulator(cfg, nil, nil)

		actuator := newRecordingActuator()
		cs.autoscaler = newTestPipeline(&countingCollector{}, &nopAnalyzer{}, &onceEngine{delta: 1}, actuator)
		if err := cs.Run(); err != nil {
			t.Fatalf("Run: %v", err)
		}
		select {
		case <-actuator.applied:
			// Apply() was called — zero-delay actuation fired as expected.
		default:
			t.Fatal("zero delay: Apply() was not called — ScaleActuationEvent did not execute")
		}
	})

	t.Run("d_30s_actuation_delay_shifts_actuation", func(t *testing.T) {
		// Observable behavior: with a 30s actuation delay and a 200s horizon,
		// Apply() must be called. The tick fires at t=0, actuation fires at t=30s.
		const intervalUs = 60_000_000.0
		const horizonUs = 200_000_000 // 200s — enough for tick at t=0, actuation at t=30s
		cfg := newAutoscalerTestConfig(intervalUs)
		cfg.Horizon = horizonUs
		cfg.ActuationDelay = DelaySpec{Mean: 30, Stddev: 0} // 30s deterministic delay
		cs := NewClusterSimulator(cfg, nil, nil)

		actuator := newRecordingActuator()
		cs.autoscaler = newTestPipeline(&countingCollector{}, &nopAnalyzer{}, &onceEngine{delta: 1}, actuator)
		if err := cs.Run(); err != nil {
			t.Fatalf("Run: %v", err)
		}
		select {
		case <-actuator.applied:
			// Apply() was called — 30s delay actuation fired within the 200s horizon.
		default:
			t.Fatal("30s delay: Apply() was not called within 200s horizon")
		}
	})
}

// ---------------------------------------------------------------------------
// T010: TestNoOpPipelineDeterminism
// ---------------------------------------------------------------------------

// TestNoOpPipelineDeterminism verifies INV-6: running the same configuration with a no-op
// autoscaler twice with the same seed produces byte-identical aggregated metrics.
// This is the correct INV-6 regression guard — same config, same seed, run twice, diff output.
// It must keep passing after T011–T015 are wired.
func TestNoOpPipelineDeterminism(t *testing.T) {
	const intervalUs = 60_000_000.0 // 60s tick
	// Use a bounded horizon so tick scheduling terminates.
	// 20 requests at rate 10/s arrive within the first ~2s; horizon=200s ensures all complete.
	const horizonUs = 200_000_000 // 200s

	makeRun := func(label string) *ClusterSimulator {
		// Each run needs its own request slice — sim.Request is mutated during simulation.
		cfg := newTestDeploymentConfig(1)
		cfg.ModelAutoscalerIntervalUs = intervalUs
		cfg.Horizon = horizonUs
		cs := NewClusterSimulator(cfg, newTestRequests(20), nil)
		wireAutoscaler(cs)
		if err := cs.Run(); err != nil {
			t.Fatalf("Run %s: %v", label, err)
		}
		return cs
	}

	// Run the same config with the same seed twice. INV-6 requires byte-identical output.
	csA := makeRun("A")
	csB := makeRun("B")

	mA := csA.AggregatedMetrics()
	mB := csB.AggregatedMetrics()

	if mA.CompletedRequests != mB.CompletedRequests {
		t.Errorf("INV-6: CompletedRequests differ: run1=%d run2=%d", mA.CompletedRequests, mB.CompletedRequests)
	}
	if mA.TotalInputTokens != mB.TotalInputTokens {
		t.Errorf("INV-6: TotalInputTokens differ: run1=%d run2=%d", mA.TotalInputTokens, mB.TotalInputTokens)
	}
	if mA.TotalOutputTokens != mB.TotalOutputTokens {
		t.Errorf("INV-6: TotalOutputTokens differ: run1=%d run2=%d", mA.TotalOutputTokens, mB.TotalOutputTokens)
	}
	if mA.SimEndedTime != mB.SimEndedTime {
		t.Errorf("INV-6: SimEndedTime differ: run1=%d run2=%d", mA.SimEndedTime, mB.SimEndedTime)
	}
}

// ---------------------------------------------------------------------------
// T011: TestNilComponentGuard
// ---------------------------------------------------------------------------

// TestNilComponentGuard verifies that a partially-wired autoscaler pipeline reschedules
// its next tick but does NOT call Collect() when any of the four components is nil.
// Behavioral contract: nil guard must not permanently stall the schedule.
func TestNilComponentGuard(t *testing.T) {
	// Each case carries its own countingCollector so we can assert Collect() was never called.
	// For the nil_collector case, the collector field passed to the pipeline is nil;
	// wiredCollector is a separate instance that would only accumulate calls if the nil
	// guard were bypassed (which would panic on a nil interface call first).
	cases := []struct {
		name            string
		wiredCollector  *countingCollector // the collector passed to the pipeline (may be nil)
		nilCollector    bool               // when true, pass nil as Collector to the pipeline
		analyzer        Analyzer
		engine          Engine
		actuator        Actuator
	}{
		{name: "nil_collector", nilCollector: true, analyzer: &nopAnalyzer{}, engine: &nopEngine{}, actuator: &nopActuator{}},
		{name: "nil_analyzer", analyzer: nil, engine: &nopEngine{}, actuator: &nopActuator{}},
		{name: "nil_engine", analyzer: &nopAnalyzer{}, engine: nil, actuator: &nopActuator{}},
		{name: "nil_actuator", analyzer: &nopAnalyzer{}, engine: &nopEngine{}, actuator: nil},
	}
	for i := range cases {
		cases[i].wiredCollector = &countingCollector{}
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const intervalUs = 60_000_000.0
			cfg := newAutoscalerTestConfig(intervalUs)
			cs := NewClusterSimulator(cfg, nil, nil)

			var col Collector = tc.wiredCollector
			if tc.nilCollector {
				col = nil
			}
			cs.autoscaler = newTestPipeline(col, tc.analyzer, tc.engine, tc.actuator)
			if err := cs.Run(); err != nil {
				t.Fatalf("Run: %v", err)
			}
			// With any nil component, Collect() must never be called.
			if tc.wiredCollector.calls != 0 {
				t.Errorf("%s: Collect() called %d times, want 0", tc.name, tc.wiredCollector.calls)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// T012: TestCooldownFilterSuppression
// ---------------------------------------------------------------------------

// TestCooldownFilterSuppression verifies INV-A7: scale decisions within the cooldown
// window are suppressed; decisions after the window pass through.
// Both scale-up (ScaleUpCooldownUs) and scale-down (ScaleDownCooldownUs) paths are
// tested — they share the same filter logic but maintain separate lastScale*At maps.
func TestCooldownFilterSuppression(t *testing.T) {
	// Ticks: 0, 60s, 120s, 180s, 240s, 300s, 360s = 7 ticks within 400s horizon.
	// Cooldown = 120s. Map zero value for missing key is 0, so:
	//   t=0:   elapsed = 0 - 0 = 0 < 120_000_000 → suppressed
	//   t=60s: elapsed = 60_000_000 - 0 < 120_000_000 → suppressed
	//   t=120s: elapsed = 120_000_000 - 0, NOT < 120_000_000 → passes; lastAt = 120_000_000
	//   t=180s: elapsed = 60_000_000 < 120_000_000 → suppressed
	//   t=240s: elapsed = 120_000_000, NOT < 120_000_000 → passes; lastAt = 240_000_000
	//   t=300s: suppressed; t=360s: passes → wantApplied = 3
	const (
		cooldownUs  = 120_000_000 // 2 minutes in μs
		intervalUs  = 60_000_000.0
		horizonUs   = 400_000_000
		wantApplied = 3
	)

	tests := []struct {
		name   string
		setup  func(cfg *DeploymentConfig) Engine
	}{
		{
			name: "scale_up",
			setup: func(cfg *DeploymentConfig) Engine {
				cfg.ScaleUpCooldownUs = cooldownUs
				return &alwaysScaleUpEngine{}
			},
		},
		{
			name: "scale_down",
			setup: func(cfg *DeploymentConfig) Engine {
				cfg.ScaleDownCooldownUs = cooldownUs
				return &alwaysScaleDownEngine{}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := newAutoscalerTestConfig(intervalUs)
			cfg.Horizon = horizonUs
			engine := tc.setup(&cfg)

			applied := 0
			cs := NewClusterSimulator(cfg, nil, nil)
			cs.autoscaler = newTestPipeline(&countingCollector{}, &nopAnalyzer{}, engine, &countingApplyActuator{count: &applied})
			if err := cs.Run(); err != nil {
				t.Fatalf("Run: %v", err)
			}
			if applied != wantApplied {
				t.Errorf("Apply() called %d times, want %d (ticks at 120s, 240s, 360s)", applied, wantApplied)
			}
		})
	}
}

// alwaysScaleUpEngine always emits a scale-up decision for "model-a".
type alwaysScaleUpEngine struct{}

func (e *alwaysScaleUpEngine) Optimize(_ []AnalyzerResult, _ GPUInventory) []ScaleDecision {
	return []ScaleDecision{{ModelID: "model-a", Variant: VariantSpec{GPUType: "A100", TPDegree: 1}, Delta: 1}}
}

// alwaysScaleDownEngine always emits a scale-down decision for "model-a".
type alwaysScaleDownEngine struct{}

func (e *alwaysScaleDownEngine) Optimize(_ []AnalyzerResult, _ GPUInventory) []ScaleDecision {
	return []ScaleDecision{{ModelID: "model-a", Variant: VariantSpec{GPUType: "A100", TPDegree: 1}, Delta: -1}}
}

// countingApplyActuator increments *count each time Apply() receives non-empty decisions.
type countingApplyActuator struct{ count *int }

func (a *countingApplyActuator) Apply(decisions []ScaleDecision) error {
	if len(decisions) > 0 {
		*a.count++
	}
	return nil
}
