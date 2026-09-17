// autoscaler_invariant_test.go — companion invariant tests for INV-19 (scale
// decisions carry a non-zero delta), promoted by #1772.
//
// INV-19 is the one entry in the enforcement-anchored set whose enforcement is a
// WARNING rather than a panic, and that choice is the substance of the invariant: a
// zero delta is inert, so skipping it leaves cluster state exactly as a correct engine
// would have, unlike the negative activeTransfers count in pd_events.go which fails
// the run. These tests pin both halves of "warn and skip": the decision does not
// actuate and does not fail the run, and — the part that is not obvious from the
// `continue` — it does not sustain a stabilization timer either.
package cluster

import (
	"testing"
)

// zeroDeltaEngine emits exactly one ScaleDecision per tick, always with Delta == 0.
// This is the engine-contract violation INV-19 describes.
type zeroDeltaEngine struct{ ticks int }

func (e *zeroDeltaEngine) Optimize(_ []AnalyzerResult, _ GPUInventory) []ScaleDecision {
	e.ticks++
	return []ScaleDecision{{ModelID: "model-a", Variant: VariantSpec{GPUType: "A100", TPDegree: 1}, Delta: 0}}
}

// zeroDeltaAtTickEngine emits a scale-up decision on every tick except zeroTick
// (0-indexed), where it emits a Delta == 0 decision instead. It is the direct analogue
// of interruptAtTickEngine (which emits NOTHING on the skipped tick) — the pair is what
// makes the timer-reset assertion discriminating rather than merely consistent.
type zeroDeltaAtTickEngine struct {
	tick     int
	zeroTick int
	delta    int
}

func (e *zeroDeltaAtTickEngine) Optimize(_ []AnalyzerResult, _ GPUInventory) []ScaleDecision {
	current := e.tick
	e.tick++
	delta := e.delta
	if current == e.zeroTick {
		delta = 0
	}
	return []ScaleDecision{{ModelID: "model-a", Variant: VariantSpec{GPUType: "A100", TPDegree: 1}, Delta: delta}}
}

// TestINV19_ZeroDeltaDecisionSkippedNotFatal asserts the warn-and-skip policy: a
// zero-delta decision reaches the pipeline on every tick, and the actuator is never
// called while the run still completes normally. The engine tick count is asserted too,
// so a run that silently stopped ticking cannot pass by producing zero actuations.
func TestINV19_ZeroDeltaDecisionSkippedNotFatal(t *testing.T) {
	const intervalUs = 60_000_000 // 60s
	cfg := newAutoscalerTestConfig(intervalUs)
	cfg.Horizon = 400_000_000 // 400s ⇒ ticks at 0,60,…,360
	cfg.ScaleUpStabilizationWindowUs = 0
	cfg.ScaleDownStabilizationWindowUs = 0

	applied := 0
	engine := &zeroDeltaEngine{}
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
	cs.autoscaler = newTestPipeline(&countingCollector{}, &nopAnalyzer{}, engine, &countingApplyActuator{count: &applied})

	if err := cs.Run(); err != nil {
		t.Fatalf("INV-19 violated: a zero-delta decision failed the run (%v) — the policy is warn-and-skip, not fatal", err)
	}
	if applied != 0 {
		t.Errorf("INV-19 violated: actuator applied %d times, want 0 — a zero-delta decision must be skipped", applied)
	}
	// Non-vacuity: the engine really was consulted, so "0 actuations" is the skip and
	// not an autoscaler that never ran.
	if engine.ticks == 0 {
		t.Fatal("engine was never consulted — the test proves nothing about the skip")
	}
}

// TestINV19_ZeroDeltaDoesNotSustainStabilizationTimer pins the consequence the entry
// calls out as non-obvious: because the skip happens before the modelsWithScaleUp set
// is consumed, a tick whose only decision was a zero delta counts as signal LOSS and
// resets the model's stabilization timer.
//
// The assertion is discriminating by construction. Against the same 400s / 60s / 120s
// window fixture, TestAutoscalerStabilizationWindow's scenario (b) — an unbroken
// scale-up stream — actuates twice, and scenario (d) — a tick emitting NOTHING —
// actuates once. Injecting a zero delta at the same tick must give the (d) answer: if
// a zero delta instead sustained the timer, this would read 2.
func TestINV19_ZeroDeltaDoesNotSustainStabilizationTimer(t *testing.T) {
	const (
		intervalUs = 60_000_000  // 60s
		horizonUs  = 400_000_000 // 400s ⇒ ticks at 0,60,120,180,240,300,360
		windowUs   = 120_000_000 // 120s
	)

	run := func(t *testing.T, engine Engine) int {
		t.Helper()
		cfg := newAutoscalerTestConfig(intervalUs)
		cfg.Horizon = horizonUs
		cfg.ScaleUpStabilizationWindowUs = windowUs

		applied := 0
		cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
		cs.autoscaler = newTestPipeline(&countingCollector{}, &nopAnalyzer{}, engine, &countingApplyActuator{count: &applied})
		if err := cs.Run(); err != nil {
			t.Fatalf("Run: %v", err)
		}
		return applied
	}

	// Baseline A: unbroken scale-up signal ⇒ 2 actuations (at 120s and 300s).
	unbroken := run(t, &alwaysScaleUpEngine{})
	// Baseline B: tick 1 emits no decision at all ⇒ timer reset ⇒ 1 actuation (at 240s).
	noDecision := run(t, &interruptAtTickEngine{skipTick: 1, delta: 1})
	// Under test: tick 1 emits a ZERO-DELTA decision instead of no decision.
	zeroDelta := run(t, &zeroDeltaAtTickEngine{zeroTick: 1, delta: 1})

	// The two baselines must differ, or the comparison below cannot discriminate.
	if unbroken == noDecision {
		t.Fatalf("fixture is not discriminating: unbroken and no-decision streams both actuated %d times", unbroken)
	}
	if zeroDelta != noDecision {
		t.Errorf("INV-19 violated: a zero-delta tick actuated %d times, want %d (same as a tick emitting NO decision) — the skipped decision must not sustain the stabilization timer; an unbroken signal gives %d",
			zeroDelta, noDecision, unbroken)
	}
}
