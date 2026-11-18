package sim

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

// The driver reports where the engine said a held handoff sits (engine.Handoff.At: the rank of a
// data-parallel group that computed the KV), nil when the engine stated nothing or holds none.
func TestEngineDriver_HandoffAtIsWhatTheEngineStated(t *testing.T) {
	eng, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 1, TokenBudget: 64, StepBaseUs: 100})
	if err != nil {
		t.Fatal(err)
	}
	d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 1000})
	at := plan.Endpoint{Node: 2, Rack: 1}
	d.noteHandoffs([]engine.Handoff{{ID: "owned", At: &at}, {ID: "plain"}})
	if got := d.HandoffAt(&Request{ID: "owned"}); got == nil || *got != at {
		t.Errorf("HandoffAt(owned) = %v, want %+v", got, at)
	}
	if got := d.HandoffAt(&Request{ID: "plain"}); got != nil {
		t.Errorf("HandoffAt(plain) = %+v, want nil: the engine stated nothing", got)
	}
	if got := d.HandoffAt(&Request{ID: "none"}); got != nil {
		t.Errorf("HandoffAt(none) = %+v, want nil: the engine holds no handoff for it", got)
	}
}
