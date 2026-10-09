package sim

import (
	"math"
	"testing"
)

type fixedFactor float64

func (f fixedFactor) LoadLatency(string) float64            { return 0 }
func (f fixedFactor) StepOverheadFactor([]*Request) float64 { return float64(f) }
func (f fixedFactor) AdapterReservedBytes() float64         { return 0 }

// The adapter wrapper multiplies a step by the batch's overhead factor and touches nothing
// else; a nil accessor, or a factor that is not finite or not above one, leaves the price as
// it was, so a run without adapters is byte-identical (INV-6).
func TestWithAdapterOverhead_ScalesOnlyTheStep(t *testing.T) {
	base := &fixedStepModel{stepTime: 1000}
	if WithAdapterOverhead(base, nil) != LatencyModel(base) {
		t.Error("a nil accessor wrapped the model")
	}
	for _, c := range []struct {
		factor float64
		want   int64
	}{{1.5, 1500}, {1.0, 1000}, {0.5, 1000}, {math.NaN(), 1000}, {math.Inf(1), 1000}} {
		m := WithAdapterOverhead(base, fixedFactor(c.factor))
		if got := m.StepTime(nil); got != c.want {
			t.Errorf("factor %v: step %d, want %d", c.factor, got, c.want)
		}
		if m.OutputTokenProcessingTime() != base.OutputTokenProcessingTime() ||
			m.PostDecodeFixedOverhead() != base.PostDecodeFixedOverhead() {
			t.Errorf("factor %v: the wrapper changed a non-step cost", c.factor)
		}
	}
}
