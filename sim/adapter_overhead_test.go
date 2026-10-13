package sim

import (
	"math"
	"testing"
)

type fixedFactor float64

func (f fixedFactor) LoadLatency(string) float64            { return 0 }
func (f fixedFactor) StepOverheadFactor([]*Request) float64 { return float64(f) }
func (f fixedFactor) AdapterReservedBytes() float64         { return 0 }

// The adapter wrapper multiplies a step by the batch's overhead factor and leaves every other
// cost alone; a nil accessor returns the model itself, and a factor of exactly one (no
// adapters in the batch) leaves the price as it was, so a run without adapters is
// byte-identical (INV-6). An accessor breaking its contract -- a factor below one or not
// finite -- is a bug, and panics rather than being read as "no overhead".
func TestWithAdapterOverhead_ScalesOnlyTheStep(t *testing.T) {
	base := &fixedStepModel{stepTime: 1000}
	if WithAdapterOverhead(base, nil) != LatencyModel(base) {
		t.Error("a nil accessor wrapped the model")
	}
	req := &Request{InputTokens: make([]TokenID, 8)}
	for _, c := range []struct {
		factor float64
		want   int64
	}{{1.5, 1500}, {1.0, 1000}, {2.0, 2000}} {
		m := WithAdapterOverhead(base, fixedFactor(c.factor))
		if got := m.StepTime(nil); got != c.want {
			t.Errorf("factor %v: step %d, want %d", c.factor, got, c.want)
		}
		if m.QueueingTime(req) != base.QueueingTime(req) ||
			m.OutputTokenProcessingTime() != base.OutputTokenProcessingTime() ||
			m.PostDecodeFixedOverhead() != base.PostDecodeFixedOverhead() {
			t.Errorf("factor %v: the wrapper changed a non-step cost", c.factor)
		}
	}
	for _, bad := range []float64{0.5, math.NaN(), math.Inf(1), math.Inf(-1)} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("factor %v broke the contract and was accepted", bad)
				}
			}()
			WithAdapterOverhead(base, fixedFactor(bad)).StepTime(nil)
		}()
	}
}
