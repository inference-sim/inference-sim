package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// countingModel is a stateless latency model that counts how often it priced a step, so a
// test can tell which pool's model served which work.
type countingModel struct {
	step  int64
	calls *int
}

func (m countingModel) StepTime([]*sim.Request) int64    { *m.calls++; return m.step }
func (m countingModel) QueueingTime(*sim.Request) int64  { return 1 }
func (m countingModel) OutputTokenProcessingTime() int64 { return 0 }
func (m countingModel) PostDecodeFixedOverhead() int64   { return 0 }

// An injected transfer price replaces the formula: every completed handoff takes exactly what
// the pricer returned, for the blocks' token capacity, from the prefill instance the request
// ran on to the decode instance it was handed to. The simulator owns when; the backend owns
// how long.
func TestPDTransfer_AnInjectedPriceIsTheTransferTime(t *testing.T) {
	cfg := newTestDisaggDeploymentConfig(4, 2, 2)
	type call struct {
		tokens   int64
		from, to InstanceID
	}
	var calls []call
	cfg.PDTransferTime = func(tokens int64, from, to InstanceID) int64 {
		calls = append(calls, call{tokens, from, to})
		return 1000 + tokens // a price no formula in this package would produce
	}
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(newTestRequests(12)), nil)
	if err := cs.Run(); err != nil {
		t.Fatal(err)
	}
	if len(calls) == 0 {
		t.Fatal("no transfer was priced; the law was checked against nothing")
	}
	checked := 0
	for _, p := range cs.ParentRequests() {
		if p.TransferCompleteTime == 0 {
			continue
		}
		want := 1000 + p.NumKVBlocks*cfg.BlockSizeTokens
		if got := p.TransferCompleteTime - p.TransferStartTime; got != want {
			t.Errorf("%s: transfer took %d ticks, the pricer said %d", p.ID, got, want)
		}
		checked++
	}
	for _, c := range calls {
		if c.tokens%cfg.BlockSizeTokens != 0 || c.from == c.to || c.from == "" || c.to == "" {
			t.Errorf("priced %+v: tokens must be whole blocks between two distinct instances", c)
		}
	}
	if checked == 0 {
		t.Fatal("no handoff completed")
	}
}

// Each role is priced by its own pool's model: in a cluster whose every instance is prefill or
// decode, the global model never prices a step, and both pool models do.
func TestPoolOverrides_EachRoleIsPricedByItsOwnModel(t *testing.T) {
	cfg := newTestDisaggDeploymentConfig(4, 2, 2)
	var global, prefill, decode int
	cfg.LatencyModelOverride = countingModel{step: 50, calls: &global}
	cfg.PrefillOverrides.LatencyModel = countingModel{step: 70, calls: &prefill}
	cfg.DecodeOverrides.LatencyModel = countingModel{step: 30, calls: &decode}
	cfg.PDTransferTime = func(int64, InstanceID, InstanceID) int64 { return 10 }
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(newTestRequests(8)), nil)
	if err := cs.Run(); err != nil {
		t.Fatal(err)
	}
	if global != 0 {
		t.Errorf("the global model priced %d steps in a fully disaggregated cluster", global)
	}
	if prefill == 0 || decode == 0 {
		t.Errorf("pool models priced prefill=%d decode=%d steps; both pools ran work", prefill, decode)
	}
}

// The fair-share contention divisor cannot be applied to a price the simulator did not
// compose, so the combination is refused at construction rather than silently mis-scaled.
func TestPDTransfer_ContentionIsRefusedWithAnInjectedPrice(t *testing.T) {
	cfg := newTestDisaggDeploymentConfig(2, 1, 1)
	cfg.PDTransferTime = func(int64, InstanceID, InstanceID) int64 { return 10 }
	cfg.PDTransferContention = true
	defer func() {
		if recover() == nil {
			t.Error("contention with an injected price was accepted")
		}
	}()
	NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
}
