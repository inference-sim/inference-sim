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
// the pricer returned, for the blocks' token capacity, priced from the prefill instance the
// request ran on to the decode instance it was handed to. The simulator owns when; the backend owns
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
	membership := BuildPoolMembershipFromIndices(4, 2, 2, 0, 0)
	priced := map[call]bool{}
	for _, c := range calls {
		priced[c] = true
		if c.tokens%cfg.BlockSizeTokens != 0 {
			t.Errorf("priced %+v: tokens must be whole blocks", c)
		}
		if membership[string(c.from)] != PoolRolePrefill || membership[string(c.to)] != PoolRoleDecode {
			t.Errorf("priced %+v: a handoff runs from a prefill instance to a decode instance", c)
		}
	}
	for _, p := range cs.ParentRequests() {
		if p.TransferCompleteTime > 0 && !priced[call{p.NumKVBlocks * cfg.BlockSizeTokens, p.PrefillInstanceID, p.DecodeInstanceID}] {
			t.Errorf("%s's handoff %s -> %s was not priced between those instances", p.ID, p.PrefillInstanceID, p.DecodeInstanceID)
		}
	}
	if checked == 0 {
		t.Fatal("no handoff completed")
	}
}

// observingModel records what kind of work it was asked to price: whether any request in a
// batch was a prefill (computed short of its prompt) or a decode, and the largest batch.
type observingModel struct {
	step                  int64
	sawPrefill, sawDecode *bool
	largestBatch          *int
}

func (m observingModel) StepTime(batch []*sim.Request) int64 {
	for _, r := range batch {
		if r.ProgressIndex < r.InputLen() {
			*m.sawPrefill = true
		} else {
			*m.sawDecode = true
		}
	}
	*m.largestBatch = max(*m.largestBatch, len(batch))
	return m.step
}
func (m observingModel) QueueingTime(*sim.Request) int64  { return 1 }
func (m observingModel) OutputTokenProcessingTime() int64 { return 0 }
func (m observingModel) PostDecodeFixedOverhead() int64   { return 0 }

func newObservingModel(step int64) (observingModel, *bool, *bool, *int) {
	var p, d bool
	var n int
	return observingModel{step: step, sawPrefill: &p, sawDecode: &d, largestBatch: &n}, &p, &d, &n
}

// Each role runs its own pool's engine: the prefill pool's model only ever prices prefill
// work and the decode pool's only decode work -- so wiring either to the other's model is
// caught -- the global model prices nothing in a fully disaggregated cluster, and a pool's own
// admission cap binds its instances (a decode pool capped at one sequence never batches two).
func TestPoolOverrides_EachRoleRunsItsOwnPoolsEngine(t *testing.T) {
	cfg := newTestDisaggDeploymentConfig(4, 2, 2)
	var global int
	cfg.LatencyModelOverride = countingModel{step: 50, calls: &global}
	prefill, pPrefill, pDecode, _ := newObservingModel(70)
	decode, dPrefill, dDecode, dLargest := newObservingModel(30)
	one := int64(1)
	cfg.PrefillOverrides.LatencyModel = prefill
	cfg.DecodeOverrides.LatencyModel = decode
	cfg.DecodeOverrides.MaxNumSeqs = &one
	cfg.PDTransferTime = func(int64, InstanceID, InstanceID) int64 { return 10 }
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(newTestRequests(8)), nil)
	if err := cs.Run(); err != nil {
		t.Fatal(err)
	}
	if global != 0 {
		t.Errorf("the global model priced %d steps in a fully disaggregated cluster", global)
	}
	if !*pPrefill || *pDecode {
		t.Errorf("prefill pool's model: saw prefill=%t decode=%t; want prefill work only", *pPrefill, *pDecode)
	}
	if !*dDecode || *dPrefill {
		t.Errorf("decode pool's model: saw prefill=%t decode=%t; want decode work only", *dPrefill, *dDecode)
	}
	if *dLargest > 1 {
		t.Errorf("a decode pool capped at max_num_seqs=1 batched %d requests", *dLargest)
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
