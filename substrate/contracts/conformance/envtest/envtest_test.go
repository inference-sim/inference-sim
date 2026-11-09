package envtest

import (
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/engine/env"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/units"
)

func reference() *Reference {
	return &Reference{
		Graphs:   map[string]*model.Graph{"m": {}},
		Chips:    map[string]*hardware.Chip{"h100": {}},
		Profiles: map[[3]string]acceptance.Profile{{"m", "d", "w"}: {Model: "m"}},
		StepBase: 100 * time.Microsecond, PerToken: 2 * time.Microsecond,
		ActivationPerToken: 1000, BytesPerSecond: 1e9, BytesPerToken: 4096,
	}
}

func goodCase() Case {
	p := plan.PoolPlan{
		Hardware: "h100",
		Layout:   plan.Layout{Tensor: 2, RanksPerNode: 8, Nodes: 1},
		Layers:   []plan.LoweredLayer{{Primitive: plan.PrimGEMM, KernelImpl: plan.KernelImpl{Library: "lib"}}},
		KV:       []plan.KVGroup{{BytesPerPage: 4096, TokensPerPage: 1}},
		Weights:  []plan.ResidentTensor{{Name: "w", Elements: 500, Format: units.QuantFormat{PayloadBits: 16}}},
	}
	step := plan.StepShape{
		Mode:   plan.StepMixed,
		Ranks:  []plan.RankStep{{Requests: []plan.RequestStep{{Scheduled: 8, Prompt: 8}}}},
		Padded: []plan.PaddedShape{{Tokens: 8, Launch: plan.LaunchGraph}},
	}
	return Case{Name: "good", Plan: p, Steps: []plan.StepShape{step}, Transfer: plan.Transfer{Tokens: 10, Kind: plan.TransferPeer}}
}

func TestTheReferenceKeepsTheEnvironmentContract(t *testing.T) {
	Check(t, reference(), []Case{goodCase()})
}

func TestTheReferencePricesWhatItSays(t *testing.T) {
	r, c := reference(), goodCase()
	est, err := r.Price(c.Plan, c.Steps[0])
	if err != nil || est.NoOverlap != 100*time.Microsecond+8*2*time.Microsecond {
		t.Fatalf("Price = %+v, %v", est, err)
	}
	m, err := r.Memory(c.Plan)
	if err != nil || m.Weights != 1000 || m.ActivationPeak != 2000 {
		t.Fatalf("Memory = %+v, %v", m, err)
	}
	d, err := r.TransferDemand(c.Plan, c.Transfer)
	if err != nil || d.Bytes != 40960 || len(d.Links) != 1 {
		t.Fatalf("TransferDemand = %+v, %v", d, err)
	}
	if _, err := r.Graph("none"); err == nil {
		t.Error("an unknown model was returned")
	}
	if _, err := r.Chip("none"); err == nil {
		t.Error("an unknown chip was returned")
	}
	if _, err := r.Acceptance("m", "d", "other"); err == nil {
		t.Error("an unknown acceptance key was returned")
	}
}

// X18: a fake that breaks each law fails Check. Each fake wraps the reference and breaks one.
type impure struct {
	*Reference
	n int
}

func (i *impure) Price(p plan.PoolPlan, s plan.StepShape) (kernel.StepEstimate, error) {
	i.n++
	e, err := i.Reference.Price(p, s)
	e.NoOverlap += time.Duration(i.n)
	return e, err
}

type cheaperWhenPadded struct{ *Reference }

func (c cheaperWhenPadded) Price(p plan.PoolPlan, s plan.StepShape) (kernel.StepEstimate, error) {
	e, err := c.Reference.Price(p, s)
	if len(s.Padded) > 0 && s.Padded[0].Tokens > 40 {
		e.NoOverlap = 0
	}
	return e, err
}

type pricesAnything struct{ *Reference }

func (pricesAnything) Price(plan.PoolPlan, plan.StepShape) (kernel.StepEstimate, error) {
	return kernel.StepEstimate{NoOverlap: time.Microsecond}, nil
}

type emptyDemand struct{ *Reference }

func (emptyDemand) TransferDemand(plan.PoolPlan, plan.Transfer) (kernel.TransferDemand, error) {
	return kernel.TransferDemand{}, nil
}

type sizesAnything struct{ *Reference }

func (sizesAnything) Memory(plan.PoolPlan) (kernel.MemoryBreakdown, error) {
	return kernel.MemoryBreakdown{}, nil
}

type recorder struct {
	testing.TB
	failed bool
}

func (r *recorder) Errorf(string, ...any) { r.failed = true }
func (r *recorder) Fatalf(string, ...any) { r.failed = true }
func (r *recorder) Fatal(...any)          { r.failed = true }
func (r *recorder) Helper()               {}

func TestAnEnvironmentThatBreaksALawFails(t *testing.T) {
	for name, e := range map[string]env.Environment{
		"impure pricing":                &impure{Reference: reference()},
		"padding that lowers the price": cheaperWhenPadded{reference()},
		"pricing a refused plan":        pricesAnything{reference()},
		"a transfer with no demand":     emptyDemand{reference()},
		"sizing a refused plan":         sizesAnything{reference()},
	} {
		rec := &recorder{TB: t}
		Check(rec, e, []Case{goodCase()})
		if !rec.failed {
			t.Errorf("%s passed Check", name)
		}
	}
	rec := &recorder{TB: t}
	Check(rec, reference(), nil)
	if !rec.failed {
		t.Error("no cases passed Check; zero cases is not a pass (X22)")
	}
}
