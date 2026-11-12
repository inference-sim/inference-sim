package envtest

import (
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/engine"
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

// tierCase is goodCase with a move to a tier the plan lists.
func tierCase() Case {
	c := goodCase()
	c.Name = "tier"
	c.Plan.Tiers = []plan.KVTier{{Device: "cpu_dram"}}
	c.Transfer = plan.Transfer{Tokens: 10, Kind: plan.TransferToTier, Tier: "cpu_dram"}
	return c
}

func TestTheReferenceKeepsTheEnvironmentContract(t *testing.T) {
	Check(t, reference(), []Case{goodCase(), tierCase()})
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

// ignoresThePlan demands a move to any tier, as an Environment that checks the move alone does.
type ignoresThePlan struct{ *Reference }

func (i ignoresThePlan) TransferDemand(p plan.PoolPlan, t plan.Transfer) (kernel.TransferDemand, error) {
	if err := t.Validate(); err != nil {
		return kernel.TransferDemand{}, err
	}
	return i.Reference.TransferDemand(plan.PoolPlan{Tiers: []plan.KVTier{{Device: t.Tier}}}, t)
}

// chargesForNothing bills a pull that moves nothing, as an Environment that divides or adds a floor
// without looking at the payload would.
type chargesForNothing struct{ *Reference }

func (c chargesForNothing) TransferDemand(p plan.PoolPlan, t plan.Transfer) (kernel.TransferDemand, error) {
	if t.Tokens == 0 {
		return kernel.TransferDemand{Bytes: 4096, Links: []kernel.LinkUse{{Link: "ref", BytesPerSecond: 1e9}}}, nil
	}
	return c.Reference.TransferDemand(p, t)
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
		"impure pricing":                 &impure{Reference: reference()},
		"padding that lowers the price":  cheaperWhenPadded{reference()},
		"pricing a refused plan":         pricesAnything{reference()},
		"a transfer with no demand":      emptyDemand{reference()},
		"sizing a refused plan":          sizesAnything{reference()},
		"a tier the plan does not list":  ignoresThePlan{reference()},
		"a charge for a pull of nothing": chargesForNothing{reference()},
		"a charge for no tokens":         admitsNothingForFree{reference()},
	} {
		rec := &recorder{TB: t}
		Check(rec, e, []Case{goodCase(), tierCase()})
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

// recordingEnv remembers what TransferDemand was asked, to show the simulator's side hands the
// Environment exactly the engine's Move and the engine's plan.
type recordingEnv struct {
	*Reference
	plans []plan.PoolPlan
	moves []plan.Transfer
}

func (r *recordingEnv) TransferDemand(p plan.PoolPlan, t plan.Transfer) (kernel.TransferDemand, error) {
	r.plans, r.moves = append(r.plans, p), append(r.moves, t)
	return r.Reference.TransferDemand(p, t)
}

func loadsRequest() []enginetest.Arrival {
	r := engine.Request{ID: "loads", Prompt: []int32{1, 2, 3, 4}, Output: []int32{9}, Tags: map[string]string{"fifo.load_tokens": "4"}}
	return []enginetest.Arrival{{Req: r}}
}

func fifo() (engine.Engine, error) {
	return enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 2, TokenBudget: 8, StepBaseUs: 10, StepPerTokenUs: 1})
}

// The finding: an engine's Batch.TransfersStarted and Environment.TransferDemand were two
// contracts nothing joined, so each could pass its own tests. This test joins them. A transfer the
// reference FIFO engine returns is taken, with the plan the engine states, through env.DemandFor
// to a demand, and the Environment is asked about the engine's Move and nothing else.
func TestATransferTheEngineReturnsReachesADemand(t *testing.T) {
	e := &recordingEnv{Reference: reference()}
	enginetest.TestEngine(t, fifo, loadsRequest(), enginetest.Options{Boundary: Boundary(e)})
	if len(e.moves) == 0 {
		t.Fatal("the Environment was asked about no transfer; the check processes at least one (X22)")
	}
	f, _ := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 1, TokenBudget: 1})
	for i, m := range e.moves {
		want := plan.Transfer{Tokens: 4, Kind: plan.TransferFromTier, Tier: enginetest.FIFOTierDevice}
		if m != want {
			t.Errorf("TransferDemand received %+v, want the engine's Move %+v", m, want)
		}
		if got := e.plans[i]; len(got.Tiers) != 1 || got.Tiers[0] != f.Plan().Tiers[0] {
			t.Errorf("TransferDemand received plan %+v, want the engine's", got)
		}
	}
	d, err := e.Reference.TransferDemand(f.Plan(), e.moves[0])
	if err != nil || d.Bytes != 4*4096 {
		t.Errorf("demand %+v, %v: want 4 tokens of 4096 bytes", d, err)
	}
}

// X18: engines that compile against both contracts and are not connected. Each is a FIFO
// that reports a transfer the Environment cannot demand; each fails the same test.
type lostMove struct{ *enginetest.FIFO }

func (e lostMove) Schedule(now engine.Time) (engine.Batch, error) {
	b, err := e.FIFO.Schedule(now)
	for i := range b.TransfersStarted {
		b.TransfersStarted[i].Move = plan.Transfer{}
	}
	return b, err
}

type namesItsOwnTier struct{ *enginetest.FIFO }

func (e namesItsOwnTier) Schedule(now engine.Time) (engine.Batch, error) {
	b, err := e.FIFO.Schedule(now)
	for i := range b.TransfersStarted {
		b.TransfersStarted[i].Move.Tier = b.TransfersStarted[i].Tier
	}
	return b, err
}

type noPlan struct{ engine.Engine }

func TestAnEngineWhoseTransfersCannotBeDemandedFails(t *testing.T) {
	wrap := map[string]func(*enginetest.FIFO) engine.Engine{
		"a transfer with no Move":                 func(f *enginetest.FIFO) engine.Engine { return lostMove{f} },
		"a Move naming the engine's own tier":     func(f *enginetest.FIFO) engine.Engine { return namesItsOwnTier{f} },
		"transfers and no plan to demand them by": func(f *enginetest.FIFO) engine.Engine { return noPlan{f} },
	}
	for name, w := range wrap {
		_, err := enginetest.Drive(func() (engine.Engine, error) {
			f, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 2, TokenBudget: 8, StepBaseUs: 10, StepPerTokenUs: 1})
			if err != nil {
				return nil, err
			}
			return w(f), nil
		}, loadsRequest(), enginetest.Options{Boundary: Boundary(reference())})
		if err == nil {
			t.Errorf("%s was driven to completion", name)
		}
	}
}

// An Environment that prices a Move its plan cannot make, or that is given a report that
// disagrees with its Move, is not reached: DemandFor refuses first.
func TestDemandForRefusesWhatTheEngineDidNotConnect(t *testing.T) {
	r := &recordingEnv{Reference: reference()}
	f, _ := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 1, TokenBudget: 1})
	p := f.Plan()
	good := engine.Transfer{ID: 1, Request: "q", Kind: engine.TransferLoad, Tokens: 4, Tier: enginetest.FIFOTier,
		Move: plan.Transfer{Tokens: 4, Kind: plan.TransferFromTier, Tier: enginetest.FIFOTierDevice}}
	if _, err := env.DemandFor(r, p, good); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*engine.Transfer){
		"zero Move":             func(x *engine.Transfer) { x.Move = plan.Transfer{} },
		"unlisted tier":         func(x *engine.Transfer) { x.Move.Tier = "nvme_gen4" },
		"report and Move split": func(x *engine.Transfer) { x.Tokens = 5 },
	} {
		bad := good
		mutate(&bad)
		before := len(r.moves)
		if _, err := env.DemandFor(r, p, bad); err == nil {
			t.Errorf("%s: DemandFor returned a demand", name)
		}
		if len(r.moves) != before {
			t.Errorf("%s: the Environment was asked to price a transfer DemandFor should have refused", name)
		}
	}
}

// A pull that moves nothing reaches a demand of zero bytes through the same boundary, and a tier
// move of nothing is refused by DemandFor before the Environment is asked.
func TestAPullOfNothingDemandsZeroBytes(t *testing.T) {
	r := &recordingEnv{Reference: reference()}
	f, _ := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 1, TokenBudget: 1})
	empty := engine.Transfer{ID: 1, Request: "q", Kind: engine.TransferLoad,
		Move: plan.Transfer{Kind: plan.TransferPeer, From: plan.Endpoint{Node: 1}}}
	d, err := env.DemandFor(r, f.Plan(), empty)
	if err != nil || d.Bytes != 0 || len(d.Links) != 0 || d.Floor != 0 {
		t.Fatalf("DemandFor = %+v, %v; want the zero demand", d, err)
	}
	if err := Boundary(r)(f.Plan(), empty); err != nil {
		t.Errorf("Boundary refused a pull of nothing: %v", err)
	}
	tier := engine.Transfer{ID: 2, Request: "q", Kind: engine.TransferLoad, Tier: enginetest.FIFOTier,
		Move: plan.Transfer{Kind: plan.TransferFromTier, Tier: enginetest.FIFOTierDevice}}
	before := len(r.moves)
	if _, err := env.DemandFor(r, f.Plan(), tier); err == nil || len(r.moves) != before {
		t.Errorf("a tier load of nothing was priced: %v", err)
	}
}

// admitsNothingForFree charges a delay even to a request of no tokens.
type admitsNothingForFree struct{ *Reference }

func (a admitsNothingForFree) HostDemand(p plan.PoolPlan, n int) (kernel.HostDemand, error) {
	return kernel.HostDemand{Delay: time.Millisecond}, nil
}

func TestTheReferenceSplitsAdmissionIntoServiceAndDelay(t *testing.T) {
	r := reference()
	r.AdmissionPerToken, r.AdmissionDelay = 350*time.Nanosecond, 40*time.Millisecond
	d, err := r.HostDemand(goodCase().Plan, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if d.Service != 350*time.Microsecond || d.Delay != 40*time.Millisecond {
		t.Fatalf("got %+v", d)
	}
}
