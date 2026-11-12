package shapetest

import (
	"errors"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
)

// fake is a kernel with settable answers, to show each check fails on the input it exists for.
type fake struct {
	kernel.Kernel
	fixed  kernel.MemoryBreakdown
	seqFix int64
	varBy  func(int) int64
	step   kernel.StepEstimate
	admit  time.Duration
}

func (f fake) FixedBytes() kernel.MemoryBreakdown { return f.fixed }
func (f fake) SequenceFixedBytes() int64          { return f.seqFix }
func (f fake) SequenceVariableBytes(n int) int64  { return f.varBy(n) }
func (f fake) StepTime(kernel.Batch) kernel.StepEstimate {
	return f.step
}
func (f fake) AdmissionOverhead(int) time.Duration { return f.admit }
func (f fake) OutputTokenOverhead() time.Duration  { return 0 }
func (f fake) CompletionOverhead() time.Duration   { return 0 }

func good() fake {
	return fake{
		fixed: kernel.MemoryBreakdown{Weights: 1 << 30},
		varBy: func(n int) int64 { return int64(n) * 4096 },
		step:  kernel.StepEstimate{Overlap: time.Millisecond, NoOverlap: 2 * time.Millisecond},
		admit: time.Microsecond,
	}
}

// Every shape is a valid graph: a shape that fails Validate tests nothing about the engine.
func TestEveryShapeIsAValidGraph(t *testing.T) {
	if got := len(Shapes()); got != 6 {
		t.Fatalf("%d shapes, want the six of SUB-10", got)
	}
	names := map[string]bool{}
	for _, s := range Shapes() {
		if err := s.Validate(); err != nil {
			t.Error(err)
		}
		if names[s.Name] {
			t.Errorf("duplicate shape %q", s.Name)
		}
		names[s.Name] = true
		if s.Why == "" {
			t.Errorf("%s: says nothing about what it tests", s.Name)
		}
	}
}

// The shapes are structurally what their names claim, and CacheOf reads each from its graph.
func TestTheShapesAreWhatTheyClaim(t *testing.T) {
	want := map[string]Cache{
		"every layer windowed": CacheBounded, "every layer recurrent": CacheState,
		"MLA only": CacheGrowing, "a single layer": CacheGrowing,
		"no attention at all": CacheNone, "a draft layer declared last": CacheGrowing,
	}
	by := map[string]Shape{}
	for _, s := range Shapes() {
		by[s.Name] = s
		if got := CacheOf(s.Graph); got != want[s.Name] {
			t.Errorf("%s: CacheOf = %v, want %v", s.Name, got, want[s.Name])
		}
	}
	if n := by["a single layer"].Graph.Stack.Layers(); n != 1 {
		t.Errorf("single-layer shape has %d layers", n)
	}
	d := by["a draft layer declared last"].Graph
	if d.Speculator == nil || d.LayerKinds[len(d.LayerKinds)-1].ID != d.Speculator.Stack.Pattern[0] {
		t.Error("the draft shape's speculator layer must be the last layer kind declared")
	}
}

func TestCheckSizesAndPricesAcceptAGoodKernelAndFailEachWay(t *testing.T) {
	if err := CheckSizes(good(), CacheGrowing); err != nil {
		t.Fatalf("a good kernel failed: %v", err)
	}
	if err := CheckPrices(good()); err != nil {
		t.Fatalf("a good kernel failed: %v", err)
	}
	mut := func(f func(*fake)) fake { k := good(); f(&k); return k }
	sizeBad := map[string]fake{
		"negative weights":                   mut(func(k *fake) { k.fixed.Weights = -1 }),
		"zero weights":                       mut(func(k *fake) { k.fixed.Weights = 0 }),
		"negative graph pool":                mut(func(k *fake) { k.fixed.CUDAGraph = -5 }),
		"negative seq state":                 mut(func(k *fake) { k.seqFix = -1 }),
		"shrinking KV":                       mut(func(k *fake) { k.varBy = func(n int) int64 { return int64(1e9 - n) } }),
		"a growing cache that does not grow": mut(func(k *fake) { k.varBy = func(int) int64 { return 4096 } }),
	}
	for name, k := range sizeBad {
		if CheckSizes(k, CacheGrowing) == nil {
			t.Errorf("sizes: %s accepted", name)
		}
	}
	if CheckSizes(good(), CacheNone) == nil {
		t.Error("sizes: a shape with no cache that reports KV bytes was accepted")
	}
	if CheckSizes(good(), CacheBounded) == nil {
		t.Error("sizes: a windowed shape whose cache grows with context was accepted")
	}
	zero := good()
	zero.varBy = func(int) int64 { return 0 }
	for _, c := range []Cache{CacheState, CacheBounded} {
		if CheckSizes(zero, c) == nil {
			t.Errorf("sizes: a %v shape that holds nothing was accepted", c)
		}
	}
	if err := CheckSizes(zero, CacheNone); err != nil {
		t.Errorf("sizes: a shape with no cache that holds nothing was rejected: %v", err)
	}
	state := zero
	state.seqFix = 4096
	if err := CheckSizes(state, CacheState); err != nil {
		t.Errorf("sizes: a state-only shape with state was rejected: %v", err)
	}
	if CheckSizes(state, CacheNone) == nil {
		t.Error("sizes: a shape with no cache that reports state bytes was accepted")
	}
	priceBad := map[string]fake{
		"zero overlap":             mut(func(k *fake) { k.step.Overlap = 0 }),
		"negative no-overlap":      mut(func(k *fake) { k.step.NoOverlap = -1 }),
		"absurd step":              mut(func(k *fake) { k.step.NoOverlap = 2 * time.Hour }),
		"overlap above no-overlap": mut(func(k *fake) { k.step.Overlap = 3 * time.Millisecond }),
		"negative resource":        mut(func(k *fake) { k.step.PerResource = map[kernel.Resource]time.Duration{kernel.ResourceSM: -1} }),
		"negative host cost":       mut(func(k *fake) { k.admit = -1 }),
	}
	for name, k := range priceBad {
		if CheckPrices(k) == nil {
			t.Errorf("prices: %s accepted", name)
		}
	}
}

// Run: a shape is run or refused, never both and never neither, and a refusal states a reason.
func TestRunHoldsAnImplementationToRunOrRefuse(t *testing.T) {
	var fails int
	// A kernel that reports a KV cache for a shape that has none (and none for ones that do) is
	// caught on the shapes where it is wrong: this stub reports KV bytes for every shape.
	Run(&recorder{onError: func() { fails++ }}, func(Shape) Result { return Result{Kernel: good()} })
	// good() grows with context and has no fixed state. That is right for the three growing
	// shapes, and passes the state-only shape (it holds something). It is wrong for the
	// windowed shape, whose cache must stop growing, and for the shape with no cache at all.
	if fails != 2 {
		t.Errorf("a stub that grows with context on every shape failed %d times, want 2", fails)
	}
	for name, r := range map[string]Result{
		"both":            {Kernel: good(), Refusal: errors.New("x")},
		"neither":         {},
		"an empty reason": {Refusal: errors.New("")},
	} {
		fails = 0
		Run(&recorder{onError: func() { fails++ }}, func(Shape) Result { return r })
		if fails != len(Shapes()) {
			t.Errorf("%s: %d failures over %d shapes", name, fails, len(Shapes()))
		}
	}
	fails = 0
	Run(&recorder{onError: func() { fails++ }}, func(Shape) Result { return Result{Refusal: errors.New("this shape is not supported: reason")} })
	if fails != 0 {
		t.Errorf("a stated refusal of every shape failed %d times", fails)
	}
}

type recorder struct {
	testing.TB
	onError func()
}

func (r *recorder) Helper()                       {}
func (r *recorder) Errorf(string, ...interface{}) { r.onError() }
func (r *recorder) Fatal(...interface{})          { r.onError() }
