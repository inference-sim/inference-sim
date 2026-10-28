// Moved from blis-latency-kernel/ceiling_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"math"
	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/vocab"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// A DECODE STEP DOES NOT LAUNCH THE GRAPH'S PREFILL-ONLY NODES. DeepSeek-V4.1 dequantizes and
// gathers its compressed KV, and its indexer gathers keys, only when the batch has prefills
// (vllm/models/deepseek_v41/nvidia/flashmla.py:143-165, 305, 317;
// vllm/model_executor/layers/sparse_attn_indexer.py:416-417, 481), and the graph marks those
// nodes phase: prefill. With no graph (NONE) every launch is a host charge, so adding one
// prefill row that samples nothing (no drafter pass joins it) must add exactly the eager plus
// dispatch cost of those launches, counted here from the graph -- and nothing on a graph
// that marks no node.
func TestADecodeStepDoesNotLaunchPrefillOnlyNodes(t *testing.T) {
	for _, c := range []struct {
		scenario string
		prefill  bool
	}{{v41, true}, {"minimax-m25-h200-tp8.yaml", false}} {
		in := fixtureInputs(t, c.scenario)
		editEngine(t, &in, func(e *engineconfig.Engine) { e.CUDAGraphMode = "NONE" })
		want := 0
		for _, id := range in.Model.Stack.Expand() {
			for _, lk := range in.Model.LayerKinds {
				if lk.ID != id {
					continue
				}
				for _, n := range lk.Nodes {
					if n.Context != nil && n.Context.Phase == "prefill" {
						want++
					}
				}
			}
		}
		if (want > 0) != c.prefill {
			t.Fatalf("%s: %d prefill-only nodes in the graph", c.scenario, want)
		}
		k := build(t, in)
		dec := decodeBatch(4, 6, 90000)
		mixed := dec
		mixed.Reqs = append(append([]kernel.ReqShape{}, dec.Reqs...),
			kernel.ReqShape{Scheduled: 512, Computed: 1024, PromptLen: 8192})
		got := k.StepTime(mixed).PerResource[kernel.ResourceHost] -
			k.StepTime(dec).PerResource[kernel.ResourceHost]
		per := k.launchEager + k.launchPerKernel
		if d := got - time.Duration(want)*per; d < -time.Nanosecond || d > time.Nanosecond {
			t.Errorf("%s: a prefill row added %v of host time; its %d prefill-only "+
				"launches at %v each are %v", c.scenario, got, want, per,
				time.Duration(want)*per)
		}
	}
}

// THE EAGER COST CAN BE A MODEL'S OWN. With no graph DeepSeek-V4.1 pays Python dispatch per
// op, since vLLM runs it uncompiled (vllm/config/vllm.py:120), so the registry may scope the
// eager constant to the model. A model-scoped entry must win over the part-wide one whatever
// the file or set order, under either name; one scoped to another model must not apply; and
// host_launch_eager_per_kernel, the name that says what the value is, wins over the
// per-layer name at equal scope. The per-launch cost is read through the step: under NONE a
// decode step's host time is launches x (eager + dispatch) plus nothing else.
func TestAModelScopedEagerCostWinsWhateverItsOrder(t *testing.T) {
	entry := func(name string, v float64, model string) coefficient.Entry {
		e := coefficient.Entry{Name: name, Value: v, Units: "us_per_transfer",
			Method: "assumed", Rationale: "test", Scope: coefficient.Scope{
				Hardware: []string{"h200"}}}
		if model != "" {
			e.Scope.Model = []string{model}
		}
		return e
	}
	eager := func(before bool, extra ...coefficient.Entry) time.Duration {
		t.Helper()
		// blis-registry v0.2.0 commits a model-scoped host_launch_eager_per_kernel for
		// V4.1, which would outrank every entry below; the cases supply their own.
		in := withoutEntries(fixtureInputs(t, v41),
			map[string]bool{"host_launch_eager_per_kernel": true})
		editEngine(t, &in, func(e *engineconfig.Engine) { e.CUDAGraphMode = "NONE" })
		set := &coefficient.Set{Kind: "CoefficientSet", Name: "test-eager",
			Coefficients: extra}
		switch {
		case len(extra) == 0:
		case before:
			in.Coefficients = append([]*coefficient.Set{set}, in.Coefficients...)
		default:
			in.Coefficients = append(append([]*coefficient.Set{}, in.Coefficients...), set)
		}
		k := build(t, in)
		b := decodeBatch(1, 6, 90000)
		host := k.StepTime(b).PerResource[kernel.ResourceHost]
		_, draft := k.priceDraft(b, &stepShape{}, &[numResources]float64{})
		n := time.Duration(k.stepKernels(b) + draft)
		return host/n - k.launchPerKernel
	}
	us := func(v float64) time.Duration { return time.Duration(v * float64(time.Microsecond)) }
	generic := eager(false)
	const m = "deepseek-v4.1-flash"
	for _, c := range []struct {
		name   string
		before bool
		extra  []coefficient.Entry
		want   time.Duration
	}{
		{"per-layer, model-scoped, listed first", true,
			[]coefficient.Entry{entry("host_launch_eager_per_layer", 93.2, m)}, us(93.2)},
		{"per-layer, model-scoped, listed last", false,
			[]coefficient.Entry{entry("host_launch_eager_per_layer", 93.2, m)}, us(93.2)},
		{"per-layer, another model", false,
			[]coefficient.Entry{entry("host_launch_eager_per_layer", 93.2, "kimi-k3")}, generic},
		{"per-kernel, model-scoped, listed first", true,
			[]coefficient.Entry{entry("host_launch_eager_per_kernel", 91, m)}, us(91)},
		{"per-kernel generic beside per-layer generic", false,
			[]coefficient.Entry{entry("host_launch_eager_per_kernel", 11, "")}, us(11)},
		{"per-layer model-scoped beats per-kernel generic", false, []coefficient.Entry{
			entry("host_launch_eager_per_kernel", 11, ""),
			entry("host_launch_eager_per_layer", 93.2, m)}, us(93.2)},
	} {
		if got := eager(c.before, c.extra...); got-c.want > time.Nanosecond ||
			c.want-got > time.Nanosecond {
			t.Errorf("%s: eager cost per launch %v, want %v", c.name, got, c.want)
		}
	}
}

// THE DSPARK DRAFTER RUNS UNDER ITS OWN GRAPHS. Its manager captures FULL graphs only, and
// only when the target's decode mode is FULL ("PIECEWISE cudagraphs are not supported for
// dflash", vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:126-150); otherwise it runs
// eagerly. A replay covers the context rows of one verify per request (:309-311), so a step
// with a prefill row stores the rest of its context eagerly first (:480-482). Each case is
// the drafter's host share: the step's host time less the same step's target-only host
// time (the target's launches, dispatch included, do not depend on the drafter).
func TestTheDrafterRunsUnderItsOwnGraphs(t *testing.T) {
	size := 64
	draftShare := func(mode string, b kernel.Batch) (got time.Duration, k *Kernel) {
		t.Helper()
		in := fixtureInputs(t, v41)
		e := mustDecode(t, in.Deployment.Pools[0])
		e.CUDAGraphMode, e.MaxCUDAGraphCaptureSize = mode, &size
		setEngine(t, &in.Deployment.Pools[0], e)
		k = build(t, in)
		_, kernels := k.priceDraft(b, &stepShape{}, &[numResources]float64{})
		host := k.StepTime(b).PerResource[kernel.ResourceHost]
		target := k.targetHost(b, 0) + seconds(k.launchSeconds(b, kernels))
		return host - target, k
	}
	d := func(k *Kernel) *price.DraftPlan { return k.plan.Draft }
	// A decode of 4 requests drafts 20 tokens: one replay under FULL_AND_PIECEWISE, every
	// launch eager under PIECEWISE, whose decode mode is not FULL.
	dec := decodeBatch(4, 6, 90000)
	if got, k := draftShare("FULL_AND_PIECEWISE", dec); got != k.replayPerStep {
		t.Errorf("FULL_AND_PIECEWISE decode: drafter host %v, want one replay %v", got,
			k.replayPerStep)
	}
	got, k := draftShare("PIECEWISE", dec)
	all := d(k).Kernels + d(k).ContextKernels + 5*d(k).SequentialKernels
	if want := time.Duration(all) * k.launchEager; got != want {
		t.Errorf("PIECEWISE decode: drafter host %v, want %d eager launches, %v", got, all,
			want)
	}
	// A decode plus one prefill chunk that completes its prompt: the draft still replays,
	// and its context kernels beyond the graph's rows launch eagerly.
	mixed := dec
	mixed.Reqs = append(append([]kernel.ReqShape{}, dec.Reqs...),
		kernel.ReqShape{Scheduled: 1808, Computed: 88192, PromptLen: 90000})
	got, k = draftShare("FULL_AND_PIECEWISE", mixed)
	if want := k.replayPerStep + time.Duration(d(k).ContextKernels)*k.launchEager; got != want {
		t.Errorf("mixed step: drafter host %v, want a replay plus %d eager context "+
			"launches, %v", got, d(k).ContextKernels, want)
	}
}

// vendoredHostEntries replicates blis-registry c45c71c's cost-model-host-overheads.yaml order
// for the entries DeepSeek-V4.1 scopes (docs/COEFF_CONTRACT.md v2.1): each model-scoped entry
// follows its part-wide namesake in the file, and the eager constant is derived on h200
// (assumed) and copied to the other eight parts (copied).
func vendoredHostEntries() []coefficient.Entry {
	all := []string{"a100-80", "a100-sxm", "b200", "b300", "gb200-nvl72", "gb300", "h100",
		"h200", "l40s"}
	others := []string{"a100-80", "a100-sxm", "b200", "b300", "gb200-nvl72", "gb300", "h100",
		"l40s"}
	m := []string{"deepseek-v4.1-flash"}
	e := func(name string, v float64, units, method string, hw, model []string) coefficient.Entry {
		x := coefficient.Entry{Name: name, Value: v, Units: vocab.Unit(units),
			Method: vocab.Method(method), Scope: coefficient.Scope{Hardware: hw, Model: model}}
		switch method {
		case "measured":
			x.Sources = []coefficient.Source{{Kind: "discussion", Cite: "test", Role: "primary"}}
		default:
			x.Rationale = "test"
		}
		if method == "copied" {
			x.CopiedFrom = "hardware=h200,model=deepseek-v4.1-flash"
		}
		return x
	}
	return []coefficient.Entry{
		e("host_admission_per_request", 40000, "us_per_request", "assumed", all, nil),
		e("host_admission_per_request", 0, "us_per_request", "not_charged", all, m),
		e("host_admission_per_token", 0.35, "us_per_token", "assumed", all, nil),
		e("host_admission_per_token", 2.67, "us_per_token", "measured", all, m),
		e("host_launch_eager_per_layer", 7, "us_per_layer", "assumed", all, nil),
		e("host_launch_eager_per_kernel", 93.8, "us_per_transfer", "assumed", []string{"h200"}, m),
		e("host_launch_eager_per_kernel", 93.8, "us_per_transfer", "copied", others, m),
	}
}

// THE TWO RESOLUTION RULES AGREE ON THE VENDORED SET, AND A COPY SAYS IT IS ONE. The registry
// resolves last-in-file wins with model-scoped placed last; the kernel never lets a part-wide
// entry displace a model-scoped one. On the vendored order the two must pick the same entry
// for every name on every part, for this model and another. And Provenance must carry the
// resolved entry's method and where a copy came from, so a report can show h100's eager
// constant as copied from h200.
func TestTheVendoredHostSetResolvesTheSameUnderBothRules(t *testing.T) {
	set := &coefficient.Set{Kind: "CoefficientSet", Name: "cost-model-host-overheads",
		Coefficients: vendoredHostEntries()}
	for _, chip := range []string{"h100", "h200", "b200"} {
		for _, model := range []string{"deepseek-v4.1-flash", "kimi-k3"} {
			scope := resolve.Scope{Hardware: chip, Model: model}
			got, err := resolve.Load([]*coefficient.Set{set}, scope)
			if err != nil {
				t.Fatal(err)
			}
			last := map[string]coefficient.Entry{}
			for _, e := range set.Coefficients {
				if scope.Admits(e.Scope) {
					last[e.Name] = e
				}
			}
			for name, want := range last {
				r, ok := got.Entry(name)
				if !ok || r.Entry.Value != want.Value || r.Entry.Method != want.Method ||
					r.Entry.CopiedFrom != want.CopiedFrom {
					t.Errorf("%s/%s %s: kernel resolved %+v, last-in-file gives %+v", chip,
						model, name, r.Entry, want)
				}
			}
		}
	}
	for _, c := range []struct{ chip, method, scope string }{
		{"h200", "assumed", "hardware: h200, model: deepseek-v4.1-flash"},
		{"h100", "copied", "hardware: h100, model: deepseek-v4.1-flash; copied from " +
			"hardware=h200,model=deepseek-v4.1-flash"},
	} {
		in := fixtureInputs(t, v41)
		if c.chip != "h200" {
			in.Chip, in.Scenario.Cluster.Hardware = chipInputs(t, c.chip), c.chip
		}
		in.Coefficients = append(append([]*coefficient.Set{}, in.Coefficients...), set)
		k := build(t, in)
		if want := time.Duration(93.8 * float64(time.Microsecond)); k.launchEager != want {
			t.Errorf("%s: eager launch %v, want the model's %v", c.chip, k.launchEager, want)
		}
		found := false
		for _, o := range k.Provenance() {
			if o.Name == "host_launch_eager_per_kernel" {
				found = true
				if o.Method != c.method || o.Scope != c.scope {
					t.Errorf("%s: Provenance records %s %q, want %s %q", c.chip, o.Method,
						o.Scope, c.method, c.scope)
				}
			}
		}
		if !found {
			t.Errorf("%s: host_launch_eager_per_kernel is not in Provenance", c.chip)
		}
	}
}

// eagerAt is the V4.1 fixture with the model's eager launch cost at us microseconds, in the
// graph mode given, without speculation where spec is false.
func eagerAt(t *testing.T, us float64, mode string, spec bool, capture int) *Kernel {
	t.Helper()
	in := fixtureInputs(t, v41)
	e := mustDecode(t, in.Deployment.Pools[0])
	e.CUDAGraphMode = mode
	if capture > 0 {
		e.MaxCUDAGraphCaptureSize = &capture
	}
	if !spec {
		e.Speculative = nil
	}
	setEngine(t, &in.Deployment.Pools[0], e)
	in.Coefficients = append(append([]*coefficient.Set{}, in.Coefficients...),
		&coefficient.Set{Kind: "CoefficientSet", Name: "test-eager", Coefficients: []coefficient.Entry{{
			Name: "host_launch_eager_per_kernel", Value: us, Units: "us_per_transfer",
			Method: "assumed", Rationale: "test", Scope: coefficient.Scope{
				Hardware: []string{"h200"}, Model: []string{"deepseek-v4.1-flash"}}}}})
	return build(t, in)
}

// ANCHOR, decode, eager: H200 TP8 under cudagraph_mode NONE, no speculation, one request at
// a 90k context, steps in 96.56 ms. Decode is launch-bound, so the step is its launches at
// the eager cost plus the dispatch gap, plus its device time: the registry's eager constant
// is backed out of this anchor as (96.56 ms - device) / launches - dispatch.
func TestTheEagerDecodeAnchorIsAdditive(t *testing.T) {
	probe := eagerAt(t, 1, "NONE", false, 0)
	b := decodeBatch(1, 1, 90000)
	launches := probe.stepKernels(b)
	base := probe.StepTime(b).NoOverlap - time.Duration(launches)*time.Microsecond
	us := (96.56e6 - float64(base)) / float64(launches) / 1e3
	got := eagerAt(t, us, "NONE", false, 0).StepTime(b).NoOverlap
	if d := math.Abs(float64(got)-96.56e6) / 96.56e6; d > 1e-3 {
		t.Errorf("at %.2f us a launch over %d launches the step is %v, want the anchor's "+
			"96.56 ms", us, launches, got)
	}
	if us < 90 || us > 95 {
		t.Errorf("the anchor backs out %.2f us a launch, outside the registry basis's 90-95", us)
	}
	t.Logf("eager constant backed out of the anchor: %.2f us over %d launches", us, launches)
}

// ANCHOR, mixed, eager: the eager launch cost is never hidden behind a step's device work.
// On the held-out H200 TP8 transfer arm (FULL_AND_PIECEWISE, capture 64, DSpark K5) vLLM
// collapses at c32 (ITL 16.18 ms), which BLIS reproduces only if mixed steps past the
// capture ceiling pay every eager launch on top of their device time; hiding min(launches,
// device) gives 5.75 ms. So each extra microsecond of launch cost adds exactly one per
// launch, prefill chunk or not.
func TestEagerLaunchesAreNotHiddenBehindAMixedStep(t *testing.T) {
	b := decodeBatch(20, 6, 90000)
	b.DecodeThreshold = 6
	b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: 4096, Computed: 50000, PromptLen: 54096})
	lo, hi := eagerAt(t, 50, "", true, 64), eagerAt(t, 100, "", true, 64)
	launches := lo.stepKernels(b)
	if launches == 0 {
		t.Fatal("the step launches no kernels")
	}
	got := hi.StepTime(b).NoOverlap - lo.StepTime(b).NoOverlap
	if want := time.Duration(launches) * 50 * time.Microsecond; got < want {
		t.Errorf("50 us more a launch adds %v to a 20-decode, 4,096-prefill step past the "+
			"capture ceiling, under %d target launches x 50 us = %v: launches are hidden",
			got, launches, want)
	}
}
