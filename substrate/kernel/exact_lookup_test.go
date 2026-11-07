package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
	"github.com/inference-sim/blis-substrate/load"
)

// VAL-8 at the pricing boundary: a coefficient that lists kernel implementations prices a
// primitive only when the implementations the pool runs for THAT primitive are ones it lists
// (Inputs.Implementations, the PoolPlan's ImplementationsByPrimitive()). The test builds a real
// kernel, finds a coefficient the kernel reads for a primitive, keys it to the kernel that
// executes that primitive, adds a transferred value for another build, and runs the pool with a
// second kernel on a different primitive.

var (
	// buildX is the kernel that executes the primitive under test, buildZ another build of it,
	// and buildY a kernel that executes a different primitive in the same pool.
	buildX = plan.KernelImpl{Library: "nccl", Variant: "ncclAllReduce", CodeHash: "aaaaaaaaaaaa", Toolkit: "cuda13.0"}
	buildZ = plan.KernelImpl{Library: "nccl", Variant: "ncclAllReduce", CodeHash: "bbbbbbbbbbbb", Toolkit: "cuda13.0"}
	buildC = plan.KernelImpl{Library: "nccl", Variant: "ncclAllReduce", CodeHash: "cccccccccccc"}
	buildY = plan.KernelImpl{Library: "cublas", Variant: "gemm", CodeHash: "dddddddddddd"}
)

// keyedPair returns the sets with the entry keyed to build X (measured), and a transferred value
// for build Z, measured on X, at 0.8 times the measured value.
func keyedPair(sets []*coefficient.Set, r resolve.Resolved) []*coefficient.Set {
	implX := coefficient.KernelImpl{Library: buildX.Library, Variant: buildX.Variant, CodeHash: buildX.CodeHash}
	implZ := coefficient.KernelImpl{Library: buildZ.Library, Variant: buildZ.Variant, CodeHash: buildZ.CodeHash}
	out := make([]*coefficient.Set, len(sets))
	for i, s := range sets {
		c := *s
		c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
		if s.Name == r.Set {
			for j, e := range c.Coefficients {
				if e.Identity() != r.Entry.Identity() {
					continue
				}
				keyed := e
				keyed.KernelImpls = []coefficient.KernelImpl{implX}
				c.Coefficients[j] = keyed
				moved := e
				moved.Value = e.Value * 0.8
				moved.Method, moved.Fitted = vocab.MethodTransferred, false
				moved.KernelImpls, moved.MeasuredOn = []coefficient.KernelImpl{implZ}, []coefficient.KernelImpl{implX}
				moved.TransferredFrom, moved.TransferRule = "hardware=h200", "a new build of the same kernel, same shapes"
				moved.Rationale = "test"
				moved.Measures, moved.FitRange, moved.CI95, moved.Sources = nil, nil, nil, nil
				c.Coefficients = append(c.Coefficients, moved)
				break
			}
		}
		out[i] = &c
	}
	return out
}

func TestTheKernelImplementationThatRunsDecidesWhichCoefficientPrices(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a catalog scenario for several kernels")
	}
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	cases := perturbationScenarios(t, d, d.Models()[0])
	if len(cases) == 0 {
		t.Fatal("no scenario builds")
	}
	in := cases[0].in
	base, err := New(in)
	if err != nil {
		t.Fatal(err)
	}
	want := pricesOf(base)
	// The entry to key: a measured one the kernel reads for a lowered primitive, unscoped to a
	// fabric, whose value moves a price.
	var pick resolve.Resolved
	var movedBy string
	for _, sel := range base.selected {
		if sel.Entry.Method != vocab.MethodMeasured || len(sel.Entry.Scope.Fabric) > 0 || sel.Entry.Value == 0 ||
			coefficient.PrimitiveOf(sel.Entry.Name) == "" {
			continue
		}
		scaled := in
		scaled.Coefficients = perturbed(in.Coefficients, sel.Entry.Name)
		if k, err := New(scaled); err == nil && pricesOf(k) != want {
			pick, movedBy = sel, pricesOf(k)
			break
		}
	}
	if pick.Entry.Name == "" {
		t.Fatal("no measured entry that prices a primitive moves the price of the scenario")
	}
	name := pick.Entry.Name
	prim := plan.Primitive(coefficient.PrimitiveOf(name))
	t.Logf("%s: %s prices %s", cases[0].label, name, prim)
	// The primitive a name prices is stated here again, from the names' own prefixes, so a wrong
	// binding in coefficient.PrimitiveOf is caught rather than agreed with.
	oracle := map[string]plan.Primitive{"attention_decode_": plan.PrimAttentionDecode, "attention_prefill_": plan.PrimAttentionPrefill,
		"recurrent_": plan.PrimRecurrentUpdate, "host_gather_": plan.PrimGather, "moe_a2a_": plan.PrimAllToAll,
		"gemm_": plan.PrimGEMM, "collective_": ""}
	for pre, want := range oracle {
		grouped := pre == "gemm_" && prim == plan.PrimMoEGroupedGEMM
		if strings.HasPrefix(name, pre) && want != "" && prim != want && !grouped {
			t.Fatalf("%s is bound to %s, but its name prices %s", name, prim, want)
		}
	}
	// other is a primitive the entry does not price.
	other := plan.PrimGEMM
	if prim == other {
		other = plan.PrimAttentionDecode
	}
	keyed := in
	keyed.Coefficients = keyedPair(in.Coefficients, pick)
	build := func(run map[plan.Primitive][]plan.KernelImpl) (*Kernel, error) {
		x := keyed
		x.Implementations = run
		return New(x)
	}
	matchOf := func(k *Kernel) string {
		for _, o := range k.Provenance() {
			if o.Name == name && o.Set != KernelAssumptionSet {
				return o.KernelMatch
			}
		}
		return "missing"
	}

	// The kernel that executes the primitive is the one the entry is keyed to: it prices, as it
	// did before it was keyed, and the provenance names the primitive and what ran it.
	k, err := build(map[plan.Primitive][]plan.KernelImpl{prim: {buildX}, other: {buildY}})
	if err != nil {
		t.Fatalf("build X: %v", err)
	}
	if pricesOf(k) != want || matchOf(k) != "exact" {
		t.Errorf("build X: prices equal the unkeyed kernel's = %v, match %q", pricesOf(k) == want, matchOf(k))
	}
	for _, o := range k.Provenance() {
		if o.Name == name && o.Set != KernelAssumptionSet &&
			(o.Primitive != string(prim) || o.Running != "nccl/ncclAllReduce/aaaaaaaaaaaa") {
			t.Errorf("provenance of %s: primitive %q, running %v", name, o.Primitive, o.Running)
		}
	}

	// Transferred evidence for a known target: the target prices from the transferred value,
	// 0.8 times the measured one, and only it.
	k, err = build(map[plan.Primitive][]plan.KernelImpl{prim: {buildZ}, other: {buildY}})
	if err != nil {
		t.Fatalf("build Z: %v", err)
	}
	if pricesOf(k) != movedBy || matchOf(k) != "exact" {
		t.Errorf("build Z: prices come from the transferred value = %v, match %q", pricesOf(k) == movedBy, matchOf(k))
	}

	// No sibling fallback. Neither entry is used when the primitive runs another build; when the
	// kernel the entry is keyed to runs ANOTHER primitive (the binding swapped); when nothing is
	// said about the primitive; and when the primitive runs the keyed kernel and a second one.
	// The kernel either refuses, naming what it refused, or, for a coefficient it reads as
	// optional, prices without it. It never prices from the value of a kernel that does not
	// execute the primitive.
	for label, run := range map[string]map[plan.Primitive][]plan.KernelImpl{
		"another build of the kernel":             {prim: {buildC}, other: {buildY}},
		"the keyed kernel runs another primitive": {prim: {buildY}, other: {buildX}},
		"nothing said about the primitive":        {other: {buildX}},
		"no implementation stated":                nil,
		"the primitive also runs a second kernel": {prim: {buildX, buildY}},
	} {
		k, err := build(run)
		switch {
		case err != nil:
			if !strings.Contains(err.Error(), "VAL-8") {
				t.Errorf("%s: the refusal should name VAL-8: %v", label, err)
			}
		case pricesOf(k) == want || pricesOf(k) == movedBy:
			t.Errorf("%s: priced from a kernel that does not execute %s", label, prim)
		case matchOf(k) != "missing":
			t.Errorf("%s: %s is still in the provenance with match %q", label, name, matchOf(k))
		}
	}

	// The kernel is built for one pool: two builds of one kernel cannot both run.
	if _, err := build(map[plan.Primitive][]plan.KernelImpl{prim: {buildX}, other: {buildZ}}); err == nil ||
		!strings.Contains(err.Error(), "two builds of one kernel") {
		t.Errorf("two builds of one kernel: %v", err)
	}
	// A malformed hash is refused rather than quietly matching nothing.
	if _, err := build(map[plan.Primitive][]plan.KernelImpl{prim: {{Library: "nccl", Variant: "ncclAllReduce", CodeHash: "AAAA"}}}); err == nil {
		t.Errorf("a malformed hash was accepted")
	}

	// The registry as committed lists no kernel implementation: stating what runs changes no
	// price, and every coefficient is reported unkeyed.
	stated := in
	stated.Implementations = map[plan.Primitive][]plan.KernelImpl{prim: {buildX}, other: {buildY}}
	k, err = New(stated)
	if err != nil {
		t.Fatal(err)
	}
	if pricesOf(k) != want {
		t.Errorf("stating the running implementations changed a price while no entry lists one")
	}
	for _, o := range k.Provenance() {
		if o.Set != KernelAssumptionSet && o.KernelMatch != "unkeyed" {
			t.Errorf("%s: match %q, want unkeyed", o.Name, o.KernelMatch)
		}
		if o.Set == KernelAssumptionSet && o.KernelMatch != "" {
			t.Errorf("%s: the kernel's own assumption has match %q", o.Name, o.KernelMatch)
		}
	}
}

// The binding table names only primitives the plan defines, and every coefficient name the
// registry carries is bound to one or deliberately to none.
func TestEveryBoundPrimitiveIsAPlanPrimitive(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	bound := map[string]int{}
	for _, set := range []string{"cost-model-primitives", "cost-model-collectives", "cost-model-host-overheads",
		"cost-model-attention", "cost-model-recurrent", "cost-model-memory", "cost-model-fabric",
		"cost-model-collectives-inferencex"} {
		cs, err := d.CoefficientSet(set)
		if err != nil {
			continue
		}
		for _, e := range cs.Coefficients {
			if p := coefficient.PrimitiveOf(e.Name); p != "" {
				if !plan.Primitive(p).Valid() {
					t.Errorf("%s binds to %q, which is not a plan primitive", e.Name, p)
				}
				bound[p]++
			}
		}
	}
	for _, p := range []string{"gemm", "moe.grouped_gemm", "attention.decode", "attention.prefill", "recurrent.update", "all_reduce", "all_gather", "reduce_scatter", "all2all"} {
		if bound[p] == 0 {
			t.Errorf("no committed coefficient binds to %s: the binding table has drifted from the registry's names", p)
		}
	}
}
