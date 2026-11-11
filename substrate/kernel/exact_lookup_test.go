package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/units"
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

// The review's counterexample at the pricing boundary: full attention runs kernel A and the
// sliding-window layers run kernel B. attention_decode_rate_swa is read only by the
// sliding-window layers, so a value keyed to B prices them; one keyed to A does not. Stating the
// pool as the primitive's pooled implementations (A and B for every attention coefficient)
// refuses the B-keyed value, which is the behaviour this binding replaces.
func TestAValueForTheKernelThatRunsOnlyTheWindowedLayersPricesThem(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	var in Inputs
	for _, c := range perturbationScenarios(t, d, "gpt-oss-120b") {
		if strings.HasPrefix(c.label, "gpt-oss-120b/h200/tp") {
			in = c.in
			break
		}
	}
	if in.Model == nil {
		t.Fatal("no gpt-oss-120b scenario builds on h200")
	}
	const swaRate = "attention_decode_rate_swa"
	kernelA := plan.KernelImpl{Library: "flashinfer", Variant: "decode_full", CodeHash: "aaaaaaaaaaaa"}
	kernelB := plan.KernelImpl{Library: "flashinfer", Variant: "decode_window", CodeHash: "bbbbbbbbbbbb"}
	// Every layer's attention runs A, except the layers whose attention is sliding-window, which
	// run B.
	kinds := map[string]model.LayerKind{}
	for _, lk := range in.Model.LayerKinds {
		kinds[lk.ID] = lk
	}
	var ops []plan.Operation
	windowed := 0
	for i, id := range in.Model.Stack.Expand() {
		impl := kernelA
		for _, n := range kinds[id].Nodes {
			if n.Op == model.OpAttention && n.AttentionKind == model.AttentionSWA {
				impl = kernelB
			}
		}
		if impl == kernelB {
			windowed++
		}
		ops = append(ops, plan.Operation{Layer: i, Primitive: plan.PrimAttentionDecode, Impl: impl})
	}
	if windowed == 0 || windowed == len(ops) {
		t.Fatalf("gpt-oss-120b has %d windowed layers of %d; the test needs a mix", windowed, len(ops))
	}
	base, err := New(in)
	if err != nil {
		t.Fatal(err)
	}
	if base.Provenance() == nil || !hasSelected(base, swaRate) {
		t.Fatalf("the unkeyed registry does not price the windowed layers with %s", swaRate)
	}
	want := pricesOf(base)

	keyTo := func(impl plan.KernelImpl) Inputs {
		x := in
		x.Coefficients = nil
		for _, s := range in.Coefficients {
			c := *s
			c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
			for j, e := range c.Coefficients {
				if e.Name == swaRate && e.Method == vocab.MethodMeasured {
					e.KernelImpls = []coefficient.KernelImpl{{Library: impl.Library, Variant: impl.Variant, CodeHash: impl.CodeHash}}
					c.Coefficients[j] = e
				}
			}
			x.Coefficients = append(x.Coefficients, &c)
		}
		return x
	}
	provenance := func(k *Kernel) (coefficient.KernelMatch, string, bool) {
		for _, o := range k.Provenance() {
			if o.Name == swaRate && o.Set != KernelAssumptionSet {
				return coefficient.KernelMatch(o.KernelMatch), o.Running, true
			}
		}
		return "", "", false
	}

	// Keyed to the kernel that runs the windowed layers: priced, matched against that kernel alone.
	x := keyTo(kernelB)
	x.Operations = ops
	k, err := New(x)
	if err != nil {
		t.Fatalf("the value keyed to the windowed layers' kernel: %v", err)
	}
	match, running, ok := provenance(k)
	if !ok || match != coefficient.KernelExact || running != "flashinfer/decode_window/bbbbbbbbbbbb" {
		t.Errorf("%s: selected %v, match %q, matched against %q; want exact against the windowed kernel alone", swaRate, ok, match, running)
	}
	if pricesOf(k) != want {
		t.Errorf("a value keyed to the kernel that runs the windowed layers changed the price")
	}

	// Keyed to the kernel that runs full attention: not used for the windowed layers.
	x = keyTo(kernelA)
	x.Operations = ops
	if k, err := New(x); err == nil {
		if _, _, ok := provenance(k); ok || pricesOf(k) == want {
			t.Errorf("a value keyed to the full-attention kernel priced the windowed layers")
		}
	}

	// The pool stated as pooled implementations: B-keyed is refused because A also runs attention.
	x = keyTo(kernelB)
	x.Implementations = map[plan.Primitive][]plan.KernelImpl{plan.PrimAttentionDecode: {kernelA, kernelB}}
	if k, err := New(x); err == nil {
		if _, _, ok := provenance(k); ok || pricesOf(k) == want {
			t.Errorf("the pooled form priced the windowed layers from a value valid for one of two kernels")
		}
	}

	// Operations and Implementations that disagree are refused.
	x = keyTo(kernelB)
	x.Operations = ops
	x.Implementations = map[plan.Primitive][]plan.KernelImpl{plan.PrimAttentionDecode: {kernelA}}
	if _, err := New(x); err == nil || !strings.Contains(err.Error(), "different kernel implementations") {
		t.Errorf("disagreeing Operations and Implementations: %v", err)
	}
	// An operation that names a layer the graph lacks is refused.
	x = keyTo(kernelB)
	x.Operations = append(append([]plan.Operation(nil), ops...), plan.Operation{Layer: len(ops), Primitive: plan.PrimGEMM, Impl: kernelA})
	if _, err := New(x); err == nil || !strings.Contains(err.Error(), "names layer") {
		t.Errorf("an operation past the graph's last layer: %v", err)
	}
}

func hasSelected(k *Kernel, name string) bool {
	for _, s := range k.selected {
		if s.Entry.Name == name {
			return true
		}
	}
	return false
}

// keyMeasured returns the inputs with every measured entry of the named coefficients keyed to
// the implementation given for its name.
func keyMeasured(in Inputs, impls map[string]plan.KernelImpl) Inputs {
	var sets []*coefficient.Set
	for _, s := range in.Coefficients {
		c := *s
		c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
		for j, e := range c.Coefficients {
			if i, ok := impls[e.Name]; ok && e.Method == vocab.MethodMeasured {
				e.KernelImpls = []coefficient.KernelImpl{{Library: i.Library, Variant: i.Variant, CodeHash: i.CodeHash}}
				c.Coefficients[j] = e
			}
		}
		sets = append(sets, &c)
	}
	in.Coefficients = sets
	return in
}

// selectedMatch is how the named coefficient met the implementations that run, and which ones,
// or ok false when no entry of that name priced the kernel.
func selectedMatch(k *Kernel, name string) (match string, running string, ok bool) {
	for _, o := range k.Provenance() {
		if o.Name == name && o.Set != KernelAssumptionSet {
			return o.KernelMatch, o.Running, true
		}
	}
	return "", "", false
}

// One layer can run two nodes of the same primitive: DeepSeek-V4.1's sparse layers run a
// block-index scorer beside the primary sparse-MLA attention, and the registry prices them apart
// (attention_decode_*_block_index_scores, attention_decode_rate_sparse_mla). A plan that lowers
// the scorer to one kernel and the primary to another binds each coefficient to its own node: the
// scorer's pair to the scorer's kernel, the primary's to the primary's, and a pair keyed to the
// other node's kernel is not used.
func TestTheScorerAndThePrimaryOfOneLayerAreBoundToTheirOwnKernels(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	var in Inputs
	for _, c := range perturbationScenarios(t, d, "deepseek-v4.1-flash") {
		if strings.HasPrefix(c.label, "deepseek-v4.1-flash/h200/tp") {
			in = c.in
			break
		}
	}
	if in.Model == nil {
		t.Fatal("no deepseek-v4.1-flash scenario builds on h200")
	}
	primary := plan.KernelImpl{Library: "flashmla", Variant: "sparse_decode", CodeHash: "aaaaaaaaaaaa"}
	scorer := plan.KernelImpl{Library: "deepgemm", Variant: "paged_mqa_logits", CodeHash: "bbbbbbbbbbbb"}
	kinds := map[string]model.LayerKind{}
	for _, lk := range in.Model.LayerKinds {
		kinds[lk.ID] = lk
	}
	scored := 0
	for i, id := range in.Model.Stack.Expand() {
		in.Operations = append(in.Operations, plan.Operation{Layer: i, Primitive: plan.PrimAttentionDecode, Impl: primary})
		for _, n := range kinds[id].Nodes {
			if n.Op == model.OpAttention && !n.ReadsEngineKV() {
				in.Operations = append(in.Operations, plan.Operation{Layer: i, Scorer: true, Primitive: plan.PrimAttentionDecode, Impl: scorer})
				scored++
				break
			}
		}
	}
	if scored == 0 {
		t.Fatal("the graph has no block-index scorer")
	}
	names := []string{"attention_decode_rate_sparse_mla", "attention_decode_floor_block_index_scores", "attention_decode_rate_block_index_scores"}
	base, err := New(in)
	if err != nil {
		t.Fatal(err)
	}
	have := map[string]bool{}
	for _, n := range names {
		_, _, have[n] = selectedMatch(base, n)
	}
	if !have[names[0]] || !have[names[1]] {
		t.Fatalf("the unkeyed registry does not price the sparse layers with %v (selected %v)", names, have)
	}
	right := map[string]plan.KernelImpl{names[0]: primary, names[1]: scorer, names[2]: scorer}
	k, err := New(keyMeasured(in, right))
	if err != nil {
		t.Fatalf("each pair keyed to the kernel of its own node: %v", err)
	}
	for _, n := range names {
		if !have[n] {
			continue
		}
		want := primary
		if n != names[0] {
			want = scorer
		}
		m, run, ok := selectedMatch(k, n)
		if n == names[2] {
			// h200 carries this rate under a method that may not state kernels; it must still be
			// used, matched against the scorer's kernel alone.
			if !ok || run != want.Library+"/"+want.Variant+"/"+want.CodeHash {
				t.Errorf("%s: selected %v, matched against %q; want the scorer's kernel", n, ok, run)
			}
			continue
		}
		if !ok || m != string(coefficient.KernelExact) || run != want.Library+"/"+want.Variant+"/"+want.CodeHash {
			t.Errorf("%s: selected %v, match %q, matched against %q; want exact against its own node's kernel", n, ok, m, run)
		}
	}
	if pricesOf(k) != pricesOf(base) {
		t.Error("keying the pairs to the kernels that run them changed a price")
	}
	// Swapped: the scorer's pair keyed to the primary's kernel is not used for the scorer.
	if k, err := New(keyMeasured(in, map[string]plan.KernelImpl{names[1]: primary})); err == nil {
		if _, _, ok := selectedMatch(k, names[1]); ok {
			t.Error("the scorer's floor keyed to the primary's kernel priced the scorer")
		}
	}
	// With the pooled form (one set of kernels for the primitive) the same keys are refused, as
	// the two nodes cannot be told apart.
	pooled := keyMeasured(in, right)
	pooled.Operations = nil
	pooled.Implementations = map[plan.Primitive][]plan.KernelImpl{plan.PrimAttentionDecode: {primary, scorer}}
	if k, err := New(pooled); err == nil {
		if _, _, ok := selectedMatch(k, names[0]); ok {
			t.Error("the pooled form priced the primary from a value valid for one of two kernels")
		}
	}
}

// A GEMM value is bound to the GEMMs of its format: a BF16 ramp keyed to the kernel that runs the
// BF16 GEMMs is used when an FP8 GEMM runs another kernel, and an FP8-keyed value is not used
// for the BF16 ones.
func TestAGemmValueIsBoundToTheKernelOfItsFormat(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	var in Inputs
	for _, c := range perturbationScenarios(t, d, "gpt-oss-120b") {
		if strings.HasPrefix(c.label, "gpt-oss-120b/h200/tp") {
			in = c.in
			break
		}
	}
	if in.Model == nil {
		t.Fatal("no gpt-oss-120b scenario builds on h200")
	}
	bf16K := plan.KernelImpl{Library: "cublas", Variant: "gemm_bf16", CodeHash: "aaaaaaaaaaaa"}
	fp8K := plan.KernelImpl{Library: "cutlass", Variant: "gemm_fp8_block", CodeHash: "bbbbbbbbbbbb"}
	bf16F := units.QuantFormat{PayloadBits: 16}
	fp8F := units.QuantFormat{PayloadBits: 8, ScaleBits: 32, BlockElements: 16384}
	in.Operations = []plan.Operation{
		{Layer: 0, Primitive: plan.PrimGEMM, Format: bf16F, Impl: bf16K},
		{Layer: 1, Primitive: plan.PrimGEMM, Format: fp8F, Impl: fp8K},
	}
	const bf16Ramp = "gemm_eps_max_bf16"
	base, err := New(in)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, ok := selectedMatch(base, bf16Ramp); !ok {
		t.Fatalf("the unkeyed registry does not price gpt-oss-120b with %s", bf16Ramp)
	}
	k, err := New(keyMeasured(in, map[string]plan.KernelImpl{bf16Ramp: bf16K}))
	if err != nil {
		t.Fatal(err)
	}
	if m, run, ok := selectedMatch(k, bf16Ramp); !ok || m != string(coefficient.KernelExact) || run != "cublas/gemm_bf16/aaaaaaaaaaaa" {
		t.Errorf("%s keyed to the BF16 GEMMs' kernel: selected %v, match %q, matched against %q", bf16Ramp, ok, m, run)
	}
	if pricesOf(k) != pricesOf(base) {
		t.Error("keying the ramp to the kernel that runs the BF16 GEMMs changed a price")
	}
	// Keyed to the kernel that runs the FP8 GEMM, it is not used for the BF16 ones.
	if k, err := New(keyMeasured(in, map[string]plan.KernelImpl{bf16Ramp: fp8K})); err == nil {
		if _, _, ok := selectedMatch(k, bf16Ramp); ok {
			t.Error("the BF16 ramp keyed to the FP8 GEMM's kernel priced the BF16 GEMMs")
		}
	}
	// Pooled, the same key is refused: the primitive runs two kernels.
	pooled := keyMeasured(in, map[string]plan.KernelImpl{bf16Ramp: bf16K})
	pooled.Operations = nil
	pooled.Implementations = map[plan.Primitive][]plan.KernelImpl{plan.PrimGEMM: {bf16K, fp8K}}
	if k, err := New(pooled); err == nil {
		if _, _, ok := selectedMatch(k, bf16Ramp); ok {
			t.Error("the pooled form priced the BF16 ramp from a value valid for one of two kernels")
		}
	}
}

// The part-wide attention pair is the fallback for a kind with no pair of its own
// (step.go: attentionByKind), so it is read only by the layers whose kind has none. With the
// sliding-window layers priced from their own pair, a part-wide pair keyed to the kernel that
// runs full attention alone prices the full layers, and is matched against that kernel only.
func TestThePartWidePairIsReadOnlyByTheLayersWithNoPairOfTheirOwn(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	var in Inputs
	for _, c := range perturbationScenarios(t, d, "gpt-oss-120b") {
		if strings.HasPrefix(c.label, "gpt-oss-120b/h200/tp") {
			in = c.in
			break
		}
	}
	if in.Model == nil {
		t.Fatal("no gpt-oss-120b scenario builds on h200")
	}
	full := plan.KernelImpl{Library: "flashinfer", Variant: "decode_full", CodeHash: "aaaaaaaaaaaa"}
	win := plan.KernelImpl{Library: "flashinfer", Variant: "decode_window", CodeHash: "bbbbbbbbbbbb"}
	kinds := map[string]model.LayerKind{}
	for _, lk := range in.Model.LayerKinds {
		kinds[lk.ID] = lk
	}
	for i, id := range in.Model.Stack.Expand() {
		impl := full
		for _, n := range kinds[id].Nodes {
			if n.Op == model.OpAttention && n.AttentionKind == model.AttentionSWA {
				impl = win
			}
		}
		in.Operations = append(in.Operations, plan.Operation{Layer: i, Primitive: plan.PrimAttentionDecode, Impl: impl})
	}
	base, err := New(in)
	if err != nil {
		t.Fatal(err)
	}
	k, err := New(keyMeasured(in, map[string]plan.KernelImpl{"attention_decode_rate_swa": win, "attention_decode_floor_swa": win,
		"attention_decode_rate": full, "attention_decode_floor": full}))
	if err != nil {
		t.Fatalf("the part-wide pair keyed to full attention's kernel, the swa pair to the windowed one: %v", err)
	}
	for _, name := range []string{"attention_decode_rate", "attention_decode_floor"} {
		if m, run, ok := selectedMatch(k, name); !ok || m != string(coefficient.KernelExact) || run != "flashinfer/decode_full/aaaaaaaaaaaa" {
			t.Errorf("%s: selected %v, match %q, matched against %q; want exact against full attention's kernel alone", name, ok, m, run)
		}
	}
	if pricesOf(k) != pricesOf(base) {
		t.Error("keying the pairs to the kernels that run their layers changed a price")
	}
}
