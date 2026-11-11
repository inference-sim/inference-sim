package latencykernel

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/units"
	"github.com/inference-sim/blis-substrate/load"
)

// VAL-8: the trtllm-gen prefill lane is priced from the kernel implementation the caller reports
// for the prefill primitive, and from nothing an engine setting says. These tests price
// MiniMax-M3 (full and sliding-window attention) on the data-center Blackwell parts that carry the
// lane, and on one that does not.

// laneCase is a built perturbation scenario for the model on the chip, with the engine settings
// the perturbation chose and the tensor-parallel width it priced.
func laneCase(t *testing.T, d *load.Data, chip string) Inputs {
	t.Helper()
	for _, c := range perturbationScenarios(t, d, "minimax-m3") {
		if strings.HasPrefix(c.label, "minimax-m3/"+chip+"/tp4/dp1/ept") || strings.HasPrefix(c.label, "minimax-m3/"+chip+"/tp4/dp1/epf") {
			return c.in
		}
	}
	t.Fatalf("no minimax-m3 scenario builds on %s at tp 4", chip)
	return Inputs{}
}

// quantCase is a built perturbation scenario for the catalog model on the chip.
func quantCase(t *testing.T, d *load.Data, name, chip string) Inputs {
	t.Helper()
	for _, c := range perturbationScenarios(t, d, name) {
		if strings.HasPrefix(c.label, name+"/"+chip+"/tp") {
			return c.in
		}
	}
	t.Fatalf("no %s scenario builds on %s", name, chip)
	return Inputs{}
}

func flag(b bool) *bool { return &b }

// The registry states no kernel hash for the trtllm-gen prefill (none is published), so its lane
// is not priced. These tests key the lane to fixture implementations, in test data only: the
// hashes below are not any real kernel's. The bf16 pair is valid for laneBF16 and the fp8 pair
// for laneFP8, two builds of one library.
var (
	laneBF16 = plan.KernelImpl{Library: "flashinfer", Variant: "fixture-trtllm-prefill-bf16", CodeHash: "0123456789ab"}
	laneFP8  = plan.KernelImpl{Library: "flashinfer", Variant: "fixture-trtllm-prefill-fp8", CodeHash: "123456789abc"}
	// laneDecode is a decode kernel of the same library and the same hash digits as laneBF16: a pool
	// that runs it for decode must not admit a pair that prices prefill.
	laneDecode = plan.KernelImpl{Library: "flashinfer", Variant: "fixture-trtllm-decode", CodeHash: laneBF16.CodeHash}
	// laneSource stands for the build the 0.25.0 sweep ran, which the entries record as the
	// implementation their data came from.
	laneSource = coefficient.KernelImpl{Library: "flashinfer", Variant: "fixture-trtllm-prefill-bf16", CodeHash: "ba9876543210"}
)

func asKey(i plan.KernelImpl) coefficient.KernelImpl {
	return coefficient.KernelImpl{Library: i.Library, Variant: i.Variant, CodeHash: i.CodeHash}
}

// keyedLane states, on the lane's entries of the scenario's sets, that the bf16 pair is valid for
// laneBF16 and the fp8 pair for laneFP8, measured on the fixture source, and reports prefill as
// the implementations of the attention.prefill primitive.
func keyedLane(in Inputs, prefill ...plan.KernelImpl) Inputs {
	return withPrefill(keyLaneEntries(in, map[string]plan.KernelImpl{"bf16": laneBF16, "fp8": laneFP8}), prefill...)
}

// keyLaneEntries keys the pair of each cache dtype to the implementation given for it.
func keyLaneEntries(in Inputs, byDType map[string]plan.KernelImpl) Inputs {
	out := make([]*coefficient.Set, len(in.Coefficients))
	for i, s := range in.Coefficients {
		c := *s
		c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
		for j := range c.Coefficients {
			e := &c.Coefficients[j]
			for dtype, impl := range byDType {
				if slices.Contains(laneNames(dtype), e.Name) {
					e.KernelImpls = []coefficient.KernelImpl{asKey(impl)}
					e.MeasuredOn = []coefficient.KernelImpl{laneSource}
				}
			}
		}
		out[i] = &c
	}
	in.Coefficients = out
	return in
}

// withPrefill reports the implementations of the attention.prefill primitive, leaving the other
// primitives as they were; none removes the primitive.
func withPrefill(in Inputs, impls ...plan.KernelImpl) Inputs {
	return withRunning(in, plan.PrimAttentionPrefill, impls...)
}

// withRunning sets the implementations the pool reports for one primitive (none removes it).
func withRunning(in Inputs, prim plan.Primitive, impls ...plan.KernelImpl) Inputs {
	m := map[plan.Primitive][]plan.KernelImpl{}
	for k, v := range in.Implementations {
		m[k] = v
	}
	if len(impls) == 0 {
		delete(m, prim)
	} else {
		m[prim] = impls
	}
	in.Implementations = m
	return in
}

// laneAssumed reports whether the kernel recorded that the lane was not priced, and why.
func laneAssumed(k *Kernel) (string, bool) {
	for _, a := range k.assumptions {
		if a.Name == "trtllm_gen_prefill_lane" {
			return a.Scope, true
		}
	}
	return "", false
}

// prefillAndDecode prices a prefill step and a decode step separately, each as one comparable
// string.
func prefillAndDecode(k *Kernel) (prefill, decode string) {
	price := func(r kernel.ReqShape) string {
		e := k.StepTime(kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{r}})
		return fmt.Sprintf("%d %d", e.Overlap, e.NoOverlap)
	}
	return price(kernel.ReqShape{Scheduled: 2048, PromptLen: 2048}),
		price(kernel.ReqShape{Scheduled: 1, Computed: 4095, PromptLen: 4096})
}

func withEngine(in Inputs, f func(*enginesettings.Settings)) Inputs {
	e := in.Engine.Clone()
	f(&e)
	in.Engine = &e
	return in
}

func mustKernel(t *testing.T, in Inputs) *Kernel {
	t.Helper()
	k, err := New(in)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return k
}

// scaleEntry returns the scenario's sets with every entry of the given name scaled by f.
func scaleEntry(in Inputs, name string, f float64) Inputs {
	out := make([]*coefficient.Set, len(in.Coefficients))
	for i, s := range in.Coefficients {
		c := *s
		c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
		for j := range c.Coefficients {
			if c.Coefficients[j].Name == name {
				c.Coefficients[j].Value *= f
				if iv := c.Coefficients[j].CI95; iv != nil {
					c.Coefficients[j].CI95 = &coefficient.Interval{Low: iv.Low * f, High: iv.High * f}
				}
			}
		}
		out[i] = &c
	}
	in.Coefficients = out
	return in
}

// laneNames are the lane's coefficient names at a cache dtype.
func laneNames(dtype string) []string {
	return []string{trtllmGenPrefillFloor + dtype, trtllmGenPrefillScale + dtype}
}

func withCache(in Inputs, dtype string, block int) Inputs {
	return withEngine(in, func(e *enginesettings.Settings) {
		e.CacheDType = dtype
		if block > 0 {
			e.BlockSize = block
		}
	})
}

// movedBy reports whether scaling any of the coefficients by 0.8 moves a prefill price of in.
func movedBy(t *testing.T, in Inputs, names []string) bool {
	t.Helper()
	want, _ := prefillAndDecode(mustKernel(t, in))
	for _, n := range names {
		if got, _ := prefillAndDecode(mustKernel(t, scaleEntry(in, n, 0.8))); got != want {
			return true
		}
	}
	return false
}

// The lane is chosen by the implementation the caller reports for the prefill primitive, and by
// nothing the settings say: not the cache dtype or the block size. The bf16
// pair is valid for one build and the fp8 pair for another, so the reported build picks the pair.
// Each lane coefficient moves the prices that read it and no others (the perturbation test of the
// layer).
func TestTRTLLMGenLaneIsChosenByTheReportedImplementation(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, chip := range []string{"b200", "b300", "gb200-nvl72", "gb300"} {
		t.Run(chip, func(t *testing.T) {
			raw := laneCase(t, d, chip)
			pooled := pricesOf(mustKernel(t, raw))
			bf16, fp8 := laneNames("bf16"), laneNames("fp8")

			onBF16 := keyedLane(raw, laneBF16)
			if pricesOf(mustKernel(t, onBF16)) == pooled {
				t.Error("the reported bf16 build runs and the lane is not read")
			}
			if !movedBy(t, onBF16, bf16) || movedBy(t, onBF16, fp8) {
				t.Error("the bf16 build must read the bf16 pair and not the fp8 one")
			}
			onFP8 := keyedLane(raw, laneFP8)
			if !movedBy(t, onFP8, fp8) || movedBy(t, onFP8, bf16) {
				t.Error("the fp8 build must read the fp8 pair and not the bf16 one")
			}
			// Nothing the settings say moves the choice: with an fp8 cache or a block size of 128 the
			// bf16 build still reads the bf16 pair and not the fp8 one. (The cache dtype changes
			// other prices, so the lane is detected by its coefficients.)
			for name, in := range map[string]Inputs{
				"an fp8 cache":     withCache(onBF16, "fp8", 0),
				"a block size 128": withCache(onBF16, "", 128),
			} {
				if !movedBy(t, in, bf16) || movedBy(t, in, fp8) {
					t.Errorf("%s: the choice of pair follows the setting, not the reported implementation", name)
				}
			}
			up, ud := prefillAndDecode(mustKernel(t, raw))
			sp, sd := prefillAndDecode(mustKernel(t, onBF16))
			if up == sp || ud != sd {
				t.Errorf("the lane moves prefill %v->%v and decode %v->%v; it prices prefill only", up, sp, ud, sd)
			}
		})
	}
}

// Latent attention is another kernel and keeps the pooled pair, so reporting the lane's build
// changes nothing on a model with no non-latent layer.
func TestTheLaneChangesNothingOnLatentOnlyModels(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	mla := quantCase(t, d, "deepseek-v3", "b200")
	want := pricesOf(mustKernel(t, mla))
	if got := pricesOf(mustKernel(t, keyedLane(mla, laneBF16))); got != want {
		t.Error("reporting the lane's build moves a latent-only model")
	}
	if movedBy(t, keyedLane(mla, laneBF16), append(laneNames("bf16"), laneNames("fp8")...)) {
		t.Error("the lane is read on a latent-only model")
	}
}

// The lane removed from the sets, or whose pairs list another build, is priced on the pooled pair
// and says why; the lane absent from the sets leaves the deployment on the pooled pair, silently.
func TestTheLaneAbsentOrKeyedToAnotherBuildLeavesThePooledPair(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	base := keyedLane(laneCase(t, d, "b200"), laneBF16)
	strip := func(in Inputs, names ...string) Inputs {
		var sets []*coefficient.Set
		for _, s := range in.Coefficients {
			c := *s
			c.Coefficients = nil
			for _, e := range s.Coefficients {
				if !slices.Contains(names, e.Name) {
					c.Coefficients = append(c.Coefficients, e)
				}
			}
			sets = append(sets, &c)
		}
		in.Coefficients = sets
		return in
	}
	all := append(laneNames("bf16"), laneNames("fp8")...)
	pooled := pricesOf(mustKernel(t, strip(base, all...)))
	// The bf16 pair removed: the fp8 pair is keyed to another build than the one reported.
	k := mustKernel(t, strip(base, laneNames("bf16")...))
	if pricesOf(k) != pooled {
		t.Error("a pair keyed to another build than the one reported prices the prefill")
	}
	why, ok := laneAssumed(k)
	if !ok {
		t.Fatal("the lane was not priced and no assumption says so")
	}
	for _, want := range []string{"trtllm-gen", trtllmGenPrefillFloor + "fp8", "blis-registry#73"} {
		if !strings.Contains(why, want) {
			t.Errorf("the assumption does not name %q: %s", want, why)
		}
	}
	// The lane absent from the sets: silent.
	k = mustKernel(t, strip(base, all...))
	if why, ok := laneAssumed(k); ok {
		t.Errorf("an assumption is recorded for a lane the sets do not carry: %s", why)
	}
}

// A model that mixes latent and plain layers prices only the plain ones on the lane: the MLA and
// sparse-MLA backends are other kernels.
func TestTheLaneIsForPlainLayersOnly(t *testing.T) {
	k := &Kernel{laneRuns: true, lanePair: prefillPair{20, 0.34}, attentionPrefillFloor: 16, attentionPrefillScale: 0.32}
	for _, kind := range []model.AttentionKind{model.AttentionMLA, model.AttentionSparseMLA} {
		if got := k.prefillFor(kind); got != (prefillPair{16, 0.32}) {
			t.Errorf("%s layers are priced on the lane: %v", kind, got)
		}
	}
	for _, kind := range []model.AttentionKind{model.AttentionGQA, model.AttentionSWA} {
		if got := k.prefillFor(kind); got != (prefillPair{20, 0.34}) {
			t.Errorf("%s layers are not priced on the lane: %v", kind, got)
		}
	}
}

// VAL-8: the lane is priced only when its pair lists the implementation of the primitive it
// prices. The known-bad inputs are the siblings (the same library and variant at another hash, the
// same hash under another variant, another library) and the mixed pool, where the entry is valid
// for a decode kernel A and prefill runs B. Each is refused through the real kernel.New.
func TestTRTLLMGenLaneIsPricedOnlyForTheImplementationOfThePrimitiveItPrices(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	base := withCache(laneCase(t, d, "b200"), "", 0)
	keyed := keyedLane(base, laneBF16)
	pooled := pricesOf(mustKernel(t, base))

	// The exact implementation: the lane prices, differently from the pooled pair, and the
	// selected entries report exact for the prefill primitive.
	{
		k := mustKernel(t, keyed)
		if pricesOf(k) == pooled {
			t.Error("the exact implementation runs and the lane is not read")
		}
		if why, ok := laneAssumed(k); ok {
			t.Errorf("the lane is priced and an assumption says it is not: %s", why)
		}
		found := false
		for _, o := range k.origins {
			if o.Name == trtllmGenPrefillFloor+"bf16" {
				found = true
				if o.KernelMatch != string(coefficient.KernelExact) || o.Primitive != "attention.prefill" {
					t.Errorf("%s is selected as %q for %q, want exact for attention.prefill", o.Name, o.KernelMatch, o.Primitive)
				}
			}
		}
		if !found {
			t.Error("the lane's floor is not among the selected entries")
		}
	}

	siblingHash := plan.KernelImpl{Library: laneBF16.Library, Variant: laneBF16.Variant, CodeHash: "ffffffffffff"}
	siblingVariant := plan.KernelImpl{Library: laneBF16.Library, Variant: "fixture-native-prefill", CodeHash: laneBF16.CodeHash}
	otherLibrary := plan.KernelImpl{Library: "other-lib", Variant: laneBF16.Variant, CodeHash: laneBF16.CodeHash}
	cases := []struct {
		name    string
		running map[plan.Primitive][]plan.KernelImpl
	}{
		{"a sibling hash", map[plan.Primitive][]plan.KernelImpl{plan.PrimAttentionPrefill: {siblingHash}}},
		{"a sibling variant", map[plan.Primitive][]plan.KernelImpl{plan.PrimAttentionPrefill: {siblingVariant}}},
		{"another library", map[plan.Primitive][]plan.KernelImpl{plan.PrimAttentionPrefill: {otherLibrary}}},
		{"no implementation reported", nil},
		// The mixed pool: the pair is valid for A and the pool runs A for decode and B for prefill.
		{"A for decode and B for prefill", map[plan.Primitive][]plan.KernelImpl{
			plan.PrimAttentionDecode: {laneBF16}, plan.PrimAttentionPrefill: {siblingVariant}}},
		// A decode-only report: the pair prices prefill and the pool says nothing about it.
		{"A for decode only", map[plan.Primitive][]plan.KernelImpl{plan.PrimAttentionDecode: {laneBF16}}},
		// Two prefill kernels: one value cannot price both.
		{"A and B for prefill", map[plan.Primitive][]plan.KernelImpl{plan.PrimAttentionPrefill: {laneBF16, siblingVariant}}},
	}
	for _, c := range cases {
		in := keyed
		in.Implementations = c.running
		// The pair is not priced for another implementation: the pooled pair prices the prefill, and
		// the assumption names the entry, its hash and what prefill runs.
		k, err := New(in)
		if err != nil {
			t.Errorf("%s: refused: %v", c.name, err)
			continue
		}
		if got := pricesOf(k); got != pooled {
			t.Errorf("%s: a pair valid for another implementation prices the prefill", c.name)
		}
		why, ok := laneAssumed(k)
		if !ok || !strings.Contains(why, "pooled") {
			t.Errorf("%s: the unpriced lane is not recorded as an assumption: %q", c.name, why)
		}
		for _, want := range []string{trtllmGenPrefillFloor + "bf16", laneBF16.CodeHash} {
			if ok && !strings.Contains(why, want) {
				t.Errorf("%s: the assumption does not name %q: %s", c.name, want, why)
			}
		}
	}

	// The regression of the review: the pairs are valid for a decode kernel A, and the pool runs A
	// for decode and a different kernel B for prefill. The pool-wide union holds A; the primitive
	// the pair prices does not.
	forA := keyLaneEntries(base, map[string]plan.KernelImpl{"bf16": laneDecode, "fp8": laneDecode})
	forA.Implementations = map[plan.Primitive][]plan.KernelImpl{
		plan.PrimAttentionDecode: {laneDecode}, plan.PrimAttentionPrefill: {laneBF16}}
	kA := mustKernel(t, forA)
	if got := pricesOf(kA); got != pooled {
		t.Error("a pair valid for the decode kernel only prices the prefill that runs B")
	}
	if why, ok := laneAssumed(kA); !ok || !strings.Contains(why, "attention.prefill runs") {
		t.Errorf("the lane valid for the decode kernel only is not recorded with what prefill runs: %q", why)
	}
	// Both pairs listing the build leave the cache dtype unknown.
	both := keyLaneEntries(keyed, map[string]plan.KernelImpl{"bf16": laneBF16, "fp8": laneBF16})
	if _, err := New(both); err == nil || !strings.Contains(err.Error(), "both list") {
		t.Errorf("both pairs list the reported build: %v", err)
	}
}

// The registry's lane states no implementation, so it is not priced whatever the pool reports: the
// pooled pair prices the prefill, the lane is recorded as not priced, and its entries move no price.
func TestTheCommittedTRTLLMGenLaneIsNotPricedWithoutAKernelHash(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, chip := range []string{"b200", "b300", "gb200-nvl72", "gb300"} {
		base := withCache(laneCase(t, d, chip), "", 0)
		for _, in := range []Inputs{base, withPrefill(base, laneBF16), withPrefill(base, laneFP8)} {
			k := mustKernel(t, in)
			if _, ok := laneAssumed(k); !ok {
				t.Errorf("%s: the unpriced lane is not recorded", chip)
			}
			if movedBy(t, in, append(laneNames("bf16"), laneNames("fp8")...)) {
				t.Errorf("%s: an unkeyed lane entry moves a price", chip)
			}
		}
	}
}

// prefillOps reports the prefill of every layer of the model that has attention as one operation
// on the given kernel, reading a cache of the given format.
func prefillOps(t *testing.T, in Inputs, f units.QuantFormat, dtype plan.Dtype, impl plan.KernelImpl) []plan.Operation {
	t.Helper()
	kinds := map[string]model.LayerKind{}
	for _, lk := range in.Model.LayerKinds {
		kinds[lk.ID] = lk
	}
	var ops []plan.Operation
	for i, id := range in.Model.Stack.Expand() {
		for _, n := range kinds[id].Nodes {
			if n.Op == model.OpAttention && n.ReadsEngineKV() {
				ops = append(ops, plan.Operation{Layer: i, Primitive: plan.PrimAttentionPrefill, Format: f, Dtype: dtype, Impl: impl})
				break
			}
		}
	}
	if len(ops) == 0 {
		t.Fatal("the model has no attention layer")
	}
	return ops
}

// dtypeOf is the element type a test cache of the format holds: the sizes the tests use are bf16
// and fp8.
func dtypeOf(f units.QuantFormat) plan.Dtype {
	if f.PayloadBits == 8 {
		return "fp8"
	}
	return "bf16"
}

var (
	bf16Cache = units.QuantFormat{PayloadBits: 16}
	fp8Cache  = units.QuantFormat{PayloadBits: 8}
)

// One kernel serves both cache formats, so both pairs list it. The operations state the format
// of the cache each prefill reads, which picks the pair: bf16 reads the bf16 pair, fp8 the fp8
// pair, and neither is ambiguous. Two formats among the non-latent prefill are refused.
func TestOneKernelServingBothCacheFormatsPricesTheLaneOfTheFormatTheOperationsRead(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	raw := laneCase(t, d, "b200")
	both := keyLaneEntries(raw, map[string]plan.KernelImpl{"bf16": laneBF16, "fp8": laneBF16})
	pooled := pricesOf(mustKernel(t, raw))
	bf16, fp8 := laneNames("bf16"), laneNames("fp8")
	at := func(f units.QuantFormat) Inputs {
		in := both
		in.Operations = prefillOps(t, in, f, dtypeOf(f), laneBF16)
		return in
	}
	onBF16, onFP8 := at(bf16Cache), at(fp8Cache)
	if pricesOf(mustKernel(t, onBF16)) == pooled || pricesOf(mustKernel(t, onFP8)) == pooled {
		t.Fatal("a kernel both pairs list, reported with the cache format, does not price the lane")
	}
	if !movedBy(t, onBF16, bf16) || movedBy(t, onBF16, fp8) {
		t.Error("a bf16 cache must read the bf16 pair and not the fp8 one")
	}
	if !movedBy(t, onFP8, fp8) || movedBy(t, onFP8, bf16) {
		t.Error("an fp8 cache must read the fp8 pair and not the bf16 one")
	}
	// The pooled form cannot tell the formats apart and is refused, as before.
	pooledForm := both
	pooledForm.Implementations = map[plan.Primitive][]plan.KernelImpl{plan.PrimAttentionPrefill: {laneBF16}}
	if _, err := New(pooledForm); err == nil || !strings.Contains(err.Error(), "both list") {
		t.Errorf("both pairs listing the kernel reported only by primitive: %v", err)
	}
	// Two formats among the non-latent prefill.
	mixed := both
	ops := prefillOps(t, both, bf16Cache, "bf16", laneBF16)
	ops[len(ops)-1].Format, ops[len(ops)-1].Dtype = fp8Cache, "fp8"
	mixed.Operations = ops
	if _, err := New(mixed); err == nil || !strings.Contains(err.Error(), "element type") {
		t.Errorf("prefill reading two cache formats: %v", err)
	}
	// No format stated: the lane is not priced, and the pooled pair is recorded as pricing.
	none := both
	none.Operations = prefillOps(t, both, units.QuantFormat{}, "", laneBF16)
	k := mustKernel(t, none)
	if pricesOf(k) != pooled {
		t.Error("a prefill that states no cache format priced the lane")
	}
	if why, ok := laneAssumed(k); !ok || !strings.Contains(why, "element type") {
		t.Errorf("the unpriced lane is not recorded with its reason: %q", why)
	}
}

// A pool with latent and non-latent layers: the non-latent prefill runs trtllm-gen (kernel B)
// and the latent prefill another kernel (A). The lane's pair lists B alone: it is read by the
// non-latent layers only, so it is exact, where a union over the primitive's kernels would have
// refused it.
func TestALaneEntryForTheNonLatentKernelIsExactBesideALatentPrefillOnAnother(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	raw := laneCase(t, d, "b200")
	// The model with one of its attention kinds made latent: a graph the catalog does not hold,
	// since no catalog model mixes the two, but the binding does not depend on that.
	g, err := raw.Model.ForCapability(10)
	if err != nil {
		t.Fatal(err)
	}
	c := *g
	c.LayerKinds = append([]model.LayerKind(nil), g.LayerKinds...)
	latentKind := ""
	for i := range c.LayerKinds {
		c.LayerKinds[i].Nodes = append([]model.Node(nil), g.LayerKinds[i].Nodes...)
		if latentKind != "" {
			continue
		}
		for j, n := range c.LayerKinds[i].Nodes {
			if n.Op == model.OpAttention && n.ReadsEngineKV() {
				c.LayerKinds[i].Nodes[j].AttentionKind, c.LayerKinds[i].Nodes[j].NumKVHeads = model.AttentionMLA, 1
				latentKind = c.LayerKinds[i].ID
				break
			}
		}
	}
	if latentKind == "" {
		t.Fatal("no attention node to make latent")
	}
	in := raw
	in.Model = &c
	latentK := plan.KernelImpl{Library: "flashinfer", Variant: "mla_prefill", CodeHash: "abababababab"}
	var ops []plan.Operation
	nonLatent, latent := 0, 0
	for i, id := range c.Stack.Expand() {
		for _, lk := range c.LayerKinds {
			if lk.ID != id {
				continue
			}
			for _, n := range lk.Nodes {
				if n.Op != model.OpAttention || !n.ReadsEngineKV() {
					continue
				}
				impl := laneBF16
				if id == latentKind {
					impl = latentK
					latent++
				} else {
					nonLatent++
				}
				ops = append(ops, plan.Operation{Layer: i, Primitive: plan.PrimAttentionPrefill, Format: bf16Cache, Dtype: "bf16", Impl: impl})
			}
		}
	}
	if latent == 0 || nonLatent == 0 {
		t.Fatalf("the pool needs both kinds, has %d latent and %d non-latent layers", latent, nonLatent)
	}
	in = keyLaneEntries(in, map[string]plan.KernelImpl{"bf16": laneBF16, "fp8": laneFP8})
	in.Operations = ops
	k, err := New(in)
	if err != nil {
		t.Fatalf("the catalog graph with an attention kind made latent does not build: %v", err)
	}
	if why, ok := laneAssumed(k); ok {
		t.Fatalf("the non-latent prefill runs the kernel the lane lists, and the lane is not priced: %s", why)
	}
	if !k.laneRuns {
		t.Error("the lane is not priced for the non-latent layers")
	}
	// Reported by primitive alone, the primitive runs two kernels and the B-only pair is refused.
	pooledForm := in
	pooledForm.Operations = nil
	pooledForm.Implementations = map[plan.Primitive][]plan.KernelImpl{plan.PrimAttentionPrefill: {laneBF16, latentK}}
	if k, err := New(pooledForm); err == nil {
		if _, ok := laneAssumed(k); !ok {
			t.Error("the pooled form priced the lane from a pair that lists one of two prefill kernels")
		}
	}
}

// QuantFormat is a storage size: a bf16 cache and an fp16 cache are both 16 bits. The lane was
// fitted on a bfloat16 cache, so a float16 cache must not consume the pair that is listed for the
// kernel that runs it: the lane is left unpriced, the part-wide family prices the prefill, and the
// assumption names the element type. The same kernel at a bf16 cache is priced from the lane.
func TestAnFP16CacheDoesNotConsumeTheBF16MeasuredLane(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	raw := laneCase(t, d, "b200")
	pooled := pricesOf(mustKernel(t, raw))
	keyed := keyLaneEntries(raw, map[string]plan.KernelImpl{"bf16": laneBF16, "fp8": laneBF16})
	at := func(dtype plan.Dtype) Inputs {
		in := keyed
		in.Operations = prefillOps(t, in, bf16Cache, dtype, laneBF16) // 16 bits whichever element type
		return in
	}
	if pricesOf(mustKernel(t, at("bf16"))) == pooled {
		t.Fatal("a bf16 cache is not priced from the lane that lists its kernel")
	}
	for _, dtype := range []plan.Dtype{"fp16", ""} {
		k := mustKernel(t, at(dtype))
		if pricesOf(k) != pooled || k.laneRuns {
			t.Errorf("a %q cache consumed the bf16-measured lane", dtype)
		}
		if dtype == "fp16" {
			if why, ok := laneAssumed(k); !ok || !strings.Contains(why, "fp16") {
				t.Errorf("the unpriced lane does not name the element type: %q", why)
			}
		}
	}
	// 8 bits that are not fp8 are outside the lane too.
	in := keyed
	in.Operations = prefillOps(t, in, fp8Cache, "int8", laneBF16)
	if k := mustKernel(t, in); k.laneRuns || pricesOf(k) != pooled {
		t.Error("an int8 cache consumed the fp8-measured lane")
	}
	// A mix of a lane element type and another is refused: the lane has one pair per kind.
	mixed := keyed
	ops := prefillOps(t, keyed, bf16Cache, "bf16", laneBF16)
	ops[len(ops)-1].Dtype = "fp16"
	mixed.Operations = ops
	if _, err := New(mixed); err == nil || !strings.Contains(err.Error(), "more than one element type") {
		t.Errorf("bf16 and fp16 layers together: %v", err)
	}
}

// The plan's dtype words are the catalog's: every catalog dtype is a plan dtype with the same
// payload size, and every plan dtype is a catalog dtype.
func TestThePlanDtypesAreTheCatalogsDtypes(t *testing.T) {
	catalog := []model.DType{model.DTypeBF16, model.DTypeFP16, model.DTypeFP8, model.DTypeNVFP4,
		model.DTypeMXFP4, model.DTypeINT8, model.DTypeINT4, model.DTypeFP32}
	if len(plan.Dtypes()) != len(catalog) {
		t.Errorf("%d plan dtypes, %d catalog dtypes", len(plan.Dtypes()), len(catalog))
	}
	for _, c := range catalog {
		d := plan.Dtype(c)
		if !d.Valid() || float64(d.Bits()) != c.Bytes()*8 {
			t.Errorf("catalog dtype %s: plan says %d bits, valid %v; the catalog %g bits", c, d.Bits(), d.Valid(), c.Bytes()*8)
		}
	}
	for _, d := range plan.Dtypes() {
		if !model.DType(d).Valid() {
			t.Errorf("plan dtype %s is not a catalog dtype", d)
		}
	}
}

// The kernel does not rely on the host having validated the plan: operations whose cache element
// type is not a catalog dtype, contradicts the stated format, or is set on a primitive that has no
// cache are refused at construction, and no price is made from the label.
func TestOperationsWithAMalformedDtypeAreRefused(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	keyed := keyLaneEntries(laneCase(t, d, "b200"), map[string]plan.KernelImpl{"bf16": laneBF16, "fp8": laneFP8})
	for name, c := range map[string]struct {
		dtype  plan.Dtype
		format units.QuantFormat
		want   string
	}{
		"a word that is not a dtype":  {"bf116", bf16Cache, "not a catalog dtype"},
		"bf16 beside an 8-bit format": {"bf16", fp8Cache, "different caches"},
	} {
		in := keyed
		in.Operations = prefillOps(t, in, c.format, c.dtype, laneBF16)
		if _, err := New(in); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want a refusal containing %q", name, err, c.want)
		}
	}
	in := keyed
	in.Operations = append(prefillOps(t, in, bf16Cache, "bf16", laneBF16), plan.Operation{Layer: 0, Primitive: plan.PrimGEMM, Dtype: "bf16", Impl: laneBF16})
	if _, err := New(in); err == nil || !strings.Contains(err.Error(), "only an attention primitive") {
		t.Errorf("a dtype on a GEMM: %v", err)
	}
}
