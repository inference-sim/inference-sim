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
