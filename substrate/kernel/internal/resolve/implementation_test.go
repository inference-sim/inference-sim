package resolve

import (
	"errors"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
	"github.com/inference-sim/blis-substrate/validate/dataset"
)

// VAL-8: a lookup matches a coefficient only for the exact kernel implementation that executes
// the primitive the coefficient prices.

const (
	// attn prices the attention.decode primitive; gemm prices the gemm primitive.
	attn = "attention_decode_rate"
	gemm = "gemm_eps_max_bf16"
)

var (
	nccl = func(hash string) coefficient.KernelImpl {
		return coefficient.KernelImpl{Library: "nccl", Variant: "ncclAllReduce", CodeHash: hash}
	}
	hashA, hashB = nccl("aaaaaaaaaaaa"), nccl("bbbbbbbbbbbb")
	// kernelA runs attention in the mixed plans; kernelB runs the GEMMs.
	kernelA = coefficient.KernelImpl{Library: "flashinfer", Variant: "decode", CodeHash: "cccccccccccc"}
	kernelB = coefficient.KernelImpl{Library: "cublas", Variant: "gemm", CodeHash: "dddddddddddd"}
)

// h100 is the scope the review's regression used.
var h100 = coefficient.Scope{Hardware: []string{"h100"}}

func entryNamed(name string, method vocab.Method, v float64, impls ...coefficient.KernelImpl) coefficient.Entry {
	e := coefficient.Entry{Name: name, Value: v, Units: vocab.UnitDimensionless, Method: method,
		Fitted: method == vocab.MethodMeasured, Scope: h100, KernelImpls: impls}
	if method == vocab.MethodAssumed {
		e.Rationale = "r"
	}
	return e
}

func measuredKeyed(v float64, impls ...coefficient.KernelImpl) coefficient.Entry {
	return entryNamed(attn, vocab.MethodMeasured, v, impls...)
}

// transferredKeyed is a value measured on source and moved to target by a stated rule.
func transferredKeyed(v float64, target, source coefficient.KernelImpl) coefficient.Entry {
	e := entryNamed(attn, vocab.MethodTransferred, v, target)
	e.MeasuredOn = []coefficient.KernelImpl{source}
	e.TransferredFrom, e.TransferRule, e.Rationale = "hardware=h100", "same kernel family, new build", "r"
	return e
}

func assumedFallback(v float64) coefficient.Entry { return entryNamed(attn, vocab.MethodAssumed, v) }

func setOf(es ...coefficient.Entry) []*coefficient.Set {
	return []*coefficient.Set{{Kind: "CoefficientSet", Name: "s", Coefficients: es}}
}

// on says what attention.decode runs; with a second argument list it also says what gemm runs.
func on(impls ...coefficient.KernelImpl) Scope {
	if len(impls) == 0 {
		return Scope{Hardware: "h100"}
	}
	return Scope{Hardware: "h100", Implementations: map[string][]coefficient.KernelImpl{"attention.decode": impls}}
}

// mixed is a pool that runs kernel A for attention and kernel B for the GEMMs.
func mixed() Scope {
	return Scope{Hardware: "h100", Implementations: map[string][]coefficient.KernelImpl{
		"attention.decode": {kernelA}, "gemm": {kernelB}}}
}

// valueAt looks name up for a deployment and returns the entry's value and how it matched.
func valueAt(t *testing.T, name string, sets []*coefficient.Set, d Scope) (float64, coefficient.KernelMatch, error) {
	t.Helper()
	c, err := Load(sets, d)
	if err != nil {
		return 0, "", err
	}
	v, err := c.Value(name)
	if err != nil {
		return 0, "", err
	}
	r, _ := c.Entry(name)
	return v, r.Match, nil
}

// The review's regression: an H100 measured entry keyed to aaaaaaaaaaaa was returned for a
// target at bbbbbbbbbbbb.
func TestAMeasuredEntryIsNotReturnedForAnotherBuildOfItsKernel(t *testing.T) {
	sets := setOf(measuredKeyed(1, hashA))
	if v, m, err := valueAt(t, attn, sets, on(hashA)); err != nil || v != 1 || m != coefficient.KernelExact {
		t.Fatalf("the build the entry is keyed to: value %v match %q err %v", v, m, err)
	}
	_, _, err := valueAt(t, attn, sets, on(hashB))
	if err == nil {
		t.Fatalf("the entry keyed to aaaaaaaaaaaa was returned for bbbbbbbbbbbb")
	}
	for _, want := range []string{"VAL-8", "aaaaaaaaaaaa", "bbbbbbbbbbbb"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the error should name %q:\n%v", want, err)
		}
	}
}

// The second review: in a pool that runs kernels A and B, an entry valid only for A must not
// price a primitive lowered to B, and the reverse.
func TestAnEntryKeyedToOneKernelCannotPriceAPrimitiveLoweredToAnother(t *testing.T) {
	// attn is priced by kernel A and gemm by kernel B. An attention entry keyed only to B (the
	// GEMM kernel) and a GEMM entry keyed only to A (the attention kernel) both run in the pool.
	wrongA := entryNamed(attn, vocab.MethodMeasured, 1, kernelB)
	wrongG := entryNamed(gemm, vocab.MethodMeasured, 2, kernelA)
	sets := setOf(wrongA, wrongG)
	c, err := Load(sets, mixed())
	if err == nil {
		for _, name := range []string{attn, gemm} {
			if v, e := c.Value(name); e == nil {
				t.Errorf("%s was priced (%v) by an entry keyed to a kernel that does not execute that primitive", name, v)
			}
		}
	}
	// Keyed to the kernel that executes each primitive, both are used.
	right := setOf(entryNamed(attn, vocab.MethodMeasured, 1, kernelA), entryNamed(gemm, vocab.MethodMeasured, 2, kernelB))
	c, err = Load(right, mixed())
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]float64{attn: 1, gemm: 2} {
		if v, err := c.Value(name); err != nil || v != want {
			t.Errorf("%s: %v %v, want %v", name, v, err, want)
		}
		if r, _ := c.Entry(name); r.Match != coefficient.KernelExact || r.Primitive == "" || len(r.Running) != 1 {
			t.Errorf("%s: match %q primitive %q running %v", name, r.Match, r.Primitive, r.Running)
		}
	}
	// A primitive lowered to B with only an A-keyed entry falls to the next entry, and to a
	// refusal that names what was refused when there is none.
	fall := setOf(entryNamed(gemm, vocab.MethodMeasured, 2, kernelA), entryNamed(gemm, vocab.MethodAssumed, 9),
		entryNamed(attn, vocab.MethodMeasured, 1, kernelA))
	c, err = Load(fall, mixed())
	if err != nil {
		t.Fatal(err)
	}
	if v, err := c.Value(gemm); err != nil || v != 9 {
		t.Errorf("gemm lowered to B with an A-keyed measured entry: %v %v, want the assumed 9", v, err)
	}
	only := setOf(entryNamed(gemm, vocab.MethodMeasured, 2, kernelA), entryNamed(attn, vocab.MethodMeasured, 1, kernelA))
	c, err = Load(only, mixed())
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Value(gemm)
	var missing *MissingError
	if !errors.As(err, &missing) || !strings.Contains(missing.Refused, "cccccccccccc") ||
		!strings.Contains(missing.Refused, "dddddddddddd") || !strings.Contains(missing.Refused, "gemm") {
		t.Errorf("want a MissingError naming the refused A-keyed entry and the kernel B that runs gemm, got %v", err)
	}
}

// One primitive lowered to two kernels: the kernel resolves one value per name, so an entry
// valid for A alone cannot price the layers lowered to B. An entry that lists both can.
func TestAPrimitiveLoweredToTwoKernelsNeedsAnEntryThatCoversBoth(t *testing.T) {
	both := Scope{Hardware: "h100", Implementations: map[string][]coefficient.KernelImpl{
		"attention.decode": {kernelA, hashA}}}
	if _, _, err := valueAt(t, attn, setOf(measuredKeyed(1, kernelA)), both); err == nil {
		t.Errorf("an entry valid for A alone priced a primitive that also runs another kernel")
	}
	if _, _, err := valueAt(t, attn, setOf(measuredKeyed(1, hashA)), both); err == nil {
		t.Errorf("an entry valid for the second kernel alone priced a primitive that also runs the first")
	}
	if v, m, err := valueAt(t, attn, setOf(measuredKeyed(3, kernelA, hashA)), both); err != nil || v != 3 || m != coefficient.KernelExact {
		t.Errorf("an entry that lists both: %v %q %v", v, m, err)
	}
}

// No sibling fallback: the same hash of another variant, another library, and the same kernel
// with no hash are all refused.
func TestNoSiblingImplementationMatches(t *testing.T) {
	sets := setOf(measuredKeyed(1, hashA))
	for name, running := range map[string][]coefficient.KernelImpl{
		"another variant":                     {{Library: "nccl", Variant: "ncclAllGather", CodeHash: "aaaaaaaaaaaa"}},
		"another library":                     {{Library: "rccl", Variant: "ncclAllReduce", CodeHash: "aaaaaaaaaaaa"}},
		"the kernel with no identified build": {{Library: "nccl", Variant: "ncclAllReduce"}},
		"nothing stated":                      nil,
	} {
		if _, _, err := valueAt(t, attn, sets, on(running...)); err == nil {
			t.Errorf("%s: the entry keyed to aaaaaaaaaaaa was returned", name)
		}
	}
	// The kernel runs, but for another primitive.
	other := Scope{Hardware: "h100", Implementations: map[string][]coefficient.KernelImpl{"gemm": {hashA}}}
	if _, _, err := valueAt(t, attn, sets, other); err == nil {
		t.Errorf("an attention entry was returned because its kernel runs the GEMMs")
	}
}

// When another entry of the lookup survives, a refused name is missing from the resolved set and
// the error says which entries were refused and why.
func TestAMissingCoefficientNamesTheEntriesRefusedForItsImplementation(t *testing.T) {
	other := measuredKeyed(3)
	other.Name = "attention_decode_floor"
	c, err := Load(setOf(measuredKeyed(1, hashA), other), on(hashB))
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Value(attn)
	var missing *MissingError
	if !errors.As(err, &missing) || !strings.Contains(missing.Refused, "aaaaaaaaaaaa") ||
		!strings.Contains(err.Error(), "bbbbbbbbbbbb") {
		t.Fatalf("got %v", err)
	}
}

// A refused entry lets the next entry of the lookup win: the design's fallback, at worst the
// family's assumed entry, and the price records that method (VAL-3).
func TestARefusedEntryFallsToTheNextRankNotToASibling(t *testing.T) {
	sets := setOf(measuredKeyed(1, hashA), assumedFallback(7))
	v, m, err := valueAt(t, attn, sets, on(hashB))
	if err != nil || v != 7 || m != coefficient.KernelUnkeyed {
		t.Fatalf("got value %v match %q err %v, want the assumed 7 unkeyed", v, m, err)
	}
	v, m, err = valueAt(t, attn, sets, on(hashA))
	if err != nil || v != 1 || m != coefficient.KernelExact {
		t.Fatalf("got value %v match %q err %v, want the measured 1 exact", v, m, err)
	}
	c, _ := Load(sets, on(hashB))
	got := c.Refused()
	if len(got) != 1 || got[0].Entry.Value != 1 || len(got[0].Running) != 1 || got[0].Running[0] != hashB {
		t.Fatalf("the refusal is recorded with the build that ran the primitive: %+v", got)
	}
}

// Transferred evidence for a known target: measured on one build, valid for another, matching
// only that other. The measured_on source is never matched.
func TestATransferredEntryMatchesOnlyItsTarget(t *testing.T) {
	hashC := nccl("cccccccccccc")
	sets := setOf(transferredKeyed(2, hashB, hashA), assumedFallback(7))
	if v, m, err := valueAt(t, attn, sets, on(hashB)); err != nil || v != 2 || m != coefficient.KernelExact {
		t.Fatalf("the target: value %v match %q err %v", v, m, err)
	}
	for name, d := range map[string]Scope{"the source build": on(hashA), "a third build": on(hashC), "unset": on()} {
		if v, _, err := valueAt(t, attn, sets, d); err != nil || v != 7 {
			t.Errorf("%s: value %v err %v, want the assumed 7: the transferred value is valid for its target only", name, v, err)
		}
	}
}

// Two builds of a kernel are two coefficients: each matches its own and the other is refused,
// in both load orders.
func TestEachBuildOfAKernelGetsItsOwnValue(t *testing.T) {
	ea, eb := measuredKeyed(1, hashA), measuredKeyed(2, hashB)
	for _, es := range [][]coefficient.Entry{{ea, eb}, {eb, ea}} {
		sets := setOf(es...)
		for hash, want := range map[coefficient.KernelImpl]float64{hashA: 1, hashB: 2} {
			if v, _, err := valueAt(t, attn, sets, on(hash)); err != nil || v != want {
				t.Errorf("build %s: value %v err %v, want %v", hash.CodeHash, v, err, want)
			}
		}
	}
}

// Unset: only an entry that states no implementation matches, and it is reported unkeyed, so a
// reader never counts it as shown to match the kernel that runs. Every price that did not
// depend on an implementation is unchanged.
func TestWithNoImplementationStatedOnlyUnkeyedEntriesMatch(t *testing.T) {
	y := measuredKeyed(3)
	y.Name = "attention_decode_floor"
	sets := setOf(measuredKeyed(1, hashA), y)
	c, err := Load(sets, on())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Value(attn); err == nil {
		t.Errorf("the keyed entry was returned with no implementation stated")
	}
	if v, err := c.Value("attention_decode_floor"); err != nil || v != 3 {
		t.Errorf("an unkeyed entry: %v %v", v, err)
	}
	if r, _ := c.Entry("attention_decode_floor"); r.Match != coefficient.KernelUnkeyed {
		t.Errorf("match = %q, want unkeyed", r.Match)
	}
	// With an implementation stated, the unkeyed entry still applies to it and is still unkeyed.
	c, _ = Load(sets, on(hashB))
	if r, ok := c.Entry("attention_decode_floor"); !ok || r.Match != coefficient.KernelUnkeyed {
		t.Errorf("an unkeyed entry under a stated implementation: %+v %v", r, ok)
	}
}

// X15: the registry validator and the lookup agree about which keyed entries can be chosen
// between. For every pair of entries of one name and scope that differ in their kernel key and
// method, and every running set a primitive can have, the lookup refuses exactly when the
// validator flags the pair, in both load orders.
func TestLookupAgreesWithTheValidatorOnKernelKeys(t *testing.T) {
	keys := [][]coefficient.KernelImpl{nil, {hashA}, {hashB}, {kernelA}, {hashA, kernelA}, {hashB, kernelA}}
	// Running sets one primitive can have: nothing, one build, or two kernels together.
	runnings := [][]coefficient.KernelImpl{nil, {hashA}, {hashB}, {kernelA}, {hashA, kernelA}, {hashB, kernelA}}
	var accepted, refused, apart int
	for i, ka := range keys {
		for _, kb := range keys[i:] {
			for _, bMethod := range []vocab.Method{vocab.MethodMeasured, vocab.MethodTransferred} {
				a, b := measuredKeyed(1, ka...), measuredKeyed(2, kb...)
				b.Method = bMethod
				sa := &coefficient.Set{Kind: "CoefficientSet", Name: "a", Coefficients: []coefficient.Entry{a}}
				sb := &coefficient.Set{Kind: "CoefficientSet", Name: "b", Coefficients: []coefficient.Entry{b}}
				flagged := len(dataset.CheckOrdered(map[string]*coefficient.Set{"a": sa, "b": sb})) > 0
				witnessed := false
				for _, run := range runnings {
					d := Scope{Hardware: "h100"}
					if run != nil {
						d.Implementations = map[string][]coefficient.KernelImpl{"attention.decode": run}
					}
					both := a.MatchKernels(d.Implementations) != coefficient.KernelRefused &&
						b.MatchKernels(d.Implementations) != coefficient.KernelRefused
					if !both {
						apart++
						continue
					}
					witnessed = true
					for _, order := range [][]*coefficient.Set{{sa, sb}, {sb, sa}} {
						_, err := Load(order, d)
						if (err != nil) != flagged {
							t.Fatalf("keys %v and %v running %v: the validator flags the pair = %v, the lookup error = %v",
								ka, kb, run, flagged, err)
						}
					}
					if flagged {
						refused++
					} else {
						accepted++
					}
				}
				// The other direction: a pair the validator flags must be one some pool can match
				// together, or the registry refuses data that no lookup would find ambiguous.
				if flagged && !witnessed {
					t.Fatalf("keys %v and %v: the validator flags a pair that no running set matches together", ka, kb)
				}
			}
		}
	}
	if accepted == 0 || refused == 0 || apart == 0 {
		t.Fatalf("not exercised: %d accepted, %d refused, %d not matching one running set", accepted, refused, apart)
	}
}
