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

// One primitive lowered to two kernels, with the pool given only as the primitive's pooled
// implementations (a caller that cannot say which operation runs which): the kernel resolves one
// value per name, so an entry valid for A alone cannot price the layers lowered to B, and an
// entry that lists both can. A pool given as operations narrows this to the operations that
// consume the name (TestAnEntryIsMatchedAgainstTheOperationsThatConsumeIt).
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

// A coefficient is matched against the kernels of the operations that read it. The pool: full
// attention on A, sliding-window attention on B, MLA in the draft model on C, a sparse-MLA
// primary on D with its block-index scorer on E, a BF16 GEMM on A, an FP8 GEMM on B and an MXFP8
// GEMM that is priced on its own ramp, then the BF16 one, on D.
func TestAnEntryIsMatchedAgainstTheOperationsThatConsumeIt(t *testing.T) {
	kernelC := coefficient.KernelImpl{Library: "flashinfer", Variant: "mla_decode", CodeHash: "eeeeeeeeeeee"}
	kernelD := coefficient.KernelImpl{Library: "flashmla", Variant: "sparse_decode", CodeHash: "ffffffffffff"}
	kernelE := coefficient.KernelImpl{Library: "flashinfer", Variant: "scorer", CodeHash: "abababababab"}
	op := func(prim string, impl coefficient.KernelImpl, draft bool, chain ...string) Operation {
		return Operation{Primitive: prim, Impl: impl, Chain: chain, Draft: draft}
	}
	pool := Scope{Hardware: "h100", Operations: []Operation{
		op("attention.decode", kernelA, false),
		op("attention.decode", kernelB, false, "swa"),
		op("attention.decode", kernelC, true, "mla"),
		op("attention.decode", kernelD, false, "sparse_mla"),
		{Primitive: "attention.decode", Impl: kernelE, Chain: []string{"block_index_scores"}, Scorer: true},
		op("gemm", kernelA, false, "bf16"),
		op("gemm", kernelB, false, "fp8"),
		op("gemm", kernelD, false, "w8a16_mxfp8", "bf16"),
	}}
	named := func(name string, v float64, impls ...coefficient.KernelImpl) coefficient.Entry {
		return entryNamed(name, vocab.MethodMeasured, v, impls...)
	}
	const (
		swa, mla, sp, blk = "attention_decode_rate_swa", "attention_decode_rate_mla", "attention_decode_rate_sparse_mla", "attention_decode_rate_block_index_scores"
		bf16, fp8, mx     = "gemm_eps_max_bf16", "gemm_eps_max_fp8", "gemm_eps_max_w8a16_mxfp8"
		floor             = "attention_decode_floor"
	)
	// Every kind and role has a pair of its own, keyed to the kernel that runs it.
	own := []coefficient.Entry{named(swa, 1, kernelB), named(mla, 1, kernelC), named(sp, 1, kernelD), named(blk, 1, kernelE)}
	with := func(es ...coefficient.Entry) []*coefficient.Set {
		return setOf(append(append([]coefficient.Entry(nil), own...), es...)...)
	}
	for _, c := range []struct {
		name   string
		sets   []*coefficient.Set
		look   string
		priced bool
		why    string
	}{
		{"swa keyed to the kernel that runs the swa layers", setOf(named(swa, 1, kernelB)), swa, true, "B runs every swa layer"},
		{"swa keyed to the kernel that runs full attention", setOf(named(swa, 1, kernelA)), swa, false, "A runs no swa layer"},
		{"mla keyed to the draft model's kernel", setOf(named(mla, 1, kernelC)), mla, true, "only the draft model has MLA layers"},
		{"mla keyed to the target's attention kernel", setOf(named(mla, 1, kernelA)), mla, false, "A runs no MLA layer"},
		{"the scorer's pair keyed to the scorer's kernel", setOf(named(blk, 1, kernelE)), blk, true, "E runs the scorer"},
		{"the scorer's pair keyed to the primary's kernel", setOf(named(blk, 1, kernelD)), blk, false, "D runs no scorer"},
		{"the sparse pair keyed to the primary, not the scorer", setOf(named(sp, 1, kernelD)), sp, true, "the scorer is a node of its own"},
		// The part-wide pair is read only by what no pair of its own prices.
		{"the part-wide rate keyed to full attention, no kind has its own pair", setOf(named(attn, 1, kernelA)), attn, false, "swa, mla, sparse and the scorer all fall back to it"},
		{"the part-wide rate keyed to full attention, every kind has its own", with(named(attn, 1, kernelA)), attn, true, "only full attention reads it"},
		{"the part-wide rate keyed to the swa kernel, swa has its own pair", with(named(attn, 1, kernelB)), attn, false, "full attention, on A, reads it"},
		{"the part-wide floor keyed to full attention, swa has a rate and no floor", with(named(floor, 1, kernelA)), floor, false, "the swa layers inherit the floor"},
		{"the part-wide floor keyed to full attention and swa, every kind has a rate and none a floor", with(named(floor, 1, kernelA, kernelB)), floor, false, "mla, sparse and the scorer inherit it too"},
		{"the part-wide floor keyed to every kernel that inherits it", with(named(floor, 1, kernelA, kernelB, kernelC, kernelD, kernelE)), floor, true, "each kind with a rate and no floor inherits it"},
		{"the part-wide floor keyed to full attention, every kind has a floor", with(named(floor, 1, kernelA),
			named("attention_decode_floor_swa", 1, kernelB), named("attention_decode_floor_mla", 1, kernelC),
			named("attention_decode_floor_sparse_mla", 1, kernelD), named("attention_decode_floor_block_index_scores", 1, kernelE)), floor, true, "only full attention reads it"},
		// GEMMs are read by the operations of the format that selects the ramp.
		{"bf16 keyed to the BF16 GEMM's kernel, mxfp8 not carried", setOf(named(bf16, 1, kernelA)), bf16, false, "the MXFP8 GEMM on D falls back to the BF16 ramp"},
		{"bf16 keyed to the BF16 GEMM's kernel, mxfp8 carried", setOf(named(bf16, 1, kernelA), named(mx, 1, kernelD)), bf16, true, "the MXFP8 GEMM has its own ramp"},
		{"fp8 keyed to the FP8 GEMM's kernel", setOf(named(fp8, 1, kernelB)), fp8, true, "B runs the FP8 GEMM"},
		{"fp8 keyed to the BF16 GEMM's kernel", setOf(named(fp8, 1, kernelA)), fp8, false, "A runs no FP8 GEMM"},
		{"mxfp8 keyed to its kernel", setOf(named(mx, 1, kernelD)), mx, true, "D runs the MXFP8 GEMM"},
	} {
		t.Run(c.name, func(t *testing.T) {
			v, m, err := valueAt(t, c.look, c.sets, pool)
			switch {
			case c.priced && (err != nil || v != 1 || m != coefficient.KernelExact):
				t.Errorf("not priced (%v %q %v): %s", v, m, err, c.why)
			case !c.priced && err == nil:
				t.Errorf("priced: %s", c.why)
			}
		})
	}
	// What the entry was matched against is what consumed it, not the primitive's pool.
	c, err := Load(with(named(attn, 3, kernelA)), pool)
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]coefficient.KernelImpl{swa: kernelB, attn: kernelA, blk: kernelE} {
		r, _ := c.Entry(name)
		if len(r.Running) != 1 || r.Running[0] != want {
			t.Errorf("%s was matched against %v, want only %v", name, r.Running, want)
		}
	}
	// A name no operation reads (no sliding-window layer) cannot be priced by a keyed entry.
	none := Scope{Hardware: "h100", Operations: []Operation{op("attention.decode", kernelA, false)}}
	if _, _, err := valueAt(t, swa, setOf(named(swa, 1, kernelA)), none); err == nil {
		t.Errorf("a keyed swa entry priced a pool with no sliding-window layer")
	}
	// A GEMM that states no format reads every ramp.
	wild := Scope{Hardware: "h100", Operations: []Operation{{Primitive: "gemm", Impl: kernelA, Wild: true}, op("gemm", kernelB, false, "fp8")}}
	if _, _, err := valueAt(t, fp8, setOf(named(fp8, 1, kernelB)), wild); err == nil {
		t.Errorf("a GEMM that states no format was left out of the fp8 ramp's consumers")
	}
	// The pooled form is the projection of the operations.
	got := pool.PooledImplementations()
	if len(got["attention.decode"]) != 5 || len(got["gemm"]) != 3 {
		t.Errorf("pooled implementations %v", got)
	}
	// The recurrent mixers narrow the same way.
	rec := Scope{Hardware: "h100", Operations: []Operation{
		op("recurrent.update", kernelA, false, "kda"), op("recurrent.update", kernelB, false, "mamba2")}}
	if _, _, err := valueAt(t, "recurrent_decode_rate_kda", setOf(named("recurrent_decode_rate_kda", 1, kernelA)), rec); err != nil {
		t.Errorf("a kda entry keyed to the kernel that runs the kda layers: %v", err)
	}
	if _, _, err := valueAt(t, "recurrent_decode_rate_kda", setOf(named("recurrent_decode_rate_kda", 1, kernelB)), rec); err == nil {
		t.Errorf("a kda entry keyed to the mamba2 kernel priced the kda layers")
	}
}

// The trtllm-gen lane's pair is read by the prefill of the non-latent layers whose cache has its
// format, and shadows the part-wide prefill pair only where it is carried exact.
func TestTheLanePairIsReadByTheNonLatentPrefillOfItsCacheFormat(t *testing.T) {
	lane := func(dtype string) string { return "attention_prefill_floor_trtllm_gen_kv_" + dtype }
	pre := func(impl coefficient.KernelImpl, chain ...string) Operation {
		return Operation{Primitive: "attention.prefill", Impl: impl, Chain: chain}
	}
	// Non-latent bf16 prefill on B, non-latent fp8 prefill on C, latent prefill on A.
	kernelC := coefficient.KernelImpl{Library: "flashinfer", Variant: "fp8_prefill", CodeHash: "eeeeeeeeeeee"}
	pool := Scope{Hardware: "h100", Operations: []Operation{pre(kernelA, "sparse_mla"), pre(kernelB, "kv_bf16"), pre(kernelC, "kv_fp8")}}
	named := func(name string, impls ...coefficient.KernelImpl) coefficient.Entry {
		return entryNamed(name, vocab.MethodMeasured, 1, impls...)
	}
	for name, c := range map[string]struct {
		e     coefficient.Entry
		exact bool
	}{
		"the bf16 pair keyed to the bf16 kernel":         {named(lane("bf16"), kernelB), true},
		"the bf16 pair keyed to the latent kernel":       {named(lane("bf16"), kernelA), false},
		"the fp8 pair keyed to the fp8 kernel":           {named(lane("fp8"), kernelC), true},
		"the fp8 pair keyed to the bf16 kernel":          {named(lane("fp8"), kernelB), false},
		"the bf16 pair keyed to both non-latent kernels": {named(lane("bf16"), kernelB, kernelC), true},
	} {
		_, m, err := valueAt(t, c.e.Name, setOf(c.e), pool)
		if got := err == nil && m == coefficient.KernelExact; got != c.exact {
			t.Errorf("%s: exact %v (%v)", name, got, err)
		}
	}
	// The part-wide prefill floor keyed to the latent kernel is read by the latent layers; the
	// non-latent ones are priced from the lane when it is carried exact, and from the part-wide
	// pair when it is not.
	wide := named("attention_prefill_floor", kernelA)
	if _, _, err := valueAt(t, wide.Name, setOf(wide), pool); err == nil {
		t.Error("the part-wide floor priced non-latent prefill that runs other kernels, with no lane carried")
	}
	exactLane := []coefficient.Entry{wide, named(lane("bf16"), kernelB), named(lane("fp8"), kernelC)}
	if _, m, err := valueAt(t, wide.Name, setOf(exactLane...), pool); err != nil || m != coefficient.KernelExact {
		t.Errorf("the part-wide floor, with an exact lane for both formats: %v %q", err, m)
	}
	unkeyed := []coefficient.Entry{wide, entryNamed(lane("bf16"), vocab.MethodTransferred, 1), entryNamed(lane("fp8"), vocab.MethodTransferred, 1)}
	for i := 1; i < 3; i++ {
		unkeyed[i].MeasuredOn = []coefficient.KernelImpl{kernelA}
		unkeyed[i].TransferredFrom, unkeyed[i].TransferRule, unkeyed[i].Rationale = "hardware=h100", "r", "r"
	}
	if _, _, err := valueAt(t, wide.Name, setOf(unkeyed...), pool); err == nil {
		t.Error("an unkeyed lane shadowed the part-wide floor, which the non-latent layers still read")
	}
}
