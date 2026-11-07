package coefficient

import (
	"fmt"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// The exact kernel-implementation lookup (design D5, VAL-8). An entry names two things about
// kernels, and they are different facts:
//
//   - KernelImpls are the implementations the value is valid for. A lookup matches the entry
//     only when one of them is what runs. This is the key.
//   - MeasuredOn are the implementations the fit data came from. It is evidence about a
//     transferred value, and a lookup never reads it.
//
// For a measured or calibrated value the two are the same by D5, so only KernelImpls is stated.
// A transferred value is measured on one implementation and used for another, so it states
// both; without MeasuredOn there would be no way to record the target of a transfer.

// KernelSlot is a kernel without its build: the library and its own exported kernel name. Two
// implementations in one slot differ only in their code hash.
type KernelSlot struct {
	Library string
	Variant string
}

// Slot returns the kernel's library and variant.
func (k KernelImpl) Slot() KernelSlot { return KernelSlot{k.Library, k.Variant} }

// KernelMatch says how an entry met the implementations that run.
type KernelMatch string

const (
	// KernelUnkeyed: the entry states no implementation. It applies to every implementation,
	// and nothing shows it was measured on the one that runs. A reader deciding whether a price
	// rests on the running kernel (the G1 hash condition) must not count it as a match.
	KernelUnkeyed KernelMatch = "unkeyed"
	// KernelExact: the entry lists an implementation that runs, with the same library, variant
	// and code hash.
	KernelExact KernelMatch = "exact"
	// KernelRefused: the entry lists implementations and none of them runs. A lookup does not
	// return it. A different code hash of the same kernel does not match: there is no fallback
	// to a sibling implementation (VAL-8).
	KernelRefused KernelMatch = "refused"
)

// PrimitiveOf returns the lowered primitive (a plan.Primitive word) that a coefficient of this
// name prices, or "" for a name that prices no lowered kernel: host overheads, hbm_derate, fabric,
// memory sizes, routing and the mHC terms. It is the one place the binding is decided. A lookup
// compares an entry's kernel_impls with the implementations of this primitive only, so a value
// valid for one kernel cannot price a primitive another kernel executes (design VAL-8, 6.2).
func PrimitiveOf(name string) string {
	has := func(prefix string) bool { return strings.HasPrefix(name, prefix) }
	switch {
	case has("gemm_eps_max_"), has("gemm_m_half_"), has("gemm_n_half_"), has("gemm_k_half_"):
		if strings.Contains(name, "_grouped") {
			return "moe.grouped_gemm"
		}
		return "gemm"
	case has("attention_decode_"):
		return "attention.decode"
	case has("attention_prefill_"):
		return "attention.prefill"
	case has("recurrent_decode_"), has("recurrent_prefill_"):
		return "recurrent.update"
	case has("collective_floor_"), has("collective_peak_rate_"), has("collective_transition_rate_"):
		rest := name
		for _, pre := range []string{"collective_floor_", "collective_peak_rate_", "collective_transition_rate_"} {
			rest = strings.TrimPrefix(rest, pre)
		}
		for _, op := range [][2]string{{"all_reduce_", "all_reduce"}, {"all_gather_", "all_gather"},
			{"reduce_scatter_", "reduce_scatter"}, {"alltoall_", "all2all"}} {
			if strings.HasPrefix(rest, op[0]) {
				return op[1]
			}
		}
	case has("moe_a2a_"):
		return "all2all"
	case has("host_gather_"):
		return "gather"
	}
	return ""
}

// MatchKernels says how the entry meets the implementations that run. running holds them by
// lowered primitive (a plan.Primitive word). An entry that lists none is unkeyed. A keyed entry
// is exact only when its name binds to a primitive (PrimitiveOf), the plan supplies at least one
// implementation for that primitive, and every one of them is listed by the entry, by library,
// variant and code hash together. Otherwise it is refused: a different hash, another kernel,
// nothing supplied for the primitive, or a primitive that also runs a kernel the entry does not
// cover. The last case is why "every": the kernel resolves one value per coefficient name, so a
// value valid for kernel A cannot price the layers of the same primitive lowered to kernel B.
func (e Entry) MatchKernels(running map[string][]KernelImpl) KernelMatch {
	if len(e.KernelImpls) == 0 {
		return KernelUnkeyed
	}
	impls := running[PrimitiveOf(e.Name)]
	if PrimitiveOf(e.Name) == "" || len(impls) == 0 {
		return KernelRefused
	}
	for _, have := range impls {
		found := false
		for _, want := range e.KernelImpls {
			if want == have {
				found = true
				break
			}
		}
		if !found {
			return KernelRefused
		}
	}
	return KernelExact
}

// Identity is what tells two entries of one set apart: the name, the scope, the method and the
// implementations the entry is valid for (design 3.3). Two entries that differ only in the
// kernel build they are valid for are different coefficients: a kernel whose source changed gets
// a new key and its own value (design 6.2). The implementations are in sorted order, so listing
// them differently does not change it.
func (e Entry) Identity() string {
	return e.Name + "\x00" + e.Scope.Key() + "\x00" + string(e.Method) + "\x00" + DescribeImpls(e.KernelImpls)
}

// TargetImpls are the implementations the entry is valid for.
func (e Entry) TargetImpls() []KernelImpl { return e.KernelImpls }

// SourceImpls are the implementations the entry's value was measured on: for a measured or
// calibrated entry its own, since D5 makes the fit data exercise the keyed implementation; for a
// transferred entry the ones it states in MeasuredOn; none for any other method.
func (e Entry) SourceImpls() []KernelImpl {
	switch e.Method {
	case vocab.MethodMeasured, vocab.MethodCalibrated:
		return e.KernelImpls
	case vocab.MethodTransferred:
		return e.MeasuredOn
	}
	return nil
}

// ImplsCompatible reports whether one set of running implementations can make both entries
// match: either is unkeyed, or they list an implementation in common (a primitive that runs only
// that one is covered by both). Two entries of one name that list disjoint implementations never
// both match. The registry validator uses it to decide whether two entries can be chosen
// between, so that the validator and the lookup agree (X15).
func ImplsCompatible(a, b Entry) bool {
	if len(a.KernelImpls) == 0 || len(b.KernelImpls) == 0 {
		return true
	}
	for _, x := range a.KernelImpls {
		for _, y := range b.KernelImpls {
			if x == y {
				return true
			}
		}
	}
	return false
}

// MayStateKernelImpls reports whether an entry of method m may list the implementations it is
// valid for. A measured or calibrated value is keyed by the implementation its fit data
// exercised (D5); a transferred one names the implementation it is moved to, and records where
// it came from in measured_on; a literature figure names the kernel its authors measured. A
// vendor_spec, roofline, assumed or not_charged value states nothing about an implementation, and
// is what a lookup falls back to when no entry matches the one that runs (design section 6.2).
// This is the one place the rule is decided: the registry's entry validator and the Result's
// validator of a selected entry both call it.
func MayStateKernelImpls(m vocab.Method) bool {
	switch m {
	case vocab.MethodMeasured, vocab.MethodCalibrated, vocab.MethodTransferred, vocab.MethodLiterature:
		return true
	}
	return false
}

// MayStateMeasuredOn reports whether an entry of method m may record the implementations its
// value was measured on, apart from the ones it is valid for. Only a transferred value is used
// on an implementation other than the one it was measured on; for a measured or calibrated one
// the two are the same (D5) and a second list could only contradict the first.
func MayStateMeasuredOn(m vocab.Method) bool { return m == vocab.MethodTransferred }

// ValidateKernelImpls checks a list of kernel implementations: each states its library, its
// variant and a lower-case hexadecimal code hash of at least 12 digits, and none repeats. at
// names the list in a message. A registry entry and a Result's record of a selected entry are
// checked by this one function.
func ValidateKernelImpls(p *validate.Problems, at string, impls []KernelImpl) {
	seen := map[KernelImpl]bool{}
	for i, k := range impls {
		kat := fmt.Sprintf("%s[%d]", at, i)
		if strings.TrimSpace(k.Library) == "" {
			p.Field(kat+".library", "required")
		}
		if strings.TrimSpace(k.Variant) == "" {
			p.Field(kat+".variant", "required: the library's own exported kernel name")
		}
		if !hexHash(k.CodeHash) {
			p.Field(kat+".code_hash", "%q must be a lower-case hexadecimal hash of at least 12 digits", k.CodeHash)
		}
		if seen[k] {
			p.Field(kat, "duplicate kernel implementation %s/%s/%s", k.Library, k.Variant, k.CodeHash)
		}
		seen[k] = true
	}
}

// ValidateRunning checks the implementations a pool runs, as an engine states them, by lowered
// primitive. It differs from ValidateKernelImpls in two ways. An implementation the engine could
// not identify has an empty code hash and is allowed: it matches no keyed entry, which is the
// honest answer. And a pool runs one build of each kernel, so two hashes for one library and
// variant anywhere in the pool are refused rather than matched against whichever entry lists
// either. A primitive named with an empty list, or a word that is not a primitive a coefficient
// binds to, is refused.
func ValidateRunning(running map[string][]KernelImpl) error {
	prims := make([]string, 0, len(running))
	for p := range running {
		prims = append(prims, p)
	}
	sort.Strings(prims)
	bySlot := map[KernelSlot]string{}
	for _, p := range prims {
		if len(running[p]) == 0 {
			return fmt.Errorf("primitive %q is listed with no implementation", p)
		}
		for i, k := range running[p] {
			at := fmt.Sprintf("implementations[%s][%d] (%s/%s)", p, i, k.Library, k.Variant)
			switch {
			case strings.TrimSpace(k.Library) == "":
				return fmt.Errorf("%s: the library is required", at)
			case k.CodeHash != "" && strings.TrimSpace(k.Variant) == "":
				return fmt.Errorf("%s: a code hash needs the library's own kernel name as variant, since a hash identifies one kernel of the library", at)
			case k.CodeHash != "" && !hexHash(k.CodeHash):
				return fmt.Errorf("%s: code hash %q must be a lower-case hexadecimal hash of at least 12 digits", at, k.CodeHash)
			}
			if prev, ok := bySlot[k.Slot()]; ok && prev != k.CodeHash {
				return fmt.Errorf("%s: two builds of one kernel run in one pool (code hash %q and %q); a lookup matches the build that runs and a pool runs one", at, prev, k.CodeHash)
			}
			bySlot[k.Slot()] = k.CodeHash
		}
	}
	return nil
}

// DescribeImpls renders implementations for an error or a report, sorted so the text is stable.
func DescribeImpls(impls []KernelImpl) string {
	out := make([]string, len(impls))
	for i, k := range impls {
		out[i] = k.Library + "/" + k.Variant + "/" + k.CodeHash
	}
	sort.Strings(out)
	return strings.Join(out, ", ")
}
