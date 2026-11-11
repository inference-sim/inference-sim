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

// ClassOf returns the class of lowered operation a coefficient of this name is read by within
// its primitive, or "" when the name is a part-wide term read by every operation of the
// primitive that has no class of its own to read instead.
//
//   - The per-kind terms of attention and the recurrent update are read by the layers of that
//     kind: `attention_decode_rate_swa`, `..._mla`, `..._sparse_mla`, `recurrent_decode_rate_kda`,
//     `..._mamba2`, `..._gdn`.
//   - `..._block_index_scores` terms are read by the block-index scorer of a layer, a node of its
//     own beside the layer's primary attention (a lowered operation with that role).
//   - `attention_prefill_*_trtllm_gen_kv_bf16` and `_kv_fp8` are read by the prefill of the
//     non-latent layers whose cache has that format (`kv_bf16`, `kv_fp8`).
//   - `gemm_*_<format>` terms (`bf16`, `fp8`, `fp8_block`, `nvfp4`, `w8a16_mxfp8`,
//     `w4a16_mxfp4_grouped`, ...) are read by the GEMMs whose weight format selects that ramp.
//
// The unsuffixed attention terms are the fallback for every kind or role with no pair of its own
// (ConsumesFallback). This is the one place the binding is decided, as PrimitiveOf is for the
// primitive.
func ClassOf(name string) string {
	switch PrimitiveOf(name) {
	case "attention.decode", "attention.prefill":
		for _, c := range []struct{ suffix, class string }{
			{"_trtllm_gen_kv_bf16", "kv_bf16"}, {"_trtllm_gen_kv_fp8", "kv_fp8"},
			{"_sparse_mla", "sparse_mla"}, {"_block_index_scores", "block_index_scores"},
			{"_mla", "mla"}, {"_swa", "swa"}} {
			if strings.HasSuffix(name, c.suffix) {
				return c.class
			}
		}
	case "recurrent.update":
		for _, c := range []string{"mamba2", "kda", "gdn"} {
			if strings.HasSuffix(name, "_"+c) {
				return c
			}
		}
	case "gemm", "moe.grouped_gemm":
		for _, pre := range []string{"gemm_eps_max_", "gemm_m_half_", "gemm_n_half_", "gemm_k_half_"} {
			if rest, ok := strings.CutPrefix(name, pre); ok {
				return rest
			}
		}
	}
	return ""
}

// ClassFamily lists the coefficients of a class that must ALL be carried for the kernel to price
// an operation from the class: the kernel uses a class's terms only when its whole family is
// present, and otherwise reads the part-wide terms. A per-kind decode pair needs its rate (the
// floor may be inherited); the sparse-MLA and scorer prefill forms their rate, floor and
// per-token terms (and the sparse form its per-request term), fitted jointly; the trtllm-gen lane
// its floor and work scale; a recurrent mixer its floor and rate; a GEMM ramp its asymptote and
// half-max. Nil for a class the kernel has no terms for.
func ClassFamily(primitive, class string) []string {
	switch primitive {
	case "attention.decode":
		return []string{"attention_decode_rate_" + class}
	case "attention.prefill":
		switch {
		case strings.HasPrefix(class, "kv_"):
			return []string{"attention_prefill_floor_trtllm_gen_" + class, "attention_prefill_work_scale_trtllm_gen_" + class}
		case class == "sparse_mla":
			return []string{"attention_prefill_rate_" + class, "attention_prefill_floor_" + class,
				"attention_prefill_per_token_" + class, "attention_prefill_per_request_" + class}
		case class == "block_index_scores":
			return []string{"attention_prefill_rate_" + class, "attention_prefill_floor_" + class,
				"attention_prefill_per_token_" + class}
		}
	case "recurrent.update":
		return []string{"recurrent_decode_floor_" + class, "recurrent_decode_rate_" + class}
	case "gemm", "moe.grouped_gemm":
		return []string{"gemm_eps_max_" + class, "gemm_m_half_" + class}
	}
	return nil
}

// ClassUsable is the one rule for whether a class prices its operations: every member of its
// family is carried (have), and, for the trtllm-gen lane, every member is shown valid for the
// implementation that runs (exact). The resolver's decision of which operations read the
// part-wide terms, applyAttentionLane and the prefill forms all use it.
func ClassUsable(primitive, class string, have, exact func(name string) bool) bool {
	fam := ClassFamily(primitive, class)
	if len(fam) == 0 {
		return false
	}
	for _, n := range fam {
		if !have(n) || (strings.HasPrefix(class, "kv_") && !exact(n)) {
			return false
		}
	}
	return true
}

// FloorInheritsFromClass reports whether a part-wide term named so is also read by the
// operations whose class has a rate but no floor of its own: the per-kind decode pair may supply
// a rate alone and inherit the part-wide floor.
func FloorInheritsFromClass(name string) bool { return name == "attention_decode_floor" }

// MatchKernels says how the entry meets the implementations that run. running holds them by
// lowered primitive (a plan.Primitive word), every operation of the primitive pooled; it is the
// form a caller gives that cannot say which operation runs which. See MatchImpls.
func (e Entry) MatchKernels(running map[string][]KernelImpl) KernelMatch {
	if len(e.KernelImpls) == 0 {
		return KernelUnkeyed
	}
	if PrimitiveOf(e.Name) == "" {
		return KernelRefused
	}
	return e.MatchImpls(running[PrimitiveOf(e.Name)])
}

// MatchImpls says how the entry meets the implementations of the operations that consume it:
// the lowered operations of the entry's primitive, narrowed to its class where the name states
// one (ClassOf). An entry that lists none is unkeyed. A keyed entry is exact only when its name
// binds to a primitive (PrimitiveOf), at least one consuming operation is supplied, and every
// implementation among them is listed by the entry, by library, variant and code hash together.
// Otherwise it is refused: a different hash, another kernel, nothing supplied for its
// operations, or an operation that consumes the value and runs a kernel the entry does not
// cover. The last case is why "every": the kernel resolves one value per coefficient name, so a
// value valid for kernel A cannot price the operations that consume it and run kernel B. An
// operation that does not consume the name does not count: a sliding-window layer lowered to B
// does not stop a value valid for A alone from pricing full attention.
func (e Entry) MatchImpls(consuming []KernelImpl) KernelMatch {
	if len(e.KernelImpls) == 0 {
		return KernelUnkeyed
	}
	if PrimitiveOf(e.Name) == "" || len(consuming) == 0 {
		return KernelRefused
	}
	for _, have := range consuming {
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
