package coefficient

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

var (
	implA = KernelImpl{Library: "nccl", Variant: "ncclAllReduce", CodeHash: "aaaaaaaaaaaa"}
	implB = KernelImpl{Library: "nccl", Variant: "ncclAllReduce", CodeHash: "bbbbbbbbbbbb"}
	// implOtherVariant is the same build of the same library as implA, another kernel of it.
	implOtherVariant = KernelImpl{Library: "nccl", Variant: "ncclAllGather", CodeHash: "aaaaaaaaaaaa"}
	implFlash        = KernelImpl{Library: "flashinfer", Variant: "decode", CodeHash: "cccccccccccc"}
)

// attn is a coefficient name that prices the attention.decode primitive.
const attn = "attention_decode_rate"

func keyed(impls ...KernelImpl) Entry {
	return Entry{Name: attn, Method: vocab.MethodMeasured, KernelImpls: impls}
}

// on says which implementations a pool runs for attention.decode.
func on(impls ...KernelImpl) map[string][]KernelImpl {
	if len(impls) == 0 {
		return nil
	}
	return map[string][]KernelImpl{"attention.decode": impls}
}

// VAL-8: a keyed entry matches only the implementation that runs the primitive it prices, by
// library, variant and code hash together, and there is no fallback to a sibling.
func TestMatchKernels(t *testing.T) {
	for _, c := range []struct {
		name    string
		e       Entry
		running map[string][]KernelImpl
		want    KernelMatch
	}{
		{"the listed implementation runs the primitive", keyed(implA), on(implA), KernelExact},
		{"one of several listed runs", keyed(implA, implB), on(implB), KernelExact},
		{"a list that covers every build the primitive runs", keyed(implA, implFlash), on(implA, implFlash), KernelExact},
		{"a different hash of the same kernel", keyed(implA), on(implB), KernelRefused},
		{"the same hash of another variant", keyed(implA), on(implOtherVariant), KernelRefused},
		{"another library", keyed(implA), on(implFlash), KernelRefused},
		{"nothing is said to run", keyed(implA), nil, KernelRefused},
		{"the engine could not identify the build", keyed(implA),
			on(KernelImpl{Library: "nccl", Variant: "ncclAllReduce"}), KernelRefused},
		// The pool runs A for one primitive and B for another: the entry is valid for A only.
		{"the kernel runs, but another primitive's", keyed(implA),
			map[string][]KernelImpl{"gemm": {implA}, "attention.decode": {implFlash}}, KernelRefused},
		{"the kernel runs only another primitive", keyed(implA), map[string][]KernelImpl{"gemm": {implA}}, KernelRefused},
		{"the primitive also runs a kernel the entry does not list", keyed(implA), on(implA, implFlash), KernelRefused},
		{"a name that prices no primitive", Entry{Name: "hbm_derate", KernelImpls: []KernelImpl{implA}},
			map[string][]KernelImpl{"": {implA}}, KernelRefused},
		{"an entry that lists none, nothing runs", keyed(), nil, KernelUnkeyed},
		{"an entry that lists none applies to whatever runs", keyed(), on(implB), KernelUnkeyed},
	} {
		if got := c.e.MatchKernels(c.running); got != c.want {
			t.Errorf("%s: got %q, want %q", c.name, got, c.want)
		}
	}
}

// The binding of a coefficient name to the primitive it prices.
func TestPrimitiveOf(t *testing.T) {
	for name, want := range map[string]string{
		"gemm_eps_max_bf16": "gemm", "gemm_m_half_fp8_block": "gemm", "gemm_n_half_bf16": "gemm",
		"gemm_eps_max_w4a16_mxfp4_grouped": "moe.grouped_gemm", "gemm_m_half_w4a16_mxfp4_grouped": "moe.grouped_gemm",
		"attention_decode_rate_mla": "attention.decode", "attention_decode_floor_swa": "attention.decode",
		"attention_prefill_floor": "attention.prefill", "attention_prefill_rate_sparse_mla": "attention.prefill",
		"recurrent_decode_rate_kda": "recurrent.update", "recurrent_prefill_rate_gdn": "recurrent.update",
		"collective_floor_all_reduce_fp16_8rank_h100":               "all_reduce",
		"collective_peak_rate_all_gather_bf16_4rank_h200":           "all_gather",
		"collective_transition_rate_reduce_scatter_fp16_2rank_b200": "reduce_scatter",
		"collective_floor_alltoall_fp16_8rank_h100":                 "all2all",
		"moe_a2a_rate_deepep_low_latency_8rank":                     "all2all", "host_gather_rate_uva": "gather",
		"hbm_derate": "", "host_launch_per_kernel": "", "fabric_bandwidth_efficiency": "", "mhc_rate_pre": "",
		"engine_workspace_bytes": "", "nccl_communicator_bytes_8rank": "", "moe_routing_imbalance_median": "",
	} {
		if got := PrimitiveOf(name); got != want {
			t.Errorf("PrimitiveOf(%q) = %q, want %q", name, got, want)
		}
	}
}

// A transferred value is valid for its target and measured on its source. The lookup reads the
// target only: a transferred entry matches the implementation it is moved to and not the one it
// was measured on.
func TestATransferredEntryMatchesItsTargetAndNotItsSource(t *testing.T) {
	e := Entry{Name: attn, Method: vocab.MethodTransferred, KernelImpls: []KernelImpl{implB}, MeasuredOn: []KernelImpl{implA}}
	if got := e.MatchKernels(on(implB)); got != KernelExact {
		t.Errorf("the target: got %q", got)
	}
	if got := e.MatchKernels(on(implA)); got != KernelRefused {
		t.Errorf("the source implementation: got %q, want refused (measured_on is never matched)", got)
	}
	if got := e.SourceImpls(); len(got) != 1 || got[0] != implA {
		t.Errorf("source: %v", got)
	}
	if got := e.TargetImpls(); len(got) != 1 || got[0] != implB {
		t.Errorf("target: %v", got)
	}
	m := keyed(implA)
	if got := m.SourceImpls(); len(got) != 1 || got[0] != implA {
		t.Errorf("a measured entry's source is its key (D5): %v", got)
	}
	a := Entry{Method: vocab.MethodAssumed}
	if a.SourceImpls() != nil {
		t.Errorf("an assumed entry has no source implementation")
	}
}

func TestImplsCompatible(t *testing.T) {
	for _, c := range []struct {
		name string
		a, b []KernelImpl
		want bool
	}{
		{"an unkeyed entry beside a keyed one", nil, []KernelImpl{implA}, true},
		{"two unkeyed entries", nil, nil, true},
		{"the same implementation", []KernelImpl{implA}, []KernelImpl{implA}, true},
		{"two builds of one kernel never both run", []KernelImpl{implA}, []KernelImpl{implB}, false},
		// A primitive's running set must be covered by each entry, so entries with nothing in
		// common never both match.
		{"two kernels of one library, nothing in common", []KernelImpl{implA}, []KernelImpl{implOtherVariant}, false},
		{"different libraries, nothing in common", []KernelImpl{implA}, []KernelImpl{implFlash}, false},
		{"lists with one implementation in common", []KernelImpl{implA, implFlash}, []KernelImpl{implB, implFlash}, true},
	} {
		if got := ImplsCompatible(keyed(c.a...), keyed(c.b...)); got != c.want {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
		if got := ImplsCompatible(keyed(c.b...), keyed(c.a...)); got != c.want {
			t.Errorf("%s, reversed: got %v, want %v", c.name, got, c.want)
		}
	}
}

// The rule by method is one function, and the table of it is the documentation.
func TestMethodsThatMayKeyAnEntry(t *testing.T) {
	want := map[vocab.Method][2]bool{ // {may state kernel_impls, may state measured_on}
		vocab.MethodMeasured: {true, false}, vocab.MethodCalibrated: {true, false},
		vocab.MethodTransferred: {true, true}, vocab.MethodLiterature: {true, false},
		vocab.MethodVendorSpec: {false, false}, vocab.MethodRoofline: {false, false},
		vocab.MethodAssumed: {false, false}, vocab.MethodNotCharged: {false, false},
	}
	for _, ms := range vocab.AllMethods() {
		m := vocab.Method(ms)
		w, ok := want[m]
		if !ok {
			t.Errorf("method %q is not in the table", m)
			continue
		}
		if MayStateKernelImpls(m) != w[0] || MayStateMeasuredOn(m) != w[1] {
			t.Errorf("%s: kernel_impls %v measured_on %v, want %v", m, MayStateKernelImpls(m), MayStateMeasuredOn(m), w)
		}
	}
}

// Two entries of one name, scope and method that differ only in the build they are valid for
// are two coefficients: a kernel whose source changed gets its own value.
func TestEntriesForTwoBuildsOfOneKernelAreBothValidInOneSet(t *testing.T) {
	s := validSet()
	a, b := s.Coefficients[0], s.Coefficients[0]
	a.KernelImpls, b.KernelImpls = []KernelImpl{implA}, []KernelImpl{implB}
	b.Value = 0.5
	s.Coefficients = []Entry{a, b}
	if p := s.Validate(); !p.OK() {
		t.Fatalf("two builds of one kernel were rejected:\n%s", p.Error())
	}
	b.KernelImpls = []KernelImpl{implA}
	s.Coefficients = []Entry{a, b}
	if p := s.Validate(); p.OK() || !strings.Contains(p.Error(), "duplicate entry name") {
		t.Fatalf("two entries with one key must be a duplicate, got %v", p)
	}
	// Listing the implementations in another order is the same identity.
	a.KernelImpls = []KernelImpl{implA, implB}
	b.KernelImpls = []KernelImpl{implB, implA}
	s.Coefficients = []Entry{a, b}
	if p := s.Validate(); p.OK() {
		t.Fatalf("the order of kernel_impls must not make two entries different")
	}
}

func TestValidateRunning(t *testing.T) {
	for _, c := range []struct {
		name    string
		impls   map[string][]KernelImpl
		wantErr string
	}{
		{"none", nil, ""},
		{"one build of each kernel", map[string][]KernelImpl{"all_reduce": {implA}, "attention.decode": {implFlash}}, ""},
		{"a build the engine could not identify", on(KernelImpl{Library: "nccl", Variant: "ncclAllReduce"}), ""},
		{"a library alone", on(KernelImpl{Library: "cublas"}), ""},
		{"the same build for two primitives", map[string][]KernelImpl{"all_reduce": {implA}, "all_gather": {implA}}, ""},
		{"no library", on(KernelImpl{Variant: "v", CodeHash: "aaaaaaaaaaaa"}), "library is required"},
		{"a hash with no variant", on(KernelImpl{Library: "l", CodeHash: "aaaaaaaaaaaa"}), "variant"},
		{"a malformed hash", on(KernelImpl{Library: "l", Variant: "v", CodeHash: "AAAA"}), "lower-case hexadecimal"},
		{"two builds of one kernel in one primitive", on(implA, implB), "two builds of one kernel"},
		{"two builds of one kernel across primitives", map[string][]KernelImpl{"all_reduce": {implA}, "all_gather": {implB}}, "two builds of one kernel"},
		{"a primitive with no implementation", map[string][]KernelImpl{"gemm": nil}, "no implementation"},
	} {
		err := ValidateRunning(c.impls)
		switch {
		case c.wantErr == "" && err != nil:
			t.Errorf("%s: %v", c.name, err)
		case c.wantErr != "" && (err == nil || !strings.Contains(err.Error(), c.wantErr)):
			t.Errorf("%s: got %v, want an error containing %q", c.name, err, c.wantErr)
		}
	}
}

// The class of lowered operation a name is consumed by within its primitive.
func TestClassOf(t *testing.T) {
	for name, want := range map[string]string{
		"attention_decode_rate_swa": "swa", "attention_decode_floor_swa": "swa",
		"attention_decode_rate_mla": "mla", "attention_prefill_rate_sparse_mla": "sparse_mla",
		"attention_decode_rate_sparse_mla": "sparse_mla", "attention_decode_floor_mla": "mla",
		"recurrent_decode_rate_kda": "kda", "recurrent_decode_floor_gdn": "gdn", "recurrent_decode_rate_mamba2": "mamba2",
		// consumed by every operation of the primitive
		"attention_decode_rate": "", "attention_decode_floor": "", "attention_prefill_work_scale": "",
		"recurrent_kda_packed_decode":                 "",
		"collective_floor_all_reduce_fp16_8rank_h100": "", "host_launch_per_kernel": "",
		// the block-index scorer is a node of its own
		"attention_decode_rate_block_index_scores": "block_index_scores", "attention_prefill_floor_block_index_scores": "block_index_scores",
		// the trtllm-gen lane, by the cache format it reads
		"attention_prefill_floor_trtllm_gen_kv_bf16": "kv_bf16", "attention_prefill_work_scale_trtllm_gen_kv_fp8": "kv_fp8",
		// a GEMM's ramp, by the weight format
		"gemm_eps_max_bf16": "bf16", "gemm_m_half_fp8": "fp8", "gemm_k_half_w4a16_mxfp4_grouped": "w4a16_mxfp4_grouped",
	} {
		if got := ClassOf(name); got != want {
			t.Errorf("ClassOf(%q) = %q, want %q", name, got, want)
		}
	}
}

// MatchImpls meets the entry against the operations that consume it: the ones another
// primitive or class runs do not count.
func TestMatchImplsIgnoresOperationsThatDoNotConsumeTheName(t *testing.T) {
	swa := Entry{Name: "attention_decode_rate_swa", KernelImpls: []KernelImpl{implB}}
	if got := swa.MatchImpls([]KernelImpl{implB}); got != KernelExact {
		t.Errorf("the sliding-window kernel: %q", got)
	}
	if got := swa.MatchImpls([]KernelImpl{implA, implB}); got != KernelRefused {
		t.Errorf("a consuming operation on a kernel the entry does not list: %q", got)
	}
	if got := swa.MatchImpls(nil); got != KernelRefused {
		t.Errorf("no consuming operation: %q", got)
	}
	if got := (Entry{Name: "attention_decode_rate_swa"}).MatchImpls(nil); got != KernelUnkeyed {
		t.Errorf("unkeyed: %q", got)
	}
	if got := (Entry{Name: "hbm_derate", KernelImpls: []KernelImpl{implA}}).MatchImpls([]KernelImpl{implA}); got != KernelRefused {
		t.Errorf("a name that prices no primitive: %q", got)
	}
}
