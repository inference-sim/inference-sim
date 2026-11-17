// Original to blis-vllm.

package blisvllm

import (
	"bytes"
	"os"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/kernelimpls"
)

// kernel_impls.json is the table design 6.2 asks each engine release to publish. This test holds
// the committed file to the document type blis-substrate defines for it:
//
//   - it loads through the substrate's strict loader (kernelimpls.Load), which refuses an unknown
//     key, an unsorted or repeated row, a row of a library version or architecture the table
//     does not list, and a code hash that is not kernelimpls.CodeHash of the row: the generator's
//     Python restatement of the hash rule cannot drift from the Go rule;
//   - it is for the release this build models (Upstream);
//   - it is in canonical form, the bytes the substrate's Marshal writes;
//   - it is the file this build embeds.
func TestTheKernelImplTableLoadsAndNamesThisRelease(t *testing.T) {
	tab, err := kernelimpls.Load("kernel_impls.json")
	if err != nil {
		t.Fatal(err)
	}
	if tab.Engine.Kind != Kind || tab.Engine.Version != Upstream.Version || tab.Engine.Commit != Upstream.Commit {
		t.Errorf("kernel_impls.json is for %+v; this build models %s %s at %s", tab.Engine, Kind, Upstream.Version, Upstream.Commit)
	}
	raw, err := os.ReadFile("kernel_impls.json")
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := tab.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, canonical) {
		t.Error("kernel_impls.json is not in the substrate's canonical form; regenerate it with kernels/gen_kernel_impls.py")
	}
	if !bytes.Equal(KernelImplsJSON, raw) {
		t.Error("KernelImplsJSON differs from kernel_impls.json")
	}
	if _, err := KernelImpls(); err != nil {
		t.Errorf("the embedded table does not load: %v", err)
	}
}

// The lowering states the build that runs on the chip, and the cost model's exact lookup (VAL-8)
// matches a coefficient keyed by that build and refuses it for any other: the trtllm-gen prefill
// kernel on a B200 (compute capability 10.0) and on a B300 (10.3) are two builds, and a value
// measured on one does not price the other.
func TestTheLoweringStatesTheBuildTheLookupMatches(t *testing.T) {
	const lib, variant = "flashinfer", "trtllm_batch_context_with_kv_cache"
	sel := AttentionSelection("FLASHINFER")
	b200, err := Implementation(plan.PrimAttentionPrefill, sel, lib, variant, 10, 0)
	if err != nil {
		t.Fatal(err)
	}
	b300, err := Implementation(plan.PrimAttentionPrefill, sel, lib, variant, 10, 3)
	if err != nil {
		t.Fatal(err)
	}
	if b200.CodeHash == "" || b300.CodeHash == "" || b200.Toolkit == "" {
		t.Fatalf("the trtllm-gen prefill build is not identified: b200 %+v, b300 %+v", b200, b300)
	}
	if b200.CodeHash == b300.CodeHash {
		t.Errorf("one code hash for two architectures: the hash covers the device the build runs on")
	}
	key := func(k plan.KernelImpl) coefficient.KernelImpl {
		return coefficient.KernelImpl{Library: k.Library, Variant: k.Variant, CodeHash: k.CodeHash}
	}
	entry := coefficient.Entry{Name: "attention_prefill_floor_trtllm_gen_kv_bf16", KernelImpls: []coefficient.KernelImpl{key(b200)}}
	if got := entry.MatchImpls([]coefficient.KernelImpl{key(b200)}); got != coefficient.KernelExact {
		t.Errorf("an entry keyed by the B200 build, priced on a B200: %s, want exact", got)
	}
	if got := entry.MatchImpls([]coefficient.KernelImpl{key(b300)}); got != coefficient.KernelRefused {
		t.Errorf("an entry keyed by the B200 build, priced on a B300: %s, want refused", got)
	}
}

// A build the table does not know is stated with no code hash, which matches no keyed entry: a
// primitive or architecture the table does not cover, a backend it names as a gap, and a kernel a
// gap inside a listed backend names.
func TestABuildTheTableDoesNotKnowHasNoHash(t *testing.T) {
	for name, c := range map[string]struct {
		prim         plan.Primitive
		backend      string
		lib, variant string
		major, minor int
	}{
		"a primitive the table does not cover":     {plan.PrimGEMM, "FLASHINFER", "cublas", "gemm", 10, 0},
		"an architecture the table does not cover": {plan.PrimAttentionPrefill, "FLASHINFER", "flashinfer", "trtllm_batch_context_with_kv_cache", 12, 0},
		"a backend the table names as a gap":       {plan.PrimAttentionDecode, "FLASHINFER_MLA", "flashinfer", "trtllm_batch_decode_with_kv_cache_mla", 10, 0},
		"a kernel a gap of a listed backend names": {plan.PrimAttentionPrefill, "FLASHINFER", "flashinfer", "BatchAttentionWithAttentionSinkWrapper", 9, 0},
	} {
		got, err := Implementation(c.prim, AttentionSelection(c.backend), c.lib, c.variant, c.major, c.minor)
		if err != nil {
			t.Errorf("%s: %v", name, err)
		}
		if got.CodeHash != "" || got.Library != c.lib || got.Variant != c.variant {
			t.Errorf("%s: %+v, want %s/%s with no code hash", name, got, c.lib, c.variant)
		}
	}
}

// A build the table knows the release cannot run is an error, not a silent unidentified
// implementation: FA3 on a B200 (vLLM runs FA3 on SM90 only), and a misspelt library.
func TestABuildTheReleaseCannotRunIsAnError(t *testing.T) {
	for name, c := range map[string]struct {
		lib, variant string
		major, minor int
	}{
		"FA3 on SM100":       {"vllm-flash-attn", "_vllm_fa3_C.fwd", 10, 0},
		"a misspelt library": {"vllm-flashattn", "_vllm_fa2_C.varlen_fwd", 9, 0},
	} {
		if got, err := Implementation(plan.PrimAttentionPrefill, AttentionSelection("FLASH_ATTN"), c.lib, c.variant, c.major, c.minor); err == nil {
			t.Errorf("%s: %+v and no error", name, got)
		}
	}
}

// On an H100 (9.0), vLLM's FlashAttention selection moves a layer FA3 cannot serve (a head size
// above 256, or attention sinks it does not support) to FA4 when FA4 is available
// (vllm/v1/attention/backends/fa_utils.py, get_flash_attn_version). The lowering must find
// FA4's SM90 build, not report it unidentified.
func TestFA4IsABuildOnSM90(t *testing.T) {
	got, err := Implementation(plan.PrimAttentionPrefill, AttentionSelection("FLASH_ATTN"), "vllm-flash-attn", "cute.flash_attn_varlen_func", 9, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got.CodeHash == "" {
		t.Errorf("FA4 on SM90 is not identified: %+v", got)
	}
}

// Sparse MLA on SM90 (FLASHINFER_MLA_SPARSE_SM90, the backend vLLM selects for a sparse-MLA model
// on an H200) runs FlashInfer's FA3 MLA template for prefill and decode, and only on SM90: the
// lowering finds the H200 build, and a B200 build of it is an error.
func TestSparseMLAOnSM90IsABuildOnH200Only(t *testing.T) {
	sel := AttentionSelection("FLASHINFER_MLA_SPARSE_SM90")
	for _, prim := range []plan.Primitive{plan.PrimAttentionPrefill, plan.PrimAttentionDecode} {
		got, err := Implementation(prim, sel, "flashinfer", "BatchMLAPagedAttentionWrapper[fa3]", 9, 0)
		if err != nil || got.CodeHash == "" {
			t.Errorf("%s on SM90: %+v, %v; want an identified build", prim, got, err)
		}
		if _, err := Implementation(prim, sel, "flashinfer", "BatchMLAPagedAttentionWrapper[fa3]", 10, 0); err == nil {
			t.Errorf("%s on SM100: no error; the backend runs on SM90 only", prim)
		}
	}
}
