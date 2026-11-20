// Original to blis-vllm.
// Moved from blis-substrate's kernel tests (kernel/sparse_mla_test.go, int4_test.go and
// refusals_v41_test.go, served_test.go, namedrafter_test.go and internal/resolve/layout_test.go at 00817b10798cbc00f560e51921c30c7084fa5e0c) with the resolution they test
// (blis-substrate#58).

package lowering

import (
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"

	"github.com/inference-sim/blis-vllm/compat/vllmsettings"
	vllmrules "github.com/inference-sim/blis-vllm/rules"
)

// A model-specific MTP name is the method vLLM runs as mtp, so a deployment naming mtp
// drafts with a graph's deepseek_mtp speculator.
func TestModelSpecificMTPNamesAreMTP(t *testing.T) {
	if got := specMethod("deepseek_mtp"); got != "mtp" {
		t.Errorf("deepseek_mtp normalises to %q", got)
	}
	if got := specMethod("dspark"); got != "dspark" {
		t.Errorf("dspark normalises to %q", got)
	}
}

// Layouts no catalog fixture reaches, asserted on the selection directly. A 128-head DSA
// model (DeepSeek-V3.2) at tp=8 holds 16 heads a rank: at dcp=8 they gather to 128, which pads
// to a different fp8 decode size from the local 16, and FlashMLA-sparse refuses it; at dcp=4
// they gather to 64 and it starts (flashmla_sparse.py:436-460, :684-687). With PCP the gathered
// count is heads x tp when dcp spans tp x pcp: at tp4/pcp2/dcp8, 32 local heads gather to
// 128 and are refused, where pcp2/dcp2 leaves them at 32 (vLLM admits dcp only as 1, pcp or
// tp x pcp under PCP, vllm/config/parallel.py:571-577). The a2a combine is refused on
// FlashMLA-sparse (:416-422), and a dtype no backend lists has none.
func TestSparseMLABackendRefusesWhatNoBackendServes(t *testing.T) {
	starts := func(r sparseMLABackendRequest) bool {
		if r.dcpComm == "" {
			r.dcpComm = DCPAllGatherReduceScatter
		}
		_, _, err := sparseMLABackend(r)
		return err == nil
	}
	for _, c := range []struct {
		name string
		r    sparseMLABackendRequest
		want bool
	}{
		{"128 heads, tp8 dcp8", sparseMLABackendRequest{cache: "fp8", headsPerRank: 16,
			tp: 8, dcp: 8, pcp: 1}, false},
		{"128 heads, tp8 dcp4", sparseMLABackendRequest{cache: "fp8", headsPerRank: 16,
			tp: 8, dcp: 4, pcp: 1}, true},
		{"128 heads, tp4 pcp2 dcp8", sparseMLABackendRequest{cache: "fp8", headsPerRank: 32,
			tp: 4, dcp: 8, pcp: 2}, false},
		{"128 heads, tp8 pcp2 dcp2", sparseMLABackendRequest{cache: "fp8", headsPerRank: 16,
			tp: 8, dcp: 2, pcp: 2}, true},
		{"a2a combine", sparseMLABackendRequest{cache: "fp8", headsPerRank: 8, tp: 8, dcp: 2,
			pcp: 1, dcpComm: DCPAllToAll}, false},
		{"fp8_e5m2 on hopper", sparseMLABackendRequest{cache: "fp8_e5m2", headsPerRank: 8,
			tp: 8, dcp: 1, pcp: 1}, false},
	} {
		if got := starts(c.r); got != c.want {
			t.Errorf("%s: starts %v, want %v", c.name, got, c.want)
		}
	}
}

// The generation of a part comes from its stated compute capability, never from its name:
// a part of any name with SM8.x is pre-Hopper, SM9 and SM10 are not, and a part that states
// none, or states it malformed, is refused with its name.
func TestPreHopperComesFromTheComputeCapability(t *testing.T) {
	for _, c := range []struct {
		name, cc string
		want     bool
	}{
		{"rtx-6000-ada", "8.9", true},
		{"a100-80", "8.0", true},
		{"my-sm80-part", "8.6", true},
		{"my-sm90-part", "9.0", false},
		{"a100-lookalike", "9.0", false},
		{"my-sm100-part", "10.0", false},
		{"my-sm103-part", "10.3", false},
	} {
		chip := hardware.Chip{Name: c.name, ComputeCapability: c.cc}
		family, _ := chip.CapabilityMajor()
		got, err := (&lowerContext{chip: chip, family: family}).preHopper()
		if err != nil || got != c.want {
			t.Errorf("%s (SM%s): preHopper = %v, %v; want %v", c.name, c.cc, got, err, c.want)
		}
	}
	for _, cc := range []string{"", "eight", "-8.0", "08.0", "9"} {
		chip := hardware.Chip{Name: "mystery-part", ComputeCapability: cc}
		family, _ := chip.CapabilityMajor()
		_, err := (&lowerContext{chip: chip, family: family}).preHopper()
		if err == nil || !strings.Contains(err.Error(), "mystery-part") {
			t.Errorf("capability %q: err = %v, want a refusal naming the chip", cc, err)
		}
	}
}

// Every part of the catalog keeps the answers the name lists gave: A100 and L40S are
// pre-Hopper, the rest (whatever parts the catalog gains) are not, and the A100 defaults hold for the two A100 parts alone.
func TestCatalogPartsKeepTheirGenerationAndBatchDefaults(t *testing.T) {
	d := data
	names := d.Chips()
	for _, want := range []string{"a100-80", "a100-sxm", "l40s", "h100", "h200", "b200"} {
		if !slices.Contains(names, want) {
			t.Fatalf("the catalog's chips %v lack %s", names, want)
		}
	}
	for _, n := range names {
		chip, _ := d.Chip(n)
		family, _ := chip.CapabilityMajor()
		pre, err := (&lowerContext{chip: *chip, family: family}).preHopper()
		if err != nil {
			t.Fatalf("%s: %v", n, err)
		}
		wantPre := n == "a100-80" || n == "a100-sxm" || n == "l40s"
		if pre != wantPre {
			t.Errorf("%s: preHopper = %v, want %v", n, pre, wantPre)
		}
		if isA100(*chip) != strings.Contains(n, "a100") {
			t.Errorf("%s: isA100 = %v", n, isA100(*chip))
		}
	}
}

// A DSA model on a part stating SM8.x is refused whatever its name (here the catalog name stays
// h100), where it used to price on the SM90 backend.
func TestADSAModelOnAnArbitrarilyNamedPreHopperPartIsRefused(t *testing.T) {
	for _, cc := range []string{"8.9", "9.0", "10.0", ""} {
		chip, err := data.Chip("h100")
		if err != nil {
			t.Fatal(err)
		}
		c := *chip
		c.ComputeCapability = cc
		_, err = tryLowerOn(t, where{"glm-5", "h100", "tp: 8, pp: 1, dp: 1"}, &c,
			vllmsettings.Settings{BlockSize: 64})
		switch cc {
		case "8.9":
			if err == nil || !strings.Contains(err.Error(), "below compute capability 9.0") {
				t.Errorf("SM89: err = %v, want the pre-Hopper refusal", err)
			}
		case "":
			if err == nil || !strings.Contains(err.Error(), "h100") {
				t.Errorf("no capability: err = %v, want a refusal naming the chip", err)
			}
		default:
			if err != nil && strings.Contains(err.Error(), "below compute capability 9.0") {
				t.Errorf("SM%s: refused as pre-Hopper: %v", cc, err)
			}
		}
	}
}

// An "auto" KV cache follows the model's COMPUTE dtype, not its weight storage width.
//
// A W4A16 checkpoint stores weights at four bits and computes in bf16. Following the
// storage width gave a half-byte KV element, which rounded to a per-block figure of
// zero once paged, and the kernel refused twelve Kimi-K2.5 sweeps rather than divide a
// budget by it. vLLM sizes an auto cache at the model dtype (vllm/platforms/interface.py:
// 861-862 at v0.31.0) unless the checkpoint's quantization config declares a KV algorithm,
// which the graph does not carry; see cacheDTypeBytes.
func TestAnAutoCacheFollowsTheComputeWidthNotTheStorageWidth(t *testing.T) {
	for _, tc := range []struct {
		served model.DType
		want   float64
		why    string
	}{
		{model.DTypeINT4, 2, "W4A16 computes in bf16"},
		{model.DTypeNVFP4, 2, "a 4-bit float is a storage format too"},
		{model.DTypeMXFP4, 2, "likewise"},
		{model.DTypeFP8, 2, "fp8 linears return bf16 (out_dtype=x.dtype), so the model dtype is bf16"},
		{model.DTypeBF16, 2, "unquantized passes through"},
		{model.DTypeFP32, 4, "and so does fp32"},
	} {
		if got := cacheDTypeBytes("auto", tc.served); got != tc.want {
			t.Errorf("auto cache on a %s model = %.1f bytes, want %.1f (%s)",
				tc.served, got, tc.want, tc.why)
		}
		// An empty setting means the engine's default, which is auto.
		if got := cacheDTypeBytes("", tc.served); got != tc.want {
			t.Errorf("empty cache dtype on a %s model = %.1f, want %.1f",
				tc.served, got, tc.want)
		}
	}
	// An explicitly named narrow cache is still honoured: the rule above is about what
	// "auto" resolves to, not a floor on every cache.
	if got := cacheDTypeBytes("fp8", model.DTypeINT4); got != 1 {
		t.Errorf("an explicit fp8 cache = %.1f bytes, want 1", got)
	}
	// nvfp4 carries an fp8 scale per 16 elements beside its four-bit data:
	// head_size/2 + head_size/16 bytes a head (vllm/utils/torch_utils.py:543-545).
	if got := cacheDTypeBytes("nvfp4", model.DTypeBF16); got != 0.5625 {
		t.Errorf("an explicit nvfp4 cache = %.4f bytes, want 0.5625", got)
	}
}

// The custom all-reduce request is resolved, not obeyed: the layout may decline it, and a
// declined request records why. A prediction that assumed the request honoured would charge SM
// occupancy that does not occur and omit NIC traffic that does.
func TestCustomAllReduceRequestIsResolvedNotObeyed(t *testing.T) {
	no := false
	rules := vllmrules.Lookup("0.33.0")
	for _, c := range []struct {
		what        string
		gpusPerNode int
		tp          int
		rackDomain  bool
		wantCustom  bool
		wantReason  string
	}{
		{"tp 8 within a node: honoured", 8, 8, false, true, ""},
		{"tp 3 is an unsupported width: declined", 8, 3, false, false, "rank counts"},
		{"tp 16 spans nodes without multi-node NVLink: declined", 8, 16, false, false, "spans more than"},
		{"tp 16 spans nodes with multi-node NVLink: honoured", 8, 16, true, true, ""},
		{"stated off", 8, 8, false, false, ""},
	} {
		t.Run(c.what, func(t *testing.T) {
			s := vllmsettings.Settings{DisableCustomAllReduce: &no}
			if c.what == "stated off" {
				s.AllReduceBackend = "nccl"
			}
			got, overrides := customAllReduce(s, layout{TP: c.tp, DP: 1, GPUsPerNode: c.gpusPerNode},
				c.rackDomain, rules)
			if got != c.wantCustom {
				t.Fatalf("custom all-reduce %v, want %v", got, c.wantCustom)
			}
			if c.wantReason == "" {
				if len(overrides) != 0 {
					t.Errorf("an honoured or unrequested custom all-reduce records %+v", overrides)
				}
				return
			}
			if len(overrides) != 1 || !strings.Contains(overrides[0].Reason, c.wantReason) {
				t.Errorf("overrides %+v, want one naming %q", overrides, c.wantReason)
			}
		})
	}
}

// A declined all-reduce request reaches the Result on the all_reduce_backend row, whatever field
// name the lowering gives it: the conversion in package result is checked against the lowering.
func TestADeclinedAllReduceRequestReachesOneRowOfTheResult(t *testing.T) {
	ok, overrides := customAllReduce(vllmsettings.Settings{}, layout{TP: 3, DP: 1}, false, vllmrules.Lookup("0.33.0"))
	if ok || len(overrides) != 1 {
		t.Fatalf("tp 3 should decline the custom all-reduce: %v %v", ok, overrides)
	}
	o := overrides[0]
	var rows []result.Resolution
	for _, x := range result.ResolutionsOf(kernel.Resolution{AllReduceBackend: "nccl",
		Overrides: []kernel.Override{{Field: o.Field, Requested: o.Requested, Resolved: o.Resolved, Reason: o.Reason}}}) {
		if strings.Contains(x.Setting, "reduce") {
			rows = append(rows, x)
		}
	}
	if len(rows) != 1 || rows[0].Setting != "all_reduce_backend" || rows[0].Host != "custom" || rows[0].Reason == "" {
		t.Errorf("rows %+v, want one all_reduce_backend row with the host value and the reason", rows)
	}
}

// The cost model recognises no quantization spelling beyond the ones servedDType prices: a
// ModelOpt or microscaling spelling is resolved by the checkpoint's recipe (package
// quantization) before the lowering, and reaches it empty.
func TestAQuantizationSpellingTheCostModelDoesNotPriceIsRefused(t *testing.T) {
	for _, spelling := range []string{"modelopt_mixed", "mxfp8", "modelopt_mxfp8"} {
		_, err := tryLower(t, where{"minimax-m3-nvfp4", "b200", "tp: 4, pp: 1, dp: 1"},
			vllmsettings.Settings{Quantization: spelling})
		if err == nil || !strings.Contains(err.Error(), "not one this kernel prices") {
			t.Errorf("quantization %q: %v", spelling, err)
		}
	}
}

// A deployment method that is not the drafter's describes another drafter, and is refused.
func TestADeploymentMethodThatIsNotTheDraftersIsRefused(t *testing.T) {
	target, err := data.Model("minimax-m3-mxfp8")
	if err != nil {
		t.Fatal(err)
	}
	draft, err := data.Model("minimax-m3-eagle3")
	if err != nil {
		t.Fatal(err)
	}
	at := where{"minimax-m3-mxfp8", "b200", "tp: 4, pp: 1, dp: 1"}
	in := func(method string) error {
		_, err := tryLowerWithDraft(t, at, target, draft, vllmsettings.Settings{Speculative: &vllmsettings.Speculative{
			Method: method, NumSpecTokens: 3, DraftModel: draft.Name}})
		return err
	}
	if err := in("eagle3"); err != nil {
		t.Fatalf("the drafter's own method was refused: %v", err)
	}
	l, err := tryLowerWithDraft(t, at, target, draft, vllmsettings.Settings{Speculative: &vllmsettings.Speculative{
		Method: "eagle3", NumSpecTokens: 3, DraftModel: draft.Name}})
	if err != nil || l.Drafter != draft.Name {
		t.Errorf("Drafter %q (%v), want the named %q", l.Drafter, err, draft.Name)
	}
	if err := in("mtp"); err == nil || !strings.Contains(err.Error(), "a mtp drafter is not described") {
		t.Errorf("err %v, want the refusal of a method the drafter does not describe", err)
	}
}

// vLLM puts FlashInfer's SM90 sparse-MLA backend first when the MLA layer's head size is 512 and
// FlashAttention's first otherwise (_get_backend_priorities, cuda.py:137-145 at v0.31.0 and
// :139-147 at v0.33.0). The head size is kv_lora_rank + qk_rope_head_dim: 576 for a rope-64 model
// such as GLM-5 and DeepSeek-V3.2, 512 for GLM-5.3-Flash, a NoPE latent (blis-vllm#278). This
// assumes FlashInfer 0.6.18 or later is installed (the backend's own gate).
func TestSparseMLAOnSM90PrefersFlashInferAtHeadSize512(t *testing.T) {
	base := sparseMLABackendRequest{cache: "auto", headsPerRank: 8, tp: 8, dcp: 1, pcp: 1}
	for _, c := range []struct {
		name     string
		headSize int
		dcp      int
		want     string
	}{
		{"head size 512 selects FlashInfer SM90", 512, 1, backendFlashInferSM90},
		{"head size 576 selects FlashAttention sparse", 576, 1, backendFlashAttnMLASparse},
		{"no head size stated keeps FlashAttention sparse", 0, 1, backendFlashAttnMLASparse},
		{"head size 512 under DCP falls to FlashMLA sparse (no DCP in FlashInfer SM90 or FlashAttention)", 512, 2, ""},
	} {
		r := base
		r.headSize, r.dcp = c.headSize, c.dcp
		r.dcpComm = DCPAllGatherReduceScatter
		got, _, err := sparseMLABackend(r)
		if c.want == "" {
			// FlashMLA-sparse takes a 16-bit cache under DCP only with an fp8_ds_mla cache, so the layout refuses it.
			if err == nil {
				t.Errorf("%s: selected %s, want a refusal", c.name, got)
			}
			continue
		}
		if err != nil || got != c.want {
			t.Errorf("%s: selected %q (%v), want %q", c.name, got, err, c.want)
		}
	}
}

// Through the lowering, from the catalog graphs: GLM-5.3-Flash (kv_lora_rank 512, qk_rope_head_dim 0)
// on an H200 is lowered on FlashInfer's SM90 backend and says what it assumes; GLM-5 (rope 64)
// keeps FlashAttention's sparse MLA.
func TestTheLoweringSelectsTheSparseMLABackendFromTheHeadSize(t *testing.T) {
	tp8 := "tp: 8, pp: 1, dp: 1, enable_expert_parallel: true"
	flash := lowered(t, where{"glm-5.3-flash", "h200", tp8}, vllmsettings.Settings{})
	if v, ok := noted(flash, "sparse_mla_backend"); !ok || v != backendFlashInferSM90 {
		t.Errorf("glm-5.3-flash on H200: sparse_mla_backend note %q (%t), want %s", v, ok, backendFlashInferSM90)
	}
	for _, m := range []string{"glm-5", "glm-5.1-fp8", "glm-5.2"} {
		l := lowered(t, where{m, "h200", tp8}, vllmsettings.Settings{})
		if v, ok := noted(l, "sparse_mla_backend"); ok {
			t.Errorf("%s on H200: sparse_mla_backend note %q, want none (FlashAttention sparse is first at head size 576)", m, v)
		}
	}
}

// The kernel a sparse-MLA layer runs on FLASHINFER_MLA_SPARSE_SM90 is stated for its prefill and its
// decode (FlashInferMLASparseSM90Impl builds a BatchMLAPagedAttentionWrapper with backend="fa3",
// compute capability 9.x), so the cost model reads the registry family keyed to that build. A model
// whose backend is another one, a part on another family, and a pipelined pool state none.
func TestTheLoweringStatesTheSparseMLAKernelItsSM90BackendRuns(t *testing.T) {
	tp8 := "tp: 8, pp: 1, dp: 1, enable_expert_parallel: true"
	flash := lowered(t, where{"glm-5.3-flash", "h200", tp8}, vllmsettings.Settings{})
	byPrim := map[plan.Primitive]int{}
	for _, l := range flash.Plan.Layers {
		byPrim[l.Primitive]++
		k := l.KernelImpl
		if k.Library != "flashinfer" || k.Variant != "BatchMLAPagedAttentionWrapper[fa3]" || k.CodeHash == "" {
			t.Errorf("layer %d %s: kernel %+v, want FlashInfer's BatchMLAPagedAttentionWrapper[fa3] with its code hash", l.Layer, l.Primitive, k)
		}
		if l.Scorer || l.Stage != 0 {
			t.Errorf("layer %d %s: scorer %t, stage %d; the primary on stage 0 is stated", l.Layer, l.Primitive, l.Scorer, l.Stage)
		}
	}
	if byPrim[plan.PrimAttentionPrefill] == 0 || byPrim[plan.PrimAttentionPrefill] != byPrim[plan.PrimAttentionDecode] {
		t.Errorf("GLM-5.3-Flash on H200 states %v, want the same layers for the prefill and the decode", byPrim)
	}
	for _, tc := range []struct{ name, model, chip, pool string }{
		{"a model whose first backend is FlashAttention sparse", "glm-5", "h200", tp8},
		{"a pipelined pool", "glm-5.3-flash", "h200", "tp: 4, pp: 2, dp: 1, enable_expert_parallel: true"},
	} {
		l := lowered(t, where{tc.model, tc.chip, tc.pool}, vllmsettings.Settings{})
		if len(l.Plan.Layers) != 0 {
			t.Errorf("%s states %d lowered layers, want none", tc.name, len(l.Plan.Layers))
		}
	}
	pipelined := lowered(t, where{"glm-5.3-flash", "h200", "tp: 4, pp: 2, dp: 1, enable_expert_parallel: true"}, vllmsettings.Settings{})
	if v, ok := noted(pipelined, "sparse_mla_kernel"); !ok || v != "unstated" {
		t.Errorf("a pipelined pool notes sparse_mla_kernel %q (%t), want it unstated by name", v, ok)
	}
}
