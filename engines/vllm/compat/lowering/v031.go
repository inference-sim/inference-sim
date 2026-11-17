// Original to blis-vllm.
// Moved unchanged from blis-substrate's kernel (kernel/new.go,
// kernel/sparse_mla.go and kernel/draft.go at 00817b10798cbc00f560e51921c30c7084fa5e0c), when the
// engine-settings resolution left the cost model (blis-substrate#58).

package lowering

import (
	"fmt"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// attentionKinds is every attention kind a layer of the target stack reads the engine's
// cache with. A speculator's layers are not the target's, and a kind declared but not
// stacked runs nowhere.
func attentionKinds(g *model.Graph) map[model.AttentionKind]bool {
	byID := map[string]model.LayerKind{}
	for _, lk := range g.LayerKinds {
		byID[lk.ID] = lk
	}
	kinds := map[model.AttentionKind]bool{}
	for _, id := range g.Stack.Expand() {
		for _, n := range byID[id].Nodes {
			if n.ReadsEngineKV() {
				kinds[n.AttentionKind] = true
			}
		}
	}
	return kinds
}

// dsMLAStateBytes is the packed page cell, in bytes per token per layer, of the cache
// layouts that state their own size.
var dsMLAStateBytes = map[string]float64{"fp8_ds_mla": 656, "nvfp4_ds_mla": 352}

// latentAttention reports whether an attention kind keeps a latent (MLA) cache, which is
// what the engine's MLA attention implementations serve.
func latentAttention(kind model.AttentionKind) bool {
	return kind == model.AttentionMLA || kind == model.AttentionSparseMLA
}

// a100Parts are the catalog's A100 parts, by catalog key. vLLM v0.31.0's batch defaults
// exclude an A100 by its device name (EngineArgs.get_batch_defaults, vllm/engine/arg_utils.py:
// 2858-2887: the 70 GiB tier applies to a device not named A100), an identity the compute
// capability does not carry -- SM80 is every A100, and SM89 parts such as the L40S are not
// A100s but are not SM80 either -- so the parts are named exactly rather than matched by a
// substring of the name.
var a100Parts = map[string]bool{"a100-80": true, "a100-sxm": true}

// isA100 reports whether the chip is one of the catalog's A100 parts.
func isA100(chip hardware.Chip) bool { return a100Parts[chip.Name] }

// defaultMaxNumSeqs is vLLM v0.31.0's default sequence cap for a part in the
// OpenAI-API-server context (EngineArgs.get_batch_defaults, vllm/engine/arg_utils.py:
// 2858-2887): 1,024 on a part with at least 70 GiB that is not an A100, 256 otherwise.
func defaultMaxNumSeqs(chip hardware.Chip) int {
	if chip.MemoryGiB >= 70 && !isA100(chip) {
		return 1024
	}
	return 256
}

// defaultMaxNumBatchedTokens is vLLM v0.31.0's default token budget for a part, in the
// OpenAI-API-server usage context an llm-d deployment serves through
// (EngineArgs.get_batch_defaults, vllm/engine/arg_utils.py:2858-2887): 16,384 on a part
// with at least 160 GiB, 8,192 on one with at least 70 GiB that is not an A100, 2,048
// otherwise.
func defaultMaxNumBatchedTokens(chip hardware.Chip) int {
	switch {
	case chip.MemoryGiB >= 160:
		return 16384
	case chip.MemoryGiB >= 70 && !isA100(chip):
		return 8192
	}
	return 2048
}

// servedDType resolves the format the linear layers are served in.
//
// A quantization the kernel does not recognize is an error rather than a fallback to the
// checkpoint's width: silently serving a request for, say, awq at bf16 would report weight
// bytes and a compute peak that are both wrong, with nothing saying so.
func servedDType(checkpoint model.DType, quantization string) (model.DType, error) {
	switch quantization {
	case "":
		return checkpoint, nil
	case "fp8", "ptpc_fp8", "fbgemm_fp8", "modelopt":
		return model.DTypeFP8, nil
	case "modelopt_fp4", "nvfp4":
		return model.DTypeNVFP4, nil
	case "bitsandbytes", "awq", "awq_marlin", "gptq", "gptq_marlin", "compressed-tensors":
		// These are weight-only integer formats whose packed width depends on a
		// per-checkpoint bit count the flag does not carry. A checkpoint serving one of
		// them records the width in its own quantization_config, which the graph already
		// read, so the flag adds nothing and the graph governs.
		return checkpoint, nil
	}
	return "", fmt.Errorf(
		"quantization %q is not one this kernel prices; add it to servedDType with its "+
			"weight width rather than letting it fall through to %s",
		quantization, checkpoint)
}

// mxfp4ExpertsOnly reports whether a stated quantization is one of vLLM's MXFP4 methods, which
// quantize the routed experts and nothing else, on a checkpoint whose routed experts are stored in
// MXFP4. It returns vLLM's refusal for a stated MXFP4 method the checkpoint cannot be served with.
//
// In v0.33.0 both methods leave every linear layer unquantized: Mxfp4Config.get_quant_method returns
// UnquantizedLinearMethod for a LinearBase ("MXFP4 linear layer is not implemented",
// vllm/model_executor/layers/quantization/mxfp4.py:91-107@4ba787ec) and an MXFP4 MoE method for the
// routed experts; GptOssMxfp4Config is that config with the GPT-OSS MoE method (:116-149). So the
// linear layers run at the checkpoint's own width, and the experts at the width the graph states for
// them. "gpt_oss_mxfp4" is the method GptOssMxfp4Config.override_quantization_method names for a
// GPT-OSS checkpoint whose quant_method is mxfp4 (:128-146), and the name vLLM logs as the engine's
// quantization; it is the checkpoint's own method only for model_type gpt_oss.
func mxfp4ExpertsOnly(g *model.Graph, quantization string) (bool, error) {
	if quantization != "mxfp4" && quantization != "gpt_oss_mxfp4" {
		return false, nil
	}
	if !routedExpertsIn(g, model.DTypeMXFP4) {
		return false, fmt.Errorf("quantization %q quantizes the routed experts as MXFP4, and %s stores no "+
			"routed experts in MXFP4, so vLLM cannot load it with this method "+
			"(vllm/model_executor/layers/quantization/mxfp4.py:91-149@4ba787ec)", quantization, g.Name)
	}
	if quantization == "gpt_oss_mxfp4" && g.Global.Architecture != "GptOssForCausalLM" {
		return false, fmt.Errorf("quantization %q is the GPT-OSS checkpoints' method, and %s is a %s: vLLM "+
			"accepts it only for model_type gpt_oss (GptOssMxfp4Config.override_quantization_method, "+
			"vllm/model_executor/layers/quantization/mxfp4.py:128-146@4ba787ec)",
			quantization, g.Name, g.Global.Architecture)
	}
	return true, nil
}

// routedExpertsIn reports whether every routed-expert GroupedGEMM of the graph stores its weights
// in d, and the graph has at least one.
func routedExpertsIn(g *model.Graph, d model.DType) bool {
	found := false
	for _, lk := range g.LayerKinds {
		for _, n := range lk.Nodes {
			if n.Op != model.OpGroupedGEMM || n.IsSharedExpert() {
				continue
			}
			w := n.WeightDType
			if w == "" {
				w = g.Global.WeightDType
			}
			if w != d {
				return false
			}
			found = true
		}
	}
	return found
}

// cacheDTypeBytes resolves the KV or state cache width.
//
// "auto" follows the model's COMPUTE dtype: vLLM sizes an auto cache at
// model_config.dtype (vllm/platforms/interface.py:861-862, vllm/utils/torch_utils.py:527-529
// at v0.31.0), which is bf16 or fp16 whatever the weights are stored in. That is not the
// weight storage width on a quantized checkpoint, and the difference is not cosmetic. A
// W4A16 checkpoint -- Kimi-K2.5 is compressed-tensors int4 at group_size 32 -- stores
// weights at four bits and computes in bf16, so following the weight width gave a
// half-byte KV element and, once paged, a per-block figure of zero. An fp8-served model is
// the same case: its linears return out_dtype=x.dtype, so its activations and an auto
// cache stay bf16, and an earlier form that sized it at one byte halved its cache.
//
// WHAT THIS CANNOT SEE. "auto" is resolved against the checkpoint before it reaches the
// cache: a quantization config declaring a KV algorithm (a ModelOpt checkpoint with
// kv_cache_quant_algo) turns it into that dtype (resolve_kv_cache_dtype_string,
// vllm/utils/torch_utils.py:503-518), and DeepSeek-V4 turns it into fp8_ds_mla when its
// backend uses that layout (vllm/models/deepseek_v4/attention.py:107-120). (DeepSeek-V3.2
// keeps "auto" at the model dtype, vllm/model_executor/models/config.py:37-39, whatever the
// cache.py:127 docstring says.) The graph carries neither, so such a deployment should
// state its cache dtype; "auto" here is the model dtype.
func cacheDTypeBytes(declared string, fallback model.DType) float64 {
	switch declared {
	case "", "auto":
		// A weight format narrower than 16 bits is a storage width, not a compute width:
		// the model computes in bf16 (or fp16) and the cache follows that. A 16- or 32-bit
		// format is its own compute width and passes through.
		if fallback.Bytes() < 2 {
			return model.DTypeBF16.Bytes()
		}
		return fallback.Bytes()
	case "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc", "fp8_ds_mla":
		return 1
	case "bfloat16", "float16":
		return 2
	case "nvfp4", "nvfp4_4over6":
		// Four-bit data plus an fp8 scale per 16 elements: head_size/2 + head_size/16
		// bytes a head (nvfp4_kv_cache_full_dim, vllm/utils/torch_utils.py:543-545).
		return 0.5 + 1.0/16
	case "nvfp4_ds_mla":
		// Its 352-byte cell is applied where it is served (dsMLAStateBytes); this is only
		// the per-element width a latent model never prices with.
		return 0.5
	case "float32":
		return 4
	}
	// checkCacheDTypes has refused every other name, so this is unreachable from New.
	return fallback.Bytes()
}

// The KV cache dtypes vLLM v0.31.0 accepts (CacheDType, vllm/config/cache.py:39-58), split by
// whether this kernel prices them. The turboquant and per-token-head formats are real but
// carry per-token scales or packings whose page size this kernel does not compute
// (MLACommonBackend.customize_spec, mla_attention.py:1580-1593, for the per-token-head
// ones), so they are refused rather than priced at a width they do not have. Some of the
// priced ones are refused for a model as well (checkCacheDTypes).
var (
	pricedCacheDTypes = map[string]bool{
		"": true, "auto": true, "float16": true, "bfloat16": true, "fp8": true,
		"fp8_e4m3": true, "fp8_e5m2": true, "fp8_inc": true, "fp8_ds_mla": true,
		"nvfp4_ds_mla": true, "nvfp4": true, "nvfp4_4over6": true,
	}
	unpricedCacheDTypes = map[string]bool{
		"turboquant_k8v4": true, "turboquant_4bit_nc": true, "turboquant_k3v4_nc": true,
		"turboquant_3bit_nc": true, "int4_per_token_head": true,
		"int8_per_token_head": true, "fp8_per_token_head": true,
	}
	// The recurrent state dtypes (MambaDType, vllm/config/cache.py:61).
	mambaCacheDTypes = map[string]bool{
		"": true, "auto": true, "float32": true, "float16": true, "bfloat16": true,
	}
)

// checkCacheDTypes refuses a cache dtype the engine does not accept, one it accepts that
// this kernel does not price, and one no backend serves for this model's attention kind:
//
//   - fp8_inc is listed by no CUDA attention backend (it appears only in
//     vllm/config/cache.py:46 and vllm/utils/torch_utils.py:46);
//   - fp8_ds_mla and nvfp4_ds_mla are listed only by the sparse-MLA backends;
//   - plain nvfp4 on a latent model is refused outright (validate_nvfp4_kv_cache_with_mla,
//     vllm/config/vllm.py:3414-3430).
//
// An unchecked name used to fall through to the weight width, which on an nvfp4-served
// model charged an unknown cache at half a byte.
//
// The question is asked of every kind the target stack holds, not of one kind standing for
// the model: a dtype must be served for each of them. An earlier form took the kind from
// the LAST-declared layer kind, so DeepSeek-V4.1-Flash, whose last declaration is a DSpark
// draft layer outside the target stack, was refused fp8_ds_mla as if it were plain MLA.
func checkCacheDTypes(cache, mamba string, kinds map[model.AttentionKind]bool) error {
	latent := false
	var notSparse model.AttentionKind
	for _, kind := range sortedKinds(kinds) {
		latent = latent || latentAttention(kind)
		if kind != model.AttentionSparseMLA && notSparse == "" {
			notSparse = kind
		}
	}
	switch {
	case cache == "fp8_inc":
		return fmt.Errorf("cache_dtype fp8_inc is served by no CUDA attention backend in " +
			"vLLM v0.31.0")
	case strings.HasSuffix(cache, "_ds_mla") && (len(kinds) == 0 || notSparse != ""):
		return fmt.Errorf("cache_dtype %q is served only by the sparse-MLA backends, and "+
			"this model's attention includes %s", cache, notSparse)
	case strings.HasPrefix(cache, "nvfp4") && !strings.HasSuffix(cache, "_ds_mla") &&
		latent:
		return fmt.Errorf("cache_dtype %q on a latent-attention model: vLLM v0.31.0 "+
			"refuses plain nvfp4 with MLA (vllm/config/vllm.py:3414-3430)", cache)
	case unpricedCacheDTypes[cache]:
		return fmt.Errorf("cache_dtype %q is a vLLM v0.31.0 format this kernel does not "+
			"price: its page carries scales or a packing the kernel does not size", cache)
	case !pricedCacheDTypes[cache]:
		return fmt.Errorf("cache_dtype %q is not a vLLM v0.31.0 cache dtype "+
			"(vllm/config/cache.py:39-58); the engine refuses it", cache)
	case !mambaCacheDTypes[mamba]:
		return fmt.Errorf("mamba_cache_dtype %q is not a vLLM v0.31.0 recurrent state "+
			"dtype (vllm/config/cache.py:61); the engine refuses it", mamba)
	}
	return nil
}

// sortedKinds returns a set's kinds in a fixed order, so a refusal names the same kind
// every time.
func sortedKinds(kinds map[model.AttentionKind]bool) []model.AttentionKind {
	out := make([]model.AttentionKind, 0, len(kinds))
	for kind := range kinds {
		out = append(out, kind)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// dsaHeadDim is the head width that marks a DSA sparse-MLA layer: kv_lora_rank 512 plus
// qk_rope_head_dim 64.
const dsaHeadDim = 576

// dsaBlockSize is the kernel block every DSA deployment runs on CUDA, and the block size the
// engine picks when none is stated. The indexer's backend accepts exactly 64
// (DeepseekV32IndexerBackend.get_supported_kernel_block_sizes,
// vllm/v1/attention/backends/mla/indexer.py:203-204), and every sparse-MLA backend accepts 64,
// so with no stated size the platform picks 64 (update_block_size_for_backend,
// vllm/platforms/interface.py:666-689, through _preferred_block_size_for_backends,
// :616-656). A stated size must be a multiple of 64: a multiple runs 64-token kernel blocks
// inside it (select_common_block_size, vllm/v1/worker/utils.py:330-392), and anything else is
// refused -- at backend selection, where the size fails every candidate's
// supports_block_size (vllm/v1/attention/backend.py:121-137; "No valid attention backend",
// vllm/platforms/cuda.py:489-493), or, where a sparse backend takes it (32 on FlashInfer,
// SM100), at select_common_block_size, which the indexer fails ("No common block size").
const dsaBlockSize = 64

// isDSA reports whether a graph's attention is the DSA family's sparse MLA: some layer of
// its target stack reads the engine's cache with a 576-wide sparse-MLA head.
func isDSA(g *model.Graph) bool {
	if g == nil {
		return false
	}
	byID := map[string]model.LayerKind{}
	for _, lk := range g.LayerKinds {
		byID[lk.ID] = lk
	}
	for _, id := range g.Stack.Expand() {
		for _, n := range byID[id].Nodes {
			if n.ReadsEngineKV() &&
				n.AttentionKind == model.AttentionSparseMLA && n.HeadDim == dsaHeadDim {
				return true
			}
		}
	}
	return false
}

// deepseekV4Family reports whether a graph is DeepSeek-V4 or V4.1, whose attention runs on
// that family's own FlashMLA backends (vllm/models/deepseek_v4, deepseek_v41) rather than
// the DSA ones. By the architecture the graph names, and nothing else: a 512-wide sparse-MLA
// head is not the family's mark, since GLM-5.3-Flash's sparse layers have one too.
func deepseekV4Family(g *model.Graph) bool {
	switch g.Global.Architecture {
	case "DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM":
		return true
	}
	return false
}

// v4BlockSize is the one kernel block a DeepSeek-V4-family model's sparse backends accept.
// DeepSeek-V4.1's run 64 on SM90 and 128 elsewhere (vllm/models/deepseek_v41/
// sparse_mla.py:89-90, vllm/v1/attention/backends/mla/indexer.py:277-278 at v0.31.0);
// DeepSeek-V4's run 256 on every part, its indexer pages holding 64 rows of a 4:1 stream
// (vllm/models/deepseek_v4/sparse_mla.py:59-61, indexer.py:259-262). hopper is consulted
// only for V4.1.
func v4BlockSize(architecture string, hopper func() bool) int {
	if architecture != "DeepseekV41ForCausalLM" {
		return 256
	}
	if hopper() {
		return 64
	}
	return 128
}

// The sparse-MLA backends v0.31.0 ships for SM90 and SM100, by AttentionBackendEnum name.
const (
	backendFlashAttnMLASparse  = "FLASH_ATTN_MLA_SPARSE"
	backendFlashMLASparse      = "FLASHMLA_SPARSE"
	backendFlashInferMLASparse = "FLASHINFER_MLA_SPARSE"
	backendFlashInferSM90      = "FLASHINFER_MLA_SPARSE_SM90"
)

// sparseMLABackendRequest is what backend selection reads.
type sparseMLABackendRequest struct {
	cache        string // the stated cache dtype; "" is "auto"
	headsPerRank int    // query heads per tensor-parallel rank
	sm100        bool
	preHopper    bool // a part below compute capability 9.0 (see Kernel.preHopper)
	tp, dcp, pcp int
	dcpComm      string // the resolved DCP combine backend
}

// sparseMLABackend returns the backend v0.31.0 runs a DSA layer on and the cache layout that
// backend allocates, or an error naming why the engine would not start.
//
// SELECTION is the platform's priority list, each candidate filtered by its own checks
// (CudaPlatform.get_valid_backends and validate_configuration; the lists at
// vllm/platforms/cuda.py:95-153). For a DSA head the lists are:
//
//   - SM100: FlashInfer then FlashMLA-sparse for a quantized cache, and for a 16-bit one
//     when a rank holds at most 16 heads; FlashMLA-sparse first otherwise.
//   - SM90: FlashAttention's sparse MLA, FlashMLA-sparse, FlashInfer's SM90 sparse MLA.
//
// and a candidate is skipped where it does not list the cache dtype, or the layout uses a
// context parallelism it does not support:
//
//   - FlashAttention sparse: auto, float16, bfloat16 (flashattn_mla_sparse.py:36-40); no DCP
//     (:99-100).
//   - FlashMLA-sparse: auto, bfloat16, fp8_ds_mla, "fp8" as an alias for it, nvfp4_ds_mla
//     (flashmla_sparse.py:133-139), the last on SM100 only (:218-222); supports DCP (:680).
//   - FlashInfer (SM100): auto, float16, bfloat16, fp8, fp8_e4m3, and not fp8_ds_mla
//     (flashinfer_mla_sparse.py:76-82, :126-129); not PCP with DCP (:117-125).
//   - FlashInfer SM90: auto, bfloat16, fp8, fp8_e4m3 (flashinfer_mla_sparse_sm90.py:141-146);
//     no DCP (its implementation inherits supports_dcp = False,
//     vllm/v1/attention/backend.py:845, checked at :345-346).
//
// LAYOUT. FlashMLA-sparse serves any quantized cache other than nvfp4_ds_mla as fp8_ds_mla
// (_canonicalize_sparse_mla_kv_cache_dtype, mla_attention.py:358-375), which is a 656-byte
// cell rather than the 576 bytes "fp8" would hold.
//
// FLASHMLA-SPARSE'S OWN DCP CHECKS run after it is chosen, so they refuse the layout rather
// than pass it to another backend. Its implementation refuses a 16-bit cache under DCP ("DCP
// for FlashMLA sparse requires an fp8_ds_mla kv-cache", FlashMLASparseImpl.__init__,
// flashmla_sparse.py:745-751). Its metadata builder (:409-460) refuses four more:
//
//   - a combine other than ag_rs;
//   - PCP with DCP on a cache that is not fp8_ds_mla, the format its gather upconverts;
//   - DCP without PCP at 32 or more heads a rank (MIN_HEADS_FOR_BF16_PREFILL, :73), where
//     the mixed-batch path that returns a log-sum-exp for every row is not taken;
//   - a layout whose local and DCP-gathered head counts pad to different fp8 decode sizes,
//     64 up to 64 heads and 128 above (:684-687). Without PCP the gathered count is
//     heads x dcp; with PCP it is heads x tp where dcp exceeds pcp, else heads.
//
// COVERAGE LIMITS: an "auto" cache is taken as 16-bit, which it is unless the checkpoint
// states a KV quantization -- a ModelOpt checkpoint's kv_cache_quant_algo turns "auto" into
// fp8_e4m3 or nvfp4 before any backend is chosen (MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP and
// resolve_kv_cache_dtype_string, vllm/utils/torch_utils.py:67-70, :448-521, called from
// vllm/engine/arg_utils.py:2199-2202), and the graph does not record it; FlashInfer's SM90
// backend needs FlashInfer 0.6.18 or later (flashinfer_mla_sparse_sm90.py:194-199), which
// this assumes is installed; FlashInfer (SM100) checks the model's qk_nope_head_dim, which
// the DSA family satisfies (128 and 192, within its [128, 192]) and
// the graph does not state; and the SM120 parts' backend, which repacks "auto" too, has no
// part in the catalog.
func sparseMLABackend(r sparseMLABackendRequest) (backend, layout string, err error) {
	if r.preHopper {
		return "", "", fmt.Errorf("no sparse-MLA backend in vLLM v0.31.0 runs below compute " +
			"capability 9.0 -- each requires SM90, SM100 or SM120 -- so the engine refuses a " +
			"DSA model on this part at startup")
	}
	cache := r.cache
	if cache == "" {
		cache = "auto"
	}
	quantized := strings.HasPrefix(cache, "fp8") || strings.HasPrefix(cache, "nvfp4") ||
		strings.HasSuffix(cache, "per_token_head")
	lists := func(dtypes ...string) bool {
		for _, d := range dtypes {
			if d == cache {
				return true
			}
		}
		return false
	}
	accepts := map[string]bool{
		backendFlashAttnMLASparse: r.dcp <= 1 && lists("auto", "float16", "bfloat16"),
		backendFlashMLASparse: lists("auto", "bfloat16", "fp8_ds_mla", "fp8") ||
			(r.sm100 && lists("nvfp4_ds_mla")),
		backendFlashInferMLASparse: (r.pcp <= 1 || r.dcp <= 1) &&
			lists("auto", "float16", "bfloat16", "fp8", "fp8_e4m3"),
		backendFlashInferSM90: r.dcp <= 1 && lists("auto", "bfloat16", "fp8", "fp8_e4m3"),
	}
	var order []string
	switch {
	case !r.sm100:
		order = []string{backendFlashAttnMLASparse, backendFlashMLASparse, backendFlashInferSM90}
	case quantized || r.headsPerRank <= 16:
		order = []string{backendFlashInferMLASparse, backendFlashMLASparse}
	default:
		order = []string{backendFlashMLASparse, backendFlashInferMLASparse}
	}
	for _, b := range order {
		if accepts[b] {
			backend = b
			break
		}
	}
	if backend == "" {
		return "", "", fmt.Errorf("no sparse-MLA backend in vLLM v0.31.0 serves a %q cache "+
			"with dcp %d, pcp %d on this part (candidates %s): the engine refuses this "+
			"layout at startup", cache, r.dcp, r.pcp, strings.Join(order, ", "))
	}
	layout = cache
	if backend == backendFlashMLASparse && quantized && cache != "nvfp4_ds_mla" {
		layout = "fp8_ds_mla"
	}
	if backend == backendFlashMLASparse && r.dcp > 1 {
		padded := func(heads int) int {
			if heads <= 64 {
				return 64
			}
			return 128
		}
		gathered := r.headsPerRank * r.dcp
		if r.pcp > 1 {
			gathered = r.headsPerRank
			if r.dcp > r.pcp {
				gathered = r.headsPerRank * r.tp
			}
		}
		switch {
		case !quantized:
			return "", "", fmt.Errorf("%s runs decode-context parallelism only on a "+
				"quantized cache, and %q is 16-bit: the engine refuses this layout at startup",
				backend, cache)
		case r.dcpComm != DCPAllGatherReduceScatter:
			return "", "", fmt.Errorf("%s runs decode-context parallelism only with the %q "+
				"combine, and %q is resolved: the engine refuses this layout at startup",
				backend, DCPAllGatherReduceScatter, r.dcpComm)
		case r.pcp > 1 && layout != "fp8_ds_mla":
			return "", "", fmt.Errorf("%s runs prefill- with decode-context parallelism "+
				"only on an fp8_ds_mla cache, and this one is %q: the engine refuses this "+
				"layout at startup", backend, layout)
		case r.pcp <= 1 && r.headsPerRank >= 32:
			return "", "", fmt.Errorf("%s runs decode-context parallelism only with fewer "+
				"than 32 heads a rank, and this layout puts %d on each: the engine refuses "+
				"this layout at startup", backend, r.headsPerRank)
		case padded(r.headsPerRank) != padded(gathered):
			return "", "", fmt.Errorf("%s runs decode-context parallelism only where the "+
				"local and gathered head counts pad alike, and %d heads a rank pad to %d "+
				"while the %d gathered pad to %d: the engine refuses this layout at startup",
				backend, r.headsPerRank, padded(r.headsPerRank), gathered, padded(gathered))
		}
	}
	return backend, layout, nil
}

// specMethod is a speculative method's name as vLLM v0.31.0 runs it: every model-specific
// MTP name (deepseek_mtp, nemotron_h_mtp, ... -- MTPModelTypes) is replaced with mtp
// (vllm/config/speculative.py:39-67, 1135-1139).
func specMethod(m string) string {
	if strings.HasSuffix(m, "_mtp") {
		return "mtp"
	}
	return m
}
