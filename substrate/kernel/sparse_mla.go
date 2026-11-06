package latencykernel

import (
	"fmt"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// Which attention backend serves a DSA sparse-MLA layer, as vLLM v0.31.0 chooses it, and
// what that choice decides about the deployment: the cache layout the engine allocates, the
// block size it runs, and the context-parallel layouts it refuses.
//
// SCOPE. This is the DSA family -- DeepSeek-V3.2 and GLM-5, served by DeepseekV32Attention
// (vllm/models/deepseek_v32/attention.py) -- which the catalog states as a sparse_mla layer
// with a 576-wide head (kv_lora_rank 512 plus a 64-wide rope; MLAAttention sets head_size
// to that sum, vllm/model_executor/layers/attention/mla_attention.py:479). DeepSeek-V4's
// sparse_mla layers, 512 wide, are served by that model's own backends
// (vllm/models/deepseek_v4/sparse_mla.py), which this does not describe.
//
// PARTS. The catalog's parts that run a sparse-MLA backend at all are Hopper (SM90) and
// data-center Blackwell (SM100). Every CUDA sparse backend v0.31.0 ships requires one of the
// two, except FlashInfer's SM120 backend, which requires SM120 and has no part in the catalog
// (supports_compute_capability in vllm/v1/attention/backends/mla/flashattn_mla_sparse.py,
// flashmla_sparse.py, flashinfer_mla_sparse.py, flashinfer_mla_sparse_sm90.py). The catalog
// states each part's compute capability, so a part below 9.0 is recognised by that
// capability and not by its name; data-center Blackwell is recognised, as elsewhere in this
// kernel, by native NVFP4 support. A part whose capability is missing or unparseable is
// refused on this path, naming the part. A DSA model on a pre-Hopper part is refused: no
// sparse-MLA backend runs there.

// preHopper reports whether the chip's compute capability is below 9.0, which no sparse-MLA
// backend runs on. The capability is the kernel's family (the major version of the chip's
// stated ComputeCapability); a chip that states none, or states it malformed, is refused,
// because guessing a generation from a name or from peaks would price a part this path
// cannot place.
func (k *Kernel) preHopper() (bool, error) {
	if k.family <= 0 {
		return false, fmt.Errorf("chip %q states no usable ComputeCapability (%q), which the "+
			"DSA sparse-MLA backend choice needs to place the part below or at SM90",
			k.chip.Name, k.chip.ComputeCapability)
	}
	return k.family < 9, nil
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
			if n.Op == model.OpAttention && n.Role == "" &&
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

// hopper reports whether the chip is the SM90 family, which decides the kernels vLLM
// selects for several formats and layouts. The catalog states the compute capability; for a
// chip that does not, it is inferred -- data-center Blackwell by native NVFP4, everything
// else as Hopper -- and that inference is recorded the first time it decides anything.
func (k *Kernel) hopper() bool {
	if k.family > 0 {
		return k.family == 9
	}
	h := k.chip.NVFP4Peak <= 0
	if !k.familyAssumed {
		k.familyAssumed = true
		guess := "sm100"
		if h {
			guess = "sm90"
		}
		k.assume("compute_capability", guess,
			"the catalog states no compute capability for "+k.chip.Name+", so its generation "+
				"is inferred from its native-format peaks, which decide kernel "+
				"selection")
	}
	return h
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
		backendFlashInferMLASparse: !(r.pcp > 1 && r.dcp > 1) &&
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
		case r.dcpComm != resolve.DCPAllGatherReduceScatter:
			return "", "", fmt.Errorf("%s runs decode-context parallelism only with the %q "+
				"combine, and %q is resolved: the engine refuses this layout at startup",
				backend, resolve.DCPAllGatherReduceScatter, r.dcpComm)
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
