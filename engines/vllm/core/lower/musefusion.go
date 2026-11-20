// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/models/muse_spark/nvidia/model.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5):
// MuseSpark1xAttention (:1138-1146, :1167-1238), MuseSpark1xMoE (:505-539, :613-617, :638-644),
// MuseSpark1xDecoderLayer (:1456-1497), MuseSpark1xModel (:1533, :1577-1610), gather_columns
// (:158-167), _all_gather_norm_residual_then_norm (:214-252) and norm_residual_then_norm (:255-271). Also
// vllm/models/muse_spark/nvidia/muse_moe_runner.py (MusePostExpertFinalize, :151-313),
// vllm/models/muse_spark/nvidia/ops/mnnvl_fusions.py (:20-58, :61-82, :105-126),
// vllm/models/muse_spark/common/ops/fused_qk_norm_rope_kv.py (:17-38),
// vllm/models/muse_spark/common/moe.py (needs_pre_norm_all_reduce, :114-119) and
// vllm/platforms/cuda.py (_get_backend_priorities, :156-181).

package lower

import (
	"fmt"
	"slices"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// MuseSparkArchitectures are the architectures vLLM v0.33.0 builds with MuseSpark1xModel, the
// decoder these fusions belong to (MuseSpark1xForCausalLM and MuseSparkForConditionalGeneration,
// vllm/config/vllm.py:101-102).
var MuseSparkArchitectures = []string{"MuseSpark1xForCausalLM", "MuseSparkForConditionalGeneration"}

// MuseSparkLayout is what vLLM v0.33.0 reads to decide which of Muse Spark's fused kernels it
// builds. Each field is a resolved value: a setting the deployment left unstated carries vLLM's
// default.
type MuseSparkLayout struct {
	// CapabilityMajor is the GPU's compute-capability major: 9 for Hopper, 10 for datacenter
	// Blackwell. Other parts are refused (MuseSparkFusionsFor).
	CapabilityMajor int
	// TP, PP, DP, PCP and DCP are the tensor, pipeline, data, prefill-context and decode-context
	// parallel widths; 0 reads as 1.
	TP, PP, DP, PCP, DCP int
	// ExpertParallel is enable_expert_parallel.
	ExpertParallel bool
	// KVCacheDType is the KV-cache dtype vLLM runs with, in vLLM's spelling (cache_config's
	// cache_dtype: auto, bfloat16, fp8, fp8_e4m3, nvfp4, ...).
	KVCacheDType string
	// TPSpansNodes is tp_group_spans_nodes (mnnvl_fusions.py:20-23): the tensor-parallel group's
	// ranks sit on more than one host, whatever links join them.
	TPSpansNodes bool
	// Multicast is FlashInfer's is_multicast_supported on the device (mnnvl_fusions.py:40-46):
	// the GPUs share NVLink multicast memory. It decides whether either MNNVL workspace can be
	// built on capability family 10, and is read nowhere else.
	Multicast bool
}

func atLeastOne(n int) int { return max(n, 1) }

// MuseSparkFusions is which of the three fused kernels vLLM v0.33.0 builds for a Muse Spark
// deployment, and the step sizes at which each runs.
type MuseSparkFusions struct {
	// QKNormRopeKVInsert is MuseSpark1xAttention.fused_qk_norm_rope_kv_insert on the target
	// model's layers: QK-norm, RoPE and the KV-cache write run as one launch
	// (ops.fused_muse_spark_qknorm_rope_kv_insert, model.py:1178-1189), and the attention layer is
	// given no k and v so its backend writes nothing (:1235-1236). It does not depend on the step size.
	QKNormRopeKVInsert bool
	// MoEFinalizeAllReduceMaxTokens is the largest step that runs the MoE finalize, its
	// post-expert norm and the expert-parallel combine all-reduce as one launch
	// (moe_finalize_rmsnorm_allreduce, muse_moe_runner.py:252-274); 0 when the workspace is not
	// built. A larger step finalizes and then all-reduces separately.
	MoEFinalizeAllReduceMaxTokens int
	// ColumnGatherBoundary is that every sharded MoE layer's unpooler hands its columns to the
	// next boundary, which gathers them, applies the gated residual and the next RMSNorm as one
	// launch (all_gather_gated_residual_next_rmsnorm, model.py:214-252, :1577-1610).
	ColumnGatherBoundary bool
	// UnpoolerShardedMinTokens and UnpoolerShardedMaxTokens bound the steps on which the MoE
	// unpooler's output columns are split over the tensor-parallel ranks (shards_unpooler,
	// model.py:638-644); both 0 when it never is.
	UnpoolerShardedMinTokens, UnpoolerShardedMaxTokens int
}

// MoEFinalizeAllReduce reports whether a step of tokens runs the fused finalize all-reduce.
func (f MuseSparkFusions) MoEFinalizeAllReduce(tokens int) bool {
	return tokens > 0 && tokens <= f.MoEFinalizeAllReduceMaxTokens
}

// UnpoolerSharded reports whether a step of tokens splits the unpooler's output columns.
func (f MuseSparkFusions) UnpoolerSharded(tokens int) bool {
	return f.UnpoolerShardedMaxTokens > 0 && f.UnpoolerShardedMinTokens <= tokens && tokens <= f.UnpoolerShardedMaxTokens
}

// GatherDeferred reports whether a step of tokens defers the unpooler's gather to the fused
// boundary (_defers_column_gather, model.py:1493-1497). A sharded step that does not defer it
// gathers the columns with gather_columns before the boundary (model.py:263-271).
func (f MuseSparkFusions) GatherDeferred(tokens int) bool {
	return f.ColumnGatherBoundary && f.UnpoolerSharded(tokens)
}

// The bounds vLLM v0.33.0 compares a step's tokens with.
const (
	// _SHARDED_PROJECTION_MAX_TOKENS (model.py:151) and _MULTI_NODE_SHARDED_UNPOOLER_MIN_TOKENS
	// (:154).
	museShardedProjectionMaxTokens  = 256
	museMultiNodeShardedUnpoolerMin = 32
	museFusedAllReduceMaxTokens     = 256 // _FUSED_ALL_REDUCE_MAX_TOKENS, muse_moe_runner.py:151
	museMultiNodeFusedAllReduceMax  = 128 // _MULTI_NODE_FUSED_ALL_REDUCE_MAX_TOKENS, :152
)

// museFusedQKNormRopeHeadDims are the head sizes the fused QK-norm, RoPE and KV-insert kernel is
// built for (model.py:1143).
var museFusedQKNormRopeHeadDims = []int{64, 128, 256}

// museFusedKVInsertDTypes is SUPPORTED_KV_CACHE_DTYPES (fused_qk_norm_rope_kv.py:17-20): NVFP4
// pages a different layout and keeps the unfused path.
var museFusedKVInsertDTypes = []string{"auto", "bfloat16", "float16", "fp8", "fp8_e4m3", "fp8_e5m2"}

// MuseSparkFusionsFor resolves which fused kernels vLLM v0.33.0 builds for a Muse Spark target
// model whose attention heads are headDim wide, under the layout l, with vLLM's default
// attention backend, MoE backend and model options.
//
// The attention backend is the first valid one in the CUDA priority list: FLASHINFER on
// capability 10 and FLASH_ATTN on 9 (_get_backend_priorities, platforms/cuda.py:156-181; Muse
// Spark is not a multimodal-prefix model, and its target attention is causal). On 9 with an fp8
// KV cache and no FA3, FLASHINFER is chosen instead. Each of these keeps the KV write out of its
// forward and supports the bypass (flashinfer.py:555, :737; flash_attn.py:287, :369), so the fused
// insert runs unless the KV cache is NVFP4 or decode-context parallelism is on
// (supports_fused_kv_insert, fused_qk_norm_rope_kv.py:23-38). A stated attention_backend is not
// modelled (fields.json), so it is not read here. On 9 no backend takes an NVFP4 KV cache, so
// vLLM does not start, and that is refused.
//
// The two Blackwell fusions need tep_default (model.py:505-517): tensor parallelism above 1 with
// replicated tokens -- no MegaMoE (moe_backend deep_gemm_mega_moe, not modelled), so no sequence
// parallelism (:361-370); data parallelism 1, so no sequence-parallel MoE
// (config/parallel.py:768-785); no batch invariance (VLLM_BATCH_INVARIANT, unset) -- on capability
// 10 or above. The checkpoint's config states none of the muse_spark_tep_* overrides. Each MNNVL
// workspace is then built only on capability family 10 with NVLink multicast
// (_create_on_all_tp_ranks, mnnvl_fusions.py:32-58).
//
// The fused finalize all-reduce also needs the expert-parallel combine to be the late all-reduce
// of replicated tokens (_create_all_reduce, muse_moe_runner.py:200-216): expert parallelism over
// exactly the tensor-parallel ranks (ep_size = tp x pcp x dp equals tp), data and prefill-context
// parallelism 1, no all-to-all kernels and no sequence parallelism. Under expert parallelism the
// experts are not tensor-sharded (FusedMoEParallelConfig.make sets tp_size 1), so no pre-norm
// all-reduce is needed (needs_pre_norm_all_reduce, common/moe.py:114-119).
//
// The fused gather boundary also needs the model on one pipeline stage (fuse_across_layers,
// model.py:1533); muse_spark_tep_fused_boundary defaults to true (:1589).
func MuseSparkFusionsFor(headDim int, l MuseSparkLayout) (MuseSparkFusions, error) {
	var f MuseSparkFusions
	major := l.CapabilityMajor
	if major != 9 && major != 10 {
		return f, fmt.Errorf("lower: Muse Spark's fused kernels are resolved for compute capability 9 and 10, not %d: "+
			"another part's attention-backend priority list is not ported (inference-sim/blis-vllm#191)", major)
	}
	if major == 9 && l.KVCacheDType == "nvfp4" {
		return f, fmt.Errorf("lower: no attention backend vLLM v0.33.0 builds on compute capability 9 takes an NVFP4 KV cache " +
			"for Muse Spark's head size, so vLLM does not start (platforms/cuda.py get_valid_backends)")
	}
	tp, dp, pcp := atLeastOne(l.TP), atLeastOne(l.DP), atLeastOne(l.PCP)
	f.QKNormRopeKVInsert = atLeastOne(l.DCP) == 1 && slices.Contains(museFusedKVInsertDTypes, l.KVCacheDType) &&
		slices.Contains(museFusedQKNormRopeHeadDims, headDim)

	tepDefault := tp > 1 && dp == 1 && major >= 10
	workspace := major == 10 && tp > 1 && l.Multicast
	if !tepDefault {
		return f, nil
	}
	// shard_unpooler (model.py:521-523) and min_sharded_unpooler_tokens (:535-539).
	f.UnpoolerShardedMinTokens, f.UnpoolerShardedMaxTokens = 1, museShardedProjectionMaxTokens
	if l.TPSpansNodes {
		f.UnpoolerShardedMinTokens = museMultiNodeShardedUnpoolerMin
	}
	f.ColumnGatherBoundary = workspace && atLeastOne(l.PP) == 1
	epOverTP := l.ExpertParallel && tp*pcp*dp == tp
	if workspace && epOverTP && pcp == 1 {
		f.MoEFinalizeAllReduceMaxTokens = museFusedAllReduceMaxTokens
		if l.TPSpansNodes {
			f.MoEFinalizeAllReduceMaxTokens = museMultiNodeFusedAllReduceMax
		}
	}
	return f, nil
}

// The graph roles these fusions act on, as the catalog's Muse Spark graphs name them, and the
// launches the cost model's per-primitive table gives each (kernelsPerPrimitive,
// blis-substrate kernel/internal/price/plan.go:659-669 at f14a5415c5479a2d8ba995b787ed9a4312f1d5b3:
// one per Elementwise and AllReduce node; two per Attention node, the kernel and its cache write).
const (
	museRoleQKNormRope     = "qk_norm_rope"
	museRolePostExpertNorm = "post_expert_norm"
	museRoleMoECombine     = "moe_combine"
	museRoleInputNorm      = "input_norm"
	museRoleFinalNorm      = "final_norm"
)

// NodeLaunches is how many kernel launches one graph node makes in a step: as the cost model's
// per-primitive table counts them (Graph), and as vLLM v0.33.0 launches them (Engine). A launch a
// fused kernel makes is counted on the first node it replaces, and the nodes it absorbs launch
// nothing of their own.
type NodeLaunches struct {
	// Role is the node's graph role; "attention (cache write)" is the Attention node's second
	// launch, and "unpooler gather" the all-gather the graph prices as a time and counts no
	// launch for.
	Role          string
	Graph, Engine int
	// Why cites the vLLM code that decides Engine.
	Why string
}

// MuseSparkLayerLaunches is, for one Muse Spark decoder layer in a decode step of tokens, the
// launches of every node one of the three fusions decides, under the resolved fusions f.
// ep is enable_expert_parallel over more than one tensor-parallel rank with no all-to-all
// dispatch (data and prefill-context parallelism 1), the layout under which the graph's
// moe_combine all-reduce is emitted (expert_parallel_without_dispatch); without it the
// all-reduce and the post-expert norm are left as the graph counts them.
//
// A node a fusion leaves unfused is counted as the graph counts it, with two exceptions. The
// post-expert norm never launches on its own: it runs inside the MoE finalize on every finalize
// path an expert-parallel layer takes (CuTeDSL moe_finalize_with_rmsnorm on capability family 10,
// the Triton fused_post_expert_finalize otherwise, muse_moe_runner.py:288-313). And the unfused
// gather, which the graph counts no launch for, is gather_columns: one all-gather and, for more
// than one token, the copy its transpose needs (model.py:158-167). The unfused QK-norm and RoPE
// launches are not counted here (inference-sim/blis-vllm#191).
func MuseSparkLayerLaunches(f MuseSparkFusions, tokens int, ep bool) []NodeLaunches {
	out := make([]NodeLaunches, 0, 6)
	if f.QKNormRopeKVInsert {
		out = append(out,
			NodeLaunches{museRoleQKNormRope, 1, 1, "fused_qk_norm_rope_kv_insert, model.py:1178-1189"},
			NodeLaunches{"attention (cache write)", 1, 0, "k and v withheld from the attention layer, model.py:1235-1236"})
	} else {
		out = append(out,
			NodeLaunches{museRoleQKNormRope, 1, 1, "unfused QK-norm and RoPE, not counted (model.py:1198-1222, inference-sim/blis-vllm#191)"},
			NodeLaunches{"attention (cache write)", 1, 1, "the backend's KV-cache update"})
	}
	if ep {
		out = append(out, NodeLaunches{museRolePostExpertNorm, 1, 0,
			"inside the MoE finalize, muse_moe_runner.py:288-313"})
		if f.MoEFinalizeAllReduce(tokens) {
			out = append(out, NodeLaunches{museRoleMoECombine, 1, 0,
				"inside the finalize, moe_finalize_rmsnorm_allreduce, muse_moe_runner.py:252-274"})
		} else {
			out = append(out, NodeLaunches{museRoleMoECombine, 1, 1, "the separate combine all-reduce"})
		}
	}
	gather := 0
	why := "the unpooler is not sharded at this step size (shards_unpooler, model.py:638-644)"
	switch {
	case f.GatherDeferred(tokens):
		why = "inside the next boundary, all_gather_gated_residual_next_rmsnorm, model.py:214-252"
	case f.UnpoolerSharded(tokens):
		gather = 1
		if tokens > 1 {
			gather = 2
		}
		why = "gather_columns before the boundary: an all-gather, and a copy above one token, model.py:158-167"
	}
	out = append(out, NodeLaunches{"unpooler gather", 0, gather, why})
	return out
}

// MuseSparkStepLaunchChange is the change, for a decode step of tokens, from the launches the
// cost model's per-primitive table counts for graph g to those vLLM v0.33.0 makes, over the nodes
// the three fusions decide (MuseSparkLayerLaunches), summed over every layer of the target stack.
// It refuses a graph whose architecture is not Muse Spark's, or whose layers lack a node the
// fusions act on.
func MuseSparkStepLaunchChange(g *model.Graph, l MuseSparkLayout, tokens int) (int, error) {
	if g == nil || !slices.Contains(MuseSparkArchitectures, g.Global.Architecture) {
		return 0, fmt.Errorf("lower: Muse Spark's fusions apply to %v graphs only", MuseSparkArchitectures)
	}
	headDim := 0
	byID := map[string]model.LayerKind{}
	for _, lk := range g.LayerKinds {
		byID[lk.ID] = lk
		for _, role := range []string{museRoleInputNorm, museRoleQKNormRope, museRolePostExpertNorm, museRoleMoECombine} {
			if !slices.ContainsFunc(lk.Nodes, func(n model.Node) bool { return n.Role == role }) {
				return 0, fmt.Errorf("lower: layer kind %q has no %s node", lk.ID, role)
			}
		}
		for _, n := range lk.Nodes {
			if n.Op == model.OpAttention {
				if headDim != 0 && n.HeadDim != headDim {
					return 0, fmt.Errorf("lower: layer kind %q has head size %d, another has %d", lk.ID, n.HeadDim, headDim)
				}
				headDim = n.HeadDim
			}
		}
	}
	if !slices.ContainsFunc(g.Head, func(n model.Node) bool { return n.Role == museRoleFinalNorm }) {
		return 0, fmt.Errorf("lower: the head has no %s node, which the last layer's boundary feeds", museRoleFinalNorm)
	}
	f, err := MuseSparkFusionsFor(headDim, l)
	if err != nil {
		return 0, err
	}
	ep := l.ExpertParallel && atLeastOne(l.TP) > 1 && atLeastOne(l.DP) == 1 && atLeastOne(l.PCP) == 1
	per := 0
	for _, n := range MuseSparkLayerLaunches(f, tokens, ep) {
		per += n.Engine - n.Graph
	}
	layers := 0
	for _, id := range g.Stack.Expand() {
		if _, ok := byID[id]; !ok {
			return 0, fmt.Errorf("lower: the stack names layer kind %q, which the graph does not declare", id)
		}
		layers++
	}
	return per * layers, nil
}
