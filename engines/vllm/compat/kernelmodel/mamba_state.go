// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/model_executor/layers/mamba/mamba_utils.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// The recurrent layers' state shapes and dtypes: MambaStateShapeCalculator.mamba2_state_shape
// (:195-221), gated_delta_net_state_shape (:280-302), kda_state_shape (:304-327) and
// extra_groups_for_head_shards (:269-278); MambaStateDtypeCalculator._mamba_state_dtype (:98-110)
// and kda_state_dtype (:133-153). The spec each layer builds: MambaBase.get_kv_cache_spec
// (vllm/model_executor/layers/mamba/abstract.py:65-85), MambaMixer2.get_kv_cache_spec
// (mamba_mixer2.py:1098-1108), KimiK3DeltaAttention.get_kv_cache_spec
// (vllm/models/kimi_k3/nvidia/kda.py:829-837) with resolve_kda_prefill_backend (:463-502) and
// kda_prefill_checkpoint_alignment (vllm/model_executor/layers/mamba/kda_checkpoint.py:16-17).
// The engine settings the model hooks set: MambaModelConfig, NemotronHForCausalLMConfig and
// Qwen3_5ForConditionalGenerationConfig (vllm/model_executor/models/config.py:672-733, 853-878);
// the hybrid block alignment is core/vllm.AlignHybridBlockSize.

package kernelmodel

import (
	"errors"
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-vllm/core/vllm"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// kdaGateLowerBoundAssumption is the one fact the KDA prefill backend turns on that the model
// graph does not carry: whether the checkpoint states linear_attn_config.gate_lower_bound
// (FlashKDA needs a bounded gate, kda.py:341-357@5df63a72). The catalog's kimi-k3 config.json
// states -5.0, and this package takes every KDA graph to be a checkpoint that does.
const kdaGateLowerBoundAssumption = "kda_prefill_backend = flashkda: the graph does not carry the checkpoint's " +
	"linear_attn_config.gate_lower_bound, which FlashKDA needs bounded (vllm/models/kimi_k3/nvidia/kda.py:341-357@5df63a72); " +
	"the catalog's Kimi-K3 config states -5.0, and a KDA graph is taken to state one. A checkpoint without it runs the " +
	"triton backend, which takes no prefill checkpoint block"

// mambaEnv is what the recurrent layers' spec turns on beside the graph and the engine block.
type mambaEnv struct {
	// tp is the pool's tensor-parallel width; dcp and pcp its decode and prefill context
	// parallel widths, which a recurrent layer does not shard under (MambaBase.supports_dcp is
	// False, abstract.py:30).
	tp, dcp, pcp int
	// capability is the chip's compute capability major version, 0 when unknown.
	capability int
	// maxModelLen is the window the engine serves when it is stated or the model's default, 0 when
	// the engine fits it to memory.
	maxModelLen int
}

// mambaPlan is what the model's recurrent layers resolve to: the one KV cache spec each builds,
// and the engine block size the hybrid alignment leaves.
type mambaPlan struct {
	// node is the geometry every recurrent layer of the model shares.
	node geometry
	// blockSize is the engine's block_size after the hybrid alignment; source says how.
	blockSize int
	source    string
	// run is the "mamba" run one layer is (Count 1).
	run KVLayerRun
	// assumptions are the facts the plan could not read from the graph, as pricing assumptions.
	assumptions []string
}

// geometry is the fields of a recurrent node that decide its state, so two layers share one spec
// only when they agree.
type geometry struct {
	kind                                                   model.RecurrentKind
	heads, stateSize, groups, convKernel, intermediateSize int
	stateDType                                             model.DType
}

func recurrentGeometry(n model.Node) geometry {
	return geometry{n.RecurrentKind, n.NumHeads, n.StateSize, n.NumGroups, n.ConvKernel, n.IntermediateSize, n.StateDType}
}

// stackNodes calls f for every node of the layer kinds the stack names, in model order.
func stackNodes(g *model.Graph, stack model.Stack, f func(kind string, n model.Node)) {
	kinds := map[string]model.LayerKind{}
	for _, k := range g.LayerKinds {
		kinds[k.ID] = k
	}
	for _, id := range stack.Expand() {
		for _, n := range kinds[id].Nodes {
			f(id, n)
		}
	}
}

// hasRecurrent is whether the target's stack holds a recurrent layer.
func hasRecurrent(g *model.Graph) bool {
	if g == nil {
		return false
	}
	found := false
	stackNodes(g, g.Stack, func(_ string, n model.Node) { found = found || n.Op == model.OpRecurrentUpdate })
	return found
}

// planMamba is the spec vLLM v0.32.0 builds for each of a graph's recurrent layers, and the block
// size its platform leaves the engine, for the engine block e (its unstated cache dtype already
// resolved). It is nil for a graph with no recurrent layer. What it does not read it refuses by
// name, naming the setting or the graph field.
func planMamba(g *model.Graph, e engineconfig.Engine, env mambaEnv) (*mambaPlan, error) {
	if !hasRecurrent(g) {
		return nil, nil
	}
	refuse := func(format string, a ...any) (*mambaPlan, error) {
		return nil, fmt.Errorf("recurrent layers: "+format, a...)
	}
	var first *model.Node
	var mixed error
	stackNodes(g, g.Stack, func(id string, n model.Node) {
		if n.Op != model.OpRecurrentUpdate || mixed != nil {
			return
		}
		if first == nil {
			c := n
			first = &c
		} else if recurrentGeometry(*first) != recurrentGeometry(n) {
			mixed = fmt.Errorf("layer kind %s: a %s state of another geometry than the model's other recurrent layers; "+
				"vLLM sizes one Mamba page for the model (Platform._align_hybrid_block_size, "+
				"vllm/platforms/interface.py:1040-1056@5df63a72), so a model of several is not mapped", id, n.RecurrentKind)
		}
	})
	if mixed != nil {
		return refuse("%v", mixed)
	}
	n := *first
	if sp := g.Speculator; sp != nil && e.Speculative != nil && e.Speculative.NumSpecTokens > 0 {
		var drafter error
		stackNodes(g, sp.Stack, func(id string, d model.Node) {
			if d.Op == model.OpRecurrentUpdate && drafter == nil {
				drafter = fmt.Errorf("drafter layer kind %s holds a recurrent layer, whose state is not mapped", id)
			}
		})
		if drafter != nil {
			return refuse("%v", drafter)
		}
	}
	if env.dcp > 1 || env.pcp > 1 {
		return refuse("decode_context_parallel_size %d and prefill_context_parallel_size %d: a recurrent layer's state is "+
			"replicated across context-parallel ranks (MambaBase.supports_dcp is False, "+
			"vllm/model_executor/layers/mamba/abstract.py:26@5df63a72), which is not modelled", env.dcp, env.pcp)
	}

	// The cache mode that runs: MambaModelConfig sets "align" when prefix caching is on and "none"
	// when it is off (config.py:685-709@5df63a72). The mode's values in v0.32.0 are align and none
	// (vllm/config/cache.py:63), so any other is refused rather than run as one of them.
	switch e.MambaCacheMode {
	case "", "none", "align":
	default:
		return refuse("mamba_cache_mode %q is not a v0.32.0 mode (align and none, vllm/config/cache.py:63@5df63a72)", e.MambaCacheMode)
	}
	mode := "none"
	if e.EnablePrefixCaching == nil || *e.EnablePrefixCaching {
		mode = "align"
	}

	numSpec := 0
	if sp := e.Speculative; sp != nil && sp.NumSpecTokens > 0 {
		numSpec = sp.NumSpecTokens
	}
	modelBytes, modelBF16 := modelDType(g)
	convBytes, ssmBytes, err := stateDTypes(n, e, modelBytes)
	if err != nil {
		return refuse("%v", err)
	}
	conv, ssm, err := stateElems(n, env.tp, numSpec)
	if err != nil {
		return refuse("%v", err)
	}
	state := conv*int64(convBytes) + ssm*int64(ssmBytes)

	plan := &mambaPlan{node: recurrentGeometry(n)}
	run := KVLayerRun{Count: 1, Spec: "mamba", StateBytes: int(state), MambaCacheMode: mode,
		NumSpeculativeBlocks: numSpec}
	switch n.RecurrentKind {
	case model.RecurrentMamba2:
		run.NumPrefillCheckpointBlocks, run.PrefillCheckpointAlignment = 1, 1
	case model.RecurrentKDA:
		if kdaPrefillBackend(env.capability, modelBF16, n.StateSize, ssmBytes) == "flashkda" {
			run.NumPrefillCheckpointBlocks, run.PrefillCheckpointAlignment = 1, 16
			plan.assumptions = append(plan.assumptions, kdaGateLowerBoundAssumption)
		}
	}

	// The window a "none" mode block covers is max_model_len (config.py:707-709@5df63a72).
	mambaBlock := 0
	if mode == "none" {
		if env.maxModelLen <= 0 {
			return refuse("mamba_cache_mode none sets mamba_block_size to max_model_len, which is not stated and which " +
				"the engine fits to the KV budget that this block size sizes")
		}
		mambaBlock = env.maxModelLen
	}

	attn, err := attentionOf(g, e, env.tp)
	if err != nil {
		return refuse("%v", err)
	}
	if !attn.present {
		// No attention layer, so no attention backend: update_block_size_for_backend returns
		// before it chooses or aligns a block size (interface.py:763-766@5df63a72), and the
		// block_size is the stated one or CacheConfig.DEFAULT_BLOCK_SIZE (cache.py:72,345).
		plan.blockSize, plan.source = e.BlockSize, "stated"
		if plan.blockSize <= 0 {
			plan.blockSize, plan.source = 16, "CacheConfig.DEFAULT_BLOCK_SIZE (vllm/config/cache.py:72@5df63a72): the model has no "+
				"attention layer, so no attention backend aligns it (vllm/platforms/interface.py:763-766@5df63a72)"
		}
		run.BlockSize = mambaBlock
		if mode == "align" {
			run.BlockSize = plan.blockSize
		}
		plan.run = run
		return plan, nil
	}
	block := e.BlockSize
	if block <= 0 {
		switch {
		case attn.mla:
			// An MLA model's alignment is at least 128 tokens, and every dense MLA backend's smallest
			// kernel block (16, 32, 64 or 128) divides it, so the size the backends would have chosen
			// first, at most 128, does not change the result.
			block = 16
		case attn.kernelErr != nil:
			return refuse("block_size is not stated, and %v (%s)", attn.kernelErr, vllm.BlockSizeIssue)
		default:
			// Every non-MLA CUDA attention backend accepts a 16-token block (FlashAttention
			// flash_attn.py:355-368, FlashInfer flashinfer.py:460-478, Triton triton_attn.py:315-316
			// @5df63a72), so the platform keeps CacheConfig.DEFAULT_BLOCK_SIZE
			// (_preferred_block_size_for_backends, vllm/platforms/interface.py:700-745@5df63a72), for
			// the head sizes attentionOf accepts.
			block = 16
			plan.assumptions = append(plan.assumptions, "block_size = 16 before the hybrid alignment: unstated, and every "+
				"non-MLA CUDA attention backend accepts a 16-token block, so vLLM keeps CacheConfig.DEFAULT_BLOCK_SIZE "+
				"(vllm/platforms/interface.py:700-745@5df63a72); the block-size oracle does not run for a model with recurrent layers")
		}
	}
	if attn.kernelErr != nil {
		return refuse("%v", attn.kernelErr)
	}
	if attn.mla && block > 128 && block%128 != 0 {
		return refuse("block_size %d beside MLA attention: above 128 and not a multiple of it, the result depends on the least "+
			"common multiple of the MLA backends' smallest kernel blocks, which the install selects", block)
	}
	hb, err := vllm.AlignHybridBlockSize(vllm.HybridBlockInput{
		Block: block, AttnPage1Token: attn.page1Token, MambaPage: state,
		KernelBlockMin: attn.kernelMin, BackendBlockLCM: attn.kernelMin, MLA: attn.mla,
		MambaCacheMode: mode, MambaBlockSize: mambaBlock,
	})
	if err != nil {
		return refuse("%v", err)
	}
	plan.blockSize = hb.BlockSize
	plan.source = fmt.Sprintf("vLLM v0.32.0 aligns a hybrid model's block_size so the attention page (%d bytes a token) holds the "+
		"%d-byte Mamba page (Platform._align_hybrid_block_size, vllm/platforms/interface.py:939-1126@5df63a72)", attn.page1Token, state)
	run.BlockSize, run.PageSizePadded = hb.MambaBlockSize, int(hb.PageSizePadded)
	plan.run = run
	return plan, nil
}

// modelDType is the width of the model's activations, which the conv state takes under
// mamba_cache_dtype auto (MambaStateDtypeCalculator._mamba_state_dtype,
// mamba_utils.py:98-110@5df63a72), and whether they are bfloat16. The graph states the weights'
// dtype, not the model's: a weight format narrower than 16 bits is a storage width and the model
// computes in bf16, as the kernel reads it; a 16- or 32-bit format is its own compute width.
func modelDType(g *model.Graph) (bytes int, bf16 bool) {
	w := g.Global.WeightDType
	switch w {
	case model.DTypeFP32:
		return 4, false
	case model.DTypeFP16:
		return 2, false
	}
	return 2, true
}

// dtypeBytes is the width of a stored state dtype the graph names.
func dtypeBytes(d model.DType) (int, bool) {
	switch d {
	case model.DTypeFP32:
		return 4, true
	case model.DTypeBF16, model.DTypeFP16:
		return 2, true
	}
	return 0, false
}

// stateDTypes is the bytes of the conv state and of the temporal (SSM) state, as the layer's
// state-dtype calculator resolves mamba_cache_dtype and mamba_ssm_cache_dtype.
//
// The conv state is mamba_cache_dtype, the model's dtype under auto. The temporal state is
// mamba_ssm_cache_dtype when stated. Under auto, KDA's is float32 (kda_state_dtype,
// mamba_utils.py:133-153@5df63a72); a Mamba-2 layer's (NemotronH) and a GDN layer's (Qwen3.5) is
// what the model's hook sets, the checkpoint's value (config.py:712-733, 853-866@5df63a72),
// which the graph states as the node's state_dtype, and a graph without it is refused by name.
func stateDTypes(n model.Node, e engineconfig.Engine, modelBytes int) (conv, ssm int, err error) {
	stated := map[string]int{"float32": 4, "float16": 2, "bfloat16": 2}
	switch c := e.MambaCacheDType; c {
	case "", "auto":
		conv = modelBytes
	default:
		b, ok := stated[c]
		if !ok {
			return 0, 0, fmt.Errorf("mamba_cache_dtype %q is not auto, float32, float16 or bfloat16 (vllm/config/cache.py:62@5df63a72)", c)
		}
		conv = b
	}
	s := e.MambaSSMCacheDType
	if s == "" || s == "auto" {
		node, named := dtypeBytes(n.StateDType)
		switch {
		case n.RecurrentKind == model.RecurrentKDA:
			if n.StateDType != "" && n.StateDType != model.DTypeFP32 {
				return 0, 0, fmt.Errorf("the %s node states state_dtype %s, and KDA's recurrent state is float32 under "+
					"mamba_ssm_cache_dtype auto (vllm/model_executor/layers/mamba/mamba_utils.py:133-153@5df63a72)", n.RecurrentKind, n.StateDType)
			}
			return conv, 4, nil
		case !named:
			return 0, 0, fmt.Errorf("the %s node states no state_dtype (fp32, bf16 or fp16), which the model's hook reads from the "+
				"checkpoint to resolve mamba_ssm_cache_dtype auto (vllm/model_executor/models/config.py:712-733, 853-866@5df63a72)", n.RecurrentKind)
		}
		return conv, node, nil
	}
	b, ok := stated[s]
	if !ok {
		return 0, 0, fmt.Errorf("mamba_ssm_cache_dtype %q is not auto, float32, float16 or bfloat16 (vllm/config/cache.py:62@5df63a72)", s)
	}
	if n.RecurrentKind == model.RecurrentKDA && s != "float32" && s != "bfloat16" {
		return 0, 0, fmt.Errorf("mamba_ssm_cache_dtype %q: KDA recurrent state supports only auto, float32, and bfloat16 "+
			"(vllm/model_executor/layers/mamba/mamba_utils.py:142-150@5df63a72)", s)
	}
	return conv, b, nil
}

// stateElems is the elements of the conv state and of the temporal state one rank holds for one
// layer, from the shape calculators (mamba_utils.py:195-327@5df63a72), in the "SD" layout whose
// element count the layout does not change. numSpec widens the conv state's window.
func stateElems(n model.Node, tp, numSpec int) (conv, ssm int64, err error) {
	tp = max(tp, 1)
	div := func(what string, v int) (int64, error) {
		if v <= 0 || v%tp != 0 {
			return 0, fmt.Errorf("%s %d is not divisible by tensor_parallel_size %d", what, v, tp)
		}
		return int64(v / tp), nil
	}
	need := func(field string, v int) error {
		if v <= 0 {
			return fmt.Errorf("the %s node states no %s", n.RecurrentKind, field)
		}
		return nil
	}
	for _, f := range []struct {
		name string
		v    int
	}{{"n_heads", n.NumHeads}, {"state_size", n.StateSize}, {"conv_kernel", n.ConvKernel}} {
		if err := need(f.name, f.v); err != nil {
			return 0, 0, err
		}
	}
	width := int64(n.ConvKernel - 1 + numSpec)
	switch n.RecurrentKind {
	case model.RecurrentMamba2:
		if err := errors.Join(need("n_groups", n.NumGroups), need("intermediate_size", n.IntermediateSize)); err != nil {
			return 0, 0, err
		}
		if n.IntermediateSize%n.NumHeads != 0 {
			return 0, 0, fmt.Errorf("the mamba2 node's intermediate_size %d is not divisible by n_heads %d, so its head dimension "+
				"is not whole", n.IntermediateSize, n.NumHeads)
		}
		headDim := n.IntermediateSize / n.NumHeads
		groups := n.NumGroups + extraGroupsForHeadShards(n.NumGroups, tp)
		cdim, err := div("the conv state's width", n.IntermediateSize+2*groups*n.StateSize)
		if err != nil {
			return 0, 0, err
		}
		heads, err := div("n_heads", n.NumHeads)
		if err != nil {
			return 0, 0, err
		}
		return cdim * width, heads * int64(headDim) * int64(n.StateSize), nil
	case model.RecurrentKDA:
		// num_k_heads = num_heads and head_k_dim = head_dim = state_size (kda_state_shape,
		// mamba_utils.py:304-327@5df63a72), so the conv state spans q, k and v.
		cdim, err := div("the conv state's width", 3*n.NumHeads*n.StateSize)
		if err != nil {
			return 0, 0, err
		}
		heads, err := div("n_heads", n.NumHeads)
		if err != nil {
			return 0, 0, err
		}
		return cdim * width, heads * int64(n.StateSize) * int64(n.StateSize), nil
	case model.RecurrentGDN:
		// The conv state spans q, k and v: intermediate_size is its whole width, 2 x
		// num_k_heads x head_k_dim + num_v_heads x head_v_dim, and the temporal state is
		// (num_v_heads, head_v_dim, head_k_dim) with n_heads the value heads and state_size both
		// head dimensions (gated_delta_net_state_shape, mamba_utils.py:280-302@5df63a72).
		if err := need("intermediate_size", n.IntermediateSize); err != nil {
			return 0, 0, err
		}
		cdim, err := div("the conv state's width", n.IntermediateSize)
		if err != nil {
			return 0, 0, err
		}
		heads, err := div("n_heads", n.NumHeads)
		if err != nil {
			return 0, 0, err
		}
		return cdim * width, heads * int64(n.StateSize) * int64(n.StateSize), nil
	}
	return 0, 0, fmt.Errorf("recurrent kind %q is not mapped", n.RecurrentKind)
}

// extraGroupsForHeadShards is MambaStateShapeCalculator.extra_groups_for_head_shards
// (mamba_utils.py:269-278@5df63a72): the groups added so every head's group is sharded with it.
func extraGroupsForHeadShards(groups, tp int) int {
	if groups%tp == 0 {
		return 0
	}
	return tp - groups
}

// kdaPrefillBackend is resolve_kda_prefill_backend for the default additional_config (backend
// "auto": the engine block cannot state kda_prefill_backend), vllm/models/kimi_k3/nvidia/kda.py:
// 463-502@5df63a72. "auto" takes FlashKDA where the chip is compute capability 9, 10 or 12, the
// head is 128 wide, the model computes in bfloat16, the recurrent state is bfloat16 or float32 and
// the gate is bounded (is_flashkda_supported, :341-357), and otherwise triton. FlashInfer is chosen
// only when named. The gate bound is the graph's missing fact (kdaGateLowerBoundAssumption).
func kdaPrefillBackend(capability int, modelBF16 bool, headDim, ssmBytes int) string {
	switch capability {
	case 9, 10, 12:
		if headDim == 128 && modelBF16 && (ssmBytes == 2 || ssmBytes == 4) {
			return "flashkda"
		}
	}
	return "triton"
}

// attentionFacts is what the hybrid alignment reads of the model's attention layers.
type attentionFacts struct {
	present bool
	mla     bool
	// kernelErr is why the attention backend's smallest kernel block is not known.
	kernelErr error
	// page1Token is the page of one token on a rank in bytes; kernelMin the smallest kernel block
	// of the attention backends, 16 for every non-MLA CUDA backend (FlashAttention
	// flash_attn.py:355-368, FlashInfer flashinfer.py:460-478 and Triton triton_attn.py:315-316
	// @5df63a72).
	page1Token int64
	kernelMin  int
}

// attentionOf reads the model-level attention geometry _align_hybrid_block_size uses
// (model_config.get_num_kv_heads, get_head_size, use_mla, interface.py:970-1038@5df63a72) from
// the target's and its drafter's attention nodes, and refuses a model whose attention it cannot
// reduce to one.
func attentionOf(g *model.Graph, e engineconfig.Engine, tp int) (attentionFacts, error) {
	var out attentionFacts
	var shape [3]int
	var err error
	visit := func(id string, n model.Node) {
		if n.Op != model.OpAttention || err != nil {
			return
		}
		mla := n.AttentionKind == model.AttentionMLA
		kv, head := max(n.NumKVHeads/max(tp, 1), 1), n.HeadDim
		if mla {
			kv = 1
		}
		geo := [3]int{kv, head, 0}
		if mla {
			geo[2] = 1
		}
		switch {
		case n.AttentionKind != model.AttentionMLA && n.AttentionKind != model.AttentionGQA && n.AttentionKind != model.AttentionSWA:
			err = fmt.Errorf("layer kind %s: %s attention beside recurrent layers is not mapped", id, n.AttentionKind)
		case len(n.Caches) > 0:
			err = fmt.Errorf("layer kind %s: an attention node stating caches beside recurrent layers is not mapped", id)
		case head <= 0 || n.NumKVHeads <= 0:
			err = fmt.Errorf("layer kind %s: the attention node states no n_kv or d_h", id)
		case !mla && head > 128 && out.kernelErr == nil:
			// FlashAttention 4 selects its own kernel block for head sizes of 256 and for fp8 KV at
			// head size 512 (flash_attn.py:296-352@5df63a72), which the install decides. It is
			// reported after an unstated block_size, which is the more basic gap.
			out.kernelErr = fmt.Errorf("layer kind %s: an attention head size %d beside recurrent layers: the attention backend's "+
				"kernel block depends on the FlashAttention version the install selects, which is not modelled", id, head)
		case out.present && geo != shape:
			err = fmt.Errorf("layer kind %s: attention layers of different geometry beside recurrent layers: vLLM aligns the "+
				"block size from one model-level page (model_config.get_head_size)", id)
		}
		out.present, out.mla, shape = true, mla, geo
	}
	stackNodes(g, g.Stack, visit)
	if sp := g.Speculator; sp != nil && e.Speculative != nil && e.Speculative.NumSpecTokens > 0 {
		stackNodes(g, sp.Stack, visit)
	}
	if err != nil || !out.present {
		return out, err
	}
	elem, err := attentionElemBytes(e.CacheDType, g)
	if err != nil {
		return out, err
	}
	kv, head := int64(shape[0]), int64(shape[1])
	// FullAttentionSpec: K and V, MLAAttentionSpec: the latent alone
	// (vllm/v1/kv_cache_interface.py:511-575, 686-700@5df63a72).
	out.page1Token = kv * head * elem * 2
	out.kernelMin = 16
	if out.mla {
		out.page1Token = head * elem
		out.kernelMin = 64
	}
	return out, nil
}

// attentionElemBytes is the width of one element of the attention KV cache at cache_dtype: the
// model's dtype under auto, 2 for float16 and bfloat16, 1 for the fp8 formats. A packed format
// (nvfp4, the ds_mla layouts, turboquant, the per-token-head formats) has a page this alignment
// does not size, so it is refused.
func attentionElemBytes(cacheDType string, g *model.Graph) (int64, error) {
	switch cacheDType {
	case "", "auto":
		b, _ := modelDType(g)
		return int64(b), nil
	case "float16", "bfloat16":
		return 2, nil
	case "fp8", "fp8_e4m3", "fp8_e5m2":
		return 1, nil
	}
	return 0, fmt.Errorf("cache_dtype %q beside recurrent layers: its attention page is not sized by the hybrid block alignment", cacheDType)
}

// mambaRun is the "mamba" run a recurrent node in a layer kind is, or the reason it is not mapped.
func (p *mambaPlan) mambaRun(kind string, n model.Node) (KVLayerRun, error) {
	if p == nil {
		return KVLayerRun{}, fmt.Errorf("layer kind %s: recurrent state is not mapped to a KV cache spec: no recurrent plan "+
			"was resolved for this engine", kind)
	}
	if recurrentGeometry(n) != p.node {
		return KVLayerRun{}, fmt.Errorf("layer kind %s: a %s state of another geometry than the model's recurrent layers", kind, n.RecurrentKind)
	}
	return p.run, nil
}
