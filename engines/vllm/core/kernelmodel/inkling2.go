// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

package kernelmodel

import (
	"fmt"
	"math/bits"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"

	"github.com/inference-sim/inference-sim/engines/vllm/core/engineconfig"
)

// archInkling2 is the architecture of Inkling 2 Small.
const archInkling2 = "Inkling2ForConditionalGeneration"

// isInkling2 is whether g is Inkling 2.
func isInkling2(g *model.Graph) bool {
	return g != nil && g.Global.Architecture == archInkling2
}

// inkling2KVLayers is the KV cache specs Inkling 2 registers, in model order. Each decoder layer
// registers two caches, the conv state first (Inkling2DecoderLayer builds InklingConvState before
// Inkling2Attention, vllm/models/inkling2/nvidia/model.py@0fda86a3):
//
//   - the conv state, one sliding-window AttentionSpec of window and block size sconv_kernel_size,
//     holding the layer's four short-convolution streams (K, V, attention output, MLP output)
//     packed head-major, num_kv_heads/TP heads of one padded width a head
//     (InklingConvState, vllm/models/inkling/nvidia/sconv_swa_attn.py:122-209): the width is the
//     next power of two over 2 x head_dim + 2 x hidden_size / (local KV heads), because Inkling 2
//     keeps the attention- and MLP-output streams replicated (replicate_hidden=True) so each rank
//     spreads the full hidden width over its local heads. The spec has no value half
//     (head_size_v=0). Its page is far above the attention pages; unify_kv_cache_spec_page_size
//     scales the attention blocks up to it, which kv.GroupLayers does;
//   - the attention cache: a SlidingWindowSpec of sliding_window_size on a local layer, full
//     attention on the others (Inkling2Attention.get_kv_cache_config, attention.py:230-241), in
//     the model dtype (the layer asserts a cache dtype of auto or bfloat16).
//
// A DSpark drafter's layers register after the target's: each is a plain Attention of the
// target's cache config (Inkling2DraftSelfAttention, dspark.py:118-181), a sliding window on a
// local layer and full attention on the other.
func inkling2KVLayers(g *model.Graph, pool deployment.Pool, engine engineconfig.Engine, ds deepseekServing) ([]KVLayerRun, error) {
	if d := ds.cacheDType; d != "" && d != "auto" && d != "bfloat16" {
		return nil, fmt.Errorf("cache dtype %q: Inkling 2's attention asserts a cache dtype of auto or bfloat16 "+
			"(vllm/models/inkling2/nvidia/attention.py:215@0fda86a3)", d)
	}
	if sp := engine.Speculative; sp != nil && sp.NumSpecTokens > 0 {
		switch sp.KVCacheDType {
		case "", "auto", "bfloat16":
		default:
			return nil, fmt.Errorf("speculative kv_cache_dtype %q: the draft's cache is the model dtype's, and this mapping "+
				"sizes it bf16 only", sp.KVCacheDType)
		}
	}
	tp := max(pool.Parallel.TP, 1)
	kinds := map[string]model.LayerKind{}
	for _, k := range g.LayerKinds {
		kinds[k.ID] = k
	}
	var runs []KVLayerRun
	add := func(r KVLayerRun) {
		if l := len(runs) - 1; l >= 0 && sameRun(runs[l], r) {
			runs[l].Count++
			return
		}
		runs = append(runs, r)
	}
	attention := func(id string, n model.Node, draft bool) (KVLayerRun, error) {
		if n.AttentionKind != model.AttentionSWA && n.AttentionKind != model.AttentionGQA || len(n.Caches) > 0 {
			return KVLayerRun{}, fmt.Errorf("layer kind %s: only a grouped-query attention layer, windowed or full, without stated caches is mapped", id)
		}
		if n.NumKVHeads%tp != 0 || tp > n.NumKVHeads {
			return KVLayerRun{}, fmt.Errorf("layer kind %s: %d KV heads over tensor_parallel_size %d: Inkling 2 asserts tp_size <= num_kv_heads "+
				"and whole heads per rank (attention.py:137, sconv_swa_attn.py:144@0fda86a3)", id, n.NumKVHeads, tp)
		}
		r := KVLayerRun{Count: 1, Spec: "full", NumKVHeads: n.NumKVHeads / tp, HeadSize: n.HeadDim, ElemBytes: 2, Draft: draft}
		if n.AttentionKind == model.AttentionSWA {
			r.Spec, r.SlidingWindow = "sliding_window", n.Window
		}
		return r, nil
	}
	for _, id := range g.Stack.Expand() {
		var attn *model.Node
		kernel := 0
		for _, n := range kinds[id].Nodes {
			switch {
			case n.Op == model.OpAttention:
				if attn != nil {
					return nil, fmt.Errorf("layer kind %s: two attention nodes in a layer are not mapped", id)
				}
				c := n
				attn = &c
			case n.Op == model.OpRecurrentUpdate:
				if n.RecurrentKind != model.RecurrentShortConv || n.ConvKernel <= 0 || kernel != 0 && n.ConvKernel != kernel {
					return nil, fmt.Errorf("layer kind %s: a recurrent node that is not a short convolution of the layer's one kernel size is not mapped", id)
				}
				kernel = n.ConvKernel
			}
		}
		if attn == nil || kernel == 0 {
			return nil, fmt.Errorf("layer kind %s: an Inkling 2 layer has an attention node and short convolutions", id)
		}
		ar, err := attention(id, *attn, false)
		if err != nil {
			return nil, err
		}
		local := ar.NumKVHeads
		perHead := g.Global.HiddenSize / local
		width := 1 << bits.Len(uint(2*attn.HeadDim+2*perHead-1))
		add(KVLayerRun{Count: 1, Spec: "sliding_window", SlidingWindow: kernel, BlockSize: kernel, NumKVHeads: local,
			HeadSize: width, StateBytes: width * 2, ElemBytes: 2})
		add(ar)
	}
	if sp := engine.Speculative; sp != nil && sp.NumSpecTokens > 0 && g.Speculator != nil {
		for _, id := range g.Speculator.Stack.Expand() {
			for _, n := range kinds[id].Nodes {
				if n.Op == model.OpRecurrentUpdate {
					return nil, fmt.Errorf("drafter layer kind %s holds a recurrent layer, whose state is not mapped", id)
				}
				if n.Op != model.OpAttention {
					continue
				}
				r, err := attention(id, n, true)
				if err != nil {
					return nil, err
				}
				add(r)
			}
		}
	}
	return runs, nil
}
