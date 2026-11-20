// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/models/mistral/nvidia/attention.py at vLLM v0.34.0+0fda86a35419 (0fda86a35419c9c70dbfada224c838c10fbb62ad).

package kernelmodel

import (
	"fmt"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"

	"github.com/inference-sim/inference-sim/engines/vllm/core/engineconfig"
	"github.com/inference-sim/inference-sim/engines/vllm/core/rulepack/v0_33"
)

// isMistralLarge4 is whether g is Mistral-Large-4 (a mistral_large_4 text model under
// PixtralForConditionalGeneration).
func isMistralLarge4(g *model.Graph) bool {
	return g != nil && g.Global.Architecture == v0_33.ArchPixtral
}

// mistralKVLayers is the KV cache specs Mistral-Large-4 registers, in model order
// (vllm/models/mistral/nvidia/attention.py@0fda86a3, indexer.py):
//
//   - a layer that runs its own indexer registers the indexer's key cache first, an
//     MLAAttentionSpec of one head holding a 132 B uint8 record (MistralIndexerCache, indexer.py:42-80);
//   - a layer that owns its latent KV registers an MLAAttentionSpec of one head holding the
//     record of the cache dtype that runs (656 B under fp8_ds_mla, 1152 B under auto);
//   - a layer that shares another's KV (kv_sharing_target_layer_name) and one that reuses another
//     layer's top-k indices register nothing.
//
// Both specs are full attention to the grouping with unequal pages, which vLLM's uniform-type
// spec groups together, so the runs state the record in bytes (one-byte elements).
//
// A DSpark drafter's layers register after the target's (MistralLarge4DSparkModel,
// vllm/models/mistral/mistral_large4/nvidia/dspark.py@0fda86a3): each is a dense MLA layer of the
// draft block, a SlidingWindowMLASpec of window sliding_window + num_speculative_tokens whose
// latent record is the draft's own cache dtype, "auto" (bf16) unless speculative_config states
// kv_cache_dtype (adapt_draft_model_config, v1/worker/gpu/spec_decode/dspark/config.py:79-86, because the
// target's fp8_ds_mla layout does not apply to dense draft layers). vLLM refuses fp8_ds_mla and
// nvfp4_ds_mla on a sliding-window MLA layer (mla_attention.py:524-530), and so does this mapping.
func mistralKVLayers(g *model.Graph, _ deployment.Pool, engine engineconfig.Engine, ds deepseekServing) ([]KVLayerRun, error) {
	kinds := map[string]model.LayerKind{}
	for _, k := range g.LayerKinds {
		kinds[k.ID] = k
	}
	dtype := ds.cacheDType
	var runs []KVLayerRun
	add := func(r KVLayerRun) {
		if l := len(runs) - 1; l >= 0 && sameRun(runs[l], r) {
			runs[l].Count++
			return
		}
		runs = append(runs, r)
	}
	for _, id := range g.Stack.Expand() {
		for _, n := range kinds[id].Nodes {
			if n.Op != model.OpAttention {
				continue
			}
			for _, c := range n.Caches {
				if !c.Owned {
					continue
				}
				switch c.Kind {
				case model.CacheIndex:
					add(KVLayerRun{Count: 1, Spec: "indexer", NumKVHeads: 1, HeadSize: c.EntryBytes(dtype), ElemBytes: 1})
				case model.CacheKV:
					add(KVLayerRun{Count: 1, Spec: "mla", NumKVHeads: 1, HeadSize: c.EntryBytes(dtype), ElemBytes: 1})
				default:
					return nil, fmt.Errorf("layer kind %s: its %s cache is not mapped to a KV cache spec", id, c.Kind)
				}
			}
		}
	}
	if sp := engine.Speculative; sp != nil && sp.NumSpecTokens > 0 && g.Speculator != nil {
		elem := 0
		switch d := sp.KVCacheDType; d {
		case "", "auto", "bfloat16", "float16":
			elem = 2
		case "fp8", "fp8_e4m3", "fp8_e5m2":
			elem = 1
		default:
			return nil, fmt.Errorf("speculative kv_cache_dtype %q: vLLM refuses it on the drafter's sliding-window MLA layers "+
				"(vllm/model_executor/layers/attention/mla_attention.py:524-530@0fda86a3)", d)
		}
		for _, id := range g.Speculator.Stack.Expand() {
			for _, n := range kinds[id].Nodes {
				if n.Op != model.OpAttention {
					continue
				}
				if n.AttentionKind != model.AttentionMLA || len(n.Caches) > 0 || n.Window <= 0 {
					return nil, fmt.Errorf("drafter layer kind %s: only a dense sliding-window MLA layer without stated caches is mapped", id)
				}
				add(KVLayerRun{Count: 1, Spec: "sliding_window_mla", SlidingWindow: n.Window + sp.NumSpecTokens,
					NumKVHeads: 1, HeadSize: n.HeadDim * elem, ElemBytes: 1, Draft: true})
			}
		}
	}
	return runs, nil
}
