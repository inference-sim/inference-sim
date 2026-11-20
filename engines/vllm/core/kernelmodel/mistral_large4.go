// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
// The drafter's layers are not mapped: a stated drafter makes the caller refuse.
func mistralKVLayers(g *model.Graph, _ deployment.Pool, _ engineconfig.Engine, ds deepseekServing) ([]KVLayerRun, error) {
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
	return runs, nil
}
