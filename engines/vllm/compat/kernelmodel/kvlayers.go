// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/model_executor/models/gpt_oss.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// The KV cache spec each attention layer builds: gpt_oss.py:176 (per_layer_sliding_window),
// vllm/models/minimax_m3/nvidia/model.py:562-571 (MiniMaxM3SparseAttention.get_kv_cache_spec)
// and vllm/models/minimax_m3/common/indexer.py:145-154 (the indexer's).

package kernelmodel

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"
)

// KVLayerRun is a run of consecutive attention layers of one KV cache spec, in core/vllm's
// kv_layers terms: Spec is "full", "sliding_window" or "indexer", and NumKVHeads is per
// tensor-parallel rank.
type KVLayerRun struct {
	Count         int
	Spec          string
	SlidingWindow int
	NumKVHeads    int
	HeadSize      int
}

// KVLayers is the pool's attention layers by the KV cache spec vLLM v0.32.0 builds for each, in
// model order:
//
//   - a sliding-window attention node is SlidingWindowSpec (gpt-oss's even layers,
//     gpt_oss.py:176), except in a layer whose indexer selects positions from the whole
//     context: that layer keeps full-attention KV, and the window bounds only what its sparse
//     kernel reads (MiniMax-M3's sparse layers, minimax_m3/nvidia/model.py:562-571);
//   - a grouped-query node owning an index cache is the indexer's key-only spec, one head
//     (indexer.py:145-154);
//   - any other attention node is full attention.
//
// It is nil when every layer is full attention, which is one group. What it does not map is an
// error naming it: recurrent layers, window and state caches, another kind's indexer, shared KV
// beside these, and a drafter's layers beside them.
func (m *Model) KVLayers() ([]KVLayerRun, error) {
	return kvLayersOf(m.graph, m.k.Deployment())
}

func kvLayersOf(g *model.Graph, pool deployment.Pool) ([]KVLayerRun, error) {
	if g == nil {
		return nil, nil
	}
	kinds := map[string]model.LayerKind{}
	for _, k := range g.LayerKinds {
		kinds[k.ID] = k
	}
	tp := max(pool.Parallel.TP, 1)
	var runs []KVLayerRun
	plain, shared := true, false
	for _, id := range g.Stack.Expand() {
		indexed := false
		for _, n := range kinds[id].Nodes {
			for _, c := range n.Caches {
				indexed = indexed || c.Kind == model.CacheIndex && c.Owned
			}
		}
		for _, n := range kinds[id].Nodes {
			if n.Op == model.OpRecurrentUpdate {
				return nil, fmt.Errorf("layer kind %s: recurrent state is not mapped to a KV cache spec", id)
			}
			if n.Op != model.OpAttention {
				continue
			}
			r := KVLayerRun{Count: 1, Spec: "full", NumKVHeads: max(n.NumKVHeads/tp, 1), HeadSize: n.HeadDim}
			for _, c := range n.Caches {
				switch {
				case c.Kind == model.CacheKV:
					shared = shared || !c.Owned
				case c.Kind == model.CacheIndex && c.Owned && n.AttentionKind == model.AttentionGQA:
					r.Spec, r.NumKVHeads = "indexer", 1
				default:
					return nil, fmt.Errorf("layer kind %s: its %s cache is not mapped to a KV cache spec", id, c.Kind)
				}
			}
			if n.AttentionKind == model.AttentionSWA && !indexed {
				r.Spec, r.SlidingWindow = "sliding_window", n.Window
			}
			plain = plain && r.Spec == "full"
			if l := len(runs) - 1; l >= 0 && runs[l].Spec == r.Spec && runs[l].SlidingWindow == r.SlidingWindow &&
				runs[l].NumKVHeads == r.NumKVHeads && runs[l].HeadSize == r.HeadSize {
				runs[l].Count++
			} else {
				runs = append(runs, r)
			}
		}
	}
	switch sp := pool.Engine.Speculative; {
	case plain:
		return nil, nil
	case shared:
		return nil, fmt.Errorf("a layer reading another layer's KV beside these KV cache specs is not mapped")
	case sp != nil && sp.NumSpecTokens > 0:
		return nil, fmt.Errorf("a %s drafter's KV layers beside these KV cache specs are not mapped", sp.Method)
	}
	return runs, nil
}
