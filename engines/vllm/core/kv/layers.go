// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/kv_cache_utils.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be):
// get_kv_cache_groups (:2159-2253) and _get_kv_cache_groups_uniform_page_size (:1362-1502),
// with UniformTypeKVCacheSpecs.is_uniform_type (vllm/v1/kv_cache_interface.py:1282-1295).

package kv

import (
	"errors"
	"fmt"
)

// LayerSpec is the KV cache spec class one attention layer builds.
type LayerSpec int

const (
	// LayerFull is FullAttentionSpec.
	LayerFull LayerSpec = iota
	// LayerSlidingWindow is SlidingWindowSpec.
	LayerSlidingWindow
	// LayerIndexer is a key-only MLAAttentionSpec with the INDEXER role, as a lightning
	// indexer builds it (vllm/models/minimax_m3/common/indexer.py:145-154@5df63a72). It is a
	// FullAttentionSpec subclass, so it shares full attention's uniform type
	// (register_all_kvcache_specs, vllm/v1/core/single_type_kv_cache_manager.py:2633-2710).
	LayerIndexer
)

// Layer is one attention layer's KV cache spec, at the engine's one block size.
type Layer struct {
	Spec LayerSpec
	// Window is SlidingWindowSpec.sliding_window, in tokens.
	Window int
	// KVHeads and HeadSize set the layer's page size with its spec.
	KVHeads, HeadSize int
}

// GroupsOfLayers is get_kv_cache_groups (kv_cache_utils.py:2159-2253@5df63a72) over a model's
// attention layers, in model order, all at one block size:
//
//   - one spec for every layer, or one uniform type (full attention and its MLA indexers, or
//     sliding windows of one size), is one group (:2185-2193);
//   - otherwise, at one page size, each spec type is a bucket in order of first appearance,
//     the group size is the fewest layers of a type (the most, when that is under 1.5 times
//     the fewest), and a bucket of n layers is ceil(n / size) groups of its type, its last one
//     padded (_get_kv_cache_groups_uniform_page_size, :1362-1502).
//
// With disableHybrid (unify_hybrid_kv_cache_specs, :1799-1822) sliding windows are promoted to
// full attention when full attention is present, which leaves one group. Layers whose pages
// differ are refused: vLLM packs them into groups of several block sizes
// (_get_packed_kv_cache_groups, :1860-2014), which this port does not model.
func GroupsOfLayers(layers []Layer, disableHybrid bool) ([]Group, error) {
	if len(layers) == 0 {
		return nil, errors.New("kv: a model with no KV cache layers")
	}
	hasFull := false
	for _, l := range layers {
		if l.Spec != LayerSlidingWindow {
			hasFull = true
		}
	}
	if disableHybrid && hasFull {
		return []Group{{Kind: FullAttention}}, nil
	}
	if uniformType(layers) {
		return []Group{groupOf(layers[0])}, nil
	}
	for _, l := range layers {
		if page(l) != page(layers[0]) {
			return nil, fmt.Errorf("kv: KV layers of unequal page sizes (%d and %d elements per token) "+
				"need vLLM's packed KV cache groups of several block sizes, which are not modelled",
				page(layers[0]), page(l))
		}
	}
	var buckets []Group
	count := map[Group]int{}
	for _, l := range layers {
		g := groupOf(l)
		if count[g] == 0 {
			buckets = append(buckets, g)
		}
		count[g]++
	}
	fewest, most := len(layers), 0
	for _, g := range buckets {
		fewest, most = min(fewest, count[g]), max(most, count[g])
	}
	size := fewest
	if 2*most < 3*fewest {
		size = most
	}
	var out []Group
	for _, g := range buckets {
		for i := 0; i < (count[g]+size-1)/size; i++ {
			out = append(out, g)
		}
	}
	return out, nil
}

// uniformType is whether every layer is one spec type: UniformTypeKVCacheSpecs.is_uniform_type
// at one block size, where full attention takes its indexers and a sliding window only windows
// of its own size (vllm/v1/kv_cache_interface.py:898-903@5df63a72).
func uniformType(layers []Layer) bool {
	for _, l := range layers {
		if groupOf(l) != groupOf(layers[0]) {
			return false
		}
	}
	return true
}

// groupOf is the group a layer's spec type makes: an indexer is a full-attention layer.
func groupOf(l Layer) Group {
	if l.Spec == LayerSlidingWindow {
		return Group{Kind: SlidingWindow, Window: l.Window}
	}
	return Group{Kind: FullAttention}
}

// page is a layer's page in elements per token: keys and values for attention, keys only for
// an indexer (MLAAttentionSpec's single latent, vllm/v1/kv_cache_interface.py:680-700@5df63a72).
func page(l Layer) int {
	if l.Spec == LayerIndexer {
		return l.KVHeads * l.HeadSize
	}
	return 2 * l.KVHeads * l.HeadSize
}

// layerSpecNames are the spec names of an engine block's kv_layers and of the oracle's
// "kv_layers" scenarios (oracle/run_scheduler.py layer_specs).
var layerSpecNames = map[string]LayerSpec{
	"full":           LayerFull,
	"sliding_window": LayerSlidingWindow,
	"indexer":        LayerIndexer,
}

// ParseLayerSpec is the LayerSpec a kv_layers spec name names.
func ParseLayerSpec(name string) (LayerSpec, error) {
	if s, ok := layerSpecNames[name]; ok {
		return s, nil
	}
	return 0, fmt.Errorf("kv: unknown KV layer spec %q", name)
}
