// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/kv_cache_utils.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be):
// get_kv_cache_groups (:2159-2253) and _get_kv_cache_groups_uniform_page_size (:1362-1502),
// with UniformTypeKVCacheSpecs.is_uniform_type (vllm/v1/kv_cache_interface.py:1282-1295), and
// the DeepSeek V4 specs: MLAAttentionSpec with tokens_per_state (:686-760),
// SlidingWindowMLASpec (:949-1033) and CircularBufferSpec (:909-946).

package kv

import (
	"errors"
	"fmt"
	"slices"
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
	// LayerMLA is a compressed MLAAttentionSpec: one latent per TokensPerState tokens, as
	// DeepSeek V4's attention and indexer key caches build it
	// (vllm/models/deepseek_v4/attention.py:856-925, vllm/models/deepseek_v41/attention.py:
	// 1138-1240@5df63a72). A FullAttentionSpec subclass, so full attention's uniform type.
	LayerMLA
	// LayerSlidingWindowMLA is SlidingWindowMLASpec: DeepSeek V4's sliding-window KV
	// (DeepseekV4SWACache, vllm/v1/attention/backends/mla/sparse_swa.py:116-140) and its
	// compressor state (vllm/models/deepseek_v4/compressor.py:179-193@5df63a72). Its own base
	// spec: a sliding-window manager, a type apart from SlidingWindowSpec.
	LayerSlidingWindowMLA
	// LayerCircularBuffer is CircularBufferSpec: one ring block per request, never prefix
	// cached, as DeepSeek V4.1's compressor state builds it
	// (vllm/models/deepseek_v41/compressor.py:136-167@5df63a72).
	LayerCircularBuffer
)

// Layer is one attention layer's KV cache spec, at the engine's one block size.
type Layer struct {
	Spec LayerSpec
	// Window is SlidingWindowSpec.sliding_window, in tokens.
	Window int
	// KVHeads and HeadSize set the layer's page size with its spec.
	KVHeads, HeadSize int
	// BlockSize is the spec's block_size, in tokens; 0 is the engine's block size. DeepSeek
	// V4's sliding-window and compressor caches state their own.
	BlockSize int
	// TokensPerState is tokens_per_state, the tokens one stored state covers; 0 is 1.
	TokensPerState int
	// StateBytes is state_content_bytes, a packed state's bytes; 0 is HeadSize elements of
	// ElemBytes, keys and values for attention, one latent for MLA, an indexer and a ring.
	StateBytes int
	// ElemBytes is the dtype's size; 0 is 1 (the uint8 caches). Only the page byte size of
	// layers with unequal pages reads it.
	ElemBytes int
	// Alignment is the spec's alignment, which pads its page (_apply_alignment_padding,
	// vllm/v1/kv_cache_interface.py:670-676@5df63a72); 0 is none.
	Alignment int
	// BoundedReplay is SlidingWindowMLASpec.bounded_replay: DeepSeek V4.1's sliding window
	// kept out of prefix caching (vllm/v1/kv_cache_interface.py:1003-1009@5df63a72).
	BoundedReplay bool
	// StrideAlignment is the spec's get_block_stride_alignment(), the bytes a stride between
	// physical blocks must be a multiple of (vllm/v1/kv_cache_interface.py:183-193@5df63a72),
	// which sizes the KV pool (PoolBytesPerBlock); 0 is 1.
	StrideAlignment int
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
// full attention when full attention is present, which leaves one group. Layers whose page
// bytes differ, DeepSeek V4's, are packed into groups of several block sizes, one block of each
// sharing a physical block (_get_packed_kv_cache_groups, :1860-2014, packedGroups); other
// layers whose pages differ are refused, as vLLM unifies their pages another way
// (unify_kv_cache_spec_page_size, :2213-2221), which is not modelled.
func GroupsOfLayers(layers []Layer, disableHybrid bool) ([]Group, error) {
	return GroupLayers(layers, GroupOptions{DisableHybrid: disableHybrid})
}

// GroupOptions are what get_kv_cache_groups reads besides the layers.
type GroupOptions struct {
	// DisableHybrid is disable_hybrid_kv_cache_manager.
	DisableHybrid bool
	// TrailingDrafter is _uses_trailing_mtp_layers with EAGLE block drop on
	// (kv_cache_utils.py:2017-2033@5df63a72): an MTP drafter, or any EAGLE-family drafter of
	// DeepSeek V4/V4.1, whose KV layers register after the target's. The group holding the
	// model's last layer is then the drafter's (_annotate_eagle_groups, :2051-2109), and only
	// it drops the EAGLE block.
	TrailingDrafter bool
}

// GroupLayers is GroupsOfLayers with the drafter's position.
func GroupLayers(layers []Layer, o GroupOptions) ([]Group, error) {
	groups, _, err := groupLayers(layers, o)
	return groups, err
}

// groupLayers is GroupLayers with each group's layers, by index: vLLM's KVCacheGroupSpec
// layer_names, which get_kv_cache_config_from_groups sizes the pool from.
func groupLayers(layers []Layer, o GroupOptions) ([]Group, [][]int, error) {
	disableHybrid := o.DisableHybrid
	if len(layers) == 0 {
		return nil, nil, errors.New("kv: a model with no KV cache layers")
	}
	all := make([]int, len(layers))
	for i := range all {
		all[i] = i
	}
	hasFull := false
	for _, l := range layers {
		if l.Spec != LayerSlidingWindow {
			hasFull = true
		}
	}
	if disableHybrid && hasFull && !deepseek(layers) {
		return []Group{{Kind: FullAttention}}, [][]int{all}, nil
	}
	if uniformType(layers) {
		return []Group{groupOf(layers[0])}, [][]int{all}, nil
	}
	if deepseek(layers) {
		if disableHybrid {
			return nil, nil, errors.New("kv: disable_hybrid_kv_cache_manager with DeepSeek V4's sliding-window MLA and " +
				"circular-buffer caches is not modelled")
		}
		if g, members := packedGroups(layers); g != nil {
			if o.TrailingDrafter {
				// _groups_partition_layers_exactly holds: every layer is in one group.
				for i, m := range members {
					if slices.Contains(m, len(layers)-1) {
						g[i].Eagle = true
						break
					}
				}
			}
			return g, members, nil
		}
		return nil, nil, errors.New("kv: DeepSeek V4's KV layers of one page size are not modelled")
	}
	for _, l := range layers {
		if page(l) != page(layers[0]) {
			return nil, nil, fmt.Errorf("kv: KV layers of unequal page sizes (%d and %d elements per token) "+
				"need vLLM's packed KV cache groups of several block sizes, which are not modelled",
				page(layers[0]), page(l))
		}
	}
	var buckets []Group
	of := map[Group][]int{}
	for i, l := range layers {
		g := groupOf(l)
		if len(of[g]) == 0 {
			buckets = append(buckets, g)
		}
		of[g] = append(of[g], i)
	}
	fewest, most := len(layers), 0
	for _, g := range buckets {
		fewest, most = min(fewest, len(of[g])), max(most, len(of[g]))
	}
	size := fewest
	if 2*most < 3*fewest {
		size = most
	}
	var out []Group
	var members [][]int
	for _, g := range buckets {
		// The i-th of a bucket's n groups takes its layers [i::n] (:1495-1501).
		in := of[g]
		n := (len(in) + size - 1) / size
		for i := 0; i < n; i++ {
			out = append(out, g)
			var m []int
			for j := i; j < len(in); j += n {
				m = append(m, in[j])
			}
			members = append(members, m)
		}
	}
	return out, members, nil
}

// deepseek is whether the layers have DeepSeek V4's spec kinds, which vLLM groups by page
// bytes and block size (packedGroups) rather than by elements per token.
func deepseek(layers []Layer) bool {
	for _, l := range layers {
		switch l.Spec {
		case LayerMLA, LayerSlidingWindowMLA, LayerCircularBuffer:
			return true
		}
		if l.BlockSize != 0 {
			return true
		}
	}
	return false
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

// groupOf is the group a layer's spec type makes: an indexer and a compressed MLA cache are
// full-attention layers.
func groupOf(l Layer) Group { return typeOf(l).group() }

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
	"full":               LayerFull,
	"sliding_window":     LayerSlidingWindow,
	"indexer":            LayerIndexer,
	"mla":                LayerMLA,
	"sliding_window_mla": LayerSlidingWindowMLA,
	"circular_buffer":    LayerCircularBuffer,
}

// ParseLayerSpec is the LayerSpec a kv_layers spec name names.
func ParseLayerSpec(name string) (LayerSpec, error) {
	if s, ok := layerSpecNames[name]; ok {
		return s, nil
	}
	return 0, fmt.Errorf("kv: unknown KV layer spec %q", name)
}
