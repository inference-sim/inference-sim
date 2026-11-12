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
	// LayerMamba is MambaSpec: the fixed-size recurrent state of a Mamba-2, GDN or KDA layer
	// (vllm/v1/kv_cache_interface.py:1035-1108@5df63a72). One state is StateBytes (the sum of
	// its conv and temporal tensors, per tensor-parallel rank), padded up to PageSizePadded
	// when the hybrid block-size alignment pads it to the attention page
	// (vllm/platforms/interface.py:1118-1126). Its own base spec: a MambaManager, a type
	// apart from every attention spec.
	LayerMamba
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
	// The fields below are MambaSpec's, read only by LayerMamba. StateBytes is
	// state_content_bytes and BlockSize is mamba_block_size, in tokens: the engine's block size
	// in align mode, max_model_len in none mode (MambaModelConfig, vllm/model_executor/models/
	// config.py:672-706@5df63a72).
	//
	// PageSizePadded is page_size_padded: 0 is the state's own bytes.
	PageSizePadded int
	// MambaCacheMode is mamba_cache_mode as it runs, "none" or "align" ("all" is refused).
	MambaCacheMode string
	// NumSpeculativeBlocks is num_speculative_blocks, the draft tokens of speculative decoding
	// (vllm/model_executor/layers/mamba/abstract.py:69-85).
	NumSpeculativeBlocks int
	// NumPrefillCheckpointBlocks and PrefillCheckpointAlignment are the spec's prefill
	// checkpoint (Mamba-2: 1 and 1, mamba_mixer2.py:1107; KDA with the flashkda backend: 1 and
	// 16, kda_checkpoint.py:16). Read only in align mode.
	NumPrefillCheckpointBlocks int
	PrefillCheckpointAlignment int
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
	// BlockSize is the engine's block size, in tokens: the one a layer that states none has.
	// Layers whose pages are compared or packed in bytes (a Mamba layer beside attention, packed
	// groups) read it; 0 is unknown, and a comparison that needs it is then skipped or refused.
	// The sizing functions (PoolBytesPerBlock and the others) fill it from their blockSize.
	BlockSize int
	// BlockOutermost is cache_config.get_resolved_kv_cache_layout().is_block_outermost: the
	// layout on which _get_packed_kv_cache_groups packs layers of unequal page sizes
	// (kv_cache_utils.py:1875-1878@5df63a72). DeepSeek V4's layers imply it (see
	// PoolBytesPerBlock); a model with Mamba layers states it here.
	BlockOutermost bool
}

// at is o with the engine's block size, when it states none.
func (o GroupOptions) at(blockSize int) GroupOptions {
	if o.BlockSize == 0 {
		o.BlockSize = blockSize
	}
	return o
}

// blockOuter is whether the pool's layout is block-outermost: stated, or implied by DeepSeek V4's
// layers, whose indexer backend declares it and which are then packed unless the hybrid manager
// is disabled.
func (o GroupOptions) blockOuter(layers []Layer) bool {
	return o.BlockOutermost || deepseek(layers) && !o.DisableHybrid
}

// GroupLayers is GroupsOfLayers with the drafter's position.
func GroupLayers(layers []Layer, o GroupOptions) ([]Group, error) {
	groups, _, err := groupLayers(layers, o)
	return groups, err
}

// groupLayers is GroupLayers with each group's layers, by index: vLLM's KVCacheGroupSpec
// layer_names, which get_kv_cache_config_from_groups sizes the pool from.
func groupLayers(layers []Layer, o GroupOptions) ([]Group, [][]int, error) {
	if len(layers) == 0 {
		return nil, nil, errors.New("kv: a model with no KV cache layers")
	}
	for _, l := range layers {
		if l.Spec == LayerMamba && l.MambaCacheMode != "" && l.MambaCacheMode != "none" && l.MambaCacheMode != "align" {
			return nil, nil, fmt.Errorf("kv: mamba_cache_mode %q is not modelled: only \"none\" and \"align\" are", l.MambaCacheMode)
		}
	}
	all := make([]int, len(layers))
	for i := range all {
		all[i] = i
	}
	work := layers
	if o.DisableHybrid {
		var err error
		if work, err = unifyHybrid(layers); err != nil {
			return nil, nil, err
		}
	}
	if uniformType(work) {
		return []Group{groupOf(work[0])}, [][]int{all}, nil
	}
	if deepseek(work) && o.DisableHybrid {
		// vLLM v0.32.0 does not start this deployment: unify_hybrid_kv_cache_specs promotes each
		// SlidingWindowMLASpec to an MLAAttentionSpec beside the compressed MLA caches
		// (_promote_local_kv_cache_specs, vllm/v1/core/kv_cache_utils.py:1731-1754@5df63a72) with
		// replace_as, which passes every field of the old spec except extra_retained_tokens to
		// the new class (vllm/v1/kv_cache_interface.py:119-138), and MLAAttentionSpec has no
		// bounded_replay field (:686-702, :956). So this is vLLM's refusal, not a gap of ours.
		return nil, nil, errors.New("vLLM does not start DeepSeek V4 or V4.1 with disable_hybrid_kv_cache_manager: " +
			"true: promoting the sliding-window MLA caches to MLA attention raises TypeError (unexpected keyword " +
			"argument 'bounded_replay', unify_hybrid_kv_cache_specs, vllm/v1/core/kv_cache_utils.py:1746-1754 at " +
			"vLLM v0.32.0), so there is no deployment to price; drop disable_hybrid_kv_cache_manager from the " +
			"engine block, or run a vLLM release that starts this layout")
	}
	if o.blockOuter(work) {
		g, members, err := packedGroups(work, o.BlockSize)
		if err != nil {
			return nil, nil, err
		}
		if g != nil {
			if o.TrailingDrafter {
				// _groups_partition_layers_exactly holds: every layer is in one group.
				for i, m := range members {
					if slices.Contains(m, len(work)-1) {
						g[i].Eagle = true
						break
					}
				}
			}
			return g, members, nil
		}
	}
	if deepseek(work) {
		return nil, nil, errors.New("kv: DeepSeek V4's KV layers of one page size are not modelled")
	}
	if err := checkEqualPages(work, o.BlockSize); err != nil {
		return nil, nil, err
	}
	var buckets []Group
	of := map[Group][]int{}
	for i, l := range work {
		g := groupOf(l)
		if len(of[g]) == 0 {
			buckets = append(buckets, g)
		}
		of[g] = append(of[g], i)
	}
	fewest, most := len(work), 0
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

// unifyHybrid is unify_hybrid_kv_cache_specs (kv_cache_utils.py:1799-1822@5df63a72) with
// _promote_local_kv_cache_specs (:1684-1761): the layers are split into the replicated attention
// specs and the rest, Mamba's, and each part is promoted by itself. Attention gets its sliding
// windows promoted to full attention when full attention is present, which keeps every layer's
// page (the promoted spec has the same page bytes). A part that is neither one spec nor one
// uniform type afterwards raises "Failed to promote local KV cache specs to one unified type"
// (:1759-1760): for Mamba, layers of more than one MambaSpec, which no promotion merges. The
// DeepSeek V4 kinds are left to the caller, which refuses them under disableHybrid.
func unifyHybrid(layers []Layer) ([]Layer, error) {
	var mamba []Layer
	for _, l := range layers {
		if l.Spec == LayerMamba {
			mamba = append(mamba, l)
		}
	}
	if len(mamba) > 0 && !uniformType(mamba) {
		return nil, errors.New("kv: disable_hybrid_kv_cache_manager with Mamba layers of more than one spec: vLLM's " +
			"unify_hybrid_kv_cache_specs raises \"Failed to promote local KV cache specs to one unified type\" " +
			"(vllm/v1/core/kv_cache_utils.py:1759-1760@5df63a72)")
	}
	attn := make([]Layer, 0, len(layers))
	for _, l := range layers {
		if l.Spec != LayerMamba {
			attn = append(attn, l)
		}
	}
	hasFull := false
	for _, l := range attn {
		if l.Spec != LayerSlidingWindow {
			hasFull = true
		}
	}
	if len(attn) == 0 || !hasFull || deepseek(attn) {
		return layers, nil
	}
	out := slices.Clone(layers)
	for i, l := range out {
		if l.Spec == LayerSlidingWindow {
			out[i].Spec, out[i].Window = LayerFull, 0
		}
	}
	return out, nil
}

// checkEqualPages refuses what vLLM's uniform-page-size grouping assumes and this port does not
// unify: the attention layers are of one page in elements per token at the engine's one block
// size, and a Mamba layer's page, padded to the attention page at start-up
// (vllm/platforms/interface.py:1118-1126@5df63a72), is that page. Mamba against attention can be
// compared only where the engine's block size and every attention layer's element size are
// known; otherwise the padded page is taken as stated. Unequal pages are unify_kv_cache_spec_page_size's
// (:1292-1354), which pads a smaller Mamba page and scales an attention block: not modelled.
func checkEqualPages(layers []Layer, blockSize int) error {
	var first *Layer
	known := blockSize > 0
	attnPage := 0
	for i, l := range layers {
		if l.Spec == LayerMamba {
			continue
		}
		if first == nil {
			first = &layers[i]
		} else if page(l) != page(*first) {
			return fmt.Errorf("kv: KV layers of unequal page sizes (%d and %d elements per token) "+
				"need vLLM's packed KV cache groups of several block sizes, which are not modelled",
				page(*first), page(l))
		}
		if l.ElemBytes <= 0 {
			known = false
		}
		if l.BlockSize == 0 {
			l.BlockSize = blockSize
		}
		attnPage = pageBytes(l)
	}
	if first == nil || !known {
		return nil
	}
	for _, l := range layers {
		if l.Spec == LayerMamba && pageBytes(l) != attnPage {
			return fmt.Errorf("kv: a Mamba layer's page of %d bytes beside attention pages of %d bytes: vLLM unifies "+
				"them with unify_kv_cache_spec_page_size (vllm/v1/core/kv_cache_utils.py:1292-1354@5df63a72), "+
				"which is not modelled", pageBytes(l), attnPage)
		}
	}
	return nil
}

// deepseek is whether the layers have DeepSeek V4's spec kinds, which vLLM groups by page
// bytes and block size (packedGroups) rather than by elements per token. A Mamba layer states
// its own block size and is not one.
func deepseek(layers []Layer) bool {
	for _, l := range layers {
		switch l.Spec {
		case LayerMLA:
			// A plain MLAAttentionSpec (one latent a token) is full attention to the grouping,
			// at whatever block size it states (its page is then compared in bytes);
			// DeepSeek V4's states more.
			if l.TokensPerState > 1 || l.StateBytes > 0 || l.Alignment > 0 {
				return true
			}
			continue
		case LayerSlidingWindowMLA, LayerCircularBuffer:
			return true
		case LayerMamba:
			continue
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
	if l.Spec == LayerIndexer || l.Spec == LayerMLA {
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
	"mamba":              LayerMamba,
}

// ParseLayerSpec is the LayerSpec a kv_layers spec name names.
func ParseLayerSpec(name string) (LayerSpec, error) {
	if s, ok := layerSpecNames[name]; ok {
		return s, nil
	}
	return 0, fmt.Errorf("kv: unknown KV layer spec %q", name)
}
