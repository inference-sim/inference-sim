// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/kv_cache_utils.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be):
// get_kv_cache_groups (:2159-2253), which the engine core runs over the model's layer specs
// before it builds the scheduler (vllm/v1/engine/core.py:292-330), and _uses_trailing_mtp_layers
// (:1997-2013), which decides whether the drafter's group is the last layer's.

package vllm

import (
	"fmt"
	"slices"

	"github.com/inference-sim/blis-vllm/core/kv"
)

// KVLayerRun is a run of a model's consecutive attention layers that build one KV cache spec,
// as an engine block's kv_layers states them in model order: the spec ("full", "sliding_window",
// "indexer", "mla", "sliding_window_mla" or "circular_buffer", kv.ParseLayerSpec), a sliding
// window's window, and each layer's KV heads per rank and head size. DeepSeek V4's specs also
// state their own block size, the tokens one state covers, a packed state's bytes, the dtype's
// bytes, the page alignment, the block stride alignment and the window's bounded replay
// (kv.Layer). An engine block without kv_layers is one full-attention group.
type KVLayerRun struct {
	Count           int    `yaml:"count"`
	Spec            string `yaml:"spec"`
	SlidingWindow   int    `yaml:"sliding_window,omitempty"`
	NumKVHeads      int    `yaml:"num_kv_heads"`
	HeadSize        int    `yaml:"head_size"`
	BlockSize       int    `yaml:"block_size,omitempty"`
	TokensPerState  int    `yaml:"tokens_per_state,omitempty"`
	StateBytes      int    `yaml:"state_bytes,omitempty"`
	ElemBytes       int    `yaml:"elem_bytes,omitempty"`
	Alignment       int    `yaml:"alignment,omitempty"`
	StrideAlignment int    `yaml:"stride_alignment,omitempty"`
	BoundedReplay   bool   `yaml:"bounded_replay,omitempty"`
}

// KVGroups is the scheduler's KV cache groups for a model's layers, as vLLM groups them
// (kv.GroupLayers); nil for no layers. disableHybrid is disable_hybrid_kv_cache_manager, and
// trailingDrafter is _uses_trailing_mtp_layers (kv_cache_utils.py:1997-2013@5df63a72): the
// drafter's layers are the last ones, so the group holding the last layer is the drafter's.
func KVGroups(runs []KVLayerRun, disableHybrid, trailingDrafter bool) ([]kv.Group, error) {
	if len(runs) == 0 {
		return nil, nil
	}
	layers, err := KVLayers(runs)
	if err != nil {
		return nil, err
	}
	groups, err := kv.GroupLayers(layers, kv.GroupOptions{DisableHybrid: disableHybrid, TrailingDrafter: trailingDrafter})
	if err != nil {
		return nil, fmt.Errorf("vllm: kv_layers: %w", err)
	}
	return groups, nil
}

// KVPoolBlocks is the KV pool's block count over the layers' KV cache groups, as
// get_kv_cache_config_from_groups sizes it (vllm/v1/core/kv_cache_utils.py:1622-1628@5df63a72):
// availableBytes, the memory the KV cache may take, over the widest group's bytes per block
// (kv.PoolBytesPerBlock).
//
// A layer that states no bytes of its own (no state_content_bytes, no element size) takes the
// cache dtype's from the caller's memory model: kernelPerBlock is its bytes for one block of
// every layer's page, block_size x the layers' elements a token x the dtype's bytes for these
// specs. One group of every layer is then kernelPerBlock exactly, so it keeps the memory model's
// count. A kernelPerBlock that is not a whole number of bytes for each of the layers' elements
// is refused, naming both, rather than sized from a guessed dtype. Layers that state their bytes,
// DeepSeek V4's, are sized as stated.
func KVPoolBlocks(runs []KVLayerRun, disableHybrid bool, blockSize int, kernelPerBlock, availableBytes int64) (int64, error) {
	layers, err := KVLayers(runs)
	if err != nil {
		return 0, err
	}
	if !slices.ContainsFunc(layers, func(l kv.Layer) bool { return l.StateBytes > 0 || l.ElemBytes > 0 }) {
		elems := int64(0)
		for _, l := range layers {
			l.ElemBytes, l.BlockSize = 1, blockSize
			one, err := kv.PoolBytesPerBlock([]kv.Layer{l}, kv.GroupOptions{}, blockSize)
			if err != nil {
				return 0, fmt.Errorf("vllm: kv_layers: %w", err)
			}
			elems += one
		}
		if kernelPerBlock <= 0 || kernelPerBlock%elems != 0 {
			return 0, fmt.Errorf("vllm: kv_layers: the memory model's %d bytes a block are not a whole number of bytes "+
				"for each of the layers' %d elements a block", kernelPerBlock, elems)
		}
		for i := range layers {
			layers[i].ElemBytes = int(kernelPerBlock / elems)
		}
	}
	per, err := kv.PoolBytesPerBlock(layers, kv.GroupOptions{DisableHybrid: disableHybrid}, blockSize)
	if err != nil {
		return 0, fmt.Errorf("vllm: kv_layers: %w", err)
	}
	return availableBytes / per, nil
}

// KVLayers is the runs' layers, one per layer in model order.
func KVLayers(runs []KVLayerRun) ([]kv.Layer, error) {
	var layers []kv.Layer
	for _, r := range runs {
		spec, err := kv.ParseLayerSpec(r.Spec)
		if err != nil {
			return nil, fmt.Errorf("vllm: kv_layers: %w", err)
		}
		windowed := spec == kv.LayerSlidingWindow || spec == kv.LayerSlidingWindowMLA
		if r.Count < 1 || windowed && r.SlidingWindow < 1 {
			return nil, fmt.Errorf("vllm: kv_layers: a run of %d %s layers with window %d", r.Count, r.Spec, r.SlidingWindow)
		}
		for range r.Count {
			layers = append(layers, kv.Layer{Spec: spec, Window: r.SlidingWindow, KVHeads: r.NumKVHeads,
				HeadSize: r.HeadSize, BlockSize: r.BlockSize, TokensPerState: r.TokensPerState,
				StateBytes: r.StateBytes, ElemBytes: r.ElemBytes, Alignment: r.Alignment,
				StrideAlignment: r.StrideAlignment, BoundedReplay: r.BoundedReplay})
		}
	}
	return layers, nil
}

// trailingDrafter is _uses_trailing_mtp_layers (vllm/v1/core/kv_cache_utils.py:1997-2013@5df63a72):
// an EAGLE-family drafter (use_eagle, vllm/config/speculative.py:1944-1948) whose KV layers
// register after the target's, which is MTP on any model and every EAGLE-family method on
// DeepSeek V4 and V4.1. The engine block does not name the model, so a DeepSeek V4 layer spec
// stands for its model type.
func trailingDrafter(method string, runs []KVLayerRun) bool {
	switch method {
	case "mtp":
		return true
	case "eagle", "eagle3", "dflash", "dspark":
		for _, r := range runs {
			switch r.Spec {
			case "mla", "sliding_window_mla", "circular_buffer":
				return true
			}
		}
	}
	return false
}
