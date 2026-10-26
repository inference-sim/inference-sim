// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/kv_cache_utils.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be):
// get_kv_cache_groups (:2159-2253), which the engine core runs over the model's layer specs
// before it builds the scheduler (vllm/v1/engine/core.py:292-330).

package vllm

import (
	"fmt"
	"slices"

	"github.com/inference-sim/blis-vllm/core/kv"
)

// KVLayerRun is a run of a model's consecutive attention layers that build one KV cache spec,
// as an engine block's kv_layers states them in model order: the spec ("full", "sliding_window"
// or "indexer", kv.ParseLayerSpec), a sliding window's window, and each layer's KV heads per
// rank and head size. An engine block without kv_layers is one full-attention group.
type KVLayerRun struct {
	Count         int    `yaml:"count"`
	Spec          string `yaml:"spec"`
	SlidingWindow int    `yaml:"sliding_window,omitempty"`
	NumKVHeads    int    `yaml:"num_kv_heads"`
	HeadSize      int    `yaml:"head_size"`
}

// KVGroups is the scheduler's KV cache groups for a model's layers, as vLLM groups them
// (kv.GroupsOfLayers); nil for no layers. disableHybrid is disable_hybrid_kv_cache_manager.
func KVGroups(runs []KVLayerRun, disableHybrid bool) ([]kv.Group, error) {
	if len(runs) == 0 {
		return nil, nil
	}
	layers, err := kvLayers(runs)
	if err != nil {
		return nil, err
	}
	groups, err := kv.GroupsOfLayers(layers, disableHybrid)
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
// is refused, naming both, rather than sized from a guessed dtype.
func KVPoolBlocks(runs []KVLayerRun, disableHybrid bool, blockSize int, kernelPerBlock, availableBytes int64) (int64, error) {
	layers, err := kvLayers(runs)
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

// kvLayers is the runs' layers, one per layer in model order.
func kvLayers(runs []KVLayerRun) ([]kv.Layer, error) {
	var layers []kv.Layer
	for _, r := range runs {
		spec, err := kv.ParseLayerSpec(r.Spec)
		if err != nil {
			return nil, fmt.Errorf("vllm: kv_layers: %w", err)
		}
		if r.Count < 1 || spec == kv.LayerSlidingWindow && r.SlidingWindow < 1 {
			return nil, fmt.Errorf("vllm: kv_layers: a run of %d %s layers with window %d", r.Count, r.Spec, r.SlidingWindow)
		}
		for range r.Count {
			layers = append(layers, kv.Layer{Spec: spec, Window: r.SlidingWindow, KVHeads: r.NumKVHeads, HeadSize: r.HeadSize})
		}
	}
	return layers, nil
}
