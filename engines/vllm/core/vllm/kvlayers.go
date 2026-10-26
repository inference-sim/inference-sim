// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/kv_cache_utils.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be):
// get_kv_cache_groups (:2159-2253), which the engine core runs over the model's layer specs
// before it builds the scheduler (vllm/v1/engine/core.py:292-330).

package vllm

import (
	"fmt"

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
	groups, err := kv.GroupsOfLayers(layers, disableHybrid)
	if err != nil {
		return nil, fmt.Errorf("vllm: kv_layers: %w", err)
	}
	return groups, nil
}
