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
// every layer's page, block_size x the layers' elements a token x each layer's dtype bytes. A
// layer that states its element size keeps it -- MiniMax-M3's indexer key cache, whose dtype is
// attention_config.indexer_kv_dtype rather than the cache dtype (bf16 unless stated,
// vllm/models/minimax_m3/nvidia/model.py:505-509@5df63a72) -- and the others share what remains
// of kernelPerBlock. One group of every layer is then kernelPerBlock exactly, so it keeps the
// memory model's count. A remainder that is not a whole number of bytes for each of the other
// layers' elements is refused, naming both, rather than sized from a guessed dtype. Layers that
// state a packed state's bytes, DeepSeek V4's, are sized as stated.
func KVPoolBlocks(runs []KVLayerRun, disableHybrid bool, blockSize int, kernelPerBlock, availableBytes int64) (int64, error) {
	per, err := KVPoolBytesPerBlock(runs, disableHybrid, blockSize, kernelPerBlock)
	if err != nil {
		return 0, err
	}
	return availableBytes / per, nil
}

// KVPoolBytesPerBlock is the bytes one block takes in the KV pool KVPoolBlocks sizes: the widest
// KV cache group's (kv.PoolBytesPerBlock), with the layers' element sizes resolved from
// kernelPerBlock as KVPoolBlocks resolves them. It is the worker_kv_bytes_per_block an
// OffloadingConnector sizes its CPU tier from (the KV tensor's bytes over its blocks,
// vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py:136-147@5df63a72).
func KVPoolBytesPerBlock(runs []KVLayerRun, disableHybrid bool, blockSize int, kernelPerBlock int64) (int64, error) {
	layers, err := KVLayers(runs)
	if err != nil {
		return 0, err
	}
	if !slices.ContainsFunc(layers, func(l kv.Layer) bool { return l.StateBytes > 0 }) {
		if err := deriveElemBytes(layers, blockSize, kernelPerBlock); err != nil {
			return 0, err
		}
	}
	per, err := kv.PoolBytesPerBlock(layers, kv.GroupOptions{DisableHybrid: disableHybrid}, blockSize)
	if err != nil {
		return 0, fmt.Errorf("vllm: kv_layers: %w", err)
	}
	return per, nil
}

// deriveElemBytes gives every layer that states no element size the bytes kernelPerBlock leaves
// after the layers that state one, split evenly over their elements (KVPoolBlocks). With no
// layer stating one, each layer takes kernelPerBlock over every layer's elements, as before.
func deriveElemBytes(layers []kv.Layer, blockSize int, kernelPerBlock int64) error {
	var stated, elems int64
	for _, l := range layers {
		bytes := l.ElemBytes
		l.BlockSize, l.ElemBytes = blockSize, max(l.ElemBytes, 1)
		one, err := kv.PoolBytesPerBlock([]kv.Layer{l}, kv.GroupOptions{}, blockSize)
		if err != nil {
			return fmt.Errorf("vllm: kv_layers: %w", err)
		}
		if bytes > 0 {
			stated += one
		} else {
			elems += one
		}
	}
	if elems == 0 {
		return nil
	}
	rest := kernelPerBlock - stated
	if rest <= 0 || rest%elems != 0 {
		if stated > 0 {
			return fmt.Errorf("vllm: kv_layers: the memory model's %d bytes a block, less the %d the layers stating "+
				"their dtype take, are not a whole number of bytes for each of the other layers' %d elements a block",
				kernelPerBlock, stated, elems)
		}
		return fmt.Errorf("vllm: kv_layers: the memory model's %d bytes a block are not a whole number of bytes "+
			"for each of the layers' %d elements a block", kernelPerBlock, elems)
	}
	for i := range layers {
		if layers[i].ElemBytes == 0 {
			layers[i].ElemBytes = int(rest / elems)
		}
	}
	return nil
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

// checkKVLayersFit is vLLM's start-up check over a model's KV cache groups (get_kv_cache_configs,
// vllm/v1/core/kv_cache_utils.py:2627-2646@5df63a72, kv.CheckEnoughKVCacheMemory): a rank's pool
// must hold one request of max_model_len tokens in every group at once, each sliding window at
// its admission cap of window - 1 + extra_retained_tokens + max_in_flight_tokens tokens.
// inFlight is max_in_flight_tokens (max_concurrent_batches x max_num_batched_tokens,
// vllm/config/vllm.py:589-598) and prefillLookahead num_prefill_lookahead_tokens (:644-664),
// whose extra_retained_tokens is one less. A block without kv_layers is one full-attention group,
// which resolve's own check covers.
func checkKVLayersFit(c Config, blockSize, maxModelLen, inFlight, prefillLookahead int, method string) error {
	if len(c.KVLayers) == 0 {
		return nil
	}
	layers, err := KVLayers(c.KVLayers)
	if err != nil {
		return err
	}
	o := kv.GroupOptions{DisableHybrid: c.DisableHybridKVCacheManager != nil && *c.DisableHybridKVCacheManager,
		TrailingDrafter: trailingDrafter(method, c.KVLayers)}
	var perBlock int64
	if c.KVAvailableBytes > 0 {
		if _, err = kv.PoolBytesPerBlock(layers, o, blockSize); err != nil {
			return fmt.Errorf("vllm: kv_layers: %w", err)
		}
		// The pool's own bytes a block: the caller sized num_gpu_blocks from kv_available_bytes
		// (num_blocks = available // page_size, get_kv_cache_config_from_groups,
		// kv_cache_utils.py:1622-1628@5df63a72), at the cache dtype and per-rank width it prices.
		// The layer runs' element bytes are the engine's defaults, not the served cache dtype, so
		// they would put the needed and the available memory in different units.
		perBlock = c.KVAvailableBytes / int64(c.NumGPUBlocks)
	}
	limits := kv.RequestLimits{MaxModelLen: maxModelLen, MaxInFlightTokens: inFlight,
		ExtraRetained: max(0, prefillLookahead-1)}
	if err := kv.CheckEnoughKVCacheMemory(layers, o, blockSize, limits, int64(c.NumGPUBlocks), perBlock,
		c.KVAvailableBytes); err != nil {
		return fmt.Errorf("vllm: %w", err)
	}
	return nil
}
