// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/kv_cache_utils.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be):
// the start-up memory check of get_kv_cache_configs (:2627-2646), _check_enough_kv_cache_memory
// (:1003-1040), _max_memory_usage_bytes_from_groups (:2337-2366) and
// _estimate_max_model_len_from_groups (:2369-2412); with each spec's max_memory_usage_bytes
// (vllm/v1/kv_cache_interface.py: FullAttentionSpec :602-607, SlidingWindowSpec :859-896,
// CircularBufferSpec :923-926, UniformTypeKVCacheSpecs :1315-1319).

package kv

import (
	"fmt"
	"math"
)

// RequestLimits are the scheduler settings one request's KV memory is bounded by at start-up.
type RequestLimits struct {
	// MaxModelLen is model_config.max_model_len.
	MaxModelLen int
	// MaxInFlightTokens is max_in_flight_tokens (vllm/config/vllm.py:589-598@5df63a72):
	// max_concurrent_batches x max_num_batched_tokens, what a sliding window holds beyond its
	// window while those tokens settle.
	MaxInFlightTokens int
	// ExtraRetained is a sliding window's extra_retained_tokens, num_prefill_lookahead_tokens - 1
	// (get_kv_cache_configs, kv_cache_utils.py:2583-2589@5df63a72).
	ExtraRetained int
}

// MaxRequestBlocks is the pool blocks one request of limits.MaxModelLen tokens takes at most,
// over the layers' KV cache groups: _max_memory_usage_bytes_from_groups over the pool's bytes
// per block (kv_cache_utils.py:2337-2366@5df63a72). Every group claims its own blocks from the
// one pool, so a request's blocks are the sum over groups of what its spec holds, in pages:
//
//   - full attention (and MLA, and an indexer): every token, cdiv(max_model_len, block_size);
//   - a sliding window: its admission cap, cdiv(min(window - 1 + extra_retained +
//     max_in_flight_tokens, max_model_len), block_size) + 1 (max_admission_blocks_per_request);
//   - a circular buffer: its one ring block.
//
// A group of several layers is UniformTypeKVCacheSpecs, the most pages any of its layers takes
// (max_memory_usage_pages); the layers of one group share its block size and kind, so that is
// the group's own count.
func MaxRequestBlocks(layers []Layer, o GroupOptions, blockSize int, limits RequestLimits) (int64, error) {
	groups, err := GroupLayers(layers, o)
	if err != nil {
		return 0, err
	}
	var total int64
	for _, g := range groups {
		bs := g.BlockSize
		if bs == 0 {
			bs = blockSize
		}
		if bs <= 0 {
			return 0, fmt.Errorf("kv: a KV cache group of block size %d", bs)
		}
		total += int64(groupRequestPages(g, bs, limits))
	}
	return total, nil
}

// groupRequestPages is one group's pages for one request at limits (the spec's
// max_memory_usage_bytes over its page_size_bytes).
func groupRequestPages(g Group, blockSize int, limits RequestLimits) int {
	switch g.Kind {
	case CircularBuffer:
		return 1
	case SlidingWindow, SlidingWindowMLA:
		tokens := min(g.Window-1+limits.ExtraRetained+limits.MaxInFlightTokens, limits.MaxModelLen)
		return cdiv(tokens, blockSize) + 1
	default:
		return cdiv(limits.MaxModelLen, blockSize)
	}
}

// EstimateMaxModelLen is _estimate_max_model_len_from_groups (kv_cache_utils.py:2369-2412@5df63a72):
// the longest max_model_len, at most limits.MaxModelLen, whose one request fits availBlocks pool
// blocks, by vLLM's binary search; 0 when one token does not fit.
func EstimateMaxModelLen(layers []Layer, o GroupOptions, blockSize int, limits RequestLimits, availBlocks int64) (int, error) {
	fits := func(n int) (bool, error) {
		l := limits
		l.MaxModelLen = n
		need, err := MaxRequestBlocks(layers, o, blockSize, l)
		return need <= availBlocks, err
	}
	ok, err := fits(1)
	if err != nil || !ok {
		return 0, err
	}
	left, right, result := 1, limits.MaxModelLen, 1
	for left <= right {
		mid := (left + right) / 2
		ok, err := fits(mid)
		if err != nil {
			return 0, err
		}
		if ok {
			result, left = mid, mid+1
		} else {
			right = mid - 1
		}
	}
	return result, nil
}

// CheckEnoughKVCacheMemory is vLLM's start-up check that a rank's KV cache holds one request of
// max_model_len tokens (get_kv_cache_configs, kv_cache_utils.py:2627-2646@5df63a72, calling
// _check_enough_kv_cache_memory, :1003-1040): the pool less the null block BlockPool holds back
// must take MaxRequestBlocks. In whole blocks of the pool, needed = bytes_per_block x blocks <=
// available - bytes_per_block is blocks <= num_blocks - 1, where num_blocks =
// available // bytes_per_block (get_kv_cache_config_from_groups, :1622-1628).
//
// numBlocks is the rank's pool. bytesPerBlock is the pool's bytes per block (PoolBytesPerBlock)
// when the caller knows the pool's bytes, which gives vLLM's message in GiB; 0 states the
// numbers in blocks.
func CheckEnoughKVCacheMemory(layers []Layer, o GroupOptions, blockSize int, limits RequestLimits,
	numBlocks, bytesPerBlock, availableBytes int64) error {
	usable := numBlocks - 1
	if usable <= 0 {
		return fmt.Errorf("kv: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` " +
			"when initializing the engine (vllm/v1/core/kv_cache_utils.py:1009-1017@5df63a72)")
	}
	need, err := MaxRequestBlocks(layers, o, blockSize, limits)
	if err != nil {
		return err
	}
	if need <= usable {
		return nil
	}
	estimated, err := EstimateMaxModelLen(layers, o, blockSize, limits, usable)
	if err != nil {
		return err
	}
	estimatedMsg := ""
	if estimated > 0 {
		estimatedMsg = fmt.Sprintf("Based on the available memory, the estimated maximum model length is %d. ", estimated)
	}
	if bytesPerBlock > 0 {
		// format_gib (vllm/utils/mem_utils.py:30-31@5df63a72): GiB rounded to two places; the
		// available memory is the pool's less the null block.
		gib := func(b int64) float64 { return math.Round(float64(b)/(1<<30)*100) / 100 }
		return fmt.Errorf("kv: To serve at least one request with the model's max seq len (%d), (%v GiB KV cache is "+
			"needed, which is larger than the available KV cache memory (%v GiB). %sTry increasing "+
			"`gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine "+
			"(vllm/v1/core/kv_cache_utils.py:2636-2646@5df63a72)",
			limits.MaxModelLen, gib(need*bytesPerBlock), gib(availableBytes-bytesPerBlock), estimatedMsg)
	}
	return fmt.Errorf("kv: To serve at least one request with the model's max seq len (%d), %d KV cache blocks are "+
		"needed, which is more than the %d blocks the pool holds beside the null block. %sTry increasing "+
		"`gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine "+
		"(vllm/v1/core/kv_cache_utils.py:2636-2646@5df63a72)", limits.MaxModelLen, need, usable, estimatedMsg)
}
