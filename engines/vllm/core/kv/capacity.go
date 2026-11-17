// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/kv_cache_utils.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5):
// get_max_concurrency_for_kv_cache_config (:1061-1093) and get_kv_cache_capacity (:1120-1128).

package kv

import "fmt"

// Capacity is the KV cache size an engine core logs at start-up: "GPU KV cache size: N tokens,
// Maximum concurrency for M tokens per request: Cx" (update_kv_cache_capacity,
// vllm/v1/core/kv_cache_utils.py:1136-1154@4ba787ec).
type Capacity struct {
	// Tokens is int(Concurrency x max_model_len), the figure the log prints as the cache's size.
	// It is not NumBlocks x block size: the groups of a hybrid model draw blocks from one pool
	// and a request holds blocks in each of them, so the pool's tokens are the requests it holds
	// at max_model_len, times that length.
	Tokens int64
	// Concurrency is num_blocks over the blocks one max_model_len request holds across the
	// groups.
	Concurrency float64
	// BlocksPerRequest is the blocks one request of max_model_len tokens takes from the pool,
	// summed over the groups.
	BlocksPerRequest int64
}

// KVCapacity is get_kv_cache_capacity over a pool of numBlocks blocks (kv_cache_config.num_blocks,
// the null block included): a request at limits.MaxModelLen holds cdiv(max_memory_usage_bytes,
// page_size_bytes) blocks in each group, all groups draw from the one pool, so the per-request
// total is the sum over groups (get_max_concurrency_for_kv_cache_config, :1061-1093), which is
// MaxRequestBlocks. A model without stated layers is one full-attention group of blockSize.
func KVCapacity(layers []Layer, o GroupOptions, blockSize int, limits RequestLimits, numBlocks int64) (Capacity, error) {
	if limits.MaxModelLen <= 0 {
		return Capacity{}, fmt.Errorf("kv: capacity of a pool at max_model_len %d", limits.MaxModelLen)
	}
	var per int64
	if len(layers) == 0 {
		if blockSize <= 0 {
			return Capacity{}, fmt.Errorf("kv: capacity of a pool of block size %d", blockSize)
		}
		per = int64(cdiv(limits.MaxModelLen, blockSize))
	} else {
		var err error
		if per, err = MaxRequestBlocks(layers, o, blockSize, limits); err != nil {
			return Capacity{}, err
		}
	}
	if per <= 0 {
		return Capacity{}, fmt.Errorf("kv: a max_model_len request holds %d pool blocks", per)
	}
	conc := float64(numBlocks) / float64(per)
	// int() of a Python float truncates toward zero.
	return Capacity{Tokens: int64(conc * float64(limits.MaxModelLen)), Concurrency: conc, BlocksPerRequest: per}, nil
}
