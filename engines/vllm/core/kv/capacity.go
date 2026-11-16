// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/kv_cache_utils.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5):
// get_max_concurrency_for_kv_cache_config (:1061-1093), get_kv_cache_capacity (:1120-1128) and
// update_kv_cache_capacity (:1136-1155); and vllm/v1/core/kv_cache_planning.py: the padding layers
// _get_kv_cache_groups_uniform_page_size warns of (:534-541).

package kv

import "fmt"

// Capacity is the KV pool as vLLM reports it at start-up: "GPU KV cache size: Tokens tokens,
// Maximum concurrency for max_model_len tokens per request: Concurrency x"
// (update_kv_cache_capacity, kv_cache_utils.py:1136-1155@4ba787ec).
//
// Tokens is not the pool's blocks times the block size. A request of max_model_len tokens takes
// blocks from every KV cache group at once, each group as many as its spec holds for one request
// (RequestBlocks), and every group draws them from the one pool. On a hybrid model each
// sliding-window group holds a window's blocks for every request beside the full-attention
// group's, so the pool holds fewer max_model_len requests than blocks x block_size / max_model_len,
// and a drafter whose layers add a group lowers the figure again.
type Capacity struct {
	// Tokens is kv_cache_size_tokens, int(Concurrency x max_model_len).
	Tokens int64
	// Concurrency is the pool's blocks over RequestBlocks (get_max_concurrency_for_kv_cache_config).
	Concurrency float64
	// NumBlocks is the pool's num_blocks, the null block included, as KVCacheConfig counts it.
	NumBlocks int64
	// RequestBlocks is the blocks one request of max_model_len tokens takes over every group
	// (MaxRequestBlocks).
	RequestBlocks int64
	// Groups is the number of KV cache groups.
	Groups int
}

// StartupCapacity is vLLM's start-up KV pool figure for a pool of numBlocks blocks (the null
// block included) over the layers' KV cache groups at limits: get_kv_cache_capacity
// (kv_cache_utils.py:1120-1128@4ba787ec) over get_max_concurrency_for_kv_cache_config (:1061-1093).
// A pool on the host (HiSparse) is not modelled.
func StartupCapacity(layers []Layer, o GroupOptions, blockSize int, limits RequestLimits, numBlocks int64) (Capacity, error) {
	groups, err := GroupLayers(layers, o.at(blockSize))
	if err != nil {
		return Capacity{}, err
	}
	return GroupsCapacity(groups, blockSize, limits, numBlocks)
}

// GroupsCapacity is StartupCapacity over groups already formed: a request's blocks are the sum
// over the groups of each one's pages for one request (the spec's max_memory_usage_bytes over
// its page_size_bytes), and the pool holds numBlocks of them.
func GroupsCapacity(groups []Group, blockSize int, limits RequestLimits, numBlocks int64) (Capacity, error) {
	if len(groups) == 0 {
		groups = []Group{{Kind: FullAttention}}
	}
	if numBlocks < 1 {
		return Capacity{}, fmt.Errorf("kv: a pool of %d blocks", numBlocks)
	}
	if limits.MaxModelLen < 1 {
		return Capacity{}, fmt.Errorf("kv: max_model_len %d", limits.MaxModelLen)
	}
	var per int64
	for _, g := range groups {
		bs := g.BlockSize
		if bs == 0 {
			bs = blockSize
		}
		if bs <= 0 {
			return Capacity{}, fmt.Errorf("kv: a KV cache group of block size %d", bs)
		}
		per += int64(groupRequestPages(g, bs, limits))
	}
	concurrency := float64(numBlocks) / float64(per)
	return Capacity{Tokens: int64(concurrency * float64(limits.MaxModelLen)), Concurrency: concurrency,
		NumBlocks: numBlocks, RequestBlocks: per, Groups: len(groups)}, nil
}

// StartupCapacity is the manager's pool as vLLM reports it at start-up (GroupsCapacity): its
// pool, its groups, max_model_len, max_in_flight_tokens and the sliding windows'
// extra_retained_tokens, as the manager was built with them.
func (m *Manager) StartupCapacity() (Capacity, error) {
	groups := make([]Group, len(m.groups))
	for i, g := range m.groups {
		groups[i] = g.spec
		groups[i].BlockSize = g.blockSize
	}
	limits := RequestLimits{MaxModelLen: m.maxModelLen, MaxInFlightTokens: m.maxInFlight, ExtraRetained: m.reprefillable}
	return GroupsCapacity(groups, m.blockSize, limits, int64(m.Pool.NumBlocks()))
}

// PaddingLayers is the layers vLLM adds to fill the KV cache groups of one page size, which it
// warns of at start-up ("Add N padding layers, may waste at most P% KV cache memory",
// _get_kv_cache_groups_uniform_page_size, kv_cache_planning.py:534-541@4ba787ec): every group is
// as wide as the widest, so each group short of it pages the missing layers' memory in every
// block. On packed groups of several page sizes, or a single group, it is 0.
func PaddingLayers(layers []Layer, o GroupOptions, blockSize int) (int, error) {
	o = o.at(blockSize)
	groups, members, err := groupLayers(layers, o)
	if err != nil {
		return 0, err
	}
	if len(groups) < 2 {
		return 0, nil
	}
	if o.blockOuter(layers) && !o.DisableHybrid {
		if packed, _, err := packedGroups(layers, o.BlockSize); err == nil && packed != nil {
			return 0, nil
		}
	}
	widest := 0
	for _, m := range members {
		widest = max(widest, len(m))
	}
	return widest*len(members) - len(layers), nil
}
