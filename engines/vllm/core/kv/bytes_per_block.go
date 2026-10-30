// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/kv_cache_utils.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be):
// _get_layer_block_stride (:1516-1518), _get_kv_cache_bytes_per_block (:1521-1553) and the
// num_blocks of get_kv_cache_config_from_groups (:1622-1628).

package kv

import (
	"errors"
	"fmt"
)

// PageLayer is one layer's page in a KV cache group, as get_kv_cache_config_from_groups sees
// it: PageBytes is its spec's page_size_bytes (padding included) and Alignment its
// get_block_stride_alignment(), the bytes a stride between physical blocks must be a multiple
// of (1 when the spec states none).
type PageLayer struct {
	PageBytes int64
	Alignment int64
}

// PageGroup is one KV cache group's layers.
type PageGroup []PageLayer

// BytesPerBlock is _get_kv_cache_bytes_per_block (kv_cache_utils.py:1521-1553@5df63a72): the
// bytes one block of the shared pool takes, which is the largest group's. Every group's
// blocks come from one pool and alias from byte 0, so the widest group sets the stride.
//
//   - layer-outer layouts (blockOutermost false): each layer's block stride is its own page,
//     padded to its alignment (_get_layer_block_stride, :1516-1518), summed over the group;
//   - block-outer layouts: the group's pages are summed, and the largest sum is rounded up to
//     the least common multiple of every layer's alignment. (vLLM adds HiSparse hot groups'
//     pages to those alignments; no spec here is one.)
func BytesPerBlock(groups []PageGroup, blockOutermost bool) (int64, error) {
	if len(groups) == 0 {
		return 0, errors.New("kv: bytes per block of no KV cache groups")
	}
	var widest int64
	lcm := int64(1)
	for _, g := range groups {
		var sum int64
		for _, l := range g {
			if l.PageBytes <= 0 || l.Alignment <= 0 {
				return 0, fmt.Errorf("kv: a KV cache layer of %d page bytes and %d alignment", l.PageBytes, l.Alignment)
			}
			if blockOutermost {
				sum += l.PageBytes
				lcm = lcmOf(lcm, l.Alignment)
			} else {
				sum += roundUp(l.PageBytes, l.Alignment)
			}
		}
		widest = max(widest, sum)
	}
	if widest <= 0 {
		return 0, errors.New("kv: KV cache groups with no layers")
	}
	if !blockOutermost {
		return widest, nil
	}
	return roundUp(widest, lcm), nil
}

// NumBlocks is get_kv_cache_config_from_groups' block count (kv_cache_utils.py:1622-1628@5df63a72):
// the memory available for the KV cache over BytesPerBlock, floored.
func NumBlocks(availableBytes int64, groups []PageGroup, blockOutermost bool) (int64, error) {
	per, err := BytesPerBlock(groups, blockOutermost)
	if err != nil {
		return 0, err
	}
	return availableBytes / per, nil
}

// roundUp is vllm.utils.math_utils.round_up: x rounded up to a multiple of m.
func roundUp(x, m int64) int64 { return (x + m - 1) / m * m }

func lcmOf(a, b int64) int64 { return a / gcdOf(a, b) * b }

func gcdOf(a, b int64) int64 {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// PoolBytesPerBlock is the shared pool's bytes per block for a model's attention layers as
// vLLM groups them (GroupLayers, with the layers each group holds): BytesPerBlock over every
// group's page_size_bytes (pageBytes, at the engine's blockSize where a layer states none).
//
// The layout is the one the grouping implies. Packed groups of several page sizes exist only on
// a block-outermost layout (_get_packed_kv_cache_groups returns None on any other,
// kv_cache_utils.py:1875-1878@5df63a72), so DeepSeek V4's layers are sized block-outer; every
// other model's on vLLM's default for backends that state none, LBNHC, which is layer-outer
// (_DEFAULT_LAYOUT_PREFERENCE, vllm/v1/attention/backends/utils.py:195-202). A layer's stride
// alignment is its StrideAlignment, 1 when unstated: a backend's whole-kernel-block alignment
// (customize_attention_spec, vllm/v1/worker/utils.py:478-486) is 1 whenever its kernel block
// divides the block size, and otherwise a fraction of the page that divides it, so it changes no
// page of these specs.
func PoolBytesPerBlock(layers []Layer, o GroupOptions, blockSize int) (int64, error) {
	_, members, err := groupLayers(layers, o)
	if err != nil {
		return 0, err
	}
	groups := make([]PageGroup, len(members))
	for i, m := range members {
		for _, j := range m {
			l := layers[j]
			if l.BlockSize == 0 {
				l.BlockSize = blockSize
			}
			groups[i] = append(groups[i], PageLayer{PageBytes: int64(pageBytes(l)), Alignment: int64(max(l.StrideAlignment, 1))})
		}
	}
	return BytesPerBlock(groups, deepseek(layers) && !o.DisableHybrid)
}

// StagePoolBytesPerBlock is a pipeline stage's pool bytes per block: the model's KV cache
// groups formed ONCE over every layer (GroupLayers), each projected onto the layers the stage
// holds (on[i] for layers[i]), and BytesPerBlock over the projections. That is vLLM's per-worker
// sizing under pipeline parallelism: get_kv_cache_groups runs over the merged specs of every
// worker, _project_kv_cache_groups_to_worker keeps each group's layers on that worker, and the
// worker's pool is its memory over the projected groups' bytes per block
// (vllm/v1/core/kv_cache_utils.py:2486-2528, 2595-2603, 2650-2660@5df63a72). A group with
// none of the stage's layers contributes no bytes, and a stage holding no layer has none: 0.
//
// Grouping a stage's layers by themselves is not the same: 20 full and 30 sliding-window
// layers group as one full group of 20 and two window groups of 15, so a stage of 20 full
// and 5 window layers pages max(20, 3, 2) = 20 layers a block, where grouping its own layers
// would form one group of each kind and page 20 + 5.
func StagePoolBytesPerBlock(layers []Layer, on []bool, o GroupOptions, blockSize int) (int64, error) {
	if len(on) != len(layers) {
		return 0, fmt.Errorf("kv: %d stage flags for %d layers", len(on), len(layers))
	}
	_, members, err := groupLayers(layers, o)
	if err != nil {
		return 0, err
	}
	var groups []PageGroup
	for _, m := range members {
		var g PageGroup
		for _, j := range m {
			if !on[j] {
				continue
			}
			l := layers[j]
			if l.BlockSize == 0 {
				l.BlockSize = blockSize
			}
			g = append(g, PageLayer{PageBytes: int64(pageBytes(l)), Alignment: int64(max(l.StrideAlignment, 1))})
		}
		if len(g) > 0 {
			groups = append(groups, g)
		}
	}
	if len(groups) == 0 {
		return 0, nil
	}
	return BytesPerBlock(groups, deepseek(layers) && !o.DisableHybrid)
}

// StageGroupCopyBytes is, for each of the model's KV cache groups in GroupLayers' order, the bytes
// the stage's worker copies for one block of that group when it offloads it: vLLM's CPU offload
// worker copies a group's blocks one (block, layer) page at a time over the group's layers on that
// worker, each at its unpadded page (_fill_direct_ops, vllm/v1/kv_offload/cpu/gpu_worker.py:395-416;
// group_data_refs, offloading/worker.py:202-207@5df63a72), so a group of fewer layers than the
// widest moves fewer bytes a block than the pool's stride. A packed layout (DeepSeek V4's) copies
// the whole packed block for every group (offloading/worker.py:144-161), the stage's pool bytes
// per block (StagePoolBytesPerBlock).
func StageGroupCopyBytes(layers []Layer, on []bool, o GroupOptions, blockSize int) ([]int64, error) {
	if len(on) != len(layers) {
		return nil, fmt.Errorf("kv: %d stage flags for %d layers", len(on), len(layers))
	}
	_, members, err := groupLayers(layers, o)
	if err != nil {
		return nil, err
	}
	out := make([]int64, len(members))
	if deepseek(layers) && !o.DisableHybrid {
		per, err := StagePoolBytesPerBlock(layers, on, o, blockSize)
		if err != nil {
			return nil, err
		}
		for i := range out {
			out[i] = per
		}
		return out, nil
	}
	for i, m := range members {
		for _, j := range m {
			if !on[j] {
				continue
			}
			l := layers[j]
			if l.BlockSize == 0 {
				l.BlockSize = blockSize
			}
			out[i] += int64(pageBytes(l))
		}
	}
	return out, nil
}
