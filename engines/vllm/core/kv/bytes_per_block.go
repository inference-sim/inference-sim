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
