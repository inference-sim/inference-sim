// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/kv_cache_utils.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5):
// update_kv_cache_capacity (:1136-1155), which the engine core runs once its KV cache is sized.

package vllm

import "github.com/inference-sim/blis-vllm/core/kv"

// KVCacheSize is the engine's KV pool as vLLM logs it at start-up, "GPU KV cache size: N tokens,
// Maximum concurrency for max_model_len tokens per request: Cx" (kv.Capacity): the pool's blocks
// over the blocks one max_model_len request takes in every KV cache group, times max_model_len.
//
// It is the figure to set beside the engine's start-up log. Snapshot's KVBlocksTotal times
// KVBlockSize is not: on a hybrid model each request also holds a window's blocks in every
// sliding-window group, so the pool holds fewer tokens of context than its blocks times the
// block size, and a drafter whose layers add a group lowers the figure again.
func (e *Engine) KVCacheSize() (kv.Capacity, error) { return e.s.KV.StartupCapacity() }

// KVCacheSize is one rank's start-up figure (Engine.KVCacheSize). Every rank of the group sizes
// the same pool from the same settings, and each logs the same figure for its own pool.
func (g *DPGroup) KVCacheSize() (kv.Capacity, error) { return g.ranks[0].e.KVCacheSize() }
