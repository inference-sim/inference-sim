// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/kv_cache_utils.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be):
// get_kv_cache_groups (:2159-2253) and _get_kv_cache_bytes_per_block (:1521-1553), through core/kv.

package lower

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/units"

	"github.com/inference-sim/blis-vllm/core/kv"
)

// KVGroups is the KV groups a PoolPlan states for a model's layers: one entry for each group the
// scheduler forms, with the bytes of one block of the group, the tokens a block covers and the
// window. The page is sized here, by the engine's own spec classes, because a cache's page size is
// an engine decision; the cost model prices the figures it is handed and the
// substrate's memory-closure check compares the plan's bytes with the engine's log.
func KVGroups(layers []kv.Layer, o kv.GroupOptions, blockSize int) ([]plan.KVGroup, error) {
	pages, err := kv.GroupPages(layers, o, blockSize)
	if err != nil {
		return nil, fmt.Errorf("lower: %w", err)
	}
	out := make([]plan.KVGroup, len(pages))
	for i, p := range pages {
		out[i] = plan.KVGroup{BytesPerPage: units.Bytes(p.BytesPerBlock), TokensPerPage: units.Tokens(p.BlockTokens),
			Window: units.Tokens(p.Window)}
	}
	return out, nil
}
