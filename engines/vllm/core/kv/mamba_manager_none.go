// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/single_type_kv_cache_manager.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be):
// MambaManager with mamba_cache_mode "none" (:1415-2112): get_num_blocks_to_allocate (:1676-1716),
// allocate_new_blocks (:1780-1800) and get_num_skipped_tokens (:1954-1960), over the
// SingleTypeKVCacheManager they extend (:174-251, 672-710).

package kv

import "fmt"

// mambaAligned is whether g is a Mamba group in mamba_cache_mode "align", whose block table the
// align-mode manager (mamba_manager_align.go) runs. A group of the default mode, "none", is
// managed by SingleTypeKVCacheManager's own rules, which the helpers below add to.
func (g *group) mambaAligned() bool {
	return g.spec.Kind == Mamba && mambaMode(g.spec.MambaCacheMode) == "align"
}

// noneSpeculativeBlocks is the blocks a Mamba group in mode "none" holds beyond the ones its
// tokens fill: num_speculative_blocks. get_num_blocks_to_allocate and allocate_new_blocks each add
// block_size x num_speculative_blocks to the tokens that need a slot before the base class
// counts blocks (single_type_kv_cache_manager.py:1692-1699, 1790-1795@5df63a72), which is
// exactly cdiv(tokens, block_size) + num_speculative_blocks blocks. Align mode and every other
// group add none here.
func (g *group) noneSpeculativeBlocks() int {
	if g.spec.Kind != Mamba || g.mambaAligned() {
		return 0
	}
	return g.spec.MambaSpeculativeBlocks
}

// slotBlocks is the blocks a group needs for numTokens tokens that need a slot:
// cdiv(num_tokens, block_size) (SingleTypeKVCacheManager.get_num_blocks_to_allocate,
// single_type_kv_cache_manager.py:212 and allocate_new_blocks, :487-488@5df63a72), and a
// Mamba group's speculative blocks besides.
func (g *group) slotBlocks(numTokens int) int {
	return cdiv(numTokens, g.blockSize) + g.noneSpeculativeBlocks()
}

// CheckGroups refuses the groups a manager cannot run for the cache setting: a Mamba group whose
// mamba_cache_mode is not "none" or "align", and a Mamba group in mode "none" under prefix
// caching. vLLM cannot reach the second:
// MambaModelConfig.verify_and_update_config turns "none" into "align" when prefix caching is on
// and sets "none" when it is off (vllm/model_executor/models/config.py:672-706@5df63a72). The
// mode "none" caches nothing, so this port, which models the cached-state lookup only for
// "align", refuses the pair by name.
func CheckGroups(groups []Group, enableCaching bool) error {
	for i, g := range groups {
		if g.Kind != Mamba {
			continue
		}
		switch mode := mambaMode(g.MambaCacheMode); mode {
		case "align":
		case "none":
			if enableCaching {
				return fmt.Errorf("kv: KV cache group %d is a Mamba group in mamba_cache_mode \"none\" with prefix caching on: "+
					"vLLM sets the mode to \"align\" when prefix caching is on, so this setting is unreachable "+
					"(vllm/model_executor/models/config.py:672-706@5df63a72)", i)
			}
		default:
			return fmt.Errorf("kv: KV cache group %d has mamba_cache_mode %q, which is not modelled: only \"none\" and \"align\" are", i, mode)
		}
	}
	return nil
}
