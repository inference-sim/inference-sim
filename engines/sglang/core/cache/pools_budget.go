// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/kv_cache_configurator.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Also derived from, at the same release: python/sglang/srt/model_executor/pool_configurator.py
// (DefaultPoolConfigurator.calculate_pool_sizes and calculate_pool_sizes_from_max_tokens).

package cache

import "fmt"

// Profile is what _profile_available_bytes reads (kv_cache_configurator.py:2990-3060@297e54fd):
// the device's free memory after the weights are loaded and its free memory before, both in GiB
// as get_available_gpu_memory reports them, the static memory fraction, and the multimodal
// runtime reservation (zero for a text model). The Mamba state cache, whose size this also
// charges, is refused with its pools.
type Profile struct {
	AvailableGiB, PreModelLoadGiB float64
	MemFractionStatic             float64
	MMReservationGiB              float64
}

// KVBudgetBytes is _profile_available_bytes (kv_cache_configurator.py:2990-3060@297e54fd): the
// free memory less the runtime slack, pre-load memory times one minus the static fraction.
func KVBudgetBytes(p Profile) (int64, error) {
	slack := p.PreModelLoadGiB * (1 - p.MemFractionStatic)
	rest := p.AvailableGiB - slack - p.MMReservationGiB
	if rest <= 0 {
		minimum := 1 - p.AvailableGiB/p.PreModelLoadGiB
		return 0, fmt.Errorf("loaded weights leave no GPU memory for the KV cache under "+
			"--mem-fraction-static=%v. Raise --mem-fraction-static above %.4f (minimum viable = 1 - available/pre)",
			p.MemFractionStatic, minimum)
	}
	return int64(rest * float64(1<<30)), nil
}

// TokenCapacity is config_from_budget (kv_cache_configurator.py:3194-3229@297e54fd) for the
// default pool configurator: the budget over the per-token cell, in whole pages, capped by
// --max-total-tokens (_apply_token_constraints, :3090-3117) and floored to a page again. A pool
// with no KV per token gets --max-total-tokens, or the context length when it is unset
// (_zero_kv_max_tokens, pool_configurator.py:240-250; the unbounded case, a hybrid model whose
// attention layers sit on another pipeline stage, is refused with its pools). The fixed overhead
// of page-interleave sharding is zero: that sharding is refused. Pipeline parallelism, whose
// ranks agree on the minimum, is the caller's.
func TokenCapacity(budgetBytes, cellBytes int64, pageSize int, maxTotalTokens *int, contextLen int) (int, error) {
	if pageSize < 1 || cellBytes < 0 {
		return 0, fmt.Errorf("page size %d, cell %d bytes", pageSize, cellBytes)
	}
	// calculate_pool_sizes (pool_configurator.py:637-647@297e54fd).
	var tokens int
	switch {
	case cellBytes > 0:
		tokens = int(max(budgetBytes, 0) / cellBytes)
	case maxTotalTokens != nil && *maxTotalTokens != 0:
		tokens = *maxTotalTokens
	default:
		tokens = contextLen
	}
	tokens = tokens / pageSize * pageSize
	capped := tokens
	if maxTotalTokens != nil {
		capped = min(capped, *maxTotalTokens)
	}
	if capped != tokens {
		// calculate_pool_sizes_from_max_tokens (pool_configurator.py:649-653@297e54fd).
		capped = capped / pageSize * pageSize
	}
	return capped, nil
}

// MaxNumReqs is resolve_max_num_reqs (kv_cache_configurator.py:3119-3175@297e54fd) without a
// Mamba state cache: the request pool's rows per data-parallel worker. maxRunningRequests nil
// is unset.
func MaxNumReqs(tokenCapacity, contextLen int, maxRunningRequests *int, attnDPSize int) int {
	estimated := int(float64(tokenCapacity) / float64(contextLen) * 512)
	estimated = max(min(estimated, 4096), 2048)
	if maxRunningRequests != nil {
		perWorker := max(1, *maxRunningRequests/attnDPSize)
		return min(perWorker, tokenCapacity/2)
	}
	return min(estimated, tokenCapacity/2)
}
