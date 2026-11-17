// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/mem_cache/kv_cache_configurator.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90):
// _handle_max_mamba_cache (:3229-3411), _calculate_mamba_ratio (:3062-3088) and resolve_max_num_reqs (:3119-3175).

package cache

import (
	"fmt"
	"math"
)

// mambaBase is MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO (kv_cache_configurator.py:227): state
// slots per running request before the extra buffer's tracking slots.
const mambaBase = 3

// MambaStrategy is the mamba radix cache strategy once resolved (arg_groups/overrides.py:455-485).
type MambaStrategy string

// The strategies --mamba-radix-cache-strategy resolves to. "auto" never reaches the pools.
const (
	MambaNoBuffer        MambaStrategy = "no_buffer"
	MambaExtraBuffer     MambaStrategy = "extra_buffer"
	MambaExtraBufferLazy MambaStrategy = "extra_buffer_lazy"
)

// MambaBudget is what _handle_max_mamba_cache reads. Pipeline parallelism, PD prefill, fused draft
// state and the Kimi-linear gate are refused by the caller. MaxRunningRequests and
// MaxMambaCacheSize nil are unset; FullMemoryRatio is the resolved --mamba-full-memory-ratio
// (0.9 when unset, schedule.py:221-226).
type MambaBudget struct {
	TotalRestGiB       float64
	State              LinearStateParams
	MaxRunningRequests *int
	MaxMambaCacheSize  *int
	DisableRadixCache  bool
	AttnDPSize         int
	FullMemoryRatio    float64
	Strategy           MambaStrategy
	DisableOverlap     bool
	SpecDraftTokens    int // 0: no speculation
	ReplaySpec         bool
	ReplayCacheLen     int
}

// MambaRatio is _calculate_mamba_ratio (kv_cache_configurator.py:3062-3088@297e54fd): the state
// slots one running request is charged. SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK is not an input.
func MambaRatio(disableRadixCache bool, s MambaStrategy, disableOverlap bool) (int, error) {
	if disableRadixCache {
		return 1, nil
	}
	extra := 0
	if s == MambaExtraBuffer || s == MambaExtraBufferLazy {
		lazy := s == MambaExtraBufferLazy
		if disableOverlap && lazy {
			return 0, fmt.Errorf("Lazy extra buffer requires overlap schedule (--disable-overlap-schedule is incompatible)")
		}
		extra = MambaTrackingSlots(true, !disableOverlap, lazy)
	}
	return mambaBase + extra, nil
}

// pyFloorDiv is Python's float // (floatobject.c float_floor_div).
func pyFloorDiv(a, b float64) float64 {
	mod := math.Mod(a, b)
	div := (a - mod) / b
	if mod != 0 && (b < 0) != (mod < 0) {
		div--
	}
	if div == 0 {
		return math.Copysign(0, a/b)
	}
	f := math.Floor(div)
	if div-f > 0.5 {
		f++
	}
	return f
}

// MambaCacheSize is _handle_max_mamba_cache (kv_cache_configurator.py:3229-3411@297e54fd): the
// number of mamba state slots per data-parallel worker and the memory left for the KV cache, in
// GiB. A speculative pool also reserves the per-draft intermediate state, which the solve
// charges against the same budget.
func MambaCacheSize(b MambaBudget) (size int, restGiB float64, err error) {
	if b.AttnDPSize < 1 || len(b.State.Layers) == 0 {
		return 0, 0, fmt.Errorf("mamba budget: attention dp %d, %d linear layers", b.AttnDPSize, len(b.State.Layers))
	}
	perReq := b.State.PerReqBytes()
	hasSpec := b.SpecDraftTokens > 0
	if hasSpec && b.MaxRunningRequests == nil {
		return 0, 0, fmt.Errorf("speculative decoding needs max_running_requests (kv_cache_configurator.py:3299)")
	}
	var ring int64
	if b.ReplaySpec {
		ring = b.State.ReplayRingBytesPerReq(b.ReplayCacheLen)
	}
	var fixed int64
	if b.ReplaySpec { // Gated DeltaNet: fixed request scratch, sized by active requests plus one
		fixed = ring * int64(max(1, *b.MaxRunningRequests/b.AttnDPSize)+1)
		ring = 0
	}
	extraPerSlot := ring
	reservesSpec := hasSpec && !b.ReplaySpec
	ratio, err := MambaRatio(b.DisableRadixCache, b.Strategy, b.DisableOverlap)
	if err != nil {
		return 0, 0, err
	}
	rest := b.TotalRestGiB
	const gib = float64(1 << 30)
	capped := func(size int) int {
		return min(max(1, *b.MaxRunningRequests/b.AttnDPSize), size/ratio)
	}
	switch {
	case b.MaxMambaCacheSize != nil:
		size = *b.MaxMambaCacheSize / b.AttnDPSize
		if reservesSpec {
			rest -= float64(perReq*int64(capped(size)+1)*int64(b.SpecDraftTokens)) / gib
		}
	case b.DisableRadixCache && b.MaxRunningRequests != nil:
		size = max(1, *b.MaxRunningRequests/b.AttnDPSize)
		if reservesSpec {
			rest -= float64(perReq*int64(size+1)*int64(b.SpecDraftTokens)) / gib
		}
	default:
		if perReq <= 0 {
			return 0, 0, fmt.Errorf("a linear-attention model with no state per request")
		}
		budgetBytes := rest * b.FullMemoryRatio / (1 + b.FullMemoryRatio) * gib
		if reservesSpec {
			d := float64(b.SpecDraftTokens)
			num := budgetBytes - float64(perReq)*(1+d) - float64(extraPerSlot)
			den := float64(perReq)*(1+d/float64(ratio)) + float64(extraPerSlot)
			size = int(pyFloorDiv(num, den))
			rest -= float64(perReq*int64(capped(size)+1)*int64(b.SpecDraftTokens)) / gib
		} else {
			perSlot := float64(perReq + extraPerSlot)
			size = int(pyFloorDiv(budgetBytes-float64(fixed)-perSlot, perSlot))
		}
	}
	if size <= 0 {
		return 0, 0, fmt.Errorf("Not enough GPU memory for hybrid (mamba/linear-attention) state cache. "+
			"Computed max_mamba_cache_size=%d (total_rest_memory=%.2f GB, mamba_cache_per_req=%.2f MB)",
			size, rest, float64(perReq)/float64(1<<20))
	}
	state := (float64(int64(size+1)*(perReq+extraPerSlot)) + float64(fixed)) / gib
	return size, rest - state, nil
}

// MaxNumReqsMamba is resolve_max_num_reqs with a mamba state cache (kv_cache_configurator.py:
// 3119-3175@297e54fd): the request slots, capped by the state slots over the per-request ratio.
func MaxNumReqsMamba(tokenCapacity, contextLen int, maxRunningRequests *int, attnDPSize, mambaSize, ratio int) (int, error) {
	n := MaxNumReqs(tokenCapacity, contextLen, maxRunningRequests, attnDPSize)
	n = min(n, mambaSize/ratio)
	if n <= 0 {
		return 0, fmt.Errorf("Hybrid (mamba/linear-attention) state cache is too small to serve any requests. "+
			"max_mamba_cache_size=%d, mamba_ratio=%d, resulting max_num_reqs=%d", mambaSize, ratio, n)
	}
	return n, nil
}
