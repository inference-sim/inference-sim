// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/arg_groups/memory_hook.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/arg_groups/cuda_graph_hook.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package config

import (
	"math"
	"sort"
	"strconv"
)

// handleCudaGraphConfig is handle_cuda_graph_config (cuda_graph_hook.py:392-407): parse the
// configuration, disable what is incompatible, align DeepEP buckets, and apply the P/D roles.
func (r *resolver) handleCudaGraphConfig() {
	r.parseCudaGraphConfig()
	r.applyCudaGraphCompatibility()
	r.applyDeepEPAdjustments()
	r.applyCudaGraphDisaggregationRoles()
}

// parseCudaGraphConfig is parse_cuda_graph_config (cuda_graph_hook.py:32-101). None of the
// per-phase flags it reads is accepted, so the configuration is the default and nothing is
// locked.
func (r *resolver) parseCudaGraphConfig() {
	r.declare("_parse_cuda_graph_config", "cuda_graph_config", defaultCudaGraphConfig())
}

// applyCudaGraphCompatibility is apply_cuda_graph_compatibility (cuda_graph_hook.py:136-167) with
// pp_size 1.
func (r *resolver) applyCudaGraphCompatibility() {
	if r.cgLocked[[2]string{"prefill", "backend"}] {
		return
	}
	if r.cfg.CudaGraphConfig.Prefill.Backend == backendBreakable {
		r.disableBreakableIfIncompatible()
	}
}

// disablePrefillCudaGraphForDeepseekTrtllmMla is disable_prefill_cuda_graph_for_deepseek_trtllm_mla
// (cuda_graph_hook.py:279-312): a captured prefill graph makes trtllm_mla fall back to FlashAttention
// for prefill, so DeepseekV3ForCausalLM on that backend captures none unless the user chose the
// prefill backend.
func (r *resolver) disablePrefillCudaGraphForDeepseekTrtllmMla() {
	if r.cgLocked[[2]string{"prefill", "backend"}] || r.cfg.CudaGraphConfig.Prefill.Backend == backendDisabled ||
		r.model.Architecture != "DeepseekV3ForCausalLM" {
		return
	}
	prefill := r.cfg.AttentionBackend
	if r.cfg.PrefillAttentionBackend != nil && *r.cfg.PrefillAttentionBackend != "" {
		prefill = r.cfg.PrefillAttentionBackend
	}
	if prefill == nil || *prefill != "trtllm_mla" {
		return
	}
	r.declare("_disable_prefill_cuda_graph_for_deepseek_trtllm_mla", "cuda_graph_config",
		r.cfg.CudaGraphConfig.withPhase("prefill", func(p *PhaseConfig) { p.Backend = backendDisabled }))
}

// disableBreakableIfIncompatible is disable_breakable_cudagraph_if_incompatible
// (cuda_graph_hook.py:170-255). Prefill context parallelism is not accepted, so
// supports_prefill_cp_bcg is false.
func (r *resolver) disableBreakableIfIncompatible() {
	rules := []bool{
		r.model.UsesKDAAttention && r.model.Architecture != "Glm5NextForConditionalGeneration",
		r.isDeepseekV4(),
		r.cfg.AttnCpSize > 1,
		r.cfg.DcpSize > 1,
		r.cfg.EnableTwoBatchOverlap,
		!contains([]any{"none", "deepep", "megamoe", "flashinfer"}, r.cfg.MoeA2aBackend),
		r.model.IsMultimodal && !r.model.MultimodalBreakableGraphSupported,
	}
	for _, hit := range rules {
		if hit {
			r.declare("_disable_breakable_cudagraph_if_incompatible", "cuda_graph_config",
				r.cfg.CudaGraphConfig.withPhase("prefill", func(p *PhaseConfig) { p.Backend = backendDisabled }))
			return
		}
	}
}

// applyDeepEPAdjustments is apply_deepep_adjustments (cuda_graph_hook.py:315-346): DeepEP
// captures breakable prefill buckets only at multiples of 8.
func (r *resolver) applyDeepEPAdjustments() {
	if r.cfg.MoeA2aBackend != "deepep" || r.cfg.CudaGraphConfig.Prefill.Backend != backendBreakable {
		return
	}
	pre := r.cfg.CudaGraphConfig.Prefill
	bs := pre.Bs
	if bs == nil {
		maxBs := 2048
		if pre.MaxBs != nil && *pre.MaxBs != 0 {
			maxBs = *pre.MaxBs
		}
		bs = generatePrefillBatchSizes(maxBs)
	}
	set := map[int]bool{}
	for _, b := range bs {
		set[(b+7)/8*8] = true
	}
	aligned := make([]int, 0, len(set))
	for b := range set {
		aligned = append(aligned, b)
	}
	sort.Ints(aligned)
	sorted := append([]int{}, bs...)
	sort.Ints(sorted)
	if !equalJSON(aligned, sorted) {
		r.declare("_apply_deepep_adjustments", "cuda_graph_config",
			r.cfg.CudaGraphConfig.withPhase("prefill", func(p *PhaseConfig) {
				p.Bs = aligned
				p.MaxBs = ptr(aligned[len(aligned)-1])
			}))
	}
}

// applyCudaGraphDisaggregationRoles is apply_cuda_graph_disaggregation_roles
// (cuda_graph_hook.py:557-576): a prefill server captures no decode graph, a decode server no
// prefill graph.
func (r *resolver) applyCudaGraphDisaggregationRoles() {
	const src = "_apply_cuda_graph_disaggregation_roles"
	switch r.cfg.DisaggregationMode {
	case "prefill":
		if !r.cgLocked[[2]string{"decode", "backend"}] {
			r.declare(src, "cuda_graph_config", r.cfg.CudaGraphConfig.withPhase("decode", func(p *PhaseConfig) { p.Backend = backendDisabled }))
		}
	case "decode":
		if !r.cgLocked[[2]string{"prefill", "backend"}] {
			r.declare(src, "cuda_graph_config", r.cfg.CudaGraphConfig.withPhase("prefill", func(p *PhaseConfig) { p.Backend = backendDisabled }))
		}
	}
}

// handleGPUMemorySettings is handle_gpu_memory_settings (memory_hook.py:47-354): the chunked
// prefill size and decode graph size from the device memory, the CUDA-graph batch sizes, and
// mem_fraction_static from what activations and graphs reserve.
func (r *resolver) handleGPUMemorySettings() {
	const src = "_handle_gpu_memory_settings"
	gpuMem := r.chip.MemoryMiB
	cg := r.cfg.CudaGraphConfig.clone()
	dec, pre := &cg.Decode, &cg.Prefill
	decodeMaxBsIsDefault := dec.MaxBs == nil

	setChunk := func(n int) {
		if r.cfg.ChunkedPrefillSize == nil {
			r.declare(src, "chunked_prefill_size", n)
		}
	}
	setMaxBs := func(n int) {
		if dec.MaxBs == nil {
			dec.MaxBs = ptr(n)
		}
	}
	tpLarge := func(small, large int) int {
		if r.cfg.TpSize < 4 {
			return small
		}
		return large
	}
	switch {
	case gpuMem < 20*1024:
		setChunk(2048)
		setMaxBs(8)
	case gpuMem < 35*1024:
		setChunk(4096)
		setMaxBs(tpLarge(48, 160))
	case gpuMem < 60*1024:
		setChunk(4096)
		setMaxBs(tpLarge(32, 160))
	case gpuMem < 90*1024:
		setChunk(8192)
		setMaxBs(tpLarge(256, 512))
	case gpuMem < 160*1024:
		setChunk(8192)
		setMaxBs(tpLarge(256, 512))
	default:
		setChunk(16384)
		setMaxBs(512)
	}
	if decodeMaxBsIsDefault && r.cfg.MaxRunningRequests != nil && dec.MaxBs != nil && *dec.MaxBs > *r.cfg.MaxRunningRequests {
		dec.MaxBs = ptr(*r.cfg.MaxRunningRequests)
	}
	// expand_multimodal_decode_graph_to_running_limit (model_overrides/qwen3_vl.py:34-62) applies
	// to Qwen3-VL alone.

	// The device is CUDA, so the decode batch sizes are the CUDA graph's.
	if dec.Bs == nil {
		dec.Bs = generateDecodeBatchSizes(r.cfg.DisableCudaGraphPadding, r.cfg.SpeculativeAlgorithm != nil, *dec.MaxBs)
	} else {
		dec.MaxBs = ptr(maxInt(dec.Bs))
	}
	if pre.MaxBs == nil {
		chunk := *r.cfg.ChunkedPrefillSize
		if r.model.useMLA() {
			chunk = min(8192, chunk)
		}
		pre.MaxBs = ptr(chunk)
		if r.cfg.MaxTotalTokens != nil {
			pre.MaxBs = ptr(min(*pre.MaxBs, *r.cfg.MaxTotalTokens))
		}
		// The Llama-2 cap reads the model path, which names no Llama-2 model among the
		// accepted architectures.
	}
	if pre.Bs == nil {
		pre.Bs = generatePrefillBatchSizes(*pre.MaxBs)
	}
	if !equalJSON(cg, r.cfg.CudaGraphConfig) {
		r.declare(src, "cuda_graph_config", cg)
	}

	if r.cfg.MemFractionStatic == nil {
		// post_capture_kv_sizing_planned is false: SGLANG_ENABLE_POST_CAPTURE_KV_SIZING is off
		// (environ.py:640). A multimodal model's fraction is scaled by its vision tower (qwen35.go).
		var activation int
		switch {
		case r.cfg.DisaggregationMode == "decode":
			running := 1
			if r.cfg.MaxRunningRequests != nil && *r.cfg.MaxRunningRequests != 0 {
				running = *r.cfg.MaxRunningRequests
			} else if cg.Decode.MaxBs != nil && *cg.Decode.MaxBs != 0 {
				running = *cg.Decode.MaxBs
			}
			draft := 1
			if r.cfg.SpeculativeNumDraftTokens != nil && *r.cfg.SpeculativeNumDraftTokens != 0 {
				draft = *r.cfg.SpeculativeNumDraftTokens
			}
			activation = max(running*draft, 2048)
		case *r.cfg.ChunkedPrefillSize > 0:
			activation = max(*r.cfg.ChunkedPrefillSize, 2048)
		default:
			activation = max(r.cfg.MaxPrefillTokens, 2048)
		}
		reserved := 512.0
		reserved += float64(activation) * 1.5
		reserved += float64(r.cfg.TpSize) * float64(r.cfg.PpSize) / 8 * 1024
		reserved += r.reserveForGraphMB()
		if gpuMem > 60*1024 {
			reserved = math.Max(reserved, 10*1024)
		}
		reserved += r.reserveForDeepEPA2AMB()
		frac := pyRound3((gpuMem - reserved) / gpuMem)
		if r.isVLM() {
			frac = r.adjustMemFractionForVLM(frac)
		}
		r.declare(src, "mem_fraction_static", frac)
	}
	// Symmetric memory (enable_symm_mem) is not accepted.
}

// reserveForGraphMB is reserve_for_graph_mb (memory_hook.py:357-399).
func (r *resolver) reserveForGraphMB() float64 {
	cg := r.cfg.CudaGraphConfig
	dec, pre := cg.Decode, cg.Prefill
	reserved := 0.0
	if r.cfg.DisaggregationMode != "prefill" && dec.Backend != backendDisabled {
		reserved += float64(*dec.MaxBs) * 2
	}
	if r.attnDPEnabled() && r.cfg.DisaggregationMode != "prefill" {
		reserved += float64(*dec.MaxBs) * float64(r.cfg.AttnDpSize) * 3
		if *dec.MaxBs > 300 {
			reserved += float64(*dec.MaxBs) * float64(r.cfg.AttnDpSize) * 1.5
		}
	}
	if r.cfg.DisaggregationMode != "decode" && pre.Backend != backendDisabled {
		if !r.model.useMLA() {
			reserved += float64(len(pre.Bs)) * 8
		} else {
			reserved += 1.5 * 1024
		}
		if pre.Backend == backendBreakable && r.cfg.MoeA2aBackend == "deepep" {
			reserved += 1 * 1024
		}
	}
	return reserved
}

// reserveForDeepEPA2AMB is reserve_for_deepep_a2a_mb (memory_hook.py:402-414).
func (r *resolver) reserveForDeepEPA2AMB() float64 {
	if r.cfg.DisaggregationMode != "prefill" && r.cfg.CudaGraphConfig.Decode.Backend != backendDisabled &&
		r.cfg.MoeA2aBackend == "deepep" {
		return 2 * 1024
	}
	return 0
}

// pyRound3 is Python's round(x, 3): the exact binary value of x rounded to three decimals, a
// tie to even. strconv rounds the exact value the same way.
func pyRound3(x float64) float64 {
	v, err := strconv.ParseFloat(strconv.FormatFloat(x, 'f', 3, 64), 64)
	if err != nil {
		panic(err)
	}
	return v
}

func maxInt(l []int) int {
	m := l[0]
	for _, v := range l[1:] {
		m = max(m, v)
	}
	return m
}
