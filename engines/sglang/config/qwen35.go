// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/arg_groups/model_overrides/qwen3_5.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).
// Derived from python/sglang/srt/arg_groups/model_overrides/qwen3_moe.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).
// Derived from python/sglang/srt/arg_groups/overrides.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).
// Derived from python/sglang/srt/arg_groups/mamba_hook.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).
// Derived from python/sglang/srt/arg_groups/model_hook.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).
// Derived from python/sglang/srt/arg_groups/memory_hook.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).

package config

import "fmt"

// The hybrid (Gated DeltaNet plus attention) Qwen3.5 family. Resolve models
// Qwen3_5MoeForConditionalGeneration; the other members of each set are listed as SGLang lists
// them so that a set is the same set, and only the architectures acceptedModels names resolve.

// mambaRadixCacheArchs is _MAMBA_RADIX_CACHE_ARCHS (overrides.py:371-407).
var mambaRadixCacheArchs = map[string]bool{
	"KimiLinearForCausalLM": true, "KimiK3ForConditionalGeneration": true, "BailingMoeV2_5ForCausalLM": true,
	"BailingMoeV3ForCausalLM": true, "BailingMoeV3VLForConditionalGeneration": true, "Qwen3NextForCausalLM": true,
	"Qwen3_5MoeForConditionalGeneration": true, "InternS2PreviewForConditionalGeneration": true,
	"InternS2MobiusForConditionalGeneration": true, "Qwen3_5ForConditionalGeneration": true,
	"Qwen3_5MoeForCausalLM": true, "Qwen3_5ForCausalLM": true, "Qwen4ExpForConditionalGeneration": true,
	"MiniCPMV4_6ForConditionalGeneration": true, "MiniCPMV4_7ForConditionalGeneration": true,
	"NemotronHForCausalLM": true, "NemotronHPuzzleForCausalLM": true, "NemotronH_Omni_Reasoning_V3": true,
	"FalconH1ForCausalLM": true, "JetNemotronForCausalLM": true, "JetVLMForConditionalGeneration": true,
	"Lfm2ForCausalLM": true, "Lfm2MoeForCausalLM": true, "ZayaForCausalLM": true,
	"Glm5NextForConditionalGeneration": true,
}

// mambaExtraBufferArchs is _MAMBA_EXTRA_BUFFER_ARCHS (overrides.py:410-433), the architectures that
// support the extra_buffer strategy without a registry spec opting in.
var mambaExtraBufferArchs = map[string]bool{
	"KimiLinearForCausalLM": true, "Qwen3_5ForConditionalGeneration": true,
	"Qwen3_5MoeForConditionalGeneration": true, "Qwen3_5MoeForCausalLM": true, "Qwen3_5ForCausalLM": true,
	"Qwen3NextForCausalLM": true, "Qwen4ExpForConditionalGeneration": true,
	"InternS2PreviewForConditionalGeneration": true, "MiniCPMV4_6ForConditionalGeneration": true,
	"MiniCPMV4_7ForConditionalGeneration": true, "BailingMoeV2_5ForCausalLM": true,
	"BailingMoeV3ForCausalLM": true, "BailingMoeV3VLForConditionalGeneration": true,
	"FalconH1ForCausalLM": true,
}

// qwen3MoeFamily is the registration of _qwen3_moe_family_overrides (qwen3_moe.py:29-37).
var qwen3MoeFamily = map[string]bool{
	"Qwen3MoeForCausalLM": true, "Qwen3VLMoeForConditionalGeneration": true, "Qwen3NextForCausalLM": true,
	"Qwen3_5MoeForConditionalGeneration": true, "InternS2PreviewForConditionalGeneration": true,
	"Qwen3_5ForConditionalGeneration": true, "Qwen4ExpForConditionalGeneration": true,
}

// qwen35Hybrid is the registration of _qwen3_5_hybrid_overrides (qwen3_5.py:15-22).
var qwen35Hybrid = map[string]bool{
	"Qwen3NextForCausalLM": true, "Qwen3_5MoeForConditionalGeneration": true,
	"InternS2PreviewForConditionalGeneration": true, "InternS2MobiusForConditionalGeneration": true,
	"Qwen3_5ForConditionalGeneration": true,
}

// qwen35Overrides runs the model overrides SGLang registers for the Qwen3.5 family, in the order
// the registry collects them (qwen3_5.py before qwen3_moe.py), then the Mamba radix-cache
// resolution with its validation (model_hook.py:730-737). The shared-expert fusion switch for the
// MoE runner they chose runs later, in handle_moe_kernel_config (moe_fusion.go).
func (r *resolver) qwen35Overrides() {
	if qwen35Hybrid[r.model.Architecture] {
		r.qwen35HybridOverrides()
	}
	if qwen3MoeFamily[r.model.Architecture] {
		r.qwen3MoeFamilyOverrides()
	}
	r.handleMambaRadixCache()
}

// qwen35HybridOverrides is _qwen3_5_hybrid_overrides (qwen3_5.py:24-52): on SM100 with no
// attention backend stated, Triton with page size 1, or TRT-LLM MHA with page size 64 when the
// default backend is TRT-LLM MHA and a Mamba radix cache is not the only reason for page size 1.
func (r *resolver) qwen35HybridOverrides() {
	const src = "_qwen3_5_hybrid_overrides"
	if !r.chip.isSM100() || r.cfg.AttentionBackend != nil {
		return
	}
	backend := "triton"
	if r.defaultAttentionBackend() == "trtllm_mha" &&
		(r.mambaExtraBuffer() || r.cfg.DisableRadixCache || r.cfg.SpeculativeAlgorithm != nil) {
		backend = "trtllm_mha"
	}
	page := 1
	if backend == "trtllm_mha" {
		page = 64
	}
	r.declare(src, "attention_backend", backend, "page_size", page)
}

// mambaExtraBuffer is mamba_extra_buffer_of (model_override_base.py:260-265).
func (r *resolver) mambaExtraBuffer() bool {
	s := r.cfg.MambaRadixCacheStrategy
	return !r.cfg.DisableRadixCache && (s == "extra_buffer" || s == "extra_buffer_lazy")
}

// qwen3MoeFamilyOverrides is _qwen3_moe_family_overrides (qwen3_moe.py:38-87) on SM100: the
// checkpoint's quantization method becomes the quantization, and the MoE runner is chosen. The
// W4A16 NVFP4 branch reads the quant_algo of the checkpoint's expert layers.
func (r *resolver) qwen3MoeFamilyOverrides() {
	const src = "_qwen3_moe_family_overrides"
	if !r.chip.isSM100() {
		return
	}
	var quantization *string
	if r.cfg.Quantization != nil {
		q := *r.cfg.Quantization
		quantization = &q
	}
	if quantization == nil && r.model.QuantMethod != nil {
		// _quantization_explicitly_unset is false: Resolve accepts no quantization field that the
		// user states as unset.
		r.declare(src, "quantization", *r.model.QuantMethod)
		q := *r.model.QuantMethod
		quantization = &q
	}
	hasW4A16 := quantization != nil && *quantization == "modelopt_mixed" && contains2(r.model.MixedPrecisionMoEQuantAlgos, "W4A16_NVFP4")
	switch {
	case hasW4A16:
		// SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16 is an environment variable Resolve does not accept
		// (the default, 0, holds), so the CuTe DSL opt-in is off.
		if r.cfg.MoeRunnerBackend != "auto" && r.cfg.MoeRunnerBackend != "marlin" {
			r.raise("model_overrides/qwen3_moe.py:_qwen3_moe_family_overrides",
				"W4A16_NVFP4 MoE layers require --moe-runner-backend=marlin, or flashinfer_cutedsl with --moe-a2a-backend=none/flashinfer and SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16=1.")
		}
		if r.cfg.MoeRunnerBackend == "auto" {
			r.declare(src, "moe_runner_backend", "marlin")
		}
	case (quantization == nil || *quantization == "fp8" || *quantization == "modelopt_fp4" || *quantization == "modelopt_mixed") &&
		r.cfg.MoeA2aBackend == "none" && r.cfg.MoeRunnerBackend == "auto":
		r.declare(src, "moe_runner_backend", "flashinfer_trtllm")
	}
}

func contains2(l []string, v string) bool {
	for _, x := range l {
		if x == v {
			return true
		}
	}
	return false
}

// supportsMambaCacheExtraBuffer is supports_mamba_cache_extra_buffer (overrides.py:~440-452) on
// CUDA: the architectures that support the strategy do so with the Triton linear-attention backend.
func (r *resolver) supportsMambaCacheExtraBuffer() bool {
	if mambaExtraBufferArchs[r.model.Architecture] {
		return r.cfg.LinearAttnBackend == "triton"
	}
	return false
}

// handleMambaRadixCache is handle_mamba_radix_cache (model_hook.py:1048-1076): the resolution pass
// and, for a model that uses the Mamba radix cache, the validation of the strategy it chose.
func (r *resolver) handleMambaRadixCache() {
	if !mambaRadixCacheArchs[r.model.Architecture] {
		return
	}
	r.mambaRadixCacheResolution()
	if !r.cfg.UsesMambaRadixCache {
		return
	}
	if r.mambaExtraBuffer() {
		r.validateMambaExtraBuffer()
	} else {
		r.validateMambaNoBuffer()
	}
}

// mambaRadixCacheResolution is _mamba_radix_cache_resolution (overrides.py:454-488).
func (r *resolver) mambaRadixCacheResolution() {
	const src = "_mamba_radix_cache_resolution"
	if r.cfg.DisableRadixCache {
		return
	}
	r.declare(src, "uses_mamba_radix_cache", true)
	if r.cfg.MambaRadixCacheStrategy == "auto" {
		wantsOverlap := !r.cfg.DisableOverlapSchedule
		wantsPaging := r.cfg.PageSize != nil && *r.cfg.PageSize > 1
		if (wantsOverlap || wantsPaging) && r.supportsMambaCacheExtraBuffer() {
			r.declare(src, "mamba_radix_cache_strategy", "extra_buffer")
		} else {
			r.declare(src, "mamba_radix_cache_strategy", "no_buffer", "disable_overlap_schedule", true)
		}
	}
}

// validateMambaExtraBuffer is validate_mamba_extra_buffer (mamba_hook.py:115-153) on CUDA, where
// the platform supports the Mamba cache snapshots and the Qwen3.5 architectures support the strategy.
func (r *resolver) validateMambaExtraBuffer() {
	const site = "mamba_hook.py:validate_mamba_extra_buffer"
	if !r.supportsMambaCacheExtraBuffer() {
		r.raise(site, fmt.Sprintf("extra_buffer is not supported for %s; use no_buffer.", r.model.Architecture))
	}
	if r.cfg.MambaRadixCacheStrategy == "extra_buffer_lazy" && r.cfg.DisaggregationMode != "null" {
		r.raise(site, "extra_buffer_lazy unsupported under PD disaggregation; use --mamba-radix-cache-strategy extra_buffer.")
	}
	if n := r.cfg.SpeculativeNumDraftTokens; n != nil && r.cfg.MambaTrackInterval < *n {
		r.raise(site, "")
	}
	if p := r.cfg.PageSize; p != nil {
		if r.cfg.MambaTrackInterval%*p != 0 {
			r.raise(site, "")
		}
		// mamba_cache_chunk_size: the larger of FLA's chunk size (64) and the page size, which
		// must divide one another.
		chunk := 64
		hi, lo := max(chunk, *p), min(chunk, *p)
		if hi%lo != 0 {
			r.raise("overrides.py:mamba_cache_chunk_size", fmt.Sprintf(
				"For SSM models, either chunk_size or page_size must be divisible by the other, got chunk_size=%d, page_size=%d", chunk, *p))
		}
	}
}

// validateMambaNoBuffer is validate_mamba_no_buffer (mamba_hook.py:156-163).
func (r *resolver) validateMambaNoBuffer() {
	const site = "mamba_hook.py:validate_mamba_no_buffer"
	if p := r.cfg.PageSize; p != nil && *p != 1 {
		r.raise(site, "no_buffer only supports page_size=1.")
	}
	if !r.cfg.DisableOverlapSchedule {
		r.raise(site, "no_buffer do not support overlap schedule. Try to set disable_overlap_schedule=True.")
	}
	if r.cfg.AttentionBackend != nil && *r.cfg.AttentionBackend == "trtllm_mha" {
		r.raise(site, "no_buffer do not support trtllm_mha attention backend.")
	}
}

// adjustMemFractionForVLM is adjust_mem_fraction_for_vlm (memory_hook.py:419-466) without
// post-capture KV sizing (SGLANG_ENABLE_POST_CAPTURE_KV_SIZING is off): the static fraction is
// scaled by the size of the vision tower, relative to ViT-L/14.
func (r *resolver) adjustMemFractionForVLM(frac float64) float64 {
	if !r.model.HasVisionConfig {
		return frac
	}
	const base = 0.95
	layers, hidden := 24, 1024
	if r.model.VisionNumHiddenLayers != nil {
		layers = *r.model.VisionNumHiddenLayers
	}
	if r.model.VisionHiddenSize != nil {
		hidden = *r.model.VisionHiddenSize
	}
	cur := float64(layers * hidden * hidden)
	baseline := float64(24 * 1024 * 1024)
	ratio := 1.0
	if baseline > 0 {
		ratio = cur / baseline
	}
	dyn := 1.0 - 0.1*(ratio-1.0)
	dyn = max(0.8, min(1.05, dyn))
	return frac * (base * dyn)
}

// isVLM is the is_vlm of _handle_gpu_memory_settings (memory_hook.py:283-288).
func (r *resolver) isVLM() bool {
	return r.model.IsMultimodal && !r.cfg.LanguageOnly && !r.cfg.LanguageModelOnly && r.cfg.DisaggregationMode != "decode"
}
