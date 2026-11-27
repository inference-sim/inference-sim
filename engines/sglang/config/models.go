// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/arg_groups/model_hook.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/arg_groups/model_overrides/deepseek_v4.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/arg_groups/deepseek_v4_hook.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/arg_groups/overrides.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package config

// deepseekFamily is _DEEPSEEK_FAMILY_ARCHS (overrides.py:745-761), the architectures
// _deepseek_family_overrides is registered for (model_overrides/deepseek_v2.py:37-50).
var deepseekFamily = map[string]bool{
	"DeepseekV3ForCausalLM": true, "DeepseekV32ForCausalLM": true, "KimiK25ForConditionalGeneration": true,
	"MistralLarge3ForCausalLM": true, "PixtralForConditionalGeneration": true, "GlmMoeDsaForCausalLM": true,
	"Glm5NextForConditionalGeneration": true, "HYV4ForCausalLM": true, "HYV4ForCausalLMNextN": true,
	"LongcatFlashForCausalLM": true, "LongcatFlashForCausalLMNextN": true, "Dots3NoteForCausalLM": true,
}

// flashinferAllreduceFusionArchs is _FLASHINFER_ALLREDUCE_FUSION_ARCHS (overrides.py:929-951).
var flashinferAllreduceFusionArchs = map[string]bool{
	"DeepseekV3ForCausalLM": true, "DeepseekV32ForCausalLM": true, "DeepseekV4ForCausalLM": true,
	"GptOssForCausalLM": true, "GlmMoeDsaForCausalLM": true, "Glm4MoeForCausalLM": true,
	"Glm4MoeLiteForCausalLM": true, "MistralLarge3ForCausalLM": true, "Qwen3MoeForCausalLM": true,
	"Qwen3VLMoeForConditionalGeneration": true, "Qwen3NextForCausalLM": true,
	"Qwen4ExpForConditionalGeneration": true, "KimiK25ForConditionalGeneration": true,
	"Qwen3_5MoeForConditionalGeneration": true, "InternS2PreviewForConditionalGeneration": true,
	"Qwen3_5ForConditionalGeneration": true, "NemotronHForCausalLM": true,
	"NemotronHPuzzleForCausalLM": true, "NemotronH_Omni_Reasoning_V3": true,
}

func (r *resolver) isDeepseekV4() bool { return r.model.Architecture == "DeepseekV4ForCausalLM" }

// handleModelSpecificAdjustments is handle_model_specific_adjustments (model_hook.py:140-748)
// for the accepted architectures: the model-override registry's declarations, then the DeepSeek
// family's and DeepSeek-V4's own steps, then the FlashInfer all-reduce fusion default.
func (r *resolver) handleModelSpecificAdjustments() {
	// uses_mamba_radix_cache=False is declared first; it is the field's default.
	switch {
	case deepseekFamily[r.model.Architecture]:
		r.deepseekFamilyOverrides()
	case r.isDeepseekV4():
		r.deepseekV4Overrides()
		r.applyDeepseekV4Defaults()
	case qwen35Hybrid[r.model.Architecture] || qwen3MoeFamily[r.model.Architecture]:
		r.qwen35Overrides()
	}
	if deepseekFamily[r.model.Architecture] && r.model.IsDeepseekDSA {
		// SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD is an environment default, not a field.
		r.dsaKVCacheDtypeDefault()
		r.dsaSplitBackendResolution()
	}
	if r.isDeepseekV4() {
		r.validateDeepseekV41Features()
	}
	if deepseekFamily[r.model.Architecture] {
		r.deepseekMoeQuantResolution()
	}
	r.flashinferAllreduceFusionAutoEnable()
}

// deepseekFamilyOverrides is _deepseek_family_overrides (model_overrides/deepseek_v2.py:51-221) on
// SM90 CUDA (the family on SM100 is not accepted).
func (r *resolver) deepseekFamilyOverrides() {
	const src = "_deepseek_family_overrides"
	if !r.model.IsDeepseekDSA {
		// DeepSeek V3/R1/V3.1: trtllm_mla is SM100's (model_overrides/deepseek_v2.py:181-186);
		// prefill context parallelism is not accepted.
		if r.chip.isSM100() && r.cfg.AttentionBackend == nil && r.cfg.PrefillAttentionBackend == nil &&
			r.cfg.DecodeAttentionBackend == nil {
			r.declare(src, "attention_backend", "trtllm_mla")
		}
		return
	}
	if r.cfg.AttentionBackend == nil && r.cfg.PrefillAttentionBackend == nil && r.cfg.DecodeAttentionBackend == nil {
		r.declare(src, "attention_backend", "dsa")
	}
	// A logical page must own a whole pooled index group (deepseek_v2.py:162-172).
	kpool := 1
	if r.model.DSAIndexKpool != nil {
		kpool = *r.model.DSAIndexKpool
	}
	r.declare(src, "page_size", 64*kpool)
}

// dsaKVCacheDtypeDefault is _dsa_kv_cache_dtype_default (overrides.py:490-547) on CUDA.
func (r *resolver) dsaKVCacheDtypeDefault() {
	kv := r.cfg.KvCacheDtype
	sinks := r.model.LearnableSink
	if sinks && kv != "auto" && kv != "bf16" && kv != "bfloat16" {
		r.raise("overrides.py:_dsa_kv_cache_dtype_default",
			"Learnable DSA attention sinks require a bfloat16 KV cache; got kv_cache_dtype="+kv+".")
	}
	if kv == "auto" {
		kv = "bfloat16"
		if r.chip.Capability[0] >= 10 && !sinks {
			kv = "fp8_e4m3"
		}
	}
	if kv == "bf16" {
		kv = "bfloat16"
	}
	if kv != "bfloat16" && kv != "fp8_e4m3" {
		r.raise("overrides.py:_dsa_kv_cache_dtype_default", "DeepSeek DSA only supports bf16/bfloat16 or fp8_e4m3 kv_cache_dtype")
	}
	if kv != r.cfg.KvCacheDtype {
		r.declare("_dsa_kv_cache_dtype_default", "kv_cache_dtype", kv)
	}
}

// dsaSplitBackendResolution is _dsa_split_backend_resolution (overrides.py:600-741) on CUDA, with
// HiSparse off (not accepted).
func (r *resolver) dsaSplitBackendResolution() {
	const src = "_dsa_split_backend_resolution"
	major := r.chip.Capability[0]
	kv := r.cfg.KvCacheDtype
	userPrefill, userDecode := r.cfg.DsaPrefillBackend != nil, r.cfg.DsaDecodeBackend != nil
	set := func(prefill, decode string) {
		if !userPrefill {
			r.declare(src, "dsa_prefill_backend", prefill)
		}
		if !userDecode {
			r.declare(src, "dsa_decode_backend", decode)
		}
	}
	switch {
	case r.model.LearnableSink:
		set("flashmla_sparse", "flashmla_sparse")
		return
	case r.model.Architecture == "GlmMoeDsaForCausalLM" && major == 12 && kv == "fp8_e4m3":
		set("flashinfer_sparse_mla", "flashinfer_sparse_mla")
		return
	case kv == "fp8_e4m3":
		// Blackwell FP8 defaults to trtllm; Hopper FP8 to flashmla_kv.
		d := "flashmla_kv"
		if major >= 10 {
			d = "trtllm"
		}
		set(d, d)
	default:
		d := "fa3"
		if major >= 10 {
			d = "trtllm"
		}
		set("flashmla_sparse", d)
	}
	// _check_dsa_backend_constraints (overrides.py:550-583): the ROCm-only triton backend and
	// CUDA TileLang FP8 are not among the defaults chosen above.
}

// flashinferAllreduceFusionAutoEnable is _flashinfer_allreduce_fusion_auto_enable
// (overrides.py:954-1002), and _enforce_disable_allreduce_fusion after it (its switch is off).
func (r *resolver) flashinferAllreduceFusionAutoEnable() {
	arch := r.model.Architecture
	hidden := 0
	if r.model.HiddenSize != nil {
		hidden = *r.model.HiddenSize
	}
	preferCustomDSV41 := r.model.ModelType == "deepseek_v41" && hidden == 5120 && r.chip.Capability[0] >= 10 &&
		r.cfg.TpSize == 4 && r.cfg.Nnodes == 1 && !r.cfg.DisableCustomAllReduce
	if r.cfg.FlashinferAllreduceFusionBackend == nil && flashinferAllreduceFusionArchs[arch] && !preferCustomDSV41 &&
		(r.chip.isSM90() || r.chip.isSM100()) && r.cfg.TpSize > 1 && !r.attnDPEnabled() &&
		(r.cfg.Nnodes == 1 || r.chip.isSM100()) && r.cfg.MoeA2aBackend == "none" {
		r.declare("_flashinfer_allreduce_fusion_auto_enable", "flashinfer_allreduce_fusion_backend", "auto")
	}
}

// deepseekV4Overrides is _deepseek_v4_overrides (model_overrides/deepseek_v4.py:18-89) on CUDA.
func (r *resolver) deepseekV4Overrides() {
	const src = "_deepseek_v4_overrides"
	r.declare(src, "attention_backend", "dsv4")
	// MXFP8 serves DeepSeek-V4.1's 32-wide ue8m0 blocks on SM100 (deepseek_v4.py:25-40);
	// flashinfer is installed in SGLang's CUDA image.
	if r.model.ModelType == "deepseek_v41" && r.chip.isSM100() && r.cfg.Fp8GemmRunnerBackend == "auto" &&
		is(r.model.QuantMethod, "fp8") && equalJSON(r.model.QuantWeightBlockSize, []int{32, 32}) &&
		is(r.model.QuantScaleFmt, "ue8m0") {
		r.declare(src, "fp8_gemm_runner_backend", "flashinfer_cutedsl")
	}
	if r.cfg.SwaFullTokensRatio == nil && r.model.ModelType != "deepseek_v41" {
		r.declare(src, "swa_full_tokens_ratio", 0.1)
	}
	r.declare(src, "page_size", 256)
	if r.cfg.MoeRunnerBackend == "auto" {
		switch {
		case r.model.HasNVFP4MoEMeta:
			r.declare(src, "moe_runner_backend", "flashinfer_trtllm_routed")
		case r.cfg.MoeA2aBackend == "none" && r.model.IsFP4Experts && (r.chip.isSM90() || r.chip.isSM100()):
			// SGLANG_DSV4_FP4_DEQUANT is off.
			r.declare(src, "moe_runner_backend", "flashinfer_mxfp4")
		}
	}
}

// applyDeepseekV4Defaults is apply_deepseek_v4_defaults (deepseek_v4_hook.py:63-158) on CUDA,
// with _deepseek_v4_kv_cache_dtype (overrides.py:891-913) at its slot.
func (r *resolver) applyDeepseekV4Defaults() {
	const src = "apply_deepseek_v4_defaults"
	kv := r.cfg.KvCacheDtype
	if kv == "auto" {
		kv = "fp8_e4m3"
	}
	if kv != "fp8_e4m3" && kv != "bfloat16" {
		r.raise("overrides.py:_deepseek_v4_kv_cache_dtype", kv+" is not supported for DeepseekV4ForCausalLM")
	}
	if kv != r.cfg.KvCacheDtype {
		r.declare("_deepseek_v4_kv_cache_dtype", "kv_cache_dtype", kv)
	}
	r.resolveEngramConfig()
	if r.cfg.Dsv4AttnBackend == "auto" {
		r.declare(src, "dsv4_attn_backend", r.defaultDSV4AttentionBackend())
	}
	if r.cfg.MaxRunningRequests == nil {
		r.declare(src, "max_running_requests", 256)
	}
	if a := r.cfg.SpeculativeAlgorithm; a != nil {
		if *a != "EAGLE" && *a != "DSPARK" {
			r.raise("deepseek_v4_hook.py:apply_deepseek_v4_defaults",
				"Only EAGLE and DSPARK speculative algorithms are supported for DeepseekV4ForCausalLM")
		}
		if *a == "EAGLE" && (r.cfg.SpeculativeEagleTopk == nil || *r.cfg.SpeculativeEagleTopk != 1) {
			r.raise("deepseek_v4_hook.py:apply_deepseek_v4_defaults",
				"Only EAGLE speculative algorithm with topk == 1 is supported for DeepseekV4ForCausalLM")
		}
	}
}

// defaultDSV4AttentionBackend is _default_dsv4_attention_backend (deepseek_v4_hook.py:45-60), with
// SGLANG_DSV4_KV_LAYOUT unset.
func (r *resolver) defaultDSV4AttentionBackend() string {
	if r.model.ModelType != "deepseek_v41" || !r.chip.isSM100() || r.cfg.KvCacheDtype != "fp8_e4m3" ||
		(r.cfg.DisaggregationMode == "null" && r.attentionIsTensorParallel()) {
		return "flashmla"
	}
	return "flashmla_mega"
}

// attentionIsTensorParallel is _attention_is_tensor_parallel (deepseek_v4_hook.py:33-42), with
// derive_attn_tp_size's tp_size // (attn_cp_size * attn_dp_size) (runtime_context.py).
func (r *resolver) attentionIsTensorParallel() bool {
	if r.cfg.EnablePrefillCp {
		return false
	}
	return r.cfg.TpSize/(r.cfg.AttnCpSize*r.cfg.AttnDpSize) > 1
}

// validateDeepseekV41Features is validate_deepseek_v41_features (deepseek_v4_hook.py:270-432) on
// CUDA, for DeepSeek-V4.1, with the encoder SWA replay off and the unified KV layout unset.
func (r *resolver) validateDeepseekV41Features() {
	const site = "deepseek_v4_hook.py:validate_deepseek_v41_features"
	if r.model.ModelType != "deepseek_v41" {
		return
	}
	unsupported := []struct {
		feature string
		on      bool
	}{
		{"speculative decoding other than DSpark", r.cfg.SpeculativeAlgorithm != nil && upper(*r.cfg.SpeculativeAlgorithm) != "DSPARK"},
		{"HiSparse", r.cfg.EnableHisparse},
		{"two-batch overlap", r.cfg.EnableTwoBatchOverlap},
		{"pipeline parallelism", r.cfg.PpSize > 1},
	}
	for _, u := range unsupported {
		if u.on {
			r.raise(site, "DeepSeek-V4.1 does not support "+u.feature+" yet; disable it to serve this model.")
		}
	}
	// enable_decoder_swa_bounded_replay is not accepted, so it is unset and resolves here: on
	// unless context parallelism, returned hidden states, a prefill graph other than breakable,
	// or DP attention other than attention TP 1 over the none or megamoe all-to-all rules it out.
	pre := r.cfg.CudaGraphConfig.Prefill.Backend
	graphIncompatible := pre != backendDisabled && pre != backendBreakable
	dpIncompatible := r.attnDPEnabled() && (r.cfg.AttnDpSize != r.cfg.TpSize || r.cfg.AttnCpSize != 1 ||
		(r.cfg.MoeA2aBackend != "none" && r.cfg.MoeA2aBackend != "megamoe"))
	enabled := !r.cfg.EnablePrefillCp && r.cfg.AttnCpSize == 1 && r.cfg.DcpSize == 1 &&
		!r.cfg.EnableReturnHiddenStates && !graphIncompatible && !dpIncompatible
	r.declare("validate_deepseek_v41_features", "enable_decoder_swa_bounded_replay", enabled)
}

// resolveEngramConfig is _resolve_engram_config (deepseek_v4_hook.py:69-112@c4af57ff). DeepSeek-V4.1
// resolves --engram-config to an EngramConfig, whose defaults an unset input takes; the input
// itself is not an accepted field (accepted.go), so only the defaults are reachable. Every other
// model leaves it unset.
func (r *resolver) resolveEngramConfig() {
	if r.model.ModelType != "deepseek_v41" {
		return
	}
	r.declare("_resolve_engram_config", "engram_config", map[string]any{
		"group_size": 0, "group_store": "", "lookup": "auto", "placement": "gpu", "replication": "none",
	})
}
