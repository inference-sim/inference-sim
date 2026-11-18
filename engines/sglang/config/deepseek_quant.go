// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/arg_groups/overrides.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/utils/common.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package config

// deepseekMoeQuantResolution is _deepseek_moe_quant_resolution (overrides.py:763-843) for the
// accepted DeepSeek-family architectures on SM100: the quantization the checkpoint declares (or,
// for DeepseekV3ForCausalLM with none declared, fp8 when its first expert tensor is float8_e4m3fn,
// utils/common.py has_fp8_weights_in_checkpoint) becomes the quantization, and the MoE runner is
// FlashInfer's TRT-LLM kernel unless an all-to-all backend or a runner is stated. The Kimi K2 int4
// branch (a compressed-tensors checkpoint) and the LongCat fp8 GEMM branch belong to
// architectures Resolve does not accept.
func (r *resolver) deepseekMoeQuantResolution() {
	const src = "_deepseek_moe_quant_resolution"
	if !r.chip.isSM100() {
		return
	}
	quantization := r.cfg.Quantization
	if quantization == nil {
		// _quantization_explicitly_unset is false: Resolve accepts no quantization field that the
		// user states as unset.
		if r.model.QuantMethod == nil && r.model.Architecture == "DeepseekV3ForCausalLM" {
			if r.model.HasFP8WeightsInCheckpoint {
				r.declare(src, "quantization", "fp8")
				q := "fp8"
				quantization = &q
			}
		} else if r.model.QuantMethod != nil {
			r.declare(src, "quantization", *r.model.QuantMethod)
			q := *r.model.QuantMethod
			quantization = &q
		}
	}
	runnerQuant := quantization == nil
	if quantization != nil {
		switch *quantization {
		case "fp8", "modelopt_fp8", "modelopt_fp4", "modelopt_mixed":
			runnerQuant = true
		}
	}
	if r.cfg.MoeA2aBackend == "none" && r.cfg.MoeRunnerBackend == "auto" && runnerQuant {
		r.declare(src, "moe_runner_backend", "flashinfer_trtllm")
	}
}
