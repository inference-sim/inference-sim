// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/arg_groups/speculative_hook.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/arg_groups/overrides.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package config

// eagleDraftInTarget are the architectures whose MTP draft layers ship inside the target
// checkpoint, so the draft model path defaults to the target's (speculative_hook.py:1121-1136).
var eagleDraftInTarget = map[string]bool{
	"DeepseekV32ForCausalLM": true, "DeepseekV3ForCausalLM": true, "DeepseekV4ForCausalLM": true,
	"Glm4MoeForCausalLM": true, "Glm4MoeLiteForCausalLM": true, "GlmMoeDsaForCausalLM": true,
	"BailingMoeForCausalLM": true, "BailingMoeV2ForCausalLM": true, "BailingMoeV2_5ForCausalLM": true,
	"MistralLarge3ForCausalLM": true, "PixtralForConditionalGeneration": true, "HYV3ForCausalLM": true,
	"HYV4ForCausalLM": true, "Qwen4ExpForConditionalGeneration": true,
}

// handleSpeculativeDecoding is handle_speculative_decoding (speculative_hook.py:121-284) for the
// accepted algorithms, EAGLE and its alias NEXTN, with no separate draft checkpoint.
func (r *resolver) handleSpeculativeDecoding() {
	const src = "handle_speculative_decoding"
	// _speculative_moe_runner_default (overrides.py:1631-1656).
	if r.cfg.SpeculativeMoeRunnerBackend == nil {
		d := r.cfg.MoeRunnerBackend
		switch {
		case d == "flashinfer_megamoe":
			d = "flashinfer_trtllm"
		case len(d) >= 5 && d[:5] == "lora_":
			d = "auto"
		}
		r.declare("_speculative_moe_runner_default", "speculative_moe_runner_backend", d)
	}
	if r.cfg.SpeculativeAlgorithm == nil {
		return
	}
	// The algorithm upper-cased, then _resolve_speculative_algorithm_alias
	// (speculative_hook.py:81-118): NEXTN is EAGLE.
	algo := upper(*r.cfg.SpeculativeAlgorithm)
	if algo == "NEXTN" {
		algo = "EAGLE"
	}
	r.declare(src, "speculative_algorithm", algo)
	r.handleEagleFamily()
}

// handleEagleFamily is _handle_eagle_family (speculative_hook.py:1072-1282), which the EAGLE
// algorithm's registry entry runs (handle_server_args).
func (r *resolver) handleEagleFamily() {
	const src = "_handle_eagle_family"
	if r.cfg.MaxRunningRequests == nil {
		r.declare(src, "max_running_requests", 48)
	}
	if eagleDraftInTarget[r.model.Architecture] && r.cfg.SpeculativeDraftModelPath == nil {
		r.declare(src, "speculative_draft_model_path", r.cfg.ModelPath,
			"speculative_draft_model_revision", clonePtr(r.cfg.Revision))
	}
	topk, steps, draft := *r.cfg.SpeculativeEagleTopk, *r.cfg.SpeculativeNumSteps, *r.cfg.SpeculativeNumDraftTokens
	if p, d := r.attentionBackends(); (is(p, "trtllm_mha") || is(d, "trtllm_mha")) && topk > 1 {
		r.raise("speculative_hook.py:_handle_eagle_family", "trtllm_mha backend only supports topk = 1 for speculative decoding.")
	}
	if topk == 1 && draft != steps+1 {
		r.declare(src, "speculative_num_draft_tokens", steps+1)
	}
	if topk > 1 && *r.cfg.PageSize > 1 {
		a := ""
		if r.cfg.AttentionBackend != nil {
			a = *r.cfg.AttentionBackend
		}
		if a != "flashinfer" && a != "fa3" && a != "triton" {
			r.raise("speculative_hook.py:_handle_eagle_family",
				"speculative_eagle_topk > 1 with page_size > 1 is only supported on ('flashinfer', 'fa3', 'triton'); got attention_backend='"+a+"'. Use page_size == 1 or one of those backends.")
		}
	}
}
