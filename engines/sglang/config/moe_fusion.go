// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/arg_groups/overrides.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).
// Derived from python/sglang/srt/arg_groups/moe_hook.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).

package config

// handleMoeKernelConfig is handle_moe_kernel_config (moe_hook.py:68-190) as far as the accepted
// inputs reach it: the quantization-driven runner resolutions change nothing for the quantizations
// Resolve resolves, the runner compatibility asserts hold for the runners it resolves, and the
// shared-expert fusion write is _moe_runner_fusion_disable.
func (r *resolver) handleMoeKernelConfig() {
	r.moeRunnerFusionDisable()
}

// moeRunnerFusionDisable is _moe_runner_fusion_disable (overrides.py:1529-1553): shared-expert
// fusion is switched off for the MoE runners that do not fuse them.
func (r *resolver) moeRunnerFusionDisable() {
	switch r.cfg.MoeRunnerBackend {
	case "lora_triton", "lora_cutedsl", "lora_marlin", "flashinfer_cutedsl",
		"flashinfer_trtllm", "experimental_sgl_trtllm", "flashinfer_trtllm_routed":
		r.declare("_moe_runner_fusion_disable", "disable_shared_experts_fusion", true)
	}
}
