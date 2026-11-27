// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/arg_groups/serving_hook.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/arg_groups/hicache_hook.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/arg_groups/pd_disaggregation_hook.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/arg_groups/parallel_hook.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/arg_groups/kv_cache_hook.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/arg_groups/model_path_hook.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/arg_groups/validation_hook.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/arg_groups/boundary_reduction.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/server_args.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package config

import "strconv"

// The resolution steps of the serving, HiCache, disaggregation, parallelism, KV-cache and
// model-path hooks, each as far as an accepted input reaches it.

// handleHicacheRatioDefault is handle_hicache_ratio_default (hicache_hook.py:100-126).
func (r *resolver) handleHicacheRatioDefault() {
	const src = "_handle_hicache_ratio_default"
	if f := r.cfg.HicacheHostMemoryFraction; f != nil && !(0 < *f && *f <= 1) {
		r.raise("hicache_hook.py:handle_hicache_ratio_default", "--hicache-host-memory-fraction must be in (0, 1].")
	}
	// Both conditions read the view before either declaration, as SGLang collects them first.
	setRatio := r.cfg.HicacheRatio == nil && r.cfg.DisaggregationMode != "decode"
	clearFraction := r.cfg.HicacheRatio != nil || r.cfg.HicacheSize > 0
	if setRatio {
		ratio := 2.0
		if r.cfg.HicacheHostMemoryMode == "buffer_only" {
			ratio = 1.2
		}
		r.declare(src, "hicache_ratio", ratio)
	}
	if clearFraction {
		r.declare(src, "hicache_host_memory_fraction", nil)
	}
}

// handleGrpcArgs is handle_grpc_args (serving_hook.py:268-). Only its unconditional declaration
// reaches an accepted input: grpc_worker_threads takes SGLANG_GRPC_WORKER_THREADS, whose default
// is 4 (environ.py:SGLANG_GRPC_WORKER_THREADS). The deprecated-flag renames this step carried at
// v0.5.23 are gone: a removed name is refused by checkRemovedOptions.
func (r *resolver) handleGrpcArgs() {
	r.declare("_handle_grpc_args", "grpc_worker_threads", 4)
}

// handleMissingDefaultValues is handle_missing_default_values (serving_hook.py:632-709).
func (r *resolver) handleMissingDefaultValues() {
	const src = "_handle_missing_default_values"
	if r.cfg.TokenizerPath == nil {
		r.declare(src, "tokenizer_path", r.cfg.ModelPath)
	}
	if r.cfg.ServedModelName == nil {
		r.declare(src, "served_model_name", r.cfg.ModelPath)
	}
	if r.cfg.Device == nil {
		// get_device() on the CUDA platform.
		r.declare(src, "device", "cuda")
	}
	// random_seed: SGLang draws one when it is unset; Resolve does not (Resolved.RandomSeedDrawn).
	if r.cfg.MmProcessConfig == nil {
		r.declare(src, "mm_process_config", map[string]any{})
	}
	if r.internal.SpeculativeDraftQuantizationExplicitlySet == nil {
		r.internal.SpeculativeDraftQuantizationExplicitlySet = ptr(r.cfg.SpeculativeDraftModelQuantization != nil)
	}
	if r.cfg.SpeculativeDraftModelQuantization == nil {
		r.declare(src, "speculative_draft_model_quantization", clonePtr(r.cfg.Quantization))
	}
}

// handlePDDisaggregation is handle_pd_disaggregation (pd_disaggregation_hook.py:23-196) for the
// decode and prefill roles over the nixl transfer backend.
func (r *resolver) handlePDDisaggregation() {
	const src = "handle_pd_disaggregation"
	switch r.cfg.DisaggregationMode {
	case "decode":
		// disaggregation_decode_enable_radix_cache is not accepted, so it is False.
		r.declare(src, "disable_radix_cache", true)
		if r.cfg.DisaggregationDecodeExtraSlots == nil {
			extra := 0
			if r.cfg.MaxRunningRequests != nil {
				perWorker := max(1, *r.cfg.MaxRunningRequests/r.numDPRanks())
				if perWorker <= 32 {
					extra = perWorker * 2
				}
			}
			r.declare(src, "disaggregation_decode_extra_slots", extra)
		}
	case "prefill":
		// The prefill role's checks (the fake backend, the Rust server) do not reach nixl.
	}
}

// numDPRanks is num_dp_ranks_of (runtime_context.py:202-207): replicas times attention-DP groups.
func (r *resolver) numDPRanks() int { return r.cfg.DpSize * r.cfg.AttnDpSize }

// attnDPEnabled is attn_dp_enabled_of (runtime_context.py:191-199); an elastic-EP scale joiner
// (ep_join_mode, not accepted) is the other way in.
func (r *resolver) attnDPEnabled() bool { return r.cfg.AttnDpSize > 1 }

// handleDecodeContextParallelism is handle_decode_context_parallelism (parallel_hook.py:240-273)
// with decode context parallelism off: _dcp_comm_backend_default (overrides.py:1373-1396)
// resolves ag_rs.
func (r *resolver) handleDecodeContextParallelism() {
	if r.cfg.DcpCommBackend == nil && r.cfg.DcpSize <= 1 {
		r.declare("_dcp_comm_backend_default", "dcp_comm_backend", "ag_rs")
	}
}

// defaultUnsetPrefillDecodeInterval is default_unset_prefill_decode_interval
// (validation_hook.py:665-675).
func (r *resolver) defaultUnsetPrefillDecodeInterval() {
	if r.cfg.PrefillDecodeInterval == nil {
		r.declare("prefill_decode_interval_default", "prefill_decode_interval", 0)
	}
}

// samplingBackendDefault is _sampling_backend_default (overrides.py:1018-1026): flashinfer, which
// SGLang's CUDA image installs.
func (r *resolver) samplingBackendDefault() {
	if r.cfg.SamplingBackend == nil {
		r.declare("_sampling_backend_default", "sampling_backend", "flashinfer")
	}
}

// pageSizeDefault is _page_size_default (overrides.py:1340-1363) on CUDA.
func (r *resolver) pageSizeDefault() {
	if r.cfg.PageSize == nil {
		r.declare("_page_size_default", "page_size", 1)
	}
}

// handleGrammarBackend is handle_grammar_backend (serving_hook.py:239-244).
func (r *resolver) handleGrammarBackend() {
	if r.cfg.GrammarBackend == nil {
		r.declare("_handle_grammar_backend", "grammar_backend", "xgrammar")
	}
}

// handleDataParallelism is handle_data_parallelism (parallel_hook.py:276-373).
func (r *resolver) handleDataParallelism() {
	const src = "_handle_data_parallelism"
	if r.cfg.AttnDpSize < 1 {
		r.raise("parallel_hook.py:handle_data_parallelism", "--attn-dp-size must be positive (got "+strconv.Itoa(r.cfg.AttnDpSize)+").")
	}
	if r.cfg.AttnDpSize > 1 && r.cfg.DpSize > 1 {
		r.raise("parallel_hook.py:handle_data_parallelism", "--dp-size "+strconv.Itoa(r.cfg.DpSize)+" with --attn-dp-size "+
			strconv.Itoa(r.cfg.AttnDpSize)+": data-parallel replicas combined with attention data parallelism are not supported.")
	}
	// _data_parallelism_defaults (overrides.py:1366-1370).
	if r.numDPRanks() == 1 && (r.cfg.EpJoinMode == nil || *r.cfg.EpJoinMode != "scale") {
		r.declare("_data_parallelism_defaults", "enable_dp_lm_head", false)
	}
	if r.attnDPEnabled() {
		r.declare(src, "schedule_conservativeness", r.cfg.ScheduleConservativeness*0.3)
		if r.cfg.TpSize%r.cfg.AttnDpSize != 0 {
			r.raise("parallel_hook.py:handle_data_parallelism", "")
		}
		chunk := pyFloorDiv(*r.cfg.ChunkedPrefillSize, r.cfg.AttnDpSize)
		r.declare(src, "chunked_prefill_size", chunk)
		// Re-clamp the prefill graph to the per-rank chunk (parallel_hook.py:338-365).
		pre := r.cfg.CudaGraphConfig.Prefill
		if pre.Backend != backendDisabled && pre.MaxBs != nil && *pre.MaxBs > chunk &&
			!r.cgLocked[[2]string{"prefill", "max_bs"}] {
			lockedBs := r.cgLocked[[2]string{"prefill", "bs"}]
			r.declare(src, "cuda_graph_config", r.cfg.CudaGraphConfig.withPhase("prefill", func(p *PhaseConfig) {
				p.MaxBs = ptr(chunk)
				if !lockedBs {
					p.Bs = generatePrefillBatchSizes(chunk)
				}
			}))
		}
	}
	// _tp_lm_head_all_to_all_default (overrides.py:1399-1418).
	if r.cfg.EnableTpLmHeadAllToAll == nil {
		enable := r.cfg.DisaggregationMode == "decode" && r.cfg.AttnDpSize > 1 &&
			r.cfg.TpSize == r.cfg.AttnDpSize && r.cfg.AttnCpSize == 1 && !r.cfg.EnableDpLmHead
		r.declare("_tp_lm_head_all_to_all_default", "enable_tp_lm_head_all_to_all", enable)
	}
}

// handleLoadBalanceMethod is handle_load_balance_method (serving_hook.py:217-236).
func (r *resolver) handleLoadBalanceMethod() {
	if r.cfg.LoadBalanceMethod == "auto" {
		m := "round_robin"
		if r.cfg.DisaggregationMode == "prefill" {
			m = "follow_bootstrap_room"
		}
		r.declare("_handle_load_balance_method", "load_balance_method", m)
	}
}

// handleA2AMoE is handle_a2a_moe (moe_hook.py:284-621) as far as the accepted all-to-all
// backends reach it: _a2a_ep_size (overrides.py:1610-1620) spans the experts over the TP group.
func (r *resolver) handleA2AMoE() {
	if r.cfg.MoeA2aBackend == "deepep" {
		r.declare("_a2a_ep_size", "ep_size", r.cfg.TpSize)
	}
}

// handleExpertDistributionMetrics is handle_expert_distribution_metrics (parallel_hook.py:794-817).
func (r *resolver) handleExpertDistributionMetrics() {
	// should_report_expert_balancedness is false: expert_balancedness_report_mode is off.
	if r.cfg.ExpertDistributionRecorderBufferSize == nil {
		r.declare("_handle_expert_distribution_metrics", "expert_distribution_recorder_buffer_size",
			r.cfg.EplbRebalanceNumIterations)
	}
}

// handleLoadFormat is handle_load_format (model_path_hook.py:163-287) for a local checkpoint in
// the auto format.
func (r *resolver) handleLoadFormat() {
	if r.cfg.CustomWeightLoader == nil {
		r.declare("_handle_load_format", "custom_weight_loader", []string{})
	}
}

// handleEncoderDisaggregation is handle_encoder_disaggregation (pd_disaggregation_hook.py:298-386)
// for a server that is neither encoder-only nor language-only; resolve_encoder_transfer_backend
// is server_args.py:571-578.
func (r *resolver) handleEncoderDisaggregation() {
	if r.cfg.EncoderTransferBackend == "auto" {
		b := "zmq_to_scheduler"
		if r.model.Architecture == "KimiK3ForConditionalGeneration" && r.cfg.TpSize > 1 {
			b = "zmq_to_tokenizer"
		}
		r.declare("_handle_encoder_disaggregation", "encoder_transfer_backend", b)
	}
}

// handleEnvironmentVariables is handle_environment_variables (serving_hook.py:435-493); the one
// field it declares is handle_multimodal_feature_transport's (serving_hook.py:800-942). On a single
// node with no SGLANG_USE_CUDA_IPC_TRANSPORT every arm of it resolves cpu.
func (r *resolver) handleEnvironmentVariables() {
	t := "cpu"
	if r.cfg.MmFeatureTransport != nil {
		t = *r.cfg.MmFeatureTransport
	}
	r.declare("_handle_multimodal_feature_transport", "mm_feature_transport", t)
}

// handleCacheCompatibility is handle_cache_compatibility (kv_cache_hook.py:351-474) as far as
// the accepted inputs reach it.
func (r *resolver) handleCacheCompatibility() {
	if r.cfg.EnableHierarchicalCache && r.cfg.DisableRadixCache {
		r.raise("kv_cache_hook.py:handle_cache_compatibility",
			"The arguments enable-hierarchical-cache and disable-radix-cache are mutually exclusive and cannot be used at the same time. Please use only one of them.")
	}
	if r.internal.SwaFullTokensRatioExplicitlySet == nil {
		r.internal.SwaFullTokensRatioExplicitlySet = ptr(r.cfg.SwaFullTokensRatio != nil)
	}
	// kv_cache_hook.py:545-551@c4af57ff: read from the input, since a model's default (DeepSeek-V4's)
	// is declared earlier and leaves it false.
	if r.internal.MaxRunningRequestsExplicitlySet == nil {
		r.internal.MaxRunningRequestsExplicitlySet = ptr(r.raw.MaxRunningRequests != nil)
	}
}

// resolveBoundaryReduction is resolve_boundary_reduction (boundary_reduction.py:57-92). A draft
// from another checkpoint is not accepted, so the draft resolves as the target does.
func (r *resolver) resolveBoundaryReduction() {
	target := r.cfg.BoundaryReduction
	draft := target
	if target == "auto" {
		target = r.modelBoundaryReduction()
		draft = target
	}
	r.declare("resolve_boundary_reduction", "boundary_reduction", target, "speculative_boundary_reduction", draft)
}

// modelBoundaryReduction is model_boundary_reduction (boundary_reduction.py:28-54) over the
// architecture and the text model's type.
func (r *resolver) modelBoundaryReduction() string {
	switch r.model.Architecture {
	case "BailingMoELinearForCausalLM", "BailingMoeV2_5ForCausalLM", "LongcatFlashForCausalLM",
		"LongcatFlashForCausalLMNextN", "Step3VLForConditionalGeneration":
		return "rsv"
	case "KimiK3ForConditionalGeneration", "KimiK3LinearForCausalLM", "Qwen3ForCausalLM",
		"Qwen3ForSequenceClassification", "Qwen3Model", "Qwen3ForRewardModel", "MossVLForCausalLM",
		"MossVLForConditionalGeneration", "Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLMMTP":
		return "ar"
	}
	for _, t := range []string{r.model.ModelType, r.model.TextModelType} {
		if t == "qwen3" || t == "qwen3_vl_text" {
			return "ar"
		}
	}
	return "rs+rsv"
}

// pyFloorDiv is Python's // on ints.
func pyFloorDiv(a, b int) int {
	q := a / b
	if (a%b != 0) && ((a < 0) != (b < 0)) {
		q--
	}
	return q
}
