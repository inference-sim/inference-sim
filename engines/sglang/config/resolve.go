// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/arg_groups/pipeline.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/arg_groups/overrides.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

// Package config is SGLang's ServerArgs and its default resolution, ported from SGLang's own
// resolution pipeline (python/sglang/srt/arg_groups/pipeline.py, run by ServerArgs.resolve_once).
//
// A user configures BLIS's SGLang engine with SGLang's own field names (X1). Resolve takes those
// fields, the chip's facts and the model's facts, and returns every ServerArgs field as SGLang
// resolves it, each with the SGLang step that decided it, or SGLang's own refusal. The oracle
// (oracle/run_serverargs.py) runs SGLang's pipeline on the same inputs, and the test compares the
// two field by field.
//
// Resolve models part of SGLang's input space, the part the oracle's cases cover. An input
// outside it is refused by name (Refusal), never resolved by a guess: see accepted.go.
package config

import (
	"fmt"
	"sort"
)

// Chip is what SGLang's resolution reads of the GPU it starts on.
type Chip struct {
	// Capability is the CUDA compute capability, major and minor (torch.cuda.get_device_capability).
	Capability [2]int `json:"capability"`
	// MemoryMiB is the device memory as nvidia-smi reports memory.total, which is what
	// get_device_memory_capacity returns on CUDA (python/sglang/srt/utils/common.py@297e54fd).
	MemoryMiB float64 `json:"nvidia_smi_memory_total_mib"`
}

func (c Chip) sm() int       { return c.Capability[0]*10 + c.Capability[1] }
func (c Chip) isSM90() bool  { return c.sm() == 90 }
func (c Chip) isSM100() bool { return c.Capability[0] == 10 }

// isHopperWithCUDA123 is the platform fact is_hopper_with_cuda_12_3: SM90 with CUDA 12.3 or
// newer. SGLang's CUDA image ships CUDA 13.0 (docker/Dockerfile:1@297e54fd).
func (c Chip) isHopperWithCUDA123() bool { return c.isSM90() }

// ModelFacts is what SGLang's resolution reads of the model, as SGLang's ModelConfig derives it
// from the model's configuration (python/sglang/srt/configs/model_config.py@297e54fd). The
// oracle records these facts with each case (oracle/run_serverargs.py model_facts).
type ModelFacts struct {
	// Architecture is hf_config.architectures[0] after SGLang's own config translation
	// (DeepSeek-V4.1's DeepseekV41ForCausalLM is read as DeepseekV4ForCausalLM,
	// configs/deepseek_v41.py:38-39).
	Architecture  string `json:"architecture"`
	ModelType     string `json:"model_type"`
	TextModelType string `json:"text_model_type"`
	// AttentionArch is ModelConfig.attention_arch: MHA or MLA.
	AttentionArch                     string  `json:"attention_arch"`
	IsMultimodal                      bool    `json:"is_multimodal"`
	MultimodalBreakableGraphSupported bool    `json:"is_multimodal_breakable_cuda_graph_supported"`
	HasVisionConfig                   bool    `json:"has_vision_config"`
	IsDeepseekDSA                     bool    `json:"is_deepseek_dsa"`
	DSAIndexKpool                     *int    `json:"dsa_index_kpool"`
	LearnableSink                     bool    `json:"learnable_sink"`
	HasAttentionSinks                 bool    `json:"has_attention_sinks"`
	HasAsymmetricKV                   bool    `json:"has_asymmetric_kv"`
	UsesKDAAttention                  bool    `json:"uses_kda_attention"`
	HasLinearAttnSpec                 bool    `json:"has_linear_attn_spec"`
	QuantMethod                       *string `json:"quant_method"`
	QuantWeightBlockSize              []int   `json:"quant_weight_block_size"`
	QuantScaleFmt                     *string `json:"quant_scale_fmt"`
	HiddenSize                        *int    `json:"hidden_size"`
	IsFP4Experts                      bool    `json:"is_fp4_experts"`
	HasNVFP4MoEMeta                   bool    `json:"has_nvfp4_moe_meta"`
	// HasFP8WeightsInCheckpoint is has_fp8_weights_in_checkpoint(model_path): whether the first
	// expert tensor of the checkpoint's safetensors header is float8_e4m3fn (utils/common.py).
	HasFP8WeightsInCheckpoint bool `json:"has_fp8_weights_in_checkpoint"`
	// VisionNumHiddenLayers and VisionHiddenSize are getattr(vision_config, "num_hidden_layers", 24)
	// and getattr(vision_config, "hidden_size", 1024) of a model with a vision config, which
	// adjust_mem_fraction_for_vlm reads (memory_hook.py:440-441).
	VisionNumHiddenLayers *int `json:"vision_num_hidden_layers"`
	VisionHiddenSize      *int `json:"vision_hidden_size"`
	// MixedPrecisionMoEQuantAlgos is _mixed_precision_moe_quant_algos(hf_config): the quant_algo
	// of the checkpoint's expert layers, upper case (model_overrides/qwen3_moe.py:17-26).
	MixedPrecisionMoEQuantAlgos []string `json:"mixed_precision_moe_quant_algos"`
}

func (m ModelFacts) useMLA() bool { return m.AttentionArch == "MLA" }

// Input is what a user hands SGLang: ServerArgs fields by their SGLang names, with the values
// JSON or YAML decodes them to, and the facts of the chip and the model.
type Input struct {
	Args  map[string]any
	Chip  Chip
	Model ModelFacts
}

// Internal holds the four ServerArgs fields whose names start with an underscore. SGLang's
// resolution writes them; no user sets them (arg_groups/fields/memory.py:45, schedule.py:38,
// spec.py:226@c4af57ff).
type Internal struct {
	// MaxRunningRequestsExplicitlySet records whether the input stated max_running_requests, not
	// whether resolution gave one (a model's default leaves it false).
	MaxRunningRequestsExplicitlySet *bool
	RadixEvictionPolicyExplicitlySet          bool
	SwaFullTokensRatioExplicitlySet           *bool
	SpeculativeDraftQuantizationExplicitlySet *bool
}

// Resolved is ServerArgs after SGLang's resolution.
type Resolved struct {
	Args     ServerArgs
	Internal Internal
	// DeclaredBy names, for each field whose value resolution changed, the SGLang step that
	// last decided it (the source SGLang's declare_resolution records).
	DeclaredBy map[string]string
	// RandomSeedDrawn is true when the input set no random_seed: SGLang then draws one
	// (random.randint(0, 1 << 30), serving_hook.py:660-665@297e54fd). Resolve draws nothing;
	// Args.RandomSeed stays nil and the simulator names its own seed.
	RandomSeedDrawn bool
}

// Value is a resolved field's value as JSON holds it, by its SGLang name, including the four
// internal fields.
func (r *Resolved) Value(name string) (any, bool) {
	switch name {
	case "_max_running_requests_explicitly_set":
		return jsonPtr(r.Internal.MaxRunningRequestsExplicitlySet), true
	case "_radix_eviction_policy_explicitly_set":
		return r.Internal.RadixEvictionPolicyExplicitlySet, true
	case "_swa_full_tokens_ratio_explicitly_set":
		return jsonPtr(r.Internal.SwaFullTokensRatioExplicitlySet), true
	case "_speculative_draft_quantization_explicitly_set":
		return jsonPtr(r.Internal.SpeculativeDraftQuantizationExplicitlySet), true
	}
	if !IsField(name) {
		return nil, false
	}
	return r.Args.value(name), true
}

func jsonPtr(p *bool) any {
	if p == nil {
		return nil
	}
	return *p
}

// UpstreamRefusal is SGLang's own refusal of a configuration: the message SGLang raises, and the
// function that raises it. BLIS refuses what SGLang refuses (SIM-9).
type UpstreamRefusal struct {
	Site    string
	Message string
}

func (e *UpstreamRefusal) Error() string { return "SGLang refuses this configuration: " + e.Message }

// NotModelled refuses an input Resolve does not model: a ServerArgs field, a model or a chip
// outside the input space the oracle's cases cover. It names the setting by its SGLang name
// and the open issue that models it (X1, X28).
type NotModelled struct {
	Name   string
	Value  any
	Reason string
	Issue  string
}

func (e *NotModelled) Error() string {
	return fmt.Sprintf("%s=%v is not modelled: %s (%s)", e.Name, e.Value, e.Reason, e.Issue)
}

// UnknownField refuses a name that is not a field of SGLang's ServerArgs (X1).
type UnknownField struct{ Name string }

func (e *UnknownField) Error() string {
	return fmt.Sprintf("%s is not a field of SGLang's ServerArgs", e.Name)
}

// refusal carries SGLang's refusal from the step that raises it up to Resolve.
type refusal struct{ err *UpstreamRefusal }

// resolver is the resolution state: the raw input and the resolving view over it. SGLang keeps
// the raw record and a list of declarations, the last of which wins (overrides.py:110-135,
// model_override_base.py:96-122); cfg is that view.
type resolver struct {
	raw      ServerArgs
	cfg      ServerArgs
	internal Internal
	by       map[string]string
	chip     Chip
	model    ModelFacts
	// cgLocked is _cuda_graph_config_locked: the (phase, key) pairs of the CUDA-graph
	// configuration a user set (cuda_graph_hook.py:32-101).
	cgLocked map[[2]string]bool
}

// declare is declare_resolution(server_args, source, name=value, ...) (overrides.py:110-135).
func (r *resolver) declare(source string, kv ...any) {
	for i := 0; i+1 < len(kv); i += 2 {
		name := kv[i].(string)
		if err := r.cfg.set(name, kv[i+1]); err != nil {
			panic("config: " + source + ": " + err.Error())
		}
		r.by[name] = source
	}
}

// raise stops resolution with SGLang's refusal, raised at site.
func (r *resolver) raise(site, message string) {
	panic(refusal{&UpstreamRefusal{Site: "sglang/srt/arg_groups/" + site, Message: message}})
}

// Resolve resolves input as SGLang v0.5.23's ServerArgs.resolve_once does on the input's chip.
func Resolve(in Input) (res *Resolved, err error) {
	if err := checkRemovedKwargs(in.Args); err != nil {
		return nil, err
	}
	if err := checkRemovedOptions(in.Args); err != nil {
		return nil, err
	}
	raw := Defaults()
	names := make([]string, 0, len(in.Args))
	for name := range in.Args {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if !IsField(name) {
			return nil, &UnknownField{Name: name}
		}
		if err := raw.set(name, in.Args[name]); err != nil {
			return nil, err
		}
	}
	if err := checkAccepted(&raw, in); err != nil {
		return nil, err
	}
	r := &resolver{raw: raw, cfg: cloneArgs(raw), by: map[string]string{}, chip: in.Chip, model: in.Model,
		cgLocked: map[[2]string]bool{}}
	defer func() {
		if p := recover(); p != nil {
			if rf, ok := p.(refusal); ok {
				res, err = nil, rf.err
				return
			}
			panic(p)
		}
	}()
	r.run()
	out := &Resolved{Args: r.cfg, Internal: r.internal, DeclaredBy: map[string]string{},
		RandomSeedDrawn: raw.RandomSeed == nil}
	for name, src := range r.by {
		if IsField(name) && differs(&r.raw, &r.cfg, name) {
			out.DeclaredBy[name] = src
		}
	}
	return out, nil
}

// cloneArgs copies a ServerArgs so that no pointer or slice is shared with the original.
func cloneArgs(s ServerArgs) ServerArgs {
	c := Defaults()
	for _, f := range fieldTable {
		if err := c.set(f.name, deref(f.ref(&s))); err != nil {
			panic(err)
		}
	}
	return c
}

// deref turns a field pointer into a value set accepts.
func deref(p any) any {
	switch v := p.(type) {
	case *bool:
		return *v
	case *int:
		return *v
	case *float64:
		return *v
	case *string:
		return *v
	case **bool:
		return clonePtr(*v)
	case **int:
		return clonePtr(*v)
	case **float64:
		return clonePtr(*v)
	case **string:
		return clonePtr(*v)
	case *[]string:
		if *v == nil {
			return nil
		}
		return *v
	case *[]int:
		if *v == nil {
			return nil
		}
		return *v
	case *[]float64:
		if *v == nil {
			return nil
		}
		return *v
	case **CudaGraphConfig:
		if *v == nil {
			return nil
		}
		return *v
	case *any:
		return *v
	}
	panic(fmt.Sprintf("config: deref of %T", p))
}

// run is run_resolution_pipeline (pipeline.py:20-377), in its order. A step that changes no
// field for any input Resolve accepts (accepted.go) is named in a comment rather than ported;
// SGLang's start-up checks in those steps are not yet ported either (inference-sim/inference-sim#2344).
func (r *resolver) run() {
	// validate_removed_options runs in Resolve (checkRemovedOptions), on the input as given.
	// handle_mega_moe, handle_return_hidden_states_mode, handle_media_url_security (it declares
	// allowed_media_domains, the [] it holds), handle_watermark_config: no change here.
	r.handleHicacheRatioDefault()
	// handle_offload_compatibility, the validate_* steps, handle_aux_hidden_state_capture,
	// validate_prefill_cp_platform, handle_hardware_runtime_validation, handle_model_source_paths,
	// apply_muse_spark_env_defaults, handle_multimodal, handle_ssl_validation and
	// handle_asr_validation: no change here.
	r.handleGrpcArgs()
	r.handleMissingDefaultValues()
	// handle_expert_pack: no change here.
	r.handlePDDisaggregation()
	// validate_prefill_only_disable_kv_cache_args: no change here.
	r.handleDecodeContextParallelism()
	// apply_inkling_prefill_cuda_graph_default, apply_muse_glimmer_prefill_cuda_graph_max_bs_default
	// (other architectures) and handle_dwdp: no change here.
	r.handleCudaGraphConfig()
	// handle_{hpu,cpu,npu,mps,xpu}_backends, handle_symm_mem_device_support and
	// handle_platform_defaults (an out-of-tree platform's hook; CUDA has none): no change here.
	r.handleGPUMemorySettings()
	r.handleModelSpecificAdjustments()
	r.defaultUnsetPrefillDecodeInterval()
	r.samplingBackendDefault()
	// handle_deterministic_inference, handle_nvfp4_prefill_kv_dequant_dtype: no change here.
	r.handleAttentionBackendCompatibility()
	r.disablePrefillCudaGraphForDeepseekTrtllmMla()
	// handle_mamba_backend, handle_int8_mamba_checkpoint, handle_linear_attn_backend,
	// handle_kv4_compatibility, handle_mxfp8_kv_cache_compatibility: no change here.
	r.pageSizeDefault()
	// handle_amd_specifics, handle_nccl_pre_warm: no change here.
	r.handleGrammarBackend()
	// handle_multi_item_scoring, handle_prefill_only_disable_kv_cache, handle_hicache: no change here.
	r.handleDataParallelism()
	r.handleLoadBalanceMethod()
	// handle_context_parallelism, handle_kv_cache_sharding, handle_moe_kernel_config: no change here.
	r.handleMoeKernelConfig()
	r.handleA2AMoE()
	// handle_shared_experts_tp, handle_eplb_and_dispatch: no change here.
	r.handleExpertDistributionMetrics()
	// handle_elastic_ep, validate_experimental_sgl_marlin, _pipeline_parallel_overlap_disable
	// (pp_size is 1): no change here.
	r.handleSpeculativeDecoding()
	r.resolveBoundaryReduction()
	// handle_layernorm_sp and the token-budget validations: no change here.
	r.handleLoadFormat()
	r.handleEncoderDisaggregation()
	// handle_tokenizer_batching: no change here.
	r.handleEnvironmentVariables()
	r.handleCacheCompatibility()
	// handle_page_major_kv_layout, handle_unified_memory_pool, handle_dllm_inference,
	// handle_crash_dump_env, handle_debug_utils (SGLANG_IS_IN_CI unset), handle_other_validations,
	// handle_model_capability_adjustments, the kernel-plugin bindings,
	// finalize_cuda_graph_prefill_max_context (no max context set), handle_kv_cache_sharding and
	// the DeepEP-v2 validations: no change here.
}
