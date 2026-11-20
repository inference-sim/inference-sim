// Original to blis-sglang.

package config

import (
	"fmt"
	"sort"
	"strings"
)

// The input space Resolve models: the inputs the oracle's cases cover (oracle/serverargs/cases.json).
// Every other input is refused by name with the open issue that models it, so Resolve never
// resolves a configuration by a guess (X1, X28).
const issueRest = "inference-sim/blis-sglang#49"

// acceptedFields are the ServerArgs fields a user may set away from their defaults. A field
// with a value list accepts only those values; nil accepts any value of the field's type.
var acceptedFields = map[string][]any{
	"model_path":                      nil,
	"random_seed":                     nil,
	"tp_size":                         nil,
	"dp_size":                         nil,
	"enable_dp_attention":             nil,
	"ep_size":                         nil,
	"moe_a2a_backend":                 {"none", "deepep"},
	"chunked_prefill_size":            nil,
	"max_running_requests":            nil,
	"page_size":                       nil,
	"schedule_policy":                 nil,
	"disable_radix_cache":             nil,
	"enable_hierarchical_cache":       nil,
	"hicache_storage_backend":         {nil, "file"},
	"hicache_ratio":                   nil,
	"disaggregation_mode":             {"null", "prefill", "decode"},
	"disaggregation_transfer_backend": {"nixl"},
	"mem_fraction_static":             nil,
	// EAGLE3 resolves as EAGLE does: both are is_eagle() and run _handle_eagle_family
	// (speculative_spec_info.py:104-118, :255-266@297e54fd); NEXTN is renamed EAGLE first. The flag
	// enable_multi_layer_eagle is read by no resolution step modelled here (it selects the worker,
	// speculative_spec_info.py:354, and a DP-coordination check).
	"speculative_algorithm":        {nil, "EAGLE", "NEXTN", "EAGLE3"},
	"speculative_num_steps":        nil,
	"speculative_eagle_topk":       nil,
	"speculative_num_draft_tokens": nil,
	"enable_multi_layer_eagle":     nil,
	// The InferenceX h200 recipes of DeepSeek-R1 and GLM-5 (oracle cases dsr1-h200-tp8-recipe-*,
	// glm5-h200-tp8-recipe, dsv3-h200-tp8-fa3-explicit, dsv3-h200-tp1-fusion-flag). Each is either
	// read by the scheduler or the lowering (fields/status.json cites the traces and tests) or proven
	// inert to every decision there.
	"max_prefill_tokens":                 nil,
	"tokenizer_worker_num":               nil,
	"kv_cache_dtype":                     {"auto", "bf16", "bfloat16", "fp8_e4m3", "fp8_e5m2"},
	"quantization":                       {nil, "fp8", "modelopt_fp4"},
	"tokenizer_path":                     nil,
	"model_loader_extra_config":          nil,
	"dsa_prefill_backend":                {nil, "flashmla_sparse", "flashmla_kv", "trtllm", "fa3"},
	"dsa_decode_backend":                 {nil, "flashmla_sparse", "flashmla_kv", "trtllm", "fa3"},
	"cuda_graph_max_bs_decode":           nil,
	"attention_backend":                  {nil, "flashinfer", "fa3", "dsa", "nsa"},
	"stream_interval":                    nil,
	"decode_log_interval":                nil,
	"enable_flashinfer_allreduce_fusion": nil,
	"host":                               nil,
	"port":                               nil,
	"trust_remote_code":                  nil,
	"served_model_name":                  nil,
	"tool_call_parser":                   nil,
	"reasoning_parser":                   nil,
	"context_length":                     nil,
	"pp_size":                            {1},
	// The hybrid (Mamba) pool settings of the Qwen3.5 family (qwen35.go): the strategy of the Mamba
	// radix cache, the dtype of its SSM states, and the vision tower's attention backend.
	"mamba_radix_cache_strategy": {"auto", "no_buffer", "extra_buffer", "extra_buffer_lazy"},
	"mamba_ssm_dtype":            {nil, "float32", "bfloat16", "float16"},
	"mm_attention_backend":       {nil, "sdpa", "fa3", "fa4", "triton_attn", "flashinfer_cudnn"},
	// The scheduler's queue, priority, truncation and retraction settings: each is read by the
	// scheduler port (core/sched) and covered by its oracle goldens (fields/status.json cites them);
	// no resolution step reads any of them (CheckServerArgs carries the two start-up rules).
	"enable_priority_scheduling":               nil,
	"priority_scheduling_preemption_threshold": nil,
	"schedule_low_priority_values_first":       nil,
	"abort_on_priority_when_disabled":          nil,
	"max_queued_requests":                      nil,
	"prefill_max_requests":                     nil,
	"allow_auto_truncate":                      nil,
	"retraction_policy":                        {"length", "priority"},
}

// refusedFieldIssue names, for a field another open issue covers, that issue instead of the
// general one. A family is the settings of one feature the engine does not model: the issue that
// models the feature is the one a user waits on.
var refusedFieldIssue = func() map[string]string {
	const repo = "inference-sim/blis-sglang#"
	m := map[string]string{}
	put := func(issue string, names ...string) {
		for _, n := range names {
			m[n] = repo + issue
		}
	}
	put("26", "attn_cp_size", "dcp_size", "enable_prefill_cp", "cp_strategy", "dcp_comm_backend")
	put("64", "disaggregation_mode", "disaggregation_bootstrap_port", "disaggregation_decode_allocation_policy",
		"disaggregation_decode_extra_slots", "disaggregation_decode_polling_interval", "disaggregation_enable_kv_checksum")
	put("65", "disaggregation_decode_enable_offload_kvcache", "disaggregation_decode_enable_radix_cache",
		"disaggregation_decode_host_receive_threshold", "disaggregation_decode_retraction_backup")
	put("66", "disaggregation_ib_device", "disaggregation_transfer_backend")
	put("38", "enable_hierarchical_cache", "hicache_ratio", "hicache_host_memory_fraction", "hicache_host_memory_mode", "hicache_io_backend",
		"hicache_mem_layout", "hicache_size", "hicache_write_policy")
	put("40", "hicache_storage_backend", "hicache_storage_prefetch_policy", "hicache_storage_prefetch_retry_max_attempts",
		"hicache_storage_prefetch_retry_poll_interval", "file_storage_path")
	put("87", "hicache_storage_backend_extra_config")
	put("70", "enable_dp_attention", "load_balance_method", "attn_dp_size", "moe_dp_size", "enable_dp_lm_head", "enable_dp_attention_local_control_broadcast")
	put("71", "deepep_config", "deepep_dispatcher_output_dtype", "deepep_mode", "deepep_v2_mode", "fuseep_mode",
		"enable_eplb", "eplb_algorithm", "eplb_min_rebalancing_utilization_threshold", "eplb_rebalance_layers_per_chunk",
		"eplb_rebalance_num_iterations", "ep_dispatch_algorithm", "ep_num_redundant_experts", "init_expert_location",
		"enable_two_batch_overlap", "tbo_token_distribution_threshold", "expert_distribution_recorder_buffer_size",
		"expert_distribution_recorder_mode", "expert_balancedness_report_mode", "moe_a2a_backend")
	put("72", "pp_size", "pp_max_micro_batch_size", "pp_async_batch_depth")
	put("73", "swa_full_tokens_ratio", "disable_hybrid_swa_memory", "swa_prefix_tails")
	put("94", "mamba_backend", "mamba_full_memory_ratio", "mamba_max_states_per_path", "mamba_track_interval", "max_mamba_cache_size")
	for _, n := range FieldNames() {
		switch {
		case strings.HasPrefix(n, "speculative_dspark_"):
			put("69", n)
		case strings.HasPrefix(n, "speculative_"):
			put("68", n)
		}
	}
	return m
}()

// refusedValueIssue names, for a field Resolve models at some values only, the open issue that
// models one value it refuses; a value absent here takes the field's issue.
var refusedValueIssue = map[string]map[string]string{
	"moe_a2a_backend": {"mooncake": "inference-sim/blis-sglang#30"},
	"hicache_storage_backend": {
		"mooncake": "inference-sim/blis-sglang#14", "aibrix": "inference-sim/blis-sglang#16", "eic": "inference-sim/blis-sglang#16",
		"simm": "inference-sim/blis-sglang#16", "mori": "inference-sim/blis-sglang#16", "tensorcast": "inference-sim/blis-sglang#16",
	},
	"speculative_algorithm": {"DSPARK": "inference-sim/blis-sglang#69"},
	"disaggregation_transfer_backend": {
		"mooncake": "inference-sim/blis-sglang#18", "mooncake_tcp": "inference-sim/blis-sglang#18",
		// The fake backend completes a transfer at the step a scenario names; the P/D queues that use it are S25.
		"fake": "inference-sim/blis-sglang#64",
	},
}

// refusalIssue is the open issue that models name at value v: the value's own, else the field's,
// else the general one.
func refusalIssue(name string, v any) string {
	if sv, isString := v.(string); isString {
		if i, ok := refusedValueIssue[name][sv]; ok {
			return i
		}
	}
	if i, ok := refusedFieldIssue[name]; ok {
		return i
	}
	return issueRest
}

// acceptedModels are the architectures (after SGLang's config translation) and the chips each
// is modelled on, by compute capability major. The DeepSeek family on SM100 reads the checkpoint's
// first expert tensor (_deepseek_moe_quant_resolution, overrides.py:763-843@297e54fd); that is the
// model fact HasFP8WeightsInCheckpoint.
var acceptedModels = map[string][]int{
	"Qwen2ForCausalLM":                   {9, 10},
	"Qwen3ForCausalLM":                   {9, 10},
	"LlamaForCausalLM":                   {9, 10},
	"MistralForCausalLM":                 {9, 10},
	"GlmMoeDsaForCausalLM":               {9, 10},
	"DeepseekV3ForCausalLM":              {9, 10},
	"DeepseekV4ForCausalLM":              {9, 10},
	"Qwen3_5MoeForConditionalGeneration": {9, 10},
}

// checkAccepted refuses an input outside the space Resolve models.
func checkAccepted(args *ServerArgs, in Input) error {
	if args.ModelPath == "" {
		return fmt.Errorf("model_path: SGLang's ServerArgs requires it (python/sglang/srt/arg_groups/fields/model.py:51@297e54fd)")
	}
	def := Defaults()
	for _, name := range FieldNames() {
		if !differs(args, &def, name) {
			continue
		}
		// An empty custom_weight_loader is the value SGLang's resolution gives an unset one
		// (model_path_hook.py handle_load_format: `if custom_weight_loader is None: = []`), and a server's
		// own record of its resolved arguments states it; it loads no custom weight loader, so it is the
		// default, not a setting.
		if name == "custom_weight_loader" && len(args.CustomWeightLoader) == 0 {
			continue
		}
		v := args.value(name)
		allowed, ok := acceptedFields[name]
		if !ok {
			return &NotModelled{Name: name, Value: v,
				Reason: "Resolve does not model this field away from its default", Issue: refusalIssue(name, v)}
		}
		if allowed != nil && !contains(allowed, v) {
			return &NotModelled{Name: name, Value: v,
				Reason: fmt.Sprintf("Resolve models only %v for this field", allowed), Issue: refusalIssue(name, v)}
		}
	}
	if args.SpeculativeAlgorithm != nil &&
		(args.SpeculativeNumSteps == nil || args.SpeculativeEagleTopk == nil || args.SpeculativeNumDraftTokens == nil) {
		return &NotModelled{Name: "speculative_num_steps", Value: nil,
			Reason: "Resolve models speculative decoding with speculative_num_steps, speculative_eagle_topk and speculative_num_draft_tokens all stated",
			Issue:  issueRest}
	}
	if args.DisaggregationMode != "null" && args.DisaggregationTransferBackend != "nixl" {
		return &NotModelled{Name: "disaggregation_transfer_backend", Value: args.DisaggregationTransferBackend,
			Reason: "Resolve models P/D disaggregation over nixl", Issue: refusalIssue("disaggregation_transfer_backend", args.DisaggregationTransferBackend)}
	}
	if args.SpeculativeAlgorithm != nil && !eagleDraftInTarget[in.Model.Architecture] {
		return &NotModelled{Name: "speculative_algorithm", Value: *args.SpeculativeAlgorithm,
			Reason: "Resolve models speculative decoding only for a model whose MTP draft ships in its own checkpoint (a separate draft checkpoint, which EAGLE3 normally names, resolves speculative_draft_model_revision and is not modelled)",
			Issue:  refusalIssue("speculative_algorithm", *args.SpeculativeAlgorithm)}
	}
	chips, ok := acceptedModels[in.Model.Architecture]
	if !ok {
		return &NotModelled{Name: "architecture", Value: in.Model.Architecture,
			Reason: "Resolve does not model this architecture's resolution", Issue: issueRest}
	}
	cc := in.Chip.Capability
	// SM90 (H100, H200), SM100 (B200) and SM103 (B300): SGLang's resolution reads the major only.
	if (cc != [2]int{9, 0} && cc != [2]int{10, 0} && cc != [2]int{10, 3}) || !containsInt(chips, cc[0]) {
		return &NotModelled{Name: "chip", Value: fmt.Sprintf("sm%d%d", cc[0], cc[1]),
			Reason: "Resolve does not model " + in.Model.Architecture + " on this chip", Issue: issueRest}
	}
	if in.Chip.MemoryMiB <= 0 {
		return &NotModelled{Name: "chip", Value: in.Chip.MemoryMiB,
			Reason: "the chip's memory must be stated", Issue: issueRest}
	}
	if in.Model.IsMultimodal && in.Model.HasVisionConfig && in.Model.VisionHiddenSize == nil {
		return &NotModelled{Name: "architecture", Value: in.Model.Architecture,
			Reason: "Resolve does not model the vision-encoder memory adjustment", Issue: issueRest}
	}
	return nil
}

func contains(l []any, v any) bool {
	for _, x := range l {
		if x == v {
			return true
		}
	}
	return false
}

func containsInt(l []int, v int) bool {
	for _, x := range l {
		if x == v {
			return true
		}
	}
	return false
}

// RefusalIssue is the open issue a refusal of the field by name cites when the field is set away
// from its default and Resolve does not model it, and false when the field has no issue of its own.
func RefusalIssue(name string) (string, bool) {
	i, ok := refusedFieldIssue[name]
	return i, ok
}

// ValueRefusalIssue is the open issue a refusal cites when the field is set to the value v, away
// from its default and not modelled: the value's own issue, else the field's, else the general one.
// It is what the start-up refusal prints, so the surface list (surface.json) can be held to it.
func ValueRefusalIssue(name string, v any) string { return refusalIssue(name, v) }

// AcceptsValue reports whether Resolve accepts the field set to the value: the field is accepted and
// its value list, if it has one, holds the value.
func AcceptsValue(name string, v any) bool {
	allowed, ok := acceptedFields[name]
	return ok && (allowed == nil || contains(allowed, v))
}

// AcceptedFields lists the fields Resolve accepts away from their defaults, sorted.
func AcceptedFields() []string {
	out := make([]string, 0, len(acceptedFields))
	for k := range acceptedFields {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
