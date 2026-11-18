// Original to blis-sglang.

package config

import (
	"fmt"
	"sort"
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
	"speculative_algorithm":           {nil, "EAGLE", "NEXTN"},
	"speculative_num_steps":           nil,
	"speculative_eagle_topk":          nil,
	"speculative_num_draft_tokens":    nil,
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
}

// refusedFieldIssue names, for a field another open issue covers, that issue instead of the
// general one.
var refusedFieldIssue = map[string]string{
	"attn_cp_size":      "inference-sim/blis-sglang#26",
	"dcp_size":          "inference-sim/blis-sglang#26",
	"enable_prefill_cp": "inference-sim/blis-sglang#26",
	"cp_strategy":       "inference-sim/blis-sglang#26",
	"dcp_comm_backend":  "inference-sim/blis-sglang#26",
}

// acceptedModels are the architectures (after SGLang's config translation) and the chips each
// is modelled on, by compute capability major. The DeepSeek family on SM100 reads the checkpoint's
// first expert tensor (_deepseek_moe_quant_resolution, overrides.py:763-843@297e54fd); that is the
// model fact HasFP8WeightsInCheckpoint.
var acceptedModels = map[string][]int{
	"Qwen2ForCausalLM":                   {9, 10},
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
			issue := refusedFieldIssue[name]
			if issue == "" {
				issue = issueRest
			}
			return &NotModelled{Name: name, Value: v,
				Reason: "Resolve does not model this field away from its default", Issue: issue}
		}
		if allowed != nil && !contains(allowed, v) {
			return &NotModelled{Name: name, Value: v,
				Reason: fmt.Sprintf("Resolve models only %v for this field", allowed), Issue: issueRest}
		}
	}
	if args.SpeculativeAlgorithm != nil &&
		(args.SpeculativeNumSteps == nil || args.SpeculativeEagleTopk == nil || args.SpeculativeNumDraftTokens == nil) {
		return &NotModelled{Name: "speculative_num_steps", Value: nil,
			Reason: "Resolve models speculative decoding with speculative_num_steps, speculative_eagle_topk and speculative_num_draft_tokens all stated",
			Issue:  issueRest}
	}
	if args.DisaggregationMode != "null" && args.DisaggregationTransferBackend != "nixl" {
		issue := issueRest
		if args.DisaggregationTransferBackend == "mooncake" {
			issue = "inference-sim/blis-sglang#18"
		}
		return &NotModelled{Name: "disaggregation_transfer_backend", Value: args.DisaggregationTransferBackend,
			Reason: "Resolve models P/D disaggregation over nixl", Issue: issue}
	}
	if args.SpeculativeAlgorithm != nil && !eagleDraftInTarget[in.Model.Architecture] {
		return &NotModelled{Name: "speculative_algorithm", Value: *args.SpeculativeAlgorithm,
			Reason: "Resolve models speculative decoding only for a model whose MTP draft ships in its own checkpoint",
			Issue:  issueRest}
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
