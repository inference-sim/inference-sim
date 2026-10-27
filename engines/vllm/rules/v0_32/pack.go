// Original to blis-vllm. Begun as a copy of the v0_31 pack (moved from
// inference-sim/blis-schemas/rules/v0_31/pack.go at c197a5ad5b2091f003ad2c3d016f3ef9738a6459),
// then re-derived against vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).

// Package v0_32 is the rules pack for vLLM 0.32.0 (tag v0.32.0, commit 5df63a72), the
// release blis-vllm executes.
//
// It began as a copy of v0_31. What moved between the two releases, each checked in the
// engine source at both tags by source_test.go:
//
//   - all2all_backend gains passthrough (vllm/config/parallel.py:44-59@5df63a72): no
//     all-to-all at all, bound automatically for MoE backends that dispatch and combine
//     themselves, not selectable from the command line (:231-233). It joins the subset that
//     makes the MoE input sequence-parallel (use_sequence_parallel_moe, :764-782).
//   - The cache dtypes gain ultraquant_4bit (vllm/config/cache.py:39-59@5df63a72).
//   - mamba_cache_mode loses all: the literal is Literal["align", "none"] (cache.py:63).
//   - The MoE backends lose flashinfer_moe_ep_mega_cutedsl and gain flashinfer_moe_ep_cutedsl
//     and aiter_mega_moe (vllm/config/kernel.py:122-146@5df63a72). MEGA_MOE_BACKENDS is now
//     NATIVE_MEGA_MOE_BACKENDS | FLASHINFER_MOE_EP_BACKENDS (:148-169):
//     deep_gemm_mega_moe, flashinfer_moe_ep_cutedsl and flashinfer_moe_ep_mega_deep_gemm.
//     aiter_mega_moe is not a member.
//   - DeepSeek-V4 and V4.1 decide sequence parallelism by the same predicate as before, now
//     spelled through a moe_needs_token_sharded_input variable
//     (vllm/models/deepseek_v41/nvidia/model.py:246-256@5df63a72;
//     vllm/models/deepseek_v4/nvidia/model.py:1047-1059@5df63a72, which folds dp > 1 into
//     the variable); the rule data is unchanged.
//   - The V4 shared expert reduces its own result only under the native mega-MoE backend
//     (reduce_results=self.use_native_mega_moe, deepseek_v4/nvidia/model.py:787@5df63a72),
//     where v0.31.0 said use_mega_moe. NATIVE_MEGA_MOE_BACKENDS is the one-member set
//     {deep_gemm_mega_moe}, which was v0.31.0's only native member, so no rule changes.
//   - The vocab-parallel embedding's tensor-parallel all-reduce is now behind
//     reduce_results and a parallel group
//     (vllm/model_executor/layers/vocab_parallel_embedding.py:558-563@5df63a72); it still
//     reduces under the default arguments the models here use.
//   - _get_kv_cache_bytes_per_block gained a layout argument and two branches
//     (vllm/v1/core/kv_cache_utils.py:1520-1551@5df63a72) that apply only to KV cache specs
//     whose layout is not block-outermost and to HiSparseHotSpec; the full-attention, MLA
//     and sliding-window specs these packs group keep v0.31.0's per-block bytes.
//   - Cascade attention is removed: disable_cascade_attn is a deprecated no-op
//     (vllm/config/model.py:312-314@5df63a72) and the runner hard-sets it off
//     (vllm/v1/worker/gpu_model_runner.py:531-533@5df63a72), so this pack has no
//     cascade-attention-is-opt-in rule and CascadeAttnOptIn is false.
//
// Everything else source_test.go checks is unchanged from v0.31.0 and passes against the
// v0.32.0 tree as transcribed there.
//
// Each rule states, in its Because field, what breaks without it. That is not
// decoration: a reader deciding whether to waive a rule needs to know what the rule
// is protecting, and a CI job that waives rules by name needs the list to be
// self-describing.
package v0_32

import (
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"
)

// Version is the engine release this pack describes.
const Version = "0.32.0"

func set[T comparable](vs ...T) map[T]bool {
	m := make(map[T]bool, len(vs))
	for _, v := range vs {
		m[v] = true
	}
	return m
}

// mtpMethods is the engine's MTPModelTypes literal, in declaration order. Every name but
// mtp is accepted and rewritten to mtp with a deprecation warning
// (vllm/config/speculative.py), so each is a name a launch line may carry.
var mtpMethods = []string{"deepseek_mtp", "dots3_note_mtp", "mimo_mtp", "mimo_v2_mtp",
	"glm4_moe_mtp", "glm4_moe_lite_mtp", "glm_ocr_mtp", "ernie_mtp", "nemotron_h_mtp",
	"exaone_moe_mtp", "exaone4_5_mtp", "qwen3_next_mtp", "qwen4_exp_mtp", "qwen3_5_mtp",
	"longcat_flash_mtp", "bailing_hybrid_v3_mtp", "minimax_m3_mtp", "bailing_hybrid_mtp",
	"mtp", "kimi_k3_mtp", "pangu_ultra_moe_mtp", "step3p5_mtp", "hy_v3_mtp", "hy_v4_mtp",
	"gemma4_mtp", "inkling_mtp", "glm5_next_mtp"}

// mlaPrefillBackends are the in-tree members of the engine's MLAPrefillBackendEnum
// (vllm/v1/attention/backends/mla/prefill/registry.py:37-65@5df63a72). attention_config parses
// mla_prefill_backend by looking the upper-cased name up in it and refuses one it lacks
// (vllm/config/attention.py:195-201@5df63a72). A plugin can add a member under a new name
// (registry.py:190), and vllm serve loads plugins while it builds its parser, before it parses
// the flag (vllm/engine/arg_utils.py:3202-3204@5df63a72), so a name outside this set may be a
// plugin's. CUSTOM is the member an out-of-tree backend can take instead.
var mlaPrefillBackends = []string{"FLASH_ATTN", "FLASHINFER", "TRTLLM_RAGGED", "TOKENSPEED_MLA",
	"ROCM_AITER_FA", "CPU", "ZEN_CPU", "CUSTOM"}

// pythonIdentifier reports whether name could be a Python identifier, which
// MLAPrefillBackendEnum.register requires of a backend a plugin adds
// (vllm/v1/attention/backends/mla/prefill/registry.py:85-86@5df63a72). It is str.isidentifier for
// ASCII: a letter or an underscore, then letters, digits or underscores. A name with any non-ASCII
// character is taken to be one, since Python admits Unicode letters this check does not classify;
// that errs toward the warning, never refusing a name vLLM could serve.
func pythonIdentifier(name string) bool {
	if name == "" {
		return false
	}
	for i, r := range name {
		switch {
		case r > 0x7f:
			return true
		case r == '_', r >= 'A' && r <= 'Z', r >= 'a' && r <= 'z':
		case r >= '0' && r <= '9' && i > 0:
		default:
			return false
		}
	}
	return true
}

// Pack returns the rules pack for this version. It is a function rather than a
// package-level variable so a caller can hold two versions at once — useful when
// comparing what a scenario would mean across releases.
func Pack() *rules.Pack {
	p := &rules.Pack{
		Version: Version,

		// Transcribed from the engine's All2AllBackend literal. Fourteen members:
		// v0.31.0's thirteen and passthrough. A name outside this set is priced with the
		// wrong volume basis.
		All2AllBackends: set("naive", "allgather_reducescatter", "pplx",
			"deepep_high_throughput", "deepep_low_latency", "deepep_v2",
			"flashinfer_all2allv", "flashinfer_nvlink_one_sided",
			"flashinfer_nvlink_two_sided", "mori_high_throughput",
			"mori_low_latency", "moonep", "nixl_ep", "passthrough"),
		// The subset for which the engine makes the MoE input sequence-parallel,
		// which replaces the layer's all-reduce with a reduce-scatter/all-gather
		// pair. passthrough joined it in this release; moonep is not a member.
		SPMoEBackends: set("allgather_reducescatter", "deepep_high_throughput",
			"deepep_low_latency", "deepep_v2", "flashinfer_nvlink_one_sided",
			"mori_high_throughput", "mori_low_latency", "nixl_ep", "passthrough"),

		// The engine exposes no allreduce-backend name; see v0_29. Unchanged.
		AllReduceBackends: set("custom", "nccl"),

		// Transcribed from the CUDAGraphMode enum, unchanged from v0.29.0.
		CUDAGraphModes: set("NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY",
			"FULL_AND_PIECEWISE"),

		// Transcribed from the CacheDType literal: nineteen members, v0.31.0's eighteen
		// and ultraquant_4bit.
		CacheDTypes: set("auto", "float16", "bfloat16", "fp8", "fp8_e4m3",
			"fp8_e5m2", "fp8_inc", "fp8_ds_mla", "nvfp4_ds_mla", "turboquant_k8v4",
			"turboquant_4bit_nc", "turboquant_k3v4_nc", "turboquant_3bit_nc",
			"ultraquant_4bit", "int4_per_token_head", "int8_per_token_head",
			"fp8_per_token_head", "nvfp4", "nvfp4_4over6"),

		DCPCommBackends: set("ag_rs", "a2a"),

		// The MambaDType literal: a recurrent state's cache dtype is checked against
		// this, not against CacheDTypes, which lacks float32.
		MambaDTypes:     set("auto", "float32", "float16", "bfloat16"),
		MambaCacheModes: set("none", "align"),

		// The MoEBackend literal of vllm/config/kernel.py, and its MEGA_MOE_BACKENDS
		// (NATIVE_MEGA_MOE_BACKENDS | FLASHINFER_MOE_EP_BACKENDS).
		MoEBackends: set("auto", "triton", "batched_triton", "deep_gemm", "deep_gemm_mega_moe", "cutlass",
			"flashinfer_trtllm", "flashinfer_cutlass", "flashinfer_cutedsl", "flashinfer_moe_ep_cutedsl",
			"flashinfer_b12x", "b12x", "flashinfer_moe_ep_mega_deep_gemm",
			"marlin", "humming", "triton_unfused", "aiter", "aiter_mega_moe", "aiter_triton_mxfp4_bf16",
			"flydsl", "hpc", "emulation", "rdna3"),
		MegaMoEBackends: set("deep_gemm_mega_moe", "flashinfer_moe_ep_cutedsl",
			"flashinfer_moe_ep_mega_deep_gemm"),

		// DeepSeek-V4.1's _use_sequence_parallel on CUDA (vllm/models/deepseek_v41/nvidia/
		// model.py; the ROCm module's is always False, see SequenceParallelCondition):
		// pipeline_parallel_size == 1 and enable_expert_parallel and tensor_parallel_size
		// > 1 and (use_mega_moe or data_parallel_size > 1).
		// DeepSeek-V4's is the same function (vllm/models/deepseek_v4/nvidia/model.py).
		ModelSequenceParallelRules: map[string]rules.SequenceParallelCondition{
			rules.ArchDeepseekV41: rules.SequenceParallelCondition{PipelineParallelOne: true, ExpertParallel: true,
				TensorParallelAboveOne: true, MegaMoEOrDataParallelAboveOne: true},
			rules.ArchDeepseekV4: rules.SequenceParallelCondition{PipelineParallelOne: true, ExpertParallel: true,
				TensorParallelAboveOne: true, MegaMoEOrDataParallelAboveOne: true},
		},

		// DeepSeek-V4.1's packed layout rewrites auto and fp8 to its backend's record
		// (vllm/models/deepseek_v41/attention.py): mega attention's nvfp4_ds_mla where it
		// serves the layout -- SM100, tp dividing the 8 output groups -- and FlashMLA's
		// fp8_ds_mla otherwise. SM120 runs a plain-row FlashInfer cache, which no
		// resolution here describes.
		//
		// TPDivides 8 is DeepSeek-V4.1-Flash's configuration, o_groups 8 over 64 heads:
		// mega attention needs tp to divide both and to leave 8 heads per group
		// (flash_mla_mega_attn.py is_available_for), which every width dividing 8 does
		// here. The pack keys this by architecture, and a checkpoint of the same
		// architecture with another o_groups or head count would need its own entry.
		CacheDTypeResolutions: map[string]rules.CacheDTypeResolutions{
			rules.ArchDeepseekV41: {
				{CapabilityFamily: 10, Requested: []string{"auto", "fp8"}, ResolvesTo: "nvfp4_ds_mla", TPDivides: 8},
				{CapabilityFamily: 10, Requested: []string{"auto", "fp8"}, ResolvesTo: "fp8_ds_mla"},
				{CapabilityFamily: 9, Requested: []string{"auto", "fp8"}, ResolvesTo: "fp8_ds_mla"},
			},
		},

		SchedulerPolicies: set("fcfs", "priority"),

		// The whole SpeculativeMethod literal of vllm/config/speculative.py with its
		// nested literals flattened: six named methods, the EAGLE family with its
		// twenty-seven multi-token-prediction names and dflash, ngram_gpu, and dspark.
		//
		// Membership is not the whole story for DeepSeek-V4.1: the engine refuses every
		// MTP-family method for it and serves dspark ("DeepSeek V4.1 has no classic-MTP
		// draft"), which the deepseek-v41-refuses-mtp rule enforces for a graph that
		// names its architecture.
		SpeculativeMethods: set(append([]string{"ngram", "medusa", "mlp_speculator",
			"draft_model", "suffix", "custom_class", "eagle", "eagle3",
			"extract_hidden_states", "dflash", "ngram_gpu", "dspark"},
			mtpMethods...)...),
		// Methods that leave async scheduling enabled: the EAGLE family (every MTP name
		// and dflash among them), ngram_gpu, draft_model and dspark -- the set
		// vllm/config/vllm.py admits when async scheduling is requested and keeps
		// enabled when it is left to the engine. Any other method disables it when it is
		// left to the engine, which moves the host term from hidden to exposed, and is
		// refused at startup when it is requested. Pipeline parallelism above one
		// also disables it on the V1 runner in this release, which is the resolver's
		// to report rather than a property of the method.
		AsyncCompatibleSpec: set(append([]string{"eagle", "eagle3",
			"extract_hidden_states", "dflash", "ngram_gpu", "draft_model", "dspark"},
			mtpMethods...)...),

		OffloadSpecs: set("CPUOffloadingSpec", "TieringOffloadingSpec"),

		// The QuantizationMethods literal, unchanged from v0.29.0.
		Quantizations: set("awq", "auto_awq", "fp8", "fbgemm_fp8", "fp_quant", "modelopt",
			"modelopt_fp4", "modelopt_mxfp8", "modelopt_mixed", "auto_gptq", "gptq",
			"gptq_marlin", "awq_marlin", "humming", "compressed-tensors", "experts_int8",
			"quark", "moe_wna16", "torchao", "inc", "mxfp4", "gpt_oss_mxfp4",
			"deepseek_v4_fp8", "online", "fp8_per_tensor", "fp8_per_block",
			"fp8_per_channel", "int8_per_channel_weight_only", "nvfp4_per_token", "mxfp8"),

		// The KVConnectorFactory registrations: v0.29.0's sixteen and HiSparseConnector.
		Connectors: set("DecodeBenchConnector", "ExampleConnector",
			"ExampleHiddenStatesConnector", "FlexKVConnectorV1", "HF3FSKVConnector",
			"HiSparseConnector", "LMCacheConnectorV1", "LMCacheMPConnector",
			"MoRIIOConnector", "MooncakeConnector", "MooncakeStoreConnector",
			"MultiConnector", "NixlConnector", "NixlPullConnector", "NixlPushConnector",
			"OffloadingConnector", "SimpleCPUOffloadConnector"),

		EvictionPolicies: set("lru", "arc"),

		CustomAllReduceWorldSizes: set(2, 4, 6, 8, 16),

		DefaultDBODecodeThreshold:  32,
		DefaultDBOPrefillThreshold: 512,

		TritonFetchPageBytes: 28 * 1024,
		TritonFetchSMs:       12,

		// CascadeAttnOptIn stays false: v0.32.0 removed cascade attention. disable_cascade_attn is
		// a deprecated no-op (vllm/config/model.py:312-314) and the runner hard-sets it off
		// (vllm/v1/worker/gpu_model_runner.py:531-533).

		// all2all_backend: All2AllBackend = "allgather_reducescatter" (vllm/config/parallel.py),
		// and the two accepted names it replaces with that one ("has been removed. Falling
		// back to 'allgather_reducescatter'").
		DefaultAll2AllBackend: "allgather_reducescatter",
		All2AllBackendRenames: map[string]string{
			"pplx": "allgather_reducescatter", "naive": "allgather_reducescatter"},

		// _NGRAM_LAYER_FIELDS (vllm/config/engram.py), and EngramConfig.cpu_offload: bool = True.
		EngramArchitectures: set(rules.ArchDeepseekV41, "Qwen4ExpForCausalLM",
			"Qwen4ExpForConditionalGeneration"),
		EngramCPUOffloadDefault: true,

		// get_kv_cache_groups (vllm/v1/core/kv_cache_utils.py:2300-2398): one uniform-type
		// group, else _get_packed_kv_cache_groups (:2001-2120), whose is_state_bucket takes
		// MambaSpec and CircularBufferSpec buckets, and a sliding-window bucket when no
		// repeat pattern is fixed; _get_kv_cache_bytes_per_block (:1606-1632) rounds the
		// widest group to the lcm of every block_stride_alignment.
		// The scheduler waives the cap when the request is the only running or waiting
		// one (vllm/v1/core/sched/scheduler.py:611-620).
		LongPrefillExemptsLoneRequest: true,

		KVCachePacking: rules.KVCachePacking{Modeled: true, StateCacheBuckets: true,
			LoneWindowIsState: true, BlockStrideAlignment: true, StateRingFromSpecTokens: true},
	}
	p.Rules = ruleList(p)
	return p
}

func ruleList(p *rules.Pack) []rules.Rule {
	return []rules.Rule{
		{
			Name:    "all2all-backend-known",
			Because: "an unrecognized backend name would be priced with the default volume basis, which differs by a factor of top_k; an accepted name the engine replaces (pplx, naive) is priced as what runs, which the warning names",
			Check: func(in rules.Input, out *rules.Problems) {
				forEachEngine(in, func(at string, e deployment.Engine) {
					if e.All2AllBackend == "" {
						return
					}
					if !p.All2AllBackends[e.All2AllBackend] {
						out.RuleErrorf("all2all-backend-known",
							"%s.engine.all2all_backend: %q is not accepted by engine %s",
							at, e.All2AllBackend, p.Version)
					} else if to, ok := p.All2AllBackendRenames[e.All2AllBackend]; ok {
						out.RuleWarnf("all2all-backend-known",
							"%s.engine.all2all_backend: engine %s accepts %q but runs %q in its place; price that one",
							at, p.Version, e.All2AllBackend, to)
					}
				})
			},
		},
		{
			Name:    "allreduce-backend-known",
			Because: "the backend decides whether a reduction consumes SMs or the NIC, so an unknown name leaves the cost on no resource",
			Check: func(in rules.Input, out *rules.Problems) {
				forEachEngine(in, func(at string, e deployment.Engine) {
					if e.AllReduceBackend == "" {
						return
					}
					if !p.AllReduceBackends[e.AllReduceBackend] {
						out.RuleErrorf("allreduce-backend-known",
							"%s.engine.allreduce_backend: %q is not accepted by engine %s",
							at, e.AllReduceBackend, p.Version)
					}
				})
			},
		},
		{
			Name:    "enum-values-known",
			Because: "graph mode, cache dtype, mamba cache dtype and mode, MoE backend and scheduler policy each change a cost term, and the engine types each as a closed literal, refusing any other value",
			Check: func(in rules.Input, out *rules.Problems) {
				forEachEngine(in, func(at string, e deployment.Engine) {
					check := func(field, value string, allowed map[string]bool) {
						if value == "" || allowed[value] {
							return
						}
						out.RuleErrorf("enum-values-known",
							"%s.engine.%s: %q is not accepted by engine %s",
							at, field, value, p.Version)
					}
					check("cudagraph_mode", e.CUDAGraphMode, p.CUDAGraphModes)
					check("cache_dtype", e.CacheDType, p.CacheDTypes)
					check("mamba_cache_dtype", e.MambaCacheDType, p.MambaDTypes)
					check("mamba_ssm_cache_dtype", e.MambaSSMCacheDType, p.MambaDTypes)
					check("moe_backend", rules.NormalizeMoEBackend(e.MoEBackend), p.MoEBackends)
					check("mamba_cache_mode", e.MambaCacheMode, p.MambaCacheModes)
					check("scheduling_policy", e.SchedulingPolicy, p.SchedulerPolicies)
				})
			},
		},
		{
			Name:    "speculative-method-known",
			Because: "an unknown draft method cannot be priced, and the draft pass of an MoE target is a second MoE rather than a scalar multiplier",
			Check: func(in rules.Input, out *rules.Problems) {
				forEachEngine(in, func(at string, e deployment.Engine) {
					if e.Speculative == nil {
						return
					}
					if !p.SpeculativeMethods[e.Speculative.Method] {
						out.RuleErrorf("speculative-method-known",
							"%s.engine.speculative.method: %q is not accepted by engine %s",
							at, e.Speculative.Method, p.Version)
					}
				})
				if in.Model != nil && in.Model.Speculator != nil {
					if !p.SpeculativeMethods[in.Model.Speculator.Method] {
						out.RuleErrorf("speculative-method-known",
							"model.speculator.method: %q is not accepted by engine %s",
							in.Model.Speculator.Method, p.Version)
					}
				}
			},
		},
		{
			Name:    "dcp-comm-backend-known",
			Because: "the engine types this setting as a closed literal and refuses any other name at startup",
			Check: func(in rules.Input, out *rules.Problems) {
				forEachEngine(in, func(at string, e deployment.Engine) {
					if e.DCPCommBackend != "" && !p.DCPCommBackends[e.DCPCommBackend] {
						out.RuleErrorf("dcp-comm-backend-known",
							"%s.engine.dcp_comm_backend %q is not one engine %s accepts (%s)",
							at, e.DCPCommBackend, Version, strings.Join(sortedKeys(p.DCPCommBackends), ", "))
					}
				})
			},
		},
		{
			Name: "mla-prefill-backend-known",
			Because: "attention_config parses mla_prefill_backend by looking the upper-cased name up in " +
				"MLAPrefillBackendEnum and refuses one it lacks at startup (vllm/config/attention.py:195-201@5df63a72); " +
				"a name a plugin could register, which vllm serve does before it parses the flag, is a warning, and " +
				"one no plugin can register, because MLAPrefillBackendEnum.register refuses a name that is not a " +
				"Python identifier (vllm/v1/attention/backends/mla/prefill/registry.py:85-86@5df63a72), is an error",
			Check: func(in rules.Input, out *rules.Problems) {
				known := set(mlaPrefillBackends...)
				forEachEngine(in, func(at string, e deployment.Engine) {
					name := strings.ToUpper(e.MLAPrefillBackend)
					switch {
					case e.MLAPrefillBackend == "" || known[name]:
					case !pythonIdentifier(name):
						out.RuleErrorf("mla-prefill-backend-known",
							"%s.engine.mla_prefill_backend %q is not a backend of engine %s (%s), and no plugin can "+
								"register it: a backend name must be a Python identifier, so vLLM refuses it at startup",
							at, e.MLAPrefillBackend, Version, strings.Join(sortedKeys(known), ", "))
					default:
						out.RuleWarnf("mla-prefill-backend-known",
							"%s.engine.mla_prefill_backend %q is not an in-tree backend of engine %s (%s); if a plugin "+
								"registers it this is expected, otherwise vLLM refuses it at startup: check the spelling",
							at, e.MLAPrefillBackend, Version, strings.Join(sortedKeys(known), ", "))
					}
				})
			},
		},
		{
			Name: "dcp-divides-tp",
			Because: "decode-context parallelism reuses the tensor-parallel ranks when prefill-context parallelism is off, " +
				"so the engine refuses a dcp that does not divide tp at startup; with prefill-context parallelism on, dcp " +
				"must be off, span the pcp axis, or span the whole tp x pcp block (vllm/config/parallel.py:590-599@5df63a72)",
			Check: func(in rules.Input, out *rules.Problems) {
				forEachPool(in, func(at string, pool deployment.Pool) {
					pl := pool.Parallel
					tp, pcp, dcp := pl.TP, max(pl.PCP, 1), max(pl.DCP, 1)
					if tp < 1 || pl.PCP < 0 || pl.DCP < 0 {
						return // a malformed width is the field layer's to report
					}
					switch {
					case pcp == 1 && tp%dcp != 0:
						out.RuleErrorf("dcp-divides-tp",
							"%s.parallel: tp %d must be divisible by dcp %d; engine %s refuses it at startup", at, tp, dcp, Version)
					case pcp > 1 && dcp != 1 && dcp != pcp && dcp != tp*pcp:
						out.RuleErrorf("dcp-divides-tp",
							"%s.parallel: with pcp %d, dcp %d must be 1, %d or %d; engine %s refuses it at startup",
							at, pcp, dcp, pcp, tp*pcp, Version)
					}
				})
			},
		},
		{
			Name:    "cp-kv-cache-interleave-fits-block",
			Because: "with decode-context parallelism on and no NixlConnector configured, this release asserts at startup that block_size is at least, and divisible by, cp_kv_cache_interleave_size (vllm/config/vllm.py validate_block_size), so a layout breaking it does not start; unlike v0.29.0, any other connector leaves the assert in force. The block size it checks is the smallest prefix-cacheable KV cache group's, which can be a model's own window-cache page size rather than the stated one",
			Check: func(in rules.Input, out *rules.Problems) {
				// Only where the assert is certain to run on the values stated:
				//   - a NixlConnector, directly or as a MultiConnector child, skips the
				//     assert, and the schema cannot see a MultiConnector's children, so
				//     either skips the check. Any other connector does not: this release
				//     pins the size to the block size only under NIXL;
				//   - a hybrid model may have its block size re-aligned by the platform,
				//     and an unknown model may be hybrid, so the stated size is final only
				//     for a known model with no recurrent layer;
				//   - an unstated block size is chosen by the platform.
				if in.Deployment == nil || mayUseNixl(in.Deployment) || in.Model == nil ||
					modelHasRecurrent(in) {
					return
				}
				forEachPool(in, func(at string, pool deployment.Pool) {
					e := pool.Engine
					size, stated := e.CPKVCacheInterleaveSize, e.BlockSize
					if pool.Parallel.DCP <= 1 || size < 1 || stated < 1 {
						return
					}
					// The assert reads the block size after the engine has replaced the
					// stated one with the smallest KV cache group's (vllm/v1/engine/core.py),
					// which a cache with its own page size lowers. One test covers both
					// halves of the assert: a size above the block leaves the block itself
					// as the remainder.
					blocks := effectiveBlocks(in.Model, stated, pool.Parallel.PCP)
					failing := 0
					for _, b := range blocks {
						if b%size != 0 {
							failing++
						}
					}
					switch {
					case failing == len(blocks):
						checked := fmt.Sprintf("block_size %d", blocks[0])
						if blocks[0] != stated {
							checked = fmt.Sprintf("block_size %d (the engine checks %d, not the stated %d)", blocks[0], blocks[0], stated)
						}
						out.RuleErrorf("cp-kv-cache-interleave-fits-block",
							"%s.engine: %s must be at least, and divisible by, cp_kv_cache_interleave_size %d when dcp is %d; engine %s refuses it at startup",
							at, checked, size, pool.Parallel.DCP, Version)
					case failing > 0:
						out.RuleWarnf("cp-kv-cache-interleave-fits-block",
							"%s.engine: cp_kv_cache_interleave_size %d fits some of the block sizes the engine may check, %v, and not others; which applies turns on whether the window cache takes part in prefix caching, so engine %s may refuse it at startup",
							at, size, blocks, Version)
					}
				})
			},
		},
		{
			Name:    "expert-divisibility-under-eplb",
			Because: "with load balancing on, the engine asserts that the expert-parallel width divides the physical expert count, so an indivisible layout does not start",
			Check: func(in rules.Input, out *rules.Problems) {
				if in.Model == nil {
					return
				}
				experts := totalExperts(in)
				if experts == 0 {
					return
				}
				forEachPool(in, func(at string, pool deployment.Pool) {
					e := pool.Engine
					if e.EPLB == nil || !e.EPLB.Enabled {
						return
					}
					ep := pool.Parallel.ExpertParallelWidth()
					if ep < 2 {
						return
					}
					physical := experts + e.EPLB.NumRedundantExperts
					if physical%ep != 0 {
						need := (ep - physical%ep) % ep
						out.RuleErrorf("expert-divisibility-under-eplb",
							"%s: %d experts plus %d redundant is %d, which expert-parallel width %d does not divide; %d more redundant experts would make it divide",
							at, experts, e.EPLB.NumRedundantExperts, physical, ep, need)
					}
				})
			},
		},
		{
			Name:    "expert-imbalance-without-eplb",
			Because: "without load balancing an indivisible split is legal but uneven, and the resulting static imbalance is a step-time term rather than an error; a width larger than the expert count leaves ranks with no experts, which is an error",
			Check: func(in rules.Input, out *rules.Problems) {
				if in.Model == nil {
					return
				}
				experts := totalExperts(in)
				if experts == 0 {
					return
				}
				forEachPool(in, func(at string, pool deployment.Pool) {
					e := pool.Engine
					if e.EPLB != nil && e.EPLB.Enabled {
						return
					}
					ep := pool.Parallel.ExpertParallelWidth()
					if ep < 2 || experts%ep == 0 {
						return
					}
					base := experts / ep
					if base == 0 {
						out.RuleErrorf("expert-imbalance-without-eplb",
							"%s: expert-parallel width %d exceeds the model's %d experts, leaving ranks with none",
							at, ep, experts)
						return
					}
					out.RuleWarnf("expert-imbalance-without-eplb",
						"%s: %d experts over width %d leaves %d ranks with %d experts and the rest with %d, a %.2fx static imbalance the estimate must carry",
						at, experts, ep, experts%ep, base+1, base,
						float64(base+1)/float64(base))
				})
			},
		},
		{
			Name:    "custom-allreduce-reachable",
			Because: "the SM-consuming kernel supports only certain rank counts and only within one node absent multi-node NVLink, so a request outside those falls back to the NIC",
			Check: func(in rules.Input, out *rules.Problems) {
				gpusPerNode := 0
				if in.Scenario != nil {
					gpusPerNode = in.Scenario.Cluster.GPUsPerNode
				}
				forEachPool(in, func(at string, pool deployment.Pool) {
					if pool.Engine.AllReduceBackend != "custom" {
						return
					}
					tp := pool.Parallel.TP
					if !p.CustomAllReduceWorldSizes[tp] {
						out.RuleWarnf("custom-allreduce-reachable",
							"%s: tp %d is not a supported world size for the custom all-reduce, so the engine will fall back to NCCL and the cost moves from SMs to the NIC",
							at, tp)
					}
					if gpusPerNode > 0 && tp > gpusPerNode {
						out.RuleWarnf("custom-allreduce-reachable",
							"%s: tp %d spans more than the %d GPUs in a node, so the custom all-reduce needs multi-node NVLink on every rank or it falls back to NCCL",
							at, tp, gpusPerNode)
					}
				})
			},
		},
		{
			Name:    "offload-spec-known",
			Because: "the spec decides whether a fetch consumes SMs or a copy engine, which is the only way the offload path enters step time",
			Check: func(in rules.Input, out *rules.Problems) {
				if in.Deployment == nil || in.Deployment.Offload == nil {
					return
				}
				s := in.Deployment.Offload.Spec
				if s != "" && !p.OffloadSpecs[s] {
					out.RuleErrorf("offload-spec-known",
						"offload.spec: %q is not a registered spec in engine %s",
						s, p.Version)
				}
			},
		},
		{
			Name:    "quantization-known",
			Because: "the served weight format sets weight-read bytes, compute peak and the GEMM efficiency envelope at once, so a misspelled in-tree method is priced against three wrong constants; it is a warning because the engine admits out-of-tree methods this set cannot enumerate",
			Check: func(in rules.Input, out *rules.Problems) {
				forEachEngine(in, func(at string, e deployment.Engine) {
					if e.Quantization == "" || p.Quantizations[e.Quantization] {
						return
					}
					out.RuleWarnf("quantization-known",
						"%s.engine.quantization: %q is not an in-tree method of engine %s; if it is an out-of-tree method this is expected, otherwise check the spelling. In-tree methods are %v",
						at, e.Quantization, p.Version, sortedKeys(p.Quantizations))
				})
			},
		},
		{
			Name:    "connector-known",
			Because: "the connector names the offload/PD-transfer implementation whose spec decides whether a fetch consumes SMs or a copy engine, so an unknown name leaves the transfer on no modeled resource; a warning because the KV connector registry is extensible out of tree",
			Check: func(in rules.Input, out *rules.Problems) {
				if in.Deployment == nil {
					return
				}
				check := func(field, value string) {
					if value == "" || p.Connectors[value] {
						return
					}
					out.RuleWarnf("connector-known",
						"%s: %q is not a connector registered in engine %s; if it is an out-of-tree connector this is expected, otherwise check the spelling. Registered connectors are %v",
						field, value, p.Version, sortedKeys(p.Connectors))
				}
				if in.Deployment.Offload != nil {
					check("offload.connector", in.Deployment.Offload.Connector)
				}
				if in.Deployment.PDTransfer != nil {
					check("pd_transfer.connector", in.Deployment.PDTransfer.Connector)
				}
			},
		},
		{
			Name:    "eviction-policy-known",
			Because: "a supplied name the engine does not recognize does not silently fall back — CachePolicyFactory.get_cache_policy_cls raises ValueError at launch unless cache_policy_module_path names an out-of-tree policy; this rule catches that typo earlier, as a warning, while still permitting a real out-of-tree name the set cannot enumerate",
			Check: func(in rules.Input, out *rules.Problems) {
				if in.Deployment == nil || in.Deployment.Offload == nil {
					return
				}
				ep := in.Deployment.Offload.EvictionPolicy
				if ep == "" || p.EvictionPolicies[ep] {
					return
				}
				out.RuleWarnf("eviction-policy-known",
					"offload.eviction_policy: %q is not an in-tree policy of engine %s; if it is an out-of-tree policy this is expected, otherwise check the spelling. In-tree policies are %v",
					ep, p.Version, sortedKeys(p.EvictionPolicies))
			},
		},
		{
			Name:    "mamba-cache-mode-matches-model",
			Because: "a cache mode for recurrent layers on a model with none is a setting with no effect, and its absence on a model with them leaves the state unsized",
			Check: func(in rules.Input, out *rules.Problems) {
				if in.Model == nil {
					return
				}
				hasRecurrent := modelHasRecurrent(in)
				forEachEngine(in, func(at string, e deployment.Engine) {
					if e.MambaCacheMode != "" && !hasRecurrent {
						out.RuleWarnf("mamba-cache-mode-matches-model",
							"%s.engine.mamba_cache_mode is set but the model has no recurrent layer, so it has no effect", at)
					}
					if e.MambaCacheMode == "" && hasRecurrent {
						out.RuleWarnf("mamba-cache-mode-matches-model",
							"%s.engine: the model has recurrent layers but no mamba_cache_mode is stated, so whether their state is fixed or context-proportional is unresolved", at)
					}
				})
			},
		},
		{
			Name:    "dbo-thresholds-stated-when-enabled",
			Because: "which threshold applies turns on batch uniformity, and an unstated threshold silently takes a default that differs by 16x between the two",
			Check: func(in rules.Input, out *rules.Problems) {
				forEachEngine(in, func(at string, e deployment.Engine) {
					if e.DBO == nil || !e.DBO.Enabled {
						return
					}
					if e.DBO.DecodeTokenThreshold == nil && e.DBO.PrefillTokenThreshold == nil {
						out.RuleWarnf("dbo-thresholds-stated-when-enabled",
							"%s.engine.dbo: enabled with no thresholds stated; engine %s defaults to %d for a uniform-decode batch and %d otherwise",
							at, p.Version, p.DefaultDBODecodeThreshold,
							p.DefaultDBOPrefillThreshold)
					}
				})
			},
		},
		{
			Name:    "dbo-thresholds-at-least-microbatches",
			Because: "with dual-batch overlap on, this release refuses a decode or prefill token threshold below its two microbatches at startup (vllm/config/parallel.py: \"must be at least the number of microbatches\")",
			Check: func(in rules.Input, out *rules.Problems) {
				const microbatches = 2 // num_ubatches is 2 whenever enable_dbo is set
				forEachEngine(in, func(at string, e deployment.Engine) {
					if e.DBO == nil || !e.DBO.Enabled {
						return
					}
					// An unstated threshold takes the default, which passes; a stated 0 does
					// not. Negative values are a field error, reported before rules run.
					for _, f := range []struct {
						name string
						v    *int
					}{
						{"dbo_decode_token_threshold", e.DBO.DecodeTokenThreshold},
						{"dbo_prefill_token_threshold", e.DBO.PrefillTokenThreshold},
					} {
						if f.v != nil && *f.v < microbatches {
							out.RuleErrorf("dbo-thresholds-at-least-microbatches",
								"%s.engine.dbo.%s: %d is below the %d microbatches dual-batch overlap splits into; engine %s refuses it at startup",
								at, f.name, *f.v, microbatches, p.Version)
						}
					}
				})
			},
		},
		{
			Name:    "engram-excludes-dbo",
			Because: "this release refuses dual-batch overlap and microbatching at startup for a DeepseekV41ForCausalLM with Engram layers (vllm/config/vllm.py: \"DeepSeek V4.1 Engram does not support DBO or microbatching\"); a graph with Engram lookups that names no architecture draws a warning, since the refusal turns on the architecture",
			Check: func(in rules.Input, out *rules.Problems) {
				if !modelHasEngram(in) {
					return
				}
				arch := in.Model.Global.Architecture
				if arch != "" && arch != rules.ArchDeepseekV41 {
					return
				}
				forEachEngine(in, func(at string, e deployment.Engine) {
					if e.DBO == nil || !e.DBO.Enabled {
						return
					}
					if arch == rules.ArchDeepseekV41 {
						out.RuleErrorf("engram-excludes-dbo",
							"%s.engine.dbo: enabled on a %s with Engram lookups; engine %s refuses dual-batch overlap with Engram",
							at, arch, p.Version)
						return
					}
					out.RuleWarnf("engram-excludes-dbo",
						"%s.engine.dbo: enabled on a model with Engram lookups that names no architecture; engine %s refuses this for %s",
						at, p.Version, rules.ArchDeepseekV41)
				})
			},
		},
		{
			Name:    "engram-config-valid",
			Because: "this release refuses an Engram config at startup on a model without Engram layers or of an architecture outside _NGRAM_LAYER_FIELDS, dp_shared_memory without cpu_offload or with one data-parallel rank, and use_thp without cpu_offload or with dp_shared_memory (vllm/config/engram.py); a graph with Engram lookups that names no architecture draws a warning, since the refusal turns on the architecture",
			Check: func(in rules.Input, out *rules.Problems) {
				forEachPool(in, func(at string, pool deployment.Pool) {
					ec := pool.Engine.EngramConfig
					if ec == nil {
						return
					}
					if in.Model != nil {
						arch := in.Model.Global.Architecture
						switch {
						case !modelHasEngram(in):
							out.RuleErrorf("engram-config-valid",
								"%s.engine.engram_config: the model has no Engram lookups; engine %s refuses an Engram config for it",
								at, p.Version)
						case arch != "" && !p.EngramArchitectures[arch]:
							out.RuleErrorf("engram-config-valid",
								"%s.engine.engram_config: engine %s accepts an Engram config only for %v, not %s",
								at, p.Version, sortedKeys(p.EngramArchitectures), arch)
						case arch == "":
							out.RuleWarnf("engram-config-valid",
								"%s.engine.engram_config: the model names no architecture; engine %s accepts an Engram config only for %v",
								at, p.Version, sortedKeys(p.EngramArchitectures))
						}
					}
					offloadOff := ec.CPUOffload != nil && !*ec.CPUOffload
					shared := ec.DPSharedMemory != nil && *ec.DPSharedMemory
					if ec.UseTHP != nil && *ec.UseTHP && (offloadOff || shared) {
						out.RuleErrorf("engram-config-valid",
							"%s.engine.engram_config: use_thp requires cpu_offload and not dp_shared_memory; engine %s refuses it", at, p.Version)
					}
					if !shared {
						return
					}
					if offloadOff {
						out.RuleErrorf("engram-config-valid",
							"%s.engine.engram_config: dp_shared_memory requires cpu_offload; engine %s refuses it", at, p.Version)
					}
					if pool.Parallel.DP <= 1 {
						out.RuleErrorf("engram-config-valid",
							"%s.engine.engram_config: dp_shared_memory requires dp above 1, got %d; engine %s refuses it",
							at, pool.Parallel.DP, p.Version)
					}
				})
			},
		},
		{
			Name:    "cache-record-dtypes-known",
			Because: "a cache's per-dtype record is looked up by the cache dtype the engine runs, so an entry keyed by a name this release does not accept, or by auto, which always resolves to a concrete dtype, can never be read, and a misspelled nvfp4_ds_mla would leave its cache priced at the default record",
			Check: func(in rules.Input, out *rules.Problems) {
				if in.Model == nil {
					return
				}
				check := func(at string, records map[string]int) {
					for _, dt := range sortedIntKeys(records) {
						if dt == "auto" || !p.CacheDTypes[dt] {
							out.RuleErrorf("cache-record-dtypes-known",
								"model %s.bytes_by_cache_dtype: %q is not a cache dtype engine %s accepts",
								at, dt, p.Version)
						}
					}
				}
				in.Model.EachNode(func(at string, n model.Node) {
					for i, c := range n.Caches {
						check(fmt.Sprintf("%s.caches[%d]", at, i), c.BytesByCacheDType)
					}
				})
				for i, o := range in.Model.ArchOverrides {
					for j, c := range o.Caches {
						check(fmt.Sprintf("arch_overrides[%d].caches[%d]", i, j), c.BytesByCacheDType)
					}
				}
			},
		},
		{
			Name:    "engram-residency-default",
			Because: "a graph states where an Engram table lives when the deployment leaves cpu_offload alone, and this release's default is cpu_offload true, host memory (vllm/config/engram.py); a graph stating the other residency would price every such deployment against the wrong memory",
			Check: func(in rules.Input, out *rules.Problems) {
				if in.Model == nil {
					return
				}
				want := model.ResidencyDevice
				if p.EngramCPUOffloadDefault {
					want = model.ResidencyHost
				}
				in.Model.EachNode(func(at string, n model.Node) {
					if n.Op != model.OpGather || n.ResidencyControl != model.ResidencyEngramCPUOffload {
						return
					}
					if got := n.Residency.Effective(); got != want {
						out.RuleErrorf("engram-residency-default",
							"model %s.residency: %s, but under engine %s an Engram table defaults to %s",
							at, got, p.Version, want)
					}
				})
			},
		},
		{
			Name:    "deepseek-v41-refuses-mtp",
			Because: "this release refuses a multi-token-prediction draft for DeepSeek-V4.1 at startup: it rewrites every MTP-family method name to mtp, rewrites the V4.1 draft config to deepseek_mtp, and then raises \"DeepSeek V4.1 has no classic-MTP draft\" (vllm/config/speculative.py); dspark is the method it serves. A launch escapes only where the draft model's path (containing eagle-, eagle3, dflash or dspark) or its own configuration (a DSpark, DFlash, Medusa or MLP-speculator draft) selects another method first, which a deployment does not record",
			Check: func(in rules.Input, out *rules.Problems) {
				if in.Model == nil || in.Model.Global.Architecture != rules.ArchDeepseekV41 {
					return
				}
				isMTP := map[string]bool{}
				for _, m := range mtpMethods {
					isMTP[m] = true
				}
				forEachEngine(in, func(at string, e deployment.Engine) {
					if e.Speculative != nil && isMTP[e.Speculative.Method] {
						out.RuleErrorf("deepseek-v41-refuses-mtp",
							"%s.engine.speculative.method: %q on %s; engine %s refuses an MTP draft for it and serves dspark",
							at, e.Speculative.Method, rules.ArchDeepseekV41, p.Version)
					}
				})
				if s := in.Model.Speculator; s != nil && isMTP[s.Method] {
					out.RuleErrorf("deepseek-v41-refuses-mtp",
						"model.speculator.method: %q on %s; engine %s refuses an MTP draft for it and serves dspark",
						s.Method, rules.ArchDeepseekV41, p.Version)
				}
			},
		},
		{
			Name:    "max-model-len-within-context-window",
			Because: "this release refuses a stated max_model_len above the window it derives from the model configuration, unless the configuration's model_max_length admits it or VLLM_ALLOW_LONG_MAX_MODEL_LEN is set (vllm/config/model.py); a warning because the schema sees neither",
			Check: func(in rules.Input, out *rules.Problems) {
				if in.Model == nil || in.Model.Global.ContextWindow < 1 {
					return
				}
				w := in.Model.Global.ContextWindow
				forEachEngine(in, func(at string, e deployment.Engine) {
					if n, ok := e.MaxModelLen.Stated(); ok && n > w {
						out.RuleWarnf("max-model-len-within-context-window",
							"%s.engine.max_model_len %d exceeds the model's %d-token window; engine %s refuses it at startup unless model_max_length admits it or VLLM_ALLOW_LONG_MAX_MODEL_LEN is set",
							at, n, w, p.Version)
					}
				})
			},
		},
		{
			Name:    "sequence-parallel-architecture-known",
			Because: "a graph's sequence_parallel emit conditions and under_sequence_parallel splits are decided by its architecture's own code, so on an architecture this release has no such rule for they never apply and the graph prices tensor parallelism alone",
			Check: func(in rules.Input, out *rules.Problems) {
				if in.Model == nil {
					return
				}
				if _, ok := p.ModelSequenceParallelRules[in.Model.Global.Architecture]; ok {
					return
				}
				reported := false
				in.Model.EachNode(func(at string, n model.Node) {
					field := n.SequenceParallelField()
					if reported || field == "" {
						return
					}
					reported = true
					value := string(n.Emit)
					if field != "emit" {
						value = string(n.UnderSequenceParallel)
					}
					out.RuleWarnf("sequence-parallel-architecture-known",
						"model %s.%s: %q, but engine %s has no sequence-parallelism rule for architecture %q; architectures with one are %v (their CUDA path; on ROCm it is always off)",
						at, field, value, p.Version, in.Model.Global.Architecture, ruleArchitectures(p))
				})
			},
		},
		{
			Name:    "model-sequence-parallel-collectives",
			Because: "an architecture that decides its own sequence parallelism never consults the engine-wide sequence-parallel-MoE rule -- DeepSeek-V4.1 passes its own _use_sequence_parallel to its MoE and shared expert, and FusedMoE then skips the final tensor-parallel all-reduce (vllm/model_executor/layers/fused_moe/runner/moe_runner.py, \"Skipped when sequence-parallel is active\") -- so a decoder-layer reduction conditioned on that rule fires or not on the wrong layouts; and a plain tensor-parallel all-reduce inside a decoder layer is replaced under the model's sequence parallelism, which shards the residual stream by token",
			Check: func(in rules.Input, out *rules.Problems) {
				if in.Model == nil {
					return
				}
				arch := in.Model.Global.Architecture
				if _, ok := p.ModelSequenceParallelRules[arch]; !ok {
					return
				}
				// Only decoder layers run on the sharded stream: the embedding reduces before
				// the stream is sharded and the head after it is gathered back.
				for i, lk := range in.Model.LayerKinds {
					for j, n := range lk.Nodes {
						at := fmt.Sprintf("model layer_kinds[%d].nodes[%d]", i, j)
						switch {
						case n.Emit == model.EmitTensorParallelUnlessSequenceParallelMoE:
							out.RuleErrorf("model-sequence-parallel-collectives",
								"%s.emit: %q, but %s decides sequence parallelism itself and never consults that rule in engine %s; use %q (on ROCm its sequence parallelism is always off)",
								at, n.Emit, arch, p.Version, model.EmitTensorParallelUnlessSequenceParallel)
						case n.Op == model.OpAllReduce && n.Emit == model.EmitTensorParallel:
							out.RuleWarnf("model-sequence-parallel-collectives",
								"%s: a tensor-parallel all-reduce in a decoder layer of %s, which engine %s replaces under the model's sequence parallelism; emit it %q, with its replacement under %q",
								at, arch, p.Version, model.EmitTensorParallelUnlessSequenceParallel, model.EmitSequenceParallel)
						}
					}
				}
			},
		},
		{
			Name:    "moe-ep-backend-binds-passthrough",
			Because: "this release binds all2all_backend passthrough (no framework all-to-all) to the MoE backends that dispatch and combine themselves, the FlashInfer MoE-EP megakernels, and refuses at startup any other explicit all2all_backend beside one and passthrough beside any other MoE backend (vllm/config/kernel.py:194-227@5df63a72), so such a pool does not run; left at the default, the backend is rebound to passthrough, which prices no all-to-all",
			Check: func(in rules.Input, out *rules.Problems) {
				forEachEngine(in, func(at string, e deployment.Engine) {
					moe := rules.NormalizeMoEBackend(e.MoEBackend)
					selfCommunicating := moe == "flashinfer_moe_ep_cutedsl" || moe == "flashinfer_moe_ep_mega_deep_gemm"
					// The binding reads the backend after ParallelConfig has renamed the removed
					// pplx and naive to allgather_reducescatter (vllm/config/parallel.py:532-538),
					// so those start beside an EP megakernel, as the default does.
					a2a := p.EffectiveAll2AllBackend(e.All2AllBackend)
					switch {
					case selfCommunicating && a2a != "passthrough" && a2a != p.DefaultAll2AllBackend:
						out.RuleErrorf("moe-ep-backend-binds-passthrough",
							"%s.engine: moe_backend %q dispatches and combines itself; engine %s refuses it beside all2all_backend %q (leave all2all_backend unset)",
							at, moe, p.Version, e.All2AllBackend)
					case !selfCommunicating && e.All2AllBackend == "passthrough":
						out.RuleErrorf("moe-ep-backend-binds-passthrough",
							"%s.engine: all2all_backend passthrough is valid on engine %s only beside a MoE backend that dispatches and combines itself (flashinfer_moe_ep_cutedsl, flashinfer_moe_ep_mega_deep_gemm), not moe_backend %q",
							at, p.Version, moe)
					}
				})
			},
		},
		{
			Name:    "moonep-requirements",
			Because: "this release refuses the moonep all2all backend at startup for a quantized model, a model whose runtime dtype is not BF16, and with EPLB (vllm/config/vllm.py), so such a pool does not run; an FP16 checkpoint is a warning, since --dtype bfloat16 casts it and a deployment does not record the dtype; the refusal of a non-linear expert placement needs no check, because a deployment cannot state one and the default is linear",
			Check: func(in rules.Input, out *rules.Problems) {
				forEachEngine(in, func(at string, e deployment.Engine) {
					if e.All2AllBackend != "moonep" {
						return
					}
					// The engine checks the runtime dtype, and --dtype bfloat16 casts an FP16
					// checkpoint (vllm/config/model.py: "Casting between float16 and
					// bfloat16 is allowed"), which the deployment does not record.
					if in.Model != nil && in.Model.Global.WeightDType == model.DTypeFP16 {
						out.RuleWarnf("moonep-requirements",
							"%s.engine: all2all_backend moonep on an FP16 checkpoint; engine %s supports moonep for BF16 models only, so it starts only if served with --dtype bfloat16",
							at, p.Version)
					}
					if e.Quantization != "" {
						out.RuleErrorf("moonep-requirements",
							"%s.engine: all2all_backend moonep with quantization %q; engine %s supports moonep for unquantized BF16 models only",
							at, e.Quantization, p.Version)
					} else if dt, ok := modelQuantizedDType(in); ok {
						out.RuleErrorf("moonep-requirements",
							"%s.engine: all2all_backend moonep on a checkpoint stored in %s; engine %s supports moonep for unquantized BF16 models only",
							at, dt, p.Version)
					}
					if e.EPLB != nil && e.EPLB.Enabled {
						out.RuleErrorf("moonep-requirements",
							"%s.engine: all2all_backend moonep with EPLB enabled; engine %s does not support EPLB on moonep",
							at, p.Version)
					}
				})
			},
		},
		{
			Name:    "sequence-parallel-moe-implied",
			Because: "on this backend with tensor and data parallelism both above one, the engine makes the MoE input sequence-parallel, which replaces the layer's all-reduce with a different pair of collectives",
			Check: func(in rules.Input, out *rules.Problems) {
				// A model that decides its own sequence parallelism never consults the
				// engine-wide rule, so the rule says nothing about it.
				if in.Model != nil {
					if _, own := p.ModelSequenceParallelRules[in.Model.Global.Architecture]; own {
						return
					}
				}
				forEachPool(in, func(at string, pool deployment.Pool) {
					pl := pool.Parallel
					if !p.SequenceParallelMoE(pool.Engine.All2AllBackend, pl.EnableExpertParallel, pl.TP, pl.DP) {
						return
					}
					backend := p.EffectiveAll2AllBackend(pool.Engine.All2AllBackend)
					out.RuleWarnf("sequence-parallel-moe-implied",
						"%s: engine %s makes the MoE input sequence-parallel here (backend %q, tp %d, dp %d), so the model graph must emit a reduce-scatter and all-gather pair rather than an all-reduce",
						at, p.Version, backend, pl.TP, pl.DP)
				})
			},
		},
		{
			Name:    "data-parallel-load-balancing",
			Because: "this release refuses external data-parallel load balancing for a model without experts and hybrid without data_parallel_size_local, switches hybrid with one rank per node to external, and runs hybrid with every rank on one node as internal (vllm/engine/arg_utils.py create_engine_config); the mode decides which endpoints a router can choose between",
			Check: func(in rules.Input, out *rules.Problems) {
				forEachPool(in, func(at string, pool deployment.Pool) {
					pl := pool.Parallel
					stated := pl.DataParallelLoadBalancing.Mode()
					if stated == deployment.DPLBInternal || pl.DP <= 1 {
						return
					}
					if stated == deployment.DPLBHybrid && pl.DPLocal < 1 {
						out.RuleErrorf("data-parallel-load-balancing",
							"%s.parallel: hybrid load balancing needs dp_local, the ranks on each node; engine %s refuses it without --data-parallel-size-local",
							at, p.Version)
						return
					}
					runs := p.ResolveDPLoadBalancing(pl)
					if runs != stated {
						out.RuleWarnf("data-parallel-load-balancing",
							"%s.parallel: hybrid load balancing with dp_local %d of dp %d runs as %s on engine %s",
							at, pl.DPLocal, pl.DP, runs, p.Version)
					}
					if runs == deployment.DPLBExternal && in.Model != nil && totalExperts(in) == 0 {
						out.RuleErrorf("data-parallel-load-balancing",
							"%s.parallel: external data-parallel load balancing on a model without experts; engine %s refuses it and asks for independent instances instead",
							at, p.Version)
					}
				})
			},
		},
		{
			Name:    "long-prefill-threshold-within-max-model-len",
			Because: "this release refuses at startup a long_prefill_token_threshold greater than max_model_len (vllm/config/scheduler.py verify_max_model_len)",
			Check: func(in rules.Input, out *rules.Problems) {
				forEachEngine(in, func(at string, e deployment.Engine) {
					n, ok := e.MaxModelLen.Stated()
					if ok && e.LongPrefillTokenThreshold > n {
						out.RuleErrorf("long-prefill-threshold-within-max-model-len",
							"%s.engine: long_prefill_token_threshold %d exceeds max_model_len %d; engine %s refuses it at startup",
							at, e.LongPrefillTokenThreshold, n, p.Version)
					}
				})
			},
		},
		{
			Name:    "cudagraph-capture-sizes",
			Because: "this release refuses a stated max_cudagraph_capture_size that is not the largest of the stated cudagraph_capture_sizes it keeps, caps the maximum at max_num_batched_tokens, drops sizes above it, and truncates a maximum off its capture stride to the largest size it generates (vllm/config/vllm.py _set_cudagraph_sizes); a step larger than the size that runs is executed eager, so the size that runs is the one to price",
			Check: func(in rules.Input, out *rules.Problems) {
				forEachEngine(in, func(at string, e deployment.Engine) {
					stated := e.MaxCUDAGraphCaptureSize != nil || e.CUDAGraphCaptureSizes != nil
					if !stated {
						return
					}
					if e.CUDAGraphMode == "NONE" {
						out.RuleWarnf("cudagraph-capture-sizes",
							"%s.engine: cudagraph_mode NONE captures no graph, so engine %s ignores the stated capture sizes",
							at, p.Version)
						return
					}
					c, _, err := p.CUDAGraphCaptureSizes(rules.CUDAGraphCaptureInput{Mode: e.CUDAGraphMode,
						MaxCaptureSize: e.MaxCUDAGraphCaptureSize, CaptureSizes: e.CUDAGraphCaptureSizes,
						MaxNumBatchedTokens: e.MaxNumBatchedTokens})
					switch {
					case errors.Is(err, rules.ErrCUDAGraphCaptureInconsistent):
						out.RuleErrorf("cudagraph-capture-sizes",
							"%s.engine: max_cudagraph_capture_size %d is not the largest of the cudagraph_capture_sizes engine %s keeps (%d); it refuses the pair at startup",
							at, *e.MaxCUDAGraphCaptureSize, p.Version, c.Max)
						return
					case err != nil:
						// max_num_batched_tokens is unstated, so the cap cannot be applied here.
						return
					}
					if c.Capped {
						out.RuleWarnf("cudagraph-capture-sizes",
							"%s.engine: max_cudagraph_capture_size %d exceeds max_num_batched_tokens %d; engine %s captures up to %d",
							at, *e.MaxCUDAGraphCaptureSize, e.MaxNumBatchedTokens, p.Version, c.Max)
					}
					if c.Truncated {
						out.RuleWarnf("cudagraph-capture-sizes",
							"%s.engine: max_cudagraph_capture_size %d is off the capture stride; engine %s truncates it to %d",
							at, *e.MaxCUDAGraphCaptureSize, p.Version, c.Max)
					}
					if len(c.Dropped) > 0 {
						out.RuleWarnf("cudagraph-capture-sizes",
							"%s.engine: cudagraph_capture_sizes %v exceed max_num_batched_tokens %d; engine %s drops them",
							at, c.Dropped, e.MaxNumBatchedTokens, p.Version)
					}
				})
			},
		},
	}
}

func forEachPool(in rules.Input, fn func(at string, pool deployment.Pool)) {
	if in.Deployment == nil {
		return
	}
	for i, pool := range in.Deployment.Pools {
		fn(poolPath(i), pool)
	}
}

func forEachEngine(in rules.Input, fn func(at string, e deployment.Engine)) {
	forEachPool(in, func(at string, pool deployment.Pool) { fn(at, pool.Engine) })
}

// sortedKeys returns a set's members in sorted order, for a membership diagnostic that
// enumerates the valid alternatives (#19 item 3): an error that names what IS accepted is
// self-correcting in one turn, where "check the spelling" leaves a generator to guess.
func sortedKeys(set map[string]bool) []string {
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func poolPath(i int) string {
	return "pools[" + itoa(i) + "]"
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [20]byte
	n := len(b)
	for i > 0 {
		n--
		b[n] = byte('0' + i%10)
		i /= 10
	}
	return string(b[n:])
}

// totalExperts returns the largest expert count of any grouped-GEMM node the graph
// declares. Where layers differ it is the target's count, as for DeepSeek-V4.1, whose
// draft layers carry 128 experts beside the target's 384. A draft's own, smaller count
// is not checked.
func totalExperts(in rules.Input) int {
	if in.Model == nil {
		return 0
	}
	best := 0
	for _, lk := range in.Model.LayerKinds {
		for _, n := range lk.Nodes {
			if n.Experts > best {
				best = n.Experts
			}
		}
	}
	return best
}

// mayUseNixl reports whether the engine's KV transfer config may contain a
// NixlConnector: one is named directly, or a MultiConnector is, whose children the
// deployment does not record.
func mayUseNixl(d *deployment.Deployment) bool {
	for _, c := range []string{connectorOf(d.Offload), pdConnectorOf(d.PDTransfer)} {
		if c == "NixlConnector" || c == "MultiConnector" {
			return true
		}
	}
	return false
}

func sortedIntKeys(m map[string]int) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func ruleArchitectures(p *rules.Pack) []string {
	out := make([]string, 0, len(p.ModelSequenceParallelRules))
	for a := range p.ModelSequenceParallelRules {
		out = append(out, a)
	}
	sort.Strings(out)
	return out
}

func connectorOf(o *deployment.Offload) string {
	if o == nil {
		return ""
	}
	return o.Connector
}

func pdConnectorOf(t *deployment.PDTransfer) string {
	if t == nil {
		return ""
	}
	return t.Connector
}

// modelHasEngram reports whether the graph carries an Engram lookup: a gather whose
// residency the engine's Engram config controls. With the architecture, it is the
// schema's statement of the engine's test for the DBO refusal (a DeepseekV41ForCausalLM
// with engram_layer_ids); alone it is also true of other models EngramConfig governs.
func modelHasEngram(in rules.Input) bool {
	if in.Model == nil {
		return false
	}
	found := false
	in.Model.EachNode(func(_ string, n model.Node) {
		if n.Op == model.OpGather && n.ResidencyControl == model.ResidencyEngramCPUOffload {
			found = true
		}
	})
	return found
}

// modelQuantizedDType returns a sub-16-bit storage format the checkpoint uses, if any:
// such a checkpoint carries a quantization config, which sets the engine's quantization
// whether or not the deployment states one.
func modelQuantizedDType(in rules.Input) (model.DType, bool) {
	if in.Model == nil {
		return "", false
	}
	quantized := func(d model.DType) bool { return d != "" && d.Bytes() < 2 }
	if quantized(in.Model.Global.WeightDType) {
		return in.Model.Global.WeightDType, true
	}
	var found model.DType
	in.Model.EachNode(func(_ string, n model.Node) {
		if found == "" && quantized(n.WeightDType) {
			found = n.WeightDType
		}
	})
	return found, found != ""
}

func modelHasRecurrent(in rules.Input) bool {
	if in.Model == nil {
		return false
	}
	for _, lk := range in.Model.LayerKinds {
		for _, n := range lk.Nodes {
			if n.RecurrentKind != "" {
				return true
			}
		}
	}
	return false
}

// effectiveBlocks is the block size validate_block_size reads, as the set it may be. This
// release replaces the stated block size with the smallest page size among the KV cache
// groups that take part in prefix caching (vllm/v1/engine/core.py). A kv or index cache
// with its own page size always does; a state cache never does; a window cache does unless
// bounded replay is on, which needs the V2 model runner and no prefill-context
// parallelism (vllm/models/deepseek_v41/attention.py). Under prefill-context parallelism
// it certainly takes part; otherwise either answer is possible.
func effectiveBlocks(g *model.Graph, stated, pcp int) []int {
	sure, window := stated, stated
	g.EachNode(func(_ string, n model.Node) {
		for _, c := range n.Caches {
			if c.BlockSize < 1 {
				continue
			}
			switch c.Kind {
			case model.CacheKV, model.CacheIndex:
				sure = min(sure, c.BlockSize)
			case model.CacheWindow:
				if pcp > 1 {
					sure = min(sure, c.BlockSize)
				} else {
					window = min(window, c.BlockSize)
				}
			}
		}
	})
	if with := min(sure, window); with != sure {
		return []int{sure, with}
	}
	return []int{sure}
}
