// Moved from inference-sim/blis-schemas/rules/v0_29/pack.go at c197a5ad5b2091f003ad2c3d016f3ef9738a6459: the rules pack moved
// unchanged to blis-vllm (inference-sim/inference-sim#2021 B4). Imports of blis-schemas'
// internal/validate are rewritten to the rules aliases.

// Package v0_29 is the rules pack for vLLM 0.29.x.
//
// Everything here is a fact about that release, transcribed from its configuration
// surface. A later release gets its own package: copy this one, change the
// constants and the rule bodies that moved, and register it under the new version.
// Nothing in spec/ or kernel/ changes when an engine ships.
//
// Each rule states, in its Because field, what breaks without it. That is not
// decoration: a reader deciding whether to waive a rule needs to know what the rule
// is protecting, and a CI job that waives rules by name needs the list to be
// self-describing.
package v0_29

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
const Version = "0.29.0"

func set[T comparable](vs ...T) map[T]bool {
	m := make(map[T]bool, len(vs))
	for _, v := range vs {
		m[v] = true
	}
	return m
}

// Pack returns the rules pack for this version. It is a function rather than a
// package-level variable so a caller can hold two versions at once — useful when
// comparing what a scenario would mean across releases.
func Pack() *rules.Pack {
	p := &rules.Pack{
		Version: Version,

		// Transcribed from the engine's All2AllBackend literal. Twelve members:
		// the all-gather family, the point-to-point family, and the vendor
		// variants. A name outside this set is priced with the wrong volume basis.
		All2AllBackends: set("naive", "allgather_reducescatter", "pplx",
			"deepep_high_throughput", "deepep_low_latency", "deepep_v2",
			"flashinfer_all2allv", "flashinfer_nvlink_one_sided",
			"flashinfer_nvlink_two_sided", "mori_high_throughput",
			"mori_low_latency", "nixl_ep"),
		// The subset for which the engine makes the MoE input sequence-parallel,
		// which replaces the layer's all-reduce with a reduce-scatter/all-gather
		// pair. The default backend is a member, so this is not an exotic path.
		SPMoEBackends: set("allgather_reducescatter", "deepep_high_throughput",
			"deepep_low_latency", "flashinfer_nvlink_one_sided",
			"mori_high_throughput", "mori_low_latency", "nixl_ep"),

		// The engine exposes no allreduce-backend name. What it has is a boolean,
		// disable_custom_all_reduce, plus an environment-selected communicator; an
		// earlier draft of this pack invented a backend enum, which would have
		// rejected every deployment stating the real field. Kept as a set of the two
		// values the deployment field accepts so the rule shape stays uniform.
		AllReduceBackends: set("custom", "nccl"),

		// Transcribed from the CUDAGraphMode enum. Five members: the fifth,
		// FULL_DECODE_ONLY, captures decode and runs prefill eagerly, and omitting
		// it would reject a configuration the engine accepts.
		CUDAGraphModes: set("NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY",
			"FULL_AND_PIECEWISE"),

		// Transcribed from the CacheDType literal: seventeen members. The narrow
		// subset an earlier draft carried would have rejected nvfp4, which published
		// serving reports use, along with every per-token-head and turboquant format.
		CacheDTypes: set("auto", "float16", "bfloat16", "fp8", "fp8_e4m3",
			"fp8_e5m2", "fp8_inc", "fp8_ds_mla", "turboquant_k8v4",
			"turboquant_4bit_nc", "turboquant_k3v4_nc", "turboquant_3bit_nc",
			"int4_per_token_head", "int8_per_token_head", "fp8_per_token_head",
			"nvfp4", "nvfp4_4over6"),

		// Transcribed from the DCPCommBackend literal: an all-gather/reduce-scatter
		// set, or one all-to-all in place of the last two collectives.
		DCPCommBackends: set("ag_rs", "a2a"),

		// The MambaDType literal: a recurrent state's cache dtype is checked against
		// this, not against CacheDTypes, which lacks float32.
		MambaDTypes:     set("auto", "float32", "float16", "bfloat16"),
		MambaCacheModes: set("none", "align", "all"),

		// The MoEBackend literal of vllm/config/kernel.py, and its MEGA_MOE_BACKENDS.
		MoEBackends: set("auto", "triton", "batched_triton", "deep_gemm", "deep_gemm_mega_moe", "cutlass",
			"flashinfer_trtllm", "flashinfer_cutlass", "flashinfer_cutedsl", "flashinfer_b12x",
			"b12x", "flashinfer_moe_ep_mega_deep_gemm", "flashinfer_moe_ep_mega_cutedsl",
			"marlin", "humming", "triton_unfused", "aiter", "flydsl", "hpc", "emulation"),
		MegaMoEBackends: set("deep_gemm_mega_moe", "flashinfer_moe_ep_mega_deep_gemm",
			"flashinfer_moe_ep_mega_cutedsl"),

		SchedulerPolicies: set("fcfs", "priority"),

		// Speculative methods accepted by this release. The multi-token-prediction
		// family is large and per-model; the members here are the ones the three
		// design documents and the published report corpus exercise.
		SpeculativeMethods: set("ngram", "ngram_gpu", "eagle", "eagle3", "medusa",
			"mlp_speculator", "draft_model", "deepseek_mtp", "glm4_moe_mtp",
			"kimi_k3_mtp", "inkling_mtp", "qwen3_next_mtp", "nemotron_h_mtp",
			"mtp", "dflash", "dspark"),
		// Methods that leave async scheduling enabled. A method outside this set
		// disables it, which moves the host term from hidden to exposed.
		AsyncCompatibleSpec: set("eagle", "eagle3", "deepseek_mtp", "glm4_moe_mtp",
			"kimi_k3_mtp", "inkling_mtp", "qwen3_next_mtp", "nemotron_h_mtp",
			"mtp", "dflash", "ngram_gpu", "draft_model", "dspark"),

		OffloadSpecs: set("CPUOffloadingSpec", "TieringOffloadingSpec"),

		// Transcribed from the QuantizationMethods literal in
		// vllm/model_executor/layers/quantization/__init__.py: thirty members, the named
		// methods and the online-quant shorthands. The engine extends QUANTIZATION_METHODS
		// at runtime for out-of-tree methods, so a name outside this set is a warning, not
		// an error — the field is still the one that decides weight width, compute peak and
		// GEMM efficiency, and a typo among the in-tree names is worth catching.
		Quantizations: set("awq", "auto_awq", "fp8", "fbgemm_fp8", "fp_quant", "modelopt",
			"modelopt_fp4", "modelopt_mxfp8", "modelopt_mixed", "auto_gptq", "gptq",
			"gptq_marlin", "awq_marlin", "humming", "compressed-tensors", "experts_int8",
			"quark", "moe_wna16", "torchao", "inc", "mxfp4", "gpt_oss_mxfp4",
			"deepseek_v4_fp8", "online", "fp8_per_tensor", "fp8_per_block",
			"fp8_per_channel", "int8_per_channel_weight_only", "nvfp4_per_token", "mxfp8"),

		// Transcribed from the KVConnectorFactory.register_connector calls in
		// vllm/distributed/kv_transfer/kv_connector/factory.py: the names an offload or
		// PD-transfer block may select. Like a quantization method the registry is
		// extensible out of tree, so an unknown name is a warning.
		Connectors: set("DecodeBenchConnector", "ExampleConnector",
			"ExampleHiddenStatesConnector", "FlexKVConnectorV1", "HF3FSKVConnector",
			"LMCacheConnectorV1", "LMCacheMPConnector", "MoRIIOConnector",
			"MooncakeConnector", "MooncakeStoreConnector", "MultiConnector",
			"NixlConnector", "NixlPullConnector", "NixlPushConnector",
			"OffloadingConnector", "SimpleCPUOffloadConnector"),

		// Transcribed from the CachePolicyFactory.register_cache_policy calls in
		// vllm/v1/kv_offload/cpu/policies/factory.py: the KV-offload eviction policies. Two
		// in-tree (lru, arc); the factory also admits out-of-tree policies, so like the
		// connector and quantization registries an unknown name is a warning, not an error.
		EvictionPolicies: set("lru", "arc"),

		CustomAllReduceWorldSizes: set(2, 4, 6, 8, 16),

		DefaultDBODecodeThreshold:  32,
		DefaultDBOPrefillThreshold: 512,

		TritonFetchPageBytes: 28 * 1024,
		TritonFetchSMs:       12,

		CascadeAttnOptIn: true,

		// all2all_backend: All2AllBackend = "allgather_reducescatter" (vllm/config/parallel.py),
		// and the two accepted names it replaces with that one ("has been removed. Falling
		// back to 'allgather_reducescatter'").
		DefaultAll2AllBackend: "allgather_reducescatter",

		// DeepSeek-V4 decides its own sequence parallelism in this release too, by the
		// condition 0.31.0 keeps (vllm/models/deepseek_v4/nvidia/model.py
		// _use_sequence_parallel). DeepSeek-V4.1 does not exist here.
		ModelSequenceParallelRules: map[string]rules.SequenceParallelCondition{
			rules.ArchDeepseekV4: rules.SequenceParallelCondition{PipelineParallelOne: true, ExpertParallel: true,
				TensorParallelAboveOne: true, MegaMoEOrDataParallelAboveOne: true},
		},
		All2AllBackendRenames: map[string]string{
			"pplx": "allgather_reducescatter", "naive": "allgather_reducescatter"},

		// get_kv_cache_groups and _get_packed_kv_cache_groups (vllm/v1/core/kv_cache_utils.py
		// :1668-1797) already pack mixed pages, but is_state_bucket takes MambaSpec alone, the
		// block is the widest group unrounded (:1358-1370), and DeepSeek-V4's compressor
		// state is a SlidingWindowMLASpec (vllm/models/deepseek_v4/compressor.py:178).
		KVCachePacking: rules.KVCachePacking{Modeled: true},
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
			Name:    "cp-kv-cache-interleave-fits-block",
			Because: "with decode-context parallelism on and no KV connector configured, this release asserts at startup that block_size is at least, and divisible by, cp_kv_cache_interleave_size (vllm/config/vllm.py validate_block_size), so a layout breaking it does not start. The block size it checks is the smallest KV cache group's, which can be a model's own cache page size rather than the stated one",
			Check: func(in rules.Input, out *rules.Problems) {
				// Only where the assert is certain to run on the values stated:
				//   - a KV connector of any kind makes this release pin the size to the
				//     block size instead, so any PD transfer or offload skips the check;
				//   - a hybrid model may have its block size re-aligned by the platform,
				//     and an unknown model may be hybrid, so the stated size is final only
				//     for a known model with no recurrent layer;
				//   - an unstated block size is chosen by the platform.
				if in.Deployment == nil || in.Deployment.PDTransfer != nil ||
					in.Deployment.Offload != nil || in.Model == nil || modelHasRecurrent(in) {
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
			Name:    "pcp-excludes-data-parallelism",
			Because: "this release refuses prefill-context parallelism combined with data parallelism at startup (\"PCP does not support data parallelism yet\", vllm/config/parallel.py), so such a layout does not run; later releases support it from e6dc16cebd, which is why the refusal is here and not a field check",
			Check: func(in rules.Input, out *rules.Problems) {
				forEachPool(in, func(at string, pool deployment.Pool) {
					if pool.Parallel.PCP > 1 && pool.Parallel.DP > 1 {
						out.RuleErrorf("pcp-excludes-data-parallelism",
							"%s: pcp %d with dp %d; engine %s refuses prefill-context parallelism combined with data parallelism",
							at, pool.Parallel.PCP, pool.Parallel.DP, Version)
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
			Name:    "cascade-attention-is-opt-in",
			Because: "cascade attention is off unless requested, and a request for it alongside speculative decoding, with async scheduling left to the engine, is one the engine overrides once async scheduling resolves on",
			Check: func(in rules.Input, out *rules.Problems) {
				if !p.CascadeAttnOptIn {
					return
				}
				forEachPool(in, func(at string, pool deployment.Pool) {
					e := pool.Engine
					if e.DisableCascadeAttn != nil && !*e.DisableCascadeAttn &&
						e.Speculative != nil && e.AsyncScheduling == nil {
						out.RuleWarnf("cascade-attention-is-opt-in",
							"%s: cascade attention is requested alongside speculative decoding; with async scheduling resolved on, the engine disables cascade regardless", at)
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
			Name:    "engram-config-unsupported",
			Because: "this release has no Engram config (no vllm/config/engram.py, no --engram-config), so a launch line carrying one does not parse",
			Check: func(in rules.Input, out *rules.Problems) {
				forEachEngine(in, func(at string, e deployment.Engine) {
					if e.EngramConfig != nil {
						out.RuleErrorf("engram-config-unsupported",
							"%s.engine.engram_config: engine %s has no Engram config", at, p.Version)
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
			Because: "an architecture that decides its own sequence parallelism never consults the engine-wide sequence-parallel-MoE rule -- DeepSeek-V4 passes its own _use_sequence_parallel to its MoE and shared expert, and FusedMoE then skips the final tensor-parallel all-reduce (vllm/model_executor/layers/fused_moe/runner/moe_runner.py, \"Skipped when sequence-parallel is active\") -- so a decoder-layer reduction conditioned on that rule fires or not on the wrong layouts; and a plain tensor-parallel all-reduce inside a decoder layer is replaced under the model's sequence parallelism, which shards the residual stream by token",
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
					c, _, err := p.CUDAGraphCaptureSizes(e)
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

// effectiveBlocks is the block size validate_block_size reads: this release replaces the
// stated block size with the smallest page size among all KV cache groups
// (vllm/v1/engine/core.py), so every cache the graph pages at its own size lowers it.
func effectiveBlocks(g *model.Graph, stated, _ int) []int {
	block := stated
	g.EachNode(func(_ string, n model.Node) {
		for _, c := range n.Caches {
			if c.BlockSize > 0 && c.BlockSize < block {
				block = c.BlockSize
			}
		}
	})
	return []int{block}
}
