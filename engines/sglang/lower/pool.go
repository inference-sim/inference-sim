// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/layers/quantization/fp8.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/model_executor/cuda_graph_config.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).
// Derived from python/sglang/srt/distributed/device_communicators/custom_all_reduce.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package lower

import (
	"fmt"
	"slices"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/coefficient"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/hardware"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
	"github.com/inference-sim/inference-sim/substrate/contracts/units"

	blissglang "github.com/inference-sim/inference-sim/engines/sglang"
	"github.com/inference-sim/inference-sim/engines/sglang/config"
)

// PoolInput is the documents the lowering of one pool reads: the deployment and the pool it
// selects, the catalog graph of the model and the catalog chip. The caller loads them; this
// package imports no loader (X7).
type PoolInput struct {
	Deployment *deployment.Deployment
	PoolIndex  int
	Model      *model.Graph
	Chip       *hardware.Chip
}

// Args are the pool's ServerArgs as the lowering and the engine both read them: the engine block's
// fields by SGLang's own names, with the pool's tensor- and data-parallel widths (they are the
// pool's, not the block's: a block that states them must agree). A pool with parallelism beyond
// tensor and data is refused by name.
func Args(in PoolInput) (map[string]any, error) {
	if in.Deployment == nil {
		return nil, fmt.Errorf("lowering a pool needs its deployment")
	}
	if in.PoolIndex < 0 || in.PoolIndex >= len(in.Deployment.Pools) {
		return nil, fmt.Errorf("pool index %d is outside the deployment's %d pool(s)", in.PoolIndex, len(in.Deployment.Pools))
	}
	pool := in.Deployment.Pools[in.PoolIndex]
	args := map[string]any{}
	if !pool.Engine.IsZero() {
		if err := pool.Engine.Decode(&args); err != nil {
			return nil, fmt.Errorf("the pool's engine block: %w", err)
		}
	}
	if args == nil {
		args = map[string]any{}
	}
	pl := pool.Parallel
	if pl.PP > 1 || pl.DCP > 1 || pl.PCP > 1 || pl.EnableExpertParallel {
		return nil, &NotLowered{Name: "pool parallelism beyond tensor and data (pp, dcp, pcp, expert parallel)", Issue: issueLower}
	}
	for name, want := range map[string]int{"tp_size": max(pl.TP, 1), "dp_size": max(pl.DP, 1)} {
		if got, ok := args[name]; ok {
			n, isInt := asInt(got)
			if !isInt || n != want {
				return nil, fmt.Errorf("the engine block states %s %v and the pool's parallelism states %d: they must agree", name, got, want)
			}
		}
		args[name] = want
	}
	if _, ok := args["model_path"]; !ok {
		// The catalog names the model; SGLang's ServerArgs requires a path and the resolution reads
		// only the model's facts, which the graph supplies.
		args["model_path"] = "<model>"
	}
	return args, nil
}

// Resolve resolves the pool's engine block as SGLang resolves a ServerArgs (Args, the chip and the
// model). Anything outside the input space config models is refused by name (config.NotModelled).
func Resolve(in PoolInput) (*config.Resolved, error) {
	if in.Model == nil || in.Chip == nil {
		return nil, fmt.Errorf("lowering a pool needs its model graph and chip")
	}
	args, err := Args(in)
	if err != nil {
		return nil, err
	}
	chip, err := ChipOf(in.Chip)
	if err != nil {
		return nil, err
	}
	facts, err := ModelFactsOf(in.Model)
	if err != nil {
		return nil, err
	}
	return config.Resolve(config.Input{Args: args, Chip: chip, Model: facts})
}

func asInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int64:
		return int(n), true
	case uint64:
		return int(n), true
	case float64:
		return int(n), float64(int(n)) == n
	}
	return 0, false
}

// Pool lowers one pool: it resolves the engine block (Resolve) and states, in the plan's neutral
// words, every setting SGLang resolved that the cost model prices. Each is read from the SGLang
// code its comment cites; the cost model applies no default of its own (SUB-6).
func Pool(in PoolInput) (*plan.Lowered, error) {
	res, err := Resolve(in)
	if err != nil {
		return nil, err
	}
	return Lower(&res.Args, in.Model, in.Chip)
}

// Lower is Pool over arguments already resolved.
func Lower(a *config.ServerArgs, g *model.Graph, chip *hardware.Chip) (*plan.Lowered, error) {
	if err := check(a); err != nil {
		return nil, err
	}
	out := &plan.Lowered{}
	p := &out.Plan
	// The identity of the code that does the host and the launch work (Table DC): the engine-path
	// coefficients are keyed by it, so a release that changes that code is priced by its own.
	p.Host.CodeID = blissglang.CodeIDOf(coefficient.CodePathHost)
	p.Graphs.CodeID = blissglang.CodeIDOf(coefficient.CodePathGraphs)
	note := func(name, value, reason string) {
		out.Notes = append(out.Notes, plan.Note{Name: name, Value: value, Reason: reason})
	}
	override := func(field, requested, resolved, reason string) {
		out.Overrides = append(out.Overrides, plan.Override{Field: field, Requested: requested, Resolved: resolved, Reason: reason})
	}
	hasRecurrent, moe := false, false
	g.EachNode(func(_ string, n model.Node) {
		hasRecurrent = hasRecurrent || n.Op == model.OpRecurrentUpdate
		moe = moe || n.Op == model.OpGroupedGEMM
	})
	if a.EnableTwoBatchOverlap || a.EnableSingleBatchOverlap {
		return nil, &NotLowered{Name: "two-batch / single-batch overlap", Issue: issueLower}
	}

	// The served weight format. A checkpoint is served in the format it was quantized in unless
	// --quantization names a method; a method that quantizes at load (a bf16 checkpoint served
	// as fp8, or its experts as nvfp4) changes the weights the graph states, which is not modelled.
	// modelopt_fp4 on a checkpoint whose graph stores nvfp4 nodes (ModelOptFp4Config's serialized path:
	// layers/quantization/modelopt_quant.py) serves each node as the graph states it, as blis-vllm's
	// servedDType does (storesNode): the plan names no format and the nvfp4 nodes are priced per node.
	served := ""
	if a.Quantization != nil {
		q := *a.Quantization
		switch {
		case q == "fp8" && g.Global.WeightDType == model.DTypeFP8:
			served = "fp8"
		case q == "modelopt_fp4" && storesNVFP4(g):
		default:
			why := " on a " + string(g.Global.WeightDType) + " checkpoint"
			if q == "modelopt_fp4" {
				why = " on a checkpoint that stores no nvfp4 weights (SGLang would quantize them on load)"
			}
			return nil, &NotLowered{Name: "quantization=" + q + why, Issue: issueLower}
		}
	}
	p.WeightFormat = served

	// The KV cache format: kv_cache_dtype (kv_cache_hook.py:40-47@297e54fd) names the element.
	switch a.KvCacheDtype {
	case "auto", "bfloat16", "float16", "half":
		p.Cache.Record, p.Cache.ElementBytes = "", 2
	case "fp8_e4m3", "fp8_e5m2":
		p.Cache.Record, p.Cache.ElementBytes = "fp8", 1
	default:
		return nil, &NotLowered{Name: "kv_cache_dtype=" + a.KvCacheDtype, Issue: issueLower}
	}
	// The kernel requires the recurrent state's width stated even for a model with no recurrent
	// layer (SUB-6); a model with one states it from mamba_ssm_dtype (qwen35.go).
	p.Cache.StateElementBytes = 4
	if hasRecurrent {
		if err := lowerRecurrent(a, g, &p.Cache, note); err != nil {
			return nil, err
		}
	}

	// The DSA attention backends: the plan carries no attention backend, and the substrate prices DSA
	// attention from the model graph's sparse-MLA lane (the part-wide attention pair), whatever kernel
	// the backend names. The pool states the substitution, so a result can name it (X19).
	if a.DsaPrefillBackend != nil {
		override("dsa_prefill_backend", *a.DsaPrefillBackend, dsaPricedAs, dsaReason)
	}
	if a.DsaDecodeBackend != nil {
		override("dsa_decode_backend", *a.DsaDecodeBackend, dsaPricedAs, dsaReason)
	}

	// Batch limits and the page.
	if a.PageSize == nil || a.MemFractionStatic == nil {
		return nil, fmt.Errorf("page_size and mem_fraction_static must resolve")
	}
	p.PageTokens = units.Tokens(*a.PageSize)
	p.MemoryFraction = *a.MemFractionStatic
	if a.ContextLength != nil {
		p.MaxContext = units.Tokens(*a.ContextLength)
	}
	decode := a.CudaGraphConfig.Decode
	reqs := 0
	if a.MaxRunningRequests != nil {
		reqs = *a.MaxRunningRequests
	} else {
		// SGLang derives the request limit from the pool the model runner sizes after the
		// weights load (resolve_max_num_reqs, kv_cache_configurator.py@297e54fd), which the
		// cost model cannot know before it prices the weights; the decode graphs' largest batch
		// stands in as the bound the step limit is read for.
		reqs = maxInt(decode.Bs)
		note("max_running_requests", fmt.Sprint(reqs),
			"stated by no field: SGLang sizes it from the KV pool; the largest decode CUDA-graph batch bounds a step")
	}
	if reqs <= 0 {
		return nil, fmt.Errorf("max_running_requests resolves to %d", reqs)
	}
	p.Batch.Requests = reqs
	// The tokens one step schedules: a prefill chunk (chunked_prefill_size) or, with chunking off,
	// the prefill budget (max_prefill_tokens); a decode step schedules one token per request.
	budget := a.MaxPrefillTokens
	if a.ChunkedPrefillSize != nil && *a.ChunkedPrefillSize > 0 {
		budget = min(budget, *a.ChunkedPrefillSize)
		if budget <= 0 {
			budget = *a.ChunkedPrefillSize
		}
	}
	p.Batch.Tokens = units.Tokens(max(budget, reqs))

	// CUDA graphs (cuda_graph_config.py:119-193@297e54fd): decode captures by batch size; prefill
	// is captured per token count.
	p.Graphs.Decode = launchOf(decode.Backend)
	p.Graphs.Other = launchOf(a.CudaGraphConfig.Prefill.Backend)
	if p.Graphs.Decode == plan.LaunchPiecewise {
		return nil, &NotLowered{Name: "a piecewise decode CUDA graph", Issue: issueLower}
	}
	if p.Graphs.Decode != plan.LaunchEager {
		p.Graphs.Mode = plan.GraphByBatch
		for _, bs := range decode.Bs {
			if bs <= reqs {
				p.Graphs.Buckets = append(p.Graphs.Buckets, bs)
			}
		}
		slices.Sort(p.Graphs.Buckets)
	}
	ceiling := 0
	if p.Graphs.Decode != plan.LaunchEager {
		ceiling = maxInt(p.Graphs.Buckets)
	}
	if p.Graphs.Other != plan.LaunchEager {
		ceiling = max(ceiling, maxInt(a.CudaGraphConfig.Prefill.Bs))
	}
	p.Graphs.Ceiling = units.Tokens(ceiling)
	if (p.Graphs.Decode != plan.LaunchEager || p.Graphs.Other != plan.LaunchEager) && ceiling <= 0 {
		return nil, fmt.Errorf("cuda_graph_config captures no size")
	}
	switch {
	case p.Graphs.Decode == plan.LaunchGraph && p.Graphs.Other == plan.LaunchEager,
		p.Graphs.Decode == plan.LaunchGraph && p.Graphs.Other == plan.LaunchPiecewise,
		p.Graphs.Decode == plan.LaunchGraph && p.Graphs.Other == plan.LaunchGraph,
		p.Graphs.Decode == plan.LaunchEager && p.Graphs.Other == plan.LaunchEager:
	default:
		return nil, &NotLowered{Name: fmt.Sprintf("decode graphs %s with prefill graphs %s", p.Graphs.Decode, p.Graphs.Other), Issue: issueLower}
	}

	// The overlap schedule keeps one step in flight while the next is scheduled
	// (disable_overlap_schedule, scheduler.py event_loop_overlap@297e54fd).
	p.InFlight = 1
	if !a.DisableOverlapSchedule {
		p.InFlight = 2
	}

	// The routed experts' kernel (layers/quantization/fp8.py:2962-2972@297e54fd): under
	// moe_runner_backend auto, a block-fp8 MoE runs Triton's fused_moe unless the DeepEP-family
	// all-to-all selects DeepGEMM; the all-to-all backends are refused above (config).
	if moe {
		experts, routing, err := expertsKernel(a, g)
		if err != nil {
			return nil, err
		}
		p.MoE.Experts, p.MoE.Routing = experts, routing
		if a.MoeRunnerBackend == "flashinfer_trtllm" {
			if routedExpertsDType(g) == model.DTypeNVFP4 {
				note("nvfp4_experts_ramp", "flashinfer_trtllm",
					"the routed experts run FlashInfer's TRT-LLM NVFP4 fused MoE (trtllm_fp4_block_scale_moe, layers/quantization/modelopt_quant.py); "+
						"the registry holds no NVFP4 grouped-expert ramp for the B200 or B300, so the substrate prices them on the dense NVFP4 "+
						"linear ramp (measured on FlashInfer's CuTe-DSL kernel), a ramp whose signed median TPOT error on a vLLM NVFP4 model is "+
						"+10.4 percent on the B200 and +19.8 percent on the B300 (inference-sim/blis-registry#51); the price is borrowed and not corrected")
			} else {
				note("moe_runner_backend", "flashinfer_trtllm",
					"the plan names no experts kernel: the substrate prices FlashInfer's TRT-LLM grouped GEMM on block-FP8 experts by its "+
						"assumed ramp (graded assumed: the engine release lists no sm_100a build to name a code hash); expert selection "+
						"runs inside the kernel only for a DeepSeek-V3-method router")
			}
		}
	}

	// Allreduce: SGLang's CustomAllreduce handles a width in {2, 4, 6, 8} unless disabled
	// (custom_all_reduce.py:41,90@297e54fd); the cost model prices that backend as it does the SM
	// all-reduce.
	out.Reported.SMAllReduce = a.TpSize > 1 && !a.DisableCustomAllReduce &&
		slices.Contains([]int{2, 4, 6, 8}, a.TpSize)

	// FlashInfer all-reduce fusion. Resolution turns it on (backend "auto") for the architectures
	// of overrides.py:929 on SM90/SM100 at tp > 1 without attention DP or an all-to-all
	// (overrides.py:984-1001@297e54fd), and --enable-flashinfer-allreduce-fusion states the same
	// backend (serving_hook.py:284-297). A step of at most FUSE_ALLREDUCE_MAX_BATCH_SIZE tokens
	// (layer_boundary/residual/add_norm.py:141-157) then runs each layer's two all-reduces
	// (attention to FFN, FFN to the next layer's attention: fusions/allreduce.py:68-163) with the
	// residual add and the norm that follows in one kernel, so two of a layer's norm nodes run
	// inside the all-reduce kernel the layer already launches. The first layer has no all-reduce
	// before its input norm; the plan states the per-layer figure, so that one launch is counted
	// as fused.
	if b := a.FlashinferAllreduceFusionBackend; b != nil && *b != "" {
		if *b != "auto" {
			return nil, &NotLowered{Name: "flashinfer_allreduce_fusion_backend=" + *b, Issue: issueLower}
		}
		if a.TpSize > 1 && !ParallelOf(a).AttnDPEnabled {
			p.Launches = []plan.LaunchChange{{FromTokens: 1, ToTokens: allreduceFusionMaxTokens, PerLayer: -2}}
			note("flashinfer_allreduce_fusion", "auto",
				"two norm launches per layer run inside the all-reduce kernel for steps of at most "+
					"2048 tokens; the first layer's input norm follows no all-reduce and is counted as fused too")
		}
	}
	return out, nil
}

// dsaPricedAs and dsaReason are the substitution a DSA pool states for its attention backends.
const (
	dsaPricedAs = "the part-wide sparse-MLA attention pair"
	dsaReason   = "flashmla_sparse, flashmla_kv, trtllm and fa3 are different kernels; the plan has no attention-backend " +
		"slot and the substrate prices DSA attention from the graph's sparse-MLA lane with the registry's part-wide attention " +
		"pair, so the backend the engine runs is not what is priced; that price is a transfer from the vLLM-measured kernel, " +
		"not a measurement of SGLang's (attention_backend nsa is the deprecated alias of dsa: " +
		"layers/attention/attention_registry.py:166-176@297e54fd)"
)

// allreduceFusionMaxTokens is FUSE_ALLREDUCE_MAX_BATCH_SIZE (layer_boundary/residual/add_norm.py:141@297e54fd).
const allreduceFusionMaxTokens = 2048

// launchOf is the plan's launch word for a CUDA-graph backend: a full graph, a breakable graph
// (attention eager between captured pieces, the plan's piecewise), or none.
func launchOf(backend string) plan.Launch {
	switch backend {
	case "full":
		return plan.LaunchGraph
	case "breakable":
		return plan.LaunchPiecewise
	}
	return plan.LaunchEager
}

func maxInt(l []int) int {
	m := 0
	for _, v := range l {
		m = max(m, v)
	}
	return m
}

// expertsKernel is the kernel SGLang runs the routed experts with, for a block-fp8 or bf16
// checkpoint under the resolved moe_runner_backend. Every backend this does not name is refused
// by name until its kernel is lowered.
func expertsKernel(a *config.ServerArgs, g *model.Graph) (plan.KernelImpl, plan.MoERouting, error) {
	if a2aBackend(a) != "none" {
		return plan.KernelImpl{}, "", &NotLowered{Name: "moe_a2a_backend=" + a2aBackend(a), Issue: issueLower}
	}
	switch a.MoeRunnerBackend {
	case "auto", "triton":
		if g.Global.WeightDType != model.DTypeFP8 {
			return plan.KernelImpl{}, "", &NotLowered{Name: "moe_runner_backend=" + a.MoeRunnerBackend + " on " + string(g.Global.WeightDType) + " experts", Issue: issueLower}
		}
		return plan.ExpertsTriton, plan.RoutingSeparate, nil
	case "flashinfer_trtllm":
		// FlashInfer's TRT-LLM monolithic MoE (layers/moe/moe_runner/flashinfer_trtllm.py): router logits
		// in, expert selection inside the kernel. The substrate prices it where the plan names no experts
		// kernel, from the routed experts' format and its grouped-GEMM ramp; the block-fp8 form and the
		// nvfp4 form (trtllm_fp4_block_scale_moe) are priced.
		switch routedExpertsDType(g) {
		case model.DTypeFP8, model.DTypeNVFP4:
		default:
			return plan.KernelImpl{}, "", &NotLowered{Name: "moe_runner_backend=flashinfer_trtllm on " + string(routedExpertsDType(g)) + " experts", Issue: issueLower}
		}
		// SGLang selects the experts inside the kernel only when it can bypass its own top-k
		// (_can_bypass_trtllm_routing, layers/moe/topk.py:613-650, 794-803@297e54fd). That depends on the
		// router's method, grouped top-k, correction bias, scoring function, renormalisation and shared
		// slots, which the model graph does not carry; they are the architecture's own, so the answer is
		// stated per architecture from the model code, and an architecture not listed is refused by name
		// rather than guessed:
		//   - DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM: the DeepSeek-V3 method (grouped top-k with a
		//     correction bias, renormalised): the kernel takes the logits, selection is fused.
		//   - DeepseekV4ForCausalLM: sqrt-softplus scoring (config.sqrtsoftplus_log1p), which the bypass
		//     test refuses: SGLang runs its own top-k, a launch of its own.
		switch g.Global.Architecture {
		case "DeepseekV3ForCausalLM", "GlmMoeDsaForCausalLM":
			return plan.KernelImpl{}, plan.RoutingFused, nil
		case "DeepseekV4ForCausalLM":
			return plan.KernelImpl{}, plan.RoutingSeparate, nil
		}
		return plan.KernelImpl{}, "", &NotLowered{Name: "moe_runner_backend=flashinfer_trtllm on " + g.Global.Architecture +
			" (whether SGLang bypasses its own top-k depends on the router's method, scoring and renormalisation, which the graph does not carry; " +
			"a plain Renormalize softmax router, such as Qwen3 MoE's, bypasses it only when norm_topk_prob holds)", Issue: issueLower}
	}
	return plan.KernelImpl{}, "", &NotLowered{Name: "moe_runner_backend=" + a.MoeRunnerBackend, Issue: issueLower}
}

// mainStackKinds are the layer kinds the model's own stack uses (its prologue, pattern and epilogue); the
// speculator's kinds (an MTP module) are not part of what a step without speculation prices.
func mainStackKinds(g *model.Graph) []model.LayerKind {
	used := map[string]bool{}
	for _, list := range [][]string{g.Stack.Prologue, g.Stack.Pattern, g.Stack.Epilogue} {
		for _, id := range list {
			used[id] = true
		}
	}
	var out []model.LayerKind
	for _, lk := range g.LayerKinds {
		if used[lk.ID] {
			out = append(out, lk)
		}
	}
	return out
}

// storesNVFP4 reports whether any weight node of the model's own stack states nvfp4 as its stored width:
// the checkpoint is NVFP4-serialized (blis-vllm's storesNode).
func storesNVFP4(g *model.Graph) bool {
	for _, lk := range mainStackKinds(g) {
		for _, n := range lk.Nodes {
			if (n.Op == model.OpGEMM || n.Op == model.OpGroupedGEMM) && n.WeightDType == model.DTypeNVFP4 {
				return true
			}
		}
	}
	return false
}

// routedExpertsDType is the stored width of the routed experts of the model's own stack: the weight dtype
// of its routed GroupedGEMM nodes (a shared expert is not routed), the checkpoint's own where a node states
// none. Mixed widths, or no routed expert, give the empty dtype, which no experts kernel is lowered for.
func routedExpertsDType(g *model.Graph) model.DType {
	var d model.DType
	seen, mixed := false, false
	for _, lk := range mainStackKinds(g) {
		for _, n := range lk.Nodes {
			if n.Op != model.OpGroupedGEMM || n.IsSharedExpert() {
				continue
			}
			w := n.WeightDType
			if w == "" {
				w = g.Global.WeightDType
			}
			if seen && w != d {
				mixed = true
			}
			d, seen = w, true
		}
	}
	if mixed || !seen {
		return ""
	}
	return d
}
