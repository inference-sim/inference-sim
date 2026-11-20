// Original to blis-vllm.
// The engine-settings resolution the cost model ran until engine lowering took it over, moved
// unchanged from blis-substrate's kernel/new.go, its async-scheduling file, draft.go and internal/resolve at
// 00817b10798cbc00f560e51921c30c7084fa5e0c (blis-substrate#58).

// Package lowering resolves a vLLM pool's settings into the neutral settings of the pool's plan
// that the cost model prices (plan.Lowered): every default, backend choice, graph mode and
// capture size, in the plan's words, with the record of how each was resolved (SUB-6, design
// §6.1). The cost model applies no default to the plan and reads no vLLM setting.
//
// KNOWN DIVERGENCE. The resolution is the one the substrate's kernel ran, moved here unchanged so
// that every price stays what it was: it encodes vLLM v0.31.0's behaviour, whatever release a
// scenario declares, and says so in its notes. Moving it to v0.32.0 is blis-vllm#78.
package lowering

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/enginefacts"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/coefficient"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/hardware"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/scenario"
	"github.com/inference-sim/inference-sim/substrate/contracts/units"

	blisvllm "github.com/inference-sim/inference-sim/engines/vllm"
	"github.com/inference-sim/inference-sim/engines/vllm/core/vllmsettings"
)

// Input is what the lowering of one pool reads: the documents the cost model is built from, the
// release's rules, and the pool's settings as blis-vllm decoded and pre-resolved them.
type Input struct {
	Scenario   *scenario.Scenario
	Deployment *deployment.Deployment
	PoolIndex  int
	// Model is the target's graph as the catalog states it, and Draft the graph of the drafter
	// the pool names as a checkpoint of its own (speculative.draft_model), nil for none.
	Model *model.Graph
	Draft *model.Graph
	Chip  *hardware.Chip
	Rules enginefacts.EngineRules
	// Settings are the pool's settings.
	Settings vllmsettings.Settings
}

// lowerContext is what the resolution reads beside the settings: the pool, the chip and its
// generation, the layout, and the drafter's layer kinds.
type lowerContext struct {
	pool            deployment.Pool
	chip            hardware.Chip
	family          int
	lay             layout
	rackIsOneDomain bool
	// grafted are the layer kinds of the target with its drafter: the target's, less those only a
	// replaced speculator of its own ran, and the drafter's.
	grafted []model.LayerKind
	// speculatorKinds are the layer kinds the running speculator's stacks name.
	speculatorKinds []model.LayerKind
}

// hopper reports whether the chip is the SM90 family: the stated compute capability, or for a
// chip that states none, no native NVFP4.
func (c *lowerContext) hopper() bool {
	if c.family > 0 {
		return c.family == 9
	}
	return c.chip.NVFP4Peak <= 0
}

// preHopper reports whether the chip's compute capability is below 9.0, which no sparse-MLA
// backend runs on; a chip that states none is refused.
func (c *lowerContext) preHopper() (bool, error) {
	if c.family <= 0 {
		return false, fmt.Errorf("chip %q states no usable ComputeCapability (%q), which the "+
			"DSA sparse-MLA backend choice needs to place the part below or at SM90",
			c.chip.Name, c.chip.ComputeCapability)
	}
	return c.family < 9, nil
}

// Lower resolves one pool's settings into the plan's resolved settings.
//
// The rules are the caller's code, so a rules value that cannot answer, or a rules method that
// panics while the pool is lowered, is a refusal naming the rules, not a panic out of Lower: a
// configuration search gets an error it can report and move past.
func Lower(in Input) (low *plan.Lowered, err error) {
	if in.Scenario == nil || in.Deployment == nil || in.Model == nil || in.Chip == nil {
		return nil, fmt.Errorf("lowering a pool needs its scenario, deployment, model graph and chip")
	}
	if in.PoolIndex < 0 || in.PoolIndex >= len(in.Deployment.Pools) {
		return nil, fmt.Errorf("pool index %d is outside the deployment's %d pool(s)",
			in.PoolIndex, len(in.Deployment.Pools))
	}
	if isNil(in.Rules) {
		return nil, fmt.Errorf("lowering a pool needs engine rules for version %q",
			in.Scenario.EngineRelease())
	}
	if err := rulesAnswer(in.Rules); err != nil {
		return nil, fmt.Errorf("the engine rules for version %q cannot answer: %w",
			in.Scenario.EngineRelease(), err)
	}
	defer func() {
		if p := recover(); p != nil {
			rp, ok := p.(rulesPanic)
			if !ok {
				panic(p)
			}
			low, err = nil, fmt.Errorf("the engine rules for version %q cannot answer: %v",
				in.Scenario.EngineRelease(), rp.v)
		}
	}()
	in.Rules = panicTagged{in.Rules}
	c := lowerContext{pool: in.Deployment.Pools[in.PoolIndex], chip: *in.Chip}
	c.family, _ = in.Chip.CapabilityMajor()
	g, err := in.Model.ForCapability(c.family)
	if err != nil {
		return nil, fmt.Errorf("resolving the model graph for %s: %w", in.Chip.Name, err)
	}
	sp := g.Speculator
	c.speculatorKinds = g.LayerKinds
	c.grafted = g.LayerKinds
	if in.Draft != nil {
		d, err := in.Draft.ForCapability(c.family)
		if err != nil {
			return nil, fmt.Errorf("resolving the drafter graph for %s: %w", in.Chip.Name, err)
		}
		sp, c.speculatorKinds = d.Speculator, d.LayerKinds
		c.grafted = append(withoutSpeculatorKinds(g), d.LayerKinds...)
	}
	pk, err := in.Scenario.Cluster.Packaging(in.Chip)
	if err != nil {
		return nil, fmt.Errorf("resolving the cluster's packaging: %w", err)
	}
	c.rackIsOneDomain = pk.GPUsPerRack > 0 && in.Chip.IntraRackBwGBps > 0
	if c.lay, err = geometry(pk, c.pool); err != nil {
		return nil, fmt.Errorf("resolving the layout: %w", err)
	}
	return lower(in, in.Settings.Clone(), g, sp, c)
}

// rulesAnswer asks a rules value the one question every value answers, its release, and reports
// a value that cannot: a forwarding struct around a nil rules value is not nil, but its first
// call dereferences nil.
func rulesAnswer(r enginefacts.EngineRules) (err error) {
	defer func() {
		if p := recover(); p != nil {
			err = fmt.Errorf("EngineVersion panicked: %v", p)
		}
	}()
	r.EngineVersion()
	return nil
}

// isNil reports whether v is nil or an interface holding a nil pointer, map, slice, func or
// channel: a value that cannot answer for a release.
func isNil(v any) bool {
	if v == nil {
		return true
	}
	switch rv := reflect.ValueOf(v); rv.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Func, reflect.Chan, reflect.Interface:
		return rv.IsNil()
	}
	return false
}

// withoutSpeculatorKinds is the target's layer kinds less those that only its own speculator ran,
// which a named drafter replaces.
func withoutSpeculatorKinds(target *model.Graph) []model.LayerKind {
	out := append([]model.LayerKind{}, target.LayerKinds...)
	sp := target.Speculator
	if sp == nil {
		return out
	}
	runs := map[string]bool{}
	for _, id := range target.Stack.Expand() {
		runs[id] = true
	}
	only := map[string]bool{}
	for _, id := range sp.Stack.Expand() {
		only[id] = !runs[id]
	}
	if sp.ContextStack != nil {
		for _, id := range sp.ContextStack.Expand() {
			only[id] = !runs[id]
		}
	}
	kept := out[:0]
	for _, lk := range out {
		if !only[lk.ID] {
			kept = append(kept, lk)
		}
	}
	return kept
}

func lower(in Input, eng vllmsettings.Settings, g *model.Graph, sp *model.Speculator, c lowerContext) (*plan.Lowered, error) {
	out := &plan.Lowered{}
	p := &out.Plan
	note := func(name, value, reason string) {
		out.Notes = append(out.Notes, plan.Note{Name: name, Value: value, Reason: reason})
	}
	override := func(field, requested, resolved, reason string) {
		out.Overrides = append(out.Overrides, plan.Override{Field: field, Requested: requested,
			Resolved: resolved, Reason: reason})
	}
	pl := c.pool.Parallel
	arch := g.Global.Architecture

	// The engine's decisions that follow from the layout.
	lay := c.lay
	lay.SequenceParallelMoE = in.Rules.SequenceParallelMoE(eng.All2AllBackend,
		pl.EnableExpertParallel, pl.TP, pl.DP)
	if arch != "" {
		lay.ModelSequenceParallel = in.Rules.ModelSequenceParallel(arch, pl, eng.MoEBackend)
	}
	var ar []plan.Override
	lay.CustomAllReduce, ar = customAllReduce(eng, lay, c.rackIsOneDomain, in.Rules)
	for _, o := range ar {
		override(o.Field, o.Requested, o.Resolved, o.Reason)
	}
	p.Collectives = plan.CollectivePlan{SequenceParallelMoE: lay.SequenceParallelMoE,
		SequenceParallelModel: lay.ModelSequenceParallel}
	out.Reported.SMAllReduce = lay.CustomAllReduce
	out.Encoder = encoderRun(eng, g, pl.TP, pl.PP)

	// The served weight format.
	quantization := eng.Quantization
	if deepseekV4Family(g) && g.Global.WeightDType == model.DTypeFP8 && quantization != "" {
		if quantization != "deepseek_v4_fp8" {
			override("quantization", quantization, "deepseek_v4_fp8",
				"vLLM v0.31.0 serves a DeepSeek-V4-family FP8 checkpoint with its own "+
					"quantization method whatever is requested")
		}
		quantization = ""
	}
	// An MXFP4 method quantizes the routed experts only, which the graph already states in MXFP4,
	// so the linear layers are served at the checkpoint's width (mxfp4ExpertsOnly).
	if ok, err := mxfp4ExpertsOnly(g, quantization); err != nil {
		return nil, err
	} else if ok {
		quantization = ""
	}
	if sw := eng.ServedWeights; sw != nil {
		out.ServedLayout = &plan.WeightLayout{Linear: sw.Linear, LinearScaleElements: sw.LinearScaleElements,
			RoutedExperts: sw.RoutedExperts, Router: sw.Router, Head: sw.Head, Embedding: sw.Embedding}
	}
	served, err := servedDType(g, quantization)
	if err != nil {
		return nil, err
	}
	p.WeightFormat = string(served)

	// The cache formats.
	if err := checkCacheDTypes(eng.CacheDType, eng.MambaCacheDType, attentionKinds(g)); err != nil {
		return nil, err
	}
	p.Cache.ElementBytes = cacheDTypeBytes(eng.CacheDType, served)
	record := in.Rules.ResolveCacheDType(arch, c.family, lay.TP, eng.CacheDType)
	if stated := eng.CacheDType; stated != "" && record != stated {
		override("cache_dtype", stated, record, "the attention backend of "+arch+
			" serves this cache as "+record+" (rules pack CacheDTypeResolutions)")
	}
	p.Cache.Record = record
	p.Cache.StateElementBytes = cacheDTypeBytes(eng.MambaCacheDType, model.DTypeFP32)
	out.DefaultCacheRecord = in.Rules.ResolveCacheDType(arch, c.family, lay.TP, "")
	if s := eng.Speculative; s != nil {
		out.Drafter = s.DraftModel
	}
	if err := refuseV4Layouts(in, c.grafted, eng, record); err != nil {
		return nil, err
	}

	// The recurrent cache mode as it runs. The config's default is "none"
	// (vllm/config/cache.py:190), and for a hybrid model vLLM v0.31.0 turns "none" into
	// "align" whenever prefix caching is on (vllm/model_executor/models/config.py:640-642) --
	// a STATED "none" included, since the hook cannot tell it from the default. Prefix caching
	// is on unless disabled (vllm/config/cache.py:142).
	hybrid := stackHasRecurrent(g)
	prefixCaching := eng.EnablePrefixCaching == nil || *eng.EnablePrefixCaching
	stated := eng.MambaCacheMode
	mode := stated
	if mode == "" {
		mode = "none"
	}
	if hybrid && prefixCaching && mode == "none" {
		mode = "align"
	}
	if hybrid {
		switch {
		case stated == "":
			note("mamba_cache_mode", mode,
				"vLLM v0.31.0's choice for a hybrid model: align with prefix caching on, "+
					"none with it off (vllm/model_executor/models/config.py:640-642)")
		case stated != mode:
			override("mamba_cache_mode", stated, mode,
				"with prefix caching on, vLLM v0.31.0 runs a hybrid model's "+
					"recurrent cache in align mode whatever none was stated "+
					"(vllm/model_executor/models/config.py:640-642)")
		}
	}
	switch mode {
	case "none":
		p.Cache.Recurrent = plan.RecurrentLatest
	case "align":
		p.Cache.Recurrent = plan.RecurrentPageAligned
	case "all":
		p.Cache.Recurrent = plan.RecurrentEveryPage
	default:
		return nil, fmt.Errorf("mamba_cache_mode %q is not one this kernel prices: it keeps the "+
			"latest state (none), one a page boundary (align) or one a page (all)", mode)
	}

	// The context bound: a count, auto, or unstated.
	maxLen := lowerMaxModelLen(eng.MaxModelLen, g, note)
	p.MaxContext = units.Tokens(maxLen)

	// The KV block size.
	block := eng.BlockSize
	dsa := isDSA(g)
	v4 := deepseekV4Family(g)
	blockFinal := !hybrid && (block > 0 || dsa || v4)
	switch {
	case v4 && block > 0 && block%v4BlockSize(arch, c.hopper) != 0:
		return nil, fmt.Errorf("block_size %d on a DeepSeek-V4-family model: its sparse "+
			"attention and indexer run %d-token kernel blocks on this part, so a stated size "+
			"must be a multiple of it, and the engine refuses this layout at startup",
			block, v4BlockSize(arch, c.hopper))
	case v4 && block <= 0:
		block = v4BlockSize(arch, c.hopper)
		note("block_size", strconv.Itoa(block),
			"what vLLM v0.31.0 picks for a DeepSeek-V4-family model on this part, the only "+
				"kernel block its sparse backends accept (vllm/models/deepseek_v41/"+
				"sparse_mla.py:89-90; vllm/models/deepseek_v4/sparse_mla.py:59-61)")
	case dsa && block > 0 && block%dsaBlockSize != 0:
		return nil, fmt.Errorf("block_size %d on a DSA sparse-MLA model: its indexer runs "+
			"on %d-token kernel blocks, so a stated size must be a multiple of %d, and the "+
			"engine refuses this layout at startup (see dsaBlockSize)",
			block, dsaBlockSize, dsaBlockSize)
	case dsa && block <= 0:
		block = dsaBlockSize
		note("block_size", strconv.Itoa(dsaBlockSize),
			"what vLLM v0.31.0 picks for a DSA sparse-MLA model, whose indexer accepts "+
				"only 64-token blocks (vllm/v1/attention/backends/mla/indexer.py:203-204; "+
				"vllm/platforms/interface.py:666-689)")
	case block <= 0:
		block = 16
		note("block_size", "16",
			"CacheConfig.DEFAULT_BLOCK_SIZE (vllm/config/cache.py:71 at v0.31.0); the "+
				"platform substitutes a size every attention backend supports where the "+
				"default is not one, and aligns a hybrid model's block to its recurrent "+
				"page even over a stated size (vllm/platforms/interface.py:683-702), "+
				"neither of which the kernel can see")
	}
	p.PageTokens = units.Tokens(block)

	// Decode-context parallelism. A replicated query projection is not priced, so it is
	// refused where it would take effect (dcp > 1, pcp <= 1, on a latent layer).
	if q := eng.DCPQReplicate; q != nil && *q && pl.DCP > 1 && pl.PCP <= 1 {
		for _, a := range stackPrimaryAttentions(g) {
			if a.NumQHeads > 0 && latentAttention(a.AttentionKind) {
				return nil, fmt.Errorf("resolving decode-context parallelism: "+
					"dcp_q_replicate is requested with dcp %d on a latent-attention model; "+
					"the replicated query projection it costs is not priced by this "+
					"kernel, so the knob is refused rather than credited for the "+
					"collective it saves", pl.DCP)
			}
		}
	}
	accepted := in.Rules.DCPCommBackendNames()
	if b := eng.DCPCommBackend; b != "" && accepted == nil {
		note("dcp_comm_backend", b,
			"stated, and checked against the backends this kernel prices, but not against "+
				"the names the engine release accepts: the rules value carries no list")
	}
	dc, dcOverrides, err := resolveDecodeContext(c.pool, eng, in.Deployment, block,
		blockFinal, accepted)
	if err != nil {
		return nil, fmt.Errorf("resolving decode-context parallelism: %w", err)
	}
	for _, o := range dcOverrides {
		override(o.Field, o.Requested, o.Resolved, o.Reason)
	}
	if dc.CommBackend == DCPAllToAll {
		p.DecodeContext.Combine = plan.CombineAllToAll
	}
	p.DecodeContext.Interleave = units.Tokens(dc.Interleave)
	if pl.DCP > 1 && eng.DCPCommBackend == "" {
		note("dcp_comm_backend", dc.CommBackend,
			"the engine's stock default (ParallelConfig.set_dcp_defaults, "+
				"vllm/config/parallel.py:581-594 at v0.31.0), with the query projection not "+
				"replicated; a model's configuration hook runs first and may choose "+
				"otherwise -- GlmMoeDsaForCausalLM selects a2a with dcp_q_replicate "+
				"(vllm/model_executor/models/config.py:43-50) -- and the kernel cannot see "+
				"which model class serves this graph, so state both to price it exactly. On "+
				"that model, where FlashMLA-sparse serves the layout, the difference "+
				"decides whether it starts: that backend runs DCP only with ag_rs "+
				"(flashmla_sparse.py:416-422)")
	}

	// Asynchronous scheduling (vllm/config/vllm.py:1544-1635 at v0.31.0).
	spec := 0
	method := ""
	if s := eng.Speculative; s != nil && s.NumSpecTokens > 0 {
		spec, method = s.NumSpecTokens, s.Method
	}
	var compatible, known bool
	if method != "" {
		compatible, known = in.Rules.AsyncCompatibleSpeculative(method)
	}
	knownIncompatible := method != "" && known && !compatible
	if method != "" && !known {
		note("async_scheduling_spec", method,
			"the rules value carries no async-compatible draft list, so the draft method is "+
				"taken as compatible with async scheduling")
	}
	async := !knownIncompatible
	if a := eng.AsyncScheduling; a != nil {
		if *a && knownIncompatible {
			return nil, fmt.Errorf("async_scheduling is requested with draft method %q, which "+
				"vLLM v0.31.0 refuses: async scheduling is only supported with "+
				"EAGLE/MTP/Draft Model/NGram GPU/DSpark/DFlash speculative decoding "+
				"(vllm/config/vllm.py:1553-1567)", method)
		}
		async = *a
	}
	p.InFlight = 1
	if async {
		p.InFlight = 2
	}

	p.Host.NativeFrontend = eng.RustFrontend()

	// The identity of the code that does the host and the launch work (Table DC): the engine-path
	// coefficients are keyed by it, so a release that changes that code is priced by its own.
	p.Host.CodeID = blisvllm.CodeIDOf(coefficient.CodePathHost)
	p.Graphs.CodeID = blisvllm.CodeIDOf(coefficient.CodePathGraphs)

	// The CUDA graph mode, per batch (vllm/config/compilation.py:604-640;
	// vllm/v1/cudagraph_dispatcher.py:233-310 at v0.31.0). FULL_AND_PIECEWISE is v0.31.0's
	// default.
	if err := refuseAll2AllBinding(in.Rules, eng.All2AllBackend, eng.MoEBackend); err != nil {
		return nil, err
	}
	backend := resolvedAll2AllBackend(in.Rules, eng.All2AllBackend, eng.MoEBackend)
	graphMode := strings.ToUpper(eng.CUDAGraphMode)
	if graphMode == "" {
		graphMode = "FULL_AND_PIECEWISE"
		note("cudagraph_mode", graphMode,
			"vLLM v0.31.0's default (vllm/config/compilation.py:614); the engine "+
				"downgrades it where an attention backend does not support full graphs, "+
				"which the kernel cannot see")
	}
	// Two MoE dispatch backends cannot be captured, and the engine turns graphs off for them
	// whatever was stated: DeepEP high-throughput with dp above one
	// (vllm/config/compilation.py:1237-1253), and MoonEP wherever the all-to-all runs
	// (vllm/config/vllm.py:2098-2106; use_all2all, vllm/config/parallel.py:738-743).
	useAll2All := lay.DP > 1 || lay.SequenceParallelMoE || (pl.EnableExpertParallel && lay.PCP > 1)
	if graphMode != "NONE" && ((backend == "deepep_high_throughput" && lay.DP > 1) ||
		(backend == "moonep" && useAll2All)) {
		override("cudagraph_mode", graphMode, "NONE",
			"the "+backend+" all-to-all cannot be captured, so vLLM v0.31.0 "+
				"runs this layout with no CUDA graphs")
		graphMode = "NONE"
	}
	// enforce_eager turns compilation and CUDA graphs off whatever cudagraph_mode was stated
	// (vllm/config/vllm.py:1766-1769, :2004-2009 at 4ba787ec9959).
	if eng.EnforceEager && graphMode != "NONE" {
		override("cudagraph_mode", graphMode, "NONE",
			"enforce_eager is stated, so vLLM runs every step eagerly with no CUDA graph "+
				"(vllm/config/vllm.py:2004-2009 at 4ba787ec9959)")
		graphMode = "NONE"
	}
	switch graphMode {
	case "PIECEWISE":
		p.Graphs.Decode, p.Graphs.Other = plan.LaunchPiecewise, plan.LaunchPiecewise
	case "FULL":
		p.Graphs.Decode, p.Graphs.Other = plan.LaunchGraph, plan.LaunchGraph
	case "FULL_AND_PIECEWISE":
		p.Graphs.Decode, p.Graphs.Other = plan.LaunchGraph, plan.LaunchPiecewise
	case "FULL_DECODE_ONLY":
		p.Graphs.Decode, p.Graphs.Other = plan.LaunchGraph, plan.LaunchEager
	case "NONE":
		p.Graphs.Decode, p.Graphs.Other = plan.LaunchEager, plan.LaunchEager
	default:
		return nil, fmt.Errorf(
			"cudagraph_mode %q is not one this kernel prices; add it with its launch "+
				"count rather than letting it fall through", eng.CUDAGraphMode)
	}
	uniformDecodeWidth := 1 + spec

	// The sequence cap and the capture ceiling (VllmConfig._set_cudagraph_sizes,
	// vllm/config/vllm.py:2388-2660 at v0.31.0).
	seqs := eng.MaxNumSeqs
	if seqs <= 0 {
		seqs = defaultMaxNumSeqs(c.chip)
		note("max_num_seqs", strconv.Itoa(seqs),
			"vLLM v0.31.0's OpenAI-API-server default for this part's memory "+
				"(EngineArgs.get_batch_defaults, vllm/engine/arg_utils.py:2858-2887); it sets "+
				"the cudagraph capture ceiling and the MLA context-gather chunk")
	}
	p.Batch.Requests = seqs
	batchedDPMoE := (backend == "deepep_low_latency" || backend == "nixl_ep") &&
		pl.EnableExpertParallel && lay.DP > 1
	defaultBatched := defaultMaxNumBatchedTokens(c.chip)
	if batchedDPMoE {
		defaultBatched = 256
	}
	if p.Graphs.Decode != plan.LaunchEager || p.Graphs.Other != plan.LaunchEager {
		e := enginefacts.CUDAGraphCaptureInput{Mode: graphMode, MaxCaptureSize: eng.MaxCUDAGraphCaptureSize,
			CaptureSizes: eng.CUDAGraphCaptureSizes, MaxNumBatchedTokens: eng.MaxNumBatchedTokens}
		if e.MaxNumBatchedTokens <= 0 {
			e.MaxNumBatchedTokens = defaultBatched
		}
		capture, stated, err := in.Rules.CUDAGraphCaptureSizes(e)
		if err != nil {
			return nil, fmt.Errorf("resolving the CUDA graph capture sizes: %w", err)
		}
		if stated {
			p.Graphs.Ceiling = units.Tokens(capture.Max)
			p.Graphs.Buckets = capture.Sizes
			if m := eng.MaxCUDAGraphCaptureSize; m != nil && *m != capture.Max {
				override("max_cudagraph_capture_size", strconv.Itoa(*m), strconv.Itoa(capture.Max),
					"vLLM v0.31.0 caps the capture at max_num_batched_tokens and "+
						"captures only the sizes on its ladder (1, 2, 4, every 8 below 256, "+
						"every 16 from it), so the largest graph that runs is this")
			}
			if len(capture.Dropped) > 0 {
				override("cudagraph_capture_sizes", fmt.Sprint(eng.CUDAGraphCaptureSizes),
					fmt.Sprint(capture.Sizes), "vLLM v0.31.0 drops the sizes above max_num_batched_tokens")
			}
		} else {
			platform := 512
			if c.chip.NVFP4Peak > 0 {
				platform = 1024
			}
			ceiling := min(seqs*uniformDecodeWidth*2, platform)
			p.Graphs.Ceiling = units.Tokens(ceiling)
			defaulted := e
			defaulted.MaxCaptureSize = &ceiling
			if d, _, err := in.Rules.CUDAGraphCaptureSizes(defaulted); err == nil {
				p.Graphs.Buckets = d.Sizes
			}
			note("max_cudagraph_capture_size", strconv.Itoa(ceiling),
				"vLLM v0.31.0's default, min(max_num_seqs x decode_query_len x 2, "+
					strconv.Itoa(platform)+") (VllmConfig._set_cudagraph_sizes, "+
					"vllm/config/vllm.py:2442-2460), as the deployment states neither "+
					"max_cudagraph_capture_size nor cudagraph_capture_sizes; a step above it "+
					"runs with no graph")
		}
	}

	// A stated dispatch backend that cannot take effect: with expert parallelism on and none
	// of dp > 1, pcp > 1 or a sequence-parallel MoE input, no backend's kernels are built
	// (vllm/model_executor/layers/fused_moe/config.py:1024-1027, all2all_utils.py:188-214).
	if b := eng.All2AllBackend; b != "" && stackRoutedExperts(g) > 0 &&
		pl.EnableExpertParallel && !lay.moeDispatches() {
		override("all2all_backend", b, "none",
			"expert parallelism with dp 1, pcp 1 and no sequence-parallel MoE input "+
				"runs no dispatch in vLLM v0.31.0: the experts run on the replicated tokens "+
				"and the tensor-parallel all-reduce combines them")
	}
	if e := eng.EPLB; e != nil && e.Enabled {
		p.MoE.RedundantExperts = e.NumRedundantExperts
	}
	p.MoE.Dispatch = dispatchKernel(backend)
	if stackRoutedExperts(g) > 0 && backend != eng.All2AllBackend && (eng.All2AllBackend != "" || backend != in.Rules.EffectiveAll2AllBackend("")) {
		// The plan states the backend the engine runs, which is not the one stated: a deprecated
		// alias or the binding to passthrough beside a self-communicating MoE backend.
		why := "the backend vLLM runs for the stated all2all_backend " +
			strconv.Quote(eng.All2AllBackend) + " beside moe_backend " + strconv.Quote(eng.MoEBackend) +
			" (vllm/config/parallel.py:537, vllm/config/kernel.py:194-227, " +
			"vllm/distributed/device_communicators/cuda_communicator.py:253-262 at 4ba787ec9959)"
		if backend == "passthrough" {
			why += "; the MoE kernel dispatches and combines the tokens itself, so that communication is inside the " +
				"kernel's price and is not priced as a separate collective"
		}
		note("all2all_backend", backend, why)
	}
	if p.MoE.Experts, err = expertsKernel(g, &c, eng.MoEBackend); err != nil {
		return nil, err
	}
	p.DecodeShards = decodeShards(arch, c.family, lay, in.Rules.MegaMoEBackend(eng.MoEBackend))
	if r, why := routingMode(g, &c, eng.MoEBackend); r != plan.RoutingNotStated || why != "" {
		p.MoE.Routing = r
		if why != "" {
			note("moe_routing", string(r), why)
		}
	}

	// The packed latent cell: the stated layout, or on a DSA model the one its backend
	// allocates, which can differ from the stated dtype; a layout no backend serves is refused
	// (sparseMLABackend).
	layout := eng.CacheDType
	if dsa || nopeSparseMLA(g, v4) {
		heads, headSize := 0, 0
		for _, a := range stackPrimaryAttentions(g) {
			if a.AttentionKind == model.AttentionSparseMLA && a.NumQHeads > 0 {
				heads = max(heads, a.NumQHeads)
				// The head size vLLM's MLA layer hands the selector (mla_attention.py:479-480).
				headSize = max(headSize, a.KVLoRARank+a.QKRopeHeadDim)
			}
		}
		preHopper, err := c.preHopper()
		if err != nil {
			return nil, err
		}
		be, dsaLayout, err := sparseMLABackend(sparseMLABackendRequest{
			cache:        eng.CacheDType,
			headsPerRank: heads / max(lay.TP, 1),
			headSize:     headSize,
			sm100:        c.chip.NVFP4Peak > 0,
			preHopper:    preHopper,
			tp:           lay.TP,
			dcp:          lay.DCP,
			pcp:          lay.PCP,
			dcpComm:      dc.CommBackend,
		})
		if err != nil {
			return nil, err
		}
		if be == backendFlashInferSM90 {
			note("sparse_mla_backend", be,
				"chosen as vLLM would choose it (head size "+strconv.Itoa(headSize)+"), which assumes "+
					"FlashInfer 0.6.18 or later is installed, the backend's own gate "+
					"(has_flashinfer_sm90_nope_mla; flashinfer_mla_sparse_sm90.py:194-199); "+
					"without it vLLM runs "+backendFlashAttnMLASparse+" (blis-vllm#278)")
			if p.Layers, err = sparseMLAKernels(g, &c, pl.PP, be, note); err != nil {
				return nil, err
			}
		}
		// The layout a backend allocates is stated for the DSA family; a NoPE sparse-MLA model
		// (GLM-5.3-Flash) is given the backend vLLM selects and the refusals it makes, and keeps the
		// stated cache and the graph's own cell (its block size and cell are not DSA's).
		if dsa {
			if st := eng.CacheDType; dsaLayout != st && st != "" {
				override("cache_dtype", st, dsaLayout, be+" serves this cache as "+dsaLayout+
					" (_canonicalize_sparse_mla_kv_cache_dtype, "+
					"vllm/model_executor/layers/attention/mla_attention.py:358-375)")
			}
			layout = dsaLayout
		}
	}
	p.Cache.LatentCellBytes = dsMLAStateBytes[layout]

	// Speculative decoding.
	if spec > 0 {
		s := eng.Speculative
		p.Drafts.Chain = spec
		p.Drafts.Adaptive = s.EnableAdaptiveVerification
		p.Drafts.OwnWorkspace = s.Method == "dspark"
		// A different method is a different drafter, which this graph does not describe, so it
		// is refused rather than priced as the graph's.
		if sp != nil {
			if s.Method != "" && sp.Method != "" && specMethod(s.Method) != specMethod(sp.Method) {
				return nil, fmt.Errorf("the deployment drafts with %q and the model graph states a %q "+
					"speculator; a %s drafter is not described by this graph, so it cannot be priced",
					s.Method, sp.Method, s.Method)
			}
			// extra_retained_tokens, max(0, num_prefill_lookahead_tokens - 1)
			// (vllm/v1/core/kv_cache_utils.py:2738-2744): num_speculative_tokens - 1 for
			// multi-module MTP, zero for every other drafter.
			if s.Method == "mtp" && min(len(sp.Stack.Expand()), spec) > 1 {
				p.Drafts.RetainedTokens = units.Tokens(spec - 1)
			}
		}
	}

	// Where an Engram table lives (blis-schemas deployment.EngramConfig).
	onHost, shards := lowerEngram(in, c.pool, eng, lay)
	p.Engram = plan.EngramPlacement{Host: onHost, Shards: shards}
	if graphRunsEngram(g, sp, c.speculatorKinds, spec > 0) {
		ec := eng.EngramConfig
		if ec == nil || ec.CPUOffload == nil {
			where := "HBM"
			if onHost {
				where = "host memory"
			}
			note("engram_cpu_offload", strconv.FormatBool(onHost),
				"unstated, so the release's EngramConfig.cpu_offload default places the Engram "+
					"tables in "+where+" (vllm/config/engram.py)")
		}
		if (ec == nil || ec.DPSharedMemory == nil) && lay.DP > 1 {
			note("engram_dp_shared_memory", strconv.Itoa(shards)+" shards",
				"unstated, so it resolves as the engine resolves it: on when cpu_offload is on, "+
					"use_thp off, dp above one and elastic EP off (resolve_dp_shared_memory, "+
					"vllm/config/engram.py:92-97); the kernel cannot see "+
					"whether /dev/shm holds the tables, where the engine falls back to splitting them")
		}
	}

	p.Microbatch = eng.DBO != nil && eng.DBO.Enabled
	out.Reported.CascadeAttention = eng.DisableCascadeAttn != nil && !*eng.DisableCascadeAttn
	p.MemoryFraction = eng.GPUMemoryUtilization

	// The token budget.
	p.Batch.Tokens = units.Tokens(eng.MaxNumBatchedTokens)
	if p.Batch.Tokens <= 0 {
		p.Batch.Tokens = units.Tokens(defaultBatched)
		why := "vLLM v0.31.0's OpenAI-API-server default for this part's memory " +
			"(EngineArgs.get_batch_defaults, vllm/engine/arg_utils.py:2858-2887); " +
			"the offline LLM class defaults higher on every part below 160 GiB"
		if batchedDPMoE {
			why = "vLLM v0.31.0's default under a batched data-parallel MoE " +
				"(DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP, vllm/config/scheduler.py:43; " +
				"vllm/engine/arg_utils.py:3046-3051), which also sizes the DeepEP " +
				"low-latency buffer"
		}
		note("max_num_batched_tokens", strconv.Itoa(defaultBatched), why)
	}
	return out, nil
}

// lowerMaxModelLen settles the context bound: a count is the bound; unstated, the model's
// derived window; auto, that window as the cap (vllm/config/model.py _get_and_verify_max_len).
func lowerMaxModelLen(mml vllmsettings.MaxModelLen, g *model.Graph, note func(name, value, reason string)) int {
	if n, ok := mml.Stated(); ok {
		return n
	}
	n := g.Global.DefaultWindow()
	switch {
	case mml.Auto() && n > 0:
		note("max_model_len", strconv.Itoa(n),
			"auto fits the bound to the KV memory the engine finds at startup, capped at the "+
				"model's derived window (vllm/config/model.py _get_and_verify_max_len); the "+
				"kernel cannot profile memory, so it takes the cap, which the engine lowers "+
				"where the cache cannot hold one sequence of it")
	case mml.Auto():
		note("max_model_len", "auto",
			"auto, on a graph that states no context_window: the bound is unknown, and the "+
				"terms that read it say what they take instead")
	case n > 0:
		note("max_model_len", strconv.Itoa(n),
			"unstated, so the engine derives it from the model configuration, which the "+
				"graph states as its context window")
	}
	return n
}

// lowerEngram settles where an engram_cpu_offload table lives and how many ways its heads split.
func lowerEngram(in Input, pool deployment.Pool, eng vllmsettings.Settings, lay layout) (onHost bool, shards int) {
	onHost = in.Rules.EngramOnHostByDefault()
	ec := eng.EngramConfig
	if ec != nil && ec.CPUOffload != nil {
		onHost = *ec.CPUOffload
	}
	dp := max(lay.DP, 1)
	shared := onHost && dp > 1
	if ec != nil && ec.UseTHP != nil && *ec.UseTHP {
		shared = false
	}
	if ec != nil && ec.DPSharedMemory != nil {
		shared = *ec.DPSharedMemory
	}
	local := pool.Parallel.DPLocal
	if local <= 0 {
		local = dp
	}
	shards = max(lay.TP, 1)
	if !shared && local > 1 {
		shards *= local
	}
	return onHost, shards
}

// expertsKernel names the kernel the routed experts' grouped GEMMs run on, in the plan's words,
// where the lowering knows it: block-FP8 experts on an SM90 part. vLLM v0.33.0 orders the FP8
// MoE backends and, for 128-block FP8 weights on SM90, moves Triton to the front at
// expert-parallel width 1 and FlashInfer's CUTLASS experts above it
// (model_executor/layers/fused_moe/oracle/fp8.py:79-130@4ba787ec); a stated moe_backend is
// taken as stated (:274-282). A stated backend outside those two is named by its own word, and
// the cost model prices it by weight format and records that. Any other format or part states
// none.
func expertsKernel(g *model.Graph, c *lowerContext, stated string) (plan.KernelImpl, error) {
	if !c.hopper() || stackRoutedExperts(g) == 0 || !blockFP8Experts(g) {
		return plan.KernelImpl{}, nil
	}
	// The engine normalises the name before it validates it (vllm/config/kernel.py
	// _normalize_moe_backend): lower case, dashes as underscores.
	backend := strings.ReplaceAll(strings.ToLower(stated), "-", "_")
	if backend == "" || backend == "auto" {
		backend = "triton"
		if c.lay.ExpertWidth > 1 {
			backend = "flashinfer_cutlass"
		}
	}
	// The build the release runs, from its kernel table (kernel_impls.json): the coefficient
	// lookup binds the routed term's ramp to it (VAL-8). A part the table does not cover states
	// the kernel unidentified, and the cost model reads no keyed entry for it.
	var member, library, variant string
	switch backend {
	case "triton":
		member, library, variant = "TRITON", "vllm-csrc", plan.ExpertsTriton.Variant
	case "flashinfer_cutlass":
		member, library, variant = "FLASHINFER_CUTLASS", plan.ExpertsFlashInferCutlass.Library, plan.ExpertsFlashInferCutlass.Variant
	default:
		return plan.KernelImpl{Library: backend}, nil
	}
	major, minor := capability(c.chip)
	return blisvllm.Implementation(plan.PrimMoEGroupedGEMM, blisvllm.MoESelection("fp8", member), library, variant, major, minor)
}

// sparseMLAKernels states, for each layer whose primary attention is a sparse MLA run by
// FLASHINFER_MLA_SPARSE_SM90, the kernel that backend runs for the prefill and for the decode: its
// FlashInferMLASparseSM90Impl builds a BatchMLAPagedAttentionWrapper with backend="fa3"
// (v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py:257-288@4ba787ec) and is supported on
// compute capability 9.x (:189-190). The registry keys the GLM-5.3 sparse-MLA family to that build
// (VAL-8), so without the statement the cost model reads no entry of it. The stage of a layer is
// the pipeline partition's, which the lowering does not compute: at pipeline width above 1 nothing is
// stated (the family is then not read, as before), and the lowering says so. No other sparse-MLA
// backend has a registry family keyed to its kernel, so none is stated. The draft model's layers
// (DraftLayers) are not stated either. The decode is always this kernel (forward_mqa); a prefill of at most
// the top-k tokens is dense multi-head attention in vLLM, which the pool names (kernelmodel
// sparseMLAPrefillSubstitution) because the plan states the primitive for the pool, not for a step.
func sparseMLAKernels(g *model.Graph, c *lowerContext, pp int, backend string, note func(name, value, reason string)) ([]plan.LoweredLayer, error) {
	major, minor := capability(c.chip)
	if backend != backendFlashInferSM90 || major != 9 {
		return nil, nil
	}
	if pp > 1 {
		note("sparse_mla_kernel", "unstated",
			"the pipeline stage of each layer is not computed here, so at pipeline-parallel width above 1 the "+
				"sparse-MLA kernel is not stated and no coefficient keyed to it is read")
		return nil, nil
	}
	byID := map[string]model.LayerKind{}
	for _, lk := range g.LayerKinds {
		byID[lk.ID] = lk
	}
	var out []plan.LoweredLayer
	for i, id := range g.Stack.Expand() {
		sparse := false
		for _, n := range byID[id].Nodes {
			if n.Op == model.OpAttention && n.ReadsEngineKV() {
				sparse = n.AttentionKind == model.AttentionSparseMLA
				break
			}
		}
		if !sparse {
			continue
		}
		for _, prim := range []plan.Primitive{plan.PrimAttentionPrefill, plan.PrimAttentionDecode} {
			k, err := blisvllm.Implementation(prim, blisvllm.AttentionSelection(backend), "flashinfer",
				"BatchMLAPagedAttentionWrapper[fa3]", major, minor)
			if err != nil {
				return nil, err
			}
			out = append(out, plan.LoweredLayer{Layer: i, Primitive: prim, KernelImpl: k})
		}
	}
	if len(out) > 0 {
		note("sparse_mla_kernel", out[0].KernelImpl.Library+" "+out[0].KernelImpl.Variant,
			"the kernel FlashInferMLASparseSM90Impl runs for the prefill and the decode of each sparse-MLA layer; the "+
				"registry's GLM-5.3 family is keyed to its build")
	}
	return out, nil
}

// capability is the chip's compute capability, major and minor; 0, 0 when it states none.
func capability(chip hardware.Chip) (int, int) {
	major, ok := chip.CapabilityMajor()
	if !ok {
		return 0, 0
	}
	_, m, _ := strings.Cut(chip.ComputeCapability, ".")
	minor, _ := strconv.Atoi(m)
	return major, minor
}

// blockFP8Experts reports whether every routed expert of the stack stores block-scaled FP8 weights:
// FP8 whose scale is not the 32-element microscaling one (MXFP8 runs another family of kernels).
func blockFP8Experts(g *model.Graph) bool {
	found := false
	for _, lk := range stackKinds(g.LayerKinds, g.Stack) {
		for _, n := range lk.Nodes {
			if n.Op != model.OpGroupedGEMM || n.TopK <= 0 || n.IsSharedExpert() {
				continue
			}
			d := n.WeightDType
			if d == "" {
				d = g.Global.WeightDType
			}
			if d != model.DTypeFP8 || (g.Global.WeightScale != nil && g.Global.WeightScale.Elements == 32) {
				return false
			}
			found = true
		}
	}
	return found
}

// resolvedAll2AllBackend is the backend the engine runs for the stated one beside the stated MoE
// backend. A pack that knows the binding to passthrough beside a self-communicating MoE backend
// (core/rulepack/pack ResolvedAll2AllBackend) applies it; any other rules value gives the effective backend.
func resolvedAll2AllBackend(r enginefacts.EngineRules, backend, moeBackend string) string {
	if b, ok := r.(interface {
		ResolvedAll2AllBackend(backend, moeBackend string) string
	}); ok {
		return b.ResolvedAll2AllBackend(backend, moeBackend)
	}
	return r.EffectiveAll2AllBackend(backend)
}

// refuseAll2AllBinding refuses what bind_passthrough_all2all_backend refuses
// (vllm/config/kernel.py:206-227@4ba787ec9959e566d3b587a796dff07673324da5): a FlashInfer MoE-EP
// backend beside an explicit all2all_backend other than the default or passthrough, and passthrough
// beside any other MoE backend. The rules pack refuses both too (moe-ep-backend-binds-passthrough),
// but a caller can reach the lowering without it (blis-vllm#271). A release whose pack does not know
// the binding refuses nothing here.
func refuseAll2AllBinding(r enginefacts.EngineRules, backend, moeBackend string) error {
	b, ok := r.(interface {
		SelfCommunicatingMoEBackend(string) (self, known bool)
	})
	if !ok {
		return nil
	}
	self, known := b.SelfCommunicatingMoEBackend(moeBackend)
	if !known {
		return nil
	}
	stated := r.EffectiveAll2AllBackend(backend)
	switch {
	case self && stated != r.EffectiveAll2AllBackend("") && stated != "passthrough":
		return fmt.Errorf("all2all_backend %q beside moe_backend %q: the MoE backend dispatches and combines itself, "+
			"and vLLM refuses any explicit all2all_backend with it; leave all2all_backend unset "+
			"(vllm/config/kernel.py:206-215, https://github.com/inference-sim/blis-vllm/issues/271)", backend, moeBackend)
	case !self && stated == "passthrough":
		return fmt.Errorf("all2all_backend passthrough beside moe_backend %q: passthrough is valid only beside a MoE backend "+
			"that dispatches and combines itself, and vLLM refuses it otherwise "+
			"(vllm/config/kernel.py:216-227, https://github.com/inference-sim/blis-vllm/issues/271)", moeBackend)
	}
	return nil
}

// dispatchKernel names the all-to-all kernel an MoE dispatch backend runs, in the plan's words:
// none for the gather-and-scatter dispatches, DeepEP's three by their own names, and any other
// by its library.
func dispatchKernel(backend string) plan.KernelImpl {
	switch backend {
	case "", "naive", "allgather_reducescatter", "passthrough":
		// passthrough runs no framework all-to-all: the MoE backend dispatches and combines
		// itself (vllm/config/parallel.py:232-234@4ba787ec).
		return plan.KernelImpl{}
	case "deepep_high_throughput":
		return plan.DispatchDeepEPHighThroughput
	case "deepep_low_latency":
		return plan.DispatchDeepEPLowLatency
	case "deepep_v2":
		return plan.DispatchDeepEPV2
	}
	return plan.KernelImpl{Library: backend}
}

// refuseV4Layouts refuses the DeepSeek-V4-family layouts vLLM v0.31.0 refuses at startup,
// which no field validation catches and which would otherwise price as a deployment that
// runs.
//
//   - V4.1's packed attention serves only fp8 caches: a cache dtype that resolves to neither
//     packed record is a ValueError ("DeepseekV4 packed KV layouts only support fp8
//     kv-cache", vllm/models/deepseek_v41/attention.py:147-155).
//   - A mega-MoE backend needs expert parallelism ("DeepSeek V4 MegaMoE currently requires
//     expert parallel", vllm/models/deepseek_v41/nvidia/model.py:630-635; the V4 model is
//     the same).
//   - V4.1 with Engram layers refuses dual-batch overlap ("DeepSeek V4.1 Engram does not
//     support DBO or microbatching", vllm/config/vllm.py).
//   - V4.1 refuses MTP drafting: its drafter is DSpark (the rules pack's
//     deepseek-v41-refuses-mtp).
func refuseV4Layouts(in Input, grafted []model.LayerKind, eng vllmsettings.Settings, cacheDType string) error {
	arch := in.Model.Global.Architecture
	if arch != enginefacts.ArchDeepseekV4 && arch != enginefacts.ArchDeepseekV41 {
		return nil
	}
	pool := in.Deployment.Pools[in.PoolIndex]
	family, _ := in.Chip.CapabilityMajor()
	if in.Rules.MegaMoEBackend(eng.MoEBackend) {
		if !pool.Parallel.EnableExpertParallel {
			return fmt.Errorf("moe_backend %q on %s needs expert parallelism: vLLM v0.31.0 "+
				"refuses mega-MoE without it", eng.MoEBackend, arch)
		}
		if family != 10 {
			return fmt.Errorf("moe_backend %q on %s needs an SM100 part: \"DeepGEMM MegaMoE "+
				"requires SM100 GPUs\" (vllm/models/deepseek_v4/nvidia/model.py:366-369)",
				eng.MoEBackend, in.Chip.Name)
		}
	}
	// The family's sparse attention backends declare no implementation class, so PCP support
	// reads as no and the engine's assertion stops it (vllm/v1/worker/cp_utils.py:31-38). Nor
	// does the family's attention implement decode-context sharding.
	if pool.Parallel.PCP > 1 {
		return fmt.Errorf("prefill-context parallelism (pcp %d) on %s: its sparse attention "+
			"backends do not support PCP, and vLLM v0.31.0 refuses it at startup",
			pool.Parallel.PCP, arch)
	}
	if pool.Parallel.DCP > 1 {
		return fmt.Errorf("decode-context parallelism (dcp %d) on %s: its attention "+
			"implements no DCP sharding, and its window cache asserts dcp_world_size == 1",
			pool.Parallel.DCP, arch)
	}
	if arch != enginefacts.ArchDeepseekV41 {
		return nil
	}
	if family == 9 && cacheDType == "nvfp4_ds_mla" {
		return fmt.Errorf("cache_dtype nvfp4_ds_mla on %s on SM90: \"nvfp4_ds_mla needs the "+
			"V4.1 KV records, which FlashMLA decodes only on SM100\" "+
			"(vllm/models/deepseek_v41/attention.py:513-517)", arch)
	}
	if (family == 9 || family == 10) && !strings.HasSuffix(cacheDType, "_ds_mla") {
		return fmt.Errorf("cache_dtype %q on %s: its packed attention serves only fp8 "+
			"caches, as fp8_ds_mla or nvfp4_ds_mla, and vLLM v0.31.0 refuses any other",
			eng.CacheDType, arch)
	}
	if s := eng.Speculative; s != nil && s.NumSpecTokens > 0 &&
		strings.Contains(strings.ToLower(s.Method), "mtp") {
		return fmt.Errorf("speculative method %q on %s: its drafter is DSpark, and vLLM "+
			"v0.31.0 refuses MTP for it", s.Method, arch)
	}
	if d := eng.DBO; d != nil && d.Enabled && kindsHaveEngram(grafted) {
		return fmt.Errorf("dual-batch overlap on %s with Engram layers: vLLM v0.31.0 "+
			"refuses it at startup", arch)
	}
	return nil
}

// stackKinds is the layer kinds of a stack in first-use order, as the plan builder walks them.
func stackKinds(kinds []model.LayerKind, st model.Stack) []model.LayerKind {
	byID := map[string]model.LayerKind{}
	for _, lk := range kinds {
		byID[lk.ID] = lk
	}
	seen := map[string]bool{}
	var out []model.LayerKind
	for _, id := range st.Expand() {
		if !seen[id] {
			seen[id] = true
			out = append(out, byID[id])
		}
	}
	return out
}

// stackHasRecurrent reports whether the target stack has a recurrent layer.
func stackHasRecurrent(g *model.Graph) bool {
	for _, lk := range stackKinds(g.LayerKinds, g.Stack) {
		for _, n := range lk.Nodes {
			if n.Op == model.OpRecurrentUpdate {
				return true
			}
		}
	}
	return false
}

// stackPrimaryAttentions is each target layer kind's primary attention: its first attention
// node that reads the engine's KV cache. A kind with none is left out.
func stackPrimaryAttentions(g *model.Graph) []model.Node {
	var out []model.Node
	for _, lk := range stackKinds(g.LayerKinds, g.Stack) {
		for _, n := range lk.Nodes {
			if n.Op == model.OpAttention && n.ReadsEngineKV() {
				out = append(out, n)
				break
			}
		}
	}
	return out
}

// stackRoutedExperts is the routed expert count of the target stack's widest routed layer.
func stackRoutedExperts(g *model.Graph) int {
	experts := 0
	for _, lk := range stackKinds(g.LayerKinds, g.Stack) {
		for _, n := range lk.Nodes {
			if n.Op == model.OpGroupedGEMM && n.TopK > 0 {
				experts = max(experts, n.Experts)
			}
		}
	}
	return experts
}

// graphRunsEngram reports whether a layer a step runs reads an Engram table: the target's
// stack, embedding and head, and the drafter's while the pool speculates. sp is the speculator
// that runs, and spKinds the layer kinds its stacks name.
func graphRunsEngram(g *model.Graph, sp *model.Speculator, spKinds []model.LayerKind, speculating bool) bool {
	engram := func(nodes []model.Node) bool {
		for _, n := range nodes {
			if n.Op == model.OpGather && n.ResidencyControl == model.ResidencyEngramCPUOffload {
				return true
			}
		}
		return false
	}
	groups := [][]model.Node{g.Head, g.Embedding}
	for _, lk := range stackKinds(g.LayerKinds, g.Stack) {
		groups = append(groups, lk.Nodes)
	}
	if sp != nil && speculating {
		for _, lk := range stackKinds(spKinds, sp.Stack) {
			groups = append(groups, lk.Nodes)
		}
		if sp.ContextStack != nil {
			for _, lk := range stackKinds(spKinds, *sp.ContextStack) {
				groups = append(groups, lk.Nodes)
			}
		}
		groups = append(groups, sp.Embedding, sp.Head, sp.SequentialHead)
	}
	for _, nodes := range groups {
		if engram(nodes) {
			return true
		}
	}
	return false
}

// kindsHaveEngram reports whether any of the layer kinds runs an Engram lookup.
func kindsHaveEngram(kinds []model.LayerKind) bool {
	for _, lk := range kinds {
		for _, n := range lk.Nodes {
			if n.Op == model.OpGather && n.ResidencyControl == model.ResidencyEngramCPUOffload {
				return true
			}
		}
	}
	return false
}
