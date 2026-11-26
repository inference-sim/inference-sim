// Original to blis-sglang.

package sglangengine

import (
	"fmt"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"

	blissglang "github.com/inference-sim/inference-sim/engines/sglang"
	"github.com/inference-sim/inference-sim/engines/sglang/config"
	"github.com/inference-sim/inference-sim/engines/sglang/core/sglang"
	"github.com/inference-sim/inference-sim/engines/sglang/lower"
)

// Pool is what a host holds for the pool a spec selects and an engine cannot read through the
// contracts: the pool's cost model, the device and the model it runs, as SGLang's resolution
// reads them. The host opens its catalog and registry; the engine imports neither (X7).
type Pool struct {
	Kernel kernel.Kernel
	Chip   config.Chip
	Model  config.ModelFacts
	// PreModelLoadGiB is the device's free memory before the weights load.
	PreModelLoadGiB float64
	ContextLen      int
	EOSTokenIDs     []int64
	VocabSize       int
	// Args are the pool's ServerArgs fields by SGLang's names, as the host lowered the pool with
	// them (the engine block with the pool's tensor- and data-parallel widths: lower.Args). Nil
	// reads the spec's engine block alone.
	Args map[string]any
	// Env is the process environment the settings read; the zero value is the defaults.
	Env *sglang.Env
}

// RootScenario is the key of engine.Spec.Roots that names the scenario file a host opens the
// pool from.
const RootScenario = "scenario"

// Host opens the pool a spec selects.
type Host interface {
	Open(s engine.Spec) (Pool, error)
}

// RegisterWith makes the factory available under its kind, building each engine from the pool the
// host opens for its spec.
func RegisterWith(h Host) { engine.Register(FactoryFor(h)) }

// FactoryFor builds engines of kind "sglang" from a deployment document's engine block.
func FactoryFor(h Host) engine.Factory { return factory{host: h} }

// resolution is a pool's configuration as SGLang resolves it: the ServerArgs, the process
// environment its settings read, the scheduler's arguments and the pool sizes they produce. The
// engine and the memory report are both built from it, so neither describes a pool the other did
// not size.
type resolution struct {
	pool Pool
	args config.ServerArgs
	env  sglang.Env
	sa   sglang.SchedArgs
	rt   sglang.Runtime
}

// resolvePool opens the pool s selects and resolves it (resolveFromPool).
func resolvePool(h Host, s engine.Spec) (resolution, error) {
	// The release check runs before anything is opened: a spec naming a release this build does
	// not model is refused without sizing (and recording) a pool the run can never use.
	if err := checkUpstreamVersion(s); err != nil {
		return resolution{}, err
	}
	pool, err := h.Open(s)
	if err != nil {
		return resolution{}, err
	}
	return resolveFromPool(pool, s)
}

// checkUpstreamVersion refuses a spec that names a release this build does not model. It is
// called before a pool is opened and again where an already-opened pool is resolved, so both
// paths refuse by name.
func checkUpstreamVersion(s engine.Spec) error {
	if s.UpstreamVersion != "" && s.UpstreamVersion != blissglang.Upstream.Version {
		return fmt.Errorf("sglangengine: this build models SGLang %s, not %s", blissglang.Upstream.Version, s.UpstreamVersion)
	}
	return nil
}

// resolveFromPool resolves an opened pool's ServerArgs as SGLang resolves them, and sizes the
// pools. Anything the engine does not model is refused by name.
func resolveFromPool(pool Pool, s engine.Spec) (resolution, error) {
	if err := checkUpstreamVersion(s); err != nil {
		return resolution{}, err
	}
	args := pool.Args
	var err error
	if args == nil {
		if err := s.Config.Decode(&args); err != nil {
			return resolution{}, fmt.Errorf("sglangengine: the engine block: %w", err)
		}
	}
	// The engine block's env map is the process environment the scheduler's settings read: it is
	// not a ServerArgs field, so it comes out of the arguments before they are resolved.
	env := sglang.DefaultEnv()
	if pool.Env != nil {
		env = *pool.Env
	}
	if raw, ok := args["env"]; ok {
		m, isMap := raw.(map[string]any)
		if !isMap {
			return resolution{}, fmt.Errorf("sglangengine: the engine block's env must be a map of variable names to values, got %T", raw)
		}
		if env, err = sglang.EnvFrom(env, m); err != nil {
			return resolution{}, fmt.Errorf("sglangengine: %w", err)
		}
		rest := make(map[string]any, len(args))
		for k, v := range args {
			if k != "env" {
				rest[k] = v
			}
		}
		args = rest
	}
	res, err := config.Resolve(config.Input{Args: args, Chip: pool.Chip, Model: pool.Model})
	if err != nil {
		return resolution{}, err
	}
	a := res.Args
	if err := config.CheckServerArgs(&a); err != nil {
		return resolution{}, err
	}
	if err := refuseUnmodelled(&a); err != nil {
		return resolution{}, err
	}
	sa, err := schedArgs(&a, s.Seed)
	if err != nil {
		return resolution{}, err
	}
	in := sglang.PoolInput{PreModelLoadGiB: pool.PreModelLoadGiB, MemFractionStatic: *a.MemFractionStatic,
		PageSize: sa.PageSize, MaxTotalTokens: a.MaxTotalTokens, MaxRunningRequests: a.MaxRunningRequests,
		ContextLen: pool.ContextLen, AttnDPSize: max(a.AttnDpSize, 1)}
	rt, err := sglang.PoolPlan(pool.Kernel, in)
	if err != nil {
		return resolution{}, err
	}
	rt.EOSTokenIDs, rt.VocabSize = pool.EOSTokenIDs, pool.VocabSize
	return resolution{pool: pool, args: a, env: env, sa: sa, rt: rt}, nil
}

// newFromDocument resolves the document's ServerArgs as SGLang resolves them, sizes the pools,
// and builds the engine (resolvePool), then derives the scheduler's settings and capture buckets
// from that one resolution.
func newFromDocument(h Host, s engine.Spec) (engine.Engine, error) {
	r, err := resolvePool(h, s)
	if err != nil {
		return nil, err
	}
	a, rt := r.args, r.rt
	cfg, err := sglang.SchedConfig(r.sa, rt, r.env)
	if err != nil {
		return nil, err
	}
	var buckets []int
	// A verify pass carries the target's whole draft width per request (num_draft_tokens), and the
	// decode runner captures its batch sizes at that width (base_cuda_graph_runner.py:65-110).
	width := max(r.sa.SpeculativeNumDraftTokens, 1)
	if !a.DisableCudaGraph {
		if buckets, err = lower.CaptureBatchSizes(&a, rt.ReqPoolSize, width); err != nil {
			return nil, err
		}
	}
	list, err := blissglang.Fields()
	if err != nil {
		return nil, err
	}
	return sglang.New(cfg, &pricer{k: r.pool.Kernel, buckets: buckets, width: width}, list)
}

type factory struct{ host Host }

func (factory) Kind() string { return Kind }

func (f factory) New(s engine.Spec) (engine.Engine, error) { return newFromDocument(f.host, s) }

// refuseUnmodelled names what a resolved configuration asks for that the engine does not run.
func refuseUnmodelled(a *config.ServerArgs) error {
	switch {
	case a.SpeculativeEagleTopk != nil && *a.SpeculativeEagleTopk > 1:
		return fmt.Errorf("sglangengine: tree drafting and tree verification (speculative_eagle_topk %d) are not modelled by the engine (inference-sim/inference-sim#2355)", *a.SpeculativeEagleTopk)
	case a.EnableDpAttention || a.DpSize > 1:
		return fmt.Errorf("sglangengine: DP attention is not modelled by the engine (inference-sim/inference-sim#2356)")
	case a.PpSize > 1:
		return fmt.Errorf("sglangengine: pipeline parallelism is not modelled by the engine (inference-sim/inference-sim#2358)")
	case a.DisaggregationMode != "null":
		return fmt.Errorf("sglangengine: P/D disaggregation is not modelled by the engine (inference-sim/inference-sim#2351)")
	case a.EnableHierarchicalCache:
		return fmt.Errorf("sglangengine: HiCache is not modelled by the engine (inference-sim/inference-sim#2341)")
	}
	return nil
}

func schedArgs(a *config.ServerArgs, seed int64) (sglang.SchedArgs, error) {
	if a.PageSize == nil || a.MemFractionStatic == nil {
		return sglang.SchedArgs{}, fmt.Errorf("sglangengine: page_size and mem_fraction_static must resolve")
	}
	rs := int(seed)
	if a.RandomSeed != nil {
		rs = *a.RandomSeed
	}
	return sglang.SchedArgs{
		PageSize: *a.PageSize, SchedulePolicy: a.SchedulePolicy, MaxPrefillTokens: a.MaxPrefillTokens,
		ChunkedPrefillSize: a.ChunkedPrefillSize, EnableMixedChunk: a.EnableMixedChunk,
		DisableRadixCache: a.DisableRadixCache, DisableOverlapSchedule: a.DisableOverlapSchedule,
		ScheduleConservativeness: a.ScheduleConservativeness, PrefillDecodeInterval: a.PrefillDecodeInterval,
		PrefillDecodeIntervalMinBs: a.PrefillDecodeIntervalMinBs,
		EnablePriorityScheduling:   a.EnablePriorityScheduling, DisablePriorityPreemption: a.DisablePriorityPreemption,
		PriorityPreemptionThreshold: a.PrioritySchedulingPreemptionThreshold, ScheduleLowPriorityValuesFirst: a.ScheduleLowPriorityValuesFirst,
		AbortOnPriorityWhenDisabled: a.AbortOnPriorityWhenDisabled, MaxQueuedRequests: a.MaxQueuedRequests,
		PrefillMaxRequests: a.PrefillMaxRequests, RetractionPolicy: a.RetractionPolicy, AllowAutoTruncate: a.AllowAutoTruncate,
		RandomSeed: rs, DisaggregationMode: a.DisaggregationMode, PPSize: a.PpSize, PPMaxMicroBatchSize: a.PpMaxMicroBatchSize,
		RadixEvictionPolicy: a.RadixEvictionPolicy, EnableMetrics: a.EnableMetrics, DecodeLogInterval: a.DecodeLogInterval,
		SpeculativeNumSteps:       specValue(a.SpeculativeAlgorithm, a.SpeculativeNumSteps),
		SpeculativeNumDraftTokens: specValue(a.SpeculativeAlgorithm, a.SpeculativeNumDraftTokens),
	}, nil
}

// specValue is a speculative setting when an algorithm is on, and zero otherwise.
func specValue(algorithm *string, v *int) int {
	if algorithm == nil || v == nil {
		return 0
	}
	return *v
}

// pricer prices a pass with the host's cost model. A decode-only pass is padded to the CUDA graph
// bucket that holds it (decode_cuda_graph_runner.py, base_cuda_graph_runner.py:166-180).
type pricer struct {
	k       kernel.Kernel
	buckets []int
	// width is the tokens a decode request carries in a pass: 1, or num_draft_tokens under
	// speculation (the verify pass feeds the pending token and the drafts after it).
	width int
}

func (p *pricer) StepTime(b engine.Batch) engine.Duration {
	// NoOverlap is the edge the vLLM engine measured nearly unbiased (engines/vllm/core/kernelmodel); microseconds,
	// at least one so a pass always takes time.
	return max(engine.Duration(p.k.StepTime(p.shape(b)).NoOverlap.Microseconds()), 1)
}

// shape is the batch the cost model prices for a pass.
func (p *pricer) shape(b engine.Batch) kernel.Batch {
	// A request that schedules its decode width is a decode: SGLang's decode pass runs one token per
	// request, a verify pass num_draft_tokens, so the kernel counts a request as a prefill only above it.
	// With the threshold left at zero the kernel prices every decode request through the prefill
	// attention path (kernel/step.go: `Scheduled > DecodeThreshold`). the vLLM engine sets 8 for its own
	// classifier (engines/vllm/core/kernelmodel DecodeThreshold); SGLang has no such short-chunk class.
	width := max(p.width, 1)
	kb := kernel.Batch{Reqs: make([]kernel.ReqShape, len(b.Reqs)), DecodeThreshold: width}
	decode := len(b.Reqs) > 0
	for i, r := range b.Reqs {
		kb.Reqs[i] = kernel.ReqShape{Scheduled: r.Tokens, Computed: r.Computed, PromptLen: r.PromptLen, CachedTokens: r.CachedTokens}
		decode = decode && r.Tokens == width && r.Drafts == width-1
	}
	if decode && len(p.buckets) > 0 {
		if padded, ok := lower.PadToBucket(len(b.Reqs), p.buckets); ok {
			kb.PadTokens = (padded - len(b.Reqs)) * width
		}
	}
	return kb
}
