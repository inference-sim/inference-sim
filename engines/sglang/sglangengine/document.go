// Original to blis-sglang.

package sglangengine

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/kernel"

	blissglang "github.com/inference-sim/blis-sglang"
	"github.com/inference-sim/blis-sglang/config"
	"github.com/inference-sim/blis-sglang/core/sglang"
	"github.com/inference-sim/blis-sglang/lower"
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
	// Env is the process environment the settings read; the zero value is the defaults.
	Env *sglang.Env
}

// Host opens the pool a spec selects.
type Host interface {
	Open(s engine.Spec) (Pool, error)
}

// RegisterWith makes the factory available under its kind, building each engine from the pool the
// host opens for its spec.
func RegisterWith(h Host) { engine.Register(FactoryFor(h)) }

// FactoryFor builds engines of kind "sglang" from a deployment document's engine block.
func FactoryFor(h Host) engine.Factory { return factory{host: h} }

// newFromDocument resolves the document's ServerArgs as SGLang resolves them, sizes the pools,
// and builds the engine. Anything the engine does not model is refused by name.
func newFromDocument(h Host, s engine.Spec) (engine.Engine, error) {
	if s.UpstreamVersion != "" && s.UpstreamVersion != blissglang.Upstream.Version {
		return nil, fmt.Errorf("sglangengine: this build models SGLang %s, not %s", blissglang.Upstream.Version, s.UpstreamVersion)
	}
	var args map[string]any
	if err := s.Config.Decode(&args); err != nil {
		return nil, fmt.Errorf("sglangengine: the engine block: %w", err)
	}
	pool, err := h.Open(s)
	if err != nil {
		return nil, err
	}
	res, err := config.Resolve(config.Input{Args: args, Chip: pool.Chip, Model: pool.Model})
	if err != nil {
		return nil, err
	}
	a := res.Args
	if err := refuseUnmodelled(&a); err != nil {
		return nil, err
	}
	sa, err := schedArgs(&a, s.Seed)
	if err != nil {
		return nil, err
	}
	in := sglang.PoolInput{PreModelLoadGiB: pool.PreModelLoadGiB, MemFractionStatic: *a.MemFractionStatic,
		PageSize: sa.PageSize, MaxTotalTokens: a.MaxTotalTokens, MaxRunningRequests: a.MaxRunningRequests,
		ContextLen: pool.ContextLen, AttnDPSize: max(a.AttnDpSize, 1)}
	rt, err := sglang.PoolPlan(pool.Kernel, in)
	if err != nil {
		return nil, err
	}
	rt.EOSTokenIDs, rt.VocabSize = pool.EOSTokenIDs, pool.VocabSize
	env := sglang.DefaultEnv()
	if pool.Env != nil {
		env = *pool.Env
	}
	cfg, err := sglang.SchedConfig(sa, rt, env)
	if err != nil {
		return nil, err
	}
	var buckets []int
	if !a.DisableCudaGraph {
		if buckets, err = lower.CaptureBatchSizes(&a, rt.ReqPoolSize, 1); err != nil {
			return nil, err
		}
	}
	list, err := blissglang.Fields()
	if err != nil {
		return nil, err
	}
	return sglang.New(cfg, &pricer{k: pool.Kernel, buckets: buckets}, list)
}

type factory struct{ host Host }

func (factory) Kind() string { return Kind }

func (f factory) New(s engine.Spec) (engine.Engine, error) { return newFromDocument(f.host, s) }

// refuseUnmodelled names what a resolved configuration asks for that the engine does not run.
func refuseUnmodelled(a *config.ServerArgs) error {
	switch {
	case a.SpeculativeAlgorithm != nil:
		return fmt.Errorf("sglangengine: speculative decoding is not modelled by the engine (inference-sim/blis-sglang#68)")
	case a.EnableDpAttention || a.DpSize > 1:
		return fmt.Errorf("sglangengine: DP attention is not modelled by the engine (inference-sim/blis-sglang#70)")
	case a.PpSize > 1:
		return fmt.Errorf("sglangengine: pipeline parallelism is not modelled by the engine (inference-sim/blis-sglang#72)")
	case a.DisaggregationMode != "null":
		return fmt.Errorf("sglangengine: P/D disaggregation is not modelled by the engine (inference-sim/blis-sglang#64)")
	case a.EnableHierarchicalCache:
		return fmt.Errorf("sglangengine: HiCache is not modelled by the engine (inference-sim/blis-sglang#37)")
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
		EnablePriorityScheduling: a.EnablePriorityScheduling, DisablePriorityPreemption: a.DisablePriorityPreemption,
		PriorityPreemptionThreshold: a.PrioritySchedulingPreemptionThreshold, ScheduleLowPriorityValuesFirst: a.ScheduleLowPriorityValuesFirst,
		AbortOnPriorityWhenDisabled: a.AbortOnPriorityWhenDisabled, MaxQueuedRequests: a.MaxQueuedRequests,
		PrefillMaxRequests: a.PrefillMaxRequests, RetractionPolicy: a.RetractionPolicy, AllowAutoTruncate: a.AllowAutoTruncate,
		RandomSeed: rs, DisaggregationMode: a.DisaggregationMode, PPSize: a.PpSize, PPMaxMicroBatchSize: a.PpMaxMicroBatchSize,
		RadixEvictionPolicy: a.RadixEvictionPolicy, EnableMetrics: a.EnableMetrics, DecodeLogInterval: a.DecodeLogInterval,
	}, nil
}

// pricer prices a pass with the host's cost model. A decode-only pass is padded to the CUDA graph
// bucket that holds it (decode_cuda_graph_runner.py, base_cuda_graph_runner.py:166-180).
type pricer struct {
	k       kernel.Kernel
	buckets []int
}

func (p *pricer) StepTime(b engine.Batch) engine.Duration {
	// A request that schedules one token is a decode: SGLang's decode pass runs one token per request
	// (no speculation is modelled, #68), so the kernel counts a request as a prefill only above one.
	// With the threshold left at zero the kernel prices every decode request through the prefill
	// attention path (kernel/step.go: `Scheduled > DecodeThreshold`). blis-vllm sets 8 for vLLM's own
	// classifier (compat/kernelmodel DecodeThreshold); SGLang has no such short-chunk class.
	kb := kernel.Batch{Reqs: make([]kernel.ReqShape, len(b.Reqs)), DecodeThreshold: 1}
	decode := len(b.Reqs) > 0
	for i, r := range b.Reqs {
		kb.Reqs[i] = kernel.ReqShape{Scheduled: r.Tokens, Computed: r.Computed, PromptLen: r.PromptLen, CachedTokens: r.CachedTokens}
		decode = decode && r.Tokens == 1
	}
	if decode && len(p.buckets) > 0 {
		if padded, ok := lower.PadToBucket(len(b.Reqs), p.buckets); ok {
			kb.PadTokens = padded - len(b.Reqs)
		}
	}
	// NoOverlap is the edge blis-vllm measured nearly unbiased (compat/kernelmodel); microseconds,
	// at least one so a pass always takes time.
	return max(engine.Duration(p.k.StepTime(kb).NoOverlap.Microseconds()), 1)
}
