// Original to blis-vllm.

package kernelmodel

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	latencykernel "github.com/inference-sim/blis-substrate/kernel"

	"github.com/inference-sim/blis-vllm/core/vllm"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// resolvedEngine is a pool's engine block with the settings its release resolves filled in,
// and where each resolved setting came from.
type resolvedEngine struct {
	engine engineconfig.Engine
	// sources names, for each setting the release resolves, "stated" or the rule that set it:
	// gpu_memory_utilization for every release this package knows (0.29.0, 0.31.0, 0.32.0), and
	// max_num_seqs, max_num_batched_tokens and block_size for 0.32.0.
	sources map[string]string
	// blockSizeErr is why an unstated block_size is not resolved, naming blis-vllm#68; nil when
	// it is stated or resolved.
	blockSizeErr error
}

// resolveEngine resolves the pool's engine settings as its release does. For vLLM 0.32.0 an
// unstated max_num_seqs or max_num_batched_tokens takes vLLM's own default for the GPU
// (vllm.ResolveBatch, vllm/engine/arg_utils.py:2893-2985, 3097-3186@5df63a72). A deployment
// generated under X19 states only what the run stated, so these defaults are the engine's to
// fill, not the generator's. For 0.29.0 and 0.31.0 the block is returned as stated, as before.
func resolveEngine(in latencykernel.Inputs, e engineconfig.Engine) resolvedEngine {
	r := resolvedEngine{engine: e}
	release := in.Scenario.EngineRelease()
	if d, ok := vllm.DefaultMemoryUtilization(release); ok {
		r.sources = map[string]string{"gpu_memory_utilization": vllm.SourceStated}
		if e.GPUMemoryUtilization == 0 {
			// 0 reads as unstated: the field is a number with omitempty, and the schema says
			// "0 means unset" (rules/engineconfig/validate.go:65). A negative value or one above 1
			// is not unstated, and stays what it is, for KVMemory to refuse.
			r.engine.GPUMemoryUtilization = d.Value
			r.sources["gpu_memory_utilization"] = d.Source
		}
	}
	if release != vllm.Version {
		return r
	}
	pool := in.Deployment.Pools[in.PoolIndex]
	maxLen, _ := e.MaxModelLen.Stated()
	if maxLen == 0 {
		// An unstated window is the model's own, which the graph states
		// (vllm/config/model.py _get_and_verify_max_len@5df63a72).
		maxLen = in.Model.Global.DefaultWindow()
	}
	backend := e.All2AllBackend
	if in.Rules != nil {
		backend = in.Rules.EffectiveAll2AllBackend(backend)
	}
	b := vllm.ResolveBatch(vllm.BatchInput{
		MaxNumSeqs:          e.MaxNumSeqs,
		MaxNumBatchedTokens: e.MaxNumBatchedTokens,
		DeviceMemoryBytes:   int64(in.Chip.MemoryGiB * float64(int64(1)<<30)),
		DeviceName:          in.Chip.Name,
		// use_batched_dp_moe (vllm/config/parallel.py:792-801@5df63a72).
		BatchedDPMoE: (backend == "deepep_low_latency" || backend == "nixl_ep") &&
			pool.Parallel.EnableExpertParallel && pool.Parallel.DP > 1,
		// The deployment schema has no performance_mode, so it is vLLM's default, balanced
		// (vllm/config/vllm.py:433@5df63a72).
		ThroughputMode: false,
		// The v0.32.0 engine runs chunked prefill on (core/vllm config.go).
		ChunkedPrefill: true,
		MaxModelLen:    maxLen,
	})
	r.engine.MaxNumSeqs, r.engine.MaxNumBatchedTokens = b.MaxNumSeqs, b.MaxNumBatchedTokens
	r.sources["max_num_seqs"] = b.MaxNumSeqsSource
	r.sources["max_num_batched_tokens"] = b.MaxNumBatchedTokensSource
	r.sources["block_size"] = vllm.SourceStated
	if e.BlockSize <= 0 {
		bs, err := resolveBlockSize(in, e, pool)
		if err != nil {
			r.blockSizeErr = err
			delete(r.sources, "block_size")
		} else {
			r.engine.BlockSize, r.sources["block_size"] = bs.Size, bs.Source
		}
	}
	return r
}

// resolveBlockSize resolves an unstated block_size as vLLM v0.32.0 does, for the models, GPUs
// and KV dtypes the block-size oracle resolves (core/vllm.ResolveBlockSize).
func resolveBlockSize(in latencykernel.Inputs, e engineconfig.Engine, pool deployment.Pool) (vllm.BlockSize, error) {
	kv := e.CacheDType
	if kv == "" {
		kv = "auto"
	}
	if in.Rules != nil {
		// The KV dtype that runs: the release's rules resolve a requested one for the model's
		// architecture, the chip's family and the TP width (rules.EngineRules.ResolveCacheDType).
		fam, _ := in.Chip.CapabilityMajor()
		kv = in.Rules.ResolveCacheDType(in.Model.Global.Architecture, fam, max(pool.Parallel.TP, 1), kv)
	}
	return vllm.ResolveBlockSize(vllm.BlockSizeInput{
		Model:               in.Scenario.Model,
		Capability:          in.Chip.ComputeCapability,
		KVCacheDType:        kv,
		Speculative:         e.Speculative != nil,
		ContextParallel:     max(pool.Parallel.DCP, 1) > 1 || max(pool.Parallel.PCP, 1) > 1,
		Env:                 e.Env,
		UsesTRTLLMAttention: e.UseTRTLLMAttention != nil,
		MLAPrefillBackend:   e.MLAPrefillBackend,
	})
}

// Engine reports the engine settings of the pool this kernel prices, as its release resolves
// them.
//
// A caller configuring a simulator needs the same block size, token budget and sequence cap
// the kernel resolved against. Otherwise the two would describe different deployments, and the
// numbers would not reveal it: a scheduler admitting 512 sequences while the kernel priced a
// 256-sequence engine.
//
// An unstated gpu_memory_utilization is the release's default, 0.92, in all three releases
// (memoryUtilizationDefaults). For vLLM 0.32.0 an unstated max_num_seqs or max_num_batched_tokens is vLLM's own default
// for the GPU (resolveEngine), and EngineSources says which values were stated. An unstated
// block_size is vLLM's own too, for the models, GPUs and KV dtypes the block-size oracle
// resolves (core/vllm.ResolveBlockSize); for any other it is refused, naming why and
// https://github.com/inference-sim/blis-vllm/issues/68. For 0.29.0 and 0.31.0 it returns the
// deployment's own values, and refuses an unstated max_num_seqs, max_num_batched_tokens or
// block_size, as before.
//
// The settings live on the DEPLOYMENT: engine knobs are tunable configuration chosen against
// a scenario, not part of the problem it states, so the error text says "deployment".
func (m *Model) Engine() (engineconfig.Engine, error) {
	e := m.resolved.engine
	if e.BlockSize <= 0 {
		if m.resolved.blockSizeErr != nil {
			return e, fmt.Errorf("kernelmodel: deployment %w", m.resolved.blockSizeErr)
		}
		return e, fmt.Errorf("kernelmodel: deployment states no block_size")
	}
	if e.MaxNumSeqs <= 0 {
		return e, fmt.Errorf("kernelmodel: deployment states no max_num_seqs")
	}
	if e.MaxNumBatchedTokens <= 0 {
		return e, fmt.Errorf("kernelmodel: deployment states no max_num_batched_tokens")
	}
	return e, nil
}

// EngineSources reports where each setting Engine resolves came from: "stated", or the rule
// that set it, with its citation. It names max_num_seqs and max_num_batched_tokens for a
// vLLM 0.32.0 pool, and nothing for a release whose settings are returned as stated.
func (m *Model) EngineSources() map[string]string {
	out := make(map[string]string, len(m.resolved.sources))
	for k, v := range m.resolved.sources {
		out[k] = v
	}
	return out
}
