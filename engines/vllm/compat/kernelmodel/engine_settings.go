// Original to blis-vllm.

package kernelmodel

import (
	"fmt"

	latencykernel "github.com/inference-sim/blis-substrate/kernel"

	"github.com/inference-sim/blis-vllm/core/vllm"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// resolvedEngine is a pool's engine block with the settings its release resolves filled in,
// and where each resolved setting came from.
type resolvedEngine struct {
	engine engineconfig.Engine
	// sources names, for each of max_num_seqs and max_num_batched_tokens, "stated" or the rule
	// that set it. Empty for a release this package does not resolve (0.29.0, 0.31.0).
	sources map[string]string
}

// resolveEngine resolves the pool's engine settings as its release does. For vLLM 0.32.0 an
// unstated max_num_seqs or max_num_batched_tokens takes vLLM's own default for the GPU
// (vllm.ResolveBatch, vllm/engine/arg_utils.py:2893-2985, 3097-3186@5df63a72). A deployment
// generated under X19 states only what the run stated, so these defaults are the engine's to
// fill, not the generator's. For 0.29.0 and 0.31.0 the block is returned as stated, as before.
func resolveEngine(in latencykernel.Inputs, e engineconfig.Engine) resolvedEngine {
	r := resolvedEngine{engine: e}
	if in.Scenario.EngineRelease() != vllm.Version {
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
	r.sources = map[string]string{
		"max_num_seqs":           b.MaxNumSeqsSource,
		"max_num_batched_tokens": b.MaxNumBatchedTokensSource,
	}
	return r
}

// Engine reports the engine settings of the pool this kernel prices, as its release resolves
// them.
//
// A caller configuring a simulator needs the same block size, token budget and sequence cap
// the kernel resolved against, or the two would describe different deployments: a scheduler
// admitting 512 sequences while the kernel priced a 256-sequence engine is not a
// disagreement the numbers would reveal.
//
// For vLLM 0.32.0 an unstated max_num_seqs or max_num_batched_tokens is vLLM's own default
// for the GPU (resolveEngine), and EngineSources says which values were stated. An unstated
// block_size is still refused: vLLM v0.32.0 picks it from the attention backends it selects
// for the model, GPU and KV dtype, and aligns a hybrid model's block to its recurrent page
// (Platform.update_block_size_for_backend, vllm/platforms/interface.py:745-785@5df63a72),
// which is not yet ported (https://github.com/inference-sim/blis-vllm/issues/68). For 0.29.0
// and 0.31.0 it returns the deployment's own values, and refuses any of the three unstated,
// as before.
//
// The settings live on the DEPLOYMENT: engine knobs are tunable configuration chosen against
// a scenario, not part of the problem it states, so the error text says "deployment".
func (m *Model) Engine() (engineconfig.Engine, error) {
	e := m.resolved.engine
	if e.BlockSize <= 0 {
		if m.resolved.sources != nil {
			return e, fmt.Errorf("kernelmodel: deployment states no block_size; vLLM v0.32.0 resolves it from the " +
				"attention backends it selects (vllm/platforms/interface.py:745-785@5df63a72), which blis-vllm " +
				"does not resolve yet (https://github.com/inference-sim/blis-vllm/issues/68)")
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
