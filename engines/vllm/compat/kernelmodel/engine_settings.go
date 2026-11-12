// Original to blis-vllm.

package kernelmodel

import (
	"fmt"
	"strconv"

	subrules "github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	latencykernel "github.com/inference-sim/blis-substrate/kernel"

	"github.com/inference-sim/blis-vllm/core/vllm"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
	"github.com/inference-sim/blis-vllm/rules/v0_32"
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
	// mamba is the spec the model's recurrent layers build, and the block size vLLM aligns a
	// hybrid model to (planMamba); nil for a model with no recurrent layer. mambaErr is why they
	// are not mapped.
	mamba    *mambaPlan
	mambaErr error
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
	// An unstated cache dtype is "auto", which an architecture's own hook can change on the
	// chip that runs it (GlmMoeDsaForCausalLM on SM10x: fp8_e4m3, rules/v0_32). A stated dtype is
	// never changed here. A DeepSeek-V4 or V4.1 graph is left to the kernel, which resolves that
	// packed layout itself. Every other graph is resolved here, whether it names its architecture
	// (the catalog's GLM-5 graphs do since the layer-10 pin) or is named by its config digest.
	if in.Rules != nil && in.Model != nil && !kernelResolvesCacheDType(in.Model.Global.Architecture) && (e.CacheDType == "" || e.CacheDType == "auto") {
		fam, _ := in.Chip.CapabilityMajor()
		if arch := architectureOf(in.Model); arch != "" {
			if kv := in.Rules.ResolveCacheDType(arch, fam, max(in.Deployment.Pools[in.PoolIndex].Parallel.TP, 1), "auto"); kv != "auto" {
				r.engine.CacheDType = kv
				r.sources["cache_dtype"] = "vllm/model_executor/models/config.py:43-55@5df63a72 changes an unstated cache dtype to " + kv +
					" for " + arch + " on compute capability " + strconv.Itoa(fam) + ".x (rules pack CacheDTypeResolutions)"
			}
		}
	}
	if release != vllm.Version {
		if hasRecurrent(in.Model) {
			r.mambaErr = fmt.Errorf("recurrent layers: their KV cache spec is mapped for vLLM %s only, and this pool runs %s",
				vllm.Version, release)
		}
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
		// The stated enable_chunked_prefill, else vLLM's default, on (core/vllm ResolveChunkedPrefill).
		ChunkedPrefill: vllm.ResolveChunkedPrefill(e.EnableChunkedPrefill),
		MaxModelLen:    maxLen,
	})
	r.engine.MaxNumSeqs, r.engine.MaxNumBatchedTokens = b.MaxNumSeqs, b.MaxNumBatchedTokens
	r.sources["max_num_seqs"] = b.MaxNumSeqsSource
	r.sources["max_num_batched_tokens"] = b.MaxNumBatchedTokensSource
	r.sources["block_size"] = vllm.SourceStated
	if hasRecurrent(in.Model) {
		// A model with recurrent layers is a hybrid (or attention-free) model: the platform aligns
		// its block size to the Mamba page instead of reading it from the attention backends
		// (Platform.update_block_size_for_backend, vllm/platforms/interface.py:750-794@5df63a72), and
		// the recurrent layers' spec is built with that block size.
		fam, _ := in.Chip.CapabilityMajor()
		window, _ := e.MaxModelLen.Stated()
		if window == 0 && !e.MaxModelLen.Auto() {
			window = in.Model.Global.DefaultWindow()
		}
		r.mamba, r.mambaErr = planMamba(in.Model, r.engine, mambaEnv{tp: max(pool.Parallel.TP, 1),
			dcp: max(pool.Parallel.DCP, 1), pcp: max(pool.Parallel.PCP, 1), capability: fam, maxModelLen: window})
		switch {
		case r.mambaErr == nil:
			r.engine.BlockSize, r.sources["block_size"] = r.mamba.blockSize, r.mamba.source
			if e.BlockSize == r.mamba.blockSize {
				r.sources["block_size"] = vllm.SourceStated
			}
		case e.BlockSize <= 0:
			r.blockSizeErr = r.mambaErr
			delete(r.sources, "block_size")
		}
		return r
	}
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
		kv = in.Rules.ResolveCacheDType(architectureOf(in.Model), fam, max(pool.Parallel.TP, 1), kv)
	}
	return vllm.ResolveBlockSize(vllm.BlockSizeInput{
		Model:        in.Scenario.Model,
		Capability:   in.Chip.ComputeCapability,
		TP:           max(pool.Parallel.TP, 1),
		KVCacheDType: kv,
		// Only a drafter with a model adds attention layers to the selection.
		Speculative:         e.Speculative != nil && vllm.SpeculationAddsAttentionLayers(e.Speculative.Method, e.Speculative.NumSpecTokens),
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

// kernelResolvesCacheDType says whether the kernel itself resolves an unstated cache dtype for the
// architecture: DeepSeek-V4 and V4.1 keep a packed cache whose layout the kernel owns.
func kernelResolvesCacheDType(architecture string) bool {
	return architecture == subrules.ArchDeepseekV4 || architecture == subrules.ArchDeepseekV41
}

// glmMoeDsaConfigs are the sha256 of the vendor config.json each catalog model whose
// architectures[0] is GlmMoeDsaForCausalLM was derived from (graph.yaml derived_from). The catalog
// graph of these models stated no global.architecture before the layer-10 pin and states it now;
// a graph derived from one of these configs that still states none (a copy kept by a user) is
// named by the hash, since the rules key behaviour on the architecture. TestGLMConfigHashesNameTheArchitecture checks each against the
// pinned catalog's config.json and that no other catalog model is of the class.
var glmMoeDsaConfigs = map[string]string{
	"5ea2d79d3a77fe90836c18893de00996d096cd01516b52b88c6fc18993cb4df8": "glm-5",
	"185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a": "glm-5.2",
	"22e49334abf8562fecf70ca3292ba3f5b33f5602fb2bf10b52dd64a66cfe65ff": "glm-5.2-fp8",
	"3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69": "glm-5.3",
}

// architectureOf is the architecture the rules key on: the graph's own, else GlmMoeDsaForCausalLM
// for a graph derived from one of glmMoeDsaConfigs.
func architectureOf(g *model.Graph) string {
	if g == nil {
		return ""
	}
	if g.Global.Architecture != "" {
		return g.Global.Architecture
	}
	if _, ok := glmMoeDsaConfigs[g.DerivedFrom.SHA256]; ok {
		return v0_32.ArchGlmMoeDsa
	}
	return ""
}
