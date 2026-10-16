package kernelmodel

import (
	"fmt"
	"slices"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/inference-sim/sim"
)

// eagleMethods are vLLM v0.31.0's use_eagle() methods, whose prefix hits drop the trailing
// block of an EAGLE group unless disable_eagle_block_drop (config/speculative.py:1938-1946).
var eagleMethods = []string{"eagle", "eagle3", "mtp", "dflash", "dspark"}

// prefixLayout is what of the model's KV cache groups decides how a prefix hit is cut.
type prefixLayout struct {
	// known is false where the layout is not one this adapter maps; the hit then stays the
	// full-attention prefix and why is recorded in note.
	known bool
	note  string
	// v41 is DeepSeek-V4.1's layout: a full-attention group (compressed KV and indexer keys
	// at the engine block), sliding-window groups and compressor rings.
	v41                 bool
	window, windowBlock int
	// eagle: the pool speculates with a use_eagle() method. lastDraftKind is the kind of the
	// last cache of the draft's last layer, the group vLLM flags as the EAGLE group.
	eagle         bool
	lastDraftKind model.CacheKind
	pcp           int
}

func prefixLayoutOf(g *model.Graph, pool deployment.Pool) prefixLayout {
	l := prefixLayout{pcp: max(pool.Parallel.PCP, 1)}
	sp := pool.Engine.Speculative
	l.eagle = sp != nil && sp.NumSpecTokens > 0 && slices.Contains(eagleMethods, sp.Method)
	kinds := map[string]model.LayerKind{}
	for _, k := range g.LayerKinds {
		kinds[k.ID] = k
	}
	owned := func(ids []string) []model.Cache {
		var out []model.Cache
		for _, id := range ids {
			for _, n := range kinds[id].Nodes {
				for _, c := range n.Caches {
					if c.Owned {
						out = append(out, c)
					}
				}
			}
		}
		return out
	}
	caches := owned(g.Stack.Expand())
	if l.eagle && g.Speculator != nil {
		layers := g.Speculator.Stack.Expand()
		if len(layers) > 0 {
			if last := owned(layers[len(layers)-1:]); len(last) > 0 {
				l.lastDraftKind = last[len(last)-1].Kind
			}
			caches = append(caches, owned(layers)...)
		}
	}
	var windows, states []model.Cache
	for _, c := range caches {
		switch c.Kind {
		case model.CacheWindow:
			windows = append(windows, c)
		case model.CacheState:
			states = append(states, c)
		}
	}
	switch {
	case len(windows) == 0 && len(states) == 0:
		// Full-attention groups only (UnitaryKVCacheCoordinator, or several full-attention
		// groups whose EAGLE bit the coordinator sets on all when none is annotated,
		// kv_cache_coordinator.py:104-110).
		l.known = true
	case g.Global.Architecture == "DeepseekV41ForCausalLM" && len(windows) > 0:
		w := windows[0]
		for _, c := range windows {
			if c.Window != w.Window || c.BlockSize != w.BlockSize {
				l.note = "the window caches differ in window or page"
				return l
			}
		}
		l.known, l.v41, l.window, l.windowBlock = true, true, w.Window, w.BlockSize
	default:
		l.note = fmt.Sprintf("the %s layout of window and state caches is not mapped", g.Global.Architecture)
	}
	return l
}

// v1OnlySpecMethods are the speculative methods vLLM v0.31.0's V2 model runner does not
// support (config/vllm.py:3073-3083).
var v1OnlySpecMethods = []string{"ngram", "ngram_gpu", "suffix", "medusa", "mlp_speculator", "custom_class"}

// ModelRunner is the model runner vLLM v0.31.0 picks for the pool: V2 unless
// VllmConfig._get_v2_model_runner_unsupported_features reports a feature
// (VllmConfig.use_v2_model_runner, config/vllm.py:700-760, with HAS_TRITON on CUDA). It
// returns "v1" or "v2" and the features that forced V1. Of the checks at
// config/vllm.py:3051-3103 a deployment states the speculative method (:3073-3083), dual
// batch overlap (:3094-3095, which without VLLM_USE_V2_MODEL_RUNNER always reports "dual
// batch overlap", :3245-3246) and the Mamba cache mode (:3100-3101). The rest are at their
// defaults, which pass: compilation mode VLLM_COMPILE, not STOCK_TORCH_COMPILE (:3056-3057);
// sequence parallelism off (:3059-3063; enable_sp defaults to IS_DENSE = False at every
// optimization level, :147, :261-330); no external_launcher (:3067-3071); no parallel
// drafting outside dflash/dspark (:3088-3092); no elastic EP (:3097-3098). KV connectors,
// P/D roles, DP/EP, CUDA graph modes and enforce_eager are not on the list. The environment
// variable, hisparse and watermarking (:702-720), and ROCm's per-architecture V1 default
// (:722-743) are outside what a deployment states (BLIS's chips are CUDA parts).
func (m *Model) ModelRunner() (runner string, forcedBy []string) {
	e := m.k.Deployment().Engine
	if sp := e.Speculative; sp != nil && slices.Contains(v1OnlySpecMethods, sp.Method) {
		forcedBy = append(forcedBy, "speculative method "+sp.Method)
	}
	if e.DBO != nil && e.DBO.Enabled {
		forcedBy = append(forcedBy, "dual batch overlap")
	}
	if e.MambaCacheMode == "all" {
		forcedBy = append(forcedBy, "mamba cache mode all")
	}
	if len(forcedBy) > 0 {
		return "v1", forcedBy
	}
	return "v2", nil
}

// PrefixHitResolution is how a pool's prefix hits are cut and why.
type PrefixHitResolution struct {
	Hit sim.KVPrefixHit
	// Runner is ModelRunner's answer and RunnerForcedBy the features that forced V1.
	Runner         string
	RunnerForcedBy []string
	// BoundedReplay is DeepSeek-V4.1's SWA bounded replay as it resolved, nil for a model
	// without it; BoundedReplayFrom says what decided it.
	BoundedReplay     *bool
	BoundedReplayFrom string
	// Note says why the hit stays the full-attention prefix when the layout is not mapped.
	Note string
}

// PrefixHit is how the pool's KV cache groups cut a prefix-cache hit in vLLM v0.31.0.
// DeepSeek-V4.1's SWA bounded replay (cache.py:242, on by default) is kept only under the V2
// model runner and without prefill context parallelism (deepseek_v41/attention.py:519-546);
// override, when non-nil, replaces that resolution (an engine build before the feature runs
// as false).
func (m *Model) PrefixHit(override *bool) PrefixHitResolution {
	l := m.id.prefix
	r := PrefixHitResolution{Note: l.note}
	r.Runner, r.RunnerForcedBy = m.ModelRunner()
	if !l.known {
		return r
	}
	if !l.v41 {
		// One full-attention group kind: an EAGLE drafter's hit drops its last block.
		r.Hit = sim.KVPrefixHit{FullEagle: l.eagle}
		return r
	}
	block := m.k.Deployment().Engine.BlockSize
	if block <= 0 || block%l.windowBlock != 0 {
		r.Note = fmt.Sprintf("engine block %d is not the scheduler block (the LCM with the "+
			"%d-token window page)", block, l.windowBlock)
		return r
	}
	bounded, from := true, "v0.31.0 default"
	switch {
	case override != nil:
		bounded, from = *override, "--swa-bounded-replay"
	case r.Runner != "v2":
		bounded, from = false, "V1 model runner (attention.py:520-526)"
	case l.pcp > 1:
		bounded, from = false, "prefill context parallelism (attention.py:528-537)"
	}
	r.BoundedReplay, r.BoundedReplayFrom = &bounded, from
	// DeepSeek-V4/V4.1 flag the group of the last registered layer, the draft's last
	// (kv_cache_utils.py:2138-2226); every window group shares one spec, so the coordinator
	// decides the EAGLE drop for all of them (kv_cache_coordinator.py:735-780).
	windowEagle := l.eagle && l.lastDraftKind == model.CacheWindow
	fullEagle := l.eagle && (l.lastDraftKind == model.CacheKV || l.lastDraftKind == model.CacheIndex)
	if bounded {
		// The window groups are not prefix-cacheable (kv_cache_interface.py:930-955) and drop
		// out of the lookup (kv_cache_coordinator.py:745-752).
		r.Hit = sim.KVPrefixHit{FullEagle: fullEagle, ReplayTokens: int64(l.window)}
		return r
	}
	r.Hit = sim.KVPrefixHit{
		FullEagle: fullEagle, Window: int64(l.window), WindowBlock: int64(l.windowBlock),
		WindowEagle: windowEagle, EagleBoundaries: l.eagle,
	}
	return r
}
