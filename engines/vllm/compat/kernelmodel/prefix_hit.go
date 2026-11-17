// Moved from inference-sim/sim/kernelmodel/prefix_hit.go at ca532abca209af41425c007974a5f3ef336f5981.

package kernelmodel

import (
	"fmt"
	"slices"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	sim "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// eagleDropMethods are vLLM v0.33.0's needs_eagle_block_drop methods: an EAGLE-style drafter
// builds a position's draft KV from the next token, so its groups drop the trailing block of a
// prefix hit (vllm/config/speculative.py:2122-2129@4ba787ec). DFlash and DSpark build it from the
// target's hidden state alone and drop nothing, which v0.31.0 and v0.32.0 did not distinguish.
var eagleDropMethods = []string{"eagle", "eagle3", "mtp"}

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
	// eagle: the coordinator drops the EAGLE block (drop_eagle_group_ids is not empty): the pool
	// speculates with an eagleDropMethods method and keeps the legacy block hashes, which it does
	// only beside a KV connector (num_hash_lookahead_tokens, vllm/config/vllm.py:677-692@4ba787ec;
	// Scheduler.__init__, vllm/v1/core/sched/scheduler.py:342-345). Without one each block's hash
	// covers the tokens the drafter reads past it, and no block is dropped.
	eagle bool
	// draftWindow and draftFull are whether the drafter's layers own a window cache and a full
	// (kv or index) cache: the groups holding them are the ones vLLM tags is_eagle_group
	// (get_global_kv_cache_groups, vllm/v1/core/kv_cache_planning.py:1650-1656@4ba787ec). With
	// none tagged the coordinator tags every group (kv_cache_coordinator.py:129-135).
	draftWindow, draftFull bool
	pcp                    int
}

func prefixLayoutOf(g *model.Graph, dep *deployment.Deployment, pool deployment.Pool, engine engineconfig.Engine) prefixLayout {
	l := prefixLayout{pcp: max(pool.Parallel.PCP, 1)}
	sp := engine.Speculative
	drafting := sp != nil && sp.NumSpecTokens > 0
	// kv_transfer_config: the deployment's offload connector, or its P/D connector on a pool of
	// either role (vllmengine sets both from the deployment).
	connector := dep != nil && (dep.Offload != nil || (dep.PDTransfer != nil && pool.Role != deployment.RoleColocated))
	l.eagle = drafting && slices.Contains(eagleDropMethods, sp.Method) && connector
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
	if drafting && g.Speculator != nil {
		// The drafter's layers register their caches beside the target's, in groups of their own
		// specs, whether or not any group drops a block.
		draft := owned(g.Speculator.Stack.Expand())
		for _, c := range draft {
			l.draftWindow = l.draftWindow || c.Kind == model.CacheWindow
			l.draftFull = l.draftFull || c.Kind == model.CacheKV || c.Kind == model.CacheIndex
		}
		caches = append(caches, draft...)
	}
	if !l.draftWindow && !l.draftFull {
		l.draftWindow, l.draftFull = true, true
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

// v1OnlySpecMethods are the speculative methods vLLM v0.33.0's V2 model runner does not
// support (vllm/config/vllm.py:3066-3068@4ba787ec9959e566d3b587a796dff07673324da5). ngram moved
// to V2, and medusa and mlp_speculator are no longer methods.
var v1OnlySpecMethods = []string{"suffix", "custom_class"}

// ModelRunner is the model runner vLLM v0.33.0 picks for the pool: V2 unless
// VllmConfig._get_v2_model_runner_unsupported_features reports a feature
// (VllmConfig.use_v2_model_runner, vllm/config/vllm.py:730-767@4ba787ec, with HAS_TRITON on CUDA).
// It returns "v1" or "v2" and the features that forced V1. Of the checks at
// vllm/config/vllm.py:3053-3087, a deployment states two: the speculative method (:3066-3068)
// and dual batch overlap (:3084-3085). Without VLLM_USE_V2_MODEL_RUNNER, overlap always reports
// "dual batch overlap" (_get_dbo_unsupported_features, :3222-3233). The rest are at their
// defaults, which pass:
//   - no external_launcher pipeline parallelism (:3060-3064);
//   - parallel drafting only for dflash and dspark, which the check exempts. vLLM sets it for
//     those two (vllm/config/speculative.py:1581-1582) and no deployment sets it for another
//     method (:3069-3076);
//   - no heterogeneous-vocabulary draft (:3081-3082).
//
// v0.33.0 dropped the Mamba cache mode from the list, so mamba_cache_mode "all" runs on V2.
// KV connectors, P/D roles, DP/EP, CUDA graph modes and enforce_eager are not on the list.
// Three things are outside what a deployment states: the environment variable, and HiSparse
// and watermarking, which force V2 (:731-746).
func (m *Model) ModelRunner() (runner string, forcedBy []string) {
	e := m.engine
	if sp := e.Speculative; sp != nil && slices.Contains(v1OnlySpecMethods, sp.Method) {
		forcedBy = append(forcedBy, "speculative method "+sp.Method)
	}
	if e.DBO != nil && e.DBO.Enabled {
		forcedBy = append(forcedBy, "dual batch overlap")
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
// model runner and without prefill context parallelism (deepseek_v41/attention.py:519-546).
// The pool's engine.swa_bounded_replay, when stated, is CacheConfig.swa_bounded_replay:
// false turns it off, and true still yields to those two conditions, as in vLLM. override
// (the --swa-bounded-replay flag), when non-nil, replaces the whole resolution (an engine
// build before the feature runs as false).
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
	block := m.engine.BlockSize
	if block <= 0 || block%l.windowBlock != 0 {
		r.Note = fmt.Sprintf("engine block %d is not the scheduler block (the LCM with the "+
			"%d-token window page)", block, l.windowBlock)
		return r
	}
	bounded, from := true, "v0.31.0 default"
	if stated := m.engine.SWABoundedReplay; stated != nil {
		bounded, from = *stated, "engine.swa_bounded_replay"
	}
	switch {
	case override != nil:
		bounded, from = *override, "--swa-bounded-replay"
	case !bounded:
	case r.Runner != "v2":
		bounded, from = false, "V1 model runner (attention.py:520-526)"
	case l.pcp > 1:
		bounded, from = false, "prefill context parallelism (attention.py:528-537)"
	}
	r.BoundedReplay, r.BoundedReplayFrom = &bounded, from
	// The groups holding the drafter's caches drop the EAGLE block; every window group shares one
	// spec, so the coordinator decides the drop for all of them (kv_cache_coordinator.py:725-780
	// @4ba787ec).
	windowEagle := l.eagle && l.draftWindow
	fullEagle := l.eagle && l.draftFull
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
