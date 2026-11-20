// Moved from inference-sim/sim/kernelmodel/prefix_hit_test.go at ca532abca209af41425c007974a5f3ef336f5981.

package kernelmodel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	latencykernel "github.com/inference-sim/blis-substrate/kernel"

	sim "github.com/inference-sim/blis-vllm/core/runconfig"

	"github.com/inference-sim/blis-vllm/core/engineconfig"
)

func boolPtr(b bool) *bool { return &b }

// DeepSeek-V4.1 with DSpark: DSpark drops no EAGLE block at vLLM v0.33.0
// (needs_eagle_block_drop, vllm/config/speculative.py:2122-2129@4ba787ec), so no group is an
// EAGLE group and the replay boundary stays at the prompt's end, as without speculation. v0.31.0
// tagged the window group of DSpark's draft layers and dropped its block. With SWA bounded replay
// the window groups leave the lookup and a hit replays its last 128 tokens; without it they cut
// the hit to a retained window tail.
func TestPrefixHit_DeepSeekV41LayoutFollowsTheDraftsWindowCache(t *testing.T) {
	m, err := openDrafting(t, dspark, &engineconfig.Speculative{Method: "dspark", NumSpecTokens: 5})
	if err != nil {
		t.Fatal(err)
	}
	r := m.PrefixHit(nil)
	if r.Hit != (sim.KVPrefixHit{ReplayTokens: 128}) || r.Runner != "v2" || r.BoundedReplay == nil || !*r.BoundedReplay {
		t.Errorf("default: got %+v, want the V2 runner with only a 128-token replay", r)
	}
	want := sim.KVPrefixHit{Window: 128, WindowBlock: 32}
	if r := m.PrefixHit(boolPtr(false)); r.Hit != want || r.BoundedReplayFrom != "--swa-bounded-replay" {
		t.Errorf("override false: got %+v, want %+v from the override", r, want)
	}
	plain, err := openDrafting(t, dspark, nil)
	if err != nil {
		t.Fatal(err)
	}
	want = sim.KVPrefixHit{Window: 128, WindowBlock: 32}
	if r := plain.PrefixHit(boolPtr(false)); r.Hit != want {
		t.Errorf("no speculation: got %+v, want %+v (no EAGLE group, prompt-end boundary)", r.Hit, want)
	}
}

// Each feature vLLM v0.33.0's V2 model runner does not support (vllm/config/vllm.py:3053-3087@4ba787ec)
// that a deployment can state forces the V1 runner, which turns SWA bounded replay off
// (deepseek_v41/attention.py:520-526), and so does prefill context parallelism
// (attention.py:528-537) under the V2 runner.
func TestPrefixHit_V1RunnerAndPCPTurnBoundedReplayOff(t *testing.T) {
	r := repos()
	r.Scenarios = "../../testdata/scenarios"
	for _, tt := range []struct {
		name   string
		edit   func(p *deployment.Pool)
		runner string
	}{
		{"V1-only speculative method", func(p *deployment.Pool) {
			editPoolEngine(t, p, func(e *engineconfig.Engine) {
				e.Speculative = &engineconfig.Speculative{Method: "suffix", NumSpecTokens: 3}
			})
		}, "v1"},
		{"dual batch overlap", func(p *deployment.Pool) {
			editPoolEngine(t, p, func(e *engineconfig.Engine) { e.DBO = &engineconfig.DBO{Enabled: true} })
		}, "v1"},
		{"prefill context parallelism", func(p *deployment.Pool) { p.Parallel.PCP = 2 }, "v2"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			pool := poolWith(t, engineconfig.Engine{BlockSize: 64})
			tt.edit(&pool)
			m := &Model{k: fixedDeployment{pool: pool}, engine: mustDecode(t, pool), id: identity{prefix: prefixLayout{
				known: true, v41: true, window: 128, windowBlock: 32, pcp: max(pool.Parallel.PCP, 1)}}}
			got := m.PrefixHit(nil)
			if got.Runner != tt.runner {
				t.Errorf("runner %q (forced by %v), want %q", got.Runner, got.RunnerForcedBy, tt.runner)
			}
			if got.BoundedReplay == nil || *got.BoundedReplay || got.Hit.ReplayTokens != 0 || got.Hit.Window != 128 {
				t.Errorf("got %+v, want bounded replay off and the window group in the lookup", got)
			}
		})
	}
}

// fixedDeployment is a kernel that answers only Deployment.
type fixedDeployment struct {
	kernel.Kernel
	pool deployment.Pool
}

func (f fixedDeployment) Deployment() deployment.Pool { return f.pool }

// A pool's engine.swa_bounded_replay is vLLM's CacheConfig.swa_bounded_replay: false turns
// the replay off, true still yields to the V1 runner (attention.py:519-526), and the
// --swa-bounded-replay override replaces both. Each resolution names what decided it.
func TestPrefixHit_ThePoolsStatedBoundedReplayIsTheCacheConfig(t *testing.T) {
	for _, tt := range []struct {
		name         string
		stated       *bool
		v1, override *bool
		bounded      bool
		from         string
	}{
		{"unstated", nil, nil, nil, true, "v0.31.0 default"},
		{"stated false", boolPtr(false), nil, nil, false, "engine.swa_bounded_replay"},
		{"stated true", boolPtr(true), nil, nil, true, "engine.swa_bounded_replay"},
		{"stated true under the V1 runner", boolPtr(true), boolPtr(true), nil, false,
			"V1 model runner (attention.py:520-526)"},
		{"stated false under the V1 runner", boolPtr(false), boolPtr(true), nil, false, "engine.swa_bounded_replay"},
		{"the flag over a stated false", boolPtr(false), nil, boolPtr(true), true, "--swa-bounded-replay"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			pool := poolWith(t, engineconfig.Engine{BlockSize: 64, SWABoundedReplay: tt.stated})
			if tt.v1 != nil {
				editPoolEngine(t, &pool, func(e *engineconfig.Engine) { e.DBO = &engineconfig.DBO{Enabled: true} })
			}
			m := &Model{k: fixedDeployment{pool: pool}, engine: mustDecode(t, pool), id: identity{prefix: prefixLayout{
				known: true, v41: true, window: 128, windowBlock: 32, pcp: 1}}}
			got := m.PrefixHit(tt.override)
			if got.BoundedReplay == nil || *got.BoundedReplay != tt.bounded || got.BoundedReplayFrom != tt.from {
				t.Fatalf("bounded replay %v from %q, want %v from %q", got.BoundedReplay,
					got.BoundedReplayFrom, tt.bounded, tt.from)
			}
			if tt.bounded != (got.Hit.ReplayTokens == 128) {
				t.Errorf("layout %+v does not follow bounded replay %v", got.Hit, tt.bounded)
			}
		})
	}
}

// mustDecode is a pool's engine block as this package decodes it at open.
func mustDecode(t testing.TB, pool deployment.Pool) engineconfig.Engine {
	t.Helper()
	e, err := decodeEngine(pool)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

// An MTP drafter drops the last block of a prefix hit at vLLM v0.33.0 only beside a KV
// connector. Without one, with prefix caching on, each block's hash covers the tokens the
// drafter reads past it (num_hash_lookahead_tokens, vllm/config/vllm.py:677-692@4ba787ec) and the
// scheduler turns the drop off (vllm/v1/core/sched/scheduler.py:342-345), so a colocated pool's
// hit keeps its last block. An offload connector, or a P/D connector on a prefill or decode pool,
// keeps the legacy hashes and the drop. glm-5 is MLA only: one full-attention group kind.
func TestPrefixHit_MTPDropsTheLastBlockOnlyBesideAKVConnector(t *testing.T) {
	r := repos()
	r.Scenarios = "../../testdata/scenarios"
	for _, tt := range []struct {
		name string
		edit func(d *deployment.Deployment)
		drop bool
	}{
		{"colocated, no connector", func(*deployment.Deployment) {}, false},
		{"an offload connector", func(d *deployment.Deployment) {
			d.Offload = &deployment.Offload{Connector: "OffloadingConnector"}
		}, true},
		{"a P/D connector on a prefill pool", func(d *deployment.Deployment) {
			d.PDTransfer = &deployment.PDTransfer{Connector: "NixlConnector"}
			d.Pools[0].Role = deployment.RolePrefill
		}, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			in, err := latencykernel.OpenInputs("glm-5-h200-tp8-mtp3.yaml", r, 0)
			if err != nil {
				t.Fatal(err)
			}
			// The connector is edited in after the documents validated, so the layout is built as
			// the pool's open builds it, from the edited deployment.
			tt.edit(in.Deployment)
			pool := in.Deployment.Pools[0]
			e := mustDecode(t, pool)
			m := &Model{k: fixedDeployment{pool: pool}, engine: e,
				id: identity{prefix: prefixLayoutOf(in.Model, in.Deployment, in.PoolIndex, pool, e)}}
			got := m.PrefixHit(nil)
			if want := (sim.KVPrefixHit{FullEagle: tt.drop}); got.Hit != want {
				t.Errorf("got %+v (%s), want %+v", got.Hit, got.Note, want)
			}
		})
	}
}

// vLLM v0.33.0 picks the model runner by its own list (vllm/config/vllm.py:3053-3087@4ba787ec).
// ngram and the Mamba cache mode "all", which v0.31.0's list sent to V1, now run on V2. suffix,
// custom_class and dual batch overlap still force V1.
func TestModelRunner_V033List(t *testing.T) {
	spec := func(method string) func(e *engineconfig.Engine) {
		return func(e *engineconfig.Engine) {
			e.Speculative = &engineconfig.Speculative{Method: method, NumSpecTokens: 3}
		}
	}
	for _, tt := range []struct {
		name   string
		edit   func(e *engineconfig.Engine)
		runner string
	}{
		{"no feature", func(*engineconfig.Engine) {}, "v2"},
		{"ngram", spec("ngram"), "v2"},
		{"ngram_gpu", spec("ngram_gpu"), "v2"},
		{"eagle3", spec("eagle3"), "v2"},
		{"mamba cache mode all", func(e *engineconfig.Engine) { e.MambaCacheMode = "all" }, "v2"},
		{"suffix", spec("suffix"), "v1"},
		{"custom_class", spec("custom_class"), "v1"},
		{"dual batch overlap", func(e *engineconfig.Engine) { e.DBO = &engineconfig.DBO{Enabled: true} }, "v1"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			e := engineconfig.Engine{BlockSize: 64}
			tt.edit(&e)
			got, forcedBy := (&Model{engine: e}).ModelRunner()
			if got != tt.runner {
				t.Errorf("runner %q (forced by %v), want %q", got, forcedBy, tt.runner)
			}
		})
	}
}
