package kernelmodel

import (
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/deployment"

	sim "github.com/inference-sim/inference-sim/sim/engine"
)

func boolPtr(b bool) *bool { return &b }

// DeepSeek-V4.1's DSpark draft layers own only a window cache, so vLLM v0.31.0 flags a window
// group as the EAGLE group and the 64-token full-attention group never drops a block. With
// SWA bounded replay the window groups leave the lookup and a hit replays its last 128
// tokens; without it they cut the hit to a retained window tail.
func TestPrefixHit_DeepSeekV41LayoutFollowsTheDraftsWindowCache(t *testing.T) {
	m, err := openDrafting(t, dspark, &deployment.Speculative{Method: "dspark", NumSpecTokens: 5})
	if err != nil {
		t.Fatal(err)
	}
	r := m.PrefixHit(nil)
	if r.Hit != (sim.KVPrefixHit{ReplayTokens: 128}) || r.Runner != "v2" || r.BoundedReplay == nil || !*r.BoundedReplay {
		t.Errorf("default: got %+v, want the V2 runner with only a 128-token replay", r)
	}
	want := sim.KVPrefixHit{Window: 128, WindowBlock: 32, WindowEagle: true, EagleBoundaries: true}
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

// Each feature vLLM v0.31.0's V2 model runner does not support (config/vllm.py:3051-3103)
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
			p.Engine.Speculative = &deployment.Speculative{Method: "ngram", NumSpecTokens: 3}
		}, "v1"},
		{"dual batch overlap", func(p *deployment.Pool) { p.Engine.DBO = &deployment.DBO{Enabled: true} }, "v1"},
		{"mamba cache mode all", func(p *deployment.Pool) { p.Engine.MambaCacheMode = "all" }, "v1"},
		{"prefill context parallelism", func(p *deployment.Pool) { p.Parallel.PCP = 2 }, "v2"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			pool := deployment.Pool{Engine: deployment.Engine{BlockSize: 64}}
			tt.edit(&pool)
			m := &Model{k: fixedDeployment{pool: pool}, id: identity{prefix: prefixLayout{
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
			pool := deployment.Pool{Engine: deployment.Engine{BlockSize: 64, SWABoundedReplay: tt.stated}}
			if tt.v1 != nil {
				pool.Engine.DBO = &deployment.DBO{Enabled: true}
			}
			m := &Model{k: fixedDeployment{pool: pool}, id: identity{prefix: prefixLayout{
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
