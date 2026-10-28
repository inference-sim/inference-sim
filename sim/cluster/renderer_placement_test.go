package cluster

import (
	cengine "github.com/inference-sim/blis-vllm/compat/engine"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// renderStages builds the cluster and returns each instance's render stage.
func renderStages(t *testing.T, build func() DeploymentConfig) (*ClusterSimulator, []*sim.Renderer) {
	t.Helper()
	cs := newTestCluster(build(), NewSliceRequestSource(nil), nil)
	if cs.startupErr != nil {
		t.Fatal(cs.startupErr)
	}
	out := make([]*sim.Renderer, len(cs.instances))
	for i, inst := range cs.instances {
		out[i] = inst.drv.Renderer()
	}
	return cs, out
}

// INV-R1 placement: the render stage belongs to an endpoint's API servers. A replica behind
// vLLM's internal or hybrid balancer is one endpoint whose api_server_count servers (dp
// unless stated, vllm/entrypoints/cli/serve.py:105-127) render for every rank, so its ranks
// share one stage. Under external balancing, and for a P/D pool's ranks, every rank is its
// own `vllm serve` endpoint with its own stage.
func TestRenderStage_BelongsToTheEndpoint(t *testing.T) {
	frontend := sim.FrontendConfig{RendererNumWorkers: 1}
	for _, mode := range []string{DPLoadBalancingInternal, DPLoadBalancingHybrid, DPLoadBalancingExternal} {
		_, r := renderStages(t, func() DeploymentConfig {
			cfg := newTestDeploymentConfig(4)
			cfg.DPLockstepRanks, cfg.DPLoadBalancing, cfg.Frontend = 2, mode, frontend
			return cfg
		})
		shared := mode != DPLoadBalancingExternal
		if got := r[0] == r[1] && r[2] == r[3]; got != shared {
			t.Errorf("%s: ranks of one replica share a render stage = %v, want %v", mode, got, shared)
		}
		if r[0] == r[2] {
			t.Errorf("%s: two replicas share a render stage", mode)
		}
	}
	_, pd := renderStages(t, func() DeploymentConfig {
		cfg := newTestDisaggDeploymentConfig(4, 2, 2)
		cfg.DPLockstepRanks, cfg.Frontend = 2, frontend
		return cfg
	})
	for i := range pd {
		for j := i + 1; j < len(pd); j++ {
			if pd[i] == pd[j] {
				t.Errorf("P/D pool ranks %d and %d share a render stage; each is its own endpoint", i, j)
			}
		}
	}
	_, inert := renderStages(t, func() DeploymentConfig {
		cfg := newTestDeploymentConfig(2)
		cfg.DPLockstepRanks = 2
		return cfg
	})
	if inert[0] != nil || inert[1] != nil {
		t.Error("an inert front end built a render stage")
	}
}

// A replica endpoint's dp API servers each take every dp-th request, so a burst of dp
// requests at an internal-balancer replica renders in parallel; an explicit
// api_server_count of 1 serializes the same burst on one thread.
func TestRenderStage_AReplicaRendersOnItsDPServers(t *testing.T) {
	const render = 40_000
	ttfts := func(apiServers int) []float64 {
		cfg := newTestDeploymentConfig(2)
		cfg.DPLockstepRanks = 2
		cfg.Frontend = sim.FrontendConfig{RendererNumWorkers: 1, APIServerCount: apiServers}
		editEngine(&cfg, func(c *simConfig) { c.LatencyModel = queueingFake(render) })
		reqs := []*sim.Request{dpReq("a", 0, 64, 4), dpReq("b", 0, 64, 4)}
		cs := newTestCluster(cfg, NewSliceRequestSource(reqs), nil)
		mustRun(t, cs)
		var out []float64
		for _, inst := range cs.instances {
			for _, v := range inst.Metrics().RequestTTFTs {
				out = append(out, v)
			}
		}
		return out
	}
	spread := func(v []float64) float64 {
		lo, hi := v[0], v[0]
		for _, x := range v {
			lo, hi = min(lo, x), max(hi, x)
		}
		return hi - lo
	}
	if d := spread(ttfts(0)); d >= render/2 {
		t.Errorf("default (dp) API servers: TTFTs spread %.0f ticks; the two renders should run in parallel", d)
	}
	if d := spread(ttfts(1)); d < render/2 {
		t.Errorf("one API server: TTFTs spread %.0f ticks; the second render should wait ~%d", d, render)
	}
}

// queueingFake is testFakeLatency with an admission overhead of ticks per request.
func queueingFake(ticks int64) cengine.LatencyModel {
	c := testutil.DefaultFakeLatency()
	c.QueueingTicks = ticks
	return fakeLatencyWithCoeffs(c)
}
