package cluster

import (
	cengine "github.com/inference-sim/blis-vllm/compat/engine"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// admissionStages builds the cluster and returns each instance's admission stage.
func admissionStages(t *testing.T, build func() DeploymentConfig) (*ClusterSimulator, []*sim.AdmissionStage) {
	t.Helper()
	cs := newTestCluster(build(), NewSliceRequestSource(nil), nil)
	if cs.startupErr != nil {
		t.Fatal(cs.startupErr)
	}
	out := make([]*sim.AdmissionStage, len(cs.instances))
	for i, inst := range cs.instances {
		out[i] = inst.drv.AdmissionStage()
	}
	return cs, out
}

// INV-R1 placement: the admission stage belongs to an endpoint's API servers. A replica behind
// vLLM's internal or hybrid balancer is one endpoint whose api_server_count servers (dp
// unless stated, vllm/entrypoints/cli/serve.py:105-127) render for every rank, so its ranks
// share one stage. Under external balancing, and for a P/D pool's ranks, every rank is its
// own `vllm serve` endpoint with its own stage.
func TestAdmissionStage_BelongsToTheEndpoint(t *testing.T) {
	frontend := cengine.FrontendConfig{RendererNumWorkers: 1}
	for _, mode := range []string{DPLoadBalancingInternal, DPLoadBalancingHybrid, DPLoadBalancingExternal} {
		_, r := admissionStages(t, func() DeploymentConfig {
			cfg := newTestDeploymentConfig(4)
			cfg.DPLockstepRanks, cfg.DPLoadBalancing = 2, mode
			editEngine(&cfg, func(c *simConfig) { c.Frontend = frontend })
			return cfg
		})
		shared := mode != DPLoadBalancingExternal
		if got := r[0] == r[1] && r[2] == r[3]; got != shared {
			t.Errorf("%s: ranks of one replica share a admission stage = %v, want %v", mode, got, shared)
		}
		if r[0] == r[2] {
			t.Errorf("%s: two replicas share a admission stage", mode)
		}
	}
	_, pd := admissionStages(t, func() DeploymentConfig {
		cfg := newTestDisaggDeploymentConfig(4, 2, 2)
		cfg.DPLockstepRanks = 2
		editEngine(&cfg, func(c *simConfig) { c.Frontend = frontend })
		return cfg
	})
	for i := range pd {
		for j := i + 1; j < len(pd); j++ {
			if pd[i] == pd[j] {
				t.Errorf("P/D pool ranks %d and %d share a admission stage; each is its own endpoint", i, j)
			}
		}
	}
	_, inert := admissionStages(t, func() DeploymentConfig {
		cfg := newTestDeploymentConfig(2)
		cfg.DPLockstepRanks = 2
		return cfg
	})
	if inert[0] != nil || inert[1] != nil {
		t.Error("an inert front end built a admission stage")
	}
}

// A replica endpoint admits on the queue its engine states: with two queues a burst of two
// requests is served in parallel, with one it is served in turn.
func TestAdmissionStage_AReplicaAdmitsOnTheQueueItsEngineStates(t *testing.T) {
	const render = 40_000
	ttfts := func(apiServers int) []float64 {
		cfg := newTestDeploymentConfig(2)
		cfg.DPLockstepRanks = 2
		editEngine(&cfg, func(c *simConfig) {
			c.Frontend = cengine.FrontendConfig{RendererNumWorkers: 1, APIServerCount: apiServers}
			c.LatencyModel = queueingFake(render)
		})
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
	if d := spread(ttfts(2)); d >= render/2 {
		t.Errorf("two queues: TTFTs spread %.0f ticks; the two renders should run in parallel", d)
	}
	if d := spread(ttfts(1)); d < render/2 {
		t.Errorf("one queue: TTFTs spread %.0f ticks; the second render should wait ~%d", d, render)
	}
}

// queueingFake is testFakeLatency with an admission overhead of ticks per request.
func queueingFake(ticks int64) cengine.LatencyModel {
	c := testutil.DefaultFakeLatency()
	c.QueueingTicks = ticks
	return fakeLatencyWithCoeffs(c)
}
