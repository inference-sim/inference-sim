package cluster

import (
	"testing"

	cengine "github.com/inference-sim/inference-sim/engines/vllm/core/runconfig"
	"github.com/inference-sim/inference-sim/sim"
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
}
