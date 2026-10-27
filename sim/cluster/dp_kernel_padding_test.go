package cluster

import (
	"fmt"
	"github.com/inference-sim/blis-vllm/compat/dplb"
	cengine "github.com/inference-sim/blis-vllm/compat/engine"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/sim"
)

// plainLatency hides a latency model's DPStepPricer, so a lockstep group prices each rank
// at its own batch with no padding.
type plainLatency struct{ cengine.LatencyModel }

// Through the cluster, a lockstep replica priced by the kernel pads its ranks: a
// long-context decode on one rank, stepping beside short prefills on the other, costs the
// padded token count, so it finishes later than the same run priced without padding -- and
// never sooner, since padding only adds work.
func TestDPLockstep_KernelPaddingReachesTheCluster(t *testing.T) {
	m, err := kernelmodel.Open("minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2.yaml", kernelmodel.DefaultRepos())
	if err != nil {
		t.Fatal(err)
	}
	longE2E := func(lm cengine.LatencyModel) float64 {
		reqs := []*sim.Request{dpReq("long", 0, 8000, 200)}
		for i := 1; i <= 20; i++ {
			reqs = append(reqs, dpReq(fmt.Sprintf("s%d", i), int64(i)*dplb.StatsIntervalUs+1, 200, 1))
		}
		cfg := newTestDeploymentConfig(2)
		cfg.DPLockstepRanks = 2
		editEngine(&cfg, func(c *simConfig) { c.LatencyModel = lm })
		cs := newTestCluster(cfg, NewSliceRequestSource(reqs), nil)
		mustRun(t, cs)
		if reqs[0].AssignedInstance != "instance_0" || reqs[1].AssignedInstance != "instance_1" {
			t.Fatalf("long on %s, first short on %s; want ranks 0 and 1", reqs[0].AssignedInstance, reqs[1].AssignedInstance)
		}
		return cs.AggregatedMetrics().RequestE2Es["long"]
	}
	padded, unpadded := longE2E(m), longE2E(plainLatency{m})
	if padded <= unpadded {
		t.Errorf("the long request's E2E is %v padded and %v unpadded; padding must slow it", padded, unpadded)
	}
}
