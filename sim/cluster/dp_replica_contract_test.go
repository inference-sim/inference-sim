package cluster

import (
	"fmt"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/contract"
	"github.com/inference-sim/inference-sim/sim"
)

// driveDPReplicas builds cfg's data-parallel replicas through the engine contract: one
// compatibility DPGroup per replica, each rank's instance driven over its rank's view, rank 0's
// driver leading the group, as the build site wires them (cmd/engine_build.go).
func driveDPReplicas(t *testing.T, cfg *DeploymentConfig) {
	t.Helper()
	ranks := cfg.DPLockstepRanks
	var group *contract.DPGroup
	var leader *sim.EngineDriver
	cfg.NewEngine = func(index int, c sim.SimConfig) (*sim.EngineDriver, error) {
		k := index % ranks
		if k == 0 {
			ids := make([]string, ranks)
			for j := range ids {
				ids[j] = fmt.Sprintf("instance_%d", index+j)
			}
			g, err := contract.NewDPGroup(ids, c)
			if err != nil {
				return nil, err
			}
			group = g
		}
		d := sim.NewEngineDriver(group.View(k), sim.EngineDriverConfig{Horizon: c.Horizon,
			ResumeGapFromLastToken: true, ScrapeEngineMetrics: true, ApplyAtStepStart: true,
			PostDecodeFixedOverhead:   c.LatencyModel.PostDecodeFixedOverhead(),
			OutputTokenProcessingTime: c.LatencyModel.OutputTokenProcessingTime(),
			QueueingTime:              c.LatencyModel.QueueingTime})
		if k == 0 {
			leader = d
		}
		d.JoinGroup(leader, k)
		return d, nil
	}
}

// A replica endpoint driven through the engine contract is the host loop's: one endpoint, rank
// 0's instance, over both ranks, whose group picks each request's rank. The pick is vLLM's
// load-aware one, not round robin: with one rank serving a long request, the later arrivals go to
// the other rank every time. Every request lands where the host loop's balancer put it, and
// finishes at the same instant.
func TestContractEngine_DPReplicaPicksAsTheHostLoop(t *testing.T) {
	run := func(driven bool) ([]*sim.Request, *ClusterSimulator) {
		reqs := []*sim.Request{
			dpReq("long", 0, 64, 4000), dpReq("short", 0, 64, 2),
			dpReq("c", 1_000_000, 64, 2), dpReq("d", 2_000_000, 64, 2), dpReq("e", 3_000_000, 64, 2),
		}
		cfg := newTestDeploymentConfig(2)
		cfg.DPLockstepRanks = 2
		if driven {
			driveDPReplicas(t, &cfg)
		}
		cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
		mustRun(t, cs)
		return reqs, cs
	}
	hostReqs, host := run(false)
	reqs, cs := run(true)
	if len(cs.EndpointRanks("instance_0")) != 2 || cs.EndpointRanks("instance_1") != nil {
		t.Fatal("a driven replica must be one endpoint, rank 0's instance, over both ranks")
	}
	for _, r := range reqs[2:] {
		if r.AssignedInstance == reqs[0].AssignedInstance {
			t.Errorf("%s went to %s, the rank serving the long request; the group's pick must use the idle rank",
				r.ID, r.AssignedInstance)
		}
	}
	want, got := host.AggregatedMetrics(), cs.AggregatedMetrics()
	for k, r := range reqs {
		if r.AssignedInstance != hostReqs[k].AssignedInstance {
			t.Errorf("%s ran on %s, the host loop's on %s", r.ID, r.AssignedInstance, hostReqs[k].AssignedInstance)
		}
		if got.RequestCompletionTimes[r.ID] != want.RequestCompletionTimes[r.ID] {
			t.Errorf("%s completed at %v, the host loop's at %v", r.ID,
				got.RequestCompletionTimes[r.ID], want.RequestCompletionTimes[r.ID])
		}
	}
}
