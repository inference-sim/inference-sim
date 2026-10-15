package cluster

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func dpReq(id string, arrival int64, in, out int) *sim.Request {
	return &sim.Request{
		ID:           id,
		ArrivalTime:  arrival,
		InputTokens:  make([]sim.TokenID, in),
		OutputTokens: make([]sim.TokenID, out),
		State:        sim.StateQueued,
	}
}

func runDP(t *testing.T, instances, lockstep int, reqs []*sim.Request) *ClusterSimulator {
	t.Helper()
	cfg := newTestDeploymentConfig(instances)
	cfg.DPLockstepRanks = lockstep
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)
	return cs
}

// The ranks of a replica step in lockstep (#1909): a probe request's latency on one rank
// grows when the OTHER rank is loaded, because every step lasts the slowest rank's step.
// Independent instances (the pre-#1909 behavior) are unaffected by the neighbour's load.
func TestDPLockstep_LoadOnOneRankSlowsTheOther(t *testing.T) {
	// The balancer sends the first arrival to rank 0 and the second to rank 1.
	probeE2E := func(lockstep int, neighbourInput int) float64 {
		reqs := []*sim.Request{dpReq("probe", 0, 64, 32), dpReq("neighbour", 0, neighbourInput, 32)}
		cs := runDP(t, 2, lockstep, reqs)
		if got := cs.instances[0].Metrics().CompletedRequests; got != 1 {
			t.Fatalf("lockstep=%d: rank 0 completed %d requests, want the probe alone", lockstep, got)
		}
		return cs.instances[0].Metrics().RequestE2Es["probe"]
	}
	light, heavy := probeE2E(2, 64), probeE2E(2, 8192)
	if heavy <= light {
		t.Errorf("lockstep: the probe's E2E did not rise with its neighbour's load (%v light, %v heavy)", light, heavy)
	}
	if a, b := probeE2E(0, 64), probeE2E(0, 8192); a != b {
		t.Errorf("independent instances: the probe's E2E moved with another instance's load (%v vs %v)", a, b)
	}
}

// A replica is one router endpoint, listed under rank 0's ID, whose queue and running
// counts are its ranks' sums and whose KV utilization is the largest rank's.
func TestDPReplica_IsOneEndpointWithAggregatedSignals(t *testing.T) {
	cfg := newTestDeploymentConfig(4)
	cfg.DPLockstepRanks = 2
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
	state := buildRouterState(cs, nil)
	var ids []string
	for _, s := range state.Snapshots {
		ids = append(ids, s.ID)
	}
	if want := []string{"instance_0", "instance_2"}; !reflect.DeepEqual(ids, want) {
		t.Fatalf("endpoints = %v, want %v", ids, want)
	}
	cs.inFlightRequests["instance_0"], cs.inFlightRequests["instance_1"] = 2, 3
	snap := cs.endpointSnapshot(cs.replicaOf["instance_0"])
	if snap.InFlightRequests != 5 {
		t.Errorf("endpoint in-flight = %d, want the ranks' sum 5", snap.InFlightRequests)
	}
	if want := cs.instances[0].TotalKvCapacityTokens() * 2; snap.TotalKvCapacityTokens != want {
		t.Errorf("endpoint KV capacity = %d, want %d", snap.TotalKvCapacityTokens, want)
	}
}

// vLLM's internal balancer spreads simultaneous arrivals across the ranks (the front-end
// bumps the chosen rank's waiting count between coordinator publishes), and every request
// is served by a rank of the replica the router chose.
func TestDPReplica_InternalBalancerSpreadsArrivals(t *testing.T) {
	var reqs []*sim.Request
	for i := 0; i < 8; i++ {
		reqs = append(reqs, dpReq(fmt.Sprintf("r%d", i), 0, 32, 8))
	}
	cs := runDP(t, 2, 2, reqs)
	for k, inst := range cs.instances {
		if got := inst.Metrics().CompletedRequests; got != 4 {
			t.Errorf("rank %d served %d of 8 simultaneous requests, want 4", k, got)
		}
	}
	for _, r := range reqs {
		if r.AssignedInstance != "instance_0" && r.AssignedInstance != "instance_1" {
			t.Errorf("%s assigned to %q, not a rank", r.ID, r.AssignedInstance)
		}
	}
}

func clusterOutcome(cs *ClusterSimulator) []any {
	m := cs.AggregatedMetrics()
	return []any{m.CompletedRequests, m.RequestE2Es, m.RequestTTFTs, m.SimEndedTime, m.TotalOutputTokens}
}

func steadyRequests() []*sim.Request {
	return testGenerateRequests(7, 20_000_000, 0.0001, 40, 0, 256, 64, 32, 1024, 64, 16, 8, 256)
}

// A width of 1 is no data parallelism: the run is identical to one with no lockstep at all.
func TestDPLockstep_WidthOneIsTheIndependentEngine(t *testing.T) {
	a := runDP(t, 3, 0, steadyRequests())
	b := runDP(t, 3, 1, steadyRequests())
	if !reflect.DeepEqual(clusterOutcome(a), clusterOutcome(b)) {
		t.Error("DPLockstepRanks=1 changed the outcome of an independent-engine run")
	}
}

// INV-6: a lockstep run is deterministic, and conserves its requests (INV-1).
func TestDPLockstep_DeterministicAndConserving(t *testing.T) {
	a := runDP(t, 4, 2, steadyRequests())
	b := runDP(t, 4, 2, steadyRequests())
	if !reflect.DeepEqual(clusterOutcome(a), clusterOutcome(b)) {
		t.Error("two identical lockstep runs differ")
	}
	if got := a.AggregatedMetrics().CompletedRequests; got != len(steadyRequests()) {
		t.Errorf("completed %d of %d requests", got, len(steadyRequests()))
	}
	for id, n := range a.inFlightRequests {
		if n != 0 {
			t.Errorf("inFlightRequests[%s] = %d after a drained run", id, n)
		}
	}
}

// Ranks of one replica share a clock: each step ends at the same instant on every rank, so
// their last steps end together and the wave's trailing dummy steps keep them aligned.
func TestDPLockstep_RanksEndTogether(t *testing.T) {
	cs := runDP(t, 2, 2, []*sim.Request{dpReq("a", 0, 64, 4), dpReq("b", 0, 4096, 64)})
	if a, b := cs.instances[0].Metrics().SimEndedTime, cs.instances[1].Metrics().SimEndedTime; a != b {
		t.Errorf("ranks ended at %d and %d; a lockstep wave ends on every rank at once", a, b)
	}
}
