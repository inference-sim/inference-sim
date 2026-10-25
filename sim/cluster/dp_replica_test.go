package cluster

import (
	"reflect"
	"strings"
	"testing"

	"pgregory.net/rapid"

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

// A replica is one router endpoint, under rank 0's ID, behind which every rank serves; the
// other ranks are no endpoint of their own. The router reads the endpoint's signals from all of
// its ranks (sim/llmd renders one series per engine).
func TestDPReplica_IsOneEndpointOverItsRanks(t *testing.T) {
	cfg := newTestDeploymentConfig(4)
	cfg.DPLockstepRanks = 2
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
	var endpoints []string
	for _, inst := range cs.Instances() {
		ranks := cs.EndpointRanks(inst.ID())
		if ranks == nil {
			continue
		}
		endpoints = append(endpoints, string(inst.ID()))
		if len(ranks) != 2 || ranks[0] != inst {
			t.Errorf("endpoint %s serves %d ranks led by %v, want its 2 ranks led by itself", inst.ID(), len(ranks), ranks[0].ID())
		}
	}
	if want := []string{"instance_0", "instance_2"}; !reflect.DeepEqual(endpoints, want) {
		t.Fatalf("endpoints = %v, want %v", endpoints, want)
	}
}

// vLLM's balancer is load-aware, not round-robin: with one rank still serving a long
// request, later arrivals go to the other rank every time, though a rotating round-robin
// would hand the busy rank its turn.
func TestDPReplica_BalancerIsLoadAwareNotRoundRobin(t *testing.T) {
	reqs := []*sim.Request{
		dpReq("long", 0, 64, 4000), dpReq("short", 0, 64, 2),
		dpReq("c", 1_000_000, 64, 2), dpReq("d", 2_000_000, 64, 2), dpReq("e", 3_000_000, 64, 2),
	}
	cs := runDP(t, 2, 2, reqs)
	if reqs[0].AssignedInstance != "instance_0" {
		t.Fatalf("the first arrival went to %s, want rank 0", reqs[0].AssignedInstance)
	}
	if end := float64(reqs[0].ArrivalTime) + cs.AggregatedMetrics().RequestE2Es["long"]; end <= 3_000_000 {
		t.Fatalf("the long request ended at %v, before the last probe; the test is vacuous", end)
	}
	for _, r := range reqs[1:] {
		if r.AssignedInstance != "instance_1" {
			t.Errorf("%s went to %s while rank 0 was busy; want the idle rank 1", r.ID, r.AssignedInstance)
		}
	}
}

// The balancer's own rule (staleness between publishes, the local bump, the KV-pressure
// penalty, the in-flight floor) is tested with it, in blis-vllm compat/dplb; the tests here
// check what the cluster does with its choice.

// The ranks a replica's requests land on do not change the outcome: two identical requests
// arriving together get identical TTFTs whichever rank serves each, and reordering
// same-tick arrivals (which swaps their ranks) leaves every request's latency unchanged.
func TestDPLockstep_RankAssignmentIsOutcomeNeutral(t *testing.T) {
	twins := runDP(t, 2, 2, []*sim.Request{dpReq("x", 0, 512, 16), dpReq("y", 0, 512, 16)})
	m := twins.AggregatedMetrics()
	if m.RequestTTFTs["x"] != m.RequestTTFTs["y"] || m.RequestE2Es["x"] != m.RequestE2Es["y"] {
		t.Errorf("identical same-tick requests on two ranks: TTFT %v vs %v, E2E %v vs %v",
			m.RequestTTFTs["x"], m.RequestTTFTs["y"], m.RequestE2Es["x"], m.RequestE2Es["y"])
	}
	ab := runDP(t, 2, 2, []*sim.Request{dpReq("a", 0, 64, 32), dpReq("b", 0, 3000, 8)})
	ba := runDP(t, 2, 2, []*sim.Request{dpReq("b", 0, 3000, 8), dpReq("a", 0, 64, 32)})
	if ab.instances[0].Metrics().CompletedRequests != 1 || ab.AggregatedMetrics().RequestE2Es["a"] == 0 {
		t.Fatal("each rank must serve one request")
	}
	for _, f := range []func(*sim.Metrics) map[string]float64{
		func(m *sim.Metrics) map[string]float64 { return m.RequestTTFTs },
		func(m *sim.Metrics) map[string]float64 { return m.RequestE2Es },
	} {
		if x, y := f(ab.AggregatedMetrics()), f(ba.AggregatedMetrics()); !reflect.DeepEqual(x, y) {
			t.Errorf("swapping the ranks changed per-request latency: %v vs %v", x, y)
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

// Trailing dummy steps are not work: a lockstep run ends at its last real event, exactly as
// far past the last request's completion as an independent run does, rather than a wave's
// dummy tail later.
func TestDPLockstep_DummyTailDoesNotStretchTheRun(t *testing.T) {
	gap := func(lockstep int) float64 {
		reqs := []*sim.Request{dpReq("a", 0, 64, 4), dpReq("b", 0, 4096, 64)}
		cs := runDP(t, 2, lockstep, reqs)
		m := cs.AggregatedMetrics()
		last := 0.0
		for _, r := range reqs {
			last = max(last, float64(r.ArrivalTime)+m.RequestE2Es[r.ID])
		}
		return float64(m.SimEndedTime) - last
	}
	if l, i := gap(2), gap(0); l != i {
		t.Errorf("SimEndedTime overshoots the last completion by %v in lockstep, %v independently", l, i)
	}
}

// A deployment a replica cannot be built from is refused when the run starts, naming why:
// a rank whose startup placement was deferred, or a count that is not whole replicas.
func TestDPReplica_UnbuildableDeploymentIsAnError(t *testing.T) {
	short := newTestDeploymentConfig(4)
	short.DPLockstepRanks = 2
	short.ModelHardwareConfig = sim.NewModelHardwareConfig(testModelConfig(), "test-model", "H100", 4, 1, false, 0)
	short.NodePools = []NodePoolConfig{{Name: "p", GPUType: "H100", GPUsPerNode: 8, InitialNodes: 1, MaxNodes: 1, GPUMemoryGiB: 80}}
	ragged := newTestDeploymentConfig(3)
	ragged.DPLockstepRanks = 2
	for name, c := range map[string]struct {
		cfg  DeploymentConfig
		want string
	}{"deferred rank": {short, "could not be placed"}, "ragged count": {ragged, "whole number"}} {
		err := NewClusterSimulator(c.cfg, NewSliceRequestSource(nil), nil).Run()
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: Run() = %v, want an error mentioning %q", name, err, c.want)
		}
	}
}

// P/D pools step in lockstep too: every request is conserved, each prefill and decode
// completion is detected (the decode sub-request's arrival wakes its idle group), and the
// run is deterministic.
func TestDPLockstep_PDPools(t *testing.T) {
	run := func() *ClusterSimulator {
		cfg := newTestDisaggDeploymentConfig(4, 2, 2)
		cfg.DPLockstepRanks = 2
		cs := NewClusterSimulator(cfg, NewSliceRequestSource(steadyRequests()), nil)
		mustRun(t, cs)
		return cs
	}
	a, b := run(), run()
	n := len(steadyRequests())
	if got := a.AggregatedMetrics().CompletedRequests; got != n {
		t.Errorf("completed %d of %d disaggregated requests", got, n)
	}
	if a.pdDecodeCompletedCount != n || a.pdPrefillCompletedCount != n {
		t.Errorf("detected %d prefill and %d decode completions of %d", a.pdPrefillCompletedCount, a.pdDecodeCompletedCount, n)
	}
	for _, rep := range a.dpReplicas {
		if served := rep.ranks[0].Metrics().CompletedRequests + rep.ranks[1].Metrics().CompletedRequests; served == 0 {
			t.Errorf("replica %s served nothing", rep.id)
		}
	}
	if !reflect.DeepEqual(clusterOutcome(a), clusterOutcome(b)) {
		t.Error("two identical P/D lockstep runs differ")
	}
}

func runDPMode(t *testing.T, instances, lockstep int, mode string, reqs []*sim.Request) *ClusterSimulator {
	t.Helper()
	cfg := newTestDeploymentConfig(instances)
	cfg.DPLockstepRanks = lockstep
	cfg.DPLoadBalancing = mode
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)
	return cs
}

// Without data parallelism the load-balancing mode is moot: dp 1 runs identically in every
// mode, and identically to a deployment that states none (today's behavior). In particular a
// hybrid replica of one rank is an external one.
func TestDPLoadBalancing_ModeIsMootAtDPOne(t *testing.T) {
	want := clusterOutcome(runDP(t, 3, 0, steadyRequests()))
	for _, mode := range []string{DPLoadBalancingInternal, DPLoadBalancingHybrid, DPLoadBalancingExternal} {
		for _, width := range []int{0, 1} {
			if got := clusterOutcome(runDPMode(t, 3, width, mode, steadyRequests())); !reflect.DeepEqual(got, want) {
				t.Errorf("mode %s, width %d: outcome differs from a dp-1 run", mode, width)
			}
		}
	}
}

// Under external DP load balancing every rank is its own routable endpoint, under its own
// ID with its own signals, whatever the replica layout; under internal and hybrid each
// replica is one.
func TestDPLoadBalancing_ExternalEveryRankIsAnEndpoint(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		dp := rapid.SampledFrom([]int{2, 4}).Draw(rt, "dp")
		replicas := rapid.IntRange(1, 3).Draw(rt, "replicas")
		mode := rapid.SampledFrom([]string{DPLoadBalancingInternal, DPLoadBalancingHybrid, DPLoadBalancingExternal}).Draw(rt, "mode")
		cfg := newTestDeploymentConfig(dp * replicas)
		cfg.DPLockstepRanks = dp
		cfg.DPLoadBalancing = mode
		cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
		var snaps []InstanceID
		for _, inst := range cs.Instances() {
			if cs.EndpointRanks(inst.ID()) != nil {
				snaps = append(snaps, inst.ID())
			}
		}
		want := replicas
		if mode == DPLoadBalancingExternal {
			want = dp * replicas
		}
		if len(snaps) != want {
			rt.Fatalf("mode %s, %d replicas of dp %d: %d endpoints, want %d", mode, replicas, dp, len(snaps), want)
		}
		if mode == DPLoadBalancingExternal {
			for i, id := range snaps {
				if id != cs.instances[i].ID() {
					rt.Fatalf("endpoint %d is %s, want rank %s", i, id, cs.instances[i].ID())
				}
			}
		}
	})
}

// External load balancing changes who picks the rank, not how ranks step: the router places
// each request on a rank, and load on one rank still slows its lockstep neighbour.
func TestDPLoadBalancing_ExternalKeepsLockstep(t *testing.T) {
	probe := func(neighbourInput int) float64 {
		reqs := []*sim.Request{dpReq("probe", 0, 64, 32), dpReq("neighbour", 0, neighbourInput, 32)}
		cs := runDPMode(t, 2, 2, DPLoadBalancingExternal, reqs)
		if reqs[0].AssignedInstance != "instance_0" || reqs[1].AssignedInstance != "instance_1" {
			t.Fatalf("round-robin routing placed probe on %s and neighbour on %s", reqs[0].AssignedInstance, reqs[1].AssignedInstance)
		}
		return cs.AggregatedMetrics().RequestE2Es["probe"]
	}
	if light, heavy := probe(64), probe(8192); heavy <= light {
		t.Errorf("external mode: the probe's E2E did not rise with its neighbour's load (%v, %v)", light, heavy)
	}
}

// An unknown mode is refused when the run starts.
func TestDPLoadBalancing_UnknownModeIsAnError(t *testing.T) {
	cfg := newTestDeploymentConfig(2)
	cfg.DPLockstepRanks = 2
	cfg.DPLoadBalancing = "random"
	if err := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil).Run(); err == nil || !strings.Contains(err.Error(), "random") {
		t.Errorf("Run() = %v, want the unknown mode refused", err)
	}
}
