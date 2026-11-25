// Original to this repository: H2/H3 of the DP external load-balancing RFC (inference-sim#2699).
// The faithfulness check for the whole H2+H3 rewrite: an externally-balanced DP group, registered
// as N router-visible per-rank endpoints over one shared engine (the dprankview views the build
// site constructs, driven by the sim leader/member path), neither creates nor drops requests
// relative to the one collapsed instance the internal balancer builds, and every request completes
// with the same output it would collapsed (BC-H2-1..5, BC-H3-2, INV-1).

package cluster

import (
	"fmt"
	"testing"

	runconfig "github.com/inference-sim/inference-sim/engines/vllm/core/runconfig"
	vllm "github.com/inference-sim/inference-sim/engines/vllm/core/vllm"
	"github.com/inference-sim/inference-sim/engines/vllm/core/vllm/dprankview"
	"github.com/inference-sim/inference-sim/internal/vllmtest"
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
)

// dpExternalTestConfig is the engine configuration both the collapsed and the external deployments
// of the differential build their group from, a small MoE-shaped group whose steps the fake
// latency model prices. async_scheduling off keeps one lockstep step in flight at a time, so the
// collapsed normal-kick path and the leader kickGroup path price the same per-step timeline.
func dpExternalTestConfig() simConfig {
	c := newTestSimConfig()
	// A roomy KV pool and modest batch so the group makes progress without preemption churn.
	c.KVCacheConfig = runconfig.NewKVCacheConfig(4000, 16, 0, 0, 0, 0)
	c.BatchConfig = runconfig.NewBatchConfig(8, 256, 0)
	return c
}

// buildDPGroup builds a shared vLLM data-parallel group of dp ranks from cfg, external when
// external is set. It is the engine the build site (cmd) hands to dprankview; here a test stands in
// for the build site.
func buildDPGroup(t *testing.T, cfg simConfig, dp int, external bool) *vllm.DPGroup {
	t.Helper()
	extra := fmt.Sprintf("data_parallel_size: %d\n", dp)
	if external {
		extra += "data_parallel_external_lb: true\n"
	}
	eng, err := vllmtest.Engine(cfg, vllmtest.Options{Extra: extra})
	if err != nil {
		t.Fatalf("building the dp=%d group: %v", dp, err)
	}
	g, ok := eng.(*vllm.DPGroup)
	if !ok {
		t.Fatalf("a dp=%d engine is %T, not a *DPGroup", dp, eng)
	}
	return g
}

// dpDriver wraps an engine in a driver with the host overheads of cfg's latency model, as the
// build site does for every instance.
func dpDriver(eng engine.Engine, cfg simConfig) *sim.EngineDriver {
	dcfg := sim.EngineDriverConfig{Horizon: cfg.Horizon}
	if m := cfg.LatencyModel; m != nil {
		dcfg.PostDecodeFixedOverhead = m.PostDecodeFixedOverhead()
		dcfg.OutputTokenProcessingTime = m.OutputTokenProcessingTime()
	}
	return sim.NewEngineDriver(eng, dcfg)
}

// collapsedDPDeployment is the internal-balancer build: one router endpoint whose engine is the
// whole dp-rank group (one instance, DPLockstepRanks unset), exactly as a run builds a DP group
// today.
func collapsedDPDeployment(t *testing.T, cfg simConfig, dp int) DeploymentConfig {
	t.Helper()
	d := compatDeployment(cfg, 1)
	g := buildDPGroup(t, cfg, dp, false)
	d.NewEngine = func(int, sim.InstanceConfig) (*sim.EngineDriver, error) { return dpDriver(g, cfg), nil }
	return d
}

// externalDPDeployment is the external-balancer build this RFC adds: dp router endpoints, one per
// rank, over one shared group through the dprankview views, registered through buildDPReplicas
// (DPLockstepRanks = dp, DPLoadBalancing external). The build site (cmd) is stood in for here.
func externalDPDeployment(t *testing.T, cfg simConfig, dp int) DeploymentConfig {
	t.Helper()
	d := compatDeployment(cfg, dp)
	d.DPLockstepRanks = dp
	d.DPLoadBalancing = DPLoadBalancingExternal
	g := buildDPGroup(t, cfg, dp, true)
	views := dprankview.NewEndpointViews(g)
	if len(views) != dp {
		t.Fatalf("a dp=%d external group exposes %d endpoints, want %d", dp, len(views), dp)
	}
	drivers := make([]*sim.EngineDriver, dp)
	for k, v := range views {
		drivers[k] = dpDriver(v, cfg)
	}
	d.NewEngine = func(index int, _ sim.InstanceConfig) (*sim.EngineDriver, error) {
		if index < 0 || index >= dp {
			return nil, fmt.Errorf("external dp build: instance index %d out of range for dp=%d", index, dp)
		}
		return drivers[index], nil
	}
	return d
}

// completionsByID is each completed request's generated-token count, keyed by request id — the
// part of the result that is fixed by the workload (the pre-specified output), so it is equal
// across balancing modes whichever rank served the request.
func completionsByID(reqs []*sim.Request) map[string]int {
	out := map[string]int{}
	for _, r := range reqs {
		if r.State == sim.StateCompleted {
			out[r.ID] = r.GeneratedTokens
		}
	}
	return out
}

// TestExternalDP_ConservesAgainstCollapsed is the H2+H3 faithfulness check (BC-H2-4, BC-H3-2,
// INV-1): the external N-endpoint build and the collapsed single-instance build, run on the same
// workload, complete the same set of requests with the same per-request output. The external build
// exposes dp router endpoints over one shared group and drives them through the sim leader/member
// path (JoinGroup in buildDPReplicas); the collapsed build is one instance behind the internal
// balancer. Routing differs (that is the feature), so timings legitimately differ, but no request
// is created or dropped and each produces the output the workload fixed.
func TestExternalDP_ConservesAgainstCollapsed(t *testing.T) {
	const dp = 4
	cfg := dpExternalTestConfig()

	collapsedReqs := newTestRequests(40)
	collapsed := newTestCluster(collapsedDPDeployment(t, cfg, dp), NewSliceRequestSource(collapsedReqs), nil)
	mustRun(t, collapsed)

	externalReqs := newTestRequests(40)
	external := newTestCluster(externalDPDeployment(t, cfg, dp), NewSliceRequestSource(externalReqs), nil)
	mustRun(t, external)

	cc, ec := collapsed.AggregatedMetrics().CompletedRequests, external.AggregatedMetrics().CompletedRequests
	if cc == 0 {
		t.Fatal("the collapsed build completed no requests; the differential is vacuous")
	}
	if cc != ec {
		t.Fatalf("external build completed %d requests, collapsed %d: the N endpoints must neither create nor drop requests", ec, cc)
	}
	collapsedDone, externalDone := completionsByID(collapsedReqs), completionsByID(externalReqs)
	if len(collapsedDone) != len(externalDone) {
		t.Fatalf("external completed %d distinct requests, collapsed %d", len(externalDone), len(collapsedDone))
	}
	for id, tokens := range collapsedDone {
		got, ok := externalDone[id]
		if !ok {
			t.Errorf("request %s completed collapsed but not external", id)
			continue
		}
		if got != tokens {
			t.Errorf("request %s generated %d tokens external, %d collapsed: the output a request produces is the workload's, not the balancer's", id, got, tokens)
		}
	}
}

// TestExternalDP_EachEndpointReportsOnlyItsOwn is BC-H2-2's cluster-visible half: one leader steps
// the shared group, and each router endpoint's per-instance report counts only the requests routed
// to it — no rank's completions leak into another's report. (The leader-only-steps property itself
// is pinned at the driver level in package sim.)
func TestExternalDP_EachEndpointReportsOnlyItsOwn(t *testing.T) {
	const dp = 4
	cfg := dpExternalTestConfig()
	reqs := newTestRequests(32)
	cs := newTestCluster(externalDPDeployment(t, cfg, dp), NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)

	per := cs.PerInstanceMetrics()
	total := 0
	for _, m := range per {
		total += m.CompletedRequests
	}
	if got := cs.AggregatedMetrics().CompletedRequests; got != total {
		t.Fatalf("the endpoints' reports sum to %d completions but the cluster aggregate is %d: each endpoint must count only its own requests, with none double-counted", total, got)
	}
	if total == 0 {
		t.Fatal("no endpoint completed a request")
	}
}
