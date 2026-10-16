package llmd

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// evictionConfig is flow control with in-flight eviction (enableEviction): when higher-priority
// requests are blocked by pool saturation, in-flight sheddable (priority < 0) requests are
// revoked to reclaim capacity.
const evictionConfig = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
- type: queue-scorer
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: max-score-picker
  - pluginRef: queue-scorer
flowControl:
  enableEviction: true
`

// runEviction runs long sheddable requests that saturate the pool, then a burst of critical ones.
func runEviction(t *testing.T, config string) (runResult, map[string]int) {
	t.Helper()
	var reqs []*sim.Request
	for i, r := range testRequests(40, 5, 1000, 0) {
		r.ID, r.SLOClass = fmt.Sprintf("sheddable_%d", i), "sheddable"
		r.OutputTokens = sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(int64(i))), 2000) // long-running
		r.MaxOutputLen = len(r.OutputTokens)
		reqs = append(reqs, r)
	}
	for i, r := range testRequests(40, 6, 500, 0) {
		r.ID, r.SLOClass = fmt.Sprintf("critical_%d", i), "critical"
		r.ArrivalTime += 200_000
		reqs = append(reqs, r)
	}
	dep := testDeployment(2)
	dep.BatchConfig = sim.NewBatchConfig(8, 2048, 0)
	cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(reqs), nil)
	gw := New(Options{
		Model:     "test-model",
		EPPFlags:  []string{"--pool-name=vllm", "--config-text=" + config},
		Manifests: []byte(poolManifests),
		PodLabels: func(cluster.InstanceID, cluster.PoolRole) map[string]string { return map[string]string{"app": "vllm"} },
	})
	evicted := map[string]int{}
	gw.observe = func(kind, detail string) {
		if kind == "evict" {
			evicted[detail]++
		}
	}
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	return runResult{cs: cs, gw: gw, reqs: reqs, metrics: cs.AggregatedMetrics()}, evicted
}

// TestFlowControl_InFlightEviction (Discussion #1933 §5.7): with enableEviction, the router
// revokes in-flight sheddable requests while critical ones wait on the saturated pool. Each
// victim is a dispatched sheddable request, revoked once and counted gateway-evicted (INV-1); its
// engine frees its KV blocks; the router's response lifecycle for it ends (nothing stays tracked
// as in flight); and the run is deterministic.
func TestFlowControl_InFlightEviction(t *testing.T) {
	res, evicted := runEviction(t, evictionConfig)
	assertConservation(t, res, len(res.reqs))
	if res.cs.GatewayEvicted() == 0 {
		t.Fatal("no request was evicted in flight; the test does not exercise the path")
	}
	if len(evicted) != res.cs.GatewayEvicted() {
		t.Errorf("%d requests revoked, %d counted gateway-evicted", len(evicted), res.cs.GatewayEvicted())
	}
	byID := map[string]*sim.Request{}
	for _, r := range res.reqs {
		byID[r.ID] = r
	}
	for id, n := range evicted {
		r := byID[id]
		if n != 1 || r.SLOClass != "sheddable" || r.AssignedInstance == "" {
			t.Errorf("%s revoked %d times (class %s, instance %q); want each victim a dispatched sheddable request, once", id, n, r.SLOClass, r.AssignedInstance)
		}
	}
	for _, inst := range res.cs.Instances() {
		if u := inst.KVUtilization(); u != 0 {
			t.Errorf("%s: KV utilization %g at the end; evicted requests must free their blocks", inst.ID(), u)
		}
	}
	if inFlight, _ := res.gw.emb.RequestEvictor.Stats(); inFlight != 0 {
		t.Errorf("the router still tracks %d requests as in flight", inFlight)
	}
	if res.gw.LiveCoroutines() != 0 || len(res.gw.flights) != 0 {
		t.Errorf("%d coroutines, %d flights left after the run", res.gw.LiveCoroutines(), len(res.gw.flights))
	}
	again, evicted2 := runEviction(t, evictionConfig)
	if len(evicted2) != len(evicted) || again.metrics.CompletedRequests != res.metrics.CompletedRequests {
		t.Fatalf("two identical runs differ: %d vs %d evicted", len(evicted), len(evicted2))
	}
	for i := range res.reqs {
		if res.reqs[i].AssignedInstance != again.reqs[i].AssignedInstance || res.reqs[i].GatewayDispatchTime != again.reqs[i].GatewayDispatchTime {
			t.Fatalf("%s: placed differently in two identical runs", res.reqs[i].ID)
		}
	}
	t.Logf("evicted %d sheddable requests in flight; completed %d", len(evicted), res.metrics.CompletedRequests)
}
