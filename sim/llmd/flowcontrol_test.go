package llmd

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// softCeilingConfig is llm-d's deploy/config/soft-reflective-ceiling-epp-config.yaml.
const softCeilingConfig = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
- type: queue-scorer
- type: kv-cache-utilization-scorer
- type: prefix-cache-scorer
- type: soft-reflective-ceiling-policy
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: queue-scorer
    weight: 2.0
  - pluginRef: kv-cache-utilization-scorer
    weight: 2.0
  - pluginRef: prefix-cache-scorer
    weight: 3.0
flowControl:
  usageLimitPolicyPluginRef: soft-reflective-ceiling-policy
`

// globalStrictConfig is llm-d's test/testdata/plugins/stable/v1/global-strict-fairness-policy.yaml.
const globalStrictConfig = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
- type: single-profile-handler
- type: max-score-picker
- type: global-strict-fairness-policy
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: max-score-picker
flowControl:
  priorityBands:
  - priority: 0
    fairnessPolicyRef: global-strict-fairness-policy
`

func runBurst(t *testing.T, config string, n, numReqs int) runResult {
	t.Helper()
	reqs := testRequests(numReqs, 11, 1000, 0) // one arrival per millisecond
	dep := testDeployment(n)
	dep.BatchConfig = sim.NewBatchConfig(4, 2048, 0) // small batches: queues build and pools saturate
	cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(reqs), nil)
	gw := New(Options{Model: "test-model", EPPFlags: append(append([]string{}, DefaultEPPFlags...), "--allow-experimental-plugins", "--config-text="+config)})
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	return runResult{cs: cs, gw: gw, reqs: reqs, metrics: cs.AggregatedMetrics()}
}

func TestFlowControl_QueuesUnderSaturationAndDrains(t *testing.T) {
	for name, cfg := range map[string]string{"soft-reflective-ceiling": softCeilingConfig, "global-strict": globalStrictConfig} {
		t.Run(name, func(t *testing.T) {
			res := runBurst(t, cfg, 2, 120)
			assertConservation(t, res, 120)
			if res.metrics.CompletedRequests != 120 {
				t.Errorf("completed %d of 120", res.metrics.CompletedRequests)
			}
			held := 0
			for _, r := range res.reqs {
				if r.GatewayDispatchTime > r.ArrivalTime {
					held++
				}
			}
			if held == 0 {
				t.Errorf("no request waited in flow control under a 1 ms arrival interval on 2 saturated instances")
			}
			t.Logf("%d of 120 requests waited in flow control", held)
			if live := res.gw.LiveCoroutines(); live != 0 {
				t.Errorf("%d router coroutines alive after Stop", live)
			}
		})
	}
}

func TestFlowControl_Deterministic(t *testing.T) {
	a := runBurst(t, softCeilingConfig, 3, 90)
	b := runBurst(t, softCeilingConfig, 3, 90)
	for i := range a.reqs {
		ra, rb := a.reqs[i], b.reqs[i]
		if ra.AssignedInstance != rb.AssignedInstance || ra.GatewayDispatchTime != rb.GatewayDispatchTime {
			t.Fatalf("request %d: (%s, %d) then (%s, %d)", i, ra.AssignedInstance, ra.GatewayDispatchTime, rb.AssignedInstance, rb.GatewayDispatchTime)
		}
	}
}

func assertConservation(t *testing.T, res runResult, total int) {
	t.Helper()
	m := res.metrics
	sum := m.CompletedRequests + m.StillQueued + m.StillRunning + m.DroppedUnservable + m.TimedOutRequests +
		res.cs.RejectedRequests() + res.cs.RoutingRejections() + res.cs.GatewayQueueDepth() +
		res.cs.GatewayQueueRejected() + res.cs.GatewayEvicted() + res.cs.GatewayExpired() + res.cs.EncodeRoutingRejections()
	if sum != total {
		t.Errorf("INV-1: %d requests injected, %d accounted for (completed %d, queued %d, running %d, dropped %d, timed out %d, "+
			"rejected %d, routing-rejected %d, in gateway %d, gateway-rejected %d, evicted %d, expired %d)",
			total, sum, m.CompletedRequests, m.StillQueued, m.StillRunning, m.DroppedUnservable, m.TimedOutRequests,
			res.cs.RejectedRequests(), res.cs.RoutingRejections(), res.cs.GatewayQueueDepth(),
			res.cs.GatewayQueueRejected(), res.cs.GatewayEvicted(), res.cs.GatewayExpired())
	}
}

// boundedQueueConfig caps flow control at 10 queued requests with a 150 ms queue-wait budget,
// so an overload both rejects at capacity and expires queued requests.
const boundedQueueConfig = `apiVersion: llm-d.ai/v1
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
  - pluginRef: queue-scorer
  - pluginRef: max-score-picker
flowControl:
  maxRequests: 10
  defaultRequestTTL: 150ms
`

func TestFlowControl_INV1_UnderCapacityRejectionAndTTLExpiry(t *testing.T) {
	reqs := testRequests(300, 5, 200, 0) // 5000 req/s on 2 small instances
	dep := testDeployment(2)
	dep.BatchConfig = sim.NewBatchConfig(4, 2048, 0)
	cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(reqs), nil)
	gw := New(Options{Model: "test-model", EPPFlags: append(append([]string{}, DefaultEPPFlags...), "--config-text="+boundedQueueConfig)})
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	res := runResult{cs: cs, gw: gw, reqs: reqs, metrics: cs.AggregatedMetrics()}
	assertConservation(t, res, len(reqs))
	if cs.GatewayQueueRejected() == 0 {
		t.Errorf("no capacity rejection under a 10-request queue cap at 5000 req/s")
	}
	if cs.GatewayExpired() == 0 {
		t.Errorf("no TTL expiry under a 150 ms queue-wait budget at 5000 req/s")
	}
	t.Logf("completed %d, rejected %d, expired %d", res.metrics.CompletedRequests, cs.GatewayQueueRejected(), cs.GatewayExpired())
	if live := gw.LiveCoroutines(); live != 0 {
		t.Errorf("%d router coroutines alive after Stop", live)
	}
}
