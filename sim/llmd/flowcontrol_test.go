package llmd

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/blis-schemas/embed"

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
	editEngine(&dep, func(c *cengine.SimConfig) { c.BatchConfig = cengine.NewBatchConfig(4, 2048, 0) }) // small batches: queues build and pools saturate
	cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), nil)
	gw := New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--allow-experimental-plugins", "--config-text="+config)})
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
	editEngine(&dep, func(c *cengine.SimConfig) { c.BatchConfig = cengine.NewBatchConfig(4, 2048, 0) })
	cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), nil)
	gw := New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--config-text="+boundedQueueConfig)})
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

// orderingConfig is globalStrictConfig with the band's ordering policy set to policy.
func orderingConfig(policy string) string {
	return `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
- type: single-profile-handler
- type: max-score-picker
- type: global-strict-fairness-policy
- type: ` + policy + `
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: max-score-picker
flowControl:
  priorityBands:
  - priority: 0
    fairnessPolicyRef: global-strict-fairness-policy
    orderingPolicyRef: ` + policy + `
`
}

// TestFlowControl_SLODeadlineOrdersBySimulatedTargets: a request's SLO target reaches the router
// as x-llm-d-slo-ttft-ms, so slo-deadline-ordering-policy dispatches SLO-bound requests ahead of
// requests without a target, where fcfs-ordering-policy does not.
func TestFlowControl_SLODeadlineOrdersBySimulatedTargets(t *testing.T) {
	gap := func(policy string) float64 {
		reqs := testRequests(120, 11, 1000, 0)
		for i, r := range reqs {
			if i%2 == 1 {
				r.SLOTargetUs = 2000
			}
		}
		dep := testDeployment(2)
		editEngine(&dep, func(c *cengine.SimConfig) { c.BatchConfig = cengine.NewBatchConfig(4, 2048, 0) })
		cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), nil)
		gw := New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--config-text="+orderingConfig(policy))})
		cs.SetGateway(gw)
		if err := cs.Run(); err != nil {
			t.Fatalf("%s: Run: %v", policy, err)
		}
		assertConservation(t, runResult{cs: cs, gw: gw, reqs: reqs, metrics: cs.AggregatedMetrics()}, len(reqs))
		var wait [2]float64
		for i, r := range reqs {
			wait[i%2] += float64(r.GatewayDispatchTime - r.ArrivalTime)
		}
		return (wait[0] - wait[1]) / float64(len(reqs)/2) // untargeted minus targeted mean wait, µs
	}
	fcfs, slo := gap("fcfs-ordering-policy"), gap("slo-deadline-ordering-policy")
	t.Logf("mean wait advantage of SLO-bound requests: fcfs %.0f µs, slo-deadline %.0f µs", fcfs, slo)
	if slo <= 0 || slo <= fcfs {
		t.Errorf("slo-deadline-ordering-policy gave SLO-bound requests no advantage over fcfs (%.0f vs %.0f µs)", slo, fcfs)
	}
}

// TestFlowControl_RequestTTLHeaderOrdersByClientDeadlines: a request's client lifetime reaches
// the router as x-llm-d-inference-ttl, so edf-ordering-policy (earliest enqueue + TTL first)
// dispatches requests with a deadline ahead of requests without one, where fcfs-ordering-policy
// does not (the header only shortens the configured budget, 60 s by default); no deadline is
// reached, and the run is deterministic.
func TestFlowControl_RequestTTLHeaderOrdersByClientDeadlines(t *testing.T) {
	run := func(policy string) (float64, runResult) {
		reqs := testRequests(120, 11, 1000, 0)
		for i, r := range reqs {
			if i%2 == 1 {
				r.Deadline = r.ArrivalTime + 30_000_000 // shorter than the 60 s default TTL; never reached
			}
		}
		dep := testDeployment(2)
		editEngine(&dep, func(c *cengine.SimConfig) { c.BatchConfig = cengine.NewBatchConfig(4, 2048, 0) })
		cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), nil)
		gw := New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--config-text="+orderingConfig(policy))})
		cs.SetGateway(gw)
		if err := cs.Run(); err != nil {
			t.Fatalf("%s: Run: %v", policy, err)
		}
		res := runResult{cs: cs, gw: gw, reqs: reqs, metrics: cs.AggregatedMetrics()}
		assertConservation(t, res, len(reqs))
		if res.metrics.CompletedRequests != len(reqs) {
			t.Fatalf("%s: completed %d of %d", policy, res.metrics.CompletedRequests, len(reqs))
		}
		var wait [2]float64
		for i, r := range reqs {
			wait[i%2] += float64(r.GatewayDispatchTime - r.ArrivalTime)
		}
		return (wait[0] - wait[1]) / float64(len(reqs)/2), res // no-deadline minus deadline mean wait, µs
	}
	fcfs, _ := run("fcfs-ordering-policy")
	edf, a := run("edf-ordering-policy")
	_, b := run("edf-ordering-policy")
	t.Logf("mean wait advantage of requests with a deadline: fcfs %.0f µs, edf %.0f µs", fcfs, edf)
	if edf <= 0 || edf <= fcfs {
		t.Errorf("edf-ordering-policy gave requests with a deadline no advantage over fcfs (%.0f vs %.0f µs)", edf, fcfs)
	}
	for i := range a.reqs {
		if a.reqs[i].GatewayDispatchTime != b.reqs[i].GatewayDispatchTime || a.reqs[i].AssignedInstance != b.reqs[i].AssignedInstance {
			t.Fatalf("%s: dispatched differently in two identical runs", a.reqs[i].ID)
		}
	}
}

// TestFlowControl_QueueDelayRecorded: under flow control each dispatched request carries the
// time it entered the router's queue, so gateway_queue_delay_ms is reported; without flow control
// there is no queue and no enqueue time.
func TestFlowControl_QueueDelayRecorded(t *testing.T) {
	res := runBurst(t, globalStrictConfig, 2, 60)
	positive := 0
	for _, r := range res.reqs {
		if r.GatewayEnqueueTime != r.ArrivalTime {
			t.Fatalf("%s: enqueued at %d, arrived at %d with no gateway latency", r.ID, r.GatewayEnqueueTime, r.ArrivalTime)
		}
		if r.GatewayDispatchTime > r.GatewayEnqueueTime {
			positive++
		}
	}
	if positive == 0 {
		t.Errorf("no request waited in flow control")
	}
	plain := runCluster(t, 2, 20, Options{})
	for _, r := range plain.reqs {
		if r.GatewayEnqueueTime != 0 {
			t.Fatalf("%s: enqueue time %d without flow control", r.ID, r.GatewayEnqueueTime)
		}
	}
}

// TestFlowControl_ClientDeadlineWhileQueued: a request whose client deadline passes while flow
// control holds it leaves the queue at its deadline and counts as timed out, not as a router
// expiry or capacity rejection.
func TestFlowControl_ClientDeadlineWhileQueued(t *testing.T) {
	// Repeated: a cancellation delivered asynchronously (a context type the stdlib must watch
	// from a goroutine) races the simulation, so one run alone may not show it.
	for range 8 {
		clientDeadlineWhileQueued(t)
		if t.Failed() {
			return
		}
	}
}

func clientDeadlineWhileQueued(t *testing.T) {
	t.Helper()
	reqs := testRequests(150, 11, 1000, 0)
	for _, r := range reqs {
		r.Deadline = r.ArrivalTime + 20000 // the client waits at most 20 ms
	}
	dep := testDeployment(2)
	editEngine(&dep, func(c *cengine.SimConfig) { c.BatchConfig = cengine.NewBatchConfig(4, 2048, 0) })
	cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), nil)
	gw := New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--config-text="+globalStrictConfig)})
	// Each held request's cancellation reaches the router at its client deadline, and the
	// router's ContextCancelled outcome lands at that same simulated instant (INV-6: no
	// cancellation is delivered later, by a goroutine outside the runtime).
	byID := map[string]*sim.Request{}
	for _, r := range reqs {
		byID[r.ID] = r
	}
	cancelled := 0
	gw.observe = func(kind, detail string) {
		if kind != "rejected" {
			return
		}
		var id, k string
		var at int64
		if _, err := fmt.Sscanf(detail, "%s %s %d", &id, &k, &at); err != nil {
			t.Fatalf("observe %q: %v", detail, err)
		}
		if k == string(embed.ContextCancelled) {
			cancelled++
			if at != byID[id].Deadline {
				t.Errorf("%s: cancelled at %d, its client deadline is %d", id, at, byID[id].Deadline)
			}
		}
	}
	cs.SetGateway(gw)
	var final sim.ProgressSnapshot
	cs.SetProgressHook(progressFunc(func(s sim.ProgressSnapshot) { final = s }), 1_000_000)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	res := runResult{cs: cs, gw: gw, reqs: reqs, metrics: cs.AggregatedMetrics()}
	assertConservation(t, res, len(reqs))
	if final.TotalTimedOut != res.metrics.TimedOutRequests {
		t.Errorf("final progress snapshot: %d timed out, aggregated metrics %d", final.TotalTimedOut, res.metrics.TimedOutRequests)
	}
	if cs.GatewayExpired() != 0 || cs.GatewayQueueRejected() != 0 {
		t.Errorf("client deadlines counted as router outcomes: expired %d, rejected %d", cs.GatewayExpired(), cs.GatewayQueueRejected())
	}
	timedOutInGateway := 0
	for _, r := range reqs {
		if r.AssignedInstance == "" {
			timedOutInGateway++
		}
		if r.GatewayDispatchTime > r.Deadline {
			t.Errorf("%s dispatched at %d, after its client deadline %d", r.ID, r.GatewayDispatchTime, r.Deadline)
		}
	}
	if timedOutInGateway == 0 || cancelled == 0 {
		t.Fatalf("%d requests timed out while held in flow control, %d cancelled in the router; the test does not exercise the path", timedOutInGateway, cancelled)
	}
	if res.metrics.TimedOutRequests < timedOutInGateway {
		t.Errorf("timed_out_requests %d < %d requests whose deadline passed in the gateway", res.metrics.TimedOutRequests, timedOutInGateway)
	}
}

type progressFunc func(sim.ProgressSnapshot)

func (f progressFunc) OnProgress(s sim.ProgressSnapshot) { f(s) }

// TestHeldUntil: the latest time the router's own timers can release a held request: its client
// deadline, or its queue-wait budget plus one expiry sweep, whichever comes first; requests with
// neither, and requests no longer held, do not count.
func TestHeldUntil(t *testing.T) {
	g := &Gateway{flights: map[string]*flight{}, maxQueueTTL: 60_000_000, expirySweepUs: 1_000_000}
	add := func(id string, held bool, enqueue, deadline int64) {
		g.flights[id] = &flight{req: &sim.Request{ID: id, GatewayEnqueueTime: enqueue, Deadline: deadline}, held: held}
	}
	add("ttl-first", true, 1_000_000, 7_200_000_000)   // TTL at 61 s + sweep
	add("deadline-first", true, 2_000_000, 10_000_000) // deadline at 10 s
	add("dispatched", false, 0, 9_000_000_000)
	if got, want := g.HeldUntil(0), int64(1_000_000+60_000_000+1_000_000); got != want {
		t.Errorf("HeldUntil = %d, want %d", got, want)
	}
	// A later event (a change of pool regime) starts a fresh budget.
	if got, want := g.HeldUntil(30_000_000), int64(30_000_000+60_000_000+1_000_000); got != want {
		t.Errorf("HeldUntil after a later event = %d, want %d", got, want)
	}
	g.maxQueueTTL = 0 // no bounded budget: only deadlines release
	if got, want := g.HeldUntil(0), int64(7_200_000_000); got != want {
		t.Errorf("no bounded TTL: HeldUntil = %d, want %d", got, want)
	}
}

// TestFlowControl_TTLOverAnHourIsNotAStall: a request flow control holds with no client deadline,
// under a queue-wait budget over an hour, expires at that budget; the one-hour "no progress" limit
// does not fail the run first. An unbounded budget elsewhere (the empty-pool regime, another band)
// does not erase the bounded one (reviews 6205113308, 6205596017).
func TestFlowControl_TTLOverAnHourIsNotAStall(t *testing.T) {
	if testing.Short() {
		t.Skip("ticks flow control through an hour of simulated time")
	}
	for name, flowControl := range map[string]string{
		"global budget":        "  defaultRequestTTL: 61m\n",
		"unbounded empty-pool": "  defaultRequestTTL: 61m\n  noEndpointRequestTTL: 0s\n",
		"unbounded other band": "  defaultRequestTTL: 61m\n  priorityBands:\n  - priority: 7\n    defaultRequestTTL: 0s\n",
	} {
		t.Run(name, func(t *testing.T) {
			config := `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
- type: single-profile-handler
- type: max-score-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: max-score-picker
flowControl:
` + flowControl
			reqs := testRequests(1, 3, 0, 0)
			reqs[0].ArrivalTime = 1_000_000 // after the startup scrape has gone stale
			cs := newCluster(testDeployment(1), cluster.NewSliceRequestSource(reqs), nil)
			// Metrics refreshed every three hours: after the startup scrape they are stale, which the
			// utilization detector reads as saturated, so flow control holds the request.
			cs.SetGateway(New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--refresh-metrics-interval=3h", "--config-text="+config)}))
			if err := cs.Run(); err != nil {
				t.Fatalf("Run: %v", err)
			}
			if reqs[0].AssignedInstance != "" || cs.GatewayExpired() != 1 {
				t.Fatalf("dispatched=%q, expired=%d: want the held request to expire at its 61-minute budget", reqs[0].AssignedInstance, cs.GatewayExpired())
			}
		})
	}
}

// TestClientDeadline_SameInstantAsTheDecision: when a request's client deadline falls at the
// instant the Director decides it, the decision stands: the request is dispatched with its
// context live (response hooks see a live request), never cancelled after the decision.
func TestClientDeadline_SameInstantAsTheDecision(t *testing.T) {
	reqs := testRequests(5, 3, 1000, 0)
	for _, r := range reqs {
		r.Deadline = r.ArrivalTime // due the instant the router decides
	}
	cs := newCluster(testDeployment(2), cluster.NewSliceRequestSource(reqs), nil)
	gw := New(Options{Model: "test-model"})
	dispatched := 0
	gw.observe = func(kind, detail string) {
		if kind == "dispatch" {
			dispatched++
			if !strings.HasSuffix(detail, " <nil>") {
				t.Errorf("dispatched with a cancelled context: %s", detail)
			}
		}
	}
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if dispatched != len(reqs) {
		t.Fatalf("%d of %d dispatched", dispatched, len(reqs))
	}
}

// TestRequestTTL: a client lifetime is sent as a positive Go duration, up to the longest a Go
// duration holds; a longer one (which would overflow negative) and a non-positive one send none.
func TestRequestTTL(t *testing.T) {
	for _, tc := range []struct {
		life int64
		want time.Duration
		ok   bool
	}{
		{30_000_000, 30 * time.Second, true},
		{1, time.Microsecond, true},
		{maxTTLMicros, time.Duration(maxTTLMicros) * time.Microsecond, true},
		{maxTTLMicros + 1, 0, false},
		{math.MaxInt64, 0, false},
		{0, 0, false},
		{-5, 0, false},
	} {
		got, ok := requestTTL(tc.life)
		if got != tc.want || ok != tc.ok {
			t.Errorf("requestTTL(%d) = %v, %v; want %v, %v", tc.life, got, ok, tc.want, tc.ok)
		}
		if ok && got <= 0 {
			t.Errorf("requestTTL(%d) = %v: not a positive Go duration", tc.life, got)
		}
	}
}
