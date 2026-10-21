package llmd

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// These tests exercise rows of the migration table in docs/guide/routing.md ("From BLIS's former
// routing features"): each former BLIS feature's llm-d configuration does, in a simulation, what
// the former feature was for.

// prefix-affinity scorer -> approx-prefix-cache-producer + prefix-cache-scorer: requests that
// share a prefix group stick to the instance that served the group first.
func TestMapping_PrefixAffinityKeepsPrefixGroupsTogether(t *testing.T) {
	const config = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: approx-prefix-cache-producer
- type: prefix-cache-scorer
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: prefix-cache-scorer
  - pluginRef: max-score-picker
`
	var reqs []*sim.Request
	for g := 0; g < 3; g++ {
		for i, r := range testRequests(20, int64(30+g), 0, 256) { // one 256-token prefix per group
			r.ID = fmt.Sprintf("g%d_%d", g, i)
			r.ArrivalTime = int64(i*3+g) * 20000 // groups interleaved, 20 ms apart
			reqs = append(reqs, r)
		}
	}
	cs := cluster.NewClusterSimulator(testDeployment(4), cluster.NewSliceRequestSource(reqs), nil)
	cs.SetGateway(New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--config-text="+config)}))
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	for g := 0; g < 3; g++ {
		where := map[string]int{}
		for _, r := range reqs[g*20 : (g+1)*20] {
			where[r.AssignedInstance]++
		}
		if len(where) != 1 {
			t.Errorf("prefix group %d spread over %v; the prefix scorer should keep it on one instance", g, where)
		}
	}
}

// --pd-decider prefix-threshold -> disagg-profile-handler with prefix-based-pd-decider
// (nonCachedTokens): a request whose uncached prompt is at most the threshold is not
// disaggregated, a longer one is, where "uncached" is what the router's prefix cache holds.
func TestMapping_PrefixBasedPDDecider(t *testing.T) {
	const config = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: approx-prefix-cache-producer
- type: prefix-cache-scorer
- type: queue-scorer
- type: prefill-filter
- type: decode-filter
- type: max-score-picker
- type: disagg-profile-handler
  parameters:
    deciders:
      prefill: prefix-based-pd-decider
- type: prefix-based-pd-decider
  parameters:
    nonCachedTokens: 200
schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: queue-scorer
  - pluginRef: max-score-picker
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: prefix-cache-scorer
  - pluginRef: queue-scorer
  - pluginRef: max-score-picker
`
	short := testRequests(30, 3, 5000, 0) // unique prompts of 32 to 95 tokens: below the threshold
	var unique []*sim.Request             // unique prompts of 300+ tokens: above it
	for i := 0; i < 15; i++ {
		r := testRequests(1, int64(100+i), 0, 300)[0]
		r.ID, r.ArrivalTime = fmt.Sprintf("unique_%d", i), 200000+int64(i)*5000
		unique = append(unique, r)
	}
	shared := testRequests(15, 4, 5000, 300) // one shared 300-token prefix: cached after the first
	for i, r := range shared {
		r.ID, r.ArrivalTime = fmt.Sprintf("shared_%d", i), 400000+int64(i)*5000
	}
	reqs := append(append(short, unique...), shared...)
	dep := testDeployment(4)
	dep.PrefillInstances, dep.DecodeInstances = 2, 2
	dep.PDTransferTime = func(tokens int64, _, _ cluster.InstanceID) int64 { return 50 + tokens/50 }
	cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(reqs), nil)
	cs.SetGateway(New(Options{Model: "test-model", PodLabels: roleLabels, EPPFlags: append(DefaultEPPFlags(), "--config-text="+config)}))
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	disagg := map[string]bool{}
	for _, p := range cs.ParentRequests() {
		disagg[p.OriginalRequest.ID] = true
	}
	check := func(group []*sim.Request, want func(i int) bool) {
		for i, r := range group {
			if disagg[r.ID] != want(i) {
				t.Errorf("%s (%d prompt tokens): disaggregated=%v, want %v", r.ID, len(r.InputTokens), disagg[r.ID], want(i))
			}
		}
	}
	check(short, func(int) bool { return false })
	check(unique, func(int) bool { return true })
	// The first shared-prefix request has 300+ uncached tokens; once its prefix is cached, the
	// others have only their 32 to 95-token suffix uncached.
	check(shared, func(i int) bool { return i == 0 })
}

// --fairness-policy / tenant budgets -> round-robin-fairness-policy over the fairness ID (the
// request's tenant): a tenant that floods the queue does not starve another.
func TestMapping_RoundRobinFairnessProtectsALightTenant(t *testing.T) {
	config := func(policy string) string {
		return `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
- type: single-profile-handler
- type: max-score-picker
- type: ` + policy + `
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: max-score-picker
flowControl:
  priorityBands:
  - priority: 0
    fairnessPolicyRef: ` + policy + `
`
	}
	lightWait := func(policy string) float64 {
		reqs := testRequests(400, 17, 500, 0)
		for i, r := range reqs {
			r.TenantID = "heavy"
			if i%10 == 0 {
				r.TenantID = "light"
			}
		}
		dep := testDeployment(2)
		dep.BatchConfig = sim.NewBatchConfig(4, 2048, 0)
		cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(reqs), nil)
		cs.SetGateway(New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--config-text="+config(policy))}))
		if err := cs.Run(); err != nil {
			t.Fatalf("%s: Run: %v", policy, err)
		}
		var sum float64
		n := 0
		for _, r := range reqs {
			if r.TenantID == "light" && r.AssignedInstance != "" {
				sum += float64(r.GatewayDispatchTime - r.ArrivalTime)
				n++
			}
		}
		return sum / float64(max(n, 1))
	}
	strict, rr := lightWait("global-strict-fairness-policy"), lightWait("round-robin-fairness-policy")
	t.Logf("light tenant's mean queue wait: global-strict %.0f us, round-robin %.0f us", strict, rr)
	if strict == 0 || rr >= strict/2 {
		t.Errorf("round-robin fairness did not protect the light tenant (wait %.0f us vs %.0f us under global-strict)", rr, strict)
	}
}
