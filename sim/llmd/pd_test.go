package llmd

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// pdConfig is llm-d's deploy/config/pd-epp-config.yaml.
const pdConfig = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: approx-prefix-cache-producer
  parameters:
    maxPrefixTokensToMatch: 16384
    lruCapacityPerServer: 31250
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
    nonCachedTokens: 16
schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: max-score-picker
  - pluginRef: prefix-cache-scorer
    weight: 2
  - pluginRef: queue-scorer
    weight: 1
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: max-score-picker
  - pluginRef: prefix-cache-scorer
    weight: 2
  - pluginRef: queue-scorer
    weight: 1
`

func roleLabels(_ cluster.InstanceID, role cluster.PoolRole) map[string]string {
	switch role {
	case cluster.PoolRolePrefill:
		return map[string]string{"llm-d.ai/role": "prefill"}
	case cluster.PoolRoleDecode:
		return map[string]string{"llm-d.ai/role": "decode"}
	case cluster.PoolRoleEncode:
		return map[string]string{"llm-d.ai/role": "encode"}
	}
	return nil
}

func TestPD_DisaggProfileHandlerPlacesPrefillAndDecode(t *testing.T) {
	reqs := testRequests(60, 3, 2000, 0)
	dep := testDeployment(4)
	dep.PrefillInstances, dep.DecodeInstances = 2, 2
	dep.PDTransferTime = func(tokens int64, _, _ cluster.InstanceID) int64 { return 50 + tokens/50 }
	cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), nil)
	gw := New(Options{Model: "test-model", PodLabels: roleLabels,
		EPPFlags: append(DefaultEPPFlags(), "--config-text="+pdConfig)})
	usage := map[string]int{}
	gw.observe = func(kind, detail string) {
		if kind == "usage" {
			var id string
			var n int
			_, _ = fmt.Sscanf(detail, "%s %d", &id, &n)
			usage[id] = n
		}
	}
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	m := cs.AggregatedMetrics()
	if m.CompletedRequests != 60 {
		t.Fatalf("completed %d of 60 (rejected %d, routing-rejected %d, dropped %d)", m.CompletedRequests,
			cs.RejectedRequests(), cs.RoutingRejections(), m.DroppedUnservable)
	}
	parents := cs.ParentRequests()
	if len(parents) == 0 {
		t.Fatal("the decider never disaggregated, although every prompt has more than 16 uncached tokens")
	}
	role := cs.PoolMembership()
	for _, p := range parents {
		if !role[string(p.PrefillInstanceID)].Has(cluster.PoolRolePrefill) || !role[string(p.DecodeInstanceID)].Has(cluster.PoolRoleDecode) {
			t.Fatalf("parent %s: prefill on %s (%v), decode on %s (%v)", p.ID, p.PrefillInstanceID, role[string(p.PrefillInstanceID)],
				p.DecodeInstanceID, role[string(p.DecodeInstanceID)])
		}
	}
	for _, p := range parents {
		// The router's response usage reports the tokens the decode child generated.
		if want := len(p.OriginalRequest.OutputTokens); usage[p.OriginalRequest.ID] != want {
			t.Errorf("parent %s: router saw completion_tokens %d, want %d", p.ID, usage[p.OriginalRequest.ID], want)
		}
	}
	t.Logf("%d of 60 requests disaggregated", len(parents))
	if live := gw.LiveCoroutines(); live != 0 {
		t.Errorf("%d router coroutines alive after Stop", live)
	}
}

var _ = sim.StateQueued
