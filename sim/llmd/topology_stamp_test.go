package llmd

import (
	"maps"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/cluster"
	request "github.com/llm-d/llm-d-router/pkg/common/request"
)

// This exercises the real EPP's response-header plugin. BLIS runs one EPP, so
// the stamped header is observable but is not forwarded to a second decode EPP.
func TestGateway_TopologyStampSelectedPrefiller(t *testing.T) {
	for _, tc := range []struct {
		name        string
		skipPrefill bool
	}{
		{name: "prefill selected"},
		{name: "prefill skipped", skipPrefill: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dep := testDeployment(2)
			dep.PrefillInstances, dep.DecodeInstances = 1, 1
			dep.PDTransferTime = func(int64, cluster.InstanceID, cluster.InstanceID) int64 { return 50 }
			dep.ScenarioEndpointTopology = cluster.StaticEndpointTopology(map[cluster.InstanceID]cluster.EndpointTopology{
				"instance_0": {Hostname: "prefill-a", Rack: "rack-a", NodeCount: 1},
				"instance_1": {Hostname: "decode-b", Rack: "rack-b", NodeCount: 1},
			})
			cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(testRequests(1, 11, 0, 0)), nil)
			gw := New(Options{Model: "test-model", PodLabels: roleLabels,
				Manifests: []byte(declaredNode("prefill-a", "zone-a", "region-a") + declaredNode("decode-b", "zone-b", "region-b")),
				EPPFlags:  append(DefaultEPPFlags(), "--allow-experimental-plugins", "--config-text="+stampPDConfig(tc.skipPrefill))})
			var got []map[string]string
			gw.responseHeaderObserved = func(_ string, headers map[string]string) {
				got = append(got, maps.Clone(headers))
			}
			cs.SetGateway(gw)
			if err := cs.Run(); err != nil {
				t.Fatalf("topology stamp run: %v", err)
			}
			if len(got) != 1 {
				t.Fatalf("observed %d responses, want 1", len(got))
			}
			stamp := got[0][request.PeerTopologyHeaderKey]
			if tc.skipPrefill {
				if len(cs.ParentRequests()) != 0 {
					t.Fatalf("prefill ran despite the decider skipping it")
				}
				if stamp != "" {
					t.Fatalf("response stamped topology with no prefill selection: %q", stamp)
				}
				return
			}
			if len(cs.ParentRequests()) != 1 || cs.ParentRequests()[0].PrefillInstanceID != "instance_0" {
				t.Fatalf("prefill placement = %v, want instance_0", cs.ParentRequests())
			}
			if want := "host=prefill-a,rack=rack-a,zone=zone-a,region=region-a"; stamp != want {
				t.Fatalf("stamped topology = %q, want %q", stamp, want)
			}
		})
	}
}

func stampPDConfig(skipPrefill bool) string {
	deciderPlugin, deciderRef, stageOrder := "", "", "prefill-first"
	if skipPrefill {
		deciderPlugin = "- type: prefix-based-pd-decider\n  parameters:\n    nonCachedTokens: 0\n"
		deciderRef = "    deciders:\n      prefill: prefix-based-pd-decider\n"
		stageOrder = "decode-first"
	}
	return `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: prefill-filter
- type: decode-filter
- type: max-score-picker
- type: topology-extractor
- type: topology-stamp-handler
  parameters:
    profileName: prefill
` + deciderPlugin + `- type: disagg-profile-handler
  parameters:
    stageOrder: ` + stageOrder + "\n" + deciderRef + `schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: max-score-picker
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: max-score-picker
`
}

func TestTopologyConfig_StampReferences(t *testing.T) {
	for _, tc := range []struct{ name, plugins, want string }{
		{"missing extractor", "- type: topology-stamp-handler\n", "needs topology-extractor"},
		{"wrong producer", "- type: topology-extractor\n  name: locations\n- type: topology-stamp-handler\n  parameters:\n    topologyProducerName: other\n", "other"},
		{"wrong header", "- type: topology-extractor\n- type: topology-stamp-handler\n  parameters:\n    headerName: x-other\n", "headerName"},
		{"wrong profile", "- type: topology-extractor\n- type: topology-stamp-handler\n  parameters:\n    profileName: missing\n", "profileName"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := []byte("plugins:\n" + tc.plugins + "schedulingProfiles:\n- name: prefill\n")
			_, err := topologyConfig(raw)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("topologyConfig error = %v, want %q", err, tc.want)
			}
		})
	}
}
