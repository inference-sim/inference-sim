package llmd

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/cluster"
	corev1 "k8s.io/api/core/v1"
)

const topologyPDConfig = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: prefill-filter
- type: decode-filter
- type: max-score-picker
- type: topology-extractor
- type: topology-affinity-filter
  parameters:
    minAffinity: host
- type: disagg-profile-handler
  parameters:
    stageOrder: prefill-first
schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: max-score-picker
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: topology-affinity-filter
  - pluginRef: max-score-picker
`

// The real EPP must read the synthetic Pod's labels as endpoint topology and
// change its decode choice when only the matching host changes. Requests merely
// succeeding would not catch the filter's fail-open behavior.
func TestGateway_TopologyExtractorChangesDecodeChoice(t *testing.T) {
	for _, matchingDecode := range []cluster.InstanceID{"instance_1", "instance_2"} {
		t.Run(string(matchingDecode), func(t *testing.T) {
			dep := testDeployment(3)
			dep.PrefillInstances, dep.DecodeInstances = 1, 2
			dep.PDTransferTime = func(int64, cluster.InstanceID, cluster.InstanceID) int64 { return 50 }
			placement := map[cluster.InstanceID]cluster.EndpointTopology{
				"instance_0": {Hostname: "host-a", NodeCount: 1},
				"instance_1": {Hostname: "host-b", NodeCount: 1},
				"instance_2": {Hostname: "host-b", NodeCount: 1},
			}
			placement[matchingDecode] = cluster.EndpointTopology{Hostname: "host-a", NodeCount: 1}
			dep.ScenarioEndpointTopology = cluster.StaticEndpointTopology(placement)
			cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(testRequests(6, 11, 50_000, 0)), nil)
			gw := New(Options{Model: "test-model", PodLabels: roleLabels,
				EPPFlags: append(DefaultEPPFlags(), "--allow-experimental-plugins", "--config-text="+topologyPDConfig)})
			cs.SetGateway(gw)
			if err := cs.Run(); err != nil {
				t.Fatalf("topology-aware run: %v", err)
			}
			for _, p := range cs.ParentRequests() {
				if p.DecodeInstanceID != matchingDecode {
					t.Errorf("request %s chose %s, want local %s", p.ID, p.DecodeInstanceID, matchingDecode)
				}
			}
			if len(cs.ParentRequests()) != 6 {
				t.Fatalf("disaggregated %d of 6 requests", len(cs.ParentRequests()))
			}
			if len(gw.endpoints) != 3 {
				t.Fatalf("the router was given %d endpoints, want 3", len(gw.endpoints))
			}
			for id, ep := range gw.endpoints {
				if ep.ep.Labels[corev1.LabelHostname] == "" || ep.ep.Labels[corev1.LabelHostname] != ep.ep.NodeName {
					t.Errorf("endpoint %s hostname label %q, node %q; want both the simulated host", id, ep.ep.Labels[corev1.LabelHostname], ep.ep.NodeName)
				}
			}
		})
	}
}

const topologyExtractorOnly = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: topology-extractor
- type: max-score-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: max-score-picker
`

// BLIS must not silently omit placement data that a configured topology
// extractor expects, or claim one host for a worker spanning multiple hosts.
func TestGateway_TopologyPlacementRefusals(t *testing.T) {
	for _, tc := range []struct {
		name, want string
		setup      func(*cluster.DeploymentConfig) Options
	}{
		{"unknown", "no simulated host placement", func(*cluster.DeploymentConfig) Options { return Options{} }},
		{"hostname without single-node placement", "no simulated host placement", func(dep *cluster.DeploymentConfig) Options {
			dep.ScenarioEndpointTopology = cluster.StaticEndpointTopology(map[cluster.InstanceID]cluster.EndpointTopology{"instance_0": {Hostname: "blis-node-0"}})
			return Options{}
		}},
		{"multi-node TP", "spans 2 nodes", func(dep *cluster.DeploymentConfig) Options {
			dep.ScenarioEndpointTopology = cluster.StaticEndpointTopology(map[cluster.InstanceID]cluster.EndpointTopology{"instance_0": {NodeCount: 2}})
			return Options{}
		}},
		{"NodePool placement without kernel alignment", "no simulated host placement", func(dep *cluster.DeploymentConfig) Options {
			dep.NodePools = []cluster.NodePoolConfig{{Name: "p", GPUType: "H100", GPUsPerNode: 1, GPUMemoryGiB: 80, InitialNodes: 1, MaxNodes: 1}}
			return Options{}
		}},
		{"kernel and NodePool placement", "both NodePool placement and independent kernel", func(dep *cluster.DeploymentConfig) Options {
			dep.NodePools = []cluster.NodePoolConfig{{Name: "p", GPUType: "H100", GPUsPerNode: 1, GPUMemoryGiB: 80, InitialNodes: 1, MaxNodes: 1}}
			dep.ScenarioEndpointTopology = cluster.StaticEndpointTopology(map[cluster.InstanceID]cluster.EndpointTopology{"instance_0": {Hostname: "blis-node-0", NodeCount: 1}})
			return Options{}
		}},
		{"conflicting label", "conflicts with simulated placement", func(dep *cluster.DeploymentConfig) Options {
			dep.ScenarioEndpointTopology = cluster.StaticEndpointTopology(map[cluster.InstanceID]cluster.EndpointTopology{"instance_0": {Hostname: "blis-node-0", NodeCount: 1}})
			return Options{PodLabels: func(cluster.InstanceID, cluster.PoolRole) map[string]string {
				return map[string]string{corev1.LabelHostname: "a-different-node"}
			}}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dep := testDeployment(1)
			opts := tc.setup(&dep)
			opts.Model = "test-model"
			opts.EPPFlags = append(DefaultEPPFlags(), "--allow-experimental-plugins", "--config-text="+topologyExtractorOnly)
			cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(testRequests(1, 9, 500, 0)), nil)
			cs.SetGateway(New(opts))
			if err := cs.Run(); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("run error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestTopologyConfigRefusesUnsupportedParameterizations(t *testing.T) {
	for _, tc := range []struct{ name, plugins, want string }{
		{"custom hostname label", "- type: topology-extractor\n  parameters:\n    hostname: example.com/node\n", "hostname label"},
		{"peer header", "- type: topology-extractor\n- type: topology-affinity-filter\n  parameters:\n    peerTopologyHeader: x-peer-topology\n", "requires a coordinator"},
		{"filter without extractor", "- type: topology-affinity-filter\n", "needs topology-extractor"},
		{"wrong producer name", "- type: topology-extractor\n  name: topology-a\n- type: topology-affinity-filter\n  parameters:\n    topologyProducerName: topology-b\n", "topology-b"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := refusal(t, placedDeployment(), topologyConfigText(tc.plugins), nil, "--allow-experimental-plugins")
			if err == nil || !strings.Contains(err.Error(), tc.want) || !strings.Contains(err.Error(), "cannot be simulated faithfully") {
				t.Errorf("startup error = %v, want %q", err, tc.want)
			}
		})
	}
	valid := "- type: topology-extractor\n  name: topology-a\n- type: topology-affinity-filter\n  parameters:\n    topologyProducerName: topology-a\n"
	config := "apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\nplugins:\n" + valid +
		"- type: single-profile-handler\n- type: max-score-picker\nschedulingProfiles:\n- name: default\n  plugins:\n  - pluginRef: topology-affinity-filter\n  - pluginRef: max-score-picker\n"
	if err := refusal(t, placedDeployment(), config, nil, "--allow-experimental-plugins"); err != nil {
		t.Errorf("matching named producer must run: %v", err)
	}
}

// placedDeployment is one instance on one simulated host, so topology checks pass placement.
func placedDeployment() cluster.DeploymentConfig {
	dep := testDeployment(1)
	dep.ScenarioEndpointTopology = cluster.StaticEndpointTopology(map[cluster.InstanceID]cluster.EndpointTopology{
		"instance_0": {Hostname: "blis-node-0", NodeCount: 1},
	})
	return dep
}

// topologyConfigText is an EndpointPickerConfig with plugins and a single default profile.
func topologyConfigText(plugins string) string {
	return "apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\nplugins:\n" + plugins +
		"- type: max-score-picker\nschedulingProfiles:\n- name: default\n  plugins:\n  - pluginRef: max-score-picker\n- name: prefill\n  plugins:\n  - pluginRef: max-score-picker\n"
}

// Internal data-parallel load balancing exposes one Pod for multiple ranks.
// Their individual host placements must not become that Pod's claimed locality.
func TestGateway_GroupedEndpointHasNoSingleHostTopology(t *testing.T) {
	placement := cluster.StaticEndpointTopology(map[cluster.InstanceID]cluster.EndpointTopology{
		"instance_0": {Hostname: "host-a", NodeCount: 1},
		"instance_1": {Hostname: "host-b", NodeCount: 1},
	})
	for _, topologyEnabled := range []bool{false, true} {
		name := "plain"
		if topologyEnabled {
			name = "topology"
		}
		t.Run(name, func(t *testing.T) {
			dep := testDeployment(2)
			dep.DPLockstepRanks = 2
			dep.ScenarioEndpointTopology = placement
			flags := DefaultEPPFlags()
			if topologyEnabled {
				flags = append(flags, "--allow-experimental-plugins", "--config-text="+topologyExtractorOnly)
			}
			gw := New(Options{Model: "test-model", EPPFlags: flags})
			cs := cluster.NewClusterSimulator(dep, cluster.NewSliceRequestSource(testRequests(1, 9, 500, 0)), nil)
			cs.SetGateway(gw)
			err := cs.Run()
			if topologyEnabled {
				if err == nil || !strings.Contains(err.Error(), "endpoint spans multiple data-parallel ranks") {
					t.Fatalf("run error = %v, want grouped-endpoint refusal", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("plain run: %v", err)
			}
			if len(gw.endpoints) != 1 {
				t.Fatalf("got %d EPP endpoints, want one grouped endpoint", len(gw.endpoints))
			}
			for _, ep := range gw.endpoints {
				if ep.ep.NodeName != "" || ep.ep.Labels[corev1.LabelHostname] != "" {
					t.Errorf("grouped Pod claims one host: node=%q labels=%v", ep.ep.NodeName, ep.ep.Labels)
				}
			}
		})
	}
}
