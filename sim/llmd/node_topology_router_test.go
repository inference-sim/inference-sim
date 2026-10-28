package llmd

import (
	"fmt"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// The synthetic Nodes supply explicit geography; the kernel placement still
// decides each worker's host and rack. This runs the real topology extractor
// and filter, rather than asserting only that labels were copied.
func TestGateway_DeclaredZoneAndRegionChangeDecodeChoice(t *testing.T) {
	for _, tc := range []struct {
		level       string
		policy      string
		matchingID  cluster.InstanceID
		decode1Zone string
		decode1Reg  string
		decode2Zone string
		decode2Reg  string
	}{
		{"zone", "filter", "instance_1", "zone-a", "region-a", "zone-b", "region-a"},
		{"zone", "filter", "instance_2", "zone-b", "region-a", "zone-a", "region-a"},
		{"region", "filter", "instance_1", "zone-b", "region-a", "zone-c", "region-b"},
		{"region", "filter", "instance_2", "zone-c", "region-b", "zone-b", "region-a"},
		{"zone", "scorer", "instance_1", "zone-a", "region-a", "zone-b", "region-a"},
		{"region", "scorer", "instance_2", "zone-c", "region-b", "zone-b", "region-a"},
	} {
		t.Run(tc.level+"/"+tc.policy+"/"+string(tc.matchingID), func(t *testing.T) {
			dep := testDeployment(3)
			dep.PrefillInstances, dep.DecodeInstances = 1, 2
			dep.PDTransferTime = func(int64, cluster.InstanceID, cluster.InstanceID) int64 { return 50 }
			dep.ScenarioEndpointTopology = cluster.StaticEndpointTopology(map[cluster.InstanceID]cluster.EndpointTopology{
				"instance_0": {Hostname: "blis-node-0", Rack: "blis-rack-0", NodeCount: 1},
				"instance_1": {Hostname: "blis-node-1", Rack: "blis-rack-1", NodeCount: 1},
				"instance_2": {Hostname: "blis-node-2", Rack: "blis-rack-2", NodeCount: 1},
			})
			manifests := declaredNode("blis-node-0", "zone-a", "region-a") +
				declaredNode("blis-node-1", tc.decode1Zone, tc.decode1Reg) +
				declaredNode("blis-node-2", tc.decode2Zone, tc.decode2Reg)
			config := strings.Replace(topologyPDConfig, "minAffinity: host", "minAffinity: "+tc.level, 1)
			if tc.policy == "scorer" {
				config = strings.Replace(config, "- type: topology-affinity-filter\n  parameters:\n    minAffinity: "+tc.level, "- type: topology-affinity-scorer", 1)
				config = strings.Replace(config, "- pluginRef: topology-affinity-filter", "- pluginRef: topology-affinity-scorer\n    weight: 1", 1)
			}
			gw := New(Options{Model: "test-model", PodLabels: roleLabels, Manifests: []byte(manifests),
				EPPFlags: append(DefaultEPPFlags(), "--allow-experimental-plugins", "--config-text="+config)})
			cs := newCluster(dep, cluster.NewSliceRequestSource(testRequests(6, 11, 50_000, 0)), nil)
			cs.SetGateway(gw)
			if err := cs.Run(); err != nil {
				t.Fatalf("topology-aware run: %v", err)
			}
			if len(cs.ParentRequests()) != 6 {
				t.Fatalf("disaggregated %d of 6 requests", len(cs.ParentRequests()))
			}
			for _, req := range cs.ParentRequests() {
				if req.DecodeInstanceID != tc.matchingID {
					t.Errorf("request %s chose %s, want %s", req.ID, req.DecodeInstanceID, tc.matchingID)
				}
			}
			for id, ep := range gw.endpoints {
				want := map[cluster.InstanceID]struct{ zone, region string }{
					"instance_0": {"zone-a", "region-a"},
					"instance_1": {tc.decode1Zone, tc.decode1Reg},
					"instance_2": {tc.decode2Zone, tc.decode2Reg},
				}[id]
				if ep.ep.Labels["topology.kubernetes.io/zone"] != want.zone || ep.ep.Labels["topology.kubernetes.io/region"] != want.region {
					t.Errorf("pod %s geography = %v, want %+v", id, ep.ep.Labels, want)
				}
			}
		})
	}
}

func declaredNode(name, zone, region string) string {
	return fmt.Sprintf("apiVersion: v1\nkind: Node\nmetadata:\n  name: %s\n  labels:\n    topology.kubernetes.io/zone: %s\n    topology.kubernetes.io/region: %s\n---\n", name, zone, region)
}

func TestGateway_DeclaredGeographyMustMatchKernelRack(t *testing.T) {
	dep := testDeployment(2)
	dep.ScenarioEndpointTopology = cluster.StaticEndpointTopology(map[cluster.InstanceID]cluster.EndpointTopology{
		"instance_0": {Hostname: "blis-node-0", Rack: "blis-rack-0", NodeCount: 1},
		"instance_1": {Hostname: "blis-node-1", Rack: "blis-rack-0", NodeCount: 1},
	})
	gw := New(Options{Model: "test-model",
		Manifests: []byte(declaredNode("blis-node-0", "zone-a", "region-a") + declaredNode("blis-node-1", "zone-b", "region-a")),
		EPPFlags:  append(DefaultEPPFlags(), "--allow-experimental-plugins", "--config-text="+topologyExtractorOnly)})
	cs := newCluster(dep, cluster.NewSliceRequestSource(testRequests(1, 11, 0, 0)), nil)
	cs.SetGateway(gw)
	if err := cs.Run(); err == nil || !strings.Contains(err.Error(), "rack \"blis-rack-0\" spans zones") {
		t.Fatalf("rack/zone mismatch error = %v", err)
	}
}

// Node manifests were previously inert without topology plugins. They must not
// make an ordinary router run fail merely because the Nodes lack geography.
func TestGateway_OrdinaryNodeManifestWithoutTopology(t *testing.T) {
	dep := testDeployment(1)
	dep.ScenarioEndpointTopology = cluster.StaticEndpointTopology(map[cluster.InstanceID]cluster.EndpointTopology{
		"instance_0": {Hostname: "blis-node-0", NodeCount: 1},
	})
	gw := New(Options{Model: "test-model", Manifests: []byte("apiVersion: v1\nkind: Node\nmetadata:\n  name: blis-node-0\n")})
	cs := newCluster(dep, cluster.NewSliceRequestSource(testRequests(1, 11, 0, 0)), nil)
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("ordinary router with unlabeled Node: %v", err)
	}
}

func TestGateway_OrdinaryPodGeographyWithoutTopology(t *testing.T) {
	dep := testDeployment(1)
	gw := New(Options{Model: "test-model", PodLabels: func(cluster.InstanceID, cluster.PoolRole) map[string]string {
		return map[string]string{"topology.kubernetes.io/zone": "zone-a"}
	}})
	cs := newCluster(dep, cluster.NewSliceRequestSource(testRequests(1, 11, 0, 0)), nil)
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("ordinary router with partial Pod geography: %v", err)
	}
}

func TestGateway_NodeManifestDoesNotReachObjectiveReconciler(t *testing.T) {
	dep := testDeployment(1)
	gw := New(Options{Model: "test-model", EPPFlags: []string{"--pool-name=vllm"},
		Manifests: []byte(poolManifests + "---\napiVersion: v1\nkind: Node\nmetadata:\n  name: blis-node-0\n"),
		PodLabels: func(cluster.InstanceID, cluster.PoolRole) map[string]string { return map[string]string{"app": "vllm"} },
	})
	cs := newCluster(dep, cluster.NewSliceRequestSource(testRequests(1, 11, 0, 0)), nil)
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("pool-mode router with simulated Node: %v", err)
	}
}
