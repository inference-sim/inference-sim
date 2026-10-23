package llmd

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/cluster"
	corev1 "k8s.io/api/core/v1"
)

func TestNodeManifestsProvideOnlyDeclaredGeography(t *testing.T) {
	const docs = `apiVersion: v1
kind: Node
metadata:
  name: blis-node-0
  labels:
    topology.kubernetes.io/zone: zone-a
    topology.kubernetes.io/region: region-1
---
apiVersion: v1
kind: Node
metadata:
  name: blis-node-1
  labels:
    topology.kubernetes.io/zone: zone-b
    topology.kubernetes.io/region: region-1
`
	res, err := parseManifests(newScheme(), []byte(docs), "default")
	if err != nil {
		t.Fatal(err)
	}
	locations, err := nodeLocations(res.nodes)
	if err != nil {
		t.Fatal(err)
	}
	if len(locations) != 2 || res.nodes[0].Namespace != "" {
		t.Fatalf("simulated Nodes: locations=%v namespace=%q", locations, res.nodes[0].Namespace)
	}
	base := cluster.EndpointTopology{Hostname: "blis-node-0", Rack: "blis-rack-0", NodeCount: 1}
	got, err := overlayNodeLocation(base, locations)
	if err != nil {
		t.Fatal(err)
	}
	if got.Rack != base.Rack || got.Zone != "zone-a" || got.Region != "region-1" {
		t.Fatalf("resolved placement = %+v", got)
	}
	if _, err := overlayNodeLocation(cluster.EndpointTopology{Hostname: "blis-node-2", NodeCount: 1}, locations); err == nil || !strings.Contains(err.Error(), "no matching simulated Node") {
		t.Fatalf("missing Node manifest error = %v", err)
	}
}

func TestNodeManifestHierarchyAndConflictRefusals(t *testing.T) {
	for _, tc := range []struct {
		name, want string
		nodes      []*corev1.Node
	}{
		{"duplicate Node", "duplicate simulated Node", []*corev1.Node{{}, {}}},
		{"missing region", "must declare both", []*corev1.Node{{}}},
		{"one zone in two regions", "spans regions", []*corev1.Node{{}, {}}},
		{"one rack in two zones", "spans zones", []*corev1.Node{{}, {}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for n, node := range tc.nodes {
				node.Name = "blis-node-" + string(rune('0'+n))
				node.Labels = map[string]string{corev1.LabelTopologyZone: "zone-a", corev1.LabelTopologyRegion: "region-1"}
			}
			switch tc.name {
			case "duplicate Node":
				tc.nodes[1].Name = tc.nodes[0].Name
			case "missing region":
				delete(tc.nodes[0].Labels, corev1.LabelTopologyRegion)
			case "one zone in two regions":
				tc.nodes[1].Labels[corev1.LabelTopologyRegion] = "region-2"
			case "one rack in two zones":
				tc.nodes[0].Labels[rackLabel] = "rack-a"
				tc.nodes[1].Labels[rackLabel] = "rack-a"
				tc.nodes[1].Labels[corev1.LabelTopologyZone] = "zone-b"
			}
			if _, err := nodeLocations(tc.nodes); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("nodeLocations error = %v, want %q", err, tc.want)
			}
		})
	}
	locs := map[string]cluster.EndpointTopology{"blis-node-0": {Hostname: "blis-node-0", Rack: "rack-b", Zone: "zone-a", Region: "region-1"}}
	_, err := overlayNodeLocation(cluster.EndpointTopology{Hostname: "blis-node-0", Rack: "rack-a", NodeCount: 1}, locs)
	if err == nil || !strings.Contains(err.Error(), "kernel rack") {
		t.Fatalf("conflicting rack error = %v", err)
	}
	_, err = overlayNodeLocation(cluster.EndpointTopology{Hostname: "blis-node-0", NodeCount: 1}, locs)
	if err == nil || !strings.Contains(err.Error(), "declares no rack placement") {
		t.Fatalf("unpriced rack error = %v", err)
	}
}
