package cluster

import "testing"

func TestStaticEndpointTopologySnapshotsPlacement(t *testing.T) {
	entries := map[InstanceID]EndpointTopology{"instance_0": {Hostname: "node-0", NodeCount: 1}}
	lookup := StaticEndpointTopology(entries)
	entries["instance_0"] = EndpointTopology{Hostname: "wrong-node", NodeCount: 1}
	if got := lookup("instance_0").Hostname; got != "node-0" {
		t.Errorf("placement changed after construction: %q", got)
	}
}
