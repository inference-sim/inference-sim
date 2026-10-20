package kernelmodel

import "testing"

// A rank may cross a node boundary even when it uses fewer GPUs than one node
// holds. PlacementOf names its first node for pricing; one Pod must not claim
// that node as the rank's sole host.
func TestEndpointPlacementDetectsCrossingRank(t *testing.T) {
	m := &Model{id: identity{gpusPerNode: 4, rankGPUs: 3}}
	first, nodes, declaredRack := m.EndpointPlacement(0)
	if first.Node != 0 || nodes != 1 || declaredRack {
		t.Errorf("first rank: placement=%+v nodes=%d declaredRack=%v", first, nodes, declaredRack)
	}
	second, nodes, declaredRack := m.EndpointPlacement(1)
	if second.Node != 0 || nodes != 2 || declaredRack {
		t.Errorf("crossing rank: placement=%+v nodes=%d declaredRack=%v", second, nodes, declaredRack)
	}
}
