package cmd

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// A colocated run's instances are told where each peer sits, so a p2p tier's WRITE between two of
// them ends on the peer's own node (inference-sim/inference-sim#2439). Before, a colocated run gave no peer endpoints
// at all and every WRITE between its instances was refused. The fixture's prefill pool, read as one
// colocated pool, holds four one-tray groups: instance n is group n, on tray n.
func TestPeerEndpoint_AColocatedRunPlacesEachInstanceOnItsGroupsNode(t *testing.T) {
	dir, _ := dpPoolScenarios(t, 4, 4)
	pools, _ := openDPPools(t, dir, 2, 2)
	m := pools.prefill
	at := peerEndpoint(nil, m)
	if at == nil {
		t.Fatal("a colocated run gives its instances no peer endpoints")
	}
	for n := 0; n < m.GroupCapacity(); n++ {
		id := cluster.InstanceID(fmt.Sprintf("instance_%d", n))
		g := m.GroupPlacementOf(n)
		if got := at(id); got.Node != g.Node-poolOrigin(m).Node || got.Rack != g.Rack-poolOrigin(m).Rack {
			t.Errorf("%s is placed at %+v, want its group's node %d, rack %d, in the pool's numbering", id, got, g.Node, g.Rack)
		}
	}
	if at("instance_0") == at("instance_1") {
		t.Error("two instances on different trays are given one endpoint, so a WRITE between them would cross no link")
	}
}

// The peer endpoints are the places the router is told and the engines use: for a colocated run,
// the hostname of each instance's tray in the run's endpoint topology.
func TestPeerEndpoint_IsTheColocatedRunsEndpointTopology(t *testing.T) {
	dir, _ := dpPoolScenarios(t, 4, 4)
	pools, _ := openDPPools(t, dir, 2, 2)
	m := pools.prefill
	topo, err := scenarioEndpointTopology(m, nil, m.GroupCapacity(), 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	at := peerEndpoint(nil, m)
	for n := 0; n < m.GroupCapacity(); n++ {
		id := cluster.InstanceID(fmt.Sprintf("instance_%d", n))
		want := fmt.Sprintf("blis-node-%d", poolOrigin(m).Node+at(id).Node)
		if topo[id].Hostname != want {
			t.Errorf("%s: the router is told %q, its peers are told node %d (%s)", id, topo[id].Hostname, at(id).Node, want)
		}
	}
}

// A disaggregated run's peer endpoints are its instance endpoints, unchanged.
func TestPeerEndpoint_ADisaggregatedRunKeepsItsInstanceEndpoints(t *testing.T) {
	dir, _ := dpPoolScenarios(t, 4, 4)
	pools, _ := openDPPools(t, dir, 2, 2)
	want, got := pools.instanceEndpoint(), peerEndpoint(pools, nil)
	for n := 0; n < 4; n++ {
		id := cluster.InstanceID(fmt.Sprintf("instance_%d", n))
		if got(id) != want(id) {
			t.Errorf("%s: peer endpoint %+v, instance endpoint %+v", id, got(id), want(id))
		}
	}
	if peerEndpoint(nil, nil) != nil {
		t.Error("a run with no kernel open is given peer endpoints")
	}
}
