// Peer endpoints: where each instance sits, for the engines that send it KV (blis-vllm#204).

package cmd

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// peerEndpoint is where each instance's workers sit, in the numbering of its own pool, for an
// engine that sends it KV directly (engine.PeerPlaced; a p2p tier's WRITE ends there). A
// disaggregated run's instances are where instanceEndpoint puts them. A colocated run's instance n
// is the pool's n-th group, as the run's endpoint topology places it (build.EndpointGroupPlacements):
// the node holding the group's first GPU, counted from the pool's lowest node and rack
// (poolOrigin). Two instances on one node get one endpoint, and a WRITE between them is a move
// within the node. Nil when no kernel is open.
func peerEndpoint(pd *kernelPools, colocated *kernelmodel.Model) func(cluster.InstanceID) plan.Endpoint {
	if pd != nil {
		return pd.instanceEndpoint()
	}
	if colocated == nil {
		return nil
	}
	origin := poolOrigin(colocated)
	return func(id cluster.InstanceID) plan.Endpoint {
		n, err := strconv.Atoi(strings.TrimPrefix(string(id), "instance_"))
		if err != nil || n < 0 || n >= colocated.GroupCapacity() {
			panic(fmt.Sprintf("peer endpoint: instance %q is not one of the pool's %d groups", id, colocated.GroupCapacity()))
		}
		at := colocated.GroupPlacementOf(n)
		local, err := origin.Local(plan.Endpoint{Node: at.Node, Rack: at.Rack})
		if err != nil {
			panic(fmt.Sprintf("peer endpoint: instance %q: %v", id, err))
		}
		return local
	}
}

// placePeerInstance states where a colocated run's instance sits when its engine exchanges KV with
// peers (engine.Peered with an address): such an engine states its WRITEs from its own endpoint, so
// left as built every instance would state the pool's first node. Other colocated engines are left as
// built (placeInstance), and a disaggregated run's are placed there.
func placePeerInstance(eng engine.Engine, colocated *kernelmodel.Model, index int) error {
	p, ok := eng.(engine.Peered)
	if !ok || p.PeerAddr() == "" || colocated == nil {
		return nil
	}
	at := peerEndpoint(nil, colocated)(cluster.InstanceID(fmt.Sprintf("instance_%d", index)))
	if err := engine.Place(eng, at, colocated.EnginePlacement().Nodes); err != nil {
		return fmt.Errorf("instance_%d: %w", index, err)
	}
	return nil
}
