package cluster

import "github.com/inference-sim/inference-sim/sim"

// recordUnserved reports a request the cluster ended without serving to the
// session driver (onRequestDone), so closed-loop and agentic drivers learn the
// request ended and can continue. The dsv41/agentx branch defines the same hook
// (recordUnserved, with an explicit unserved-request callback and measured-window
// accounting); when the branches merge, its definition replaces this file and the
// EPP rejection sites keep calling it unchanged.
func (cs *ClusterSimulator) recordUnserved(req *sim.Request) {
	if cs.sessionCallback == nil || req == nil {
		return
	}
	for _, next := range cs.sessionCallback(req, cs.clock) {
		cs.pushArrival(next, next.ArrivalTime)
	}
}
