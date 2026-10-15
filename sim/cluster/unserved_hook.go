package cluster

import "github.com/inference-sim/inference-sim/sim"

// The unserved-request hook, defined identically to the dsv41/agentx branch
// (sim/cluster/arrival_gate.go, commits fd4d225c/0a8287ea) so the branches compose:
// on merge keep agentx's definitions (its recordUnserved also counts the measured
// window's failures) and delete this file. EPP routing rejections call
// recordUnserved exactly like agentx's other refusal sites.

// SetUnservedHook installs a callback told of every request the cluster itself ends
// without serving -- admission and routing rejections, flow-control rejections, sheds,
// TTL expiries, in-flight evictions and P/D decode-KV drops: every site that records the
// request as unserved (recordUnserved). The engine's own refusals and every completion
// still reach onRequestDone. Follow-ups it returns are routed like onRequestDone's. Must be
// called before Run(); nil clears it.
func (cs *ClusterSimulator) SetUnservedHook(h func(req *sim.Request, now int64) []*sim.Request) {
	if cs.hasRun {
		panic("ClusterSimulator: SetUnservedHook must be called before Run()")
	}
	cs.unservedHook = h
}

func (cs *ClusterSimulator) notifyUnserved(req *sim.Request) {
	if cs.unservedHook == nil || req == nil {
		return
	}
	for _, m := range cs.unservedHook(req, cs.clock) {
		cs.pushArrival(m, m.ArrivalTime)
	}
}

// recordUnserved reports a request the cluster never served. On this branch it only
// notifies the hook; agentx's version also counts measured-window failures.
func (c *ClusterSimulator) recordUnserved(req *sim.Request) {
	c.notifyUnserved(req)
}
