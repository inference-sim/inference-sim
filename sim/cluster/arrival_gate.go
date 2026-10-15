package cluster

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// ArrivalGate decides, when a pushed arrival comes due, whether the client actually sends
// it. It is the client-side half of a closed-loop workload whose send times are not fixed
// when a follow-up is pushed: a timer the client may later pull in (aiperf's global idle
// cap), cancel (the end of the sending window) or hold behind a dependency (aiperf's
// replay barriers). The driver re-pushes a request through the follow-up path
// (onRequestDone's return value) whenever it moves a send; the gate refuses the arrivals
// it superseded. A refused arrival is not a request: it never reaches admission, is not
// counted as injected, and fires no arrival hook, so INV-1 is unaffected.
//
// more are further requests the client sends at the same instant (e.g. children a parent's
// send spawns), routed like follow-ups. Called on the cluster's single Run goroutine.
type ArrivalGate func(req *sim.Request, now int64) (send bool, more []*sim.Request)

// SetArrivalGate installs the arrival gate. Must be called before Run(); nil clears it.
func (cs *ClusterSimulator) SetArrivalGate(g ArrivalGate) {
	if cs.hasRun {
		panic("ClusterSimulator: SetArrivalGate must be called before Run()")
	}
	cs.arrivalGate = g
}

// SetHorizon moves the simulation horizon during Run, for a workload whose end is known
// only once it is running (an agentic replay's profiling window starts when its warm-up
// drains). Events after the new horizon do not execute; requests still in flight at it are
// left unfinished, as at any horizon. It may not move the horizon before the current clock.
func (cs *ClusterSimulator) SetHorizon(h int64) {
	if h < cs.clock {
		panic(fmt.Sprintf("ClusterSimulator.SetHorizon(%d) before the clock %d", h, cs.clock))
	}
	cs.config.Horizon = h
}

// gateArrival applies the arrival gate, pushing any requests it adds at now. It reports
// whether the arrival proceeds.
func (cs *ClusterSimulator) gateArrival(req *sim.Request, now int64) bool {
	if cs.arrivalGate == nil {
		return true
	}
	send, more := cs.arrivalGate(req, now)
	for _, m := range more {
		cs.pushArrival(m, m.ArrivalTime)
	}
	return send
}

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

// Horizon returns the simulation horizon in effect.
func (cs *ClusterSimulator) Horizon() int64 { return cs.config.Horizon }

func (cs *ClusterSimulator) notifyUnserved(req *sim.Request) {
	if cs.unservedHook == nil || req == nil {
		return
	}
	for _, m := range cs.unservedHook(req, cs.clock) {
		cs.pushArrival(m, m.ArrivalTime)
	}
}
