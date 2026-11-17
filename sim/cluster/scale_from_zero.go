package cluster

import (
	"sort"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
)

// Scale from zero (inference-sim#908): a request for a model that has no replica serving or
// starting starts one, and is held at the cluster until a replica is routable, so it is served
// after the cold start instead of being rejected. It is opt-in: ScaleFromZeroHoldUs above 0 is how
// long a request may be held; 0 changes nothing. The replica is the variant the model last ran
// (a model that never had a replica has no variant to start, and its request is not held), placed
// through the actuator, so a full pool queues the placement for the cluster autoscaler (#741, #740).

const defaultColdStartPollUs = 100_000

// holdForColdStart reports whether the arrival e was held, for retry after the poll interval. It
// is called as an arrival executes, before the request reaches the gateway.
func (cs *ClusterSimulator) holdForColdStart(e *ClusterArrivalEvent) bool {
	hold := cs.config.ScaleFromZeroHoldUs
	if hold <= 0 || cs.placement == nil {
		return false
	}
	req := e.request
	model := req.Model
	if model == "" {
		model = cs.config.Model
	}
	var last *InstanceSimulator
	starting := false
	for _, inst := range cs.instances {
		if inst.Model != model {
			continue
		}
		if inst.IsRoutable() && inst.State != sim.InstanceStateDraining {
			delete(cs.heldSince, req.ID)
			delete(cs.coldStarted, model)
			return false // a replica is serving
		}
		if inst.State == sim.InstanceStateLoading {
			starting = true
		}
		if last == nil || string(inst.ID()) > string(last.ID()) {
			last = inst
		}
	}
	if last == nil {
		return false // the model never had a replica: no variant to start
	}
	if !starting && !cs.coldStarted[model] {
		if cs.coldStarted == nil {
			cs.coldStarted = map[string]bool{}
		}
		cs.coldStarted[model] = true
		v := NewVariantSpec(last.GPU(), max(last.TPDegree, 1))
		logrus.Infof("[scale-from-zero] request %s for model %q finds no replica: starting one (%v)", req.ID, model, v)
		if err := NewDirectActuator(cs).Apply([]ScaleDecision{{ModelID: model, Variant: v, Delta: 1}}); err != nil {
			logrus.Warnf("[scale-from-zero] starting a replica for model %q: %v", model, err)
		}
	}
	cs.noteDemand(req) // a held request is demand: the replica it starts is not drained before it lands
	if cs.heldSince == nil {
		cs.heldSince = map[string]int64{}
	}
	since, held := cs.heldSince[req.ID]
	if !held {
		since = e.time
		cs.heldSince[req.ID] = since
	}
	if float64(e.time-since) >= hold {
		delete(cs.heldSince, req.ID)
		return false // held long enough: it goes to the gateway, which rejects it if still nothing serves
	}
	poll := int64(cs.config.ScaleFromZeroPollUs)
	if poll <= 0 {
		poll = defaultColdStartPollUs
	}
	cs.schedule(&ClusterArrivalEvent{time: e.time + poll, request: req}, &cs.workloadSender)
	cs.pendingArrivals++
	return true
}

// heldRequests are the IDs of the requests currently held for a cold start, sorted.
func (cs *ClusterSimulator) heldRequests() []string {
	out := make([]string, 0, len(cs.heldSince))
	for id := range cs.heldSince {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}
