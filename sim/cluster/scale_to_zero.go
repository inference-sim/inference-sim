package cluster

import (
	"math"
	"sort"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
)

// Scale to zero (inference-sim#909): when a model has had no request for ScaleToZeroIdleUs of
// simulated time and has nothing in flight, the autoscaler drains all of its replicas, and the
// nodes they leave empty are flagged as reclaimable. It is opt-in; 0 never drains. How the replicas
// are drained (what happens to the work they hold) is the configured drain policy (#910); this only
// decides when. Bringing a model back from zero on a request is scale-from-zero (#908).

// noteDemand records that a request for the model arrived now.
func (cs *ClusterSimulator) noteDemand(req *sim.Request) {
	if cs.config.ScaleToZeroIdleUs <= 0 || req.Redirected {
		return
	}
	model := req.Model
	if model == "" {
		model = cs.config.Model
	}
	if cs.lastDemandAt == nil {
		cs.lastDemandAt = map[string]int64{}
	}
	cs.lastDemandAt[model] = cs.clock
}

// servedModels are the models with at least one replica serving or starting, sorted (R2).
func (cs *ClusterSimulator) servedModels() []string {
	seen := map[string]bool{}
	for _, inst := range cs.instances {
		switch inst.State {
		case sim.InstanceStateActive, sim.InstanceStateWarmingUp, sim.InstanceStateLoading:
			seen[inst.Model] = true
		}
	}
	models := make([]string, 0, len(seen))
	for m := range seen {
		models = append(models, m)
	}
	sort.Strings(models)
	return models
}

// scaleToZeroPending reports whether some model could still be drained, so the scaling tick keeps
// running past the end of the workload to do it.
func (cs *ClusterSimulator) scaleToZeroPending() bool {
	return cs.config.ScaleToZeroIdleUs > 0 && len(cs.servedModels()) > 0
}

// idleForScaleToZero reports whether model has been without a request for the idle period and has
// no request in flight on any replica.
func (cs *ClusterSimulator) idleForScaleToZero(model string) bool {
	if float64(cs.clock-cs.lastDemandAt[model]) <= cs.config.ScaleToZeroIdleUs {
		return false
	}
	for _, inst := range cs.instances {
		if inst.Model != model || inst.State == sim.InstanceStateTerminated {
			continue
		}
		if cs.inFlightRequests[string(inst.ID())] > 0 || inst.QueueDepth() > 0 || inst.BatchSize() > 0 {
			return false
		}
	}
	return true
}

// checkScaleToZero drains every replica of each idle model, and flags the nodes that are left empty.
func (cs *ClusterSimulator) checkScaleToZero() {
	if cs.config.ScaleToZeroIdleUs <= 0 || math.IsNaN(cs.config.ScaleToZeroIdleUs) {
		return
	}
	actuator := NewDirectActuator(cs)
	for _, model := range cs.servedModels() {
		if !cs.idleForScaleToZero(model) {
			continue
		}
		n := 0
		for _, inst := range cs.instances {
			if inst.Model == model && (inst.State == sim.InstanceStateActive || inst.State == sim.InstanceStateWarmingUp) {
				actuator.drainPolicy.Drain(inst, cs)
				n++
			}
		}
		if n > 0 {
			logrus.Infof("[scale-to-zero] model %q idle for more than %gμs: draining %d replica(s)", model, cs.config.ScaleToZeroIdleUs, n)
		}
	}
}

// IdleNodes are the Ready nodes with no instance on them, sorted: the nodes a scale-to-zero (or
// any scale-down) left empty, which the cluster autoscaler's scale-down may reclaim.
func (cs *ClusterSimulator) IdleNodes() []string {
	if cs.placement == nil {
		return nil
	}
	var out []string
	for _, nodes := range cs.placement.nodesByPool {
		for _, n := range nodes {
			if n.State == NodeStateReady && n.freeCount() == n.TotalGPUs {
				out = append(out, n.ID)
			}
		}
	}
	sort.Strings(out)
	return out
}
