// direct_actuator.go implements DirectActuator — applies scale decisions directly
// to the cluster by calling PlacementManager for scale-up and transitioning instances
// to Draining for scale-down (WaitDrain semantics).
package cluster

import (
	"errors"
	"fmt"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
)

// DirectActuator applies scale decisions to the cluster. Scale-up calls
// PlacementManager.PlaceInstance(). Scale-down transitions the oldest active
// instance for the target model+variant to Draining. Must not block.
type DirectActuator struct {
	cluster     *ClusterSimulator
	nextInstSeq int // monotonic counter for generating unique instance IDs
	// drainPolicy is how scale-down drains an instance (InstanceLifecycleConfig.DrainPolicy),
	// resolved once at construction; WAIT when the config states none (#910).
	drainPolicy DrainPolicy
}

// NewDirectActuator constructs a DirectActuator. Panics if cluster is nil.
func NewDirectActuator(cluster *ClusterSimulator) *DirectActuator {
	if cluster == nil {
		panic("NewDirectActuator: cluster must not be nil")
	}
	name := DrainPolicyName(cluster.config.InstanceLifecycle.DrainPolicy)
	if name == "" {
		name = DrainPolicyWait
	}
	return &DirectActuator{cluster: cluster, drainPolicy: NewDrainPolicy(name)}
}

// Apply processes scale decisions. For Delta > 0, places new instances.
// For Delta < 0, drains existing instances. Returns an error if any sub-operation
// fails; partial success is possible (some decisions applied, others failed).
// Individual failures are always logged (R1, INV-A2).
func (a *DirectActuator) Apply(decisions []ScaleDecision) error {
	var errs []error
	for _, d := range decisions {
		if d.Delta > 0 {
			if err := a.scaleUp(d); err != nil {
				errs = append(errs, err)
			}
		} else if d.Delta < 0 {
			if err := a.scaleDown(d); err != nil {
				errs = append(errs, err)
			}
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("actuator: %d operation(s) failed; first: %w", len(errs), errs[0])
	}
	return nil
}

// scaleUp places new instance(s) for the given decision. Returns error if any placement fails.
func (a *DirectActuator) scaleUp(d ScaleDecision) error {
	if a.cluster.placement == nil {
		err := fmt.Errorf("scale-up for model %q: PlacementManager not available (INV-A2)", d.ModelID)
		logrus.Errorf("[actuator] %v", err)
		return err
	}

	var lastErr error
	for i := 0; i < d.Delta; i++ {
		// The sequence number is committed only once the placement succeeds, so a refused
		// placement leaves no gap in the autoscale-<model>-NNNNNN IDs.
		id := InstanceID(fmt.Sprintf("autoscale-%s-%06d", d.ModelID, a.nextInstSeq+1))

		// Reserve GPU slots. GPUType comes from the scale decision (variant-authoritative).
		nodeID, gpuIDs, matchedGPU, err := a.cluster.placement.PlaceInstance(
			id, d.ModelID, d.Variant.GPUType, d.Variant.TPDegree,
		)
		if err != nil {
			if errors.Is(err, ErrNoCapacity) {
				// No Ready node has room: the replica waits in the pending queue for a node (#741),
				// and is placed, in arrival order, when one becomes Ready. It is not dropped.
				if a.queuePending(id, d, err) {
					continue
				}
				// The model already has as many placements waiting as this decision asks for, so a
				// repeat of the same decision does not queue the same replica again.
				a.nextInstSeq--
				continue
			}
			logrus.Errorf("[actuator] scale-up for model %q variant %v: PlaceInstance failed: %v (INV-A2)", d.ModelID, d.Variant, err)
			lastErr = err
			continue
		}
		a.nextInstSeq++

		// Build simCfg: start from the cluster default, then apply the pool-authoritative GPU
		// type (SC-004, mirrors NewClusterSimulator startup path).
		// PoolRole(0) returns the global SimConfig unchanged — autoscaler does not yet
		// support PD disaggregation. If PD + autoscaler are combined in future, this
		// will need to resolve the correct prefill/decode role for the pool.
		simCfg := a.cluster.config.resolveConfigForRole(PoolRole(0))
		simCfg.GPU = matchedGPU

		// Record the instance's node span (mirrors the startup + deferred paths).
		a.cluster.recordNodeSpan(gpuIDs)

		// #1529: cost = distinct-nodes-spanned × pool cost_per_hour (mirrors startup +
		// deferred paths). Single-node instances are unchanged (1×).
		instCost := a.cluster.placement.PlacedInstanceCostPerHour(gpuIDs)

		// Construct, register, and activate the instance; the gateway sees its pod when it
		// becomes ready.
		a.cluster.addLiveInstance(id, d.ModelID, simCfg, nodeID, gpuIDs, d.Variant.TPDegree, instCost)

		logrus.Infof("[actuator] scale-up: placed instance %s for model %q on node %s (gpu=%s, tp=%d)",
			id, d.ModelID, nodeID, matchedGPU, d.Variant.TPDegree)
	}
	return lastErr
}

// queuePending queues one replica of d in the placement manager's pending queue, unless the queue
// already holds as many placements for the same model and variant as d's remaining demand asks
// for. It reports whether it queued one.
func (a *DirectActuator) queuePending(id InstanceID, d ScaleDecision, cause error) bool {
	waiting := 0
	for _, p := range a.cluster.placement.Pending() {
		if p.ModelID == d.ModelID && p.Variant == d.Variant {
			waiting++
		}
	}
	if waiting >= d.Delta {
		logrus.Debugf("[actuator] scale-up for model %q variant %v: %d placement(s) already pending", d.ModelID, d.Variant, waiting)
		return false
	}
	simCfg := a.cluster.config.resolveConfigForRole(PoolRole(0))
	a.cluster.placement.AddPendingAt(id, d.ModelID, d.Variant.GPUType, d.Variant.TPDegree, simCfg, a.cluster.clock)
	logrus.Infof("[actuator] scale-up for model %q variant %v: %v; queued as pending placement %s", d.ModelID, d.Variant, cause, id)
	return true
}

// scaleDown drains existing instance(s) for the given decision. Returns error if
// any iteration finds no active instance to drain.
// TODO(1C-4b): cancel pending placements for the same model before draining (specs/010).
//
// The decision was computed from the cluster as it stood at its tick; with an HPA scrape delay
// it is applied later, after other decisions computed from the same view may have drained
// replicas already. So the actuator applies the minimum again, on the replicas still routable
// now: it drains none that would leave the model below the analyzer's MinReplicas, and logs
// each removal it declines (#2015).
func (a *DirectActuator) scaleDown(d ScaleDecision) error {
	minReplicas := effectiveAnalyzerConfig(a.cluster.config.AutoscalerAnalyzerConfig).MinReplicas
	var lastErr error
	for i := 0; i < -d.Delta; i++ {
		if n := a.routableReplicas(d.ModelID); n <= minReplicas {
			logrus.Warnf("[actuator] scale-down for model %q: %d of %d removals not applied; %d routable replica(s) "+
				"remain and the minimum is %d (the decision predates earlier removals)", d.ModelID, -d.Delta-i, -d.Delta, n, minReplicas)
			break
		}
		inst := a.findOldestActive(d.ModelID, d.Variant)
		if inst == nil {
			err := fmt.Errorf("scale-down for model %q variant %v: no active instance found", d.ModelID, d.Variant)
			logrus.Warnf("[actuator] %v", err)
			lastErr = err
			continue // keep trying remaining iterations (next iteration may find different variant match)
		}
		a.drainPolicy.Drain(inst, a.cluster)
		logrus.Infof("[actuator] scale-down: draining instance %s for model %q (%T)", inst.ID(), d.ModelID, a.drainPolicy)
	}
	return lastErr
}

// routableReplicas counts the model's instances the router can send requests to now.
func (a *DirectActuator) routableReplicas(model string) int {
	n := 0
	for _, inst := range a.cluster.instances {
		if inst.Model == model && inst.IsRoutable() {
			n++
		}
	}
	return n
}

// findOldestActive returns the first (oldest) active instance for the given model+variant.
// Candidates are sorted by sequence number embedded in the ID for determinism (R2).
func (a *DirectActuator) findOldestActive(model string, variant VariantSpec) *InstanceSimulator {
	var candidates []*InstanceSimulator
	for _, inst := range a.cluster.instances {
		if inst.Model != model {
			continue
		}
		if inst.State != sim.InstanceStateActive {
			continue
		}
		gpuType := inst.GPU()
		tp := inst.TPDegree
		if tp < 1 {
			tp = 1
		}
		if gpuType == variant.GPUType && tp == variant.TPDegree {
			candidates = append(candidates, inst)
		}
	}

	if len(candidates) == 0 {
		return nil
	}

	// Sort by ID for determinism (R2). Zero-padded sequence numbers (%06d) ensure
	// lexicographic order matches creation order.
	sort.Slice(candidates, func(i, j int) bool {
		return string(candidates[i].ID()) < string(candidates[j].ID())
	})
	return candidates[0]
}
