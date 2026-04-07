// direct_actuator.go implements DirectActuator — applies scale decisions directly
// to the cluster by calling PlacementManager for scale-up and transitioning instances
// to Draining for scale-down (WaitDrain semantics).
package cluster

import (
	"fmt"
	"sort"

	"github.com/sirupsen/logrus"
)

// DirectActuator applies scale decisions to the cluster. Scale-up calls
// PlacementManager.PlaceInstance(). Scale-down transitions the oldest active
// instance for the target model+variant to Draining. Must not block.
type DirectActuator struct {
	cluster     *ClusterSimulator
	nextInstSeq int // monotonic counter for generating unique instance IDs
}

// NewDirectActuator constructs a DirectActuator. Panics if cluster is nil.
func NewDirectActuator(cluster *ClusterSimulator) *DirectActuator {
	if cluster == nil {
		panic("NewDirectActuator: cluster must not be nil")
	}
	return &DirectActuator{cluster: cluster}
}

// Apply processes scale decisions. For Delta > 0, places new instances.
// For Delta < 0, drains existing instances. Errors are logged, not silently dropped (INV-A2).
func (a *DirectActuator) Apply(decisions []ScaleDecision) error {
	for _, d := range decisions {
		if d.Delta > 0 {
			a.scaleUp(d)
		} else if d.Delta < 0 {
			a.scaleDown(d)
		}
	}
	return nil
}

// scaleUp places new instance(s) for the given decision.
func (a *DirectActuator) scaleUp(d ScaleDecision) {
	for i := 0; i < d.Delta; i++ {
		a.nextInstSeq++
		id := InstanceID(fmt.Sprintf("autoscale-%s-%d", d.ModelID, a.nextInstSeq))

		if a.cluster.placement == nil {
			logrus.Errorf("[actuator] scale-up for model %q: PlacementManager not available (INV-A2)", d.ModelID)
			continue
		}

		nodeID, gpuIDs, matchedGPU, err := a.cluster.placement.PlaceInstance(
			id, d.ModelID, d.Variant.GPUType, d.Variant.TPDegree,
		)
		if err != nil {
			logrus.Errorf("[actuator] scale-up for model %q variant %v failed: %v (INV-A2)", d.ModelID, d.Variant, err)
			continue
		}
		logrus.Infof("[actuator] scale-up: placed instance %s for model %q on node %s (gpus=%v, matchedGPU=%s)",
			id, d.ModelID, nodeID, gpuIDs, matchedGPU)
	}
}

// scaleDown drains existing instance(s) for the given decision.
func (a *DirectActuator) scaleDown(d ScaleDecision) {
	for i := 0; i < -d.Delta; i++ {
		inst := a.findOldestActive(d.ModelID, d.Variant)
		if inst == nil {
			logrus.Warnf("[actuator] scale-down for model %q variant %v: no active instance found", d.ModelID, d.Variant)
			return
		}
		inst.TransitionTo(InstanceStateDraining)
		logrus.Infof("[actuator] scale-down: draining instance %s for model %q", inst.ID(), d.ModelID)
	}
}

// findOldestActive returns the first (oldest) active instance for the given model+variant.
// Instances are sorted by ID for determinism (R2).
func (a *DirectActuator) findOldestActive(model string, variant VariantSpec) *InstanceSimulator {
	// Collect candidates
	var candidates []*InstanceSimulator
	for _, inst := range a.cluster.instances {
		if inst.Model != model {
			continue
		}
		if inst.State != InstanceStateActive {
			continue
		}
		// Match variant
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

	// Sort by ID for determinism (R2) — oldest first
	sort.Slice(candidates, func(i, j int) bool {
		return string(candidates[i].ID()) < string(candidates[j].ID())
	})
	return candidates[0]
}
