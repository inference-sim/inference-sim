// direct_actuator.go implements DirectActuator — applies scale decisions directly
// to the cluster by calling PlacementManager for scale-up and transitioning instances
// to Draining for scale-down (WaitDrain semantics).
package cluster

import (
	"container/heap"
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
}

// NewDirectActuator constructs a DirectActuator. Panics if cluster is nil.
func NewDirectActuator(cluster *ClusterSimulator) *DirectActuator {
	if cluster == nil {
		panic("NewDirectActuator: cluster must not be nil")
	}
	return &DirectActuator{cluster: cluster}
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

// scaleUp places new instance(s) for the given decision and wires them into the cluster.
// Mirrors the NodeReadyEvent.Execute() creation path: place → construct → register → schedule.
func (a *DirectActuator) scaleUp(d ScaleDecision) error {
	if a.cluster.placement == nil {
		err := fmt.Errorf("scale-up for model %q: PlacementManager not available (INV-A2)", d.ModelID)
		logrus.Errorf("[actuator] %v", err)
		return err
	}

	cs := a.cluster
	tpDegree := d.Variant.TPDegree
	if tpDegree < 1 {
		tpDegree = 1
	}

	var lastErr error
	for i := 0; i < d.Delta; i++ {
		a.nextInstSeq++
		id := InstanceID(fmt.Sprintf("autoscale-%s-%06d", d.ModelID, a.nextInstSeq))

		nodeID, gpuIDs, matchedGPU, err := cs.placement.PlaceInstance(
			id, d.ModelID, d.Variant.GPUType, tpDegree,
		)
		if err != nil {
			logrus.Errorf("[actuator] scale-up for model %q variant %v failed: %v (INV-A2)", d.ModelID, d.Variant, err)
			lastErr = err
			continue
		}

		// Build SimConfig for the new instance (mirrors NodeReadyEvent.Execute).
		simCfg := cs.config.resolveConfigForRole(0)
		simCfg.GPU = matchedGPU
		if hc, ok := cs.config.HWConfigByGPU[matchedGPU]; ok {
			simCfg.HWConfig = hc
		}
		var costPerHour float64
		for i := range cs.config.NodePools {
			if cs.config.NodePools[i].GPUType == matchedGPU {
				costPerHour = cs.config.NodePools[i].CostPerHour
				break
			}
		}

		inst := NewInstanceSimulator(id, simCfg)
		inst.Model = d.ModelID
		inst.nodeID = nodeID
		inst.allocatedGPUIDs = gpuIDs
		inst.TPDegree = tpDegree
		inst.CostPerHour = costPerHour
		inst.warmUpRemaining = cs.config.InstanceLifecycle.WarmUpRequestCount
		inst.TransitionTo(InstanceStateLoading)

		// Register with snapshot provider so the instance is routable.
		csp, ok := cs.snapshotProvider.(*CachedSnapshotProvider)
		if !ok {
			logrus.Warnf("[actuator] scale-up: snapshotProvider is not *CachedSnapshotProvider for instance %s — releasing GPUs and skipping", id)
			cs.releaseInstanceGPUs(inst)
			continue
		}
		csp.AddInstance(id, inst)
		cs.scheduleInstanceLoadedEvent(inst)
		cs.instances = append(cs.instances, inst)
		cs.inFlightRequests[string(id)] = 0

		if cs.cacheQueryFn != nil {
			cs.registerInstanceCacheQueryFn(id, inst)
		}
		if cs.tenantTracker != nil || cs.sessionCallback != nil {
			onRequestDone := cs.sessionCallback
			inst.sim.OnRequestDone = func(req *sim.Request, tick int64) []*sim.Request {
				if cs.tenantTracker != nil {
					cs.tenantTracker.OnComplete(req.TenantID)
				}
				if onRequestDone == nil {
					return nil
				}
				nextReqs := onRequestDone(req, tick)
				for _, next := range nextReqs {
					heap.Push(&cs.clusterEvents, clusterEventEntry{
						event: &ClusterArrivalEvent{time: next.ArrivalTime, request: next},
						seqID: cs.nextSeqID(),
					})
				}
				return nil
			}
		}

		logrus.Infof("[actuator] scale-up: placed instance %s for model %q on node %s (gpus=%v, matchedGPU=%s)",
			id, d.ModelID, nodeID, gpuIDs, matchedGPU)
	}
	return lastErr
}

// scaleDown drains existing instance(s) for the given decision. Returns error if
// any iteration finds no active instance to drain.
// TODO(1C-4b): cancel pending placements for the same model before draining (specs/010).
func (a *DirectActuator) scaleDown(d ScaleDecision) error {
	var lastErr error
	for i := 0; i < -d.Delta; i++ {
		inst := a.findOldestActive(d.ModelID, d.Variant)
		if inst == nil {
			err := fmt.Errorf("scale-down for model %q variant %v: no active instance found", d.ModelID, d.Variant)
			logrus.Warnf("[actuator] %v", err)
			lastErr = err
			continue // keep trying remaining iterations (next iteration may find different variant match)
		}
		inst.TransitionTo(InstanceStateDraining)
		logrus.Infof("[actuator] scale-down: draining instance %s for model %q", inst.ID(), d.ModelID)
	}
	return lastErr
}

// findOldestActive returns the first (oldest) active instance for the given model+variant.
// Candidates are sorted by sequence number embedded in the ID for determinism (R2).
func (a *DirectActuator) findOldestActive(model string, variant VariantSpec) *InstanceSimulator {
	var candidates []*InstanceSimulator
	for _, inst := range a.cluster.instances {
		if inst.Model != model {
			continue
		}
		if inst.State != InstanceStateActive {
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
