package cluster

import (
	"fmt"
	"math"
	"sort"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim/trace"
)

// The cluster autoscaler (design SIM-14, inference-sim#740): a control loop beside the router that
// decides how many nodes each pool needs. It is separate from the model autoscaler, which decides
// replica counts: the two meet only in the pending-placement queue (#741). The model autoscaler
// asks for a replica, no node has room, the placement waits in the queue; the cluster autoscaler
// reads the queue, with a read-only view of each pool, and answers with node decisions; a node
// that becomes Ready retries the queue. It acts only on those published values.

// NodeDecision is a request to add or remove a node of a pool. Delta > 0 provisions nodes.
type NodeDecision struct {
	PoolName string
	Delta    int // +1 provisions one node, -1 drains one
}

// NodePoolState is a read-only snapshot of one node pool, as the cluster autoscaler sees it.
type NodePoolState struct {
	PoolName          string
	GPUType           string
	GPUsPerNode       int
	TotalNodes        int     // nodes that are not terminated, provisioning ones included
	ProvisioningNodes int     // of those, nodes not yet Ready
	MaxNodes          int     // the pool's configured ceiling (0: none)
	MinNodes          int     // the pool's configured floor
	GPUCapacity       int     // GPUs across the pool's non-terminated nodes
	GPUAllocated      int     // GPUs occupied by instances
	CostPerHour       float64 // per node-hour
}

// ClusterAutoscaler decides node counts. Decide is called with the pools' state and the pending
// placements in arrival order, now in microseconds of simulated time. It must not touch the model
// autoscaler or the cluster; it returns decisions.
type ClusterAutoscaler interface {
	Decide(now int64, pools []NodePoolState, pending []PendingPlacement) []NodeDecision
}

// PendingDrivenConfig configures the pending-driven provisioner.
type PendingDrivenConfig struct {
	// PendingThresholdUs is how long a placement may wait before a node is added for it.
	PendingThresholdUs float64 `yaml:"pending_threshold_us"`
	// MaxNodesPerPool caps the nodes of every pool (0: only each pool's own max_nodes).
	MaxNodesPerPool int `yaml:"max_nodes_per_pool"`
}

// ClusterAutoscalerConfig selects and configures the cluster autoscaler. The zero value is none.
type ClusterAutoscalerConfig struct {
	// Policy is "" (no scale-up) or "pending-driven".
	Policy              string `yaml:"policy"`
	PendingDrivenConfig `yaml:",inline"`
	// ScaleDown is "" (nodes are never removed) or "bin-pack".
	ScaleDown     string `yaml:"scale_down"`
	BinPackConfig `yaml:",inline"`
}

// ClusterAutoscalerBinPack is the scale-down policy that removes nodes of an underused pool.
const ClusterAutoscalerBinPack = "bin-pack"

// BinPackConfig configures the bin-pack optimizer.
type BinPackConfig struct {
	// LowWaterMark: a pool whose GPU utilization is below it is a candidate for losing a node.
	LowWaterMark float64 `yaml:"low_water_mark"`
	// MinNodesPerPool: never remove below this many nodes (a pool's own min_nodes also holds).
	MinNodesPerPool int `yaml:"min_nodes_per_pool"`
}

// ClusterAutoscalerPendingDriven is the policy name that adds a node when a placement has waited.
const ClusterAutoscalerPendingDriven = "pending-driven"

// Validate refuses a config that cannot run.
func (c ClusterAutoscalerConfig) Validate() error {
	switch c.ScaleDown {
	case "":
	case ClusterAutoscalerBinPack:
		if math.IsNaN(c.LowWaterMark) || c.LowWaterMark <= 0 || c.LowWaterMark > 1 {
			return fmt.Errorf("cluster_autoscaler.low_water_mark must be in (0, 1], got %v", c.LowWaterMark)
		}
		if c.MinNodesPerPool < 0 {
			return fmt.Errorf("cluster_autoscaler.min_nodes_per_pool must be >= 0, got %d", c.MinNodesPerPool)
		}
	default:
		return fmt.Errorf("cluster_autoscaler.scale_down %q is not a policy; the policy is %s", c.ScaleDown, ClusterAutoscalerBinPack)
	}
	switch c.Policy {
	case "":
		return nil
	case ClusterAutoscalerPendingDriven:
	default:
		return fmt.Errorf("cluster_autoscaler.policy %q is not a policy; the policy is %s", c.Policy, ClusterAutoscalerPendingDriven)
	}
	if math.IsNaN(c.PendingThresholdUs) || math.IsInf(c.PendingThresholdUs, 0) || c.PendingThresholdUs < 0 {
		return fmt.Errorf("cluster_autoscaler.pending_threshold_us must be a finite number >= 0, got %v", c.PendingThresholdUs)
	}
	if c.MaxNodesPerPool < 0 {
		return fmt.Errorf("cluster_autoscaler.max_nodes_per_pool must be >= 0, got %d", c.MaxNodesPerPool)
	}
	return nil
}

// PendingDrivenProvisioner adds a node to a pool when a placement for its GPU type has waited
// longer than the threshold, unless a node of the pool is already on its way or the pool is at its
// ceiling. It adds at most one node per pool per call, so a burst of requests does not provision
// a node for each (the first node's arrival retries the whole queue).
type PendingDrivenProvisioner struct{ cfg PendingDrivenConfig }

// NewPendingDrivenProvisioner returns the provisioner.
func NewPendingDrivenProvisioner(cfg PendingDrivenConfig) *PendingDrivenProvisioner {
	return &PendingDrivenProvisioner{cfg: cfg}
}

// Decide implements ClusterAutoscaler.
func (p *PendingDrivenProvisioner) Decide(now int64, pools []NodePoolState, pending []PendingPlacement) []NodeDecision {
	var out []NodeDecision
	for _, pool := range pools {
		if pool.ProvisioningNodes > 0 {
			continue // a node is already coming; its arrival retries the queue
		}
		ceiling := pool.MaxNodes
		if p.cfg.MaxNodesPerPool > 0 && (ceiling == 0 || p.cfg.MaxNodesPerPool < ceiling) {
			ceiling = p.cfg.MaxNodesPerPool
		}
		if ceiling > 0 && pool.TotalNodes >= ceiling {
			continue
		}
		for _, w := range pending {
			if w.Variant.GPUType != pool.GPUType && w.Variant.GPUType != "" {
				continue
			}
			if float64(now-w.RequestedAt) >= p.cfg.PendingThresholdUs {
				out = append(out, NodeDecision{PoolName: pool.PoolName, Delta: 1})
				break
			}
		}
	}
	return out
}

// BinPackOptimizer removes a node from the most expensive pool whose GPU utilization is below the
// low-water mark and whose allocated GPUs would still fit on the remaining nodes. It emits at most
// one decision per call (the next call looks again), and none while a placement waits or a node is
// on its way, since the pool is then growing, not idle.
type BinPackOptimizer struct{ cfg BinPackConfig }

// NewBinPackOptimizer returns the optimizer.
func NewBinPackOptimizer(cfg BinPackConfig) *BinPackOptimizer { return &BinPackOptimizer{cfg: cfg} }

// Decide implements ClusterAutoscaler.
func (b *BinPackOptimizer) Decide(_ int64, pools []NodePoolState, pending []PendingPlacement) []NodeDecision {
	if len(pending) > 0 {
		return nil
	}
	var candidates []NodePoolState
	for _, p := range pools {
		if p.ProvisioningNodes > 0 || p.GPUCapacity == 0 || p.GPUsPerNode == 0 {
			continue
		}
		floor := max(b.cfg.MinNodesPerPool, p.MinNodes)
		if p.TotalNodes <= floor || p.TotalNodes <= 0 {
			continue
		}
		if float64(p.GPUAllocated)/float64(p.GPUCapacity) >= b.cfg.LowWaterMark {
			continue
		}
		if p.GPUAllocated > p.GPUCapacity-p.GPUsPerNode { // the work would not fit on the other nodes
			continue
		}
		candidates = append(candidates, p)
	}
	if len(candidates) == 0 {
		return nil
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].CostPerHour != candidates[j].CostPerHour {
			return candidates[i].CostPerHour > candidates[j].CostPerHour
		}
		return candidates[i].PoolName < candidates[j].PoolName
	})
	return []NodeDecision{{PoolName: candidates[0].PoolName, Delta: -1}}
}

// composite runs the scale-up and the scale-down policy and joins their decisions.
type composite struct{ up, down ClusterAutoscaler }

func (c composite) Decide(now int64, pools []NodePoolState, pending []PendingPlacement) []NodeDecision {
	var out []NodeDecision
	if c.up != nil {
		out = append(out, c.up.Decide(now, pools, pending)...)
	}
	if c.down != nil {
		out = append(out, c.down.Decide(now, pools, pending)...)
	}
	return out
}

// newClusterAutoscaler builds the configured policies, nil for none.
func newClusterAutoscaler(c ClusterAutoscalerConfig) ClusterAutoscaler {
	var up, down ClusterAutoscaler
	if c.Policy == ClusterAutoscalerPendingDriven {
		up = NewPendingDrivenProvisioner(c.PendingDrivenConfig)
	}
	if c.ScaleDown == ClusterAutoscalerBinPack {
		down = NewBinPackOptimizer(c.BinPackConfig)
	}
	if up == nil && down == nil {
		return nil
	}
	return composite{up: up, down: down}
}

// poolStates is the placement manager's pools as the cluster autoscaler sees them, in pool order.
func (pm *PlacementManager) poolStates() []NodePoolState {
	out := make([]NodePoolState, 0, len(pm.pools))
	for _, ps := range pm.pools {
		s := NodePoolState{PoolName: ps.config.Name, GPUType: ps.config.GPUType, GPUsPerNode: ps.config.GPUsPerNode,
			MaxNodes: ps.config.MaxNodes, MinNodes: ps.config.MinNodes, CostPerHour: ps.config.CostPerHour}
		for _, n := range pm.nodesByPool[ps.config.Name] {
			if n.State == NodeStateTerminated {
				continue
			}
			s.TotalNodes++
			s.GPUCapacity += n.TotalGPUs
			s.GPUAllocated += n.TotalGPUs - n.freeCount()
			if n.State == NodeStateProvisioning {
				s.ProvisioningNodes++
			}
		}
		out = append(out, s)
	}
	return out
}

// poolSnapshot is the named pool's state now.
func (cs *ClusterSimulator) poolSnapshot(name string) NodePoolState {
	for _, p := range cs.placement.poolStates() {
		if p.PoolName == name {
			return p
		}
	}
	return NodePoolState{PoolName: name}
}

// gpuUtilization is a pool's allocated GPUs over its GPUs, 0 for a pool with none.
func gpuUtilization(p NodePoolState) float64 {
	if p.GPUCapacity == 0 {
		return 0
	}
	return float64(p.GPUAllocated) / float64(p.GPUCapacity)
}

// ClusterAutoscalerCheckEvent re-runs the cluster autoscaler when a queued placement reaches the
// threshold: a placement that was under the threshold when it was queued is acted on once it has
// waited long enough, without waiting for another placement to be queued.
type ClusterAutoscalerCheckEvent struct{ At int64 }

func (e *ClusterAutoscalerCheckEvent) Timestamp() int64 { return e.At }
func (e *ClusterAutoscalerCheckEvent) Priority() int    { return 8 }

// Execute runs the cluster autoscaler.
func (e *ClusterAutoscalerCheckEvent) Execute(cs *ClusterSimulator) { cs.runClusterAutoscaler() }

// runClusterAutoscaler asks the cluster autoscaler for decisions on the current queue and applies
// them: a +1 provisions a node and schedules its NodeReadyEvent after the pool's provisioning
// delay; a -1 is not applied (scale-down belongs to the node scale-down policy, #907).
func (cs *ClusterSimulator) runClusterAutoscaler() {
	if cs.clusterAutoscaler == nil || cs.placement == nil {
		return
	}
	pending := cs.placement.Pending()
	if len(pending) == 0 && cs.config.ClusterAutoscaler.ScaleDown == "" {
		return
	}
	decisions := cs.clusterAutoscaler.Decide(cs.clock, cs.placement.poolStates(), pending)
	sort.SliceStable(decisions, func(i, j int) bool { return decisions[i].PoolName < decisions[j].PoolName })
	for _, d := range decisions {
		if d.Delta < 0 {
			cs.removeNode(d.PoolName)
			continue
		}
		for i := 0; i < d.Delta; i++ {
			before := cs.poolSnapshot(d.PoolName)
			node, readyAt := cs.placement.ProvisionNode(d.PoolName, cs.clock)
			if cs.trace != nil {
				cs.trace.RecordClusterScale(trace.ClusterScaleRecord{Clock: cs.clock, PoolName: d.PoolName,
					OldNodeCount: before.TotalNodes, NewNodeCount: before.TotalNodes + 1, TriggerReason: "pending_threshold",
					PendingCount: len(pending), GPUUtilization: gpuUtilization(before)})
			}
			cs.schedule(&NodeReadyEvent{timestamp: readyAt, nodeID: node.ID}, &cs.lifecycleSender)
			logrus.Infof("[cluster-autoscaler] provisioning node %s in pool %q for %d pending placement(s); ready at %d",
				node.ID, d.PoolName, len(pending), readyAt)
		}
	}
	// A placement still under the threshold is looked at again when its wait reaches it.
	if cfg := cs.config.ClusterAutoscaler; cfg.Policy == ClusterAutoscalerPendingDriven {
		next := int64(math.MaxInt64)
		for _, w := range pending {
			if at := w.RequestedAt + int64(cfg.PendingThresholdUs); at > cs.clock && at < next {
				next = at
			}
		}
		if next != math.MaxInt64 {
			cs.schedule(&ClusterAutoscalerCheckEvent{At: next}, &cs.lifecycleSender)
		}
	}
}

// removeNode takes one node out of the pool: the newest Ready node that holds no instance. A node
// with instances on it is not removed (moving them is the drain policies' work, #1C-4); the
// decision is then dropped and the next call looks again.
func (cs *ClusterSimulator) removeNode(pool string) {
	nodes := cs.placement.nodesByPool[pool]
	for i := len(nodes) - 1; i >= 0; i-- {
		n := nodes[i]
		if n.State != NodeStateReady || n.freeCount() != n.TotalGPUs {
			continue
		}
		nodeID := n.ID
		err := cs.placement.DrainNode(nodeID, func() {
			cs.schedule(&NodeDrainedEvent{timestamp: cs.clock, nodeID: nodeID}, &cs.lifecycleSender)
		})
		if err != nil {
			logrus.Warnf("[cluster-autoscaler] drain node %s: %v", nodeID, err)
			return
		}
		logrus.Infof("[cluster-autoscaler] removing idle node %s from pool %q", nodeID, pool)
		return
	}
	logrus.Debugf("[cluster-autoscaler] pool %q has no idle node to remove", pool)
}
