// infra_node.go defines the node/GPU infrastructure types for Phase 1A.
// NodeState and InstanceState enums, Node and GPU runtime entities.
package cluster

import "fmt"

// NodeState represents the lifecycle state of a cluster node.
type NodeState string

const (
	NodeStateProvisioning NodeState = "Provisioning"
	NodeStateReady        NodeState = "Ready"
	NodeStateDraining     NodeState = "Draining"
	NodeStateTerminated   NodeState = "Terminated"
)

// validNodeStates is the unexported validation map (R8).
var validNodeStates = map[NodeState]struct{}{
	NodeStateProvisioning: {},
	NodeStateReady:        {},
	NodeStateDraining:     {},
	NodeStateTerminated:   {},
}

// IsValidNodeState returns true if s is a known NodeState value.
func IsValidNodeState(s string) bool {
	_, ok := validNodeStates[NodeState(s)]
	return ok
}

// InstanceState represents the lifecycle state of an instance (model server).
type InstanceState string

const (
	InstanceStateScheduling InstanceState = "Scheduling"
	InstanceStateLoading    InstanceState = "Loading"
	InstanceStateWarmingUp  InstanceState = "WarmingUp"
	InstanceStateActive     InstanceState = "Active"
	InstanceStateDraining   InstanceState = "Draining"
	InstanceStateTerminated InstanceState = "Terminated"
)

// validInstanceStates is the unexported validation map (R8).
var validInstanceStates = map[InstanceState]struct{}{
	InstanceStateScheduling: {},
	InstanceStateLoading:    {},
	InstanceStateWarmingUp:  {},
	InstanceStateActive:     {},
	InstanceStateDraining:   {},
	InstanceStateTerminated: {},
}

// IsValidInstanceState returns true if s is a known InstanceState value.
func IsValidInstanceState(s string) bool {
	_, ok := validInstanceStates[InstanceState(s)]
	return ok
}

// Node represents a single machine in a node pool.
// Carries a fixed inventory of GPUs tracked as allocated or free.
type Node struct {
	ID            string    // deterministic: "{pool-name}-{index}"
	PoolName      string    // parent pool name
	GPUType       string    // GPU model (from pool config)
	TotalGPUs     int       // fixed at node creation
	GPUs          []*GPU    // ordered by index; len(GPUs) == TotalGPUs
	State         NodeState // current lifecycle state
	CostStartTime int64     // simulation clock (microseconds) when provisioning began

	// drainCallback is called when the last allocated GPU on this node is released
	// while the node is in Draining state. Nil otherwise.
	drainCallback func()
}

// freeCount returns the number of unallocated GPUs on this node.
func (n *Node) freeCount() int {
	count := 0
	for _, g := range n.GPUs {
		if g.AllocatedTo == "" {
			count++
		}
	}
	return count
}

// allocatedCount returns the number of GPUs with a non-empty AllocatedTo field.
func (n *Node) allocatedCount() int {
	count := 0
	for _, g := range n.GPUs {
		if g.AllocatedTo != "" {
			count++
		}
	}
	return count
}

// GPU represents a single accelerator device.
type GPU struct {
	ID          string     // deterministic: "{node-id}-gpu-{index}"
	NodeID      string     // parent node ID
	PoolName    string     // parent pool name
	Type        string     // GPU model name
	MemoryGiB   float64    // memory capacity in GiB
	AllocatedTo InstanceID // empty = free; non-empty = allocated to this instance
}

// validNodeTransitions maps valid source → target state pairs.
var validNodeTransitions = map[NodeState]map[NodeState]struct{}{
	NodeStateProvisioning: {NodeStateReady: {}, NodeStateTerminated: {}},
	NodeStateReady:        {NodeStateDraining: {}, NodeStateTerminated: {}},
	NodeStateDraining:     {NodeStateTerminated: {}},
	NodeStateTerminated:   {},
}

// transitionNode validates and applies a node state transition.
// Panics on invalid transition (invariant violation per Principle V).
func transitionNode(n *Node, to NodeState) {
	targets, ok := validNodeTransitions[n.State]
	if !ok {
		panic(fmt.Sprintf("transitionNode %s: unknown source state %q", n.ID, n.State))
	}
	if _, valid := targets[to]; !valid {
		panic(fmt.Sprintf("transitionNode %s: invalid transition %q → %q", n.ID, n.State, to))
	}
	n.State = to
}
