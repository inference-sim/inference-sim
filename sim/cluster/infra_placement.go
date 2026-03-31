// infra_placement.go implements PlacementManager: node/GPU inventory management
// and first-fit bin-packing instance placement. Phase 1A.
package cluster

import (
	"fmt"
	"math/rand"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// PlacementManager manages node/GPU inventory and instance placement decisions.
// Uses first-fit bin-packing within matching pools (pool declaration order, then node index order).
//
// Thread-safety: NOT goroutine-safe. All calls must come from the simulation event loop.
type PlacementManager struct {
	pools        []*nodePoolState
	nodesByID    map[string]*Node    // all nodes by ID
	nodesByPool  map[string][]*Node  // pool name → nodes in index order
	pendingInsts []pendingInstance   // instances awaiting a Ready node
	provisionRng *rand.Rand          // RNG for provisioning delays (subsystemNodeProvisioning)
	loadingRng   *rand.Rand          // RNG for loading delays (subsystemInstanceLoading)
	nextNodeIdx  map[string]int      // pool name → next sequential node index counter
}

// nodePoolState bundles a NodePoolConfig for internal use.
type nodePoolState struct {
	config NodePoolConfig
}

// pendingInstance records an instance that could not be placed immediately.
type pendingInstance struct {
	id       InstanceID
	model    string
	gpuType  string
	tpDegree int
	simCfg   sim.SimConfig // per-instance simulator configuration
}

// placedInstance records a successfully placed instance with its node and GPU assignments.
type placedInstance struct {
	id      InstanceID
	nodeID  string
	gpuIDs  []string
	gpuType string         // gpu_type from the matched pool config
	simCfg  sim.SimConfig  // per-instance simulator configuration
}

// NewPlacementManager creates a PlacementManager from the given node pool configs.
// Materializes initial nodes and GPUs for each pool's InitialNodes count.
// Panics if any pool config is invalid or pool names are duplicated (R3, constructor invariant).
func NewPlacementManager(pools []NodePoolConfig, provisionRng, loadingRng *rand.Rand, clock int64) *PlacementManager {
	pm := &PlacementManager{
		nodesByID:    make(map[string]*Node),
		nodesByPool:  make(map[string][]*Node),
		nextNodeIdx:  make(map[string]int),
		provisionRng: provisionRng,
		loadingRng:   loadingRng,
	}

	// Validate and store pools; check for duplicate names.
	seen := make(map[string]struct{})
	for i := range pools {
		if err := pools[i].IsValid(); err != nil {
			panic(fmt.Sprintf("NewPlacementManager: pool[%d]: %v", i, err))
		}
		if _, dup := seen[pools[i].Name]; dup {
			panic(fmt.Sprintf("NewPlacementManager: duplicate pool name %q", pools[i].Name))
		}
		seen[pools[i].Name] = struct{}{}
		pm.pools = append(pm.pools, &nodePoolState{config: pools[i]})
		pm.nodesByPool[pools[i].Name] = nil
		pm.nextNodeIdx[pools[i].Name] = 0
	}

	// Create initial nodes — no provisioning delay applies to initial nodes;
	// they start in Ready state immediately (available at simulation start).
	for _, p := range pm.pools {
		for i := 0; i < p.config.InitialNodes; i++ {
			node := pm.newNode(&p.config, NodeStateReady, clock)
			pm.nodesByID[node.ID] = node
			pm.nodesByPool[p.config.Name] = append(pm.nodesByPool[p.config.Name], node)
		}
	}

	return pm
}

// newNode allocates a new Node for the given pool with the given initial state.
// Assigns deterministic IDs based on sequential pool index counters (INV-6).
func (pm *PlacementManager) newNode(cfg *NodePoolConfig, state NodeState, clock int64) *Node {
	idx := pm.nextNodeIdx[cfg.Name]
	pm.nextNodeIdx[cfg.Name]++
	nodeID := fmt.Sprintf("%s-%d", cfg.Name, idx)

	gpus := make([]*GPU, cfg.GPUsPerNode)
	for g := 0; g < cfg.GPUsPerNode; g++ {
		gpus[g] = &GPU{
			ID:        fmt.Sprintf("%s-gpu-%d", nodeID, g),
			NodeID:    nodeID,
			PoolName:  cfg.Name,
			Type:      cfg.GPUType,
			MemoryGiB: cfg.GPUMemoryGiB,
		}
	}

	return &Node{
		ID:            nodeID,
		PoolName:      cfg.Name,
		GPUType:       cfg.GPUType,
		TotalGPUs:     cfg.GPUsPerNode,
		GPUs:          gpus,
		State:         state,
		CostStartTime: clock,
	}
}

// NodeCount returns the total number of non-Terminated nodes across all pools.
func (pm *PlacementManager) NodeCount() int {
	count := 0
	for _, node := range pm.nodesByID {
		if node.State != NodeStateTerminated {
			count++
		}
	}
	return count
}

// GPUCount returns the total GPU capacity (free + allocated) for non-Terminated nodes
// in the named pool. Returns 0 for unknown pool names.
func (pm *PlacementManager) GPUCount(poolName string) int {
	nodes := pm.nodesByPool[poolName]
	total := 0
	for _, n := range nodes {
		if n.State != NodeStateTerminated {
			total += n.TotalGPUs
		}
	}
	return total
}

// FreeGPUCount returns the number of free GPUs on the named node.
// Returns 0 for unknown node IDs.
func (pm *PlacementManager) FreeGPUCount(nodeID string) int {
	node, ok := pm.nodesByID[nodeID]
	if !ok {
		return 0
	}
	return node.freeCount()
}

// VerifyConservation checks allocated + free == total for every node.
// Returns nil if all nodes pass; a descriptive error on the first violation (INV-A).
// Iterates nodes in sorted ID order for determinism (R2).
func (pm *PlacementManager) VerifyConservation() error {
	ids := make([]string, 0, len(pm.nodesByID))
	for id := range pm.nodesByID {
		ids = append(ids, id)
	}
	sort.Strings(ids) // R2: deterministic iteration

	for _, id := range ids {
		node := pm.nodesByID[id]
		allocated := node.allocatedCount()
		free := node.freeCount()
		if allocated+free != node.TotalGPUs {
			return fmt.Errorf("INV-A violation on node %s: allocated=%d free=%d total=%d (sum=%d)",
				node.ID, allocated, free, node.TotalGPUs, allocated+free)
		}
	}
	return nil
}

// PlaceInstance attempts to place an instance using first-fit bin-packing.
// Considers only Ready nodes in pools matching gpuType, in pool declaration order.
// Select-then-commit atomicity (R5): GPUs are only mutated after full selection succeeds.
// Returns (nodeID, gpuIDs, matchedGPUType, nil) on success; ("", nil, "", error) when no capacity found.
// matchedGPUType is the gpu_type value from the matched pool config.
func (pm *PlacementManager) PlaceInstance(id InstanceID, model, gpuType string, tpDegree int) (nodeID string, gpuIDs []string, matchedGPUType string, err error) {
	if tpDegree < 1 {
		return "", nil, "", fmt.Errorf("PlaceInstance %s: tpDegree must be ≥1, got %d", id, tpDegree)
	}

	for _, poolState := range pm.pools {
		if poolState.config.GPUType != gpuType {
			continue // type mismatch — skip pool
		}

		nodes := pm.nodesByPool[poolState.config.Name]
		for _, node := range nodes {
			if node.State != NodeStateReady {
				continue
			}
			if node.freeCount() < tpDegree {
				continue
			}

			// Phase 1 — select tpDegree free GPUs (no mutation yet)
			selected := make([]*GPU, 0, tpDegree)
			for _, gpu := range node.GPUs {
				if gpu.AllocatedTo == "" {
					selected = append(selected, gpu)
					if len(selected) == tpDegree {
						break
					}
				}
			}
			if len(selected) < tpDegree {
				// Shouldn't happen given freeCount check, but defensive
				continue
			}

			// Phase 2 — commit: mark GPUs as allocated
			resultIDs := make([]string, tpDegree)
			for i, gpu := range selected {
				gpu.AllocatedTo = id
				resultIDs[i] = gpu.ID
			}
			return node.ID, resultIDs, poolState.config.GPUType, nil
		}
	}

	return "", nil, "", fmt.Errorf("PlaceInstance %s: no Ready node has %d free %s GPUs", id, tpDegree, gpuType)
}

// ReleaseInstance returns GPUs allocated to id back to the free pool.
// Returns error if no GPUs are found for id (R1: no silent data loss).
// Checks for drain completion after release (R5: transactional semantics).
func (pm *PlacementManager) ReleaseInstance(id InstanceID) error {
	released := false
	var nodeToCheck *Node

	for _, node := range pm.nodesByID {
		for _, gpu := range node.GPUs {
			if gpu.AllocatedTo == id {
				gpu.AllocatedTo = ""
				released = true
				nodeToCheck = node
			}
		}
	}

	if !released {
		return fmt.Errorf("ReleaseInstance %s: no GPUs found for this instance", id)
	}

	// Drain completion check: if node is Draining and all GPUs are now free,
	// call the drain callback to schedule NodeDrainedEvent.
	if nodeToCheck != nil &&
		nodeToCheck.State == NodeStateDraining &&
		nodeToCheck.allocatedCount() == 0 &&
		nodeToCheck.drainCallback != nil {
		cb := nodeToCheck.drainCallback
		nodeToCheck.drainCallback = nil
		cb()
	}

	return nil
}

// ProvisionNode creates a new Provisioning-state node for the named pool.
// Samples the provisioning delay and returns (node, readyTime).
// The caller must schedule NodeReadyEvent at readyTime (clock + sampled delay).
func (pm *PlacementManager) ProvisionNode(poolName string, clock int64) (*Node, int64) {
	var poolState *nodePoolState
	for _, p := range pm.pools {
		if p.config.Name == poolName {
			poolState = p
			break
		}
	}
	if poolState == nil {
		panic(fmt.Sprintf("ProvisionNode: unknown pool %q", poolName))
	}

	node := pm.newNode(&poolState.config, NodeStateProvisioning, clock)
	pm.nodesByID[node.ID] = node
	pm.nodesByPool[poolName] = append(pm.nodesByPool[poolName], node)

	delay := poolState.config.ProvisioningDelay.Sample(pm.provisionRng)
	return node, clock + delay
}

// DrainNode initiates draining of the named node.
// Transitions the node Ready → Draining.
// If no instances are allocated, calls callback immediately.
// Otherwise, stores callback for invocation when the last instance is released.
// Returns error if node is unknown or not in Ready state.
func (pm *PlacementManager) DrainNode(nodeID string, callback func()) error {
	node, ok := pm.nodesByID[nodeID]
	if !ok {
		return fmt.Errorf("DrainNode: unknown node %q", nodeID)
	}
	if node.State != NodeStateReady {
		return fmt.Errorf("DrainNode %s: node is in state %q, expected Ready", nodeID, node.State)
	}

	transitionNode(node, NodeStateDraining)

	if node.allocatedCount() == 0 {
		callback() // no instances to wait for — fire immediately
	} else {
		node.drainCallback = callback
	}
	return nil
}

// MarkNodeTerminated transitions a node from Draining → Terminated
// and defensively frees any remaining GPU allocations.
// Also invokes and clears any pending drain callback to prevent memory leaks.
func (pm *PlacementManager) MarkNodeTerminated(nodeID string) error {
	node, ok := pm.nodesByID[nodeID]
	if !ok {
		return fmt.Errorf("MarkNodeTerminated: unknown node %q", nodeID)
	}
	transitionNode(node, NodeStateTerminated)
	for _, gpu := range node.GPUs {
		gpu.AllocatedTo = ""
	}
	// Clear drain callback if present to prevent memory leak when node is
	// terminated through a different path than normal drain completion.
	if node.drainCallback != nil {
		cb := node.drainCallback
		node.drainCallback = nil
		cb()
	}
	return nil
}

// MarkNodeReady transitions a Provisioning node → Ready.
func (pm *PlacementManager) MarkNodeReady(nodeID string) error {
	node, ok := pm.nodesByID[nodeID]
	if !ok {
		return fmt.Errorf("MarkNodeReady: unknown node %q", nodeID)
	}
	transitionNode(node, NodeStateReady)
	return nil
}

// RetryPendingInstances attempts placement for all pending instances now that a
// new node is Ready. Uses index-based iteration (R21: slice can shrink when instances placed).
// Bounded by initial pending count (R19: circuit breaker preventing livelock). Each instance
// is tried at most once per call; unplaced instances remain in pendingInsts for the next
// NodeReadyEvent. This ensures O(pending) per call with no unbounded retry.
// Returns the list of instances that were successfully placed, with their node and GPU assignments.
func (pm *PlacementManager) RetryPendingInstances() []placedInstance {
	if len(pm.pendingInsts) == 0 {
		return nil
	}

	var nowPlaced []placedInstance
	maxIter := len(pm.pendingInsts) // R19: at most one pass through pending list per call
	i := 0
	for i < len(pm.pendingInsts) && i < maxIter {
		p := pm.pendingInsts[i]
		nodeID, gpuIDs, matchedGPUType, err := pm.PlaceInstance(p.id, p.model, p.gpuType, p.tpDegree)
		if err == nil {
			nowPlaced = append(nowPlaced, placedInstance{id: p.id, nodeID: nodeID, gpuIDs: gpuIDs, gpuType: matchedGPUType, simCfg: p.simCfg})
			// Remove from pending: swap with last and shrink (R21).
			// Swap-remove pattern: move last element to position i, then truncate.
			// This is O(1) removal vs O(N) for shifting all elements left.
			pm.pendingInsts[i] = pm.pendingInsts[len(pm.pendingInsts)-1]
			pm.pendingInsts = pm.pendingInsts[:len(pm.pendingInsts)-1]
			// Do NOT increment i — the element now at position i (previously last)
			// needs to be checked in the next iteration.
		} else {
			i++
		}
	}
	return nowPlaced
}

// AddPending registers an instance as pending (placement deferred until a node is ready).
func (pm *PlacementManager) AddPending(id InstanceID, model, gpuType string, tpDegree int, simCfg sim.SimConfig) {
	pm.pendingInsts = append(pm.pendingInsts, pendingInstance{
		id:       id,
		model:    model,
		gpuType:  gpuType,
		tpDegree: tpDegree,
		simCfg:   simCfg,
	})
}

// SampleLoadingDelay samples and returns a loading delay in microsecond ticks.
func (pm *PlacementManager) SampleLoadingDelay(cfg *InstanceLifecycleConfig) int64 {
	return cfg.LoadingDelay.Sample(pm.loadingRng)
}
