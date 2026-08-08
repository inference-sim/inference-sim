// infra_placement.go implements PlacementManager: node/GPU inventory management
// and first-fit bin-packing instance placement. Phase 1A.
package cluster

import (
	"fmt"
	"math/rand"
	"sort"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
)

// PlacementManager manages node/GPU inventory and instance placement decisions.
// Uses first-fit bin-packing within matching pools (pool declaration order, then node index order).
//
// Thread-safety: NOT goroutine-safe. All calls must come from the simulation event loop.
type PlacementManager struct {
	pools        []*nodePoolState
	nodesByID    map[string]*Node   // all nodes by ID
	nodesByPool  map[string][]*Node // pool name → nodes in index order
	gpusByID     map[string]*GPU    // all GPUs by ID (authoritative GPU-ID → node resolver, #1529)
	pendingInsts []pendingInstance  // instances awaiting a Ready node
	provisionRng *rand.Rand         // RNG for provisioning delays (subsystemNodeProvisioning)
	loadingRng   *rand.Rand         // RNG for loading delays (subsystemInstanceLoading)
	nextNodeIdx  map[string]int     // pool name → next sequential node index counter
	// spanWarned latches true the first time an instance is placed across more than one
	// node (#1529). Gates a one-time warning that cross-node TP all-reduce is priced with
	// the intra-node term until #1530 prices the interconnect. Never reset.
	spanWarned bool
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
	id       InstanceID
	nodeID   string
	gpuIDs   []string
	gpuType  string        // gpu_type from the matched pool config
	tpDegree int           // tensor-parallel degree from the pending instance request
	simCfg   sim.SimConfig // per-instance simulator configuration
}

// NewPlacementManager creates a PlacementManager from the given node pool configs.
// Materializes initial nodes and GPUs for each pool's InitialNodes count.
// Panics if any pool config is invalid or pool names are duplicated (R3, constructor invariant).
func NewPlacementManager(pools []NodePoolConfig, provisionRng, loadingRng *rand.Rand, clock int64) *PlacementManager {
	pm := &PlacementManager{
		nodesByID:    make(map[string]*Node),
		nodesByPool:  make(map[string][]*Node),
		gpusByID:     make(map[string]*GPU),
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
		// Register in the authoritative GPU-ID index (#1529). newNode is the single GPU
		// construction site (R4), so the index cannot drift from the inventory.
		pm.gpusByID[gpus[g].ID] = gpus[g]
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

// PlaceInstance attempts to place an instance using two global passes over the
// matching pools (#1529). Considers only Ready nodes in pools matching gpuType, in
// pool declaration order.
//
// Pass 1 (single-node first-fit): the original behavior — bin-pack tpDegree GPUs
// onto a single Ready node. Tried across ALL matching pools first; on success the
// instance occupies exactly one node.
//
// Pass 2 (cross-node packing): reached ONLY when no single node in any matching
// pool can satisfy tpDegree. Packs tpDegree GPUs across multiple Ready nodes of a
// single pool (multi-node tensor parallelism). Because Pass 1 runs to completion
// across all pools before Pass 2 begins, single-node placement is strictly
// preferred to spanning anywhere — an instance spans only when nothing fits on one
// node (BC-1, BC-2). Cross-pool spanning is not attempted; a spanning instance
// stays within one pool (homogeneous interconnect domain).
//
// When gpuType is empty (""), all pools are considered (match-any semantics) — used
// by the NodePools construction path (SC-004) where the pool's gpu_type is
// authoritative, not the CLI --gpu flag. When gpuType is non-empty, only pools whose
// GPUType equals gpuType are tried.
//
// Select-then-commit atomicity (R5): GPUs are only mutated after full selection
// succeeds — a Pass-2 pool that falls short of tpDegree mutates nothing.
// Returns (nodeID, gpuIDs, matchedGPUType, nil) on success; ("", nil, "", error) when
// no capacity found. nodeID is the primary (lowest-index) node the instance occupies;
// for a spanning instance the full node set is derivable from gpuIDs via
// distinctNodesForGPUs. matchedGPUType is the gpu_type value from the matched pool config.
func (pm *PlacementManager) PlaceInstance(id InstanceID, model, gpuType string, tpDegree int) (nodeID string, gpuIDs []string, matchedGPUType string, err error) {
	if tpDegree < 1 {
		return "", nil, "", fmt.Errorf("PlaceInstance %s: tpDegree must be ≥1, got %d", id, tpDegree)
	}

	// ── Pass 1: single-node first-fit across all matching pools (unchanged) ──
	for _, poolState := range pm.pools {
		// Empty gpuType means "match any pool" — used when NodePools are configured
		// and the CLI --gpu flag is not used as a pool filter (SC-004).
		if gpuType != "" && poolState.config.GPUType != gpuType {
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

	// ── Pass 2: cross-node packing within a single pool (multi-node TP) ──
	// Reached only when Pass 1 found no single-node fit in any matching pool.
	for _, poolState := range pm.pools {
		if gpuType != "" && poolState.config.GPUType != gpuType {
			continue // type mismatch — skip pool
		}

		// Select-then-commit: gather free GPUs from Ready nodes of THIS pool in node
		// index order until tpDegree is reached. Only Ready nodes and only free GPUs
		// are eligible (never pack onto a Draining/Provisioning node). If the pool
		// cannot reach tpDegree, nothing is mutated and we try the next pool.
		nodes := pm.nodesByPool[poolState.config.Name]
		selected := make([]*GPU, 0, tpDegree)
		for _, node := range nodes {
			if node.State != NodeStateReady {
				continue
			}
			for _, gpu := range node.GPUs {
				if gpu.AllocatedTo == "" {
					selected = append(selected, gpu)
					if len(selected) == tpDegree {
						break
					}
				}
			}
			if len(selected) == tpDegree {
				break
			}
		}
		if len(selected) < tpDegree {
			continue // this pool cannot satisfy tpDegree even by spanning — no mutation
		}

		// Commit: mark all selected GPUs as allocated (atomic — we reached tpDegree).
		resultIDs := make([]string, tpDegree)
		for i, gpu := range selected {
			gpu.AllocatedTo = id
			resultIDs[i] = gpu.ID
		}
		// selected[0] is on the lowest-index touched node (nodes walked in order) —
		// report it as the primary node for bookkeeping/logging.
		primaryNode := selected[0].NodeID

		// One-time warning: a spanning instance's TP all-reduce is priced with the
		// intra-node NVLink term until #1530 prices the cross-node interconnect.
		if !pm.spanWarned && len(pm.distinctNodesForGPUs(resultIDs)) > 1 {
			pm.spanWarned = true
			logrus.Warnf("[cluster] instance %s spans multiple nodes for TP=%d: cross-node TP "+
				"all-reduce is priced with the intra-node term until #1530 prices the interconnect; "+
				"latency/throughput for spanning instances is optimistic", id, tpDegree)
		}
		return primaryNode, resultIDs, poolState.config.GPUType, nil
	}

	gpuTypeDisplay := gpuType
	if gpuTypeDisplay == "" {
		gpuTypeDisplay = "any"
	}
	return "", nil, "", fmt.Errorf("PlaceInstance %s: no Ready node has %d free %s GPUs (single-node or spanning within a pool)", id, tpDegree, gpuTypeDisplay)
}

// distinctNodesForGPUs resolves each GPU ID to its owning node and returns the
// distinct node IDs in sorted order (#1529). Used to derive how many nodes an
// instance spans — for cost accounting and the cross-node latency warning —
// without parsing GPU-ID strings (a node ID itself contains "-", so the
// "{nodeID}-gpu-{i}" format is not safely splittable). Resolution goes through
// the authoritative gpusByID index. An unknown GPU ID is skipped defensively;
// this never happens for GPUs that were actually placed. Sorted output is
// required for deterministic cost/logging (R2).
func (pm *PlacementManager) distinctNodesForGPUs(gpuIDs []string) []string {
	seen := make(map[string]struct{}, len(gpuIDs))
	for _, id := range gpuIDs {
		gpu, ok := pm.gpusByID[id]
		if !ok {
			continue
		}
		seen[gpu.NodeID] = struct{}{}
	}
	nodes := make([]string, 0, len(seen))
	for nodeID := range seen {
		nodes = append(nodes, nodeID)
	}
	sort.Strings(nodes)
	return nodes
}

// InstanceCostPerHour returns the per-instance hourly cost for an instance holding
// the given GPUs at the given per-node pool cost (#1529): distinct-nodes-spanned ×
// poolCostPerHour. A single-node instance is 1 × poolCostPerHour (unchanged from the
// pre-#1529 behavior, INV-6); a spanning instance is billed for every node it
// occupies. Used identically at all three placement sites (startup, deferred
// NodeReadyEvent, autoscaler scale-up) so the cost rule lives in one place.
func (pm *PlacementManager) InstanceCostPerHour(gpuIDs []string, poolCostPerHour float64) float64 {
	nodes := pm.distinctNodesForGPUs(gpuIDs)
	span := len(nodes)
	if span < 1 {
		span = 1 // defensive: an unplaced/empty GPU set bills as a single node, never zero
	}
	return poolCostPerHour * float64(span)
}

// ReleaseInstance returns GPUs allocated to id back to the free pool.
// Returns error if no GPUs are found for id (R1: no silent data loss).
// Checks for drain completion on EVERY node the instance's GPUs were freed from
// (R5: transactional semantics). A spanning instance (#1529) holds GPUs on more
// than one node, so the drain-completion check must run per touched node — the
// pre-#1529 code checked only the last node visited during release, which for a
// spanning instance would leave a fully-freed draining node's callback unfired.
func (pm *PlacementManager) ReleaseInstance(id InstanceID) error {
	released := false
	touched := make(map[string]*Node)

	for _, node := range pm.nodesByID {
		for _, gpu := range node.GPUs {
			if gpu.AllocatedTo == id {
				gpu.AllocatedTo = ""
				released = true
				touched[node.ID] = node
			}
		}
	}

	if !released {
		return fmt.Errorf("ReleaseInstance %s: no GPUs found for this instance", id)
	}

	// Drain completion check per touched node, in sorted ID order for determinism
	// (R2): each fired callback schedules a NodeDrainedEvent (and consumes a seqID),
	// so the firing order must be deterministic. For a node that is Draining and now
	// fully free, fire and clear its callback (each node owns a distinct callback, so
	// this cannot double-fire any single node).
	ids := make([]string, 0, len(touched))
	for nodeID := range touched {
		ids = append(ids, nodeID)
	}
	sort.Strings(ids)
	for _, nodeID := range ids {
		node := touched[nodeID]
		if node.State == NodeStateDraining &&
			node.allocatedCount() == 0 &&
			node.drainCallback != nil {
			cb := node.drainCallback
			node.drainCallback = nil
			cb()
		}
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
			nowPlaced = append(nowPlaced, placedInstance{id: p.id, nodeID: nodeID, gpuIDs: gpuIDs, gpuType: matchedGPUType, tpDegree: p.tpDegree, simCfg: p.simCfg})
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
