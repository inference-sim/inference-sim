# Data Model: Phase 1A — Infrastructure: Nodes, GPUs, Instances

**Branch**: `001-infra-nodes-gpus-instances`
**Date**: 2026-03-13

---

## Entities

### 1. `NodeState` (enum, `sim/cluster/infra_node.go`)

```text
Provisioning → Ready → Draining → Terminated
```

| Value | Description |
|-------|-------------|
| `NodeStateProvisioning` | Node is being VM-spun-up; not yet available for placement |
| `NodeStateReady` | Node is available for instance placement |
| `NodeStateDraining` | Node is waiting for all instances to terminate |
| `NodeStateTerminated` | Node no longer exists; GPUs released from inventory |

**Validation**: `IsValidNodeState(s string) bool` exposes validity check. Internal validation map unexported (R8).

**Transitions**:
- `Provisioning → Ready`: triggered by `NodeReadyEvent` after provisioning delay elapses
- `Ready → Draining`: triggered by cluster autoscaler or explicit drain command
- `Draining → Terminated`: triggered when all instances on node reach `Terminated`
- `Ready → Terminated`: shortcut when node has zero instances at drain time

---

### 2. `InstanceState` (enum, `sim/cluster/infra_node.go`)

```text
Scheduling → Loading → WarmingUp → Active → Draining → Terminated
```

| Value | Description |
|-------|-------------|
| `InstanceStateScheduling` | Placement pending; no node assigned yet |
| `InstanceStateLoading` | Placed on node; loading model weights (configurable delay) |
| `InstanceStateWarmingUp` | Model loaded; serving warm-up requests with TTFT penalty |
| `InstanceStateActive` | Fully ready; receives all new requests |
| `InstanceStateDraining` | No new requests accepted; existing requests completing |
| `InstanceStateTerminated` | Simulation complete; GPUs returned |

**Validation**: `IsValidInstanceState(s string) bool`. Internal map unexported (R8).

**Routing eligibility**: Only `InstanceStateActive` instances appear in `buildRouterState()` snapshots.

---

### 3. `NodePoolConfig` (config type, `sim/cluster/infra_config.go`)

Configuration for a pool of homogeneous nodes. Parsed from YAML with strict field parsing (R10).

| Field | Type | Validation | Description |
|-------|------|------------|-------------|
| `Name` | `string` | non-empty | Pool identifier (used in node/GPU IDs) |
| `GPUType` | `string` | non-empty | GPU model name (e.g., "H100", "A100") |
| `GPUsPerNode` | `int` | ≥1 | Number of GPUs on each node |
| `GPUMemoryGiB` | `float64` | >0 | Memory capacity per GPU (GiB) |
| `InitialNodes` | `int` | ≥0 | Nodes provisioned at simulation start |
| `MinNodes` | `int` | ≥0, ≤MaxNodes | Floor for autoscaler |
| `MaxNodes` | `int` | ≥InitialNodes | Ceiling for autoscaler |
| `ProvisioningDelay` | `DistSpec` | valid DistSpec | VM spin-up time distribution (seconds) |

**`IsValid()` method**: validates all fields, returns descriptive error. Factory `NewNodePoolConfig()` calls `IsValid()` and panics on invalid input (constructor invariant).

---

### 4. `InstanceLifecycleConfig` (config type, `sim/cluster/infra_config.go`)

Per-instance timing parameters for lifecycle transitions.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `LoadingDelay` | `DistSpec` | Constant(0) | Model weight loading time |
| `WarmUpRequestCount` | `int` | 0 | Number of warm-up requests (TTFT penalty applies) |
| `WarmUpTTFTFactor` | `float64` | 1.0 | Multiplier applied to TTFT during warm-up (e.g., 2.0) |
| `DrainPolicy` | `string` | `"WAIT"` | One of: `IMMEDIATE`, `WAIT`, `REDIRECT` |

**Validation**: `WarmUpTTFTFactor ≥ 1.0`, `WarmUpRequestCount ≥ 0`, drain policy in valid set.

---

### 5. `Node` (runtime entity, `sim/cluster/infra_node.go`)

Represents a single machine. Created by `PlacementManager` at simulation start or on scale-out.

| Field | Type | Description |
|-------|------|-------------|
| `ID` | `string` | Deterministic: `"{pool-name}-{index}"` |
| `PoolName` | `string` | Parent pool identifier |
| `GPUType` | `string` | GPU model (from pool config) |
| `TotalGPUs` | `int` | Fixed at node creation |
| `GPUs` | `[]*GPU` | Ordered slice of all GPUs on this node |
| `State` | `NodeState` | Current lifecycle state |
| `CostStartTime` | `int64` | Simulation clock when provisioning began |

**Conservation invariant**: `countFreeGPUs(node) + countAllocatedGPUs(node) == node.TotalGPUs` at all times.

---

### 6. `GPU` (runtime entity, `sim/cluster/infra_node.go`)

Single accelerator device.

| Field | Type | Description |
|-------|------|-------------|
| `ID` | `string` | Deterministic: `"{node-id}-gpu-{index}"` |
| `NodeID` | `string` | Parent node identifier |
| `PoolName` | `string` | Parent pool (for type lookup) |
| `Type` | `string` | GPU model name (copied from pool) |
| `MemoryGiB` | `float64` | Memory capacity |
| `AllocatedTo` | `InstanceID` | Empty string = free; non-empty = allocated |

---

### 7. `PlacementRecord` (trace record, `sim/cluster/infra_node.go` or `sim/trace/record.go`)

Captures the outcome of a placement decision for trace output.

| Field | Type | Description |
|-------|------|-------------|
| `Timestamp` | `float64` | Simulation clock when placement was attempted |
| `InstanceID` | `InstanceID` | Instance being placed |
| `Model` | `string` | Target model name |
| `TPDegree` | `int` | Required GPU count |
| `GPUType` | `string` | Required GPU type |
| `Outcome` | `string` | `"placed"`, `"pending"`, `"evicted"` |
| `NodeID` | `string` | Node assigned (empty if pending) |
| `GPUIDs` | `[]string` | GPUs assigned (empty if pending) |

---

### 8. Updated `sim.Request` (modified, `sim/request.go`)

**Added field**:
```go
Model string // Target model identifier for multi-model clusters; empty = default/single-model
```

**Backward compatibility**: All existing tests where `Model` is empty will route to the only instance (single-model clusters). The `buildRouterState()` filter is a no-op when all instances have the same model as the request.

---

### 9. Updated `sim.RoutingSnapshot` (modified, `sim/routing.go`)

**Added field**:
```go
Model string // Model served by this instance; used by buildRouterState() for filtering
```

---

### 10. Updated `InstanceSimulator` (modified, `sim/cluster/instance.go`)

**Added fields**:
```go
Model           string
State           InstanceState
warmUpRemaining int
nodeID          string
allocatedGPUIDs []string
```

**Added methods**:
- `IsRoutable() bool` — true iff `State == InstanceStateActive || State == InstanceStateWarmingUp`
- `IsWarmingUp() bool` — true iff `State == InstanceStateWarmingUp && warmUpRemaining > 0`
- `ConsumeWarmUpRequest()` — decrements `warmUpRemaining`; transitions to Active when 0
- `TransitionTo(state InstanceState)` — validates transition, panics on invalid transition

---

### 11. `PlacementManager` (runtime manager, `sim/cluster/infra_placement.go`)

Owns node/GPU inventory and placement logic.

**State**:
```go
pools     []*NodePool         // ordered by declaration order
nodesByID map[string]*Node    // lookup by node ID
rng       *sim.PartitionedRNG // "node-provisioning" subsystem
```

**Key methods**:
- `PlaceInstance(id InstanceID, model string, tpDegree int, gpuType string) (nodeID string, gpuIDs []string, err error)` — first-fit placement, returns error if no node has capacity (R5 transactional)
- `ReleaseInstance(id InstanceID) error` — returns GPUs to free pool
- `VerifyConservation() error` — checks `allocated + free == total` per node (INV-A)
- `FreeGPUCount(nodeID string) int` — current free GPU count for a node
- `ProvisionNode(pool *NodePool, clock float64) (*Node, float64)` — creates node, returns node + ready-time

---

### 12. `ModelMetrics` (new type, `sim/cluster/metrics.go`)

Per-model aggregated metrics for JSON output (FR-011).

| Field | Type | Description |
|-------|------|-------------|
| `Model` | `string` | Model name |
| `TTFT` | `Distribution` | Time-to-first-token distribution |
| `E2E` | `Distribution` | End-to-end latency distribution |
| `ThroughputRPS` | `float64` | Requests per second |
| `ThroughputTokensPerSec` | `float64` | Output tokens per second |
| `TotalRequests` | `int64` | Completed requests |

---

## State Transition Diagrams

### Node Lifecycle

```
[Provisioning] --NodeReadyEvent--> [Ready]
    |                                 |
    | (cost accrues)                  | --drain command--> [Draining]
    |                                 |                        |
    |                                 |--no instances-->       |
    |                                 |                        | --all instances terminated-->
    v                                 v                        v
   cost                            [Ready]              [Terminated]
 (from entry)                   (cost accrues)           (cost stops)
```

### Instance Lifecycle

```
[Scheduling] --placement found--> [Loading] --LoadingDelay elapses--> [WarmingUp]
                |                                                           |
                | (no placement = stays Scheduling)                        | --N warm-up reqs--> [Active]
                |                                                           |                        |
                v                                                           v                        | --drain--> [Draining]
           (pending)                                                    (TTFT×factor)                |                |
                                                                                                     v                |
                                                                                                  [Active]            | --in-flight done--> [Terminated]
```

---

## Invariants

| ID | Invariant | Location |
|----|-----------|----------|
| INV-A (new) | `allocated_gpus + free_gpus == total_gpus` per node at all times | `PlacementManager.VerifyConservation()` |
| INV-1 (existing) | Request conservation: `injected == completed + queued + running + dropped` | `ClusterSimulator` post-run check |
| INV-6 (existing) | Determinism: same seed → byte-identical stdout | Regression test |

---

## Configuration Impact

`DeploymentConfig` gains new fields (in `sim/cluster/deployment.go`):

```go
NodePools              []NodePoolConfig       // one or more node pools (FR-001)
InstanceLifecycle      InstanceLifecycleConfig // loading/warmup params
```

If `NodePools` is empty, the simulator behaves exactly as before (backward-compatible): instances are created without node/GPU tracking, and placement is a no-op.

**Backward-compatibility rule**: All existing tests pass with empty `NodePools`. The presence of node pools activates the placement machinery.
