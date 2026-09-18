# Data Model: Defer Instance Construction Until After Placement (#888)

## Changed Structs

### `pendingInstance` (infra_placement.go)

Records an instance that could not be placed immediately. Gains `simCfg` to enable construction at placement time.

| Field | Type | Change | Purpose |
|-------|------|--------|---------|
| `id` | `InstanceID` | unchanged | Instance identifier |
| `model` | `string` | unchanged | Model name for pool matching |
| `gpuType` | `string` | unchanged | GPU type filter for pool matching |
| `tpDegree` | `int` | unchanged | Tensor-parallel degree |
| `simCfg` | `sim.SimConfig` | **new** | Pre-resolved config (GPU field intentionally empty; injected at placement time) |

**Invariant**: `simCfg.GPU` is always `""` when stored in `pendingInstance`. The GPU field is set to `gpuType` (from matched pool) immediately before calling `NewInstanceSimulator`.

**Construction site**: `PlacementManager.AddPending(id, model, gpuType, tpDegree, simCfg)` — one site in `cluster.go`.

---

### `placedInstance` (infra_placement.go)

Records a successfully placed instance. Gains `gpuType` so the caller can construct the `InstanceSimulator` with the correct GPU.

| Field | Type | Change | Purpose |
|-------|------|--------|---------|
| `id` | `InstanceID` | unchanged | Instance identifier |
| `nodeID` | `string` | unchanged | Assigned node |
| `gpuIDs` | `[]string` | unchanged | Assigned GPU IDs |
| `gpuType` | `string` | **new** | Matched pool's GPU type (authoritative source for HardwareCalib) |

**Construction site**: inside `RetryPendingInstances` after successful `PlaceInstance` call — one site.

---

## Changed Function Signatures

### `PlacementManager.PlaceInstance`

```
Before: PlaceInstance(id InstanceID, model, gpuType string, tpDegree int) (nodeID string, gpuIDs []string, err error)
After:  PlaceInstance(id InstanceID, model, gpuType string, tpDegree int) (nodeID string, gpuIDs []string, matchedGPUType string, err error)
```

`matchedGPUType` is `poolState.config.GPUType` — the pool's own GPU type string, populated on success. Empty string on error.

**Callers to update**:
1. `cluster.go` — synchronous placement loop in `NewClusterSimulator`
2. `infra_placement.go` — `RetryPendingInstances` internal call (populates `placedInstance.gpuType`)
3. `infra_placement_test.go` — all test call sites

---

### `PlacementManager.AddPending`

```
Before: AddPending(id InstanceID, model, gpuType string, tpDegree int)
After:  AddPending(id InstanceID, model, gpuType string, tpDegree int, simCfg sim.SimConfig)
```

Stores `simCfg` (with `GPU` field empty) alongside existing fields.

**Callers to update**:
1. `cluster.go` — pending path in `NewClusterSimulator`

---

## New Methods

### `CachedSnapshotProvider.AddInstance`

```go
func (p *CachedSnapshotProvider) AddInstance(id InstanceID, inst *InstanceSimulator)
```

Inserts `inst` into `p.instances`, initializes `p.cache[id]` with a fresh `sim.NewRoutingSnapshot(string(id))`, and initializes `p.lastRefresh[id]` with zero timestamps.

**Callers**: `NodeReadyEvent.Execute` in `infra_lifecycle_event.go` after constructing the deferred `InstanceSimulator`.

---

## Construction Order Change (cluster.go)

### Before (NodePools path)

```
1. NewInstanceSimulator(id, simCfg)       ← GPU locked in here
2. PlaceInstance(...)                     ← placement after construction
3. inst.nodeID = nodeID
4. inst.allocatedGPUIDs = gpuIDs
```

### After (unified path — both NodePools and no-NodePools)

```
1. [determine gpuType: pool's GPU if NodePools, else config.GPU]
2. PlaceInstance(id, model, gpuType, tp)  ← placement first
3. simCfg = resolveConfigForRole(role)
4. simCfg.GPU = matchedGPUType            ← pool's authoritative GPU type
5. inst = NewInstanceSimulator(id, simCfg) ← GPU correct
6. inst.nodeID = nodeID
7. inst.allocatedGPUIDs = gpuIDs
```

### Deferred Path (NodeReadyEvent.Execute)

```
1. RetryPendingInstances() → []placedInstance{id, nodeID, gpuIDs, gpuType}
2. For each placed:
   a. Retrieve pendingInstance record (for simCfg)  ← stored in pendingInstance
   b. simCfg.GPU = placedInstance.gpuType
   c. inst = NewInstanceSimulator(id, simCfg)
   d. inst.nodeID, inst.allocatedGPUIDs = ...
   e. cs.snapshotProvider.(*CachedSnapshotProvider).AddInstance(id, inst)
   f. cs.instances = append(cs.instances, inst)
   g. cs.inFlightRequests[string(id)] = 0
   h. Wire OnRequestDone callback
   i. TransitionTo(InstanceStateLoading); scheduleInstanceLoadedEvent(inst)
```

---

## State Transition Invariants

- An `InstanceSimulator` is constructed exactly once per instance ID.
- After construction, the instance is immediately in `InstanceStateScheduling` (before transition) or the placement-success state — never in an uninitialised limbo.
- The `snapshotProvider`, `cs.instances`, and `cs.inFlightRequests` are always consistent: an entry in one implies entries in all three.

## No-NodePools Backward Compatibility

When `len(config.NodePools) == 0`, the unified path passes `gpuType = config.GPU` (CLI value) to a no-op "placement" that always succeeds immediately with `matchedGPUType = config.GPU`. `simCfg.GPU` is set to `config.GPU` — same as today. Stdout is byte-identical for the same seed (INV-6).
