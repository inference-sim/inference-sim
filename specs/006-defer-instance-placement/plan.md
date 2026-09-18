# Implementation Plan: Defer Instance Construction Until After Placement

**Branch**: `006-defer-instance-placement` | **Date**: 2026-03-31 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/006-defer-instance-placement/spec.md`

## Summary

When `NodePools` are configured, `NewClusterSimulator` currently constructs `InstanceSimulator` objects (locking in `HardwareCalib` via `GetHWConfig(hwConfigFile, config.GPU)`) before calling `PlaceInstance`. This means every instance uses the CLI `--gpu` flag for latency estimation, not the actual GPU type from the matched node pool. The fix defers instance construction to after `PlaceInstance` succeeds, injecting the matched pool's `GPUType` into `SimConfig.GPU` before calling `NewInstanceSimulator`. The same correction applies to the deferred path (`NodeReadyEvent`). The no-NodePools path is unified with the new path and remains byte-identical (INV-6).

## Technical Context

**Language/Version**: Go 1.22+
**Primary Dependencies**: `gopkg.in/yaml.v3`, `gonum`, `cobra`, `logrus` — no new dependencies
**Storage**: N/A (in-memory simulation)
**Testing**: `go test ./...` (table-driven, BDD/TDD); `golangci-lint run ./...` (v2.9.0, zero tolerance)
**Target Platform**: Linux/macOS CPU-only simulation binary
**Project Type**: Library (`sim/cluster/`) + CLI (`cmd/`)
**Performance Goals**: No change — this is a correctness fix; no hot paths affected
**Constraints**: INV-6 (byte-identical stdout for no-NodePools path); all 23 antipattern rules (R1-R23)
**Scale/Scope**: 5 files modified, 1 new method added; no new packages

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| **I. Architecture & Layering** | PASS | All changes in `sim/cluster/`; no import direction changes; `sim/` not touched |
| **II. Determinism (INV-6)** | PASS | No-NodePools path produces byte-identical output; deferred construction uses same RNG subsystems |
| **III. Interface & Module Design** | PASS | `AddInstance` added to concrete type only (not `SnapshotProvider` interface); `PlaceInstance` return type extended without breaking behavioral contract |
| **IV. BDD/TDD** | PASS | Tests written before implementation (see task breakdown below) |
| **V. Error Handling** | PASS | `PlaceInstance` error return preserved; new construction errors panic (constructor invariant) |
| **VI. Configuration Discipline** | PASS | No new config fields; `simCfg.GPU` override is internal to construction, not a new config parameter |
| **VII. System Invariants** | PASS | INV-4 (KV conservation) unaffected; INV-6 (determinism) explicitly tested; INV-1 (request conservation) unaffected |
| **VIII. Antipattern Prevention** | PASS | R4 (single construction site), R5 (transactional placement), R21 (no shrink-during-range), R1 (no silent drop) all preserved |

**Post-design re-check**: Constitution Check re-evaluated after Phase 1 — all gates still pass. See Complexity Tracking below.

## Project Structure

### Documentation (this feature)

```text
specs/006-defer-instance-placement/
├── plan.md              # This file
├── research.md          # Phase 0 output (complete)
├── data-model.md        # Phase 1 output (complete)
└── tasks.md             # Phase 2 output (/speckit.tasks command — NOT created here)
```

### Source Code (affected files)

```text
sim/cluster/
├── infra_placement.go          # PlaceInstance returns gpuType; placedInstance gains gpuType;
│                               # pendingInstance gains simCfg; AddPending signature extended
├── infra_config.go             # No change (pendingInstance lives in infra_placement.go)
├── infra_lifecycle_event.go    # NodeReadyEvent.Execute: construct InstanceSimulator after retry
├── snapshot.go                 # CachedSnapshotProvider.AddInstance (new method)
└── cluster.go                  # Unified construction loop; instance construction after PlaceInstance

sim/cluster/ (tests — written first)
├── infra_placement_test.go     # Update call sites for new PlaceInstance signature; add gpuType assertions
├── infra_node_test.go          # Unchanged (no signature changes here)
├── snapshot_test.go            # Add TestCachedSnapshotProvider_AddInstance
└── cluster_test.go             # Add deferred-construction correctness test; backward-compat golden check
```

## Phase 0: Research (Complete)

All decisions resolved in [research.md](research.md). Key decisions:

1. `PlaceInstance` returns `(nodeID, gpuIDs, matchedGPUType, err)` — pool GPU type is authoritative
2. `placedInstance` struct gains `gpuType` for deferred construction path
3. `pendingInstance` gains `simCfg sim.SimConfig` (GPU field empty until placement)
4. No-NodePools and NodePools paths unified — single construction site
5. `CachedSnapshotProvider.AddInstance` on concrete type only (not interface)
6. `NodeReadyEvent.Execute` constructs + registers instance after `RetryPendingInstances`

## Phase 1: Design (Complete)

Full data model and construction order in [data-model.md](data-model.md).

## Phase 2: Implementation Tasks

### Task T1 — Extend `PlaceInstance` return signature and `placedInstance`

**File**: `sim/cluster/infra_placement.go`

**Behavioral contract**:
- GIVEN a pool with `gpu_type: A100`, WHEN `PlaceInstance` succeeds, THEN the returned `matchedGPUType` equals `"A100"` (the pool's own value, not the caller's input)
- GIVEN `PlaceInstance` fails (no capacity), WHEN the error is returned, THEN `matchedGPUType` is `""` and `nodeID`/`gpuIDs` are empty — same as today
- GIVEN the `placedInstance` returned by `RetryPendingInstances`, THEN `gpuType` equals the matched pool's `GPUType`

**Changes**:
1. Update `PlaceInstance` signature: add `matchedGPUType string` as third return value
2. Set `matchedGPUType = poolState.config.GPUType` on the success path (`return node.ID, resultIDs, poolState.config.GPUType, nil`)
3. Update error return to `return "", nil, "", fmt.Errorf(...)`
4. Add `gpuType string` field to `placedInstance` struct
5. In `RetryPendingInstances`: update the internal `PlaceInstance` call to capture `gpuType`; populate `placedInstance.gpuType`

**TDD sequence**: Write `TestPlaceInstance_ReturnsMatchedPoolGPUType` (table-driven: pool gpu_type vs caller gpuType, success + failure cases) → red → implement → green → lint.

---

### Task T2 — Extend `pendingInstance` and `AddPending`

**File**: `sim/cluster/infra_placement.go`

**Behavioral contract**:
- GIVEN `AddPending(id, model, gpuType, tpDegree, simCfg)` is called, WHEN `RetryPendingInstances` succeeds later, THEN the caller can retrieve `simCfg` from the placement result to construct the instance

**Changes**:
1. Add `simCfg sim.SimConfig` field to `pendingInstance` struct
2. Update `AddPending` signature to accept `simCfg sim.SimConfig`
3. Store `simCfg` in the `pendingInstance` literal (R4: grep all construction sites)
4. Update `RetryPendingInstances` to propagate `simCfg` in `placedInstance` — or keep it on a parallel lookup (see note)

**Note on simCfg propagation**: `placedInstance` already carries `gpuType`; `simCfg` is needed by `NodeReadyEvent.Execute`. Two options:
- (A) Add `simCfg sim.SimConfig` to `placedInstance` as well — simpler, one return trip
- (B) Keep a separate map in `ClusterSimulator` from `InstanceID → simCfg` for pending instances

Option A is chosen: `placedInstance` gains `simCfg` (from `pendingInstance.simCfg`) alongside `gpuType`. The caller sets `simCfg.GPU = gpuType` before construction.

**TDD sequence**: Write `TestAddPending_StoresSimCfg` (verify simCfg retrievable through retry path) → red → implement → green → lint.

---

### Task T3 — `CachedSnapshotProvider.AddInstance`

**File**: `sim/cluster/snapshot.go`

**Behavioral contract**:
- GIVEN a `CachedSnapshotProvider` initialized with N instances, WHEN `AddInstance(id, inst)` is called with a new ID, THEN subsequent `Snapshot(id, clock)` calls return a valid snapshot for that instance
- GIVEN `AddInstance` is called for an ID already present, THEN the method panics (constructor invariant — no duplicate registration)

**Changes**:
1. Add `AddInstance(id InstanceID, inst *InstanceSimulator)` to `CachedSnapshotProvider`
2. Insert into `p.instances[id]`, `p.cache[id]`, `p.lastRefresh[id]`
3. Panic if `id` already present

**TDD sequence**: Write `TestCachedSnapshotProvider_AddInstance` (new instance appears in snapshot; duplicate panics) → red → implement → green → lint.

---

### Task T4 — Unified construction loop in `NewClusterSimulator`

**File**: `sim/cluster/cluster.go`

**Behavioral contract**:
- GIVEN `NodePools` configured with `gpu_type: A100`, WHEN `NewClusterSimulator` runs, THEN each placed instance has `latencyModel` calibrated for A100 (not CLI GPU type)
- GIVEN `NodePools` is empty, WHEN `NewClusterSimulator` runs, THEN stdout for the same seed is byte-identical to the current behavior (INV-6)
- GIVEN a placement failure, WHEN `AddPending` is called, THEN `simCfg` (with GPU empty) is stored for deferred construction

**Changes**:
1. Remove the two-step construction-then-placement loop
2. Replace with a unified loop:
   ```
   for each instance slot i:
       id = InstanceID(fmt.Sprintf("instance_%d", i))
       role = prePoolMembership[id] (or 0)
       simCfg = resolveConfigForRole(role)
       if len(config.NodePools) > 0:
           nodeID, gpuIDs, gpuType, err = cs.placement.PlaceInstance(id, model, gpuType_hint, tpDegree)
       else:
           gpuType = config.GPU; nodeID, gpuIDs = "", nil; err = nil (no-op placement)
       if err != nil:
           simCfg.GPU = ""  // leave GPU blank; will be set when placed
           cs.placement.AddPending(id, model, gpuType_hint, tpDegree, simCfg)
           // create placeholder inst in Scheduling state for instanceMap bookkeeping
           // (OR: skip creating inst entirely — deferred construction)
           ...
       else:
           simCfg.GPU = gpuType  // pool-authoritative
           inst = NewInstanceSimulator(id, simCfg)
           inst.nodeID = nodeID; inst.allocatedGPUIDs = gpuIDs
           ...
   ```

**Important implementation note**: The current code creates ALL instances upfront, including pending ones. The snapshot provider and `cs.instances` slice are built from this pre-created list. After the refactor, pending instances are NOT created upfront. The snapshot provider and `cs.instances` receive only placed instances at startup; deferred instances are added in T5.

This means `cs.instances` becomes a dynamic slice (not fixed at construction). `cs.inFlightRequests` must be initialized only for placed instances at construction time. Pending instance IDs that are not yet in `cs.instances` must not appear in routing state. This is correct behavior: an unplaced instance should not be routed to.

**TDD sequence**: Write `TestNewClusterSimulator_UsesPoolGPUType` (with NodePools; assert latency model uses pool GPU); `TestNewClusterSimulator_NoNodePools_DeterminismPreserved` (golden check, byte-identical stdout) → red → implement → green → lint.

---

### Task T5 — Deferred construction in `NodeReadyEvent.Execute`

**File**: `sim/cluster/infra_lifecycle_event.go`

**Behavioral contract**:
- GIVEN a pending instance successfully placed via `NodeReadyEvent`, WHEN `Execute` runs, THEN a new `InstanceSimulator` is constructed with `simCfg.GPU = matchedPoolGPUType`
- GIVEN the new instance is constructed, THEN it is registered with `snapshotProvider`, appended to `cs.instances`, and `cs.inFlightRequests[id]` is initialized to 0
- GIVEN the new instance is constructed, THEN `OnRequestDone` callback is wired (matching the startup path)
- GIVEN the new instance is constructed, THEN `warmUpRemaining` is set and the loading event is scheduled (matching the startup path)

**Changes**:
1. Remove the `for _, inst := range cs.instances { if inst.ID() == p.id { ... } }` loop that finds the pre-created instance
2. Replace with construction of a new `InstanceSimulator` from `p.simCfg` with `p.simCfg.GPU = p.gpuType` set
3. Call `cs.snapshotProvider.(*CachedSnapshotProvider).AddInstance(id, inst)`
4. Append to `cs.instances`
5. Initialize `cs.inFlightRequests[string(id)] = 0`
6. Wire `OnRequestDone` callback (mirror the startup path in `cluster.go`)
7. Set `warmUpRemaining`; `TransitionTo(InstanceStateLoading)`; `scheduleInstanceLoadedEvent(inst)`

**TDD sequence**: Write `TestNodeReadyEvent_DeferredConstruction_UsesPoolGPUType` (pending instance, then NodeReady fires, assert new instance uses pool GPU type and is routable) → red → implement → green → lint.

---

### Task T6 — Update all callers of changed signatures

**Files**: `sim/cluster/infra_placement_test.go`, `sim/cluster/cluster_test.go`, `sim/cluster/infra_node_test.go`

**Changes**:
1. Update all `PlaceInstance(...)` call sites to capture the new `matchedGPUType` return value (use `_` where not needed in tests)
2. Update all `AddPending(...)` call sites to pass `simCfg` (use `sim.SimConfig{}` as zero-value where not needed in tests)
3. Verify `go test ./...` passes

---

### Task T7 — Verification gate

**Behavioral contract** (all must hold):
- `go build ./...` exits 0
- `go test ./... -count=1` exits 0
- `golangci-lint run ./...` exits 0
- Grep confirms no `config.GPU` reads in the NodePools-active construction path
- No-NodePools golden output unchanged (INV-6)

## Complexity Tracking

No constitution violations. All changes are contained within `sim/cluster/` and touch ≤6 files. No new interfaces or packages introduced.

## Acceptance Criteria Mapping

| Criterion | Verified by |
|-----------|-------------|
| Pool GPU type used for roofline (SC-001) | T4: `TestNewClusterSimulator_UsesPoolGPUType` |
| No-NodePools byte-identical (SC-002, INV-6) | T4: `TestNewClusterSimulator_NoNodePools_DeterminismPreserved` |
| Deferred instance is routable (SC-003) | T5: `TestNodeReadyEvent_DeferredConstruction_UsesPoolGPUType` |
| No `config.GPU` read in NodePools path (SC-004) | T7: grep check |
| All tests pass, lint clean (SC-005) | T7: verification gate |
