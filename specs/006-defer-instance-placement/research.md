# Research: Defer Instance Construction Until After Placement (#888)

## Decision 1: PlaceInstance Return Signature

**Decision**: Extend `PlaceInstance` to return `(nodeID string, gpuIDs []string, gpuType string, err error)` — the matched pool's `GPUType` is added as a third return value.

**Rationale**: The matched pool's `GPUType` is the authoritative source for hardware calibration. Currently `PlaceInstance` takes `gpuType` from the caller (CLI flag) and filters pools by that value — but the caller should receive back the pool's own value to use for instance construction, not the CLI value. This eliminates the coupling between CLI config and the hardware calibration lookup.

**Alternatives considered**:
- Have the caller look up the pool GPU type separately via a new `GPUTypeForInstance()` query: requires extra state, adds a second lookup, and races if placement changes.
- Pass a mutable `*string` out-param to `PlaceInstance`: non-idiomatic Go.

---

## Decision 2: PlacedInstance Carries GPUType

**Decision**: The `placedInstance` struct returned by `RetryPendingInstances` gains a `gpuType string` field, populated from the matched pool at placement time.

**Rationale**: `NodeReadyEvent.Execute` consumes `[]placedInstance` and must construct the `InstanceSimulator` using the actual GPU type. Without the field on the struct, the handler would need a second lookup.

---

## Decision 3: PendingInstance Carries Pre-resolved SimConfig

**Decision**: `pendingInstance` gains a `simCfg sim.SimConfig` field holding the role-resolved config (without `GPU` set — that is injected at placement time). The existing `gpuType` field stays (needed to filter pools during retry). PoolRole is NOT stored separately; instead, the simCfg without GPU is stored directly — `GPU` is the only field that changes at placement time.

**Rationale**: `NodeReadyEvent.Execute` currently finds the pre-created `InstanceSimulator` and updates its fields. After the refactor, the instance does not exist yet — it must be constructed at this point. The handler needs the full SimConfig to call `NewInstanceSimulator`. Storing `simCfg` on `pendingInstance` avoids re-reading global config in the event handler (which would re-introduce the dependency on `config.GPU`).

**Alternatives considered**:
- Store `PoolRole` and re-call `resolveConfigForRole` in the event handler: re-introduces access to `config` in the event handler; handler must carry a config reference.
- Store nothing and pass the full config to `RetryPendingInstances`: changes the `PlacementManager` API to carry cluster-level concerns.

---

## Decision 4: Unified Construction Path (No-NodePools and NodePools)

**Decision**: Treat the no-NodePools path as "placement always succeeds immediately with `gpuType = config.GPU`", making the construction loop identical in both modes. The `if len(config.NodePools) > 0` branch is restructured so the instance construction happens inside the same placement-success block in both paths.

**Rationale**: A single code path eliminates the dual-maintenance risk (the same bug recurring in the legacy path). The no-NodePools path behavior is byte-identical to today since `config.GPU` is still used — just through the same construction site.

**Alternatives considered**:
- Keep the two paths fully separate: simpler short-term, but diverges over time and already caused this bug.

---

## Decision 5: CachedSnapshotProvider.AddInstance

**Decision**: Add `AddInstance(id InstanceID, inst *InstanceSimulator)` to `CachedSnapshotProvider` (concrete type, not the `SnapshotProvider` interface). The method inserts the instance into `p.instances`, initializes a fresh cache entry, and initializes a fresh `fieldTimestamps` entry.

**Rationale**: The `SnapshotProvider` interface is defined by behavioral contracts for consumers (routing). `AddInstance` is a management operation for the cluster constructor and event handlers — it belongs on the concrete type, not the interface (which would require all mock implementations to carry it).

**Alternatives considered**:
- Add `AddInstance` to the `SnapshotProvider` interface: forces all test doubles to implement it; breaks the single-method interface principle (Constitution III).
- Re-initialize the entire snapshot provider on each new instance: O(N) cost; breaks periodic cache state for existing instances.

---

## Decision 6: Instance Registration in NodeReadyEvent

**Decision**: After constructing the `InstanceSimulator` in `NodeReadyEvent.Execute`, call `cs.snapshotProvider.(*CachedSnapshotProvider).AddInstance(id, inst)` to register it, then append `inst` to `cs.instances` and add it to `cs.inFlightRequests`. The type assertion is acceptable here because `NodeReadyEvent` already operates within the `cluster` package where `CachedSnapshotProvider` is the only concrete implementation.

**Rationale**: The deferred instance must be visible to routing immediately after construction. Appending to `cs.instances` and registering with the snapshot provider are the two registration points.

**Alternatives considered**:
- Add a `RegisterInstance(inst)` method on `ClusterSimulator`: valid but adds indirection for a single call site — YAGNI until autoscaler (Phase 1C) needs it.

---

## Decision 7: No Change to SnapshotProvider Interface

**Decision**: The `SnapshotProvider` interface keeps two methods: `Snapshot` and `RefreshAll`. `AddInstance` is NOT added to the interface.

**Rationale**: The interface is used by the routing policy (reads only). Dynamic registration is a management concern of the cluster constructor and event handlers, not the routing consumer. Constitution Principle III: interfaces must accommodate ≥2 implementations — a read-only `SnapshotProvider` interface is the right contract for consumers.

---

## Key Invariants to Preserve

- **INV-6 (Determinism)**: The no-NodePools path must produce byte-identical stdout for the same seed. Verified by running existing cluster golden tests after the change with no modifications.
- **INV-4 (KV conservation)**: GPU allocation/release bookkeeping is unchanged; only the timing of instance construction moves.
- **R4 (Construction sites)**: `NewInstanceSimulator` has exactly one construction site inside the new unified placement-success block. `pendingInstance` struct literal has one construction site in `AddPending`. Both must be updated.
- **R5 (Transactional)**: The construction loop already rolls back on `PlaceInstance` failure (instance goes to pending). This property is preserved — construction only happens on the success branch.
- **R21 (Slice shrink during iteration)**: `RetryPendingInstances` already uses the swap-remove pattern with an index-based loop — no change needed here.
