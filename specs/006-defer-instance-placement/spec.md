# Feature Specification: Defer Instance Construction Until After Placement

**Feature Branch**: `006-defer-instance-placement`
**Created**: 2026-03-31
**Status**: Draft
**Input**: User description: "fix(cluster): defer instance construction until after placement so roofline uses the pool's GPU type (issue #888/#893)"

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Accurate Roofline Latency When Node Pools Define GPU Type (Priority: P1)

A capacity planner configures a cluster with node pools that each specify a GPU type (e.g., `gpu_type: H100-SXM5`). The CLI flag `--gpu` is either omitted or set to a different value. When the simulation runs, every instance placed on an H100 node must estimate step latency using H100 peak throughput and bandwidth — not the CLI GPU fallback.

**Why this priority**: This is the core correctness gap. Without this fix, capacity planning results are silently wrong whenever node-pool GPU type differs from the CLI flag. All other stories depend on this working first.

**Independent Test**: Configure a cluster with one node pool specifying `gpu_type: A100` and a CLI flag `--gpu H100`. Run the simulation. Verify that the roofline latency model on each placed instance uses A100 hardware coefficients for step-time computation.

**Acceptance Scenarios**:

1. **Given** a deployment with `NodePools` configured with `gpu_type: A100` and `--gpu H100` on the CLI, **When** the simulation places an instance on an A100 node, **Then** that instance's latency model uses A100 peak TFLOPS and memory bandwidth — not H100 values.

2. **Given** a deployment with heterogeneous pools (some A100, some H100), **When** instances are placed across both pool types, **Then** each instance's latency model reflects the GPU type of its assigned node pool, not the CLI flag.

3. **Given** a deployment where `NodePools` is empty (no pools configured), **When** the simulation runs, **Then** behavior is byte-identical to the current behavior — the CLI `--gpu` value is used, and stdout output is unchanged for the same seed (INV-6 determinism preserved).

---

### User Story 2 — Deferred Placement Also Uses Correct GPU Type (Priority: P2)

Some instances cannot be placed immediately at simulation start because no ready capacity exists in the matching node pool. These instances are queued as pending and placed later when a node becomes ready. A user expects that when such an instance is eventually placed, it also uses the GPU type of the node it lands on — not a stale value captured at simulation startup.

**Why this priority**: The deferred path is a known correctness gap. If only the synchronous path is fixed, pending instances remain broken. This is less urgent than P1 only because it requires a node-startup event to trigger, which many configs don't exercise.

**Independent Test**: Configure a cluster where initial node capacity is insufficient for all instances. Observe that after a node-ready event fires, the newly placed instance uses the pool's GPU type in its latency model.

**Acceptance Scenarios**:

1. **Given** a node pool with insufficient initial capacity for all instances, **When** a `NodeReadyEvent` fires and a pending instance is placed, **Then** that instance is constructed using the GPU type of the pool it was placed in — not a GPU type captured before placement.

2. **Given** a pending instance that retries placement via `NodeReadyEvent`, **When** it is placed on a node belonging to a pool with `gpu_type: B200`, **Then** its roofline latency model uses B200 hardware calibration.

---

### User Story 3 — Dynamic Instance Registration Supports Deferred Construction (Priority: P3)

When an instance is constructed after simulation startup (deferred path), it must be registered with the snapshot provider so that the cluster router can observe it for load-balancing decisions. Without this registration, the new instance is invisible to routing.

**Why this priority**: This is an enabler for P2. The deferred construction path requires a dynamic registration mechanism that does not currently exist. This also unblocks the Phase 1C autoscaler.

**Independent Test**: After a `NodeReadyEvent` places a pending instance, send a request to the cluster. Verify that the request can be routed to the newly constructed instance (it appears in routing snapshots).

**Acceptance Scenarios**:

1. **Given** an instance constructed after simulation startup via the deferred path, **When** a routing snapshot is taken for load balancing, **Then** the new instance appears in the snapshot and is eligible to receive requests.

2. **Given** the snapshot provider initialized with the set of instances known at startup, **When** a new instance is dynamically added after startup, **Then** subsequent snapshots include the new instance without restart or reconfiguration.

---

### Edge Cases

- What happens when a node pool specifies a `gpu_type` not present in the hardware calibration file? The simulation must fail fast with an explicit error — not silently use wrong coefficients.
- What happens when `--gpu` is not provided and `NodePools` is empty? The existing default GPU fallback must apply (backward-compatible path unchanged).
- What happens when TP degree exceeds the GPUs available on a node? Placement fails and the instance is queued as pending — same behavior as today.
- What happens when all pending instances are eventually placed but the snapshot provider was never updated? Routing must not silently exclude the new instances.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: When `NodePools` is configured, the latency model for each instance MUST be initialized using the `gpu_type` of the node pool the instance is placed on, not the CLI `--gpu` value.
- **FR-002**: Instance construction MUST occur after `PlaceInstance` succeeds, so the matched pool's GPU type can be injected before the instance simulator is created.
- **FR-003**: The placement subsystem MUST return the matched pool's GPU type alongside node ID and GPU IDs, so the caller can build the correct configuration before constructing the instance.
- **FR-004**: The deferred placement path MUST also construct the instance after successful placement, using the GPU type of the assigned node pool.
- **FR-005**: When `NodePools` is empty, the construction and placement behavior MUST be byte-identical to the current behavior for the same seed (INV-6: determinism preserved).
- **FR-006**: The snapshot provider MUST support dynamic registration of instances constructed after simulation startup, so deferred instances are visible to the routing layer.
- **FR-007**: The pending instance record MUST carry enough context (pre-resolved configuration and pool role, excluding GPU) to construct the instance at placement time without re-reading global config.
- **FR-008**: No code path in the NodePools-active mode MUST derive a placed instance's GPU type from the CLI flag after this change — the pool's GPU type MUST be the sole source.

### Key Entities

- **InstanceSimulator**: A simulated inference server instance. Its latency model (hardware calibration) is locked in at construction time via GPU type lookup. After this fix, construction is deferred until after placement.
- **PlacementManager**: Assigns instances to nodes in pools. After this fix, it returns the matched pool's GPU type to the caller alongside node and GPU identifiers.
- **PendingInstance**: A record of an instance awaiting capacity. After this fix, it carries pre-resolved configuration (minus GPU) and pool role so the full instance can be constructed when placement succeeds.
- **SnapshotProvider**: Provides routing snapshots from the set of known instances. After this fix, it supports dynamic registration of instances added after simulation startup.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: When `NodePools` specifies a GPU type that differs from `--gpu`, every placed instance reports latency estimates derived from the pool's GPU hardware — verifiable by comparing step-time outputs with expected per-GPU roofline values.
- **SC-002**: When `NodePools` is empty, simulation output (stdout) for the same seed and config is byte-for-byte identical before and after the change.
- **SC-003**: After a deferred placement event places a previously-pending instance, that instance is routable — a request dispatched after the event can land on the new instance.
- **SC-004**: No CLI GPU flag read occurs in the NodePools-active construction path after the refactoring (verifiable by static inspection).
- **SC-005**: All existing tests pass without modification and no new lint findings are introduced.

## Assumptions

- The hardware calibration file contains entries for all `gpu_type` values used in `NodePools`. If a pool references an unknown GPU type, an error at placement time is acceptable.
- The TP degree is the same for all instances in the current design; no per-pool TP override is in scope for this fix.
- The no-NodePools path is treated as "placement always succeeds immediately with the CLI GPU value," unifying the construction logic without changing observable behavior.
- The dynamic `AddInstance` method added to the snapshot provider is intentionally designed to be the same entry point the Phase 1C autoscaler will use — this fix unblocks that work.

## Dependencies

- Placement subsystem: `PlaceInstance` signature change to also return matched pool GPU type.
- Pending instance record: gains pre-resolved configuration and pool role fields.
- Lifecycle event handler: constructs the instance simulator after retry placement succeeds.
- Snapshot provider: gains dynamic instance registration capability.
- Cluster constructor: instance construction loop moves inside/after placement; unified across NodePools and no-NodePools paths.
