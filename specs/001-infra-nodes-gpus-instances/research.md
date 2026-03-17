# Research Notes: Phase 1A — Infrastructure: Nodes, GPUs, Instances

**Branch**: `001-infra-nodes-gpus-instances`
**Date**: 2026-03-13
**Phase**: 0 — Resolve unknowns before design

---

## Decision 1: Package Placement for New Infrastructure Types

**Decision**: Add all new types (`Node`, `GPU`, `NodePool`, `PlacementManager`, `InstanceState`) to `sim/cluster/` — no new sub-package.

**Rationale**: Consistent with existing cluster-level types (`InstanceSimulator`, `ClusterSimulator`, `DeploymentConfig`). Adding a `sim/cluster/infra/` sub-package would increase import depth without architectural benefit; the infra types are tightly coupled to cluster orchestration.

**Alternatives considered**:
- `sim/cluster/infra/` sub-package — rejected; unnecessary for 5–6 new files.
- Promoting to `sim/` — rejected; node/GPU placement is cluster-level (instance-level code must not see cluster state).

---

## Decision 2: Node Provisioning Delay Distribution

**Decision**: Reuse `sim/workload.DistSpec` type for provisioning delay and instance loading delay distributions, sampled via `PartitionedRNG` with named subsystems (`"node-provisioning"`, `"instance-loading"`).

**Rationale**: `DistSpec` already models durations (Gaussian, Exponential, Weibull, Constant) and is validated by `sim/workload` constructors. Reuse avoids duplication and keeps the delay model consistent with arrival-process modeling.

**Alternatives considered**:
- Fixed `float64` delay field — rejected; spec explicitly requires distribution (FR-006).
- New `DurationDist` type — rejected; `DistSpec` already covers this use case exactly.

---

## Decision 3: GPU and Node ID Encoding

**Decision**: Deterministic, human-readable string IDs:
- Node ID: `"{pool-name}-{zero-padded-index}"` → `"h100-pool-0"`, `"h100-pool-1"`
- GPU ID: `"{node-id}-gpu-{index}"` → `"h100-pool-0-gpu-3"`
- Instance ID for new multi-model clusters: `"{model-slug}-{index}"` → `"qwen-0"`, `"llama-1"`

**Rationale**: IDs must be traceable to pool/node of origin (FR-002), deterministic (INV-6), and human-readable in trace output. Sequential indexing within each pool guarantees determinism.

**Alternatives considered**:
- UUID — rejected; non-deterministic, violates INV-6.
- Integer tuple — rejected; not self-documenting, harder to trace in logs.

---

## Decision 4: Multi-Model Routing — Where to Filter

**Decision**: Filter `RoutingSnapshot` list by `req.Model` inside `buildRouterState()` in `cluster_event.go`. Routing policies (`WeightedScoring`, `RoundRobin`, etc.) receive only the snapshots of instances serving the request's model and never need to know about model names.

**Rationale**: Separation of concerns (R14). Routing policies are model-agnostic; model filtering is a cluster-level routing pipeline concern. Adding a `Model string` field to `RoutingSnapshot` enables the filter and also lets trace records capture which model was being routed.

**Alternatives considered**:
- Each routing policy filters internally — rejected; cross-cutting concern, violates R14.
- Separate `RouterState` per model — rejected; over-engineering for first pass; `buildRouterState()` already reconstructs state per request.

**Required code changes**:
1. Add `Model string` to `sim.RoutingSnapshot` (in `sim/routing.go`).
2. Add `Model string` to `sim.Request` (in `sim/request.go`).
3. `buildRouterState()` filters `cs.instances` to those with `inst.Model == req.Model`.
4. `InstanceSimulator` gains a `Model string` field set at construction.

---

## Decision 5: Instance Lifecycle State Machine Placement

**Decision**: `InstanceState` enum and the `state` field live in `sim/cluster/instance.go` (the `InstanceSimulator` wrapper). The underlying `sim.Simulator` is unaware of lifecycle states.

**Rationale**: Lifecycle states (Loading, WarmingUp, Active, Draining, Terminated) are cluster-level orchestration concerns — they affect routing eligibility and placement decisions, not the core DES execution loop. `sim.Simulator` is a library and must remain lifecycle-agnostic.

**Alternatives considered**:
- Push states into `sim.Simulator` — rejected; violates the two-layer architecture; `sim/` must not contain cluster-level logic.

---

## Decision 6: Warm-Up Penalty Model

**Decision**: Additive TTFT multiplier for the first N requests served by a newly-Active instance. The multiplier is a configurable `WarmUpTTFTFactor float64` (default 2.0x), applied uniformly to all warm-up requests. No exponential decay for v1.

**Rationale**: Simplest model satisfying SC-004 and the spec assumption. The factor is applied after latency model calculation: `effectiveTTFT = rawTTFT * warmUpFactor` while `inst.warmUpRemaining > 0`. After N requests, `warmUpRemaining` reaches 0 and the penalty is removed.

**Implementation hook**: The cluster layer intercepts TTFT before recording by checking `inst.WarmUpRemaining()`. Because `sim.Simulator` computes TTFT internally, the cluster layer post-processes the TTFT from `sim.Metrics()` by annotating requests completed during warm-up. Alternatively, an `ExtraLatencyFn` hook on the instance can inject the penalty at step-time. Prefer a simpler approach: `InstanceSimulator` tracks warm-up state and exposes `IsWarmingUp() bool` + `ConsumeWarmUpRequest()`. The cluster event loop applies the TTFT factor before recording in per-request metrics.

**Alternatives considered**:
- Exponential decay warm-up curve — over-engineered for v1; rejected.
- Modify `sim.Simulator`'s latency model — rejected; crosses the layer boundary.

---

## Decision 7: Bin-Packing Placement Strategy

**Decision**: First-Fit across nodes within a pool (nodes ordered by node index, lowest first). Among all pools, try pools whose GPU type matches the instance's requirement, in pool declaration order.

**Rationale**: Deterministic (node iteration order is fixed), simple, sufficient for v1. First-fit is O(N) and predictable.

**Alternatives considered**:
- Best-fit (minimize remaining capacity after placement) — rejected; marginal benefit for simulation, adds complexity.
- Round-robin across nodes — rejected; may leave nodes with insufficient free GPUs when a TP=8 instance needs all GPUs of one node.

---

## Decision 8: GPU Allocation Transactional Rollback (R5)

**Decision**: `PlacementManager.PlaceInstance()` uses a collect-then-commit pattern. It first selects `tp_degree` free GPUs without modifying state, then atomically marks them as allocated in a single loop. If the selection fails at any point (e.g., mid-loop node state change), no GPU is allocated. Because Go is single-threaded within a simulation step, there is no concurrency concern; the rollback is a defensive pattern for future-proofing and rule compliance.

---

## Decision 9: Drain Policies

**Decision**: Three implementations of a `DrainPolicy` interface, all in `sim/cluster/infra_lifecycle_event.go` or `infra_node.go`:
- `DrainImmediate`: Instance moves to Terminated immediately; in-flight requests complete but no new step events are scheduled.
- `DrainWait`: Instance stops accepting new requests (routing excludes it); existing in-flight requests complete; instance terminates when batch empties.
- `DrainRedirect`: Like Wait, but pending (queued but not yet scheduled) requests are extracted from the instance's wait queue and re-injected into the cluster event queue as new `ClusterArrivalEvent`s at the current clock time.

**Rationale**: R13 requires ≥2 implementations for interfaces — three drain policies satisfy this. REDIRECT migration is approximated via re-injection (correct causal semantics for a discrete-event simulation).

---

## Decision 10: Per-Model Metrics

**Decision**: Add `PerModelMetrics map[string]*ModelMetrics` to `EvaluationResult`. `ModelMetrics` mirrors the key latency/throughput fields of `RawMetrics`. During `CollectRawMetrics()`, requests are partitioned by `req.Model` and per-model distributions are computed.

**Rationale**: FR-011 requires per-model TTFT p50/p99, E2E p50/p99, and throughput in JSON output. Reusing existing `Distribution` type avoids duplication.

---

## Decision 11: New GPU Conservation Invariant

**Decision**: Define INV-A: `allocated_gpus + free_gpus = total_gpus` per node, verified at every placement event. Companion test exists alongside every golden test that exercises placement.

**Verification strategy**: `PlacementManager.VerifyConservation() error` iterates all nodes, checks the sum; returns error if violated. Called in tests after every placement/release operation.

---

## Summary of Required `sim/` Layer Changes

| Change | File | Reason |
|--------|------|--------|
| Add `Model string` to `Request` | `sim/request.go` | Multi-model routing (FR-010) |
| Add `Model string` to `RoutingSnapshot` | `sim/routing.go` | Model-aware router state |
| No other changes to `sim/` | — | Lifecycle is cluster-layer concern |

All `sim/` changes are additive, backward-compatible with existing single-model cluster tests.
