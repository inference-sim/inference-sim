# Implementation Plan: Phase 1A — Infrastructure: Nodes, GPUs, Instances

**Branch**: `001-infra-nodes-gpus-instances` | **Date**: 2026-03-13 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-infra-nodes-gpus-instances/spec.md`

---

## Summary

Phase 1A introduces three first-class infrastructure entities — `NodePool`, `Node`, and `GPU` — that make cluster hardware topology explicit and addressable. The simulator gains: (1) a `PlacementManager` that bin-packs instances onto nodes by GPU type and TP degree; (2) node and instance lifecycle state machines with configurable provisioning/loading delays modeled as DES timer events; (3) three drain policies (IMMEDIATE, WAIT, REDIRECT); and (4) multi-model cluster routing with per-model metrics in the JSON output. All 14 functional requirements are implemented in `sim/cluster/` with backward-compatible changes to `sim/request.go` and `sim/routing.go`. The GPU conservation invariant (`allocated + free == total`) is enforced transactionally and verified by companion invariant tests alongside every placement-related golden test.

---

## Technical Context

**Language/Version**: Go 1.22+
**Primary Dependencies**: `gopkg.in/yaml.v3` (strict parsing), `gonum` (stats), `cobra`, `logrus`
**Storage**: In-memory node/GPU inventory maps; no external storage
**Testing**: `go test ./...` — table-driven, BDD-style; total suite under 60 seconds
**Target Platform**: Linux server (CI), macOS (dev)
**Project Type**: Library (`sim/`) + CLI (`cmd/`)
**Performance Goals**: No per-request overhead increase >5%; simulation remains CPU-bound
**Constraints**: `go test ./...` must complete under 60 s; zero `golangci-lint` failures; backward-compatible
**Scale/Scope**: Up to 100 nodes, 800 GPUs, 32 instances, 3 models in a single simulation run

---

## Constitution Check

*GATE: Must pass before implementation. Re-check after Phase 1 design.*

### Principle I — Architecture & Layering ✅

- `NodePool`, `Node`, `GPU`, `PlacementManager`, `InstanceState` → `sim/cluster/` ✅
- `Request.Model`, `RoutingSnapshot.Model` → `sim/` (routing interface types live in `sim/`) ✅
- Dependency direction: `cmd/ → sim/cluster/ → sim/` — no new cycles introduced ✅
- `sim/` remains a library; no `os.Exit` or `logrus.Fatalf` added ✅
- Bridge types added to `sim/routing.go` not `sim/cluster/` ✅

### Principle II — Determinism ✅

- Node IDs: `"{pool-name}-{sequential-index}"` — deterministic ✅
- GPU IDs: `"{node-id}-gpu-{index}"` — deterministic ✅
- Provisioning/loading delays sampled via `PartitionedRNG` with named subsystems (`"node-provisioning"`, `"instance-loading"`) ✅
- All map iteration over node/GPU maps uses sorted keys (R2) ✅
- `ClusterSimulator` already uses deterministic instance creation order; lifecycle events ordered by `(timestamp, priority, seqID)` ✅

### Principle III — Interface & Module Design ✅

- `DrainPolicy` interface with 3 implementations (IMMEDIATE, WAIT, REDIRECT) satisfies R13 ✅
- `PlacementManager` as concrete type (no interface needed — only one placement algorithm in spec) ✅
- `InstanceState` and `NodeState` validation via `IsValidInstanceState()` / `IsValidNodeState()` — maps unexported (R8) ✅
- New lifecycle event types implement existing `ClusterEvent` interface — no new interface needed ✅
- Factory function `NewNodePoolConfig()` validates, panics on invalid input ✅

### Principle IV — BDD/TDD ✅

- All 15 acceptance scenarios in spec become table-driven test cases ✅
- GPU conservation invariant (INV-A) companion test alongside every placement golden test ✅
- Warm-up TTFT penalty is directly measurable (SC-004) via per-request TTFT comparison ✅
- Behavioral assertions only: `assert.Equal(inst.State(), InstanceStateActive)` not `inst.(*ConcreteType)` ✅
- Total test budget: 60 s; individual tests ≤5 s ✅

### Principle V — Error Handling ✅

- `PlaceInstance()` returns `error` when no node has capacity; no silent drop (R1) ✅
- GPU allocation is transactional: select-then-commit; rollback if any GPU unavailable (R5) ✅
- `DrainRedirect` re-injection returns error on re-enqueue failure ✅
- All new `sim/cluster/` code returns errors to callers; no `logrus.Fatalf` (R6) ✅
- Loops over pending instances bounded by max iteration count = len(instances) (R19) ✅

### Principle VI — Configuration Discipline ✅

- `NodePoolConfig` is a new cluster-level sub-config; does NOT go into `SimConfig` (R16) ✅
- `DeploymentConfig` gains `NodePools []NodePoolConfig` and `InstanceLifecycle InstanceLifecycleConfig` ✅
- `*float64` not needed here (all config fields have clear non-zero defaults; zero is not a valid user value for `GPUsPerNode` or `GPUMemoryGiB`) ✅
- `yaml.KnownFields(true)` on all YAML parsing of new config (R10) ✅
- `cmd.Flags().Changed()` guard for any new CLI flags before applying defaults (R18) ✅

### Principle VII — System Invariants ✅

- **INV-A (new)**: `allocated_gpus + free_gpus == total_gpus per node` — companion test in `infra_placement_test.go` ✅
- INV-1 (request conservation): preserved; REDIRECT re-injects requests into event queue, they count as injected ✅
- INV-2 (request lifecycle): preserved; routing still only targets Active instances ✅
- INV-6 (determinism): PartitionedRNG subsystems added deterministically ✅
- INV-8 (work-conserving): preserved; placement manager doesn't affect event queue logic ✅
- INV-9 (oracle boundary): preserved; placement decisions never read `Request.OutputTokens` ✅

### Principle VIII — Antipattern Prevention

| Rule | Status | Notes |
|------|--------|-------|
| R1 | ✅ | `PlaceInstance` returns error; no silent drop |
| R2 | ✅ | Node map iteration sorted by node ID |
| R3 | ✅ | `NodePoolConfig.IsValid()` validates all numeric fields |
| R4 | ✅ | `Request` construction sites audited (grep before adding `Model` field) |
| R5 | ✅ | GPU allocation uses select-then-commit; rollback on failure |
| R6 | ✅ | No `logrus.Fatalf` in any `sim/cluster/` new code |
| R7 | ✅ | Every placement golden test has companion INV-A invariant test |
| R8 | ✅ | `validNodeStates`, `validInstanceStates` maps unexported |
| R9 | N/A | No YAML float64 fields where zero is meaningful (GPUsPerNode ≥1, etc.) |
| R10 | ✅ | `yaml.KnownFields(true)` in cluster config parsing |
| R11 | ✅ | `FreeGPUCount()` never divides; `KVUtilization` guard already in place |
| R13 | ✅ | `DrainPolicy` interface has 3 backends; `ClusterEvent` has 5+ implementations |
| R14 | ✅ | `PlaceInstance` only handles placement; `TransitionTo` only handles state |
| R16 | ✅ | `NodePoolConfig` in `DeploymentConfig`, not `SimConfig` |
| R19 | ✅ | Pending instance retry bounded by `len(pendingInstances)` max iterations |
| R21 | ✅ | Pending instances slice uses index-based iteration (slice can shrink when placed) |
| R22 | ✅ | Pre-check `FreeGPUCount ≥ tp_degree` matches actual allocation loop condition |
| R23 | ✅ | Warm-up TTFT factor applied consistently in all paths that record TTFT |

---

## Project Structure

### Documentation (this feature)

```text
specs/001-infra-nodes-gpus-instances/
├── spec.md              # Feature specification (input)
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 — architectural decisions and unknowns resolved
├── data-model.md        # Phase 1 — entities, fields, state transitions
├── quickstart.md        # Phase 1 — user-facing configuration guide
├── contracts/
│   └── node-pool-config.md  # YAML schema + CLI + output contracts
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created here)
```

### Source Code (repository root)

**New files in `sim/cluster/`:**

```text
sim/cluster/
├── infra_config.go          # NodePoolConfig, InstanceLifecycleConfig, DrainPolicy interface
├── infra_node.go            # Node, GPU, NodeState, InstanceState, PlacementRecord
├── infra_placement.go       # PlacementManager (bin-packing, GPU inventory)
├── infra_lifecycle_event.go # NodeReadyEvent, InstanceActiveEvent (DES timer events)
├── infra_node_test.go       # Node/GPU state machine + conservation invariant tests
├── infra_placement_test.go  # Placement algorithm tests (all 4 acceptance scenarios US2)
└── instance_lifecycle_test.go # Instance state machine + warm-up TTFT tests
```

**Modified files:**

```text
sim/request.go                  # +Model string field
sim/routing.go                  # +Model string to RoutingSnapshot
sim/cluster/instance.go         # +InstanceState, warmUpRemaining, Model, nodeID, allocatedGPUIDs
sim/cluster/cluster.go          # +PlacementManager integration, lifecycle event handling
sim/cluster/cluster_event.go    # +model-filtered buildRouterState()
sim/cluster/deployment.go       # +NodePools, InstanceLifecycle fields
sim/cluster/metrics.go          # +ModelMetrics, per-model aggregation in CollectRawMetrics()
sim/cluster/evaluation.go       # +PerModelMetrics in EvaluationResult
```

**Modified test files** (additive changes only):

```text
sim/cluster/cluster_test.go     # Add multi-model routing test cases
sim/cluster/metrics_test.go     # Add per-model metrics test cases
```

**No changes to:**
- `sim/kv/`, `sim/latency/`, `sim/workload/`, `sim/trace/`
- `cmd/` (no new CLI flags in Phase 1A; cluster config passed via `--cluster-config`)

**Structure Decision**: Single project layout (`sim/cluster/` extension). All new types are in `sim/cluster/` per existing convention. No new sub-packages. Six new files + nine modified files.

---

## Complexity Tracking

No constitution violations requiring justification. All new interfaces satisfy R13 (≥2 implementations). Architecture stays within existing two-layer model.

---

## Key Behavioral Contracts (for TDD)

The following GIVEN/WHEN/THEN contracts drive the TDD task breakdown. Full test scenarios are in the spec; abbreviated contracts here for plan-time orientation.

### Contract 1: Node Materialization (US1)

**GIVEN** `NodePools: [{name: "h100-pool", GPUType: "H100", GPUsPerNode: 8, InitialNodes: 2}]`
**WHEN** `ClusterSimulator` is constructed
**THEN** exactly 16 distinct GPU IDs of type H100 exist; `PlacementManager.VerifyConservation()` returns nil

### Contract 2: Bin-Packing Placement (US2)

**GIVEN** one 8-GPU node, two TP=4 instance requests
**WHEN** `PlaceInstance()` called twice
**THEN** both succeed; node has 0 free GPUs; INV-A holds

**GIVEN** one 8-GPU node fully allocated, one TP=4 instance request
**WHEN** `PlaceInstance()` called
**THEN** returns error "no node has capacity"; instance state is Scheduling

### Contract 3: GPU Release on Termination (US2, SC3)

**GIVEN** an instance occupying GPUs 0–3 of a node
**WHEN** `ReleaseInstance(id)` called
**THEN** GPUs 0–3 are free; `FreeGPUCount(nodeID) == 4`; INV-A holds

### Contract 4: Node Provisioning Delay (US3)

**GIVEN** a pool with `ProvisioningDelay: constant(120s)`
**WHEN** `ProvisionNode()` called at clock T
**THEN** `NodeReadyEvent` scheduled at T+120s; node state is `Provisioning` until T+120s; placement attempts before T+120s fail

### Contract 5: Instance Loading → WarmingUp → Active (US4)

**GIVEN** an instance with `LoadingDelay: constant(60s)`, `WarmUpRequestCount: 5`
**WHEN** placed at clock T
**THEN** `InstanceActiveEvent` for Loading→WarmingUp fired at T+60s; instance enters Active after 5 warm-up requests; warm-up TTFT = raw TTFT × `WarmUpTTFTFactor`

### Contract 6: Routing Filters by State (US4, US5)

**GIVEN** one Active instance and one Loading instance for model M
**WHEN** `buildRouterState()` called for a model-M request
**THEN** RouterState has exactly 1 snapshot (the Active instance)

### Contract 7: Multi-Model Isolation (US5)

**GIVEN** 2 llama instances and 2 qwen instances, all Active
**WHEN** a qwen request arrives
**THEN** RouterState contains only qwen snapshots; llama instances are never considered

### Contract 8: Per-Model Metrics in Output (US5, FR-011)

**GIVEN** 100 llama requests and 100 qwen requests completed
**WHEN** `CollectRawMetrics()` called
**THEN** `result.PerModel["llama"]` and `result.PerModel["qwen"]` both contain non-nil `ModelMetrics` with correct request counts

---

## DES Event Classification

New events added to the cluster event queue (all endogenous):

| Event | Priority | Trigger | Effect |
|-------|----------|---------|--------|
| `NodeReadyEvent` | 0 | Provisioning timer elapses | Node transitions Provisioning → Ready; pending instances attempted |
| `InstanceLoadedEvent` | 1 | Loading timer elapses | Instance transitions Loading → WarmingUp |
| `InstanceWarmUpCompleteEvent` | 1 | N-th warm-up request completes | Instance transitions WarmingUp → Active |
| `NodeDrainedEvent` | 0 | Last instance on node terminates | Node transitions Draining → Terminated |

Priority 0 for node events ensures they fire before admission/routing (priority 1/2) to prevent routing to a node that has just become unavailable.

---

## PartitionedRNG Subsystems Added

| Subsystem Name | Used For |
|---------------|----------|
| `"node-provisioning"` | Sampling provisioning delay from `ProvisioningDelay` DistSpec |
| `"instance-loading"` | Sampling loading delay from `LoadingDelay` DistSpec |

Both subsystems are registered with `PartitionedRNG` at `ClusterSimulator` construction time, preserving INV-6 (determinism).

---

## Backward Compatibility

All changes are additive and backward-compatible:

1. `Request.Model = ""` → routing behaves as before (all instances match empty model or no model filtering)
2. `RoutingSnapshot.Model = ""` → buildRouterState passes all snapshots when request model is empty
3. `DeploymentConfig.NodePools = nil/empty` → placement manager is a no-op; instances created without node assignment
4. `InstanceLifecycle.WarmUpRequestCount = 0` → warm-up phase skipped; instances immediately Active after Loading
5. Existing tests that construct `Request{}` without `Model` continue to pass

**Required construction site audit** (R4): Before adding `Model string` to `Request`, grep for all `Request{` and `sim.Request{` literals across the codebase and verify they compile without `Model` (Go zero-value for string is `""`, so existing literals are unaffected).
