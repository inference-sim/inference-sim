# Phase 1C: Two-Level Autoscaling — Design Document

**Status:** In progress  
**Tracking issue:** [#696](https://github.com/inference-sim/inference-sim/issues/696)  
**Sprint plan:** [Discussion #402](https://github.com/inference-sim/inference-sim/discussions/402#discussioncomment-15901661)  
**Reference implementation:** [llm-d WVA](https://github.com/llm-d/llm-d-workload-variant-autoscaler)

---

## 1. Problem

BLIS can simulate a multi-instance cluster with admission control, routing, and KV cache dynamics, but instance counts are fixed at startup. This prevents BLIS from modeling the dynamic replica management that production LLM serving systems perform, and from acting as a simulation substrate for autoscaling algorithm research (Phase 2: OpenEvolve/AlphaEvolve).

---

## 2. Design goals

1. **WVA algorithm parity** — implement the same pipeline structure and reference algorithms as [llm-d WVA](https://github.com/llm-d/llm-d-workload-variant-autoscaler).
2. **Model autoscaler first** — the model-level pipeline (`Collector → Analyzer → Engine → Actuator`) is the immediate focus. The cluster autoscaler (node provisioning) is deferred until the model autoscaler baseline is validated with the WVA/llm-d team. See Section 10.
3. **Two-level isolation** — model-level pipeline is independently testable with a fixed node pool, matching how WVA works in production. The cluster autoscaler and coordinator are additive layers on top, added only after model autoscaler baseline is established.
4. **Interface-per-module** — one single-method interface per module, multiple swappable implementations. Same pattern as BLIS's existing `Router`, `AdmissionController`, `Scheduler`.
5. **Realistic actuation** — model the three delays that make autoscaling policy design non-trivial.
6. **Two research hooks** — both `Analyzer` and `Engine` are targets for Phase 2 evolutionary search. They are independent dimensions: a new `Analyzer` improves capacity modeling; a new `Engine` improves allocation. See Section 10.

---

## 3. Module map

### 3.1 Model-level pipeline (WVA-equivalent)

```
┌─────────────────────────────────────────────────────────────────┐
│  ScalingTickEvent fires (ModelAutoscalerIntervalUs)             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  Collector interface    │
              │  DefaultCollector       │  ← wraps RouterState into
              └────────────┬────────────┘    per-model ModelSignals
                           │  []ModelSignals
                           ▼
              ┌─────────────────────────┐
              │  Analyzer interface     │  ← called once per model
              │  SaturationAnalyzer     │  ← KV+queue spare capacity
              │  UtilizationAnalyzer    │  ← single-signal KV baseline
              │  QueueAnalyzer          │  ← single-signal queue baseline
              │  QueueingModelAnalyzer  │  ← M/M/1 token model (future)
              └────────────┬────────────┘
                           │  []AnalyzerResult
                           │  (model-level supply/demand + per-variant breakdown)
                           ▼
              ┌─────────────────────────┐
              │  Engine interface       │  ← variant allocation
              │  GreedyEngine           │  ← respects GPU inventory
              │  UnlimitedEngine        │  ← fixed-node testing
              │  ← OpenEvolve target    │
              └────────────┬────────────┘
                           │  []ScaleDecision
                           │
              HPAScrapeDelay elapsed
                           │
                           ▼
              ┌─────────────────────────┐
              │  Actuator interface     │
              │  DirectActuator         │  ← calls PlacementEngine
              └─────────────────────────┘
```

### 3.2 Infrastructure layer (Karpenter/K8s equivalent, no WVA counterpart)

```
              ┌──────────────────────────┐
              │  ClusterAutoscaler       │  ← node provisioning
              │  PendingDrivenProvisioner│
              │  BinPackOptimizer        │
              └──────────────────────────┘

              ┌──────────────────────────┐
              │  DrainPolicy interface   │  ← instance shutdown
              │  ImmediateDrain          │
              │  WaitDrain               │
              │  RedirectDrain           │
              └──────────────────────────┘
```

---

## 4. Data flow and types

```
RouterState (existing)
    │
    │ Collector.Collect()
    ▼
ModelSignals                     // per model
  ModelID  string
  Replicas []ReplicaMetrics      // one per active replica
    InstanceID    string
    Variant       VariantSpec    // GPUType + TPDegree
    KVUtilization float64
    QueueDepth    int
    InFlightCount int
    TTFT          float64        // for QueueingModelAnalyzer
    DispatchRate  float64        // for QueueingModelAnalyzer
    │
    │ Analyzer.Analyze()  — called once per model
    ▼
AnalyzerResult                   // per model, aggregated from replica states
  ModelID           string
  TotalSupply       float64      // aggregate serving capacity (model-level)
  TotalDemand       float64      // aggregate load (model-level)
  Utilization       float64      // TotalDemand / TotalSupply
  RequiredCapacity  float64      // scale-up signal
  SpareCapacity     float64      // scale-down signal
  VariantCapacities []VariantCapacity   // per-variant breakdown for Engine
    Variant         VariantSpec
    Supply          float64
    Demand          float64
    ReplicaCount    int
    CostPerReplica  float64
    │
    │ Engine.Optimize()  — all models at once, with GPU inventory
    ▼
[]ScaleDecision
  ModelID  string
  Variant  VariantSpec
  Delta    int          // +N = add replicas, -N = remove replicas
    │
    │ (HPAScrapeDelay elapses)
    │
    │ Actuator.Apply()
    ▼
PlacementEngine.TryPlace() / DrainPolicy
```

**Key invariant on supply/demand:** `TotalSupply` and `TotalDemand` are **model-level aggregates**, computed by the `Analyzer` by summing across all replicas serving that model (grouped by variant for `VariantCapacities`). The `Engine` never reads raw `KVUtilization` or `QueueDepth` — it only reads `AnalyzerResult`.

**`GPUInventory` definition:**
```
GPUInventory.byVariant[v] = total GPU slots for variant v
                          - slots held by Loading instances (reserved, not yet serving)
                          - slots held by WarmingUp instances
                          - slots held by Active instances
                          - slots held by Draining instances (hold GPUs until drain completes)
```
Pending (Scheduling) placements are NOT subtracted (no GPU committed yet). Terminated instances are NOT subtracted. This is a committed-state snapshot — optimistic relative to in-flight operations.

**Zero-replica edge case:** When a model has no active replicas, `Collector` returns `ModelSignals{Replicas: []}`. `Analyzer.Analyze()` must return `AnalyzerResult{TotalSupply: 0, TotalDemand: 0, RequiredCapacity: 0, SpareCapacity: 0}` — no division by zero. Scale-from-zero is triggered by a separate path outside the `Analyzer` (see #908, deferred).

---

## 5. Interface contracts

### `Collector`
```go
type Collector interface {
    Collect(state *RouterState) []ModelSignals
}
```
**Observes:** `RouterState` (per-instance signals, already populated every tick).  
**Produces:** one `ModelSignals` per active model, grouping replicas by model ID.  
**Must not:** modify state, filter models, or apply thresholds — raw data only.

### `Analyzer`
```go
type Analyzer interface {
    Name() string
    Analyze(metrics ModelSignals) AnalyzerResult
}
```
**Observes:** `ModelSignals` for one model.  
**Produces:** model-level `TotalSupply`, `TotalDemand`, `RequiredCapacity`, `SpareCapacity`, and per-variant `VariantCapacities`.  
**Must not:** read `RouterState` directly, access GPU inventory, or emit `ScaleDecision`.  
**Zero-replica contract:** when `metrics.Replicas` is empty, return `AnalyzerResult` with all numeric fields zero — no panics, no divisions.  
**Invariant:** `sum(VariantCapacity.Supply) == TotalSupply`; `sum(VariantCapacity.Demand) == TotalDemand`.

### `Engine`
```go
type Engine interface {
    Optimize(results []AnalyzerResult, inventory GPUInventory) []ScaleDecision
}
```
**Observes:** `[]AnalyzerResult` (one per model) + `GPUInventory` (free slots per variant).  
**Produces:** `[]ScaleDecision` — at most one per model per call (conservative, reassess next tick).  
**Scale-up rule:** target cheapest available variant (`CostPerReplica` ascending).  
**Scale-down rule:** target most expensive active variant (`CostPerReplica` descending).  
**Cross-model priority:** when GPU inventory is insufficient to satisfy all scale-up requests, models are served in descending `RequiredCapacity` order (highest need first). This is the `GreedyEngine` default; other `Engine` implementations may use different priority rules (e.g. SLO tier).  
**Must not:** read `RouterState` or `ModelSignals` directly — only `AnalyzerResult`.

### `Actuator`
```go
type Actuator interface {
    Apply(decisions []ScaleDecision) error
}
```
**Observes:** `[]ScaleDecision`.  
**Effect:** calls `PlacementEngine.TryPlace()` for `Delta > 0`; initiates `DrainPolicy` for `Delta < 0`.  
**PendingPlacement cancellation:** before applying `Delta < 0` for a model, cancel any `PendingPlacement` entries queued for that model. This prevents the race where a pending scale-up and an active scale-down coexist for the same model, causing wasteful churn when the node eventually provisions.  
**Default scale-down behavior (`DirectActuator`):** in the immediate scope, before the full `DrainPolicy` interface is wired (`specs/010`, deferred), `DirectActuator` applies `WaitDrain` semantics for `Delta < 0`: set the instance to `Draining`, stop routing new requests to it (router already skips `Draining` instances), and free its GPUs only after `InFlightCount == 0`. This is safe, matches Kubernetes' default graceful termination (`terminationGracePeriodSeconds`), and requires no new code — instance states `Running`/`Draining`/`Terminated` already exist from Phase 1A.  
**Must not:** block — fires and forgets; placement delays and drain are handled by subsequent events.

### Pipeline cooldown (orchestrator-level, not an interface)

WVA delegates cooldown to Kubernetes' reconciliation stabilization window. BLIS owns the full pipeline and must implement it explicitly. Cooldown lives in `autoscalerPipeline.tick()` in `autoscaler.go`, not inside any interface — keeping `Engine` stateless:

```go
// In DeploymentConfig:
ScaleUpStabilizationWindowUs   float64 // HPA scale-up stabilization window in μs; 0 = pass immediately
ScaleDownStabilizationWindowUs float64 // HPA scale-down stabilization window in μs; 0 = pass immediately
```

The handler tracks `lastScaleUpAt[modelID]` and `lastScaleDownAt[modelID]`. A `ScaleDecision` from the Engine is suppressed (not forwarded to Actuator) if `now - lastScaleAt[modelID] < CooldownUs`. The cooldown timestamp is recorded when the decision passes the cooldown filter and is forwarded to ScaleActuationEvent (not on actuation).

---

## 6. Actuation chain — three delays

WVA's actuator emits a Prometheus metric; HPA/KEDA reads it and calls the Kubernetes scale API. BLIS models this entire pipeline as three configurable `DistributionSpec` delays:

| Field | Models | Typical range |
|-------|--------|---------------|
| `HPAScrapeDelay` | HPA/KEDA scrape + eval lag (WVA → scale API) | 15s–90s |
| `NodeProvisioningDelayUs` | VM boot time (Karpenter) | 30s–5min |
| `InstanceLoadingDelayUs` | Model weight load onto GPU | 10s–2min |

```
ScalingTickEvent
  → Collector.Collect()
  → Analyzer.Analyze() per model
  → Engine.Optimize()
  → ScaleActuationEvent scheduled at (now + HPAScrapeDelay)

ScaleActuationEvent
  → Actuator.Apply()
  → PlacementEngine.TryPlace()
      success: InstanceLoadingDelay → instance Running
      failure: PendingPlacement queued → ClusterAutoscaler.Decide()
                → NodeProvisioningDelay → NodeReadyEvent
                → placement retried
```

`HPAScrapeDelay = 0` (default) preserves all existing test output (INV-6).  
`HPAScrapeDelay > 0` enables oscillation research (H-Oscillation in Phase 1D).

---

## 7. Cross-cutting invariants

| Invariant | Description |
|-----------|-------------|
| **INV-A1** (instance conservation) | `active + draining + loading == sum(ScaleDecisions applied) - terminated` at all times |
| **INV-A2** (no silent drops) | Every `PlacementEngine.TryPlace()` failure produces a `PendingPlacement`. No `ScaleDecision` is silently discarded. |
| **INV-A3** (GPU conservation) | INV-4: `free + allocated == total` per node after every add, remove, place, and drain. |
| **INV-A4** (actuation ordering) | With `HPAScrapeDelay > 0`, no placement fires before `decision_time + sampled_delay`. INV-5 (causality) is preserved. |
| **INV-A5** (drain completeness) | Every instance that enters `Draining` eventually reaches `Terminated`. No instance stranded. Pairs with INV-11. |
| **INV-A6** (analyzer aggregation) | `sum(VariantCapacity.Supply) == AnalyzerResult.TotalSupply` and `sum(VariantCapacity.Demand) == AnalyzerResult.TotalDemand` for every result. |
| **INV-A7** (stabilization window gate) | A scale-up `ScaleDecision` for model M is forwarded to the Actuator only after the scale-up signal has been continuously present for `ScaleUpStabilizationWindowUs`. Timer resets if the signal disappears. Window=0 passes on first signal. Same semantics for scale-down with `ScaleDownStabilizationWindowUs`. |

**INV-1 interaction with drain policies:**

INV-1 currently reads: `injected == completed + queued + running + dropped + timed_out + deferred + ...`

Two drain policies add new terms:

- **`ImmediateDrain`** — in-flight requests are abandoned. These must be counted as a new terminal state `drained_dropped` (distinct from admission `dropped` and request `timed_out`). INV-1 becomes: `injected == completed + queued + running + dropped + timed_out + drained_dropped + ...`

- **`RedirectDrain`** — queued requests are re-enqueued to another instance. This is a **state transition, not a new injection**: the request object moves, `EnqueueTime` is updated, but `injected_requests` is NOT incremented. The request remains in the `queued` term throughout.

- **`WaitDrain`** — no new terms; in-flight requests complete normally and appear in `completed`.

---

## 8. WVA alignment

| WVA component | BLIS equivalent |
|--------------|-----------------|
| `internal/collector/` — pulls metrics from K8s pods + Prometheus | `Collector` interface + `DefaultCollector` (wraps `RouterState`) |
| `internal/interfaces/Analyzer` — `Analyze(AnalyzerInput) *AnalyzerResult` | `Analyzer` interface (same name, same contract) |
| `internal/engines/analyzers/saturation_v2/` — KV+queue spare capacity | `SaturationAnalyzer` (#905) |
| `pkg/analyzer/` — M/M/1 queueing model | `QueueingModelAnalyzer` (future issue) |
| `internal/interfaces/VariantAutoscalingsEngine` — variant allocation | `Engine` interface |
| `pkg/solver/greedy.go` — constrained allocation | `GreedyEngine` + `GreedySolver` (#new) |
| `pkg/solver/solver.go` SolveUnlimited — separable greedy | `UnlimitedEngine` (#new) |
| `pkg/solver` TODO: MIP solver | Future `Engine` implementation — Phase 2 OpenEvolve target |
| `internal/actuator/direct_actuator.go` — calls K8s scale subresource | `DirectActuator` (calls `PlacementEngine` directly) |
| HPA/KEDA scrape + eval lag | `HPAScrapeDelay` distribution (#742) |
| K8s `terminationGracePeriodSeconds` | `WaitDrain` (#910) |
| Pod preStop / service mesh drain | `RedirectDrain` (#911) |
| SIGKILL | `ImmediateDrain` (#910) |
| Karpenter node provisioning | `ClusterAutoscaler` + `NodeProvisioningDelayUs` (#740, #742) |
| `internal/engines/scalefromzero/` | `ScaleFromZeroEngine` (#908) |

---

## 9. Speckit feature scope

| Feature | Speckit dir | Issues | Status |
|---------|-------------|--------|--------|
| Model autoscaler | `specs/007-model-autoscaler/` | #692, #905, #906, #918 | **Active** |
| Cluster autoscaler | `specs/008-cluster-autoscaler/` | #740, #907 | Deferred — see Section 10 |
| Coordinator | `specs/009-coordinator/` | #741, #908, #909 | Deferred — see Section 10 |
| Actuation model | `specs/010-actuation-model/` | #742, #910, #911 | Deferred — see Section 10 |
| Observability | `specs/011-observability/` | #743, #912 | Deferred — see Section 10 |

---

## 10. Phasing and research hooks

### Immediate scope: model autoscaler baseline

The cluster autoscaler, coordinator, actuation model, and observability features are **deferred**. The immediate goal is a working, testable model autoscaler pipeline that the WVA/llm-d team can recognize and validate.

The minimal viable pipeline for that conversation:

```
DefaultCollector → SaturationAnalyzer → UnlimitedEngine → DirectActuator
```

- `SaturationAnalyzer` — WVA's own algorithm; the team already knows it
- `UnlimitedEngine` — simplest allocation (fixed-node, no inventory constraints)
- Fixed-node assumption throughout; no node provisioning, no drain policies needed

Once this baseline is running and validated with Lionel/team, we iterate: swap in `GreedyEngine`, add `QueueingModelAnalyzer`, introduce actuation delays, then tackle the cluster autoscaler layer.

### Two research hooks for Phase 2

`Analyzer` and `Engine` are **independent** research dimensions. A new `Analyzer` improves capacity modeling (how supply and demand are characterized); a new `Engine` improves allocation (how capacity is distributed across variants given the signals). OpenEvolve/AlphaEvolve can target either or both.

```
                  GreedyEngine   UnlimitedEngine   Evolved Engine
SaturationAnalyzer    ✓               ✓ (start)        Phase 2
QueueingModelAnalyzer ✓               ✓                Phase 2
Evolved Analyzer      Phase 2         —                Phase 2
```

**Evolving the Analyzer** — research question: *what is the right capacity model?*
- Discover better signal combinations (KV + queue + TTFT + dispatch rate, nonlinear)
- Adapt saturation thresholds per workload type (short vs long output, as WVA's `classifyOutputLength` already hints at)
- Improve or replace the M/M/1 queueing model parameters

**Evolving the Engine** — research question: *given signals, how do you allocate?*
- Improve on greedy for multi-model GPU scarcity (cross-model priority)
- Look-ahead allocation (account for actuation delay — don't scale up for load that will pass)
- Cost-performance Pareto frontier across variants
- This is where WVA's MIP solver TODO lives and where OpenEvolve fits most naturally

The two layers are decoupled by `AnalyzerResult` — any `Analyzer` implementation works with any `Engine` implementation without modification.
