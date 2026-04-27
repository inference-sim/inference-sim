# Queueing Model Autoscaler Engine for inference-sim

**Date:** 2026-04-22  
**Updated:** 2026-04-23  
**Status:** In Review  
**Repos:** [inference-sim](https://github.com/inference-sim/inference-sim) (target), [llm-inferno/control-loop](https://github.com/llm-inferno/control-loop) (source of components)  
**Reference:** [llm-d WVA queueingmodel analyzer](https://github.com/llm-d/llm-d-workload-variant-autoscaler/tree/main/internal/engines/analyzers/queueingmodel)

---

## Context

inference-sim has a pluggable autoscaler pipeline: `Collector → Analyzer → Engine → Actuator`. The current threshold-based `V2SaturationAnalyzer` + `GreedyEngine` scales based on KV-cache utilization. The goal is to add an SLO-driven, model-based, self-tuning alternative by porting the tuner (EKF-based parameter learning) and queue analyzer (capacity prediction at target latency) from llm-inferno.

inference-sim is designed to mirror the [llm-d Workload Variant Autoscaler (WVA)](https://github.com/llm-d/llm-d-workload-variant-autoscaler) and act as a DES-based testing environment for it. The new `QueueingModelAnalyzer` directly mirrors WVA's `internal/engines/analyzers/queueingmodel` package — same algorithm, same naming, same feature set — importing the same llm-inferno libraries as Go modules rather than vendoring older versions.

The result is a new `QueueingModelAnalyzer` and a modified `GreedyEngine` that together replace the threshold-based pipeline. `QueueingModelAnalyzer` is a drop-in for the `Analyzer` interface. `GreedyEngine` is extended — not replaced — to support exact-N replica scaling in addition to its existing ±1 behavior.

---

## Terminology Mapping

The repos use different vocabulary for the same concepts:

| llm-inferno | inference-sim / WVA | Notes |
|---|---|---|
| **model** | **model** (`ModelID`) | The LLM being served (e.g. `llama_13b`) |
| **accelerator** | **variant** (`VariantSpec.GPUType`) | GPU hardware type. inference-sim adds `TPDegree` as a second dimension of variant. |
| **server** | **(model, variant) pair** | In llm-inferno a server is (model + accelerator + service class). In inference-sim the autoscaler groups replicas by model, with hardware heterogeneity as variants within a model's `VariantCapacity` list. No service-class dimension in inference-sim yet. |
| **replica** (pod) | **replica** (`InstanceID`) | One running instance. |
| **service class** | *(not modeled)* | Deferred extension point. |

**EKF keying:** The model-tuner in llm-inferno keys EKF state by **(model, accelerator)** pair, since Alpha/Beta/Gamma differ by GPU type. `QueueingModelAnalyzer` follows the same approach: one estimation pipeline per **(ModelID, VariantSpec)** pair.

---

## Architecture

One new struct implements the `Analyzer` interface; one existing struct is extended:

```
Collector → QueueingModelAnalyzer → GreedyEngine (extended) → Actuator
             (new, Analyzer)          (modified, Engine)
```

`V2SaturationAnalyzer` continues to work with `GreedyEngine` unchanged — the extension is backward-compatible. All other existing components (`UnlimitedEngine`, `DefaultCollector`, `DirectActuator`) are untouched.

---

## Components

### QueueingModelAnalyzer (implements `Analyzer`)

**Purpose:** Learns per-(model, variant) performance parameters (Alpha/Beta/Gamma) from observed latency metrics using the same three-phase pipeline as `TunerService` in llm-inferno (NM bootstrap → sliding-window NM as primary → EKF as fallback), resolves SLO targets (from config, inferred from learned parameters, or estimated from observed latencies), and uses queue-analysis to compute max sustainable RPS per replica at target SLO.

**State:**
```go
type QueueingModelAnalyzer struct {
    config       QMConfig
    variantState map[modelVariantKey]*perVariantState  // keyed by (ModelID, VariantSpec)
}

type modelVariantKey struct {
    ModelID string
    Variant VariantSpec
}

type perVariantState struct {
    ie          *estimator.InitEstimator             // Phase 1: NM bootstrap (always active)
    swe         *estimator.SlidingWindowEstimator    // Phase 2a: sliding-window NM (nil until ie.IsReady())
    tuner       *core.Tuner                          // Phase 2b: EKF fallback (nil unless UseSliding=false or ekfFallback=true)
    ekfFallback bool                                 // true if SW init fit quality was too poor
    ekfUpdates  int                                  // accepted EKF updates (for NIS gate; Phase 2b only)
    alpha, beta, gamma float64                       // current best estimates; 0 until first successful fit
}

type QMConfig struct {
    SLOTargets        map[string]SLOTarget  // explicit per-model SLO targets; optional (guessed if absent)
    SLOMultiplier     float64               // queueing delay multiplier k for theory-based SLO guessing (default: 3.0; ρ = 1−1/k ≈ 0.67)
    TuningEnabled     bool                  // enable online parameter learning (default: true)
    InitObs           int                   // observations for IE bootstrap (default: 5)
    UseSliding        bool                  // use SlidingWindowEstimator as primary estimator (default: true)
    WindowSize        int                   // SWE window capacity (default: 20)
    ResidualThreshold float64               // SWE outlier rejection threshold (default: 0.3; 0 = disabled)
    InitFitThreshold  float64               // max NM func value for SW quality gate; 0 = no gate (default: 0)
    WarmUpCycles      int                   // EKF NIS gate disable count (default: 5; EKF path only)
}

type SLOTarget struct {
    TargetTTFT float32  // target time-to-first-token, milliseconds
    TargetITL  float32  // target inter-token latency, milliseconds
}
```

**Defaults:** `DefaultMaxBatchSize = 256`, `DefaultMaxQueueSize = 100`, `DefaultSLOMultiplier = 3.0`, `DefaultFallbackHeadroom = 1.5`.

SLO targets are optional in config — if absent they are inferred automatically (see SLO Resolution below). Token averages (`AvgInTokens`, `AvgOutTokens`) and `MaxBatchSize` are collected per replica at runtime — see Collector Extension. Each (model, variant) pair maintains its own estimation pipeline initialized lazily on first observation, converging independently to the variant's hardware-specific Alpha/Beta/Gamma.

**`Analyze(ModelSignals) AnalyzerResult` flow:**
1. If `ModelSignals.Replicas` is empty: return neutral result
2. Group replicas by variant; compute per-variant `workloadMetrics` (arrival-rate-weighted averages of AvgInTokens, AvgOutTokens, AvgTTFT, AvgITL across replicas with valid observations)
3. If `TuningEnabled`: for each variant group, for each replica with valid observations (ITL > 0, DispatchRate > 0, AvgInTokens > 0, AvgOutTokens > 0): build env (`Lambda = DispatchRate × 60`, AvgInputTokens, AvgOutputTokens, AvgTTFT, AvgITL, `MaxBatchSize = Variant.MaxBatchSize`), call `updateVariantParameters(state, env)` which implements the three-phase pipeline:
   - **Phase 1** (`InitEstimator`): `ie.AddObservation(env)`; if not `ie.IsReady()`: return (alpha/beta/gamma remain 0)
   - **Phase 2a** (SlidingWindowEstimator, when `UseSliding=true` and `!ekfFallback`):
     - On first call after IE ready: create SWE seeded from IE observations; run `ie.Fit()` for NM warm start; if `InitFitThreshold > 0` and fit quality poor: set `ekfFallback=true`, fall through to Phase 2b
     - On subsequent calls: `swe.AddObservation(env)`
     - If not `swe.IsReady()`: return; otherwise `swe.Fit()` → update alpha/beta/gamma
   - **Phase 2b** (EKF, when `UseSliding=false` or `ekfFallback=true`):
     - On first call: create tuner seeded with `ie.Fit()` result (or `GuessInitState` if fit failed)
     - `tuner.RunWithValidation(env, skipNIS)` where `skipNIS = (ekfUpdates < WarmUpCycles)`; update alpha/beta/gamma on accepted result
4. Resolve SLO target via `getSLOTarget(modelID, workloadMetrics)` — see SLO Resolution below; if no SLO can be determined: return neutral result
5. For each variant:
   a. If `alpha == 0` (no successful fit yet): **exclude this variant** from supply and demand
   b. Otherwise: call queue-analysis with (alpha, beta, gamma, TargetTTFT, TargetITL, workloadMetrics.AvgInTokens, workloadMetrics.AvgOutTokens, MaxBatchSize) → `maxRPS_per_replica`
6. For each variant with alpha > 0:
   - `variantSupply = maxRPS_per_replica × replicaCount` (all replicas of the variant, including those with zero observations)
   - `variantDemand = Σ DispatchRate` across replicas of this variant
   - Add to `VariantCapacities`
7. Compute model-level totals, `RequiredCapacity`, and `SpareCapacity` with N-1 safety check (same logic as V2SaturationAnalyzer)
8. Return `AnalyzerResult`

New (model, variant) pairs are initialized on first `Analyze()` call — no pre-registration required.

---

### SLO Resolution (`getSLOTarget`)

SLO targets are resolved in three priority levels, matching WVA's `guessSLOFromMetrics()`:

**Priority 1 — Explicit config:** Check `QMConfig.SLOTargets[modelID]`. If present, use directly.

**Priority 2 — Theory-based (from learned parameters):** For each variant with alpha > 0, apply the queueing model formulas using `SLOMultiplier` k:
```
TargetTTFT = k×α + (β+γ)×AvgInTokens
TargetITL  = k×α + β + γ×(AvgInTokens + (AvgOutTokens+1)/2)
```
Take the maximum across all variants with learned parameters. This produces SLO targets consistent with utilization ρ = 1−1/k (at k=3.0, ρ≈0.67).

**Priority 3 — Observation-based fallback:** Use `workloadMetrics.AvgTTFT` and `workloadMetrics.AvgITL` × `DefaultFallbackHeadroom` (1.5), capped at TTFT≤10000ms and ITL≤500ms.

If none of the three priorities yields a result (no config, no learned params, no observations yet), return neutral — no scaling decision is emitted until SLO can be resolved.

---

### GreedyEngine and UnlimitedEngine (extended)

**Purpose:** Both engines currently scale by ±1 per tick. The extension makes them compute the exact replica delta N from the supply/demand gap when `VariantCapacities` carry per-replica capacity information — which `QueueingModelAnalyzer` provides. The logic is not SLO-specific; it is generic and works with any analyzer that populates `VariantCapacities`. `GreedyEngine` respects `GPUInventory`; `UnlimitedEngine` does not — this distinction is unchanged.

This mirrors WVA's `CostAwareOptimizer`, which applies the same exact-N formula to any `AnalyzerResult` regardless of which analyzer produced it.

**Modified `Optimize([]AnalyzerResult, GPUInventory) []ScaleDecision` flow:**
```
for each model:
    perReplicaCapacity = vc.Supply / vc.ReplicaCount  (from VariantCapacities; 0 if no active replicas)
    if RequiredCapacity > 0:
        N = ceil(RequiredCapacity / perReplicaCapacity)  if perReplicaCapacity > 0, else 1
        → ScaleDecision{ModelID, cheapest variant with N free GPU slots, Delta: +N}
    if SpareCapacity > 0:
        N = floor(SpareCapacity / perReplicaCapacity), clamped to [1, replicaCount]  if perReplicaCapacity > 0, else 1
        → ScaleDecision{ModelID, most expensive active variant, Delta: -N}
```

**Backward compatibility:** when `perReplicaCapacity == 0` (no active replicas or `VariantCapacities` unpopulated), N falls back to 1, preserving the existing ±1 behavior for `V2SaturationAnalyzer` and any other analyzer that does not populate per-replica supply.

Scale-up: exact N replicas (fast convergence under step load changes).  
Scale-down: exact N replicas (avoids prolonged resource waste when spare capacity is large).  
GPU inventory constraint enforcement is unchanged.

---

## Collector Extension

TTFT, ITL, DispatchRate, and average token counts are **per-instance simulator metrics**, not observable from the routing layer. `DefaultCollector` reads from `RoutingSnapshot` which is populated by `CachedSnapshotProvider` — a router-side view that does not have access to server-internal latency statistics. Four changes are needed to surface these metrics into the autoscaler pipeline.

**`sim/cluster/autoscaler.go`** — extend `VariantSpec` with `MaxBatchSize`:
```go
type VariantSpec struct {
    GPUType      string
    TPDegree     int
    MaxBatchSize int  // server-configured max batch size; static per variant
}
```

Add `ITL`, `AvgInTokens`, `AvgOutTokens` to `ReplicaMetrics` (alongside the existing zero-filled `TTFT` and `DispatchRate`):
```go
ITL          float64  // μs; 0 if not yet available
AvgInTokens  float64  // average input tokens per request; 0 if not yet available
AvgOutTokens float64  // average output tokens per request; 0 if not yet available
```

**`sim/instance_simulator.go`** (or equivalent) — add a `LatencyStats()` method that computes rolling window averages of TTFT, ITL, DispatchRate, AvgInTokens, and AvgOutTokens from the completed-request data already tracked in `sim.Simulator.Metrics` (`RequestTTFTs`, `RequestITLs`, `AllITLs`). This is the supplier that bridges the simulator's per-request records into the per-instance aggregate view the autoscaler needs. For the initial implementation a simple sliding window over the most recent N completed requests is sufficient.

**`sim/routing.go`** — add to `RoutingSnapshot`:
```go
TTFT         float64  // μs; 0 if not yet available
ITL          float64  // μs; 0 if not yet available
DispatchRate float64  // req/s completed by this instance; 0 if not yet available
AvgInTokens  float64  // average input tokens per completed request; 0 if not yet available
AvgOutTokens float64  // average output tokens per completed request; 0 if not yet available
```
`buildRouterState()` populates these by calling `LatencyStats()` on the corresponding `InstanceSimulator`.

**`sim/cluster/default_collector.go`** — map the new `RoutingSnapshot` fields into `ReplicaMetrics`. The existing zero-fill behavior is preserved for any field the simulator has not yet populated (backward-compatible).

---

## Configuration

Add a single `QMConfig` field to `DeploymentConfig` in `sim/cluster/deployment.go`:
```go
QMConfig QMConfig  // queueing model analyzer config; zero value uses all defaults
```

`MaxBatchSize` per variant is set on `VariantSpec` at instance placement time, not via `QMConfig`. Token averages are collected at runtime from each instance.

The user selects `QueueingModelAnalyzer` at startup by constructing `autoscalerPipeline` with it in place of `V2SaturationAnalyzer`. `GreedyEngine` is used as-is — no change to pipeline wiring needed. No structural changes to autoscaler setup code.

---

## Error Handling & Edge Cases

| Condition | Behavior |
|---|---|
| `ModelSignals.Replicas` is empty | Skip all estimation updates, return neutral result |
| Zero ITL, DispatchRate, AvgInTokens, or AvgOutTokens for a replica | Exclude that replica from estimation (metrics not yet available); still count it in replicaCount for supply once that variant has alpha > 0 |
| `TuningEnabled = false` | Skip estimation entirely; alpha/beta/gamma remain 0 unless seeded externally |
| IE still accumulating (`ie.ObsCount() < InitObs`) | Exclude variant from both supply and demand; alpha/beta/gamma remain 0 |
| SWE not yet ready (window not full) | Exclude variant from both supply and demand |
| SW init fit quality poor (`InitFitThreshold > 0` and NM func value exceeds threshold) | Set `ekfFallback=true`; activate EKF path for this variant |
| No accepted fit/EKF update yet (alpha == 0) | Exclude variant from both supply and demand; other variants in the same model are unaffected |
| alpha/beta/gamma zero or negative after fit | Clamp to configurable minimum (e.g., 1 RPS) to prevent division-by-zero in engine |
| No SLO target determinable (no config, no learned params, no observations) | Return neutral result; no scaling decision emitted until SLO resolves |
| GPUInventory exhausted on scale-up | Skip decision for that model; existing `GreedyEngine` behavior unchanged |
| `perReplicaCapacity == 0` in `GreedyEngine` (no active replicas) | Falls back to N=1, preserving original ±1 behavior |
| New variant added to existing model | Its pipeline initializes independently; existing variants with alpha > 0 are unaffected |

---

## Files

### New files (inference-sim)

| File | Contents |
|---|---|
| `sim/cluster/queueing_model_analyzer.go` | `QueueingModelAnalyzer`, `modelVariantKey`, `perVariantState`, `QMConfig`, `SLOTarget`, `updateVariantParameters()`, `getSLOTarget()` |

### Modified files (inference-sim)

| File | Change |
|---|---|
| `sim/cluster/engine.go` | Extend `GreedyEngine.Optimize()` and `UnlimitedEngine.Optimize()` to compute exact N from `VariantCapacities`; falls back to N=1 when `perReplicaCapacity == 0` |
| `sim/cluster/autoscaler.go` | Add `MaxBatchSize` to `VariantSpec`; add `ITL`, `AvgInTokens`, `AvgOutTokens` to `ReplicaMetrics` |
| `sim/routing.go` | Add `TTFT`, `ITL`, `DispatchRate`, `AvgInTokens`, `AvgOutTokens` to `RoutingSnapshot` |
| `sim/instance_simulator.go` | Add `LatencyStats()` method returning rolling-window averages of TTFT, ITL, DispatchRate, AvgInTokens, AvgOutTokens from completed-request metrics |
| `sim/cluster/cluster_event.go` | Populate new `RoutingSnapshot` fields by calling `LatencyStats()` in `buildRouterState()` |
| `sim/cluster/default_collector.go` | Map new `RoutingSnapshot` fields into `ReplicaMetrics` |
| `sim/cluster/deployment.go` | Add `QMConfig QMConfig` to `DeploymentConfig` |
| `go.mod` | Add `github.com/llm-inferno/model-tuner` (direct), promote `github.com/llm-inferno/queue-analysis` (direct) |

### Unchanged (inference-sim)

`Analyzer` interface, `Engine` interface, `DirectActuator`, `V2SaturationAnalyzer` — untouched.

---

## Verification

**Unit tests:**
- `QueueingModelAnalyzer`: synthetic `ModelSignals` with known TTFT/ITL/DispatchRate → assert `AnalyzerResult` supply/demand values once IE bootstrap completes
- Bootstrap exclusion: assert variant excluded from supply/demand for first `InitObs - 1` calls; assert variant included from call `InitObs` (SWE seeded and fit succeeds); assert other variants in the same model are unaffected during exclusion
- EKF fallback: configure `InitFitThreshold` low enough to trigger fallback; assert `ekfFallback=true` is set and EKF path activates
- SLO guessing: assert theory-based target is computed once alpha > 0; assert observation-based fallback applies headroom multiplier; assert explicit `SLOTargets` config overrides both
- `GreedyEngine` and `UnlimitedEngine` exact-N: various RequiredCapacity/SpareCapacity/VariantCapacity inputs → assert correct Delta; assert fallback to N=1 when perReplicaCapacity=0; assert existing `V2SaturationAnalyzer` tests still pass

**Integration tests:**
- Step load increase with `QueueingModelAnalyzer` + `GreedyEngine` → assert scale-out within expected cycles with Delta > 1
- Compare convergence speed vs. `V2SaturationAnalyzer` + `GreedyEngine`
- Assert no `ScaleDecision` emitted during bootstrap period (`alpha == 0`)

**Regression:**
- All existing sim tests with `V2SaturationAnalyzer` + `GreedyEngine` must pass unchanged

---

## Open Questions

- **Multi-service-class support:** Single SLO per model for now. Extension point: `SLOTargets` can be extended to a list of `SLOTarget` entries (one per service class), with the analyzer tracking capacity per class and the engine distributing replicas across classes. Deferred.

---

## WVA Alignment Reference

[llm-d Workload Variant Autoscaler (WVA)](https://github.com/llm-d/llm-d-workload-variant-autoscaler) is the reference implementation that `QueueingModelAnalyzer` mirrors. inference-sim serves as a DES-based testing environment for WVA. The table below maps WVA names and concepts to this implementation.

| WVA | inference-sim (this spec) | Notes |
|---|---|---|
| `QueueingModelAnalyzer` | `QueueingModelAnalyzer` | Identical name |
| `QMConfig` | `QMConfig` | Identical name; embedded as `DeploymentConfig.QMConfig` |
| `SLOTarget` | `SLOTarget` | Identical name; `TargetTTFT`, `TargetITL` in float32 ms |
| `SLOTargets map[string]*SLOTarget` | `SLOTargets map[string]SLOTarget` | WVA keys by "namespace/modelID"; inference-sim keys by `ModelID` only |
| `SLOMultiplier` | `SLOMultiplier` | Default 3.0; ρ = 1−1/k ≈ 0.67 |
| `TuningEnabled` | `TuningEnabled` | Guards online parameter learning |
| `LearnedParameters` (Alpha, Beta, Gamma, NIS, Covariance) | `perVariantState.alpha/beta/gamma` + EKF covariance | WVA stores in `ParameterStore`; inference-sim embeds in `perVariantState` alongside the full estimation pipeline |
| `updateVariantParameters()` | `updateVariantParameters()` | Identical name; inference-sim uses three-phase pipeline (NM bootstrap → SWE → EKF) vs. WVA's direct EKF |
| `computeAllVariantCapacities()` | inline in `Analyze()` steps 5–6 | WVA separates into a named helper; same logic |
| `guessSLOFromMetrics()` | `getSLOTarget()` | Same three-priority resolution: explicit config → theory-based → observation-based fallback |
| `aggregateWorkloadMetrics()` | `workloadMetrics` per variant in step 2 | Arrival-rate-weighted averaging across replicas |
| `DefaultMaxBatchSize = 256` | `DefaultMaxBatchSize = 256` | Identical |
| `DefaultMaxQueueSize = 100` | `DefaultMaxQueueSize = 100` | Identical |
| `DefaultFallbackHeadroom = 1.5` | `DefaultFallbackHeadroom = 1.5` | TTFT/ITL multiplier for observation-based fallback |
| Direct EKF via `llm-inferno/kalman-filter` | Three-phase pipeline via `llm-inferno/model-tuner` | inference-sim uses richer estimation; model-tuner bundles NM bootstrap + sliding-window NM + EKF internally |
| Vendors llm-inferno source (older versions) | Imports `llm-inferno/model-tuner`, `llm-inferno/queue-analysis` as Go modules | inference-sim uses current module versions |
| `CostAwareOptimizer` — exact-N scale-up/down (`ceil`/`floor`) | `GreedyEngine` extended with exact-N logic | WVA has a separate optimizer layer; inference-sim extends the existing `Engine` directly. Same arithmetic: `ceil(RequiredCapacity / perReplicaCapacity)` / `floor(SpareCapacity / perReplicaCapacity)` |
| `GreedyByScoreOptimizer` — fair-share GPU across models | *(not in scope)* | WVA's limited-GPU fair-share optimizer; inference-sim defers this |
