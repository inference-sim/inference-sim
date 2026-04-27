# Windowed Statistical Collector for Autoscaler Pipeline

**Status:** Draft  
**Date:** 2026-04-23  
**Feature area:** `sim/cluster` — autoscaler pipeline  
**Related design:** `docs/plans/2026-04-01-phase1c-autoscaling-design.md`

---

## 1. Problem

The autoscaler's `DefaultCollector` produces point-in-time snapshots of replica state at each tick. Two gaps result:

1. **No temporal smoothing.** A single momentary spike in `QueueDepth` or `KVUtilization` can trigger a scale-up decision; a momentary dip can trigger scale-down. The only damping mechanism today is the cooldown window — a coarse gate, not a signal smoother.

2. **No statistical richness.** The `Analyzer` receives only flat scalars. It cannot distinguish steady-state high utilization from highly variable load. Future evolved analyzers need distributions (mean, stddev, P90/P95) to reason about variability, not just level.

Additionally, `ReplicaMetrics.TTFT` and `DispatchRate` are permanently zero — no mechanism computes them. `ITL` is absent entirely.

---

## 2. Goals

1. Add a `WindowedCollector` that maintains a per-replica circular buffer of the last N tick snapshots, exposing the full history in `ModelSignals.Window` for distribution-aware Analyzers.
2. Populate `ReplicaMetrics.TTFT`, `DispatchRate`, and a new `ITL` field via per-tick deltas of per-instance cumulative completion metrics.
3. Keep `DefaultCollector` and all existing `Analyzer` implementations unchanged — `Window` is ignored when nil.
4. Keep `CollectorWindowSize=0` (default) as a zero-behavioral-change config value.

---

## 3. Design

### 3.1 Data model changes

**`ReplicaMetrics`** (`autoscaler.go`) — two fields populated, one added:

```go
type ReplicaMetrics struct {
    // ... existing fields unchanged ...
    TTFT         float64 // μs — mean TTFT of requests completing this tick; 0 if none
    DispatchRate float64 // req/s — completions this tick / tick interval; 0 if none
    ITL          float64 // μs — mean ITL of requests completing this tick; 0 if none (NEW)
}
```

`TTFT` and `DispatchRate` were declared but permanently zero. `WindowedCollector` populates them from per-tick deltas. `ITL` is new. The comment *"zero until QueueingModelAnalyzer"* is removed from both existing fields — they are now populated by the collector when `WindowedCollector` is used.

**`ModelSignals`** (`autoscaler.go`) — one field added:

```go
type ModelSignals struct {
    ModelID  string
    Replicas []ReplicaMetrics   // current-tick snapshot (unchanged)
    Window   [][]ReplicaMetrics // [tick][replica]; nil when DefaultCollector is used; oldest tick first
}
```

`Window[0]` is the oldest retained tick; `Window[len-1]` is the current tick (same data as `Replicas`). Analyzers that want distributions iterate `Window`. Analyzers that don't (including `V2SaturationAnalyzer`) ignore it entirely.

**`DeploymentConfig`** (`deployment.go`) — one field added:

```go
CollectorWindowSize int `yaml:"collector_window_size,omitempty"`
// ticks to retain; 0 or 1 = DefaultCollector (point-in-time, no behavioral change)
```

### 3.2 WindowedCollector

`WindowedCollector` implements `Collector` as a stateful decorator over `DefaultCollector`.

```go
type WindowedCollector struct {
    inner          Collector
    windowSize     int
    tickIntervalUs float64
    metricsQueryFn map[string]func() instanceMetricSnapshot

    // circular buffer per model
    buffer  map[string][][]ReplicaMetrics
    head    map[string]int
    filled  map[string]bool

    // TTFT/ITL delta baselines per instance
    prevCompleted map[string]int64
    prevTTFTSum   map[string]float64
    prevITLSum    map[string]float64
}
```

`instanceMetricSnapshot` is an unexported value type: `{completedCount int64, ttftSumUs float64, itlSumUs float64}`.

**`Collect(state)` per tick:**

1. Call `inner.Collect(state)` to get current-tick `[]ModelSignals` with scalar `Replicas`.
2. For each replica: call `metricsQueryFn[instanceID]()`, compute deltas vs `prev*` maps, populate `TTFT`, `DispatchRate`, `ITL`. Update `prev*` maps.
3. Append enriched `Replicas` to the circular buffer for that model at index `head[modelID] % windowSize`. Advance `head`.
4. Set `ModelSignals.Window` to buffer contents in oldest-first order (reordered into a fresh `[][]ReplicaMetrics`).
5. Return enriched `[]ModelSignals`.

**`AddInstance(id, inst)`** mirrors `CachedSnapshotProvider.AddInstance`: adds a new closure to `metricsQueryFn` when a deferred instance comes online mid-run.

### 3.3 Pipeline wiring

In `cluster.go`, where the default pipeline is constructed:

```go
var collector Collector
if config.CollectorWindowSize >= 2 {
    collector = NewWindowedCollector(
        &DefaultCollector{},
        config.CollectorWindowSize,
        config.ModelAutoscalerIntervalUs,
        buildMetricsQueryFn(cs),
    )
} else {
    collector = &DefaultCollector{}
}
```

`buildMetricsQueryFn` is a private helper on `ClusterSimulator` that iterates `cs.instances` and captures each `InstanceSimulator` in a closure returning `instanceMetricSnapshot` from `inst.Metrics()`. It follows the same pattern as the `cacheQueryFn` builder (`cluster.go:458`).

When `NodeReadyEvent` fires for a deferred instance, `cs.autoscaler.collector` is type-asserted to `*WindowedCollector` (when non-nil) and `AddInstance` is called alongside `CachedSnapshotProvider.AddInstance`.

### 3.4 Edge cases

| Scenario | Behavior |
|---|---|
| New instance (not yet in `prev*`) | Delta = 0 for TTFT/ITL/DispatchRate on first tick. Baseline set. No false spike. |
| Instance disappearing (draining) | `metricsQueryFn` returns last-known values; delta is 0. Old buffer entries remain as historical data. |
| Zero completions this tick | `TTFT`, `ITL`, `DispatchRate` all 0.0. Analyzer guards against zero before dividing (existing contract). |
| Partial window (first N-1 ticks) | `Window` has fewer than `WindowSize` rows. `filled[modelID]` distinguishes partial from full window. |
| `CollectorWindowSize=1` | `Window` has exactly 1 row matching `Replicas`. Equivalent to point-in-time. |
| `CollectorWindowSize=0` (default) | `DefaultCollector` used; `Window` is nil; no behavioral change. |

---

## 4. Configuration

All autoscaler config lives flat in `DeploymentConfig`. The new field follows the same pattern:

```yaml
model_autoscaler_interval_us: 10000000   # 10s tick
collector_window_size: 5                  # last 5 ticks = 50s effective window
scale_up_cooldown_us: 60000000
scale_down_cooldown_us: 120000000
```

Effective window duration = `collector_window_size × model_autoscaler_interval_us`.

---

## 5. Testing

### Unit: `windowed_collector_test.go`

Table-driven tests covering:

- Buffer growth: `Window` has 1 row after tick 1, N rows after tick N, N rows after tick N+1 (oldest evicted).
- TTFT/ITL/DispatchRate delta: fake `metricsQueryFn` returns controlled cumulative values; assert per-tick deltas across multiple ticks including zero-completion ticks.
- New instance mid-run: first tick produces zero deltas; subsequent ticks produce correct deltas.
- `WindowSize=1`: `Window` has exactly 1 row matching `Replicas`.
- Determinism (INV-6): same call sequence with same fake state produces identical output.

### Unit: `default_collector_test.go`

No changes. `DefaultCollector` is unchanged.

### Integration: `autoscaler_test.go`

- End-to-end tick sequence with `CollectorWindowSize=3`: `ModelSignals.Window` depth grows correctly; `V2SaturationAnalyzer` still produces correct decisions (backward-compat verification).
- `CollectorWindowSize=0` (default): `Window` is nil; existing tests unaffected.

### Invariant tests

- `Window[len-1]` always equals `Replicas` (current tick consistency).
- `ReplicaMetrics.TTFT >= 0`, `DispatchRate >= 0`, `ITL >= 0` always.
- `len(Window) <= CollectorWindowSize` always.

---

## 6. Out of scope

- Changes to `Analyzer`, `Engine`, or `Actuator` interfaces.
- A distribution-aware Analyzer implementation (separate feature; this PR provides the data).
- CLI flags — `collector_window_size` is bundle YAML only.
- `QueueingModelAnalyzer` — deferred; this PR unblocks it by populating `TTFT` and `DispatchRate`.
