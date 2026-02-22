# Design: LatencyModel Interface Extraction

**Status:** Approved
**Date:** 2026-02-21
**Issue:** #241
**Species:** Specification (behavioral contract for a new interface boundary)
**Design Guidelines:** Section 4.2 (Target module map), Section 5.4 (Backend Swap recipe), Section 6.2 (Monolith Method anti-pattern)

## Problem

The latency estimation logic is 6 private methods on `Simulator` with no interface. A boolean branch (`if sim.roofline`) selects between two estimation strategies. Adding a third model (SGLang, TensorRT-LLM, neural surrogate) requires modifying `simulator.go` core — the "backend swap" extension type described in design guidelines Section 5.4 is not possible today.

Compounding the problem: `runningBatchFeatures` (a `RegressionFeatures` struct) is accumulated incrementally inside `makeRunningBatch()` through 12 scattered increment statements. The feature accumulation is coupled to batch formation, not to the latency model.

## Approach

**Phase A only** (this PR): Extract `LatencyModel` interface from existing code. Move all 6 latency methods behind the interface. Existing tests pass unchanged. No new backends.

Phase B (future PR): Implement alternative backends behind the extracted interface.

## Interface Contract

```go
type LatencyModel interface {
    StepTime(batch []*Request) int64
    QueueingTime(req *Request) int64
    OutputTokenProcessingTime() int64
    SchedulingProcessingTime() int64
    PreemptionProcessingTime() int64
}
```

**Key decision:** `StepTime` takes `[]*Request` (the batch), not `RegressionFeatures`. Each implementation extracts its own features from the batch. This means:
- The blackbox implementation internally computes `TotalCacheMissTokens`, `TotalDecodeTokens`, etc. from request states
- The roofline implementation builds its own `StepConfig` from the requests
- A future SGLang model can extract whatever features it needs

**Feature extraction equivalence:** The current code accumulates `runningBatchFeatures` incrementally during `makeRunningBatch()`. Moving to snapshot-based computation from the final batch produces identical results because all features are commutative: sums of cache miss tokens, counts of decode requests, and max of prefill tokens are all order-independent.

## Implementations

### BlackboxLatencyModel

Uses alpha/beta regression coefficients.

```go
type BlackboxLatencyModel struct {
    betaCoeffs  []float64
    alphaCoeffs []float64
}
```

- `StepTime(batch)`: Walks the batch to compute regression features, applies `beta0 + beta1*cacheMissTokens + beta2*decodeTokens`
- `QueueingTime(req)`: `alpha0 + alpha1 * len(req.InputTokens)`
- `OutputTokenProcessingTime()`: returns `alpha2`
- `SchedulingProcessingTime()`: returns `0` (placeholder)
- `PreemptionProcessingTime()`: returns `0` (placeholder)

### RooflineLatencyModel

Analytical FLOPs/bandwidth estimation.

```go
type RooflineLatencyModel struct {
    modelConfig ModelConfig
    hwConfig    HardwareCalib
    tp          int
    alphaCoeffs []float64
}
```

- `StepTime(batch)`: Builds `StepConfig` from batch, delegates to `rooflineStepTime()`
- `QueueingTime(req)`: Same alpha-based estimate (shared with blackbox)
- `OutputTokenProcessingTime()`: Same alpha-based estimate
- `SchedulingProcessingTime()`: returns `0`
- `PreemptionProcessingTime()`: returns `0`

### Factory

```go
func NewLatencyModel(cfg SimConfig) (LatencyModel, error)
```

Returns `RooflineLatencyModel` if `cfg.Roofline`, else `BlackboxLatencyModel`.

## Simulator Field Changes

**Remove from `Simulator`:** `betaCoeffs`, `alphaCoeffs`, `runningBatchFeatures`, `roofline`, `modelConfig`, `hwConfig`, `tp` (7 fields)

**Add to `Simulator`:** `latencyModel LatencyModel` (1 field)

**Remove from `Simulator`:** all 6 `get*` private methods

**Changes in `Step()`:**
- `if sim.roofline { ... } else { ... }` → `sim.latencyModel.StepTime(sim.RunningBatch.Requests)`
- `sim.getOutputTokenProcessingTime()` → `sim.latencyModel.OutputTokenProcessingTime()`

**Changes in `makeRunningBatch()`:**
- Remove all 12 `runningBatchFeatures` increment lines
- `RegressionFeatures` struct stays (used internally by `BlackboxLatencyModel`)

**Changes in `event.go`:**
- `sim.getQueueingTime(req)` → `sim.latencyModel.QueueingTime(req)`

**Changes in `preempt()`:**
- `sim.getPreemptionProcessingTime()` → `sim.latencyModel.PreemptionProcessingTime()`

## File Organization

**New file:** `sim/latency_model.go` — interface, both implementations, factory

**Modified files:**
- `sim/simulator.go` — field removal, method removal, Step()/makeRunningBatch() simplification
- `sim/event.go` — call-site update (1 line)

**Unchanged:**
- `sim/roofline_step.go` — pure functions, already well-tested
- `sim/model_hardware_config.go` — configuration types
- `sim/cluster/` — delegates to Simulator, no changes needed

## Testing Strategy

**New test file:** `sim/latency_model_test.go`

1. **Blackbox StepTime equivalence:** Known batch → verify formula `beta0 + beta1*cacheMissTokens + beta2*decodeTokens`
2. **Roofline StepTime delegation:** Verify correct `StepConfig` construction and delegation to `rooflineStepTime()`
3. **QueueingTime formula:** Both models produce `alpha0 + alpha1*inputLen`
4. **Factory selection:** `Roofline=true` → `RooflineLatencyModel`, `Roofline=false` → `BlackboxLatencyModel` (behavioral, not type assertion)
5. **Feature extraction equivalence:** Verify snapshot-based extraction matches incremental accumulation

**Existing tests:** All must pass unchanged. The golden dataset test exercises Step() end-to-end. No golden dataset regeneration needed.

## Complexity

Medium. ~200-300 lines of new code, ~150 lines removed from simulator.go. Net simplification.
