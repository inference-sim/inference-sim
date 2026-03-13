# Prefill-Decode Interference Model Design

**Issue:** #635
**Parent:** #577 (PD Disaggregation Phase 2)
**Building Block:** 8 (Interference Model)
**Extension Type:** Tier composition (wraps LatencyModel)
**Base branch:** `pd`

## Problem

When prefill and decode phases co-locate on the same instance, they compete for GPU compute and memory bandwidth. Without an interference model, BLIS cannot quantify the break-even point between disaggregation transfer cost and co-location interference cost.

## Approach

A `LatencyModel` wrapper (`InterferenceLatencyModel`) that applies a multiplicative slowdown to `StepTime()` based on batch phase composition. Other LatencyModel methods pass through unchanged. The wrapper lives in `sim/cluster/` — the single-instance simulator remains unaware of interference.

## InterferenceLatencyModel

**File:** `sim/cluster/interference.go`

Implements `sim.LatencyModel`. Wraps an inner model with two configurable interference factors.

**Fields:**
- `inner sim.LatencyModel` — the base latency model
- `prefillInterference float64` — slowdown factor for prefill-dominant batches (minority is decode)
- `decodeInterference float64` — slowdown factor for decode-dominant batches (minority is prefill)
- `lastMultiplier float64` — most recently applied multiplier (BC-P2-12), initialized to 1.0

**StepTime algorithm:**
1. Classify each request: prefill if `ProgressIndex < int64(len(InputTokens))`, decode otherwise. (Note: `ProgressIndex` is `int64`, `len()` returns `int` — explicit cast required.)
2. If batch is empty or phase-pure (minority count = 0): multiplier = 1.0.
3. Determine majority phase. minority_count = min(prefill_count, decode_count).
4. Select factor: majority prefill → `prefillInterference`; majority decode → `decodeInterference`; exact tie → max of both factors (conservative).
5. `multiplier = 1.0 + factor * (float64(minority_count) / float64(total_count))`
6. Store multiplier in `lastMultiplier`.
7. Return `int64(math.Round(float64(inner.StepTime(batch)) * multiplier))`, clamped to minimum of 1 (INV-3).

**Effective range:** Since minority_count / total_count ranges from 0 to 0.5 (minority is always <= half), a factor of 1.0 produces at most a 50% slowdown when the batch is evenly split.

**Pass-through methods:** `QueueingTime`, `OutputTokenProcessingTime`, `PostDecodeFixedOverhead` delegate directly to inner.

**Constructor:** `NewInterferenceLatencyModel(inner, prefillFactor, decodeFactor)` validates:
- `inner != nil`
- Both factors >= 0
- Both factors are finite (not NaN or Inf)

Returns `(*InterferenceLatencyModel, error)`. Returns concrete type so callers can access `LastAppliedMultiplier()`.

**Accessor:** `LastAppliedMultiplier() float64` returns `lastMultiplier` (initialized to 1.0).

## Configuration

**DeploymentConfig fields** (in `sim/cluster/deployment.go`):
```
PDInterferencePrefill float64  // default 0, interference for prefill-dominant batches
PDInterferenceDecode  float64  // default 0, interference for decode-dominant batches
```

**CLI flags** (in `cmd/root.go`):
- `--pd-interference-prefill` (float64, default 0)
- `--pd-interference-decode` (float64, default 0)

## Injection Point

**File:** `sim/cluster/instance.go`

To avoid changing the `NewInstanceSimulator(id, cfg)` public signature (which has 19+ call sites across test files — R4), the injection uses an unexported constructor variant:

```go
// Public API unchanged — all existing call sites (tests, etc.) keep working.
func NewInstanceSimulator(id InstanceID, cfg sim.SimConfig) *InstanceSimulator {
    return newInstanceSimulatorCore(id, cfg, 0, 0)
}

// Unexported: used only by cluster.go with actual interference factors.
func newInstanceSimulatorCore(id InstanceID, cfg sim.SimConfig,
    prefillInterference, decodeInterference float64) *InstanceSimulator {
    kvStore := kv.NewKVStore(cfg.KVCacheConfig)
    latencyModel, err := latency.NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
    // ... error handling ...
    if prefillInterference > 0 || decodeInterference > 0 {
        latencyModel, err = NewInterferenceLatencyModel(latencyModel, prefillInterference, decodeInterference)
        // ... error handling ...
    }
    s, err := sim.NewSimulator(cfg, kvStore, latencyModel)
    // ...
}
```

**Call site update** in `sim/cluster/cluster.go` `NewClusterSimulator` (line ~90): the single internal call switches from `NewInstanceSimulator` to `newInstanceSimulatorCore` with the factors from `config.PDInterferencePrefill` / `config.PDInterferenceDecode`.

## Behavioral Guarantees

- **BC-P2-9:** When interference factors are 0 (default), step time is identical to base latency model.
- **BC-P2-10:** When batch is phase-pure (all prefill or all decode), multiplier = 1.0.
- **BC-P2-11 / INV-P2-3:** Multiplier >= 1.0 always (interference never speeds up execution).
- **BC-P2-12:** `LastAppliedMultiplier()` records the applied multiplier per StepTime call.

## New Invariant

**INV-P2-3 (Interference monotonicity):** Multiplier >= 1.0 (interference never speeds up execution). Multiplier = 1.0 when batch is phase-pure. This is in the P2 numbering series (following INV-P2-1 pool-config consistency and INV-P2-2 transfer fair-share), not the PD series (INV-PD-1 through INV-PD-5).

## Testing Strategy

- **Unit tests** (`sim/cluster/interference_test.go`): Table-driven tests covering:
  - Factors = 0 → identity (BC-P2-9)
  - Phase-pure batch (all prefill, all decode) → multiplier 1.0 (BC-P2-10)
  - Mixed batch (e.g., 3 prefill + 1 decode, factor 0.5) → correct multiplier with linear interpolation
  - Tied batch (2 prefill + 2 decode) → uses max factor
  - Empty batch → multiplier 1.0
  - INV-P2-3 invariant: multiplier >= 1.0 for all factor/composition combinations (property test across ranges)
  - Constructor rejects negative, NaN, Inf factors (R3)
  - LastAppliedMultiplier returns 1.0 before first call
- **Integration test**: 2-instance cluster with 10 requests (mixed prefill/decode batch composition), interference factor 0.5. Assert that total simulation time exceeds the zero-interference baseline and that per-request step times are larger.

## Files Changed

| File | Change |
|------|--------|
| `sim/cluster/interference.go` | New: InterferenceLatencyModel wrapper |
| `sim/cluster/interference_test.go` | New: Unit + invariant + integration tests |
| `sim/cluster/instance.go` | Modified: Extract `newInstanceSimulatorCore` with interference params; `NewInstanceSimulator` delegates with 0,0 |
| `sim/cluster/cluster.go` | Modified: Call `newInstanceSimulatorCore` with interference config |
| `sim/cluster/deployment.go` | Modified: Add PDInterferencePrefill/Decode fields |
| `cmd/root.go` | Modified: Add CLI flags, wire to DeploymentConfig |
| `CLAUDE.md` | Modified: INV-P2-3, CLI flags, file organization |
| `docs/contributing/templates/design-guidelines.md` | Modified: Section 4.2 module map — add interference model entry |
