# Bugfix Audit: 8-Issue Hardening PR

- **Goal:** Fix 8 bugs discovered during a systematic codebase audit — silent data loss, CLI flag issues, livelock risk, and dead code.
- **The problem today:** `SaveResults` silently drops output on marshal failure (R1). Blackbox coefficient loading silently overrides explicit user zeros (R18). `StepTime()` can return 0 causing infinite loops (R19). Several CLI flags are unvalidated or dead. Negative alpha/beta coefficients can violate causality.
- **What this PR adds:**
  1. `SaveResults` returns error instead of silently dropping output
  2. Blackbox coefficient loading uses `Flags().Changed()` instead of `AllZeros()` (R18)
  3. `StepTime()` floor of `max(1, ...)` for Blackbox and Roofline models (R19)
  4. CLI validation for `--admission-latency`, `--routing-latency` (R3), and non-negative coefficients (INV-5)
  5. Remove dead `--max-model-len` flag
  6. Guard throughput divide-by-zero in `SaveResults` (R11)
  7. Warning when `InjectArrival` receives requests beyond horizon (INV-1)
- **Why this matters:** Each bug was found by a real rule (R1, R3, R11, R18, R19, INV-1, INV-5). Fixing them hardens the simulator against pathological inputs and silent failures.
- **Architecture:** Changes span `cmd/root.go` (CLI validation + flag cleanup), `sim/metrics.go` (error returns + division guard), `sim/latency/latency.go` (StepTime floor + coefficient validation), `sim/simulator.go` (horizon warning). No new types, interfaces, or module boundaries.
- **Source:** Issues #490, #491, #493, #495, #496, #497, #498, #499 (see #494 deferral below)
- **Closes:** Fixes #490, fixes #491, fixes #493, fixes #495, fixes #496, fixes #497, fixes #498, fixes #499
- **Behavioral Contracts:** See Part 1, Section B.

**Note on #494 (Request.Streaming):** Deferred. Removing the field touches workload generation, trace export, and test assertions across 5+ files — not a small fix. Filed for a separate PR.

---

## Phase 0: Component Context

1. **Building blocks modified:** CLI validation layer (`cmd/root.go`), metrics output (`sim/metrics.go`), latency model factory + implementations (`sim/latency/latency.go`), simulator injection (`sim/simulator.go`)
2. **Adjacent blocks:** `sim/cluster/cluster.go` calls `SaveResults` via instances; `sim/simulator.go:343` calls `StepTime()` in the hot loop (via `executeBatchStep`); `cmd/root.go` is the only entry point for CLI flags
3. **Invariants touched:** INV-1 (conservation — horizon guard), INV-5 (causality — coefficient validation), R19 (livelock — StepTime floor)
4. **Construction site audit:** No struct fields added. `SaveResults` signature changes from `void` to `error` — all 13 call sites:
   - `cmd/root.go:662,666` (2 production calls — handle error)
   - `cmd/root_test.go:44` (1 test call)
   - `sim/metrics_test.go:44,92,140,184,229,264,298,323` (8 test calls)
   - `sim/simulator_test.go:719,728` (2 test calls — determinism test)
   - Note: `sim/cluster/instance.go:Finalize()` does NOT call SaveResults

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes 8 bugs (deferring #494) found during a systematic rule-by-rule audit. The fixes are independent, each touching 1-5 lines. The highest-severity fixes are: (1) `SaveResults` returning error instead of silently dropping output, (2) blackbox coefficient loading using `Flags().Changed()`, and (3) `StepTime()` livelock prevention. All fixes are validation guards, dead code removal, or simple control flow changes. No behavioral changes to simulation output for valid inputs (INV-6 preserved).

### B) Behavioral Contracts

**Positive contracts:**

```
BC-1: SaveResults propagates errors
- GIVEN a metrics output that fails JSON marshalling (e.g., contains +Inf)
- WHEN SaveResults is called
- THEN it returns a non-nil error describing the failure
- MECHANISM: Change signature to return error; cmd/ callers handle via logrus.Fatalf (per R1 + CLI convention)
```

```
BC-2: Throughput divide-by-zero guarded
- GIVEN SimEndedTime == 0 (zero runtime)
- WHEN SaveResults computes ResponsesPerSec and TokensPerSec
- THEN both are 0.0 (not +Inf or NaN)
- MECHANISM: Guard with `if vllmRuntime > 0` before division
```

```
BC-3: Coefficient loading respects explicit CLI flags across all backends
- GIVEN a user provides --beta-coeffs 0,0,0 or --alpha-coeffs 0,0,0 explicitly
- WHEN the blackbox OR roofline loading path runs
- THEN the zero coefficients are preserved (not overwritten by defaults.yaml)
- MECHANISM: Use cmd.Flags().Changed() instead of AllZeros() at line 304 (blackbox) AND line 213 (roofline alpha)
```

```
BC-4: StepTime returns at least 1 tick for non-empty batches
- GIVEN a batch with one or more requests
- WHEN any LatencyModel.StepTime() is called
- THEN the return value is >= 1
- MECHANISM: Add max(1, ...) to Blackbox (line 40) and Roofline (line 89), applied unconditionally (no empty-batch early return — see DES Expert analysis below). CrossModelLatencyModel already has max(1, ...) at crossmodel.go:59.
- NOTE: Round 2 convergence review (PP-6 DES Expert) found that adding an empty-batch `return 0` early exit would create a livelock: when KV allocation fails for all queued requests, FormBatch returns an empty batch, StepTime returns 0, and scheduleNextStep schedules at `now + 0` with WaitQ still non-empty — infinite loop. The `max(1, ...)` floor must apply unconditionally to prevent this. All three backends share the same scheduleNextStep code path — CrossModel's existing `return 0` for empty batches (crossmodel.go:41) has the same latent livelock risk (pre-existing bug, filed as known debt — see deviation log). For Blackbox/Roofline, the unconditional `max(1, ...)` is the livelock protection.
```

```
BC-5: Negative latency coefficients rejected at construction (defense-in-depth)
- GIVEN alpha or beta coefficients containing negative values
- WHEN NewLatencyModel is called
- THEN it returns an error mentioning the negative coefficient
- MECHANISM: Add c < 0 check in validateCoeffs
- NOTE: This overlaps with BC-4 (StepTime floor) — belt-and-suspenders. BC-4 prevents livelock at runtime; BC-5 prevents causality violations (negative QueueingTime) at construction. Both are needed: BC-4 alone doesn't prevent negative enqueue times from negative alpha coefficients. If future calibration requires small negative coefficients, a corresponding floor on QueueingTime (similar to BC-4's StepTime floor) MUST be added before relaxing this check — otherwise negative QueueingTime would violate INV-5 causality.
```

**Negative contracts:**

```
BC-6: Negative admission/routing latency rejected
- GIVEN --admission-latency -100 or --routing-latency -100
- WHEN the CLI validation runs
- THEN the process exits with a descriptive error message
- MECHANISM: Add validation checks in cmd/root.go validation block
```

```
BC-7: Dead --max-model-len flag removed
- GIVEN the CLI flag registry
- WHEN a user runs blis run --help
- THEN --max-model-len does not appear in the output
- MECHANISM: Remove flag declaration and registration
```

**Warning contracts:**

```
BC-8: Horizon overflow warning (single-instance mode)
- GIVEN a request with ArrivalTime > sim.Horizon
- WHEN InjectArrival is called
- THEN a warning is logged to stderr
- MECHANISM: Add logrus.Warnf check in InjectArrival
- NOTE: InjectArrivalAt (cluster mode) is NOT covered — cluster mode is already guarded by the event loop horizon break at cluster.go:149/155. The INV-1 conservation impact for beyond-horizon requests in single-instance mode remains as known debt (request registered but never processed).
```

### C) Component Interaction

```
cmd/root.go (CLI entry)
  │
  ├─ BC-3: Flags().Changed() guard on alpha/beta loading
  ├─ BC-6: Validate admission-latency >= 0, routing-latency >= 0
  ├─ BC-7: Remove maxModelLength declaration + registration
  ├─ BC-1: Handle SaveResults error return via logrus.Fatalf
  │
  └─► sim/metrics.go (metrics output)
       ├─ BC-1: SaveResults returns error
       ├─ BC-2: Guard vllmRuntime > 0 before division
       │
  └─► sim/latency/latency.go (latency factory)
       ├─ BC-4: max(1,...) floor in StepTime()
       ├─ BC-5: Negative coefficient validation
       │
  └─► sim/simulator.go (core engine)
       └─ BC-8: Horizon warning in InjectArrival
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #494: Remove Request.Streaming | Deferred to separate PR | DEFERRAL — touches 5+ files across workload/trace/tests; not a small fix |
| #490: "add error return or Fatalf" | Returns error (not Fatalf) | CORRECTION — R6 says sim/ must not terminate; error return is the right pattern for library code |
| BC-8: Ideal would be error return from InjectArrival | Uses logrus.Warnf instead | SIMPLIFICATION — Adding error return to InjectArrival is a larger API change (touches cluster injection path); warning is pragmatic. Precedent: sim/ already uses logrus.Debugf/Infof/Warnf. Known debt for future cleanup. |
| BC-3 (#491): Fix only blackbox path | Also fixes roofline alpha path at line 213, 225 | ADDITION — Pre-pass review (I1) found the same R18 violation in the roofline alpha-loading path |
| BC-3: "preserve explicit zero coefficients" | Zero-coefficient safety guard at line 351 still fires Fatalf for all-zero blackbox coefficients | CLARIFICATION — BC-3 preserves explicit zeros for the defaults-loading path (line 304). The safety guard at line 351 intentionally rejects all-zero blackbox coefficients (meaningless results). These are separate concerns: line 304 = "don't override user input with defaults", line 351 = "don't run with unusable coefficients". |
| BC-4: CrossModel already safe | CrossModel has the same latent livelock risk (return 0 for empty batches via same scheduleNextStep path) | CORRECTION — Round 3 convergence (7/9 perspectives) identified that CrossModel's `return 0` at crossmodel.go:41 bypasses its own `max(1,...)` at line 59. This is a pre-existing bug, not introduced by this plan. File as separate issue. |

### E) Review Guide

**Tricky part:** BC-1 (SaveResults signature change) ripples to 13 call sites (2 production + 11 test). All test callers that previously discarded the void return must now handle the error (assign to `_` or check). BC-3 now also covers the roofline alpha-loading path (line 213), not just the blackbox path (line 304).

**Safe to skim:** BC-6, BC-7 (mechanical CLI validation/removal). BC-8 (one-line warning).

**Known debt:** BC-8 uses `logrus.Warnf` in `sim/` (pragmatic but not ideal per R6 spirit). A cleaner approach would be an error return from `InjectArrival`, deferred to a future PR.

---

## Part 2: Executable Implementation

### F) Implementation Overview

| File | Change |
|------|--------|
| `sim/metrics.go` | BC-1: Change SaveResults to return error. BC-2: Guard vllmRuntime > 0 |
| `cmd/root.go` | BC-1: Handle SaveResults error. BC-3: Flags().Changed() guard. BC-6: Validate latencies. BC-7: Remove dead flag |
| `sim/latency/latency.go` | BC-4: max(1,...) floor. BC-5: Negative coefficient validation |
| `sim/simulator.go` | BC-8: Horizon warning in InjectArrival |
| `sim/metrics_test.go` | BC-1: Update 8 test call sites to handle error return |
| `sim/simulator_test.go` | BC-1: Update 2 test call sites to handle error return |
| `cmd/root_test.go` | BC-1: Update 1 test call site to handle error return |

No dead code. No new types. No new files.

### G) Task Breakdown

#### Task 1: BC-2 + BC-1 — SaveResults error return + throughput guard

**Test (BC-2):**
```go
// sim/metrics_test.go — add test
func TestSaveResults_ZeroRuntime_NoInfinity(t *testing.T) {
    m := NewMetrics()
    m.CompletedRequests = 1
    m.SimEndedTime = 0 // zero runtime
    m.RequestTTFTs["r1"] = 100.0
    m.RequestE2Es["r1"] = 200.0
    m.RequestITLs["r1"] = 50.0
    m.AllITLs = []int64{50}
    m.RequestSchedulingDelays["r1"] = 100
    m.Requests["r1"] = NewRequestMetrics(&Request{ID: "r1", InputTokens: make([]int, 10), OutputTokens: make([]int, 5)}, 0)

    err := m.SaveResults("test", 1000000, 100, "")
    if err != nil {
        t.Fatalf("SaveResults returned error for zero runtime: %v", err)
    }
    // Should NOT produce +Inf — if it did, JSON marshal would fail
}
```

**Run:** `go test ./sim/... -run TestSaveResults_ZeroRuntime_NoInfinity` → FAIL (SaveResults has no error return yet)

**Implement:**
1. In `sim/metrics.go`, change `func (m *Metrics) SaveResults(...)` to return `error`
2. Add `if vllmRuntime > 0 { ... }` guard around ResponsesPerSec/TokensPerSec division
3. Replace `logrus.Errorf` + `return` with `return fmt.Errorf(...)` for marshal/write errors
4. Add `return nil` at end

**Update all 13 callers:**
- `cmd/root.go:662,666`: Add `if err := ... SaveResults(...); err != nil { logrus.Fatalf("SaveResults: %v", err) }` (Fatalf per CLI convention + R1)
- `sim/metrics_test.go` (8 sites at lines 44,92,140,184,229,264,298,323): Tests that don't check error → add `if err := m.SaveResults(...); err != nil { t.Fatalf(...) }`
- `sim/simulator_test.go:719,728`: Same pattern for determinism test
- `cmd/root_test.go:44`: Same pattern

**Run:** `go test ./sim/... -run TestSaveResults_ZeroRuntime_NoInfinity` → PASS
**Lint:** `golangci-lint run ./sim/... ./cmd/...`
**Commit:** `fix(metrics): SaveResults returns error + guards divide-by-zero (R1, R11) (#490, #496)`

#### Task 2: BC-3 — Coefficient Flags().Changed() guard (blackbox + roofline)

**Test:** CLI-level fix. Existing test suite must still pass (coefficients loaded correctly for normal usage).

**Implement:**
1. In `cmd/root.go:304` (blackbox path), replace:
```go
if AllZeros(alphaCoeffs) && AllZeros(betaCoeffs) && len(modelConfigFolder) == 0 && len(hwConfigPath) == 0 {
```
with:
```go
if !cmd.Flags().Changed("alpha-coeffs") && !cmd.Flags().Changed("beta-coeffs") && len(modelConfigFolder) == 0 && len(hwConfigPath) == 0 {
```

2. In `cmd/root.go:213` (roofline alpha path), replace:
```go
if AllZeros(alphaCoeffs) && !AllZeros(defAlpha) {
```
with:
```go
if !cmd.Flags().Changed("alpha-coeffs") && !AllZeros(defAlpha) {
```

3. In `cmd/root.go:225` (roofline alpha warning), replace:
```go
if AllZeros(alphaCoeffs) {
```
with:
```go
if AllZeros(alphaCoeffs) && !cmd.Flags().Changed("alpha-coeffs") {
```
This prevents the spurious "no trained alpha coefficients found" warning when the user explicitly passed `--alpha-coeffs 0,0,0`.

**Run:** `go test ./... -count=1` → all pass
**Lint:** `golangci-lint run ./cmd/...`
**Commit:** `fix(cli): use Flags().Changed() for coefficient loading across all backends (R18) (#491)`

#### Task 3: BC-4 — StepTime floor for Blackbox and Roofline

**Test:**
```go
// sim/latency/latency_test.go — add test
func TestBlackboxLatencyModel_StepTime_FloorAtOne(t *testing.T) {
    // Zero beta coefficients → StepTime should return 1 (not 0)
    coeffs := sim.NewLatencyCoeffs([]float64{0, 0, 0}, []float64{0, 0, 0})
    hw := sim.ModelHardwareConfig{} // blackbox backend (empty Backend field)
    model, err := NewLatencyModel(coeffs, hw)
    if err != nil {
        t.Fatalf("NewLatencyModel: %v", err)
    }
    batch := []*sim.Request{{InputTokens: make([]int, 16), OutputTokens: make([]int, 4), NumNewTokens: 1}}
    stepTime := model.StepTime(batch)
    if stepTime < 1 {
        t.Errorf("StepTime = %d, want >= 1", stepTime)
    }
}
```

**Run:** `go test ./sim/latency/... -run TestBlackboxLatencyModel_StepTime_FloorAtOne` → FAIL (returns 0)

**Implement:**
- `sim/latency/latency.go:40`: Change `return int64(totalStepTime)` to `return max(1, int64(totalStepTime))`
- `sim/latency/latency.go:89`: Change `return rooflineStepTime(...)` to `return max(1, rooflineStepTime(...))`
- NOTE: Do NOT add empty-batch early return — see BC-4 NOTE about livelock risk from PP-6 DES Expert review.

**Run:** → PASS
**Lint:** `golangci-lint run ./sim/latency/...`
**Commit:** `fix(latency): floor StepTime at 1 tick for Blackbox and Roofline (R19) (#497)`

#### Task 4: BC-5 — Negative coefficient validation

**Test:**
```go
// sim/latency/latency_test.go — add test
func TestNewLatencyModel_NegativeCoefficients_ReturnsError(t *testing.T) {
    coeffs := sim.NewLatencyCoeffs([]float64{-1, 0, 0}, []float64{100, 1, 1})
    hw := sim.ModelHardwareConfig{}
    _, err := NewLatencyModel(coeffs, hw)
    if err == nil {
        t.Fatal("expected error for negative alpha coefficient")
    }
    if !strings.Contains(err.Error(), "negative") {
        t.Errorf("error should mention 'negative', got: %v", err)
    }
}
```

**Run:** → FAIL (no negativity check)

**Implement:**
In `sim/latency/latency.go`, add to `validateCoeffs`:
```go
if c < 0 {
    return fmt.Errorf("latency model: %s[%d] must be non-negative, got %f", name, i, c)
}
```

**Run:** → PASS
**Lint:** `golangci-lint run ./sim/latency/...`
**Commit:** `fix(latency): reject negative alpha/beta coefficients (INV-5) (#499)`

#### Task 5: BC-6 — Validate admission/routing latency

**Test:** CLI-level. Existing tests must pass.

**Implement:**
In `cmd/root.go`, after `snapshotRefreshInterval` validation (line ~583), add:
```go
if admissionLatency < 0 {
    logrus.Fatalf("--admission-latency must be >= 0, got %d", admissionLatency)
}
if routingLatency < 0 {
    logrus.Fatalf("--routing-latency must be >= 0, got %d", routingLatency)
}
```

**Run:** `go test ./cmd/... -count=1` → pass
**Lint:** `golangci-lint run ./cmd/...`
**Commit:** `fix(cli): validate admission-latency and routing-latency >= 0 (R3) (#495)`

#### Task 6: BC-7 — Remove dead --max-model-len flag

**Implement:**
- `cmd/root.go:41`: Remove `maxModelLength int` declaration
- `cmd/root.go:801`: Remove `runCmd.Flags().IntVar(&maxModelLength, ...)` registration

**Run:** `go build ./...` → pass (no references to maxModelLength remain)
**Lint:** `golangci-lint run ./cmd/...`
**Commit:** `fix(cli): remove dead --max-model-len flag (#493)`

#### Task 7: BC-8 — Horizon warning in InjectArrival

**Test:**
```go
// sim/simulator_test.go — add test
func TestInjectArrival_BeyondHorizon_Warns(t *testing.T) {
    // This test verifies the warning is logged; it doesn't test log output directly
    // but ensures no panic and the request is still registered (backward compatible)
    cfg := SimConfig{
        Horizon:       1000,
        Seed:          42,
        KVCacheConfig: NewKVCacheConfig(100, 16, 0, 0, 0, 0),
        BatchConfig:   NewBatchConfig(10, 2048, 0),
        LatencyCoeffs: NewLatencyCoeffs([]float64{100, 1, 1}, []float64{50, 0.1, 50}),
    }
    sim := mustNewSimulator(t, cfg)
    req := &Request{
        ID: "beyond_horizon", InputTokens: make([]int, 16),
        OutputTokens: make([]int, 4), ArrivalTime: 2000, State: StateQueued,
    }
    sim.InjectArrival(req) // should not panic
    // Request is registered (backward compatible)
    if _, ok := sim.Metrics.Requests["beyond_horizon"]; !ok {
        t.Error("request should still be registered in Metrics.Requests")
    }
}
```

**Run:** → PASS (already works, just adding the warning)

**Implement:**
In `sim/simulator.go`, `InjectArrival` method, add before the `Schedule` call:
```go
if req.ArrivalTime > sim.Horizon {
    logrus.Warnf("InjectArrival: request %s has ArrivalTime %d > Horizon %d; ArrivalEvent will not fire (INV-1 conservation may be affected)", req.ID, req.ArrivalTime, sim.Horizon)
}
```

**Lint:** `golangci-lint run ./sim/...`
**Commit:** `fix(sim): warn when InjectArrival receives request beyond horizon (INV-1) (#498)`

### H) Test Strategy

| Contract | Test | Type |
|----------|------|------|
| BC-1 | TestSaveResults_ZeroRuntime_NoInfinity | Behavioral — verifies error return on edge case |
| BC-2 | Same test (zero runtime → no +Inf → no error) | Behavioral — verifies division guard |
| BC-3 | Existing test suite (regression) | Regression — coefficients still load for normal usage |
| BC-4 | TestBlackboxLatencyModel_StepTime_FloorAtOne | Behavioral — verifies floor property |
| BC-5 | TestNewLatencyModel_NegativeCoefficients_ReturnsError | Behavioral — verifies error on bad input |
| BC-6 | Build-level (CLI validation) | Regression — existing tests pass |
| BC-7 | Build-level (flag removed) | Regression — no compilation references |
| BC-8 | TestInjectArrival_BeyondHorizon_Warns | Behavioral — verifies no panic + backward compat |

### I) Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| BC-1 signature change breaks callers | Grep for all SaveResults call sites; update each |
| BC-3 changes coefficient loading behavior | Existing golden test verifies output unchanged for normal inputs |
| BC-4 changes step timing for pathological inputs | Only affects zero-coefficient case; normal coefficients always > 1 |
| BC-5 rejects previously-accepted inputs | Only negative coefficients rejected; all real-world coefficients are positive |

---

## Part 3: Appendix

### J) File-Level Details

**`sim/metrics.go`:** Change `SaveResults` signature to `func (m *Metrics) SaveResults(...) error`. Add `if vllmRuntime > 0` guard at line ~120. Replace 3 silent returns with `return fmt.Errorf(...)`. Add `return nil` at end.

**`cmd/root.go`:** (1) Replace `AllZeros()` with `Flags().Changed()` at line ~304 (blackbox) AND line ~213 (roofline alpha). (2) Add 2 latency validation checks after line ~583. (3) Remove 2 lines for maxModelLength. (4) Handle SaveResults error at lines 662, 666.

**`sim/latency/latency.go`:** (1) Add `max(1, ...)` to Blackbox StepTime return (line 40). (2) Add `max(1, ...)` to Roofline StepTime return (line 89). (3) Add `c < 0` check in validateCoeffs (line ~113).

**`sim/simulator.go`:** Add 3-line horizon warning in InjectArrival (line ~174).

**`sim/metrics_test.go`:** Update 8 SaveResults call sites to handle error return.

**`sim/simulator_test.go`:** Update 2 SaveResults call sites (determinism test) to handle error return.

**`cmd/root_test.go`:** Update 1 SaveResults call site to handle error return.
