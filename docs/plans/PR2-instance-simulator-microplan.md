# PR2: InstanceSimulator Wrapper - Micro-Design Plan

**PR Title:** `feat(cluster): Add InstanceSimulator wrapper`
**Date:** 2026-02-11
**Status:** Draft
**Depends On:** PR1 (PartitionedRNG) - MERGED
**Scope Extension:** Includes GitHub Actions CI setup for automated testing

---

## A) Executive Summary

PR2 introduces `InstanceSimulator`, a thin wrapper around the existing `Simulator` that provides:
1. A composable unit for multi-replica cluster simulation (PR3)
2. An interception point for cluster-level coordination
3. An `InstanceID` type for explicit instance identification

The wrapper uses **composition over inheritance**: it delegates all simulation logic to the wrapped `Simulator` while exposing a stable interface for `ClusterSimulator` (PR3) to consume.

**Critical constraint:** All simulation metrics must be **identical** to PR1. JSON output gains one new field (`instance_id`) but all other values remain unchanged.

**Scope Extensions:**
1. GitHub Actions CI configuration to run automated tests on every PR
2. Add `instance_id` field to JSON output (additive, non-breaking change)

---

## B) Targeted Recon Summary

### B.1 Files Directly Impacted

| File | Current State | PR2 Changes |
|------|---------------|-------------|
| `sim/simulator.go` | Core simulator with 17-field struct, `Run()`, `Step()` methods | **No changes** - wrapped, not modified |
| `cmd/root.go` | Creates `Simulator` directly in `runCmd.Run()` | Route through `InstanceSimulator`, pass instance ID to `SaveResults` |
| `sim/cluster/` | Does not exist | **New package**: `instance.go` |
| `sim/metrics.go` | `SaveResults(horizon, totalBlocks, startTime, outputFilePath)` | Add `instanceID` parameter to `SaveResults` |
| `sim/metrics_utils.go` | `MetricsOutput` struct without instance ID | Add `InstanceID` field to `MetricsOutput` |
| `.github/workflows/` | Does not exist | **New**: `ci.yml` for automated testing |

### B.2 Current Behavior of Touched Codepaths

**CLI Flow (cmd/root.go:174-193):**
```go
s := sim.NewSimulator(
    simulationHorizon,
    seed,
    totalKVBlocks,
    // ... 13 more parameters
)
s.Run()
s.Metrics.SaveResults(s.Horizon, totalKVBlocks, startTime, resultsPath)
```

**Simulator.Run() (sim/simulator.go:170-186):**
- Pops events from heap, advances clock, executes events
- Terminates when `EventQueue` empty or `Clock > Horizon`
- Sets `Metrics.SimEndedTime` at completion

**Simulator.Step() (sim/simulator.go:408-528):**
- Forms running batch via `makeRunningBatch()`
- Executes model step, handles completions
- Schedules next `StepEvent` if requests remain

### B.3 Relevant Invariants

| Invariant | Location | Verification |
|-----------|----------|--------------|
| **Determinism** | `NewSimulator` seeds RNG | Same seed → identical output |
| **Clock monotonicity** | `Run()` line 175 | `sim.Clock = ev.Timestamp()` |
| **Request lifecycle** | `Step()` | queued → running → completed |
| **KV conservation** | `kvcache.go` | `used + free = total` |

### B.4 Data Flow Across Boundaries

```
CLI (cmd/root.go)
      │
      ▼
InstanceSimulator.Run()  ◄─── NEW: wrapper entry point
      │
      ▼
Simulator.Run()          ◄─── Existing: unchanged
      │
      ▼
Simulator.Step()         ◄─── Existing: unchanged
      │
      ▼
Metrics                  ◄─── Existing: unchanged
```

### B.5 Concurrency Assumptions

- **Current:** Single-threaded event loop, no goroutines in hot path
- **PR2:** Maintains single-threaded model; `InstanceSimulator` is NOT thread-safe
- **Future (PR3):** `ClusterSimulator` may coordinate multiple `InstanceSimulator` instances

### B.6 Confirmed Facts

1. `Simulator` has 17 fields, all initialized in `NewSimulator()`
2. `Run()` is the only public entry point for simulation execution
3. `Metrics` is the only output artifact (accessed via `s.Metrics`)
4. No existing `sim/cluster/` package
5. No existing GitHub Actions configuration
6. Golden dataset tests exist in `testdata/goldendataset.json`
7. Go version is 1.21 (from go.mod)

---

## C) Expanded Contracts

### C.1 Behavioral Contracts

#### MUST Happen

| Contract | Description |
|----------|-------------|
| **BC-1** | `InstanceSimulator.Run()` produces **identical** `Metrics` to `Simulator.Run()` for same inputs |
| **BC-2** | `InstanceSimulator` exposes `Metrics()` accessor returning wrapped simulator's metrics |
| **BC-3** | `InstanceSimulator` exposes `Clock()` accessor returning current simulation time |
| **BC-4** | `InstanceID` is a distinct type (not alias) preventing accidental string mixing |
| **BC-5** | `NewInstanceSimulator()` accepts same parameters as `NewSimulator()` plus `InstanceID` |
| **BC-6** | GitHub Actions runs `go test ./...` on every PR to main |
| **BC-7** | GitHub Actions runs `go build` to verify compilation |
| **BC-8** | JSON output includes `instance_id` field with value from `InstanceSimulator.ID()` |

#### MUST NOT Happen

| Contract | Description |
|----------|-------------|
| **BN-1** | `InstanceSimulator` must NOT modify `Simulator` internal state directly |
| **BN-2** | `InstanceSimulator` must NOT add fields that require initialization beyond delegation |
| **BN-3** | `InstanceSimulator` must NOT expose `Simulator` directly (encapsulation) |
| **BN-4** | PR must NOT change any existing test behavior or golden dataset values |
| **BN-5** | GitHub Actions must NOT fail on flaky tests (all tests must be deterministic) |

#### Edge Case Behavior

| Scenario | Behavior |
|----------|----------|
| Empty `InstanceID` | Valid; uses empty string as ID |
| Zero-length simulation | Wrapper delegates; `Simulator` handles gracefully |
| Nil `GuideLLMConfig` | Wrapper delegates; traces workload path activated |

#### Backward Compatibility

| Aspect | Guarantee |
|--------|-----------|
| CLI flags | All existing flags work identically |
| Output JSON | Additive change only: new `instance_id` field; all other fields unchanged |
| Golden tests | All pass without modification (golden tests check metrics, not JSON schema) |
| RNG sequences | Identical (same seed → same output) |

### C.2 API Contracts

#### New Types

```go
// InstanceID uniquely identifies a simulator instance within a cluster.
// Uses distinct type (not alias) to prevent accidental string mixing.
type InstanceID string

// InstanceSimulator wraps a Simulator for use in multi-replica clusters.
// Provides an interception point for cluster-level coordination.
type InstanceSimulator struct {
    id  InstanceID
    sim *Simulator
}
```

#### Constructor

```go
// NewInstanceSimulator creates an InstanceSimulator wrapping a new Simulator.
// All parameters except `id` are passed directly to NewSimulator.
//
// Thread-safety: NOT thread-safe. Must be called from single goroutine.
// Failure modes: Panics if internal Simulator creation fails (matches existing behavior).
func NewInstanceSimulator(
    id InstanceID,
    horizon int64,
    seed int64,
    totalKVBlocks int64,
    blockSizeTokens int64,
    maxRunningReqs int64,
    maxScheduledTokens int64,
    longPrefillTokenThreshold int64,
    betaCoeffs []float64,
    alphaCoeffs []float64,
    guideLLMConfig *GuideLLMConfig,
    modelConfig ModelConfig,
    hwConfig HardwareCalib,
    model string,
    GPU string,
    tp int,
    roofline bool,
    tracesWorkloadFilePath string,
) *InstanceSimulator
```

#### Methods

```go
// Run executes the simulation to completion.
// Delegates directly to wrapped Simulator.Run().
//
// Postconditions:
//   - Metrics() returns populated metrics
//   - Clock() returns final simulation time
func (i *InstanceSimulator) Run()

// ID returns the instance identifier.
func (i *InstanceSimulator) ID() InstanceID

// Clock returns the current simulation clock (in ticks).
func (i *InstanceSimulator) Clock() int64

// Metrics returns the simulation metrics.
// Returns pointer to wrapped Simulator's Metrics (not a copy).
func (i *InstanceSimulator) Metrics() *Metrics

// Horizon returns the simulation horizon (in ticks).
func (i *InstanceSimulator) Horizon() int64
```

#### Input/Output Invariants

| Method | Input Invariant | Output Invariant |
|--------|-----------------|------------------|
| `NewInstanceSimulator` | Same as `NewSimulator` | Non-nil `*InstanceSimulator` |
| `Run()` | None | `Metrics()` populated |
| `ID()` | None | Same value as constructor |
| `Clock()` | None | `>= 0`, monotonic during `Run()` |
| `Metrics()` | None | Same pointer throughout lifetime |

#### Thread-Safety Guarantees

- `InstanceSimulator` is **NOT** thread-safe
- All methods must be called from the same goroutine
- `Metrics()` returns a pointer; concurrent access requires external synchronization

---

## D) Detailed Implementation Plan

### D.1 New Files

#### `sim/cluster/instance.go` (~120 LOC)

```
sim/cluster/
└── instance.go    # InstanceID type, InstanceSimulator struct+methods
```

**Package Dependencies:**
```go
import (
    "github.com/inference-sim/inference-sim/sim"
)
```
This creates `sim/cluster` → `sim` dependency (valid, one-way).

**Contents:**
1. Package declaration with doc comment
2. `InstanceID` type definition
3. `InstanceSimulator` struct definition
4. `NewInstanceSimulator()` constructor
5. `Run()`, `ID()`, `Clock()`, `Metrics()`, `Horizon()` methods

**Technical Debt Acknowledgment:**
The 18-parameter constructor is inherited from `sim.NewSimulator()`. This is intentional for PR2 (preserve compatibility). Future PRs may introduce options pattern or config struct to improve ergonomics.

#### `.github/workflows/ci.yml` (~40 LOC)

```
.github/
└── workflows/
    └── ci.yml    # GitHub Actions workflow for PR testing
```

**Contents:**
1. Workflow name and trigger configuration (PR to main)
2. Go setup with version from go.mod
3. Dependency caching
4. Build verification step
5. Test execution step with verbose output

### D.2 Modified Files

#### `cmd/root.go` (~20 LOC changes)

**Changes:**
1. Add import for `github.com/inference-sim/inference-sim/sim/cluster`
2. Replace `sim.NewSimulator(...)` with `cluster.NewInstanceSimulator("default", ...)`
3. Replace `s.Run()` with instance wrapper call
4. Replace `s.Metrics` with `instance.Metrics()`
5. Replace `s.Horizon` with `instance.Horizon()`
6. Pass instance ID to `SaveResults()`

**Before (lines 174-196):**
```go
s := sim.NewSimulator(
    simulationHorizon,
    seed,
    // ... parameters
)
s.Run()
s.Metrics.SaveResults(s.Horizon, totalKVBlocks, startTime, resultsPath)
```

**After:**
```go
instance := cluster.NewInstanceSimulator(
    cluster.InstanceID("default"),
    simulationHorizon,
    seed,
    // ... same parameters
)
instance.Run()
instance.Metrics().SaveResults(string(instance.ID()), instance.Horizon(), totalKVBlocks, startTime, resultsPath)
```

#### `sim/metrics_utils.go` (~3 LOC changes)

**Changes:**
Add `InstanceID` field to `MetricsOutput` struct.

**Before:**
```go
type MetricsOutput struct {
    SimStartTimestamp     string           `json:"sim_start_timestamp"`
    SimEndTimestamp       string           `json:"sim_end_timestamp"`
    // ...
}
```

**After:**
```go
type MetricsOutput struct {
    InstanceID            string           `json:"instance_id"`
    SimStartTimestamp     string           `json:"sim_start_timestamp"`
    SimEndTimestamp       string           `json:"sim_end_timestamp"`
    // ...
}
```

#### `sim/metrics.go` (~5 LOC changes)

**Changes:**
1. Add `instanceID string` parameter to `SaveResults()`
2. Set `output.InstanceID = instanceID` in struct initialization

**Note:** `SaveResults` prints to BOTH stdout (line 114-119) AND writes to file (if `outputFilePath != ""`). Both outputs use the same `MetricsOutput` struct, so `instance_id` appears in both automatically.

**Before:**
```go
func (m *Metrics) SaveResults(horizon int64, totalBlocks int64, startTime time.Time, outputFilePath string) {
    // ...
    output := MetricsOutput{
        SimStartTimestamp:     startTime.Format("2006-01-02 15:04:05"),
        // ...
    }
}
```

**After:**
```go
func (m *Metrics) SaveResults(instanceID string, horizon int64, totalBlocks int64, startTime time.Time, outputFilePath string) {
    // ...
    output := MetricsOutput{
        InstanceID:            instanceID,
        SimStartTimestamp:     startTime.Format("2006-01-02 15:04:05"),
        // ...
    }
}
```

### D.3 Control Flow Changes

**Before:**
```
cmd/root.go:runCmd.Run()
    └── sim.NewSimulator()
    └── Simulator.Run()
    └── Metrics.SaveResults()
```

**After:**
```
cmd/root.go:runCmd.Run()
    └── cluster.NewInstanceSimulator()
        └── sim.NewSimulator()       # Internal delegation
    └── InstanceSimulator.Run()
        └── Simulator.Run()          # Internal delegation
    └── InstanceSimulator.Metrics()
        └── Simulator.Metrics        # Returns wrapped metrics
    └── Metrics.SaveResults()
```

### D.4 Validation Logic

| Check | Location | Implementation |
|-------|----------|----------------|
| Constructor delegation | `NewInstanceSimulator` | Calls `NewSimulator` with same params |
| Run delegation | `Run()` | Calls `sim.Run()` directly |
| Metrics passthrough | `Metrics()` | Returns `i.sim.Metrics` |

### D.5 Error Handling

- **No new error paths introduced**
- Constructor panics propagate from `NewSimulator` (existing behavior)
- No runtime errors possible in wrapper methods (all are simple delegations)

### D.6 Logging/Observability

- **No new logging added** (wrapper is transparent)
- Existing `logrus` calls in `Simulator` continue to work
- Future PR3 may add instance-specific logging prefix

### D.7 Dead Code Prevention Checklist

| Concern | Mitigation |
|---------|------------|
| `InstanceSimulator` unused | CLI routes through it immediately |
| `InstanceID` unused | Passed to constructor, stored, returned by `ID()`, written to JSON output |
| `Clock()` method unused | Used in future PR3; tested in unit tests |
| `Horizon()` method unused | Used by CLI for `SaveResults()` |
| `MetricsOutput.InstanceID` unused | Written to JSON on every `--results-path` invocation |
| GitHub Actions unused | Triggered on every PR |

### D.8 GitHub Actions Configuration

**Workflow triggers:**
- On push to `main` branch
- On pull request to `main` branch

**Jobs:**
1. **build-and-test** (runs on ubuntu-latest)
   - Checkout code
   - Setup Go 1.21
   - Cache Go modules
   - Run `go build ./...`
   - Run `go test -v ./...`

**Design rationale:**
- Single job keeps CI simple and fast
- Verbose test output helps debug failures
- Module caching speeds up subsequent runs
- Matches Go version from go.mod

---

## E) CLI Exercise Proof

### E.1 Basic Execution (Existing Behavior, Through Wrapper)

```bash
# Build with new wrapper
go build -o simulation_worker main.go

# Run with default parameters
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --seed 42 --max-prompts 10 --results-path /tmp/output.json

# Expected output now includes instance_id:
cat /tmp/output.json | head -5
# {
#   "instance_id": "default",
#   "sim_start_timestamp": "2026-02-11 10:30:00",
#   ...
# }

# All metric VALUES are identical to PR1 (same completed_requests, tokens, latencies)
```

### E.2 Determinism Verification

```bash
# Run twice with same seed
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --seed 42 \
  --max-prompts 50 --results-path /tmp/run1.json

./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --seed 42 \
  --max-prompts 50 --results-path /tmp/run2.json

# Compare outputs (excluding timestamps which vary)
diff /tmp/run1.json /tmp/run2.json
# Expected: Only sim_start_timestamp/sim_end_timestamp differ (wall clock)
# All metrics (completed_requests, e2e_*, ttft_*, etc.) must be identical
```

### E.3 All Existing Workload Types

```bash
# Distribution workload (default)
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --workload distribution \
  --rate 10 --max-prompts 20 --prompt-tokens 256 --output-tokens 128

# Traces workload
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --workload traces \
  --workload-traces-filepath testdata/sample_traces.csv

# Preset workloads
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --workload chatbot
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --workload summarization
```

### E.4 Roofline Mode

```bash
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct \
  --model-config-folder model_configs/llama-3.1-8b-instruct \
  --hardware-config hardware_config.json --hardware H100 --tp 1
```

### E.5 Test Execution

```bash
# All tests pass
go test -v ./...

# Specific package tests
go test -v ./sim/cluster/...

# Golden dataset tests (critical for backward compat)
go test -v ./sim/... -run TestSimulator_GoldenDataset
```

### E.6 GitHub Actions Verification

After PR is created, verify:
1. CI workflow appears in GitHub Actions tab
2. Build step succeeds
3. All tests pass
4. PR shows green checkmark

---

## F) Test Matrix

### Key Behavioral Property

The **primary behavioral guarantee** of PR2 is:

> Running a simulation through `InstanceSimulator` produces **identical results** to running directly through `Simulator`.

All tests are designed to verify this property or its corollaries.

### F.1 Behavioral Tests (`sim/cluster/instance_test.go`)

Tests follow BDD methodology with Given/When/Then structure.

#### F.1.1 Equivalence Tests (Critical for BC-1)

| Test Name | Scenario | Contract |
|-----------|----------|----------|
| `TestInstanceSimulator_GoldenDataset_Equivalence` | **GIVEN** golden dataset test cases<br>**WHEN** run through `InstanceSimulator` wrapper<br>**THEN** all metrics match golden expected values exactly | BC-1, BN-4 |
| `TestInstanceSimulator_Determinism` | **GIVEN** same seed (42) and config<br>**WHEN** simulation runs twice via `InstanceSimulator`<br>**THEN** `CompletedRequests`, `TotalInputTokens`, `TotalOutputTokens` are identical | BC-1 |

#### F.1.2 Accessor Behavior Tests

| Test Name | Scenario | Contract |
|-----------|----------|----------|
| `TestInstanceSimulator_ID_ReturnsConstructorValue` | **GIVEN** `InstanceSimulator` created with ID "replica-0"<br>**WHEN** `ID()` is called<br>**THEN** returns `InstanceID("replica-0")` | BC-4 |
| `TestInstanceSimulator_Clock_AdvancesWithSimulation` | **GIVEN** simulation with 10 requests<br>**WHEN** `Run()` completes<br>**THEN** `Clock() > 0` AND `Clock() == Metrics().SimEndedTime` | BC-3 |
| `TestInstanceSimulator_Metrics_DelegatesCorrectly` | **GIVEN** simulation runs to completion<br>**WHEN** `Metrics()` is accessed<br>**THEN** `Metrics().CompletedRequests > 0` (not a nil or empty struct) | BC-2 |

#### F.1.3 Edge Case Tests

| Test Name | Scenario | Contract |
|-----------|----------|----------|
| `TestInstanceSimulator_EmptyID_Valid` | **GIVEN** `InstanceSimulator` created with ID ""<br>**WHEN** `ID()` is called<br>**THEN** returns `InstanceID("")` (no panic, no error) | Edge |
| `TestInstanceSimulator_ZeroRequests` | **GIVEN** config with `MaxPrompts=0`<br>**WHEN** `Run()` completes<br>**THEN** `Metrics().CompletedRequests == 0` AND no panic | Edge |

### F.1.4 Output Tests (`sim/metrics_test.go`)

| Test Name | Scenario | Contract |
|-----------|----------|----------|
| `TestSaveResults_InstanceID_InJSON` | **GIVEN** `SaveResults` called with instanceID "default"<br>**WHEN** JSON file is written<br>**THEN** file contains `"instance_id": "default"` as first field | BC-8 |
| `TestSaveResults_InstanceID_InStdout` | **GIVEN** `SaveResults` called (stdout output)<br>**WHEN** output is captured<br>**THEN** stdout JSON includes `instance_id` field | BC-8 |
| `TestSaveResults_InstanceID_Empty` | **GIVEN** `SaveResults` called with instanceID ""<br>**WHEN** JSON file is written<br>**THEN** file contains `"instance_id": ""` | BC-8 |

### F.2 Integration Tests

| Test Name | Scenario | Contract |
|-----------|----------|----------|
| `TestCLI_InstanceSimulator_E2E` | **GIVEN** CLI invoked with `--results-path`<br>**WHEN** simulation completes<br>**THEN** JSON file exists AND contains `instance_id: "default"` | BC-8 |

### F.3 Regression Tests (Existing, Unmodified)

| Test Name | Contract | Description |
|-----------|----------|-------------|
| `TestSimulator_GoldenDataset` | BN-4 | Existing test continues to pass (direct `Simulator` path) |
| `TestSimulator_DeterministicWorkload` | BC-1 | Determinism maintained |
| `TestPartitionedRNG_*` | BC-1 | RNG isolation preserved |

**Note:** `TestSimulator_GoldenDataset` tests the direct `Simulator` path. `TestInstanceSimulator_GoldenDataset_Equivalence` (new) tests the same golden data through the wrapper, proving equivalence.

### F.5 GitHub Actions Tests

| Test | Description |
|------|-------------|
| PR creation triggers workflow | Workflow runs on PR open |
| Build step passes | `go build ./...` succeeds |
| Test step passes | `go test ./...` succeeds |
| Cache works | Subsequent runs use cached modules |

### F.6 Test-to-Contract Mapping

| Contract | Tests |
|----------|-------|
| BC-1 | `TestInstanceSimulator_GoldenDataset_Equivalence`, `TestInstanceSimulator_Determinism` |
| BC-2 | `TestInstanceSimulator_Metrics_DelegatesCorrectly` |
| BC-3 | `TestInstanceSimulator_Clock_AdvancesWithSimulation` |
| BC-4 | `TestInstanceSimulator_ID_ReturnsConstructorValue` |
| BC-5 | (Covered by all tests that construct `InstanceSimulator`) |
| BC-6 | GitHub Actions workflow execution |
| BC-7 | GitHub Actions build step |
| BC-8 | `TestSaveResults_InstanceID_InJSON`, `TestSaveResults_InstanceID_InStdout`, `TestCLI_InstanceSimulator_E2E` |
| BN-4 | `TestInstanceSimulator_GoldenDataset_Equivalence` (same golden data, through wrapper) |

---

## G) Risk Analysis

### G.1 Invariant Break Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Wrapper changes RNG sequence | Low | High | Test: `TestInstanceSimulator_GoldenDataset` |
| Metrics pointer differs | Low | Medium | Test: `TestInstanceSimulator_Metrics_SamePointer` |
| Clock drift between wrapper and delegate | Very Low | High | Test: `TestInstanceSimulator_Clock_AfterRun` |

### G.2 Performance Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Wrapper overhead | Very Low | Low | Wrapper is trivial delegation; benchmark in PR3 |
| Method call overhead | Negligible | Negligible | Go inlines trivial methods |

### G.3 Backward Compatibility Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Golden tests fail | Low | High | Run golden tests before merge |
| CLI output changes | Low | High | Test: diff-based verification |
| API breakage | None | N/A | No existing external API |

### G.4 Hidden Coupling Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Direct `Simulator` access elsewhere | None | N/A | Only `cmd/root.go` creates simulators |
| Test coupling to internals | Low | Low | Tests use public API only |

### G.5 Observability Gaps

| Gap | Impact | Mitigation |
|-----|--------|------------|
| No instance-specific logging | Low | Defer to PR3 (cluster logging) |
| No wrapper-level metrics | None | Wrapper is transparent |

### G.6 GitHub Actions Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Flaky tests | Low | Medium | All tests are deterministic (seeded RNG) |
| Go version mismatch | Low | Low | Explicit version in workflow matches go.mod |
| Cache invalidation issues | Low | Low | Use official actions/cache |

---

## H) Sanity Checklist

### H.1 Design Quality

| Check | Status | Notes |
|-------|--------|-------|
| No unnecessary abstractions | ✅ | Single wrapper struct, minimal methods |
| No feature creep | ✅ | Only wrapper + CI; no policies, no cluster |
| No unexercised flags | ✅ | No new CLI flags |
| No partial implementations | ✅ | Wrapper is complete; all methods implemented |
| No breaking changes | ✅ | CLI unchanged; output identical |
| No hidden global state | ✅ | Wrapper is stateless except for delegation |

### H.2 Code Quality

| Check | Status | Notes |
|-------|--------|-------|
| All new code has tests | ✅ | Unit + integration tests defined |
| Tests map to contracts | ✅ | See Test Matrix section |
| Error handling explicit | ✅ | No new error paths; delegation only |
| Thread-safety documented | ✅ | "NOT thread-safe" in doc comments |

### H.3 CI Quality

| Check | Status | Notes |
|-------|--------|-------|
| Workflow triggers correct | ✅ | PR to main, push to main |
| Go version matches project | ✅ | Uses 1.21 from go.mod |
| Tests are deterministic | ✅ | All use seeded RNG |
| Build step included | ✅ | Catches compilation errors |

### H.4 Backward Compatibility

| Check | Status | Notes |
|-------|--------|-------|
| Golden tests pass | ✅ | Required for merge |
| CLI flags unchanged | ✅ | No additions or removals |
| Output format additive only | ✅ | New `instance_id` field; no removals or type changes |
| Same seed → same metrics | ✅ | Tested explicitly; all metric values identical |

### H.5 Documentation

| Check | Status | Notes |
|-------|--------|-------|
| Package doc comment | ✅ | Explains cluster package purpose |
| Type doc comments | ✅ | `InstanceID`, `InstanceSimulator` documented |
| Method doc comments | ✅ | All public methods documented |
| Thread-safety noted | ✅ | In constructor and type doc |

---

## I) Implementation Checklist

### Phase 1: Create cluster package
- [ ] Create `sim/cluster/` directory
- [ ] Create `sim/cluster/instance.go` with types and methods
- [ ] Create `sim/cluster/instance_test.go` with unit tests

### Phase 2: Add instance_id to output
- [ ] Add `InstanceID` field to `MetricsOutput` struct in `sim/metrics_utils.go`
- [ ] Add `instanceID` parameter to `SaveResults()` in `sim/metrics.go`
- [ ] Create `sim/metrics_test.go` with instance_id tests

### Phase 3: Wire CLI
- [ ] Modify `cmd/root.go` to use `InstanceSimulator`
- [ ] Pass instance ID to `SaveResults()`
- [ ] Verify golden tests pass
- [ ] Verify determinism with diff test

### Phase 4: GitHub Actions
- [ ] Create `.github/workflows/` directory
- [ ] Create `.github/workflows/ci.yml` workflow
- [ ] Test workflow locally with `act` (optional)

### Phase 5: Validation
- [ ] Run `go test ./...` - all pass
- [ ] Run golden dataset test specifically
- [ ] Run CLI with various workload types
- [ ] Verify JSON output contains `instance_id: "default"`
- [ ] Verify all other metric values identical to PR1 (use `jq` to compare):
  ```bash
  # PR1 output (hypothetically saved)
  # PR2 output
  ./simulation_worker run --model X --seed 42 --results-path /tmp/pr2.json

  # Compare metrics (excluding instance_id and timestamps)
  jq 'del(.instance_id, .sim_start_timestamp, .sim_end_timestamp, .simulation_duration_s)' /tmp/pr2.json
  # Should match PR1 output with same seed
  ```

### Phase 6: PR
- [ ] Create PR with descriptive title
- [ ] Verify GitHub Actions runs and passes
- [ ] Request review

---

## J) Estimated LOC

| File | New LOC | Modified LOC |
|------|---------|--------------|
| `sim/cluster/instance.go` | ~120 | 0 |
| `sim/cluster/instance_test.go` | ~200 | 0 |
| `sim/metrics_test.go` | ~50 | 0 |
| `cmd/root.go` | 0 | ~20 |
| `sim/metrics.go` | 0 | ~5 |
| `sim/metrics_utils.go` | 0 | ~3 |
| `.github/workflows/ci.yml` | ~40 | 0 |
| **Total** | **~410** | **~28** |

---

## K) Appendix: GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Go
        uses: actions/setup-go@v5
        with:
          go-version: '1.21'
          cache: true

      - name: Build
        run: go build -v ./...

      - name: Test
        run: go test -v ./...
```
