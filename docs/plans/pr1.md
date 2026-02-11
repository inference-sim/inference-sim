# PR 1: PartitionedRNG and SimulationKey

**Title:** `feat(sim): Add PartitionedRNG for deterministic multi-subsystem simulation`

**Status:** Planning
**Based on:** [Macro Implementation Plan v2.2](2026-02-11-macro-implementation-plan-v2.md)

---

## A) Executive Summary

This PR introduces `PartitionedRNG` and `SimulationKey` to enable deterministic, isolated random number generation across multiple simulation subsystems.

**Why this matters:** The current simulator uses a single `*rand.Rand` instance. When we add multi-instance cluster simulation (PR 2-3), routing policies, and schedulers, each subsystem needs its own RNG stream. Without isolation, changing the number of random draws in one subsystem (e.g., routing) would alter the random sequence in another (e.g., workload generation), breaking reproducibility.

**What this PR delivers:**
- `SimulationKey` — Type representing the determinism key for a simulation run
- `PartitionedRNG` — Provides isolated, lazily-initialized RNG instances per subsystem
- Refactored `Simulator` to use `PartitionedRNG` instead of raw `*rand.Rand`
- **100% backward compatibility** — Same seed produces identical output

**What this PR does NOT do:**
- No multi-instance support (PR 2-3)
- No new CLI flags
- No policy interfaces

**LOC Estimate:** ~100 production code, ~120 test code

---

## B) Targeted Recon Summary

### B.1 Files Impacted

| File | Current State | PR Impact |
|------|---------------|-----------|
| `sim/simulator.go` | Contains `randomNumberGenerator *rand.Rand` field (line 112), initialized from seed (lines 146-147) | Replace with `PartitionedRNG`; add accessor |
| `sim/workload_config.go` | Calls `sim.randomNumberGenerator.NormFloat64()` (line 98) and `sim.randomNumberGenerator.Intn()` (line 112) | Update to use accessor method |
| `sim/rng.go` | **Does not exist** | Create with `SimulationKey`, `PartitionedRNG`, subsystem constants |

**Note:** The macro plan lists only `sim/rng.go` (new) and `sim/simulator.go` (modified). The `workload_config.go` modification is a necessary consequence—2 lines changing `sim.randomNumberGenerator` to `sim.WorkloadRNG()`.

### B.2 Current RNG Behavior

```go
// sim/simulator.go:112
randomNumberGenerator  *rand.Rand // random number generator for request tokens

// sim/simulator.go:146-147 (in NewSimulator)
src := rand.NewSource(seed)
s.randomNumberGenerator = rand.New(src)
```

**Usage (2 call sites in workload_config.go):**
1. Line 98: `sim.randomNumberGenerator.NormFloat64()` — Gaussian length sampling
2. Line 112: `sim.randomNumberGenerator.Intn(MaxTokenID)` — Random token generation

### B.3 Current Invariants

1. **Determinism:** Same `--seed` produces identical simulation results
2. **Initialization order:** RNG initialized in `NewSimulator()` before workload generation
3. **Single consumer:** Only workload generation uses RNG (currently)

### B.4 Data Flow

```
CLI --seed 42
       ↓
cmd/root.go:212 (flag parsing)
       ↓
cmd/root.go:174 → sim.NewSimulator(..., seed, ...)
       ↓
sim/simulator.go:146-147 → rand.New(rand.NewSource(seed))
       ↓
sim.randomNumberGenerator
       ↓
workload_config.go → generateWorkloadDistribution()
```

### B.5 Concurrency Assumptions

- **Single-threaded.** Event loop is synchronous.
- RNG accessed only from main goroutine during workload generation (before `Run()`).

---

## C) Expanded Contracts

### C.1 Behavioral Contracts (BDD Scenarios)

```gherkin
Feature: Partitioned RNG for Deterministic Multi-Subsystem Simulation

  Background:
    Given the FNV-1a hash function for seed derivation
    And subsystem "workload" uses master seed directly for backward compatibility

  Scenario: Backward compatibility with existing simulations
    Given the golden dataset at "testdata/goldendataset.json" with 5 test cases
    And all test cases use seed 42
    When I run each test case with the NEW implementation
    Then completed_requests MUST match exactly for each test case
    And total_input_tokens MUST match exactly for each test case
    And total_output_tokens MUST match exactly for each test case
    # These metrics prove identical RNG sequences were generated

  Scenario: Deterministic seed derivation
    Given a PartitionedRNG with master seed 42
    When I request RNG for subsystem "router"
    And I draw 3 values: [v1, v2, v3]
    And I create a NEW PartitionedRNG with master seed 42
    And I request RNG for subsystem "router"
    And I draw 3 values: [v4, v5, v6]
    Then v1 == v4 AND v2 == v5 AND v3 == v6

  Scenario: Subsystem isolation
    Given PartitionedRNG "A" with master seed 42
    And PartitionedRNG "B" with master seed 42
    When I draw 10 values from "A"'s "workload" subsystem
    And I draw 5 values from "B"'s "router" subsystem
    And I draw 1 value from "A"'s "router" subsystem as "a_router_first"
    And I draw 1 value from "B"'s "router" subsystem as "b_router_sixth"
    Then "a_router_first" is the 1st value in router's sequence
    And "b_router_sixth" is the 6th value in router's sequence
    # Proves: drawing from "workload" in A did NOT affect "router" sequence

  Scenario: Lazy initialization
    Given a PartitionedRNG with master seed 42
    When I check internal state before any ForSubsystem() call
    Then the subsystems map is empty
    When I call ForSubsystem("workload")
    Then the subsystems map contains exactly 1 entry

  Scenario: Instance caching
    Given a PartitionedRNG with master seed 42
    When I call ForSubsystem("workload") and store result as "rng1"
    And I call ForSubsystem("workload") and store result as "rng2"
    Then rng1 and rng2 are the same pointer (identical instance)

  Scenario: Edge case - empty subsystem name
    Given a PartitionedRNG with master seed 42
    When I call ForSubsystem("")
    Then I receive a valid non-nil *rand.Rand

  Scenario: Edge case - seed is zero
    Given a PartitionedRNG with master seed 0
    When I call ForSubsystem("router")
    Then I receive a valid non-nil *rand.Rand
    And the derived seed equals fnv1a64("router") XOR 0 = fnv1a64("router")

  Scenario: Edge case - seed is negative
    Given a PartitionedRNG with master seed -9223372036854775808 (MinInt64)
    When I call ForSubsystem("workload")
    Then I receive a valid non-nil *rand.Rand
```

### C.2 API Contracts

#### Type: `SimulationKey`

```go
// SimulationKey uniquely identifies a reproducible simulation run.
// Two simulations with the same SimulationKey and identical configuration
// MUST produce bit-for-bit identical results.
type SimulationKey int64

// NewSimulationKey creates a SimulationKey from a seed value.
func NewSimulationKey(seed int64) SimulationKey
```

**Invariants:**
- `SimulationKey` is a value type (not pointer)
- Equality: two `SimulationKey` values are equal iff their underlying `int64` values are equal

#### Type: `PartitionedRNG`

```go
// PartitionedRNG provides deterministic, isolated RNG instances per subsystem.
//
// Derivation formula:
//   - For SubsystemWorkload: uses masterSeed directly (backward compatibility)
//   - For all other subsystems: masterSeed XOR fnv1a64(subsystemName)
//
// Thread-safety: NOT thread-safe. Must be called from single goroutine.
type PartitionedRNG struct {
    key        SimulationKey
    subsystems map[string]*rand.Rand
}

// NewPartitionedRNG creates a PartitionedRNG from a SimulationKey.
func NewPartitionedRNG(key SimulationKey) *PartitionedRNG

// ForSubsystem returns a deterministically-seeded RNG for the named subsystem.
// The same subsystem name always returns the same *rand.Rand instance (cached).
// Never returns nil.
func (p *PartitionedRNG) ForSubsystem(name string) *rand.Rand

// Key returns the SimulationKey used to create this PartitionedRNG.
func (p *PartitionedRNG) Key() SimulationKey
```

#### Constants

```go
const (
    // SubsystemWorkload is the RNG subsystem for workload generation.
    // Uses master seed directly for backward compatibility.
    SubsystemWorkload = "workload"

    // SubsystemRouter is the RNG subsystem for routing decisions.
    // Used in PR 6+.
    SubsystemRouter = "router"
)

// SubsystemInstance returns the subsystem name for a specific instance.
// Used in PR 2+ for per-instance RNG isolation.
func SubsystemInstance(id int) string {
    return fmt.Sprintf("instance_%d", id)
}
```

#### Input/Output Invariants

| Method | Input | Output Guarantee |
|--------|-------|------------------|
| `NewSimulationKey(seed)` | Any `int64` | Valid `SimulationKey` |
| `NewPartitionedRNG(key)` | Any `SimulationKey` | Non-nil `*PartitionedRNG` with empty subsystems map |
| `ForSubsystem(name)` | Any `string` | Non-nil `*rand.Rand`; deterministic for same key+name |
| `Key()` | N/A | Original `SimulationKey` |

#### Failure Modes

None. All operations are infallible.

---

## D) Detailed Implementation Plan

### D.1 New File: `sim/rng.go` (~100 LOC)

```go
package sim

import (
    "fmt"
    "hash/fnv"
    "math/rand"
)

// === SimulationKey ===

// SimulationKey uniquely identifies a reproducible simulation run.
type SimulationKey int64

// NewSimulationKey creates a SimulationKey from a seed value.
func NewSimulationKey(seed int64) SimulationKey {
    return SimulationKey(seed)
}

// === Subsystem Constants ===

const (
    // SubsystemWorkload uses master seed directly for backward compatibility.
    SubsystemWorkload = "workload"

    // SubsystemRouter for routing decisions (PR 6+).
    SubsystemRouter = "router"
)

// SubsystemInstance returns the subsystem name for instance N.
func SubsystemInstance(id int) string {
    return fmt.Sprintf("instance_%d", id)
}

// === PartitionedRNG ===

// PartitionedRNG provides deterministic, isolated RNG instances per subsystem.
type PartitionedRNG struct {
    key        SimulationKey
    subsystems map[string]*rand.Rand
}

// NewPartitionedRNG creates a PartitionedRNG from a SimulationKey.
func NewPartitionedRNG(key SimulationKey) *PartitionedRNG {
    return &PartitionedRNG{
        key:        key,
        subsystems: make(map[string]*rand.Rand),
    }
}

// ForSubsystem returns a deterministically-seeded RNG for the named subsystem.
func (p *PartitionedRNG) ForSubsystem(name string) *rand.Rand {
    if rng, ok := p.subsystems[name]; ok {
        return rng
    }

    var derivedSeed int64
    if name == SubsystemWorkload {
        // Backward compatibility: workload uses master seed directly.
        // This ensures existing --seed behavior produces identical output.
        derivedSeed = int64(p.key)
    } else {
        // All other subsystems: XOR with hash for isolation.
        derivedSeed = int64(p.key) ^ fnv1a64(name)
    }

    rng := rand.New(rand.NewSource(derivedSeed))
    p.subsystems[name] = rng
    return rng
}

// Key returns the SimulationKey used to create this PartitionedRNG.
func (p *PartitionedRNG) Key() SimulationKey {
    return p.key
}

// fnv1a64 computes a 64-bit FNV-1a hash of the input string.
func fnv1a64(s string) int64 {
    h := fnv.New64a()
    h.Write([]byte(s))
    return int64(h.Sum64())
}
```

### D.2 Modifications: `sim/simulator.go` (~15 LOC changed)

**Change 1:** Replace field (line 112)
```go
// BEFORE:
randomNumberGenerator  *rand.Rand // random number generator for request tokens

// AFTER:
rng *PartitionedRNG // partitioned RNG for deterministic multi-subsystem simulation
```

**Change 2:** Update constructor (lines 146-147)
```go
// BEFORE:
src := rand.NewSource(seed)
s.randomNumberGenerator = rand.New(src)

// AFTER:
s.rng = NewPartitionedRNG(NewSimulationKey(seed))
```

**Change 3:** Add accessor method (new, after line 157)
```go
// WorkloadRNG returns the RNG for workload generation.
func (sim *Simulator) WorkloadRNG() *rand.Rand {
    return sim.rng.ForSubsystem(SubsystemWorkload)
}
```

### D.3 Modifications: `sim/workload_config.go` (~4 LOC changed)

**Change 1:** Line 98
```go
// BEFORE:
val := sim.randomNumberGenerator.NormFloat64()*float64(lengthStd) + float64(lengthMean)

// AFTER:
val := sim.WorkloadRNG().NormFloat64()*float64(lengthStd) + float64(lengthMean)
```

**Change 2:** Line 112
```go
// BEFORE:
tokens[i] = sim.randomNumberGenerator.Intn(MaxTokenID)

// AFTER:
tokens[i] = sim.WorkloadRNG().Intn(MaxTokenID)
```

### D.4 Dead Code Verification

| New Code | Exercised By | Status |
|----------|--------------|--------|
| `SimulationKey` | `NewSimulator()` | ✅ Every run |
| `NewPartitionedRNG()` | `NewSimulator()` | ✅ Every run |
| `ForSubsystem(SubsystemWorkload)` | `WorkloadRNG()` | ✅ Every workload generation |
| `Key()` | Tests | ✅ Test coverage |
| `fnv1a64()` | `ForSubsystem()` for non-workload | ⚠️ Not exercised in PR1 |
| `SubsystemRouter` | Future PR 6 | ⚠️ Constant only |
| `SubsystemInstance()` | Future PR 2 | ⚠️ Not exercised in PR1 |

**Dead code assessment:**
- `fnv1a64()` is called by `ForSubsystem()` but only when subsystem != "workload"
- In PR1, only "workload" is used, so `fnv1a64()` is exercised only via tests
- `SubsystemRouter` and `SubsystemInstance()` are preparatory constants for PR 2+

**Decision:** This is acceptable because:
1. The constants have zero runtime cost
2. They prevent magic strings in future PRs
3. `fnv1a64()` is tested directly
4. Alternative (adding them in PR2) would require modifying `rng.go` again

---

## E) CLI Exercise Proof

### E.1 Backward Compatibility Verification

**Using existing golden dataset:** `testdata/goldendataset.json` contains 5 pre-computed test cases with `seed: 42`.

```bash
# Test case 1: llama-3.1-8b with prefix tokens (from goldendataset.json)
./simulation_worker run --model "meta-llama/llama-3.1-8b-instruct" \
  --workload distribution --rate 10 --max-prompts 300 \
  --prefix-tokens 284 --prompt-tokens 70 --prompt-tokens-stdev 35 \
  --prompt-tokens-min 2 --prompt-tokens-max 200 \
  --output-tokens 215 --output-tokens-stdev 80 \
  --output-tokens-min 1 --output-tokens-max 512 \
  --hardware H100 --tp 1 --seed 42 \
  --max-num-running-reqs 256 --max-num-scheduled-tokens 2048 \
  --total-kv-blocks 7463 --block-size-in-tokens 16 \
  --alpha-coeffs 232.46191091038054,1.752360364195244,3357.4400353290152 \
  --beta-coeffs 5752.705191348184,17.25086436834028,5.999143920128404 \
  --results-path /tmp/pr1_test1.json

# Verify against golden metrics
# Expected: completed_requests=300, total_input_tokens=106250, total_output_tokens=64740
```

**All 5 golden test cases must pass:**

| Test | Model | Expected completed_requests | Expected total_input_tokens |
|------|-------|----------------------------|----------------------------|
| 1 | llama-3.1-8b (prefix) | 300 | 106250 |
| 2 | llama-3.1-8b (long prompt) | 300 | 545758 |
| 3 | qwen2.5-7b | 500 | 177729 |
| 4 | granite-3.1-8b | 300 | 545758 |
| 5 | codellama-34b | 499 | 177729 |

### E.2 Determinism Verification (100 runs)

```bash
# Run 100 times with same seed
for i in {1..100}; do
  ./simulation_worker run \
    --model meta-llama/llama-3.1-8b-instruct \
    --seed 42 \
    --workload distribution \
    --rate 10 \
    --max-prompts 50 \
    --results-path /tmp/det_run_$i.json 2>/dev/null
done

# Verify all outputs identical
md5sum /tmp/det_run_*.json | awk '{print $1}' | sort -u | wc -l
# Expected output: 1
```

### E.3 Different Seeds Produce Different Results

```bash
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct \
  --seed 1 --rate 10 --max-prompts 10 --results-path /tmp/seed1.json 2>/dev/null

./simulation_worker run --model meta-llama/llama-3.1-8b-instruct \
  --seed 2 --rate 10 --max-prompts 10 --results-path /tmp/seed2.json 2>/dev/null

diff /tmp/seed1.json /tmp/seed2.json >/dev/null && echo "FAIL: identical" || echo "PASS: different"
# Expected: PASS: different
```

### E.4 Trace Workload Mode (No RNG)

```bash
# CSV traces don't use RNG - should work unchanged
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --workload traces \
  --workload-traces-filepath testdata/sample_traces.csv \
  --results-path /tmp/traces_test.json
# Expected: Success (CSV parsing unchanged)
```

---

## F) Test Matrix

### F.1 Unit Tests: `sim/rng_test.go`

| Test Name | Contract | Description |
|-----------|----------|-------------|
| `TestSimulationKey_Creation` | SimulationKey type | Verify NewSimulationKey returns expected value |
| `TestPartitionedRNG_DeterministicDerivation` | Deterministic derivation | Same key+name → same sequence |
| `TestPartitionedRNG_SubsystemIsolation` | Isolation guarantee | Drawing from A doesn't affect B |
| `TestPartitionedRNG_WorkloadBackwardCompat` | Backward compatibility | "workload" uses master seed directly |
| `TestPartitionedRNG_CachesInstance` | Lazy init + caching | Same name returns same *rand.Rand |
| `TestPartitionedRNG_Key` | Key() accessor | Returns original SimulationKey |
| `TestPartitionedRNG_EmptySubsystemName` | Edge case | Empty string is valid |
| `TestPartitionedRNG_ZeroSeed` | Edge case | Seed 0 works correctly |
| `TestPartitionedRNG_NegativeSeed` | Edge case | MinInt64 works correctly |
| `TestFnv1a64_Deterministic` | Hash determinism | Same input → same hash |
| `TestFnv1a64_Collision` | Hash quality | Different inputs → different hashes (spot check) |
| `TestSubsystemInstance` | Helper function | Returns "instance_N" format |

### F.2 Integration Tests: `sim/simulator_test.go`

| Test Name | Contract | Description |
|-----------|----------|-------------|
| `TestSimulator_DeterministicWorkload` | End-to-end determinism | Same seed → same request tokens |
| `TestSimulator_WorkloadRNG_NotNil` | WorkloadRNG() accessor | Never returns nil |
| `TestSimulator_GoldenDataset` | Backward compatibility | All 5 golden test cases pass exactly |

**Golden dataset test is critical:** The `testdata/goldendataset.json` file contains 5 diverse configurations (different models, token distributions, KV block counts) all with `seed: 42`. If PR1 breaks backward compatibility, at least one of these will fail with mismatched `total_input_tokens` or `total_output_tokens`.

### F.3 Regression Tests

| Test Name | Guards Against |
|-----------|----------------|
| `TestSimulator_SeedZero` | Seed=0 edge case regression |
| `TestSimulator_SeedNegative` | Negative seed regression |
| `TestSimulator_SeedMaxInt64` | MaxInt64 seed regression |

### F.4 Golden Test

**Uses existing `testdata/goldendataset.json`** which contains 5 pre-computed test cases.

```go
func TestSimulator_GoldenDataset(t *testing.T) {
    // Load existing golden dataset
    dataset := loadGoldenDataset(t, "testdata/goldendataset.json")

    for _, tc := range dataset.Tests {
        t.Run(tc.Model, func(t *testing.T) {
            // Build simulator from test case config
            sim := NewSimulator(
                math.MaxInt64,           // horizon
                tc.Seed,                 // seed (always 42)
                tc.TotalKVBlocks,
                tc.BlockSizeInTokens,
                tc.MaxNumRunningReqs,
                tc.MaxNumScheduledTokens,
                tc.LongPrefillTokenThreshold,
                tc.BetaCoeffs,
                tc.AlphaCoeffs,
                &GuideLLMConfig{
                    Rate:               tc.Rate / 1e6,
                    MaxPrompts:         tc.MaxPrompts,
                    PrefixTokens:       tc.PrefixTokens,
                    PromptTokens:       tc.PromptTokens,
                    PromptTokensStdDev: tc.PromptTokensStdev,
                    PromptTokensMin:    tc.PromptTokensMin,
                    PromptTokensMax:    tc.PromptTokensMax,
                    OutputTokens:       tc.OutputTokens,
                    OutputTokensStdDev: tc.OutputTokensStdev,
                    OutputTokensMin:    tc.OutputTokensMin,
                    OutputTokensMax:    tc.OutputTokensMax,
                },
                // ... remaining config
            )
            sim.Run()

            // Verify against golden metrics (exact match required)
            assert.Equal(t, tc.Metrics.CompletedRequests, sim.Metrics.CompletedRequests,
                "completed_requests mismatch for %s", tc.Model)
            assert.Equal(t, tc.Metrics.TotalInputTokens, sim.Metrics.TotalInputTokens,
                "total_input_tokens mismatch for %s", tc.Model)
            assert.Equal(t, tc.Metrics.TotalOutputTokens, sim.Metrics.TotalOutputTokens,
                "total_output_tokens mismatch for %s", tc.Model)

            // Verify latency metrics (float comparison with tolerance)
            assert.InDelta(t, tc.Metrics.TTFTMeanMs,
                sim.Metrics.TTFTMean()/1000, 0.001,
                "ttft_mean mismatch for %s", tc.Model)
        })
    }
}
```

**Key verification points from golden dataset:**

| Metric | Verification | Why Critical |
|--------|--------------|--------------|
| `completed_requests` | Exact match | Proves same requests generated |
| `total_input_tokens` | Exact match | Proves same token lengths sampled |
| `total_output_tokens` | Exact match | Proves same output lengths sampled |
| `ttft_mean_ms` | Within 0.001 tolerance | Proves same scheduling behavior |

### F.5 Benchmark Test

```go
func BenchmarkPartitionedRNG_ForSubsystem(b *testing.B) {
    rng := NewPartitionedRNG(NewSimulationKey(42))
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        rng.ForSubsystem(SubsystemWorkload)
    }
}
// Target: < 50ns after initial call (map lookup only)
```

---

## G) Risk Analysis

### G.1 Invariant Break Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Hash collision in subsystem names | Very Low | Medium | FNV-1a has excellent distribution; subsystem names are few, short, distinct |
| XOR produces problematic seed | Very Low | Low | All int64 values are valid seeds |
| Breaking existing determinism | **HIGH** | **CRITICAL** | Golden test + 100-run verification |

### G.2 Performance Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Map lookup overhead | Very Low | Negligible | One lookup per call; cached after first |
| FNV-1a computation | Very Low | Negligible | One hash per subsystem lifetime |

**Benchmark target:** `ForSubsystem()` < 50ns on cache hit.

### G.3 Backward Compatibility Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Different random sequence for "workload" | **Must prevent** | Critical | Special case: "workload" uses masterSeed directly (no XOR) |
| Different sequence for non-workload subsystems | N/A | None | No other subsystems used in PR1 |

**Why backward compatibility works:**

```
BEFORE PR1:
  rand.NewSource(seed) → RNG

AFTER PR1:
  ForSubsystem("workload") → derivedSeed = masterSeed (no XOR) → rand.NewSource(masterSeed) → RNG

Result: Identical RNG state, identical sequence.
```

**Golden dataset provides strong verification:**

The existing `testdata/goldendataset.json` contains 5 diverse test cases:
- 4 different models (llama, qwen, granite, codellama)
- Different token distributions (prefix vs no-prefix, short vs long prompts)
- Different KV block configurations (2,537 to 1,281,766 blocks)
- All use `seed: 42`

If PR1 breaks RNG determinism, at least one test will fail with:
- Mismatched `total_input_tokens` (wrong prompt lengths sampled)
- Mismatched `total_output_tokens` (wrong output lengths sampled)
- Mismatched `completed_requests` (different request generation)

### G.4 Hidden Coupling Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Other code using `randomNumberGenerator` | Low | Medium | Grep found only 2 call sites; both updated |
| Tests accessing internal field | Low | Low | Search test files; update if found |

### G.5 Observability Gaps

None identified. RNG is internal implementation detail; no logging needed.

---

## H) Sanity Checklist

| Check | Status | Notes |
|-------|--------|-------|
| No unnecessary abstractions | ✅ | Minimal types: SimulationKey (type alias), PartitionedRNG (struct) |
| No feature creep beyond PR scope | ✅ | Only RNG partitioning; no multi-instance |
| No unexercised flags | ✅ | No new CLI flags |
| No partial implementations | ✅ | PartitionedRNG is complete and functional |
| No breaking changes | ✅ | Backward compatible via "workload" special case |
| No hidden global state impact | ✅ | No globals modified |
| All production code exercised via CLI | ✅ | Every run uses ForSubsystem("workload") |
| BDD scenarios comprehensive | ✅ | 8 scenarios covering all contracts |
| Follows existing code style | ✅ | Matches sim/ package conventions |
| Tests map to contracts | ✅ | Each test references specific contract |

---

## Appendix: File Summary

| File | Action | LOC |
|------|--------|-----|
| `sim/rng.go` | Create | ~100 |
| `sim/rng_test.go` | Create | ~120 |
| `sim/simulator.go` | Modify | +8 / -3 |
| `sim/workload_config.go` | Modify | +2 / -2 |
| `sim/simulator_test.go` | Modify | +60 (golden dataset test) |
| `testdata/goldendataset.json` | **Existing** | (no changes) |

**Total:** ~130 production LOC, ~180 test LOC

---

## Appendix: Dependencies

| Relationship | PR |
|--------------|-----|
| **Depends on** | None (first PR) |
| **Blocks** | PR 2 (InstanceSimulator uses SubsystemInstance) |
| **Blocks** | PR 3 (ClusterSimulator uses PartitionedRNG) |
| **Parallel with** | None (sequential in Phase 1) |
