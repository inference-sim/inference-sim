# Hardening Batch: Library-Level Input Validation (R3/R1)

- **Goal:** Add missing input validation at library boundaries for KV cache and batch configuration, and fix a thrashing detection false positive.
- **The problem today:** Several constructor functions accept invalid inputs without error. `NewTieredKVCache` accepts negative `cpuBlocks`, `NewKVStore` doesn't validate threshold range or bandwidth when tiered mode is active, and `NewSimulator` doesn't validate `BatchConfig` fields. Library callers bypass CLI validation, leading to silent misbehavior (permanently full CPU tier, zero-throughput simulations, false thrashing metrics).
- **What this PR adds:**
  1. `NewTieredKVCache` panics on `cpuBlocks <= 0` (R3)
  2. Thrashing detection skips counting when clock has never advanced past 0 (bug fix)
  3. `NewKVStore` validates threshold ∈ [0,1] and bandwidth > 0 when tiered mode is active (R3)
  4. `NewSimulator` returns error for invalid `BatchConfig` values (R3, R1)
- **Why this matters:** Library callers (tests, future API consumers) get immediate feedback on invalid configuration instead of silent misbehavior.
- **Architecture:** All changes are in existing constructors/factories — no new types, interfaces, or packages.
- **Source:** Issues #508, #509, #382, #383. Issue #384 will be closed as already resolved (WorkloadConfig is now empty).
- **Closes:** Fixes #508, fixes #509, fixes #382, fixes #383, fixes #384
- **Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building blocks modified:** KV cache constructors (`sim/kv/tiered.go`, `sim/kv/register.go`), core simulator constructor (`sim/simulator.go`)
2. **Adjacent blocks:** CLI layer (`cmd/root.go`) already validates these at the CLI boundary; this PR closes the library boundary gap.
3. **Invariants touched:** None directly — these are pre-construction validation, not runtime behavior changes.
4. **Construction Site Audit:**
   - `NewTieredKVCache` — called only from `NewKVStore` (`sim/kv/register.go:25`) + tests
   - `NewKVStore` — registered via `sim.NewKVStoreFromConfig` (`sim/kv/register.go:14`), called from `cmd/root.go` and `sim/cluster/instance.go`
   - `NewSimulator` — called from `cmd/root.go` and `sim/cluster/instance.go`

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds library-level input validation to three constructors that currently accept invalid inputs silently. The KV cache constructors (`NewTieredKVCache`, `NewKVStore`) get panic-based validation matching the existing `NewKVCacheState` pattern. The simulator constructor (`NewSimulator`) gets error-return validation matching its existing signature. A thrashing detection false positive at clock=0 is also fixed. No runtime behavior changes for valid inputs.

### B) Behavioral Contracts

**Positive contracts (what MUST happen):**

```
BC-1: NewTieredKVCache rejects non-positive cpuBlocks
- GIVEN cpuBlocks <= 0
- WHEN NewTieredKVCache is called
- THEN it panics with a message containing "cpuBlocks" and the invalid value
- MECHANISM: Explicit check before struct construction
```

```
BC-2: Thrashing detection skips clock=0
- GIVEN a TieredKVCache where SetClock has never been called (clock=0)
- WHEN a block is reloaded from CPU to GPU
- THEN thrashingCount is NOT incremented
- AND KVThrashingRate() returns 0
- MECHANISM: Guard condition t.clock > 0 before thrashing check
```

```
BC-3: NewKVStore validates threshold range for tiered mode
- GIVEN KVCPUBlocks > 0 AND KVOffloadThreshold outside [0,1] OR is NaN
- WHEN NewKVStore is called
- THEN it panics with a message containing "KVOffloadThreshold" and the invalid value
- MECHANISM: Range check + NaN guard before delegating to NewTieredKVCache
```

```
BC-4: NewKVStore validates bandwidth for tiered mode
- GIVEN KVCPUBlocks > 0 AND KVTransferBandwidth <= 0 OR is NaN/Inf
- WHEN NewKVStore is called
- THEN it panics with a message containing "KVTransferBandwidth" and the invalid value
- MECHANISM: Positive-value + NaN/Inf check before delegating to NewTieredKVCache
```

```
BC-5: NewSimulator rejects non-positive MaxRunningReqs
- GIVEN BatchConfig.MaxRunningReqs <= 0
- WHEN NewSimulator is called
- THEN it returns a non-nil error containing "MaxRunningReqs"
- MECHANISM: Validation at the start of NewSimulator
```

```
BC-6: NewSimulator rejects non-positive MaxScheduledTokens
- GIVEN BatchConfig.MaxScheduledTokens <= 0
- WHEN NewSimulator is called
- THEN it returns a non-nil error containing "MaxScheduledTokens"
- MECHANISM: Validation at the start of NewSimulator
```

```
BC-7: NewSimulator rejects negative LongPrefillTokenThreshold
- GIVEN BatchConfig.LongPrefillTokenThreshold < 0
- WHEN NewSimulator is called
- THEN it returns a non-nil error containing "LongPrefillTokenThreshold"
- MECHANISM: Validation at the start of NewSimulator (0 means disabled per vLLM semantics)
```

**Non-regression:** All existing tests must pass unchanged (or with minimal updates for tests that used zero-valued BatchConfig). This is verified by running `go test ./...`, not by a new contract.

### C) Component Interaction

```
cmd/root.go (CLI validation: logrus.Fatalf)
    │
    ├──▶ NewKVStore(KVCacheConfig)     ← BC-3, BC-4: panic on invalid tiered params
    │       └──▶ NewTieredKVCache(...)  ← BC-1: panic on cpuBlocks <= 0
    │                                     BC-2: clock=0 thrashing guard
    │
    └──▶ NewSimulator(cfg, kv, lm)     ← BC-5, BC-6, BC-7: error on invalid BatchConfig
```

CLI validation is the first line of defense (user-facing). Library validation (this PR) is the second (developer-facing). Both use the same rules but different error mechanisms (Fatalf vs panic/error).

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #383: Move bandwidth validation into NewKVStore | Add redundant bandwidth check in NewKVStore AND keep existing check in NewTieredKVCache | ADDITION: Both layers validate — NewKVStore gives KVCacheConfig-level message, NewTieredKVCache gives parameter-level message |
| #383: Change "(default X)" comments | Already done in prior PR | NO-OP: Comments already say "CLI default:" |
| #384: Add mutual exclusion validation | Close issue as resolved | SCOPE_CHANGE: WorkloadConfig is now empty — fields removed in W0-4 (#420) |

### E) Review Guide

**Tricky part:** BC-2 (thrashing false positive). The `t.clock > 0` guard means thrashing at the literal start of simulation (clock=0) goes undetected, but this is a non-scenario: Step() calls SetClock(now) before any batch formation, and offload/reload only happens during batch processing. The first realistic offload+reload cycle is at clock > 0.

**Scrutinize:** BC-3 threshold range — [0,1] is correct: 0 means "always offload" (aggressive CPU usage), 1 means "never offload" (CPU tier never used). Both are valid edge cases.

**Safe to skim:** BC-5/6/7 are straightforward nil-check-style validation in a function that already returns error.

**Known debt:** NaN/Inf validation for BatchConfig int64 fields is not needed (Go int64 cannot be NaN/Inf).

---

## Part 2: Executable Implementation

### F) Implementation Overview

| File | Action | Purpose |
|------|--------|---------|
| `sim/kv/tiered.go` | Modify | BC-1 (cpuBlocks panic), BC-2 (clock guard) |
| `sim/kv/tiered_test.go` | Modify | Tests for BC-1, BC-2 |
| `sim/kv/register.go` | Modify | BC-3 (threshold), BC-4 (bandwidth) |
| `sim/kv/register_test.go` | Create | Tests for BC-3, BC-4 |
| `sim/simulator.go` | Modify | BC-5, BC-6, BC-7 |
| `sim/simulator_test.go` | Modify | Tests for BC-5, BC-6, BC-7 |

No dead code. No new types or interfaces. No CLAUDE.md changes needed.

### G) Task Breakdown

#### Task 1: BC-1 — NewTieredKVCache cpuBlocks validation (#508)

**Contracts:** BC-1

**Test (sim/kv/tiered_test.go):**
```go
func TestNewTieredKVCache_ZeroCPUBlocks_Panics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for cpuBlocks=0")
		}
		msg := fmt.Sprintf("%v", r)
		if !strings.Contains(msg, "cpuBlocks") {
			t.Errorf("panic message should mention cpuBlocks, got: %s", msg)
		}
	}()
	NewTieredKVCache(NewKVCacheState(10, 2), 0, 0.5, 100.0, 0)
}

func TestNewTieredKVCache_NegativeCPUBlocks_Panics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for cpuBlocks=-5")
		}
		msg := fmt.Sprintf("%v", r)
		if !strings.Contains(msg, "cpuBlocks") {
			t.Errorf("panic message should mention cpuBlocks, got: %s", msg)
		}
	}()
	NewTieredKVCache(NewKVCacheState(10, 2), -5, 0.5, 100.0, 0)
}
```

**Implementation (sim/kv/tiered.go — add after threshold validation, before return):**
```go
if cpuBlocks <= 0 {
    panic(fmt.Sprintf("NewTieredKVCache: cpuBlocks must be > 0, got %d", cpuBlocks))
}
```

Also update the doc comment on `NewTieredKVCache` (line 47) from:
```
// Panics if gpu is nil, bandwidth is non-positive/NaN/Inf, or threshold is NaN/Inf.
```
To:
```
// Panics if gpu is nil, cpuBlocks is non-positive, bandwidth is non-positive/NaN/Inf, or threshold is NaN/Inf.
```

Add `"strings"` to imports in tiered_test.go (needed for `strings.Contains`).

**Commands:**
```bash
go test ./sim/kv/... -run TestNewTieredKVCache_ZeroCPUBlocks_Panics -v
go test ./sim/kv/... -run TestNewTieredKVCache_NegativeCPUBlocks_Panics -v
golangci-lint run ./sim/kv/...
```

#### Task 2: BC-2 — Thrashing detection clock=0 guard (#509)

**Contracts:** BC-2

**Test (sim/kv/tiered_test.go):**
```go
func TestTieredKVCache_ThrashingNotCounted_WhenClockNeverSet(t *testing.T) {
	// BC-2: GIVEN a TieredKVCache where SetClock has never been called
	gpu := NewKVCacheState(10, 2)
	tiered := NewTieredKVCache(gpu, 10, 0.3, 100.0, 0)
	// Note: no SetClock() call — clock stays at 0

	// Allocate and release to trigger offload
	target := &sim.Request{ID: "target", InputTokens: []int{1, 2, 3, 4}}
	tiered.AllocateKVBlocks(target, 0, 4, []int64{})
	for i := 0; i < 3; i++ {
		other := &sim.Request{ID: fmt.Sprintf("o%d", i), InputTokens: []int{i*4 + 10, i*4 + 11, i*4 + 12, i*4 + 13}}
		tiered.AllocateKVBlocks(other, 0, 4, []int64{})
	}
	tiered.ReleaseKVBlocks(target)
	if tiered.offloadCount == 0 {
		t.Fatal("setup error: offload should have triggered")
	}

	// Fill GPU to force CPU reload
	for i := 0; i < 3; i++ {
		filler := &sim.Request{ID: fmt.Sprintf("f%d", i), InputTokens: []int{i*2 + 100, i*2 + 101}}
		tiered.AllocateKVBlocks(filler, 0, 2, []int64{})
	}

	// Re-request same prefix — triggers CPU reload
	sameReq := &sim.Request{ID: "retry", InputTokens: []int{1, 2, 3, 4}}
	cached := tiered.GetCachedBlocks([]int{1, 2, 3, 4})
	start := int64(len(cached)) * tiered.BlockSize()
	tiered.AllocateKVBlocks(sameReq, start, 4, cached)

	// THEN thrashing should NOT be counted (clock was never set)
	if tiered.KVThrashingRate() != 0 {
		t.Errorf("KVThrashingRate() = %f, want 0 when clock was never set", tiered.KVThrashingRate())
	}
}
```

**Implementation (sim/kv/tiered.go line 154):**
Change:
```go
if t.clock-offloaded.OffloadTime < 1000 {
```
To:
```go
if t.clock > 0 && t.clock-offloaded.OffloadTime < 1000 {
```

**Commands:**
```bash
go test ./sim/kv/... -run TestTieredKVCache_ThrashingNotCounted -v
go test ./sim/kv/... -v  # ensure existing thrashing test still passes
golangci-lint run ./sim/kv/...
```

#### Task 3: BC-3, BC-4 — NewKVStore tiered-mode validation (#383)

**Contracts:** BC-3, BC-4

**Test (sim/kv/register_test.go — new file):**
```go
package kv

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestNewKVStore_TieredMode_ThresholdOutOfRange_Panics(t *testing.T) {
	tests := []struct {
		name      string
		threshold float64
	}{
		{"negative", -0.1},
		{"above_one", 1.1},
		{"NaN", math.NaN()},
		{"pos_inf", math.Inf(1)},
		{"neg_inf", math.Inf(-1)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for threshold=%v", tc.threshold)
				}
				msg := fmt.Sprintf("%v", r)
				if !strings.Contains(msg, "KVOffloadThreshold") {
					t.Errorf("panic message should mention KVOffloadThreshold, got: %s", msg)
				}
			}()
			cfg := sim.NewKVCacheConfig(10, 2, 5, tc.threshold, 100.0, 0)
			NewKVStore(cfg)
		})
	}
}

func TestNewKVStore_TieredMode_InvalidBandwidth_Panics(t *testing.T) {
	tests := []struct {
		name      string
		bandwidth float64
	}{
		{"zero", 0},
		{"negative", -1.0},
		{"NaN", math.NaN()},
		{"pos_inf", math.Inf(1)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected panic for bandwidth=%v", tc.bandwidth)
				}
				msg := fmt.Sprintf("%v", r)
				if !strings.Contains(msg, "KVTransferBandwidth") {
					t.Errorf("panic message should mention KVTransferBandwidth, got: %s", msg)
				}
			}()
			cfg := sim.NewKVCacheConfig(10, 2, 5, 0.5, tc.bandwidth, 0)
			NewKVStore(cfg)
		})
	}
}

func TestNewKVStore_TieredMode_ValidEdgeCases(t *testing.T) {
	// Threshold=0 (always offload) and threshold=1 (never offload) are both valid
	tests := []struct {
		name      string
		threshold float64
	}{
		{"threshold_zero", 0.0},
		{"threshold_one", 1.0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := sim.NewKVCacheConfig(10, 2, 5, tc.threshold, 100.0, 0)
			store := NewKVStore(cfg)
			if store == nil {
				t.Fatal("NewKVStore should return non-nil for valid config")
			}
		})
	}
}

func TestNewKVStore_SingleTier_SkipsValidation(t *testing.T) {
	// When KVCPUBlocks <= 0, tiered-mode validation does not apply
	cfg := sim.NewKVCacheConfig(10, 2, 0, -999.0, -999.0, 0)
	store := NewKVStore(cfg)
	if store == nil {
		t.Fatal("NewKVStore should return non-nil for single-tier mode")
	}
}
```

**Implementation (sim/kv/register.go — add before NewTieredKVCache call):**
```go
func NewKVStore(cfg sim.KVCacheConfig) sim.KVStore {
	gpu := NewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
	if cfg.KVCPUBlocks <= 0 {
		return gpu
	}
	// Validate tiered-mode parameters at the KVCacheConfig level (R3)
	if cfg.KVOffloadThreshold < 0 || cfg.KVOffloadThreshold > 1 || math.IsNaN(cfg.KVOffloadThreshold) {
		panic(fmt.Sprintf("NewKVStore: KVOffloadThreshold must be in [0,1] when KVCPUBlocks > 0, got %v", cfg.KVOffloadThreshold))
	}
	if cfg.KVTransferBandwidth <= 0 || math.IsNaN(cfg.KVTransferBandwidth) || math.IsInf(cfg.KVTransferBandwidth, 0) {
		panic(fmt.Sprintf("NewKVStore: KVTransferBandwidth must be finite and > 0 when KVCPUBlocks > 0, got %v", cfg.KVTransferBandwidth))
	}
	return NewTieredKVCache(gpu, cfg.KVCPUBlocks, cfg.KVOffloadThreshold,
		cfg.KVTransferBandwidth, cfg.KVTransferBaseLatency)
}
```

**Commands:**
```bash
go test ./sim/kv/... -run TestNewKVStore -v
golangci-lint run ./sim/kv/...
```

#### Task 4: BC-5, BC-6, BC-7 — NewSimulator BatchConfig validation (#382)

**Contracts:** BC-5, BC-6, BC-7

**Test (sim/simulator_test.go):**
```go
func TestNewSimulator_BatchConfigValidation(t *testing.T) {
	tests := []struct {
		name           string
		maxRunning     int64
		maxTokens      int64
		prefillThresh  int64
		wantErrContain string
	}{
		{"zero_max_running", 0, 2048, 0, "MaxRunningReqs"},
		{"negative_max_running", -1, 2048, 0, "MaxRunningReqs"},
		{"zero_max_tokens", 256, 0, 0, "MaxScheduledTokens"},
		{"negative_max_tokens", 256, -1, 0, "MaxScheduledTokens"},
		{"negative_prefill_threshold", 256, 2048, -1, "LongPrefillTokenThreshold"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := newTestSimConfig()
			cfg.BatchConfig = NewBatchConfig(tc.maxRunning, tc.maxTokens, tc.prefillThresh)
			kvStore := MustNewKVStoreFromConfig(cfg.KVCacheConfig)
			latencyModel, err := MustNewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
			if err != nil {
				t.Fatalf("MustNewLatencyModel: %v", err)
			}
			_, err = NewSimulator(cfg, kvStore, latencyModel)
			if err == nil {
				t.Fatalf("expected error for %s", tc.name)
			}
			if !strings.Contains(err.Error(), tc.wantErrContain) {
				t.Errorf("error %q should contain %q", err.Error(), tc.wantErrContain)
			}
		})
	}
}
```

**Implementation (sim/simulator.go — add at start of NewSimulator):**
```go
if cfg.MaxRunningReqs <= 0 {
    return nil, fmt.Errorf("NewSimulator: MaxRunningReqs must be > 0, got %d", cfg.MaxRunningReqs)
}
if cfg.MaxScheduledTokens <= 0 {
    return nil, fmt.Errorf("NewSimulator: MaxScheduledTokens must be > 0, got %d", cfg.MaxScheduledTokens)
}
if cfg.LongPrefillTokenThreshold < 0 {
    return nil, fmt.Errorf("NewSimulator: LongPrefillTokenThreshold must be >= 0, got %d", cfg.LongPrefillTokenThreshold)
}
```

**Fix existing test:** `TestNewSimulator_NonRooflineZeroNumHeads_Succeeds` at `simulator_test.go:1279` constructs SimConfig without BatchConfig (zero-valued → MaxRunningReqs=0). Add `BatchConfig: NewBatchConfig(256, 2048, 0)` to that test's SimConfig.

Add `"strings"` to `simulator_test.go` imports (needed for `strings.Contains` in the new test).

**Commands:**
```bash
go test ./sim/... -run TestNewSimulator_BatchConfigValidation -v
go test ./sim/... -run TestNewSimulator_NonRooflineZeroNumHeads -v  # verify fix
go test ./sim/... -v  # ensure all existing tests still pass
golangci-lint run ./sim/...
```

#### Task 5: Close #384 as resolved

**No code changes.** WorkloadConfig is now an empty struct — the mutual exclusion fields (`GuideLLMConfig`, `TracesWorkloadFilePath`) were removed during workload unification (W0-4). PR description will reference `Fixes #384`.

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit (panic) | TestNewTieredKVCache_ZeroCPUBlocks_Panics |
| BC-1 | Task 1 | Unit (panic) | TestNewTieredKVCache_NegativeCPUBlocks_Panics |
| BC-2 | Task 2 | Unit (behavioral) | TestTieredKVCache_ThrashingNotCounted_WhenClockNeverSet |
| BC-3 | Task 3 | Unit (panic) | TestNewKVStore_TieredMode_ThresholdOutOfRange_Panics |
| BC-4 | Task 3 | Unit (panic) | TestNewKVStore_TieredMode_ZeroBandwidth_Panics |
| BC-3,4 | Task 3 | Unit (positive) | TestNewKVStore_TieredMode_ValidEdgeCases |
| BC-3,4 | Task 3 | Unit (negative) | TestNewKVStore_SingleTier_SkipsValidation |
| BC-5 | Task 4 | Unit (error) | TestNewSimulator_BatchConfigValidation (zero/negative maxRunning) |
| BC-6 | Task 4 | Unit (error) | TestNewSimulator_BatchConfigValidation (zero/negative maxTokens) |
| BC-7 | Task 4 | Unit (error) | TestNewSimulator_BatchConfigValidation (negative prefillThreshold) |
| Non-regression | All | Regression | `go test ./...` passes (existing tests + new) |

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Existing tests construct with cpuBlocks=0 | Low | Medium (test break) | Check: `NewTieredKVCache` is only called from `NewKVStore` which guards `KVCPUBlocks > 0` | Task 1 |
| Existing tests rely on thrashing count at clock=0 | Low | Low (test assertion change) | Check existing thrashing test — it calls SetClock(100) explicitly | Task 2 |
| NewSimulator tests pass 0 for MaxRunningReqs | Medium | Medium | Grep all NewSimulator/SimConfig construction sites in tests for zero BatchConfig values | Task 4 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] R1: No silent continue/return dropping data
- [x] R3: Validation at library boundary matching CLI validation
- [x] R4: No new struct fields
- [x] R6: No Fatalf in sim/ (errors and panics only)
- [x] CLAUDE.md: No updates needed (no new files/packages/flags)

---

## Appendix: File-Level Implementation Details

**File: `sim/kv/tiered.go`**
- Add `cpuBlocks <= 0` panic after existing threshold validation (line ~57)
- Add `t.clock > 0 &&` guard to thrashing detection (line 154)
- Error handling: panic (matches existing constructor pattern)

**File: `sim/kv/tiered_test.go`**
- Add 3 new test functions (BC-1 × 2, BC-2 × 1)

**File: `sim/kv/register.go`**
- Add `fmt` and `math` imports
- Add threshold range [0,1] and bandwidth > 0 checks before `NewTieredKVCache` call
- Error handling: panic (matches `NewKVCacheState` pattern)

**File: `sim/kv/register_test.go`** (NEW)
- 4 test functions for BC-3, BC-4, edge cases, single-tier bypass

**File: `sim/simulator.go`**
- Add 3 validation checks at start of `NewSimulator` (after kvStore/latencyModel nil checks)
- Error handling: return error (matches existing `NewSimulator` signature)

**File: `sim/simulator_test.go`**
- Add `"strings"` import
- Table-driven test for BatchConfig validation (5 subtests)
- Fix `TestNewSimulator_NonRooflineZeroNumHeads_Succeeds` to include `BatchConfig`
