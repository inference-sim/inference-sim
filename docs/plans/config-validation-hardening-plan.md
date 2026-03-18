# Config Validation Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add constructor-level validation to `NewKVCacheConfig` and mutual exclusion enforcement to `WorkloadSpec`, eliminating two classes of silent misconfiguration at the library boundary.

**The problem today:** A library caller can construct a `KVCacheConfig` with `TotalKVBlocks=0` or `BlockSizeTokens=-1` and get no error until deep inside `NewKVCacheState` — with an unhelpful panic message that doesn't mention `KVCacheConfig`. Similarly, a `WorkloadSpec` with both `InferencePerf` and `ServeGenData` set silently drops one source (R1 violation) instead of erroring.

**What this PR adds:**
1. Constructor validation in `NewKVCacheConfig` — panics with clear messages for invalid parameters, matching the pattern established by `NewBatchConfig`
2. Mutual exclusion check in `GenerateRequests` — errors when multiple workload sources (Clients, ServeGenData, InferencePerf) coexist, preventing silent data loss

**Why this matters:** Defense-in-depth at the library boundary catches misconfiguration early with clear messages. The mutual exclusion fix closes the last known R1 (silent data loss) violation in the workload pipeline.

**Architecture:** `NewKVCacheConfig` in `sim/config.go` gains validation panics (library convention for constructor invariants). `GenerateRequests` in `sim/workload/generator.go` gains a pre-expansion mutual exclusion check. No new types, interfaces, or packages.

**Source:** GitHub issues #736, #737 (follow-ups from closed #383, #384)

**Closes:** Fixes #736, fixes #737

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds validation to two existing constructors/entry points:

1. **`NewKVCacheConfig`** (`sim/config.go`) — currently a pure pass-through with zero validation. Gets panic-based validation matching `NewBatchConfig`'s pattern: `TotalKVBlocks > 0`, `BlockSizeTokens > 0`, `KVCPUBlocks >= 0`, plus conditional tiered-mode checks.

2. **`GenerateRequests`** (`sim/workload/generator.go`) — currently applies silent precedence when multiple workload sources coexist. Gets an explicit mutual exclusion check before expansion.

Adjacent blocks: `NewKVStore` (sim/kv/register.go) already validates tiered-mode params — this PR adds defense-in-depth at the constructor level. `GenerateRequests` is the single entry point for all workload generation.

See Deviation Log (Section D) for justified deviations.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: KVCacheConfig rejects invalid GPU-tier parameters
- GIVEN `TotalKVBlocks <= 0` OR `BlockSizeTokens <= 0`
- WHEN `NewKVCacheConfig` is called
- THEN it panics with a message naming the invalid parameter and its value
- MECHANISM: Validation checks at top of constructor, before struct literal

BC-2: KVCacheConfig rejects negative CPU block count
- GIVEN `KVCPUBlocks < 0`
- WHEN `NewKVCacheConfig` is called
- THEN it panics with a message naming `KVCPUBlocks` and its value
- MECHANISM: Non-negative check in constructor

BC-3: KVCacheConfig rejects invalid tiered-mode parameters
- GIVEN `KVCPUBlocks > 0` AND (`KVTransferBandwidth <= 0` OR `KVTransferBandwidth` is NaN/Inf OR `KVTransferBaseLatency < 0`)
- WHEN `NewKVCacheConfig` is called
- THEN it panics with a message naming the invalid parameter
- MECHANISM: Conditional validation block when KVCPUBlocks > 0. Note: `KVOffloadThreshold` is NOT validated — it is deprecated and ignored in the vLLM v1 mirror model (sim/kv/tiered.go). Existing `NewKVStore` validates it for legacy reasons, but the constructor should not tighten a deprecated contract.

BC-4: KVCacheConfig accepts valid single-tier parameters
- GIVEN `TotalKVBlocks > 0` AND `BlockSizeTokens > 0` AND `KVCPUBlocks == 0`
- WHEN `NewKVCacheConfig` is called with any threshold/bandwidth/latency values
- THEN it returns a KVCacheConfig with all fields set to the provided values (no defaults injected)
- MECHANISM: Tiered-mode checks are conditional on KVCPUBlocks > 0; single-tier skips them

BC-5: KVCacheConfig accepts valid tiered-mode parameters
- GIVEN `TotalKVBlocks > 0`, `BlockSizeTokens > 0`, `KVCPUBlocks > 0`, bandwidth > 0 and finite, latency >= 0
- WHEN `NewKVCacheConfig` is called
- THEN it returns a KVCacheConfig with all fields set to the provided values

BC-6: WorkloadSpec mutual exclusion enforcement
- GIVEN a `WorkloadSpec` with more than one workload source set (any combination of: `len(Clients) > 0`, `ServeGenData != nil`, `InferencePerf != nil`)
- WHEN `GenerateRequests` is called
- THEN it returns an error naming the conflicting sources
- MECHANISM: Pre-expansion check at top of GenerateRequests, before InferencePerf/ServeGenData expansion

BC-7: Cohorts compose with any primary source
- GIVEN a `WorkloadSpec` with `Cohorts` set alongside any single primary source (`Clients`, `InferencePerf`, or `ServeGenData`)
- WHEN `GenerateRequests` is called
- THEN it succeeds (cohorts expand into additional clients, this is intentional composition)
- MECHANISM: Mutual exclusion check only applies to {Clients, ServeGenData, InferencePerf}

**Negative Contracts:**

BC-8: No silent data loss
- GIVEN any combination of multiple workload sources
- WHEN `GenerateRequests` is called
- THEN it MUST NOT silently skip any source — it MUST return an error
- MECHANISM: Mutual exclusion check runs before any expansion logic

**Error Handling Contracts:**

BC-9: Constructor panics use library convention
- GIVEN invalid parameters to `NewKVCacheConfig`
- WHEN the constructor panics
- THEN the panic message includes the constructor name, parameter name, and actual value
- MECHANISM: `fmt.Sprintf("NewKVCacheConfig: <Param> must be <constraint>, got %v", value)`

### C) Component Interaction

```
CLI (cmd/root.go, cmd/replay.go)
  │ validates flags via logrus.Fatalf
  ▼
NewKVCacheConfig(...)         ← THIS PR: adds constructor validation (panics)
  │
  ▼
NewKVStore(KVCacheConfig)     ← existing: validates tiered-mode params (panics)
  │
  ▼
NewKVCacheState / NewTieredKVCache  ← existing: validates blocks > 0 (panics)
```

```
WorkloadSpec (from YAML)
  │
  ▼
GenerateRequests(spec, ...)   ← THIS PR: mutual exclusion check (returns error)
  │ expands InferencePerf / ServeGenData into Clients
  ▼
spec.Validate()               ← existing: validates individual fields
```

No new interfaces, types, or state. Extension friction: 0 (no new fields).

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #736 recommends threshold/bandwidth validation in constructor | Plan validates bandwidth/latency but NOT threshold | CLARIFICATION: KVOffloadThreshold is deprecated and ignored in vLLM v1 mirror model (sim/kv/tiered.go). NewKVStore validates it for legacy reasons, but the constructor should not tighten a deprecated contract. Bandwidth and latency are still active. |
| #737 recommends validation in Validate() | Plan adds check in GenerateRequests() before expansion | CORRECTION: Validate() runs after expansion, so it can't distinguish user-set Clients from expanded ones |
| #736 does not mention KVTransferBaseLatency | Plan validates KVTransferBaseLatency >= 0 in constructor (tiered-mode only) | ADDITION: Neither CLI nor NewKVStore validates this; negative base latency is nonsensical. Constructor is the only validation point. |
| #737 recommends InferencePerf field validation | Plan does not add it | CLARIFICATION: `validateInferencePerfSpec` is already called inside `ExpandInferencePerfSpec` (sim/workload/inference_perf.go:84). Already satisfied. |
| N/A | Mutual exclusion check makes spec mutation observable | ADDITION: `GenerateRequests` already mutates `spec.Clients` in place (line 30). Pre-existing non-idempotency — second call with same InferencePerf spec would trigger the new check. This is correct behavior (exposes pre-existing mutation), not a regression. |

### E) Review Guide

**The tricky part:** BC-4 (single-tier accepts any threshold/bandwidth values). Many existing tests use `NewKVCacheConfig(X, Y, 0, 0, 0, 0)` — the zeros for threshold/bandwidth MUST NOT trigger validation when KVCPUBlocks is 0. The conditional check is critical.

**What to scrutinize:** The mutual exclusion check in BC-6 — ensure Cohorts are excluded from the check (they compose with Clients by design). Verify the error message is actionable.

**What's safe to skim:** The panic message formatting (BC-9) — mechanical pattern matching NewBatchConfig.

**Known debt:** The existing `TestNewKVCacheConfig_ZeroValues_NoDefaults` test (config_test.go:70-74) passes all zeros. This test must be replaced with a panic test, since TotalKVBlocks=0 is now invalid.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/config.go:18-29` — Add validation to `NewKVCacheConfig`
- `sim/config_test.go:70-74` — Replace zero-value test with panic tests
- `sim/workload/generator.go:23` — Add mutual exclusion check (after maxRequests validation)
- `sim/workload/generator_test.go` — Add mutual exclusion tests (existing file, same package)

**Key decisions:**
- Constructor uses `panic` (library convention, matches NewBatchConfig)
- Mutual exclusion uses `error` return (matches GenerateRequests convention)
- Tiered-mode validation conditional on `KVCPUBlocks > 0` (matches NewKVStore)

**Confirmation:** No dead code. All validation paths exercised by tests. All existing tests remain valid.

### G) Task Breakdown

---

### Task 1: NewKVCacheConfig validation — GPU-tier parameters (BC-1, BC-9)

**Contracts Implemented:** BC-1, BC-9

**Files:**
- Modify: `sim/config.go:18-29`
- Test: `sim/config_test.go`

**Step 1: Write failing test for invalid GPU-tier parameters**

Context: Table-driven test matching NewBatchConfig_PanicsOnInvalid pattern.

```go
func TestNewKVCacheConfig_PanicsOnInvalid(t *testing.T) {
	tests := []struct {
		name            string
		totalKVBlocks   int64
		blockSizeTokens int64
		kvCPUBlocks     int64
		threshold       float64
		bandwidth       float64
		baseLatency     int64
		wantContains    string
	}{
		{"zero_total_kv_blocks", 0, 16, 0, 0, 0, 0, "TotalKVBlocks"},
		{"negative_total_kv_blocks", -1, 16, 0, 0, 0, 0, "TotalKVBlocks"},
		{"zero_block_size", 100, 0, 0, 0, 0, 0, "BlockSizeTokens"},
		{"negative_block_size", 100, -1, 0, 0, 0, 0, "BlockSizeTokens"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("expected panic")
				}
				msg := fmt.Sprintf("%v", r)
				if !strings.Contains(msg, tc.wantContains) {
					t.Errorf("panic message %q should contain %q", msg, tc.wantContains)
				}
				if !strings.Contains(msg, "NewKVCacheConfig") {
					t.Errorf("panic message %q should contain constructor name", msg)
				}
			}()
			NewKVCacheConfig(tc.totalKVBlocks, tc.blockSizeTokens, tc.kvCPUBlocks,
				tc.threshold, tc.bandwidth, tc.baseLatency)
		})
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestNewKVCacheConfig_PanicsOnInvalid -v`
Expected: FAIL (no panic — constructor is a pass-through)

**Step 3: Implement validation in NewKVCacheConfig**

In `sim/config.go`, replace the existing `NewKVCacheConfig` body:

```go
func NewKVCacheConfig(totalKVBlocks, blockSizeTokens, kvCPUBlocks int64,
	kvOffloadThreshold, kvTransferBandwidth float64,
	kvTransferBaseLatency int64) KVCacheConfig {
	if totalKVBlocks <= 0 {
		panic(fmt.Sprintf("NewKVCacheConfig: TotalKVBlocks must be > 0, got %d", totalKVBlocks))
	}
	if blockSizeTokens <= 0 {
		panic(fmt.Sprintf("NewKVCacheConfig: BlockSizeTokens must be > 0, got %d", blockSizeTokens))
	}
	if kvCPUBlocks < 0 {
		panic(fmt.Sprintf("NewKVCacheConfig: KVCPUBlocks must be >= 0, got %d", kvCPUBlocks))
	}
	if kvCPUBlocks > 0 {
		// Note: KVOffloadThreshold is NOT validated here — it is deprecated and
		// ignored in the vLLM v1 mirror model. NewKVStore validates it for legacy
		// reasons, but the constructor should not tighten a deprecated contract.
		if kvTransferBandwidth <= 0 || math.IsNaN(kvTransferBandwidth) || math.IsInf(kvTransferBandwidth, 0) {
			panic(fmt.Sprintf("NewKVCacheConfig: KVTransferBandwidth must be finite and > 0 when KVCPUBlocks > 0, got %v", kvTransferBandwidth))
		}
		if kvTransferBaseLatency < 0 {
			panic(fmt.Sprintf("NewKVCacheConfig: KVTransferBaseLatency must be >= 0 when KVCPUBlocks > 0, got %d", kvTransferBaseLatency))
		}
	}
	return KVCacheConfig{
		TotalKVBlocks:         totalKVBlocks,
		BlockSizeTokens:       blockSizeTokens,
		KVCPUBlocks:           kvCPUBlocks,
		KVOffloadThreshold:    kvOffloadThreshold,
		KVTransferBandwidth:   kvTransferBandwidth,
		KVTransferBaseLatency: kvTransferBaseLatency,
	}
}
```

Add `"math"` to the import block in `sim/config.go`.

Also delete `TestNewKVCacheConfig_ZeroValues_NoDefaults` (config_test.go:70-74) — it calls `NewKVCacheConfig(0, 0, 0, 0, 0, 0)` which now panics on `TotalKVBlocks=0`. This must happen in the same commit as the validation to maintain bisectability.

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestNewKVCacheConfig_PanicsOnInvalid -v`
Expected: PASS

**Step 4b: Run full sim test suite to verify no breakage**

Run: `go test ./sim/... -count=1`
Expected: All PASS (zero-value test deleted, all other call sites use valid values)

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/config.go sim/config_test.go
git commit -m "fix(sim): add validation to NewKVCacheConfig for GPU-tier params (BC-1, BC-9)

- TotalKVBlocks must be > 0
- BlockSizeTokens must be > 0
- KVCPUBlocks must be >= 0
- Tiered-mode params validated when KVCPUBlocks > 0
- Panic messages include constructor name + parameter name (R3)
- Remove TestNewKVCacheConfig_ZeroValues_NoDefaults (bisectability)

Fixes #736

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Add tiered-mode and valid-path tests (BC-2, BC-3, BC-4, BC-5)

**Contracts Implemented:** BC-2, BC-3, BC-4, BC-5

**Files:**
- Modify: `sim/config_test.go` (extend panic table, add positive tests)
- Test: `sim/config_test.go`

**Step 1: Replace zero-value test and add cases to panic table**

Context: Extend the panic table with tiered-mode and CPU-blocks cases. Add positive path tests for BC-4 and BC-5.

Add these cases to the `TestNewKVCacheConfig_PanicsOnInvalid` table from Task 1:

```go
		{"negative_cpu_blocks", 100, 16, -1, 0, 0, 0, "KVCPUBlocks"},
		{"tiered_bandwidth_zero", 100, 16, 10, 0.5, 0, 0, "KVTransferBandwidth"},
		{"tiered_bandwidth_negative", 100, 16, 10, 0.5, -1.0, 0, "KVTransferBandwidth"},
		{"tiered_bandwidth_nan", 100, 16, 10, 0.5, math.NaN(), 0, "KVTransferBandwidth"},
		{"tiered_bandwidth_inf", 100, 16, 10, 0.5, math.Inf(1), 0, "KVTransferBandwidth"},
		{"tiered_base_latency_negative", 100, 16, 10, 0.5, 100.0, -1, "KVTransferBaseLatency"},
```

Add test for BC-4 (single-tier skips tiered checks):

```go
func TestNewKVCacheConfig_SingleTier_SkipsTieredValidation(t *testing.T) {
	// BC-4: Single-tier mode (KVCPUBlocks=0) accepts any threshold/bandwidth/latency
	// without panicking. These fields are meaningless in single-tier mode.
	cfg := NewKVCacheConfig(100, 16, 0, -999.0, -999.0, -999)
	if cfg.TotalKVBlocks != 100 {
		t.Errorf("TotalKVBlocks = %d, want 100", cfg.TotalKVBlocks)
	}
	if cfg.KVOffloadThreshold != -999.0 {
		t.Errorf("KVOffloadThreshold = %f, want -999.0 (passed through)", cfg.KVOffloadThreshold)
	}
}
```

Add test for BC-5 (valid tiered-mode):

```go
func TestNewKVCacheConfig_ValidTiered_ReturnsConfig(t *testing.T) {
	// BC-5: Valid tiered-mode parameters accepted
	cfg := NewKVCacheConfig(100, 16, 50, 0.9, 100.0, 500)
	if cfg.KVCPUBlocks != 50 {
		t.Errorf("KVCPUBlocks = %d, want 50", cfg.KVCPUBlocks)
	}
	if cfg.KVOffloadThreshold != 0.9 {
		t.Errorf("KVOffloadThreshold = %f, want 0.9", cfg.KVOffloadThreshold)
	}
}
```

**Step 2: Run tests to verify they pass**

Run: `go test ./sim/... -run "TestNewKVCacheConfig" -v`
Expected: All PASS (including the original field-equivalence test)

**Step 3: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 4: Commit**

```bash
git add sim/config_test.go
git commit -m "test(sim): add tiered-mode and edge-case tests for NewKVCacheConfig (BC-2..BC-5)

- Add negative CPU blocks, tiered bandwidth/latency cases
- Verify single-tier mode skips tiered validation (BC-4)
- Verify valid tiered-mode accepted (BC-5)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: WorkloadSpec mutual exclusion check (BC-6, BC-7, BC-8)

**Contracts Implemented:** BC-6, BC-7, BC-8

**Files:**
- Modify: `sim/workload/generator.go:17-24`
- Test: `sim/workload/generator_test.go` (new test function, or spec_test.go)

**Step 1: Write failing tests for mutual exclusion**

Context: Test that GenerateRequests rejects specs with multiple sources.

In `sim/workload/generator_test.go` (tests for `GenerateRequests` which lives in `generator.go`):

```go
func TestGenerateRequests_MutualExclusion_ClientsAndServeGen_ReturnsError(t *testing.T) {
	// BC-6: Clients + ServeGenData → error
	spec := &WorkloadSpec{
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
		ServeGenData: &ServeGenDataSpec{Path: "data/"},
	}
	_, err := GenerateRequests(spec, 1_000_000, 100)
	if err == nil {
		t.Fatal("expected error for Clients + ServeGenData, got nil")
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("error should mention mutual exclusion: %v", err)
	}
}

func TestGenerateRequests_MutualExclusion_ClientsAndInferencePerf_ReturnsError(t *testing.T) {
	// BC-6: Clients + InferencePerf → error
	spec := &WorkloadSpec{
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
		InferencePerf: &InferencePerfSpec{
			Stages: []InferencePerfStage{{Rate: 10, Duration: 60, PromptTokens: 100, OutputTokens: 50}},
			SharedPrefixTokens: 10,
		},
	}
	_, err := GenerateRequests(spec, 1_000_000, 100)
	if err == nil {
		t.Fatal("expected error for Clients + InferencePerf, got nil")
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("error should mention mutual exclusion: %v", err)
	}
}

func TestGenerateRequests_MutualExclusion_ServeGenAndInferencePerf_ReturnsError(t *testing.T) {
	// BC-6: ServeGenData + InferencePerf → error
	spec := &WorkloadSpec{
		AggregateRate: 100.0,
		ServeGenData: &ServeGenDataSpec{Path: "data/"},
		InferencePerf: &InferencePerfSpec{
			Stages: []InferencePerfStage{{Rate: 10, Duration: 60, PromptTokens: 100, OutputTokens: 50}},
			SharedPrefixTokens: 10,
		},
	}
	_, err := GenerateRequests(spec, 1_000_000, 100)
	if err == nil {
		t.Fatal("expected error for ServeGenData + InferencePerf, got nil")
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("error should mention mutual exclusion: %v", err)
	}
}

func TestGenerateRequests_MutualExclusion_AllThreeSources_ReturnsError(t *testing.T) {
	// BC-6: All three sources set → error listing all three
	spec := &WorkloadSpec{
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
		ServeGenData: &ServeGenDataSpec{Path: "data/"},
		InferencePerf: &InferencePerfSpec{
			Stages: []InferencePerfStage{{Rate: 10, Duration: 60, PromptTokens: 100, OutputTokens: 50}},
			SharedPrefixTokens: 10,
		},
	}
	_, err := GenerateRequests(spec, 1_000_000, 100)
	if err == nil {
		t.Fatal("expected error for all three sources, got nil")
	}
	if !strings.Contains(err.Error(), "clients") || !strings.Contains(err.Error(), "servegen_data") || !strings.Contains(err.Error(), "inference_perf") {
		t.Errorf("error should list all three conflicting sources: %v", err)
	}
}

func TestGenerateRequests_MutualExclusion_CohortsWithClients_Allowed(t *testing.T) {
	// BC-7: Cohorts + Clients is intentional composition, not a conflict
	spec := &WorkloadSpec{
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 0.5,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
		Cohorts: []CohortSpec{{
			ID: "cohort1", Population: 2, RateFraction: 0.5,
			Arrival:   ArrivalSpec{Process: "poisson"},
			InputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	reqs, err := GenerateRequests(spec, 1_000_000, 10)
	if err != nil {
		t.Fatalf("unexpected error for Clients + Cohorts: %v", err)
	}
	if len(reqs) == 0 {
		t.Error("expected requests from Clients + Cohorts composition")
	}
}

func TestGenerateRequests_MutualExclusion_CohortsWithInferencePerf_Allowed(t *testing.T) {
	// BC-7: Cohorts + InferencePerf (no explicit Clients) is allowed composition.
	// InferencePerf expands into Clients, then Cohorts compose with them.
	spec := &WorkloadSpec{
		AggregateRate: 100.0,
		Cohorts: []CohortSpec{{
			ID: "cohort1", Population: 1, RateFraction: 0.5,
			Arrival:   ArrivalSpec{Process: "poisson"},
			InputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
		InferencePerf: &InferencePerfSpec{
			Stages: []InferencePerfStage{{Rate: 10, Duration: 60, PromptTokens: 100, OutputTokens: 50}},
			SharedPrefixTokens: 10,
		},
	}
	reqs, err := GenerateRequests(spec, 1_000_000, 10)
	if err != nil {
		t.Fatalf("unexpected error for Cohorts + InferencePerf: %v", err)
	}
	if len(reqs) == 0 {
		t.Error("expected requests from Cohorts + InferencePerf composition")
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/workload/... -run "TestGenerateRequests_MutualExclusion" -v`
Expected: First four FAIL (no mutual exclusion check), last two PASS

**Step 3: Implement mutual exclusion check in GenerateRequests**

In `sim/workload/generator.go`, add `"strings"` to the import block, then add after the `maxRequests < 0` check (after line 23) and before the InferencePerf expansion (line 25):

```go
	// Mutual exclusion: at most one primary workload source allowed (R1).
	// Clients+Cohorts compose (cohorts expand into clients), but
	// InferencePerf and ServeGenData are exclusive alternatives.
	var sourceNames []string
	if len(spec.Clients) > 0 {
		sourceNames = append(sourceNames, "clients")
	}
	if spec.ServeGenData != nil {
		sourceNames = append(sourceNames, "servegen_data")
	}
	if spec.InferencePerf != nil {
		sourceNames = append(sourceNames, "inference_perf")
	}
	if len(sourceNames) > 1 {
		return nil, fmt.Errorf("workload sources {%s} are mutually exclusive; specify exactly one of: clients, servegen_data, inference_perf", strings.Join(sourceNames, ", "))
	}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/workload/... -run "TestGenerateRequests_MutualExclusion" -v`
Expected: All PASS

**Step 5: Run full test suite to check for regressions**

Run: `go test ./sim/workload/... -v -count=1`
Expected: All PASS (no existing tests set multiple sources)

**Step 6: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 7: Commit**

```bash
git add sim/workload/generator.go sim/workload/generator_test.go
git commit -m "fix(workload): enforce mutual exclusion of workload sources (BC-6..BC-8)

- Error when multiple of {clients, servegen_data, inference_perf} coexist
- Cohorts + Clients remains allowed (intentional composition)
- Prevents silent data loss (R1) from precedence ordering

Fixes #737

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Full test suite verification and MEMORY.md cleanup

**Contracts Implemented:** All (regression check)

**Step 1: Run full test suite**

Run: `go test ./... -count=1`
Expected: All PASS

**Step 2: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 3: Commit (if any cleanup needed)**

No code changes expected — this is a verification gate.

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Failure | TestNewKVCacheConfig_PanicsOnInvalid (zero/negative blocks) |
| BC-2 | Task 2 | Failure | TestNewKVCacheConfig_PanicsOnInvalid (negative CPU blocks) |
| BC-3 | Task 2 | Failure | TestNewKVCacheConfig_PanicsOnInvalid (tiered-mode cases) |
| BC-4 | Task 2 | Unit | TestNewKVCacheConfig_SingleTier_SkipsTieredValidation |
| BC-5 | Task 2 | Unit | TestNewKVCacheConfig_ValidTiered_ReturnsConfig |
| BC-6 | Task 3 | Failure | TestGenerateRequests_MutualExclusion_{Clients+ServeGen, Clients+InferencePerf, ServeGen+InferencePerf} |
| BC-7 | Task 3 | Unit | TestGenerateRequests_MutualExclusion_CohortsWithClients_Allowed |
| BC-8 | Task 3 | Failure | (Same as BC-6 — error returned, not silent skip) |
| BC-9 | Task 1 | Failure | TestNewKVCacheConfig_PanicsOnInvalid (all cases check constructor name) |

No golden dataset changes. No new shared test infrastructure needed.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Existing tests break from new KVCacheConfig validation | Medium | Medium | Task 2 removes the zero-value test; all other callers use valid values (verified by grep) |
| Mutual exclusion check breaks existing integration tests | Low | Medium | Grep confirms no existing test sets multiple sources |
| NaN comparison edge case in threshold check | Low | Low | `math.IsNaN` check explicit, matching NewKVStore pattern |

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
- [x] Shared test helpers: testify/assert for field-equivalence tests; raw testing.T for panic-recovery tests (matches existing NewBatchConfig pattern)
- [x] CLAUDE.md: no updates needed (no new files/packages/CLI flags)
- [x] Documentation DRY: no canonical sources modified
- [x] Deviation log reviewed — two deviations justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Task 1 → Task 2 → Task 3 → Task 4)
- [x] All contracts mapped to tasks
- [x] Construction site audit: NewKVCacheConfig has ~70 call sites; all use valid values (TotalKVBlocks > 0, BlockSizeTokens > 0). Only config_test.go:70-74 uses all zeros — Task 2 removes it.
- [x] R1: Mutual exclusion check eliminates silent data loss
- [x] R3: All new validation covers library constructors
- [x] R4: No new struct fields
- [x] R6: No logrus.Fatalf in sim/
- [x] R7: No golden tests added
- [x] R8-R23: Not applicable (no maps, YAML fields, division, interfaces, etc.)

---

## Appendix: File-Level Implementation Details

### File: `sim/config.go`

**Purpose:** Add validation panics to `NewKVCacheConfig`, matching `NewBatchConfig` pattern.

**Changes:** Replace body of `NewKVCacheConfig` (lines 18-29) with validation + struct literal. Add `"math"` import.

**Error handling:** Panics (library constructor convention). Messages include `"NewKVCacheConfig:"` prefix, parameter name, and actual value.

### File: `sim/config_test.go`

**Purpose:** Add `TestNewKVCacheConfig_PanicsOnInvalid` table-driven test. Remove `TestNewKVCacheConfig_ZeroValues_NoDefaults`. Add `TestNewKVCacheConfig_SingleTier_SkipsTieredValidation` and `TestNewKVCacheConfig_ValidTiered_ReturnsConfig`.

**Changes:** Add `"math"` import if not present. New test functions. Delete lines 70-74.

### File: `sim/workload/generator.go`

**Purpose:** Add mutual exclusion check before InferencePerf/ServeGenData expansion.

**Changes:** Insert ~12 lines after line 23 (after maxRequests check). No other changes.

### File: `sim/workload/generator_test.go`

**Purpose:** Add mutual exclusion tests for GenerateRequests.

**Changes:** Add 5 new test functions. Add `"strings"` import if not present.
