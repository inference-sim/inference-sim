# fix(roofline): Validate Hardware/Model Config Inputs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent silent NaN/Inf corruption in roofline latency estimation by validating hardware and model config inputs before they reach division operations.

**The problem today:** The roofline latency estimation path has 5 divisions where the denominator comes from user-provided config files (`config.json` from HuggingFace, `hardware_config.json`). If any required field is zero or missing — `num_attention_heads`, `TFlopsPeak`, `MfuPrefill`, `MfuDecode`, `BwPeakTBs`, or `BwEffConstant` — the result silently becomes NaN or Inf, corrupting all downstream step time calculations without any error message. Additionally, unsorted map iteration in `calculateMemoryAccessBytes` violates the determinism invariant.

**What this PR adds:**
1. Config validation — `ValidateRooflineConfig` checks all 6 critical fields at roofline mode entry, producing clear error messages instead of silent NaN/Inf
2. Deterministic float accumulation — sorted map keys in `calculateMemoryAccessBytes` ensure byte-identical output across runs
3. Clear error boundaries — validation errors returned from `sim/` library code, CLI boundary calls `logrus.Fatalf`

**Why this matters:** Roofline mode is the zero-setup path for capacity planning (no trained coefficients needed). Silent NaN corruption makes it unusable for new model/GPU combinations where configs may have missing fields.

**Architecture:** Add a `ValidateRooflineConfig(ModelConfig, HardwareCalib)` function in `sim/model_hardware_config.go` that returns an error listing all invalid fields. Call it from `NewSimulator` when `cfg.Roofline == true`. Fix the map iteration in `sim/roofline_step.go:130`.

**Source:** GitHub issue #236

**Closes:** Fixes #236

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds input validation for the roofline latency estimation subsystem. Today, 5 division operations in `sim/roofline_step.go` trust that config values are non-zero — but these values come from user-provided JSON files that may have missing or zero fields. The fix validates all critical denominators at roofline mode entry (`NewSimulator`), returning a clear error instead of silently producing NaN/Inf. A secondary fix sorts map keys in `calculateMemoryAccessBytes` to ensure deterministic float accumulation.

**Adjacent blocks:** `NewSimulator` (validation caller), `cmd/root.go` (CLI boundary — already calls `logrus.Fatalf` on config load errors), `rooflineStepTime` (consumer of validated config).

**No deviations from issue #236.**

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Zero NumHeads Rejected
- GIVEN a `ModelConfig` with `NumHeads == 0`
- WHEN `ValidateRooflineConfig` is called
- THEN it returns an error mentioning "NumHeads"
- MECHANISM: Explicit `NumHeads <= 0` check in validation function

BC-2: Invalid Hardware Fields Rejected
- GIVEN a `HardwareCalib` with any of `TFlopsPeak`, `BwPeakTBs`, `BwEffConstant`, `MfuPrefill`, or `MfuDecode` equal to zero, negative, NaN, or Inf
- WHEN `ValidateRooflineConfig` is called
- THEN it returns an error mentioning every invalid field
- MECHANISM: Check each field with `invalidPositiveFloat()` (rejects `<= 0`, NaN, Inf) and collect all errors

BC-3: Valid Config Passes
- GIVEN a `ModelConfig` and `HardwareCalib` with all required fields positive
- WHEN `ValidateRooflineConfig` is called
- THEN it returns nil (no error)

BC-4: NewSimulator Rejects Invalid Roofline Config
- GIVEN a `SimConfig` with `Roofline == true` and `NumHeads == 0`
- WHEN `NewSimulator` is called
- THEN it returns a non-nil error (not a panic)
- MECHANISM: `NewSimulator` calls `ValidateRooflineConfig` when `cfg.Roofline == true`

BC-5: Deterministic Memory Bytes
- GIVEN any `ModelConfig`
- WHEN `calculateMemoryAccessBytes` is called multiple times with the same inputs
- THEN the `"total"` field is identical across all calls
- MECHANISM: Sort map keys before accumulating `total`

**Negative Contracts:**

BC-6: No Library Fatalf
- GIVEN any input to `ValidateRooflineConfig`
- WHEN called
- THEN it MUST NOT call `logrus.Fatalf`, `os.Exit`, or any process-terminating function
- MECHANISM: Returns `error` — only `cmd/` may terminate

BC-7: Non-Roofline Mode Unaffected
- GIVEN a `SimConfig` with `Roofline == false`
- WHEN `NewSimulator` is called
- THEN roofline config validation is NOT performed (zero `NumHeads` is acceptable)

### C) Component Interaction

```
cmd/root.go:178-187    ──→ sim.GetModelConfig() ──→ ModelConfig
                        ──→ sim.GetHWConfig()    ──→ HardwareCalib
                            │
                            ▼
cmd/root.go (builds SimConfig with Roofline=true)
                            │
                            ▼
sim.NewSimulator(cfg)  ──→ ValidateRooflineConfig(cfg.ModelConfig, cfg.HWConfig)
                            │                         │
                            │ error ←─────────────────┘
                            ▼
                       rooflineStepTime() [safe: all denominators validated]
```

**API:** `ValidateRooflineConfig(mc ModelConfig, hc HardwareCalib) error` — pure validation, no side effects.

**State changes:** None. This is input validation only.

**Extension friction:** Adding a new field to `HardwareCalib` that appears as a denominator requires adding one check to `ValidateRooflineConfig`. 1 file to change — acceptable.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "validate at roofline mode entry (or config load time)" | Validate in `NewSimulator` | SIMPLIFICATION: `NewSimulator` is the single entry point where `Roofline == true` is known. Config load time (`GetModelConfig`/`GetHWConfig`) doesn't know if roofline mode is active |
| Issue mentions `roofline_step.go:16,95` lines | These are lines 16 and 95 in the current code, both `dModel / nHeads` | CORRECTION: Both are the same expression; validation catches the root cause (NumHeads=0) rather than guarding each division site |

### E) Review Guide

**The tricky part:** `NewSimulator` currently uses `panic` for validation failures (lines 155-166). This PR adds `error` return for roofline validation instead. The choice is deliberate: panics are for programming errors (invariant violations), errors are for user-facing config problems.

**What to scrutinize:** BC-4 — verify `NewSimulator` returns error (not panic) for invalid roofline config. BC-5 — verify the sort actually fixes non-determinism.

**What's safe to skim:** BC-1, BC-2, BC-3 are straightforward field checks. BC-6 is a convention check.

**Known debt:** (1) The existing `calculateTransformerFlops` function (lines 8-79) also uses `dModel / nHeads` (lines 16, 95) but those divisions are already guarded by the validation we're adding — `ValidateRooflineConfig` runs before any roofline computation. (2) `NewInstanceSimulator` in `sim/cluster/instance.go:34` converts `NewSimulator` errors into panics — so in cluster mode, invalid roofline config produces a stack trace instead of a clean error message. This is pre-existing behavior (not introduced by this PR) and would need `NewInstanceSimulator` to return `error` to fix properly.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/model_hardware_config.go` — add `ValidateRooflineConfig` function
- `sim/roofline_step.go:129-133` — sort map keys before float accumulation
- `sim/simulator.go:152-166` — call `ValidateRooflineConfig` when roofline mode
- `sim/model_hardware_config_test.go` — add validation tests
- `sim/roofline_step_test.go` (create) — add determinism test

**No files to create other than `sim/roofline_step_test.go`.**

**Key decisions:**
- Validation returns `error` (not panic) — these are user config errors, not programming bugs
- All fields checked in one pass (collect all errors, don't stop at first)
- Sort fix uses `sort.Strings` on map keys — minimal change, proven pattern

### G) Task Breakdown

---

### Task 1: Add ValidateRooflineConfig and Validation Tests

**Contracts Implemented:** BC-1, BC-2, BC-3, BC-6

**Files:**
- Modify: `sim/model_hardware_config.go` (add function at end)
- Modify: `sim/model_hardware_config_test.go` (add tests at end)

**Step 1: Write failing tests for config validation**

Context: We need to verify that zero/missing fields produce clear errors, valid configs pass, and no `os.Exit` is called.

In `sim/model_hardware_config_test.go`, append:

```go
func TestValidateRooflineConfig_ZeroNumHeads_ReturnsError(t *testing.T) {
	// GIVEN a ModelConfig with NumHeads == 0
	mc := ModelConfig{NumHeads: 0, NumLayers: 32, HiddenDim: 4096}
	hc := HardwareCalib{TFlopsPeak: 1000, BwPeakTBs: 3.35, BwEffConstant: 0.7, MfuPrefill: 0.5, MfuDecode: 0.3}

	// WHEN ValidateRooflineConfig is called
	err := ValidateRooflineConfig(mc, hc)

	// THEN it returns an error mentioning NumHeads
	if err == nil {
		t.Fatal("expected error for zero NumHeads, got nil")
	}
	if !strings.Contains(err.Error(), "NumHeads") {
		t.Errorf("error should mention NumHeads, got: %v", err)
	}
}

func TestValidateRooflineConfig_ZeroHardwareFields_ReturnsAllErrors(t *testing.T) {
	// GIVEN a HardwareCalib with all critical fields zero
	mc := ModelConfig{NumHeads: 32, NumLayers: 32, HiddenDim: 4096}
	hc := HardwareCalib{} // all zero

	// WHEN ValidateRooflineConfig is called
	err := ValidateRooflineConfig(mc, hc)

	// THEN it returns an error mentioning every zero field
	if err == nil {
		t.Fatal("expected error for zero hardware fields, got nil")
	}
	errMsg := err.Error()
	for _, field := range []string{"TFlopsPeak", "BwPeakTBs", "BwEffConstant", "MfuPrefill", "MfuDecode"} {
		if !strings.Contains(errMsg, field) {
			t.Errorf("error should mention %s, got: %v", field, errMsg)
		}
	}
}

func TestValidateRooflineConfig_NaNInfFields_ReturnsErrors(t *testing.T) {
	// GIVEN a HardwareCalib with NaN and Inf fields (bypass <= 0 check)
	mc := ModelConfig{NumHeads: 32, NumLayers: 32, HiddenDim: 4096}
	hc := HardwareCalib{
		TFlopsPeak:    math.NaN(),
		BwPeakTBs:     math.Inf(1),
		BwEffConstant: 0.7,
		MfuPrefill:    0.5,
		MfuDecode:     math.NaN(),
	}

	// WHEN ValidateRooflineConfig is called
	err := ValidateRooflineConfig(mc, hc)

	// THEN it returns an error mentioning the invalid fields
	if err == nil {
		t.Fatal("expected error for NaN/Inf hardware fields, got nil")
	}
	errMsg := err.Error()
	for _, field := range []string{"TFlopsPeak", "BwPeakTBs", "MfuDecode"} {
		if !strings.Contains(errMsg, field) {
			t.Errorf("error should mention %s, got: %v", field, errMsg)
		}
	}
}

func TestValidateRooflineConfig_ValidConfig_ReturnsNil(t *testing.T) {
	// GIVEN valid ModelConfig and HardwareCalib
	mc := ModelConfig{NumHeads: 32, NumLayers: 32, HiddenDim: 4096}
	hc := HardwareCalib{TFlopsPeak: 1000, BwPeakTBs: 3.35, BwEffConstant: 0.7, MfuPrefill: 0.5, MfuDecode: 0.3}

	// WHEN ValidateRooflineConfig is called
	err := ValidateRooflineConfig(mc, hc)

	// THEN it returns nil
	if err != nil {
		t.Errorf("expected nil error for valid config, got: %v", err)
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/... -run TestValidateRooflineConfig -v`
Expected: FAIL — `ValidateRooflineConfig` is undefined

**Step 3: Implement ValidateRooflineConfig**

Context: Pure validation function that checks all fields used as denominators in roofline computations and returns a single error listing all problems.

In `sim/model_hardware_config.go`, add at the end:

```go
// invalidPositiveFloat returns true if v is not a valid positive float64
// (i.e., v <= 0, NaN, or Inf). Used to validate roofline config denominators.
func invalidPositiveFloat(v float64) bool {
	return v <= 0 || math.IsNaN(v) || math.IsInf(v, 0)
}

// ValidateRooflineConfig checks that all fields required by the roofline latency
// model are valid positive values. Returns an error listing all invalid fields, or nil if valid.
func ValidateRooflineConfig(mc ModelConfig, hc HardwareCalib) error {
	var problems []string

	if mc.NumHeads <= 0 {
		problems = append(problems, fmt.Sprintf("ModelConfig.NumHeads must be > 0, got %d", mc.NumHeads))
	}
	if invalidPositiveFloat(hc.TFlopsPeak) {
		problems = append(problems, fmt.Sprintf("HardwareCalib.TFlopsPeak must be a valid positive number, got %v", hc.TFlopsPeak))
	}
	if invalidPositiveFloat(hc.BwPeakTBs) {
		problems = append(problems, fmt.Sprintf("HardwareCalib.BwPeakTBs must be a valid positive number, got %v", hc.BwPeakTBs))
	}
	if invalidPositiveFloat(hc.BwEffConstant) {
		problems = append(problems, fmt.Sprintf("HardwareCalib.BwEffConstant must be a valid positive number, got %v", hc.BwEffConstant))
	}
	if invalidPositiveFloat(hc.MfuPrefill) {
		problems = append(problems, fmt.Sprintf("HardwareCalib.MfuPrefill must be a valid positive number, got %v", hc.MfuPrefill))
	}
	if invalidPositiveFloat(hc.MfuDecode) {
		problems = append(problems, fmt.Sprintf("HardwareCalib.MfuDecode must be a valid positive number, got %v", hc.MfuDecode))
	}

	if len(problems) > 0 {
		return fmt.Errorf("invalid roofline config: %s", strings.Join(problems, "; "))
	}
	return nil
}
```

Note: `fmt` is already imported in `model_hardware_config.go`. Add `"math"` and `"strings"` to the import block. Also add `"math"` to the test file's import block (`model_hardware_config_test.go` currently imports `os`, `path/filepath`, `strings`, `testing` — add `"math"` for the NaN/Inf test).

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/... -run TestValidateRooflineConfig -v`
Expected: PASS (3 tests)

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/model_hardware_config.go sim/model_hardware_config_test.go
git commit -m "feat(sim): add ValidateRooflineConfig for hardware/model inputs (BC-1, BC-2, BC-3)

- Check NumHeads, TFlopsPeak, BwPeakTBs, BwEffConstant, MfuPrefill, MfuDecode > 0
- Collect all errors in one pass (user sees all problems at once)
- Return error (not panic) — these are user config errors

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Wire Validation into NewSimulator

**Contracts Implemented:** BC-4, BC-7

**Files:**
- Modify: `sim/simulator.go:152-166` (add roofline validation)
- Modify: `sim/model_hardware_config_test.go` (add NewSimulator integration test)

**Step 1: Write failing test for NewSimulator rejecting invalid roofline config**

Context: `NewSimulator` should return an error (not panic) when roofline mode has invalid config.

In `sim/model_hardware_config_test.go`, append:

```go
func TestNewSimulator_RooflineZeroNumHeads_ReturnsError(t *testing.T) {
	// GIVEN a SimConfig with Roofline=true and NumHeads=0
	cfg := SimConfig{
		Roofline:        true,
		ModelConfig:     ModelConfig{NumHeads: 0, NumLayers: 32, HiddenDim: 4096},
		HWConfig:        HardwareCalib{TFlopsPeak: 1000, BwPeakTBs: 3.35, BwEffConstant: 0.7, MfuPrefill: 0.5, MfuDecode: 0.3},
		TotalKVBlocks:   1000,
		BlockSizeTokens: 16,
		Horizon:         100000,
	}

	// WHEN NewSimulator is called
	_, err := NewSimulator(cfg)

	// THEN it returns a non-nil error mentioning NumHeads
	if err == nil {
		t.Fatal("expected error for roofline with zero NumHeads, got nil")
	}
	if !strings.Contains(err.Error(), "NumHeads") {
		t.Errorf("error should mention NumHeads, got: %v", err)
	}
}

func TestNewSimulator_NonRooflineZeroNumHeads_Succeeds(t *testing.T) {
	// GIVEN a SimConfig with Roofline=false and NumHeads=0 (irrelevant)
	cfg := SimConfig{
		Roofline:        false,
		ModelConfig:     ModelConfig{NumHeads: 0},
		HWConfig:        HardwareCalib{},
		TotalKVBlocks:   1000,
		BlockSizeTokens: 16,
		Horizon:         100000,
		BetaCoeffs:      []float64{1, 2, 3},
		AlphaCoeffs:     []float64{1, 2, 3},
	}

	// WHEN NewSimulator is called
	sim, err := NewSimulator(cfg)

	// THEN it succeeds (roofline validation not applied)
	if err != nil {
		t.Fatalf("unexpected error for non-roofline mode: %v", err)
	}
	if sim == nil {
		t.Error("expected non-nil simulator")
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/... -run "TestNewSimulator_Roofline|TestNewSimulator_NonRoofline" -v`
Expected: First test FAILS (NewSimulator currently panics or doesn't validate roofline config), second test may pass.

**Step 3: Add roofline validation to NewSimulator**

Context: Insert validation after the existing checks in `NewSimulator`, before the simulator struct is constructed.

In `sim/simulator.go`, after the existing `BlockSizeTokens` check (line ~166) and before the `s := &Simulator{` construction (line ~168), add:

```go
	if cfg.Roofline {
		if err := ValidateRooflineConfig(cfg.ModelConfig, cfg.HWConfig); err != nil {
			return nil, fmt.Errorf("roofline validation: %w", err)
		}
	}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/... -run "TestNewSimulator_Roofline|TestNewSimulator_NonRoofline" -v`
Expected: PASS (2 tests)

Also run full test suite to verify no regressions:
Run: `go test ./sim/... -v -count=1`
Expected: All existing tests still pass

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/simulator.go sim/model_hardware_config_test.go
git commit -m "feat(sim): wire ValidateRooflineConfig into NewSimulator (BC-4, BC-7)

- NewSimulator returns error (not panic) for invalid roofline config
- Validation only runs when Roofline == true
- Non-roofline mode unaffected

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Fix Non-Deterministic Map Iteration and Add Determinism Test

**Contracts Implemented:** BC-5

**Files:**
- Modify: `sim/roofline_step.go:129-133` (sort keys)
- Create: `sim/roofline_step_test.go` (determinism test)

**Step 1: Write failing test for deterministic float accumulation**

Context: Map iteration in Go is non-deterministic. We need to verify that `calculateMemoryAccessBytes` produces identical results across multiple calls. This test may not reliably fail before the fix (non-determinism is probabilistic), but it documents the contract.

Create `sim/roofline_step_test.go`:

```go
package sim

import (
	"testing"
)

func TestCalculateMemoryAccessBytes_Deterministic(t *testing.T) {
	// GIVEN a ModelConfig with multiple non-zero fields
	config := ModelConfig{
		NumLayers:       32,
		HiddenDim:       4096,
		NumHeads:        32,
		NumKVHeads:      8,
		BytesPerParam:   2,
		IntermediateDim: 14336,
	}

	// WHEN calculateMemoryAccessBytes is called 100 times
	var firstTotal float64
	for i := 0; i < 100; i++ {
		result := calculateMemoryAccessBytes(config, 1024, 64, true)

		// THEN every call produces the same "total"
		if i == 0 {
			firstTotal = result["total"]
		} else if result["total"] != firstTotal {
			t.Fatalf("non-deterministic total: call 0 got %v, call %d got %v", firstTotal, i, result["total"])
		}
	}

	// Also verify the total is positive (sanity)
	if firstTotal <= 0 {
		t.Errorf("expected positive total, got %v", firstTotal)
	}
}
```

**Step 2: Run test to verify it compiles**

Run: `go test ./sim/... -run TestCalculateMemoryAccessBytes_Deterministic -v`
Expected: PASS (or may intermittently fail due to non-determinism — that's what we're fixing)

**Step 3: Fix the non-deterministic map iteration**

Context: In `sim/roofline_step.go`, lines 129-133, replace the range loop with sorted key iteration.

Replace lines 129-133 in `sim/roofline_step.go`:
```go
	var total float64
	for _, v := range mem {
		total += v
	}
	mem["total"] = total
```

With:
```go
	// Sort keys before accumulation for deterministic float summation
	// (Go map iteration order is non-deterministic — antipattern #2)
	keys := make([]string, 0, len(mem))
	for k := range mem {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var total float64
	for _, k := range keys {
		total += mem[k]
	}
	mem["total"] = total
```

Also add `"sort"` to the import block at the top of `roofline_step.go` (currently only imports `"math"`).

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestCalculateMemoryAccessBytes_Deterministic -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/roofline_step.go sim/roofline_step_test.go
git commit -m "fix(sim): sort map keys in calculateMemoryAccessBytes for determinism (BC-5)

- Sort keys before float accumulation (antipattern #2)
- Add determinism test verifying 100 identical calls

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Final Verification and Documentation

**Contracts Implemented:** (verification of all contracts)

**Files:**
- No code changes (verification only)

**Step 1: Run full test suite**

Run: `go test ./... -count=1`
Expected: All tests pass

**Step 2: Run full lint**

Run: `golangci-lint run ./...`
Expected: 0 issues

**Step 3: Run build**

Run: `go build ./...`
Expected: Build succeeds

**Step 4: Verify contract coverage**

Manually verify each contract has a passing test:
- BC-1: `TestValidateRooflineConfig_ZeroNumHeads_ReturnsError`
- BC-2: `TestValidateRooflineConfig_ZeroHardwareFields_ReturnsAllErrors`
- BC-3: `TestValidateRooflineConfig_ValidConfig_ReturnsNil`
- BC-4: `TestNewSimulator_RooflineZeroNumHeads_ReturnsError`
- BC-5: `TestCalculateMemoryAccessBytes_Deterministic`
- BC-6: Verified by inspection — no `logrus.Fatalf` or `os.Exit` in `ValidateRooflineConfig`
- BC-7: `TestNewSimulator_NonRooflineZeroNumHeads_Succeeds`

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | `TestValidateRooflineConfig_ZeroNumHeads_ReturnsError` |
| BC-2 | Task 1 | Unit | `TestValidateRooflineConfig_ZeroHardwareFields_ReturnsAllErrors`, `TestValidateRooflineConfig_NaNInfFields_ReturnsErrors` |
| BC-3 | Task 1 | Unit | `TestValidateRooflineConfig_ValidConfig_ReturnsNil` |
| BC-4 | Task 2 | Integration | `TestNewSimulator_RooflineZeroNumHeads_ReturnsError` |
| BC-5 | Task 3 | Invariant | `TestCalculateMemoryAccessBytes_Deterministic` |
| BC-6 | Task 1 | Inspection | Code review — no process termination in library code |
| BC-7 | Task 2 | Unit | `TestNewSimulator_NonRooflineZeroNumHeads_Succeeds` |

**Golden dataset:** No updates needed. Roofline mode uses a separate code path from the alpha/beta mode tested by the golden dataset. Existing golden tests remain unchanged.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Existing tests break due to NewSimulator signature change | Low | High | NewSimulator already returns `(*, error)` — no signature change | Task 2 |
| Validation too strict (rejects valid real-world configs) | Low | Medium | Only check fields that are denominators — no cosmetic checks | Task 1 |
| Sort overhead in hot path | Low | Low | `calculateMemoryAccessBytes` is called per-step but the map has ~4-5 keys; sort of 5 strings is negligible | Task 3 |

### J) Sanity Checklist

- [x] No unnecessary abstractions — single function, no new types
- [x] No feature creep beyond PR scope — only validation + sort fix
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes — `NewSimulator` signature unchanged (`(*Simulator, error)`)
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers: N/A (no shared infra needed)
- [x] CLAUDE.md: No update needed (no new files/packages/CLI flags)
- [x] No stale references in CLAUDE.md
- [x] Deviation log reviewed — 2 deviations, both justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1 → 2 → 3 → 4)
- [x] All contracts mapped to specific tasks
- [x] Golden dataset: no regeneration needed
- [x] Construction site audit: `ValidateRooflineConfig` is a new function (no struct modification). `NewSimulator` is modified but its signature is unchanged
- [x] No new CLI flags
- [x] Every error path returns error — no silent continue
- [x] Sort fixes map iteration for float accumulation (antipattern #2)
- [x] Library code never calls logrus.Fatalf (antipattern #6)
- [x] No new resource allocation loops
- [x] No exported mutable maps
- [x] No new YAML config structs
- [x] No new YAML loading
- [x] Division denominators validated (the entire point of this PR)
- [x] No new interfaces
- [x] No multi-concern methods
- [x] No config struct modifications
- [x] Grepped for "PR 236" / "#236" references — none found in codebase

---

## Appendix: File-Level Implementation Details

### File: `sim/model_hardware_config.go`

**Purpose:** Add `ValidateRooflineConfig` function at end of file.

**Implementation:**

```go
// invalidPositiveFloat returns true if v is not a valid positive float64
// (i.e., v <= 0, NaN, or Inf). Used to validate roofline config denominators.
func invalidPositiveFloat(v float64) bool {
	return v <= 0 || math.IsNaN(v) || math.IsInf(v, 0)
}

// ValidateRooflineConfig checks that all fields required by the roofline latency
// model are valid positive values. Returns an error listing all invalid fields, or nil if valid.
// This validates fields that appear as denominators in roofline_step.go:
// - NumHeads: used in dModel/nHeads (lines 16, 95)
// - TFlopsPeak * MfuPrefill: used in prefill compute (line 167)
// - TFlopsPeak * MfuDecode: used in decode compute (line 188)
// - TFlopsPeak also guards vectorPeak (= peakFlops * 0.10) used at lines 167, 188
// - BwPeakTBs * BwEffConstant: used in memory bandwidth (lines 168, 189)
func ValidateRooflineConfig(mc ModelConfig, hc HardwareCalib) error {
	var problems []string

	if mc.NumHeads <= 0 {
		problems = append(problems, fmt.Sprintf("ModelConfig.NumHeads must be > 0, got %d", mc.NumHeads))
	}
	if invalidPositiveFloat(hc.TFlopsPeak) {
		problems = append(problems, fmt.Sprintf("HardwareCalib.TFlopsPeak must be a valid positive number, got %v", hc.TFlopsPeak))
	}
	if invalidPositiveFloat(hc.BwPeakTBs) {
		problems = append(problems, fmt.Sprintf("HardwareCalib.BwPeakTBs must be a valid positive number, got %v", hc.BwPeakTBs))
	}
	if invalidPositiveFloat(hc.BwEffConstant) {
		problems = append(problems, fmt.Sprintf("HardwareCalib.BwEffConstant must be a valid positive number, got %v", hc.BwEffConstant))
	}
	if invalidPositiveFloat(hc.MfuPrefill) {
		problems = append(problems, fmt.Sprintf("HardwareCalib.MfuPrefill must be a valid positive number, got %v", hc.MfuPrefill))
	}
	if invalidPositiveFloat(hc.MfuDecode) {
		problems = append(problems, fmt.Sprintf("HardwareCalib.MfuDecode must be a valid positive number, got %v", hc.MfuDecode))
	}

	if len(problems) > 0 {
		return fmt.Errorf("invalid roofline config: %s", strings.Join(problems, "; "))
	}
	return nil
}
```

**Imports to add:** `"math"` and `"strings"` (neither currently imported in model_hardware_config.go)

### File: `sim/simulator.go`

**Purpose:** Add roofline validation call in `NewSimulator`.

**Location:** After line ~166 (`BlockSizeTokens` check), before line ~168 (`s := &Simulator{`).

**Implementation:**

```go
	if cfg.Roofline {
		if err := ValidateRooflineConfig(cfg.ModelConfig, cfg.HWConfig); err != nil {
			return nil, fmt.Errorf("roofline validation: %w", err)
		}
	}
```

### File: `sim/roofline_step.go`

**Purpose:** Fix non-deterministic map iteration in `calculateMemoryAccessBytes`.

**Location:** Lines 129-133.

**Before:**
```go
	var total float64
	for _, v := range mem {
		total += v
	}
	mem["total"] = total
```

**After:**
```go
	// Sort keys before accumulation for deterministic float summation
	// (Go map iteration order is non-deterministic — antipattern #2)
	keys := make([]string, 0, len(mem))
	for k := range mem {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var total float64
	for _, k := range keys {
		total += mem[k]
	}
	mem["total"] = total
```

**Imports to add:** `"sort"` (currently only `"math"`)

### File: `sim/roofline_step_test.go` (new)

**Purpose:** Determinism invariant test for `calculateMemoryAccessBytes`.

```go
package sim

import (
	"testing"
)

func TestCalculateMemoryAccessBytes_Deterministic(t *testing.T) {
	config := ModelConfig{
		NumLayers:       32,
		HiddenDim:       4096,
		NumHeads:        32,
		NumKVHeads:      8,
		BytesPerParam:   2,
		IntermediateDim: 14336,
	}

	var firstTotal float64
	for i := 0; i < 100; i++ {
		result := calculateMemoryAccessBytes(config, 1024, 64, true)
		if i == 0 {
			firstTotal = result["total"]
		} else if result["total"] != firstTotal {
			t.Fatalf("non-deterministic total: call 0 got %v, call %d got %v", firstTotal, i, result["total"])
		}
	}

	if firstTotal <= 0 {
		t.Errorf("expected positive total, got %v", firstTotal)
	}
}
```

### File: `sim/model_hardware_config_test.go` (appended tests)

All new test functions shown in Tasks 1-2 above. No modifications to existing tests.
