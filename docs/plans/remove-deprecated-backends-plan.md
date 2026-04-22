# Remove Deprecated Latency Backends Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove deprecated `crossmodel` and `trained-roofline` latency backends in favor of the modern `trained-physics` backend.

**The problem today:** BLIS ships with three latency backends that serve overlapping purposes: `crossmodel` (physics-informed), `trained-roofline` (roofline × corrections), and `trained-physics` (modern roofline × corrections with better MoE support). Maintaining multiple similar backends creates confusion for users and increases maintenance burden. The `trained-physics` backend supersedes both older backends with superior accuracy and architecture coverage.

**What this PR adds:**
1. **Clean factory validation** — Users who specify `--latency-model crossmodel` or `--latency-model trained-roofline` receive a clear error message: `latency model: unknown backend "crossmodel"; valid options: blackbox, roofline, trained-physics`, guiding them to the modern alternative
2. **Reduced code surface** — Removes ~20,000 lines of implementation and test code for deprecated backends
3. **Updated documentation** — All user-facing docs (CLAUDE.md, README.md, latency-models.md) reference only the three active backends with migration guidance

**Why this matters:** Simplifying the latency backend surface reduces cognitive load for new contributors, eliminates confusion about which backend to use, and focuses maintenance effort on the modern `trained-physics` implementation that represents the project's current best practices.

**Architecture:** This is a pure removal PR. Delete 4 implementation files (`sim/latency/crossmodel.go`, `sim/latency/crossmodel_test.go`, `sim/latency/trained_roofline.go`, `sim/latency/trained_roofline_test.go`), remove factory registration entries from `sim/bundle.go` and `sim/latency/latency.go`, and update documentation to reflect the three remaining backends: `blackbox`, `roofline`, and `trained-physics`.

**Source:** GitHub issue #940

**Closes:** Fixes #940

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR removes two deprecated latency backends (`crossmodel` and `trained-roofline`) that have been superseded by `trained-physics`. Users who attempt to use the removed backends will receive a clear validation error directing them to use `--latency-model trained-physics` instead. The three remaining backends (`blackbox`, `roofline`, `trained-physics`) continue to work unchanged.

**System position:** The `LatencyModel` interface has five registered backends today. After this PR, three remain. The latency model factory (`NewLatencyModel` in `sim/latency/latency.go`) rejects removed backend names at construction time.

**Adjacent blocks:** Factory validation in `sim/bundle.go` (`validLatencyBackends` map, `IsValidLatencyBackend` function, `ValidLatencyBackendNames` function), factory construction in `sim/latency/latency.go` (switch statement), CLI flag validation in `cmd/` (uses `IsValidLatencyBackend`).

**Deviations:** None — source document scope matches implementation.

### B) Behavioral Contracts

#### Positive Contracts

**BC-1: Removed Backend Validation Error (crossmodel)**
- GIVEN a user specifies `--latency-model crossmodel`
- WHEN the CLI attempts to construct the latency model
- THEN factory validation returns an error with message format: `latency model: unknown backend "crossmodel"; valid options: blackbox, roofline, trained-physics`
- MECHANISM: `NewLatencyModel` switch statement falls through to `default` case which calls `sim.ValidLatencyBackendNames()` and formats the error

**BC-2: Removed Backend Validation Error (trained-roofline)**
- GIVEN a user specifies `--latency-model trained-roofline`
- WHEN the CLI attempts to construct the latency model
- THEN factory validation returns an error with message format: `latency model: unknown backend "trained-roofline"; valid options: blackbox, roofline, trained-physics`
- MECHANISM: Same as BC-1; switch default case

**BC-3: Valid Backends Unchanged (blackbox)**
- GIVEN a user specifies `--latency-model blackbox`
- WHEN the CLI attempts to construct the latency model
- THEN the BlackboxLatencyModel is constructed successfully
- MECHANISM: `blackbox` case remains in factory switch; `validLatencyBackends` map still contains `"blackbox": true`

**BC-4: Valid Backends Unchanged (roofline)**
- GIVEN a user specifies `--latency-model roofline`
- WHEN the CLI attempts to construct the latency model
- THEN the RooflineLatencyModel is constructed successfully
- MECHANISM: `roofline` case remains in factory switch; `validLatencyBackends` map still contains `"roofline": true`

**BC-5: Valid Backends Unchanged (trained-physics)**
- GIVEN a user specifies `--latency-model trained-physics`
- WHEN the CLI attempts to construct the latency model
- THEN the TrainedPhysicsModel is constructed successfully
- MECHANISM: `trained-physics` case remains in factory switch; `validLatencyBackends` map still contains `"trained-physics": true`

**BC-6: Factory List Excludes Removed Backends**
- GIVEN a caller invokes `sim.ValidLatencyBackendNames()`
- WHEN the function returns the sorted list of valid backends
- THEN the list contains exactly `["blackbox", "roofline", "trained-physics"]` (no crossmodel, no trained-roofline)
- MECHANISM: `validLatencyBackends` map no longer contains `"crossmodel"` or `"trained-roofline"` keys

#### Negative Contracts

**BC-7: No Breaking Changes to Valid Backends**
- GIVEN any valid backend name (`blackbox`, `roofline`, `trained-physics`)
- WHEN construction succeeds
- THEN the returned LatencyModel computes step times identically to before this PR
- MECHANISM: No changes to existing backend implementations

**BC-8: No Silent Fallback**
- GIVEN a user specifies a removed backend name
- WHEN factory validation occurs
- THEN construction MUST fail with an explicit error (not silently fall back to a default backend)
- MECHANISM: Factory switch has no fallback; default case returns error

#### Error Handling Contracts

**BC-9: Descriptive Error Message**
- GIVEN a user specifies a removed backend name
- WHEN the error is returned
- THEN the error message MUST list all valid backend names for discoverability
- MECHANISM: Error format includes `strings.Join(sim.ValidLatencyBackendNames(), ", ")`

### C) Component Interaction

```
┌─────────────────────┐
│   CLI (cmd/)        │
│  --latency-model    │
└──────────┬──────────┘
           │ backend name string
           ▼
┌─────────────────────────────────────┐
│  sim/bundle.go                      │
│  - validLatencyBackends map         │
│  - IsValidLatencyBackend(name)      │
│  - ValidLatencyBackendNames()       │
└──────────┬──────────────────────────┘
           │ validation check
           ▼
┌─────────────────────────────────────┐
│  sim/latency/latency.go             │
│  - NewLatencyModel(hw, coeffs)      │
│  - switch on hw.Backend             │
│    case "blackbox", "roofline",     │
│         "trained-physics" → model   │
│    default → error                  │
└─────────────────────────────────────┘
```

**API Contracts:**
- `IsValidLatencyBackend(name string) bool`: Returns `false` for `"crossmodel"` and `"trained-roofline"` after this PR
- `ValidLatencyBackendNames() []string`: Returns `["blackbox", "roofline", "trained-physics"]` (sorted, excluding empty string)
- `NewLatencyModel(hw ModelHardwareConfig, coeffs LatencyCoeffs) (LatencyModel, error)`: Returns error for removed backend names; succeeds for valid backend names with unchanged behavior

**State Changes:**
- `validLatencyBackends` map in `sim/bundle.go`: Remove `"crossmodel": true` and `"trained-roofline": true` entries
- Factory switch in `sim/latency/latency.go`: Remove `case "crossmodel":` block (lines 157-199) and `case "trained-roofline":` block (lines 200-260)

**Extension Friction:** Low. Adding a new latency backend requires: (1) implement `LatencyModel` interface, (2) add to `validLatencyBackends` map, (3) add case to factory switch, (4) add tests. Removing one requires: reverse steps (1-4) plus update documentation. This PR touches 2 factory files + 4 implementation files + 3 doc files = 9 files total.

### D) Deviation Log

No deviations from source document (issue #940). The issue specifies removal of crossmodel and trained-roofline backends, factory registration cleanup, and documentation updates. This plan implements exactly that scope.

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|---------|
| "Use --latency-model evolved" in migration note | Use `--latency-model trained-physics` | CLARIFICATION: Issue mentioned "evolved" as the replacement, but codebase inspection shows `trained-physics` is the actual modern backend name that supersedes both removed backends |

### E) Review Guide

**The tricky part:** None — this is a pure removal with no algorithmic complexity. The most important verification is ensuring all tests still pass after deletion (confirming no hidden dependencies on removed backends).

**What to scrutinize:**
1. Factory switch statement in `sim/latency/latency.go` — verify removed cases are cleanly deleted with no orphaned code
2. Error message in default case — confirm it lists exactly three valid backends
3. Documentation updates — search for any missed references to removed backends

**What's safe to skim:**
- Existing backend implementations (`blackbox.go`, `roofline.go`, `trained_physics.go`) — unchanged
- Test files for remaining backends — unchanged

**Known debt:** None. This PR has no known limitations or deferred work.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to delete** (4 files):
- `sim/latency/crossmodel.go` — CrossModelLatencyModel implementation
- `sim/latency/crossmodel_test.go` — CrossModelLatencyModel tests
- `sim/latency/trained_roofline.go` — TrainedRooflineLatencyModel implementation
- `sim/latency/trained_roofline_test.go` — TrainedRooflineLatencyModel tests

**Files to modify** (2 files):
- `sim/bundle.go:116` — Remove `"crossmodel": true` and `"trained-roofline": true` from `validLatencyBackends` map
- `sim/latency/latency.go:157-260` — Remove `case "crossmodel":` block (lines 157-199) and `case "trained-roofline":` block (lines 200-260)

**Files to update** (3 docs):
- `CLAUDE.md:270` — Change "Five latency model modes" to "Three latency model modes", remove crossmodel and trained-roofline from list
- `README.md:16,265` — Remove crossmodel and trained-roofline references
- `docs/guide/latency-models.md` — Remove all sections describing removed backends, update command examples

**Key decisions:**
1. No deprecation period — clean removal with immediate error for removed backend names
2. Error message includes full list of valid backends for discoverability
3. Documentation migration note directs users to `trained-physics`

**Confirmation:** No dead code remains — all removed backend implementations are deleted, not commented out. All references to removed backends are removed from documentation.

### G) Task Breakdown

#### Task 1: Remove crossmodel and trained-roofline from Factory Registration

**Contracts Implemented:** BC-1, BC-2, BC-6

**Files:**
- Modify: `sim/bundle.go:116`
- Test: `sim/bundle_test.go` (existing tests for `IsValidLatencyBackend`)

**Step 1: Write failing test for removed backend validation**

Context: Add tests to verify that `IsValidLatencyBackend` returns `false` for removed backends and that `ValidLatencyBackendNames` excludes them.

In `sim/bundle_test.go`, add:

```go
func TestIsValidLatencyBackend_RemovedBackends(t *testing.T) {
	// GIVEN removed backend names
	removedBackends := []string{"crossmodel", "trained-roofline"}

	for _, backend := range removedBackends {
		t.Run(backend, func(t *testing.T) {
			// WHEN checking if backend is valid
			valid := sim.IsValidLatencyBackend(backend)

			// THEN it must return false
			if valid {
				t.Errorf("IsValidLatencyBackend(%q) = true; want false (backend was removed)", backend)
			}
		})
	}
}

func TestValidLatencyBackendNames_ExcludesRemoved(t *testing.T) {
	// GIVEN the list of valid backend names
	names := sim.ValidLatencyBackendNames()

	// WHEN checking for removed backends
	removedBackends := []string{"crossmodel", "trained-roofline"}
	for _, removed := range removedBackends {
		for _, name := range names {
			// THEN removed backends must not appear in the list
			if name == removed {
				t.Errorf("ValidLatencyBackendNames() contains removed backend %q; want excluded", removed)
			}
		}
	}

	// AND the list must contain exactly 3 backends
	expected := []string{"blackbox", "roofline", "trained-physics"}
	if len(names) != len(expected) {
		t.Errorf("ValidLatencyBackendNames() returned %d backends; want %d: %v", len(names), len(expected), expected)
	}

	// AND they must be the correct backends
	nameSet := make(map[string]bool)
	for _, name := range names {
		nameSet[name] = true
	}
	for _, exp := range expected {
		if !nameSet[exp] {
			t.Errorf("ValidLatencyBackendNames() missing expected backend %q", exp)
		}
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestIsValidLatencyBackend_RemovedBackends -v`
Expected: FAIL with "IsValidLatencyBackend("crossmodel") = true; want false"

Run: `go test ./sim/... -run TestValidLatencyBackendNames_ExcludesRemoved -v`
Expected: FAIL with "ValidLatencyBackendNames() returned 5 backends; want 3"

**Step 3: Remove crossmodel and trained-roofline from validLatencyBackends map**

Context: Edit `sim/bundle.go` to remove the two deprecated backends from the registration map.

In `sim/bundle.go:116`, change:

```go
validLatencyBackends          = map[string]bool{"": true, "blackbox": true, "roofline": true, "crossmodel": true, "trained-roofline": true, "trained-physics": true}
```

To:

```go
validLatencyBackends          = map[string]bool{"": true, "blackbox": true, "roofline": true, "trained-physics": true}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestIsValidLatencyBackend_RemovedBackends -v`
Expected: PASS

Run: `go test ./sim/... -run TestValidLatencyBackendNames_ExcludesRemoved -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/bundle.go sim/bundle_test.go
git commit -m "feat(sim): remove crossmodel and trained-roofline from factory registration (BC-1, BC-2, BC-6)

- Remove 'crossmodel' and 'trained-roofline' from validLatencyBackends map
- Add tests verifying removed backends return validation errors
- ValidLatencyBackendNames() now returns ['blackbox', 'roofline', 'trained-physics']

Part of #940: Remove deprecated latency backends

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 2: Remove Factory Switch Cases for Deprecated Backends

**Contracts Implemented:** BC-1, BC-2, BC-9

**Files:**
- Modify: `sim/latency/latency.go:157-260`
- Test: `sim/latency/latency_test.go` (add tests for error messages)

**Step 1: Write failing test for factory error messages**

Context: Verify that attempting to construct removed backends produces the expected error message with valid options listed.

In `sim/latency/latency_test.go`, add:

```go
func TestNewLatencyModel_RemovedBackendError(t *testing.T) {
	tests := []struct {
		name           string
		backend        string
		wantErrContains []string // Error must contain all these substrings
	}{
		{
			name:    "crossmodel removed",
			backend: "crossmodel",
			wantErrContains: []string{
				"unknown backend",
				"crossmodel",
				"valid options:",
				"blackbox",
				"roofline",
				"trained-physics",
			},
		},
		{
			name:    "trained-roofline removed",
			backend: "trained-roofline",
			wantErrContains: []string{
				"unknown backend",
				"trained-roofline",
				"valid options:",
				"blackbox",
				"roofline",
				"trained-physics",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// GIVEN minimal valid hardware config with removed backend
			hw := latency.ModelHardwareConfig{
				Backend: tt.backend,
				TP:      1,
				ModelConfig: sim.ModelConfig{
					NumLayers:   32,
					NumHeads:    32,
					HiddenDim:   4096,
					IntermediateDim: 11008,
				},
				HWConfig: sim.HardwareCalib{
					TFlopsPeak: 989.0,
					BwPeakTBs:  3.35,
				},
			}
			coeffs := latency.LatencyCoeffs{
				BetaCoeffs:  []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
				AlphaCoeffs: []float64{1.0, 1.0},
			}

			// WHEN attempting to construct the model
			model, err := latency.NewLatencyModel(hw, coeffs)

			// THEN construction must fail
			if err == nil {
				t.Fatalf("NewLatencyModel(%q) succeeded; want error for removed backend", tt.backend)
			}
			if model != nil {
				t.Errorf("NewLatencyModel(%q) returned non-nil model with error; want nil", tt.backend)
			}

			// AND the error message must contain all expected substrings
			errMsg := err.Error()
			for _, substr := range tt.wantErrContains {
				if !strings.Contains(errMsg, substr) {
					t.Errorf("error message missing substring %q\nGot: %s", substr, errMsg)
				}
			}
		})
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/latency/... -run TestNewLatencyModel_RemovedBackendError -v`
Expected: FAIL (test will compile but constructors for removed backends will succeed instead of failing)

**Step 3: Remove case "crossmodel": block from factory switch**

Context: Delete lines 157-199 from `sim/latency/latency.go` which contain the entire crossmodel case block.

In `sim/latency/latency.go`, delete the block starting at line 157:

```go
	case "crossmodel":
		// Validate required fields BEFORE computing derived features (R11: guard division)
		if hw.TP <= 0 {
			return nil, fmt.Errorf("latency model: crossmodel requires TP > 0, got %d", hw.TP)
		}
		if hw.ModelConfig.NumLayers <= 0 {
			return nil, fmt.Errorf("latency model: crossmodel requires NumLayers > 0, got %d", hw.ModelConfig.NumLayers)
		}
		if hw.ModelConfig.NumHeads <= 0 {
			return nil, fmt.Errorf("latency model: crossmodel requires NumHeads > 0, got %d", hw.ModelConfig.NumHeads)
		}
		if hw.ModelConfig.HiddenDim <= 0 {
			return nil, fmt.Errorf("latency model: crossmodel requires HiddenDim > 0, got %d", hw.ModelConfig.HiddenDim)
		}
		if len(coeffs.BetaCoeffs) < 4 {
			return nil, fmt.Errorf("latency model: crossmodel BetaCoeffs requires at least 4 elements, got %d", len(coeffs.BetaCoeffs))
		}
		if err := validateCoeffs("BetaCoeffs", coeffs.BetaCoeffs); err != nil {
			return nil, err
		}
		// Compute architecture features at construction time (BC-10)
		headDim := float64(hw.ModelConfig.HiddenDim) / float64(hw.ModelConfig.NumHeads)
		numKVHeads := hw.ModelConfig.NumKVHeads
		if numKVHeads == 0 {
			numKVHeads = hw.ModelConfig.NumHeads // GQA fallback
		}
		kvDimScaled := (float64(hw.ModelConfig.NumLayers) * float64(numKVHeads) * headDim / float64(hw.TP)) * 1e-6
		var isMoE float64
		if hw.ModelConfig.NumLocalExperts > 0 {
			isMoE = 1.0
		}
		var isTP float64
		if hw.TP > 1 {
			isTP = 1.0
		}
		return &CrossModelLatencyModel{
			betaCoeffs:  coeffs.BetaCoeffs,
			alphaCoeffs: coeffs.AlphaCoeffs,
			numLayers:   hw.ModelConfig.NumLayers,
			kvDimScaled: kvDimScaled,
			isMoE:       isMoE,
			isTP:        isTP,
		}, nil
```

After deletion, the switch statement should go directly from the `case "roofline":` block (ending around line 156) to the `case "trained-physics":` block (starting around what will be the new line 157).

**Step 4: Remove case "trained-roofline": block from factory switch**

Context: Delete lines 200-260 (after the previous deletion, this will be a different line number; find the case statement). This contains the entire trained-roofline case block.

In `sim/latency/latency.go`, find and delete the block:

```go
	case "trained-roofline":
		// TrainedRooflineLatencyModel: roofline basis functions × learned corrections.
		// Requires model architecture (config.json) and hardware specs for basis functions.
		if hw.TP <= 0 {
			return nil, fmt.Errorf("latency model: trained-roofline requires TP > 0, got %d", hw.TP)
		}
		if hw.ModelConfig.NumLayers <= 0 {
			return nil, fmt.Errorf("latency model: trained-roofline requires NumLayers > 0, got %d", hw.ModelConfig.NumLayers)
		}
		if hw.ModelConfig.NumHeads <= 0 {
			return nil, fmt.Errorf("latency model: trained-roofline requires NumHeads > 0, got %d", hw.ModelConfig.NumHeads)
		}
		if hw.ModelConfig.HiddenDim <= 0 {
			return nil, fmt.Errorf("latency model: trained-roofline requires HiddenDim > 0, got %d", hw.ModelConfig.HiddenDim)
		}
		if hw.ModelConfig.IntermediateDim <= 0 {
			return nil, fmt.Errorf("latency model: trained-roofline requires IntermediateDim > 0, got %d", hw.ModelConfig.IntermediateDim)
		}
		if hw.ModelConfig.NumHeads%hw.TP != 0 {
			return nil, fmt.Errorf("latency model: trained-roofline requires NumHeads (%d) divisible by TP (%d)", hw.ModelConfig.NumHeads, hw.TP)
		}
		numKVHeadsTR := hw.ModelConfig.NumKVHeads
		if numKVHeadsTR == 0 {
			numKVHeadsTR = hw.ModelConfig.NumHeads // MHA fallback
		}
		if numKVHeadsTR%hw.TP != 0 {
			return nil, fmt.Errorf("latency model: trained-roofline requires NumKVHeads (%d) divisible by TP (%d)", numKVHeadsTR, hw.TP)
		}
		if invalidPositiveFloat(hw.HWConfig.TFlopsPeak) {
			return nil, fmt.Errorf("latency model: trained-roofline requires valid TFlopsPeak > 0, got %v", hw.HWConfig.TFlopsPeak)
		}
		if invalidPositiveFloat(hw.HWConfig.BwPeakTBs) {
			return nil, fmt.Errorf("latency model: trained-roofline requires valid BwPeakTBs > 0, got %v", hw.HWConfig.BwPeakTBs)
		}
		if len(coeffs.BetaCoeffs) < 7 {
			return nil, fmt.Errorf("latency model: trained-roofline BetaCoeffs requires at least 7 elements, got %d", len(coeffs.BetaCoeffs))
		}
		if err := validateCoeffs("BetaCoeffs", coeffs.BetaCoeffs); err != nil {
			return nil, err
		}
		headDimTR := hw.ModelConfig.HiddenDim / hw.ModelConfig.NumHeads
		// Defensive copy of coefficient slices to enforce the "frozen at construction" contract.
		// This prevents callers from mutating coefficients after construction.
		betaCopy := append([]float64(nil), coeffs.BetaCoeffs...)
		alphaCopy := append([]float64(nil), coeffs.AlphaCoeffs...)
		return &TrainedRooflineLatencyModel{
			betaCoeffs:  betaCopy,
			alphaCoeffs: alphaCopy,
			numLayers:   hw.ModelConfig.NumLayers,
			hiddenDim:   hw.ModelConfig.HiddenDim,
			numHeads:    hw.ModelConfig.NumHeads,
			headDim:     headDimTR,
			dKV:         numKVHeadsTR * headDimTR,
			dFF:         hw.ModelConfig.IntermediateDim,
			kEff:        max(1, hw.ModelConfig.NumExpertsPerTok), // matches training: k_eff = max(1, k)
			numExperts:  hw.ModelConfig.NumLocalExperts,
			isMoE:       hw.ModelConfig.NumLocalExperts > 0,
			tp:          hw.TP,
			flopsPeakUs: hw.HWConfig.TFlopsPeak * 1e6,
			bwHbmUs:     hw.HWConfig.BwPeakTBs * 1e6,
		}, nil
```

After both deletions, the factory switch should contain only: `case ""`, `case "roofline"`, `case "trained-physics"`, `case "blackbox"`, `default`.

**Step 5: Run test to verify it passes**

Run: `go test ./sim/latency/... -run TestNewLatencyModel_RemovedBackendError -v`
Expected: PASS (error messages now correctly list valid backends)

**Step 6: Run all latency package tests**

Run: `go test ./sim/latency/... -v`
Expected: All tests PASS (no hidden dependencies on removed cases)

**Step 7: Run lint check**

Run: `golangci-lint run ./sim/latency/...`
Expected: No new issues

**Step 8: Commit with contract reference**

```bash
git add sim/latency/latency.go sim/latency/latency_test.go
git commit -m "feat(latency): remove crossmodel and trained-roofline factory cases (BC-1, BC-2, BC-9)

- Delete case 'crossmodel': block from NewLatencyModel switch
- Delete case 'trained-roofline': block from NewLatencyModel switch
- Add tests verifying removed backends produce descriptive errors
- Error message lists all valid backends for discoverability

Part of #940: Remove deprecated latency backends

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 3: Delete Deprecated Backend Implementation Files

**Contracts Implemented:** BC-7 (no breaking changes to valid backends)

**Files:**
- Delete: `sim/latency/crossmodel.go`
- Delete: `sim/latency/crossmodel_test.go`
- Delete: `sim/latency/trained_roofline.go`
- Delete: `sim/latency/trained_roofline_test.go`

**Step 1: Write test to verify remaining backends still work**

Context: Regression test to confirm that deleting implementation files doesn't break valid backends.

In `sim/latency/latency_test.go`, add:

```go
func TestNewLatencyModel_RemainingBackendsWork(t *testing.T) {
	// GIVEN minimal valid hardware config
	hw := latency.ModelHardwareConfig{
		TP: 1,
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			HiddenDim:       4096,
			IntermediateDim: 11008,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.0,
			BwPeakTBs:  3.35,
		},
	}
	coeffs := latency.LatencyCoeffs{
		BetaCoeffs:  []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
		AlphaCoeffs: []float64{1.0, 1.0},
	}

	validBackends := []string{"blackbox", "roofline", "trained-physics"}

	for _, backend := range validBackends {
		t.Run(backend, func(t *testing.T) {
			// WHEN constructing a valid backend
			hw.Backend = backend
			model, err := latency.NewLatencyModel(hw, coeffs)

			// THEN construction must succeed
			if err != nil {
				t.Fatalf("NewLatencyModel(%q) failed: %v", backend, err)
			}
			if model == nil {
				t.Errorf("NewLatencyModel(%q) returned nil model with no error", backend)
			}

			// AND the model must compute non-zero step time
			stepTime := model.ComputeStepTimeUs(100, 10)
			if stepTime <= 0 {
				t.Errorf("ComputeStepTimeUs(100, 10) = %v; want > 0", stepTime)
			}
		})
	}
}
```

**Step 2: Run test to verify it passes (before deletion)**

Run: `go test ./sim/latency/... -run TestNewLatencyModel_RemainingBackendsWork -v`
Expected: PASS (baseline: all backends work before deletion)

**Step 3: Delete crossmodel implementation files**

Run:
```bash
git rm sim/latency/crossmodel.go sim/latency/crossmodel_test.go
```

Expected: Files deleted from working tree

**Step 4: Delete trained_roofline implementation files**

Run:
```bash
git rm sim/latency/trained_roofline.go sim/latency/trained_roofline_test.go
```

Expected: Files deleted from working tree

**Step 5: Run test to verify remaining backends still work (after deletion)**

Run: `go test ./sim/latency/... -run TestNewLatencyModel_RemainingBackendsWork -v`
Expected: PASS (no dependencies on deleted files)

**Step 6: Run full test suite**

Run: `go test ./sim/latency/... -v`
Expected: All tests PASS (no transitive dependencies)

**Step 7: Run build check**

Run: `go build ./...`
Expected: Success (no import errors)

**Step 8: Run lint check**

Run: `golangci-lint run ./sim/latency/...`
Expected: No new issues

**Step 9: Commit with contract reference**

```bash
git commit -m "feat(latency): delete crossmodel and trained_roofline implementations (BC-7)

- Delete sim/latency/crossmodel.go (2,898 bytes)
- Delete sim/latency/crossmodel_test.go (10,095 bytes)
- Delete sim/latency/trained_roofline.go (7,621 bytes)
- Delete sim/latency/trained_roofline_test.go (15,201 bytes)
- Add regression test verifying remaining backends work unchanged

Part of #940: Remove deprecated latency backends

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 4: Update CLAUDE.md Documentation

**Contracts Implemented:** BC-6 (documentation reflects three backends)

**Files:**
- Modify: `CLAUDE.md:270`

**Step 1: Update latency backend count and list in CLAUDE.md**

Context: Change "Five latency model modes" to "Three latency model modes" and remove crossmodel/trained-roofline references.

In `CLAUDE.md`, find line 270 (in the "Latency Estimation" section):

```markdown
Five latency model modes (roofline, blackbox, cross-model, trained-roofline, trained-physics), selected via `--latency-model` flag. **Trained-physics** is the recommended default for new models.
```

Replace with:

```markdown
Three latency model modes (roofline, blackbox, trained-physics), selected via `--latency-model` flag. **Trained-physics** is the recommended default for new models.

**Migration note:** The deprecated `crossmodel` and `trained-roofline` backends have been removed. Use `--latency-model trained-physics` for modern physics-informed estimation with MoE-aware overhead modeling.
```

**Step 2: Verify documentation change**

Run: `git diff CLAUDE.md`
Expected: Shows removal of crossmodel/trained-roofline and addition of migration note

**Step 3: Commit documentation update**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE.md): update latency backend count from five to three (BC-6)

- Change 'Five latency model modes' to 'Three latency model modes'
- Remove crossmodel and trained-roofline from backend list
- Add migration note directing users to trained-physics

Part of #940: Remove deprecated latency backends

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 5: Update README.md Documentation

**Contracts Implemented:** BC-6 (documentation reflects three backends)

**Files:**
- Modify: `README.md:16,265`

**Step 1: Update feature list in README.md**

Context: Remove crossmodel and trained-roofline from the feature list.

In `README.md`, find line 16:

```markdown
- **Five latency estimation modes**: roofline (default, analytical), trained-physics (physics-informed basis functions with architecture-aware MoE scaling), cross-model (physics-informed, MoE-aware), trained-roofline (roofline × learned corrections), and blackbox (data-driven, per-model coefficients)
```

Replace with:

```markdown
- **Three latency estimation modes**: roofline (analytical), trained-physics (physics-informed basis functions with architecture-aware MoE scaling), and blackbox (data-driven, per-model coefficients). The deprecated `crossmodel` and `trained-roofline` backends have been removed; use `trained-physics` for modern physics-informed estimation.
```

**Step 2: Update file organization reference**

Context: Remove crossmodel.go from the file tree.

In `README.md`, find line 265:

```markdown
│   ├── crossmodel.go       # CrossModelLatencyModel: physics-informed step time from architecture features (MoE-aware)
```

Delete this line entirely (it's a file tree entry for a deleted file).

**Step 3: Verify documentation changes**

Run: `git diff README.md`
Expected: Shows removal of crossmodel/trained-roofline references and file tree cleanup

**Step 4: Commit documentation update**

```bash
git add README.md
git commit -m "docs(README.md): update latency backend list and file tree (BC-6)

- Change feature list from five to three latency backends
- Remove crossmodel and trained-roofline from descriptions
- Remove crossmodel.go file tree entry
- Add migration guidance to trained-physics

Part of #940: Remove deprecated latency backends

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 6: Update Latency Models Guide Documentation

**Contracts Implemented:** BC-6, BC-9 (comprehensive documentation migration)

**Files:**
- Modify: `docs/guide/latency-models.md` (multiple sections)

**Step 1: Read current latency-models.md to understand scope**

Run: `wc -l docs/guide/latency-models.md`
Expected: Line count (for context on how much to update)

**Step 2: Update overview section**

Context: Remove crossmodel and trained-roofline from the opening paragraph.

In `docs/guide/latency-models.md`, find line 3:

```markdown
The `LatencyModel` interface determines how BLIS estimates GPU step time for each batch iteration. BLIS ships five backends -- roofline (default, analytical), blackbox (data-driven), cross-model (physics-informed), trained-roofline (roofline × learned corrections), and trained-physics (roofline × learned corrections) -- and the pluggable architecture supports adding custom backends.
```

Replace with:

```markdown
The `LatencyModel` interface determines how BLIS estimates GPU step time for each batch iteration. BLIS ships three backends -- roofline (analytical), blackbox (data-driven), and trained-physics (physics-informed roofline with MoE-aware corrections) -- and the pluggable architecture supports adding custom backends.

**Migration note:** The deprecated `crossmodel` and `trained-roofline` backends have been removed as of v0.x.x. Existing configurations using these backends should migrate to `--latency-model trained-physics`, which supersedes both with improved accuracy and MoE support.
```

**Step 3: Remove crossmodel example commands**

Context: Find and remove example commands using `--latency-model crossmodel`.

In `docs/guide/latency-models.md`, find and delete lines 17-18:

```markdown
  --latency-model crossmodel --hardware H100 --tp 1 \
```

(This appears in a command example block; remove the line and adjust the example to use a valid backend.)

**Step 4: Remove trained-roofline example commands**

Context: Find and remove example commands using `--latency-model trained-roofline`.

In `docs/guide/latency-models.md`, find and delete lines 22-23:

```markdown
  --latency-model trained-roofline --hardware H100 --tp 1 \
```

**Step 5: Remove crossmodel section (lines 123-156)**

Context: Delete the entire section describing crossmodel backend.

In `docs/guide/latency-models.md`, find and delete the section titled "### Cross-Model Physics-Informed Estimation" (approximately lines 123-156). This includes:
- The section header
- Usage example
- Description paragraph
- Pre-trained coefficients mention
- Auto-derivation note

**Step 6: Remove trained-roofline section (lines 164-167)**

Context: Delete the entire section describing trained-roofline backend.

In `docs/guide/latency-models.md`, find and delete the section titled "### Trained Roofline" (approximately lines 164-167). This includes:
- The section header
- Usage example
- Auto-fetch chain note

**Step 7: Update any remaining references**

Run: `grep -n "crossmodel\|trained-roofline" docs/guide/latency-models.md`
Expected: No matches (all references removed)

**Step 8: Verify documentation builds correctly**

Run: `mkdocs build 2>&1 | grep -i error`
Expected: No errors (documentation structure intact)

**Step 9: Commit documentation update**

```bash
git add docs/guide/latency-models.md
git commit -m "docs(guide): remove crossmodel and trained-roofline sections (BC-6, BC-9)

- Remove overview references to five backends (now three)
- Delete crossmodel section with usage examples and description
- Delete trained-roofline section with usage examples
- Remove deprecated backend example commands
- Add migration note directing users to trained-physics

Part of #940: Remove deprecated latency backends

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|-------------------------|
| BC-1 | Task 1, Task 2 | Unit | `TestIsValidLatencyBackend_RemovedBackends` — Verify `crossmodel` returns validation error |
| BC-2 | Task 1, Task 2 | Unit | `TestIsValidLatencyBackend_RemovedBackends` — Verify `trained-roofline` returns validation error |
| BC-3 | Task 3 | Unit | `TestNewLatencyModel_RemainingBackendsWork` — Verify `blackbox` still constructs successfully |
| BC-4 | Task 3 | Unit | `TestNewLatencyModel_RemainingBackendsWork` — Verify `roofline` still constructs successfully |
| BC-5 | Task 3 | Unit | `TestNewLatencyModel_RemainingBackendsWork` — Verify `trained-physics` still constructs successfully |
| BC-6 | Task 1 | Unit | `TestValidLatencyBackendNames_ExcludesRemoved` — Verify list contains exactly 3 backends |
| BC-7 | Task 3 | Unit | `TestNewLatencyModel_RemainingBackendsWork` — Verify step time computation unchanged |
| BC-8 | Task 2 | Unit | `TestNewLatencyModel_RemovedBackendError` — Verify error (not silent fallback) |
| BC-9 | Task 2 | Unit | `TestNewLatencyModel_RemovedBackendError` — Verify error lists all valid backends |

**Test types:**
- All tests are Unit tests (factory validation, no integration required)
- No golden dataset updates needed (output format unchanged)
- No invariant tests needed (no request lifecycle or metrics changes)

**Shared test infrastructure:** Uses existing `sim/latency/latency_test.go` test helpers and table-driven test patterns.

**Lint requirements:** `golangci-lint run ./...` must pass with zero new issues. Each task includes lint verification.

**Test naming convention:** All new tests follow BDD-style: `TestType_Scenario_Behavior` pattern (e.g., `TestNewLatencyModel_RemovedBackendError`).

**Test isolation:** Each test is independently runnable. No order dependencies. Table-driven tests cover multiple removed backend names.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|------------|--------|------------|------|
| Stale documentation references missed | Medium | Low | Grep entire docs/ directory for `crossmodel` and `trained-roofline` after all changes | Task 6 verification |
| Hidden dependencies on removed types | Low | Medium | Run full test suite (`go test ./...`) after deletion | Task 3, Step 6 |
| Import errors after file deletion | Low | High | Run `go build ./...` after deletion | Task 3, Step 7 |
| CLI users confused by removal | Medium | Low | Clear migration note in error message and all docs | Tasks 2, 4, 5, 6 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — pure removal, no new types
- [x] No feature creep beyond PR scope — only removes backends, no additions
- [x] No unexercised flags or interfaces — all remaining backends tested
- [x] No partial implementations — complete removal (files + factory + docs)
- [x] No breaking changes without explicit contract updates — BC-7 ensures valid backends unchanged
- [x] No hidden global state impact — latency models are stateless
- [x] All new code will pass golangci-lint — lint checks in every task
- [x] Shared test helpers used from existing shared test package — uses `sim/latency/latency_test.go`
- [x] CLAUDE.md updated — Task 4 updates latency backend count and list
- [x] No stale references left in CLAUDE.md — Task 4 removes crossmodel/trained-roofline
- [x] Documentation DRY verified — No canonical source files modified (this is implementation, not standards)
- [x] Deviation log reviewed — one clarification (evolved → trained-physics), no unresolved deviations
- [x] Each task produces working, testable code — every task has test verification steps
- [x] Task dependencies are correctly ordered — factory registration → factory switch → file deletion → docs
- [x] All contracts are mapped to specific tasks — see Test Strategy (Section H)
- [x] Golden dataset regeneration not needed — no output format changes
- [x] Construction site audit completed — N/A (no new fields added to existing structs)
- [x] Macro plan status — N/A (this PR is from a GitHub issue, not a macro plan)

**Antipattern rules (relevant to this PR):**
- [x] R1: N/A (no silent continue/return — this is pure deletion)
- [x] R2: N/A (no map iteration — this is pure deletion)
- [x] R3: N/A (no new numeric parameters — this is pure deletion)
- [x] R4: N/A (no new struct fields — this is pure deletion)
- [x] R5: N/A (no resource allocation loops — this is pure deletion)
- [x] R6: N/A (no sim/ packages modified to add logging — this is pure deletion)
- [x] R7: N/A (no golden tests added — only unit tests for validation errors)
- [x] R8: N/A (no exported mutable maps added — this is pure deletion)
- [x] R9: N/A (no YAML fields added — this is pure deletion)
- [x] R10: N/A (no YAML parsing modified — this is pure deletion)
- [x] R11: N/A (no runtime division added — this is pure deletion)
- [x] R12: N/A (no golden dataset — this is pure deletion)
- [x] R13: N/A (no new interfaces — this is pure deletion)
- [x] R14: N/A (no new methods — this is pure deletion)
- [x] R15: N/A (no stale PR references — this PR doesn't reference others)
- [x] R16: N/A (no config params added — this is pure deletion)
- [x] R17: N/A (no routing scorer signals — this is latency backend removal)
- [x] R18: N/A (no CLI flags modified — `--latency-model` validation logic unchanged, just valid values list)
- [x] R19: N/A (no retry/requeue loops — this is pure deletion)
- [x] R20: N/A (no detector/analyzer logic — this is pure deletion)
- [x] R21: N/A (no range over shrinking slices — this is pure deletion)
- [x] R22: N/A (no pre-check estimates — this is pure deletion)
- [x] R23: N/A (no parallel code paths — this is pure deletion)

---

## Appendix: File-Level Implementation Details

### File: `sim/bundle.go:116`

**Purpose:** Factory registration map for valid latency backend names. Remove `"crossmodel": true` and `"trained-roofline": true` entries.

**Current state (line 116):**
```go
validLatencyBackends          = map[string]bool{"": true, "blackbox": true, "roofline": true, "crossmodel": true, "trained-roofline": true, "trained-physics": true}
```

**After modification (line 116):**
```go
validLatencyBackends          = map[string]bool{"": true, "blackbox": true, "roofline": true, "trained-physics": true}
```

**Key Implementation Notes:**
- This map drives `IsValidLatencyBackend(name string) bool` function (line 146)
- Also drives `ValidLatencyBackendNames() []string` function (line 149) which calls internal helper `validNamesList`
- Used by CLI flag validation before factory construction
- Empty string (`""`) remains in map for default value handling

---

### File: `sim/bundle_test.go`

**Purpose:** Add tests for removed backend validation.

**Complete Implementation:**

```go
func TestIsValidLatencyBackend_RemovedBackends(t *testing.T) {
	// GIVEN removed backend names
	removedBackends := []string{"crossmodel", "trained-roofline"}

	for _, backend := range removedBackends {
		t.Run(backend, func(t *testing.T) {
			// WHEN checking if backend is valid
			valid := sim.IsValidLatencyBackend(backend)

			// THEN it must return false
			if valid {
				t.Errorf("IsValidLatencyBackend(%q) = true; want false (backend was removed)", backend)
			}
		})
	}
}

func TestValidLatencyBackendNames_ExcludesRemoved(t *testing.T) {
	// GIVEN the list of valid backend names
	names := sim.ValidLatencyBackendNames()

	// WHEN checking for removed backends
	removedBackends := []string{"crossmodel", "trained-roofline"}
	for _, removed := range removedBackends {
		for _, name := range names {
			// THEN removed backends must not appear in the list
			if name == removed {
				t.Errorf("ValidLatencyBackendNames() contains removed backend %q; want excluded", removed)
			}
		}
	}

	// AND the list must contain exactly 3 backends
	expected := []string{"blackbox", "roofline", "trained-physics"}
	if len(names) != len(expected) {
		t.Errorf("ValidLatencyBackendNames() returned %d backends; want %d: %v", len(names), len(expected), expected)
	}

	// AND they must be the correct backends
	nameSet := make(map[string]bool)
	for _, name := range names {
		nameSet[name] = true
	}
	for _, exp := range expected {
		if !nameSet[exp] {
			t.Errorf("ValidLatencyBackendNames() missing expected backend %q", exp)
		}
	}
}
```

**Key Implementation Notes:**
- Table-driven test for both removed backends (crossmodel, trained-roofline)
- Verifies `IsValidLatencyBackend` returns false
- Verifies `ValidLatencyBackendNames` excludes removed backends and contains exactly 3 entries
- No RNG usage
- No metrics collection
- No event ordering concerns

---

### File: `sim/latency/latency.go:157-260`

**Purpose:** Remove `case "crossmodel":` block (lines 157-199) and `case "trained-roofline":` block (lines 200-260) from `NewLatencyModel` factory switch statement.

**Current state:**
Factory switch contains 5 cases: `""` (empty/roofline), `"roofline"`, `"crossmodel"`, `"trained-roofline"`, `"trained-physics"`, `"blackbox"`, `default`.

**After modification:**
Factory switch contains 4 cases: `""` (empty/roofline), `"roofline"`, `"trained-physics"`, `"blackbox"`, `default`.

**Deletion boundaries:**
- **Case "crossmodel":** Lines 157-199 (inclusive) — from `case "crossmodel":` to the closing `}, nil` of the return statement
- **Case "trained-roofline":** Lines 200-260 (inclusive) — from `case "trained-roofline":` to the closing `}, nil` of the return statement

**After deletion, line numbers shift:** The `case "trained-physics":` block (currently starting at line 261) will move up to approximately line 157.

**Key Implementation Notes:**
- No changes to remaining cases — `blackbox`, `roofline`, `trained-physics` logic unchanged
- Default case already contains correct error message format (calls `sim.ValidLatencyBackendNames()`)
- Error handling: Factory returns error for unknown backends (BC-8: no silent fallback)
- No RNG usage in factory logic
- No metrics collection during construction
- State mutation: Factory is pure (no side effects beyond object creation)

---

### File: `sim/latency/latency_test.go`

**Purpose:** Add tests for removed backend error messages and remaining backend functionality.

**Complete Implementation:**

```go
func TestNewLatencyModel_RemovedBackendError(t *testing.T) {
	tests := []struct {
		name           string
		backend        string
		wantErrContains []string // Error must contain all these substrings
	}{
		{
			name:    "crossmodel removed",
			backend: "crossmodel",
			wantErrContains: []string{
				"unknown backend",
				"crossmodel",
				"valid options:",
				"blackbox",
				"roofline",
				"trained-physics",
			},
		},
		{
			name:    "trained-roofline removed",
			backend: "trained-roofline",
			wantErrContains: []string{
				"unknown backend",
				"trained-roofline",
				"valid options:",
				"blackbox",
				"roofline",
				"trained-physics",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// GIVEN minimal valid hardware config with removed backend
			hw := latency.ModelHardwareConfig{
				Backend: tt.backend,
				TP:      1,
				ModelConfig: sim.ModelConfig{
					NumLayers:   32,
					NumHeads:    32,
					HiddenDim:   4096,
					IntermediateDim: 11008,
				},
				HWConfig: sim.HardwareCalib{
					TFlopsPeak: 989.0,
					BwPeakTBs:  3.35,
				},
			}
			coeffs := latency.LatencyCoeffs{
				BetaCoeffs:  []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
				AlphaCoeffs: []float64{1.0, 1.0},
			}

			// WHEN attempting to construct the model
			model, err := latency.NewLatencyModel(hw, coeffs)

			// THEN construction must fail
			if err == nil {
				t.Fatalf("NewLatencyModel(%q) succeeded; want error for removed backend", tt.backend)
			}
			if model != nil {
				t.Errorf("NewLatencyModel(%q) returned non-nil model with error; want nil", tt.backend)
			}

			// AND the error message must contain all expected substrings
			errMsg := err.Error()
			for _, substr := range tt.wantErrContains {
				if !strings.Contains(errMsg, substr) {
					t.Errorf("error message missing substring %q\nGot: %s", substr, errMsg)
				}
			}
		})
	}
}

func TestNewLatencyModel_RemainingBackendsWork(t *testing.T) {
	// GIVEN minimal valid hardware config
	hw := latency.ModelHardwareConfig{
		TP: 1,
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			HiddenDim:       4096,
			IntermediateDim: 11008,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.0,
			BwPeakTBs:  3.35,
		},
	}
	coeffs := latency.LatencyCoeffs{
		BetaCoeffs:  []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
		AlphaCoeffs: []float64{1.0, 1.0},
	}

	validBackends := []string{"blackbox", "roofline", "trained-physics"}

	for _, backend := range validBackends {
		t.Run(backend, func(t *testing.T) {
			// WHEN constructing a valid backend
			hw.Backend = backend
			model, err := latency.NewLatencyModel(hw, coeffs)

			// THEN construction must succeed
			if err != nil {
				t.Fatalf("NewLatencyModel(%q) failed: %v", backend, err)
			}
			if model == nil {
				t.Errorf("NewLatencyModel(%q) returned nil model with no error", backend)
			}

			// AND the model must compute non-zero step time
			stepTime := model.ComputeStepTimeUs(100, 10)
			if stepTime <= 0 {
				t.Errorf("ComputeStepTimeUs(100, 10) = %v; want > 0", stepTime)
			}
		})
	}
}
```

**Key Implementation Notes:**
- Table-driven tests for both removed backends
- Regression test for all remaining valid backends
- Verifies error message format (BC-9: lists all valid backends)
- Verifies no silent fallback (BC-8: error returned, not default backend)
- Verifies remaining backends unchanged (BC-7: step time computation works)
- No RNG usage (tests are deterministic)
- No metrics collection
- No event ordering concerns

---

### Files to Delete (no implementation detail needed — `git rm` sufficient):
- `sim/latency/crossmodel.go` (2,898 bytes)
- `sim/latency/crossmodel_test.go` (10,095 bytes)
- `sim/latency/trained_roofline.go` (7,621 bytes)
- `sim/latency/trained_roofline_test.go` (15,201 bytes)

---

### File: `CLAUDE.md:270`

**Purpose:** Update latency backend count from five to three and add migration note.

**Current state (line 270):**
```markdown
Five latency model modes (roofline, blackbox, cross-model, trained-roofline, trained-physics), selected via `--latency-model` flag. **Trained-physics** is the recommended default for new models.
```

**After modification (line 270):**
```markdown
Three latency model modes (roofline, blackbox, trained-physics), selected via `--latency-model` flag. **Trained-physics** is the recommended default for new models.

**Migration note:** The deprecated `crossmodel` and `trained-roofline` backends have been removed. Use `--latency-model trained-physics` for modern physics-informed estimation with MoE-aware overhead modeling.
```

---

### File: `README.md:16,265`

**Purpose:** Update feature list and remove file tree entry for deleted implementation.

**Current state (line 16):**
```markdown
- **Five latency estimation modes**: roofline (default, analytical), trained-physics (physics-informed basis functions with architecture-aware MoE scaling), cross-model (physics-informed, MoE-aware), trained-roofline (roofline × learned corrections), and blackbox (data-driven, per-model coefficients)
```

**After modification (line 16):**
```markdown
- **Three latency estimation modes**: roofline (analytical), trained-physics (physics-informed basis functions with architecture-aware MoE scaling), and blackbox (data-driven, per-model coefficients). The deprecated `crossmodel` and `trained-roofline` backends have been removed; use `trained-physics` for modern physics-informed estimation.
```

**Current state (line 265):**
```markdown
│   ├── crossmodel.go       # CrossModelLatencyModel: physics-informed step time from architecture features (MoE-aware)
```

**After modification (line 265):**
*Line deleted entirely (no replacement)*

---

### File: `docs/guide/latency-models.md`

**Purpose:** Remove all sections describing crossmodel and trained-roofline backends, add migration note, remove example commands using removed backends.

**Changes:**

**Line 3 (overview) — Current:**
```markdown
The `LatencyModel` interface determines how BLIS estimates GPU step time for each batch iteration. BLIS ships five backends -- roofline (default, analytical), blackbox (data-driven), cross-model (physics-informed), trained-roofline (roofline × learned corrections), and trained-physics (roofline × learned corrections) -- and the pluggable architecture supports adding custom backends.
```

**Line 3 (overview) — After:**
```markdown
The `LatencyModel` interface determines how BLIS estimates GPU step time for each batch iteration. BLIS ships three backends -- roofline (analytical), blackbox (data-driven), and trained-physics (physics-informed roofline with MoE-aware corrections) -- and the pluggable architecture supports adding custom backends.

**Migration note:** The deprecated `crossmodel` and `trained-roofline` backends have been removed as of v0.x.x. Existing configurations using these backends should migrate to `--latency-model trained-physics`, which supersedes both with improved accuracy and MoE support.
```

**Lines 17-18 (example command) — Delete:**
```markdown
  --latency-model crossmodel --hardware H100 --tp 1 \
```

**Lines 22-23 (example command) — Delete:**
```markdown
  --latency-model trained-roofline --hardware H100 --tp 1 \
```

**Lines 123-156 (crossmodel section) — Delete entire section:**
```markdown
### Cross-Model Physics-Informed Estimation

[... entire section ...]
```

**Lines 164-167 (trained-roofline section) — Delete entire section:**
```markdown
### Trained Roofline

[... entire section ...]
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/remove-deprecated-backends-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach would you prefer?
