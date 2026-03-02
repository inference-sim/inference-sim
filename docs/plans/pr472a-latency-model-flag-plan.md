# Unified `--latency-model` Flag Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the boolean `--roofline` flag with a string-valued `--latency-model` flag that selects between latency model backends by name.

**The problem today:** The `--roofline` flag is a boolean that selects between exactly two latency backends (blackbox vs roofline). Adding a third backend (cross-model, coming in PR-B) requires either adding another boolean flag or switching to a string enum. The current boolean design also pollutes the `ModelHardwareConfig` struct with a latency-model-selection concern that doesn't belong there.

**What this PR adds:**
1. **Named backend selection** — `--latency-model blackbox|roofline` replaces the boolean `--roofline` flag, following the same string-enum pattern used by `--admission-policy`, `--routing-policy`, and `--scheduler`
2. **Validation infrastructure** — `IsValidLatencyBackend()` accessor and `ValidLatencyBackendNames()` list, consistent with all other policy enums in `sim/bundle.go`
3. **Extensibility** — PR-B can register `"crossmodel"` by adding one line to the validity map and one case to the factory switch

**Why this matters:** This is the prerequisite for the cross-model learned latency model (issue #472). Without a named backend selector, every new latency model would require a new boolean flag — an N-flag problem.

**Architecture:** The `ModelHardwareConfig.Roofline bool` field becomes `ModelHardwareConfig.Backend string`. The `NewModelHardwareConfig` canonical constructor changes its last parameter from `bool` to `string` (compiler catches all 45 call sites). The `NewLatencyModel` factory in `sim/latency/latency.go` switches on `hw.Backend` instead of `hw.Roofline`. The CLI replaces `--roofline` with `--latency-model`.

**Source:** GitHub issue #472 (PR-A section) + comment simplifying to eliminate `--roofline` entirely

**Closes:** N/A — #472 will be closed by PR-B when the cross-model backend is implemented

**Behavioral Contracts:** See Part 1, Section B below

---

## Phase 0: Component Context

**Building block:** Latency Model backend selection mechanism. Currently a boolean on `ModelHardwareConfig`; this PR changes it to a string enum.

**Adjacent blocks:**
- `cmd/root.go` — CLI flag parsing, config resolution (upstream consumer)
- `sim/latency/latency.go` — `NewLatencyModel` factory (downstream dispatcher)
- `sim/bundle.go` — Policy enum registries (houses the new validation map)
- `sim/config.go` — `ModelHardwareConfig` struct (owns the field)

**Invariants touched:** INV-6 (determinism) — preserved by construction (same backends, same dispatch logic). INV-L1 through INV-L4 from problem.md — preserved (backend implementations unchanged).

**Construction site audit for `NewModelHardwareConfig`:**

| File | Count | Current last arg | New last arg |
|------|-------|-----------------|-------------|
| `cmd/root.go:555` | 1 | `rooflineActive` (bool) | `latencyModelBackend` (string) |
| `sim/config.go:74` | 1 | definition | definition |
| `sim/config_test.go:43` | 1 | `true` | `"roofline"` |
| `sim/batch_formation_test.go` | 8 | `false` | `""` |
| `sim/simulator_test.go` | 13 | `false` (12), `false` (1 at :1254) | `""` |
| `sim/simulator_preempt_test.go` | 2 | `false` | `""` |
| `sim/metrics_substrate_test.go` | 1 | `false` | `""` |
| `sim/cluster/cluster_test.go` | 4 | `false` (3), `true` (1 at :61) | `""` (3), `"roofline"` (1) |
| `sim/cluster/instance_test.go` | 3 | `false` | `""` |
| `sim/cluster/metrics_substrate_test.go` | 1 | `false` | `""` |
| `sim/cluster/prefix_routing_test.go` | 1 | `false` | `""` |
| `sim/cluster/snapshot_test.go` | 1 | `false` | `""` |
| `sim/latency/latency_test.go` | 7 | `false` (4), `true` (2), `tc.roofline` (1) | `""` (4), `"roofline"` (2), `tc.backend` (1) |
| `sim/latency/config_test.go` | 2 | see file | update both |
| **Total** | **47** | | |

All sites compiler-caught by bool→string type change.

---

## Part 1: Design Validation

### A) Executive Summary

This PR replaces the boolean latency model selector (`Roofline bool`) with a string-valued enum (`Backend string`) across four core files and ~45 mechanical call site updates. It follows the established pattern for policy enums (admission, routing, scheduling, priority) already in `sim/bundle.go`.

**System context:**
- **Before:** `cmd/root.go` → `--roofline` bool flag → `ModelHardwareConfig.Roofline` → `latency.NewLatencyModel` checks `hw.Roofline`
- **After:** `cmd/root.go` → `--latency-model` string flag → `ModelHardwareConfig.Backend` → `latency.NewLatencyModel` switches on `hw.Backend`
- **Depends on:** Nothing (first PR in #472 series)
- **Enables:** PR-B (CrossModelLatencyModel) adds `"crossmodel"` to the registry

Three deviations from the issue design documented in Section D (all justified simplifications/deferrals).

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Backend Selection — Blackbox Default
- GIVEN no `--latency-model` flag is provided
- WHEN the latency model factory creates a model
- THEN the blackbox regression model is used (same as current default)
- MECHANISM: `Backend` defaults to `""`, factory treats `""` and `"blackbox"` identically

BC-2: Backend Selection — Roofline
- GIVEN `--latency-model roofline` is provided with valid `--hardware` and `--tp`
- WHEN the latency model factory creates a model
- THEN the roofline analytical model is used
- MECHANISM: `Backend = "roofline"` triggers the roofline branch in the factory switch

BC-3: Backend Validation
- GIVEN an unrecognized backend name (e.g., `--latency-model nonexistent`)
- WHEN the CLI processes the flag
- THEN an error is reported listing valid backend names
- MECHANISM: `IsValidLatencyBackend()` check in `cmd/root.go` before config construction

BC-4: Backward Compatibility — Output Identity
- GIVEN any existing simulation configuration (with `--latency-model roofline` replacing the old `--roofline`)
- WHEN the simulation runs
- THEN output is byte-identical to the previous version (INV-6 preserved)
- MECHANISM: Only the selection mechanism changes; both model implementations are untouched

**Negative Contracts:**

BC-5: No Exported Mutable Map
- GIVEN the `validLatencyBackends` map
- WHEN any code outside `sim/bundle.go` attempts to modify it
- THEN compilation fails (the map is unexported)
- MECHANISM: Map is `var validLatencyBackends` (lowercase), accessed via `IsValidLatencyBackend()` (R8)

**Error Handling:**

BC-6: Factory Error — Unknown Backend
- GIVEN `Backend` is set to an unrecognized string
- WHEN `NewLatencyModel` is called
- THEN it returns an error containing the unknown backend name
- MECHANISM: `default:` case in factory switch returns `fmt.Errorf`

### C) Component Interaction

```
cmd/root.go                   sim/config.go                    sim/latency/latency.go
┌──────────────┐              ┌──────────────────┐             ┌──────────────────┐
│ --latency-   │──validates──>│ ModelHardwareConfig│──passed──>│ NewLatencyModel() │
│  model flag  │              │   .Backend string │   to       │  switch hw.Backend│
└──────────────┘              └──────────────────┘             └──────────────────┘
                                                                     │
                              sim/bundle.go                          │
                              ┌──────────────────┐                   ▼
                              │IsValidLatency-   │            BlackboxLatencyModel
                              │   Backend()      │            RooflineLatencyModel
                              └──────────────────┘            (future: CrossModel)
```

**State changes:** None. `Backend` is a config field set at construction, immutable thereafter.

**Extension friction:** Adding one more backend: 1 line in `validLatencyBackends` + 1 case in factory switch = **2 files** (matches design guidelines target).

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue mentions `validNamesList(validLatencyBackends)` | Uses existing `validNamesList` helper | SIMPLIFICATION: helper already exists in bundle.go |
| Issue shows validation in bundle.go Validate() | Validates in cmd/root.go only | SIMPLIFICATION: PolicyBundle is for YAML policy configs; latency backend is a CLI-only concern. No YAML loading path for latency backend exists. If a future PR adds YAML-based latency backend selection, `Validate()` must be updated at that point. |
| Issue mentions updating docs | Defers MkDocs docs updates | DEFERRAL: 26 doc files reference "roofline"; updating all is a separate docs PR to avoid bloating this mechanical refactor. CLAUDE.md IS updated since it's project instructions. |

### E) Review Guide

**The tricky part:** The `cmd/root.go` logic has two activation paths for roofline: explicit (`--roofline` flag) and implicit (all-zeros coefficients + config folder present). Both must be preserved under the new flag while removing the two boolean variables (`rooflineFlag`, `rooflineActive`). Verify the implicit detection path maps correctly to `latencyModelBackend = "roofline"`.

**What to scrutinize:** BC-3 (validation) and BC-6 (factory error) — ensure invalid backends are caught at both CLI and factory levels. Also verify hfconfig.go log messages are updated from `--roofline:` to `--latency-model:`.

**What's safe to skim:** The ~44 test file call site updates are fully mechanical (`false` → `""`, `true` → `"roofline"`). Compiler enforces completeness.

**Known debt:** 26 MkDocs documentation files still reference `--roofline`. These should be updated in a follow-up docs PR.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/config.go` — `Roofline bool` → `Backend string` in struct and constructor
- `sim/config_test.go` — Update field equivalence test
- `sim/bundle.go` — Add `validLatencyBackends` map + accessors
- `sim/latency/latency.go` — Switch on `hw.Backend` instead of `hw.Roofline`
- `sim/latency/latency_test.go` — Update 7 call sites + factory test cases
- `sim/latency/config_test.go` — Update 2 call sites
- `cmd/root.go` — Replace flag, update resolution logic, update log messages
- `cmd/hfconfig.go` — Update log message prefixes
- `sim/simulator_test.go` — Update 13 call sites
- `sim/batch_formation_test.go` — Update 8 call sites
- `sim/cluster/cluster_test.go` — Update 4 call sites
- `sim/cluster/instance_test.go` — Update 3 call sites
- `sim/cluster/metrics_substrate_test.go` — Update 1 call site
- `sim/cluster/prefix_routing_test.go` — Update 1 call site
- `sim/cluster/snapshot_test.go` — Update 1 call site
- `sim/metrics_substrate_test.go` — Update 1 call site
- `sim/simulator_preempt_test.go` — Update 2 call sites
- `CLAUDE.md` — Update `--roofline` references

**Key decisions:**
- `""` and `"blackbox"` are both valid (empty = default = blackbox)
- `--roofline` flag is deleted, not deprecated (per issue comment)
- Validation happens at CLI level; factory also validates as defense-in-depth

### G) Task Breakdown

---

### Task 1: Core Struct + Constructor + Validation Registry

**Contracts Implemented:** BC-1 (default), BC-5 (unexported map)

**Files:**
- Modify: `sim/config.go:62-84`
- Modify: `sim/config_test.go:40-53`
- Modify: `sim/bundle.go` (add after line 64)

**Step 1: Write failing test for Backend field and validator**

Context: Change the struct field from bool to string and add the validation accessor. The field equivalence test will fail first because the struct changes.

In `sim/config_test.go`, replace `TestNewModelHardwareConfig_FieldEquivalence`:

```go
func TestNewModelHardwareConfig_FieldEquivalence(t *testing.T) {
	mc := ModelConfig{NumLayers: 32}
	hw := HardwareCalib{TFlopsPeak: 1000.0, MemoryGiB: 80.0}
	got := NewModelHardwareConfig(mc, hw, "llama", "H100", 2, "roofline")
	want := ModelHardwareConfig{
		ModelConfig: mc,
		HWConfig:    hw,
		Model:       "llama",
		GPU:         "H100",
		TP:          2,
		Backend:     "roofline",
	}
	assert.Equal(t, want, got)
}
```

Add new tests for backend validation in `sim/bundle_test.go` (matching existing `TestIsValidAdmissionPolicy` pattern):

```go
func TestIsValidLatencyBackend(t *testing.T) {
	assert.True(t, IsValidLatencyBackend(""))
	assert.True(t, IsValidLatencyBackend("blackbox"))
	assert.True(t, IsValidLatencyBackend("roofline"))
	assert.False(t, IsValidLatencyBackend("nonexistent"))
}

func TestValidLatencyBackendNames(t *testing.T) {
	names := ValidLatencyBackendNames()
	assert.Contains(t, names, "blackbox")
	assert.Contains(t, names, "roofline")
	assert.NotContains(t, names, "")
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestNewModelHardwareConfig_FieldEquivalence -v`
Expected: Compilation error (type mismatch: bool vs string)

**Step 3: Implement the struct change, constructor, and validation registry**

In `sim/config.go`, replace the `ModelHardwareConfig` struct and constructor:

```go
// ModelHardwareConfig groups model identity and hardware specification.
type ModelHardwareConfig struct {
	ModelConfig ModelConfig   // HuggingFace model parameters (for roofline/crossmodel modes)
	HWConfig    HardwareCalib // GPU specifications (for roofline/crossmodel modes)
	Model       string        // model name (e.g., "meta-llama/llama-3.1-8b-instruct")
	GPU         string        // GPU type (e.g., "H100")
	TP          int           // tensor parallelism degree
	Backend     string        // latency model backend: "" or "blackbox" (default), "roofline"
}

// NewModelHardwareConfig creates a ModelHardwareConfig with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
// Parameter order matches struct field order.
func NewModelHardwareConfig(modelConfig ModelConfig, hwConfig HardwareCalib,
	model, gpu string, tp int, backend string) ModelHardwareConfig {
	return ModelHardwareConfig{
		ModelConfig: modelConfig,
		HWConfig:    hwConfig,
		Model:       model,
		GPU:         gpu,
		TP:          tp,
		Backend:     backend,
	}
}
```

In `sim/bundle.go`, add after the `validSchedulers` declaration (around line 64):

```go
	validLatencyBackends = map[string]bool{"": true, "blackbox": true, "roofline": true}
```

And add accessor functions after `ValidSchedulerNames()`:

```go
// IsValidLatencyBackend returns true if name is a recognized latency model backend.
func IsValidLatencyBackend(name string) bool { return validLatencyBackends[name] }

// ValidLatencyBackendNames returns sorted valid latency backend names (excluding empty).
func ValidLatencyBackendNames() []string { return validNamesList(validLatencyBackends) }
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run "TestNewModelHardwareConfig_FieldEquivalence|TestIsValidLatencyBackend|TestValidLatencyBackendNames" -v`
Expected: PASS (but build will fail due to remaining bool call sites — that's expected, we fix those in Task 4)

Note: At this point `go build ./...` will fail because all other call sites still pass `bool`. This is expected and correct — the compiler is enforcing R4.

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...` (will report compilation errors from call sites — expected)

**Step 6: Do NOT commit yet** — Task 2 must complete first to make the build pass.

---

### Task 2: Factory Dispatch Update

**Contracts Implemented:** BC-2 (roofline selection), BC-6 (unknown backend error)

**Files:**
- Modify: `sim/latency/latency.go:122-158`

**Step 1: Write failing test for string-based backend dispatch**

Context: The factory must switch on `hw.Backend` string instead of `hw.Roofline` bool. Tests in `sim/latency/latency_test.go` that pass `true`/`false` will already fail from Task 1's type change. Add a test for unknown backend.

In `sim/latency/latency_test.go`, add:

```go
func TestNewLatencyModel_UnknownBackend_ReturnsError(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs([]float64{1000, 10, 2}, []float64{500, 1, 100})
	hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "", "", 0, "nonexistent")
	_, err := NewLatencyModel(coeffs, hw)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "nonexistent")
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/latency/... -run TestNewLatencyModel_UnknownBackend -v`
Expected: Compilation error (the factory still expects bool)

**Step 3: Implement factory switch**

In `sim/latency/latency.go`, update the docstring and replace the `NewLatencyModel` function body.

Update docstring from:
```go
// NewLatencyModel creates the appropriate LatencyModel based on config.
// Returns RooflineLatencyModel if hw.Roofline is true, BlackboxLatencyModel otherwise.
```
To:
```go
// NewLatencyModel creates the appropriate LatencyModel based on config.
// Dispatches on hw.Backend: "roofline" → RooflineLatencyModel, "" or "blackbox" → BlackboxLatencyModel.
```

Replace the function body:

```go
func NewLatencyModel(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (sim.LatencyModel, error) {
	// Both implementations index alphaCoeffs[0..2]; validate upfront.
	if len(coeffs.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("latency model: AlphaCoeffs requires at least 3 elements, got %d", len(coeffs.AlphaCoeffs))
	}
	if err := validateCoeffs("AlphaCoeffs", coeffs.AlphaCoeffs); err != nil {
		return nil, err
	}
	switch hw.Backend {
	case "roofline":
		if hw.TP <= 0 {
			return nil, fmt.Errorf("latency model: roofline requires TP > 0, got %d", hw.TP)
		}
		if err := ValidateRooflineConfig(hw.ModelConfig, hw.HWConfig); err != nil {
			return nil, fmt.Errorf("latency model: %w", err)
		}
		return &RooflineLatencyModel{
			modelConfig: hw.ModelConfig,
			hwConfig:    hw.HWConfig,
			tp:          hw.TP,
			alphaCoeffs: coeffs.AlphaCoeffs,
		}, nil
	case "", "blackbox":
		// BlackboxLatencyModel indexes betaCoeffs[0..2]; validate upfront.
		if len(coeffs.BetaCoeffs) < 3 {
			return nil, fmt.Errorf("latency model: BetaCoeffs requires at least 3 elements, got %d", len(coeffs.BetaCoeffs))
		}
		if err := validateCoeffs("BetaCoeffs", coeffs.BetaCoeffs); err != nil {
			return nil, err
		}
		return &BlackboxLatencyModel{
			betaCoeffs:  coeffs.BetaCoeffs,
			alphaCoeffs: coeffs.AlphaCoeffs,
		}, nil
	default:
		return nil, fmt.Errorf("latency model: unknown backend %q; valid options: %s",
			hw.Backend, strings.Join(sim.ValidLatencyBackendNames(), ", "))
	}
}
```

Add `"strings"` to the import block (it is not currently imported).

**Step 4: Run test to verify it passes**

Run: `go test ./sim/latency/... -run TestNewLatencyModel_UnknownBackend -v`
Expected: Still compilation errors from other test call sites — fix those in Task 3.

**Step 5: Do NOT commit yet** — Task 3 fixes remaining test call sites to make build pass.

---

### Task 3: Update All Test Construction Sites

**Contracts Implemented:** BC-4 (output identity — all tests must pass unchanged)

**Files:**
- Modify: `sim/latency/latency_test.go` — 7 call sites
- Modify: `sim/latency/config_test.go` — 2 call sites
- Modify: `sim/simulator_test.go` — 13 call sites
- Modify: `sim/batch_formation_test.go` — 8 call sites
- Modify: `sim/simulator_preempt_test.go` — 2 call sites
- Modify: `sim/metrics_substrate_test.go` — 1 call site
- Modify: `sim/cluster/cluster_test.go` — 4 call sites
- Modify: `sim/cluster/instance_test.go` — 3 call sites
- Modify: `sim/cluster/metrics_substrate_test.go` — 1 call site
- Modify: `sim/cluster/prefix_routing_test.go` — 1 call site
- Modify: `sim/cluster/snapshot_test.go` — 1 call site

**Step 1: No new test needed — existing tests ARE the verification**

Context: This is a purely mechanical transformation. Every `NewModelHardwareConfig(..., false)` becomes `NewModelHardwareConfig(..., "")` and every `NewModelHardwareConfig(..., true)` becomes `NewModelHardwareConfig(..., "roofline")`. The existing tests verify BC-4 (output identity).

**Step 2: Apply mechanical replacements**

The transformation rule is simple:
- Last argument `false` → `""`
- Last argument `true` → `"roofline"`
- Any `tc.roofline` (bool test table field) → `tc.backend` (string test table field)
- Any `Roofline: true/false` in test table structs → `backend: "roofline"/""`

Apply across all files listed above. Use the compiler as verification — `go build ./...` must succeed with zero errors after all replacements.

**Step 3: Run full test suite to verify output identity**

Run: `go test ./... -count=1`
Expected: ALL PASS — identical behavior with string-based backend selection

**Step 4: Run lint check**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 5: Commit Tasks 1-3 together**

```bash
git add sim/config.go sim/config_test.go sim/bundle.go sim/bundle_test.go \
  sim/latency/latency.go sim/latency/latency_test.go sim/latency/config_test.go \
  sim/simulator_test.go sim/batch_formation_test.go sim/simulator_preempt_test.go \
  sim/metrics_substrate_test.go \
  sim/cluster/cluster_test.go sim/cluster/instance_test.go \
  sim/cluster/metrics_substrate_test.go sim/cluster/prefix_routing_test.go \
  sim/cluster/snapshot_test.go
git commit -m "refactor(latency): replace Roofline bool with Backend string in ModelHardwareConfig (BC-1, BC-2, BC-5, BC-6)

- Change ModelHardwareConfig.Roofline bool → .Backend string
- Update NewModelHardwareConfig canonical constructor signature
- Switch NewLatencyModel factory to dispatch on hw.Backend
- Add validLatencyBackends registry with IsValidLatencyBackend/ValidLatencyBackendNames (R8)
- Add factory test for unknown backend error (BC-6)
- Update all 45 construction sites (compiler-enforced completeness)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: CLI Flag Replacement

**Contracts Implemented:** BC-1 (default), BC-2 (roofline), BC-3 (validation)

**Files:**
- Modify: `cmd/root.go`
- Modify: `cmd/hfconfig.go`

**Step 1: Write failing test for new CLI flag**

Context: Replace the boolean `--roofline` flag with a string `--latency-model` flag. The existing cmd tests should continue to pass.

No new test file needed — the existing `cmd/` tests verify CLI behavior. The key behavioral test is that `--latency-model roofline` produces the same behavior as the old `--roofline`.

**Step 2: Apply CLI changes**

In `cmd/root.go`:

1. Replace the two package-level variables with ONE Cobra-bound variable:
   ```go
   // REMOVE:
   rooflineActive bool
   rooflineFlag   bool
   // ADD:
   latencyModelBackend string  // Cobra-bound only — NEVER mutated inside Run
   ```

2. Replace the flag declaration (around line 728):
   ```go
   // REMOVE:
   runCmd.Flags().BoolVar(&rooflineFlag, "roofline", false, "Enable roofline mode...")
   // ADD:
   runCmd.Flags().StringVar(&latencyModelBackend, "latency-model", "", "Latency model backend: blackbox (default), roofline")
   ```

3. Add validation early in the RunE function (after flag parsing, before config resolution):
   ```go
   if !sim.IsValidLatencyBackend(latencyModelBackend) {
       logrus.Fatalf("unknown --latency-model %q; valid options: %s",
           latencyModelBackend, strings.Join(sim.ValidLatencyBackendNames(), ", "))
   }
   ```

4. Replace all `rooflineFlag` references with `backend == "roofline"` checks (using the LOCAL variable from step 6).

5. Replace all `rooflineActive = true` with `backend = "roofline"` (mutating the LOCAL variable, not the package-level one).

6. At the top of the RunE function (replacing `rooflineActive = false` at line 140), create a LOCAL variable that copies the Cobra-bound value:
   ```go
   // REMOVE:
   rooflineActive = false
   // ADD:
   backend := latencyModelBackend  // local copy of CLI value; may be mutated by implicit detection
   ```
   **CRITICAL:** Do NOT reset the package-level `latencyModelBackend` — Cobra sets it before Run, and resetting it would destroy the CLI value. Instead, ALL subsequent code in Run must use the local `backend` variable (not `latencyModelBackend`). The implicit detection path mutates `backend` (local), which dies when Run returns — no process-reuse leak.

7. Replace all `if rooflineActive {` with `if backend == "roofline" {`.

8. Replace the implicit detection guard at line 268:
   ```go
   // BEFORE:
   if !rooflineActive && AllZeros(alphaCoeffs) && AllZeros(betaCoeffs) {
   // AFTER:
   if backend == "" && AllZeros(alphaCoeffs) && AllZeros(betaCoeffs) {
   ```
   This MUST be `== ""` (not `!= "roofline"`), because `--latency-model blackbox` should NOT trigger implicit roofline detection — the user explicitly chose blackbox. Inside the block, replace `rooflineActive = true` with `backend = "roofline"` (local variable mutation).

8b. Add a zero-coefficients safety guard AFTER the defaults.yaml loading block and AFTER the implicit detection block. This prevents `--latency-model blackbox` (or `""`) with an unknown model from silently producing zero-step-time simulations:
   ```go
   // After implicit detection block (line ~277), before config loading:
   if backend != "roofline" && AllZeros(alphaCoeffs) && AllZeros(betaCoeffs) {
       logrus.Fatalf("No trained coefficients found for model=%s, GPU=%s, TP=%d. "+
           "Provide --alpha-coeffs/--beta-coeffs, or use --latency-model roofline with --hardware and --tp",
           model, gpu, tensorParallelism)
   }
   ```
   This preserves the current behavior where zero coefficients without roofline is always an error.

9. Update the SimConfig construction to pass the LOCAL `backend` variable:
   ```go
   ModelHardwareConfig: sim.NewModelHardwareConfig(modelConfig, hwConfig, model, gpu, tensorParallelism, backend),
   ```

10. Update ALL log messages and comments referencing `--roofline` (not just the `--roofline:` colon pattern). This includes `--roofline requires`, `--roofline replaces`, and `--roofline:` prefixes in both root.go (~9 occurrences) and hfconfig.go (~9 occurrences). Replace with `--latency-model roofline requires`, `--latency-model roofline replaces`, `--latency-model:`, respectively.

11. Update the stale design comment at lines 192-198 (currently references `rooflineActive` by name and explains the AllZeros/rooflineActive interaction). Rewrite to reference `latencyModelBackend` and the new `== ""` guard semantics.

In `cmd/hfconfig.go`:
- Replace all `--roofline:` log message prefixes with `--latency-model:` (~9 occurrences).

In `hypotheses/h19-roofline-vs-blackbox/run.sh`:
- Replace `--roofline \` with `--latency-model roofline \` (line ~120). Update any comments referencing `--roofline` flag. This preserves experiment reproducibility per `docs/contributing/standards/experiments.md`.

**Step 3: Run tests**

Run: `go test ./cmd/... -v -count=1`
Expected: ALL PASS

Run: `go test ./... -count=1`
Expected: ALL PASS

**Step 4: Run lint check**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 5: Commit**

```bash
git add cmd/root.go cmd/hfconfig.go hypotheses/h19-roofline-vs-blackbox/run.sh
git commit -m "refactor(cmd): replace --roofline flag with --latency-model string flag (BC-1, BC-2, BC-3)

- Remove --roofline boolean flag and rooflineFlag/rooflineActive variables
- Add --latency-model string flag with validation via IsValidLatencyBackend
- Preserve implicit roofline detection path (all-zeros + config folder)
- Update all log message prefixes from --roofline: to --latency-model:

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Update CLAUDE.md

**Contracts Implemented:** Documentation accuracy

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update all `--roofline` references in CLAUDE.md**

Search and replace:
- `--roofline` → `--latency-model roofline` in CLI examples
- `Roofline    bool` → `Backend     string` in File Organization comments (if present)
- Update the "Latency Estimation" section to describe the new flag
- Update the File Organization tree entry for `cmd/root.go` to list `--latency-model` instead of `--roofline`

Specifically update:
1. File Organization tree: `cmd/root.go` flags list — replace `--roofline` with `--latency-model`
2. File Organization tree: `cmd/hfconfig.go` description — replace `--roofline auto-fetch` with `--latency-model auto-fetch`
3. Latency Estimation section: replace `--roofline` flag description with `--latency-model roofline`
4. The CLI example in Build and Run: no change needed (doesn't use --roofline)

**Step 2: Run build to verify no code changes were accidentally introduced**

Run: `go build ./...`
Expected: SUCCESS

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for --latency-model flag replacement

- Replace --roofline references with --latency-model roofline
- Update File Organization tree and Latency Estimation section

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|------------------------|
| BC-1 | Task 1 | Unit | `TestNewModelHardwareConfig_FieldEquivalence` (empty Backend = default) |
| BC-2 | Task 2 | Unit | Existing roofline factory tests (Backend="roofline") |
| BC-3 | Task 4 | Manual + defense-in-depth | No automated cmd tests exist; verified manually + factory defense-in-depth (BC-6) |
| BC-4 | Task 3 | Regression | Full `go test ./...` — all existing tests pass unchanged |
| BC-5 | Task 1 | Structural | `validLatencyBackends` is unexported (compilation enforces) |
| BC-6 | Task 2 | Unit | `TestNewLatencyModel_UnknownBackend_ReturnsError` |

No golden dataset changes. No new invariant tests needed (this is a refactor with zero behavioral change).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Missed construction site | Low | High (compilation) | Compiler catches all (bool→string type change) | Task 3 |
| Implicit roofline detection broken | Medium | High | Existing cmd tests + manual verification | Task 4 |
| Log message prefix missed | Low | Low (cosmetic) | Grep for remaining `--roofline:` after changes | Task 4 |
| Test data depends on `Roofline` field name | Low | Medium | Grep for `Roofline` in YAML/JSON test data | Task 3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope (cross-model deferred to PR-B)
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without contract updates (--roofline removed, documented)
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used (existing patterns)
- [x] CLAUDE.md updated (Task 5)
- [x] No stale references in CLAUDE.md (Task 5)
- [x] Documentation DRY: MkDocs docs deferred (noted in deviation log)
- [x] Deviation log reviewed — no unresolved deviations
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Tasks 1-3 must be atomic)
- [x] All contracts mapped to tasks
- [x] No golden dataset changes needed
- [x] Construction site audit completed (45 sites listed)
- [x] R1: No silent continue/return ✓
- [x] R3: --latency-model validated at CLI ✓
- [x] R4: All 45 construction sites updated ✓
- [x] R6: No Fatalf in sim/ ✓
- [x] R8: validLatencyBackends unexported ✓

---

## Appendix: File-Level Implementation Details

### File: `sim/config.go`

**Purpose:** Change `ModelHardwareConfig.Roofline bool` to `Backend string`.

The full struct and constructor are shown in Task 1, Step 3.

### File: `sim/bundle.go`

**Purpose:** Add latency backend validation registry following established enum pattern.

Add to the `var` block:
```go
validLatencyBackends = map[string]bool{"": true, "blackbox": true, "roofline": true}
```

Add two accessor functions (matching pattern of all other policy accessors).

### File: `sim/latency/latency.go`

**Purpose:** Switch factory dispatch from `hw.Roofline` bool to `hw.Backend` string.

Full implementation shown in Task 2, Step 3. Import `"strings"` added.

### File: `cmd/root.go`

**Purpose:** Replace `--roofline` boolean flag with `--latency-model` string flag.

Key changes:
- Remove `rooflineActive bool` and `rooflineFlag bool` package vars
- Add `latencyModelBackend string` package var
- Replace flag declaration
- Add validation call
- Replace all `rooflineFlag`/`rooflineActive` references
- Update log message prefixes

### File: `cmd/hfconfig.go`

**Purpose:** Update log message prefixes from `--roofline:` to `--latency-model:`.

~12 log message string replacements. No logic changes.

### File: `CLAUDE.md`

**Purpose:** Update project documentation to reflect new flag.

Replace `--roofline` with `--latency-model roofline` in relevant sections.
