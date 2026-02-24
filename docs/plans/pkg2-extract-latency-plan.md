# PKG-2: Extract sim/latency/ Package — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the latency model implementations out of the flat `sim/` package into a dedicated `sim/latency/` sub-package, reducing `sim/`'s surface area and following the same extraction pattern established by PKG-1 (sim/kv/).

**The problem today:** The `sim/` package contains ~700 lines of latency model code across three files (`latency_model.go`, `roofline_step.go`, `model_hardware_config.go`). This mixes domain-specific latency estimation logic with the simulator kernel. Adding a new latency backend (SGLang, TensorRT-LLM) requires modifying the `sim/` package directly. PKG-1 showed that extracting implementations to sub-packages makes the codebase more modular while keeping interfaces in `sim/` for consumers.

**What this PR adds:**
1. **`sim/latency/` package** — all latency model implementations (BlackboxLatencyModel, RooflineLatencyModel) and supporting types/functions (roofline computation, hardware config parsing, validation) live here
2. **Init-based registration** — same pattern as `sim/kv/`: `sim/latency/register.go` wires the factory into `sim/`'s registration variable so test code in `package sim` can create latency models without importing `sim/latency/` directly
3. **Clean dependency graph** — `sim/` never imports `sim/latency/`; `sim/cluster/` imports `sim/latency/` for production use; `sim/` test files use registration variables

**Why this matters:** This is PKG-2 in the package extraction series (#404). Together with PKG-1 (kv), it establishes the pattern for extracting all domain modules from the monolithic `sim/` package. A smaller `sim/` package is easier to understand, test, and extend.

**Architecture:** The `LatencyModel` interface (5 methods) stays in `sim/latency_model.go`. Implementations (`BlackboxLatencyModel`, `RooflineLatencyModel`), the `NewLatencyModel` factory, roofline computation functions, config parsers (`HFConfig`, `GetHWConfig`, `GetModelConfig`), and validation (`ValidateRooflineConfig`) move to `sim/latency/`. `StepConfig`, `PrefillRequestConfig`, `DecodeRequestConfig` move to `sim/latency/` (only used by roofline implementation). `ModelConfig` and `HardwareCalib` structs stay in `sim/` (they're embedded in `ModelHardwareConfig` in `sim/config.go` — moving them would create an import cycle). `cmd/root.go`, `sim/cluster/instance.go`, and `sim/cluster/cluster_test.go` update imports from `sim` to `latency`.

**Source:** GitHub issue #406 (PKG-2), parent issue #404 (package extraction tracking)

**Closes:** Fixes #406

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR extracts latency model implementations from `sim/` into `sim/latency/`, following the exact pattern PKG-1 established for `sim/kv/`:

- **Interfaces stay in `sim/`** — `LatencyModel` interface remains in `sim/latency_model.go` (consumers like `Simulator` and `BatchFormation` use this)
- **Implementations move to `sim/latency/`** — `BlackboxLatencyModel`, `RooflineLatencyModel`, `NewLatencyModel` factory, roofline computation, hardware/model config types and parsers
- **Registration pattern** for test files in `package sim` that can't import sub-packages
- **Adjacent blocks**: `sim/simulator.go` (consumes `LatencyModel` interface), `sim/batch_formation.go` (consumes `LatencyModel` interface), `sim/cluster/instance.go` (calls factory), `cmd/root.go` (calls `GetHWConfig`/`GetModelConfig`)

No deviations from PKG-1 pattern. No behavioral changes — all output is byte-identical.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Interface stays in sim/
- GIVEN the `LatencyModel` interface is defined in `sim/latency_model.go`
- WHEN a consumer in `sim/` (Simulator, BatchFormation) uses the interface
- THEN it MUST NOT need to import `sim/latency/`
- MECHANISM: Interface definition stays in `sim/`; only implementations move

BC-2: Factory in sub-package
- GIVEN `NewLatencyModel(LatencyCoeffs, ModelHardwareConfig) (LatencyModel, error)` is the factory
- WHEN it is called with valid blackbox config (Roofline=false)
- THEN it MUST return a model whose `StepTime` produces byte-identical results to the pre-extraction code
- MECHANISM: Implementation code is moved verbatim; only package declaration changes

BC-3: Roofline path preserved
- GIVEN `NewLatencyModel` is called with valid roofline config (Roofline=true, valid ModelConfig/HardwareCalib)
- WHEN `StepTime` is called on the result
- THEN it MUST produce byte-identical results to the pre-extraction code
- MECHANISM: `rooflineStepTime()` and all helper functions move verbatim

BC-4: Registration enables sim/ tests
- GIVEN `sim/latency/register.go` has an `init()` that sets `sim.NewLatencyModelFunc`
- WHEN test code in `package sim` calls `sim.MustNewLatencyModel(coeffs, hw)`
- THEN it MUST receive a valid `LatencyModel` without importing `sim/latency/`
- MECHANISM: Same init-based registration as `sim/kv/register.go`; `sim/latency_import_test.go` blank-imports `sim/latency`

BC-5: Cluster uses sub-package directly
- GIVEN `sim/cluster/instance.go` currently calls `sim.NewLatencyModel()`
- WHEN updated to call `latency.NewLatencyModel()`
- THEN cluster behavior MUST be byte-identical
- MECHANISM: Direct import of `sim/latency` in cluster package

BC-6: Config parsing functions accessible from sub-package
- GIVEN `HFConfig`, `ValidateRooflineConfig`, `GetHWConfig`, `GetModelConfig` move to `sim/latency/` (note: `ModelConfig` and `HardwareCalib` structs stay in `sim/` — they're embedded in `ModelHardwareConfig` in `sim/config.go`)
- WHEN `cmd/root.go` calls config parsing functions
- THEN it MUST use `latency.GetHWConfig`, `latency.GetModelConfig` (struct types remain as `sim.ModelConfig`, `sim.HardwareCalib`)
- MECHANISM: Update import paths in cmd/root.go; struct literal types unchanged

BC-7: StepConfig types move to latency package
- GIVEN `StepConfig`, `PrefillRequestConfig`, `DecodeRequestConfig` are only used by `RooflineLatencyModel.StepTime()` and `rooflineStepTime()`
- WHEN they move to `sim/latency/`
- THEN `sim/simulator.go` MUST NOT reference them (they are internal to the latency package)
- MECHANISM: These types have no consumers outside the latency model

**Negative Contracts:**

BC-8: No import cycle
- GIVEN `sim/` defines the `LatencyModel` interface
- WHEN `sim/latency/` implements it
- THEN `sim/` MUST NOT import `sim/latency/` (acyclic constraint)
- MECHANISM: Registration pattern breaks the cycle; `go build ./...` validates

BC-9: No behavioral change
- GIVEN the golden dataset test exercises the full simulation pipeline
- WHEN all tests run after extraction
- THEN the golden dataset test MUST pass without regeneration (byte-identical output)
- MECHANISM: Pure refactoring — no logic changes

**Error Handling Contracts:**

BC-10: Factory validation preserved
- GIVEN `NewLatencyModel` validates alpha/beta coefficient lengths, NaN/Inf, and roofline config
- WHEN invalid inputs are provided
- THEN the same errors MUST be returned (error messages unchanged)
- MECHANISM: `validateCoeffs()` and `ValidateRooflineConfig()` move verbatim

BC-11: Registration nil guard
- GIVEN `MustNewLatencyModel` is the nil-guarded wrapper in `sim/`
- WHEN called without `sim/latency` being imported (registration variable is nil)
- THEN it MUST panic with an actionable message explaining the missing import
- MECHANISM: Same pattern as `MustNewKVCacheState`. R6 note: this panic is a programming-error guard (missing import), NOT reachable from user input. It follows the established `MustNew*` precedent — R6 prohibits `Fatalf` on user-facing failures in library code, not programming-error panics.

### C) Component Interaction

```
cmd/root.go ──────────┐
                       │ latency.GetHWConfig(), latency.GetModelConfig()
                       ▼
              ┌─────────────────┐
              │  sim/latency/   │ ── BlackboxLatencyModel, RooflineLatencyModel
              │                 │    NewLatencyModel(LatencyCoeffs, ModelHardwareConfig)
              │                 │    HFConfig, ValidateRooflineConfig()
              │                 │    rooflineStepTime(), GetHWConfig(), GetModelConfig()
              └────────┬────────┘
                       │ init() registers sim.NewLatencyModelFunc
                       ▼
              ┌─────────────────┐
              │     sim/        │ ── LatencyModel interface (5 methods)
              │                 │    NewLatencyModelFunc (registration var)
              │                 │    MustNewLatencyModel() (nil-guarded wrapper)
              └────────▲────────┘
                       │ uses LatencyModel interface
              ┌────────┴────────┐
              │  sim/cluster/   │ ── imports sim/latency directly
              │  instance.go    │    latency.NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
              └─────────────────┘
```

**API contracts:**
- `latency.NewLatencyModel(sim.LatencyCoeffs, sim.ModelHardwareConfig) (sim.LatencyModel, error)` — factory in sub-package, returns interface from parent
- `sim.NewLatencyModelFunc` — registration variable, type `func(LatencyCoeffs, ModelHardwareConfig) (LatencyModel, error)`
- `sim.MustNewLatencyModel(LatencyCoeffs, ModelHardwareConfig) LatencyModel` — nil-guarded wrapper (panics if not registered)

**State changes:** None. Pure refactoring — no new mutable state.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Design doc (#241) shows `NewLatencyModel(SimConfig)` signature | Factory uses `NewLatencyModel(LatencyCoeffs, ModelHardwareConfig)` | CORRECTION: Signature was narrowed in #381 (SimConfig decomposition). Design doc predates that change. |
| Issue #406 doesn't mention `StepConfig` types moving | Plan moves `StepConfig`, `PrefillRequestConfig`, `DecodeRequestConfig` to `sim/latency/` | ADDITION: These types are only used within the latency model (roofline implementation). Moving them eliminates unnecessary exports from `sim/`. |
| Issue #406 doesn't mention config parsing functions moving | Plan moves `HFConfig`, parsing functions (`GetHWConfig`, `GetModelConfig`), and validators (`ValidateRooflineConfig`) to `sim/latency/`. `ModelConfig` and `HardwareCalib` structs stay in `sim/` (embedded in `ModelHardwareConfig`). | ADDITION: Parsing/validation functions are only consumed by the latency factory and `cmd/root.go`. Config types stay in `sim/` to avoid import cycle. |

### E) Review Guide

**The tricky part:** The `ModelHardwareConfig` sub-config type in `sim/config.go` embeds `ModelConfig` and `HardwareCalib`. When those types move to `sim/latency/`, we need `sim/config.go` to import `sim/latency/` — BUT that creates an import cycle. The solution: `ModelConfig` and `HardwareCalib` must stay in `sim/` (as part of the config layer), while only the implementation code (parsers, validators, the `HFConfig` type) moves. This is the key architectural decision.

**What to scrutinize:** The registration variable pattern — ensure `NewLatencyModelFunc` matches the factory signature exactly, and `MustNewLatencyModel` has the nil guard.

**What's safe to skim:** File moves that are verbatim (roofline computation, validation). These are pure code motion.

**Known debt:** `ModelConfig` and `HardwareCalib` types remain in `sim/config.go` (or `sim/model_hardware_config.go`) because they're embedded in `ModelHardwareConfig` which is a sub-config of `SimConfig`. This is the correct layering — config types stay in `sim/`, implementations move to sub-packages.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/latency/latency.go` — `NewLatencyModel` factory, `BlackboxLatencyModel`, `RooflineLatencyModel`, `validateCoeffs`
- `sim/latency/roofline.go` — `rooflineStepTime`, `calculateTransformerFlops`, `calculateMemoryAccessBytes`, `StepConfig`, `PrefillRequestConfig`, `DecodeRequestConfig`
- `sim/latency/config.go` — `HFConfig`, `GetHWConfig`, `GetModelConfig`, `parseHWConfig`, `parseHFConfig`, `ValidateRooflineConfig`, `invalidPositiveFloat`
- `sim/latency/register.go` — init-based registration
- `sim/latency/latency_test.go` — tests for factory and implementations
- `sim/latency/roofline_test.go` — tests for roofline computation
- `sim/latency/config_test.go` — tests for config parsing and validation
- `sim/latency_import_test.go` — blank import for `package sim_test`

**Files to modify:**
- `sim/latency_model.go` — keep interface only, add registration variable + `MustNewLatencyModel`
- `sim/simulator.go` — remove `StepConfig`, `PrefillRequestConfig`, `DecodeRequestConfig`
- `sim/cluster/instance.go` — import `sim/latency`, call `latency.NewLatencyModel`
- `cmd/root.go` — import `sim/latency`, call `latency.GetHWConfig`, `latency.GetModelConfig`
- `CLAUDE.md` — update File Organization tree

**Files to delete:**
- `sim/roofline_step.go` — moved to `sim/latency/roofline.go`
- `sim/roofline_step_test.go` — moved to `sim/latency/roofline_test.go`
- `sim/model_hardware_config_test.go` — moved to `sim/latency/config_test.go`
- `sim/latency_model_test.go` — moved to `sim/latency/latency_test.go`

**Files to trim (not delete — struct definitions stay):**
- `sim/model_hardware_config.go` — `HFConfig`, parsers, validators move to `sim/latency/config.go`; `ModelConfig` and `HardwareCalib` structs remain

**Key decisions:**
1. `ModelConfig` and `HardwareCalib` structs stay in `sim/` (in `sim/model_hardware_config.go`) because they're fields of `ModelHardwareConfig` in `sim/config.go`. Moving them would create an import cycle.
2. `HFConfig` type and all parsing functions (`GetHWConfig`, `GetModelConfig`, `parseHWConfig`, `parseHFConfig`) move to `sim/latency/` — these are only consumed by latency model creation and `cmd/root.go`.
3. `ValidateRooflineConfig` and `invalidPositiveFloat` move to `sim/latency/` — only consumed by `NewLatencyModel` factory.
4. `StepConfig`, `PrefillRequestConfig`, `DecodeRequestConfig` move to `sim/latency/` — only consumed by `RooflineLatencyModel.StepTime()`.

### G) Task Breakdown

---

#### Task 1: Extract sim/latency/ package (atomic: create + wire + move tests)

> **IMPORTANT:** Tasks 1-3 from an earlier plan revision were combined into this single atomic task because package extraction is all-or-nothing for build correctness — moving implementations without updating call sites and test files would break the build at intermediate steps.

**Contracts Implemented:** BC-1, BC-2, BC-3, BC-4, BC-5, BC-6, BC-7, BC-8, BC-9, BC-10, BC-11

**Files:**
- Create: `sim/latency/latency.go`
- Create: `sim/latency/roofline.go`
- Create: `sim/latency/config.go`
- Create: `sim/latency/register.go`
- Create: `sim/latency/latency_test.go`
- Create: `sim/latency/roofline_test.go`
- Create: `sim/latency/config_test.go`
- Create: `sim/latency_import_test.go`
- Modify: `sim/latency_model.go` (remove implementations, keep interface + registration)
- Modify: `sim/model_hardware_config.go` (trim: keep ModelConfig + HardwareCalib structs only)
- Modify: `sim/simulator.go` (remove StepConfig types)
- Modify: `sim/cluster/instance.go` (sim.NewLatencyModel → latency.NewLatencyModel)
- Modify: `sim/cluster/cluster_test.go` (sim.NewLatencyModel → latency.NewLatencyModel)
- Modify: `cmd/root.go` (sim.GetHWConfig/GetModelConfig → latency.*)
- Modify: `sim/batch_formation_test.go` (NewLatencyModel → MustNewLatencyModel at 7 call sites + direct struct at line 440)
- Modify: `sim/simulator_test.go` (NewLatencyModel → MustNewLatencyModel at 6 call sites)
- Modify: `sim/simulator_preempt_test.go` (NewLatencyModel → MustNewLatencyModel at 2 call sites)
- Delete: `sim/roofline_step.go`
- Delete: `sim/latency_model_test.go`
- Delete: `sim/roofline_step_test.go`
- Delete: `sim/model_hardware_config_test.go` (tests that exercise NewSimulator stay in sim/ — see Step 12)

**Step 1: Create sim/latency/latency.go with implementations**

Context: Move `BlackboxLatencyModel`, `RooflineLatencyModel`, `NewLatencyModel`, and `validateCoeffs` from `sim/latency_model.go` to `sim/latency/latency.go`. The structs need exported fields now since they're in a separate package.

In `sim/latency/latency.go`:
```go
// Package latency provides latency model implementations for the BLIS simulator.
// The LatencyModel interface is defined in sim/ (parent package).
// This package provides BlackboxLatencyModel (alpha/beta regression) and
// RooflineLatencyModel (analytical FLOPs/bandwidth).
package latency

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// BlackboxLatencyModel estimates latency using trained alpha/beta regression coefficients.
// Beta coefficients estimate step time: beta0 + beta1*cacheMissTokens + beta2*decodeTokens.
// Alpha coefficients estimate overheads: alpha0 + alpha1*inputLen (queueing), alpha2 (output processing).
type BlackboxLatencyModel struct {
	betaCoeffs  []float64
	alphaCoeffs []float64
}

func (m *BlackboxLatencyModel) StepTime(batch []*sim.Request) int64 {
	var totalCacheMissTokens, totalDecodeTokens int64
	for _, req := range batch {
		if req.ProgressIndex < util.Len64(req.InputTokens) {
			totalCacheMissTokens += int64(req.NumNewTokens)
		} else if len(req.OutputTokens) > 0 {
			totalDecodeTokens += int64(req.NumNewTokens)
		}
	}
	var totalStepTime float64
	totalStepTime += m.betaCoeffs[0]
	totalStepTime += m.betaCoeffs[1] * float64(totalCacheMissTokens)
	totalStepTime += m.betaCoeffs[2] * float64(totalDecodeTokens)
	return int64(totalStepTime)
}

func (m *BlackboxLatencyModel) QueueingTime(req *sim.Request) int64 {
	var totalProcessingTime float64
	totalProcessingTime += m.alphaCoeffs[0]
	totalProcessingTime += m.alphaCoeffs[1] * float64(len(req.InputTokens))
	return int64(totalProcessingTime)
}

func (m *BlackboxLatencyModel) OutputTokenProcessingTime() int64 {
	return int64(m.alphaCoeffs[2])
}

func (m *BlackboxLatencyModel) SchedulingProcessingTime() int64 { return 0 }

func (m *BlackboxLatencyModel) PreemptionProcessingTime() int64 { return 0 }

// RooflineLatencyModel estimates latency using analytical FLOPs/bandwidth roofline model.
// Step time is computed via rooflineStepTime(); overhead estimates use alpha coefficients.
type RooflineLatencyModel struct {
	modelConfig sim.ModelConfig
	hwConfig    sim.HardwareCalib
	tp          int
	alphaCoeffs []float64
}

func (m *RooflineLatencyModel) StepTime(batch []*sim.Request) int64 {
	stepConfig := StepConfig{
		PrefillRequests: make([]PrefillRequestConfig, 0, len(batch)),
		DecodeRequests:  make([]DecodeRequestConfig, 0, len(batch)),
	}
	for _, req := range batch {
		if req.ProgressIndex < util.Len64(req.InputTokens) {
			stepConfig.PrefillRequests = append(stepConfig.PrefillRequests, PrefillRequestConfig{
				ProgressIndex:       req.ProgressIndex,
				NumNewPrefillTokens: req.NumNewTokens,
			})
		} else if len(req.OutputTokens) > 0 {
			stepConfig.DecodeRequests = append(stepConfig.DecodeRequests, DecodeRequestConfig{
				ProgressIndex:      req.ProgressIndex,
				NumNewDecodeTokens: req.NumNewTokens,
			})
		}
	}
	return rooflineStepTime(m.modelConfig, m.hwConfig, stepConfig, m.tp)
}

func (m *RooflineLatencyModel) QueueingTime(req *sim.Request) int64 {
	var totalProcessingTime float64
	totalProcessingTime += m.alphaCoeffs[0]
	totalProcessingTime += m.alphaCoeffs[1] * float64(len(req.InputTokens))
	return int64(totalProcessingTime)
}

func (m *RooflineLatencyModel) OutputTokenProcessingTime() int64 {
	return int64(m.alphaCoeffs[2])
}

func (m *RooflineLatencyModel) SchedulingProcessingTime() int64 { return 0 }

func (m *RooflineLatencyModel) PreemptionProcessingTime() int64 { return 0 }

// validateCoeffs checks for NaN or Inf in a coefficient slice.
func validateCoeffs(name string, coeffs []float64) error {
	for i, c := range coeffs {
		if math.IsNaN(c) {
			return fmt.Errorf("latency model: %s[%d] is NaN", name, i)
		}
		if math.IsInf(c, 0) {
			return fmt.Errorf("latency model: %s[%d] is Inf", name, i)
		}
	}
	return nil
}

// NewLatencyModel creates the appropriate LatencyModel based on config.
// Returns RooflineLatencyModel if hw.Roofline is true, BlackboxLatencyModel otherwise.
// Returns error if coefficient slices are too short, contain NaN/Inf, or roofline config validation fails.
func NewLatencyModel(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (sim.LatencyModel, error) {
	if len(coeffs.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("latency model: AlphaCoeffs requires at least 3 elements, got %d", len(coeffs.AlphaCoeffs))
	}
	if err := validateCoeffs("AlphaCoeffs", coeffs.AlphaCoeffs); err != nil {
		return nil, err
	}
	if hw.Roofline {
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
	}
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
}
```

**Step 2: Create sim/latency/roofline.go**

Move `rooflineStepTime`, `calculateTransformerFlops`, `calculateMemoryAccessBytes` verbatim from `sim/roofline_step.go`, plus `StepConfig`, `PrefillRequestConfig`, `DecodeRequestConfig` from `sim/simulator.go`.

In `sim/latency/roofline.go`:
```go
package latency

import (
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// PrefillRequestConfig describes a single prefill request in a batch step.
type PrefillRequestConfig struct {
	ProgressIndex       int64 `json:"progress_index"`
	NumNewPrefillTokens int   `json:"num_new_prefill_tokens"`
}

// DecodeRequestConfig describes a single decode request in a batch step.
type DecodeRequestConfig struct {
	ProgressIndex      int64 `json:"progress_index"`
	NumNewDecodeTokens int   `json:"num_new_decode_tokens"`
}

// StepConfig describes the requests in a single batch step for roofline estimation.
type StepConfig struct {
	PrefillRequests []PrefillRequestConfig `json:"prefill_requests"`
	DecodeRequests  []DecodeRequestConfig  `json:"decode_requests"`
}

// calculateTransformerFlops computes FLOPs for a transformer forward pass.
func calculateTransformerFlops(config sim.ModelConfig, sequenceLength int64, newTokens int64, includeAttention, includeMLP bool) map[string]float64 {
	// ... (verbatim from sim/roofline_step.go)
}

// calculateMemoryAccessBytes computes memory access bytes for a transformer forward pass.
func calculateMemoryAccessBytes(config sim.ModelConfig, sequenceLength int64, newTokens int64, includeKVCache bool) map[string]float64 {
	// ... (verbatim from sim/roofline_step.go)
}

// rooflineStepTime computes step latency using the roofline model.
func rooflineStepTime(modelConfig sim.ModelConfig, hwConfig sim.HardwareCalib, stepConfig StepConfig, tp int) int64 {
	// ... (verbatim from sim/roofline_step.go, with sim.ModelConfig/sim.HardwareCalib type prefixes)
}
```

**Step 3: Create sim/latency/config.go**

Move `HFConfig`, `parseHWConfig`, `GetHWConfig`, `parseHFConfig`, `GetModelConfig`, `ValidateRooflineConfig`, `invalidPositiveFloat` from `sim/model_hardware_config.go`. Keep `ModelConfig`, `HardwareCalib` structs in `sim/` (they're embedded in `ModelHardwareConfig` in `sim/config.go`).

In `sim/latency/config.go`:
```go
package latency

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
)

// HFConfig represents a flexible JSON object with dynamic fields.
type HFConfig struct {
	Raw map[string]any
}

// ... (all HFConfig methods verbatim)
// ... parseHWConfig, GetHWConfig, parseHFConfig, GetModelConfig verbatim
// ... (return sim.HardwareCalib, *sim.ModelConfig)
// ... ValidateRooflineConfig, invalidPositiveFloat verbatim (accept sim.ModelConfig, sim.HardwareCalib)
```

**Step 4: Trim sim/latency_model.go to interface + registration**

Remove all implementations, factory, and `validateCoeffs`. Add registration variable and `MustNewLatencyModel`.

In `sim/latency_model.go` (after trimming):
```go
package sim

// LatencyModel estimates execution times for the DES step loop.
// Two implementations exist in sim/latency/: BlackboxLatencyModel (alpha/beta regression)
// and RooflineLatencyModel (analytical FLOPs/bandwidth).
// All time estimates are in microseconds (ticks).
type LatencyModel interface {
	StepTime(batch []*Request) int64
	QueueingTime(req *Request) int64
	OutputTokenProcessingTime() int64
	SchedulingProcessingTime() int64
	PreemptionProcessingTime() int64
}

// NewLatencyModelFunc is a factory function for creating LatencyModel implementations.
// Set by sim/latency package's init() via registration. This breaks the import cycle
// between sim/ (which defines LatencyModel) and sim/latency/ (which implements it).
//
// Production callers should import sim/latency and use latency.NewLatencyModel() directly.
// Test code in package sim uses MustNewLatencyModel to avoid importing sim/latency.
var NewLatencyModelFunc func(coeffs LatencyCoeffs, hw ModelHardwareConfig) (LatencyModel, error)

// MustNewLatencyModel calls NewLatencyModelFunc with a nil guard. Panics with an
// actionable message if the factory has not been registered (missing sim/latency import).
func MustNewLatencyModel(coeffs LatencyCoeffs, hw ModelHardwareConfig) (LatencyModel, error) {
	if NewLatencyModelFunc == nil {
		panic("NewLatencyModelFunc not registered: import sim/latency to register it " +
			"(add: import _ \"github.com/inference-sim/inference-sim/sim/latency\")")
	}
	return NewLatencyModelFunc(coeffs, hw)
}
```

**Step 5: Remove StepConfig types from sim/simulator.go**

Delete `PrefillRequestConfig`, `DecodeRequestConfig`, `StepConfig` type definitions from `sim/simulator.go` (they've moved to `sim/latency/roofline.go`).

**Step 6: Trim sim/model_hardware_config.go**

Keep `ModelConfig` struct, `HardwareCalib` struct (they're referenced by `sim/config.go`'s `ModelHardwareConfig`). Remove everything else (`HFConfig`, all parse/get functions, `ValidateRooflineConfig`, `invalidPositiveFloat`).

**Step 7: Delete sim/roofline_step.go**

The entire file has moved to `sim/latency/roofline.go`.

**Step 8: Create sim/latency/register.go**

```go
// register.go wires sim/latency constructors into the sim package's registration
// variable (NewLatencyModelFunc). This init() runs when any package imports
// sim/latency, breaking the import cycle between sim/ (interface owner) and
// sim/latency/ (implementation). Production code imports sim/latency directly;
// test code in package sim uses latency_import_test.go for the blank import.
package latency

import "github.com/inference-sim/inference-sim/sim"

func init() {
	sim.NewLatencyModelFunc = NewLatencyModel
}
```

**Step 9: Create sim/latency_import_test.go**

```go
package sim_test

// Blank import triggers sim/latency's init(), which registers NewLatencyModelFunc.
// This allows package sim's internal test files to create latency models
// without directly importing sim/latency (which would create an import cycle).
import _ "github.com/inference-sim/inference-sim/sim/latency"
```

**Step 10: Update sim/cluster/instance.go**

Change `sim.NewLatencyModel` to `latency.NewLatencyModel`. Add import `"github.com/inference-sim/inference-sim/sim/latency"`.

**Step 11: Update sim/cluster/cluster_test.go**

Line 875: `sim.NewLatencyModel(...)` → `latency.NewLatencyModel(...)`. Add import `"github.com/inference-sim/inference-sim/sim/latency"`.

**Step 12: Update cmd/root.go**

Lines 182, 187: `sim.GetModelConfig(...)` → `latency.GetModelConfig(...)`, `sim.GetHWConfig(...)` → `latency.GetHWConfig(...)`. Add import `"github.com/inference-sim/inference-sim/sim/latency"`.

**Step 13: Move test files to sim/latency/**

Create `sim/latency/latency_test.go` (from `sim/latency_model_test.go`):
- Package: `package latency` (internal — can access unexported fields)
- `&BlackboxLatencyModel{...}` stays (same package), `NewLatencyModel(...)` stays
- `SimConfig{...}` → `sim.SimConfig{...}`, `NewLatencyCoeffs(...)` → `sim.NewLatencyCoeffs(...)`, etc.
- `testModelConfig()`/`testHardwareCalib()` from `roofline_test.go` accessible (same package)

Create `sim/latency/roofline_test.go` (from `sim/roofline_step_test.go`):
- Package: `package latency`, `ModelConfig` → `sim.ModelConfig`, etc.
- `testModelConfig()` and `testHardwareCalib()` stay (defined in this file)

Create `sim/latency/config_test.go` (from `sim/model_hardware_config_test.go`):
- Package: `package latency_test` (external — imports both `sim` and `sim/latency`)
- Imports: `sim`, `sim/latency` (no `_ sim/kv` needed — `MustNewKVCacheState` user stays in `sim/`)
- `GetHWConfig(...)` → `latency.GetHWConfig(...)`, `ValidateRooflineConfig(...)` → `latency.ValidateRooflineConfig(...)`
- `ModelConfig{...}` → `sim.ModelConfig{...}`, `HardwareCalib{...}` → `sim.HardwareCalib{...}`
- Rename: `TestNewSimulator_RooflineZeroNumHeads_ReturnsError` → `TestNewLatencyModel_RooflineZeroNumHeads_ReturnsError`
- Rename: `TestNewSimulator_RooflineZeroTP_ReturnsError` → `TestNewLatencyModel_RooflineZeroTP_ReturnsError`
- **KEEP `TestNewSimulator_NonRooflineZeroNumHeads_Succeeds` IN `sim/simulator_test.go`** (not moved — it tests `NewSimulator`, not latency config). Convert its `NewLatencyModel` → `MustNewLatencyModel`.

**Step 14: Update sim/ test files (all `NewLatencyModel` → `MustNewLatencyModel`)**

Files and call counts:
- `sim/batch_formation_test.go`: 7 calls (lines 19, 58, 115, 174, 244, 303, 352) + 1 direct struct (line 440)
- `sim/simulator_test.go`: 6 calls (lines 20, 236, 919, 972, 1019, 1085) + move `TestNewSimulator_NonRooflineZeroNumHeads_Succeeds` here
- `sim/simulator_preempt_test.go`: 2 calls (lines 18, 76)

For line 440 of `batch_formation_test.go`:
```go
lm, err := MustNewLatencyModel(
    NewLatencyCoeffs([]float64{100, 1, 1}, []float64{100, 1, 100}),
    NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, false),
)
if err != nil {
    t.Fatalf("MustNewLatencyModel: %v", err)
}
bf := &VLLMBatchFormation{latencyModel: lm}
```

**Step 15: Delete old test files**

```bash
git rm sim/latency_model_test.go sim/roofline_step_test.go sim/model_hardware_config_test.go
```

**Step 16: Run build + tests + lint**

Run: `go build ./...`
Expected: PASS

Run: `go test ./... -count=1`
Expected: ALL PASS

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 17: Verify no stale references**

Run: `grep -r "sim\.NewLatencyModel\|sim\.GetHWConfig\|sim\.GetModelConfig\|sim\.ValidateRooflineConfig\|sim\.BlackboxLatencyModel\|sim\.RooflineLatencyModel" --include="*.go" | grep -v "sim/latency/" | grep -v docs/ | grep -v archive/`
Expected: NO matches (all references updated)

**Step 18: Commit**

```bash
git add -A
git commit -m "refactor(sim): extract sim/latency/ package (PKG-2)

Move latency model implementations from sim/ to sim/latency/ following
the same pattern as PKG-1 (sim/kv/):
- LatencyModel interface stays in sim/ with registration variable
- BlackboxLatencyModel, RooflineLatencyModel → sim/latency/
- Roofline computation, config parsers → sim/latency/
- StepConfig types → sim/latency/ (internal to roofline)
- ModelConfig, HardwareCalib stay in sim/ (avoid import cycle)
- Init-based registration for sim/ test files
- All call sites updated: cluster, cmd, tests

Fixes #406

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 2: Registration nil-guard test + MustNewLatencyModel panic test

**Contracts Implemented:** BC-11

**Files:**
- Modify: `sim/simulator_test.go` (add nil-guard test)

**Step 1: Write nil-guard test**

In `sim/simulator_test.go`, add (following `TestMustNewKVCacheState_NilFunc_Panics` pattern):

```go
func TestMustNewLatencyModel_NilFunc_Panics(t *testing.T) {
	saved := NewLatencyModelFunc
	defer func() { NewLatencyModelFunc = saved }()
	NewLatencyModelFunc = nil

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for nil NewLatencyModelFunc")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("expected string panic, got %T: %v", r, r)
		}
		expected := "NewLatencyModelFunc not registered: import sim/latency to register it " +
			"(add: import _ \"github.com/inference-sim/inference-sim/sim/latency\")"
		if msg != expected {
			t.Errorf("panic message mismatch:\ngot:  %q\nwant: %q", msg, expected)
		}
	}()
	coeffs := NewLatencyCoeffs([]float64{1, 2, 3}, []float64{1, 2, 3})
	hw := NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, false)
	MustNewLatencyModel(coeffs, hw)
}
```

**Step 2: Run test**

Run: `go test ./sim/... -run TestMustNewLatencyModel_NilFunc_Panics -v`
Expected: PASS

**Step 3: Commit**

```bash
git add sim/simulator_test.go
git commit -m "test(sim): add MustNewLatencyModel nil-guard panic test (BC-11)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 3: Update CLAUDE.md and documentation

**Contracts Implemented:** Documentation update

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/extension-recipes.md`

**Step 1: Update CLAUDE.md File Organization + Latency Estimation section**

In the File Organization tree, update the `sim/` section:
- Remove `latency_model.go` description mentioning implementations
- Add `sim/latency/` section
- Remove `roofline_step.go` and `model_hardware_config.go` from sim/ listing
- Update latency_model.go description to "LatencyModel interface + registration"
- Keep `model_hardware_config.go` listed but update description to "ModelConfig, HardwareCalib structs (config types stay in sim/)"

Also update the "Latency Estimation" prose section (~line 265-271): change `roofline_step.go` reference to `sim/latency/roofline.go`, and note that `NewLatencyModel()` factory is in `sim/latency/`.

**Step 2: Update docs/extension-recipes.md**

Update references from `sim/latency_model.go` to `sim/latency/latency.go`:
```
- See `BlackboxLatencyModel` in `sim/latency/latency.go`
- See `RooflineLatencyModel` in `sim/latency/latency.go`
```

Update the factory registration note:
```
2. **Register in `NewLatencyModel` factory** in `sim/latency/latency.go`
```

**Step 3: Commit**

```bash
git add CLAUDE.md docs/extension-recipes.md
git commit -m "docs: update file organization and extension recipes for sim/latency/ extraction

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 4: Final verification

**Contracts Implemented:** All (verification gate)

**Step 1: Build**

Run: `go build ./...`
Expected: PASS

**Step 2: Test**

Run: `go test ./... -count=1`
Expected: ALL PASS (same test count as baseline)

**Step 3: Lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 4: Verify golden dataset unchanged**

The golden dataset test in `sim/cluster/cluster_test.go` must pass without regeneration — byte-identical output confirms BC-9.

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|--------------------------|
| BC-1 | Task 1 | Build | `go build ./...` — no import of sim/latency from sim/ |
| BC-2 | Task 1 | Unit | `TestNewLatencyModel_BlackboxMode` (moved to sim/latency/) |
| BC-3 | Task 1 | Unit | `TestNewLatencyModel_RooflineMode` (moved to sim/latency/) |
| BC-4 | Task 2 | Unit | `TestMustNewLatencyModel_NilFunc_Panics` |
| BC-5 | Task 1 | Integration | Cluster tests pass with latency.NewLatencyModel |
| BC-6 | Task 1 | Build | `go build ./cmd/...` — cmd/root.go compiles with latency.* |
| BC-7 | Task 1 | Build | StepConfig types not referenced from sim/simulator.go |
| BC-8 | Task 1 | Build | No import cycle (`go build ./...` succeeds) |
| BC-9 | Task 4 | Golden | Golden dataset test passes without regeneration |
| BC-10 | Task 1 | Unit | All NaN/Inf/short-coeff tests pass (moved to sim/latency/) |
| BC-11 | Task 2 | Unit | `TestMustNewLatencyModel_NilFunc_Panics` |

No golden dataset regeneration needed. No new invariant tests needed (pure refactoring).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Import cycle sim/ ↔ sim/latency/ | Low | High (build fails) | Registration pattern; `go build` validates | Task 1 |
| `ModelConfig`/`HardwareCalib` in wrong package | Medium | Medium (cycle) | Keep in sim/ — they're in ModelHardwareConfig | Task 1 |
| Test files reference unexported types | Medium | Medium (compile error) | Use factory path (MustNewLatencyModel) | Task 1 (Step 14) |
| Missing call site update | Low | High (compile error) | Grep for all `sim.NewLatencyModel`, `sim.GetHWConfig` etc. | Task 1 (Step 17) |
| sim/cluster/cluster_test.go references latency types | Low | Medium | Check for direct struct construction in cluster tests | Task 1 (Step 11) |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — pure extraction, no new interfaces
- [x] No feature creep — no behavioral changes
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes — all output byte-identical
- [x] No hidden global state impact — registration variable is the established pattern
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used (testModelConfig, testHardwareCalib)
- [x] CLAUDE.md updated (Task 3)
- [x] No stale references in CLAUDE.md
- [x] Documentation DRY: extension-recipes.md updated (Task 3)
- [x] Deviation log reviewed — 3 deviations, all justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1→2→3→4)
- [x] All contracts mapped to tasks
- [x] No golden dataset regeneration needed
- [x] Construction site audit: no new struct fields added; all `NewLatencyModel` call sites (15 in sim/ tests + 1 cluster + 1 cluster_test + 2 cmd/) explicitly enumerated
- [x] R6: `MustNewLatencyModel` panic is programming-error guard only (not user-reachable); follows `MustNewKVCacheState` precedent
- [x] R4: All construction sites for `NewLatencyModel` explicitly listed (batch_formation_test.go ×7+1, simulator_test.go ×6, simulator_preempt_test.go ×2, cluster/instance.go ×1, cluster/cluster_test.go ×1, cmd/root.go ×2)

---

## Appendix: File-Level Implementation Details

### File: `sim/latency/latency.go`

**Purpose:** BlackboxLatencyModel, RooflineLatencyModel implementations and NewLatencyModel factory.

Complete implementation is specified in Task 1, Step 1 above. Key notes:
- Package `latency`, imports `sim` and `sim/internal/util`
- Struct fields remain unexported (betaCoeffs, alphaCoeffs, etc.) — constructed only by factory
- `NewLatencyModel` returns `sim.LatencyModel` interface
- All error messages unchanged from original

### File: `sim/latency/roofline.go`

**Purpose:** Roofline analytical latency estimation functions and step config types.

Verbatim move from `sim/roofline_step.go` with package change and `sim.ModelConfig`/`sim.HardwareCalib` type prefixes. Plus `StepConfig`, `PrefillRequestConfig`, `DecodeRequestConfig` from `sim/simulator.go`.

### File: `sim/latency/config.go`

**Purpose:** HFConfig type, hardware/model config parsing, roofline validation.

Verbatim move from `sim/model_hardware_config.go` (minus `ModelConfig` and `HardwareCalib` structs which stay in sim/). Functions return `sim.ModelConfig`, `sim.HardwareCalib`, etc.

### File: `sim/latency/register.go`

**Purpose:** Wire `NewLatencyModel` into `sim.NewLatencyModelFunc` via init().

4-line file following `sim/kv/register.go` pattern exactly.

### File: `sim/latency_model.go` (modified)

**Purpose:** LatencyModel interface definition + registration variable + nil-guarded wrapper.

Reduced from ~175 lines to ~30 lines. Interface stays, everything else moves.

### File: `sim/model_hardware_config.go` (modified)

**Purpose:** `ModelConfig` and `HardwareCalib` struct definitions only.

Reduced from ~293 lines to ~35 lines. Structs stay because they're embedded in `ModelHardwareConfig` (sim/config.go).

### File: `sim/latency_import_test.go` (new)

**Purpose:** Blank import to trigger registration for `package sim_test` files.

2-line file following `sim/kv_import_test.go` pattern exactly.
