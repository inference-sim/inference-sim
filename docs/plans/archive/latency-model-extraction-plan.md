# LatencyModel Interface Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract the hardcoded latency estimation logic from the simulator's Step() method into a pluggable LatencyModel interface, enabling future alternative backends (SGLang, TensorRT-LLM) without modifying core simulation code.

**The problem today:** The simulator estimates execution times through 6 private methods hardcoded on the `Simulator` struct. A boolean flag (`sim.roofline`) selects between two strategies via if/else branching. Adding a third latency model requires modifying `simulator.go` core — the "backend swap" extension type (design guidelines Section 5.4) is impossible. Additionally, `makeRunningBatch()` accumulates 12 regression feature increments that couple batch formation to the blackbox latency model, including dead writes to fields that are never read.

**What this PR adds:**
1. **LatencyModel interface** — a 5-method contract (StepTime, QueueingTime, OutputTokenProcessingTime, SchedulingProcessingTime, PreemptionProcessingTime) that any latency backend can implement
2. **Two implementations** — BlackboxLatencyModel (alpha/beta regression, the current default) and RooflineLatencyModel (analytical FLOPs/bandwidth), extracted from existing code with identical behavior
3. **Simplified Step()** — the `if sim.roofline` branch becomes a single `sim.latencyModel.StepTime(batch)` call; `makeRunningBatch()` drops 10 lines of feature accumulation
4. **Dead code removal** — 4 unused fields in `RegressionFeatures` and 8 dead write statements eliminated

**Why this matters:** This is Phase A of the backend swap recipe (design guidelines Section 5.4). It unblocks #242 (BatchFormation extraction), #243 (state/statistics separation), and future alternative backend implementations.

**Architecture:** New file `sim/latency_model.go` contains the interface, both implementations, and the factory. The factory is called inside `NewSimulator()`, replacing 7 Simulator fields with 1 `latencyModel` field. Call sites in `Step()`, `makeRunningBatch()`, `preempt()`, and `event.go` switch from `sim.get*()` to `sim.latencyModel.*()`.

**Source:** GitHub issue #241, design doc `docs/plans/2026-02-21-latency-model-extraction-design.md`

**Closes:** Fixes #241

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR extracts latency estimation logic from `Simulator` into a `LatencyModel` interface with two implementations (Blackbox and Roofline). The extraction follows the backend swap recipe: define interface, move existing code behind it, verify identical behavior.

The LatencyModel sits between the batch formation and step execution phases. It observes the running batch composition and produces time estimates in microseconds (ticks). It interacts with `Step()` (step time), `ArrivalEvent.Execute()` (queueing time), `makeRunningBatch()` (scheduling overhead), and `preempt()` (preemption overhead).

No deviations from the design doc. Discovery during Phase 0: 4 of 6 `RegressionFeatures` fields (`NumPrefillRequests`, `TotalPrefillTokens`, `NumDecodeRequests`, `MaxPrefillTokens`) are write-only dead code. They will be removed along with the struct.

### B) Behavioral Contracts

**Positive Contracts:**

**BC-1: Blackbox StepTime Equivalence**
- GIVEN a simulator configured with blackbox mode (roofline=false) and known beta coefficients
- WHEN StepTime is called with a batch containing prefill and decode requests
- THEN the estimated step time in microseconds MUST equal the value produced by the original `getStepTime()` formula for the same batch composition
- MECHANISM: BlackboxLatencyModel walks the batch, sums cache-miss tokens and decode tokens from `NumNewTokens` per request, applies `beta0 + beta1*cacheMiss + beta2*decode`

**BC-2: Roofline StepTime Equivalence**
- GIVEN a simulator configured with roofline mode and valid model/hardware config
- WHEN StepTime is called with a batch containing prefill and decode requests
- THEN the estimated step time MUST equal `rooflineStepTime()` called with the same StepConfig that `getStepTimeRoofline()` would have constructed
- MECHANISM: RooflineLatencyModel builds StepConfig from batch request states, delegates to existing `rooflineStepTime()` pure function

**BC-3: QueueingTime Equivalence**
- GIVEN either latency model and known alpha coefficients
- WHEN QueueingTime is called with a request
- THEN the result MUST equal `alpha0 + alpha1 * len(req.InputTokens)` (the original formula)
- MECHANISM: Both implementations apply the same alpha-coefficient formula

**BC-4: Factory Selection**
- GIVEN a SimConfig with Roofline=false
- WHEN NewLatencyModel creates the model
- THEN StepTime produces results consistent with beta-coefficient estimation (not roofline)
- GIVEN a SimConfig with Roofline=true and valid model/hardware config
- WHEN NewLatencyModel creates the model
- THEN StepTime produces results consistent with roofline estimation (not beta-coefficient)

**BC-5: End-to-End Golden Dataset Preservation**
- GIVEN the existing golden dataset test configuration
- WHEN the full simulation runs after refactoring
- THEN all golden dataset output values MUST be byte-identical to pre-refactoring output
- MECHANISM: Pure refactoring with identical computation paths

**Negative Contracts:**

**BC-6: No Simulator Field Leakage**
- GIVEN the refactored Simulator struct
- WHEN inspecting Simulator's fields
- THEN `betaCoeffs`, `alphaCoeffs`, `runningBatchFeatures`, `roofline`, `modelConfig`, `hwConfig` MUST NOT exist as Simulator fields
- MECHANISM: These fields move into the LatencyModel implementations

**BC-7: No Dead Code**
- GIVEN the refactored codebase
- WHEN searching for `RegressionFeatures` struct definition
- THEN it MUST NOT exist — its useful fields are computed locally within `BlackboxLatencyModel.StepTime()`
- MECHANISM: The struct is removed; feature extraction becomes a local computation

**Error Handling Contracts:**

**BC-8: Factory Validation**
- GIVEN a SimConfig with Roofline=true but invalid model/hardware config
- WHEN NewLatencyModel is called
- THEN it MUST return a non-nil error (not panic)
- MECHANISM: Factory validates roofline config and returns error

### C) Component Interaction

```
                    ┌──────────────────┐
                    │   SimConfig      │
                    │ (BetaCoeffs,     │
                    │  AlphaCoeffs,    │
                    │  Roofline, ...)  │
                    └────────┬─────────┘
                             │ NewLatencyModel(cfg)
                             ▼
                    ┌──────────────────┐
                    │  LatencyModel    │◄─── interface
                    │  (5 methods)     │
                    └───────┬──────────┘
                   ┌────────┴────────┐
                   ▼                 ▼
          ┌──────────────┐  ┌───────────────┐
          │ Blackbox     │  │ Roofline      │
          │ (beta/alpha) │  │ (FLOPs/BW)    │
          └──────────────┘  └───────────────┘
                   │                 │
                   └────────┬────────┘
                            ▼
            ┌─────────────────────────────┐
            │         Simulator           │
            │  latencyModel LatencyModel  │
            │                             │
            │  Step() ──► StepTime(batch) │
            │  event.go ► QueueingTime()  │
            │  preempt() ► PreemptionPT() │
            └─────────────────────────────┘
```

**API contracts:**
- `NewLatencyModel(SimConfig) (LatencyModel, error)` — factory, returns error only for roofline validation failure
- `StepTime([]*Request) int64` — precondition: batch has `NumNewTokens` set per request (done by `makeRunningBatch`)
- `QueueingTime(*Request) int64` — precondition: request has `InputTokens` populated

**Extension friction:** Adding a third latency model: 1 file (new implementation + registration in factory switch). This matches the design guidelines target of ~2 touch points.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| `RegressionFeatures` struct stays (used internally by BlackboxLatencyModel) | `RegressionFeatures` struct is removed entirely | CORRECTION: Phase 0 audit found only 2 of 6 fields are read; BlackboxLatencyModel computes these 2 values locally without the struct |
| Design doc does not mention `reqNumComputedTokens` | `reqNumComputedTokens` stays on Simulator unchanged | ADDITION: This field is batch formation state, not latency model state; no change needed |
| Design doc says remove `tp` from Simulator | `tp` is removed from Simulator as specified | N/A — no deviation, micro plan matches design doc |
| Design doc testing item 5: "Feature extraction equivalence" as a distinct test | No separate equivalence test; covered by BC-1 (formula verification) + BC-5 (golden dataset) | SIMPLIFICATION: BC-1 directly verifies the formula output matches expectations; BC-5 proves the full pipeline produces identical results. A side-by-side test of old vs new paths would require keeping the old code around, defeating the purpose of the extraction. |

### E) Review Guide

**The tricky part:** BC-1 (Blackbox StepTime Equivalence). The current code accumulates `TotalCacheMissTokens` and `TotalDecodeTokens` incrementally during `makeRunningBatch()`. The new code computes them from the final batch state. These must produce identical values. The proof: both are additive sums over the same requests, and `NumNewTokens` is set before `StepTime` is called.

**What to scrutinize:** The `StepTime` method on `BlackboxLatencyModel` — verify it correctly distinguishes prefill vs decode requests using `ProgressIndex < len(InputTokens)` and uses `NumNewTokens` (not a recomputed value).

**What's safe to skim:** RooflineLatencyModel (it literally delegates to the existing `rooflineStepTime()` pure function with the same StepConfig construction). Factory is straightforward. Alpha-based methods are trivial.

**Known debt:** `SchedulingProcessingTime()` and `PreemptionProcessingTime()` return hardcoded 0. These are placeholders for future work (noted in existing code with ToDo comments). Additionally, `model` and `gpu` fields on Simulator are pre-existing dead code (written but never read in `sim/`). Removing them is out of scope for this PR — file a separate issue if desired.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/latency_model.go` — interface + 2 implementations + factory

**Files to modify:**
- `sim/simulator.go` — remove 6 methods, 7 fields, `RegressionFeatures` struct; add `latencyModel` field; simplify `Step()` and `makeRunningBatch()`
- `sim/event.go` — 1 call site update (line 31)

**Files to create (test):**
- `sim/latency_model_test.go` — behavioral tests for BC-1 through BC-8

**Key decisions:**
- `NumNewTokens` on `*Request` is the bridge: set by `makeRunningBatch()`, read by `StepTime()`
- `RegressionFeatures` struct is deleted entirely (dead fields outweigh utility)
- `roofline_step.go` unchanged (pure functions, already well-tested)

**Confirmation:** No dead code — every method is called, every field is used, all paths exercisable.

### G) Task Breakdown

---

#### Task 1: Create LatencyModel Interface and BlackboxLatencyModel

**Contracts Implemented:** BC-1, BC-3, BC-4 (blackbox path), BC-7 (partial — no RegressionFeatures in new code)

**Files:**
- Create: `sim/latency_model.go`
- Test: `sim/latency_model_test.go`

**Step 1: Write failing tests for BlackboxLatencyModel**

Context: We test that the blackbox model produces the correct step time and queueing time given known coefficients and a known batch composition.

```go
// sim/latency_model_test.go
package sim

import (
	"testing"
)

// TestBlackboxLatencyModel_StepTime_PrefillAndDecode verifies BC-1:
// StepTime produces beta0 + beta1*cacheMissTokens + beta2*decodeTokens.
func TestBlackboxLatencyModel_StepTime_PrefillAndDecode(t *testing.T) {
	// GIVEN a blackbox model with known beta coefficients
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	// AND a batch with 1 prefill request (30 new tokens) and 1 decode request
	batch := []*Request{
		{
			InputTokens:  make([]int, 100),
			ProgressIndex: 50, // < len(InputTokens), so prefill
			NumNewTokens:  30,
		},
		{
			InputTokens:  make([]int, 50),
			OutputTokens: make([]int, 20),
			ProgressIndex: 60, // >= len(InputTokens), so decode
			NumNewTokens:  1,
		},
	}

	// WHEN StepTime is called
	result := model.StepTime(batch)

	// THEN result = beta0 + beta1*30 + beta2*1 = 1000 + 300 + 5 = 1305
	expected := int64(1305)
	if result != expected {
		t.Errorf("StepTime = %d, want %d", result, expected)
	}
}

// TestBlackboxLatencyModel_StepTime_EmptyBatch verifies StepTime with no requests.
func TestBlackboxLatencyModel_StepTime_EmptyBatch(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	// WHEN StepTime is called with an empty batch
	result := model.StepTime([]*Request{})

	// THEN result = beta0 only = 1000
	if result != 1000 {
		t.Errorf("StepTime(empty) = %d, want 1000", result)
	}
}

// TestBlackboxLatencyModel_QueueingTime verifies BC-3:
// QueueingTime = alpha0 + alpha1 * len(InputTokens).
func TestBlackboxLatencyModel_QueueingTime(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	req := &Request{InputTokens: make([]int, 50)}

	// WHEN QueueingTime is called
	result := model.QueueingTime(req)

	// THEN result = alpha0 + alpha1*50 = 100 + 50 = 150
	if result != 150 {
		t.Errorf("QueueingTime = %d, want 150", result)
	}
}

// TestBlackboxLatencyModel_OutputTokenProcessingTime verifies the alpha2 overhead.
func TestBlackboxLatencyModel_OutputTokenProcessingTime(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 200},
	}

	result := model.OutputTokenProcessingTime()

	if result != 200 {
		t.Errorf("OutputTokenProcessingTime = %d, want 200", result)
	}
}

// TestBlackboxLatencyModel_PlaceholderOverheads verifies placeholders return 0.
func TestBlackboxLatencyModel_PlaceholderOverheads(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	if model.SchedulingProcessingTime() != 0 {
		t.Errorf("SchedulingProcessingTime = %d, want 0", model.SchedulingProcessingTime())
	}
	if model.PreemptionProcessingTime() != 0 {
		t.Errorf("PreemptionProcessingTime = %d, want 0", model.PreemptionProcessingTime())
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestBlackboxLatencyModel -v`
Expected: FAIL — `BlackboxLatencyModel` type not defined

**Step 3: Implement LatencyModel interface and BlackboxLatencyModel**

Context: Define the interface and the blackbox implementation. The blackbox model computes regression features from the batch on-demand.

In `sim/latency_model.go`:
```go
// sim/latency_model.go
package sim

// LatencyModel estimates execution times for the DES step loop.
// Two implementations exist: BlackboxLatencyModel (alpha/beta regression)
// and RooflineLatencyModel (analytical FLOPs/bandwidth).
// All time estimates are in microseconds (ticks).
type LatencyModel interface {
	// StepTime estimates the duration of one batch step given the running batch.
	// Precondition: each request in batch has NumNewTokens set by makeRunningBatch().
	StepTime(batch []*Request) int64

	// QueueingTime estimates the arrival-to-queue delay for a request.
	QueueingTime(req *Request) int64

	// OutputTokenProcessingTime estimates per-token post-processing time.
	OutputTokenProcessingTime() int64

	// SchedulingProcessingTime estimates scheduling overhead per request.
	SchedulingProcessingTime() int64

	// PreemptionProcessingTime estimates preemption overhead per eviction.
	PreemptionProcessingTime() int64
}

// BlackboxLatencyModel estimates latency using trained alpha/beta regression coefficients.
// Beta coefficients estimate step time: beta0 + beta1*cacheMissTokens + beta2*decodeTokens.
// Alpha coefficients estimate overheads: alpha0 + alpha1*inputLen (queueing), alpha2 (output processing).
type BlackboxLatencyModel struct {
	betaCoeffs  []float64
	alphaCoeffs []float64
}

func (m *BlackboxLatencyModel) StepTime(batch []*Request) int64 {
	var totalCacheMissTokens, totalDecodeTokens int64
	for _, req := range batch {
		if req.ProgressIndex < Len64(req.InputTokens) {
			// Prefill phase: NumNewTokens are cache-miss tokens
			totalCacheMissTokens += int64(req.NumNewTokens)
		} else if len(req.OutputTokens) > 0 {
			// Decode phase
			totalDecodeTokens += int64(req.NumNewTokens)
		}
	}
	var totalStepTime float64
	totalStepTime += m.betaCoeffs[0]
	totalStepTime += m.betaCoeffs[1] * float64(totalCacheMissTokens)
	totalStepTime += m.betaCoeffs[2] * float64(totalDecodeTokens)
	return int64(totalStepTime)
}

func (m *BlackboxLatencyModel) QueueingTime(req *Request) int64 {
	var totalProcessingTime float64
	totalProcessingTime += m.alphaCoeffs[0]
	totalProcessingTime += m.alphaCoeffs[1] * float64(len(req.InputTokens))
	return int64(totalProcessingTime)
}

func (m *BlackboxLatencyModel) OutputTokenProcessingTime() int64 {
	return int64(m.alphaCoeffs[2])
}

func (m *BlackboxLatencyModel) SchedulingProcessingTime() int64 {
	return 0
}

func (m *BlackboxLatencyModel) PreemptionProcessingTime() int64 {
	return 0
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestBlackboxLatencyModel -v`
Expected: PASS (all 5 tests)

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/latency_model.go sim/latency_model_test.go
git commit -m "feat(sim): add LatencyModel interface and BlackboxLatencyModel (BC-1, BC-3)

- Define LatencyModel interface with 5 methods
- Implement BlackboxLatencyModel using alpha/beta regression
- StepTime computes features from batch on-demand (no RegressionFeatures struct)
- Add behavioral tests for step time, queueing time, and placeholder overheads

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 2: Add RooflineLatencyModel and Factory

**Contracts Implemented:** BC-2, BC-4, BC-8

**Files:**
- Modify: `sim/latency_model.go` (add RooflineLatencyModel + NewLatencyModel factory)
- Modify: `sim/latency_model_test.go` (add roofline + factory tests)

**Step 1: Write failing tests for RooflineLatencyModel and factory**

Context: We test that the roofline model delegates correctly and the factory selects the right implementation.

```go
// Add to sim/latency_model_test.go

// TestRooflineLatencyModel_StepTime_DelegatesToRooflineStepTime verifies BC-2:
// StepTime builds StepConfig from batch and delegates to rooflineStepTime.
func TestRooflineLatencyModel_StepTime_DelegatesToRooflineStepTime(t *testing.T) {
	// GIVEN a roofline model with valid config (reuse test config from roofline_step_test.go)
	model := &RooflineLatencyModel{
		modelConfig: testModelConfig(),
		hwConfig:    testHardwareCalib(),
		tp:          2,
		alphaCoeffs: []float64{100, 1, 100},
	}

	// AND a batch with 1 prefill request
	batch := []*Request{
		{
			InputTokens:  make([]int, 100),
			ProgressIndex: 0,
			NumNewTokens:  100,
		},
	}

	// WHEN StepTime is called
	result := model.StepTime(batch)

	// THEN result must be positive and finite (same behavior as rooflineStepTime)
	if result <= 0 {
		t.Errorf("StepTime = %d, want > 0", result)
	}

	// AND it must match calling rooflineStepTime directly with the same config
	expectedConfig := StepConfig{
		PrefillRequests: []PrefillRequestConfig{
			{ProgressIndex: 0, NumNewPrefillTokens: 100},
		},
	}
	expected := rooflineStepTime(model.modelConfig, model.hwConfig, expectedConfig, model.tp)
	if result != expected {
		t.Errorf("StepTime = %d, want %d (from rooflineStepTime)", result, expected)
	}
}

// TestRooflineLatencyModel_StepTime_EmptyBatch verifies roofline handles empty batch.
func TestRooflineLatencyModel_StepTime_EmptyBatch(t *testing.T) {
	model := &RooflineLatencyModel{
		modelConfig: testModelConfig(),
		hwConfig:    testHardwareCalib(),
		tp:          2,
		alphaCoeffs: []float64{100, 1, 100},
	}

	// WHEN StepTime is called with an empty batch
	result := model.StepTime([]*Request{})

	// THEN result must match rooflineStepTime with empty config (overhead only)
	expected := rooflineStepTime(model.modelConfig, model.hwConfig, StepConfig{}, model.tp)
	if result != expected {
		t.Errorf("StepTime(empty) = %d, want %d", result, expected)
	}
}

// TestRooflineLatencyModel_QueueingTime verifies BC-3 for roofline model.
func TestRooflineLatencyModel_QueueingTime(t *testing.T) {
	model := &RooflineLatencyModel{
		modelConfig: testModelConfig(),
		hwConfig:    testHardwareCalib(),
		tp:          2,
		alphaCoeffs: []float64{100, 1, 100},
	}

	req := &Request{InputTokens: make([]int, 50)}
	result := model.QueueingTime(req)

	// Same alpha formula as blackbox: alpha0 + alpha1*50 = 150
	if result != 150 {
		t.Errorf("QueueingTime = %d, want 150", result)
	}
}

// TestNewLatencyModel_BlackboxMode verifies BC-4 (blackbox path).
func TestNewLatencyModel_BlackboxMode(t *testing.T) {
	cfg := SimConfig{
		Roofline:    false,
		BetaCoeffs:  []float64{1000, 10, 5},
		AlphaCoeffs: []float64{100, 1, 100},
	}

	model, err := NewLatencyModel(cfg)
	if err != nil {
		t.Fatalf("NewLatencyModel returned error: %v", err)
	}

	// THEN the model produces blackbox-consistent results
	batch := []*Request{
		{
			InputTokens:  make([]int, 100),
			ProgressIndex: 50,
			NumNewTokens:  30,
		},
	}
	result := model.StepTime(batch)
	// beta0 + beta1*30 = 1000 + 300 = 1300
	if result != 1300 {
		t.Errorf("StepTime = %d, want 1300 (blackbox mode)", result)
	}
}

// TestNewLatencyModel_RooflineMode verifies BC-4 (roofline path).
func TestNewLatencyModel_RooflineMode(t *testing.T) {
	cfg := SimConfig{
		Roofline:    true,
		AlphaCoeffs: []float64{100, 1, 100},
		ModelConfig: testModelConfig(),
		HWConfig:    testHardwareCalib(),
		TP:          2,
	}

	model, err := NewLatencyModel(cfg)
	if err != nil {
		t.Fatalf("NewLatencyModel returned error: %v", err)
	}

	// THEN the model produces roofline-consistent results (positive, finite)
	batch := []*Request{
		{
			InputTokens:  make([]int, 100),
			ProgressIndex: 0,
			NumNewTokens:  100,
		},
	}
	result := model.StepTime(batch)
	if result <= 0 {
		t.Errorf("StepTime = %d, want > 0 (roofline mode)", result)
	}
}

// TestNewLatencyModel_InvalidRoofline verifies BC-8.
func TestNewLatencyModel_InvalidRoofline(t *testing.T) {
	cfg := SimConfig{
		Roofline:    true,
		AlphaCoeffs: []float64{100, 1, 100},
		// ModelConfig and HWConfig left empty — invalid for roofline
	}

	_, err := NewLatencyModel(cfg)
	if err == nil {
		t.Fatal("expected error for invalid roofline config, got nil")
	}
}
```

We also need test helpers. Check if `testModelConfig()` and `testHardwareCalib()` exist:

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run "TestRooflineLatencyModel|TestNewLatencyModel" -v`
Expected: FAIL — `RooflineLatencyModel` and `NewLatencyModel` not defined

**Step 3: Implement RooflineLatencyModel and NewLatencyModel factory**

Context: Add the roofline implementation (delegates to existing `rooflineStepTime()`) and the factory.

Append to `sim/latency_model.go`:
```go
import "fmt"

// RooflineLatencyModel estimates latency using analytical FLOPs/bandwidth roofline model.
// Step time is computed via rooflineStepTime(); overhead estimates use alpha coefficients.
type RooflineLatencyModel struct {
	modelConfig ModelConfig
	hwConfig    HardwareCalib
	tp          int
	alphaCoeffs []float64
}

func (m *RooflineLatencyModel) StepTime(batch []*Request) int64 {
	stepConfig := StepConfig{
		PrefillRequests: make([]PrefillRequestConfig, 0, len(batch)),
		DecodeRequests:  make([]DecodeRequestConfig, 0, len(batch)),
	}
	for _, req := range batch {
		if req.ProgressIndex < Len64(req.InputTokens) {
			stepConfig.PrefillRequests = append(stepConfig.PrefillRequests, PrefillRequestConfig{
				ProgressIndex:       req.ProgressIndex,
				NumNewPrefillTokens: req.NumNewTokens,
			})
		} else {
			stepConfig.DecodeRequests = append(stepConfig.DecodeRequests, DecodeRequestConfig{
				ProgressIndex:      req.ProgressIndex,
				NumNewDecodeTokens: req.NumNewTokens,
			})
		}
	}
	return rooflineStepTime(m.modelConfig, m.hwConfig, stepConfig, m.tp)
}

func (m *RooflineLatencyModel) QueueingTime(req *Request) int64 {
	var totalProcessingTime float64
	totalProcessingTime += m.alphaCoeffs[0]
	totalProcessingTime += m.alphaCoeffs[1] * float64(len(req.InputTokens))
	return int64(totalProcessingTime)
}

func (m *RooflineLatencyModel) OutputTokenProcessingTime() int64 {
	return int64(m.alphaCoeffs[2])
}

func (m *RooflineLatencyModel) SchedulingProcessingTime() int64 {
	return 0
}

func (m *RooflineLatencyModel) PreemptionProcessingTime() int64 {
	return 0
}

// NewLatencyModel creates the appropriate LatencyModel based on SimConfig.
// Returns RooflineLatencyModel if cfg.Roofline is true, BlackboxLatencyModel otherwise.
// Returns error if roofline config validation fails.
func NewLatencyModel(cfg SimConfig) (LatencyModel, error) {
	if cfg.Roofline {
		if cfg.TP <= 0 {
			return nil, fmt.Errorf("latency model: roofline requires TP > 0, got %d", cfg.TP)
		}
		if err := ValidateRooflineConfig(cfg.ModelConfig, cfg.HWConfig); err != nil {
			return nil, fmt.Errorf("latency model: %w", err)
		}
		return &RooflineLatencyModel{
			modelConfig: cfg.ModelConfig,
			hwConfig:    cfg.HWConfig,
			tp:          cfg.TP,
			alphaCoeffs: cfg.AlphaCoeffs,
		}, nil
	}
	return &BlackboxLatencyModel{
		betaCoeffs:  cfg.BetaCoeffs,
		alphaCoeffs: cfg.AlphaCoeffs,
	}, nil
}
```

Also check that `testModelConfig()` and `testHardwareCalib()` exist in `sim/roofline_step_test.go` — if they do, they're available since they're in the same package. If not, add them to `latency_model_test.go`.

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run "TestRooflineLatencyModel|TestNewLatencyModel|TestBlackboxLatencyModel" -v`
Expected: PASS (all tests)

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/latency_model.go sim/latency_model_test.go
git commit -m "feat(sim): add RooflineLatencyModel and NewLatencyModel factory (BC-2, BC-4, BC-8)

- Implement RooflineLatencyModel delegating to rooflineStepTime()
- Add NewLatencyModel factory with roofline config validation
- Add behavioral tests for factory selection and roofline delegation

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 3: Wire LatencyModel into Simulator and Remove Old Methods

**Contracts Implemented:** BC-5 (partial), BC-6, BC-7

**Files:**
- Modify: `sim/simulator.go` — replace fields and methods, update Step()/makeRunningBatch()/preempt()
- Modify: `sim/event.go` — update QueueingTime call site

**Step 1: Write failing test to verify the integration compiles**

Context: We don't need new tests here — the existing test suite (especially the golden dataset tests and `TestInjectArrival_RequestCompletes`) verifies BC-5. But we need to make the changes and verify everything still passes.

No new test file needed — existing tests serve as the regression suite.

**Step 2: Modify NewSimulator to use NewLatencyModel**

In `sim/simulator.go`, modify `NewSimulator`:

1. Remove the roofline validation block (lines 197-204) — this moves into `NewLatencyModel`
2. After creating the simulator struct, call `NewLatencyModel(cfg)` and store result
3. Remove fields from Simulator struct: `betaCoeffs`, `alphaCoeffs`, `runningBatchFeatures`, `roofline`, `modelConfig`, `hwConfig`, `tp`
4. Add field: `latencyModel LatencyModel`
5. Remove the `RegressionFeatures` struct definition
6. Remove the 6 `get*` methods

In `sim/simulator.go`, `Step()`:
- Replace `if sim.roofline { currStepAdvance = sim.getStepTimeRoofline() } else { currStepAdvance = sim.getStepTime() }` with `currStepAdvance = sim.latencyModel.StepTime(sim.RunningBatch.Requests)`
- Replace `sim.getOutputTokenProcessingTime()` with `sim.latencyModel.OutputTokenProcessingTime()`
- Remove `sim.runningBatchFeatures = RegressionFeatures{...}` reset at top of Step()

In `sim/simulator.go`, `makeRunningBatch()`:
- Remove all 10 `sim.runningBatchFeatures.*` increment lines (lines 484, 486-488, 507-508, 563, 565-567)
- The `reqNumComputedTokens` tracking stays — that's batch formation state

In `sim/simulator.go`, `preempt()`:
- Replace `sim.getPreemptionProcessingTime()` with `sim.latencyModel.PreemptionProcessingTime()`

In `sim/simulator.go`, `makeRunningBatch()`:
- Replace `sim.getSchedulingProcessingTime()` with `sim.latencyModel.SchedulingProcessingTime()`

In `sim/event.go`:
- Replace `sim.getQueueingTime(e.Request)` with `sim.latencyModel.QueueingTime(e.Request)`

In `sim/simulator.go`, `getStepTimeRoofline()` method:
- This method is deleted — its logic now lives in `RooflineLatencyModel.StepTime()`

**Step 3: Run full test suite to verify BC-5 (golden dataset preservation)**

Run: `go test ./sim/... -v -count=1`
Expected: PASS (all existing tests including golden dataset)

Run: `go test ./sim/cluster/... -v -count=1`
Expected: PASS

Run: `go test ./... -count=1`
Expected: PASS

**Step 4: Run lint check**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/simulator.go sim/event.go
git commit -m "refactor(sim): wire LatencyModel into Simulator, remove old methods (BC-5, BC-6, BC-7)

- Replace 7 Simulator fields with single latencyModel LatencyModel field
- Remove 6 private get* methods from Simulator
- Remove RegressionFeatures struct (dead code: 4 of 6 fields were never read)
- Remove 10 runningBatchFeatures increment lines from makeRunningBatch()
- Simplify Step(): if/else roofline branch → latencyModel.StepTime(batch)
- Update event.go QueueingTime call site
- All existing tests pass unchanged — pure refactoring

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 4: Update Test Construction Sites and Workload Config

**Contracts Implemented:** BC-5 (full verification)

**Files:**
- Verify: `sim/workload_config_test.go` — confirm `&Simulator{}` literals still compile (they only reference `Metrics`, which stays)
- Verify: All `SimConfig{}` construction sites in test files still compile

**Step 1: Verify all test construction sites compile**

Context: `SimConfig` still has the same fields (BetaCoeffs, AlphaCoeffs, Roofline, etc.) — only `Simulator` struct changed. Test code constructing `SimConfig{}` should be unaffected. Test code constructing `Simulator{}` directly (workload_config_test.go) may need updating.

Check `sim/workload_config_test.go` — it constructs `&Simulator{...}` directly with a `Metrics` field. This should still work since `Metrics` stays on `Simulator`. But it may need `latencyModel` set if any method path requires it.

**Step 2: Run all tests to verify construction sites**

Run: `go test ./... -count=1`
Expected: PASS — if any test constructs `Simulator{}` with removed fields, it will fail to compile

**Step 3: Fix any compilation issues**

If `workload_config_test.go` needs updating (unlikely since it only accesses `Metrics.RequestRate`), update the construction sites. The tests construct minimal `Simulator{}` structs with only the fields they use.

**Step 4: Run lint check**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 5: Commit (if changes were needed)**

```bash
git add sim/workload_config_test.go  # only if changed
git commit -m "fix(sim): update test construction sites for Simulator field removal

- Update Simulator construction in workload_config_test.go (removed fields)
- All tests pass

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 5: Update CLAUDE.md and Documentation

**Contracts Implemented:** None (documentation task)

**Files:**
- Modify: `CLAUDE.md` — update architecture description

**Step 1: Update CLAUDE.md**

In the "Core Simulation Engine (sim/)" section, add after the scheduler line:
```
- **latency_model.go**: `LatencyModel` interface (5 methods: StepTime, QueueingTime, OutputTokenProcessingTime, SchedulingProcessingTime, PreemptionProcessingTime), `BlackboxLatencyModel` (alpha/beta regression), `RooflineLatencyModel` (analytical FLOPs/bandwidth), `NewLatencyModel` factory
```

Update the `simulator.go` description to remove "step execution" and note the latencyModel field.

In the "Latency Estimation" section, update to mention the LatencyModel interface:
```
Two modes, selected by `NewLatencyModel()` factory based on `--model-config-folder` presence:
```

In the File Organization section, add `latency_model.go` entry.

**Step 2: Verify no stale references**

Run: Search for "getStepTime" or "runningBatchFeatures" in documentation to ensure no stale references.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for LatencyModel extraction

- Add latency_model.go to file organization and architecture sections
- Update Latency Estimation section to reference LatencyModel interface
- Remove references to hardcoded latency methods

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | TestBlackboxLatencyModel_StepTime_PrefillAndDecode |
| BC-1 | Task 1 | Unit | TestBlackboxLatencyModel_StepTime_EmptyBatch |
| BC-2 | Task 2 | Unit | TestRooflineLatencyModel_StepTime_DelegatesToRooflineStepTime |
| BC-2 | Task 2 | Unit | TestRooflineLatencyModel_StepTime_EmptyBatch |
| BC-3 | Task 1 | Unit | TestBlackboxLatencyModel_QueueingTime |
| BC-3 | Task 2 | Unit | TestRooflineLatencyModel_QueueingTime |
| BC-4 | Task 2 | Unit | TestNewLatencyModel_BlackboxMode |
| BC-4 | Task 2 | Unit | TestNewLatencyModel_RooflineMode |
| BC-5 | Task 3-4 | Golden+Invariant | Existing golden dataset tests + existing invariant tests |
| BC-6 | Task 3 | Structural (compile-time) | Removed fields cause compile errors if referenced |
| BC-7 | Task 3 | Structural (compile-time) | Removed RegressionFeatures causes compile error if referenced |
| BC-8 | Task 2 | Unit | TestNewLatencyModel_InvalidRoofline |

**Golden dataset:** No regeneration needed. Output is byte-identical since computation paths are equivalent.

**Invariant tests:** Existing invariant tests (request conservation, KV block conservation, determinism) serve as BC-5 regression guards. No new invariant tests needed since this is a pure refactoring with no behavioral change.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Feature extraction produces different values than incremental accumulation | Low | High | BC-1 test directly computes expected values; golden dataset test catches any drift | Task 1, 3 |
| `NumNewTokens` not set before `StepTime` called | Low | High | Code path guarantees: `makeRunningBatch()` runs before `StepTime()` in `Step()` | Task 3 |
| Test construction sites reference removed fields | Medium | Low | Compile-time error; fixed in Task 4 | Task 4 |
| Roofline StepConfig construction differs from original | Low | Medium | BC-2 test directly compares with `rooflineStepTime()` | Task 2 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — LatencyModel interface has exactly the methods currently called
- [x] No feature creep — Phase A only, no new backends
- [x] No unexercised flags or interfaces — both implementations used, all 5 methods called
- [x] No partial implementations — every method fully implemented
- [x] No breaking changes — pure refactoring, identical behavior
- [x] No hidden global state impact — no global state involved
- [x] All new code will pass golangci-lint
- [x] Shared test helpers: reuse `testModelConfig()` / `testHardwareCalib()` from roofline_step_test.go
- [x] CLAUDE.md updated in Task 5
- [x] No stale references — checked in Task 5
- [x] Deviation log reviewed — 3 deviations, all justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered: 1→2→3→4→5
- [x] All contracts mapped to tasks
- [x] Golden dataset: no regeneration needed
- [x] Construction site audit: SimConfig unchanged; Simulator has 1 construction site (NewSimulator)

**Antipattern rules:**
- [x] R1: No silent continue/return — no new error paths
- [x] R2: No map iteration for ordered output in new code
- [x] R3: No new CLI flags
- [x] R4: Simulator construction site (NewSimulator) updated in Task 3
- [x] R5: No new resource allocation loops
- [x] R6: No logrus.Fatalf in sim/ — factory returns error
- [x] R7: Existing golden tests already have companion invariant tests
- [x] R8: No exported mutable maps
- [x] R9: No YAML config in new code
- [x] R10: No YAML parsing in new code
- [x] R11: No division in new code (beta formula is multiply-and-add)
- [x] R12: No golden dataset change needed
- [x] R13: LatencyModel has 2 implementations (Blackbox + Roofline)
- [x] R14: Each method has single responsibility
- [x] R15: No stale PR references — this is a new PR
- [x] R16: Latency config params grouped in LatencyModel implementations
- [x] R17: N/A — no routing signals involved

---

## Appendix: File-Level Implementation Details

### File: `sim/latency_model.go`

**Purpose:** Defines the LatencyModel interface, BlackboxLatencyModel, RooflineLatencyModel, and NewLatencyModel factory.

See Tasks 1 and 2 for complete implementation code.

**Key implementation notes:**
- BlackboxLatencyModel.StepTime uses `req.ProgressIndex < Len64(req.InputTokens)` to distinguish prefill from decode — same logic as the original code in `makeRunningBatch()` and `getStepTimeRoofline()`
- RooflineLatencyModel.StepTime is nearly identical to the old `getStepTimeRoofline()` — the only difference is it receives `batch []*Request` instead of reading `sim.RunningBatch.Requests`
- QueueingTime and OutputTokenProcessingTime use identical alpha-coefficient formulas in both implementations — could be factored into a shared helper, but YAGNI (they're 2-line methods)
- Factory validates roofline config and returns error (R6: no Fatalf in library code)

### File: `sim/simulator.go` (modifications)

**Purpose:** Remove latency model fields/methods from Simulator, add `latencyModel` field, simplify Step() and makeRunningBatch().

**Removals:**
- Struct `RegressionFeatures` (lines 33-40)
- Fields: `runningBatchFeatures` (line 157), `betaCoeffs` (line 153), `alphaCoeffs` (line 154), `roofline` (line 168), `modelConfig` (line 170), `hwConfig` (line 171), `tp` (line 167)
- Methods: `getStepTime()` (lines 377-383), `getStepTimeRoofline()` (lines 386-406), `getQueueingTime()` (lines 350-355), `getOutputTokenProcessingTime()` (lines 358-361), `getSchedulingProcessingTime()` (lines 364-368), `getPreemptionProcessingTime()` (lines 371-374)

**Additions:**
- Field: `latencyModel LatencyModel` on Simulator struct

**Modifications in NewSimulator:**
- Remove roofline validation block (moved to NewLatencyModel)
- Add: `latencyModel, err := NewLatencyModel(cfg)` and `s.latencyModel = latencyModel`
- Remove: field initializations for removed fields

**Modifications in Step():**
- Remove: `sim.runningBatchFeatures = RegressionFeatures{...}` (line 581)
- Replace: `if sim.roofline { ... } else { ... }` with `currStepAdvance = sim.latencyModel.StepTime(sim.RunningBatch.Requests)`
- Replace 3 call sites of `sim.getOutputTokenProcessingTime()` with `sim.latencyModel.OutputTokenProcessingTime()`:
  - Line 627: decode ITL accumulation
  - Line 631: TTFT calculation
  - Line 659: completion ITL accumulation

**Modifications in makeRunningBatch():**
- Remove: all `sim.runningBatchFeatures.*` increment lines (10 lines in makeRunningBatch + 1 reset in Step)
- Replace: `sim.getSchedulingProcessingTime()` with `sim.latencyModel.SchedulingProcessingTime()`

**Modifications in preempt():**
- Replace: `sim.getPreemptionProcessingTime()` with `sim.latencyModel.PreemptionProcessingTime()`

### File: `sim/event.go` (modifications)

**Purpose:** Update QueueingTime call site.

**Line 31:** `sim.getQueueingTime(e.Request)` → `sim.latencyModel.QueueingTime(e.Request)`
