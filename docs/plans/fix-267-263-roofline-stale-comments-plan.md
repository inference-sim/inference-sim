# Fix: Remove Unused Roofline Parameter + Stale Comments Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean up the codebase by removing an unused function parameter and stale TODO comments that mislead contributors.

**The problem today:** The `rooflineStepTime` function accepts a `gpu string` parameter that is never used in the function body, confusing contributors who might assume GPU-specific branching exists. Additionally, `simulator.go` contains 4 stale `ToDo` comments that describe work already completed (scheduler reordering, metrics refinement, preemption loop, preemption handling), creating false impressions of missing functionality.

**What this PR adds:**
1. Removes the unused `gpu` parameter from `rooflineStepTime` — the function signature now reflects its actual dependencies (model config, hardware config, step config, tensor parallelism degree)
2. Removes 4 stale ToDo comments in `simulator.go` that describe already-implemented behavior, while preserving legitimate future-work ToDos

**Why this matters:** Code hygiene — unused parameters and stale comments are contributor friction. New contributors waste time investigating "missing" features that already exist.

**Architecture:** Pure refactoring within `sim/` package. Two files modified: `sim/roofline_step.go` (function signature), `sim/simulator.go` (call site + comments). Test file `sim/roofline_step_test.go` updated mechanically (remove `gpu` argument from 4 call sites).

**Source:** GitHub issues #267, #263

**Closes:** Fixes #267, fixes #263

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR is a pure cleanup — no behavioral changes, no new types, no interface modifications. It removes one unused function parameter (`gpu string` from `rooflineStepTime`) and 4 stale ToDo comments from `simulator.go`. The function's output is byte-identical before and after the parameter removal since the parameter was never referenced in the function body.

**Adjacent blocks:** `Simulator.getStepTimeRoofline()` (sole production call site), `roofline_step_test.go` (4 test call sites). No other packages reference `rooflineStepTime` (it's unexported).

**No DEVIATION flags** — the issues describe exactly what the code shows.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Roofline Output Preservation
- GIVEN a valid model config, hardware config, step config, and TP value
- WHEN `rooflineStepTime` is called (without the removed `gpu` parameter)
- THEN it MUST produce identical output to the previous version called with any `gpu` string
- MECHANISM: The `gpu` parameter was never referenced in the function body; removing it changes no computation.

BC-2: Stale Comment Removal Preserves Behavior
- GIVEN the simulator with stale ToDo comments removed
- WHEN the full test suite runs
- THEN all tests MUST pass with identical results
- MECHANISM: Comment removal has zero effect on compiled Go code.

BC-3: Legitimate ToDos Preserved
- GIVEN `simulator.go` after cleanup
- WHEN searching for remaining ToDo comments
- THEN the legitimate future-work ToDos (lines 327, 334, 502, 573) MUST still be present
- MECHANISM: Only the 4 stale comments identified in #263 are removed.

**Negative Contracts:**

BC-4: No Compilation Regression
- GIVEN the parameter removal in `rooflineStepTime`
- WHEN `go build ./...` runs
- THEN it MUST succeed with zero errors
- MECHANISM: All call sites (1 production, 4 test) are updated in the same task.

### C) Component Interaction

```
Simulator.getStepTimeRoofline() ──calls──▶ rooflineStepTime(modelConfig, hwConfig, stepConfig, tp)
                                                    ▲
                                                    │ (gpu parameter removed)
roofline_step_test.go ────────────calls─────────────┘
```

No new types, no new interfaces, no state changes. Pure signature simplification.

**Extension friction:** N/A — no new types or fields.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #263 mentions line 114 ToDo about "vLLM logic for reordering" | Remove it | InstanceScheduler.OrderQueue() (PR7) handles request ordering. The RunningBatch reordering is not a real concern in BLIS's DES model. |
| #263 mentions line 374 ToDo about "while true" loop | Remove it | The `for {}` loop on line 372 already implements this exact behavior. |
| #263 mentions line 528 ToDo about "pre-emption logic" | Remove it | Preemption is fully implemented at lines 370-400. |
| #263 mentions lines 116-117 ToDo about "metrics calculations" | Remove it | Metrics have been refined through PRs 1-13+. |

No deviations from source documents.

### E) Review Guide

1. **THE TRICKY PART:** Nothing tricky — this is mechanical cleanup. The only risk is missing a call site for the parameter removal.
2. **WHAT TO SCRUTINIZE:** Verify all 4 stale ToDos are genuinely stale (check the referenced behavior exists). Verify the 4 preserved ToDos are genuinely future-work.
3. **WHAT'S SAFE TO SKIM:** The roofline function body (unchanged), test assertions (unchanged).
4. **KNOWN DEBT:** The remaining ToDos at lines 327, 334, 502, 573 are legitimate future-work items (scheduling/preemption time modeling, cache block management).

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/roofline_step.go:131` — remove `gpu string` from function signature
- `sim/simulator.go:366` — remove `sim.gpu` argument from call site
- `sim/simulator.go:114,116-117,374,528` — remove 4 stale ToDo comments
- `sim/roofline_step_test.go:207,208,259,273` — remove `"H100"` argument from 4 test call sites

**Key decisions:**
- Remove ONLY the 4 stale ToDos listed in #263. Do not touch other ToDos.
- The `gpu` field remains in `Simulator` struct and `SimConfig` — it may be used elsewhere or in future work. We only remove it from `rooflineStepTime`'s parameter list.

**Confirmation:** No dead code introduced. All changes are removals of existing dead code/comments.

### G) Task Breakdown

---

### Task 1: Remove unused `gpu` parameter from `rooflineStepTime`

**Contracts Implemented:** BC-1, BC-4

**Files:**
- Modify: `sim/roofline_step.go:131` (function signature)
- Modify: `sim/roofline_step.go:128-130` (doc comment)
- Modify: `sim/simulator.go:366` (call site)
- Modify: `sim/roofline_step_test.go:207,208,259,273` (test call sites)

**Step 1: Update function signature and doc comment**

Context: Remove the unused `gpu string` parameter from `rooflineStepTime`. The parameter is the first positional argument.

In `sim/roofline_step.go`, change line 131 from:
```go
func rooflineStepTime(gpu string, modelConfig ModelConfig, hwConfig HardwareCalib, stepConfig StepConfig, tp int) int64 {
```
to:
```go
func rooflineStepTime(modelConfig ModelConfig, hwConfig HardwareCalib, stepConfig StepConfig, tp int) int64 {
```

**Step 2: Update production call site**

In `sim/simulator.go`, change line 366 from:
```go
	stepTime := rooflineStepTime(sim.gpu, sim.modelConfig, sim.hwConfig, stepConfig, sim.tp)
```
to:
```go
	stepTime := rooflineStepTime(sim.modelConfig, sim.hwConfig, stepConfig, sim.tp)
```

**Step 3: Update test call sites**

In `sim/roofline_step_test.go`, update all 4 call sites:

Line 207: `rooflineStepTime("H100", mc, hc, step, 1)` → `rooflineStepTime(mc, hc, step, 1)`
Line 208: `rooflineStepTime("H100", mc, hc, step, 2)` → `rooflineStepTime(mc, hc, step, 2)`
Line 259: `rooflineStepTime("H100", mc, hc, tt.step, 1)` → `rooflineStepTime(mc, hc, tt.step, 1)`
Line 273: `rooflineStepTime("H100", mc, hc, step, 1)` → `rooflineStepTime(mc, hc, step, 1)`

**Step 4: Build and run tests**

Run: `go build ./... && go test ./sim/... -count=1 -v -run "TestRoofline"`
Expected: BUILD SUCCESS, all roofline tests PASS

**Step 5: Run lint**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/roofline_step.go sim/simulator.go sim/roofline_step_test.go
git commit -m "refactor(roofline): remove unused gpu parameter from rooflineStepTime (#267, BC-1, BC-4)

The gpu string parameter was never referenced in the function body.
Remove it from the signature and all 5 call sites (1 production, 4 test).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Remove stale ToDo comments from simulator.go

**Contracts Implemented:** BC-2, BC-3

**Files:**
- Modify: `sim/simulator.go:114,116-117,374,528`

**Step 1: Remove stale ToDo at line 114**

Context: `// ToDo: Add vLLM logic for reordering requests in RunningBatch before model execution` — this is stale because `InstanceScheduler.OrderQueue()` (added in PR7) handles request ordering before batch formation.

In `sim/simulator.go`, remove line 114:
```go
	// ToDo: Add vLLM logic for reordering requests in RunningBatch before model execution
```

**Step 2: Remove stale ToDo at lines 116-117**

Context: `// ToDo: We have a data structure, but this is where we need to make metrics calculations accurate` — stale because metrics have been significantly refined through PRs 1-13+.

In `sim/simulator.go`, remove lines 116-117:
```go
	// ToDo: We have a data structure, but this is where we need to
	// make metrics calculations accurate
```

**Step 3: Remove stale ToDo at line 374**

Context: `// ToDo: add while true here, because we will keep preempting until we are good` — stale because the `for {}` infinite loop on line 372 already implements exactly this behavior.

In `sim/simulator.go`, remove line 374:
```go
			// ToDo: add while true here, because we will keep preempting until we are good
```

**Step 4: Remove stale ToDo at line 528**

Context: `// ToDo: Understand and handle pre-emption logic, if need be.` — stale because preemption is fully implemented at lines 370-400.

In `sim/simulator.go`, change the comment on the `Step` function. The current comment block is:
```go
// In vllm, the processing of requests proceeds iteratively in steps.
// Step simulates a single vllm step(), which roughly corresponds to a single scheduler.schedule()
// to construct a batch, model execution of the batch and scheduler.update().
// ToDo: Understand and handle pre-emption logic, if need be.
func (sim *Simulator) Step(now int64) {
```

Remove the `// ToDo:` line, keeping the other comment lines:
```go
// In vllm, the processing of requests proceeds iteratively in steps.
// Step simulates a single vllm step(), which roughly corresponds to a single scheduler.schedule()
// to construct a batch, model execution of the batch and scheduler.update().
func (sim *Simulator) Step(now int64) {
```

**Step 5: Verify legitimate ToDos still present**

Run: `grep -n "ToDo\|TODO" sim/simulator.go`

Expected output should show the 4 preserved legitimate ToDos at approximately:
- ~line 325: `// ToDo: incorporate some alphas here or constant?` (getSchedulingProcessingTime)
- ~line 332: `// ToDo: incorporate some alphas here or maybe constat` (getPreemptionProcessingTime)
- ~line 500: `// ToDo: there are some minor processing time above` (scheduling delay)
- ~line 571: `// ToDo: Go through the newly allocated blocks` (cache optimization)

**Step 6: Build and run full test suite**

Run: `go build ./... && go test ./... -count=1`
Expected: BUILD SUCCESS, ALL tests PASS

**Step 7: Run lint**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 8: Commit**

```bash
git add sim/simulator.go
git commit -m "fix(sim): remove 4 stale ToDo comments in simulator.go (#263, BC-2, BC-3)

Remove stale comments that describe already-implemented behavior:
- Line 114: scheduler reordering (implemented by InstanceScheduler.OrderQueue)
- Lines 116-117: metrics accuracy (refined through PRs 1-13+)
- Line 374: preemption loop (for{} already implements this)
- Line 528: preemption handling (fully implemented at lines 370-400)

Preserve 4 legitimate future-work ToDos (scheduling time, preemption time,
scheduling delay modeling, cache block management).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|--------------------------|
| BC-1     | Task 1 | Existing | `TestRooflineStepTime_*` — all 3 existing tests verify identical behavior after param removal |
| BC-2     | Task 2 | Existing | Full `go test ./...` — comment removal cannot change behavior |
| BC-3     | Task 2 | Manual | `grep` verification that legitimate ToDos survive |
| BC-4     | Task 1 | Build | `go build ./...` succeeds |

No golden dataset changes. No new tests needed — this is pure cleanup with no behavioral change.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Miss a call site for `gpu` param removal | Low | Low (build fails immediately) | Grep for all call sites — only 5 exist (1 production + 4 test) | Task 1 |
| Remove a legitimate ToDo | Low | Low (can be re-added) | Issue #263 specifically identifies the 4 stale ones; verify each against implemented code | Task 2 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — pure removal
- [x] No feature creep beyond PR scope — only #267 and #263
- [x] No unexercised flags or interfaces — N/A (removal only)
- [x] No partial implementations — complete
- [x] No breaking changes — unexported function, internal refactoring
- [x] No hidden global state impact — N/A
- [x] All new code will pass golangci-lint — no new code
- [x] Shared test helpers — N/A (no new tests)
- [x] CLAUDE.md — no update needed (no new files/packages/flags)
- [x] No stale references in CLAUDE.md — verified
- [x] Deviation log reviewed — no deviations
- [x] Each task produces working, testable code — yes
- [x] Task dependencies correctly ordered — Task 1 and 2 are independent
- [x] All contracts mapped to tasks — BC-1,4→Task1; BC-2,3→Task2
- [x] Golden dataset regeneration — not needed
- [x] Construction site audit — no fields added
- [x] No new CLI flags
- [x] No new error paths
- [x] No map iteration changes
- [x] No logrus.Fatalf in library code
- [x] No resource allocation loops
- [x] No exported mutable maps
- [x] No YAML config changes
- [x] No division operations added
- [x] No new interfaces
- [x] No multi-concern methods
- [x] No config parameter additions
- [x] Grepped for stale PR references — N/A (no PR number references to this fix)
- [x] Not part of a macro plan — standalone issue fix

---

## Appendix: File-Level Implementation Details

### File: `sim/roofline_step.go`

**Purpose:** Remove unused `gpu string` parameter from function signature.

**Change:** Lines 128-131 only.

Before:
```go
// rooflineStepTime computes step latency using the roofline model.
// Precondition: ValidateRooflineConfig(modelConfig, hwConfig) must return nil
// and tp must be > 0. Callers must validate before first call.
func rooflineStepTime(gpu string, modelConfig ModelConfig, hwConfig HardwareCalib, stepConfig StepConfig, tp int) int64 {
```

After:
```go
// rooflineStepTime computes step latency using the roofline model.
// Precondition: ValidateRooflineConfig(modelConfig, hwConfig) must return nil
// and tp must be > 0. Callers must validate before first call.
func rooflineStepTime(modelConfig ModelConfig, hwConfig HardwareCalib, stepConfig StepConfig, tp int) int64 {
```

No other changes in this file. Function body is untouched.

### File: `sim/simulator.go`

**Purpose:** (1) Update `rooflineStepTime` call site. (2) Remove 4 stale ToDo comments.

**Call site change (line 366):**
```go
// Before:
stepTime := rooflineStepTime(sim.gpu, sim.modelConfig, sim.hwConfig, stepConfig, sim.tp)
// After:
stepTime := rooflineStepTime(sim.modelConfig, sim.hwConfig, stepConfig, sim.tp)
```

**Stale comment removals:**
- Line 114: Remove `// ToDo: Add vLLM logic for reordering requests in RunningBatch before model execution`
- Lines 116-117: Remove `// ToDo: We have a data structure, but this is where we need to` / `// make metrics calculations accurate`
- Line 374: Remove `// ToDo: add while true here, because we will keep preempting until we are good`
- Line 528: Remove `// ToDo: Understand and handle pre-emption logic, if need be.`

**Preserved ToDos (DO NOT REMOVE):**
- ~Line 327: `// ToDo: incorporate some alphas here or constant?` (getSchedulingProcessingTime)
- ~Line 334: `// ToDo: incorporate some alphas here or maybe constat` (getPreemptionProcessingTime)
- ~Line 502: `scheduledDelay := sim.getSchedulingProcessingTime() // ToDo: there are some minor processing time above - model it or constant?`
- ~Line 573: `// ToDo: Go through the newly allocated blocks for this request;`

### File: `sim/roofline_step_test.go`

**Purpose:** Remove `"H100"` first argument from 4 `rooflineStepTime` call sites.

**Changes (mechanical):**
- Line 207: `rooflineStepTime("H100", mc, hc, step, 1)` → `rooflineStepTime(mc, hc, step, 1)`
- Line 208: `rooflineStepTime("H100", mc, hc, step, 2)` → `rooflineStepTime(mc, hc, step, 2)`
- Line 259: `rooflineStepTime("H100", mc, hc, tt.step, 1)` → `rooflineStepTime(mc, hc, tt.step, 1)`
- Line 273: `rooflineStepTime("H100", mc, hc, step, 1)` → `rooflineStepTime(mc, hc, step, 1)`

No test logic changes. Assertions unchanged.
