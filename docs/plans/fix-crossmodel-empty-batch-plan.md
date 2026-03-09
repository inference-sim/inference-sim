# Fix crossmodel StepTime empty-batch livelock Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix a livelock where the crossmodel latency backend returns 0 for empty batches, preventing simulation clock advancement.

**The problem today:** When all requests are preempted from the running batch and the front-of-queue request cannot allocate KV blocks, the simulator schedules a new StepEvent. The crossmodel backend returns 0 for this empty batch, creating an infinite loop at the same timestamp — the simulation hangs indefinitely. This violates INV-3 (clock monotonicity) and makes crossmodel mode unusable under KV pressure scenarios.

**What this PR adds:**
1. **Crossmodel empty-batch fix** — crossmodel `StepTime` returns the architecture-dependent base overhead for empty batches (same pattern as blackbox), guaranteeing ≥ 1
2. **Defensive floor in simulator** — `executeBatchStep` floors `currStepAdvance` at 1 tick, protecting against any future backend returning 0
3. **Interface contract documentation** — `LatencyModel.StepTime()` doc comment specifies the ≥ 1 postcondition

**Why this matters:** This unblocks crossmodel mode for KV-pressure workloads (the primary use case for capacity planning with MoE models), and establishes a safety net against future livelock regressions from new latency backends.

**Architecture:** Three small edits across `sim/latency/crossmodel.go` (remove early return), `sim/simulator.go` (add floor), and `sim/latency_model.go` (document contract). Tests updated in `sim/latency/crossmodel_test.go` and `sim/latency/latency_test.go`.

**Source:** GitHub issue #569

**Closes:** Fixes #569

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes a livelock in the crossmodel latency backend when `StepTime` is called with an empty batch. Today, crossmodel returns 0 for empty batches (line 40-42 of `crossmodel.go`), while blackbox returns ~6910 (its beta0 intercept) and roofline returns ≥ 1 (explicit `max(1, ...)`). When the simulator schedules a next step at `now + 0`, it fires immediately at the same timestamp — infinite loop.

The fix removes the `return 0` early exit in crossmodel (letting the formula compute the base overhead, floored by the existing `max(1, ...)`), adds a defense-in-depth floor in `simulator.go`, and documents the interface contract. An existing test `TestCrossModelLatencyModel_StepTime_EmptyBatch_Zero` encodes the buggy behavior and must be updated.

No deviations from the issue description.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: StepTime minimum floor (crossmodel)
- GIVEN a CrossModelLatencyModel with valid coefficients
- WHEN `StepTime` is called with an empty batch
- THEN the return value MUST be ≥ 1
- MECHANISM: Remove the `if len(batch) == 0 { return 0 }` early return; the formula computes `β₀·numLayers + β₃·isTP` (token-dependent terms are 0), and the existing `max(1, int64(stepTime))` at line 59 floors it.

BC-2: StepTime minimum floor (all backends, defense-in-depth)
- GIVEN any LatencyModel implementation
- WHEN `executeBatchStep` computes `currStepAdvance`
- THEN the returned value MUST be ≥ 1
- MECHANISM: `currStepAdvance = max(1, currStepAdvance)` added after StepTime + transfer latency computation in `executeBatchStep`.

BC-3: Interface contract documentation
- GIVEN the `LatencyModel` interface definition
- WHEN a developer reads the `StepTime` doc comment
- THEN the postcondition (return value ≥ 1 for all inputs including empty batch) MUST be documented.

**Negative Contracts:**

BC-4: No clock stall on empty batch
- GIVEN a simulator using crossmodel latency backend
- WHEN `Step()` executes with an empty running batch and non-empty WaitQ
- THEN the next StepEvent MUST be scheduled at a timestamp strictly greater than `now`
- MECHANISM: BC-1 + BC-2 jointly guarantee `currStepAdvance ≥ 1`, so `now + currStepAdvance > now`.

**Backward Compatibility:**

BC-5: Existing non-empty-batch behavior unchanged
- GIVEN any LatencyModel implementation
- WHEN `StepTime` is called with a non-empty batch
- THEN the return value MUST be identical to the pre-fix value
- MECHANISM: The crossmodel formula path for non-empty batches is untouched; the defensive floor in `executeBatchStep` is a no-op when StepTime already returns ≥ 1 (which all backends do for non-empty batches).

### C) Component Interaction

```
LatencyModel.StepTime(batch)  ←── crossmodel fix: return ≥ 1 for empty batch
        │
        ▼
executeBatchStep(now)          ←── defense floor: max(1, currStepAdvance)
        │
        ▼
scheduleNextStep(now, advance) ──→ StepEvent{time: now + advance}  (advance ≥ 1, INV-3 preserved)
```

No new interfaces, types, or state. No extension friction change.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Affected file: `sim/latency/latency.go` for interface contract documentation | Targets `sim/latency_model.go` instead | CORRECTION: The `LatencyModel` interface is defined in `sim/latency_model.go` (package `sim`), not `sim/latency/latency.go` (package `latency`, which contains implementations). Issue #569 listed the wrong file path. |

**Behavioral magnitude note:** Removing the `return 0` early exit changes crossmodel empty-batch return value from 0 to `max(1, int64(β₀·numLayers + β₃·isTP))`. With the test's coefficients (β₀=116.110, numLayers=32, isTP=0.0): ~3715 ticks. With production llama-3.1-8b/H100/TP=2 coefficients (isTP=1.0): ~13160 ticks. This matches blackbox's pattern (returns β₀ ~6910 for empty batches) — both represent the architecture-dependent base overhead when no tokens are processed.

### E) Review Guide

1. **THE TRICKY PART:** The crossmodel formula for empty batches computes `β₀·numLayers + β₃·isTP`. For a config with `isTP=0.0` and small `numLayers`, the raw value could truncate to 0 in `int64()`. The existing `max(1, ...)` at line 59 handles this, but verify it's actually reached (no other early returns).
2. **WHAT TO SCRUTINIZE:** BC-1 — trace the empty-batch path through the modified crossmodel code to confirm `max(1, ...)` is reached.
3. **WHAT'S SAFE TO SKIM:** BC-3 (doc comment update) and BC-5 (non-empty paths are untouched).
4. **KNOWN DEBT:** The blackbox (`latency_test.go:62`) and roofline (`latency_test.go:191`) empty-batch tests assert `>= 0`, not `>= 1`. This PR tightens them to `>= 1` to match the interface contract.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/latency/crossmodel.go` — Remove empty-batch early return (2 lines deleted)
- `sim/latency/crossmodel_test.go` — Update empty-batch test to assert ≥ 1 instead of == 0
- `sim/latency/latency_test.go` — Tighten blackbox and roofline empty-batch tests to assert ≥ 1
- `sim/simulator.go` — Add defensive floor in `executeBatchStep`
- `sim/latency_model.go` — Update `StepTime` doc comment with postcondition

**Key decisions:**
- Remove early return rather than changing it to `return 1` — the formula gives a meaningful architecture-dependent overhead
- Defensive floor in `executeBatchStep` rather than `scheduleNextStep` — catches the issue at the source before the value propagates

**Confirmation:** No dead code, all paths exercisable immediately.

### G) Task Breakdown

---

### Task 1: Fix crossmodel empty-batch StepTime and update tests

**Contracts Implemented:** BC-1, BC-4, BC-5

**Files:**
- Modify: `sim/latency/crossmodel.go:39-42`
- Modify: `sim/latency/crossmodel_test.go:53-65`
- Modify: `sim/latency/latency_test.go:53-65,180-193`

**Step 1: Write failing test for BC-1 (update existing test)**

Context: The existing test `TestCrossModelLatencyModel_StepTime_EmptyBatch_Zero` asserts the buggy `== 0` behavior. Update it to assert the correct contract: empty batch returns ≥ 1.

In `sim/latency/crossmodel_test.go`, replace the test:
```go
func TestCrossModelLatencyModel_StepTime_EmptyBatch_FloorAtOne(t *testing.T) {
	// BC-1: empty batch → StepTime ≥ 1 (clock must always advance)
	m := &CrossModelLatencyModel{
		betaCoeffs:  []float64{116.110, 1226.868, 19.943, 9445.157},
		alphaCoeffs: []float64{13732.0, 0.0, 860.6},
		numLayers:   32,
		kvDimScaled: 0.001,
		isMoE:       0.0,
		isTP:        0.0,
	}
	result := m.StepTime([]*sim.Request{})
	assert.GreaterOrEqual(t, result, int64(1), "BC-1: empty batch must return >= 1 to guarantee clock advancement")

	// Regression anchor: empty batch computes β₀·numLayers + β₃·isTP = 116.110*32 + 9445.157*0.0 = 3715.52 → int64(3715)
	assert.Equal(t, int64(3715), result, "empty batch regression anchor: β₀·numLayers overhead")
}
```

Also tighten the blackbox and roofline empty-batch tests in `sim/latency/latency_test.go`:

For `TestBlackboxLatencyModel_StepTime_EmptyBatch` (line 53), replace `result < 0` check with:
```go
	// THEN empty batch produces StepTime >= 1 (interface contract: clock must advance)
	assert.GreaterOrEqual(t, result, int64(1), "empty batch must return >= 1 per LatencyModel contract")
```

For `TestRooflineLatencyModel_StepTime_EmptyBatch` (line 180), replace `emptyResult < 0` check with:
```go
	// THEN empty batch result must be >= 1 (interface contract: clock must advance)
	assert.GreaterOrEqual(t, emptyResult, int64(1), "empty batch must return >= 1 per LatencyModel contract")
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/latency/... -run "TestCrossModelLatencyModel_StepTime_EmptyBatch_FloorAtOne" -v`
Expected: FAIL (crossmodel still returns 0)

**Step 3: Implement fix — remove early return in crossmodel**

Context: Delete the `if len(batch) == 0 { return 0 }` early return. The formula then computes `β₀·numLayers + 0 + 0 + β₃·isTP`, and the existing `max(1, int64(stepTime))` at line 59 floors the result.

In `sim/latency/crossmodel.go`, replace:
```go
func (m *CrossModelLatencyModel) StepTime(batch []*sim.Request) int64 {
	if len(batch) == 0 {
		return 0
	}
	var totalPrefillTokens, totalDecodeTokens int64
```
with:
```go
func (m *CrossModelLatencyModel) StepTime(batch []*sim.Request) int64 {
	var totalPrefillTokens, totalDecodeTokens int64
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/latency/... -run "TestCrossModelLatencyModel_StepTime_EmptyBatch|TestBlackboxLatencyModel_StepTime_EmptyBatch|TestRooflineLatencyModel_StepTime_EmptyBatch" -v`
Expected: PASS (all three backends return ≥ 1 for empty batch)

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/latency/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/latency/crossmodel.go sim/latency/crossmodel_test.go sim/latency/latency_test.go
git commit -m "fix(latency): crossmodel StepTime returns >= 1 for empty batch (BC-1)

- Remove early return 0 for empty batch in CrossModelLatencyModel.StepTime
- Formula computes base overhead (β₀·numLayers + β₃·isTP), existing max(1,...) floors it
- Update crossmodel empty-batch test from == 0 to >= 1
- Tighten blackbox and roofline empty-batch tests from >= 0 to >= 1

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Add defensive floor in simulator and document interface contract

**Contracts Implemented:** BC-2, BC-3

**Files:**
- Modify: `sim/simulator.go:352-357`
- Modify: `sim/latency_model.go:8-10`

**Step 1: Write failing test for BC-2**

Context: We need a test that verifies the simulator never produces a zero step advance, even if a hypothetical latency model returns 0. Since `executeBatchStep` is private, we test via the public `Step()` method by checking that scheduled events advance the clock. However, the existing test infrastructure already covers this implicitly through the crossmodel fix in Task 1. Instead, we add an explicit unit-level test.

In `sim/latency/latency_test.go`, add a cross-backend empty-batch floor test:
```go
func TestAllBackends_StepTime_EmptyBatch_FloorAtOne(t *testing.T) {
	// BC-2: all backends must return >= 1 for empty batch (interface contract)
	emptyBatch := []*sim.Request{}

	blackbox := &BlackboxLatencyModel{
		betaCoeffs:  []float64{0, 0, 0}, // zero coefficients — worst case
		alphaCoeffs: []float64{0, 0, 0},
	}
	assert.GreaterOrEqual(t, blackbox.StepTime(emptyBatch), int64(1),
		"blackbox with zero coefficients must still return >= 1")

	crossmodel := &CrossModelLatencyModel{
		betaCoeffs:  []float64{0, 0, 0, 0}, // zero coefficients — worst case
		alphaCoeffs: []float64{0, 0, 0},
		numLayers:   1,
		kvDimScaled: 0.0,
		isMoE:       0.0,
		isTP:        0.0,
	}
	assert.GreaterOrEqual(t, crossmodel.StepTime(emptyBatch), int64(1),
		"crossmodel with zero coefficients must still return >= 1")
}
```

Note: Roofline is tested in the existing `TestRooflineLatencyModel_StepTime_EmptyBatch` (already tightened in Task 1). The roofline model requires model/hardware config structs for construction, so it's tested separately.

**Step 2: Run test to verify it passes**

Run: `go test ./sim/latency/... -run "TestAllBackends_StepTime_EmptyBatch_FloorAtOne" -v`
Expected: PASS (both backends already return >= 1 after Task 1 fix)

**Step 3: Implement defensive floor in simulator + update interface doc**

Context: Add `max(1, ...)` floor in `executeBatchStep` as defense-in-depth (catches any future backend that returns 0). Update the `StepTime` doc comment with the postcondition.

In `sim/simulator.go`, after line 357 (`currStepAdvance += sim.KVCache.ConsumePendingTransferLatency()`), add:
```go
	// INV-3 defense-in-depth: guarantee clock advancement regardless of backend.
	// All LatencyModel implementations must return >= 1 per interface contract;
	// this floor catches violations that would cause infinite livelock.
	currStepAdvance = max(1, currStepAdvance)
```

In `sim/latency_model.go`, update the `StepTime` doc comment:
```go
	// StepTime estimates the duration of one batch step given the running batch.
	// Precondition: each request in batch has NumNewTokens set by BatchFormation.FormBatch().
	// Postcondition: return value >= 1 for all inputs (including empty batch).
	// A return value of 0 would stall the simulation clock, violating INV-3 (clock monotonicity).
	StepTime(batch []*Request) int64
```

**Step 4: Run full test suite**

Run: `go test ./sim/... -count=1`
Expected: All tests pass

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/simulator.go sim/latency_model.go sim/latency/latency_test.go
git commit -m "fix(sim): add defensive StepTime floor in executeBatchStep (BC-2, BC-3)

- Add max(1, currStepAdvance) in executeBatchStep as defense-in-depth
- Document StepTime postcondition (>= 1) in LatencyModel interface
- Add cross-backend zero-coefficient empty-batch test

Fixes #569

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | `TestCrossModelLatencyModel_StepTime_EmptyBatch_FloorAtOne` |
| BC-2 | Task 2 | Unit | `TestAllBackends_StepTime_EmptyBatch_FloorAtOne` |
| BC-4 | Task 1 | Implicit | Empty batch returning ≥ 1 prevents clock stall |
| BC-5 | Task 1 | Regression | All existing non-empty-batch tests pass unchanged |

Golden dataset update: Not needed — this fix changes empty-batch behavior only, which doesn't occur in golden dataset test scenarios (golden tests use non-empty batches with actual requests).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Crossmodel empty-batch overhead value changes existing simulation results | Low | Low | Empty batches only occur during KV pressure preemption cascades; the step time value affects recovery timing, not steady-state results. Golden dataset unaffected. |
| Future backend returns negative value | Low | Medium | Defensive `max(1, ...)` floor in simulator catches both 0 and negative values |

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
- [x] Shared test helpers: uses existing test infrastructure
- [x] CLAUDE.md: no update needed (no new files/packages/CLI flags)
- [x] No stale references in CLAUDE.md
- [x] Documentation DRY: only sim/latency_model.go doc comment updated (no working copies elsewhere)
- [x] Deviation log reviewed — no deviations
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Task 2 depends on Task 1's crossmodel fix)
- [x] All contracts mapped to tasks
- [x] Golden dataset regeneration: not needed
- [x] Construction site audit: no struct fields added

**Antipattern rules:**
- [x] R1: No silent data loss — no new error paths
- [x] R3: No new numeric parameters
- [x] R4: No struct field additions
- [x] R6: No logrus.Fatalf in sim/
- [x] R7: No golden tests added; invariant-style assertions used
- [x] R11: No new division
- [x] R19: This PR fixes an unbounded loop (the livelock itself)

---

## Appendix: File-Level Implementation Details

### File: `sim/latency/crossmodel.go`

**Purpose:** Remove the empty-batch early return that causes livelock.

**Change:** Delete lines 40-42 (`if len(batch) == 0 { return 0 }`). The remaining code computes `β₀·numLayers + β₃·isTP` for empty batches (token-dependent terms are 0) and the existing `max(1, int64(stepTime))` at line 59 provides the floor.

### File: `sim/simulator.go`

**Purpose:** Add defense-in-depth floor in `executeBatchStep`.

**Change:** After line 357 (`currStepAdvance += sim.KVCache.ConsumePendingTransferLatency()`), insert `currStepAdvance = max(1, currStepAdvance)` with an INV-3 reference comment.

### File: `sim/latency_model.go`

**Purpose:** Document the StepTime postcondition.

**Change:** Expand `StepTime` doc comment to include: "Postcondition: return value >= 1 for all inputs (including empty batch). A return value of 0 would stall the simulation clock, violating INV-3 (clock monotonicity)."

### File: `sim/latency/crossmodel_test.go`

**Purpose:** Update the existing empty-batch test from asserting `== 0` to `>= 1`.

**Change:** Rename `TestCrossModelLatencyModel_StepTime_EmptyBatch_Zero` to `TestCrossModelLatencyModel_StepTime_EmptyBatch_FloorAtOne`. Update assertion from `assert.Equal(t, int64(0), result)` to `assert.GreaterOrEqual(t, result, int64(1))` plus regression anchor `assert.Equal(t, int64(3715), result)`. Update comment from "BC-5: empty batch → 0" to "BC-1: empty batch → StepTime ≥ 1". The regression anchor verifies: `β₀·numLayers + β₃·isTP = 116.110*32 + 9445.157*0.0 = 3715.52 → int64(3715)`.

### File: `sim/latency/latency_test.go`

**Purpose:** Tighten existing empty-batch tests and add cross-backend zero-coefficient test.

**Changes:**
1. `TestBlackboxLatencyModel_StepTime_EmptyBatch`: Replace `result < 0` with `assert.GreaterOrEqual(t, result, int64(1))`
2. `TestRooflineLatencyModel_StepTime_EmptyBatch`: Replace `emptyResult < 0` with `assert.GreaterOrEqual(t, emptyResult, int64(1))`
3. Add `TestAllBackends_StepTime_EmptyBatch_FloorAtOne`: Tests blackbox and crossmodel with all-zero coefficients to verify the floor holds even in worst case
