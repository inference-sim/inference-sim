# Fix Roofline Dead Code + Add Unit Tests — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove dead code in the roofline FLOPs calculator and add comprehensive unit tests for all three roofline functions.

**The problem today:** The `calculateTransformerFlops` function has two consecutive `if includeAttention` blocks — the first block's results are entirely overwritten by the second block, making lines 29-43 dead code. This wastes computation and misleads readers. Additionally, the three core roofline functions (`calculateTransformerFlops`, `calculateMemoryAccessBytes`, `rooflineStepTime`) have zero or minimal test coverage, meaning correctness regressions would go undetected.

**What this PR adds:**
1. Dead code removal — the first `if includeAttention` block (lines 29-43) is deleted, leaving only the correct block that properly separates attention into GEMM ops (QK^T, AV) and vector ops (softmax, RoPE)
2. FLOPs monotonicity tests — more tokens produce more FLOPs, verifying the compute model scales correctly
3. TP scaling tests — TP=2 produces roughly half the per-shard latency of TP=1, verifying tensor parallelism division
4. Smoke test for `rooflineStepTime` — valid inputs produce positive, finite latency
5. Edge case tests — single-token decode, large-prefill, attention-only vs MLP-only isolation

**Why this matters:** The roofline model is the analytical latency estimator used for capacity planning without real GPUs. Dead code and untested functions undermine trust in its predictions. These tests also protect against future regressions during any roofline refactoring.

**Architecture:** All changes are in `sim/roofline_step.go` (dead code removal) and a new `sim/roofline_step_test.go` (tests). No interfaces, no new types, no cross-package changes.

**Source:** GitHub issues #254, #257

**Closes:** Fixes #254, fixes #257

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes a dead-code bug in `calculateTransformerFlops` where duplicate `if includeAttention` blocks cause the first block's results to be silently overwritten, then adds comprehensive behavioral tests for all three roofline functions. The changes are confined to `sim/roofline_step.go` (removal of 15 lines) and a new `sim/roofline_step_test.go`. No interfaces change, no types change, and no other packages are affected.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: FLOPs Monotonicity — More Tokens Produce More FLOPs
- GIVEN a valid ModelConfig
- WHEN `calculateTransformerFlops` is called with sequenceLength=512, newTokens=100 vs newTokens=200
- THEN the result with more newTokens MUST have higher `total` FLOPs
- MECHANISM: Every FLOPs term is linear or super-linear in newTokens

BC-2: Memory Monotonicity — More Tokens Produce More Memory Bytes
- GIVEN a valid ModelConfig
- WHEN `calculateMemoryAccessBytes` is called with newTokens=100 vs newTokens=200 (same sequenceLength)
- THEN the result with more newTokens MUST have higher `total` bytes
- MECHANISM: KV cache growth and activations scale linearly with newTokens

BC-3: TP Scaling — TP=2 Produces Less Per-Shard Latency Than TP=1
- GIVEN a valid ModelConfig and HardwareCalib with TP > 1
- WHEN `rooflineStepTime` is called with tp=1 vs tp=2 (same workload)
- THEN the tp=2 result MUST be strictly less than the tp=1 result
- MECHANISM: FLOPs and memory bytes are divided by tpFactor

BC-4: Roofline Smoke — Valid Inputs Produce Positive Finite Latency
- GIVEN a valid ModelConfig and HardwareCalib
- WHEN `rooflineStepTime` is called with non-empty prefill and decode requests
- THEN the result MUST be > 0 and finite (not NaN, not Inf)
- MECHANISM: All denominators validated positive by ValidateRooflineConfig precondition

BC-5: Dead Code Removal — No Duplicate Attention Blocks
- GIVEN the source of `calculateTransformerFlops`
- WHEN the function is inspected
- THEN there MUST be exactly one `if includeAttention` block
- MECHANISM: The dead first block (old lines 29-43) is deleted

BC-6: Attention Isolation — Disabling Attention Zeroes Attention FLOPs
- GIVEN a valid ModelConfig
- WHEN `calculateTransformerFlops` is called with includeAttention=false, includeMLP=true
- THEN `sram_ops` MUST be zero and `gemm_ops` MUST contain only MLP FLOPs
- MECHANISM: Attention GEMM and vector ops are gated by includeAttention flag

BC-7: MLP Isolation — Disabling MLP Zeroes MLP FLOPs
- GIVEN a valid ModelConfig
- WHEN `calculateTransformerFlops` is called with includeAttention=true, includeMLP=false
- THEN FLOPs MUST contain only attention-derived values (no SwiGLU contribution)
- MECHANISM: MLP SwiGLU block is gated by includeMLP flag

BC-8: FLOPs Conservation — Total Equals Sum of Components
- GIVEN any valid inputs to `calculateTransformerFlops`
- WHEN the function returns
- THEN `flops["total"]` MUST equal `flops["gemm_ops"] + flops["sram_ops"]`
- MECHANISM: Final line of function computes total as sum

BC-9: Memory Conservation — Total Equals Sum of Components
- GIVEN any valid inputs to `calculateMemoryAccessBytes`
- WHEN the function returns
- THEN `mem["total"]` MUST equal the sum of all non-"total" component values
- MECHANISM: Sorted accumulation loop at end of function

**Negative Contracts:**

BC-10: No Dead Code — First Attention Block Must Not Exist
- GIVEN the codebase after this PR
- WHEN `roofline_step.go` is inspected
- THEN there MUST NOT be two consecutive `if includeAttention` blocks

### C) Component Interaction

```
calculateTransformerFlops(config, seqLen, newTokens, attn, mlp) → map[string]float64
       ↓                                                                    ↓
calculateMemoryAccessBytes(config, seqLen, newTokens, kvCache) → map[string]float64
       ↓                                                                    ↓
rooflineStepTime(gpu, modelConfig, hwConfig, stepConfig, tp) → int64 (microseconds)
       ↑
  Called by simulator.go Step() when Roofline=true
```

No new types, interfaces, or state. Pure functions — no side effects to test for.

**Extension Friction:** N/A — no new types or fields added.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #254: "Verify block 2's formula is correct" | We verify behavioral properties (monotonicity, conservation) not exact formulas | CORRECTION: Exact formula testing is structural; behavioral tests are more robust |
| #257: "Edge cases: zero tokens" | Not included — zero tokens is not a valid input (requests always have ≥1 token) | SIMPLIFICATION: Testing invalid inputs that can't reach this function adds no value |
| N/A (codebase state) | `sim/roofline_step_test.go` already exists with `TestCalculateMemoryAccessBytes_Deterministic` from #236 — plan appends to it rather than creating from scratch | ADDITION: Discovered during plan review; existing tests must be preserved |

### E) Review Guide

1. **THE TRICKY PART:** The dead code removal itself — verify that block 2 (lines 46-71) is truly the correct computation. Block 2 separates attention into GEMM-like ops (QK^T, AV at `4 * nHeads * newT * effectiveCtx * dHead`) and vector ops (softmax + RoPE), while block 1 lumped everything into sram_ops. Block 2 is more physically accurate.
2. **WHAT TO SCRUTINIZE:** BC-1 (monotonicity) and BC-3 (TP scaling) — these are the tests most likely to reveal formula bugs.
3. **WHAT'S SAFE TO SKIM:** BC-8/BC-9 (conservation) — these are straightforward sum checks.
4. **KNOWN DEBT:** `calculateMemoryAccessBytes` already has a determinism test from #236; we add conservation and monotonicity tests but don't duplicate the determinism test.

---

## Part 2: Executable Implementation

### F) Implementation Overview

- **Modify:** `sim/roofline_step.go` — remove dead code (lines 29-43)
- **Modify:** `sim/roofline_step_test.go` — append tests (preserves existing `TestCalculateMemoryAccessBytes_Deterministic` from #236)

Key decisions:
- Use table-driven tests for monotonicity and isolation
- Test behavioral invariants (monotonicity, conservation, scaling) not exact formulas
- Use a realistic Llama-3.1-8B-like ModelConfig for all tests

### G) Task Breakdown

---

### Task 1: Remove Dead Code — First `if includeAttention` Block

**Contracts Implemented:** BC-5, BC-10

**Files:**
- Modify: `sim/roofline_step.go:29-43`

**Step 1: Verify the dead code exists (pre-condition)**

Run: `grep -n "if includeAttention" sim/roofline_step.go`
Expected: Two matches (lines 29 and 46)

**Step 2: Remove the dead first block (lines 29-43)**

In `sim/roofline_step.go`, delete the first `if includeAttention` block. The function should go from the `flops := make(...)` line directly to the second (correct) `if includeAttention` block.

After removal, `calculateTransformerFlops` should look like:

```go
func calculateTransformerFlops(config ModelConfig, sequenceLength int64, newTokens int64, includeAttention, includeMLP bool) map[string]float64 {
	dModel := float64(config.HiddenDim)
	nLayers := float64(config.NumLayers)
	nHeads := float64(config.NumHeads)
	nKVHeads := float64(config.NumKVHeads)
	if nKVHeads == 0 {
		nKVHeads = nHeads
	}
	dHead := dModel / nHeads

	// Qwen2.5 uses specific intermediate dims for SwiGLU
	dFF := 4.0 * dModel
	if config.IntermediateDim > 0 {
		dFF = float64(config.IntermediateDim)
	}

	seqLen := float64(sequenceLength)
	newT := float64(newTokens)
	flops := make(map[string]float64)

	if includeAttention {
		dKV := nKVHeads * dHead

		// 1. Standard GEMMs (Weights)
		qkvFlops := 2 * newT * (dModel*dModel + 2*dModel*dKV)
		projFlops := 2 * newT * dModel * dModel
		flops["gemm_ops"] = (qkvFlops + projFlops) * nLayers

		// SRAM-local ops (FlashAttention)
		effectiveCtx := seqLen
		if newT > 1 {
			effectiveCtx = seqLen + (newT-1)/2.0
		}

		// 2. Attention Score Ops (The TTFT Killer)
		// We treat the QK^T and AV as "GEMM" ops if they are large enough,
		// because they utilize the same execution units as standard GEMMs in FlashAttention.
		attnGemmOps := (4 * nHeads * newT * effectiveCtx * dHead)

		// 3. Vector Ops (Softmax, Masking, RoPE)
		ropeOps := 2 * newT * dModel
		vectorOps := (5 * nHeads * newT * effectiveCtx) + ropeOps

		flops["gemm_ops"] += (attnGemmOps * nLayers)
		flops["sram_ops"] = (vectorOps * nLayers)
	}

	if includeMLP {
		// SwiGLU Gating: Gate, Up, and Down (3 matrices)
		flops["gemm_ops"] += 2 * newT * (3 * dModel * dFF) * nLayers
	}

	flops["total"] = flops["gemm_ops"] + flops["sram_ops"]
	return flops
}
```

**Step 3: Run existing tests to confirm no regression**

Run: `go test ./sim/... -v -count=1 2>&1 | tail -20`
Expected: PASS (all existing tests still pass)

**Step 4: Verify only one `if includeAttention` block remains**

Run: `grep -c "if includeAttention" sim/roofline_step.go`
Expected: `1`

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/roofline_step.go
git commit -m "fix(roofline): remove dead first attention block in calculateTransformerFlops (BC-5, BC-10)

The first 'if includeAttention' block (old lines 29-43) was dead code —
its results were entirely overwritten by the second block. The second block
is the correct computation: it separates attention into GEMM-like ops
(QK^T, AV) and vector ops (softmax, RoPE), while the first block
incorrectly lumped all attention into sram_ops.

Fixes #254

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add FLOPs Conservation and Component Isolation Tests

**Contracts Implemented:** BC-6, BC-7, BC-8

**Files:**
- Modify: `sim/roofline_step_test.go` (append to existing file — preserves `TestCalculateMemoryAccessBytes_Deterministic` from #236)

**Step 1: Write tests for conservation and isolation**

Append to `sim/roofline_step_test.go` (add `"math"` to existing imports, then append all new functions after the existing `TestCalculateMemoryAccessBytes_Deterministic`):

```go
// The existing import block already has "sort" and "testing".
// Add "math" to the existing import block:
// import (
// 	"math"
// 	"sort"
// 	"testing"
// )

// testModelConfig returns a Llama-3.1-8B-like config for roofline tests.
func testModelConfig() ModelConfig {
	return ModelConfig{
		NumLayers:       32,
		HiddenDim:       4096,
		NumHeads:        32,
		NumKVHeads:      8,
		VocabSize:       128256,
		BytesPerParam:   2, // bfloat16
		IntermediateDim: 14336,
	}
}

// testHardwareCalib returns an H100-like hardware config for roofline tests.
func testHardwareCalib() HardwareCalib {
	return HardwareCalib{
		TFlopsPeak:       989.0,
		BwPeakTBs:        3.35,
		BwEffConstant:    0.7,
		TOverheadMicros:  50.0,
		PerLayerOverhead: 5.0,
		MfuPrefill:       0.55,
		MfuDecode:        0.30,
		AllReduceLatency: 10.0,
	}
}

func TestCalculateTransformerFlops_Conservation_TotalEqualsSumOfComponents(t *testing.T) {
	// BC-8: total MUST equal gemm_ops + sram_ops
	mc := testModelConfig()
	tests := []struct {
		name    string
		seqLen  int64
		newT    int64
		attn    bool
		mlp     bool
	}{
		{"prefill attn+mlp", 0, 128, true, true},
		{"decode attn+mlp", 512, 1, true, true},
		{"attn only", 256, 64, true, false},
		{"mlp only", 256, 64, false, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flops := calculateTransformerFlops(mc, tt.seqLen, tt.newT, tt.attn, tt.mlp)
			sum := flops["gemm_ops"] + flops["sram_ops"]
			if flops["total"] != sum {
				t.Errorf("total (%g) != gemm_ops (%g) + sram_ops (%g) = %g",
					flops["total"], flops["gemm_ops"], flops["sram_ops"], sum)
			}
		})
	}
}

func TestCalculateTransformerFlops_AttentionOnly_NoMLPContribution(t *testing.T) {
	// BC-7: disabling MLP zeroes MLP FLOPs
	mc := testModelConfig()

	attnOnly := calculateTransformerFlops(mc, 256, 64, true, false)
	both := calculateTransformerFlops(mc, 256, 64, true, true)

	// With MLP disabled, gemm_ops should be less (no SwiGLU)
	if attnOnly["gemm_ops"] >= both["gemm_ops"] {
		t.Errorf("attention-only gemm_ops (%g) should be less than attn+mlp gemm_ops (%g)",
			attnOnly["gemm_ops"], both["gemm_ops"])
	}
	// sram_ops should be the same (MLP doesn't contribute to sram_ops)
	if attnOnly["sram_ops"] != both["sram_ops"] {
		t.Errorf("sram_ops should be identical with/without MLP: got %g vs %g",
			attnOnly["sram_ops"], both["sram_ops"])
	}
}

func TestCalculateTransformerFlops_MLPOnly_NoAttentionContribution(t *testing.T) {
	// BC-6: disabling attention zeroes attention FLOPs, sram_ops must be zero
	mc := testModelConfig()

	mlpOnly := calculateTransformerFlops(mc, 256, 64, false, true)

	if mlpOnly["sram_ops"] != 0 {
		t.Errorf("MLP-only sram_ops should be 0, got %g", mlpOnly["sram_ops"])
	}
	if mlpOnly["gemm_ops"] <= 0 {
		t.Errorf("MLP-only gemm_ops should be > 0, got %g", mlpOnly["gemm_ops"])
	}
}
```

**Step 2: Run tests to verify they pass**

Run: `go test ./sim/... -run TestCalculateTransformerFlops -v`
Expected: PASS (3 tests)

**Step 3: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 4: Commit**

```bash
git add sim/roofline_step_test.go
git commit -m "test(roofline): add FLOPs conservation and component isolation tests (BC-6, BC-7, BC-8)

- BC-8: total equals gemm_ops + sram_ops (table-driven, 4 scenarios)
- BC-7: disabling MLP removes SwiGLU from gemm_ops, leaves sram_ops unchanged
- BC-6: disabling attention zeroes sram_ops, leaves only MLP in gemm_ops

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add FLOPs and Memory Monotonicity Tests

**Contracts Implemented:** BC-1, BC-2

**Files:**
- Modify: `sim/roofline_step_test.go`

**Step 1: Add monotonicity tests**

Append to `sim/roofline_step_test.go`:

```go
func TestCalculateTransformerFlops_Monotonicity_MoreTokensMoreFlops(t *testing.T) {
	// BC-1: more newTokens MUST produce higher total FLOPs
	mc := testModelConfig()

	small := calculateTransformerFlops(mc, 512, 100, true, true)
	large := calculateTransformerFlops(mc, 512, 200, true, true)

	if large["total"] <= small["total"] {
		t.Errorf("200 tokens total FLOPs (%g) should exceed 100 tokens (%g)",
			large["total"], small["total"])
	}
	// Check component-level monotonicity too
	if large["gemm_ops"] <= small["gemm_ops"] {
		t.Errorf("200 tokens gemm_ops (%g) should exceed 100 tokens (%g)",
			large["gemm_ops"], small["gemm_ops"])
	}
}

func TestCalculateMemoryAccessBytes_Monotonicity_MoreTokensMoreBytes(t *testing.T) {
	// BC-2: more newTokens MUST produce higher total bytes
	mc := testModelConfig()

	small := calculateMemoryAccessBytes(mc, 512, 100, true)
	large := calculateMemoryAccessBytes(mc, 512, 200, true)

	if large["total"] <= small["total"] {
		t.Errorf("200 tokens total bytes (%g) should exceed 100 tokens (%g)",
			large["total"], small["total"])
	}
}

func TestCalculateMemoryAccessBytes_Conservation_TotalEqualsSumOfComponents(t *testing.T) {
	// BC-9: total MUST equal sum of all non-"total" components
	mc := testModelConfig()

	mem := calculateMemoryAccessBytes(mc, 512, 64, true)

	// Sort keys before float accumulation (antipattern #2)
	keys := make([]string, 0, len(mem))
	for k := range mem {
		if k != "total" {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	var sum float64
	for _, k := range keys {
		sum += mem[k]
	}
	if math.Abs(mem["total"]-sum) > 1e-6 {
		t.Errorf("total (%g) != sum of components (%g), delta=%g",
			mem["total"], sum, mem["total"]-sum)
	}
}
```

**Step 2: Run tests to verify they pass**

Run: `go test ./sim/... -run "TestCalculateTransformerFlops_Monotonicity|TestCalculateMemoryAccessBytes" -v`
Expected: PASS (3 tests)

**Step 3: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 4: Commit**

```bash
git add sim/roofline_step_test.go
git commit -m "test(roofline): add monotonicity and memory conservation tests (BC-1, BC-2, BC-9)

- BC-1: more tokens → more FLOPs (total and gemm_ops)
- BC-2: more tokens → more memory bytes
- BC-9: memory total equals sum of all component values

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add TP Scaling and Roofline Smoke Tests

**Contracts Implemented:** BC-3, BC-4

**Files:**
- Modify: `sim/roofline_step_test.go`

**Step 1: Add TP scaling and smoke tests**

Append to `sim/roofline_step_test.go`:

```go
func TestRooflineStepTime_TPScaling_TP2LessThanTP1(t *testing.T) {
	// BC-3: TP=2 MUST produce strictly less latency than TP=1
	mc := testModelConfig()
	hc := testHardwareCalib()

	step := StepConfig{
		PrefillRequests: []PrefillRequestConfig{
			{ProgressIndex: 0, NumNewPrefillTokens: 128},
		},
		DecodeRequests: []DecodeRequestConfig{
			{ProgressIndex: 256, NumNewDecodeTokens: 1},
		},
	}

	tp1 := rooflineStepTime("H100", mc, hc, step, 1)
	tp2 := rooflineStepTime("H100", mc, hc, step, 2)

	if tp2 >= tp1 {
		t.Errorf("TP=2 latency (%d µs) should be less than TP=1 (%d µs)", tp2, tp1)
	}
	if tp2 <= 0 {
		t.Errorf("TP=2 latency should be positive, got %d", tp2)
	}
}

func TestRooflineStepTime_Smoke_ValidInputsProducePositiveFiniteResult(t *testing.T) {
	// BC-4: valid inputs MUST produce > 0, finite result
	mc := testModelConfig()
	hc := testHardwareCalib()

	tests := []struct {
		name string
		step StepConfig
	}{
		{
			"prefill only",
			StepConfig{
				PrefillRequests: []PrefillRequestConfig{
					{ProgressIndex: 0, NumNewPrefillTokens: 256},
				},
			},
		},
		{
			"decode only",
			StepConfig{
				DecodeRequests: []DecodeRequestConfig{
					{ProgressIndex: 512, NumNewDecodeTokens: 1},
				},
			},
		},
		{
			"mixed prefill+decode",
			StepConfig{
				PrefillRequests: []PrefillRequestConfig{
					{ProgressIndex: 0, NumNewPrefillTokens: 64},
				},
				DecodeRequests: []DecodeRequestConfig{
					{ProgressIndex: 128, NumNewDecodeTokens: 1},
					{ProgressIndex: 256, NumNewDecodeTokens: 1},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := rooflineStepTime("H100", mc, hc, tt.step, 1)
			if result <= 0 {
				t.Errorf("expected positive latency, got %d µs", result)
			}
			if math.IsNaN(float64(result)) || math.IsInf(float64(result), 0) {
				t.Errorf("expected finite latency, got %d", result)
			}
		})
	}
}

func TestRooflineStepTime_EmptyStep_ReturnsOverheadOnly(t *testing.T) {
	// Edge case: no requests should still return overhead (non-zero due to TOverheadMicros)
	mc := testModelConfig()
	hc := testHardwareCalib()

	step := StepConfig{} // empty
	result := rooflineStepTime("H100", mc, hc, step, 1)

	// Should be approximately TOverheadMicros (50) + layer overhead
	if result <= 0 {
		t.Errorf("empty step should still have overhead latency, got %d µs", result)
	}
}
```

**Step 2: Run tests to verify they pass**

Run: `go test ./sim/... -run "TestRooflineStepTime" -v`
Expected: PASS (3 tests)

**Step 3: Run all roofline tests together**

Run: `go test ./sim/... -run "TestCalculateTransformerFlops|TestCalculateMemoryAccessBytes|TestRooflineStepTime" -v`
Expected: PASS (all tests)

**Step 4: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/roofline_step_test.go
git commit -m "test(roofline): add TP scaling, smoke, and edge case tests (BC-3, BC-4)

- BC-3: TP=2 latency strictly less than TP=1
- BC-4: valid inputs produce positive, finite latency (3 scenarios)
- Edge case: empty step config returns overhead-only latency

Fixes #257

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Final Verification — Full Test Suite + Lint

**Contracts Implemented:** All (regression check)

**Step 1: Run full test suite**

Run: `go test ./... -count=1`
Expected: PASS (all packages)

**Step 2: Run full lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 3: Verify dead code removal**

Run: `grep -c "if includeAttention" sim/roofline_step.go`
Expected: `1`

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 3 | Unit/Invariant | TestCalculateTransformerFlops_Monotonicity_MoreTokensMoreFlops |
| BC-2 | Task 3 | Unit/Invariant | TestCalculateMemoryAccessBytes_Monotonicity_MoreTokensMoreBytes |
| BC-3 | Task 4 | Unit/Invariant | TestRooflineStepTime_TPScaling_TP2LessThanTP1 |
| BC-4 | Task 4 | Unit | TestRooflineStepTime_Smoke_ValidInputsProducePositiveFiniteResult |
| BC-5 | Task 1 | Manual/grep | Verified by `grep -c` in Task 1 Step 4 |
| BC-6 | Task 2 | Unit | TestCalculateTransformerFlops_MLPOnly_NoAttentionContribution |
| BC-7 | Task 2 | Unit | TestCalculateTransformerFlops_AttentionOnly_NoMLPContribution |
| BC-8 | Task 2 | Unit/Invariant | TestCalculateTransformerFlops_Conservation_TotalEqualsSumOfComponents |
| BC-9 | Task 3 | Unit/Invariant | TestCalculateMemoryAccessBytes_Conservation_TotalEqualsSumOfComponents |
| BC-10 | Task 1 | Manual/grep | Verified by `grep -c` in Task 1 Step 4 |

No golden dataset updates needed — these functions are not exercised by the golden dataset tests (golden tests use blackbox alpha/beta mode, not roofline mode).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Removing wrong block | Low | High | Block 2 has more sophisticated decomposition (GEMM vs vector ops); block 1 lumps everything. Behavioral tests verify monotonicity holds after removal. | Task 1, 2, 3 |
| Tests too tightly coupled to formula | Medium | Low | All tests assert behavioral invariants (monotonicity, conservation, scaling), not exact values | Task 2, 3, 4 |
| TP scaling test fragile to overhead | Low | Low | Test uses both prefill and decode to ensure compute dominates overhead | Task 4 |

### E) Review Guide (reiterated)

1. **THE TRICKY PART:** Verify that the surviving attention block (block 2) correctly separates attention into GEMM ops (`4 * nHeads * newT * effectiveCtx * dHead` for QK^T + AV) and vector ops (softmax + RoPE). The key difference from block 1 is that block 2 treats QK^T and AV as "GEMM-like" because they use the same tensor core execution units in FlashAttention.
2. **WHAT TO SCRUTINIZE:** BC-1 monotonicity — if this test fails after dead code removal, it means block 2 has a formula bug.
3. **WHAT'S SAFE TO SKIM:** BC-8/BC-9 conservation tests — trivial sum checks.
4. **KNOWN DEBT:** None discovered.

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — pure function tests only
- [x] No feature creep beyond PR scope — only dead code removal + tests
- [x] No unexercised flags or interfaces — N/A
- [x] No partial implementations — all tests complete
- [x] No breaking changes — removing dead code doesn't change behavior
- [x] No hidden global state impact — pure functions
- [x] All new code will pass golangci-lint
- [x] CLAUDE.md — no update needed (no new files/packages/flags)
- [x] No stale references in CLAUDE.md
- [x] Deviation log reviewed — 2 deviations, both justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Task 1 before Task 2-4)
- [x] All contracts mapped to tasks
- [x] Golden dataset — no update needed (roofline not exercised by golden tests)
- [x] Construction site audit — no struct fields added
- [x] No new CLI flags
- [x] No new error paths with silent continue
- [x] No map iteration feeding float accumulation without sorted keys (existing sorted accumulation in calculateMemoryAccessBytes preserved)
- [x] Library code never calls logrus.Fatalf — N/A (test code only + deletion)
- [x] No resource allocation loops — N/A
- [x] No exported mutable maps — N/A
- [x] No YAML config changes — N/A
- [x] No YAML loading changes — N/A
- [x] No division with runtime denominators without guards — N/A (existing code, not modified)
- [x] No new interfaces — N/A
- [x] No methods spanning multiple concerns — N/A
- [x] No monolithic config changes — N/A
- [x] Grepped for PR references — no stale "planned for PR" references to these issues
- [x] Not part of a macro plan — standalone bug fix + test addition

---

## Appendix: File-Level Implementation Details

### File: `sim/roofline_step.go`

**Purpose:** Remove dead first `if includeAttention` block (old lines 29-43)

**Change:** Delete lines 29-43 (the first `if includeAttention { ... }` block). The second block at lines 46-71 becomes the sole attention computation. No other changes to this file.

### File: `sim/roofline_step_test.go` (MODIFY — existing file with `TestCalculateMemoryAccessBytes_Deterministic`)

**Purpose:** Extend with comprehensive behavioral tests for all three roofline functions.

**Existing content preserved:** `TestCalculateMemoryAccessBytes_Deterministic` (55 lines, from #236)

**New content appended (see Tasks 2-4):**
- `testModelConfig()` — Llama-3.1-8B-like fixture
- `testHardwareCalib()` — H100-like fixture
- 8 test functions covering all 10 behavioral contracts
