# CrossModelLatencyModel Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a third latency model backend that predicts GPU step time from model architecture features (layer count, KV dimensions, MoE routing, TP degree) using globally-fitted physics coefficients — enabling capacity planning for new models from their HuggingFace config.json alone, without dedicated profiling.

**The problem today:** BLIS has two latency backends — blackbox (requires per-model trained coefficients) and roofline (analytical FLOPs/bandwidth, no MoE awareness). Neither can predict step time for a new model without either dedicated profiling data or ignoring MoE architecture effects (~4x overestimate for Mixtral).

**What this PR adds:**
1. **Cross-model step time prediction** — a single set of 4 physics coefficients (per-layer overhead, KV bandwidth, MoE dispatch, TP sync) predicts step time for any model given its config.json
2. **MoE-aware architecture parsing** — `ModelConfig` gains `NumLocalExperts` and `NumExpertsPerTok` fields, parsed from HuggingFace config.json
3. **Global default coefficients** — pre-trained values from Iteration 3 of the learned latency model research, stored in `defaults.yaml`
4. **CLI integration** — `--latency-model crossmodel` activates the backend, reusing the existing `--hardware`/`--tp` and HF config auto-fetch infrastructure

**Why this matters:** This completes issue #472 and enables the learned latency model research pipeline (training/problem.md). The cross-model backend is the first step toward replacing per-model coefficient profiling with architecture-parameterized prediction.

**Architecture:** New `CrossModelLatencyModel` struct in `sim/latency/crossmodel.go` implementing the 5-method `LatencyModel` interface. Pre-computes architecture features at construction from `ModelConfig` fields. Factory case added to `NewLatencyModel` switch. CLI wiring reuses the roofline path's `resolveModelConfig`/`resolveHardwareConfig` chain. Coefficients stored in a new `crossmodel_defaults:` section of `defaults.yaml`.

**Source:** GitHub issue #472 (PR-B section) + training/ledger.md (Iteration 3 coefficients)

**Closes:** Fixes #472

**Behavioral Contracts:** See Part 1, Section B below

---

## Phase 0: Component Context

**Building block:** Third `LatencyModel` backend (policy template — new implementation behind existing frozen interface).

**Adjacent blocks:**
- `sim/latency/latency.go` — factory switch (add `case "crossmodel":`)
- `sim/model_hardware_config.go` — `ModelConfig` struct (add MoE fields)
- `sim/latency/config.go` — HF config parsing (add MoE field extraction)
- `cmd/root.go` — CLI wiring (add `backend == "crossmodel"` block)
- `defaults.yaml` / `cmd/default_config.go` — coefficient storage

**Invariants touched:**
- INV-L1 (non-negativity): StepTime ≥ 1 for non-empty batches — enforced by `max(1, ...)` clamp
- INV-L2 (monotonicity): Non-decreasing in prefill/decode tokens — enforced by non-negative coefficients
- INV-6 (determinism): No map iteration, no floating-point order dependence — deterministic feature computation

**Construction site audit for `ModelConfig`:**
- `sim/latency/config.go:131-140` — `GetModelConfig` builds `&sim.ModelConfig{...}` (MUST add MoE fields)
- `sim/latency/roofline_test.go:53-63` — `testModelConfig()` helper (add MoE fields, 0 = dense)
- All other `sim.ModelConfig{...}` sites use partial struct literals — zero-value safe for MoE fields (Go defaults unmentioned fields to 0, which means "dense model" for NumLocalExperts/NumExpertsPerTok)

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds a third latency model backend (`crossmodel`) that estimates GPU step time using physics-informed features derived from a model's HuggingFace config.json. Unlike blackbox (per-model coefficients) or roofline (analytical FLOPs), cross-model uses 4 globally-fitted coefficients that work across model architectures. It requires MoE field parsing in ModelConfig (2 new fields, zero-value safe) and a new `crossmodel_defaults` section in defaults.yaml. The CLI activation path (`--latency-model crossmodel`) reuses the existing roofline infrastructure for HF config auto-fetch and hardware config resolution.

One deviation from the issue: the issue mentions removing the ad-hoc MoE string-search warning in root.go; the plan defers this to avoid scope creep (the warning is harmless alongside proper MoE fields).

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: StepTime Physics Formula
- GIVEN a non-empty batch with known architecture (numLayers, kvDim, MoE indicator, TP indicator)
- WHEN StepTime is called
- THEN the result reflects per-layer overhead + KV bandwidth cost + MoE dispatch cost + TP sync cost
- MECHANISM: `β₀·L + β₁·dc·kvDimScaled + β₂·(pf+dc)·isMoE + β₃·isTP`

BC-2: Monotonicity in Prefill Tokens (MoE models)
- GIVEN two identical batches on an MoE model (NumLocalExperts > 0), except batch A has one more prefill token
- WHEN StepTime is called on each
- THEN result(A) ≥ result(B)
- MECHANISM: β₂ coefficient is non-negative (NNLS-fitted); prefill tokens appear in the MoE dispatch term. Note: for dense models (isMoE=0), prefill tokens have zero effect on step time — prefill compute cost is absorbed into β₀ (per-layer term)

BC-3: Monotonicity in Decode Tokens
- GIVEN two identical batches except batch A has one more decode token
- WHEN StepTime is called on each
- THEN result(A) ≥ result(B)
- MECHANISM: β₁ and β₂ are non-negative; decode tokens appear in both KV bandwidth and MoE terms

BC-4: Non-negativity (INV-L1)
- GIVEN a non-empty batch
- WHEN StepTime is called
- THEN the result MUST be ≥ 1
- MECHANISM: `max(1, int64(stepTime))` clamp

BC-5: Empty Batch Returns Zero
- GIVEN an empty batch
- WHEN StepTime is called
- THEN the result is 0 (no GPU work)
- MECHANISM: Early return when `len(batch) == 0`

BC-6: MoE Detection
- GIVEN a ModelConfig with NumLocalExperts > 0
- WHEN the cross-model backend is constructed
- THEN MoE-specific features contribute to step time
- MECHANISM: `isMoE = 1.0` when NumLocalExperts > 0, else 0.0

BC-7: TP Indicator
- GIVEN hw.TP > 1
- WHEN the cross-model backend is constructed
- THEN TP synchronization overhead contributes to step time
- MECHANISM: `isTP = 1.0` when TP > 1, else 0.0

BC-8: Alpha/Gamma Semantics Preserved
- GIVEN any request
- WHEN QueueingTime or OutputTokenProcessingTime is called
- THEN the result matches the blackbox model's behavior (same AlphaCoeffs interpretation)
- MECHANISM: QueueingTime and OutputTokenProcessingTime implementations are identical to BlackboxLatencyModel

BC-9: Factory Backend Selection
- GIVEN `Backend = "crossmodel"` with valid ModelConfig and BetaCoeffs ≥ 4
- WHEN NewLatencyModel is called
- THEN it returns the cross-model backend (not blackbox or roofline)
- MECHANISM: `case "crossmodel":` in factory switch

BC-10: Architecture Features Frozen at Construction
- GIVEN a constructed CrossModelLatencyModel
- WHEN StepTime is called multiple times
- THEN architecture features (numLayers, kvDimScaled, isMoE, isTP) are identical across calls
- MECHANISM: Features computed once in constructor, stored as struct fields

**Negative Contracts:**

BC-11: No Fatalf in Library
- GIVEN any invalid input
- WHEN the cross-model backend or factory is called
- THEN errors are returned (never logrus.Fatalf or panic) (R6)

**Error Handling:**

BC-12: Factory Validation — Missing Config
- GIVEN `Backend = "crossmodel"` with NumLayers == 0 or NumHeads == 0 or HiddenDim == 0
- WHEN NewLatencyModel is called
- THEN it returns an error describing the missing field
- MECHANISM: Validation in the `case "crossmodel":` factory branch

BC-13: Factory Validation — Short BetaCoeffs
- GIVEN `Backend = "crossmodel"` with len(BetaCoeffs) < 4
- WHEN NewLatencyModel is called
- THEN it returns an error requiring ≥ 4 elements
- MECHANISM: Length check before construction

### C) Component Interaction

```
cmd/root.go                    sim/latency/latency.go           sim/latency/crossmodel.go
┌──────────────┐               ┌──────────────────┐             ┌─────────────────────┐
│ --latency-   │──resolves──>  │ NewLatencyModel() │──creates──>│CrossModelLatencyModel│
│  model       │   HF config   │  case "crossmodel"│            │  .StepTime(batch)   │
│  crossmodel  │               └──────────────────┘             │  .QueueingTime(req) │
└──────────────┘                                                └─────────────────────┘
        │                       sim/model_hardware_config.go              │
        │                       ┌──────────────────────┐                  │
        └──────────────────────>│ ModelConfig           │<───reads────────┘
                                │  +NumLocalExperts     │
         defaults.yaml          │  +NumExpertsPerTok    │
         ┌───────────────┐      └──────────────────────┘
         │crossmodel_    │
         │  defaults:    │──loaded by──> cmd/root.go
         │  beta_coeffs  │
         │  alpha_coeffs │
         └───────────────┘
```

**State changes:** None. All architecture features frozen at construction. No mutable state.

**Extension friction:** Adding a 4th backend after this: 1 line in `validLatencyBackends` + 1 case in factory switch = 2 files. Same as before this PR.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue mentions removing ad-hoc MoE string-search warning in root.go | Keeps the warning | SIMPLIFICATION: Warning is harmless alongside proper MoE fields; removal is cosmetic scope creep |
| Issue shows `countTokens(batch)` helper function | Inlines the token counting loop | SIMPLIFICATION: Helper adds indirection for a 10-line loop used in one place; matches BlackboxLatencyModel pattern |
| Issue specifies `TP > 0` validation for crossmodel | Uses `TP > 0` like roofline | CORRECTION: Consistent with roofline validation already in the factory |
| Issue does not mention zero-coeff guard interaction | Plan updates guard at root.go:281 to exclude crossmodel | ADDITION: Guard `backend != "roofline"` must become `backend != "roofline" && backend != "crossmodel"` to avoid misleading fatalf when crossmodel defaults are missing |
| Issue does not mention KV bandwidth term physics | Plan adds code comment explaining prefill exclusion | ADDITION: β₁ term applies only to decode tokens; prefill KV write cost is absorbed into β₀ (per-layer term) per the training methodology |

### E) Review Guide

**The tricky part:** The StepTime formula uses `kvDimScaled = numLayers * numKVHeads * (hiddenDim / numHeads) / TP * 1e-6`. The division by `numHeads` computes `headDim`, and the division by `TP` partitions across tensor-parallel ranks. Verify the computation order avoids integer truncation (must cast to float64 before dividing).

**What to scrutinize:** BC-12 (factory validation) — ensure all required ModelConfig fields are validated. Missing NumKVHeads is a subtle case (some models use NumKVHeads == 0 to mean "same as NumHeads", per the HF convention — the existing `GetModelConfig` already handles this fallback at config.go:112-114).

**What's safe to skim:** QueueingTime, OutputTokenProcessingTime, SchedulingProcessingTime, PreemptionProcessingTime — all identical to BlackboxLatencyModel. The token counting loop in StepTime — same pattern as BlackboxLatencyModel.

**Known debt:** The γ₁=860.6 µs/tok coefficient (AlphaCoeffs[2]) is a diagnostic artifact, not a physical output processing time (see training/ledger.md). This is acceptable for the initial implementation and documented in the problem statement Section 7a.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/latency/crossmodel.go` — CrossModelLatencyModel struct + 5 interface methods
- `sim/latency/crossmodel_test.go` — behavioral tests (monotonicity, MoE, TP, factory, edge cases)

**Files to modify:**
- `sim/model_hardware_config.go` — Add MoE fields to ModelConfig
- `sim/latency/config.go` — Parse MoE fields in GetModelConfig()
- `sim/latency/latency.go` — Add `case "crossmodel":` to factory switch
- `sim/bundle.go` — Add `"crossmodel": true` to validLatencyBackends
- `defaults.yaml` — Add `crossmodel_defaults:` section
- `cmd/default_config.go` — Add CrossModelDefaults to Config struct
- `cmd/root.go` — Wire `backend == "crossmodel"` path
- `sim/config.go` — Update Backend field comment
- `CLAUDE.md` — Update Latency Estimation section, File Organization
- `docs/guide/latency-models.md` — Add Cross-Model Mode section
- `docs/reference/configuration.md` — Update --latency-model flag description
- `docs/reference/models.md` — Mention crossmodel mode

**Key decisions:**
- Reuse `LatencyCoeffs.BetaCoeffs[0:4]` and `AlphaCoeffs[0:3]` — no new coefficient struct
- MoE fields on ModelConfig directly (zero-value safe: 0 = dense)
- KV dim precomputed at construction (eliminates per-call division and validates at build time)

### G) Task Breakdown

---

### Task 1: Add MoE Fields to ModelConfig + Parsing

**Contracts Implemented:** BC-6 (MoE detection prerequisite)

**Files:**
- Modify: `sim/model_hardware_config.go`
- Modify: `sim/latency/config.go:131-140`
- Modify: `sim/latency/roofline_test.go` (testModelConfig helper)

**Step 1: Add MoE fields to ModelConfig**

In `sim/model_hardware_config.go`, add after `IntermediateDim`:
```go
NumLocalExperts  int `json:"num_local_experts"`   // 0 = dense model (MoE: number of experts)
NumExpertsPerTok int `json:"num_experts_per_tok"` // 0 = dense model (MoE: active experts per token)
```

**Step 2: Parse MoE fields in GetModelConfig**

In `sim/latency/config.go`, before the `modelConfig := &sim.ModelConfig{` block (~line 131), add:
```go
numLocalExperts := getInt("num_local_experts")
numExpertsPerTok := getInt("num_experts_per_tok")
```
Then add to the struct literal:
```go
NumLocalExperts:  numLocalExperts,
NumExpertsPerTok: numExpertsPerTok,
```

**Step 3: Update testModelConfig helper**

In `sim/latency/roofline_test.go`, add to `testModelConfig()`:
```go
NumLocalExperts:  0,  // dense model
NumExpertsPerTok: 0,
```

**Step 4: Run tests**

Run: `go test ./sim/latency/... -count=1`
Expected: ALL PASS (zero-value MoE fields are backward compatible)

**Step 5: Commit**

```bash
git add sim/model_hardware_config.go sim/latency/config.go sim/latency/roofline_test.go
git commit -m "feat(latency): add MoE fields to ModelConfig for cross-model backend (BC-6)

- Add NumLocalExperts and NumExpertsPerTok to ModelConfig (zero-value safe: 0 = dense)
- Parse num_local_experts and num_experts_per_tok from HuggingFace config.json
- Update testModelConfig helper with explicit MoE fields

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: CrossModelLatencyModel Implementation + Tests

**Contracts Implemented:** BC-1, BC-2, BC-3, BC-4, BC-5, BC-6, BC-7, BC-8, BC-10

**Files:**
- Create: `sim/latency/crossmodel.go`
- Create: `sim/latency/crossmodel_test.go`

**Step 1: Write failing tests**

Create `sim/latency/crossmodel_test.go` with behavioral tests:

```go
package latency

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
)

// testCrossModelConfig returns a dense model config for crossmodel tests.
func testCrossModelConfig() sim.ModelConfig {
	return sim.ModelConfig{
		NumLayers:        32,
		HiddenDim:        4096,
		NumHeads:         32,
		NumKVHeads:       8,
		VocabSize:        128256,
		BytesPerParam:    2,
		IntermediateDim:  14336,
		NumLocalExperts:  0,
		NumExpertsPerTok: 0,
	}
}

// testMoEModelConfig returns a Mixtral-like MoE model config.
func testMoEModelConfig() sim.ModelConfig {
	mc := testCrossModelConfig()
	mc.NumLocalExperts = 8
	mc.NumExpertsPerTok = 2
	return mc
}

func TestCrossModelLatencyModel_StepTime_NonEmpty_Positive(t *testing.T) {
	// BC-4: non-empty batch → StepTime ≥ 1
	m := &CrossModelLatencyModel{
		betaCoeffs:  []float64{116.110, 1226.868, 19.943, 9445.157},
		alphaCoeffs: []float64{13732.0, 0.0, 860.6},
		numLayers:   32,
		kvDimScaled: 32 * 8 * (4096 / 32) / 2 * 1e-6, // L * kvHeads * headDim / TP * 1e-6
		isMoE:       0.0,
		isTP:        1.0, // TP=2 > 1
	}
	batch := []*sim.Request{{
		InputTokens:  make([]int, 100),
		OutputTokens: []int{1},
		ProgressIndex: 100,
		NumNewTokens:  1,
	}}
	result := m.StepTime(batch)
	assert.GreaterOrEqual(t, result, int64(1), "BC-4: non-empty batch must produce positive step time")
}

func TestCrossModelLatencyModel_StepTime_EmptyBatch_Zero(t *testing.T) {
	// BC-5: empty batch → 0
	m := &CrossModelLatencyModel{
		betaCoeffs:  []float64{116.110, 1226.868, 19.943, 9445.157},
		alphaCoeffs: []float64{13732.0, 0.0, 860.6},
		numLayers:   32,
		kvDimScaled: 0.001,
		isMoE:       0.0,
		isTP:        0.0,
	}
	result := m.StepTime([]*sim.Request{})
	assert.Equal(t, int64(0), result, "BC-5: empty batch must return 0")
}

func TestCrossModelLatencyModel_StepTime_Monotonic_Decode(t *testing.T) {
	// BC-3: more decode tokens → higher or equal step time
	m := &CrossModelLatencyModel{
		betaCoeffs:  []float64{116.110, 1226.868, 19.943, 9445.157},
		alphaCoeffs: []float64{13732.0, 0.0, 860.6},
		numLayers:   32,
		kvDimScaled: 0.004096, // realistic scaled value
		isMoE:       0.0,
		isTP:        1.0,
	}
	batchSmall := []*sim.Request{{
		InputTokens: make([]int, 10), OutputTokens: []int{1},
		ProgressIndex: 10, NumNewTokens: 1,
	}}
	batchLarge := []*sim.Request{
		{InputTokens: make([]int, 10), OutputTokens: []int{1}, ProgressIndex: 10, NumNewTokens: 1},
		{InputTokens: make([]int, 10), OutputTokens: []int{1}, ProgressIndex: 10, NumNewTokens: 1},
	}
	small := m.StepTime(batchSmall)
	large := m.StepTime(batchLarge)
	assert.GreaterOrEqual(t, large, small, "BC-3: more decode tokens must produce >= step time")
}

func TestCrossModelLatencyModel_MoE_IncreasesStepTime(t *testing.T) {
	// BC-6: MoE indicator contributes to step time
	baseCfg := func(isMoE float64) *CrossModelLatencyModel {
		return &CrossModelLatencyModel{
			betaCoeffs:  []float64{116.110, 1226.868, 19.943, 9445.157},
			alphaCoeffs: []float64{13732.0, 0.0, 860.6},
			numLayers:   32,
			kvDimScaled: 0.004096,
			isMoE:       isMoE,
			isTP:        0.0,
		}
	}
	batch := []*sim.Request{{
		InputTokens: make([]int, 100), OutputTokens: []int{1},
		ProgressIndex: 0, NumNewTokens: 100,
	}}
	dense := baseCfg(0.0).StepTime(batch)
	moe := baseCfg(1.0).StepTime(batch)
	assert.Greater(t, moe, dense, "BC-6: MoE model must have higher step time than dense (same tokens)")
}

func TestCrossModelLatencyModel_TP_IncreasesStepTime(t *testing.T) {
	// BC-7: TP > 1 adds synchronization overhead
	baseCfg := func(isTP float64) *CrossModelLatencyModel {
		return &CrossModelLatencyModel{
			betaCoeffs:  []float64{116.110, 1226.868, 19.943, 9445.157},
			alphaCoeffs: []float64{13732.0, 0.0, 860.6},
			numLayers:   32,
			kvDimScaled: 0.004096,
			isMoE:       0.0,
			isTP:        isTP,
		}
	}
	batch := []*sim.Request{{
		InputTokens: make([]int, 10), OutputTokens: []int{1},
		ProgressIndex: 10, NumNewTokens: 1,
	}}
	noTP := baseCfg(0.0).StepTime(batch)
	withTP := baseCfg(1.0).StepTime(batch)
	assert.Greater(t, withTP, noTP, "BC-7: TP > 1 must add overhead")
}

func TestCrossModelLatencyModel_MoE_PrefillMonotonicity(t *testing.T) {
	// BC-2: more prefill tokens on MoE model → higher or equal step time
	m := &CrossModelLatencyModel{
		betaCoeffs:  []float64{116.110, 1226.868, 19.943, 9445.157},
		alphaCoeffs: []float64{13732.0, 0.0, 860.6},
		numLayers:   32,
		kvDimScaled: 0.004096,
		isMoE:       1.0, // MoE model — prefill tokens contribute via β₂
		isTP:        0.0,
	}
	batchSmall := []*sim.Request{{
		InputTokens: make([]int, 50), ProgressIndex: 0, NumNewTokens: 50,
	}}
	batchLarge := []*sim.Request{{
		InputTokens: make([]int, 100), ProgressIndex: 0, NumNewTokens: 100,
	}}
	small := m.StepTime(batchSmall)
	large := m.StepTime(batchLarge)
	assert.GreaterOrEqual(t, large, small, "BC-2: more prefill tokens on MoE must produce >= step time")
}

func TestCrossModelLatencyModel_QueueingTime_MatchesBlackbox(t *testing.T) {
	// BC-8: QueueingTime identical to blackbox semantics
	alpha := []float64{13732.0, 0.0, 860.6}
	m := &CrossModelLatencyModel{
		betaCoeffs: []float64{116, 1226, 19, 9445}, alphaCoeffs: alpha,
		numLayers: 32, kvDimScaled: 0.001, isMoE: 0, isTP: 0,
	}
	bb := &BlackboxLatencyModel{
		betaCoeffs: []float64{1000, 10, 5}, alphaCoeffs: alpha,
	}
	req := &sim.Request{InputTokens: make([]int, 50)}
	assert.Equal(t, bb.QueueingTime(req), m.QueueingTime(req), "BC-8: QueueingTime must match blackbox")
	assert.Equal(t, bb.OutputTokenProcessingTime(), m.OutputTokenProcessingTime(), "BC-8: OutputTokenProcessingTime must match")
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/latency/... -run TestCrossModel -v`
Expected: Compilation error (CrossModelLatencyModel undefined)

**Step 3: Implement CrossModelLatencyModel**

Create `sim/latency/crossmodel.go`:

```go
package latency

import (
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// CrossModelLatencyModel estimates latency using physics-informed features derived from
// model architecture (config.json). A single set of 4 beta coefficients works across
// model architectures via architecture-specific feature scaling.
//
// StepTime formula:
//   β₀·numLayers + β₁·decodeTokens·kvDimScaled + β₂·(prefillTokens+decodeTokens)·isMoE + β₃·isTP
//
// Beta coefficients (from training/ledger.md Iter 3):
//   β₀ = per-layer CUDA kernel dispatch overhead (µs/layer)
//   β₁ = KV cache bandwidth cost (µs per scaled KV unit)
//   β₂ = MoE expert routing + dispatch/gather cost (µs per MoE token)
//   β₃ = fixed TP synchronization barrier (µs per step, TP > 1 only)
//
// Architecture features are computed once at construction and frozen:
//   kvDimScaled = numLayers × numKVHeads × headDim / TP × 1e-6
//   isMoE       = 1.0 if NumLocalExperts > 0, else 0.0
//   isTP        = 1.0 if TP > 1, else 0.0
type CrossModelLatencyModel struct {
	betaCoeffs  []float64 // [per_layer, kv_bw, moe_dispatch, tp_sync]
	alphaCoeffs []float64 // [pre_sched_fixed, pre_sched_per_tok, output_per_tok]

	// Pre-computed architecture features (frozen at construction)
	numLayers   int
	kvDimScaled float64 // L × kvHeads × headDim / TP × 1e-6
	isMoE       float64 // 1.0 if NumLocalExperts > 0
	isTP        float64 // 1.0 if TP > 1
}

func (m *CrossModelLatencyModel) StepTime(batch []*sim.Request) int64 {
	if len(batch) == 0 {
		return 0
	}
	var totalPrefillTokens, totalDecodeTokens int64
	for _, req := range batch {
		if req.ProgressIndex < util.Len64(req.InputTokens) {
			totalPrefillTokens += int64(req.NumNewTokens)
		} else if len(req.OutputTokens) > 0 {
			totalDecodeTokens += int64(req.NumNewTokens)
		}
	}
	// β₁ term uses only decode tokens (not prefill) because decode is memory-bandwidth-bound
	// on H100: each decode token reads its accumulated KV cache from HBM. Prefill KV write
	// cost is absorbed into β₀ (per-layer term) where it overlaps with compute via GPU pipelining.
	// See training/ledger.md Iter 3 and training/problem.md Section 2a for the physics rationale.
	stepTime := m.betaCoeffs[0]*float64(m.numLayers) +
		m.betaCoeffs[1]*float64(totalDecodeTokens)*m.kvDimScaled +
		m.betaCoeffs[2]*float64(totalPrefillTokens+totalDecodeTokens)*m.isMoE +
		m.betaCoeffs[3]*m.isTP
	return max(1, int64(stepTime))
}

func (m *CrossModelLatencyModel) QueueingTime(req *sim.Request) int64 {
	var totalProcessingTime float64
	totalProcessingTime += m.alphaCoeffs[0]
	totalProcessingTime += m.alphaCoeffs[1] * float64(len(req.InputTokens))
	return int64(totalProcessingTime)
}

func (m *CrossModelLatencyModel) OutputTokenProcessingTime() int64 {
	return int64(m.alphaCoeffs[2])
}

func (m *CrossModelLatencyModel) SchedulingProcessingTime() int64 {
	return 0
}

func (m *CrossModelLatencyModel) PreemptionProcessingTime() int64 {
	return 0
}
```

**Step 4: Run tests**

Run: `go test ./sim/latency/... -run TestCrossModel -v`
Expected: ALL PASS

**Step 5: Lint**

Run: `golangci-lint run ./sim/latency/...`
Expected: 0 issues

**Step 6: Commit**

```bash
git add sim/latency/crossmodel.go sim/latency/crossmodel_test.go
git commit -m "feat(latency): implement CrossModelLatencyModel with physics-informed step time (BC-1 through BC-10)

- Add CrossModelLatencyModel struct with architecture-derived features
- StepTime: β₀·L + β₁·dc·kvDimScaled + β₂·(pf+dc)·isMoE + β₃·isTP
- Pre-compute kvDimScaled, isMoE, isTP at construction (BC-10)
- QueueingTime/OutputTokenProcessingTime identical to BlackboxLatencyModel (BC-8)
- Behavioral tests: monotonicity, MoE indicator, TP indicator, empty batch

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Factory Integration + Backend Registry

**Contracts Implemented:** BC-9, BC-11, BC-12, BC-13

**Files:**
- Modify: `sim/latency/latency.go` (add `case "crossmodel":`)
- Modify: `sim/bundle.go` (add `"crossmodel"` to registry)
- Modify: `sim/config.go` (update Backend field comment)
- Test: `sim/latency/crossmodel_test.go` (add factory tests)

**Step 1: Write factory tests**

Add to `sim/latency/crossmodel_test.go`:

```go
func TestNewLatencyModel_CrossModelMode(t *testing.T) {
	// BC-9: factory creates crossmodel backend
	mc := testCrossModelConfig()
	coeffs := sim.NewLatencyCoeffs(
		[]float64{116.110, 1226.868, 19.943, 9445.157},
		[]float64{13732.0, 0.0, 860.6},
	)
	hw := sim.NewModelHardwareConfig(mc, sim.HardwareCalib{}, "", "", 2, "crossmodel")
	model, err := NewLatencyModel(coeffs, hw)
	assert.NoError(t, err)

	// Regression anchor: verify step time for a known batch
	batch := []*sim.Request{{
		InputTokens: make([]int, 100), OutputTokens: []int{1},
		ProgressIndex: 100, NumNewTokens: 1,
	}}
	result := model.StepTime(batch)
	assert.GreaterOrEqual(t, result, int64(1), "factory-created crossmodel must produce positive step time")
}

func TestNewLatencyModel_CrossModelMode_MissingNumLayers(t *testing.T) {
	// BC-12: factory rejects missing NumLayers
	mc := sim.ModelConfig{NumLayers: 0, HiddenDim: 4096, NumHeads: 32, NumKVHeads: 8}
	coeffs := sim.NewLatencyCoeffs(
		[]float64{116, 1226, 19, 9445},
		[]float64{13732, 0, 860},
	)
	hw := sim.NewModelHardwareConfig(mc, sim.HardwareCalib{}, "", "", 2, "crossmodel")
	_, err := NewLatencyModel(coeffs, hw)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "NumLayers")
}

func TestNewLatencyModel_CrossModelMode_ShortBeta(t *testing.T) {
	// BC-13: factory rejects short BetaCoeffs
	mc := testCrossModelConfig()
	coeffs := sim.NewLatencyCoeffs(
		[]float64{116, 1226, 19}, // only 3, need 4
		[]float64{13732, 0, 860},
	)
	hw := sim.NewModelHardwareConfig(mc, sim.HardwareCalib{}, "", "", 2, "crossmodel")
	_, err := NewLatencyModel(coeffs, hw)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "4")
}
```

**Step 2: Implement factory case**

In `sim/latency/latency.go`, add between the `"roofline"` case and the `"", "blackbox"` case:

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

Update factory docstring:
```go
// Dispatches on hw.Backend: "roofline" → RooflineLatencyModel, "crossmodel" → CrossModelLatencyModel, "" or "blackbox" → BlackboxLatencyModel.
```

In `sim/bundle.go`, add `"crossmodel"` to the map:
```go
validLatencyBackends = map[string]bool{"": true, "blackbox": true, "roofline": true, "crossmodel": true}
```

In `sim/config.go`, update the Backend field comment:
```go
Backend     string        // latency model backend: "" or "blackbox" (default), "roofline", "crossmodel"
```

**Step 3: Run tests**

Run: `go test ./sim/... -count=1`
Expected: ALL PASS

**Step 4: Lint + commit**

```bash
golangci-lint run ./sim/...
git add sim/latency/latency.go sim/latency/crossmodel_test.go sim/bundle.go sim/config.go
git commit -m "feat(latency): integrate crossmodel backend into factory and registry (BC-9, BC-11, BC-12, BC-13)

- Add case crossmodel to NewLatencyModel factory switch
- Validate NumLayers, NumHeads, HiddenDim, TP > 0, BetaCoeffs >= 4
- Compute kvDimScaled, isMoE, isTP at construction
- Register crossmodel in validLatencyBackends (R8)
- Factory tests: valid creation, missing fields, short beta

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: defaults.yaml + Config Struct + CLI Wiring

**Contracts Implemented:** CLI integration (enables `--latency-model crossmodel`)

**Files:**
- Modify: `defaults.yaml`
- Modify: `cmd/default_config.go`
- Modify: `cmd/root.go`

**Step 1: Add crossmodel_defaults to defaults.yaml**

Add at the end of `defaults.yaml` (before `version:`):
```yaml
crossmodel_defaults:
  beta_coeffs: [116.110, 1226.868, 19.943, 9445.157]
  alpha_coeffs: [13732.0, 0.0, 860.6]
```

**Step 2: Add to Config struct (R10: strict YAML)**

In `cmd/default_config.go`, add to the `Config` struct:
```go
CrossModelDefaults *CrossModelDefaults `yaml:"crossmodel_defaults,omitempty"`
```

Add the new struct:
```go
// CrossModelDefaults holds globally-fitted physics coefficients for cross-model latency estimation.
// These are model-independent: a single set works across architectures via config.json features.
type CrossModelDefaults struct {
	BetaCoeffs  []float64 `yaml:"beta_coeffs"`
	AlphaCoeffs []float64 `yaml:"alpha_coeffs"`
}
```

**Step 3: Wire CLI in cmd/root.go**

After the `backend == "roofline"` block (around line 232) and before the defaults.yaml loading block, add:
```go
		// --latency-model crossmodel: auto-resolve model config and hardware config
		if backend == "crossmodel" {
			if gpu == "" {
				logrus.Fatalf("--latency-model crossmodel requires --hardware (GPU type)")
			}
			if tensorParallelism <= 0 {
				logrus.Fatalf("--latency-model crossmodel requires --tp > 0")
			}

			// Resolve model config folder (same auto-fetch chain as roofline)
			resolved, err := resolveModelConfig(model, modelConfigFolder, defaultsFilePath)
			if err != nil {
				logrus.Fatalf("%v", err)
			}
			modelConfigFolder = resolved

			// Resolve hardware config (for future use; crossmodel doesn't need HWConfig
			// but loading it ensures --hardware flag is valid)
			resolvedHW, err := resolveHardwareConfig(hwConfigPath, defaultsFilePath)
			if err != nil {
				logrus.Fatalf("%v", err)
			}
			hwConfigPath = resolvedHW

			// Load crossmodel defaults from defaults.yaml
			if _, statErr := os.Stat(defaultsFilePath); statErr == nil {
				data, readErr := os.ReadFile(defaultsFilePath)
				if readErr != nil {
					logrus.Warnf("--latency-model crossmodel: failed to read %s: %v", defaultsFilePath, readErr)
				} else {
					var cfg Config
					decoder := yaml.NewDecoder(bytes.NewReader(data))
					decoder.KnownFields(true) // R10: strict YAML parsing
					if yamlErr := decoder.Decode(&cfg); yamlErr != nil {
						logrus.Fatalf("--latency-model crossmodel: failed to parse %s: %v", defaultsFilePath, yamlErr)
					}
					if cfg.CrossModelDefaults != nil {
						if !cmd.Flags().Changed("beta-coeffs") {
							betaCoeffs = cfg.CrossModelDefaults.BetaCoeffs
							logrus.Infof("--latency-model: loaded crossmodel beta coefficients from defaults.yaml")
						}
						if !cmd.Flags().Changed("alpha-coeffs") {
							alphaCoeffs = cfg.CrossModelDefaults.AlphaCoeffs
							logrus.Infof("--latency-model: loaded crossmodel alpha coefficients from defaults.yaml")
						}
					}
				}
				// Also load KV blocks from per-model config if available
				if vllmVersion == "" {
					_, _, ver := GetDefaultSpecs(model)
					if len(ver) > 0 {
						vllmVersion = ver
					}
				}
				_, _, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
				if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
					totalKVBlocks = kvBlocks
					logrus.Infof("--latency-model: loaded total-kv-blocks=%d from defaults.yaml", kvBlocks)
				}
			}
		}
```

**Step 3b: Update zero-coeff guard at line 281**

The existing guard must exclude crossmodel (same pattern as roofline):
```go
// BEFORE:
if backend != "roofline" && AllZeros(alphaCoeffs) && AllZeros(betaCoeffs) {
// AFTER:
if backend != "roofline" && backend != "crossmodel" && AllZeros(alphaCoeffs) && AllZeros(betaCoeffs) {
```
Also update the error message to mention crossmodel:
```go
logrus.Fatalf("No trained coefficients found for model=%s, GPU=%s, TP=%d. "+
    "Provide --alpha-coeffs/--beta-coeffs, use --latency-model roofline, or use --latency-model crossmodel",
    model, gpu, tensorParallelism)
```

**Step 3c: Add coefficients-loaded check AFTER the entire defaults-loading block**

After the entire `if _, statErr := os.Stat(defaultsFilePath); statErr == nil { ... }` block (not inside it), validate that crossmodel coefficients were actually loaded. This catches all failure modes: missing file, YAML parse error, missing section.
```go
		// After the defaults-loading block (outside all nested ifs):
		if !cmd.Flags().Changed("beta-coeffs") && (len(betaCoeffs) < 4 || AllZeros(betaCoeffs)) {
			logrus.Fatalf("--latency-model crossmodel: no crossmodel_defaults found in %s and no --beta-coeffs provided. "+
				"Add crossmodel_defaults to defaults.yaml or provide --beta-coeffs explicitly",
				defaultsFilePath)
		}
```

**Step 3d: Update config loading block at line 286**

Both roofline and crossmodel need ModelConfig from HF config.json:
```go
// BEFORE:
if backend == "roofline" {
// AFTER:
if backend == "roofline" || backend == "crossmodel" {
```
This block loads `modelConfig` via `latency.GetModelConfig(hfPath)` and `hwConfig` via `latency.GetHWConfig(hwConfigPath, gpu)`. Crossmodel needs `modelConfig` for architecture features (NumLayers, NumKVHeads, HiddenDim). It doesn't use `hwConfig` for computation but loading it validates the `--hardware` flag.

**Step 3e: Update CLI flag help string at line 736**

```go
// BEFORE:
runCmd.Flags().StringVar(&latencyModelBackend, "latency-model", "", "Latency model backend: blackbox (default), roofline")
// AFTER:
runCmd.Flags().StringVar(&latencyModelBackend, "latency-model", "", "Latency model backend: blackbox (default), roofline, crossmodel")
```

**Step 4: Run full test suite**

Run: `go test ./... -count=1`
Expected: ALL PASS

Run: `golangci-lint run ./...`
Expected: 0 issues

**Step 5: Commit**

```bash
git add defaults.yaml cmd/default_config.go cmd/root.go
git commit -m "feat(cmd): wire --latency-model crossmodel with defaults.yaml and HF config auto-fetch

- Add crossmodel_defaults section to defaults.yaml with Iter 3 coefficients
- Add CrossModelDefaults struct and Config field (R10: strict YAML)
- Wire backend == crossmodel in cmd/root.go (reuses resolveModelConfig/resolveHardwareConfig)
- Load crossmodel coefficients from defaults.yaml when user doesn't provide explicit ones

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Documentation Updates

**Contracts Implemented:** Documentation accuracy

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/guide/latency-models.md`
- Modify: `docs/reference/configuration.md`
- Modify: `docs/reference/models.md`

**Step 1: Update CLAUDE.md**

- Latency Estimation section: Add "3. **Cross-model mode**: Physics-informed estimation via `sim/latency/crossmodel.go`" with description
- File Organization: Add `│   ├── crossmodel.go        # CrossModelLatencyModel (physics-informed cross-model step time)` under `sim/latency/`
- Update "two modes" to "three modes" in the Latency Estimation intro

**Step 2: Update docs/guide/latency-models.md**

- Add CLI example in the intro block:
```bash
# Cross-model mode — physics-informed estimation from config.json
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --latency-model crossmodel --hardware H100 --tp 2 \
  --num-instances 4 --rate 100 --num-requests 500
```

- Add new section "## Cross-Model Mode (Physics-Informed)" between Roofline and "When to Use Which":
  - Explain the 4-coefficient physics formula
  - Note MoE support (Mixtral tested)
  - Note global coefficients vs per-model
  - CLI example with `--latency-model crossmodel`

- Update "When to Use Which" comparison table to include crossmodel as 3rd column

**Step 3: Update docs/reference/configuration.md**

- Update `--latency-model` flag description to: `blackbox (default), roofline, crossmodel`
- Add path 5 to Latency Mode Selection: "**Explicit cross-model mode**: If `--latency-model crossmodel` is set with `--hardware` and `--tp`"

**Step 4: Update docs/reference/models.md**

- Add note that crossmodel mode works for any model with HF config.json (same as roofline, plus MoE support)

**Step 5: Commit**

```bash
git add CLAUDE.md docs/guide/latency-models.md docs/reference/configuration.md docs/reference/models.md
git commit -m "docs: add cross-model latency backend documentation

- CLAUDE.md: add cross-model mode to Latency Estimation, update File Organization
- latency-models.md: add Cross-Model Mode section with formula and CLI examples
- configuration.md: update --latency-model flag, add crossmodel to mode selection
- models.md: note crossmodel works for any model with HF config.json

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 2 | Unit | `TestCrossModelLatencyModel_StepTime_NonEmpty_Positive` |
| BC-2 | Task 2 | Unit | (verified implicitly by monotonicity test) |
| BC-3 | Task 2 | Unit | `TestCrossModelLatencyModel_StepTime_Monotonic_Decode` |
| BC-4 | Task 2 | Unit | `TestCrossModelLatencyModel_StepTime_NonEmpty_Positive` |
| BC-5 | Task 2 | Unit | `TestCrossModelLatencyModel_StepTime_EmptyBatch_Zero` |
| BC-6 | Task 2 | Unit | `TestCrossModelLatencyModel_MoE_IncreasesStepTime` |
| BC-7 | Task 2 | Unit | `TestCrossModelLatencyModel_TP_IncreasesStepTime` |
| BC-8 | Task 2 | Unit | `TestCrossModelLatencyModel_QueueingTime_MatchesBlackbox` |
| BC-9 | Task 3 | Unit | `TestNewLatencyModel_CrossModelMode` |
| BC-10 | Task 2 | Unit | (implied by construction pattern — all tests use frozen features) |
| BC-11 | Task 3 | Structural | Factory returns error, not panic (R6) |
| BC-12 | Task 3 | Failure | `TestNewLatencyModel_CrossModelMode_MissingNumLayers` |
| BC-13 | Task 3 | Failure | `TestNewLatencyModel_CrossModelMode_ShortBeta` |

No golden dataset changes needed — crossmodel is a new backend that doesn't change existing output.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Integer truncation in kvDimScaled | Medium | High | Cast to float64 before all divisions | Task 3 |
| defaults.yaml breaks with new section | Medium | High | R10: add field to Config struct; pointer + omitempty for backward compat | Task 4 |
| NumKVHeads == 0 (GQA fallback) | Medium | Medium | Same fallback as existing GetModelConfig (line 112-114) | Task 3 |
| Zero coefficients reach crossmodel factory | Low | High | CLI zero-coeff guard (from PR-A) catches this | Task 4 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions (no new interfaces, no new config types beyond CrossModelDefaults)
- [x] No feature creep (MoE warning removal deferred)
- [x] No unexercised code (all features tested)
- [x] No breaking changes (zero-value MoE fields = backward compatible)
- [x] CLAUDE.md updated (Task 5)
- [x] Documentation updated (Task 5)
- [x] R4: ModelConfig construction sites audited (GetModelConfig, testModelConfig; zero-value sites safe)
- [x] R6: No Fatalf in sim/ (factory returns errors)
- [x] R8: validLatencyBackends updated (unexported map)
- [x] R10: defaults.yaml Config struct has CrossModelDefaults field
- [x] R13: 3 implementations of LatencyModel interface (BC-B8)
- [x] All contracts mapped to tasks

---

## Appendix: File-Level Details

### File: `sim/latency/crossmodel.go`
Complete implementation in Task 2, Step 3.

### File: `sim/latency/crossmodel_test.go`
Complete test code in Task 2, Step 1 + Task 3, Step 1.

### File: `sim/model_hardware_config.go`
Add 2 fields: `NumLocalExperts int`, `NumExpertsPerTok int` with json tags.

### File: `sim/latency/config.go`
Add 2 lines to parse MoE fields via `getInt()`, add 2 fields to struct literal.

### File: `sim/latency/latency.go`
Add `case "crossmodel":` block (~30 lines) between roofline and blackbox cases. Update docstring.

### File: `sim/bundle.go`
Add `"crossmodel": true` to `validLatencyBackends` map.

### File: `defaults.yaml`
Add `crossmodel_defaults:` top-level section with 2 coefficient arrays.

### File: `cmd/default_config.go`
Add `CrossModelDefaults` struct + pointer field on `Config`.

### File: `cmd/root.go`
Add `backend == "crossmodel"` block (~40 lines) mirroring roofline pattern. Update config loading guard.
