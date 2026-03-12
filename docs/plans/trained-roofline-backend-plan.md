# Trained-Roofline Latency Backend — Micro Plan

- **Goal:** Add a `trained-roofline` latency model backend that applies learned correction factors to analytical roofline basis functions, fitted from real vLLM traces across 4 model architectures.
- **The problem today:** BLIS has two latency model families: (1) blackbox, which is per-model and requires per-model coefficient training, and (2) crossmodel, which uses hand-engineered features with manually-fitted coefficients from Iter 3 (4 beta + 3 alpha). The `training/` pipeline has produced a principled 10-parameter model (7 beta + 3 alpha) fitted via NNLS from 13 experiments across 4 architectures, using roofline basis functions as analytical priors. There is no backend in BLIS that can consume these coefficients.
- **What this PR adds:**
  1. A new `TrainedRooflineLatencyModel` backend in `sim/latency/` implementing the 7-term step-time formula from `training/DESIGN.md`
  2. Registration as `--latency-model trained-roofline` with defaults loaded from `defaults.yaml`
  3. Six analytical basis functions (ported from `training/basis_functions.py`) computing prefill/decode roofline, weight loading, TP communication, per-layer overhead, and scheduling overhead
  4. Same auto-fetch chain as roofline/crossmodel (HuggingFace config + hardware config resolution)
- **Why this matters:** The trained-roofline model achieves 7% MAPE on GPU combined step time (test split) across llama-2-7b, llama-2-70b, mixtral-8x7b, and codellama-34b — fitted from 137K real vLLM requests. It provides a principled foundation (roofline prior + learned corrections) that constrains extrapolation to unseen models.
- **Architecture:** New file `sim/latency/trained_roofline.go` containing the struct, 6 basis functions, and LatencyModel interface implementation. Factory dispatch added to existing `NewLatencyModel` in `latency.go`. CLI handling follows the crossmodel pattern in `cmd/root.go`. Coefficients in `defaults.yaml` under `trained_roofline_defaults`.
- **Source:** Discussion of `training/` pipeline output, `training/DESIGN.md`, `training/output/fit/coefficients.json`.
- **Closes:** (new issue TBD — or standalone PR)
- **Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block:** New latency model backend (policy template behind existing `LatencyModel` interface) — the simplest extension type per design guidelines Section 5.
2. **Adjacent blocks:** `sim.LatencyModel` interface (3 methods), `NewLatencyModel` factory in `sim/latency/latency.go`, `validLatencyBackends` map in `sim/bundle.go`, `cmd/root.go` CLI handling (3 touch points: trained-roofline block, zero-coefficients guard at ~line 351, analytical HFConfig parsing block at ~line 358), `defaults.yaml` coefficient storage. **Adjacent documentation:** `docs/guide/latency-models.md`, `docs/reference/configuration.md`, `docs/concepts/core-engine.md`, `docs/concepts/glossary.md`.
3. **Invariants touched:** INV-3 (clock monotonicity) — StepTime must return >= 1. INV-6 (determinism) — no randomness in basis functions.
4. **Construction Site Audit:**
   - `HardwareCalib` — NOT modified (no NVLink field added; see Deviation Log).
   - `ModelHardwareConfig` — NOT modified.
   - `LatencyCoeffs` — NOT modified (uses existing BetaCoeffs/AlphaCoeffs slices).
   - `validLatencyBackends` — modified (add entry; single construction site in `sim/bundle.go:65`).
   - `Config` (cmd/default_config.go) — modified (add field; single construction site — YAML deserialization).

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds the fourth latency model backend to BLIS: `trained-roofline`. It implements the 7-term step-time formula from the `training/` pipeline, which applies learned correction coefficients (β₁-β₇) to analytical roofline basis functions. The basis functions compute prefill/decode compute/bandwidth bottlenecks, weight loading time, TP communication time, per-layer overhead, and scheduling overhead — all from model architecture (`config.json`) and hardware specs (`hardware_config.json`). The factory dispatch in `NewLatencyModel` gains a `"trained-roofline"` case. Coefficients load from `defaults.yaml` via the same pattern as `crossmodel`. One method added to the existing `LatencyModel` interface (`PostDecodeFixedOverhead()`) — existing backends return 0 (backward compatible). No new interfaces created, no struct field additions to existing types, no behavioral changes to existing backends.

**Deviation flag:** This PR adds a `PostDecodeFixedOverhead()` method to the `LatencyModel` interface to correctly model α₁ (post-decode fixed overhead) at request completion. Existing backends return 0. See Deviation Log.

### B) Behavioral Contracts

**Positive contracts (what MUST happen):**

```
BC-1: Backend Registration
- GIVEN the simulator binary is built
- WHEN IsValidLatencyBackend("trained-roofline") is called
- THEN it returns true
- MECHANISM: entry in validLatencyBackends map
```

```
BC-2: Factory Construction
- GIVEN LatencyCoeffs with BetaCoeffs of length >= 7, AlphaCoeffs of length >= 3,
  and ModelHardwareConfig with Backend="trained-roofline", valid ModelConfig, valid HardwareCalib, TP > 0
- WHEN NewLatencyModel is called
- THEN it returns a non-nil LatencyModel and nil error
```

```
BC-3: Step-Time Formula
- GIVEN a TrainedRooflineLatencyModel and a batch of requests
- WHEN StepTime(batch) is called
- THEN the result equals β₁·max(T_pf_compute, T_pf_kv) + β₂·max(T_dc_compute, T_dc_kv)
  + β₃·T_weight + β₄·T_tp + β₅·L + β₆·batchSize + β₇, rounded to int64, floored at 1
- MECHANISM: six basis functions computed from model architecture + hardware specs + batch composition
```

```
BC-4: Prefill Monotonicity
- GIVEN a model with fixed architecture and a batch with only prefill requests
- WHEN more prefill tokens are added (increasing T_pf)
- THEN StepTime is non-decreasing
```

```
BC-5: Decode Monotonicity
- GIVEN a model with fixed architecture and a batch with only decode requests
- WHEN more decode requests are added (increasing context length sum)
- THEN StepTime is non-decreasing
```

```
BC-6: Clock Safety (INV-3)
- GIVEN any inputs (including empty batch)
- WHEN StepTime is called
- THEN the result is >= 1
```

```
BC-7: QueueingTime Mapping
- GIVEN AlphaCoeffs [α₀, α₁, α₂]
- WHEN QueueingTime(req) is called
- THEN the result equals int64(α₀)
- MECHANISM: α₀ is the API processing overhead (ARRIVED→QUEUED), constant per-request
```

```
BC-8: OutputTokenProcessingTime Mapping
- GIVEN AlphaCoeffs [α₀, α₁, α₂]
- WHEN OutputTokenProcessingTime() is called
- THEN the result equals int64(α₂)
```

```
BC-15: PostDecodeFixedOverhead Mapping
- GIVEN AlphaCoeffs [α₀, α₁, α₂]
- WHEN PostDecodeFixedOverhead() is called
- THEN the result equals int64(α₁)
- MECHANISM: α₁ is the fixed per-request post-decode overhead (FINISHED→DEPARTED), added
  to E2E in recordRequestCompletion. This new LatencyModel interface method enables correct
  modeling without rolling α₁ into QueueingTime (which would inflate TTFT).
```

```
BC-9: MoE-Aware Weight Loading
- GIVEN a MoE model (NumLocalExperts > 0) and a batch with B total tokens
- WHEN StepTime is called
- THEN the weight loading basis function uses min(N, max(k, B*k)) effective experts
  (not all N experts as in the pure roofline backend)
```

```
BC-10: Defaults Loading
- GIVEN defaults.yaml with a trained_roofline_defaults section
- WHEN --latency-model trained-roofline is used without --beta-coeffs/--alpha-coeffs
- THEN coefficients are loaded from the trained_roofline_defaults section
```

**Negative contracts (what MUST NOT happen):**

```
BC-11: No MFU Scaling
- GIVEN a TrainedRooflineLatencyModel
- WHEN basis functions compute prefill/decode FLOPs → time
- THEN they divide by raw peak FLOPS (TFlopsPeak), NOT by TFlopsPeak * MfuPrefill/MfuDecode
- MECHANISM: β₁ and β₂ are the MFU corrections; applying MfuPrefill/MfuDecode would double-count
```

```
BC-12: Existing Backend Isolation
- GIVEN any existing backend (blackbox, roofline, crossmodel)
- WHEN --latency-model <existing> is used
- THEN behavior is byte-identical to before this PR
```

**Error handling contracts:**

```
BC-13: Coefficient Length Validation
- GIVEN BetaCoeffs with fewer than 7 elements
- WHEN NewLatencyModel is called with Backend="trained-roofline"
- THEN it returns a non-nil error describing the minimum requirement
```

```
BC-14: Config Validation
- GIVEN ModelConfig with NumHeads <= 0, or HiddenDim <= 0, or NumLayers <= 0,
  or IntermediateDim <= 0, or NumHeads not divisible by TP,
  or NumKVHeads not divisible by TP,
  or TFlopsPeak invalid (<=0, NaN, Inf), or BwPeakTBs invalid (<=0, NaN, Inf)
- WHEN NewLatencyModel is called with Backend="trained-roofline"
- THEN it returns a non-nil error for each invalid field
```

### C) Component Interaction

```
cmd/root.go (CLI)
    │  --latency-model trained-roofline --hardware H100 --tp 2
    │  Resolves model config (auto-fetch), hardware config, loads coefficients
    ▼
sim/latency/latency.go  NewLatencyModel()
    │  Dispatches on hw.Backend == "trained-roofline"
    │  Validates: 7 betas, 3 alphas, ModelConfig fields, TP > 0
    ▼
sim/latency/trained_roofline.go  TrainedRooflineLatencyModel
    │  Pre-computes: headDim, dKV, kEff, isMoE (frozen at construction)
    │  StepTime(batch):
    │    1. Classify requests → prefill/decode
    │    2. Compute 6 basis functions from model arch + hardware + batch
    │    3. Return β·X formula, floored at 1
    │  QueueingTime(req): int64(α₀) — API processing overhead
    │  OutputTokenProcessingTime(): int64(α₂) — per-token detokenization
    │  PostDecodeFixedOverhead(): int64(α₁) — fixed post-decode overhead (NEW method)
    ▼
sim.LatencyModel interface (4 methods after this PR)
    │  Used by sim.Simulator.executeBatchStep() for step time + per-token output processing
    │  Used by sim.Simulator.EnqueueRequest() for queueing overhead
    │  Used by sim.Simulator.recordRequestCompletion() for post-decode fixed overhead (NEW)
    ▼
sim.Simulator (event loop)

Data flow for StepTime:
  batch []*Request → classify prefill/decode → per-phase FLOPs + bytes
  → 6 basis values (µs) → 7-term β formula → int64 step time (µs)

State ownership:
  - TrainedRooflineLatencyModel: owns frozen architecture features + coefficient copies
  - No mutable state, no clock dependency, no external side effects
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Training pipeline has 3 separate alpha parameters: α₀ (pre-queue), α₁ (post-decode fixed), α₂ (per-output-token) | Adds `PostDecodeFixedOverhead()` method to LatencyModel interface; returns α₁. QueueingTime = α₀ only (constant, API overhead). OutputTokenProcessingTime = α₂ (per-token detokenization). | ADDITION — The existing LatencyModel interface lacks a per-request completion overhead method. This PR adds `PostDecodeFixedOverhead() int64` to the interface. Existing backends (blackbox, roofline, crossmodel) return 0 (backward compatible — their α model has no fixed post-decode component). Trained-roofline returns int64(α₁) = 1849. The simulator's `recordRequestCompletion` adds this to E2E: `lat = FirstTokenTime + itlSum + PostDecodeFixedOverhead()`. E2E is correct. TTFT is correct (no inflation). QueueingTime is constant (α₀ only, not input-proportional — differs from other backends which compute α₀ + α₁*inputLen). |
| Training basis_functions.py computes T_tp from NVLink bandwidth | T_tp always returns 0 when NVLink data unavailable | SIMPLIFICATION — β₄=0.0 in the fitted coefficients, so T_tp is multiplied by zero. HardwareCalib does not have NVLink bandwidth and hardware_config.json doesn't include it. Adding it would require R4 construction site audit for a zero-multiplied term. When β₄ becomes nonzero in future refitting, NVLink support can be added. |
| Training uses `prompt_tokens` (total input length) as the attention context for prefill | Uses `len(req.InputTokens)` to match training semantics exactly | SCOPE_CHANGE — Physically, ProgressIndex would be more correct for chunked prefill. But the training data is entirely single-step prefill where prompt_tokens == tokens_this_step. Using len(InputTokens) maintains coefficient compatibility. |
| Training uses 3-matrix SwiGLU: 6·d·d_ff for FFN FLOPs AND 3·d·d_ff for weight loading bytes | Same (6·d·d_ff for FLOPs, 3·d·d_ff for weights), differs from roofline.go's 2-matrix (4·d·d_ff FLOPs, 2·d·d_ff weights via `mlpMatrixCount()=2`) | CORRECTION — The coefficients were fitted against 3-matrix SwiGLU formulas for both compute and memory. Using 2-matrix for either would invalidate β₁/β₂/β₃. Implementation must include cross-reference comment to roofline.go's `mlpMatrixCount()` explaining why the difference exists (R23 documented exception). |
| Training uses `d_ff = cfg["intermediate_size"]` uniformly for all models | Uses `dFF = IntermediateDim` (not MoEExpertFFNDim) | SCOPE_CHANGE — For the 4 training models, `intermediate_size` IS the per-expert FFN dim. Models like Qwen2-MoE where `intermediate_size != moe_intermediate_size` will produce incorrect basis functions. This is a known limitation until the training data includes such models. |

### E) Review Guide

**Tricky part:** The basis function formulas must match `training/basis_functions.py` EXACTLY. The fitted coefficients are only valid with the exact features they were trained on. Pay close attention to: projection formula (2·d + 2·d_kv includes O_proj), attention FLOPs formula (s_i = total prompt tokens, NOT ProgressIndex), FFN FLOPs (k_eff = max(1, k), uses 6·d·d_ff not 4), weight loading MoE formula (min(N, max(k, B·k))), and the max(compute, memory) bottleneck per phase.

**Scrutinize:** BC-11 (no MFU scaling — the existing roofline backend applies MFU, but trained-roofline must NOT), BC-7 (α₁ roll-in semantics), and any division that could produce NaN/Inf.

**Safe to skim:** CLI wiring (follows identical crossmodel pattern), backend registration (one-line map entry), documentation updates.

**Known debt:** T_tp is hardcoded to 0 (β₄=0.0 makes this correct but not general; TP communication cost is absorbed into β₅·L, making it H100-NVLink-specific). QueueingTime is constant (α₀ only, not input-proportional — differs from other backends). The attention FLOPs formula overpredicts for chunked prefill (matches training data; future refit with chunked data would fix). **TTFT accuracy caveat:** The "7% MAPE" headline applies to GPU combined step time only. The alpha model has 93% MAPE (pre-queueing) and 54% MAPE (post-decode) — TTFT predictions will have significantly higher error than GPU step time. MoE models where `IntermediateDim != MoEExpertFFNDim` (e.g., Qwen2-MoE) will produce incorrect basis functions.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/latency/trained_roofline.go` — TrainedRooflineLatencyModel struct, 6 basis functions, 3 interface methods
- `sim/latency/trained_roofline_test.go` — behavioral tests

**Files to modify:**
- `sim/latency_model.go` — add `PostDecodeFixedOverhead() int64` method to LatencyModel interface
- `sim/simulator.go` — add `+ sim.latencyModel.PostDecodeFixedOverhead()` to `recordRequestCompletion` (~line 337)
- `sim/latency/latency.go` — add `"trained-roofline"` case to NewLatencyModel switch; add `PostDecodeFixedOverhead()` returning 0 to BlackboxLatencyModel and RooflineLatencyModel
- `sim/latency/crossmodel.go` — add `PostDecodeFixedOverhead()` returning 0 to CrossModelLatencyModel
- `sim/bundle.go` — add `"trained-roofline": true` to validLatencyBackends (line ~65)
- `sim/config.go` — update `Backend` field comment to include "trained-roofline" (~line 81)
- `defaults.yaml` — add `trained_roofline_defaults` section
- `cmd/default_config.go` — add `TrainedRooflineDefaults` struct + field on Config
- `cmd/root.go` — 4 modification sites: loading block, zero-coefficients guard, HFConfig parsing block, help text
- `CLAUDE.md` — update latency estimation (4th mode), file tree, Key Data Flow
- `docs/guide/latency-models.md` — add trained-roofline section + update comparison table
- `docs/reference/configuration.md` — update flag description + defaults.yaml section
- `docs/concepts/core-engine.md` — update "Latency Models" subsection
- `docs/concepts/glossary.md` — update "Latency Model" entry
- `docs/index.md` — update feature bullet
- `docs/reference/models.md` — mention trained-roofline in modes section

**Key decisions:**
- One method added to existing `LatencyModel` interface: `PostDecodeFixedOverhead()` — existing backends return 0 (backward compatible)
- No MFU scaling in basis functions (β₁/β₂ are the corrections)
- T_tp = 0 always (β₄=0.0, no NVLink data)
- α₁ modeled via PostDecodeFixedOverhead() at request completion (not rolled into QueueingTime)
- Formulas ported from training/basis_functions.py verbatim for coefficient compatibility

### G) Task Breakdown

#### Task 1: Register "trained-roofline" backend name (BC-1)

**Contracts:** BC-1

**Test (failing):**

File: `sim/bundle_test.go` — add to existing test functions:

```go
// In TestIsValidLatencyBackend:
assert.True(t, IsValidLatencyBackend("trained-roofline"))

// In TestValidLatencyBackendNames:
assert.Contains(t, names, "trained-roofline")
```

**Command:** `cd .worktrees/trained-roofline-backend && go test ./sim/... -run TestIsValidLatencyBackend -count=1`
**Expected:** FAIL (trained-roofline not in map)

**Implement:**

File: `sim/bundle.go` line ~65 — add to validLatencyBackends map:

```go
validLatencyBackends = map[string]bool{"": true, "blackbox": true, "roofline": true, "crossmodel": true, "trained-roofline": true}
```

**Command:** `cd .worktrees/trained-roofline-backend && go test ./sim/... -run "TestIsValidLatencyBackend|TestValidLatencyBackendNames" -count=1`
**Expected:** PASS

**Lint:** `cd .worktrees/trained-roofline-backend && golangci-lint run ./sim/...`

**Commit:** `feat(latency): register trained-roofline backend name (BC-1)`

---

#### Task 2: Add PostDecodeFixedOverhead to interface + Implement TrainedRooflineLatencyModel (BC-3, BC-6, BC-7, BC-8, BC-9, BC-11, BC-15)

**Contracts:** BC-3, BC-6, BC-7, BC-8, BC-9, BC-11, BC-15

**Pre-step: Interface addition (must compile before tests)**

1. File: `sim/latency_model.go` — add to LatencyModel interface:
```go
// PostDecodeFixedOverhead estimates the fixed per-request post-processing overhead (µs).
// This is the constant overhead at request completion (e.g., response setup, final API processing).
// Added for trained-roofline alpha model: α₁ = fixed post-decode overhead per request.
// Existing backends return 0.
PostDecodeFixedOverhead() int64
```

2. File: `sim/simulator.go` — in `recordRequestCompletion` (~line 337), change:
```go
lat := req.FirstTokenTime + itlSum
```
to:
```go
lat := req.FirstTokenTime + itlSum + sim.latencyModel.PostDecodeFixedOverhead()
```

3. File: `sim/latency/latency.go` — add to BlackboxLatencyModel and RooflineLatencyModel:
```go
func (m *BlackboxLatencyModel) PostDecodeFixedOverhead() int64 { return 0 }
func (m *RooflineLatencyModel) PostDecodeFixedOverhead() int64 { return 0 }
```

4. File: `sim/latency/crossmodel.go` — add:
```go
func (m *CrossModelLatencyModel) PostDecodeFixedOverhead() int64 { return 0 }
```

5. Verify: `go build ./... && go test ./... -count=1` — all existing tests must pass (BC-12).

**Test (failing):**

File: `sim/latency/trained_roofline_test.go` — create with behavioral tests:

```go
package latency

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- Test helpers ---

// llama7bConfig returns a ModelConfig approximating Llama-2-7b (TP=1).
func llama7bConfig() sim.ModelConfig {
	return sim.ModelConfig{
		NumLayers:       32,
		HiddenDim:       4096,
		NumHeads:        32,
		NumKVHeads:      32,
		IntermediateDim: 11008,
		BytesPerParam:   2,
	}
}

// mixtral8x7bConfig returns a ModelConfig approximating Mixtral-8x7B (MoE).
func mixtral8x7bConfig() sim.ModelConfig {
	return sim.ModelConfig{
		NumLayers:        32,
		HiddenDim:        4096,
		NumHeads:         32,
		NumKVHeads:       8,
		IntermediateDim:  14336,
		NumLocalExperts:  8,
		NumExpertsPerTok: 2,
		BytesPerParam:    2,
	}
}

// h100HWConfig returns HardwareCalib for H100 SXM.
func h100HWConfig() sim.HardwareCalib {
	return sim.HardwareCalib{
		TFlopsPeak: 989.5,
		BwPeakTBs:  3.35,
		MfuPrefill: 0.45,
		MfuDecode:  0.30,
		MemoryGiB:  80.0,
	}
}

// trainingFittedBetas returns the β₁-β₇ from training/output/fit/coefficients.json.
var trainingFittedBetas = []float64{
	0.7726491335309499,  // β₁: prefill roofline correction
	1.127489556719325,   // β₂: decode roofline correction
	1.0559901872766853,  // β₃: weight loading correction
	0.0,                 // β₄: TP communication (zeroed)
	43.500541908701074,  // β₅: per-layer overhead (µs/layer)
	48.80613214319187,   // β₆: per-request scheduling (µs/req)
	0.0,                 // β₇: per-step overhead (zeroed)
}

// trainingFittedAlphas returns the α₀-α₂ from training/output/fit/coefficients.json.
var trainingFittedAlphas = []float64{
	9315.338771116985,   // α₀: API processing overhead
	1849.5902371340574,  // α₁: post-decode fixed
	1.7079389122469397,  // α₂: per-output-token
}

func makePrefillRequest(inputLen int, newTokens int) *sim.Request {
	req := &sim.Request{
		InputTokens:   make([]int, inputLen),
		ProgressIndex: 0,
		NumNewTokens:  newTokens,
	}
	return req
}

func makeDecodeRequest(inputLen int, outputSoFar int) *sim.Request {
	req := &sim.Request{
		InputTokens:   make([]int, inputLen),
		OutputTokens:  make([]int, outputSoFar),
		ProgressIndex: int64(inputLen + outputSoFar),
		NumNewTokens:  1,
	}
	return req
}

// --- BC-6: Clock safety ---

func TestTrainedRoofline_EmptyBatch_ReturnsOne(t *testing.T) {
	model := &TrainedRooflineLatencyModel{
		betaCoeffs: trainingFittedBetas,
		alphaCoeffs: trainingFittedAlphas,
		numLayers: 32, hiddenDim: 4096, numHeads: 32,
		headDim: 128, dKV: 4096, dFF: 11008,
		kEff: 1, tp: 1,
		flopsPeakUs: 989.5e6, bwHbmUs: 3.35e6,
	}
	assert.Equal(t, int64(1), model.StepTime(nil))
	assert.Equal(t, int64(1), model.StepTime([]*sim.Request{}))
}

// --- BC-3: Step-time formula ---

func TestTrainedRoofline_PrefillOnly_PositiveStepTime(t *testing.T) {
	model := &TrainedRooflineLatencyModel{
		betaCoeffs: trainingFittedBetas,
		alphaCoeffs: trainingFittedAlphas,
		numLayers: 32, hiddenDim: 4096, numHeads: 32,
		headDim: 128, dKV: 4096, dFF: 11008,
		kEff: 1, tp: 1,
		flopsPeakUs: 989.5e6, bwHbmUs: 3.35e6,
	}
	batch := []*sim.Request{makePrefillRequest(512, 512)}
	stepTime := model.StepTime(batch)
	assert.Greater(t, stepTime, int64(1), "prefill step time should be > 1 µs")
}

func TestTrainedRoofline_DecodeOnly_PositiveStepTime(t *testing.T) {
	model := &TrainedRooflineLatencyModel{
		betaCoeffs: trainingFittedBetas,
		alphaCoeffs: trainingFittedAlphas,
		numLayers: 32, hiddenDim: 4096, numHeads: 32,
		headDim: 128, dKV: 4096, dFF: 11008,
		kEff: 1, tp: 1,
		flopsPeakUs: 989.5e6, bwHbmUs: 3.35e6,
	}
	batch := []*sim.Request{makeDecodeRequest(512, 100)}
	stepTime := model.StepTime(batch)
	assert.Greater(t, stepTime, int64(1), "decode step time should be > 1 µs")
}

// --- BC-7: QueueingTime = α₀ only (API processing overhead) ---

func TestTrainedRoofline_QueueingTime_IsAlpha0(t *testing.T) {
	model := &TrainedRooflineLatencyModel{
		alphaCoeffs: []float64{9315.0, 1850.0, 1.71},
	}
	// QueueingTime = α₀ (constant, independent of input length)
	req512 := makePrefillRequest(512, 512)
	req1024 := makePrefillRequest(1024, 1024)
	assert.Equal(t, int64(9315), model.QueueingTime(req512))
	assert.Equal(t, int64(9315), model.QueueingTime(req1024),
		"QueueingTime must be constant (α₀ only), independent of input length")
}

// --- BC-8: OutputTokenProcessingTime = α₂ (per-token detokenization) ---

func TestTrainedRoofline_OutputTokenProcessingTime_IsAlpha2(t *testing.T) {
	model := &TrainedRooflineLatencyModel{
		alphaCoeffs: []float64{9315.0, 1850.0, 1.71},
	}
	assert.Equal(t, int64(1), model.OutputTokenProcessingTime())
}

// --- BC-15: PostDecodeFixedOverhead = α₁ (fixed per-request post-decode) ---

func TestTrainedRoofline_PostDecodeFixedOverhead_IsAlpha1(t *testing.T) {
	model := &TrainedRooflineLatencyModel{
		alphaCoeffs: []float64{9315.0, 1850.0, 1.71},
	}
	assert.Equal(t, int64(1850), model.PostDecodeFixedOverhead())
}

// --- BC-9: MoE-aware weight loading ---

func TestTrainedRoofline_MoE_WeightLoading_UsesEffectiveExperts(t *testing.T) {
	// For MoE with N=8 experts, k=2 active, batch of 1 token:
	// n_eff = min(8, max(2, 1*2)) = min(8, 2) = 2
	// For batch of 10 tokens: n_eff = min(8, max(2, 10*2)) = min(8, 20) = 8
	denseBetas := []float64{0, 0, 1.0, 0, 0, 0, 0} // only β₃ (weight loading) nonzero
	alphas := []float64{0, 0, 0}

	modelSmall := &TrainedRooflineLatencyModel{
		betaCoeffs: denseBetas, alphaCoeffs: alphas,
		numLayers: 32, hiddenDim: 4096, numHeads: 32,
		headDim: 128, dKV: 1024, dFF: 14336,
		kEff: 2, numExperts: 8, isMoE: true, tp: 1,
		flopsPeakUs: 989.5e6, bwHbmUs: 3.35e6,
	}
	modelLarge := &TrainedRooflineLatencyModel{
		betaCoeffs: denseBetas, alphaCoeffs: alphas,
		numLayers: 32, hiddenDim: 4096, numHeads: 32,
		headDim: 128, dKV: 1024, dFF: 14336,
		kEff: 2, numExperts: 8, isMoE: true, tp: 1,
		flopsPeakUs: 989.5e6, bwHbmUs: 3.35e6,
	}

	// 1 decode token → n_eff=2; 10 decode tokens → n_eff=8 (all experts)
	smallBatch := []*sim.Request{makeDecodeRequest(100, 10)}
	largeBatch := make([]*sim.Request, 10)
	for i := range largeBatch {
		largeBatch[i] = makeDecodeRequest(100, 10)
	}

	smallTime := modelSmall.StepTime(smallBatch)
	largeTime := modelLarge.StepTime(largeBatch)

	// Weight loading should increase with more effective experts
	assert.Greater(t, largeTime, smallTime,
		"larger batch should load more MoE experts → higher weight loading time")
}

// --- BC-11: No MFU scaling regression anchor ---

func TestTrainedRoofline_NoMfuScaling_RegressionAnchor(t *testing.T) {
	// Construct a model where we can hand-compute the expected FLOPs.
	// Architecture: 1 layer, d=128, H=1, kv_heads=1, d_h=128, d_kv=128, d_ff=1024, TP=1
	// 1 prefill token, s_i=1 (len(InputTokens)=1), t_i=1 (NumNewTokens=1)
	//
	// FLOPs_proj = 1 * 2 * 1 * 128 * (2*128 + 2*128) / 1 = 2 * 128 * 512 = 131072
	// FLOPs_attn = 1 * 4 * 1 * 1 * (1 + 0.5) * 128 = 768
	// FLOPs_ffn  = 1 * 1 * 1 * 6 * 128 * 1024 / 1 = 786432
	// Total = 131072 + 768 + 786432 = 918272 FLOPs
	// T_pf_compute = 918272 / (1.0e6 FLOP/µs) = 0.918272 µs
	// bw very high → T_pf_kv ≈ 0 → max(T_pf_compute, T_pf_kv) = 0.918272
	// β₁=1.0 → step time = 0.918272 → int64 = 0, floored to 1
	//
	// If MFU were applied (MfuPrefill=0.45): 918272 / (0.45e6) = 2.040 µs → int64 = 2
	// Without MFU: int64 result = 1. With MFU: int64 result = 2.

	betas := []float64{1.0, 0, 0, 0, 0, 0, 0} // only β₁ nonzero
	alphas := []float64{0, 0, 0}

	model := &TrainedRooflineLatencyModel{
		betaCoeffs: betas, alphaCoeffs: alphas,
		numLayers: 1, hiddenDim: 128, numHeads: 1,
		headDim: 128, dKV: 128,
		dFF: 1024, kEff: 1, tp: 1,
		flopsPeakUs: 1.0e6, // 1 TFLOP/s → easy calculation
		bwHbmUs: 1e12,      // very high BW → compute-bound
	}

	req := makePrefillRequest(1, 1)
	st := model.StepTime([]*sim.Request{req})

	// Without MFU: FLOPs/peak gives ~0.92 µs → floored to 1
	// With MFU (0.45): FLOPs/(peak*0.45) gives ~2.04 µs → int64 = 2
	assert.Equal(t, int64(1), st,
		"step time should be 1 (no MFU scaling); if MFU were applied it would be 2")
}
```

**Command:** `cd .worktrees/trained-roofline-backend && go test ./sim/latency/... -run TestTrainedRoofline -count=1`
**Expected:** FAIL (TrainedRooflineLatencyModel undefined)

**Implement:**

File: `sim/latency/trained_roofline.go` — create with full implementation.

The struct holds frozen architecture features (computed at construction) and coefficient copies. The 6 basis functions are methods on the struct for access to architecture fields.

Step-time formula:
```
StepTime = β₁·max(T_pf_compute, T_pf_kv)
         + β₂·max(T_dc_compute, T_dc_kv)
         + β₃·T_weight
         + β₄·T_tp
         + β₅·numLayers
         + β₆·batchSize
         + β₇
```

**Struct definition:**

```go
type TrainedRooflineLatencyModel struct {
	betaCoeffs  []float64 // [β₁..β₇] from trained_roofline_defaults
	alphaCoeffs []float64 // [α₀, α₁, α₂]

	// Pre-computed architecture features (frozen at construction)
	numLayers  int
	hiddenDim  int     // d (hidden_size)
	numHeads   int     // H (num_attention_heads)
	headDim    int     // d_h = d / H
	dKV        int     // kv_heads * d_h (NOT d; differs for GQA)
	dFF        int     // intermediate_size (= IntermediateDim, NOT MoEExpertFFNDim)
	kEff       int     // max(1, NumExpertsPerTok) — FFN FLOPs multiplier
	numExperts int     // NumLocalExperts (0 for dense)
	isMoE      bool    // NumLocalExperts > 0
	tp         int     // tensor parallelism degree

	// Pre-converted hardware specs for per-call efficiency
	flopsPeakUs float64 // TFlopsPeak × 1e6 → FLOP/µs (divide FLOPs by this → µs)
	bwHbmUs     float64 // BwPeakTBs × 1e6 → bytes/µs (divide bytes by this → µs)
}
```

**Derived field formulas (computed in factory):**
- `headDim = HiddenDim / NumHeads`
- `dKV = numKVHeads * headDim` (where numKVHeads defaults to NumHeads if 0)
- `dFF = IntermediateDim` (NOT MoEExpertFFNDim — matches training's `d_ff = cfg["intermediate_size"]`)
- `kEff = max(1, NumExpertsPerTok)` (1 for dense models)
- `flopsPeakUs = TFlopsPeak * 1e6` (989.5 TFLOPS → 989.5e6 FLOP/µs)
- `bwHbmUs = BwPeakTBs * 1e6` (3.35 TB/s → 3.35e6 bytes/µs)

**Basis function formulas (ported from training/basis_functions.py):**

All use FP16 (2 bytes per element). No MFU scaling (BC-11). **All arithmetic must use float64** throughout to prevent integer overflow for large models (e.g., 70B: `L*2*T_pf*d*(2*d+2*dKV)` can exceed int64 in intermediate products). Cast all int struct fields to float64 before multiplication.

**T_pf_compute** — prefill compute time (µs):
```
For each prefill request i: t_i = NumNewTokens, s_i = len(InputTokens)
T_pf = Σ t_i (total prefill tokens)
FLOPs_proj = L * 2 * T_pf * d * (2*d + 2*dKV) / TP
FLOPs_attn = L * Σᵢ 4 * (H/TP) * t_i * (s_i + t_i/2) * d_h    ← PER-REQUEST loop
FLOPs_ffn  = L * T_pf * kEff * 6 * d * dFF / TP
result = (FLOPs_proj + FLOPs_attn + FLOPs_ffn) / flopsPeakUs
```

**T_pf_kv** — prefill KV write bandwidth (µs):
```
bytes = L * 2 * (kvHeads/TP) * d_h * T_pf * 2
result = bytes / bwHbmUs
```
Where `kvHeads/TP = dKV / d_h / TP` (integer division, must be exact).

**T_dc_compute** — decode compute time (µs):
```
T_dc = number of decode requests (each generates 1 token)
sum_ctx = Σⱼ req.ProgressIndex    ← context_length maps to ProgressIndex in BLIS
FLOPs_proj = L * 2 * T_dc * d * (2*d + 2*dKV) / TP
FLOPs_attn = L * 4 * (H/TP) * sum_ctx * d_h
FLOPs_ffn  = L * T_dc * kEff * 6 * d * dFF / TP
result = (FLOPs_proj + FLOPs_attn + FLOPs_ffn) / flopsPeakUs
```

**T_dc_kv** — decode KV read+write bandwidth (µs):
```
bytes = L * 2 * (kvHeads/TP) * d_h * 2 * (sum_ctx + T_dc)
result = bytes / bwHbmUs
```

**T_weight** — weight loading time (µs):
```
For dense: nEff = 1
For MoE: B = T_pf + T_dc; nEff = min(numExperts, max(kEff, B*kEff))
bytes_attn = L * d * (2*d + 2*dKV) * 2 / TP
bytes_ffn  = L * nEff * 3 * d * dFF * 2 / TP    ← 3 matrices (SwiGLU), NOT mlpMatrixCount()=2
result = (bytes_attn + bytes_ffn) / bwHbmUs
```

**T_tp** — TP communication (µs): returns 0.0 always (β₄=0.0, no NVLink data).

**Key implementation notes:**
- **Empty-batch guard:** `if len(batch) == 0 || batch == nil { return 1 }` — MUST be first line of StepTime. Without this, batch-independent terms (β₃·T_weight + β₅·L) produce ~5.5ms for an empty batch (physically wrong). Matches roofline.go's early return pattern.
- **Single-pass accumulation:** StepTime iterates batch ONCE, accumulating: `totalPrefillTokens`, `totalDecodeTokens`, `sumCtx` (decode ProgressIndex sum), `prefillAttentionFlops` (per-request sum), and `batchSize = len(batch)`. Then computes each basis function in O(1) from these aggregates. Zero heap allocations — all arithmetic uses stack-local float64 values.
- `batchSize = len(batch)` — total requests in the step (prefill + decode)
- Attention `s_i = len(req.InputTokens)` matches training's `entry.prompt_tokens` (total prompt, NOT ProgressIndex). See Deviation 3.
- Decode `context_length` maps to `req.ProgressIndex` — the canonical position tracker in BLIS (= inputLen + outputSoFar)
- T_tp returns 0.0 always (no NVLink data; β₄=0.0)
- `dFF = IntermediateDim` (NOT MoEExpertFFNDim) — matches training pipeline's `d_ff = cfg["intermediate_size"]`

**Command:** `cd .worktrees/trained-roofline-backend && go test ./sim/latency/... -run TestTrainedRoofline -count=1`
**Expected:** PASS

**Lint:** `cd .worktrees/trained-roofline-backend && golangci-lint run ./sim/latency/...`

**Commit:** `feat(latency): implement TrainedRooflineLatencyModel with 6 basis functions (BC-3,6,7,8,9,11)`

---

#### Task 3: Wire factory in NewLatencyModel (BC-2, BC-13, BC-14)

**Contracts:** BC-2, BC-13, BC-14

**Test (failing):**

File: `sim/latency/trained_roofline_test.go` — add factory tests:

```go
// --- BC-2: Factory construction ---

func TestNewLatencyModel_TrainedRoofline_ReturnsModel(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs(trainingFittedBetas, trainingFittedAlphas)
	hw := sim.ModelHardwareConfig{
		Backend:     "trained-roofline",
		ModelConfig: llama7bConfig(),
		HWConfig:    h100HWConfig(),
		TP:          1,
	}
	model, err := NewLatencyModel(coeffs, hw)
	require.NoError(t, err)
	require.NotNil(t, model)

	// Verify it produces positive step times
	batch := []*sim.Request{makePrefillRequest(512, 512)}
	assert.Greater(t, model.StepTime(batch), int64(1))
}

// --- BC-13: Coefficient length validation ---

func TestNewLatencyModel_TrainedRoofline_TooFewBetas_Error(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs([]float64{1, 2, 3, 4, 5, 6}, trainingFittedAlphas) // only 6 betas
	hw := sim.ModelHardwareConfig{
		Backend:     "trained-roofline",
		ModelConfig: llama7bConfig(),
		HWConfig:    h100HWConfig(),
		TP:          1,
	}
	_, err := NewLatencyModel(coeffs, hw)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "7 elements")
}

// --- BC-14: Config validation ---

func TestNewLatencyModel_TrainedRoofline_InvalidConfig_Error(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs(trainingFittedBetas, trainingFittedAlphas)

	tests := []struct {
		name   string
		modify func(*sim.ModelHardwareConfig)
		errMsg string
	}{
		{"zero TP", func(hw *sim.ModelHardwareConfig) { hw.TP = 0 }, "TP > 0"},
		{"zero NumLayers", func(hw *sim.ModelHardwareConfig) { hw.ModelConfig.NumLayers = 0 }, "NumLayers > 0"},
		{"zero NumHeads", func(hw *sim.ModelHardwareConfig) { hw.ModelConfig.NumHeads = 0 }, "NumHeads > 0"},
		{"zero HiddenDim", func(hw *sim.ModelHardwareConfig) { hw.ModelConfig.HiddenDim = 0 }, "HiddenDim > 0"},
		{"zero TFlopsPeak", func(hw *sim.ModelHardwareConfig) { hw.HWConfig.TFlopsPeak = 0 }, "TFlopsPeak"},
		{"zero BwPeakTBs", func(hw *sim.ModelHardwareConfig) { hw.HWConfig.BwPeakTBs = 0 }, "BwPeakTBs"},
		{"zero IntermediateDim", func(hw *sim.ModelHardwareConfig) { hw.ModelConfig.IntermediateDim = 0 }, "IntermediateDim > 0"},
		{"NaN TFlopsPeak", func(hw *sim.ModelHardwareConfig) { hw.HWConfig.TFlopsPeak = math.NaN() }, "TFlopsPeak"},
		{"Inf BwPeakTBs", func(hw *sim.ModelHardwareConfig) { hw.HWConfig.BwPeakTBs = math.Inf(1) }, "BwPeakTBs"},
		{"NumKVHeads not divisible by TP", func(hw *sim.ModelHardwareConfig) { hw.ModelConfig.NumKVHeads = 5; hw.TP = 2 }, "divisible"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			hw := sim.ModelHardwareConfig{
				Backend:     "trained-roofline",
				ModelConfig: llama7bConfig(),
				HWConfig:    h100HWConfig(),
				TP:          1,
			}
			tc.modify(&hw)
			_, err := NewLatencyModel(coeffs, hw)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), tc.errMsg)
		})
	}
}
```

**Command:** `cd .worktrees/trained-roofline-backend && go test ./sim/latency/... -run "TestNewLatencyModel_TrainedRoofline" -count=1`
**Expected:** FAIL (no "trained-roofline" case in NewLatencyModel)

**Implement:**

File: `sim/latency/latency.go` — add `case "trained-roofline":` to the switch in `NewLatencyModel`:

```go
case "trained-roofline":
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
	if hw.ModelConfig.NumHeads%hw.TP != 0 {
		return nil, fmt.Errorf("latency model: trained-roofline requires NumHeads (%d) divisible by TP (%d)", hw.ModelConfig.NumHeads, hw.TP)
	}
	numKVHeads := hw.ModelConfig.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = hw.ModelConfig.NumHeads
	}
	if numKVHeads%hw.TP != 0 {
		return nil, fmt.Errorf("latency model: trained-roofline requires NumKVHeads (%d) divisible by TP (%d)", numKVHeads, hw.TP)
	}
	if invalidPositiveFloat(hw.HWConfig.TFlopsPeak) {
		return nil, fmt.Errorf("latency model: trained-roofline requires valid TFlopsPeak > 0, got %v", hw.HWConfig.TFlopsPeak)
	}
	if invalidPositiveFloat(hw.HWConfig.BwPeakTBs) {
		return nil, fmt.Errorf("latency model: trained-roofline requires valid BwPeakTBs > 0, got %v", hw.HWConfig.BwPeakTBs)
	}
	if hw.ModelConfig.IntermediateDim <= 0 {
		return nil, fmt.Errorf("latency model: trained-roofline requires IntermediateDim > 0, got %d", hw.ModelConfig.IntermediateDim)
	}
	if len(coeffs.BetaCoeffs) < 7 {
		return nil, fmt.Errorf("latency model: trained-roofline BetaCoeffs requires at least 7 elements, got %d", len(coeffs.BetaCoeffs))
	}
	if err := validateCoeffs("BetaCoeffs", coeffs.BetaCoeffs); err != nil {
		return nil, err
	}
	headDim := hw.ModelConfig.HiddenDim / hw.ModelConfig.NumHeads
	dKV := numKVHeads * headDim
	dFF := hw.ModelConfig.IntermediateDim
	kEff := max(1, hw.ModelConfig.NumExpertsPerTok) // matches training: k_eff = max(1, k)
	return &TrainedRooflineLatencyModel{
		betaCoeffs:  coeffs.BetaCoeffs,
		alphaCoeffs: coeffs.AlphaCoeffs,
		numLayers:   hw.ModelConfig.NumLayers,
		hiddenDim:   hw.ModelConfig.HiddenDim,
		numHeads:    hw.ModelConfig.NumHeads,
		headDim:     headDim,
		dKV:         dKV,
		dFF:         dFF,
		kEff:        kEff,
		numExperts:  hw.ModelConfig.NumLocalExperts,
		isMoE:       hw.ModelConfig.NumLocalExperts > 0,
		tp:          hw.TP,
		flopsPeakUs: hw.HWConfig.TFlopsPeak * 1e6,
		bwHbmUs:     hw.HWConfig.BwPeakTBs * 1e6,
	}, nil
```

Also add TP divisibility tests to `TestNewLatencyModel_TrainedRoofline_InvalidConfig_Error`:
```go
{"NumHeads not divisible by TP", func(hw *sim.ModelHardwareConfig) { hw.TP = 3 }, "divisible"},
```

**Command:** `cd .worktrees/trained-roofline-backend && go test ./sim/latency/... -run "TestNewLatencyModel_TrainedRoofline" -count=1`
**Expected:** PASS

**Lint:** `cd .worktrees/trained-roofline-backend && golangci-lint run ./sim/latency/...`

**Commit:** `feat(latency): wire trained-roofline factory in NewLatencyModel (BC-2,13,14)`

---

#### Task 4: Behavioral monotonicity tests (BC-4, BC-5)

**Contracts:** BC-4, BC-5

**Test:**

File: `sim/latency/trained_roofline_test.go` — add monotonicity tests:

```go
// --- BC-4: Prefill monotonicity ---

func TestTrainedRoofline_PrefillMonotonicity(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs(trainingFittedBetas, trainingFittedAlphas)
	hw := sim.ModelHardwareConfig{
		Backend: "trained-roofline", ModelConfig: llama7bConfig(),
		HWConfig: h100HWConfig(), TP: 1,
	}
	model, err := NewLatencyModel(coeffs, hw)
	require.NoError(t, err)

	tokenCounts := []int{64, 128, 256, 512, 1024}
	var prevTime int64
	for _, n := range tokenCounts {
		batch := []*sim.Request{makePrefillRequest(n, n)}
		st := model.StepTime(batch)
		assert.GreaterOrEqual(t, st, prevTime,
			"prefill step time should be non-decreasing with more tokens: %d tokens → %d µs (prev %d µs)", n, st, prevTime)
		prevTime = st
	}
}

// --- BC-5: Decode monotonicity ---

func TestTrainedRoofline_DecodeMonotonicity(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs(trainingFittedBetas, trainingFittedAlphas)
	hw := sim.ModelHardwareConfig{
		Backend: "trained-roofline", ModelConfig: llama7bConfig(),
		HWConfig: h100HWConfig(), TP: 1,
	}
	model, err := NewLatencyModel(coeffs, hw)
	require.NoError(t, err)

	// More decode requests → higher batch size → more KV reads → longer step
	var prevTime int64
	for nReqs := 1; nReqs <= 16; nReqs *= 2 {
		batch := make([]*sim.Request, nReqs)
		for i := range batch {
			batch[i] = makeDecodeRequest(512, 100)
		}
		st := model.StepTime(batch)
		assert.GreaterOrEqual(t, st, prevTime,
			"decode step time should be non-decreasing with more requests: %d reqs → %d µs (prev %d µs)", nReqs, st, prevTime)
		prevTime = st
	}
}
```

**Command:** `cd .worktrees/trained-roofline-backend && go test ./sim/latency/... -run "TestTrainedRoofline_.*Monotonicity" -count=1`
**Expected:** PASS (implementation from Task 2/3)

**Commit:** `test(latency): add monotonicity behavioral tests for trained-roofline (BC-4,5)`

---

#### Task 5: Add defaults + CLI loading (BC-10, BC-12)

**Contracts:** BC-10, BC-12

**Implement:**

1. File: `defaults.yaml` — add `trained_roofline_defaults` section (after `crossmodel_defaults`):

```yaml
trained_roofline_defaults:
  # Globally-fitted roofline correction coefficients from training/output/fit/coefficients.json.
  # Fitted via 3-phase NNLS from 13 experiments (4 models × 3-4 profiles, 137K requests).
  # β₁-β₄ are dimensionless roofline corrections, β₅ is µs/layer, β₆ is µs/req, β₇ is µs/step.
  beta_coeffs: [0.7726491335309499, 1.127489556719325, 1.0559901872766853, 0.0, 43.500541908701074, 48.80613214319187, 0.0]
  # α₀=API processing overhead (µs), α₁=post-decode fixed (µs), α₂=per-output-token detokenization (µs/tok).
  alpha_coeffs: [9315.338771116985, 1849.5902371340574, 1.7079389122469397]
```

2. File: `cmd/default_config.go` — add struct and field:

```go
// TrainedRooflineDefaults holds globally-fitted roofline correction coefficients.
// These are model-independent: analytical basis functions from config.json scale the coefficients.
type TrainedRooflineDefaults struct {
	BetaCoeffs  []float64 `yaml:"beta_coeffs"`
	AlphaCoeffs []float64 `yaml:"alpha_coeffs"`
}
```

Add to Config struct: `TrainedRooflineDefaults *TrainedRooflineDefaults \`yaml:"trained_roofline_defaults,omitempty"\``

3. File: `cmd/root.go` — 4 modification sites:

   **Site A: Trained-roofline loading block (after crossmodel block, ~line 302).** Add a new `if backend == "trained-roofline"` block. Same structure as crossmodel: require `--hardware` and `--tp > 0`, resolve model config, resolve hardware config, load `cfg.TrainedRooflineDefaults` from defaults.yaml using `Flags().Changed()` guards (R18), validate coefficients were loaded (analogous to crossmodel line 297-301):
   ```go
   if !cmd.Flags().Changed("beta-coeffs") && (len(betaCoeffs) < 7 || allZeros(betaCoeffs)) {
       logrus.Fatalf("--latency-model trained-roofline: no trained_roofline_defaults found in %s and no --beta-coeffs provided. "+
           "Add trained_roofline_defaults to defaults.yaml or provide --beta-coeffs explicitly",
           defaultsFilePath)
   }
   ```

   **Site B: Zero-coefficients safety guard (~line 351).** Add `&& backend != "trained-roofline"` to prevent false-positive fatal error:
   ```go
   if backend != "roofline" && backend != "crossmodel" && backend != "trained-roofline" && allZeros(alphaCoeffs) && allZeros(betaCoeffs) {
   ```
   Also update the error message at ~line 352-354 to mention `trained-roofline`.

   **Site C: Analytical backends HFConfig parsing (~line 358).** Add `|| backend == "trained-roofline"`:
   ```go
   if backend == "roofline" || backend == "crossmodel" || backend == "trained-roofline" {
   ```

   **Site D: Help text (~line 972).** Update `--latency-model` flag description:
   ```go
   "Latency model backend: blackbox (default), roofline, crossmodel, trained-roofline"
   ```

**Test:** Verify existing backends unchanged (BC-12):

```bash
cd .worktrees/trained-roofline-backend && go test ./... -count=1
```

**Expected:** All existing tests PASS (no behavioral change to existing backends).

**Lint:** `cd .worktrees/trained-roofline-backend && golangci-lint run ./...`

**Commit:** `feat(latency): add trained-roofline defaults + CLI loading (BC-10,12)`

---

#### Task 6: Update documentation

**Contracts:** (documentation only)

**Implement:**

1. File: `CLAUDE.md` — update:
   - Latency Estimation section: change "Three modes" to "Four modes", add trained-roofline as 4th mode with description
   - File Organization: add `trained_roofline.go` to `sim/latency/` listing
   - Key Data Flow: update parenthetical to "(alpha/beta, roofline, cross-model, or trained-roofline)"

2. File: `sim/latency/latency.go` — update package doc comment to mention TrainedRooflineLatencyModel

3. File: `sim/config.go` — update `Backend` field comment on `ModelHardwareConfig` (~line 81) to include `"trained-roofline"`

4. File: `docs/guide/latency-models.md` — add "Trained-Roofline Mode" section:
   - Description: roofline basis functions × learned correction coefficients from real vLLM traces
   - Usage: `./blis run --model <name> --latency-model trained-roofline --hardware <GPU> --tp <N>`
   - Update comparison table and mode count
   - Note: coefficients fitted from 137K requests across 4 architectures (7% MAPE on GPU combined)

5. File: `docs/reference/configuration.md` — update `--latency-model` flag description, add `trained_roofline_defaults` YAML section

6. File: `docs/concepts/core-engine.md` — update "Latency Models" section: change "one of three" to "one of four", add trained-roofline subsection

7. File: `docs/concepts/glossary.md` — update "Latency Model" entry: change "Three modes" to "Four modes"

8. File: `docs/index.md` — update feature bullet to include "trained-roofline"

9. File: `docs/reference/models.md` — mention trained-roofline in "Roofline and Cross-Model Modes" section

**Command:** `cd .worktrees/trained-roofline-backend && go build ./... && go test ./... -count=1`
**Expected:** PASS

**Commit:** `docs: add trained-roofline to latency model documentation`

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | TestIsValidLatencyBackend (updated) |
| BC-1 | Task 1 | Unit | TestValidLatencyBackendNames (updated) |
| BC-2 | Task 3 | Unit | TestNewLatencyModel_TrainedRoofline_ReturnsModel |
| BC-3 | Task 2 | Unit | TestTrainedRoofline_PrefillOnly_PositiveStepTime |
| BC-3 | Task 2 | Unit | TestTrainedRoofline_DecodeOnly_PositiveStepTime |
| BC-4 | Task 4 | Behavioral | TestTrainedRoofline_PrefillMonotonicity |
| BC-5 | Task 4 | Behavioral | TestTrainedRoofline_DecodeMonotonicity |
| BC-6 | Task 2 | Unit | TestTrainedRoofline_EmptyBatch_ReturnsOne |
| BC-7 | Task 2 | Unit | TestTrainedRoofline_QueueingTime_IsAlpha0 |
| BC-8 | Task 2 | Unit | TestTrainedRoofline_OutputTokenProcessingTime_IsAlpha2 |
| BC-9 | Task 2 | Behavioral | TestTrainedRoofline_MoE_WeightLoading_UsesEffectiveExperts |
| BC-11 | Task 2 | Regression | TestTrainedRoofline_NoMfuScaling_RegressionAnchor |
| BC-12 | Task 5 | Integration | Full test suite passes (no regressions) |
| BC-13 | Task 3 | Error | TestNewLatencyModel_TrainedRoofline_TooFewBetas_Error |
| BC-14 | Task 3 | Error | TestNewLatencyModel_TrainedRoofline_InvalidConfig_Error |
| BC-15 | Task 2 | Unit | TestTrainedRoofline_PostDecodeFixedOverhead_IsAlpha1 |

**Invariant tests:**
- BC-6 verifies INV-3 (clock monotonicity: StepTime >= 1)
- BC-4/BC-5 verify physical monotonicity (more work → more time)
- BC-12 verifies INV-6 (determinism: no regressions in existing backends)

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Basis function formula mismatch with training pipeline | Medium | High (coefficients invalid) | Regression anchor test + exact formula port from training/basis_functions.py | Task 2 |
| MFU accidentally applied (double-counting) | Low | High (2-3x step time error) | BC-11 test explicitly checks no MFU | Task 2 |
| Division by zero in basis functions | Low | High (panic) | Validate TFlopsPeak > 0, BwPeakTBs > 0 at construction (BC-14) | Task 3 |
| Interface addition (PostDecodeFixedOverhead) | Low | Low (all existing backends return 0; 1-line change to recordRequestCompletion) | Existing backends unaffected (BC-12); trained-roofline returns α₁; verified by BC-15 test | Task 2/3 |
| defaults.yaml strict parsing (R10) rejects new section | Low | Medium (breaks existing tests) | TrainedRooflineDefaults uses `omitempty` tag | Task 5 |
| NaN/Inf in coefficients | Low | High (silent garbage) | validateCoeffs called for both alpha and beta | Task 3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — single new struct, one method added to existing LatencyModel interface (PostDecodeFixedOverhead)
- [x] No feature creep — only trained-roofline backend, no refactoring of existing backends
- [x] No unexercised flags — all 7 betas and 3 alphas are used
- [x] No partial implementations — all 4 LatencyModel methods implemented (StepTime, QueueingTime, OutputTokenProcessingTime, PostDecodeFixedOverhead)
- [x] No breaking changes — existing backends untouched (BC-12)
- [x] No hidden global state — struct is immutable after construction
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used (testify assert/require)
- [x] CLAUDE.md updated — Task 6
- [x] No stale references
- [x] Documentation DRY — CLAUDE.md file tree and latency section updated
- [x] Deviation log reviewed — 6 deviations (α mapping via PostDecodeFixedOverhead, T_tp=0, prompt_tokens semantics, SwiGLU 3-matrix, IntermediateDim vs MoEExpertFFNDim, chunked prefill overprediction), all justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1 → 2 → 3 → 4 → 5 → 6)
- [x] All contracts mapped to tasks
- [x] Golden dataset not affected (new backend, no output changes)
- [x] Construction site audit: validLatencyBackends (1 site, updated in Task 1), Config struct (YAML deserialization, updated in Task 5)

**Antipattern rules:**
- [x] R1: No silent data loss — all error paths return error
- [x] R2: No map iteration for ordered output (basis functions use direct computation)
- [x] R3: All numeric params validated (TP, NumLayers, NumHeads, HiddenDim, TFlopsPeak, BwPeakTBs)
- [x] R4: Construction site audit completed (validLatencyBackends, Config struct)
- [x] R5: No resource allocation loops
- [x] R6: No Fatalf in sim/ — returns error from NewLatencyModel
- [x] R7: Monotonicity tests alongside value tests
- [x] R8: validLatencyBackends remains unexported
- [x] R9: No new YAML pointer types needed
- [x] R10: Config uses existing KnownFields(true) parsing (omitempty on new field)
- [x] R11: Division guarded (TFlopsPeak > 0, BwPeakTBs > 0 at construction)
- [x] R12: No golden dataset changes
- [x] R13: LatencyModel interface now has 4 implementations (was 3)
- [x] R14: StepTime computes step time only (no scheduling/metrics concerns)
- [x] R15: No stale PR references
- [x] R16: Coefficients loaded from existing LatencyCoeffs (module-scoped config)
- [x] R17: N/A (no routing signals)
- [x] R18: CLI uses Flags().Changed() guard (follows crossmodel pattern)
- [x] R19: No retry loops
- [x] R20: Empty batch handled (returns 1)
- [x] R21: No range over mutable slices
- [x] R22: N/A (no pre-checks)
- [x] R23: Basis functions internally consistent. **Documented exception:** trained-roofline uses 3-matrix SwiGLU (6·d·d_ff FLOPs, 3·d·d_ff weights) while roofline.go uses 2-matrix (mlpMatrixCount()=2). This is intentional — coefficients were fitted against 3-matrix formulas. Cross-reference comment in implementation code required.

---

## Appendix: File-Level Implementation Details

### File: `sim/latency/trained_roofline.go` (CREATE)

**Purpose:** TrainedRooflineLatencyModel — applies learned correction factors to analytical roofline basis functions.

**Key implementation notes:**
- StepTime uses single-pass accumulation: one loop classifies prefill/decode and accumulates all aggregate values, then computes 6 basis functions in O(1). Zero heap allocations.
- Empty-batch guard: `if len(batch) == 0 || batch == nil { return 1 }` — first line of StepTime.
- Architecture features frozen at construction (headDim, dKV, dFF, kEff, isMoE, etc.)
- Pre-converted hardware specs: `flopsPeakUs` = TFlopsPeak × 1e6, `bwHbmUs` = BwPeakTBs × 1e6
- FP16 assumed (2 bytes per element) for all KV/weight calculations — matches training pipeline's `_BYTES_PER_ELEMENT = 2`
- No MFU scaling (BC-11)
- T_tp returns 0 (no NVLink data, β₄=0)
- Cross-reference comment for 3-matrix SwiGLU vs roofline.go's `mlpMatrixCount()=2` (R23 exception)

### File: `sim/latency/latency.go` (MODIFY)

**Purpose:** Add `"trained-roofline"` case to NewLatencyModel switch.

**Changes:**
- New case between `"crossmodel"` and `"", "blackbox"` cases
- Validates: TP > 0, NumLayers > 0, NumHeads > 0, HiddenDim > 0, IntermediateDim > 0, TFlopsPeak > 0 (via invalidPositiveFloat — catches NaN/Inf), BwPeakTBs > 0, NumHeads%TP == 0, NumKVHeads%TP == 0
- Validates: BetaCoeffs >= 7 elements
- Calls validateCoeffs for both alpha and beta
- Computes derived features (headDim, dKV, etc.) and constructs TrainedRooflineLatencyModel

### File: `sim/bundle.go` (MODIFY)

**Purpose:** Register "trained-roofline" as valid backend name.

**Changes:** Single line addition to validLatencyBackends map.

### File: `defaults.yaml` (MODIFY)

**Purpose:** Store fitted coefficients from training pipeline.

**Changes:** Add `trained_roofline_defaults` section between `crossmodel_defaults` and `version`.

### File: `cmd/default_config.go` (MODIFY)

**Purpose:** Add YAML deserialization struct for trained-roofline defaults.

**Changes:**
- Add `TrainedRooflineDefaults` struct (BetaCoeffs, AlphaCoeffs)
- Add field to Config struct with `yaml:"trained_roofline_defaults,omitempty"` tag

### File: `cmd/root.go` (MODIFY)

**Purpose:** CLI handling for `--latency-model trained-roofline`.

**Changes (4 sites):**
- **Site A (~line 302):** Add `if backend == "trained-roofline"` loading block with coefficient validation guard
- **Site B (~line 351):** Add `&& backend != "trained-roofline"` to zero-coefficients safety guard + update error message
- **Site C (~line 358):** Add `|| backend == "trained-roofline"` to analytical HFConfig parsing block
- **Site D (~line 972):** Update `--latency-model` help text: `"blackbox (default), roofline, crossmodel, trained-roofline"`

### File: `sim/config.go` (MODIFY)

**Purpose:** Update `Backend` field comment on `ModelHardwareConfig`.

**Changes:** Line ~81: add `"trained-roofline"` to the comment listing valid backend names.

### File: `CLAUDE.md` (MODIFY)

**Purpose:** Document the new backend.

**Changes:**
- Latency Estimation section: "Four modes" + trained-roofline description
- File Organization: add `trained_roofline.go` to sim/latency/ listing
- Key Data Flow parenthetical: add "trained-roofline"

### Files: `docs/guide/latency-models.md`, `docs/reference/configuration.md`, `docs/concepts/core-engine.md`, `docs/concepts/glossary.md`, `docs/index.md`, `docs/reference/models.md` (MODIFY)

**Purpose:** Update all documentation working copies that reference the number of latency backends.

**Changes:** Update mode counts from "three" to "four", add trained-roofline descriptions where appropriate. See Task 6 for per-file details.
