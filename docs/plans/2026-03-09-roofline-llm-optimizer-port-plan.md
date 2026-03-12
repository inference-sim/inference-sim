# Roofline llm-optimizer Physics Port — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace BLIS roofline's dual-ceiling + overhead physics with llm-optimizer's single-crossover model to reduce ITL MAPE from 215% to ~15-25%.

**Architecture:** Only `rooflineStepTime()` changes — the FLOPs and memory calculation functions are untouched. The function keeps its signature, phase separation, and per-request accumulation. Four deletions (dual ceiling, bandwidth haircut, overhead terms) and one update (MFU values in hardware_config.json).

**Tech Stack:** Go, JSON hardware config

---

### Task 1: Simplify rooflineStepTime to single-crossover model

**Files:**
- Modify: `sim/latency/roofline.go:221-292`
- Test: `sim/latency/roofline_test.go`

**Step 1: Write a test that verifies the new physics**

Add to `roofline_test.go` — a test that asserts: for a memory-bound decode step, step time equals `total_bytes / peak_bandwidth` (no haircut, no overhead). This will fail against the current implementation.

```go
func TestRooflineStepTime_SingleCrossover_MemoryBoundDecode(t *testing.T) {
	// llm-optimizer physics: memory-bound step time = total_bytes / peak_bandwidth
	// No bandwidth haircut, no overhead terms, single crossover (not dual ceiling).
	mc := testModelConfig()
	hc := testHardwareCalib()

	// Single decode request — decode is memory-bound on H100
	step := StepConfig{
		DecodeRequests: []DecodeRequestConfig{
			{ProgressIndex: 512, NumNewDecodeTokens: 1},
		},
	}
	result := rooflineStepTime(mc, hc, step, 1)

	// Compute expected: weights + KV + activations, all at raw peak bandwidth
	peakBW := hc.BwPeakTBs * 1e12
	peakFlops := hc.TFlopsPeak * 1e12

	baseMem := calculateMemoryAccessBytes(mc, 0, 0, false)
	dynamicMem := calculateMemoryAccessBytes(mc, 512, 1, true)
	totalBytes := baseMem["model_weights"] + (dynamicMem["total"] - dynamicMem["model_weights"])

	flops := calculateTransformerFlops(mc, 512, 1, true, true)
	totalFlops := flops["total"]

	computeS := totalFlops / (peakFlops * hc.MfuDecode)
	memoryS := totalBytes / peakBW

	// Decode should be memory-bound (verify assumption)
	if computeS >= memoryS {
		t.Skipf("decode is compute-bound with this config, skipping memory-bound test")
	}

	expectedMicros := int64(math.Round(memoryS * 1e6))
	if result != expectedMicros {
		t.Errorf("expected %d µs (total_bytes/peak_bw), got %d µs (delta=%d)",
			expectedMicros, result, result-expectedMicros)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/latency/ -run TestRooflineStepTime_SingleCrossover -v -count=1`
Expected: FAIL — current implementation adds haircut + overhead

**Step 3: Rewrite rooflineStepTime**

Replace lines 221-292 of `sim/latency/roofline.go` with:

```go
// rooflineStepTime computes step latency using the roofline model.
//
// Uses single-crossover roofline (llm-optimizer style): for each phase,
// step_time = max(total_flops / (peak * MFU), total_bytes / peak_bandwidth).
// No bandwidth haircut, no overhead terms.
//
// Precondition: ValidateRooflineConfig(modelConfig, hwConfig) must return nil
// and tp must be > 0. Callers must validate before first call.
func rooflineStepTime(modelConfig sim.ModelConfig, hwConfig sim.HardwareCalib, stepConfig StepConfig, tp int) int64 {

	tpFactor := float64(tp)
	peakFlops := hwConfig.TFlopsPeak * 1e12
	peakBW := hwConfig.BwPeakTBs * 1e12

	var prefillTimeS, decodeTimeS float64

	// 1. PREFILL PHASE
	if len(stepConfig.PrefillRequests) > 0 {
		var pTotalFlops, pDynamicBytes float64

		for _, req := range stepConfig.PrefillRequests {
			numTokens := int64(req.NumNewPrefillTokens)

			f := calculateTransformerFlops(modelConfig, req.ProgressIndex, numTokens, true, true)
			pTotalFlops += f["total"] / tpFactor

			m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, numTokens, true)
			pDynamicBytes += (m["total"] - m["model_weights"]) / tpFactor
		}

		baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
		pWeightBytes := baseMem["model_weights"] / tpFactor

		pComputeS := pTotalFlops / (peakFlops * hwConfig.MfuPrefill)
		pMemoryS := (pWeightBytes + pDynamicBytes) / peakBW

		prefillTimeS = math.Max(pComputeS, pMemoryS)
	}

	// 2. DECODE PHASE
	if len(stepConfig.DecodeRequests) > 0 {
		var dTotalFlops, dDynamicBytes float64

		for _, req := range stepConfig.DecodeRequests {
			f := calculateTransformerFlops(modelConfig, req.ProgressIndex, 1, true, true)
			dTotalFlops += f["total"] / tpFactor

			m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, 1, true)
			dDynamicBytes += (m["total"] - m["model_weights"]) / tpFactor
		}

		baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
		dWeightBytes := baseMem["model_weights"] / tpFactor

		dComputeS := dTotalFlops / (peakFlops * hwConfig.MfuDecode)
		dMemoryS := (dWeightBytes + dDynamicBytes) / peakBW

		decodeTimeS = math.Max(dComputeS, dMemoryS)
	}

	// 3. COMBINE (no overhead terms)
	totalMicros := (prefillTimeS + decodeTimeS) * 1e6

	return int64(math.Round(totalMicros))
}
```

**Step 4: Update the empty step test**

The empty step test (`TestRooflineStepTime_EmptyStep_ReturnsOverheadOnly`) currently expects positive result from overhead alone. Without overheads, an empty step returns 0. Update:

```go
func TestRooflineStepTime_EmptyStep_ReturnsZero(t *testing.T) {
	// No requests = no work = 0 µs (no overhead terms in llm-optimizer model)
	mc := testModelConfig()
	hc := testHardwareCalib()

	step := StepConfig{} // empty
	result := rooflineStepTime(mc, hc, step, 1)

	if result != 0 {
		t.Errorf("empty step should return 0 µs, got %d µs", result)
	}
}
```

**Step 5: Run all roofline tests**

Run: `go test ./sim/latency/ -run TestRooflineStepTime -v -count=1`
Expected: All PASS — SingleCrossover test passes, empty step updated, TP scaling and smoke tests still hold (physics changed but invariants preserved)

**Step 6: Run full test suite + lint**

Run: `go test ./sim/... -count=1 && golangci-lint run ./sim/...`
Expected: All PASS. The `vectorPeak` variable and `effBW` are now unused — the compiler will catch this. The `sort` import may become unused if it was only used indirectly. The linter may flag unused `HardwareCalib` fields, but those are still valid struct fields.

**Step 7: Commit**

```bash
git add sim/latency/roofline.go sim/latency/roofline_test.go
git commit -m "refactor(latency): port llm-optimizer single-crossover roofline physics

Replace dual-ceiling model (GEMM + vector ceilings) with single-crossover:
  step_time = max(total_flops / (peak * MFU), total_bytes / peak_bandwidth)

Remove bandwidth haircut (BwEffConstant no longer used in step time).
Remove all overhead terms (TOverheadMicros, PerLayerOverhead, AllReduceLatency).

Keeps BLIS's superior model-awareness: actual IntermediateDim, SwiGLU
3-matrix MLP, MoE support, FlashAttention-aware memory model.

Motivation: BLIS roofline has 215% ITL MAPE vs llm-optimizer's 36.5%.
The dual ceiling + bandwidth haircut + overhead stacking caused ~3x
systematic over-prediction for memory-bound decode steps.

Design: docs/plans/2026-03-09-roofline-llm-optimizer-port-design.md"
```

---

### Task 2: Update hardware_config.json MFU values

**Files:**
- Modify: `hardware_config.json`

**Step 1: Update MFU values to match llm-optimizer defaults**

Change all GPU entries:
- `mfuPrefill`: 0.65 → 0.45
- `mfuDecode`: 0.12 → 0.30

For H100:
```json
{
    "H100": {
        "TFlopsPeak":        989.5,
        "BwPeakTBs":         3.35,
        "BwEffConstant":     0.72,
        "TOverheadMicros":   500.0,
        "perLayerOverhead":  20.0,
        "mfuPrefill":        0.45,
        "mfuDecode":         0.30,
        "allReduceLatency":  20.0,
        "MemoryGiB":         80.0
    }
}
```

Same for A100-SXM and A100-80.

Note: `BwEffConstant`, `TOverheadMicros`, `PerLayerOverhead`, `AllReduceLatency` remain in the file unchanged. They are still valid `HardwareCalib` struct fields and may be used by other consumers (crossmodel, future models). Only `rooflineStepTime()` stopped using them.

**Step 2: Run full test suite**

Run: `go test ./... -count=1`
Expected: All PASS. The `testHardwareCalib()` helper in `roofline_test.go` uses its own hardcoded values (MfuPrefill=0.55, MfuDecode=0.30), so test behavior doesn't depend on `hardware_config.json`.

**Step 3: Run lint**

Run: `golangci-lint run ./...`
Expected: Clean

**Step 4: Commit**

```bash
git add hardware_config.json
git commit -m "config: update MFU values to llm-optimizer defaults (0.45/0.30)

MfuPrefill: 0.65 → 0.45, MfuDecode: 0.12 → 0.30 for all GPU entries.
These values match llm-optimizer's defaults which achieve 36.5% ITL MAPE
on the sim-to-real evaluation (discussion #522).

Other HardwareCalib fields (BwEffConstant, overheads) remain unchanged
for backward compatibility — they are no longer used by rooflineStepTime()
but may be consumed by other callers."
```

---

### Task 3: Verify and commit design doc

**Files:**
- Existing: `docs/plans/2026-03-09-roofline-llm-optimizer-port-design.md`

**Step 1: Run full build + test + lint**

Run: `go build -o blis main.go && go test ./... -count=1 && golangci-lint run ./...`
Expected: All pass

**Step 2: Commit design doc**

```bash
git add docs/plans/2026-03-09-roofline-llm-optimizer-port-design.md docs/plans/2026-03-09-roofline-llm-optimizer-port-plan.md
git commit -m "docs: add roofline llm-optimizer port design and implementation plan

Design doc: decision record for porting llm-optimizer's single-crossover
roofline physics into BLIS.
Implementation plan: 3 tasks (physics rewrite, MFU update, verification).

Motivation: discussion #522 sim-to-real accuracy validation."
```
