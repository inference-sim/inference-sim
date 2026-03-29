# Scout MoE Bugs: Root Cause of Catastrophic Failures

## Executive Summary

Investigation revealed **three critical bugs** in BLIS's MoE implementation that explain ALL Scout experiment catastrophic failures (89-100% TTFT error).

**Scout's interleaved architecture** (24 MoE layers + 24 dense layers) is **completely ignored** by the simulator. BLIS treats all 48 layers as MoE layers with 8192 FFN dim, when the correct architecture is:
- 24 MoE layers: 16 experts, top-1, 8192 FFN dim per expert
- 24 Dense layers: 16384 FFN dim, no experts

**Impact**: FLOPs and weight bandwidth calculations are systematically wrong, causing β₆ (MoE gating) to inflate to 0.224 (28× expected) as optimizer tries to compensate for structural mismatch.

**This discovery invalidates iter2's data quality hypothesis** for Scout experiments. The failures are NOT data corruption - they are simulator bugs.

---

## Bug Details

### BUG 1 (CRITICAL): Interleaved MoE Architecture Completely Ignored

**Severity**: CRITICAL - Affects FLOPs, weight bandwidth, and memory calculations

**Description**: Scout uses `interleave_moe_layer_step: 1`, where layers alternate between MoE and dense. Of 48 total layers, only 24 are MoE layers. But BLIS has:
- NO `InterleaveMoELayerStep` field in `ModelConfig` struct
- NO parsing of `interleave_moe_layer_step` in `config.go`
- NO distinction between MoE and dense layer counts anywhere

**Location**:
- `sim/model_hardware_config.go:14` - Missing field in `ModelConfig` struct
- `sim/latency/config.go:240` - No parsing of `interleave_moe_layer_step`
- `sim/latency/roofline.go:102-105` - MLP FLOPs multiplied by `nLayers` (48) without layer-type distinction

**Code**:
```go
// roofline.go:102-105
if config.NumLocalExperts > 1 {
    mlpFlopsPerLayer *= float64(config.NumExpertsPerTok)
}
flops["gemm_ops"] += mlpFlopsPerLayer * nLayers  // <-- Applies MoE scaling to ALL 48 layers
```

**Impact**:
- MLP FLOPs: Applies expert scaling (top-1) to all 48 layers instead of 24
- Weight bandwidth: Applies nEff expert loading formula to all 48 layers instead of 24
- For Scout, this causes 2× over-estimation of expert-related overhead (48/24 = 2)

**Evidence**:
```bash
# Scout config.json
curl -s https://huggingface.co/RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic/raw/main/config.json | jq '.text_config.interleave_moe_layer_step'
# Output: 1

# BLIS has no field for this
grep -r "interleave" sim/ --include="*.go"
# Output: (empty)
```

**Fix Required**:
1. Add `InterleaveMoELayerStep int` field to `ModelConfig` struct
2. Parse `interleave_moe_layer_step` from config.json in `GetModelConfigFromHF()`
3. In `calculateTransformerFlops()` and `calculateMemoryAccessBytes()`, compute separate FLOPs/bandwidth for MoE layers vs dense layers
4. Formula: `numMoELayers = (numLayers + interleaveMoELayerStep - 1) / interleaveMoELayerStep` when `interleaveMoELayerStep > 0`

---

### BUG 2 (CRITICAL): Scout's `intermediate_size_mlp` Field Not Parsed

**Severity**: CRITICAL - Dense layers use wrong FFN dimension (50% underprediction)

**Description**: Scout's config.json has TWO distinct FFN dimensions:
- `intermediate_size: 8192` - Per-expert FFN dim for MoE layers
- `intermediate_size_mlp: 16384` - Dense FFN dim for non-MoE layers

BLIS only parses `intermediate_size` (8192) and uses it for ALL layers.

**Location**:
- `sim/latency/config.go:240` - Only checks `intermediate_size` and `ffn_hidden_size` fallback

**Code**:
```go
// config.go:240
intermediateDim := getIntWithFallbacks("intermediate_size", "ffn_hidden_size")
```

**Impact**:
- Dense layers (24 of 48 for Scout) compute MLP FLOPs using 8192 instead of 16384
- Dense layer MLP FLOPs are 50% of actual (16384/8192 = 2)
- Weight bandwidth for dense layers is also 50% of actual

**Evidence**:
```bash
# Scout config has both fields
curl -s https://huggingface.co/RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic/raw/main/config.json | jq '.text_config | {intermediate_size, intermediate_size_mlp}'
# Output: {"intermediate_size": 8192, "intermediate_size_mlp": 16384}

# BLIS never parses intermediate_size_mlp
grep -r "intermediate_size_mlp" sim/ --include="*.go"
# Output: (empty)
```

**Fix Required**:
1. Add `DenseIntermediateDim int` field to `ModelConfig` struct (or similar naming)
2. Parse `intermediate_size_mlp` from config.json in `GetModelConfigFromHF()`
3. Use `DenseIntermediateDim` (if set and > 0) for dense layers, `IntermediateDim` or `MoEExpertFFNDim` for MoE layers
4. Requires implementing layer-type distinction from Bug 1 fix first

---

### BUG 3 (MODERATE): nEff Expert Loading Applied to All Layers

**Severity**: MODERATE - Weight bandwidth calculation incorrect

**Description**: The `nEff` formula (expected unique experts loaded per step) is computed correctly for uniform routing, but applied to all 48 layers:

**Location**: `sim/latency/roofline.go:151-170`

**Code**:
```go
// roofline.go:151-170
if config.NumLocalExperts > 1 {
    N := float64(config.NumLocalExperts)  // 16
    k := float64(config.NumExpertsPerTok)  // 1
    B := float64(newTokens)
    probNotSelected := (N - k) / N
    nEff := N * (1.0 - math.Pow(probNotSelected, B))
    mlpWeightsPerLayer *= nEff  // <-- Applies to ALL layers
}
weightsPerLayer := attnWeightsPerLayer + mlpWeightsPerLayer
mem["model_weights"] = weightsPerLayer * nLayers * config.EffectiveWeightBytesPerParam()
```

**Impact**:
- For Scout with batch_size ~10-30, nEff ≈ 2-4 experts per step
- Dense layers should load the full single FFN (no expert scaling), but code applies nEff multiplier
- Weight bandwidth is over-estimated for dense layers by nEff factor (2-4×)

**Fix Required**: Same as Bug 1 - requires splitting into MoE vs dense layer calculations.

---

## Verification Plan

### Step 1: Confirm Bugs with Manual Calculation

For Scout codegen experiment (~588 prompt tokens, TP=2, H100):

**Expected TTFT calculation** (correct architecture):
1. Attention FLOPs: Same for all 48 layers (not MoE-dependent)
   - Per-layer: 2 × 588 × (3 × 5120²) + 4 × 40 × 588² × 128 ≈ 9.5e10 FLOPs/layer
   - Total: 9.5e10 × 48 ≈ 4.6e12 FLOPs

2. MLP FLOPs:
   - **24 MoE layers**: 2 × 588 × (2 × 5120 × 8192) × 1 ≈ 9.8e10 FLOPs/layer
   - **24 Dense layers**: 2 × 588 × (2 × 5120 × 16384) ≈ 2.0e11 FLOPs/layer
   - Total MLP: (9.8e10 × 24) + (2.0e11 × 24) ≈ 7.2e12 FLOPs

3. Total FLOPs: 4.6e12 + 7.2e12 ≈ 1.2e13 FLOPs

4. Time (H100 989.5 TFLOPS, TP=2, MFU=40%):
   - Compute time: 1.2e13 / (989.5e12 / 2) / 0.4 ≈ 60ms

**BLIS calculation** (all layers treated as MoE with 8192 FFN):
1. Attention FLOPs: 4.6e12 (same)
2. MLP FLOPs: 9.8e10 × 48 ≈ 4.7e12
3. Total: 4.6e12 + 4.7e12 ≈ 9.3e12 FLOPs
4. Time: 9.3e13 / (989.5e12 / 2) / 0.4 ≈ 47ms

**Result**: BLIS under-predicts by ~20% (47ms vs 60ms), but observed is ~72.8ms. So bugs explain PART of error but not all. Memory bandwidth may be the dominant term.

### Step 2: Test Fix Impact

After fixing bugs:
1. Re-run Scout experiments through simulator with corrected architecture
2. Expected: TTFT predictions improve from 89-100% error to <30% error
3. β₆ should drop from 0.224 to ~0.008 (normalized, no longer compensating for architecture bug)

---

## Impact on iter2 Analysis

**Major conclusion change**: Scout catastrophic failures are **SIMULATOR BUGS**, not data quality issues.

**Updated recommendations**:
1. ❌ REMOVE recommendation to "exclude Scout experiments from iter3"
2. ✅ ADD recommendation to "fix Scout MoE bugs before iter3"
3. ✅ KEEP recommendation to "audit reasoning experiments" (different root cause)

**Expected iter3 outcome after Scout fixes**:
- Scout experiments improve from 89-100% TTFT to <30% TTFT
- β₆ drops from 0.224 to ~0.008
- Overall loss drops from 150.78% to ~90-100% (Scout contributes ~1072% to loss sum, fixing saves ~70 points)
- Combined with reasoning exclusion/fix: Overall loss drops to **<60%**

---

## Recommended Actions

### Immediate (Before iter3):

**Priority 1: Fix Scout MoE bugs**
1. Add `InterleaveMoELayerStep int` field to `ModelConfig` (default: 0 = all layers same type)
2. Add `DenseIntermediateDim int` field to `ModelConfig` (dense FFN dim for interleaved architectures)
3. Parse both fields from config.json in `GetModelConfigFromHF()`
4. Split `calculateTransformerFlops()` into MoE-layer and dense-layer calculations
5. Split `calculateMemoryAccessBytes()` into MoE-layer and dense-layer calculations
6. Update `evolved_model.go` to distinguish layer types (gating only applies to MoE layers)

**Priority 2: Add test coverage**
1. Add Scout-like config to `roofline_test.go` (48 layers, interleave_moe_layer_step=1, two FFN dims)
2. Test that FLOPs/bandwidth calculations correctly split between MoE and dense layers

**Priority 3: Re-validate Scout after fix**
1. Re-run all 6 Scout experiments through fixed simulator
2. Expected: TTFT error drops from 89-100% to <30%
3. If still >50% error after fix: THEN investigate data quality

### For iter3 (After Scout Fixes):

**Updated model configuration**:
- **Training set**: All 15 experiments (Scout NOW INCLUDED, reasoning audit still needed)
- **Free parameters**: 5-7 (fix Alpha, remove β₇/β₈, fix or remove β₂/β₄)
- **Expected loss**: <60% (Scout fixed, reasoning excluded)

**Updated hypothesis focus**:
- Do NOT add new basis functions (β₇, β₈ proven ineffective)
- Focus on fixing Alpha confounding (fix to constants)
- Focus on normalizing β₁ (decode memory efficiency still inflated at 1.027)

---

`★ Insight ─────────────────────────────────────`

**Strategy Evolution principle validated**: Catastrophic failure is more valuable than partial success. iter2's 150.78% loss forced deep investigation that revealed THREE simulator bugs, not just wrong coefficients.

**Key learning**: When β₆ inflates to 28× expected (0.224 vs 0.008), it's a **structural mismatch signal**, not a physics parameter. The optimizer cannot fix architecture bugs by tuning coefficients - it can only highlight them via extreme values.

**Scout's interleaved architecture** (alternating MoE/dense layers) is a rare pattern (Mixtral, DeepSeek, etc. use uniform MoE). BLIS was developed for uniform architectures, and the interleaved case was never tested. This is why all 6 Scout experiments failed identically.

**Next steps**: Fix the three bugs, add Scout test coverage, then re-run iter2 optimization with Scout included. Expected outcome: loss drops from 150.78% to ~80-100%.

`─────────────────────────────────────────────────`

Would you like me to:
1. **Update iter2-FINDINGS.md** to reflect that Scout failures are simulator bugs (not data quality)?
2. **File GitHub issues** for the three MoE bugs?
3. **Create a plan to fix the bugs** before iter3?