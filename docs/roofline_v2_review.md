# Roofline V2 Implementation Review

## Overview

This document reviews `sim/roofline_step_v2.go` against the InferSim methodology described in the technical report and problem statement.

## ✅ Correct Implementation Decisions

### 1. **Per-Operation GEMM Time Calculation** (Lines 11-20, 25-86)
```go
func computeGEMMTime(m, k, n int, peakFlops float64, mfuDB *MFUDatabase) float64 {
    flops := 2.0 * float64(m) * float64(k) * float64(n)
    mfu := mfuDB.GetGEMMmfu(m, k, n)
    return flops / (peakFlops * mfu)
}
```
✅ **Correct**: Matches InferSim formula `t = FLOPs / (FLOPS_peak × MFU)`

✅ **Correct**: Separates Q, K, V, O, Gate, Up, Down projections for individual MFU lookups

### 2. **Decode Phase Aggregation** (Lines 141-193)
```go
totalBatchSize := len(stepConfig.DecodeRequests)
maxKVLen := int64(0)
for _, req := range stepConfig.DecodeRequests {
    if req.ProgressIndex > maxKVLen {
        maxKVLen = req.ProgressIndex
    }
}
```
✅ **Correct**: Aggregates all decode requests into single batch for MFU lookup (matches InferSim)

### 3. **Prefill Bucketing Strategy** (Lines 198-266)
```go
bucketMap := make(map[int][]PrefillRequestConfig)
for _, req := range stepConfig.PrefillRequests {
    seqLen := int(req.ProgressIndex + int64(req.NumNewPrefillTokens))
    bucket := 512
    for bucket < seqLen && bucket < 65536 {
        bucket *= 2
    }
    bucketMap[bucket] = append(bucketMap[bucket], req)
}
```
✅ **Good approach**: Bucketing by power-of-2 sequence lengths makes sense for MFU lookup

### 4. **MFU Database Integration**
✅ **Correct**: Uses `mfuDB.GetAttnPrefillMFU()`, `GetAttnDecodeMFU()`, `GetGEMMmfu()` for lookups

---

## 🔴 Critical Issues

### Issue 1: **Incorrect Prefill GEMM Batch Size** (Line 226-228)

**Current Code:**
```go
gemmTimeS := computeTransformerGEMMTimes(
    modelConfig,
    batchSize,  // ← WRONG: This is number of requests in bucket
    peakFlops,
    mfuDB,
    tpScaling,
)
```

**Problem**: For prefill, the `m` dimension in GEMMs should be **total tokens**, not number of requests.

**Expected**:
```go
totalTokensInBucket := 0
for _, req := range requests {
    totalTokensInBucket += int(req.NumNewPrefillTokens)
}

gemmTimeS := computeTransformerGEMMTimes(
    modelConfig,
    totalTokensInBucket,  // ← Use total tokens as batch dimension
    peakFlops,
    mfuDB,
    tpScaling,
)
```

**Impact**: MFU lookup will use wrong `m` dimension, leading to incorrect GEMM time predictions.

---

### Issue 2: **Attention Core FLOPs Calculation Missing R_mask** (Lines 92-116)

**Current Code:**
```go
func calculateAttentionCoreFLOPs(
    nHeads int,
    nKVHeads int,
    dModel int,
    batchSize int,
    seqLen int64,
) float64 {
    qkMatMul := 2.0 * float64(nHeads) * float64(batchSize) * effectiveCtx * float64(headDim)
    softmaxOps := 4.0 * float64(nHeads) * float64(batchSize) * effectiveCtx
    avMatMul := 2.0 * float64(nHeads) * float64(batchSize) * effectiveCtx * float64(headDim)
    return qkMatMul + softmaxOps + avMatMul
}
```

**InferSim Formula:**
```
FLOPs_attn_core = 4 × n_h × S_q × S_kv × d_h × R_mask
```

Where `R_mask = 0.5` for prefill (causal masking), `1.0` for decode.

**Problem**:
1. Missing `R_mask` factor (should reduce prefill attention FLOPs by 50%)
2. FLOPs formula appears to be `2*QK + 4*softmax + 2*AV` but should be `4*n_h*S_q*S_kv*d_h`

**Expected**:
```go
func calculateAttentionCoreFLOPs(
    nHeads int,
    nKVHeads int,
    dModel int,
    batchSize int,
    seqLen int64,
    isPrefill bool,  // ← Add flag to determine R_mask
) float64 {
    headDim := dModel / nHeads

    // R_mask: 0.5 for prefill (causal), 1.0 for decode
    rMask := 1.0
    if isPrefill {
        rMask = 0.5
    }

    // InferSim formula: 4 × n_h × S_q × S_kv × d_h × R_mask
    return 4.0 * float64(nHeads) * float64(batchSize) * float64(seqLen) * float64(headDim) * rMask
}
```

**Impact**: Prefill attention FLOPs will be overestimated by 2x.

---

### Issue 3: **Mysterious 1.8 Divisor for Prefill Attention** (Line 249)

**Current Code:**
```go
// Formula: time = flops / 1.8 / (peakFlops * mfu)
// Note: /1.8 factor from InferSim (hardware-specific adjustment)
attnCoreTimeS := attnCoreFLOPs / 1.8 / (peakFlops * attnMFU) * tpScaling
```

**Problem**:
- This 1.8 factor is not documented in the InferSim technical report
- Comment says "hardware-specific" but it's hardcoded for all hardware
- Unclear if this compensates for Issue #2 (missing R_mask) or is a separate adjustment

**Questions**:
1. Where does 1.8 come from? Is it `1/0.5 = 2` approximated to 1.8?
2. Does the InferSim codebase have this factor?
3. Should this vary by hardware or workload?

**Action**: Need to verify against InferSim source code or remove if it's compensating for wrong FLOPs calculation.

---

### Issue 4: **Prefill Attention Batch Size Calculation** (Lines 236-241)

**Current Code:**
```go
attnCoreFLOPs := calculateAttentionCoreFLOPs(
    modelConfig.NumHeads,
    modelConfig.NumKVHeads,
    modelConfig.HiddenDim,
    batchSize,  // ← This is number of requests, not tokens
    int64(bucketSeqLen),
) * float64(modelConfig.NumLayers)
```

**Problem**: For prefill attention, the batch dimension should be **total tokens** being processed, not number of requests.

**Expected**:
```go
totalTokensInBucket := 0
for _, req := range requests {
    totalTokensInBucket += int(req.NumNewPrefillTokens)
}

attnCoreFLOPs := calculateAttentionCoreFLOPs(
    modelConfig.NumHeads,
    modelConfig.NumKVHeads,
    modelConfig.HiddenDim,
    totalTokensInBucket,  // ← Total tokens, not requests
    int64(bucketSeqLen),
) * float64(modelConfig.NumLayers)
```

**Impact**: Attention FLOPs and time will be incorrectly scaled by request count.

---

### Issue 5: **Incorrect Bucketing Key** (Line 206)

**Current Code:**
```go
seqLen := int(req.ProgressIndex + int64(req.NumNewPrefillTokens))
```

**Problem**: `ProgressIndex` is the number of tokens already processed. For bucketing, we should only use **new tokens** being prefilled in this step.

**Expected**:
```go
seqLen := int(req.NumNewPrefillTokens)  // ← Only bucket by new tokens
```

**Rationale**: MFU varies by sequence length being processed in this step, not total historical sequence length.

---

## ⚠️ Design Questions

### Question 1: **TP Scaling for Attention Core** (Lines 178, 249)

**Current Code:**
```go
// Decode
attnCoreTimeS := attnCoreFLOPs / (peakFlops * attnMFU) * tpScaling

// Prefill
attnCoreTimeS := attnCoreFLOPs / 1.8 / (peakFlops * attnMFU) * tpScaling
```

**Question**: Should attention core time be scaled by TP?

**InferSim Approach**: The technical report doesn't explicitly mention TP scaling for attention core, only for projections.

**Analysis**:
- FlashAttention implementations typically **do** benefit from TP because Q/K/V are already split
- However, the MFU lookup tables might already account for TP (e.g., decode CSVs have `-tp2` suffix)
- Need to verify: Are we double-counting TP benefits?

**Action**: Check if `GetAttnDecodeMFU(batchSize, kvLen, tp)` already incorporates TP scaling.

---

### Question 2: **Mixed Batch Weighting** (Lines 273-297)

**Current Code:**
```go
if prefillTokens > decodeTokens*4 && prefillTokens > 100 {
    stepHardwareS = 0.75*prefillTimeS + 0.25*decodeTimeS
} else if decodeTokens > prefillTokens*2 && decodeTokens > 50 {
    stepHardwareS = 0.35*prefillTimeS + 0.65*decodeTimeS
} else {
    prefillWeight := prefillTokens / totalTokens
    decodeWeight := decodeTokens / totalTokens
    stepHardwareS = prefillWeight*prefillTimeS + decodeWeight*decodeTimeS
}
```

**Question**: Does InferSim handle mixed batches, or does it assume pure prefill/decode?

**InferSim Report**: Section 2.1 and 2.2 discuss prefill and decode separately. No mention of mixed batches.

**Analysis**: vLLM **can** mix prefill and decode in same batch with chunked prefill. The weighting logic seems reasonable but is empirical.

**Action**: Validate against actual vLLM mixed batch measurements.

---

## 📊 Comparison with InferSim Formulas

### Attention Core (InferSim)
```
FLOPs_attn_core = 4 × n_h × S_q × S_kv × d_h × R_mask
t_attn_core = FLOPs_attn_core / (FLOPS_gpu × MFU_attn_core)
```

### Attention Core (Current Implementation)
```go
qkMatMul := 2.0 * nHeads * batchSize * effectiveCtx * headDim
softmaxOps := 4.0 * nHeads * batchSize * effectiveCtx
avMatMul := 2.0 * nHeads * batchSize * effectiveCtx * headDim
attnCoreFLOPs = qkMatMul + softmaxOps + avMatMul
attnCoreTimeS = attnCoreFLOPs / 1.8 / (peakFlops * attnMFU) * tpScaling
```

**Mismatch**: Implementation has explicit softmax ops (4x) but InferSim bundles everything into coefficient of 4.

---

### GEMM Projections (InferSim)
```
FLOPs_qkvo_proj = 4(n_h + n_kv) × d_h × d_hidden
t_attn_proj = FLOPs_qkvo_proj / (FLOPS_gpu × MFU_attn_proj)
```

### GEMM Projections (Current Implementation)
```go
// Separates into individual GEMMs:
qTime := computeGEMMTime(batchSize, dModel, dModel, peakFlops, mfuDB)
kTime := computeGEMMTime(batchSize, dModel, dKV, peakFlops, mfuDB)
vTime := computeGEMMTime(batchSize, dModel, dKV, peakFlops, mfuDB)
oTime := computeGEMMTime(batchSize, dModel, dModel, peakFlops, mfuDB)
```

**Better**: Current implementation is more granular and accurate (each GEMM gets its own MFU lookup).

---

## 🎯 Priority Fixes

### Priority 1 (Critical - Affects Accuracy)
1. ✅ Fix prefill GEMM batch size (Issue #1)
2. ✅ Fix prefill attention batch size (Issue #4)
3. ✅ Add R_mask to attention FLOPs (Issue #2)
4. ✅ Fix bucketing to use only new tokens (Issue #5)

### Priority 2 (Clarification Needed)
5. ⚠️ Investigate 1.8 divisor (Issue #3)
6. ⚠️ Verify TP scaling for attention (Question #1)

### Priority 3 (Empirical Validation)
7. 📊 Validate mixed batch weighting (Question #2)
8. 📊 Run evaluator to measure actual error rates

---

## 🔧 Recommended Fixes

### Fix 1: Update Prefill GEMM and Attention Calls

```go
// Process each bucket independently
for bucketSeqLen, requests := range bucketMap {
    // Calculate total tokens in this bucket
    totalTokens := 0
    for _, req := range requests {
        totalTokens += int(req.NumNewPrefillTokens)
    }

    // === GEMM Projections ===
    gemmTimeS := computeTransformerGEMMTimes(
        modelConfig,
        totalTokens,  // ← FIX: Use total tokens, not request count
        peakFlops,
        mfuDB,
        tpScaling,
    )

    // === Attention Core ===
    attnCoreFLOPs := calculateAttentionCoreFLOPs(
        modelConfig.NumHeads,
        modelConfig.NumKVHeads,
        modelConfig.HiddenDim,
        totalTokens,        // ← FIX: Use total tokens
        int64(bucketSeqLen),
        true,               // ← FIX: Add isPrefill flag
    ) * float64(modelConfig.NumLayers)

    attnMFU := mfuDB.GetAttnPrefillMFU(bucketSeqLen)

    // FIX: Remove 1.8 divisor (or verify its necessity)
    attnCoreTimeS := attnCoreFLOPs / (peakFlops * attnMFU) * tpScaling

    prefillComputeS += gemmTimeS + attnCoreTimeS
}
```

### Fix 2: Update calculateAttentionCoreFLOPs

```go
func calculateAttentionCoreFLOPs(
    nHeads int,
    nKVHeads int,
    dModel int,
    batchSize int,
    seqLen int64,
    isPrefill bool,  // ← ADD: Flag to determine R_mask
) float64 {
    if nKVHeads == 0 {
        nKVHeads = nHeads
    }

    headDim := dModel / nHeads

    // Causal mask ratio: 0.5 for prefill, 1.0 for decode
    rMask := 1.0
    if isPrefill {
        rMask = 0.5
    }

    // InferSim formula: 4 × n_h × S_q × S_kv × d_h × R_mask
    return 4.0 * float64(nHeads) * float64(batchSize) * float64(seqLen) * float64(headDim) * rMask
}
```

### Fix 3: Update Bucketing Logic

```go
for _, req := range stepConfig.PrefillRequests {
    seqLen := int(req.NumNewPrefillTokens)  // ← FIX: Only use new tokens

    // Find next power-of-2 bucket
    bucket := 512
    for bucket < seqLen && bucket < 65536 {
        bucket *= 2
    }
    if bucket > 65536 {
        bucket = 65536
    }

    bucketMap[bucket] = append(bucketMap[bucket], req)
}
```

---

## Next Steps

1. **Apply Priority 1 fixes** to address critical accuracy issues
2. **Review InferSim source code** to clarify 1.8 divisor and TP scaling
3. **Run evaluator** (`python_scripts/blis_evaluator.py`) to measure accuracy
4. **Compare V1 vs V2** error rates to validate improvements
5. **Investigate remaining error sources** if accuracy doesn't meet <15% target
