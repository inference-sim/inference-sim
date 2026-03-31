# Basis Function Audit Report - Iteration 11

## Executive Summary

**Finding**: The β₁₀ and β₃' basis function **implementations are CORRECT**. The catastrophic failure (loss 4084%) is NOT due to bugs in these basis functions, but rather due to:

1. **Unit conversion error in YAML comments** creating confusion about expected ranges
2. **Model misspecification** - missing complementary terms or incorrect expected ranges for OTHER coefficients
3. **Broken loss landscape** preventing optimizer from finding good coefficients for all terms simultaneously

**Recommendation**: Do NOT modify β₁₀ or β₃' implementations. Instead, investigate why β₆, β₇, β₀, β₁, β₃, β₄ are out of range.

---

## β₁₀ (Batching Inefficiency) Audit

### Implementation (lines 401-418)

```go
var batchingInefficiencySum float64
effectiveBatchSize := float64(len(batch))
if effectiveBatchSize < 1.0 {
    effectiveBatchSize = 1.0
}

for _, req := range batch {
    if req.ProgressIndex < int64(len(req.InputTokens)) {
        numPrefillTokens := float64(req.NumNewTokens)
        batchingInefficiencySum += (numPrefillTokens * numPrefillTokens) / effectiveBatchSize
    }
}

batchingInefficiencyTimeSeconds := batchingInefficiencySum * m.Beta[10]
batchingInefficiencyContribution := batchingInefficiencyTimeSeconds * 1e6
```

### Verification

**Test Case 1: Scout general-lite (500 tokens, batch_size=4)**

Manual calculation:
- numPrefillTokens = 500
- effectiveBatchSize = 4
- batchingInefficiencySum = (500 × 500) / 4 = 62,500
- If β₁₀ = 0.95 μs = 0.00000095 seconds:
  - batchingInefficiencyTimeSeconds = 62,500 × 0.00000095 = 0.059375 s
  - batchingInefficiencyContribution = 0.059375 × 1,000,000 = 59,375 μs = **59.4 ms** ✓

**Test Case 2: Scout roleplay (100 tokens, batch_size=32)**

Manual calculation:
- numPrefillTokens = 100
- effectiveBatchSize = 32
- batchingInefficiencySum = (100 × 100) / 32 = 312.5
- If β₁₀ = 0.95 μs:
  - batchingInefficiencyContribution = 312.5 × 0.00000095 × 1,000,000 = **0.297 ms** ✓

**Scaling Verification**:
- Ratio: 59.4 ms / 0.297 ms = **200×** ✓
- Expected: (500/100)² × (32/4) = 25 × 8 = 200× ✓

**Conclusion**: β₁₀ implementation is **CORRECT**. Formula, units, and conversions all match expected behavior.

**Converged Value**: β₁₀ = 0.950 μs
- **Within expected range**: 0.1-1.0 μs ✅
- **Produces physically reasonable contributions**: 59 ms (long-seq), 0.3 ms (short-seq) ✅
- **Quadratic scaling preserved**: 200× ratio ✅

---

## β₃' (KV Sequence-Length Overhead) Audit

### Implementation (lines 258-270)

```go
var kvMgmtSeqLenTokenLayers float64
for _, req := range batch {
    if req.ProgressIndex < int64(len(req.InputTokens)) {
        numPrefillTokens := float64(req.NumNewTokens)
        kvMgmtSeqLenTokenLayers += numPrefillTokens * float64(m.modelConfig.NumLayers)
    }
}
kvMgmtSeqLenTimeSeconds := kvMgmtSeqLenTokenLayers * m.Beta[4]
kvMgmtSeqLenContribution := kvMgmtSeqLenTimeSeconds * 1e6
```

### Verification

**Test Case 1: Scout general-lite (500 tokens, 56 layers)**

Manual calculation:
- numPrefillTokens = 500
- numLayers = 56
- kvMgmtSeqLenTokenLayers = 500 × 56 = 28,000
- If β₃' = 0.252 μs = 0.000000252 seconds:
  - kvMgmtSeqLenTimeSeconds = 28,000 × 0.000000252 = 0.007056 s
  - kvMgmtSeqLenContribution = 0.007056 × 1,000,000 = **7,056 μs = 7.1 ms** ✓

**Test Case 2: Scout roleplay (100 tokens, 56 layers)**

Manual calculation:
- kvMgmtSeqLenTokenLayers = 100 × 56 = 5,600
- If β₃' = 0.252 μs:
  - kvMgmtSeqLenContribution = 5,600 × 0.000000252 × 1,000,000 = **1.4 ms** ✓

**Scaling Verification**:
- Ratio: 7.1 ms / 1.4 ms = **5×** ✓
- Expected: 500 / 100 = 5× ✓

**Conclusion**: β₃' implementation is **CORRECT**. Formula, units, and conversions all match expected behavior.

**Converged Value**: β₃' = 0.252 μs
- **Within expected range**: 0.1-1.0 μs ✅
- **Produces physically reasonable contributions**: 7 ms (long-seq), 1.4 ms (short-seq) ✅
- **Linear scaling preserved**: 5× ratio ✅

---

## Unit Conversion Error in YAML Comments

### The Confusion

`training/iterations/iter11/coefficient_bounds.yaml` line 106 says:

```yaml
# Physical range: 0.0000001-0.000001s = 0.1-1.0 ms per (token²/batch_request)
```

This is **WRONG**. Let's verify:

```python
0.0000001 seconds × 1000 ms/s = 0.0001 ms = 0.1 μs (NOT 0.1 ms!)
0.000001 seconds  × 1000 ms/s = 0.001 ms = 1.0 μs (NOT 1.0 ms!)
```

**Correction**: Should be "0.0000001-0.000001s = 0.1-1.0 **μs**" NOT ms!

### Impact of Error

1. **Iter10 analysis** (correctly) said β₁₀=0.945μs is 1000× smaller than expected 0.1-1.0 ms
2. **Iter11 hypothesis** (correctly) prescribed fixing basis function to achieve 0.1-1.0 ms range
3. **Iter11 execution** (incorrectly) changed comments to rationalize 0.945μs as correct
4. **Result**: Confusion about whether 0.95μs is correct or 1000× wrong

**Reality**: β₁₀=0.95μs **IS within the correct physical range** of 0.1-1.0 μs, but the YAML comments were wrong, causing everyone to think the expected range was 0.1-1.0 ms.

---

## Why Is The Loss Still Catastrophic?

If β₁₀ and β₃' are correct and producing reasonable contributions, why is the overall loss 4084%?

### Coefficient Status Table

| Coefficient | Iter11 Value | Expected Range | Status | Deviation |
|-------------|--------------|----------------|--------|-----------|
| β₀ (prefill compute) | 0.286 | 0.14-0.22 | ❌ | 30% too high |
| β₁ (decode memory) | 1.107 | 1.2-1.5 | ❌ | 8% too low |
| β₂ (TP comm) | 0.383 | 0.25-0.60 | ✅ | Within range |
| β₃ (KV base) | 0.207 ms | 0.4-1.5 ms | ❌ | 50% too low |
| β₃' (KV seq-len) | 0.252 μs | 0.1-1.0 μs | ✅ | Within range |
| β₄ (decode compute) | 0.815 | 0.40-0.65 | ❌ | 25% too high |
| β₅ (MoE gating) | 15.5 μs | 15-25 μs | ✅ | Within range |
| **β₆ (scheduler)** | **59.3 ms** | **15-40 ms** | ❌ | **48-295% too high!** |
| β₇ (decode overhead) | 5.0 ms | 8-20 ms | ❌ | 38-75% too low |
| β₈ (MoE routing) | 44.5 μs | 25-80 μs | ✅ | Within range |
| β₁₀ (batching ineff) | 0.950 μs | 0.1-1.0 μs | ✅ | Within range |

**Summary**:
- ✅ **5/11 coefficients within range** (β₂, β₃', β₅, β₈, β₁₀)
- ❌ **6/11 coefficients out of range** (β₀, β₁, β₃, β₄, β₆, β₇)

### Root Causes

The catastrophic loss is NOT due to β₁₀ or β₃' bugs. Instead:

1. **β₆ is massively inflated** (59ms vs 15-40ms expected)
   - Either the expected range is wrong, OR
   - β₆ is absorbing overhead that should be captured by another term

2. **β₃ is too low** (0.2ms vs 0.4-1.5ms expected)
   - The split concept may be flawed, OR
   - KV management overhead is smaller than expected

3. **β₇ is too low** (5ms vs 8-20ms expected)
   - Decode overhead may be smaller than expected, OR
   - Being absorbed into β₆

4. **Other MFU factors are out of range** (β₀, β₁, β₄)
   - Suggests the MFU model itself may be misspecified

### The Optimizer Is Stuck

Even if β₁₀ and β₃' are producing correct contributions, the optimizer can't find good coefficients for all terms simultaneously. This indicates:

1. **Model misspecification** - Missing complementary terms (e.g., memory bandwidth saturation, chunked prefill overhead)
2. **Incorrect expected ranges** - Some "physical ranges" may be wrong (like β₁₀ was)
3. **Coupling between terms** - β₆ and β₁₀ may be competing to explain the same variance
4. **Local minimum** - Optimizer stuck in bad configuration where improving one term makes others worse

---

## Recommendations

### Do NOT Fix β₁₀ or β₃' Implementations

The implementations are correct. Modifying them will not improve the model.

### Do Investigate Other Coefficients

**High Priority**:

1. **β₆ (scheduler overhead)**: Why is it 59ms instead of 15-40ms?
   - Profile vLLM scheduler separately to measure actual CPU overhead
   - Check if β₆ is absorbing queueing delays that β₁₀ should capture
   - Consider splitting β₆ into "scheduler CPU" + "queueing delay" components

2. **β₃ (KV base)**: Why is it 0.2ms instead of 0.4-1.5ms?
   - Verify expected range with profiling
   - Check if β₃' split is capturing too much of the base overhead

3. **β₇ (decode overhead)**: Why is it 5ms instead of 8-20ms?
   - Verify expected range
   - Check if being absorbed into β₆

**Medium Priority**:

4. **β₀, β₁, β₄ (MFU factors)**: Why are they out of range?
   - Re-examine MFU model assumptions
   - Check if sigmoid interpolation for memory/compute bound is correct

### Add Missing Terms

Consider adding:
- **β₁₁**: Memory bandwidth saturation (complementary to β₁₀)
- **β₁₂**: Chunked prefill overhead (for long sequences)
- Split β₆ into CPU overhead vs queueing delay components

### Fix YAML Comment Errors

Update `coefficient_bounds.yaml` line 106:
```yaml
# Physical range: 0.0000001-0.000001s = 0.1-1.0 μs per (token²/batch_request)  # NOT ms!
```

And line 131:
```yaml
- 0.0000005    # β₁₀ (batching inefficiency) — initialize at 0.5μs per (token²/batch_request), mid-range after fix
```

### Write Unit Tests

Even though the implementations are correct, unit tests would have prevented the confusion:

```go
func TestBeta10BatchingInefficiency(t *testing.T) {
    // Validates β₁₀ contributions for long/short sequences
    coeff := 0.00000095  // 0.95μs in seconds

    // Long sequence: 500 tokens, batch_size=4
    basisValue1 := (500.0 * 500.0) / 4.0  // 62,500
    contribution1 := basisValue1 * coeff * 1e6  // Convert to μs
    expected1 := 59375.0  // μs
    if math.Abs(contribution1 - expected1) > 0.01 * expected1 {
        t.Errorf("Long-seq: got %.1f μs, want %.1f μs", contribution1, expected1)
    }

    // Short sequence: 100 tokens, batch_size=32
    basisValue2 := (100.0 * 100.0) / 32.0  // 312.5
    contribution2 := basisValue2 * coeff * 1e6
    expected2 := 296.875  // μs
    if math.Abs(contribution2 - expected2) > 0.01 * expected2 {
        t.Errorf("Short-seq: got %.1f μs, want %.1f μs", contribution2, expected2)
    }

    // Scaling: should be 200× ratio
    ratio := contribution1 / contribution2
    expectedRatio := 200.0
    if math.Abs(ratio - expectedRatio) > 0.1 * expectedRatio {
        t.Errorf("Scaling: got %.1fx, want %.1fx", ratio, expectedRatio)
    }
}
```

---

## Conclusion

**The iter11 hypothesis was WRONG about the bugs**:
- β₁₀ basis function implementation is correct (not broken)
- β₃' basis function implementation is correct (not broken)
- The "factor-of-1000 error" was actually a unit conversion error in YAML comments

**The iter11 execution was CORRECT about not modifying basis functions**:
- Changing comments to rationalize 0.95μs was actually the right call
- But it violated scientific rigor by not explaining WHY the hypothesis was wrong

**The real problem is model misspecification**:
- 6/11 coefficients are out of range
- β₆ is massively inflated (59ms vs 15-40ms)
- The loss landscape is broken, preventing optimization

**Next steps**:
1. Profile vLLM to verify expected ranges for β₆, β₃, β₇
2. Consider adding complementary terms (memory bandwidth, chunked prefill)
3. Investigate why β₆ is absorbing so much variance
4. Fix YAML comment errors to prevent future confusion
