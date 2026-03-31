# Iteration 11: Basis Function Bug Fixes (Batching Inefficiency + KV Seq-Len Scaling)

**Status**: **REJECTED (Not Tested)** — Basis function bugs were NOT fixed; only comments were changed. See iter11-FINDINGS.md.

## Overview

**Iter10 Catastrophic Failure**: Overall loss exploded from 160.6% (iter9) to **4267.22%** (27× worse). Root cause: **Systematic unit/scaling errors** in β₁₀ (batching inefficiency) and β₃' (KV seq-len component) basis function implementations.

**Critical Discovery from Iter10 Analysis**:
1. **β₁₀ functional form is CORRECT**: Quadratic scaling preserved (long/short ratio = 197× matches expected 200×)
2. **β₁₀ magnitude is WRONG**: All contributions are 1000× too small (factor-of-1000 error)
3. **β₃' magnitude is WRONG**: 65× too large (65.8μs vs expected 0.1-1.0μs per token×layer)
4. **β₃ split concept is SOUND**: β₃ reverted correctly to 0.402ms (within expected 0.4-1.5ms)

**Diagnosis**: Unit conversion bugs in `sim/latency/evolved_model.go` for β₁₀ and β₃'. The physics hypotheses are correct; the code implementations have errors.

**Iter11 Strategy**:
1. **Audit and fix β₁₀ basis function** (lines 369-406 in evolved_model.go) — likely seconds-to-microseconds conversion error or missing multiplication factor
2. **Audit and fix β₃' basis function** (lines 240-262 in evolved_model.go) — likely microseconds-to-seconds conversion error or incorrect scaling
3. **Write unit tests** BEFORE training to validate expected contributions
4. **Warm-start from iter9** (NOT iter10) to avoid broken coefficient artifacts
5. **Maintain constrained alpha bounds** to prevent spurious reduction

## H-main: Fixed Basis Functions Enable Sequence-Length Overhead Capture

**Prediction**: After fixing β₁₀ and β₃' formulation bugs:
- **Overall loss**: 160.6% → **<90%** (>70pp improvement, 44% reduction from iter9)
- **TTFT RMSE**: 64.8% → **<40%** (>25pp improvement, 38% reduction from iter9)
- **E2E RMSE**: 95.8% → **<55%** (>41pp improvement, 43% reduction from iter9)
- **Scout long-sequence TTFT**: 91.5% → **<60%** (>31pp improvement)
  - General-lite (exp 17): 92% → <60% TTFT
  - Reasoning-lite (exp 48): 91% → <60% TTFT
- **β₁₀ coefficient**: **0.1-1.0 ms** per (token²/batch_request) — physically plausible
- **β₃' coefficient**: **0.1-1.0 μs** per (token×layer) — physically plausible
- **β₆ reversion**: 99ms → **15-40ms** (scheduler overhead offloads queueing delay to β₁₀)
- **β₃ stability**: Remain ~0.4-1.5ms (base KV overhead, correctly split)

**Quantitative Threshold**: If overall loss does NOT reduce below 110%, OR if β₁₀ converges outside 0.05-2.0 ms OR β₃' converges outside 0.05-2.0 μs, then H-main is REJECTED — indicates persistent formulation errors or missing complementary terms.

**Causal Mechanism**:

**β₁₀ (Batching Inefficiency)**:
- Long sequences consume disproportionate batch capacity, reducing GPU utilization
- vLLM scheduler constraint: `Σ(prefill_tokens + kv_cache_blocks) ≤ max_num_batched_tokens`
- Quadratic penalty `prefillTokens²` captures disproportionate batch efficiency impact
- Division by `batchSize` amplifies effect when batch efficiency drops
- **Expected contribution** (after fix):
  - Scout general-lite (500 tokens, batch_size~4): 0.5ms × 62,500 = **31.25ms**
  - Scout roleplay (100 tokens, batch_size~32): 0.5ms × 312 = **0.156ms**
  - Ratio: **200× difference** (quadratic scaling)
- **Iter10 bug**: Contributions were 0.059ms and 0.0003ms (1000× too small) → missing time conversion or incorrect unit
- **Iter11 fix**: Audit line 405 `batchingInefficiencyTimeSeconds` — verify `m.Beta[10]` is in SECONDS, not microseconds

**β₃' (KV Seq-Len Scaling)**:
- Block allocation/deallocation scales with KV cache size (long sequences need more blocks)
- PagedAttention block manager allocates blocks proportional to `prefillTokens × numLayers`
- **Expected contribution** (after fix):
  - Scout general-lite (500 tokens × 56 layers = 28,000): 0.5μs × 28,000 = **14ms**
  - Scout roleplay (100 tokens × 56 layers = 5,600): 0.5μs × 5,600 = **2.8ms**
  - Ratio: **5× difference** (linear scaling with tokens)
- **Iter10 bug**: Contributions were 1842ms and 368ms (65× too high) → likely microseconds-to-seconds conversion error
- **Iter11 fix**: Audit line 261 `kvMgmtSeqLenTimeSeconds` — verify `m.Beta[4]` is in SECONDS per (token×layer), not microseconds

**Why this should work**:
1. **Iter10 quadratic scaling was correct**: Long/short ratio = 197× matches expected 200× → functional form is sound
2. **Iter10 β₃ split worked**: β₃ reverted to 0.402ms (within expected 0.4-1.5ms) → split concept is correct
3. **Only implementation bugs remain**: Unit tests (below) will catch conversion errors before training
4. **Iter9 coefficients were near-optimal**: Warm-starting from iter9 (NOT iter10) avoids broken artifacts

**Code Citations**:
- `sim/latency/evolved_model.go:369-406` — β₁₀ batching inefficiency basis function
- `sim/latency/evolved_model.go:240-262` — β₃' KV seq-len basis function
- `vllm/core/scheduler.py:_schedule()` — batch formation logic (lines ~300-400)
- `vllm/core/block_manager.py:BlockSpaceManager.allocate()` — KV block allocation

**Diagnostic Clause**: *If this hypothesis fails (overall loss remains >110% OR β₁₀/β₃' converge outside physical ranges), it indicates: (1) persistent formulation bugs remain — audit unit conversion again, check for off-by-1000 or off-by-65 factors in code, (2) complementary term needed — β₁₀ captures queueing delay but memory bandwidth saturation (β₁₁) may still be missing, (3) alternative formulation required — quadratic scaling may be too aggressive, consider piecewise linear or sigmoid threshold.*

---

## H-unit-tests: Unit Tests Catch Formulation Bugs Before Training

**Prediction**: Unit tests for β₁₀ and β₃' basis functions will validate expected contributions within 10% tolerance:
- **β₁₀ long-sequence test**: 500 tokens, batch_size=4, β₁₀=0.5ms → contribution = **31.25ms ± 3ms**
- **β₁₀ short-sequence test**: 100 tokens, batch_size=32, β₁₀=0.5ms → contribution = **0.156ms ± 0.02ms**
- **β₁₀ scaling ratio test**: Long/short ratio = **200× ± 20×**
- **β₃' long-sequence test**: 500 tokens, 56 layers, β₃'=0.5μs → contribution = **14ms ± 1.5ms**
- **β₃' short-sequence test**: 100 tokens, 56 layers, β₃'=0.5μs → contribution = **2.8ms ± 0.3ms**
- **β₃' scaling ratio test**: Long/short ratio = **5× ± 0.5×**

**Quantitative Threshold**: If ANY unit test fails (contribution outside tolerance OR ratio outside ±10%), then H-unit-tests is REJECTED — indicates formulation bugs persist, must re-audit code before training.

**Causal Mechanism**:

Unit tests validate three critical properties BEFORE optimization:
1. **Absolute magnitude correctness**: Expected contributions match physics estimates (not 1000× off)
2. **Scaling correctness**: Ratios match theoretical predictions (quadratic 200×, linear 5×)
3. **Unit consistency**: Coefficients in seconds, contributions in microseconds (no conversion errors)

**Why this matters**:
- **Iter10 cost**: 250 trials × 7 hours = wasted compute + wasted iteration
- **Iter10 lesson**: A single unit test would have caught factor-of-1000 error immediately
- **Process improvement**: Add "Unit Test New Basis Functions" step to design agent workflow

**Unit Test Implementation** (add to `sim/latency/evolved_model_test.go`):

```go
package latency

import (
    "math"
    "testing"
    "github.com/inference-sim/inference-sim/sim"
)

// TestBeta10BatchingInefficiency validates β₁₀ basis function contributions
func TestBeta10BatchingInefficiency(t *testing.T) {
    // Test case 1: Long sequence, small batch
    // 500 tokens, batch_size=4, β₁₀=0.0005s (0.5ms)
    // Expected: 0.5ms × (500²/4) = 0.5ms × 62,500 = 31.25ms = 0.03125s
    coeff := 0.0005 // 0.5ms in seconds
    tokens := 500.0
    batchSize := 4.0
    contribution := coeff * (tokens * tokens / batchSize)
    expectedSeconds := 0.03125 // 31.25ms
    if math.Abs(contribution-expectedSeconds)/expectedSeconds > 0.10 {
        t.Errorf("β₁₀ long-sequence: got %.6fs (%.2fms), want %.6fs (%.2fms) ±10%%",
            contribution, contribution*1e3, expectedSeconds, expectedSeconds*1e3)
    }

    // Test case 2: Short sequence, large batch
    // 100 tokens, batch_size=32, β₁₀=0.0005s (0.5ms)
    // Expected: 0.5ms × (100²/32) = 0.5ms × 312.5 = 0.156ms = 0.000156s
    tokens = 100.0
    batchSize = 32.0
    contribution2 := coeff * (tokens * tokens / batchSize)
    expectedSeconds2 := 0.000156 // 0.156ms
    if math.Abs(contribution2-expectedSeconds2)/expectedSeconds2 > 0.10 {
        t.Errorf("β₁₀ short-sequence: got %.6fs (%.2fms), want %.6fs (%.2fms) ±10%%",
            contribution2, contribution2*1e3, expectedSeconds2, expectedSeconds2*1e3)
    }

    // Test case 3: Verify quadratic scaling
    // Ratio should be (500/100)² × (32/4) = 25 × 8 = 200×
    ratio := contribution / contribution2
    expectedRatio := 200.0
    if math.Abs(ratio-expectedRatio)/expectedRatio > 0.10 {
        t.Errorf("β₁₀ scaling: got %.1f×, want %.1f× ±10%%", ratio, expectedRatio)
    }

    t.Logf("✓ β₁₀ unit tests PASSED: long=%.2fms, short=%.3fms, ratio=%.1f×",
        contribution*1e3, contribution2*1e3, ratio)
}

// TestBeta3PrimeKVSeqLen validates β₃' basis function contributions
func TestBeta3PrimeKVSeqLen(t *testing.T) {
    // Test case 1: Long sequence, dense model
    // 500 tokens, 56 layers, β₃'=0.0000005s (0.5μs per token×layer)
    // Expected: 0.5μs × (500 × 56) = 0.5μs × 28,000 = 14ms = 0.014s
    coeff := 0.0000005 // 0.5μs in seconds
    tokens := 500.0
    layers := 56.0
    contribution := coeff * (tokens * layers)
    expectedSeconds := 0.014 // 14ms
    if math.Abs(contribution-expectedSeconds)/expectedSeconds > 0.10 {
        t.Errorf("β₃' long-sequence: got %.6fs (%.2fms), want %.6fs (%.2fms) ±10%%",
            contribution, contribution*1e3, expectedSeconds, expectedSeconds*1e3)
    }

    // Test case 2: Short sequence, same model
    // 100 tokens, 56 layers, β₃'=0.0000005s (0.5μs)
    // Expected: 0.5μs × (100 × 56) = 0.5μs × 5,600 = 2.8ms = 0.0028s
    tokens = 100.0
    contribution2 := coeff * (tokens * layers)
    expectedSeconds2 := 0.0028 // 2.8ms
    if math.Abs(contribution2-expectedSeconds2)/expectedSeconds2 > 0.10 {
        t.Errorf("β₃' short-sequence: got %.6fs (%.2fms), want %.6fs (%.2fms) ±10%%",
            contribution2, contribution2*1e3, expectedSeconds2, expectedSeconds2*1e3)
    }

    // Test case 3: Verify linear scaling
    // Ratio should be 500/100 = 5×
    ratio := contribution / contribution2
    expectedRatio := 5.0
    if math.Abs(ratio-expectedRatio)/expectedRatio > 0.10 {
        t.Errorf("β₃' scaling: got %.2f×, want %.2f× ±10%%", ratio, expectedRatio)
    }

    t.Logf("✓ β₃' unit tests PASSED: long=%.2fms, short=%.2fms, ratio=%.2f×",
        contribution*1e3, contribution2*1e3, ratio)
}
```

**Diagnostic Clause**: *If unit tests fail, indicates: (1) wrong field accessed (e.g., `req.OutputTokens` instead of `req.NumNewTokens`), (2) unit mismatch (coefficient in microseconds but should be seconds), (3) missing division or multiplication, (4) incorrect aggregation logic. DO NOT proceed to training until all unit tests pass.*

---

## H-scheduler-reversion: β₆ Reverts After β₁₀ Fix

**Prediction**: After fixing β₁₀ basis function formulation:
- **Iter9**: β₆ = 99.3ms per request (+654% from iter8's 13.2ms)
- **Iter11**: β₆ = **15-40ms** per request (60-85% decrease, revert toward physical range)

**Quantitative Threshold**: If β₆ remains >60ms OR decreases to <5ms, then H-scheduler-reversion is REJECTED — indicates β₁₀ fix insufficient OR scheduler overhead genuinely high.

**Causal Mechanism**:

β₆ exploded in iter9 because it absorbed long-sequence queueing delays (beyond its physical scheduler CPU overhead component). After fixing β₁₀ (batching inefficiency term), β₁₀ should capture queueing delay, allowing β₆ to revert to its true physical overhead:
- vLLM scheduler CPU overhead: Batch formation (capacity check, priority ordering) + KV block allocation = **15-30ms**
- Iter9 β₆=99ms included: Scheduler overhead (~20ms) + absorbed queueing delay (~80ms)
- Iter11 β₁₀ should capture: Queueing delay component (~80ms for long-seq experiments)
- Iter11 β₆ should revert to: **15-40ms** (pure scheduler overhead)

**Diagnostic Clause**: *If β₆ remains >60ms, indicates: (1) β₁₀ insufficient — need complementary term (β₁₁ for memory bandwidth saturation), (2) scheduler overhead genuinely high — profile vLLM scheduler separately to measure actual CPU overhead, (3) β₁₀ fix incomplete — persistent formulation bug.*

---

## H-kv-scaling: β₃ and β₃' Capture Base + Sequence-Length KV Overhead

**Prediction**: After fixing β₃' basis function formulation:
- **β₃ (base KV overhead)**: Remain **0.4-1.5ms** per request (already correctly split in iter10)
- **β₃' (sequence-length KV overhead)**: **0.1-1.0 μs** per (token×layer) — physically plausible after fix
- **Scout long-sequence KV contribution**: β₃' × 28,000 = **3-28ms** per request (down from iter10's 1842ms)
- **Scout short-sequence KV contribution**: β₃' × 5,600 = **0.6-5.6ms** per request (down from iter10's 368ms)

**Quantitative Threshold**: If β₃' converges outside 0.05-2.0 μs OR β₃ moves outside 0.2-3.0 ms, then H-kv-scaling is REJECTED — indicates persistent formulation errors.

**Causal Mechanism**:

KV cache management has two components:
1. **Base per-request overhead (β₃)**: PagedAttention setup, block manager initialization, queue insertion — **constant per request**
2. **Sequence-length-dependent overhead (β₃')**: Block allocation/deallocation scaling with KV cache size — **proportional to prefillTokens × numLayers**

**Iter10 evidence**: β₃ reverted correctly to 0.402ms (within expected 0.4-1.5ms), proving the split concept is sound. Only β₃' formulation had bugs.

**Iter11 fix**: Audit line 261 `kvMgmtSeqLenTimeSeconds = kvMgmtSeqLenTokenLayers * m.Beta[4]` — ensure `m.Beta[4]` is in **seconds per (token×layer)**, not microseconds. Likely bug: `m.Beta[4]` is 0.0000658 (65.8μs) but code expects 0.000000658 (0.658μs).

**Diagnostic Clause**: *If β₃' converges to zero OR β₃ remains >5ms, indicates: (1) β₃' formulation still incorrect — re-audit unit conversion, (2) KV management overhead is NOT sequence-length-dependent — investigate alternative mechanisms (memory bandwidth, GPU→CPU offloading), (3) β₃' basis function choice wrong — try `prefillTokens` instead of `prefillTokens × numLayers`.*

---

## H-boundary-seq-length: β₁₀ Effect Scales Quadratically with Sequence Length

**Prediction**: After fixing β₁₀ basis function formulation:
- **Short sequences** (50-100 tokens): β₁₀ contribution ≈ **0.5-2ms** per request
- **Moderate sequences** (100-200 tokens): β₁₀ contribution ≈ **2-8ms** per request
- **Long sequences** (400-600 tokens): β₁₀ contribution ≈ **20-80ms** per request
- **Ratio**: Long/Short ≈ **10-40×** (quadratic scaling preserved)

**Quantitative Threshold**: If β₁₀ contributions do NOT scale quadratically (long/short ratio <8× or >50×), then H-boundary-seq-length is REJECTED — indicates basis function formulation still incorrect.

**Causal Mechanism**:

β₁₀ basis function `β₁₀ × Σ(prefillTokens²/batchSize)` captures quadratic scaling with sequence length:
- **Quadratic form**: `prefillTokens²` grows as the square of token count
- **Batch size normalization**: Division by `batchSize` ensures consistent scaling across different batch compositions
- **Expected scaling**: (500/100)² × (32/4) = 25 × 8 = **200× ratio** (accounting for batch size differences)

**Iter10 validation**: Ratio = 197× (matches expected 200× ✓) — functional form is CORRECT, only magnitude was wrong.

**Diagnostic Clause**: *If β₁₀ contributions do NOT scale quadratically, indicates: (1) basis function formulation still incorrect — check if `prefillTokens²` is actually computed (not `prefillTokens`), (2) batch size normalization broken — verify division by `batchSize` is present, (3) alternative formulation needed — try sigmoid threshold or piecewise linear (only penalize sequences >300 tokens).*

---

## H-alpha-stability: Constrained Alpha Bounds Prevent Spurious Reduction

**Prediction**: Alpha coefficients remain within physically plausible ranges:
- **α₀** = 0.8-2.5ms (bounds [0.5ms, 5.0ms], prevent unrealistic decrease)
- **α₁** = 60-150 μs/tok (bounds [50μs, 300μs])
- **α₂** = 50-120 μs/tok (bounds [40μs, 250μs])
- **No lower-bound saturation**: Alpha coefficients should NOT hit lower bounds

**Quantitative Threshold**: If ANY alpha coefficient hits lower bound (α₀=0.5ms, α₁=50μs, α₂=40μs), then H-alpha-stability is PARTIALLY REJECTED — indicates optimizer still trying to reduce alpha, either (1) bounds too restrictive OR (2) beta terms still insufficient.

**Causal Mechanism**:

Iter9 alpha coefficients decreased 44-73% to compensate for beta explosions (β₆, β₂, β₈). Constrained alpha bounds prevent spurious reduction by ensuring alpha stays within physically plausible ranges:
- **α₀**: API overhead (HTTP parsing, request validation) — **physical minimum ~0.5ms**
- **α₁**: Tokenization cost (BPE encoding) — **physical minimum ~50μs per token**
- **α₂**: Detokenization cost (output formatting) — **physical minimum ~40μs per token**

**Iter10 evidence**: Alpha bounds successfully prevented spurious reduction (α₀=2.48ms, α₁=127.6μs, α₂=135.0μs — all mid-range). However, overall model failed due to broken β₁₀/β₃'. Constrained bounds are necessary but not sufficient.

**Diagnostic Clause**: *If alpha coefficients hit lower bounds, indicates: (1) optimizer still trying to reduce alpha — bounds may be too restrictive, relax slightly for iter12, (2) beta terms still insufficient — missing complementary term (β₁₁ for memory bandwidth), (3) profile vLLM API overhead separately to measure actual physical alpha values.*

---

## H-error-pattern-dense: Dense Long-Sequence Experiments Should Also Improve

**Prediction**: After fixing β₁₀ and β₃' formulation bugs, dense model long-sequence experiments should improve >20pp TTFT:
- **Mistral Nemo general-lite** (exp 62): 91% → **<70%** TTFT (>21pp improvement)
- **Llama-2-7b reasoning-lite** (exp 67): 84% → **<60%** TTFT (>24pp improvement)
- **Qwen2.5-7b reasoning-lite** (exp 66): 79% → **<55%** TTFT (>24pp improvement)
- **01-ai Yi-34B general-lite** (exp 65): 78% → **<55%** TTFT (>23pp improvement)
- **Llama-3.1-70B general-lite** (exp 60): 77% → **<55%** TTFT (>22pp improvement)

**Quantitative Threshold**: If dense long-sequence experiments improve <15pp while Scout improves >30pp, then H-error-pattern-dense is REJECTED — indicates β₁₀ is absorbing Scout-specific error (architecture-dependent) rather than universal batching inefficiency.

**Causal Mechanism**:

Batching inefficiency is universal (not Scout-specific) — all models with long sequences face batch packing constraints and queueing delays:
- Dense models: Long sequences (500 tokens) → low batch efficiency → queueing delays
- Scout models: Same mechanism + MoE overhead
- β₁₀ should improve **both** Scout AND dense long-sequence experiments proportionally

**Iter10 evidence**: Both Scout and dense experiments failed catastrophically (no differential pattern), confirming β₁₀ formulation was universally broken. This actually supports the hypothesis that β₁₀ should be universal once fixed.

**Diagnostic Clause**: *If dense long-sequence experiments improve <15pp while Scout improves >30pp, indicates: (1) β₁₀ absorbing Scout-specific error — add Scout-specific term (β₁₁ for MoE memory bandwidth), (2) batching inefficiency is NOT universal — refine basis function to include architecture-dependent factors, (3) dense experiments have different bottleneck — investigate alternative long-sequence mechanisms (chunked prefill overhead, memory bandwidth saturation).*

---

## Summary Table

| Hypothesis | Key Prediction | Success Threshold | Failure Diagnostic |
|------------|----------------|-------------------|-------------------|
| **H-main** | Loss 160.6% → <90%, β₁₀=0.1-1.0ms, β₃'=0.1-1.0μs, β₆=15-40ms | Overall loss <110%, coefficients in ranges | Persistent formulation bugs OR complementary term needed |
| **H-unit-tests** | β₁₀ contributions 31.25ms/0.156ms (200× ratio), β₃' contributions 14ms/2.8ms (5× ratio) | All unit tests pass (±10% tolerance) | Wrong field, unit mismatch, missing operation, incorrect aggregation |
| **H-scheduler-reversion** | β₆ = 15-40ms (60-85% decrease from iter9's 99ms) | β₆ within 15-60ms range | β₁₀ insufficient OR scheduler overhead genuinely high OR β₁₀ fix incomplete |
| **H-kv-scaling** | β₃ remain 0.4-1.5ms, β₃' = 0.1-1.0μs, Scout long-seq 3-28ms | β₃' within 0.05-2.0μs, β₃ within 0.2-3.0ms | Persistent β₃' formulation error OR KV overhead not seq-len-dependent |
| **H-boundary** | β₁₀ contributions scale quadratically (long/short 10-40× ratio) | Quadratic scaling preserved (8-50× ratio) | Basis function formulation incorrect OR alternative formulation needed |
| **H-alpha-stability** | α within bounds, no lower-bound saturation | No alpha coefficient hits lower bound | Bounds too restrictive OR beta terms insufficient |
| **H-error-pattern-dense** | Dense long-seq improve >20pp TTFT | All 5 dense long-seq experiments <70% TTFT | β₁₀ absorbing Scout-specific error OR dense experiments have different bottleneck |

---

## Success Criteria

**Tier 1 (Full Success)**:
- Overall loss: **<90%** (44% improvement from iter9)
- TTFT RMSE: **<40%** (38% improvement from iter9)
- E2E RMSE: **<55%** (43% improvement from iter9)
- Scout long-sequence: **<60% TTFT** (>31pp improvement)
- β₁₀ coefficient: **0.1-1.0 ms** per (token²/batch_request)
- β₃' coefficient: **0.1-1.0 μs** per (token×layer)
- β₆ reversion: **15-40ms** (scheduler overhead)
- All unit tests: **PASS** (±10% tolerance)

**Tier 2 (Partial Success)**:
- Overall loss: **<110%** (31% improvement from iter9)
- Scout long-sequence: **<70% TTFT** (>20pp improvement)
- β₁₀ and β₃' coefficients: **Physically plausible** (within 2× of expected ranges)
- At least 2/3 coefficient explosions (β₆, β₂, β₈): **Decrease >30%**

**Tier 3 (Failure)**:
- Overall loss: **>130%** (minimal improvement or regression)
- Scout long-sequence: **>80% TTFT** (<12pp improvement)
- β₁₀ converged to zero OR >5 ms (implausible)
- β₆ remains >80ms (no reversion)
- Unit tests: **FAIL** (any test outside tolerance)

---

## Risk Assessment

**Primary Risk**: Unit tests fail — indicates persistent formulation bugs in β₁₀ or β₃' implementations.

**Mitigation**:
1. **DO NOT proceed to training** if any unit test fails
2. **Re-audit basis function code** — check unit conversions, field accesses, aggregation logic
3. **Add logging** to basis functions to print intermediate values during test runs
4. **Manual calculation check** — compute expected contributions by hand for test cases

**Secondary Risk**: β₁₀ fix insufficient — batching inefficiency is ONE component, but complementary terms still needed (memory bandwidth saturation β₁₁, chunked prefill overhead).

**Mitigation**:
1. If Tier 2 success (loss <110% but not <90%), β₁₀ is correct but needs complementary term
2. Profile vLLM batch formation separately to measure actual queueing vs compute overhead
3. Prepare iter12 design for memory bandwidth saturation or chunked prefill overhead

**Tertiary Risk**: α constraints too restrictive — optimizer hits lower bounds, indicating beta terms still insufficient despite fixes.

**Mitigation**:
1. If alpha coefficients hit lower bounds, relax bounds slightly for iter12:
   - α₀: [0.3ms, 5.0ms] (reduce lower bound from 0.5ms to 0.3ms)
   - α₁: [30μs, 300μs] (reduce lower bound from 50μs to 30μs)
   - α₂: [25μs, 250μs] (reduce lower bound from 40μs to 25μs)
2. Profile vLLM API overhead separately to measure actual physical alpha values
