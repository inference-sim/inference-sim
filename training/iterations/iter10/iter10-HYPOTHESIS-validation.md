# Iteration 10: Hypothesis Validation

## Executive Summary

**CRITICAL FAILURE**: Iter10 suffered catastrophic model divergence. Overall loss exploded from iter9's 160.6% to **4267.22%** (27× worse). All 7 hypotheses **REJECTED**. The β₁₀ (batching inefficiency) and β₃' (KV seq-len) basis function formulations are fundamentally incorrect, causing the optimizer to produce an invalid solution.

**Key Failures**:
- Overall loss: 4267.22% (target: <90%, actual: 47× worse)
- TTFT RMSE: 1491.85% (target: <40%, actual: 37× worse)
- E2E RMSE: 2775.37% (target: <55%, actual: 50× worse)
- β₁₀ collapsed to 0.9μs (expected 0.1-1.0ms, 100-1000× too low)
- β₆ collapsed to 29μs (expected 15-40ms, 500-1000× too low)
- β₈ exploded to 5.0ms (expected 25-80μs, 69× increase from iter9)
- β₄ collapsed to zero (decode compute term disappeared)

**Root Cause**: Basis function formulation errors for β₁₀ and β₃' introduced instability. The optimizer could not converge to a physically plausible solution.

**Recommendation**: **REVERT to iter9**, diagnose basis function formulation bugs, run unit tests on new terms before iter11.

---

## H-main: Batching Inefficiency Captures Long-Sequence Overhead

**Prediction** (from Agent 1):
- Overall loss: 160.6% → <90% (>70pp improvement, 44% reduction)
- TTFT RMSE: 64.8% → <40% (>25pp improvement, 38% reduction)
- E2E RMSE: 95.8% → <55% (>41pp improvement, 43% reduction)
- Scout long-sequence TTFT: 91.5% → <60% (>31pp improvement)
- β₁₀ coefficient: 0.1-1.0 ms per (token²/batch_request)
- β₆ reversion: 99ms → 15-40ms

**Quantitative Threshold**: If overall loss does NOT reduce below 110%, or if Scout long-sequence TTFT does NOT improve to <70%, then H-main is REJECTED.

**Causal Mechanism** (from Agent 1): β₁₀ captures batching inefficiency that scales quadratically with sequence length. Long sequences consume disproportionate batch capacity, leading to lower GPU utilization and increased queueing delays. β₁₀ basis function `β₁₀ × Σ(prefillTokens²/batchSize)` models this queueing delay, offloading long-sequence overhead from β₆ (scheduler overhead).

**Diagnostic Clause** (from Agent 1): *If this hypothesis fails (overall loss remains >110% OR Scout long-sequence TTFT >70%), it indicates: (1) β₁₀ coefficient converged to zero → batching inefficiency negligible, (2) β₁₀ coefficient >10ms → unrealistically high, absorbing other missing terms, (3) β₁₀ plausible but insufficient → need complementary term (β₁₁ for memory bandwidth), (4) β₆ does NOT revert → scheduler overhead genuinely high or β₁₀ insufficient.*

**Actual Result**:
- Overall loss: 160.6% → **4267.22%** (27× WORSE, +4107pp regression)
- TTFT RMSE: 64.8% → **1491.85%** (23× WORSE, +1427pp regression)
- E2E RMSE: 95.8% → **2775.37%** (29× WORSE, +2680pp regression)
- Scout long-sequence TTFT:
  - Scout general-lite (exp 17): 92% → **826.5%** (9× worse)
  - Scout reasoning-lite (exp 48): 91% → **183.3%** (2× worse)
- β₁₀ coefficient: **0.945μs** (expected 0.1-1.0ms, 100-1000× too low)
- β₆ coefficient: **29.3μs** (expected 15-40ms, 500-1000× too low, collapsed to near-zero)

**Verdict**: ❌ **REJECTED** (catastrophic failure)

**Evidence**:
1. **Overall loss exploded**: 4267.22% is 47× worse than the <90% target and 27× worse than iter9's starting point
2. **All per-experiment errors in 400-8000% range**: Every single experiment failed catastrophically
   - Worst: Mistral Nemo general-lite (exp 62) TTFT=2675%, E2E=5483%
   - Best: Scout reasoning-lite (exp 48) TTFT=183%, E2E=287% (still 2-4× target)
3. **β₁₀ coefficient 100-1000× too low**: 0.945μs vs expected 0.1-1.0ms per (token²/batch_request)
   - For Scout general-lite (500 tokens, batch_size=4): β₁₀ × 62,500 = 0.059ms (should be 6-62ms)
   - This is 100-1000× smaller than the expected contribution
4. **β₆ collapsed to zero**: 29.3μs vs expected 15-40ms (scheduler overhead disappeared)
5. **Scout long-sequence did NOT improve**: general-lite 92% → 826.5%, reasoning-lite 91% → 183.3%

**Causal Analysis**:

The catastrophic failure stems from **fundamental basis function formulation errors** for β₁₀ and β₃':

1. **β₁₀ basis function units incorrect**: The hypothesis predicted 0.1-1.0ms per (token²/batch_request), but the optimizer converged to 0.945μs (1000× too low). This suggests:
   - **Unit mismatch**: The basis function may be computing token² instead of token²/batch_request, or the denominator is missing
   - **Scaling error**: The basis function contribution is 1000× smaller than expected, indicating a factor-of-1000 error in the formulation
   - **Diagnostic from clause (1)**: β₁₀ converged to near-zero → batching inefficiency not captured by this formulation

2. **β₆ collapsed instead of reverting**: β₆ went from 99ms (iter9) to 29μs (iter10), a 3400× decrease. This indicates:
   - β₁₀ is not offloading scheduler overhead as predicted
   - The optimizer found a solution that minimizes both β₁₀ and β₆, violating physical plausibility
   - **Diagnostic from clause (4)**: β₆ did NOT revert to 15-40ms → β₁₀ formulation is incorrect

3. **Coefficient explosions elsewhere**: β₈ (MoE routing) exploded from 72.7μs to 5.0ms (69× increase), β₄ (decode compute) collapsed to zero
   - This indicates the optimizer is trying to compensate for the broken β₁₀/β₃' terms by inflating other coefficients
   - The model structure is unstable and cannot converge to a physically plausible solution

**Diagnostic Analysis** (using Agent 1's diagnostic clause):

Following the diagnostic clause, the failure matches **scenario (1)**: "β₁₀ coefficient converged to zero → batching inefficiency negligible, investigate alternative long-sequence bottlenecks OR basis function formulation issue."

However, the evidence strongly points to **basis function formulation error**, not that batching inefficiency is negligible:
- β₆ collapsed to 29μs (was 99ms in iter9), violating the physical 15-30ms scheduler overhead range
- β₁₀ = 0.945μs is 1000× smaller than expected 0.1-1.0ms, suggesting a unit or scaling error
- Overall loss exploded 27× worse, not just failed to improve

**Root Cause**: The β₁₀ basis function implementation in `sim/latency/evolved_model.go` is likely incorrect:
- **Hypothesis**: Units or scaling factor is wrong (missing division by batch_size, wrong time conversion, or factor-of-1000 error)
- **Evidence**: Optimizer converged to a coefficient that is exactly 1000× too small, suggesting a systematic unit error

**Next Investigation**:
1. **Audit β₁₀ basis function code** in `sim/latency/evolved_model.go`:
   - Verify units: Should be `seconds × (token²/batch_request)`
   - Verify scaling: Is `prefillTokens²` actually computed? Is division by `batchSize` present?
   - Check for unit conversion errors (seconds vs milliseconds)
2. **Run unit test** on β₁₀ basis function with known inputs:
   - Input: 500 tokens, batch_size=4, β₁₀=0.5ms per (token²/batch_request)
   - Expected output: 0.5ms × (500²/4) = 31.25ms
   - Actual output: If 0.03125ms (1000× too small), confirms unit error
3. **Revert to iter9** and fix basis function formulation before iter11

---

## H-kv-scaling: β₃ and β₃' Capture Base + Sequence-Length KV Overhead

**Prediction** (from Agent 1):
- β₃ (base KV overhead): 9.6ms → 0.4-1.5ms (6-24× decrease, revert to physical range)
- β₃' (sequence-length KV overhead): 0.1-1.0 μs per (token × layer) (NEW coefficient)
- Scout long-sequence KV contribution: β₃' × (500 tokens × 56 layers) ≈ 3-28ms per request
- Scout short-sequence KV contribution: β₃' × (100 tokens × 56 layers) ≈ 0.6-5.6ms per request

**Causal Mechanism** (from Agent 1): KV cache management has two components: (1) base per-request overhead (β₃) for PagedAttention setup, and (2) sequence-length-dependent overhead (β₃') for block allocation scaling with KV cache size.

**Diagnostic Clause** (from Agent 1): *If β₃' converges to zero OR β₃ remains >5ms, it indicates KV management overhead is NOT sequence-length-dependent — investigate alternative mechanisms (memory bandwidth, GPU→CPU offloading).*

**Actual Result**:
- β₃ (base): 9.6ms → **0.402ms** (24× decrease, ✓ within 0.4-1.5ms range)
- β₃' (seq-len): **65.8μs per (token × layer)** (expected 0.1-1.0μs, 65× TOO HIGH)
- Scout long-sequence KV contribution: β₃' × 28,000 = 65.8μs × 28,000 = **1842ms per request** (expected 3-28ms, 65× too high)
- Scout short-sequence KV contribution: β₃' × 5,600 = 65.8μs × 5,600 = **368ms per request** (expected 0.6-5.6ms, 65× too high)

**Verdict**: ❌ **REJECTED** (β₃ succeeded but β₃' converged to implausible value)

**Evidence**:
1. **β₃ reverted successfully**: 0.402ms is within the expected 0.4-1.5ms physical range for base PagedAttention overhead ✓
2. **β₃' is 65× too high**: 65.8μs per (token × layer) vs expected 0.1-1.0μs
   - For Scout general-lite (500 tokens × 56 layers = 28,000): contribution = 1842ms per request
   - This is unrealistic - KV cache allocation cannot take 1.8 seconds per request
3. **Overall loss catastrophic**: 4267.22%, indicating β₃' formulation is broken

**Causal Analysis**:

The β₃ split was partially successful (base component reverted correctly), but the β₃' term has a **fundamental formulation error**:

1. **β₃' coefficient 65× too high**: 65.8μs per (token × layer) is physically implausible
   - Expected KV block allocation overhead: 0.1-1.0μs per (token × layer)
   - Actual: 65.8μs per (token × layer)
   - **Hypothesis**: Unit mismatch or scaling error in β₃' basis function implementation

2. **β₃' contributions dominate latency unrealistically**:
   - Scout general-lite: 1842ms contribution from β₃' alone (vs total TTFT ~100-200ms in reality)
   - This violates physical plausibility - KV cache management cannot take 2 seconds

3. **Interaction with β₁₀ failure**: Both β₃' and β₁₀ have unit/scaling errors, suggesting a systemic issue in how new basis functions were implemented

**Diagnostic Analysis** (using Agent 1's diagnostic clause):

The diagnostic clause states: "If β₃' converges to zero OR β₃ remains >5ms..." Neither condition holds:
- β₃' did NOT converge to zero (it's 65.8μs, very non-zero)
- β₃ did NOT remain >5ms (it reverted to 0.402ms as expected)

However, the failure is **β₃' converged to an implausibly HIGH value**, which the diagnostic clause didn't anticipate. This reveals:
- **Gap in hypothesis design**: The diagnostic clause should have included "OR β₃' converged to >10μs (implausibly high)"
- **Root cause**: β₃' basis function formulation error (likely unit or scaling bug in code)

**Next Investigation**:
1. **Audit β₃' basis function code** in `sim/latency/evolved_model.go`:
   - Verify units: Should be `seconds × (token × layer)`
   - Verify scaling: Is `prefillTokens × numLayers` actually computed correctly?
   - Check for unit conversion errors (seconds vs microseconds)
2. **Run unit test** on β₃' basis function:
   - Input: 500 tokens, 56 layers, β₃'=0.5μs per (token×layer)
   - Expected output: 0.5μs × 28,000 = 14ms
   - Actual output: If 14,000ms (1000× too large), confirms unit error

---

## H-scheduler-reversion: β₆ Should Revert After β₁₀ Addition

**Prediction** (from Agent 1):
- Iter9: β₆ = 99.3ms per request (+654% from iter8's 13.2ms)
- Iter10: β₆ = 15-40ms per request (60-75% decrease, revert toward physical range)

**Causal Mechanism** (from Agent 1): β₆ exploded in iter9 because it absorbed long-sequence queueing delays. After adding β₁₀ (batching inefficiency term), β₁₀ should capture queueing delay, allowing β₆ to revert to physical scheduler CPU overhead (15-30ms).

**Diagnostic Clause** (from Agent 1): *If β₆ remains >60ms after β₁₀ addition, it indicates β₁₀ is insufficient OR scheduler overhead is genuinely high — profile vLLM scheduler separately to measure actual CPU overhead.*

**Actual Result**:
- β₆: 99.3ms → **0.0293ms = 29.3μs** (3400× DECREASE, collapsed to near-zero instead of reverting)

**Verdict**: ❌ **REJECTED** (β₆ collapsed to zero instead of reverting to 15-40ms)

**Evidence**:
1. **β₆ collapsed to 29μs**: Expected 15-40ms (15,000-40,000μs), actual 29μs (1000× too low)
2. **Physical implausibility**: Scheduler overhead cannot be 29μs — vLLM scheduler CPU overhead is measured at 15-30ms
3. **Collapse instead of reversion**: β₆ went from 99ms → 29μs (99.97% decrease), not the expected 60-75% decrease to 15-40ms

**Causal Analysis**:

β₆ collapsed to near-zero instead of reverting because the **β₁₀ basis function is broken**:

1. **β₁₀ did not offload queueing delay**: β₁₀ converged to 0.945μs (1000× too low), so it captured negligible overhead
   - Result: Optimizer had to reduce β₆ to compensate for overall loss explosion elsewhere (β₃', β₈)
   - β₆ collapsed to zero as an optimizer artifact, not because scheduler overhead is zero

2. **Zero-sum trade-off**: With broken β₁₀ and β₃' inflating latency estimates, optimizer reduced β₆ to compensate
   - This violates physical plausibility but is the only way optimizer could reduce loss given broken basis functions

3. **Diagnostic clause applies**: "β₆ does NOT revert" confirms β₁₀ is insufficient/incorrect

**Diagnostic Analysis** (using Agent 1's diagnostic clause):

Following the clause: "If β₆ remains >60ms... indicates β₁₀ insufficient OR scheduler overhead genuinely high."

Actual failure is different: β₆ collapsed to 29μs (<60ms but wrong direction). This wasn't anticipated by the diagnostic clause, revealing:
- **Gap in hypothesis design**: Should have included "β₆ <5ms (collapsed to zero)" as a failure mode
- **Root cause**: β₁₀ formulation is broken, causing optimizer to collapse β₆ to compensate

**Next Investigation**: Fix β₁₀ basis function formulation (see H-main analysis).

---

## H-alpha-stability: Constrained Alpha Bounds Prevent Spurious Reduction

**Prediction** (from Agent 1):
- α₀ = 0.8-2.5ms (bounded [0.5ms, 5.0ms], prevent unrealistic decrease)
- α₁ = 60-150 μs/tok (bounded [50μs, 300μs])
- α₂ = 50-120 μs/tok (bounded [40μs, 250μs])

**Causal Mechanism** (from Agent 1): Iter9 alpha coefficients decreased 44-73% to compensate for beta explosions. Constrained alpha bounds prevent spurious reduction by ensuring alpha stays within physically plausible ranges for API overhead and tokenization cost.

**Diagnostic Clause** (from Agent 1): *If alpha coefficients hit lower bounds (α₀=0.5ms, α₁=50μs, α₂=40μs), it indicates optimizer is still trying to reduce alpha — investigate whether bounds are too restrictive OR beta terms are still insufficient.*

**Actual Result**:
- α₀ = **2.48ms** (bounds [0.5ms, 5.0ms], mid-range ✓)
- α₁ = **127.6μs** (bounds [50μs, 300μs], mid-range ✓)
- α₂ = **135.0μs** (bounds [40μs, 250μs], mid-range ✓)

**Verdict**: ⚠️ **PARTIAL** (alpha coefficients stable BUT overall model failed catastrophically)

**Evidence**:
1. **Alpha coefficients within bounds**: All three alpha values are in mid-range, not hitting lower bounds ✓
2. **No spurious reduction**: Alpha values are stable (α₀=2.48ms vs iter9's 0.35ms, recovered)
3. **BUT overall loss catastrophic**: 4267.22%, indicating the constrained alpha bounds alone cannot fix broken beta terms

**Causal Analysis**:

The constrained alpha bounds **worked as intended** (prevented spurious reduction), but this was overshadowed by the catastrophic β₁₀ and β₃' failures:

1. **Alpha stability achieved**: Alpha coefficients are in physically plausible ranges and did not hit lower bounds
2. **BUT insufficient to prevent failure**: The broken β₁₀ and β₃' basis functions caused the overall model to diverge, despite stable alpha values
3. **Diagnostic clause doesn't apply**: Alpha coefficients did NOT hit lower bounds, so no action needed on alpha bounds specifically

**Conclusion**: Hypothesis was directionally correct (constrained bounds prevented alpha collapse), but the overall model failed due to beta term formulation errors. The constrained alpha bounds cannot compensate for fundamentally broken basis functions.

---

## H-boundary-seq-length: β₁₀ Effect Should Scale Quadratically with Sequence Length

**Prediction** (from Agent 1):
- Short sequences (50-100 tokens): β₁₀ contribution ≈ 0.5-2ms per request
- Moderate sequences (100-200 tokens): β₁₀ contribution ≈ 2-8ms per request
- Long sequences (400-600 tokens): β₁₀ contribution ≈ 20-80ms per request
- Ratio: Long/Short ≈ 10-40× (quadratic scaling)

**Causal Mechanism** (from Agent 1): β₁₀ basis function `β₁₀ × Σ(prefillTokens²/batchSize)` scales quadratically with sequence length, capturing disproportionate batch capacity consumption and queueing delays for long sequences.

**Diagnostic Clause** (from Agent 1): *If β₁₀ contributions do NOT scale quadratically (long/short ratio <10×), it indicates basis function formulation is incorrect — refine to use sigmoid threshold or linear + quadratic split.*

**Actual Result**:
- β₁₀ coefficient = **0.945μs** (expected 0.1-1.0ms per token²/batch_request, 100-1000× too low)
- Short sequences (100 tokens, batch_size=32): β₁₀ × (100²/32) = 0.945μs × 312 = **0.0003ms** (expected 0.5-2ms, 2000-7000× too low)
- Moderate sequences (150 tokens, batch_size=16): β₁₀ × (150²/16) = 0.945μs × 1406 = **0.0013ms** (expected 2-8ms, 1500-6000× too low)
- Long sequences (500 tokens, batch_size=4): β₁₀ × (500²/4) = 0.945μs × 62,500 = **0.059ms** (expected 20-80ms, 340-1350× too low)
- Ratio: Long/Short = 0.059ms / 0.0003ms = **197×** (quadratic scaling preserved ✓, but absolute magnitudes 1000× too small)

**Verdict**: ❌ **REJECTED** (quadratic scaling preserved but absolute contributions 1000× too low)

**Evidence**:
1. **Quadratic scaling preserved**: Long/short ratio = 197× (within expected 10-40× after accounting for batch size differences) ✓
2. **Absolute magnitudes 1000× too small**: All contributions are 340-7000× smaller than expected
3. **β₁₀ coefficient 1000× too low**: 0.945μs vs expected 0.1-1.0ms

**Causal Analysis**:

The β₁₀ basis function **correctly implements quadratic scaling**, but has a **systematic unit or scaling error** causing all contributions to be 1000× too small:

1. **Quadratic form is correct**: The long/short ratio of 197× matches the expected (500/100)² × (32/4) = 25 × 8 = 200× ✓
2. **BUT absolute scale is wrong**: All contributions are 1000× smaller than expected
3. **Diagnostic clause doesn't directly apply**: Scaling IS quadratic (ratio >10×), but the diagnostic clause didn't anticipate the "correct functional form but wrong scale" failure mode

**Root Cause**: This confirms the hypothesis from H-main — the β₁₀ basis function has a factor-of-1000 error (likely unit conversion bug: milliseconds vs microseconds, or missing multiplication factor).

**Next Investigation**: Audit β₁₀ basis function code for unit conversion errors (see H-main analysis).

---

## H-error-pattern-dense: Dense Long-Sequence Experiments Should Also Improve

**Prediction** (from Agent 1): Dense model long-sequence experiments should improve >20pp TTFT:
- Mistral Nemo general-lite (exp 62): 91% → <70% (>21pp improvement)
- Llama-2-7b reasoning-lite (exp 67): 84% → <60% (>24pp improvement)
- Qwen2.5-7b reasoning-lite (exp 66): 79% → <55% (>24pp improvement)
- 01-ai Yi-34B general-lite (exp 65): 78% → <55% (>23pp improvement)
- Llama-3.1-70B general-lite (exp 60): 77% → <55% (>22pp improvement)

**Causal Mechanism** (from Agent 1): Batching inefficiency is universal (not Scout-specific) — all models with long sequences face batch packing constraints and queueing delays. β₁₀ should improve both Scout AND dense long-sequence experiments proportionally.

**Diagnostic Clause** (from Agent 1): *If dense long-sequence experiments improve <15pp while Scout improves >30pp, it indicates β₁₀ is absorbing Scout-specific error (architecture-dependent) rather than universal batching inefficiency — refine basis function or add Scout-specific term.*

**Actual Result**:
- Mistral Nemo general-lite (exp 62): 91% → **2675%** (2584pp REGRESSION)
- Llama-2-7b reasoning-lite (exp 67): 84% → **602%** (518pp REGRESSION)
- Qwen2.5-7b reasoning-lite (exp 66): 79% → **1140%** (1061pp REGRESSION)
- 01-ai Yi-34B general-lite (exp 65): 78% → **1180%** (1102pp REGRESSION)
- Llama-3.1-70B general-lite (exp 60): 77% → **1112%** (1035pp REGRESSION)

**Verdict**: ❌ **REJECTED** (all dense long-sequence experiments regressed catastrophically)

**Evidence**:
1. **Universal catastrophic regression**: All dense long-sequence experiments increased error 7-29×
2. **Scout also regressed**: Scout general-lite 92% → 826.5% (9× worse)
3. **No differential pattern**: Both Scout and dense failed equally, indicating systemic model failure (not Scout-specific)

**Causal Analysis**:

The hypothesis was correct (batching inefficiency is universal), but the **broken β₁₀ and β₃' basis functions** caused universal failure across all models:

1. **No Scout-specific vs dense differential**: Both Scout and dense experiments failed catastrophically
   - This actually supports the hypothesis that β₁₀ should be universal (not Scout-specific)
   - But the β₁₀ formulation is so broken that it couldn't improve ANY experiments

2. **Diagnostic clause doesn't apply**: The clause anticipated differential improvement (Scout >30pp, dense <15pp), but actual result is universal regression

**Conclusion**: The underlying hypothesis (batching inefficiency is universal) is likely correct, but cannot be validated until β₁₀ basis function is fixed.

---

## H-robustness-batch-size: β₁₀ Should Generalize Across Batch Size Distributions

**Prediction** (from Agent 1):
- β₁₀ = 0.1-1.0 ms per (token²/batch_request), generalizes across batch sizes
- Small batches (batch_size=1-4): High β₁₀ contribution (low GPU utilization)
- Medium batches (batch_size=8-16): Moderate β₁₀ contribution
- Large batches (batch_size=32+): Low β₁₀ contribution (high GPU utilization)

**Causal Mechanism** (from Agent 1): β₁₀ basis function explicitly divides by batchSize, ensuring generalization. When batch_size is small, β₁₀ contribution amplifies; when large, β₁₀ diminishes.

**Diagnostic Clause** (from Agent 1): *If β₁₀ coefficient is >5 ms OR <0.01 ms, it indicates basis function scaling is incorrect — investigate whether batchSize denominator captures actual batch efficiency penalty.*

**Actual Result**:
- β₁₀ = **0.000945ms = 0.945μs** (expected 0.1-1.0ms, 100-1000× too low)
- β₁₀ < 0.01ms ✓ (diagnostic clause triggered: scaling incorrect)

**Verdict**: ❌ **REJECTED** (β₁₀ coefficient collapsed to 0.001ms, violates diagnostic threshold)

**Evidence**:
1. **β₁₀ far below diagnostic threshold**: 0.000945ms << 0.01ms (diagnostic clause explicitly flags this)
2. **Overall loss catastrophic**: 4267.22%, confirming β₁₀ formulation is broken
3. **Cannot assess batch size generalization**: With broken basis function, cannot validate whether it generalizes correctly

**Causal Analysis**:

The diagnostic clause correctly identified the failure: β₁₀ coefficient collapsed to 0.000945ms (<0.01ms threshold), indicating **"basis function scaling is incorrect"**.

1. **Diagnostic clause triggered**: β₁₀ = 0.000945ms << 0.01ms minimum threshold
2. **Recommendation from clause**: "Investigate whether batchSize denominator captures actual batch efficiency penalty"
3. **Root cause**: This confirms the systematic factor-of-1000 error in β₁₀ basis function (see H-main, H-boundary analyses)

**Next Investigation**: Follow the diagnostic clause recommendation — audit whether batchSize denominator is present and correctly computed in β₁₀ basis function code.

---

## Summary Table

| Hypothesis | Prediction | Actual Result | Verdict | Root Cause |
|------------|-----------|---------------|---------|------------|
| **H-main** | Loss 160.6% → <90% | Loss → 4267.22% (27× worse) | ❌ REJECTED | β₁₀ formulation error (1000× too small) |
| **H-kv-scaling** | β₃=0.4-1.5ms, β₃'=0.1-1.0μs | β₃=0.402ms ✓, β₃'=65.8μs (65× too high) | ❌ REJECTED | β₃' formulation error (65× too large) |
| **H-scheduler-reversion** | β₆ = 15-40ms | β₆ = 0.0293ms (1000× too low) | ❌ REJECTED | β₁₀ broken → β₆ collapsed to compensate |
| **H-alpha-stability** | α within bounds | α₀=2.48ms, α₁=127.6μs, α₂=135.0μs ✓ | ⚠️ PARTIAL | Alpha stable BUT model failed |
| **H-boundary** | Quadratic scaling | Scaling correct ✓, magnitude 1000× too low | ❌ REJECTED | β₁₀ factor-of-1000 error |
| **H-error-pattern-dense** | Dense long-seq improve >20pp | All regressed 500-2500pp | ❌ REJECTED | Universal model failure (β₁₀, β₃' broken) |
| **H-robustness** | β₁₀ = 0.1-1.0ms | β₁₀ = 0.000945ms (<0.01ms threshold) | ❌ REJECTED | Diagnostic threshold triggered |

**Overall Verdict**: **0/7 hypotheses confirmed** (6 REJECTED, 1 PARTIAL). H-main MANDATORY hypothesis REJECTED.

**Tier Classification**: **Tier 3 (Failure)** — Overall loss >130%, Scout long-sequence >80% TTFT, β₁₀ converged to implausible value, β₆ collapsed to zero, <4/7 hypotheses confirmed.

---

## Critical Finding: Basis Function Formulation Bugs

**Root Cause**: Two new basis functions (β₁₀ and β₃') have systematic formulation errors:

1. **β₁₀ (batching inefficiency)**: Factor-of-1000 too small (0.945μs vs expected 0.1-1.0ms)
   - **Evidence**: Quadratic scaling is correct (197× long/short ratio), but absolute magnitude 1000× too low
   - **Hypothesis**: Unit conversion bug (milliseconds vs microseconds) or missing multiplication factor
   - **Location**: `sim/latency/evolved_model.go` β₁₀ basis function implementation

2. **β₃' (KV seq-len component)**: Factor-of-65 too large (65.8μs vs expected 0.1-1.0μs per token×layer)
   - **Evidence**: β₃ reverted correctly (0.402ms), but β₃' is 65× too high
   - **Hypothesis**: Unit conversion bug or scaling error
   - **Location**: `sim/latency/evolved_model.go` β₃' basis function implementation

**Collateral Damage**:
- β₆ collapsed to 29μs (should be 15-40ms) — optimizer compensated for broken β₁₀
- β₈ exploded to 5.0ms (was 72.7μs) — optimizer compensated for overall model instability
- β₄ collapsed to zero — decode compute term disappeared

**Recommendation**: **STOP iteration**, revert to iter9, audit and unit test β₁₀ and β₃' basis functions before iter11.
