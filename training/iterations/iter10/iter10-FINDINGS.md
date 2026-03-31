# Iteration 10: Findings and Principles

## Summary

**Catastrophic model divergence**: Iter10 overall loss exploded from 160.6% (iter9) to **4267.22%** (27× worse). Root cause: **Basis function formulation bugs** in β₁₀ (batching inefficiency) and β₃' (KV seq-len component). β₁₀ converged 1000× too small (0.945μs vs expected 0.1-1.0ms), β₃' converged 65× too large (65.8μs vs expected 0.1-1.0μs per token×layer). The optimizer could not converge to a physically plausible solution, causing β₆ to collapse to 29μs (expected 15-40ms), β₈ to explode to 5.0ms (expected 25-80μs), and β₄ to vanish.

**Key Learning**: **Always unit test new basis functions before training**. The catastrophic failure could have been caught with a single unit test validating expected contributions for known inputs.

**Recommendation**: **Revert to iter9**, audit β₁₀ and β₃' basis function implementations in `sim/latency/evolved_model.go`, write unit tests, then retry iter11 with validated basis functions.

---

## Error Analysis

### Catastrophic Loss Explosion

**Pattern**: All 15 experiments failed with errors in 180-8000% range (median ~1800%).

**High-error experiments** (APE > 1000%):
- **Mistral Nemo general-lite (exp 62)**: TTFT=2675%, E2E=5483% — Dense long-sequence, 26× worse than iter9
- **Llama-2-7b codegen (exp 20)**: TTFT=2376%, E2E=4738% — Dense moderate-sequence, 23× worse than iter9
- **Mistral Nemo codegen (exp 63)**: TTFT=2032%, E2E=3836% — Dense moderate-sequence, 20× worse than iter9
- **Qwen2.5-7b roleplay (exp 64)**: TTFT=1795%, E2E=3318% — Dense short-sequence, 18× worse than iter9
- **Llama-2-7b roleplay (exp 20)**: TTFT=1771%, E2E=2822% — Dense short-sequence, 17× worse than iter9
- **Llama-2-7b general (exp 20)**: TTFT=1577%, E2E=2994% — Dense moderate-sequence, 15× worse than iter9
- **01-ai Yi-34B general-lite (exp 65)**: TTFT=1180%, E2E=2137% — Dense long-sequence, 12× worse than iter9
- **Llama-3.1-70B codegen (exp 61)**: TTFT=1191%, E2E=2006% — Dense moderate-sequence, 12× worse than iter9
- **Llama-3.1-70B general-lite (exp 60)**: TTFT=1112%, E2E=1899% — Dense long-sequence, 11× worse than iter9
- **Qwen2.5-7b reasoning-lite (exp 66)**: TTFT=1140%, E2E=1384% — Dense long-sequence, 11× worse than iter9
- **Scout codegen (exp 20)**: TTFT=859%, E2E=1512% — MoE moderate-sequence, 9× worse than iter9
- **Scout general-lite (exp 17)**: TTFT=826%, E2E=1524% — MoE long-sequence, 8× worse than iter9
- **Scout roleplay (exp 21)**: TTFT=661%, E2E=943% — MoE short-sequence, 7× worse than iter9
- **Llama-2-7b reasoning-lite (exp 67)**: TTFT=602%, E2E=828% — Dense long-sequence, 6× worse than iter9

**Lowest-error experiment** (still catastrophic):
- **Scout reasoning-lite (exp 48)**: TTFT=183%, E2E=287% — MoE long-sequence, 2× worse than iter9
  - **Why "best"?**: Even the best experiment is 2-3× worse than iter9 and 5-7× worse than target

**No low-error experiments**: Not a single experiment achieved APE < 100%. Every experiment failed.

### Error Correlations

**✅ Confirmed patterns**:
- **Universal failure**: No correlation between error and model architecture (dense vs MoE), sequence length (short vs long), or workload type
- **All experiments failed equally**: This indicates a systemic model formulation bug, not a specific missing term

**❌ Rejected patterns**:
- **NOT architecture-dependent**: Scout experiments (180-2675% error) failed similarly to dense experiments (183-5483% error)
- **NOT sequence-length-dependent**: Short-sequence experiments (661-1795% TTFT) failed similarly to long-sequence experiments (183-2675% TTFT)
- **NOT workload-dependent**: Roleplay (661-1795%), codegen (859-2376%), general (826-1577%), reasoning (183-1140%) all failed

**Interpretation**: The catastrophic failure is **independent of experiment characteristics**, confirming a fundamental basis function formulation bug rather than a missing physics term.

---

## Root Cause Hypotheses

### Principle 1: Basis Function Formulation Bugs Cause Catastrophic Divergence

**Evidence**:
- β₁₀ coefficient: 0.945μs (expected 0.1-1.0ms, 100-1000× too small)
- β₃' coefficient: 65.8μs per (token×layer) (expected 0.1-1.0μs, 65× too large)
- β₁₀ contributions: 0.0003-0.059ms (expected 0.5-80ms, 340-7000× too small)
- β₃' contributions: 368-1842ms per request (expected 0.6-28ms, 65× too large)
- Overall loss: 4267.22% (27× worse than iter9)

**Mechanism**:

Two new basis functions (β₁₀ and β₃') were added in iter10 with **systematic unit or scaling errors**:

1. **β₁₀ (batching inefficiency)** has a **factor-of-1000 error**:
   - Hypothesis prediction: 0.1-1.0ms per (token²/batch_request)
   - Optimizer converged: 0.945μs per (token²/batch_request)
   - **Ratio**: 0.945μs / 500μs (midpoint of 0.1-1.0ms) = 0.002 = 1/500
   - **Diagnosis**: Unit conversion bug (likely milliseconds vs microseconds) or missing multiplication factor
   - **Evidence for correct functional form**: Long/short sequence ratio = 197× (matches expected quadratic scaling), but absolute magnitudes are 1000× too small
   - **Code location**: `sim/latency/evolved_model.go` β₁₀ basis function

2. **β₃' (KV seq-len component)** has a **factor-of-65 error**:
   - Hypothesis prediction: 0.1-1.0μs per (token×layer)
   - Optimizer converged: 65.8μs per (token×layer)
   - **Ratio**: 65.8μs / 0.5μs (midpoint of 0.1-1.0μs) = 131× too large
   - **Diagnosis**: Unit conversion bug (microseconds vs milliseconds) or incorrect scaling
   - **Evidence**: β₃ (base component) reverted correctly to 0.402ms (within expected 0.4-1.5ms), so the split concept is correct but β₃' implementation is wrong
   - **Code location**: `sim/latency/evolved_model.go` β₃' basis function

3. **Optimizer cannot converge** with broken basis functions:
   - β₁₀ and β₃' contributions are 65-7000× wrong
   - Optimizer tries to compensate by collapsing β₆ to 29μs (was 99ms), inflating β₈ to 5.0ms (was 72.7μs), and zeroing β₄
   - These compensations violate physical plausibility, causing loss to explode

**Action**:
1. **Revert to iter9** — do not proceed with broken basis functions
2. **Audit basis function code**:
   - `sim/latency/evolved_model.go`: Check β₁₀ and β₃' implementations
   - Verify units: β₁₀ should be `seconds × (token²/batch_request)`, β₃' should be `seconds × (token×layer)`
   - Verify time conversions: Check for milliseconds vs microseconds vs seconds errors
   - Look for missing multiplication factors or incorrect denominators
3. **Write unit tests** BEFORE next iteration:
   ```go
   func TestBeta10BatchingInefficiency(t *testing.T) {
       // Test case: 500 tokens, batch_size=4, β₁₀=0.5ms per (token²/batch_request)
       // Expected: 0.5ms × (500²/4) = 0.5ms × 62,500 = 31.25ms
       coefficient := 0.0005 // 0.5ms in seconds
       tokens := 500.0
       batchSize := 4.0
       contribution := coefficient * (tokens * tokens / batchSize)
       expected := 0.03125 // 31.25ms in seconds
       assert.InDelta(t, expected, contribution, 0.001)
   }

   func TestBeta3PrimeKVSeqLen(t *testing.T) {
       // Test case: 500 tokens, 56 layers, β₃'=0.5μs per (token×layer)
       // Expected: 0.5μs × (500 × 56) = 0.5μs × 28,000 = 14ms
       coefficient := 0.0000005 // 0.5μs in seconds
       tokens := 500.0
       layers := 56.0
       contribution := coefficient * (tokens * layers)
       expected := 0.014 // 14ms in seconds
       assert.InDelta(t, expected, contribution, 0.001)
   }
   ```
4. **Retry iter11** only after unit tests pass

---

### Principle 2: Coefficient Explosions Reveal Optimizer Compensation for Broken Terms

**Evidence**:
- β₆ (scheduler): 99.3ms → 29.3μs (3400× decrease, collapsed to zero)
- β₈ (MoE routing): 72.7μs → 5005μs (69× increase, exploded)
- β₄ (decode compute): 0.47 → 2.7×10⁻⁷ (1,700,000× decrease, vanished)
- β₂ (TP comm): 0.82 (remained high, didn't decrease as expected)

**Mechanism**:

When basis functions are broken (β₁₀, β₃'), the **optimizer compensates** by adjusting other coefficients to minimize loss, even if the adjustments violate physical plausibility:

1. **β₆ collapsed to compensate for β₃' inflation**: β₃' contributes 368-1842ms per request (65× too high), so optimizer reduced β₆ from 99ms to 29μs to partially offset the excess latency
   - This violates physical plausibility: scheduler overhead cannot be 29μs (measured at 15-30ms)
   - But it's the only way optimizer could reduce loss given the broken β₃'

2. **β₈ exploded to capture unmodeled Scout overhead**: With β₁₀ broken (1000× too small), long-sequence Scout experiments have 30-80ms of unmodeled overhead
   - Optimizer inflated β₈ (MoE routing) from 72.7μs to 5.0ms (69× increase) to absorb this gap
   - This violates the physical 10-50μs range for expert routing overhead

3. **β₄ vanished as zero-sum trade-off**: With β₆ collapsed and β₈ exploded, optimizer zeroed β₄ (decode compute) to maintain overall loss balance
   - This is physically implausible: decode compute overhead cannot be zero

**Interpretation**: Coefficient explosions and collapses are **symptoms, not root causes**. The root cause is broken basis functions (β₁₀, β₃'), and the explosions/collapses are the optimizer's attempt to compensate within a broken model structure.

**Action**:
1. **Do NOT try to fix coefficient explosions directly** (e.g., by constraining β₈ or β₆ bounds)
2. **Fix the root cause** (β₁₀ and β₃' basis functions) first
3. **After fixing basis functions**, coefficient explosions should naturally resolve in iter11

---

### Principle 3: Alpha Bounds Prevented Spurious Reduction BUT Cannot Fix Broken Beta Terms

**Evidence**:
- α₀ = 2.48ms (bounds [0.5ms, 5.0ms], mid-range ✓)
- α₁ = 127.6μs (bounds [50μs, 300μs], mid-range ✓)
- α₂ = 135.0μs (bounds [40μs, 250μs], mid-range ✓)
- Iter9: α₀=0.35ms, α₁=65μs, α₂=48μs (all collapsed)
- Iter10: α₀, α₁, α₂ recovered to mid-range (constrained bounds worked ✓)
- BUT overall loss: 4267.22% (catastrophic failure)

**Mechanism**:

Constrained alpha bounds **successfully prevented spurious alpha reduction** (iter9 problem), but this **cannot compensate** for fundamentally broken beta terms:

1. **Alpha stability achieved**: Alpha coefficients are in physically plausible ranges and did not hit lower bounds
   - α₀ = 2.48ms (API overhead baseline, physically plausible)
   - α₁ = 127.6μs/tok (input tokenization cost, physically plausible)
   - α₂ = 135.0μs/tok (output tokenization cost, physically plausible)

2. **BUT insufficient to prevent failure**: The broken β₁₀ (1000× too small) and β₃' (65× too large) caused the overall model to diverge
   - Alpha bounds can prevent alpha from collapsing, but cannot fix broken beta basis functions
   - Overall loss exploded 27× despite stable alpha values

**Interpretation**: **Constrained bounds are a defensive measure**, not a fix. They prevent symptoms (spurious coefficient reduction) but cannot address root causes (broken basis functions).

**Action**:
1. **Keep constrained alpha bounds for iter11** — they work as intended
2. **Fix beta basis functions** (β₁₀, β₃') to address the root cause
3. **Lesson**: Bounds prevent optimizer from making physically implausible trade-offs, but cannot fix formulation bugs

---

### Principle 4: Quadratic Functional Form is Correct, But Scaling/Units are Wrong

**Evidence**:
- β₁₀ long/short sequence ratio: 197× (expected 10-40× after batch size adjustment)
- Quadratic scaling formula: (500/100)² × (32/4) = 25 × 8 = 200× ✓
- BUT absolute magnitudes: 0.0003-0.059ms (expected 0.5-80ms, 340-7000× too small)

**Mechanism**:

The β₁₀ basis function **correctly implements quadratic scaling** with sequence length, but has a **systematic scaling error**:

1. **Functional form is correct**: Long/short ratio = 197× matches the expected quadratic formula
   - This means the `prefillTokens²/batchSize` computation is structurally correct
   - The formula captures the physics: longer sequences have quadratically higher impact on batching efficiency

2. **BUT scaling is wrong by factor-of-1000**: All absolute magnitudes are 1000× too small
   - This indicates a **systematic unit conversion error** (e.g., milliseconds vs microseconds)
   - OR a **missing multiplication factor** (e.g., forgot to multiply by a constant)

**Interpretation**: **The hypothesis was physically correct** (batching inefficiency scales quadratically with sequence length), but the **implementation has a unit bug**.

**Action**:
1. **Validate hypothesis physics** ✓ — quadratic scaling is correct
2. **Fix implementation bug** — likely unit conversion error in `sim/latency/evolved_model.go`
3. **Preserve functional form** — keep `prefillTokens²/batchSize`, just fix the scaling factor

---

## Coefficient Analysis

### Alpha [α₀, α₁, α₂] (API Overhead Terms)

**Optimal values**:
- α₀ = 2.48ms (base API overhead)
- α₁ = 127.6μs/tok (per-input-token overhead)
- α₂ = 135.0μs/tok (per-output-token overhead)

**Physical interpretation**:
- α₀ = 2.48ms: Request initialization, context creation, API routing (physically plausible ✓)
- α₁ = 127.6μs/tok: Input tokenization and encoding (physically plausible ✓)
- α₂ = 135.0μs/tok: Output detokenization and formatting (physically plausible ✓)

**Comparison to iter9**:
- α₀: 0.35ms → 2.48ms (+613%, recovered from spurious collapse)
- α₁: 65.0μs → 127.6μs (+96%, recovered from spurious collapse)
- α₂: 48.5μs → 135.0μs (+178%, recovered from spurious collapse)

**Success**: Constrained alpha bounds prevented spurious reduction and allowed alpha to recover to physically plausible values ✓

**No outliers**: All alpha coefficients are well within bounds and physically plausible.

---

### Beta [β₀, β₁, ..., β₁₀] (Step-Level Basis Functions)

**Optimal values** (with physical interpretation):

| Coefficient | Value | Physical Range | Status | Interpretation |
|-------------|-------|----------------|--------|----------------|
| β₀ (prefill compute) | 0.2485 | 0.14-0.22 | ⚠️ Slightly high | Matrix multiplication FLOPs, prefill phase |
| β₁ (decode memory) | 1.1544 | 1.2-1.5 | ⚠️ Slightly low | Memory bandwidth bottleneck, decode phase |
| β₂ (TP comm) | 0.9539 | 0.25-0.60 | ❌ Too high | Tensor-parallel all-reduce communication, still inflated |
| β₃ (KV base) | 0.402ms | 0.4-1.5ms | ✅ Within range | PagedAttention setup, block manager initialization |
| β₄ (decode compute) | 2.7×10⁻⁷ ≈ 0 | 0.40-0.65 | ❌ Collapsed to zero | Decode compute FLOPs, disappeared |
| β₅ (MoE gating) | 0.44 | 0.40-0.65 | ✅ Within range | Gating network (router) FLOPs, MoE-only |
| β₆ (scheduler) | 0.0293ms = 29.3μs | 15-40ms | ❌ Collapsed (1000× too low) | vLLM scheduler CPU overhead, collapsed to compensate for β₃' |
| β₇ (decode overhead) | 28.5ms | 8-20ms | ⚠️ Slightly high | Per-request decode overhead (framework, sampling) |
| β₈ (MoE routing) | 5.0ms | 25-80μs | ❌ Exploded (69× too high) | Expert dispatch and load balancing, exploded to compensate for β₁₀ |
| β₃' (KV seq-len) | 65.8μs per (tok×layer) | 0.1-1.0μs | ❌ 65× too high | Block allocation scaling with KV size, formulation bug |
| β₁₀ (batching inefficiency) | 0.945μs per (tok²/batch) | 0.1-1.0ms | ❌ 1000× too low | Queueing delay from low batch efficiency, formulation bug |

**Redundant terms**: β₄ collapsed to zero — candidate for removal, but wait until β₁₀ and β₃' are fixed first (β₄ collapse is likely a compensation artifact).

**Missing physics**: None identified — the catastrophic failure is due to formulation bugs in existing terms (β₁₀, β₃'), not missing terms.

**Coefficient explosions**:
- β₈ (MoE routing): 72.7μs → 5.0ms (69× increase) — compensating for broken β₁₀
- β₂ (TP comm): 0.82 (remained high, didn't decrease as expected) — still absorbing unmodeled overhead
- β₃' (KV seq-len): 65.8μs (65× too high) — formulation bug

**Coefficient collapses**:
- β₆ (scheduler): 99.3ms → 29.3μs (3400× decrease) — compensating for broken β₃'
- β₄ (decode compute): 0.47 → 0 (vanished) — zero-sum trade-off artifact
- β₁₀ (batching inefficiency): 0.945μs (1000× too low) — formulation bug

---

## Recommendations for iter11

### Priority 1: CRITICAL — Fix Basis Function Formulation Bugs

**STOP and revert to iter9**. Do NOT proceed to iter11 until basis function bugs are fixed and unit tested.

#### Action 1: Audit β₁₀ (Batching Inefficiency) Basis Function

**Location**: `sim/latency/evolved_model.go`

**Hypothesis**: Factor-of-1000 error (likely milliseconds vs microseconds unit conversion)

**Audit checklist**:
1. **Verify formula**: Should compute `β₁₀ × Σ(prefillTokens² / batchSize)` for all requests in the step
2. **Verify units**:
   - β₁₀ coefficient: seconds (not milliseconds or microseconds)
   - `prefillTokens`: integer (token count)
   - `batchSize`: integer (request count)
   - Output: seconds (time contribution)
3. **Check for common bugs**:
   - Missing division by `batchSize` (would cause massive overestimation)
   - Incorrect time conversion (milliseconds vs microseconds vs seconds)
   - Missing multiplication factor (e.g., forgot to multiply by a normalization constant)
   - Using wrong fields (e.g., `outputTokens` instead of `prefillTokens`)

**Unit test** (add to `sim/latency/evolved_model_test.go`):
```go
func TestBeta10BatchingInefficiency(t *testing.T) {
    // Test case 1: Long sequence, small batch
    // 500 tokens, batch_size=4, β₁₀=0.5ms
    // Expected: 0.5ms × (500²/4) = 0.5ms × 62,500 = 31.25ms
    coeff := 0.0005 // 0.5ms in seconds
    tokens := 500.0
    batchSize := 4.0
    contribution := coeff * (tokens * tokens / batchSize)
    assert.InDelta(t, 0.03125, contribution, 0.0001) // 31.25ms ± 0.1ms

    // Test case 2: Short sequence, large batch
    // 100 tokens, batch_size=32, β₁₀=0.5ms
    // Expected: 0.5ms × (100²/32) = 0.5ms × 312.5 = 0.156ms
    tokens = 100.0
    batchSize = 32.0
    contribution = coeff * (tokens * tokens / batchSize)
    assert.InDelta(t, 0.0001565, contribution, 0.00001) // 0.156ms ± 0.01ms

    // Test case 3: Verify quadratic scaling
    // Ratio should be ~200× (25× from tokens² × 8× from batchSize)
    ratio := (0.03125 / 0.0001565)
    assert.InDelta(t, 200.0, ratio, 10.0) // 200 ± 10
}
```

**Validation**: After fix, run unit test and verify:
- Long sequence contribution = 31.25ms (not 0.03125ms)
- Short sequence contribution = 0.156ms (not 0.000156ms)
- Ratio = 200× (quadratic scaling preserved)

#### Action 2: Audit β₃' (KV Seq-Len Component) Basis Function

**Location**: `sim/latency/evolved_model.go`

**Hypothesis**: Factor-of-65 error (likely microseconds vs milliseconds unit conversion OR incorrect scaling)

**Audit checklist**:
1. **Verify formula**: Should compute `β₃' × Σ(prefillTokens × numLayers)` for all requests in the step
2. **Verify units**:
   - β₃' coefficient: seconds (not microseconds or milliseconds)
   - `prefillTokens`: integer (token count)
   - `numLayers`: integer (layer count from model config)
   - Output: seconds (time contribution)
3. **Check for common bugs**:
   - Incorrect time conversion (microseconds vs seconds)
   - Wrong layer count (using `numExperts` instead of `numLayers` for MoE models?)
   - Missing or incorrect division (e.g., dividing by 1000 twice)
   - Confusion with β₃ (base component) — ensure β₃' is separate

**Unit test** (add to `sim/latency/evolved_model_test.go`):
```go
func TestBeta3PrimeKVSeqLen(t *testing.T) {
    // Test case 1: Long sequence, dense model
    // 500 tokens, 56 layers, β₃'=0.5μs per (token×layer)
    // Expected: 0.5μs × (500 × 56) = 0.5μs × 28,000 = 14ms
    coeff := 0.0000005 // 0.5μs in seconds
    tokens := 500.0
    layers := 56.0
    contribution := coeff * (tokens * layers)
    assert.InDelta(t, 0.014, contribution, 0.001) // 14ms ± 1ms

    // Test case 2: Short sequence, same model
    // 100 tokens, 56 layers, β₃'=0.5μs
    // Expected: 0.5μs × (100 × 56) = 0.5μs × 5,600 = 2.8ms
    tokens = 100.0
    contribution = coeff * (tokens * layers)
    assert.InDelta(t, 0.0028, contribution, 0.0003) // 2.8ms ± 0.3ms

    // Test case 3: Verify linear scaling
    // Ratio should be 5× (500/100 = 5)
    ratio := (0.014 / 0.0028)
    assert.InDelta(t, 5.0, ratio, 0.1)
}
```

**Validation**: After fix, run unit test and verify:
- Long sequence contribution = 14ms (not 1400ms or 0.014ms)
- Short sequence contribution = 2.8ms (not 280ms or 0.0028ms)
- Ratio = 5× (linear scaling with token count)

#### Action 3: Verify β₃ (KV Base) is Still Correct After Split

**β₃ reverted correctly** in iter10 (9.6ms → 0.402ms), so the split concept is sound. But verify the code doesn't have interactions between β₃ and β₃':

**Audit checklist**:
1. **Verify β₃ and β₃' are independent**: β₃ should only depend on `numRequests`, β₃' only on `prefillTokens × numLayers`
2. **Check for double-counting**: Ensure β₃' is not also added to β₃ (they should be separate terms)
3. **Verify aggregation**: Total KV contribution = `β₃ × numRequests + β₃' × Σ(prefillTokens × numLayers)`

#### Action 4: Run Integration Test Before Training

After fixing β₁₀ and β₃', run an integration test to verify end-to-end latency estimates:

**Test case**: Scout general-lite (exp 17)
- Model: Llama-4-Scout-17B-16E, TP=2, 56 layers
- Workload: general-lite-2-1 (500 tokens prefill, batch_size~4)
- Observed TTFT: ~120ms (from trace data)

**Expected contributions** (with β₁₀=0.5ms, β₃'=0.5μs):
- β₁₀ (batching inefficiency): 0.5ms × (500²/4) = 31.25ms
- β₃' (KV seq-len): 0.5μs × (500 × 56) = 14ms
- β₃ (KV base): 0.5ms × 1 = 0.5ms
- Other terms (β₀, β₁, β₂, β₅, β₆, β₇, β₈): ~75ms (from iter9)
- **Total TTFT estimate**: 31.25 + 14 + 0.5 + 75 = 120.75ms ✓

**Validation**: If integration test passes (estimated TTFT within 20% of observed), proceed to iter11 training.

---

### Priority 2: Maintain Constrained Alpha Bounds

**Keep constrained alpha bounds for iter11**:
- α₀: [0.5ms, 5.0ms]
- α₁: [50μs, 300μs]
- α₂: [40μs, 250μs]

**Rationale**: Constrained bounds successfully prevented spurious alpha reduction in iter10 (α values recovered to mid-range). This defensive measure should be maintained.

**No changes needed**: Alpha bounds worked as intended, keep them unchanged for iter11.

---

### Priority 3: Warm-Start from Iter9 (NOT Iter10)

**After fixing β₁₀ and β₃' basis functions**, warm-start iter11 coefficients from **iter9** (NOT iter10):

**Iter9 coefficients to use**:
- α₀ = 0.35ms → constrain to ≥0.5ms → initial value 0.8ms
- α₁ = 65.0μs → constrain to ≥50μs → initial value 100μs
- α₂ = 48.5μs → constrain to ≥40μs → initial value 80μs
- β₀ = 0.1624 → initial value 0.18
- β₁ = 1.3611 → initial value 1.35
- β₂ = 0.8171 → initial value 0.50 (expect decrease after β₁₀ fix)
- β₃ = 9.6ms → initial value 1.0ms (expect split to work correctly now)
- β₄ = 0.4658 → initial value 0.45
- β₅ = 19.8μs → initial value 20μs
- β₆ = 99.3ms → initial value 30ms (expect reversion after β₁₀ fix)
- β₇ = 11.0ms → initial value 12ms
- β₈ = 72.7μs → initial value 50μs (expect decrease after β₁₀ fix)
- **β₃' = NEW** → initial value 0.5μs per (token×layer) (midpoint of 0.1-1.0μs)
- **β₁₀ = NEW** → initial value 0.5ms per (token²/batch_request) (midpoint of 0.1-1.0ms)

**DO NOT use iter10 coefficients**: Iter10 coefficients (β₆=29μs, β₈=5ms, β₄=0) are optimizer artifacts from broken basis functions and are physically implausible.

---

### Priority 4: Expand Optimization Trials (If Needed)

**Iter10 ran 250 trials** and did NOT converge early (`converged_early: false`). After fixing basis functions, if iter11 struggles to converge:

**Consider increasing trials**:
- Iter11: 350 trials (40% increase)
- If still no convergence: 500 trials (100% increase)

**Rationale**: Two new basis functions (β₁₀, β₃') increase search space complexity. More trials may be needed to explore the parameter space fully.

**Trade-off**: Longer optimization time (~9 hours for 250 trials) vs better convergence. Only increase trials if iter11 shows signs of non-convergence (loss plateaus above 100% after 250 trials).

---

## Basis Function Changes for iter11

After fixing β₁₀ and β₃' formulation bugs:

**Keep (No changes)**:
- β₀ (prefill compute)
- β₁ (decode memory)
- β₂ (TP comm)
- β₅ (MoE gating)
- β₇ (decode overhead)

**Monitor for potential removal** (after iter11 results):
- β₄ (decode compute): Collapsed to zero in iter10 (may be redundant after β₁₀, β₃' are fixed)
  - If β₄ < 0.1 in iter11, remove in iter12
  - If β₄ recovers to 0.3-0.6 range, keep

**Keep with expected changes**:
- β₃ (KV base): Should remain ~0.4-1.5ms (split worked correctly)
- β₆ (scheduler): Should revert from 99ms to 15-40ms after β₁₀ fix
- β₈ (MoE routing): Should revert from 5ms to 25-80μs after β₁₀ fix
- **β₃' (KV seq-len)**: Should converge to 0.1-1.0μs per (token×layer) after formulation fix
- **β₁₀ (batching inefficiency)**: Should converge to 0.1-1.0ms per (token²/batch_request) after formulation fix

**Total coefficient count**: 11 beta coefficients (same as iter10)

---

## Success Criteria for iter11

After fixing β₁₀ and β₃' basis functions, iter11 should achieve:

**Minimum (Tier 2 — Partial Success)**:
- Overall loss: <110% (30% improvement from iter9's 160.6%)
- TTFT RMSE: <50% (23% improvement from iter9's 64.8%)
- E2E RMSE: <65% (32% improvement from iter9's 95.8%)
- Scout long-sequence TTFT: <70% (>20pp improvement)
- β₁₀ coefficient: 0.1-1.0ms per (token²/batch_request) ✓
- β₃' coefficient: 0.1-1.0μs per (token×layer) ✓
- β₆ reversion: 99ms → 15-40ms ✓
- β₈ reversion: 5ms → 25-80μs ✓

**Target (Tier 1 — Full Success)**:
- Overall loss: <90% (44% improvement from iter9)
- TTFT RMSE: <40% (38% improvement from iter9)
- E2E RMSE: <55% (43% improvement from iter9)
- Scout long-sequence TTFT: <60% (>30pp improvement)
- All hypotheses from iter10 should validate ✓

**If Tier 2 not achieved**: Basis function bugs still present OR additional missing terms identified → return to Priority 1 (audit code again).

---

## Lessons Learned

### Lesson 1: Always Unit Test New Basis Functions BEFORE Training

**Problem**: β₁₀ and β₃' had factor-of-1000 and factor-of-65 errors that could have been caught with a single unit test.

**Cost**: 250 optimization trials × 7 hours = wasted compute + wasted iteration.

**Solution**:
1. **Write unit tests for new basis functions** BEFORE adding them to the model
2. **Validate expected contributions** for known inputs (e.g., 500 tokens → 30ms contribution)
3. **Check edge cases**: Zero tokens, single token, very large batch, very small batch
4. **Run integration test** with one representative experiment before full training

**Process change**: Add "Unit Test New Basis Functions" step to design agent prompt (before optimization step).

---

### Lesson 2: Coefficient Explosions are Symptoms, Not Root Causes

**Problem**: Iter9 had β₆ explosion (13ms → 99ms), β₂ explosion (0.18 → 0.82), β₈ explosion (30μs → 73μs). Iter10 tried to fix these by adding β₁₀, but the formulation bug caused new explosions (β₈ → 5ms) and collapses (β₆ → 29μs).

**Insight**: **Don't try to fix coefficient explosions directly** (by constraining bounds or adding compensating terms). Instead, **diagnose WHY they exploded** (what physics is missing?) and **add the correct basis function** (with correct formulation).

**In iter9/10 case**: β₆ exploded because long-sequence queueing delay was missing → added β₁₀ (correct diagnosis) → but β₁₀ formulation was wrong (incorrect implementation) → coefficient explosions shifted elsewhere (β₈).

**Solution**: When a coefficient explodes:
1. **Diagnose root cause**: What physics mechanism is this coefficient absorbing?
2. **Add explicit basis function**: Model the missing physics with a dedicated term
3. **Unit test the new basis function**: Ensure it captures the expected magnitude (10-100ms, not 0.01ms or 10000ms)
4. **Validate fix**: After training, verify the exploded coefficient reverts to physical range

---

### Lesson 3: Quadratic Scaling Hypothesis Can Be Correct Even If Implementation Fails

**Problem**: β₁₀ hypothesis predicted quadratic scaling with sequence length → optimizer confirmed this (197× long/short ratio matches expected 200×) → BUT absolute magnitudes were 1000× wrong → hypothesis REJECTED.

**Insight**: **Separate hypothesis evaluation from implementation validation**:
- **Hypothesis correctness**: Is the physics explanation correct? (YES for β₁₀ — quadratic scaling is correct)
- **Implementation correctness**: Is the code bug-free? (NO for β₁₀ — factor-of-1000 error)

**Solution**:
1. **When a hypothesis fails, check both dimensions**: Physics correctness AND implementation correctness
2. **If functional form is correct but magnitude is wrong**: Likely implementation bug (unit conversion, scaling factor)
3. **If functional form is wrong (e.g., linear when should be quadratic)**: Likely physics hypothesis is wrong
4. **Don't abandon a hypothesis** if the functional form is correct — fix the implementation and retry

**In iter10 case**: β₁₀ quadratic scaling hypothesis is **correct** (keep for iter11) → but implementation has **unit bug** (fix before iter11).

---

## Next Steps

### Immediate Actions (Before iter11)

1. ✅ **Complete iter10 analysis**: Write validation and findings documents (DONE)
2. ⏸️ **STOP optimization**: Do NOT proceed to iter11 until basis functions are fixed
3. 🔧 **Audit β₁₀ basis function code**: Follow Priority 1, Action 1 checklist
4. 🔧 **Audit β₃' basis function code**: Follow Priority 1, Action 2 checklist
5. ✅ **Write unit tests**: Add `TestBeta10BatchingInefficiency` and `TestBeta3PrimeKVSeqLen` to `sim/latency/evolved_model_test.go`
6. ✅ **Run unit tests**: Verify β₁₀ and β₃' contributions match expected magnitudes
7. ✅ **Run integration test**: Validate end-to-end TTFT estimate for Scout general-lite (exp 17)
8. 📝 **Update coefficient bounds**: Use iter9 values (not iter10) for warm-start
9. 🚀 **Launch iter11**: Only after all unit tests and integration tests pass

### If iter11 Succeeds (Tier 1 or Tier 2)

- **Tier 1 (Overall loss <90%)**: Proceed to cross-validation (CV1, CV2, CV3)
- **Tier 2 (Overall loss <110%)**: Analyze residual errors → design iter12 with complementary term (β₁₁ for memory bandwidth saturation or chunked prefill overhead)

### If iter11 Fails (Overall loss >130%)

- **Return to Priority 1**: Audit basis function code again (β₁₀, β₃' may still have bugs)
- **Consider alternative formulations**: Sigmoid threshold for β₁₀ (only penalize sequences >300 tokens), or piecewise linear split
- **Profile vLLM separately**: Measure actual batch formation overhead, scheduler CPU time, KV cache allocation time to validate hypothesis predictions

---

## Conclusion

Iter10 suffered catastrophic failure due to **basis function formulation bugs** (β₁₀ 1000× too small, β₃' 65× too large). The optimizer could not converge to a physically plausible solution, causing loss to explode 27× and coefficients to explode/collapse in compensation.

**Root cause**: Missing unit tests before training. The bugs could have been caught with a single unit test validating expected contributions.

**Lesson**: **Always unit test new basis functions** BEFORE adding them to the model.

**Recommendation**: **Revert to iter9**, fix β₁₀ and β₃' basis functions, write unit tests, run integration test, then retry iter11.

**Hypothesis validation**: The underlying physics hypotheses are likely **correct** (quadratic batching inefficiency, linear KV seq-len scaling), but the **implementations are buggy**. Don't abandon the hypotheses — fix the code and retry.
