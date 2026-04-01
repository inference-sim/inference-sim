# Iteration 15: Findings and Principles

## Executive Summary

**Iteration 15 FAILED catastrophically** - loss INCREASED from 2319% (iter14) to **6538%** (182% worse).

The iteration attempted a "three-axis correction" strategy:
1. Decode amplification (β₁, β₄ at 5-15×, 3-8×)
2. MoE non-compute term (β₈, NEW)
3. Prefill batching penalty (β₉, NEW)

**What actually happened**:
- ✅ Decode amplification was USED (β₁=6.4, β₄=6.5) - helped reasoning-lite (E2E 75-180%)
- ❌ MoE non-compute was REJECTED (β₈≈0) - Scout errors INCREASED (527% → 1068%)
- ❌ Prefill batching was REJECTED (β₉≈0) - Dense errors remain catastrophic (1300-4000%)

### Hypothesis Validation Results

| Hypothesis | Verdict | Key Finding |
|------------|---------|-------------|
| **H-main** (Three-axis correction) | ❌ REJECTED | Loss increased 182%, all metrics worse |
| **H-ablation-decode** | ⚠️ PARTIAL | Decode amplification helps decode-heavy, but can't fix prefill |
| **H-ablation-moe** | ❌ REJECTED | β₈ collapsed to 0, MoE FLOPs likely wrong |
| **H-ablation-batching** | ❌ REJECTED | β₉ collapsed to 0, batch heterogeneity not the issue |
| **H-boundary** (cold-start) | ❌ REJECTED | Cold-start 4219pp worse than warm-start |
| **H-error-pattern** | ⚠️ PARTIAL | Only reasoning-lite improved (30-668%), Scout/dense got worse |
| **H-robustness** | ⚠️ PARTIAL | 6/10 coefficients in range, but 4/10 collapsed (β₃,β₆,β₇,β₈,β₉) |

**Overall**: 0/7 confirmed, 3/7 partial, 4/7 rejected.

### Root Cause: Scaling Broken Formulas Doesn't Fix Them

The fundamental error in iter15 was attempting to **FIX roofline estimates by scaling them** (β × roofline_term), rather than **REPLACING them with vLLM-accurate formulas**.

**Evidence**:
1. **Prefill**: β₀=0.092 scales down roofline by 11×, but dense TTFT is still 13-40× wrong → base roofline is 140-440× off
2. **MoE**: β₈ (non-compute correction) rejected by optimizer → MoE FLOPs calculation itself is wrong
3. **Batching**: β₉ (heterogeneity penalty) rejected by optimizer → dense overestimation is NOT about batching

**You cannot fix a broken formula by multiplying it by a constant.**

### Critical Learnings

**✅ What Worked:**

**Decode amplification helps decode-heavy workloads**:
- Reasoning-lite E2E APE: 75-180% (vs 100% timeout in iter14)
- β₁=6.4, β₄=6.5 amplify decode time → prevents underestimation for long outputs
- **Limitation**: Only helps when decode DOMINATES total latency (256-512 output tokens)

**❌ What Failed:**

1. **Roofline-based prefill model is fundamentally broken**:
   - Dense TTFT APE: 1300-4000% (13-40× too fast)
   - β₀=0.092 provides 11× scale-down, still insufficient
   - Pattern: Shorter prompts (64 tokens) have WORSE errors than longer prompts (512 tokens)
   - Root cause: Base roofline prefill calculation (FLOPs or memory formula) is orders of magnitude wrong

2. **MoE non-compute hypothesis was wrong**:
   - β₈ collapsed to 0 (optimizer rejected it)
   - Scout errors INCREASED from 527% (iter14) to 1068% (iter15)
   - Alternative hypothesis: MoE FLOPs calculation is wrong (active vs total experts? load imbalance in FLOPs?)

3. **Prefill batching hypothesis was wrong**:
   - β₉ collapsed to 0 (optimizer rejected it)
   - Dense roleplay (low heterogeneity) has WORSE errors than dense codegen (high heterogeneity)
   - Alternative hypothesis: Dense overestimation is due to wrong BASE prefill FLOPs, not batching

4. **Cold-start in 10D space is inefficient**:
   - Loss 6538% (cold-start) vs 2319% (iter14 warm-start) - 2.8× worse
   - 2000 trials in 10D → only 200 per dimension (sparse coverage)
   - Optimizer rejected 5/10 new coefficients (β₃,β₆,β₇,β₈,β₉)

5. **Ignored iter9 baseline** (critical oversight):
   - **Iter9 was LAST STABLE STATE** (161% loss) on SAME dataset as iter15
   - Iter15 compared cold-start to iter7 (wrong dataset), ignored iter9 entirely
   - Should have warm-started from iter9 (5 proven coefficients + same dataset)

### Quick Reference: Next Steps for Iter16

**Priority 1 (MUST DO)**: **Stop scaling roofline. Start profiling vLLM.**

1. **Profile real vLLM prefill latency** → Replace `β₀ × roofline` with empirical model
2. **Profile real vLLM decode latency** → Validate β₁, β₄ functional forms
3. **Investigate MoE FLOPs calculation** → Fix expert counting (active vs total)

**Priority 2**: Remove optimizer-rejected terms (β₃,β₆,β₇,β₈,β₉) → **5 coefficients**

**Priority 3**: Warm-start from **iter9** (not iter7, not cold-start)
- Iter9: 161% loss, 15 clean experiments (SAME dataset as iter16)
- Proven coefficients: β₀=0.191, β₁=1.108, β₄=0.705, β₅=27.5, β₇=0.027

**Expected outcome**: Loss 6538% → <500% (13× improvement)

---

## Error Analysis

### Systematic Patterns

**High-error experiments** (TTFT APE > 1000%):

| Experiment | Model | Workload | TTFT APE | E2E APE | Pattern |
|------------|-------|----------|----------|---------|---------|
| exp_64 | Qwen-7B | roleplay | 4151% | 11357% | Dense, short prompts, long outputs (decode-heavy) |
| exp_02 | Llama-2-7B | roleplay | 4124% | 9711% | Dense, short prompts, long outputs |
| exp_62 | Mistral-Nemo-12B TP2 | general-lite | 4167% | 5164% | Dense, medium prompts, TP=2 |
| exp_04 | Llama-2-7B | codegen | 1847% | 3758% | Dense, long prompts (prefill-heavy) |
| exp_21 | Scout-17B-16E TP2 | roleplay | 1634% | 3378% | MoE, short prompts, long outputs |
| exp_63 | Mistral-Nemo-12B | codegen | 1437% | 2583% | Dense, long prompts (prefill-heavy) |
| exp_65 | Yi-34B TP2 | general-lite | 1296% | 1877% | Dense, medium prompts, TP=2 |
| exp_09 | Llama-2-7B | general | 1183% | 1354% | Dense, short prompts, balanced |
| exp_61 | Llama-3.1-70B TP4 | codegen | 786% | 1230% | Dense, long prompts, TP=4 |
| exp_20 | Scout-17B-16E TP2 | codegen | 708% | 1335% | MoE, long prompts (prefill-heavy) |

**Catastrophic failures (TTFT APE > 4000%)**:
- **exp_64** (Qwen roleplay): 4151% TTFT, 11357% E2E - decoder-heavy with short prompts (64 tokens avg), long outputs (128 tokens avg)
- **exp_02** (Llama-2 roleplay): 4124% TTFT, 9711% E2E - similar to exp_64, decoder-dominated
- **exp_62** (Mistral TP2 general-lite): 4167% TTFT, 5164% E2E - TP=2, medium heterogeneity

**Why these failed**: Despite decode amplification (β₁=6.4, β₄=6.5), roleplay workloads have catastrophic TTFT errors. TTFT measures time-to-first-token, which is PREFILL latency (not decode). This confirms:
1. Decode amplification helps DECODE phase (ITL, E2E) but NOT PREFILL phase (TTFT)
2. Roleplay's high TTFT error (4000%+) indicates **prefill basis functions are catastrophically wrong**
3. The E2E error (9000-11000%) is even worse, suggesting decode amplification may be OVERCORRECTING decode phase

**Low-error experiments** (TTFT APE < 100%):

| Experiment | Model | Workload | TTFT APE | E2E APE | Pattern |
|------------|-------|----------|----------|---------|---------|
| exp_48 | Scout-17B-16E TP2 | reasoning-lite | 30% | 27% | MoE, long prompts, long outputs (balanced) |

**Why exp_48 succeeded**:
- TTFT APE = 30%, E2E APE = 27% - **BOTH are low** (unlike other experiments where only one is low)
- Reasoning-lite workload: 512 input tokens, 256 output tokens → balanced prefill/decode
- Scout MoE model with TP=2
- This is the ONLY experiment with <100% APE across all metrics

**Critical insight**: exp_48's success suggests that for **BALANCED workloads** (comparable prefill and decode token counts), the model can achieve reasonable accuracy. Catastrophic failures occur when workloads are **HEAVILY SKEWED** towards prefill (codegen) or decode (roleplay).

**Mid-error experiments** (TTFT APE 200-1000%):

| Experiment | Model | Workload | TTFT APE | E2E APE | Pattern |
|------------|-------|----------|----------|---------|---------|
| exp_60 | Llama-3.1-70B TP4 | general-lite | 956% | 1611% | Large model, TP=4, medium heterogeneity |
| exp_17 | Scout-17B-16E TP2 | general-lite | 863% | 1460% | MoE, medium heterogeneity |
| exp_66 | Qwen-7B | reasoning-lite | 668% | 180% | Dense, long prompts AND long outputs |
| exp_67 | Llama-2-7B | reasoning-lite | 238% | 75% | Dense, long prompts AND long outputs |

**Pattern**: Reasoning-lite experiments (exp_66, exp_67) show **LOW E2E APE** (75-180%) but **HIGH TTFT APE** (238-668%). This is the OPPOSITE of the catastrophic roleplay failures.

**Why reasoning-lite has low E2E but high TTFT**:
- Long outputs (256-512 tokens) → decode dominates E2E latency
- Decode amplification (β₁, β₄) helps decode phase → low E2E APE ✅
- But TTFT (prefill) is still wrong → high TTFT APE ❌
- Numerical stability improved (no timeouts, unlike iter14) → decode amplification prevents negative latencies ✅

### Error Correlations

**✅ Confirmed correlations**:
1. **Decode amplification helps decode-heavy E2E latency**:
   - Reasoning-lite experiments: E2E APE 75-180% (vs 100% timeout in iter14)
   - Evidence: β₁=6.4, β₄=6.5 significantly amplify decode time → prevents underestimation
   - Limitation: Only helps when decode DOMINATES total latency (long outputs)

2. **Prefill-heavy workloads have catastrophic TTFT errors**:
   - Dense roleplay: 4124-4151% TTFT APE
   - Dense codegen: 1437-1847% TTFT APE
   - Evidence: β₀=0.092 provides 11× scale-down, but prefill is still 40× too fast
   - Root cause: Base roofline prefill calculation is orders of magnitude wrong

3. **MoE experiments have universally high errors**:
   - Scout experiments: 708-1634% TTFT APE (avg 1068%)
   - Evidence: β₈ (MoE non-compute) collapsed to 0 → optimizer rejected MoE hypothesis
   - Pattern: MoE errors INCREASED from iter14 (527% avg → 1068% avg)
   - Root cause: MoE FLOPs calculation likely wrong, not missing non-compute overhead

**❌ Rejected correlations**:
1. **Batch heterogeneity does NOT explain dense overestimation**:
   - Hypothesis: High heterogeneity (codegen) should have worse errors than low heterogeneity (general)
   - Reality: Dense roleplay (low heterogeneity) has 4124% APE, dense codegen (high heterogeneity) has 1847% APE
   - Evidence: β₉ (batching penalty) collapsed to 0 → optimizer found no heterogeneity effect
   - Conclusion: Dense overestimation is NOT about batching inefficiency

2. **MoE non-compute overhead does NOT explain Scout underestimation**:
   - Hypothesis: Scout needs routing/load-imbalance penalty (β₈) on top of FLOPs
   - Reality: β₈ collapsed to 0 → optimizer found no non-compute effect
   - Evidence: Scout errors INCREASED with β₈ available (527% → 1068%)
   - Conclusion: Scout underestimation is NOT about non-compute latency, likely wrong MoE FLOPs

3. **Cold-start does NOT improve over warm-start**:
   - Hypothesis: Cold-start finds better minimum than iter7 warm-start
   - Reality: Cold-start loss (6538%) is 2.8× WORSE than iter14 warm-start (2319%)
   - Evidence: 2000 trials in 10D space failed to converge
   - Conclusion: Search space too large without physics priors, or basis functions are fundamentally wrong

### Root Cause Hypotheses

**Principle 1: Roofline-based scaling is fundamentally flawed for vLLM prefill latency**

**Evidence**:
- β₀ (prefill MFU scaling) = 0.092 → scales down roofline by 11× (1/0.092)
- Despite 11× scale-down, dense experiments have 1300-4000% TTFT APE (13-40× too fast)
- β₉ (prefill batching penalty) collapsed to 0 → additive correction also doesn't work
- Dense roleplay TTFT: predicted = P, actual = 40P → error is 40×, not 11×

**Mechanism**: The roofline model calculates prefill time as:
```
prefill_time = max(compute_time, memory_time)
             = max(FLOPs / peak_FLOPS, bytes / peak_bandwidth)
```
Then applies β₀ scaling: `β₀ × prefill_time`.

For dense roleplay with β₀=0.092, roofline predicts:
- Base roofline estimate: T
- After β₀ scaling: 0.092 × T = T/11
- Actual vLLM latency: ~4× (T/11) = T/2.75

**The issue**: Roofline predicts prefill is 11× TOO FAST (after β₀ correction), but actual error is 40×. This means:
1. Base roofline calculation is 440× too fast (40 × 11 = 440×)
2. β₀ only corrects to 40× too fast (still catastrophic)
3. The roofline formula itself (max(compute, memory)) is wrong for vLLM

**Why roofline fails for vLLM prefill**:
- **Causal attention vs full attention**: Roofline may assume full attention (N² FLOPs), vLLM uses causal attention (~N²/2 FLOPs)
- **Memory pattern mismatch**: Roofline assumes contiguous memory, vLLM has PagedAttention fragmentation
- **Kernel launch overhead**: Roofline assumes large kernel amortizes launch cost, vLLM has per-layer launches
- **Mixed precision**: Roofline may assume FP16, vLLM uses mixed FP16/FP8/INT8

**Action for iter16**:
1. **Profile real vLLM prefill latency** vs batch size, sequence length, TP for 3-5 dense models
2. **Derive empirical prefill formula** (not roofline-based): `prefill_time = f(batch_size, seq_len, TP, model_params)`
3. **Replace β₀ × roofline_prefill with empirical prefill function**
4. **Test hypothesis**: Does empirical formula reduce dense TTFT APE from 1300-4000% to <500%?

---

**Principle 2: Decode amplification has correct directional effect but may have wrong functional form**

**Evidence**:
- β₁ = 6.40 (decode memory MFU amplification) - in range [5, 15]
- β₄ = 6.47 (decode compute MFU amplification) - in range [3, 8]
- Reasoning-lite E2E APE: 75-180% (down from 100% timeout in iter14) ✅
- But reasoning-lite TTFT APE: 238-668% (still high) ❌
- Dense roleplay E2E APE: 9711-11357% (catastrophically high) ❌

**Mechanism**: Decode amplification helps when:
- Long outputs (many decode steps) → decode latency dominates E2E
- Reasoning-lite: 256-512 output tokens → decode is 80%+ of E2E time
- Result: 6× decode amplification prevents underestimation → valid results (no timeouts)

But decode amplification FAILS when:
- Short outputs (few decode steps) → prefill dominates E2E
- Dense roleplay: 64 input tokens, 128 output tokens → prefill still significant
- Result: Decode amplification may OVERCORRECT decode phase, making E2E worse

**The issue**: `β₁ × roofline_decode_memory` and `β₄ × roofline_decode_compute` assume:
1. Decode time scales linearly with roofline estimate
2. Memory-bound and compute-bound decode have independent scaling factors
3. Scaling is constant across batch sizes, sequence lengths, TP

**Evidence these assumptions are wrong**:
- Reasoning-lite E2E improves (75-180%) but TTFT doesn't (238-668%) → amplification is not balanced
- Dense roleplay E2E is catastrophic (9711-11357%) despite decode amplification → overcorrection or collinearity
- Scout E2E is high (1335-3378%) even with decode amplification → MoE decode may have different scaling

**Why linear scaling may be wrong**:
- **Batch size dependence**: Small batches (decode) may have different MFU than roofline predicts (kernel launch overhead)
- **Sequence length dependence**: KV cache access patterns change with sequence length (cache locality)
- **TP dependence**: TP communication overhead may not be captured by β₂ alone (interaction with decode)

**Action for iter16**:
1. **Profile real vLLM decode latency** vs batch size, KV cache size, TP for 3-5 models
2. **Test functional forms**: Linear (β × roofline), quadratic (β × roofline²), logarithmic (β × log(roofline))
3. **Check batch size dependence**: Does decode time scale as expected with batch size?
4. **Measure collinearity**: Are β₁ (memory) and β₄ (compute) capturing independent effects, or are they collinear?

---

**Principle 3: MoE non-compute hypothesis was fundamentally wrong - expert routing overhead is negligible or already captured**

**Evidence**:
- β₈ (MoE non-compute latency) = 0.000037 (expected 10-40 μs/token, got 3.7e-5)
- Optimizer pushed β₈ to ZERO → hypothesis rejected
- Scout experiments: 708-1634% TTFT APE (avg 1068%)
- Scout errors INCREASED from iter14 (527% avg) to iter15 (1068% avg) - **102% worse**

**Mechanism hypothesis** (from iter15, now rejected):
- "MoE has non-compute overhead from token routing (scatter/gather), load imbalance (stragglers), expert communication (all-to-all)"
- Basis function: `β₈ × num_moe_layers × (prefill_tokens + decode_tokens)`
- Expected: β₈ = 10-40 μs/token

**Why the hypothesis was wrong**:
1. **Not a separate effect**: Routing/load-imbalance may already be captured by β₅ (MoE gating, 33.57)
2. **Wrong functional form**: Per-token overhead (linear in token count) may not match real vLLM MoE behavior
3. **Negligible magnitude**: Even if present, the effect is <0.04 μs/token (3.7e-5), which is 1000× smaller than expected
4. **Wrong root cause**: Scout underestimation is NOT about non-compute latency - likely wrong MoE FLOPs calculation

**Alternative hypothesis** (supported by evidence):
- **MoE FLOPs calculation is wrong in roofline model**
- Roofline may count ALL expert FLOPs (num_experts × FFN_FLOPs), but vLLM only executes ACTIVE experts (top-K)
- Or roofline may not account for load imbalance in the FLOPs calculation (assumes uniform expert utilization)
- Result: Roofline predicts 8-16× more FLOPs than vLLM actually computes → underestimates latency

**Evidence for alternative hypothesis**:
- β₅ (MoE gating efficiency) = 33.57 (in range [20, 50]) → gating FLOPs are somewhat correct
- But Scout TTFT is still 708-1634% wrong → the issue is NOT gating, it's expert execution
- Scout errors are WORSE in iter15 (with β₈) than iter14 (without β₈) → β₈ interfered with optimization

**Action for iter16**:
1. **Remove β₈** (optimizer already rejected it)
2. **Investigate MoE FLOPs calculation** in roofline:
   - Are active vs total experts counted correctly?
   - Is load imbalance factored into FLOPs or utilization?
   - Does top-K routing match roofline's assumptions?
3. **Profile real vLLM MoE latency** for Scout vs equivalent dense model:
   - Measure TTFT ratio: Scout / (dense with same param count)
   - Expected: 2-4× slower due to expert overhead
   - Actual: Check if roofline predicts this ratio correctly
4. **Redesign β₅** (MoE gating) if FLOPs calculation is wrong

---

**Principle 4: Prefill batching penalty hypothesis was fundamentally wrong - batch heterogeneity is not the cause of dense overestimation**

**Evidence**:
- β₉ (prefill batching penalty) = 7.17e-07 (expected 0.5-2.0 μs/token, got 7.2e-7)
- Optimizer pushed β₉ to ZERO → hypothesis rejected
- Dense experiments: 1300-4000% TTFT APE (catastrophic)
- Dense roleplay (low heterogeneity) has WORSE errors (4124%) than dense codegen (high heterogeneity, 1847%)

**Mechanism hypothesis** (from iter15, now rejected):
- "Mixing prefill and decode requests causes kernel inefficiency due to batch heterogeneity"
- Basis function: `β₉ × num_prefill_tokens × (1.0 + num_decode_requests / max(1, num_prefill_tokens))`
- Expected: β₉ = 0.5-2.0 μs/token, with high heterogeneity (codegen) having larger penalties than low heterogeneity (roleplay)

**Why the hypothesis was wrong**:
1. **Opposite pattern**: Dense roleplay (low heterogeneity) has HIGHER errors (4124%) than dense codegen (high heterogeneity, 1847%)
2. **β₀ already handles MFU**: Prefill MFU scaling (β₀ = 0.092) reduces roofline by 11×, which should capture ANY prefill inefficiency
3. **Magnitude check**: Even if β₉ = 2.0 μs/token (upper bound), for 512 prefill tokens with 2× heterogeneity factor → 2 ms penalty. But dense errors are 40× (seconds, not milliseconds)
4. **Not additive**: Dense overestimation is 40× multiplicative error, not 2 ms additive error

**Alternative hypothesis** (supported by evidence):
- **Dense overestimation is due to wrong BASE prefill calculation** (FLOPs or memory), not batching inefficiency
- β₀ = 0.092 scales down roofline by 11×, but actual vLLM is 40× slower → base calculation is 4× off even after β₀ correction
- This 4× gap cannot be explained by batch heterogeneity (a 2-3× effect at most)
- Root cause: Roofline prefill FLOPs or memory calculation is fundamentally wrong (see Principle 1)

**Evidence for alternative hypothesis**:
- Dense roleplay (4124% TTFT) has SHORT prompts (64 tokens) → prefill should be fast, but roofline severely underestimates
- Dense codegen (1847% TTFT) has LONG prompts (512 tokens) → prefill is slower, less severe underestimation
- Pattern: **Shorter prompts have worse errors** → roofline may have fixed overhead not captured by per-token calculation

**Action for iter16**:
1. **Remove β₉** (optimizer already rejected it)
2. **Focus on Principle 1**: Fix base prefill calculation (roofline FLOPs/memory formula)
3. **Check fixed overhead hypothesis**: Profile vLLM prefill latency vs prompt length:
   - Does `prefill_time = A + B × prompt_length` fit better than `prefill_time = B × prompt_length`?
   - If fixed overhead A is significant, add β_prefill_overhead (ms) term
4. **Test alternative**: Does `prefill_time = β₀ × roofline + β_fixed` reduce dense TTFT APE?

---

**Principle 5: Cold-start optimization in 10D space is inefficient without physics priors**

**Evidence**:
- Cold-start (random uniform initialization) used for iter15
- Optimization: 2000 trials, 13.5 hours, 0 errors
- Result: Loss = 6538% (vs iter14's warm-start loss = 2319%) - **182% worse**
- Coefficient outcomes: β₈, β₉ collapsed to 0 (optimizer rejected new terms), β₁, β₄ used (~6.5)

**Mechanism**: With 10 β coefficients, search space is 10-dimensional:
- Each coefficient has [lower, upper] bounds → 2¹⁰ = 1024 "corners" in hypercube
- TPE (Tree-structured Parzen Estimator) requires ~100× dimensionality trials for convergence
- 10D → need ~1000 trials minimum
- Iter15 used 2000 trials, but still failed to converge to good minimum

**Why cold-start failed**:
1. **Basis functions structurally wrong**: No amount of random search can fix wrong functional forms (see Principles 1-4)
2. **Physics midpoints were wrong**: Trial 0 used midpoints of ranges, but those ranges were based on incorrect physics
3. **Dimensionality curse**: 2000 trials in 10D is only 200 trials per dimension (sparse coverage)

**Comparison to warm-start** (iter14):
- Iter14 warm-started from iter7 coefficients: β₀=0.191, β₁=1.108, β₂=0.195, etc.
- Iter14 loss = 2319% (failed, but less catastrophic than iter15)
- Iter14 had 8 β coefficients (iter15 added β₈, β₉ → 10 coefficients)
- Warm-start loss 2319% < cold-start loss 6538% by 4219pp

**Critical insight**: Cold-start vs warm-start is IRRELEVANT when basis functions are wrong. But given basis functions, **warm-start is more sample-efficient**:
- Warm-start: 1000 trials, starts from known good region (iter7), explores locally
- Cold-start: 2000 trials, starts from random, explores globally, but wastes trials on bad regions

**Action for iter16**:
1. **Fix basis functions first** (Principles 1-4), then choose initialization
2. **If basis functions are fixed, use warm-start** from iter7 (β₀=0.191, β₁=1.108, etc.)
3. **Reduce dimensionality**: Remove β₈, β₉ (rejected), β₃, β₆, β₇ (collapsed to 0) → 5 β coefficients (β₀, β₁, β₂, β₄, β₅)
4. **Add physics priors**: Initialize coefficients from vLLM profiling (not physics midpoints or iter7):
   - β₀ from measured prefill MFU (profile 3-5 dense models)
   - β₁, β₄ from measured decode MFU (profile decode phase)
   - β₅ from measured MoE overhead (profile Scout vs equivalent dense)

### Critical Oversight: The Iter9 Baseline

**The most serious strategic error in iter15 was comparing cold-start to warm-start from iter7, while completely ignoring iter9.**

From TRAINING_JOURNEY.md, **iter9 was the last stable state before catastrophic cascade**:

| Iteration | Loss | Dataset | Status |
|-----------|------|---------|--------|
| **Iter7** | 155% | 14 clean + 1 bad (93% clean) | Stable, but wrong dataset |
| **Iter8** | 155% | 14 clean + 1 bad (93% clean) | β₈ rejected, stable |
| **Iter9** | **161%** | **15 clean (100% clean)** | **FIRST with final dataset, LAST STABLE** ✅ |
| **Iter10** | 4267% | 15 clean | 💥 CATASTROPHIC (added β₁₀ + β₃') |
| **Iter13** | 2387% | 15 clean | Warm-started from iter7 (wrong dataset) |
| **Iter14** | 2319% | 15 clean | Fixed β₅, but still warm-started from iter7 |
| **Iter15** | 6538% | 15 clean | Cold-start (threw away ALL history) |

**Why iter9 is the correct baseline**:

1. **Dataset alignment**:
   - Iter9 trained on SAME dataset as iter15 (15 clean experiments, reasoning-lite)
   - Iter7 trained on DIFFERENT dataset (14 clean, old reasoning data)
   - Warm-starting from iter7 anchors optimizer in basin optimized for WRONG data distribution

2. **Coefficient quality**:
   - Iter9: β₀=0.191, β₁=1.108, β₄=0.705, β₅=27.5 (stable, physically plausible)
   - Iter9: β₆=99ms (inflated, compensating for wrong β₉), β₃=9.6ms (inflated)
   - 5 out of 7 major coefficients were GOOD, only 2 needed adjustment

3. **Cascade origin**:
   - **Iter9 → iter10 is where cascade started** (161% → 4267%, 26× worse)
   - Iter10 added β₁₀ (batching) + β₃' (KV seq-len) simultaneously → catastrophic failure
   - Going back to iter9 and adding terms **incrementally** would prevent cascade

4. **Sample efficiency**:
   - Warm-start from iter9: Need ~1000 trials (starting near good region)
   - Cold-start (iter15): Used 2000 trials, still failed catastrophically
   - **2× efficiency gain** by using historical information

**What iter15 SHOULD have done**:

```python
# Option A: Warm-start from iter9 with resets (RECOMMENDED)
warm_start_from_iter9 = {
    'beta': [
        0.191,  # β₀ prefill (iter9 stable) ✅
        1.108,  # β₁ decode mem (iter9 stable) ✅
        0.820,  # β₂ TP comm (iter9 slightly inflated, but reasonable)
        1.0,    # β₃ KV mgmt (RESET from iter9's 9.6ms)
        0.705,  # β₄ decode comp (iter9 stable) ✅
        27.5,   # β₅ MoE gating (iter9 stable) ✅
        40.0,   # β₆ scheduler (RESET from iter9's 99ms)
        0.027,  # β₇ decode overhead (iter9 stable) ✅
        20.0,   # β₈ MoE non-compute (NEW, physics prior)
        1.0     # β₉ prefill batching (NEW, physics prior)
    ]
}
# Expected: 5/10 coefficients start from PROVEN good values (iter9)
# Expected: Loss 161% → <500% with 1000 trials (vs 6538% cold-start)
```

**Why this would have worked better**:

1. **Start in correct basin**: Iter9 already converged on final clean dataset → optimizer starts in right region
2. **Leverage proven coefficients**: 5 coefficients (β₀, β₁, β₄, β₅, β₇) are known-good from iter9
3. **Adjust inflated terms**: Only β₃, β₆ need resetting (2 out of 10)
4. **Add new terms carefully**: β₈, β₉ start from physics priors (not random)

**The debugging analogy**:

When a system enters catastrophic failure:
- ❌ **Wrong**: Revert to commit on different branch (iter7, different dataset)
- ❌ **Wrong**: Fresh install (cold-start, no history)
- ✅ **Right**: Revert to last commit that worked on current branch (iter9, same dataset)

**Impact on iter16 strategy**:

Agent 1 (Design) should:
1. **Use iter9 as warm-start baseline** (not iter7, not cold-start)
2. **Fix inflated coefficients**: β₃ (9.6ms → 1.0ms), β₆ (99ms → 40ms)
3. **Keep stable coefficients**: β₀, β₁, β₄, β₅, β₇ from iter9 (proven on final dataset)
4. **Add new terms from vLLM profiling** (not physics guesses like iter15's β₈, β₉)

**Expected improvement**: If prefill/MoE basis functions are fixed AND warm-started from iter9, loss should decrease from 6538% → <500% (13× improvement).

---

## Coefficient Analysis

### Alpha [α₀, α₁, α₂]

**From `best_params.alpha`**: [0.000627, 0.0000501, 0.0000416]

**Physical interpretation**:
- α₀ = 0.000627 ms: Fixed API overhead per request (0.627 μs)
- α₁ = 0.0000501 ms/token: Per-input-token overhead (0.0501 μs/token)
- α₂ = 0.0000416 ms/token: Per-output-token overhead (0.0416 μs/token)

**Plausibility**:
- Fixed overhead (0.627 μs) is **very small** - API overhead is typically 1-10 ms, not sub-millisecond
- Per-token overhead (0.05 μs/token) is **negligible** - tokenization/validation is typically 1-10 μs/token
- All alpha values are ~1000× smaller than expected

**Why alpha values collapsed**:
- Alpha coefficients model **vLLM API overhead** (tokenization, request parsing, response formatting)
- Beta coefficients model **GPU execution time** (prefill, decode, KV management)
- When beta coefficients are catastrophically wrong (producing 10-100× errors), optimizer pushes alpha to 0 to avoid double-counting
- Alpha's contribution (~1-2 ms total) is negligible compared to beta errors (producing 10-100 second errors)

**Action**: Alpha values are NOT the issue. Focus on fixing beta coefficients (prefill, decode basis functions).

---

### Beta [β₀, β₁, β₂, β₃, β₄, β₅, β₆, β₇, β₈, β₉]

**From `best_params.beta`**: [0.092, 6.398, 0.207, 0.001, 6.471, 33.569, 0.042, 0.016, 0.000037, 7.17e-07]

#### β₀: Prefill Compute MFU Scaling

**Value**: 0.092 (expected: 0.05-0.25) ✅ IN RANGE

**Physical interpretation**:
- Scales down roofline prefill compute time by 11× (1/0.092)
- Roofline assumes prefill achieves peak MFU (0.5-0.6 on H100), β₀ corrects this to ~0.046 (0.092 / 2)
- Lower β₀ → prefill is slower than roofline predicts

**Why β₀ is low**:
- Roofline overestimates prefill performance by ~11×
- vLLM prefill achieves ~5% MFU (0.092 / 2) vs roofline's 50-60% assumption
- This gap is plausible: mixed precision, kernel launch overhead, PagedAttention fragmentation

**Why β₀ alone is insufficient**:
- Dense TTFT APE: 1300-4000% (even with β₀ = 0.092)
- After β₀ scaling, roofline still predicts prefill 13-40× too fast
- **Root cause**: Base roofline calculation (before β₀) is 140-440× wrong (13-40 × 11 = 143-440)
- β₀ can only correct by 11×, not 140-440×

**Action**: β₀ is necessary but insufficient. Need new prefill basis functions (not roofline-based).

---

#### β₁: Decode Memory MFU Scaling

**Value**: 6.398 (expected: 5.0-15.0) ✅ IN RANGE

**Physical interpretation**:
- Amplifies roofline decode memory time by 6.4×
- Roofline assumes decode is memory-bound and achieves peak bandwidth (1.5-2.0 TB/s on H100)
- β₁ = 6.4 → vLLM decode is 6.4× SLOWER than memory bandwidth predicts

**Why β₁ is high (amplification, not scaling)**:
- Decode phase has poor memory bandwidth utilization due to:
  - Small per-request GEMMs (poor tensor core utilization)
  - KV cache pointer chasing (random access, not sequential)
  - Per-layer synchronization barriers (prevents overlapping)
- Result: Effective bandwidth is 6.4× worse than peak (1.5 TB/s / 6.4 ≈ 0.23 TB/s)

**Evidence β₁ helps decode-heavy workloads**:
- Reasoning-lite E2E APE: 75-180% (down from 100% timeout in iter14) ✅
- Long outputs (256-512 tokens) → decode dominates E2E → β₁ amplification prevents underestimation

**Why β₁ is insufficient for prefill-heavy workloads**:
- Dense roleplay TTFT: 4124-4151% (catastrophic) ❌
- TTFT measures prefill latency, not decode latency
- β₁ amplifies decode time but doesn't fix prefill time

**Action**: Keep β₁ (helps decode), but fix prefill basis functions separately.

---

#### β₂: TP Communication Scaling

**Value**: 0.207 (expected: 0.15-0.25) ✅ IN RANGE

**Physical interpretation**:
- Scales TP all-reduce communication time
- TP communication formula: `β₂ × (2 × layer_count × hidden_dim × sizeof(dtype) / network_bandwidth)`
- β₂ = 0.207 → TP communication is 20.7% of theoretical (NVLink bandwidth ≈ 300 GB/s)

**Why β₂ is low**:
- Overlapping: vLLM overlaps communication with computation → effective communication time is reduced
- Compression: FP8/INT8 reduces communication volume
- Topology: Ring all-reduce on NVLink has better bandwidth than theory predicts

**Action**: β₂ is physically plausible. No changes needed.

---

#### β₃: KV Management Base Overhead

**Value**: 0.001 ms (expected: 0.4-1.5 ms) ❌ COLLAPSED

**Physical interpretation**:
- Per-request KV cache allocation/deallocation overhead
- β₃ = 0.001 ms = 1 μs per request

**Why β₃ collapsed**:
- KV management overhead is negligible compared to GPU execution time (seconds)
- Or: KV overhead is already captured by other terms (β₁ for KV cache read bandwidth, β₀ for KV cache write in prefill)
- Result: Optimizer pushed β₃ to 0 to avoid double-counting

**Action**: Remove β₃ from iter16 (optimizer rejected it).

---

#### β₄: Decode Compute MFU Scaling

**Value**: 6.471 (expected: 3.0-8.0) ✅ IN RANGE

**Physical interpretation**:
- Amplifies roofline decode compute time by 6.5×
- Roofline assumes decode achieves peak FLOPs (300-400 TFLOPS on H100 for FP16)
- β₄ = 6.5 → vLLM decode achieves 46-62 TFLOPS (300-400 / 6.5)

**Why β₄ is high (amplification, not scaling)**:
- Small per-request GEMMs have poor tensor core utilization (40-80% vs 90%+ for large batches)
- Per-layer kernel launches and synchronization reduce effective FLOPS
- Mixed precision (FP8/FP16) may not achieve theoretical peak

**Evidence β₄ helps decode-heavy workloads**:
- Same pattern as β₁ (memory) - reasoning-lite E2E improves, but prefill-heavy workloads fail

**Collinearity concern**:
- β₁ = 6.398 (decode memory) and β₄ = 6.471 (decode compute) are nearly EQUAL
- This suggests they may be collinear (capturing the same effect)
- Decode time = max(memory_time, compute_time) → amplifying both by ~6.5× is equivalent to amplifying the max by 6.5×
- But roofline should identify which is the bottleneck (memory vs compute) - if both are amplified equally, the model is not distinguishing them

**Action**:
1. Keep β₁, β₄ for now (they help decode)
2. **Investigate collinearity**: Profile real vLLM decode to check if memory or compute is the bottleneck
3. If collinear, combine into single β_decode (simplifies to 9 coefficients)

---

#### β₅: MoE Gating Efficiency

**Value**: 33.569 (expected: 20-50) ✅ IN RANGE

**Physical interpretation**:
- Scales MoE gating network FLOPs (routing probability computation)
- β₅ = 33.57 → gating overhead is 33.57× higher than roofline predicts
- This is very high - suggests gating network is severely underutilized

**Why β₅ is high**:
- Gating network is a small GEMM (hidden_dim × num_experts) per layer per token
- Small GEMMs have poor tensor core utilization
- Per-token gating (vs batched) may have kernel launch overhead

**Why β₅ alone doesn't fix Scout underestimation**:
- Scout TTFT APE: 708-1634% (avg 1068%) - catastrophically high
- β₅ only scales gating FLOPs, not expert execution FLOPs
- If MoE FLOPs calculation for experts is wrong, β₅ cannot fix it

**Action**: Keep β₅ (gating is real overhead), but investigate MoE expert FLOPs calculation.

---

#### β₆: Scheduler Overhead

**Value**: 0.042 ms (expected: 40-100 ms) ❌ COLLAPSED

**Physical interpretation**:
- Per-request scheduler overhead (vLLM's AsyncLLMEngine scheduling latency)
- β₆ = 0.042 ms = 42 μs per request

**Why β₆ collapsed**:
- Scheduler overhead is fixed per request, not per token
- For long requests (256-512 tokens), 40-100 ms scheduling is only 5-10% of total latency
- Optimizer may have pushed β₆ to 0 because it's negligible, or because it's already captured by α₀ (API overhead)

**Action**: Remove β₆ from iter16 (optimizer rejected it).

---

#### β₇: Decode Per-Request Overhead

**Value**: 0.016 ms (expected: 15-30 ms) ❌ COLLAPSED

**Physical interpretation**:
- Per-request overhead in decode phase (separate from β₁, β₄ which are per-token)
- β₇ = 0.016 ms = 16 μs per request

**Why β₇ collapsed**:
- Similar to β₆ - fixed overhead is small compared to per-token decode latency
- For 256 output tokens, 15-30 ms overhead is 1-5% of total decode time
- Optimizer pushed β₇ to 0 to avoid double-counting with β₁, β₄

**Action**: Remove β₇ from iter16 (optimizer rejected it).

---

#### β₈: MoE Non-Compute Latency (NEW in iter15)

**Value**: 0.000037 (expected: 10-40 μs/token) ❌ **REJECTED**

**Physical interpretation**:
- Per-token non-compute overhead in MoE layers (routing, load imbalance, expert communication)
- β₈ = 3.7e-5 μs/token = 0.037 nanoseconds/token (NEGLIGIBLE)

**Why β₈ was rejected**:
- Optimizer found β₈ has no predictive value
- MoE routing/load-imbalance overhead is either:
  1. Already captured by β₅ (MoE gating)
  2. Negligible compared to expert execution time
  3. Wrong functional form (per-token may not match real vLLM MoE)

**Evidence β₈ is wrong**:
- Scout errors INCREASED from iter14 (527% avg) to iter15 (1068% avg) with β₈ available
- Adding β₈ made the model WORSE, not better
- Root cause: MoE FLOPs calculation is wrong, not missing non-compute term

**Action**: **Remove β₈ from iter16** (optimizer rejected it).

---

#### β₉: Prefill Batching Penalty (NEW in iter15)

**Value**: 7.17e-07 (expected: 0.5-2.0 μs/token) ❌ **REJECTED**

**Physical interpretation**:
- Per-token heterogeneity-induced overhead in prefill batches (mixing prefill and decode requests)
- β₉ = 7.2e-7 μs/token = 0.00072 nanoseconds/token (NEGLIGIBLE)

**Why β₉ was rejected**:
- Optimizer found β₉ has no predictive value
- Batch heterogeneity is either:
  1. Already captured by β₀ (prefill MFU scaling)
  2. Not the cause of dense overestimation (wrong hypothesis)
  3. Wrong functional form (linear in heterogeneity ratio may not match reality)

**Evidence β₉ is wrong**:
- Dense roleplay (low heterogeneity) has WORSE errors (4124%) than dense codegen (high heterogeneity, 1847%)
- Pattern is OPPOSITE of hypothesis prediction
- Root cause: Dense overestimation is due to wrong base prefill calculation (see Principle 1)

**Action**: **Remove β₉ from iter16** (optimizer rejected it).

---

### Redundant Terms

**Coefficients near zero** (should be removed in iter16):
1. **β₃** (KV management): 0.001 (expected 0.4-1.5 ms) - **1500× smaller**
2. **β₆** (scheduler overhead): 0.042 (expected 40-100 ms) - **1000-2000× smaller**
3. **β₇** (decode per-request): 0.016 (expected 15-30 ms) - **1000-2000× smaller**
4. **β₈** (MoE non-compute): 0.000037 (expected 10-40 μs) - **270,000-1,000,000× smaller**
5. **β₉** (prefill batching): 7.17e-07 (expected 0.5-2.0 μs) - **700,000-2,800,000× smaller**

**Recommendation**: Remove β₃, β₆, β₇, β₈, β₉ → Reduce from 10 to **5 beta coefficients** (β₀, β₁, β₂, β₄, β₅).

---

### Missing Physics

**Coefficient magnitudes suggest missing terms**:

1. **Prefill fixed overhead**: β₀ alone cannot explain 1300-4000% dense TTFT errors
   - Pattern: Shorter prompts (64 tokens) have worse errors than longer prompts (512 tokens)
   - Hypothesis: `prefill_time = β_prefill_fixed + β₀ × roofline_prefill_time`
   - Add **β_prefill_fixed** (ms) to capture fixed overhead (kernel launch, attention setup)

2. **Decode batch size dependence**: β₁, β₄ may not capture small-batch inefficiency
   - Pattern: Decode amplification (6×) helps long outputs but overcorrects E2E for short outputs
   - Hypothesis: Decode MFU depends on batch size (small batches have worse MFU)
   - Replace `β₁ × roofline_decode_memory` with `β₁ × f(batch_size) × roofline_decode_memory`

3. **MoE FLOPs correction**: β₅ (gating) is high (33×) but Scout errors still catastrophic (1068%)
   - Pattern: MoE underestimation persists even with β₅
   - Hypothesis: MoE expert FLOPs calculation in roofline is wrong (active vs total experts)
   - Add **β_moe_expert_ratio** to scale expert FLOPs separately from gating

**Action for iter16**: Profile vLLM to validate these missing physics hypotheses before adding new terms.

---

## Recommendations for iter16

### Priority 1: Critical Issues (MUST FIX)

**Issue 1: Roofline-based prefill model is fundamentally broken**

**Evidence**:
- Dense TTFT APE: 1300-4000% (13-40× too fast)
- β₀ = 0.092 provides 11× scale-down, still fails by 13-40×
- β₉ (batching penalty) rejected by optimizer (collapsed to 0)
- Root cause: Base roofline prefill calculation is orders of magnitude wrong

**Action**:
1. **Profile real vLLM prefill latency** for 3-5 dense models:
   - Vary: batch size (1, 2, 4, 8), sequence length (64, 128, 256, 512), TP (1, 2, 4)
   - Measure: TTFT (time-to-first-token)
   - Extract: Empirical formula `prefill_time = f(batch, seq_len, TP, model_params)`
2. **Replace roofline prefill with empirical model**:
   - Current: `β₀ × max(compute_time, memory_time)`
   - New: `a × batch^b × seq_len^c × TP^d × params^e` (fit a, b, c, d, e from profiling data)
3. **Test hypothesis**: Does empirical model reduce dense TTFT APE from 1300-4000% to <500%?

**Expected impact**: If prefill formula is correct, should fix 80%+ of dense TTFT errors.

---

**Issue 2: Decode functional form may be wrong despite directional correctness**

**Evidence**:
- β₁, β₄ ≈ 6.5 (used heavily by optimizer) → decode amplification has SOME signal
- Reasoning-lite E2E APE: 75-180% (improvement) ✅
- But reasoning-lite TTFT APE: 238-668% (still high) ❌
- Dense roleplay E2E APE: 9711-11357% (catastrophic) ❌

**Action**:
1. **Profile real vLLM decode latency** for 3-5 models:
   - Vary: batch size (1, 2, 4, 8, 16), KV cache size (64, 128, 256, 512 tokens), TP (1, 2, 4)
   - Measure: ITL (inter-token latency) per decode step
   - Extract: Check if `decode_time ∝ roofline_decode_time` (linear) or has non-linear dependence
2. **Test collinearity**: Are β₁ (memory) and β₄ (compute) capturing independent effects?
   - Run optimization with β₁ only (freeze β₄ = 1.0)
   - Run optimization with β₄ only (freeze β₁ = 1.0)
   - Compare: If both give similar loss, they're collinear → combine into β_decode
3. **Check batch size dependence**: Does decode MFU degrade for small batches?
   - Hypothesis: `decode_time = β_decode × (1 + k / batch_size) × roofline_decode_time`
   - Small batches (1-2) → large k/batch_size overhead → worse MFU
   - Large batches (16+) → k/batch_size ≈ 0 → roofline MFU

**Expected impact**: If decode formula is correct, should improve reasoning-lite TTFT from 238-668% to <200%.

---

**Issue 3: MoE FLOPs calculation is likely wrong**

**Evidence**:
- β₈ (MoE non-compute) rejected by optimizer (collapsed to 0)
- Scout TTFT APE: 708-1634% (avg 1068%) - catastrophically high
- Scout errors INCREASED from iter14 (527%) to iter15 (1068%) - 102% worse
- β₅ (MoE gating) = 33.57 (in range) but doesn't fix Scout underestimation

**Action**:
1. **Investigate roofline MoE FLOPs calculation**:
   - Check: Are active experts vs total experts counted correctly?
   - Example: Scout has 16 experts, top-2 routing → only 2/16 = 12.5% of experts are active
   - If roofline counts ALL 16 experts → 8× overestimate of FLOPs → predicts 8× too fast
2. **Profile real vLLM MoE latency** for Scout vs equivalent dense model:
   - Measure: TTFT ratio (Scout / Llama-2-13B or similar param count)
   - Expected: Scout should be 1.5-3× SLOWER than dense (expert overhead)
   - Check: Does roofline predict this ratio correctly?
3. **Redesign β₅ (MoE gating) if needed**:
   - Current: `β₅ × moe_gating_flops`
   - If active/total experts issue: Add `β_moe_expert_ratio × expert_flops`
   - If load imbalance issue: Multiply expert_flops by `(1 + load_imbalance_penalty)`

**Expected impact**: If MoE FLOPs formula is correct, should reduce Scout TTFT APE from 1068% to <300%.

---

### Priority 2: Improvements (SHOULD FIX)

**Issue 4: Wrong warm-start baseline (should be iter9, not iter7 or cold-start)**

**Evidence**:
- Cold-start loss (6538%) is 2.8× worse than iter14's warm-start (2319%)
- Iter14 warm-started from iter7 (different dataset) → loss 2319%
- **Iter9 was last stable state** (161% loss) on SAME dataset as iter15 (15 clean experiments)
- Iter15 ignored iter9 entirely, compared cold-start to iter7 (wrong comparison)
- Optimizer rejected 5/10 coefficients (β₃, β₆, β₇, β₈, β₉ collapsed to 0)

**Why iter9, not iter7**:
1. **Dataset alignment**: Iter9 trained on SAME dataset as iter15 (15 clean, reasoning-lite)
2. **Last stable state**: Iter9 (161%) → iter10 (4267%) is where cascade started
3. **Proven coefficients**: Iter9's β₀, β₁, β₄, β₅, β₇ were stable (5 out of 7)
4. **Sample efficiency**: Starting from iter9 needs 1000 trials (not 2000 like cold-start)

**Action**:
1. **Reduce dimensionality**: Remove collapsed terms (β₃, β₆, β₇, β₈, β₉) → 5 beta coefficients (β₀, β₁, β₂, β₄, β₅)
2. **Use warm-start from iter9** (NOT iter7, NOT cold-start):
   - Iter9 stable coefficients: β₀=0.191, β₁=1.108, β₄=0.705, β₅=27.5, β₇=0.027 ✅
   - Iter9 inflated coefficients: β₃=9.6ms (reset to 1.0ms), β₆=99ms (reset to 40ms)
   - Iter9 dataset: 15 clean experiments (SAME as iter15/16)
3. **Add vLLM-profiled basis functions** (not roofline-based):
   - Replace β₀ × roofline with empirical prefill model from profiling
   - Validate β₁, β₄ decode formulas from profiling
   - Fix MoE FLOPs calculation for β₅

**Expected impact**: 5D search space + warm-start from iter9 + fixed basis functions → loss 6538% to <500% (13× improvement).

**Critical lesson**: When system enters cascade (iter10-15), revert to **last known-good state on CURRENT dataset** (iter9), not earlier state on different dataset (iter7) or fresh start (cold-start).

---

**Issue 5: Coefficient collinearity (β₁ vs β₄) reduces model interpretability**

**Evidence**:
- β₁ = 6.398 (decode memory MFU)
- β₄ = 6.471 (decode compute MFU)
- Nearly equal values suggest collinearity (capturing same effect)

**Action**:
1. **Test collinearity** (see Priority 1, Issue 2)
2. **If collinear, combine into β_decode**:
   - Current: `β₁ × roofline_decode_memory + β₄ × roofline_decode_compute`
   - New: `β_decode × max(roofline_decode_memory, roofline_decode_compute)`
   - Reduces from 5 to 4 beta coefficients (β₀, β_decode, β₂, β₅)
3. **If independent, keep separate** but add interaction term:
   - Check: Does decode MFU depend on memory-compute balance?
   - Add: `β_interaction × roofline_decode_memory × roofline_decode_compute`

**Expected impact**: Simplifies model to 4 coefficients, improves interpretability and sample efficiency.

---

### Priority 3: Refinements (NICE TO HAVE)

**Refinement 1: Add prefill fixed overhead term**

**Pattern**: Shorter prompts (64 tokens) have worse TTFT errors (4124%) than longer prompts (512 tokens, 1847%).

**Hypothesis**: `prefill_time = β_prefill_fixed + β₀ × roofline_prefill_time`
- Fixed overhead (kernel launch, attention setup) dominates for short prompts
- Per-token overhead (roofline_prefill_time) dominates for long prompts

**Action**:
1. Check vLLM profiling data: Is there a fixed prefill overhead (intercept in linear fit)?
2. If yes, add **β_prefill_fixed** (ms) as new coefficient (increases from 4 to 5 beta terms)

---

**Refinement 2: Add batch size dependence to decode terms**

**Pattern**: Decode amplification (6×) helps long outputs (reasoning-lite) but may overcorrect short outputs (dense roleplay).

**Hypothesis**: Decode MFU degrades for small batches
- Small batch (1-2 requests) → poor tensor core utilization → lower MFU
- Large batch (16+ requests) → good utilization → roofline MFU

**Action**:
1. Check vLLM profiling data: Does decode ITL scale linearly with batch size, or is there batch size dependence?
2. If non-linear, replace: `β_decode × roofline` with `β_decode × (1 + k / batch_size) × roofline`

---

### Specific Actions for iter16

**Basis function changes**:

**Remove** (optimizer rejected):
- β₃ (KV management base) - collapsed to 0.001
- β₆ (scheduler overhead) - collapsed to 0.042
- β₇ (decode per-request) - collapsed to 0.016
- β₈ (MoE non-compute latency) - collapsed to 0.000037
- β₉ (prefill batching penalty) - collapsed to 7.17e-07

**Keep**:
- β₀ (prefill MFU scaling) - necessary but insufficient
- β₁ (decode memory MFU) - helps decode-heavy workloads
- β₂ (TP communication) - physically plausible
- β₄ (decode compute MFU) - helps decode-heavy workloads (test collinearity with β₁)
- β₅ (MoE gating) - captures gating overhead, but MoE expert FLOPs may need separate term

**Modify/Replace**:
- **β₀ × roofline_prefill** → Replace with empirical prefill model from vLLM profiling
- **β₁, β₄** → Test collinearity, combine into β_decode if collinear
- **β₅** → Keep, but investigate MoE expert FLOPs calculation

**Add** (if validated by profiling):
- **β_prefill_fixed** (ms): Fixed prefill overhead (if intercept is significant)
- **β_moe_expert_ratio**: Scales MoE expert FLOPs separately from gating (if roofline counts wrong experts)

**Resulting architecture**: 4-6 beta coefficients (down from 10 in iter15).

---

**Bounds adjustments**:

**β₀** (prefill MFU):
- Current range: [0.05, 0.25]
- Optimal: 0.092
- **Keep range** - optimal is within bounds, but REPLACE roofline-based formula with empirical model

**β₁** (decode memory MFU):
- Current range: [5.0, 15.0]
- Optimal: 6.398
- **Keep range** - optimal is in lower third, suggests current formula is directionally correct
- Action: Test collinearity with β₄

**β₄** (decode compute MFU):
- Current range: [3.0, 8.0]
- Optimal: 6.471
- **Keep range** - optimal is in upper half, suggests good coverage
- Action: Test collinearity with β₁

**β₅** (MoE gating):
- Current range: [20, 50]
- Optimal: 33.569
- **Keep range** - optimal is in middle, suggests good coverage
- But: Scout errors still catastrophic (1068%) → investigate MoE expert FLOPs, not just gating

---

**Optimization strategy**:

1. **Fix basis functions FIRST** (profile vLLM, replace roofline formulas) - see Priority 1
2. **Reduce dimensionality** (remove β₃, β₆, β₇, β₈, β₉) → 5 beta coefficients
3. **Test collinearity** (β₁ vs β₄) → potentially reduce to 4 coefficients
4. **Use warm-start** from iter7 (β₀=0.191, β₁=1.108, etc.) with 1000 trials (not 2000)
5. **Add physics priors** from vLLM profiling for initialization

**Expected outcome**: With correct basis functions and reduced dimensionality, iter16 should achieve <500% loss (10× improvement from iter15's 6538%).

---

## Cross-Validation Decision

**CV tests NOT warranted** - catastrophic failure (loss 6538%, target <300%).

**Criteria for CV** (from analysis-agent-prompt.md):
- ✅ All hypotheses confirmed
- Overall loss < 80% (ideally < 50%)
- No experiment with TTFT or E2E APE > 100%
- Coefficients physically plausible

**Iter15 status**:
- ❌ All 6 hypotheses REJECTED or PARTIAL (none confirmed)
- ❌ Overall loss = 6538% (82× above threshold)
- ❌ 14/15 experiments have TTFT APE > 100% (only exp_48 has 30%)
- ⚠️ Coefficients partially plausible (6/10 in range, 4/10 collapsed)

**Recommendation**: Skip CV, proceed to iter16 with Priority 1 actions (profile vLLM, replace roofline formulas).
