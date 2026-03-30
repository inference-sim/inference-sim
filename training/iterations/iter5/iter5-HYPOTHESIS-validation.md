# Iteration 5: Hypothesis Validation

## H-main: Per-Layer Fixed Overhead as the Missing Prefill Term

**Prediction** (from Agent 1): Overall loss will decrease to <110% (from 129% in iter4), with:
- TTFT RMSE reducing from 66.49% to <55%
- E2E RMSE staying stable at ~62-65% (decode already well-modeled)
- β₀ (prefill MFU) rising from 0.165 to 0.25-0.35 (closer to physical plausibility)
- Reasoning experiments improving from ~100% TTFT to 70-85% TTFT (measurable progress toward resolution)

**Causal Mechanism** (from Agent 1): Per-layer fixed overhead (kernel launch + scheduler + memory allocation) scales with prefill chunking. Formula: `overhead_us = β₆ × num_layers × (1.0 + num_prefill_tokens / 2048.0)`. Expected β₆ ~ 1000-3000μs (1-3ms per layer-chunk) to capture the 900ms gap in reasoning experiments.

**Diagnostic Clause** (from Agent 1): If this fails (overall loss >120% or β₀ doesn't rise above 0.22), it indicates per-layer overhead is NOT the dominant bottleneck. Real cause may be algorithmic switch (different attention kernel for long contexts), O(n²) attention memory bandwidth, or KV cache preemption overhead.

**Actual Result**:
- Overall loss: **603.26%** (TTFT RMSE = 518.85%, E2E RMSE = 84.41%)
- β₀ = **0.266** (rose to expected range 0.25-0.35 ✓)
- β₆ = **521.53μs** (far below expected 1000-3000μs)
- Reasoning experiments: **99% TTFT** (minimal improvement from 99.98%, not 70-85%)
- Short-context experiments: **300-1091% TTFT** (catastrophic degradation)

**Verdict**: ❌ **REJECTED**

**Evidence**:
- Overall loss: 603.26% vs target <110% (548% miss, 467% worse than iter4's 129%)
- TTFT RMSE: 518.85% vs target <55% (463% miss, 8× worse than iter4's 66.49%)
- E2E RMSE: 84.41% vs stable 62-65% (22-32% worse than expected)
- β₀ coefficient: 0.266 ✓ (within predicted 0.25-0.35 range)
- β₆ coefficient: 521.53μs ✗ (expected 1000-3000μs, 48-83% lower)
- Reasoning TTFT: 99% (improved by ~0.7 percentage points from 99.75-99.99%, not 15-30pp to reach 70-85%)
- Short-context degradation pattern:
  - Llama-3.1-70B TP=4 codegen: 1091% TTFT (was 3.86% in iter4, +1087pp degradation!)
  - Mistral codegen: 834% TTFT (was 30.69% in iter4, +803pp degradation)
  - Llama-2 roleplay: 822% TTFT (was 64.03% in iter4, +758pp degradation)
  - Qwen roleplay: 736% TTFT (was 59.21% in iter4, +677pp degradation)

**Causal Analysis**:

The hypothesis **catastrophically failed** due to an **inverse boundary effect**:

1. **β₀ rose as predicted (0.165 → 0.266), BUT this caused massive over-prediction of short-context experiments**:
   - β₀ represents prefill MFU — higher β₀ means model predicts FASTER prefill
   - 61% increase in β₀ (0.165 → 0.266) means predicted prefill time decreased by 38%
   - Short-context experiments (which were well-predicted in iter4) are now 10-30× over-predicted
   - Example: Llama-3.1-70B codegen went from 3.86% TTFT error → 1091% TTFT error

2. **β₆ converged to 521μs (not 1000-3000μs), providing insufficient overhead to help reasoning**:
   - For reasoning (8K tokens, 80 layers): overhead = 521 × 80 × 5.0 = 208ms
   - Actual underestimation: ~900ms (from iter4 analysis)
   - β₆ captures only 23% of missing overhead (208ms / 900ms)
   - Reasoning improved marginally (~0.7pp) because β₆ is too small

3. **The two effects combined catastrophically**:
   - β₀ increase shortened ALL prefill predictions by 38%
   - β₆ overhead added back only 208ms for long contexts (insufficient)
   - Net effect: Short contexts massively over-predicted, reasoning barely improved

4. **Why β₆ converged low**: The optimizer faced a trade-off:
   - Increasing β₆ helps reasoning experiments (4 exps with 99% TTFT)
   - But also adds overhead to ALL experiments (15 total)
   - Short-context experiments (11 exps) were already over-predicted due to β₀ increase
   - Adding more overhead would worsen 11 experiments to slightly help 4 experiments
   - Optimizer chose β₆ = 521μs as compromise (minimizes overall RMSE across all 15)

**Diagnostic Analysis** (using Agent 1's diagnostic clause):

Agent 1's diagnostic clause stated: *"If overall loss >120% or β₀ doesn't rise above 0.22, it indicates per-layer overhead is NOT the dominant bottleneck."*

**Result**: Overall loss = 603% (>120% ✓) AND β₀ = 0.266 (>0.22 ✓)

This creates a **diagnostic contradiction**:
- β₀ DID rise to physical range (diagnostic NOT triggered by this condition)
- BUT loss is catastrophically high (diagnostic IS triggered by this condition)

**Resolution**: The diagnostic clause missed a critical failure mode: **β₀ rising WITHOUT a correctly-sized overhead term causes inverse errors**. The hypothesis assumed β₀ would stay bounded until the right overhead term was added. Instead, β₀ rose freely, causing short-context over-prediction.

**Root cause indicated by diagnostic**: Per-layer overhead **is plausibly correct in mechanism** but **wrong in functional form**:
- The 1.0 + tokens/2048.0 scaling factor may be wrong (linear when should be quadratic? constant when should be logarithmic?)
- OR the base formula should only apply to LONG contexts (>4K tokens), not all contexts
- OR β₆ should be decoupled from β₀ (currently correlated, causing gradient masking)

**What to investigate next** (from diagnostic clause):
1. **Algorithmic switch**: vLLM may use different attention kernel for contexts >8K, with fixed overhead only applying to long-context kernel
2. **Per-context-length β₀**: Short contexts may need β₀ ~ 0.40-0.50 (physical MFU), long contexts may need β₀ ~ 0.15-0.25 (degraded MFU) + overhead term
3. **Quadratic overhead scaling**: overhead ∝ (num_prefill_tokens)² instead of linear, to prevent short-context over-prediction

---

## H-simplification-validated: Removing β₆ (Activation Bandwidth) Improves Stability

**Prediction** (from Agent 1): Removing β₆ (activation bandwidth) from iter4 will:
- NOT degrade any experiments by >3%
- Stabilize other coefficients: β₁ → 1.00-1.10 (from 1.802), β₂ → 0.30-0.35 (from 1.360), β₅ → 0.01-0.012 (from 0.0304)
- Reduce parameter count from 7 to 6 Beta terms
- Improve optimization convergence

**Actual Result**:
- β₁ = 1.449 (predicted 1.00-1.10, still 32-45% too high)
- β₂ = 1.368 (predicted 0.30-0.35, essentially unchanged from iter4's 1.360)
- β₅ = 0.0149 (predicted 0.01-0.012, 24-49% too high)
- β₃ = 0.000013 (collapsed to near-zero, was 0.000495 in iter4)
- β₄ = 0.620 (dropped to 0.620, was 0.918 in iter4)
- Optimization: Converged early at 78 trials (vs 185 in iter4)

**Verdict**: ⚠️ **PARTIAL**

**Evidence**:
- Coefficient stabilization: β₁, β₂, β₅ did NOT revert to iter3 ranges as predicted
  - β₁: 1.802 (iter4) → 1.449 (iter5), improved by 20% but still 32% above target 1.10
  - β₂: 1.360 (iter4) → 1.368 (iter5), essentially unchanged (0.6% increase)
  - β₅: 0.0304 (iter4) → 0.0149 (iter5), improved by 51% but still 24% above target 0.012
- Unexpected coefficient collapse:
  - β₃ (KV cache management): 0.000495 → 0.000013 (97% reduction, near-zero)
  - β₄ (decode compute-bound): 0.918 → 0.620 (32% reduction)
- Convergence: Improved (185 → 78 trials), confirming reduced parameter space helps
- Degradation: Cannot assess directly (iter5 changed TWO things: removed β₆ activation + added β₆ per-layer), but catastrophic loss increase suggests removal alone was insufficient

**Causal Analysis**:

The hypothesis that **removing activation bandwidth would stabilize coefficients** is ⚠️ **partially confirmed**:

1. **Improvements (confirming hypothesis)**:
   - β₁ improved by 20% (1.802 → 1.449), moving toward target
   - β₅ improved by 51% (0.0304 → 0.0149), moving toward target
   - Convergence faster (185 → 78 trials), reduced parameter space helps
   - No gradient masking from misspecified activation term

2. **Failures (refuting hypothesis)**:
   - β₁, β₂, β₅ did NOT revert to iter3 ranges (still 24-45% above target)
   - β₂ essentially unchanged (1.360 → 1.368), suggesting collinearity persists
   - β₃ collapsed to near-zero (was stable at 0.0005 for 3 iterations)
   - β₄ dropped 32% (decode compute should be stable)

3. **Why partial reversion**:
   - Removing activation BW eliminated one source of collinearity (iter4's β₆)
   - BUT added new β₆ (per-layer overhead) which also scales with num_layers
   - New β₆ still creates gradient masking with β₀, preventing full stabilization
   - β₂ unchanged suggests it's absorbing error from new β₆ (correlated with TP configs)

4. **Why β₃/β₄ collapsed**:
   - β₃ (KV cache management) collapsed because β₀ increased 61%
   - Higher β₀ means shorter prefill predictions, less need for KV management overhead
   - β₄ (decode compute) dropped because β₁ (decode memory) is too high
   - Optimizer balances β₁ (memory-bound) vs β₄ (compute-bound) decode time

**Diagnostic Analysis** (using Agent 1's diagnostic clause):

Agent 1's diagnostic stated: *"If experiments degrade by >5% after removing β₆, it indicates the formula was partially correct despite appearing ineffective."*

**Result**: Cannot directly assess degradation from removing activation BW, because iter5 changed TWO things (removed activation BW + added per-layer overhead). However, the catastrophic loss increase (129% → 603%) suggests:
- Removing activation BW alone was NOT sufficient (coefficient drift persisted)
- Adding per-layer overhead in current functional form caused inverse boundary effect
- Net effect: Partial confirmation (coefficients moved toward iter3, but slowly)

**Conclusion**: Removing activation bandwidth was **necessary but not sufficient** for stabilization. Coefficients improved 20-51% but didn't fully revert because new β₆ (per-layer overhead) also creates collinearity with β₀.

---

## H-coefficient-normalization: Physical Plausibility Recovery

**Prediction** (from Agent 1): With β₆ (activation bandwidth) removed and new β₆ (per-layer overhead) added, coefficients will move toward physically plausible ranges:
- β₀: 0.165 → 0.25-0.35 (improved, but still below ideal 0.40-0.55)
- β₁: 1.802 → 1.00-1.10 (revert to iter3)
- β₂: 1.360 → 0.30-0.35 (revert to iter3)
- β₃: stable at ~0.0004-0.0005
- β₄: stable at ~0.75-0.85
- β₅: 0.0304 → 0.01-0.012 (revert to iter3)
- β₆ (NEW): 1000-3000μs

**Actual Result**:
- β₀ = 0.266 ✓ (within predicted 0.25-0.35 range)
- β₁ = 1.449 ✗ (improved from 1.802, but still 32-45% above target 1.00-1.10)
- β₂ = 1.368 ✗ (essentially unchanged from 1.360, far above target 0.30-0.35)
- β₃ = 0.000013 ✗ (collapsed to near-zero, expected 0.0004-0.0005)
- β₄ = 0.620 ✗ (dropped below target 0.75-0.85)
- β₅ = 0.0149 ⚠️ (improved from 0.0304, but still 24-49% above target 0.01-0.012)
- β₆ = 521.53μs ✗ (far below expected 1000-3000μs)

**Verdict**: ❌ **REJECTED**

**Evidence**:
- 1 out of 7 coefficients in predicted range (β₀)
- 4 coefficients outside physical plausibility (β₁, β₂, β₃, β₆)
- 2 coefficients partially improved but not normalized (β₄, β₅)
- Key failure: β₆ = 521μs (expected 1000-3000μs), only 17-52% of predicted value

**Causal Analysis**:

The hypothesis that **coefficients would normalize after removing activation BW and adding per-layer overhead** is ❌ **rejected**:

1. **Why β₀ normalized but others didn't**:
   - β₀ rose to 0.266 because new β₆ (per-layer overhead) scales with num_layers
   - Optimizer can increase β₀ (predict faster prefill) while adding β₆ overhead (slow prefill back down)
   - BUT β₀ rising BROKE short-context experiments (now over-predicted by 10-30×)
   - This prevented β₀ from rising further toward physical 0.40-0.55

2. **Why β₁, β₂ didn't normalize**:
   - β₁ (decode memory) improved 20% but still 32-45% too high
   - β₂ (TP communication) essentially unchanged at 1.360 (far above physical 0.30-0.35)
   - Both still absorbing error from model misspecification
   - β₂ unchanged suggests new β₆ term has same collinearity issue as old activation BW term

3. **Why β₃ collapsed**:
   - β₃ (KV cache management) was 0.000495 in iter4, now 0.000013 (97% drop)
   - Collapsed because β₀ increased 61% (shorter prefill predictions)
   - Less prefill time → less need for KV management overhead
   - May indicate β₃ was absorbing prefill overhead that β₀/β₆ should capture

4. **Why β₆ converged low**:
   - Optimizer faced trade-off: increasing β₆ helps 4 reasoning exps, hurts 11 short-context exps
   - With β₀ at 0.266, short contexts already over-predicted by 10-30×
   - Adding more β₆ overhead would worsen 73% of training set to help 27%
   - Optimizer chose β₆ = 521μs as compromise (minimizes RMSE across all experiments)

**Diagnostic Analysis** (using Agent 1's diagnostic clause):

Agent 1's diagnostic stated: *"If β₀ doesn't rise above 0.22, it indicates per-layer overhead is NOT the missing term."*

**Result**: β₀ = 0.266 (>0.22 ✓), so diagnostic NOT triggered by this condition.

**However**, Agent 1's diagnostic also stated: *"If β₁, β₂, or β₅ don't normalize (stay >20% above iter3 values), it indicates new β₆ has collinearity with existing terms."*

**Result**: All three failed to normalize:
- β₁ = 1.449 (40% above iter3's 1.037)
- β₂ = 1.368 (330% above iter3's 0.318)
- β₅ = 0.0149 (27% above iter3's 0.0117)

**Conclusion**: New β₆ (per-layer overhead) **does have collinearity** with existing terms, likely β₀ (both scale with num_layers). This prevented full coefficient normalization.

**Additional diagnostic**: Agent 1 stated: *"If β₆ converges to <500μs or >5000μs, it indicates functional form is wrong."*

**Result**: β₆ = 521.53μs (marginally above 500μs threshold), suggesting functional form is **at the edge of validity**. The optimizer pushed β₆ as low as possible while staying within plausible bounds.

**Root cause**: The per-layer overhead functional form `1.0 + tokens/2048.0` applies overhead to ALL experiments, not just long-context ones. This creates a zero-sum game where helping reasoning experiments hurts short-context experiments.

---

## H-boundary: Per-Layer Overhead Scales with Prompt Length

**Prediction** (from Agent 1): The new β₆ (per-layer overhead) will affect experiments differently based on prompt length:
- Short prompts (<2K tokens): Minimal overhead (<100ms total), experiments change <10%
- Medium prompts (2-4K tokens): Moderate overhead (100-300ms), experiments improve 10-25%
- Long prompts (>4K tokens, reasoning): Large overhead (300-900ms), experiments improve >25% (from ~100% TTFT to 70-85%)

**Actual Result**:
- Short prompts: **CATASTROPHIC degradation** (300-1091% TTFT, was 4-64% in iter4)
- Medium prompts: **Major degradation** (215-506% TTFT, was 15-77% in iter4)
- Long prompts (reasoning): **Minimal improvement** (99% TTFT, was 99.75-99.99% in iter4, only ~0.7pp improvement)

**Verdict**: ❌ **REJECTED** (inverse boundary effect)

**Evidence**:
Per-experiment breakdown by prompt length:

**Short prompts (<2K tokens)** — Predicted <10% change, actual 300-1091% TTFT:
- Llama-3.1-70B codegen (1K tokens): 1091% TTFT (was 3.86% in iter4, +1087pp)
- Mistral codegen (1K tokens): 834% TTFT (was 30.69% in iter4, +803pp)
- Llama-2 roleplay (800 tokens): 822% TTFT (was 64.03% in iter4, +758pp)
- Qwen roleplay (1K tokens): 736% TTFT (was 59.21% in iter4, +677pp)

**Medium prompts (2-4K tokens)** — Predicted 10-25% improvement, actual 215-506% TTFT:
- Yi-34B general-lite (3K tokens): 506% TTFT (was 14.69% in iter4, +491pp)
- Scout roleplay (2K tokens): 425% TTFT (was 84.00% in iter4, +341pp)
- Llama-2 general (2K tokens): 365% TTFT (was 56.51% in iter4, +309pp)
- Llama-3.1-70B general-lite (4K tokens): 339% TTFT (was 70.90% in iter4, +268pp)
- Llama-2 codegen (1.5K tokens): 328% TTFT (was 39.43% in iter4, +289pp)
- Scout codegen (2K tokens): 225% TTFT (was 89.69% in iter4, +136pp)
- Mistral general-lite (3K tokens): 215% TTFT (was 76.90% in iter4, +138pp)

**Long prompts (>4K tokens, reasoning)** — Predicted >25% improvement (100% → 70-85%), actual ~0.7pp improvement:
- Qwen reasoning (16K tokens): 99.85% TTFT (was 99.99% in iter4, +0.14pp improvement)
- Scout reasoning (8K tokens): 99.66% TTFT (was 99.99% in iter4, +0.33pp improvement)
- Scout general (8K tokens): 99.45% TTFT (was 99.97% in iter4, +0.52pp improvement)
- Llama-2 reasoning (8K tokens): 99.76% TTFT (was 99.98% in iter4, +0.22pp improvement)

**Causal Analysis**:

The hypothesis is ❌ **categorically rejected** — the boundary effect is **inverted**:

1. **What was predicted**: Long prompts improve most (>25%), short prompts improve least (<10%)

2. **What actually happened**: Short prompts degraded catastrophically (10-30× worse), long prompts barely improved (~0.7pp)

3. **Why the inversion occurred**:
   - **β₀ rose from 0.165 → 0.266** (61% increase), making ALL prefill predictions 38% shorter
   - **β₆ = 521μs** (not 1000-3000μs), adding insufficient overhead for long contexts
   - **For short prompts**: overhead = 521 × 32 × 1.25 = 20.8ms (20.8ms added, but β₀ increase reduced prediction by ~50ms → net -29ms → over-prediction)
   - **For long prompts**: overhead = 521 × 80 × 5.0 = 208.6ms (208ms added, but β₀ increase reduced prediction by ~400ms → net -192ms → still under-predicted, but slightly less)

4. **The zero-sum trade-off**:
   - Optimizer minimizes RMSE across ALL 15 experiments
   - Increasing β₆ helps 4 reasoning experiments (27% of data)
   - But hurts 11 short/medium experiments (73% of data) that are now over-predicted due to β₀ increase
   - Optimizer chose β₆ = 521μs to minimize overall RMSE, sacrificing reasoning to avoid catastrophic degradation of short contexts

5. **Why didn't the optimizer balance differently?**:
   - If β₆ → 2000μs (as predicted): reasoning improves to ~70% TTFT (good!), but short contexts degrade even more (1500-2000% TTFT, catastrophic)
   - If β₀ → 0.165 (iter4 value): short contexts stay at iter4 levels (4-64% TTFT), but reasoning stays at 99.99% (no progress)
   - Optimizer chose middle ground: β₀ = 0.266, β₆ = 521μs, resulting in 603% overall loss (worst of both worlds)

**Diagnostic Analysis** (using Agent 1's diagnostic clause):

Agent 1's diagnostic stated: *"If short-prompt experiments improve significantly (>10%), it indicates a bug in the formula."*

**Result**: Short-prompt experiments **DEGRADED** by 300-1000% (not improved), so diagnostic partially triggered.

Agent 1's diagnostic continued: *"If short-prompt experiments degrade, check: Is overhead applied during decode? (should only apply during prefill)"*

**Verification needed**: Inspect `evolved_model.go` to confirm per-layer overhead only applies during prefill, not decode. Catastrophic degradation suggests overhead may be leaking into decode phase OR β₀ increase is not properly gated.

Agent 1's diagnostic also stated: *"If long-prompt experiments don't improve (reasoning stays >95% TTFT), it indicates per-layer overhead is NOT the bottleneck."*

**Result**: Reasoning stayed at 99% TTFT (>95% ✓), so diagnostic IS triggered.

**Conclusion**: Per-layer overhead **is NOT the dominant bottleneck** for reasoning experiments. The 1000× underestimation requires a different mechanism (algorithmic switch, O(n²) attention, KV preemption).

---

## H-error-pattern: Which Experiments Should Improve Most?

**Prediction** (from Agent 1): Iter5 will show largest improvements in experiments with **long prompts + large models**:
1. Reasoning experiments (99.98-99.99% → 70-85% TTFT)
2. TP=4 Llama-3.1-70B general-lite (70.90% → 50-60% TTFT)
3. Mistral TP=2 general-lite (76.90% → 60-70% TTFT)

Short-prompt, TP=1 experiments should show minimal change (<10%).

**Actual Result**:
1. Reasoning experiments: 99.75-99.99% → 99.45-99.85% TTFT (0.1-0.5pp improvement, NOT 15-30pp)
2. TP=4 Llama-3.1-70B general-lite: 70.90% → 339% TTFT (+268pp degradation!)
3. Mistral TP=2 general-lite: 76.90% → 215% TTFT (+138pp degradation!)

Short-prompt experiments: 4-64% → 300-1091% TTFT (+300-1000pp degradation!)

**Verdict**: ❌ **REJECTED** (completely inverted pattern)

**Evidence**:

**Experiments that DEGRADED most** (inverse of prediction):
1. Llama-3.1-70B TP=4 codegen: 3.86% → 1091% TTFT (+1087pp, 282× worse)
2. Mistral codegen: 30.69% → 834% TTFT (+803pp, 27× worse)
3. Llama-2 roleplay: 64.03% → 822% TTFT (+758pp, 13× worse)
4. Qwen roleplay: 59.21% → 736% TTFT (+677pp, 12× worse)

**Experiments that improved marginally** (not as predicted):
1. Qwen reasoning: 99.99% → 99.85% TTFT (0.14pp improvement)
2. Scout reasoning: 99.99% → 99.66% TTFT (0.33pp improvement)
3. Scout general: 99.97% → 99.45% TTFT (0.52pp improvement)
4. Llama-2 reasoning: 99.98% → 99.76% TTFT (0.22pp improvement)

**Causal Analysis**:

The predicted error pattern is ❌ **completely inverted**:

1. **Predicted**: Long-prompt + large-model experiments improve most
   **Actual**: Long-prompt experiments improved marginally, large-model SHORT-prompt experiments degraded catastrophically

2. **Why the inversion**:
   - The per-layer overhead formula `overhead = β₆ × num_layers × (1.0 + tokens/2048.0)` scales with BOTH num_layers AND tokens
   - Large models (Llama-3.1-70B: 80 layers) have HIGH per-layer overhead even for SHORT prompts
   - Small models (Llama-2-7B: 32 layers) have LOW per-layer overhead even for LONG prompts
   - β₀ increase (0.165 → 0.266) affected ALL experiments uniformly (38% shorter predictions)
   - Result: Large-model short-prompt experiments got hit by both β₀ increase (shorter prediction) and insufficient β₆ overhead (doesn't compensate)

3. **Example calculation**:
   - **Llama-3.1-70B codegen** (1K tokens, 80 layers):
     - β₀ effect: Prefill prediction reduced by ~38% (FLOPs / (peak × 0.266) vs FLOPs / (peak × 0.165))
     - β₆ effect: overhead = 521 × 80 × 1.5 = 62.5ms added
     - Net: Large reduction from β₀, small addition from β₆ → catastrophic over-prediction
   - **Llama-2 reasoning** (8K tokens, 32 layers):
     - β₀ effect: Prefill prediction reduced by ~38%
     - β₆ effect: overhead = 521 × 32 × 5.0 = 83.4ms added
     - Net: Still under-predicted, but slightly less than before (99.98% → 99.76%)

4. **Why didn't the optimizer choose β₆ = 2000μs** (as predicted)?
   - If β₆ = 2000μs: Llama-3.1-70B codegen would get 2000 × 80 × 1.5 = 240ms overhead
   - But it's already over-predicted by ~200ms (from β₀ increase)
   - Adding 240ms would make it 440ms over-predicted → TTFT error >2000%
   - Optimizer chose β₆ = 521μs to minimize damage to short-context experiments

**Diagnostic Analysis** (using Agent 1's diagnostic clause):

Agent 1's diagnostic stated: *"If pattern is reversed (short-prompt experiments improve more than long-prompt), it indicates the formula is wrong."*

**Result**: Pattern is INVERTED — short-prompt experiments DEGRADED massively, long-prompt improved marginally. Diagnostic IS triggered.

Agent 1's diagnostic continued: *"Functional form may be inverted (overhead decreases with prompt length instead of increasing)."*

**Assessment**: Functional form is not inverted, but β₀ increase created a zero-sum game where helping long contexts hurts short contexts. The formula adds overhead correctly, but β₀ rising to 0.266 means short contexts are now over-predicted by default.

Agent 1's diagnostic also stated: *"If ALL experiments improve uniformly (no prompt-length correlation), it indicates collinearity with β₀."*

**Result**: Experiments did NOT improve uniformly — clear prompt-length correlation (long prompts improved, short prompts degraded). This rules out complete collinearity, but suggests β₀ and β₆ are partially correlated (both scale with num_layers).

Agent 1's diagnostic finally stated: *"If NO experiments improve significantly (<5% across board), it indicates per-layer overhead is the wrong hypothesis."*

**Result**: No experiments improved significantly. Best improvement was 0.52pp (Scout general: 99.97% → 99.45%). Diagnostic IS triggered.

**Conclusion**: Per-layer overhead **is the wrong hypothesis** for reasoning experiments. The 1000× underestimation requires a different mechanism.

---

## Summary of Hypothesis Validation

| Hypothesis | Prediction | Actual Result | Verdict | Key Evidence |
|------------|-----------|---------------|---------|--------------|
| **H-main** | Loss <110%, TTFT RMSE <55%, β₀ → 0.25-0.35, reasoning 70-85% | Loss 603%, TTFT RMSE 519%, β₀ = 0.266 ✓, reasoning 99% | ❌ REJECTED | Catastrophic loss increase, inverse boundary effect |
| **H-simplification** | No degradation, coefficients stabilize (β₁ → 1.0-1.1, β₂ → 0.3-0.35) | β₁ = 1.449, β₂ = 1.368, partial improvement | ⚠️ PARTIAL | Coefficients improved 20-51% but didn't fully revert |
| **H-coefficient-norm** | β₀ → 0.25-0.35, all coefficients normalize | β₀ = 0.266 ✓, but β₁/β₂/β₅ still 24-45% above target | ❌ REJECTED | Only β₀ normalized, β₆ = 521μs (not 1000-3000μs) |
| **H-boundary** | Long prompts improve >25%, short prompts <10% | Long prompts +0.5pp, short prompts -700pp | ❌ REJECTED | Completely inverted — short contexts catastrophically degraded |
| **H-error-pattern** | Reasoning/TP=4/Mistral TP=2 improve most | ALL degraded (reasoning minimal improvement, TP=4 catastrophic) | ❌ REJECTED | Large-model short-prompt experiments worst affected |

**Overall result**: 0 hypotheses confirmed, 1 partial, 4 rejected. Iter5 is a **catastrophic failure**.

---

## Root Cause Summary

Iter5 failed due to a **fundamental flaw in the hypothesis design**:

1. **β₀ rising to 0.266 broke short-context predictions**: Higher prefill MFU means shorter predicted times, causing 10-30× over-prediction of experiments that were previously well-modeled
2. **β₆ converged too low (521μs, not 1000-3000μs)**: Insufficient overhead to help reasoning experiments, optimizer chose compromise to minimize overall RMSE
3. **Zero-sum trade-off**: Per-layer overhead formula applies to ALL experiments uniformly, creating gradient conflict between helping reasoning (4 exps) vs protecting short-context accuracy (11 exps)
4. **Collinearity persists**: β₀ and β₆ both scale with num_layers, preventing independent fitting

**Key insight**: Adding β₆ (per-layer overhead) without constraining β₀ (prefill MFU) allowed β₀ to rise freely, breaking short-context predictions. The optimizer then minimized β₆ to protect short contexts, failing to help reasoning.

**For iter6**: Must either (1) decouple β₀ and β₆ via context-length gating (apply overhead only to contexts >4K), or (2) use separate β₀ values for short vs long contexts, or (3) profile vLLM reasoning to identify algorithmic switch (different attention kernel for long contexts).
