# Iteration 5: Findings and Principles

## Summary

Iter5 tested the hypothesis that **per-layer fixed overhead** (kernel launch + scheduler + memory allocation) is the missing prefill term causing reasoning experiments' 1000× underestimation. The hypothesis was **catastrophically rejected**:

- Overall loss: **603.26%** (vs target <110%, 467% worse than iter4's 129%)
- TTFT RMSE: **518.85%** (vs target <55%, 8× worse than iter4's 66.49%)
- E2E RMSE: **84.41%** (vs stable 62-65%, 30% worse than expected)
- All 5 hypotheses **rejected or partial** (0 confirmed)

**Root cause**: Adding per-layer overhead β₆ without constraining prefill MFU β₀ caused β₀ to rise from 0.165 → 0.266 (61% increase), predicting 38% shorter prefill times for ALL experiments. This broke short-context predictions (300-1091% TTFT error). The optimizer then minimized β₆ to 521μs (far below expected 1000-3000μs) to protect short contexts, failing to help reasoning (only 0.5pp improvement from 99.75% → 99%).

**Critical insight**: The per-layer overhead mechanism is **plausibly correct** (β₀ rose to predicted 0.25-0.35 range), but the **functional form creates a zero-sum game** where helping reasoning experiments (4 exps, 27% of data) catastrophically degrades short-context experiments (11 exps, 73% of data). The hypothesis failed not because the physics is wrong, but because applying uniform overhead to all contexts violated the boundary condition that short contexts must remain well-predicted.

**Key learning**: Coefficients must be **context-dependent** (separate β₀_short vs β₀_long, or gated overhead terms) to model vLLM's qualitatively different behavior for short vs long contexts. Iter6 must profile vLLM reasoning to identify the actual algorithmic switch or memory pattern causing 1000× underestimation.

---

## Error Analysis

### Systematic Patterns

**By improvement/degradation from iter4**:
1. **Catastrophic degradation (>300pp worse)**: Short-context experiments (8 exps) — TTFT 4-77% → 300-1091%
2. **Moderate degradation (100-300pp worse)**: Medium-context experiments (3 exps) — TTFT 70-90% → 215-425%
3. **Marginal improvement (<1pp better)**: Reasoning experiments (4 exps) — TTFT 99.75-99.99% → 99.45-99.85%

**By workload category** (sorted by TTFT APE):
1. **Worst (>700% TTFT)**: Short-context dense models (Llama-3.1 TP=4 codegen: 1091%, Mistral codegen: 834%, Llama-2 roleplay: 822%, Qwen roleplay: 736%)
2. **Bad (300-700% TTFT)**: Medium-context dense models (Yi general-lite: 506%, Scout roleplay: 425%, Llama-2 general: 365%, Llama-3.1 TP=4 general-lite: 339%, Llama-2 codegen: 328%)
3. **Moderate (200-300% TTFT)**: Medium-context MoE/TP=2 (Scout codegen: 225%, Mistral TP=2 general-lite: 215%)
4. **Best (~99% TTFT)**: Reasoning experiments (Qwen/Scout/Llama-2 reasoning: 99.45-99.85%)

**Key observation**: Error **inversely correlates** with prompt length and **directly correlates** with model size (num_layers):
- Long prompts (>4K): ~99% TTFT (minimal improvement from iter4)
- Short prompts (<2K): 300-1091% TTFT (catastrophic degradation from iter4)
- Large models (70B, 80 layers): Worst degradation even for short prompts (1091% TTFT)
- Small models (7B, 32 layers): Moderate degradation (300-400% TTFT)

This inverse correlation indicates the **β₀ increase (0.165 → 0.266) dominated the per-layer overhead effect**, causing over-prediction proportional to num_layers × baseline_flops, with insufficient β₆ compensation.

### High-Error Experiments (APE > 700%)

**Llama-3.1-70B TP=4 codegen** (1091% TTFT, 60% E2E):
- **Context**: 1K prompt tokens, 80 layers, TP=4, decode-dominated workload
- **Iter4 baseline**: 3.86% TTFT (was BEST in training set)
- **Iter5 degradation**: +1087pp (282× worse)
- **Why it failed**:
  - β₀ increase: FLOPs / (peak × 0.266) vs FLOPs / (peak × 0.165) = 0.62× predicted time (38% reduction)
  - For 80 layers, 1K tokens, massive FLOP count → 38% reduction = ~200ms shorter prediction
  - β₆ overhead: 521 × 80 × 1.5 = 62.5ms added (only 31% of reduction)
  - Net: 200ms reduction - 62.5ms overhead = 137ms net over-prediction
  - Actual TTFT: ~150ms (from iter4 validation) → predicted ~13ms → 1091% error

**Mistral codegen** (834% TTFT, 63% E2E):
- **Context**: 1K prompt tokens, 40 layers, TP=1, decode-dominated
- **Iter4 baseline**: 30.69% TTFT
- **Iter5 degradation**: +803pp (27× worse)
- **Why it failed**: Same β₀ mechanism, but with 40 layers (not 80) → less severe but still catastrophic

**Pattern**: Short-context + large-model experiments were **best in iter4** (3-30% TTFT error), became **worst in iter5** (700-1000% TTFT error). This confirms β₀ increase broke well-modeled experiments.

### Low-Error Experiments (APE < 100%)

**Reasoning experiments** (99.45-99.85% TTFT, 98-99% E2E):
- **Context**: 8K-16K prompt tokens, 32-80 layers, prefill-dominated
- **Iter4 baseline**: 99.75-99.99% TTFT (catastrophically underestimated)
- **Iter5 improvement**: 0.14-0.52pp (marginal)
- **Why marginal improvement**:
  - β₀ increase: Reduced prediction by 38% (helps a little)
  - β₆ overhead: 521 × 80 × 5.0 = 208ms for 8K tokens, 80 layers
  - Actual underestimation: ~900ms (from iter4 profiling hypothesis)
  - β₆ captures only 23% of missing overhead (208ms / 900ms)
  - Net: Still massively underestimated (predicted ~100ms, actual ~1000ms)

**What makes reasoning "easy" to keep consistent** (not improve, just not degrade):
- Already at 99.75-99.99% TTFT baseline (floor effect — can't get much worse)
- Long contexts benefit from β₆ overhead scaling (5.0× factor for 8K tokens)
- Large num_layers amplifies β₆ contribution (80 layers × 521μs = 42ms base overhead)
- E2E error follows TTFT error (prefill-dominated workload)

**Pattern**: Reasoning experiments are **consistently bad** across iter3/iter4/iter5 (~99.7-99.9% TTFT), confirming the missing mechanism is orthogonal to activation bandwidth (iter4) and per-layer overhead (iter5).

### Error Correlations

**✅ Confirmed correlations**:

1. **Model size (num_layers) → degradation severity**: Larger models degraded more
   - 80-layer models (Llama-3.1-70B): 339-1091% TTFT for short contexts
   - 56-layer models (Scout): 225-425% TTFT for short contexts
   - 40-layer models (Mistral-Nemo): 215-834% TTFT for short contexts
   - 32-layer models (Llama-2, Qwen): 99-822% TTFT across all contexts
   - **Mechanism**: β₀ increase (0.165 → 0.266) affects predicted time ∝ FLOPs ∝ num_layers². Larger models have quadratically more FLOPs → 38% reduction = larger absolute time reduction → more over-prediction

2. **Prompt length → degree of error (inverse)**: Shorter prompts degraded more
   - <1K tokens: 736-1091% TTFT (catastrophic over-prediction)
   - 1-2K tokens: 328-834% TTFT (severe over-prediction)
   - 2-4K tokens: 215-506% TTFT (moderate over-prediction)
   - >4K tokens: 99% TTFT (slight under-prediction, unchanged from iter4)
   - **Mechanism**: β₆ overhead scales with 1.0 + tokens/2048, so short contexts get minimal overhead (1.25-2.0×) while β₀ reduction applies uniformly (38% for all)

3. **Decode-dominated workloads → E2E resilience**: E2E errors stayed moderate even when TTFT catastrophic
   - Llama-3.1-70B codegen: 1091% TTFT but only 60% E2E
   - Mistral codegen: 834% TTFT but only 63% E2E
   - Llama-2 roleplay: 822% TTFT but only 66% E2E
   - **Mechanism**: E2E dominated by decode time (well-modeled by β₁/β₄), so TTFT over-prediction has limited impact on E2E

**❌ Rejected correlations**:

1. **TP degree → error**: TP=1/TP=2/TP=4 experiments show similar error patterns (no TP-specific trend)
   - TP=1: 99-834% TTFT (wide range, context-dependent)
   - TP=2: 215-822% TTFT (wide range, context-dependent)
   - TP=4: 339-1091% TTFT (wide range, but both are Llama-3.1-70B → model size effect, not TP effect)
   - **Conclusion**: TP communication β₂ is stable (1.368, unchanged from iter4), not the error source

2. **MoE architecture → error**: Scout experiments don't show systematically different errors than dense models
   - Scout short contexts: 225-425% TTFT (similar to Mistral TP=2: 215%)
   - Scout reasoning: 99.45-99.66% TTFT (similar to dense reasoning: 99.76-99.85%)
   - **Conclusion**: MoE gating β₅ improved (0.0304 → 0.0149), moving toward stability. Scout errors are context-length dependent, not architecture-specific

3. **Workload type → error** (once controlled for context length): Codegen, roleplay, general-lite have similar errors for same context length
   - 1K-2K codegen: 225-1091% TTFT
   - 1K-2K roleplay: 425-822% TTFT
   - 2K-4K general-lite: 215-506% TTFT
   - **Conclusion**: Workload-agnostic property holds (errors driven by context length and model size, not workload patterns)

### Root Cause Hypotheses

Based on the catastrophic error patterns, five root causes emerge:

#### **Principle 1**: β₀ and β₆ must be context-dependent, not uniform coefficients

**Evidence**:
- β₀ rose from 0.165 → 0.266 (within predicted 0.25-0.35 range ✓)
- But this **broke short-context predictions**: 4-77% TTFT → 300-1091% TTFT
- β₆ converged to 521μs (far below expected 1000-3000μs)
- Reasoning experiments barely improved: 99.75-99.99% → 99.45-99.85% TTFT (0.5pp average)
- Large-model short-context experiments degraded 10-30× (worst: Llama-3.1-70B codegen +1087pp)

**Mechanism**:

The current model assumes **uniform coefficients** across all contexts:
```
prefill_time = FLOPs / (peak_TFLOPS × β₀) + β₆ × num_layers × (1.0 + tokens/2048)
```

This creates a **zero-sum gradient conflict**:
1. Increasing β₀ helps reasoning (reduces predicted time for long contexts where actual time is ~1000ms)
2. But increasing β₀ also reduces predicted time for SHORT contexts (where actual time is ~50-150ms)
3. Short contexts were already well-predicted in iter4 (4-77% TTFT) → β₀ increase over-predicts them
4. Optimizer must choose: help reasoning (4 exps) OR protect short contexts (11 exps)
5. Optimizer chose compromise: β₀ = 0.266 (moderate increase), β₆ = 521μs (low overhead)
6. Result: Short contexts catastrophically over-predicted, reasoning barely improved

**Why uniform coefficients fail**:
- vLLM has **qualitatively different prefill behavior** for short vs long contexts:
  - Short contexts (<2K): Compute-bound, achieves ~40-50% MFU, minimal scheduler overhead
  - Long contexts (>4K): Memory-bound OR algorithm-switched, achieves ~15-25% MFU, high scheduler overhead OR different attention kernel
- Single β₀ cannot represent both regimes (forced to 0.266 = bad compromise)
- Single β₆ overhead applies to ALL contexts, but overhead may only exist for long contexts

**Action for iter6**:
- **Option 1: Context-gated coefficients**:
  ```
  if num_prefill_tokens > 4096:
      β₀_long = 0.15-0.25  # Low MFU for long contexts
      β₆_long = 1000-3000μs  # High per-layer overhead
  else:
      β₀_short = 0.40-0.55  # High MFU for short contexts
      β₆_short = 0μs  # No per-layer overhead
  ```
- **Option 2: Continuous context-length scaling** (avoids discontinuity at 4K):
  ```
  context_scale = min(1.0, num_prefill_tokens / 4096.0)
  β₀_effective = β₀_short × (1 - context_scale) + β₀_long × context_scale
  β₆_effective = β₆_long × context_scale  # Only apply to long contexts
  ```
- **Option 3: Profile vLLM** to confirm algorithmic switch at >4K or >8K tokens (different FlashAttention kernel? PagedAttention chunking threshold?)

**Expected outcome**: With context-dependent coefficients:
- Short contexts: β₀ = 0.45, β₆ = 0 → predicted time matches iter4 baseline (4-77% TTFT)
- Long contexts: β₀ = 0.20, β₆ = 2000μs → reasoning improves from 99% → 60-80% TTFT (if per-layer overhead is correct mechanism)
- Overall loss: 603% → 90-110% (recovering iter4 short-context accuracy + gaining reasoning improvement)

---

#### **Principle 2**: Per-layer overhead is plausibly correct for long contexts, but wrong functional form

**Evidence from H-main validation**:
- β₀ rose to 0.266 (within predicted 0.25-0.35 ✓) — confirms overhead term allows β₀ to rise
- β₆ converged to 521μs (expected 1000-3000μs) — optimizer minimized to protect short contexts
- Reasoning improved marginally (0.5pp average) — insufficient overhead captured
- Formula: `overhead_us = β₆ × num_layers × (1.0 + num_prefill_tokens / 2048.0)`

**Evidence from H-boundary validation**:
- Short prompts degraded catastrophically (overhead applied when shouldn't be)
- Long prompts improved marginally (overhead too small)
- β₆ = 521μs means reasoning gets 208ms overhead (need ~900ms to close gap)

**Mechanism**:

The per-layer overhead hypothesis is **directionally correct** (β₀ rose as predicted), but **quantitatively wrong**:

1. **Base factor (1.0) applies overhead to ALL contexts**:
   - For short prompts (512 tokens): prefill_scale_factor = 1.0 + 512/2048 = 1.25
   - For long prompts (8K tokens): prefill_scale_factor = 1.0 + 8192/2048 = 5.0
   - The 1.0 base means short prompts get 1.25× overhead → 20-60ms added → over-prediction
   - Should be: `prefill_scale_factor = max(0, (num_prefill_tokens - 2048) / 2048.0)` (no overhead below 2K tokens)

2. **Linear scaling may be wrong functional form**:
   - Current: `overhead ∝ (1.0 + tokens/2048)` — linear in tokens
   - Alternative 1: `overhead ∝ (tokens/2048)²` — quadratic in tokens (attention O(n²) memory)
   - Alternative 2: `overhead ∝ log(tokens/2048)` — logarithmic (scheduler batching)
   - Alternative 3: `overhead = constant` for tokens > 8K (algorithmic switch, fixed cost)

3. **Chunking threshold may be wrong**:
   - Current: Assumed 2048-token chunks (vLLM default `--max-model-len`)
   - Reality: vLLM may use 512, 1024, or 4096 tokens depending on batch size and GPU memory
   - Profile vLLM to confirm actual chunking behavior

**Why β₆ converged low**:
- Optimizer penalized by 11 short-context experiments (73% of data) when overhead applied
- Had to minimize β₆ to reduce damage to short contexts
- Even β₆ = 521μs caused 300-1000% degradation → optimizer wanted to go lower but hit plausibility bound

**Action for iter6**:
1. **Remove base factor (1.0)**: `prefill_scale_factor = max(0, (num_prefill_tokens - 2048) / 2048.0)`
   - No overhead for contexts <2K tokens (protects short contexts)
   - Linear scaling above 2K tokens (overhead grows with chunking)
2. **Test alternative functional forms**: Quadratic, logarithmic, or constant overhead
3. **Profile vLLM chunking**: Confirm actual chunking threshold and chunk size
4. **Decouple from β₀**: Add constraint that β₀ cannot exceed 0.35 when β₆ is active (prevent β₀ rising too high)

**Expected outcome**: With base factor removed:
- Short contexts: No β₆ overhead applied → avoid degradation
- Long contexts: β₆ can converge to 1500-2500μs (optimizer no longer penalized by short contexts)
- Reasoning: overhead = 2000 × 80 × 4.0 = 640ms (captures 71% of 900ms gap) → 99% → 70-85% TTFT

---

#### **Principle 3**: Removing activation bandwidth partially helped, but collinearity persists

**Evidence from H-simplification validation**:
- β₁ improved by 20% (1.802 → 1.449), but still 32-45% above target 1.00-1.10
- β₂ unchanged (1.360 → 1.368), far above target 0.30-0.35
- β₅ improved by 51% (0.0304 → 0.0149), but still 24-49% above target 0.01-0.012
- β₃ collapsed 97% (0.000495 → 0.000013), near-zero
- β₄ dropped 32% (0.918 → 0.620), below target 0.75-0.85
- Convergence improved (185 → 78 trials), confirming reduced parameter space helps

**Mechanism**:

Removing iter4's β₆ (activation bandwidth) **partially reduced collinearity**, but **new β₆ (per-layer overhead) re-introduced collinearity**:

1. **Why coefficients improved but didn't revert**:
   - Iter4's β₆ (activation BW) was completely misspecified (converged to 1.818, expected 3.0-6.0)
   - Removing it eliminated one source of gradient masking
   - β₁, β₅ improved by 20-51% toward iter3 values
   - BUT: New β₆ (per-layer overhead) also scales with num_layers, creating new collinearity with β₀
   - β₂ unchanged (1.360 → 1.368) suggests new β₆ is absorbing error correlated with TP configs

2. **Why β₃ collapsed**:
   - β₃ (KV cache management) was 0.000495 in iter4 (0.5ms per request overhead)
   - Collapsed to 0.000013 in iter5 (0.013ms per request, negligible)
   - **Mechanism**: β₀ increased 61% → predicted prefill time decreased 38% → less time for KV management overhead
   - β₃ was artificially absorbing prefill overhead that should be in β₀
   - When β₀ rose, β₃ became redundant

3. **Why β₄ dropped**:
   - β₄ (decode compute-bound MFU) was 0.918 in iter4 (91.8% of theoretical time)
   - Dropped to 0.620 in iter5 (62% of theoretical time, physically implausible)
   - **Mechanism**: β₁ (decode memory-bound) still too high at 1.449 → absorbs more decode time
   - Optimizer balances β₁ vs β₄ → if β₁ high, β₄ must be low to match actual decode latency
   - β₄ = 0.620 is implausibly low (decode compute cannot be 38% faster than theoretical)

**Why β₂ didn't normalize**:
- β₂ (TP decode communication) stayed at 1.360-1.368 across iter4/iter5
- Unchanged despite removing activation BW (iter4's β₆) and adding per-layer overhead (iter5's β₆)
- **Hypothesis**: β₂ is absorbing error from a **missing TP-dependent prefill term**
- Evidence: Mistral TP=2 (215% TTFT), Llama-3.1 TP=4 (339-1091% TTFT) both have high errors
- TP communication β₂ should only affect DECODE, but it's at 1.36× theoretical (36% overhead)
- Physically implausible for NVLink all-reduce to have 36% overhead (should be <10%)

**Action for iter6**:
1. **Decouple β₀ and β₆**: Add constraint or use context gating (see Principle 1)
2. **Investigate β₂**: Profile TP=2/TP=4 experiments to confirm decode communication is well-modeled
3. **Check for missing TP prefill term**: TP may increase PREFILL overhead via:
   - All-reduce of activations during prefill (forward pass communication)
   - Larger batch formations (scheduler overhead for TP=4 may be higher)
   - Memory allocation overhead (allocating KV cache blocks across 4 GPUs)

**Expected outcome**: With collinearity reduced:
- β₁ → 1.00-1.10 (revert to iter3 physical plausibility)
- β₂ → 0.30-0.35 OR identify new TP prefill term
- β₃ → 0.0004-0.0005 (recover KV management overhead)
- β₄ → 0.75-0.85 (recover decode compute plausibility)
- β₅ → 0.01-0.012 (complete reversion to iter3)

---

#### **Principle 4**: Reasoning experiments require a different mechanism (not per-layer overhead)

**Evidence from H-main, H-boundary, H-error-pattern**:
- Reasoning improved by only 0.14-0.52pp (99.75-99.99% → 99.45-99.85% TTFT)
- 1000× underestimation persists (predicted ~100ms, actual ~1000ms)
- β₆ converged to 521μs (far below expected 1000-3000μs)
- No meaningful progress in 3 iterations (iter3: 99.71%, iter4: 99.85%, iter5: 99.65% average)

**Mechanism**:

Three iterations have conclusively ruled out **memory bandwidth bottlenecks** for reasoning:
- Iter4: Activation write bandwidth (β₆ = 1.818, expected 3.0-6.0) → 0% improvement
- Iter5: Per-layer overhead (β₆ = 521μs, expected 1000-3000μs) → 0.5pp improvement
- Both hypotheses predicted >25pp improvement (from 99.75% → 70-75% TTFT) → failed

**Physical impossibility**: 1000× underestimation cannot be explained by continuous bottlenecks:
- Memory bandwidth (HBM): max 3-5× slowdown (limited by 3.35 TB/s on H100)
- Compute throughput: max 2-3× slowdown (limited by MFU 40-55%)
- Communication (NVLink): max 2× slowdown (limited by 900 GB/s)

**A 1000× slowdown requires**:
1. **Algorithmic switch**: vLLM uses different attention kernel for contexts >8K tokens
   - FlashAttention-2 (short contexts) vs PagedAttention with CPU offload (long contexts)
   - Or: Different chunking strategy (smaller chunks for long contexts → more overhead)
   - Or: Attention computation switches from O(n²) tiled to O(n²) full materialization
2. **Quadratic memory overhead**: Attention working set ∝ (tokens)² for contexts >4K
   - Current model: prefill FLOPs ∝ tokens × num_layers (linear in tokens)
   - Reality: For n > 4K, attention may materialize full attention matrix (n² memory)
   - Memory bandwidth time ∝ n² → explains 1000× slowdown for n=8K (64× from n² alone)
3. **Scheduler batching**: Reasoning requests batched differently (larger batches → longer wait times)
   - vLLM may prioritize short requests over long requests (starvation)
   - Or: Reasoning prompts trigger KV cache eviction (swapping blocks to CPU)
   - Or: Prefix cache misses (reasoning prompts don't share prefixes with other requests)
4. **Kernel launch overhead**: Fixed cost per layer per chunk (already tested — β₆ = 521μs insufficient)

**Why per-layer overhead failed** (from Principle 2):
- β₆ = 521μs gives 208ms overhead for reasoning (8K tokens, 80 layers)
- Need ~900ms to close gap → β₆ should be 2167μs (4.2× higher)
- But optimizer chose 521μs to protect short contexts from degradation
- This suggests per-layer overhead is **a component** but **not the dominant bottleneck**

**Action for iter6** (CRITICAL — must profile before next iteration):
1. **Profile vLLM reasoning experiments**:
   ```bash
   nsys profile -o reasoning_profile vllm serve --model meta-llama/Llama-2-7b-hf --tensor-parallel-size 1
   # Send 8K-token reasoning prompt, capture timeline
   ```
2. **Measure**:
   - Kernel launch count and timing distribution (is it 50μs × 200 kernels × 80 layers?)
   - Attention kernel type for long contexts (FlashAttention vs PagedAttention?)
   - Memory allocation overhead (KV cache blocks, attention buffers)
   - Scheduler queue depth and batch formation latency
   - CPU-GPU memory transfers (KV cache offload?)
3. **Identify dominant bottleneck**:
   - If kernel launch: Add kernel-specific β_kernel term (may need >2000μs per layer)
   - If algorithmic switch: Add context-gated β₀ (β₀_short vs β₀_long)
   - If O(n²) attention: Add quadratic memory term (bandwidth ∝ tokens²)
   - If scheduler batching: Add queuing term (wait time before prefill starts)

**Expected outcome**: Profiling will reveal ONE of four mechanisms dominates (likely algorithmic switch or O(n²) memory), enabling targeted hypothesis for iter6.

---

#### **Principle 5**: Warm-starting from iter3 (not iter4) partially worked, but needs refinement

**Evidence from coefficient analysis**:
- Iter5 was warm-started from iter3 (skipping iter4's destabilized coefficients)
- β₁, β₅ moved toward iter3 values (partial reversion):
  - β₁: 1.802 (iter4) → 1.449 (iter5) → target 1.037 (iter3), 53% of way back
  - β₅: 0.0304 (iter4) → 0.0149 (iter5) → target 0.0117 (iter3), 81% of way back
- β₂ unchanged (1.360 → 1.368), suggesting collinearity with new β₆
- Convergence faster (185 → 78 trials), confirming good initialization helps

**Mechanism**:

Warm-starting from iter3 was **directionally correct** (avoid iter4's coefficient drift), but **insufficient** because:

1. **New β₆ (per-layer overhead) has similar collinearity to old β₆ (activation BW)**:
   - Both scale with num_layers
   - Both create gradient masking with β₀
   - Warm-starting from iter3 doesn't prevent collinearity in iter5

2. **β₀ should have been CONSTRAINED, not warm-started**:
   - Iter3 β₀ = 0.169 (good starting point)
   - But iter5 had no upper bound on β₀ → rose to 0.266 (broke short contexts)
   - Should have: β₀ bounds [0.15, 0.25] with β₆ active, [0.15, 0.55] with β₆ = 0
   - Prevents β₀ rising too high while per-layer overhead is being fitted

3. **Convergence at 78 trials suggests premature stopping**:
   - Iter4: 185 trials (did not converge early)
   - Iter5: 78 trials (converged early flag set)
   - But coefficients didn't fully stabilize (β₁/β₂/β₅ still 24-45% above target)
   - Early convergence may indicate optimizer got stuck in local minimum
   - Should have: Run 150-200 trials minimum to allow full exploration

**Action for iter6**:
1. **Warm-start from iter5's β₁/β₄/β₅** (improved from iter4, closer to iter3):
   - β₁: 1.449 → target 1.00-1.10 (closer than iter4's 1.802)
   - β₄: 0.620 → target 0.75-0.85 (needs to rise)
   - β₅: 0.0149 → target 0.01-0.012 (close!)
2. **Constrain β₀ based on context regime**:
   - If context-gated: β₀_short bounds [0.35, 0.55], β₀_long bounds [0.15, 0.30]
   - If uniform: β₀ bounds [0.15, 0.35] (prevent rising above physical range)
3. **Increase minimum trials**: Set n_trials = 200 (no early stopping until 150+ trials)

**Expected outcome**: With constrained β₀ and more trials, coefficients should fully stabilize to iter3 ranges by iter6 or iter7.

---

## Coefficient Analysis

**Alpha [α₀, α₁, α₂]** (request-level overheads):
- α₀ = 0.00333 ms = **3.33 ms** fixed API overhead per request (was 1.50 ms in iter4, +122%)
- α₁ = 0.000371 ms/token = **371 μs** per input token (was 125 μs in iter4, +197%)
- α₂ = 0.000381 ms/token = **381 μs** per output token (was 36 μs in iter4, +960%)

**Trend**: Alpha coefficients **exploded** in iter5, absorbing error from catastrophic TTFT over-prediction.

**Physical interpretation**:
- α₀ = 3.33 ms: Possible but high for API overhead (JSON parsing, validation)
- α₁ = 371 μs/token: **Physically implausible** (10× too high for tokenization)
- α₂ = 381 μs/token: **Physically implausible** (10× too high for detokenization)

**Why Alpha coefficients exploded**:
- Short-context TTFT over-predicted by 10-30× (predicted ~10ms, actual ~100-300ms)
- Optimizer compensated by inflating α₁/α₂ to add per-token overhead
- This "patches" TTFT by adding ~50-100ms of request-level overhead
- But breaks E2E predictions (E2E error increased from 62% → 84%)

**Outliers**: α₁ and α₂ are **severely outliers** (10× above physical range).

**Action for iter6**: Do NOT warm-start from iter5's Alpha coefficients. Revert to iter3 or iter4 Alpha values.

---

**Beta [β₀, β₁, β₂, β₃, β₄, β₅, β₆]** (step-level basis functions):

**β₀ = 0.2663** (prefill compute MFU, up 61% from 0.1654 in iter4)
- **Physical interpretation**: 26.6% MFU during prefill (was 16.5% in iter4)
- **Trend**: Rose to predicted range 0.25-0.35 ✓ (hypothesis correctly predicted this)
- **Problem**: Rise broke short-context predictions (38% reduction in predicted time)
- **Status**: In physical range BUT caused catastrophic side effects
- **Action for iter6**: Constrain to [0.15, 0.25] with per-layer overhead active, OR use context-gated β₀_short / β₀_long

**β₁ = 1.4493** (decode memory-bound MFU, down 20% from 1.8016 in iter4)
- **Physical interpretation**: 1.45× theoretical memory-bound time (45% slower than HBM bandwidth allows)
- **Trend**: Improved from 1.802 → 1.449 (moving toward iter3's 1.037)
- **Problem**: Still 32-45% above physical range 1.00-1.10
- **Status**: ⚠️ Partial recovery, needs further reduction
- **Action for iter6**: Continue warm-starting from 1.449, should converge to 1.00-1.10 with collinearity fixes

**β₂ = 1.3682** (TP decode communication, unchanged from 1.3598 in iter4)
- **Physical interpretation**: 1.37× theoretical all-reduce time (37% slower than NVLink allows)
- **Trend**: Essentially unchanged (1.360 → 1.368, 0.6% increase)
- **Problem**: Far above physical range 0.30-0.35, suggests missing TP prefill term OR collinearity
- **Status**: ❌ No progress, likely absorbing error from new β₆
- **Action for iter6**: Profile TP=2/TP=4 prefill to confirm no TP-dependent overhead exists. If β₂ remains high, may indicate fundamental model misspecification.

**β₃ = 0.00001309** (KV cache management, down 97% from 0.0004953 in iter4)
- **Physical interpretation**: 0.013 ms per request (was 0.495 ms in iter4, now negligible)
- **Trend**: Collapsed to near-zero
- **Problem**: KV cache management should cost ~0.4-0.5 ms per request (iter3/iter4 baseline), not 0.01 ms
- **Status**: ❌ Collapsed due to β₀ increase absorbing prefill overhead
- **Action for iter6**: Should recover to 0.0004-0.0005 when β₀ is constrained (prefill overhead no longer artificially absorbed)

**β₄ = 0.6199** (decode compute-bound MFU, down 32% from 0.9182 in iter4)
- **Physical interpretation**: 62% of theoretical compute time (decode compute is 38% FASTER than theoretical)
- **Trend**: Dropped from 0.918 → 0.620 (below physical range 0.75-0.85)
- **Problem**: Physically implausible (tensor cores cannot exceed theoretical TFLOPS)
- **Status**: ❌ Implausibly low, balancing β₁ being too high
- **Action for iter6**: Should rise to 0.75-0.85 when β₁ normalizes to 1.00-1.10

**β₅ = 0.01485** (MoE gating overhead, down 51% from 0.0304 in iter4)
- **Physical interpretation**: 14.85 ms per step for expert routing (was 30.4 ms in iter4)
- **Trend**: Improved from 0.0304 → 0.0149 (moving toward iter3's 0.0117)
- **Problem**: Still 24-49% above physical range 0.01-0.012
- **Status**: ⚠️ Partial recovery (81% of way back to iter3), close to target
- **Action for iter6**: Continue warm-starting from 0.0149, should converge to 0.01-0.012 with collinearity fixes

**β₆ = 521.53 μs** (NEW: per-layer prefill overhead, expected 1000-3000 μs)
- **Physical interpretation**: 521.5 μs per layer per chunk-equivalent overhead (kernel launch + scheduler + memory allocation)
- **Trend**: First iteration for this term
- **Problem**: Converged 48-83% below expected range (optimizer minimized to protect short contexts)
- **Status**: ❌ Too low to help reasoning (need 2000-3000 μs)
- **Physical plausibility**: 521 μs per layer is plausible for kernel launch alone (10-20 kernels × 50 μs), but insufficient to explain 1000× underestimation
- **Action for iter6**: Remove base factor (1.0) from formula, allow β₆ to converge to 1500-2500 μs without penalizing short contexts

---

**Redundant terms**: β₃ collapsed to near-zero (0.000013), but this is likely a side effect of β₀ increase, not true redundancy. Expect β₃ to recover when β₀ is constrained.

**Missing physics**:
1. **Context-gated coefficients**: β₀_short vs β₀_long (short contexts need ~0.45, long contexts need ~0.20)
2. **Algorithmic switch for reasoning**: Different attention kernel or memory pattern for contexts >8K
3. **TP prefill overhead**: β₂ staying at 1.37 suggests missing TP-dependent term for prefill (not just decode)
4. **Quadratic attention memory**: Attention working set ∝ tokens² for large contexts (not captured by linear FLOPs)

---

## Recommendations for iter6

### Priority 1: Critical Issues (MUST address before iter6 optimization)

**1.1 Profile vLLM reasoning experiments to identify actual bottleneck — BLOCKING**
- **Rationale**: Three iterations (iter3/iter4/iter5) failed to improve reasoning (99.7-99.9% TTFT consistently)
- **Action**: Run `nsys profile` on Llama-2-7B reasoning (8K tokens) and Qwen reasoning (16K tokens)
- **Measure**:
  - Kernel launch count and timing distribution (hypothesis: 50μs × 200 kernels × 80 layers = 800ms)
  - Attention kernel type (FlashAttention-2 vs PagedAttention vs other?)
  - Memory allocation overhead (KV cache blocks, attention buffers, CPU offload?)
  - Scheduler queue depth and batch formation latency
  - CPU-GPU memory transfers (KV cache swapping to CPU?)
- **Expected outcome**: Identify ONE of four mechanisms:
  1. Kernel launch overhead (confirms per-layer hypothesis, but need higher β₆)
  2. Algorithmic switch (different kernel for contexts >8K → context-gated β₀)
  3. O(n²) attention memory (quadratic bandwidth term needed)
  4. Scheduler batching overhead (queuing term needed before prefill)
- **Time estimate**: 2-4 hours for profiling + analysis
- **Blocking**: DO NOT proceed to iter6 hypothesis design without profiling results

**1.2 Implement context-gated coefficients to prevent short-context degradation — ARCHITECTURE**
- **Rationale**: Uniform β₀ and β₆ create zero-sum game (helping reasoning hurts short contexts)
- **Action**: Add context-length threshold (4096 tokens) for coefficient selection:
  ```go
  if num_prefill_tokens <= 4096 {
      // Short-context regime: high MFU, no per-layer overhead
      prefill_mfu = Beta[0]  // β₀_short, bounds [0.35, 0.55]
      per_layer_overhead_us = 0.0
  } else {
      // Long-context regime: low MFU, high per-layer overhead
      prefill_mfu = Beta[6]  // β₀_long, bounds [0.15, 0.30]
      chunks = 1.0 + float64(num_prefill_tokens - 4096) / 2048.0
      per_layer_overhead_us = Beta[7] * num_layers * chunks  // β₆_long
  }
  ```
- **Coefficient mapping**:
  - β₀: Short-context prefill MFU (0.35-0.55)
  - β₁-β₅: Unchanged (decode terms, KV management, MoE gating)
  - β₆: Long-context prefill MFU (0.15-0.30)
  - β₇: Long-context per-layer overhead (1000-3000 μs)
- **Total parameters**: 10 (3 alpha + 7 beta, +1 from iter5's 6 beta)
- **Expected outcome**: Short contexts recover to iter4 baseline (4-77% TTFT), reasoning can improve independently

**1.3 Remove base factor (1.0) from per-layer overhead formula — BUG FIX**
- **Rationale**: Current formula applies overhead to ALL contexts, causing short-context over-prediction
- **Action**: Change `prefill_scale_factor = 1.0 + tokens/2048` to `max(0, (tokens - 4096) / 2048)`
- **Effect**: No overhead for contexts <4K tokens, linear scaling above 4K
- **Combine with 1.2**: Use context gating (4096 threshold) for both β₀ and β₆
- **Expected outcome**: Short contexts get zero overhead, long contexts get full overhead (β₆ can converge to 2000-3000 μs)

### Priority 2: Improvements (address coefficient drift)

**2.1 Revert Alpha coefficients to iter4 values — IMMEDIATE**
- **Rationale**: Alpha exploded to absorb TTFT over-prediction error (α₁ = 371 μs, α₂ = 381 μs, physically implausible)
- **Action**: Warm-start from iter4 Alpha values:
  - α₀ = 0.001498 ms (1.5 ms fixed overhead)
  - α₁ = 0.0001247 ms/token (124.7 μs per input token)
  - α₂ = 0.00003599 ms/token (36.0 μs per output token)
- **DO NOT warm-start from iter5** Alpha (will propagate error)
- **Expected outcome**: Request-level overhead returns to physical plausibility

**2.2 Constrain β₀ bounds based on context regime — DEFENSIVE**
- **Rationale**: β₀ rising to 0.266 broke short contexts (even though 0.266 is physically plausible for short contexts)
- **Action**: Split β₀ bounds by context length:
  - β₀_short (contexts <4K): bounds [0.35, 0.55], warm-start 0.45
  - β₀_long (contexts >4K): bounds [0.15, 0.30], warm-start 0.20
- **Prevents**: β₀_short cannot drop below 0.35 (maintains short-context accuracy)
- **Allows**: β₀_long can be low (captures long-context degraded MFU)
- **Expected outcome**: Short contexts predicted correctly (4-77% TTFT), reasoning can be fitted independently

**2.3 Increase minimum trials to 200 (disable early stopping before 150 trials) — CONVERGENCE**
- **Rationale**: Iter5 converged at 78 trials, but coefficients didn't fully stabilize (β₁/β₂/β₅ still 24-45% above target)
- **Action**: Set Optuna n_trials = 200, early stopping patience = 50 (don't stop before 150 trials)
- **Expected outcome**: Full exploration of parameter space, coefficients should stabilize to iter3 ranges

**2.4 Warm-start Beta from iter5 (except β₀/β₆, which become β₀_short/β₀_long/β₆_long) — STABILITY**
- **Action**: Use iter5 Beta coefficients as starting point for iter6:
  - β₁: 1.449 (improved from iter4, closer to target 1.00-1.10)
  - β₂: 1.368 (unchanged, monitor for missing TP prefill term)
  - β₃: 0.0005 (use iter4 value, not iter5's collapsed 0.000013)
  - β₄: 0.796 (use iter3 value, not iter5's implausible 0.620)
  - β₅: 0.0149 (improved from iter4, close to target 0.01-0.012)
- **For new context-gated coefficients**:
  - β₀_short: 0.45 (physical MFU for short contexts)
  - β₀_long: 0.20 (degraded MFU for long contexts, or from profiling results)
  - β₆_long: 2000 μs (per-layer overhead, or from profiling results)
- **Expected outcome**: Coefficients continue gradual convergence toward iter3 stability

### Priority 3: Refinements (post-profiling actions)

**3.1 If profiling confirms algorithmic switch → Add context-switched β₀ (HYPOTHESIS)**
- **If profiling shows**: vLLM uses different attention kernel for contexts >8K tokens
- **Action**: Implement context-switched β₀ (already covered by Priority 1.2)
- **Formula**: `β₀_effective = β₀_short if tokens ≤ 4K else β₀_long`
- **Expected outcome**: Reasoning improves from 99% → 60-80% TTFT (if algorithmic switch is bottleneck)

**3.2 If profiling confirms O(n²) attention memory → Add quadratic memory term (HYPOTHESIS)**
- **If profiling shows**: Attention kernel materializes full attention matrix for contexts >4K (O(n²) memory)
- **Action**: Add quadratic memory bandwidth term:
  ```go
  if num_prefill_tokens > 4096 {
      attention_memory_bytes = num_prefill_tokens^2 * num_heads * bytes_per_element
      attention_bandwidth_us = Beta[8] * attention_memory_bytes / memory_bandwidth_gbps
      prefill_time_us += attention_bandwidth_us
  }
  ```
- **Coefficient**: β₈ = 3.0-6.0 (memory bandwidth scale factor, similar to iter4's activation BW hypothesis)
- **Expected outcome**: Reasoning improves from 99% → 50-70% TTFT (if O(n²) memory is bottleneck)

**3.3 If profiling confirms kernel launch overhead → Increase β₆_long bounds (REFINEMENT)**
- **If profiling shows**: Kernel launch overhead dominates (800-1000ms for 80 layers × 10-20 kernels × 50μs)
- **Action**: Keep context-gated β₆_long, but expand bounds:
  - Current: β₆_long bounds [1000, 3000] μs
  - New: β₆_long bounds [1000, 5000] μs (allow higher overhead if profiling confirms)
- **Expected outcome**: β₆_long converges to 2500-4000 μs, reasoning improves from 99% → 60-80% TTFT

**3.4 Investigate β₂ (TP communication) staying at 1.37 → Profile TP prefill (EXPLORATION)**
- **Rationale**: β₂ unchanged at 1.360-1.368 across iter4/iter5, far above physical range 0.30-0.35
- **Action**: Profile Mistral TP=2 and Llama-3.1 TP=4 experiments to confirm:
  - Decode all-reduce time matches β₂ × theoretical_time
  - No TP-dependent prefill overhead exists (all-reduce during forward pass?)
- **If TP prefill overhead found**: Add β_tp_prefill = TP_degree × num_layers × overhead_us
- **If not found**: β₂ may be absorbing error from other misspecifications (wait until β₁/β₀ stabilize)

---

## Basis Function Changes for Iter6

**Remove**:
- Nothing (all 6 beta terms from iter5 are plausibly useful, though β₃ collapsed temporarily)

**Add**:
- **β₀_short** (short-context prefill MFU): Replaces single β₀, applies to contexts <4K tokens
- **β₀_long** (long-context prefill MFU): Replaces single β₀, applies to contexts >4K tokens
- **β₆_long** (long-context per-layer overhead): Replaces single β₆, applies only to contexts >4K tokens with base factor removed
- **Optionally β₈** (if profiling finds O(n²) attention memory or other mechanism): New term based on profiling results

**Keep**:
- β₁: Decode memory-bound MFU
- β₂: TP decode communication
- β₃: KV cache management (expect recovery from 0.000013 → 0.0005 when β₀ constrained)
- β₄: Decode compute-bound MFU (expect recovery from 0.620 → 0.80 when β₁ normalizes)
- β₅: MoE gating overhead

**Total parameters**: 10-11 (3 alpha + 7-8 beta, depending on profiling results)

**Bounds for new coefficients**:
- β₀_short: [0.35, 0.55], initial 0.45 (physical MFU for short contexts)
- β₀_long: [0.15, 0.30], initial 0.20 (degraded MFU for long contexts, adjust after profiling)
- β₆_long: [1000, 5000] μs, initial 2000 μs (per-layer overhead, adjust after profiling)
- β₈ (if added): [0.0, 10.0], initial 3.0 (scale factor, based on profiling)

**Bounds for existing coefficients** (unchanged from iter5):
- β₁: [0.8, 2.0], initial 1.449
- β₂: [0.0, 1.5], initial 1.368
- β₃: [0.0, 0.01], initial 0.0005 (revert to iter4, not iter5's collapsed value)
- β₄: [0.4, 1.5], initial 0.796 (revert to iter3, not iter5's implausible value)
- β₅: [0.0, 0.05], initial 0.0149

---

## Bounds Adjustments for Iter6

**Alpha coefficients** (revert to iter4, DO NOT use iter5):
- α₀: [0.0, 0.01] ms, initial 0.001498 (1.5 ms fixed overhead)
- α₁: [0.0, 0.001] ms/token, initial 0.0001247 (124.7 μs per input token)
- α₂: [0.0, 0.001] ms/token, initial 0.00003599 (36.0 μs per output token)

**Beta coefficients** (context-gated + warm-started from iter5 where appropriate):
- β₀_short: [0.35, 0.55], initial 0.45
- β₁: [0.8, 2.0], initial 1.449 (from iter5)
- β₂: [0.0, 1.5], initial 1.368 (from iter5, monitor for missing TP prefill term)
- β₃: [0.0, 0.01], initial 0.0005 (from iter4, expect recovery)
- β₄: [0.4, 1.5], initial 0.796 (from iter3, expect to stay stable)
- β₅: [0.0, 0.05], initial 0.0149 (from iter5, close to target)
- β₀_long: [0.15, 0.30], initial 0.20 (adjust after profiling)
- β₆_long: [1000, 5000] μs, initial 2000 μs (adjust after profiling)
- β₈ (if added): [0.0, 10.0], initial 3.0 (from profiling)

**Rationale**:
- Context gating prevents β₀ rising too high for short contexts (locked at 0.35-0.55)
- β₀_long can be low (0.15-0.30) to model long-context degraded MFU
- β₆_long can be high (2000-5000 μs) without penalizing short contexts
- Warm-starting from iter5 (β₁/β₂/β₅) continues gradual convergence toward iter3 stability
- Reverting α/β₃/β₄ to iter4/iter3 avoids propagating iter5's error

---

## Cross-Validation Decision

**Criteria for CV**:
- ✅ All hypotheses confirmed (every hypothesis ✅ verdict)
- Overall loss < 80% (ideally < 50%)
- No experiment with TTFT or E2E APE > 100%
- Coefficients physically plausible (no bounds violations)

**Iter5 Status**:
- ❌ 0 out of 5 hypotheses CONFIRMED (4 rejected, 1 partial)
- ❌ Overall loss = 603.26% (far above 80% threshold, 467% worse than iter4)
- ❌ 14 experiments with TTFT > 100% (all except reasoning, which stayed at 99%)
- ❌ Coefficients not physically plausible (α₁ = 371 μs, α₂ = 381 μs, β₄ = 0.620)

**Decision**: **DO NOT proceed to CV**. Iter6 required to address catastrophic failure.

**Expected iter6 outcome** (if context gating + profiling successful):
- Overall loss: 603% → 80-100% (recovering iter4 short-context accuracy + gaining reasoning improvement)
- TTFT RMSE: 519% → 40-50% (short contexts back to 4-77%, reasoning from 99% → 60-80%)
- E2E RMSE: 84% → 55-65% (recovering from Alpha coefficient explosion)
- All coefficients: Return to physical plausibility ranges

**If iter6 fails** (loss >150%): Consider fundamental model redesign (end-to-end validation for reasoning, accept 99% error as limitation, focus on improving short/medium contexts to <20% error).

---

## Lessons Learned

**What worked**:
1. **Hypothesis prediction accuracy**: H-main correctly predicted β₀ would rise to 0.25-0.35 (actual: 0.266 ✓)
2. **Diagnostic clauses**: Partially effective (β₀ >0.22 triggered different failure mode than predicted)
3. **Warm-starting from iter3**: Convergence faster (78 vs 185 trials), coefficients partially improved (β₁ -20%, β₅ -51%)
4. **Simplification**: Removing iter4's misspecified β₆ (activation BW) allowed partial coefficient recovery

**What didn't work**:
1. **Uniform coefficients**: β₀ and β₆ applying to ALL contexts created zero-sum game (helping reasoning hurts short contexts)
2. **Unconstrained β₀**: Allowing β₀ to rise freely (0.165 → 0.266) broke short-context predictions
3. **Base factor in overhead formula**: Adding 1.0 + tokens/2048 applied overhead to short contexts (should be max(0, ...))
4. **Not profiling before hypothesizing**: Three iterations (iter3/iter4/iter5) failed to improve reasoning — should have profiled after iter4

**Key insights for iter6**:
1. **Context-dependent coefficients are essential**: Short vs long contexts have qualitatively different behavior (different MFU, different overhead)
2. **Profile before hypothesizing**: Don't add basis functions based on physics intuition — profile vLLM to identify actual bottleneck
3. **Constrain coefficients defensively**: Prevent coefficients from rising/falling into ranges that break well-modeled experiments
4. **Test for boundary effects**: H-boundary was correct to predict short vs long would differ — but didn't predict INVERSE effect
5. **Zero-sum games are failure modes**: If helping X% of data hurts (100-X)% of data, optimizer will choose bad compromise

**Strategy Evolution validation**:
- **Prediction errors ARE valuable**: H-main rejection revealed β₀ rising breaks short contexts (wasn't anticipated in hypothesis)
- **Diagnostic clauses need refinement**: Should have included "if short contexts degrade, β₀ rose too high" clause
- **Causal mechanisms were testable**: H-boundary's boundary effect prediction was clear and testable (failed with inverse effect)
- **Multiple hypotheses help**: H-coefficient-norm provided additional evidence that β₆ converged too low (not just H-main failure)

**For next iteration (iter6)**:
1. **BLOCK on profiling**: DO NOT design iter6 hypothesis until profiling results available
2. **Context gating is mandatory**: Implement β₀_short / β₀_long / β₆_long architecture
3. **Defensive constraints**: Constrain β₀_short ≥ 0.35, β₀_long ≤ 0.30
4. **Expect 2-3 more iterations**: Reasoning may require algorithmic switch term (iter6) + refinement (iter7) before <80% loss
