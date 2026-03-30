# Iteration 5: Per-Layer Fixed Overhead (Kernel Launch + Scheduler + Memory Allocation)

## H-main: Per-Layer Fixed Overhead as the Missing Prefill Term

**Prediction**: Overall loss will decrease to <110% (from 129% in iter4), with:
- TTFT RMSE reducing from 66.49% to <55%
- E2E RMSE staying stable at ~62-65% (decode already well-modeled)
- β₀ (prefill MFU) rising from 0.165 to 0.25-0.35 (closer to physical plausibility)
- Reasoning experiments improving from ~100% TTFT to 70-85% TTFT (measurable progress toward resolution)

**Causal Mechanism**:

Iter4 findings conclusively rejected the activation bandwidth hypothesis: β₆ converged to 1.818 (far below expected 3.0-6.0), reasoning experiments improved 0%, and other coefficients destabilized dramatically (β₁ +73.8%, β₂ +328%, β₅ +160%). The 3.93% overall improvement came entirely from simplification (removing ineffective β₂/β₇), not from adding β₆.

**Why per-layer fixed overhead dominates prefill for long contexts:**

The reasoning experiments show 1000× underestimation (predicted ~1ms, actual ~1000ms). This magnitude of error cannot be explained by any continuous bottleneck:
- Memory bandwidth: max 3-5× slowdown (HBM limits: 3.35 TB/s on H100)
- Compute throughput: max 2-3× slowdown (MFU limits: 40-55% typical)
- Communication: max 2× slowdown (NVLink limits: 900 GB/s)

A 1000× slowdown requires **fixed per-operation overhead** that accumulates across layers and scales with chunking. The most likely culprits:

1. **Kernel launch overhead** (~50-100μs per CUDA kernel):
   - Each transformer layer requires 10-20 kernel launches (QKV projection, attention, FFN, layer norms)
   - For 80 layers: 800-1600 kernel launches × 75μs = 60-120ms base overhead
   - For long contexts (>2K tokens), vLLM chunks prefill into 2048-token pieces
   - Each chunk repeats kernel launches: 4 chunks × 80 layers × 75μs = 24ms additional

2. **Scheduler overhead** (batch formation, memory allocation):
   - vLLM scheduler prepares KV cache blocks, allocates attention buffers
   - For long contexts, memory allocation overhead scales with context length
   - PagedAttention block allocation: ~10-20μs per block × (tokens/block_size) per layer

3. **Memory allocator overhead** (prefix cache, KV block swapping):
   - Reasoning workloads may trigger different memory allocation patterns
   - KV cache preemption for long contexts (swapping blocks to CPU/GPU)

**Proposed basis function**: Per-layer overhead that scales with prefill chunking

```go
// Prefill scale factor: captures chunking overhead
// Base cost (1.0) + additional cost per 2048-token chunk
prefill_scale_factor := 1.0 + float64(num_prefill_tokens) / 2048.0

// Per-layer overhead: kernel launch + scheduler + memory allocation
per_layer_overhead_us := Beta[6] * float64(num_layers) * prefill_scale_factor
```

**Example calculation**:
- Reasoning experiment: 8K tokens, 80 layers, Llama-2-7B
- prefill_scale_factor = 1.0 + 8192/2048 = 5.0
- If β₆ = 2000μs (2ms per layer-chunk): overhead = 2000 × 80 × 5.0 = 800ms
- Current TTFT underestimation: ~900ms → this captures the gap!

**Expected coefficient**: β₆ ~ 1000-3000μs (1-3ms per layer per chunk-equivalent)
- Physical interpretation: Fixed cost per layer during prefill
- Includes: kernel launch (50-100μs × 10-20 kernels), scheduler overhead, memory allocation
- Should be independent of batch size, model size, or workload type (workload-agnostic)

**Expected impact on β₀**:
- Current β₀ = 0.165 compensates for missing overhead by artificially lowering prefill MFU
- With per-layer overhead term added, β₀ should rise to 0.25-0.35
- Physical interpretation: prefill achieves 25-35% MFU, with remaining time absorbed by β₆
- Still below ideal 0.40-0.55, but much closer than 0.165

**Code Citations**:
- **vLLM kernel launch**: `vllm/worker/model_runner.py:execute_model()` calls CUDA kernels per layer
- **vLLM prefill chunking**: `vllm/core/scheduler.py:_schedule_prefills()` chunks long prefills into 2048-token pieces (configurable via `--max-model-len`)
- **PagedAttention block allocation**: `vllm/core/block_manager.py:BlockSpaceManager.allocate()` allocates KV cache blocks
- **BLIS current**: No per-layer fixed overhead term exists (missing mechanism)
- **BLIS iter5**: Will add β₆ (per-layer prefill overhead) to capture this fixed cost

**Diagnostic Clause**:

*If this fails (overall loss >120% or β₀ doesn't rise above 0.22), it indicates:*

1. **Per-layer overhead is NOT the dominant bottleneck**: The 1000× underestimation may be due to:
   - Algorithmic switch: vLLM uses different attention kernel for long contexts (>8K tokens) with different performance characteristics
   - Quadratic attention memory bandwidth: O(n²) working set for attention computation that dominates for n>4K
   - KV cache preemption overhead: Swapping KV blocks to CPU for long contexts (prefix cache misses)

2. **Functional form is wrong**: Real overhead may not scale linearly with chunking. Alternatives:
   - Quadratic scaling: overhead ∝ (num_prefill_tokens)²
   - Logarithmic scaling: overhead ∝ log(num_prefill_tokens)
   - Per-request constant: overhead independent of prompt length (fixed 1s delay)

3. **Chunk size assumption is wrong**: vLLM may chunk at different sizes (512, 1024, 4096) or use dynamic chunking based on batch size

4. **Coefficient collinearity**: β₆ may overlap with β₀ (both scale with num_layers), causing gradient masking. Check correlation matrix of basis functions.

*If short-prompt experiments degrade by >10%, it indicates formula has a bug* (overhead applied when it shouldn't be, or scaling factor wrong for short prompts).

*If ALL experiments improve uniformly (no prompt-length correlation), it indicates β₆ is absorbing overhead that should be in β₀* (collinearity issue).

**Next steps if H-main fails**: Profile vLLM reasoning experiments with `nsys profile` to measure:
- Actual kernel launch count and timing distribution
- Attention kernel type/algorithm for long vs short contexts
- Scheduler batch formation latency
- Memory allocator overhead (block allocation, prefix cache operations)

---

## H-simplification-validated: Removing β₆ (Activation Bandwidth) Improves Stability

**Prediction**: Removing β₆ (activation bandwidth) from iter4 will:
- **NOT degrade** any experiments by >3% (activation term provided no benefit)
- Stabilize other coefficients: β₁ → 1.00-1.10 (from 1.802), β₂ → 0.30-0.35 (from 1.360), β₅ → 0.01-0.012 (from 0.0304)
- Reduce parameter count from 7 to 6 Beta terms (simplification)
- Improve optimization convergence (fewer dimensions, less collinearity)

**Causal Mechanism**:

Iter4 validation showed that β₆ (activation bandwidth) was **decisively rejected**:
- **0% improvement** in reasoning experiments (stayed at 99.98-99.99% TTFT)
- **Coefficient explosion** in other terms: β₁ +73.8%, β₂ +328%, β₅ +160%
- **β₆ converged to 1.818**, far below expected 3.0-6.0
- **β₀ DECREASED** from 0.169 → 0.165 instead of rising to 0.25-0.35

This pattern indicates **collinearity** between β₆ (activation bandwidth) and existing terms:
- β₆ formula: `activation_bytes = tokens × hidden_dim × num_layers × k_factor`
- β₀ formula: `prefill_time = FLOPs / (peak_TFLOPS × β₀)`
- Both scale with `tokens × num_layers`, creating gradient masking

When β₆ was added, the optimizer couldn't disambiguate overhead between β₆ and β₀, causing:
- β₀ to decrease (0.169 → 0.165) instead of rising
- Other coefficients (β₁, β₂, β₅) to absorb error from misspecified β₆

**Evidence from coefficient drift**:
- **β₁ (decode memory)**: 1.037 → 1.802 (+73.8%) — physically implausible, decode cannot be 80% slower than memory bandwidth allows
- **β₂ (TP communication)**: 0.318 → 1.360 (+328%) — physically implausible, 4.3× increase in NVLink overhead
- **β₅ (MoE gating)**: 0.0117 → 0.0304 (+160%) — absorbing error correlated with MoE experiments

**Why removal will help**:
1. **Eliminates collinearity**: Removing β₆ allows β₀ to fit prefill MFU independently
2. **Reduces parameter space**: 7 → 6 Beta terms improves Bayesian optimization sample density
3. **Prevents coefficient drift**: Iter3 coefficients were stable and physically plausible
4. **Follows validated pattern**: Iter3 improved 3.06% by removing ineffective β₇/β₈; iter4 improved 3.93% by removing β₂/β₇

**Expected outcome**: Coefficients should revert to iter3 stability ranges:
- β₀: Start at 0.169 (iter3), expect to rise to 0.25-0.35 with correct per-layer term
- β₁: Revert to 1.00-1.10 (iter3: 1.037)
- β₂: Revert to 0.30-0.35 (iter3: 0.318)
- β₅: Revert to 0.01-0.012 (iter3: 0.0117)

**Diagnostic Clause**:

*If experiments degrade by >5% after removing β₆, it indicates the formula was partially correct despite appearing ineffective.* This would suggest:
- Activation bandwidth overhead exists but scale factor k=4 was wrong (should be k=1-2)
- Or collinearity masked the true coefficient (try ablation: set β₀=0, refit β₆)

More likely: No experiments will degrade, confirming β₆ was harmful noise.

---

## H-coefficient-normalization: Physical Plausibility Recovery

**Prediction**: With β₆ (activation bandwidth) removed and new β₆ (per-layer overhead) added, coefficients will move toward physically plausible ranges:
- **β₀ (prefill MFU)**: Will rise from 0.165 to 0.25-0.35 (improved, but still below ideal 0.40-0.55)
- **β₁ (decode memory MFU)**: Will revert to 1.00-1.10 (from 1.802 in iter4, back to iter3's 1.037)
- **β₂ (TP decode comm)**: Will revert to 0.30-0.35 (from 1.360 in iter4, back to iter3's 0.318)
- **β₃ (KV cache management)**: Will stay at ~0.0004-0.0005 (stable across iterations)
- **β₄ (decode compute-bound MFU)**: Will stay at ~0.75-0.85 (iter3: 0.796)
- **β₅ (MoE gating overhead)**: Will revert to 0.01-0.012 (from 0.0304 in iter4, back to iter3's 0.0117)
- **β₆ (NEW: per-layer overhead)**: Will converge to 1000-3000μs (1-3ms per layer-chunk)

**Causal Mechanism**:

**Why β₀ will rise**:
- Current β₀ = 0.165 compensates for two missing overheads: (1) activation bandwidth (wrong), (2) per-layer overhead (correct)
- Iter4 added wrong term (activation bandwidth), causing β₀ to decrease further (0.169 → 0.165)
- Iter5 adds correct term (per-layer overhead), allowing β₀ to rise toward physical plausibility
- Expected: β₀ rises to 0.25-0.35, still below ideal 0.40-0.55 but much better than 0.165
- Remaining gap (0.35 vs 0.45 ideal) may require additional terms in iter6 (memory bandwidth, attention-specific overhead)

**Why β₁, β₂, β₅ will normalize**:
- Iter4's coefficient explosion was caused by misspecified β₆ (activation bandwidth)
- Removing β₆ eliminates collinearity and gradient masking
- Optimizer can fit β₁, β₂, β₅ independently without absorbing β₆'s error
- Expected: Coefficients revert to iter3 ranges (before activation bandwidth was added)

**Why β₄ will stay stable**:
- β₄ (decode compute-bound) = 0.796 (iter3) was physically plausible
- No changes to decode modeling in iter5, so β₄ should remain at ~0.75-0.85
- Slightly below 1.0 is expected (decode compute is not perfectly efficient)

**Why new β₆ will converge to 1000-3000μs**:
- Physical estimate: 1-3ms fixed cost per layer during prefill
- For 80 layers, 4 chunks: 2000μs × 80 × 4 = 640ms overhead
- This captures the ~900ms gap in reasoning experiments
- Coefficient should be stable across model sizes, TP configs, workload types (workload-agnostic)

**Diagnostic Clause**:

*If β₀ doesn't rise above 0.22, it indicates per-layer overhead is NOT the missing term.* Next candidates:
- Algorithmic switch (different attention kernel for long contexts)
- Quadratic attention memory bandwidth O(n²)
- KV cache preemption overhead

*If β₁, β₂, or β₅ don't normalize (stay >20% above iter3 values), it indicates:*
- New β₆ (per-layer overhead) has collinearity with existing terms
- Or iter3 coefficients were already absorbing real physics that β₆ should capture

*If β₆ (per-layer overhead) converges to <500μs or >5000μs, it indicates functional form is wrong.*
- <500μs: Overhead too small to explain 1000× underestimation
- >5000μs: Physically implausible (5ms per layer = 400ms for 80 layers alone)

---

## H-boundary: Per-Layer Overhead Scales with Prompt Length

**Prediction**: The new β₆ (per-layer overhead) will affect experiments differently based on prompt length and chunking:
- **Short prompts (<2K tokens)**: Minimal overhead (<100ms total), experiments change <10%
- **Medium prompts (2-4K tokens)**: Moderate overhead (100-300ms), experiments improve 10-25%
- **Long prompts (>4K tokens, reasoning)**: Large overhead (300-900ms), experiments improve >25% (from ~100% TTFT to 70-85%)

**Causal Mechanism**:

Per-layer overhead scales with prefill chunking:
```
prefill_scale_factor = 1.0 + num_prefill_tokens / 2048.0
overhead_us = β₆ × num_layers × prefill_scale_factor
```

**For short prompts (512 tokens, 32 layers, Llama-2-7B)**:
- prefill_scale_factor = 1.0 + 512/2048 = 1.25
- overhead = β₆ × 32 × 1.25 = 40 × β₆
- If β₆ = 2000μs: 80ms overhead
- Total prefill time: ~5-10ms (roofline) + 80ms = 85-90ms
- Actual: ~50-100ms → good fit!

**For medium prompts (3072 tokens, 32 layers)**:
- prefill_scale_factor = 1.0 + 3072/2048 = 2.5
- overhead = β₆ × 32 × 2.5 = 80 × β₆
- If β₆ = 2000μs: 160ms overhead
- Total prefill time: ~15-25ms (roofline) + 160ms = 175-185ms
- Should improve experiments from 50-80% TTFT to 30-60% TTFT

**For long prompts (8192 tokens, 80 layers, Llama-3.1-70B)**:
- prefill_scale_factor = 1.0 + 8192/2048 = 5.0
- overhead = β₆ × 80 × 5.0 = 400 × β₆
- If β₆ = 2000μs: 800ms overhead
- Total prefill time: ~50-100ms (roofline) + 800ms = 850-900ms
- Actual: ~1000ms → close! Should improve from ~100% TTFT to 70-85% TTFT

**Expected behavior**:
- **Reasoning experiments** (8K-16K prompts): Should improve from 99.98-99.99% TTFT to 70-85% TTFT (measurable progress)
- **TP=4 Llama-3.1-70B general-lite** (4K prompts): Should improve from 70.90% TTFT to 50-60% TTFT
- **Mistral TP=2 general-lite** (2-3K prompts): Should improve from 76.90% TTFT to 60-70% TTFT
- **Scout experiments**: May improve moderately (10-20%), but will remain problematic (~140-180% combined loss) due to interleaved MoE+dense architecture not captured by single β₀
- **Short-prompt experiments** (<1K tokens): Minimal change (<10%), already well-predicted

**Diagnostic Clause**:

*If short-prompt experiments improve significantly (>10%), it indicates a bug in the formula*:
- prefill_scale_factor is not correctly gated on prefill phase
- Or base cost (1.0) is too high for short prompts
- Check: Is overhead applied during decode? (should only apply during prefill)

*If long-prompt experiments don't improve (reasoning stays >95% TTFT), it indicates per-layer overhead is NOT the bottleneck*. Real cause may be:
- Algorithmic switch (different attention kernel for n>8K)
- Quadratic attention memory bandwidth O(n²)
- KV cache preemption overhead (swapping to CPU)

*If improvement is uniform across all prompt lengths (no correlation), it indicates collinearity with β₀* (both terms scale with num_layers, causing gradient masking).

Next iteration must profile vLLM to identify actual bottleneck.

---

## H-error-pattern: Which Experiments Should Improve Most?

**Prediction**: Iter5 will show largest improvements in experiments with **long prompts + large models**:

1. **Reasoning experiments** (99.98-99.99% → 70-85% TTFT):
   - Qwen2.5-7B reasoning-1-1 (16K tokens, 32 layers): 99.99% → 75-85% TTFT
   - Llama-2-7B reasoning (8K tokens, 32 layers): 99.98% → 80-90% TTFT
   - Scout reasoning-2 (8K tokens, 56 layers): 99.99% → 70-80% TTFT
   - **Mechanism**: 4-8 chunks × 32-56 layers × 2ms = 250-900ms overhead captured by new β₆

2. **TP=4 Llama-3.1-70B general-lite** (70.90% → 50-60% TTFT):
   - 4K tokens, 80 layers, TP=4
   - prefill_scale_factor = 3.0 → overhead = β₆ × 80 × 3.0 = 240 × β₆
   - If β₆ = 2000μs: 480ms overhead
   - **Mechanism**: Long prompts + many layers → high per-layer overhead

3. **Mistral TP=2 general-lite** (76.90% → 60-70% TTFT):
   - 2-3K tokens, 40 layers, TP=2
   - prefill_scale_factor = 2.0-2.5 → overhead = β₆ × 40 × 2.0 = 80-100 × β₆
   - If β₆ = 2000μs: 160-200ms overhead
   - **Mechanism**: Medium prompts + moderate layers → moderate overhead

**Experiments with minimal improvement** (<10% change):
- **Short-prompt, TP=1 experiments**: Llama-2-7B codegen/roleplay, Qwen2.5 roleplay, Mistral codegen
  - Prompts <1K tokens → prefill_scale_factor ≈ 1.0-1.5 → minimal overhead
- **Already-excellent experiments**: Yi-34B (14.69% TTFT), Llama-3.1-70B TP=4 codegen (3.86% TTFT)
  - Already well-predicted, minimal headroom for improvement

**Experiments with unknown improvement** (need investigation):
- **Scout experiments** (89-100% TTFT): May improve moderately (10-25%) from per-layer overhead, but will still fail (140-180% combined loss) due to interleaved MoE+dense architecture
  - Scout's issue is architectural (single β₀ cannot represent MoE vs dense layers), not missing overhead
  - Will remain problematic until per-layer-type coefficients added (β₀_dense, β₀_moe)

**Causal Mechanism**:

Per-layer overhead is largest for:
- **Long prompts** (>4K tokens) → more chunks → more overhead per layer
- **Large models** (70B, 80 layers) → more layers → more total overhead
- **TP=2/TP=4 configs** → larger models that require TP

Small models (7B, 32 layers) with short prompts (<1K tokens) have minimal per-layer overhead:
- 32 layers × 1.5 (scale factor) × 2ms = 96ms overhead (small relative to total time)

**Diagnostic Clause**:

*If pattern is reversed (short-prompt experiments improve more than long-prompt), it indicates the formula is wrong*:
- Functional form may be inverted (overhead decreases with prompt length instead of increasing)
- Or base cost (1.0) dominates, making chunking factor irrelevant

*If ALL experiments improve uniformly (no prompt-length correlation), it indicates collinearity with β₀*:
- Both β₀ and β₆ scale with num_layers
- Optimizer cannot disambiguate which term should capture prefill time
- Check correlation matrix of basis functions

*If NO experiments improve significantly (<5% across board), it indicates per-layer overhead is the wrong hypothesis*:
- Real bottleneck may be algorithmic switch, O(n²) attention, or KV preemption
- Next iteration must profile vLLM to identify actual cause

*If Scout improves >30% (combined loss <150%), it indicates per-layer overhead was masking Scout's architectural issue*:
- Unlikely, but would suggest Scout's MoE overhead is partially captured by per-layer term
- Check if β₅ (MoE gating) decreases when β₆ is added

---

## Summary of Hypotheses

| Hypothesis | Prediction | Key Metric | Diagnostic Clause |
|------------|-----------|------------|-------------------|
| **H-main** | Loss <110%, TTFT RMSE <55%, β₀ rises to 0.25-0.35 | Reasoning TTFT: 100% → 70-85% | If fails: Try algorithmic switch, O(n²) attention, or KV preemption |
| **H-simplification-validated** | Remove β₆ (activation BW) with no degradation, coefficients stabilize | β₁: 1.802 → 1.00-1.10 | If degrades >5%: Activation BW partially correct despite appearing ineffective |
| **H-coefficient-norm** | β₀ rises to 0.25-0.35, β₁/β₂/β₅ revert to iter3 ranges | β₀: 0.165 → 0.25-0.35 | If β₀ doesn't rise: Wrong hypothesis, per-layer overhead not the issue |
| **H-boundary** | Long prompts improve >25%, short prompts <10% | Reasoning: -25 to -30pp TTFT | If reversed: Formula wrong (inverted functional form) |
| **H-error-pattern** | Reasoning, TP=4, Mistral TP=2 improve most | Reasoning: 99.98% → 70-85% | If uniform: Collinearity with β₀ (both scale with num_layers) |

**Expected outcome**: Iter5 should achieve 100-115% overall loss (improvement from iter4's 129%), with reasoning experiments showing the largest gains (25-30 percentage point reduction in TTFT APE). Scout will remain problematic (~140-180% combined loss), confirming it needs per-layer-type MFU coefficients (β₀_dense, β₀_moe) or end-to-end validation. β₀ should rise toward physical plausibility (0.25-0.35), though still below ideal (0.40-0.55), indicating additional terms may be needed in iter6.

**If iter5 fails** (loss >120% or β₀ <0.22): Next iteration must profile vLLM prefill with `nsys` to identify actual bottleneck. Candidates:
- Algorithmic switch (different attention kernel for contexts >8K)
- O(n²) attention memory bandwidth (quadratic working set for large n)
- KV cache preemption overhead (swapping KV blocks to CPU for long contexts)
- Prefix cache miss rate (re-computing attention for repeated prompts)
