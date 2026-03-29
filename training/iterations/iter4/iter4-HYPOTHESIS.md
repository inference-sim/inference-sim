# Iteration 4: Activation Memory Bandwidth + Continued Simplification

## H-main: Activation Memory Bandwidth as the Missing Prefill Term

**Prediction**: Overall loss will decrease to <110% (from 133% in iter3), with:
- TTFT RMSE reducing from 70.59% to <55%
- E2E RMSE staying stable at ~62-65% (decode already well-modeled)
- β₀ (prefill MFU) rising from 0.169 to 0.25-0.35 (closer to physical plausibility)
- Reasoning experiments improving from ~100% TTFT to 70-85% TTFT (still high, but measurable progress)

**Causal Mechanism**:

Iter3 findings reveal that β₀ = 0.169 (16.9% MFU) is far below physical plausibility (40-55% MFU for prefill). This indicates a major missing prefill overhead term that's artificially suppressing β₀.

The new β₇ (TP prefill communication) was rejected by the optimizer (coefficient ≈ 0), eliminating communication as the missing overhead. This leaves **activation memory bandwidth** as the most likely candidate.

**Why activation memory bandwidth dominates prefill**:

During prefill, each transformer layer writes large activation tensors to HBM:
1. **Residual connections**: Full hidden_dim vectors written after each sublayer
2. **Attention outputs**: Q, K, V projections before attention (batch_size × seq_len × hidden_dim)
3. **Layer norm outputs**: Normalized activations before/after each sublayer
4. **MLP intermediate activations**: Expanded dimensions (4× or 8× hidden_dim for non-MoE)

For long prompts (4K-16K tokens), these writes become bandwidth-limited:
- **Example**: Llama-3.1-70B, 8K tokens, batch_size=4
  - Per layer: ~8K tokens × 8192 hidden_dim × 2 bytes (FP16) = 131 MB
  - 80 layers: 10.5 GB of activation writes per prefill step
  - H100 HBM bandwidth: 3.35 TB/s → 3.1 ms just for activation writes
  - But total prefill time: ~300-500 ms → activation writes are 0.6-1% overhead
  - **However**: This doesn't account for read-modify-write patterns, bank conflicts, or concurrent KV cache writes

**The missing overhead**: Activation bandwidth overhead is NOT captured by current model because:
- β₀ multiplies FLOPs-based prefill compute time
- Activation writes happen concurrently with compute but add latency when memory-bound
- For long contexts (>4K tokens), activation writes dominate memory bandwidth, competing with KV cache writes

**Proposed basis function**: Activation write bandwidth per prefill step
```
activation_bytes = num_prefill_tokens × hidden_dim × bytes_per_param × num_layers × k
activation_time_us = (activation_bytes / hbm_bandwidth) × 1e6
```

Where k ≈ 4-6 accounts for:
- Residual connections (1×)
- Attention QKV projections (3×)
- Layer norms (1-2×)
- Competing KV cache writes

**Expected coefficient**: β₇ (new activation bandwidth term) ~ 3.0-6.0
- Multiplier on theoretical activation write time
- Accounts for read-modify-write overhead, bank conflicts, concurrent KV writes
- Higher values (5-6) indicate severe memory bandwidth contention

**Expected impact on β₀**:
- Current β₀ = 0.169 compensates for missing activation overhead by suppressing compute time
- With activation bandwidth term added, β₀ should rise to 0.25-0.35
- Physical interpretation: prefill achieves 25-35% MFU, with remaining overhead absorbed by β₇

**Code Citations**:
- **vLLM**: `vllm/worker/model_runner.py:_prepare_model_input()` allocates activation buffers, `vllm/model_executor/layers/attention.py` writes attention outputs to HBM
- **BLIS current**: No activation bandwidth term exists (missing mechanism)
- **BLIS iter4**: Will add β₇ (activation write bandwidth) to capture this overhead

**Diagnostic Clause**:

*If this fails (overall loss >120% or β₀ doesn't rise above 0.22), it indicates:*
1. **Activation bandwidth is NOT the missing term**: Prefill overhead may be kernel launch overhead (~50μs per CUDA kernel × 100-200 kernels per layer), scheduler overhead for large batches, or attention-specific memory patterns (O(n²) working set for n>4K)
2. **Formula is wrong**: Real activation overhead may not scale linearly with tokens (could be sublinear due to buffering/caching) or may need different k factor (2-3 instead of 4-6)
3. **KV cache write overhead already captured**: β₅ (currently 0.796) may already absorb activation writes, and adding β₇ creates collinearity. Check if β₅ decreases significantly when β₇ is added.
4. **Reasoning failures have different root cause**: Could be quadratic attention memory bandwidth O(n²), prefix cache miss rate, or KV cache preemption overhead not captured by linear terms

If TP=1/TP=2 experiments degrade by >10%, it indicates β₇ formula has a bug or is inadvertently affecting compute terms.

---

## H-simplification: Continue Removing Ineffective Terms

**Prediction**: Removing β₂ (scheduler overhead ≈ 0) and β₇ (TP prefill comm ≈ 0) will:
- NOT degrade any experiments by >3%
- Reduce parameter count from 10 to 8, improving optimization quality
- Speed up convergence (fewer dimensions: 10 → 8 Beta terms)
- Overall loss improvement of 0-2% from simplification alone (baseline before activation term helps)

**Causal Mechanism**:

Iter3 validated that removing ineffective terms improves optimization:
- Removed β₇ (very long context) and β₈ (per-request decode) from iter2
- Loss improved by 3.06% (136.19% → 133.13%)
- β₁ normalized significantly (1.553 → 1.037) after removing β₈

**Evidence from iter3 that β₂ and β₇ are ineffective**:
- β₂ = 9.97e-05 (0.0997 μs) — genuinely negligible scheduler overhead
- β₇ = 2.78e-07 (≈ 0) — TP prefill comm rejected by optimizer
- Both stayed near initial values throughout 84 trials
- No experiments rely on these terms (removing them won't degrade predictions)

**Why removal helps**:
1. **Reduces overfitting risk**: 10 params for 15 experiments = 0.67 params/experiment. Removing 2 → 8/15 = 0.53 params/experiment, even safer.
2. **Speeds convergence**: Bayesian optimization explores 8-dimensional space instead of 10-dimensional
3. **Improves interpretability**: Fewer terms make it easier to understand which physics matter

**Diagnostic Clause**:

*If any experiment degrades by >5% after removing β₂ and β₇, it indicates they captured partial effects despite appearing ineffective.* This would suggest:
- Initial values were accidentally correct via lucky initialization
- Or collinearity with other terms masked their contributions
- Next iteration should investigate why optimizer didn't adjust them (gradient masking? too narrow bounds?)

More likely: All experiments will be unaffected, confirming these terms are truly redundant.

---

## H-coefficient-normalization: Physical Plausibility Recovery

**Prediction**: With β₂/β₇ removed and activation bandwidth added, coefficients will move toward physically plausible ranges:
- **β₀ (prefill MFU)**: Will rise from 0.169 to 0.25-0.35 (improved, but still below ideal 0.40-0.55)
- **β₁ (decode memory MFU)**: Will stay at ~1.00-1.10 (already normalized in iter3)
- **β₃ (TP decode comm)**: Will stay at ~0.30-0.35 (stable across iterations)
- **β₅ (KV cache overhead)**: May decrease from 0.796 to 0.60-0.70 if it was absorbing activation overhead
- **β₆ (attention overhead)**: May decrease from 0.0117 to 0.008-0.010 if it was absorbing activation overhead
- **β₇ (NEW: activation bandwidth)**: Will converge to 3.0-6.0 (multiplier on theoretical write time)

**Causal Mechanism**:

**Why β₀ will rise**:
- Current β₀ = 0.169 compensates for missing activation overhead by artificially lowering prefill MFU
- With β₇ (activation bandwidth) capturing real overhead, optimizer can fit β₀ to actual compute efficiency
- Expected: β₀ rises to 0.25-0.35, still below ideal 0.40-0.55 but much closer than 0.169

**Why β₅ and β₆ may decrease**:
- Iter3 findings noted: β₅ increased from 0.651 → 0.796, β₆ increased from 0.008 → 0.0117
- Hypothesis: With β₀ too low and missing prefill term, optimizer pushed overhead into β₅/β₆
- If hypothesis correct, adding proper activation term (β₇) should allow β₅/β₆ to decrease back to iter2 levels

**Why β₁ will stay normalized**:
- β₁ = 1.037 (iter3) is already close to physical expectation (~1.0 for memory-bound decode)
- No changes to decode modeling in iter4, so β₁ should remain stable

**Diagnostic Clause**:

*If β₀ doesn't rise above 0.22, it indicates activation bandwidth is NOT the missing term.* Candidates for next iteration:
- Kernel launch overhead (~50μs per kernel × 100-200 kernels per layer)
- Scheduler overhead for large models (70B-specific batching costs)
- Attention-specific O(n²) memory bandwidth for n>4K tokens

*If β₅ or β₆ don't decrease (or increase further), it indicates they're NOT absorbing activation overhead.* This would refute the "coefficient drift" hypothesis and suggest β₅/β₆ increases are capturing real physics.

*If β₁ rises above 1.2, it indicates simplification degraded decode predictions.* This would be surprising but would suggest β₂ was absorbing real scheduler overhead despite appearing negligible.

---

## H-boundary: Activation Bandwidth Scales with Prompt Length

**Prediction**: The new β₇ (activation bandwidth) will affect experiments differently based on prompt length:
- **Short prompts (<1K tokens)**: Minimal effect (<5% TTFT change)
- **Medium prompts (1K-4K tokens)**: Moderate effect (10-20% TTFT improvement)
- **Long prompts (>4K tokens, reasoning workload)**: Large effect (>25% TTFT improvement, from ~100% to 70-85%)

**Causal Mechanism**:

Activation bandwidth overhead scales linearly with prompt length:
```
activation_bytes = num_prefill_tokens × hidden_dim × bytes_per_param × num_layers × k
```

For short prompts (256-512 tokens):
- Activation writes: ~128 MB (Llama-2-7B, 32 layers, k=4)
- H100 bandwidth: 3.35 TB/s → 38 μs write time
- With β₇ = 4.0: 152 μs overhead
- Total prefill time: ~10-20 ms → overhead is 0.7-1.5% (negligible)

For long prompts (8K-16K tokens):
- Activation writes: 4-8 GB (Llama-3.1-70B, 80 layers, k=4)
- H100 bandwidth: 3.35 TB/s → 1.2-2.4 ms write time
- With β₇ = 4.0: 4.8-9.6 ms overhead
- Total prefill time: 300-500 ms → overhead is 1-3% (still small, but measurable)

**Expected behavior**:
- **Reasoning experiments** (8K-16K prompts): Should improve from ~100% TTFT to 70-85% TTFT
  - Current: Model predicts ~1ms, actual ~1000ms (1000× underestimation)
  - With β₇: Model will add 5-10ms activation overhead, plus β₀ rises → prediction closer to reality
  - Still won't fully solve (need profiling to identify actual bottleneck), but should make measurable progress

- **TP=4 general-lite** (4K prompts): Should improve from 70.90% to 50-60% TTFT
  - Medium-length prompts benefit from activation term + β₀ normalization

- **Short-prompt experiments** (<1K tokens): Minimal change (<5% TTFT)
  - Activation overhead negligible for short prompts

**Diagnostic Clause**:

*If short-prompt experiments improve significantly (>10%), it indicates a bug in the formula* (activation term is non-zero when it should be negligible, or formula is inadvertently affecting other terms).

*If long-prompt experiments don't improve (reasoning stays >95% TTFT), it indicates activation bandwidth is NOT the missing overhead.* Real bottleneck may be:
- Quadratic attention memory bandwidth O(n²) for n>4K
- KV cache preemption overhead (swapping KV blocks to CPU for long contexts)
- Prefix cache miss rate (re-computing attention for repeated prompts)

Next iteration must profile vLLM reasoning experiments to identify actual bottleneck.

---

## H-error-pattern: Which Experiments Should Improve Most?

**Prediction**: Iter4 will show largest improvements in experiments with:
1. **Long prompts + large models**: Reasoning experiments (Llama-2-7B, Qwen2.5-7B, Scout reasoning) will improve from ~100% TTFT to 70-85%
2. **TP=4 + medium prompts**: Llama-3.1-70B general-lite will improve from 70.90% TTFT to 50-60%
3. **Mistral TP=2 general-lite**: Will improve from 79.61% TTFT to 65-75%

Experiments with **minimal improvement** (<10% change):
- **Short-prompt, TP=1 experiments**: Llama-2-7B codegen/roleplay, Qwen2.5 roleplay, Mistral codegen
- **Already-excellent experiments**: Yi-34B (14.69% TTFT), Llama-3.1-70B TP=4 codegen (3.86% TTFT)

Experiments with **unknown improvement** (need investigation):
- **Scout experiments** (89-100% TTFT): May improve slightly (10-20%) from activation bandwidth, but still fail due to interleaved MoE+dense architecture not captured by single β₀

**Causal Mechanism**:

Activation bandwidth overhead is largest for:
- Long prompts (>4K tokens) → more activation bytes to write
- Large models (70B, 80 layers) → more layers, more writes per token
- TP=2/TP=4 configs → larger models that need TP

Small models (7B, 32 layers) with short prompts (<1K tokens) have negligible activation overhead.

**Diagnostic Clause**:

*If pattern is reversed (short-prompt experiments improve more than long-prompt), it indicates the formula is wrong.* Activation overhead may have different functional form (nonlinear, or dependent on batch size instead of prompt length).

*If ALL experiments improve uniformly (no pattern), it indicates β₇ is absorbing overhead that should be attributed to β₀ (prefill MFU).* This would suggest collinearity between activation bandwidth and compute time.

*If NO experiments improve significantly (<5% across board), it indicates activation bandwidth is the wrong hypothesis.* Next iteration must profile to identify actual missing term.

---

## Summary of Hypotheses

| Hypothesis | Prediction | Key Metric | Diagnostic Clause |
|------------|-----------|------------|-------------------|
| **H-main** | Loss <110%, TTFT RMSE <55%, β₀ rises to 0.25-0.35 | Overall loss: 133% → <110% | If fails: Activation BW not the missing term, try kernel launch or O(n²) attention |
| **H-simplification** | Remove β₂, β₇ with no degradation | Parameter count: 10 → 8 | If degrades >5%: Terms captured partial effects despite appearing ineffective |
| **H-coefficient-norm** | β₀ rises to 0.25-0.35, β₅/β₆ decrease | β₀: 0.169 → 0.25-0.35 | If β₀ doesn't rise: Wrong hypothesis, try kernel launch overhead |
| **H-boundary** | Long prompts improve >25%, short prompts <5% | Reasoning TTFT: 100% → 70-85% | If reversed: Formula wrong (nonlinear or batch-dependent) |
| **H-error-pattern** | Reasoning, TP=4, Mistral TP=2 improve most | Reasoning: -15 to -30pp TTFT | If uniform improvement: Collinearity with β₀ (absorbing compute overhead) |

**Expected outcome**: Iter4 should achieve 100-115% overall loss (improvement from iter3's 133%), with reasoning and TP=4 experiments showing the largest gains. Scout will remain problematic (~160-190% combined loss), confirming it needs per-layer-type MFU or end-to-end validation. β₀ should rise toward physical plausibility (0.25-0.35), though still below ideal (0.40-0.55).

**If iter4 fails** (loss >120% or β₀ <0.22): Next iteration must profile vLLM prefill to identify actual bottleneck. Candidates: kernel launch overhead, O(n²) attention memory bandwidth, KV cache preemption, or attention-specific memory patterns.
