# Iteration 3: Simplification + TP-Dependent Prefill Communication

## H-main: Model Simplification and Targeted TP Prefill Overhead

**Prediction**: Overall loss will decrease to <100% (from 136% in iter2), with:
- TTFT RMSE reducing from 72.75% to <50%
- E2E RMSE reducing from 63.44% to <50%
- TP=4 experiments (Llama-3.1-70B) will see TTFT APE drop from 42-90% to <35%

**Causal Mechanism**:

Iter2's regression (136% vs iter1's 134%) stems from adding two ineffective terms (β₇, β₈) that didn't move from initial values during 51 optimization trials. This added parameter bloat (9 terms → 12 total params for 15 experiments = 0.8 params/experiment, approaching overfitting threshold) without predictive value.

Iter3 addresses this through two complementary changes:

**1. Simplification (remove β₇ and β₈)**:
- β₇ (very long context overhead) remained at 1.0 (initial value) → optimizer found no gradient
- β₈ (per-request decode overhead) remained at 30μs (initial value) → too small to matter (< 1% of step times)
- Removing them reduces model from 9 to 7 Beta terms, eliminating parameter bloat
- β₁=1.553 inflation persists WITH β₈ present, proving β₈'s ineffectiveness

**2. Targeted improvement (add β₇ new: TP-dependent prefill communication)**:
- TP=4 experiments have asymmetric errors: high TTFT (42-90%) but excellent E2E (8-11%)
- This indicates **prefill underestimation** while decode is accurate
- Current β₃ (TP communication) captures **decode** TP overhead correctly (β₃=0.394 stable)
- Missing: **Prefill** TP overhead, which scales differently from decode:

  **Decode TP communication** (captured by β₃):
  - Processes 1 token per step
  - All-reduce per layer: `hidden_dim × bytes_per_param / network_BW`
  - Fixed overhead per decode step

  **Prefill TP communication** (NEW β₇):
  - Processes L tokens in parallel (L = 500-8000)
  - All-reduce per layer **per token**: `L × hidden_dim × bytes_per_param / network_BW`
  - Overhead scales linearly with prompt length
  - For TP=4, 80 layers, 4000 tokens: ~10-20ms overhead (vs 50-500ms total prefill time)

The new β₇ formula captures this scaling:
```
β₇ × TP × num_layers × (prompt_tokens / 1000.0)
```

This naturally captures:
- TP=1: Zero overhead (TP=1 → term is zero)
- TP=2: Moderate overhead (~50% of TP=4)
- TP=4: High overhead (4× base rate)

**Expected coefficient**: β₇ ~ 5-15 microseconds per (TP × layer × K-token)

For TP=4, 80 layers, 4000 tokens:
```
β₇ × 4 × 80 × 4 = β₇ × 1280
```

At β₇=10: 12,800 μs = 12.8ms overhead (~3% of 400ms prefill for 70B model)

**Code Citations**:
- **vLLM**: `vllm/model_executor/layers/linear.py:ColumnParallelLinear.forward()` calls all_reduce after each layer. Prefill processes batch of L tokens → L parallel all-reduces.
- **BLIS current**: `sim/latency/evolved_model.go:199-209` computes TP comm for decode only (single token).
- **BLIS iter3**: Will add prefill-specific TP comm term (β₇) that scales with prompt_tokens.

**Diagnostic Clause**:

*If this fails (overall loss >100% or TP=4 TTFT remains >60%), it indicates one of:*
1. **TP prefill overhead is not communication-dominated**: Large model prefill may be memory-bandwidth-bound (activation writes) or compute-bound (low MFU) instead of communication-bound. Investigate via nsys profiling: `nsys profile vllm --model llama-3.1-70b --tp 4`.
2. **Network bandwidth formula is wrong**: Formula uses peak NVLink bandwidth (600-900 GB/s), but effective bandwidth under congestion may be 50-70% of peak. β₇ coefficient will reveal this if it converges to 1.5-2.0× expected.
3. **TP=4 prefill errors have a different root cause**: Could be large-model-specific scheduler overhead, activation memory bandwidth saturation, or KV cache write bursts not captured by existing terms.

If TP=1/TP=2 experiments degrade significantly (>10 percentage points), it indicates β₇ formula has a bug (non-zero when TP=1, or incorrect scaling for TP=2).

---

## H-simplification: Removing Ineffective Terms Improves Optimization

**Prediction**: Removing β₇ (very long context) and β₈ (per-request decode) will:
- NOT degrade reasoning experiments (they'll stay ~100% TTFT, confirming β₇ was ineffective)
- NOT degrade Scout experiments (they'll stay 160-200%, confirming β₈ was ineffective)
- Potentially improve overall loss by 1-3% by reducing overfitting risk
- Speed up optimization convergence (fewer dimensions to explore: 11 params → 10 params)

**Causal Mechanism**:

Terms that don't move during optimization provide no predictive value but increase parameter count, slowing convergence and risking overfitting.

**Evidence from iter2**:
- β₇=1.0 exactly matched initial value after 51 trials → optimizer found no benefit from adjusting it
- β₈=30μs exactly matched initial value after 51 trials → optimizer found no benefit from adjusting it
- Reasoning TTFT errors remained ~100% WITH β₇ present → β₇ wasn't helping
- β₁=1.553 remained inflated WITH β₈ present → β₈ wasn't normalizing β₁

**Why removal helps**:
1. **Reduces overfitting risk**: 12 params for 15 experiments = 0.8 params/experiment. Removing 2 params → 10/15 = 0.67 params/experiment, safer margin.
2. **Speeds convergence**: Bayesian optimization explores 10-dimensional space instead of 12-dimensional. With 250 max trials, this improves sample density.
3. **Improves interpretability**: Smaller models are easier to understand and debug. If iter3 still fails, we have fewer variables to investigate.

**Diagnostic Clause**:

*If overall loss degrades by >5% after removing β₇ and β₈, it indicates they were capturing partial effects despite not moving.* This would suggest the initial values were "accidentally correct" by lucky initialization. In this case, next iteration should keep the terms but investigate why the optimizer didn't adjust them (too narrow bounds? wrong functional form? gradient masking?).

---

## H-boundary: TP Scaling Linearity

**Prediction**: The new β₇ (TP prefill comm) will affect experiments differently based on TP:
- **TP=1 experiments**: No change (<2 percentage points TTFT difference vs iter2)
- **TP=2 experiments** (non-Scout): 5-10 percentage points TTFT improvement
- **TP=4 experiments**: >20 percentage points TTFT improvement (from 42-90% to <35%)

**Causal Mechanism**:

TP communication overhead scales linearly with TP degree because the formula is:
```
β₇ × TP × num_layers × (prompt_tokens / 1000)
```

For TP=1: The term is zero (no communication).
For TP=2: The term is 2× base rate.
For TP=4: The term is 4× base rate.

This linear scaling matches the physics of ring all-reduce: communication time is proportional to (TP-1)/TP ≈ TP for TP > 1.

**Expected behavior**:
- TP=1 experiments (Llama-2-7B, Qwen2.5-7B, Mistral TP=1): No change because β₇ term contributes zero
- TP=2 experiments (Scout excluded, just Yi-34B and Mistral TP=2): Moderate improvement (current TTFT errors 59-72% should drop to 50-60%)
- TP=4 experiments (Llama-3.1-70B): Large improvement (current TTFT errors 42-90% should drop to <35%)

**Diagnostic Clause**:

*If TP=1 experiments improve by >5%, it indicates a bug in the formula* (TP term is non-zero when it should be zero, or formula is inadvertently affecting prefill compute MFU β₀).

*If TP=2 and TP=4 experiments don't improve proportionally (TP=4 improves 2× more than TP=2), it indicates the linear scaling assumption is wrong.* Real TP overhead may be logarithmic (log₂(TP)) or have a constant + linear component (C + k×TP). Next iteration should adjust formula.

---

## H-scout-reasoning-exclusion: Systematic Failures Require Investigation, Not Formulas

**Prediction**: Removing β₇ (targeting reasoning) and β₈ (targeting Scout decode) will NOT significantly degrade reasoning or Scout experiments:
- **Reasoning experiments**: Will stay ~100% TTFT (no degradation >5%)
- **Scout experiments**: Will stay 160-200% combined loss (no degradation >10%)

**Causal Mechanism**:

Iter2 findings revealed that adding formulas without understanding root causes provides no value:

**For reasoning experiments**:
- β₇ formula `(prompt_tokens - 4096) / 1000 × num_layers` assumes linear overhead scaling
- Reasoning TTFT errors remained 99.97-99.99% WITH β₇ present
- Root cause is likely: quadratic attention memory bandwidth (O(n²) for n>4096), KV cache preemption overhead, or prefix cache miss rate
- **Next iteration must profile vLLM reasoning experiments**, not add more formulas

**For Scout experiments**:
- Simulator bugs were fixed March 28 (InterleaveMoELayerStep, DenseIntermediateDim, split FLOPs)
- FLOPs tests PASS (`TestScoutInterleavedArchitecture_EndToEnd` validates FLOPs calculation)
- Scout experiments STILL FAIL (168-197% combined loss) with bugs fixed
- Root cause: **Inadequate model structure** - single β₀ (prefill MFU) cannot represent different efficiencies for MoE layers (lower, routing overhead) vs dense layers (higher)
- **Next iteration needs end-to-end latency validation test** (not just FLOPs tests) and potentially per-layer-type basis functions (β₀_dense, β₀_moe)

**Diagnostic Clause**:

*If reasoning experiments degrade by >10% (TTFT goes from ~100% to >110%), it indicates β₇ was capturing partial effects.* However, this is unlikely because β₇=1.0 didn't move from initial value.

*If Scout experiments degrade by >20% (combined loss from 160-200% to >220%), it indicates β₈ was absorbing some Scout-specific overhead.* However, this is unlikely because β₈=30μs is negligible compared to 100-1000ms step times.

More likely: Both will remain unchanged, confirming that these failure modes require root cause investigation (profiling, end-to-end testing) not formula additions.

---

## H-coefficient-normalization: Physical Plausibility Recovery via Simplification

**Prediction**: With β₇ and β₈ removed and TP prefill comm added, coefficients will move toward physically plausible ranges:
- **β₀ (prefill MFU)**: Will rise from 0.203 to 0.30-0.45 (still below ideal 0.40-0.55, but improved)
- **β₁ (decode memory-bound MFU)**: Will stay at ~1.50-1.60 (β₈ removal won't help because β₈ was ineffective)
- **β₂ (scheduler overhead)**: Will stay at ~0.1-0.2μs (genuinely negligible or captured elsewhere)

**Causal Mechanism**:

Iter2 showed that adding β₇ and β₈ did NOT normalize β₀ and β₁ because those terms didn't move. Iter3's simplification removes the dead weight, allowing the optimizer to focus on the 7 core terms + 1 new effective term (TP prefill comm).

**Why β₀ may rise**:
- With fewer parameters (10 vs 12), optimizer has better sample density in the remaining dimensions
- TP prefill comm term (β₇ new) may absorb some overhead that was artificially lowering β₀
- Tighter bounds force β₀ into more plausible range

**Why β₁ will stay high**:
- β₈ (per-request decode overhead) removal won't affect β₁ because β₈ was too small to matter
- β₁ inflation has a different root cause: either decode FLOPs formula undercounts operations, or decode regime split is fundamentally wrong (should be mixed compute+memory, not discrete regimes)

**Diagnostic Clause**:

*If β₀ does not rise (stays <0.25), it indicates there's still a major missing prefill overhead term* beyond TP communication. Candidates: activation memory bandwidth (residual connections, attention outputs), large-model-specific scheduler overhead, or KV cache write bursts.

*If β₁ rises further (>1.7), it indicates the simplification made decode predictions worse.* This would be surprising but would suggest β₈ was absorbing real overhead despite appearing negligible.

---

## Summary of Hypotheses

| Hypothesis | Prediction | Diagnostic Clause |
|------------|-----------|-------------------|
| **H-main** | Loss <100%, TTFT RMSE <50%, TP=4 TTFT <35% | If fails: TP prefill not comm-dominated, or network BW formula wrong |
| **H-simplification** | Removing β₇,β₈ improves by 1-3% | If degrades >5%: Terms were accidentally correct via lucky init |
| **H-boundary** | TP=4 improves >20pp, TP=2 improves 5-10pp, TP=1 no change | If TP=1 improves >5%: Formula bug (non-zero when TP=1) |
| **H-scout-reasoning** | Reasoning stays ~100% TTFT, Scout stays 160-200% | If degrade >10%: Terms captured partial effects |
| **H-coefficient-norm** | β₀ rises to 0.30-0.45, β₁ stays ~1.5-1.6 | If β₀ doesn't rise: Still missing major prefill term |

**Expected outcome**: Iter3 should achieve 90-110% overall loss (improvement from iter2's 136%), with TP=4 experiments showing the largest gains. Reasoning and Scout will remain problematic, confirming they need investigation not formulas.
