# Iteration 2: Hypothesis Validation

## Executive Summary

**Result: All hypotheses REJECTED or PARTIAL**

Iteration 2 achieved **150.20% overall loss** (TTFT: 69.31%, E2E: 80.90%), representing a **12% degradation** from iter1's 134.54%. The H-main prediction of <55% loss failed catastrophically.

**Root causes**:
1. **Scout MoE validation failure** (4 experiments at 100% APE) contributes 800% to loss
2. **β₇ (very long context) ineffective**: Coefficient 1.507 but reasoning experiments still at ~99% APE
3. **β₈ (per-request decode) negligible**: Coefficient 0.000042, effectively zero
4. **Wrong mechanism**: The very long context hypothesis targeted the wrong physics

---

## H-main: Very Long Context + Per-Request Overhead Mechanism

**Prediction** (from Agent 1): Overall loss will decrease from 134.54% (iter1) to **<50%**, with TTFT RMSE <30% and E2E RMSE <25%.

**Causal Mechanism** (from Agent 1):
- Part 1: Very long contexts (>4096 tokens) have prefill overhead from quadratic attention cost, KV recomputation, and reduced prefix cache effectiveness, captured by β₇
- Part 2: Each request incurs per-request decode overhead (scheduler work, attention state setup, kernel launch), captured by β₈
- Part 3: Smooth sigmoid regime transition instead of discrete batch_size<8 split

**Diagnostic Clause** (from Agent 1): *If this fails (loss remains > 80%), it indicates β₇ or β₈ are ineffective, β₁ still inflated, or coefficient distortion persists.*

**Actual Result**:
- Overall loss: **150.20%** (target: <55%)
- TTFT RMSE: **69.31%** (target: <30%)
- E2E RMSE: **80.90%** (target: <25%)
- **Loss INCREASED by 11.7% instead of decreasing by 59%**

**Verdict**: ❌ **REJECTED**

**Evidence**:

*Loss metrics* (from `loss`):
- `loss.overall_loss` = 150.20 (target: <55, actual: 2.7× target)
- `loss.ttft_rmse` = 69.31 (target: <30, actual: 2.3× target)
- `loss.e2e_rmse` = 80.90 (target: <25, actual: 3.2× target)
- Degradation from iter1: +15.66 overall, +0.02 TTFT, +15.66 E2E

*Coefficient values* (from `best_params`):
- β₀ = 0.155 (expected: 0.4-0.5, **got worse** from iter1's 0.203)
- β₁ = 1.316 (expected: 0.6-0.9, improved from 1.553 but still inflated)
- β₇ = 1.507 (very long context, **large coefficient** but ineffective)
- β₈ = 0.000042 (per-request, **effectively zero**)

*Per-experiment breakdown*:
- **Scout MoE experiments** (4): All 100% APE → 800% contribution (53% of loss)
- **Reasoning experiments** (2): Qwen 99%, Llama-2 99% TTFT → still catastrophic
- **Other experiments** (9): 5-83% TTFT, 48-90% E2E → mixed performance

**Causal Analysis**:

**Why Part 1 (β₇ very long context) failed**:

Agent 1's hypothesis predicted that reasoning experiments have ~100% TTFT error because prompts >4096 tokens trigger additional prefill overhead (attention bandwidth, KV recomputation, reduced prefix caching). The model added β₇ with scaling `max(0, prompt_tokens - 4096) / 1000 × num_layers`.

**However**:
1. β₇ converged to 1.507 (large, non-negligible)
2. Reasoning experiments STILL have 99% TTFT APE
3. This proves **the mechanism is wrong** - the overhead is NOT linearly proportional to `(prompt_tokens - 4096)`

**Possible root causes**:
- **Wrong threshold**: 4096 may not be the cutoff; could be 2048, 8192, or no fixed threshold
- **Wrong scaling**: Overhead may be quadratic `(tokens - threshold)²`, not linear
- **Wrong feature**: May depend on KV cache state (fragmentation, eviction) rather than prompt length
- **Validation failure**: Reasoning experiments may have incorrect ground truth latencies in the dataset
- **Missing interaction**: Overhead may depend on batch composition (mixed short+long) rather than individual request length

The 99% TTFT error persisting despite β₇ indicates the very long context hypothesis is fundamentally incorrect.

**Why Part 2 (β₈ per-request decode) failed**:

Agent 1 predicted β₁ is inflated (1.553) because it's compensating for missing per-request decode overhead (~10-50μs per request for scheduler work, attention setup, kernel launch). Adding β₈ should normalize β₁ to 0.6-0.9.

**However**:
1. β₈ converged to 0.000042 (effectively zero)
2. β₁ = 1.316 (improved from 1.553 but still inflated)
3. This proves **the per-request overhead hypothesis is insufficient**

**Possible root causes**:
- **Overhead is constant**: Already captured by α₀ or β₂, not per-request
- **Overhead scales differently**: May scale with context length or KV cache size, not request count
- **Wrong order of magnitude**: 0.000042 suggests ~0.042μs per request (100× too small)
- **Missing interaction**: Overhead may depend on batch size or TP config, not just request count

The negligible β₈ coefficient indicates per-request overhead is either absent or the model captured it incorrectly.

**Why Part 3 (smooth regime transition) didn't help**:

The sigmoid interpolation between memory-bound (β₁) and compute-bound (β₅, now β₅ after removing old β₅) should smooth predictions across batch sizes. However:
1. E2E RMSE INCREASED from 65.24% to 80.90% (+24%)
2. This suggests the sigmoid introduced more error, not less

**Optimization convergence**:
- Converged early at 142/250 trials (good convergence)
- Zero errors during optimization (stable)
- But converged to a **worse solution** than iter1

**Diagnostic Analysis**:

Using Agent 1's diagnostic clause:

1. **β₈ ineffective (<0.00001)**: ✅ Confirmed - β₈ = 0.000042, proving per-request overhead is negligible or wrong scaling
2. **β₇ ineffective (<0.01)**: ❌ Not confirmed - β₇ = 1.507 is large, but **still ineffective** at fixing reasoning experiments
3. **β₁ still inflated (>1.2)**: ✅ Confirmed - β₁ = 1.316, indicating missing additional decode overhead
4. **Coefficient distortion persists**: ✅ Confirmed - β₀ = 0.155 (worse than iter1's 0.203)

**What diagnostic clause reveals**:

The diagnostic clause says "If this fails, investigate whether overhead is batch-size dependent, per-sequence-length, or wrong threshold/scaling for long context."

**Findings**:
- β₇ large but ineffective → mechanism is wrong, not just threshold
- β₈ negligible → per-request hypothesis is incorrect
- β₁ still inflated → missing a DIFFERENT decode overhead mechanism
- β₀ got worse → removing β₅ (chunking) may have eliminated a real signal

**Critical insight**: Agent 1's hypotheses tested TWO independent mechanisms (very long context + per-request decode). BOTH failed. This suggests a more fundamental issue - the additive overhead model structure may be inadequate.

---

## H-ablation-long-context: Very Long Context Term Importance

**Prediction** (from Agent 1): Removing β₇ will increase TTFT RMSE by >20%, with reasoning experiments reverting from <50% to ~100% TTFT APE.

**Actual Result**: **Cannot validate - baseline already at 100% APE**

**Verdict**: ⚠️ **INDETERMINATE** (baseline failed)

**Evidence**:

The baseline (9-term model with β₇) already has:
- Reasoning experiments at 99% TTFT APE
- TTFT RMSE at 69.31%

**Analysis**:

This hypothesis predicted β₇ would reduce reasoning TTFT from ~100% to <50%. Since baseline is at 99%, the hypothesis is already falsified - we cannot test whether removing β₇ makes it worse because it's already at maximum error.

**Recommendation**: Do not run this ablation. The baseline failure makes comparison meaningless. Instead, investigate WHY β₇ with coefficient 1.507 didn't fix reasoning experiments.

---

## H-ablation-per-request: Per-Request Decode Term Importance

**Prediction** (from Agent 1): Removing β₈ will increase E2E RMSE by >15%, with largest impact on small-batch experiments.

**Actual Result**: **Cannot validate - β₈ already negligible**

**Verdict**: ⚠️ **INDETERMINATE** (term effectively absent)

**Evidence**:

β₈ = 0.000042 (from `best_params.beta[8]`)

At this magnitude, β₈ contributes ~0.042μs per request (assuming coefficient multiplies microseconds). For a batch of 8 requests, that's ~0.3μs total - negligible compared to typical decode step times (100-1000μs).

**Analysis**:

This hypothesis predicted removing β₈ would increase E2E RMSE by >15%. However, β₈ is already effectively zero - the optimizer found it unnecessary.

Removing a coefficient that's already zero will have **zero impact**. This ablation would confirm β₈ is redundant, but we already know that from the coefficient value.

**Recommendation**: Do not run this ablation. The coefficient magnitude proves β₈ is not contributing. Instead, investigate WHY β₈ converged to zero - does per-request overhead not exist, or is the model capturing it incorrectly?

---

## H-ablation-kv-mgmt: KV Management Term Importance (Reconfirmation)

**Prediction** (from Agent 1): Removing β₄ will increase E2E RMSE by >25%, reconfirming iter1's result (+30.28% E2E degradation).

**Actual Result**: **Cannot validate - β₄ also near zero**

**Verdict**: ⚠️ **INDETERMINATE** (term effectively absent)

**Evidence**:

β₄ = 0.000043 (from `best_params.beta[4]`)

This is the same order of magnitude as β₈ - effectively zero.

**Analysis**:

Iter1 found β₄ = 0.37μs and claimed its ablation caused +30.28% E2E degradation, marking it as CRITICAL. However, iter2's optimizer drove β₄ to near-zero (0.000043), suggesting:

1. **Iter1's ablation result was spurious**: β₄ was not actually critical
2. **Removing β₅ (chunking) eliminated β₄'s role**: They were confounded
3. **New terms (β₇, β₈) absorbed β₄'s contribution**: Model restructuring changed coefficient importance

The fact that β₄ went from "CRITICAL" in iter1 to "effectively zero" in iter2 suggests instability in coefficient identification.

**Recommendation**: Do not run this ablation. Instead, investigate why β₄ importance changed so dramatically between iterations. This may indicate overfitting to spurious correlations in iter1.

---

## H-boundary-long-context-threshold: Very Long Context Activation Threshold

**Prediction** (from Agent 1): β₇ should be near-zero for experiments with max_prompt_tokens < 4096, and substantial (>20% of TTFT) for experiments with max_prompt_tokens > 4096.

**Actual Result**: **Need to check per-experiment β₇ contributions**

**Verdict**: ⚠️ **PARTIAL** - Cannot fully validate without detailed per-request analysis

**Evidence**:

From per-experiment results, reasoning experiments (which should have prompts >4096 tokens):
- Qwen2.5 reasoning: 99% TTFT APE
- Llama-2 reasoning: 99% TTFT APE

These should be the experiments where β₇ contributes most heavily. However, they still have ~100% error.

**Analysis**:

The hypothesis structure is:
```
β₇ × max(0, prompt_tokens - 4096) / 1000 × num_layers
```

For reasoning experiments with (hypothetically) ~8000 token prompts and 28-32 layers:
```
β₇=1.507 × (8000-4096)/1000 × 30 ≈ 1.507 × 3.9 × 30 ≈ 176μs overhead
```

If baseline TTFT for these experiments is ~500-1000μs, adding 176μs should reduce APE from ~100% to ~60-80%. Instead, we see 99% APE, proving:

1. **The feature value is wrong**: Actual prompt lengths may not trigger β₇ as expected
2. **The threshold is wrong**: 4096 is not the right cutoff
3. **The mechanism is wrong**: Overhead doesn't scale this way

**Recommendation**: Investigate actual prompt token distributions in reasoning experiments. If prompts are <4096 tokens, β₇ would be inactive and the hypothesis is falsified.

---

## H-robustness-tp-scaling: Cross-TP Generalization

**Prediction** (from Agent 1): Model should generalize across TP configs (TP=1, TP=2, TP=4) with <5% error variance between TP groups. β₃ should handle TP scaling without β₇ or β₈ being TP-dependent.

**Actual Result**: **Error variance exceeds 5%**

**Verdict**: ❌ **REJECTED**

**Evidence**:

Error distribution by TP group (from `per_experiment_results`):

**TP=1 experiments** (7 experiments):
- TTFT APE: 6-99% (range: 93%)
- E2E APE: 48-99% (range: 51%)
- Mean TTFT: 32.3%, Mean E2E: 63.4%

**TP=2 experiments** (6 experiments):
- TTFT APE: 10-100% (range: 90%)
- E2E APE: 60-100% (range: 40%)
- Mean TTFT: 72.2%, Mean E2E: 87.2%

**TP=4 experiments** (2 experiments):
- TTFT APE: 27-37% (range: 10%)
- E2E APE: 64-74% (range: 10%)
- Mean TTFT: 32.4%, Mean E2E: 69.2%

**Variance across TP groups**:
- TTFT: TP=2 has 2.2× higher mean error than TP=1/TP=4
- E2E: TP=2 has 1.4× higher mean error than TP=1
- This is **WAY above** 5% threshold

**Causal Analysis**:

The hypothesis predicted β₇ and β₈ are TP-orthogonal (no interaction with TP). However, TP=2 has much higher error than TP=1 or TP=4, suggesting:

1. **Scout MoE confounding**: 4 of 6 TP=2 experiments are Scout MoE (validation failure at 100% APE), inflating TP=2 mean
2. **TP=2 has different physics**: Possible interaction between TP communication and new terms
3. **Coefficient instability**: β₃ (TP communication) may not correctly capture TP=2 behavior

**Excluding Scout MoE** (validation failures), TP=2 experiments:
- Yi-34B general-lite: 10% TTFT, 70% E2E
- Mistral-Nemo general-lite: 83% TTFT, 90% E2E

Still wide variance (73% range in TTFT), exceeding 5% threshold.

**Diagnostic Analysis**:

Agent 1's diagnostic clause says: *If error variance across TP configs exceeds 10%, one of the new terms is accidentally TP-correlated (confounded variable).*

**Findings**:
- Error variance is 40-90% (far exceeding 10%)
- This suggests β₇ or β₈ may have TP-dependent effects
- OR: Scout MoE validation failure dominates TP=2 and masks true TP scaling behavior

**Recommendation**: Re-evaluate TP scaling after fixing Scout MoE validation. If high variance persists, investigate whether very long context overhead or per-request decode overhead varies with TP degree.

---

## Summary of Verdicts

| Hypothesis | Prediction | Actual Result | Verdict | Key Finding |
|------------|-----------|---------------|---------|-------------|
| **H-main** | Loss <55% (TTFT <30%, E2E <25%) | Loss 150.20% (TTFT 69.31%, E2E 80.90%) | ❌ REJECTED | Loss INCREASED 12% instead of decreasing 59% |
| **H-ablation-long-context** | Removing β₇ increases TTFT by >20% | Cannot validate (baseline at 99%) | ⚠️ INDETERMINATE | β₇ large (1.507) but ineffective |
| **H-ablation-per-request** | Removing β₈ increases E2E by >15% | Cannot validate (β₈ near zero) | ⚠️ INDETERMINATE | β₈ = 0.000042, effectively absent |
| **H-ablation-kv-mgmt** | Removing β₄ increases E2E by >25% | Cannot validate (β₄ near zero) | ⚠️ INDETERMINATE | β₄ = 0.000043, contradicts iter1 |
| **H-boundary-long-context** | β₇ zero for <4096 tokens, substantial for >4096 | Cannot validate without per-request data | ⚠️ PARTIAL | Reasoning experiments still at 99% APE |
| **H-robustness-tp-scaling** | Error variance <5% across TP groups | Error variance 40-90% | ❌ REJECTED | TP=2 has 2.2× error vs TP=1/TP=4 |

**Overall verdict**: **All hypotheses rejected or indeterminate**. Iteration 2 represents a complete failure of the very long context + per-request overhead hypothesis.

---

## Next Steps

1. **Fix Scout MoE validation**: 800% contribution (53% of loss) from validation failures
2. **Re-examine reasoning experiments**: β₇ with coefficient 1.507 did NOT fix 99% TTFT error - wrong mechanism
3. **Investigate β₀ degradation**: Prefill efficiency got worse (0.155 vs iter1's 0.203)
4. **Reconsider model structure**: Additive overhead model may be inadequate
5. **Do NOT run ablations**: Baseline failure makes ablations non-informative
