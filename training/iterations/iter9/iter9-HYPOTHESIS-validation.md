# Iteration 9: Hypothesis Validation

## H-main: FP8 Dequantization Overhead Captures Scout Residual

**Prediction** (from Agent 1):
- **Overall loss**: 155.35% → **<75%** (80pp improvement, 52% reduction)
- **TTFT RMSE**: 63.98% → **<35%** (29pp improvement, 45% reduction)
- **E2E RMSE**: 91.37% → **<45%** (46pp improvement, 50% reduction)
- **Scout TTFT error**: Avg 92% (range 79-100%) → **<40%** (>52pp improvement for all 4 Scout experiments)
- **Non-FP8 experiments**: Remain stable or improve slightly (< ±10pp change from iter8)
- **β₃, β₅, β₇ reversion**: β₃ (4.4ms → 0.4-1ms), β₅ (41μs → 10-20μs), β₇ (26ms → 10-20ms)
- **β₉ coefficient**: **17-50μs per token per layer** (FP8 dequantization overhead)

**Quantitative Threshold**: If overall loss does NOT reduce below 95%, or if Scout TTFT does NOT improve to <60%, then H-main is REJECTED.

**Causal Mechanism** (from Agent 1):

FP8 dynamic quantization introduces per-token mixed-precision coordination overhead: (1) Weight dequantization (10-30μs per token per layer): FP8 → FP16/BF16 conversion before matmul; (2) Mixed-precision coordination (5-15μs per token per layer): FP8 weights × FP16 activations; (3) Dynamic scale management (2-5μs per token per layer): Per-tensor scale factors. Expected total: 17-50μs per token per layer × 56 layers = 95-280ms per Scout prefill request.

**Diagnostic Clause** (from Agent 1):

*If this hypothesis fails (overall loss remains >95% OR Scout TTFT >60%), it indicates:*
1. β₉ coefficient converged to zero → FP8 dequantization overhead negligible
2. β₉ coefficient converged >100μs → Unrealistically high, absorbing other terms
3. β₉ is plausible but insufficient → FP8 overhead is ONE component, need additional terms
4. Non-FP8 experiments degraded >10pp → Zero-sum trade-off, check formulation

**Actual Result**:

**Loss Metrics**:
- Overall loss: **160.60%** (+5.25pp from iter8's 155.35%, WORSE)
- TTFT RMSE: **64.76%** (+0.78pp from iter8's 63.98%, unchanged)
- E2E RMSE: **95.84%** (+4.47pp from iter8's 91.37%, WORSE)

**Scout TTFT Errors** (mixed results):
- Scout general-lite (exp 17, NEW DATA): **92.28%** TTFT (target <48%, failed by 44pp)
- Scout reasoning-lite (exp 48): **90.54%** TTFT (-7.92pp from iter8's 98.46%, minimal improvement)
- Scout codegen (exp 20): **58.21%** TTFT (-33.87pp from iter8's 92.08%, significant improvement!)
- Scout roleplay (exp 21): **25.73%** TTFT (-53.37pp from iter8's 79.10%, huge improvement!)

**β₉ Coefficient**: **0.0000001365 seconds** = **0.1365 μs per token per layer** (124-366× SMALLER than predicted 17-50μs)

**Non-FP8 Experiments**: Mixed changes, largest shifts in general-lite workloads (+/- 10-20pp).

**Coefficient Changes** (iter8 → iter9):
- β₃ (KV mgmt): 4.4ms → **9.6ms** (+117.7%, DOUBLED, moving AWAY from physical)
- β₅ (MoE gating): 41μs → **20μs** (-51.9%, approaching physical 10-20μs range ✓)
- β₆ (scheduler): 13ms → **99ms** (+654.4%, EXPLODED 7.5×)
- β₇ (decode overhead): 26ms → **11ms** (-58.0%, approaching physical 10-20ms range ✓)
- β₈ (MoE routing): 30μs → **73μs** (+142.6%, now ABOVE predicted 10-50μs range)
- β₂ (TP comm): 0.18 → **0.82** (+342.6%, EXPLODED 4.5×)

**Verdict**: ❌ **REJECTED**

**Evidence**:
1. **Loss increased**: Overall loss 160.60% vs target <75% (failed by 85pp), worse than iter8 by +5.25pp
2. **β₉ essentially ZERO**: 0.1365 μs/tok/layer vs expected 17-50 μs/tok/layer (124-366× smaller)
3. **Scout results mixed**: 2/4 improved significantly (codegen -34pp, roleplay -53pp), 2/4 failed (general-lite 92%, reasoning-lite 91%)
4. **Coefficient explosions**: β₆ (+654%), β₂ (+343%), β₈ (+143%) — model absorbing error into OTHER terms
5. **No uniform Scout improvement**: Contradicts architecture-specific hypothesis (FP8 should affect all Scout uniformly)
6. **Workload-specific pattern**: Short-sequence workloads (codegen, roleplay) improved; long-sequence (general-lite, reasoning-lite) failed

**Causal Analysis**:

**Why β₉ Failed**: The optimizer learned β₉ ≈ 0, meaning FP8 dequantization overhead is either:
1. **Negligible**: Mixed-precision coordination is handled efficiently by CUDA/vLLM, not a bottleneck
2. **Already captured**: Roofline's MFU factor may already account for FP8 inefficiency
3. **Not the mechanism**: Scout's bottleneck lies elsewhere (batching, memory, different overhead)

**Evidence against the causal mechanism**:
1. **β₉ is zero**: Optimizer found no FP8 dequantization overhead worth modeling
2. **Scout improvement is workload-specific**: Roleplay and codegen improved significantly (-53pp, -34pp), but general-lite and reasoning-lite did not. If FP8 were the bottleneck, ALL Scout experiments should improve uniformly (same architecture). The workload-dependent pattern suggests the bottleneck is NOT architecture-specific (FP8) but workload-dependent (batching, scheduling, sequence length).
3. **Other coefficients absorbed error**: β₆ exploded (+654%), β₂ exploded (+343%), suggesting model is compensating via scheduler overhead and TP communication, not FP8 dequantization.
4. **β₈ inflated further**: MoE routing overhead increased to 73μs (now ABOVE the 10-50μs predicted range), suggesting MoE overhead may be larger than initially estimated, not FP8.

**Diagnostic Analysis** (using Agent 1's diagnostic clause):

**Diagnostic clause evaluation**:
- ✅ **Clause 1 applies**: β₉ coefficient converged to zero → FP8 dequantization overhead negligible
- ✅ **Alternative Scout bottlenecks**: Must investigate workload-specific mechanisms (batching delay, scheduler overhead, memory pressure for long sequences)

**Alternative Scout bottlenecks** (from diagnostic clause):
1. **Batching inefficiency**: Scout roleplay and codegen (short sequences) improved, but general-lite and reasoning-lite (long sequences) did not. Long sequences may cause batching delays or memory pressure.
2. **Scheduler overhead**: β₆ exploded from 13ms to 99ms (+654%), suggesting scheduler is absorbing Scout error for long-sequence workloads.
3. **TP=2 communication**: β₂ exploded from 0.18 to 0.82 (+343%), suggesting cross-GPU coordination overhead is larger than expected for Scout.
4. **Model config error**: InterleaveMoELayerStep=26 or NumExpertsPerTok might be incorrect, causing β₈ to inflate to 73μs (above predicted range).
5. **Sequence length dependency**: The fact that short-sequence Scout workloads improved but long-sequence did not suggests the bottleneck scales with sequence length (memory bandwidth, KV cache pressure, batching delay).

**Why workload-specific pattern contradicts H-main**: H-main predicted FP8 overhead would affect ALL Scout experiments uniformly (same FP8 architecture). Instead, we see:
- **Short-sequence Scout** (roleplay 25% TTFT, codegen 58% TTFT): Improved significantly
- **Long-sequence Scout** (general-lite 92% TTFT, reasoning-lite 91% TTFT): Failed completely

This pattern indicates the bottleneck is **sequence-length-dependent**, NOT **architecture-dependent (FP8)**. The true bottleneck likely involves:
- Memory bandwidth saturation for long sequences
- Batching inefficiency (long sequences delayed by batch formation)
- Scheduler overhead scaling with sequence length (β₆ explosion)
- KV cache management scaling with sequence length (β₃ doubled)

---

## H-ablation-beta9: β₉ Accounts for Majority of Scout Improvement

**Prediction** (from Agent 1):
- **With β₉ (full model)**: Scout TTFT avg 92% → <40% (>52pp improvement)
- **Without β₉ (ablated)**: Scout TTFT avg 92% → 80-92% (<12pp improvement, similar to iter8)
- **Difference**: β₉ contributes **>40pp** of Scout TTFT improvement

**Actual Result**: **Cannot validate — β₉ is essentially zero (mechanism not active)**

**Verdict**: ❌ **REJECTED**

**Evidence**:
- β₉ = 0.1365 μs/tok/layer (essentially zero, 124-366× smaller than predicted 17-50 μs/tok/layer)
- Since β₉ is zero, it cannot contribute >40pp to Scout improvement
- The improvements seen in Scout codegen (-34pp) and Scout roleplay (-53pp) were driven by OTHER coefficients (β₂, β₆, β₈), not β₉

**Causal Analysis**: The hypothesis assumed β₉ would be non-zero and drive Scout improvement. Instead, β₉ converged to zero, meaning the FP8 dequantization mechanism was not learned by the optimizer. Any Scout improvement observed (codegen -34pp, roleplay -53pp) must be attributed to other terms:
- β₆ (scheduler): +654% increase
- β₂ (TP comm): +343% increase
- β₈ (MoE routing): +143% increase

These terms absorbed the Scout error, not β₉.

**Recommendation**: Do not run ablation study — β₉ is already effectively ablated (zero coefficient). The hypothesis is rejected because the mechanism was not activated.

---

## H-boundary-fp8-vs-non-fp8: β₉ Effect Should Vanish for Non-FP8 Models

**Prediction** (from Agent 1):
- **Non-FP8 models** (11 experiments): β₉ contribution = 0 (BytesPerParam = 2.0, isFP8 = 0)
  - Non-FP8 TTFT change: <±10pp from iter8 (stable or slight improvement)
- **FP8 models** (4 Scout experiments): β₉ contribution = 95-280ms per request
  - Scout TTFT improvement: >52pp (92% → <40%)

**Actual Result**:

**Non-FP8 Models**: Mixed changes, some >10pp shifts (but β₉ is zero, so not caused by β₉)
- Mistral Nemo general-lite: 91% TTFT (new workload, no baseline)
- Llama-2-7b reasoning-lite: 84% TTFT (+0.99pp from iter8, stable)
- Qwen2.5-7b reasoning-lite: 79% TTFT (+0.09pp, stable)

**FP8 Models** (Scout): Mixed, NOT uniform improvement
- Scout general-lite: 92% TTFT (failed, no improvement)
- Scout reasoning-lite: 91% TTFT (-7.92pp, minimal)
- Scout codegen: 58% TTFT (-33.87pp, significant)
- Scout roleplay: 26% TTFT (-53.37pp, huge)

**β₉ Contribution**: ZERO for all models (β₉ = 0.1365 μs/tok/layer ≈ 0)

**Verdict**: ⚠️ **PARTIAL** (boundary condition holds, but Scout failed)

**Evidence**:
- ✅ **Confirmed**: β₉ ≈ 0 for all models (mechanism not active, so boundary condition trivially satisfied)
- ❌ **Rejected**: Scout experiments did NOT improve uniformly >52pp TTFT (actual: -54pp to -8pp, mixed results)
- ❌ **Rejected**: Improvement is workload-specific (short sequences improved, long sequences failed), contradicting architecture-specific (FP8) hypothesis

**Causal Analysis**: The mathematical boundary condition holds trivially because β₉ is zero (no effect on any model). However, the prediction that Scout would improve uniformly due to FP8 mechanism is FALSE. Scout improvements were:
1. **Workload-dependent** (short sequences improved, long sequences failed)
2. **Driven by other terms** (β₆ +654%, β₂ +343%, β₈ +143%)

This proves the FP8 hypothesis was incorrect — Scout's bottleneck is NOT FP8 dequantization.

---

## H-error-pattern-scout: Scout Experiments Should Improve Uniformly

**Prediction** (from Agent 1): All 4 Scout experiments should improve >52pp TTFT (proportional to sequence length):
- Scout general-lite (exp 17): 100% → <48% (>52pp improvement, longest sequence)
- Scout reasoning-lite (exp 48): 98% → <46% (>52pp improvement, long sequence)
- Scout codegen (exp 20): 92% → <40% (>52pp improvement, moderate sequence)
- Scout roleplay (exp 21): 79% → <27% (>52pp improvement, shortest sequence)

**Actual Result**: Scout experiments showed OPPOSITE pattern (short sequences improved, long sequences failed):
- Scout general-lite: 92.28% (NEW DATA, target <48%, failed by 44pp)
- Scout reasoning-lite: 90.54% (-7.92pp improvement, target >52pp, failed by 44pp)
- Scout codegen: 58.21% (-33.87pp improvement, approaching target but <52pp)
- Scout roleplay: 25.73% (-53.37pp improvement, **ONLY experiment meeting >52pp target!**)

**Verdict**: ❌ **REJECTED**

**Evidence**:
- Only 1/4 Scout experiments met >52pp improvement target (roleplay: -53.37pp)
- Scout codegen showed moderate improvement (-33.87pp, <52pp target)
- Scout reasoning-lite showed minimal improvement (-7.92pp, <<52pp target)
- Scout general-lite showed NO improvement (92%, still worst)
- **OPPOSITE pattern**: Hypothesis predicted long sequences would improve MORE (more FP8 overhead). Actual: short sequences improved MORE (roleplay > codegen >> reasoning-lite > general-lite).

**Causal Analysis**: The hypothesis assumed FP8 overhead scales uniformly with `totalTokens × numLayers`, so longer sequences would have more FP8 overhead and larger improvements. Instead, we see the OPPOSITE:

**Actual pattern (sequence length → improvement)**:
- Roleplay (shortest, ~50-100 tokens): **-53pp** (best improvement)
- Codegen (moderate, ~100-200 tokens): **-34pp** (moderate improvement)
- Reasoning-lite (long, ~200-400 tokens): **-8pp** (minimal improvement)
- General-lite (longest, ~400-600 tokens): **0pp** (no improvement)

This **INVERSE correlation** (longer sequences → worse performance) proves the bottleneck is NOT FP8 dequantization (which would scale proportionally with sequence length). Instead, the bottleneck likely involves:
1. **Memory bandwidth saturation**: Long sequences exhaust memory bandwidth, causing delays
2. **Batching inefficiency**: Long sequences cause batch formation delays (cannot fit many in batch)
3. **Scheduler overhead**: β₆ exploded (+654%), suggesting scheduler struggles with long-sequence Scout requests
4. **KV cache management**: β₃ doubled (+118%), suggesting KV allocation/eviction overhead scales with sequence length

**Diagnostic Analysis**: Since the pattern is OPPOSITE (short sequences improved, long sequences failed), the FP8 hypothesis is conclusively FALSE. Investigate sequence-length-dependent bottlenecks (batching, memory, scheduler).

---

## H-robustness-fp8-generalization: β₉ Should Generalize to All FP8 Architectures

**Prediction** (from Agent 1): β₉ mechanism should generalize to all FP8 architectures:
- Scout (56 layers FP8, MoE hybrid): β₉ = 17-50μs per token per layer
- Hypothetical FP8 dense model: β₉ should scale proportionally
- Hypothetical FP8 pure MoE: β₉ should scale proportionally

**Actual Result**: β₉ = **0.1365 μs per token per layer** (124-366× smaller than predicted, essentially zero)

**Verdict**: ❌ **REJECTED**

**Evidence**:
- ❌ β₉ coefficient NOT physically plausible: 0.1365 μs vs expected 17-50 μs (124-366× too small)
- ❌ Cannot validate generalization: β₉ is zero, so there's no mechanism to generalize
- ❌ Scout improvement failed for long sequences, proving FP8 is not the bottleneck

**Causal Analysis**: The optimizer learned β₉ ≈ 0, meaning FP8 dequantization overhead is either (1) negligible in vLLM's implementation, (2) already captured by roofline's MFU factor, or (3) not the dominant Scout bottleneck. Since β₉ is zero, the generalization hypothesis is moot — there's no FP8 mechanism to test on future FP8 models.

**Recommendation**: Do not pursue FP8 dequantization as a latency term. Focus on sequence-length-dependent bottlenecks (batching, scheduler, memory).

---

## H-reversion-inflated-coefficients: β₃, β₅, β₇ Should Revert After β₉ Addition

**Prediction** (from Agent 1):
- **Iter8**: β₃ = 4.40ms, β₅ = 41.1μs, β₇ = 26.3ms (inflated, absorbing Scout error)
- **Iter9**: β₃ = **0.4-1.0ms** (10× decrease), β₅ = **10-20μs** (2-4× decrease), β₇ = **10-20ms** (1.3-2.6× decrease)

**Actual Result**:
- β₃ (KV mgmt): 4.40ms → **9.59ms** (+117.7%, DOUBLED, moving AWAY from physical)
- β₅ (MoE gating): 41.1μs → **19.8μs** (-51.9%, approaching physical 10-20μs range ✓)
- β₇ (decode overhead): 26.3ms → **11.0ms** (-58.0%, approaching physical 10-20ms range ✓)

**Verdict**: ⚠️ **PARTIAL** (2/3 reverted, 1/3 inflated further)

**Evidence**:
- ✅ **β₅ reverted**: 41.1μs → 19.8μs (-51.9%), now within physical 10-20μs range ✓
- ✅ **β₇ reverted**: 26.3ms → 11.0ms (-58.0%), now within physical 10-20ms range ✓
- ❌ **β₃ inflated further**: 4.40ms → 9.59ms (+117.7%), moving AWAY from physical 0.4-1ms range

**Causal Analysis**: Two coefficients (β₅, β₇) reverted toward physical ranges, but β₃ (KV management) DOUBLED instead of reverting. This suggests:

**Why β₅ and β₇ reverted**: The optimizer redistributed Scout error from β₅ and β₇ to OTHER terms (β₆ +654%, β₂ +343%), allowing these coefficients to approach physical values. This is GOOD — β₅ and β₇ are no longer absorbing spurious Scout error.

**Why β₃ inflated**: β₃ (KV management) doubled from 4.4ms to 9.6ms, suggesting KV overhead is being absorbed into this term. This likely reflects:
1. **Long-sequence Scout overhead**: β₃ is per-request overhead. Long-sequence Scout experiments (general-lite, reasoning-lite) failed completely (92%, 91% TTFT), and β₃ may be absorbing this sequence-length-dependent KV pressure.
2. **Real KV overhead increase**: It's possible KV management overhead is genuinely higher than initially estimated (9.6ms vs 0.4-1ms physical), especially for large models with TP=2 (cross-GPU KV synchronization).

**Diagnostic Analysis**: The partial reversion (2/3 coefficients) confirms the model is redistributing error more efficiently, but the β₃ inflation suggests a missing term related to KV management or sequence-length-dependent memory overhead.

**Recommendation**: Investigate β₃ inflation — either (1) refine β₃ basis function (add sequence-length dependency), or (2) add new term for long-sequence memory pressure / KV cache thrashing.

---

## H-data-update-exp17: New Exp17 (General-Lite) Should Show Similar Improvement

**Prediction** (from Agent 1):
- **Old exp17** (general-2, saturated server): 100% TTFT in iter8 (worst performer)
- **New exp17** (general-lite-2-1, normal server): **<48%** TTFT in iter9 (>52pp improvement, similar to other Scout experiments)

**Actual Result**: New exp17 (general-lite-2-1) = **92.28%** TTFT (failed target by 44pp)

**Verdict**: ❌ **REJECTED**

**Evidence**:
- New exp17 TTFT: 92.28% (target <48%, failed by 44pp)
- No comparison to old exp17 possible (different workload: general-lite vs general)
- New exp17 is WORST Scout experiment (92% vs 91% reasoning-lite, 58% codegen, 26% roleplay)

**Causal Analysis**: The hypothesis assumed server saturation was the primary cause of old exp17's 100% TTFT, and that clean data would improve to <48% (similar to other Scout experiments). Instead, new exp17 shows 92% TTFT, which is:
1. **Still worst Scout experiment** (92% vs 91%, 58%, 26%)
2. **NOT improved** compared to Scout reasoning-lite (91%, also long-sequence)
3. **Confirms long-sequence pattern**: Both general-lite (92%) and reasoning-lite (91%) are long-sequence workloads and both FAILED to improve.

This proves the bottleneck is **sequence-length-dependent**, NOT server-saturation-dependent. The clean data collection did not help because the underlying bottleneck (batching delay, scheduler overhead, memory pressure for long sequences) persists.

**Diagnostic Analysis**: New exp17 failure confirms the pattern seen in H-error-pattern-scout: long-sequence Scout experiments (general-lite, reasoning-lite) fail regardless of data collection quality. Investigate:
1. Batching inefficiency for long sequences (cannot fit many in batch → delayed processing)
2. Scheduler overhead scaling with sequence length (β₆ +654%)
3. Memory bandwidth saturation for long sequences
4. KV cache management overhead for long sequences (β₃ +118%)

---

## Summary Table

| Hypothesis | Prediction | Actual Result | Verdict | Key Evidence |
|------------|-----------|---------------|---------|--------------|
| **H-main** | Overall loss 155% → <75%, Scout TTFT <40%, β₉ = 17-50μs | **Loss 160.6%** (WORSE), Scout TTFT 26-92% (mixed), **β₉ = 0.14μs** (zero) | ❌ REJECTED | β₉ is zero (124-366× too small), Scout improvement workload-specific (short improved, long failed), β₆/β₂ exploded (+654%, +343%) |
| **H-ablation** | β₉ contributes >40pp to Scout | Cannot validate (β₉ is zero) | ❌ REJECTED | β₉ = 0, mechanism not active |
| **H-boundary** | β₉ = 0 for non-FP8, 95-280ms for Scout | β₉ ≈ 0 for all models (trivial), Scout mixed | ⚠️ PARTIAL | Boundary holds trivially (β₉ is zero), but Scout failed |
| **H-error-pattern** | All 4 Scout improve >52pp uniformly | Only 1/4 met target (roleplay -53pp), **OPPOSITE pattern** (short improved, long failed) | ❌ REJECTED | Inverse correlation: short sequences improved (-53pp, -34pp), long sequences failed (0pp, -8pp) |
| **H-robustness** | β₉ = 17-50μs, generalizes to all FP8 | **β₉ = 0.14μs** (zero) | ❌ REJECTED | β₉ not physically plausible, no mechanism to generalize |
| **H-reversion** | β₃ →0.4-1ms, β₅ →10-20μs, β₇ →10-20ms | β₃ **+118%** (9.6ms), β₅ **-52%** (20μs ✓), β₇ **-58%** (11ms ✓) | ⚠️ PARTIAL | 2/3 reverted (β₅, β₇), β₃ inflated further (+118%) |
| **H-data-update** | New exp17 <48% TTFT (similar to other Scout) | **92% TTFT** (failed by 44pp, worst Scout) | ❌ REJECTED | New exp17 still worst, confirms long-sequence bottleneck |

**Overall Success**: **0/7 confirmed**, **2/7 partial**, **5/7 rejected** → **FAILURE**

**Critical Verdict**: H-main (MANDATORY) is REJECTED. Iteration 9 failed to achieve any measurable improvement and actually worsened overall loss by +5.25pp.

---

## Key Learnings

**What We Learned**:

1. **FP8 hypothesis is FALSE**: β₉ converged to zero (0.14 μs vs expected 17-50 μs), proving FP8 dequantization is NOT Scout's bottleneck.

2. **Sequence-length-dependent bottleneck discovered**: Scout improvement pattern is OPPOSITE of FP8 prediction:
   - **Short-sequence Scout** (roleplay 26% TTFT, codegen 58% TTFT): Improved significantly (-53pp, -34pp)
   - **Long-sequence Scout** (general-lite 92% TTFT, reasoning-lite 91% TTFT): Failed completely (0pp, -8pp)

3. **Coefficient explosions reveal true bottlenecks**:
   - β₆ (scheduler): +654% (13ms → 99ms) — scheduler overhead scaling with long sequences
   - β₂ (TP comm): +343% (0.18 → 0.82) — TP communication overhead larger than expected
   - β₈ (MoE routing): +143% (30μs → 73μs) — MoE routing overhead higher than predicted
   - These terms absorbed Scout error, not β₉

4. **Partial coefficient reversion** (2/3):
   - β₅ (MoE gating): 41μs → 20μs (-52%), now within physical 10-20μs range ✓
   - β₇ (decode overhead): 26ms → 11ms (-58%), now within physical 10-20ms range ✓
   - β₃ (KV mgmt): 4.4ms → 9.6ms (+118%), inflated further (sequence-length dependency?)

**What We Don't Understand**:

1. **Why does Scout bottleneck scale INVERSELY with sequence length?**
   - Hypothesis predicted: Longer sequences → more FP8 overhead → larger improvements
   - Reality: Shorter sequences improved (-53pp, -34pp), longer sequences failed (0pp, -8pp)
   - Candidates: Batching inefficiency (long sequences don't fit in batch), scheduler overhead, memory bandwidth saturation, KV cache thrashing

2. **What caused β₆ and β₂ explosions?**
   - β₆ (scheduler): +654% (13ms → 99ms) — Why did scheduler overhead explode 7.5×?
   - β₂ (TP comm): +343% (0.18 → 0.82) — Why did TP communication inflate 4.5×?
   - Are these absorbing Scout long-sequence overhead? Or is there a real increase in scheduler/TP costs?

3. **Why did β₈ inflate to 73μs (above predicted 10-50μs range)?**
   - Iter8: β₈ = 30μs (within range)
   - Iter9: β₈ = 73μs (above range)
   - Is MoE routing overhead genuinely higher? Or is β₈ absorbing other Scout overhead?

4. **What is the true Scout bottleneck for long sequences?**
   - FP8 dequantization: ❌ (β₉ is zero)
   - MoE routing: Partial (β₈ inflated but only affects MoE layers, not dense layers)
   - Candidates: Batching delay, scheduler overhead (β₆), TP coordination (β₂), KV cache management (β₃), memory bandwidth
