# Iteration 8: MoE Routing Overhead Mechanism

## Motivation

Iter7 revealed a critical discovery: **Scout MoE architecture dominates the error budget**. Four Scout experiments account for 767% combined loss (49% of total error) despite representing only 27% of training data. Meanwhile, non-Scout reasoning-lite experiments improved dramatically (99% → 54-66% TTFT), confirming that clean data retraining succeeded for dense models.

**Root Cause Diagnosis from Iter7**:
- Scout general: 100% TTFT (99% in iter6) — no improvement
- Scout reasoning-lite: 98% TTFT (NEW in iter7, same clean data as Qwen/Llama-2)
- Qwen reasoning-lite: 54% TTFT (45pp better than Scout)
- Llama-2 reasoning-lite: 66% TTFT (32pp better than Scout)

**Conclusion**: The bottleneck is NOT workload or data quality — it's **Scout MoE architecture-specific overhead** not captured by current model.

### Baseline Evidence: Roofline Underestimates Scout

**CRITICAL**: `baseline_errors.json` shows roofline has a **bifurcation pattern**:

**Scout experiments** (roofline TTFT MPE):
- exp_17 (Scout general): **-99.88%** (underestimated by 100×) → predicts 1ms, reality 100ms
- exp_20 (Scout codegen): **-50.07%** (underestimated by 2×) → predicts 50ms, reality 100ms
- exp_48 (Scout reasoning): **-92.03%** (underestimated by 12×) → predicts 8ms, reality 100ms

**Dense models** (roofline TTFT MPE):
- exp_61 (Llama-3.1 codegen): **+912%** (overestimated by 10×) → predicts 1000ms, reality 100ms
- exp_63 (Mistral codegen): **+1031%** (overestimated by 11×) → predicts 1100ms, reality 100ms

**Interpretation**:
- Negative MPE (Scout) = **missing overhead** → roofline predicts too fast
- Positive MPE (dense) = phantom overhead → roofline predicts too slow (batching efficiency?)

**This is direct physical evidence that Scout MoE has overhead beyond roofline's physics!**

**Current Model Gap**: The evolved model captures MoE gating FLOPs (β₅ = 41.1ms in iter7) but does NOT capture per-token expert routing latency:
- Gating network forward pass: Router computes expert probabilities
- Expert selection: Top-k selection from 16 experts (Scout uses top-1 or top-2)
- Expert dispatch: Token reordering and routing to selected experts
- Load balancing: Dynamic expert assignment and utilization tracking
- Expert aggregation: Weighted sum of expert outputs

**Physics Missing**: vLLM's MoE implementation (fused_moe.py) has per-token routing overhead beyond the gating network FLOPs. Scout has 26 MoE layers (InterleaveMoELayerStep=26 per iter1 fix #877), and each layer incurs routing latency.

**Iter8 Strategy**: Add β₈ to capture per-token MoE routing overhead, train on all 15 experiments (including 4 Scout) to learn MoE-specific coefficient, and validate that β₈ absorbs Scout's 767% combined loss while leaving non-Scout experiments unaffected.

---

## H-main: MoE Routing Overhead Captures Scout Residual

**Prediction**:
- **Overall loss**: 155.37% → **<80%** (75pp improvement, 48% reduction)
- **TTFT RMSE**: 64.04% → **<40%** (24pp improvement, 38% reduction)
- **E2E RMSE**: 91.33% → **<50%** (41pp improvement, 45% reduction)
- **Scout TTFT error**: Avg 90% (range 79-100%) → **<50%** (>40pp improvement for all 4 Scout experiments)
- **Non-Scout experiments**: Remain stable or improve slightly (< ±10pp change from iter7)

**Quantitative Threshold**: If overall loss does NOT reduce below 100%, or if Scout TTFT does NOT improve to <70%, then H-main is REJECTED.

**Causal Mechanism**:

**Baseline Validation**: Roofline underestimates Scout by 50-99% (negative MPE = missing overhead). This proves Scout has physics-based overhead beyond current roofline model.

Scout MoE architecture has **per-token expert routing overhead** not captured by current model:

1. **Expert Routing Cost** (10-50μs per routed token) — **THIS IS THE MISSING OVERHEAD**:
   - Gating network forward pass: Input → router → expert probabilities (softmax over 16 experts)
   - Top-k expert selection: Choose k experts per token (k=1 or k=2 for Scout)
   - Expert dispatch: Reorder tokens to selected experts (may require cross-GPU communication if TP>1)
   - Load balancing: Track expert utilization, apply auxiliary loss penalties
   - Expert aggregation: Weighted sum of expert outputs per token

2. **Scout Architecture Details** (from iter1 fix #877):
   - 56 total layers: 26 MoE layers + 30 dense layers (interleaved)
   - 16 experts per MoE layer
   - Top-k routing (k likely 1 or 2 based on DeepSeek-V2/Mixtral patterns)
   - TP=2 configuration (cross-GPU expert routing)
   - FP8 dynamic quantization (may add mixed-precision overhead)

3. **Why β₅ (MoE gating FLOPs) is Insufficient**:
   - β₅ captures gating network compute time (FLOPs / peak_TFLOPS × efficiency)
   - β₅ does NOT capture routing latency: expert selection, token dispatch, load balancing
   - Iter7 β₅ = 41.1ms (increased 10× from iter6's 4.31ms), suggesting optimizer compensating but insufficient

4. **Expected β₈ Contribution** (for Scout prefill):
   - Prefill tokens: ~100-500 tokens per request
   - MoE layers: 26 layers
   - Routed tokens: 100 × 26 = 2,600 per prefill request
   - Routing overhead: 10-50μs per routed token
   - Total contribution: β₈ × 2,600 = 26-130ms per prefill request
   - **This matches Scout TTFT residual** (predicted ~10-20ms, actual 100-200ms → gap 80-180ms)

5. **Why β₈ Won't Affect Dense Models**:
   - Dense models have `NumLocalExperts = 0` in ModelConfig
   - β₈ term: `β₈ × (numMoELayers × totalTokens × numExpertsPerTok / TP)`
   - For dense models: `numMoELayers = 0` → β₈ contribution = 0
   - **Non-Scout experiments remain unaffected** (predicted <±10pp change)

**Code Citations**:

- **vLLM MoE routing**: `vllm/model_executor/layers/fused_moe/fused_moe.py:fused_experts()`
  - Line ~150-200: Expert routing implementation (gating, selection, dispatch)
  - Top-k selection: `torch.topk()` call per token
  - Expert dispatch: Token reordering via scatter/gather operations
  - Aggregation: Weighted sum of expert outputs

- **Scout model config**: `sim/models.go` (after iter1 fix #877)
  - `InterleaveMoELayerStep = 26`: 26 MoE layers interspersed with 30 dense layers
  - `NumLocalExperts = 16`: 16 experts per MoE layer
  - `NumExpertsPerTok` (to be added): Top-k routing (likely k=1 or k=2)

- **BLIS evolved model**: `sim/latency/evolved_model.go:StepTime()`
  - Line 224-243: β₅ (MoE gating FLOPs) — captures gating network compute
  - **Missing**: β₈ (per-token routing overhead) — will be added in iter8

**Diagnostic Clause**:

*If this hypothesis fails (overall loss remains >100% OR Scout TTFT >70%), it indicates:*

1. **β₈ coefficient converged to zero** → MoE routing overhead negligible, investigate alternative Scout bottlenecks:
   - FP8 dequantization overhead (mixed-precision coordination)
   - TP=2 communication overhead (cross-GPU expert routing)
   - Model config issue (InterleaveMoELayerStep/NumExpertsPerTok incorrect)

2. **β₈ coefficient converged >100μs per routed token** → Unrealistically high, investigate:
   - Absorbing other missing terms (batching delay, memory allocation)
   - Training data distribution bias (Scout experiments dominating optimization)
   - Basis function formulation issue (scaling or units incorrect)

3. **Non-Scout experiments degraded >10pp** → β₈ absorbing non-MoE error, investigate:
   - Zero-sum trade-off: helping Scout hurts non-Scout
   - Need architecture-specific models OR more training data

**Next Investigation**: If H-main fails, profile Scout MoE overhead separately with vLLM profiler to measure:
- Expert routing latency per layer (gating, selection, dispatch)
- TP communication overhead (cross-GPU expert routing)
- FP8 dequantization latency (mixed-precision coordination)

---

## H-ablation-beta8: β₈ Accounts for Majority of Scout Improvement

**Prediction**:
- **With β₈ (full model)**: Scout TTFT avg 90% → <50% (>40pp improvement)
- **Without β₈ (ablated)**: Scout TTFT avg 90% → 80-90% (<10pp improvement)
- **Difference**: β₈ contributes **>30pp** of Scout TTFT improvement

**Causal Mechanism**: β₈ is the only term that scales with MoE architecture parameters (numMoELayers, numExpertsPerTok). Removing β₈ should eliminate most Scout improvement while leaving non-Scout experiments unaffected.

**Validation**: After iter8 training, run ablation experiment:
1. Train iter8 with β₈ (9 beta coefficients)
2. Evaluate on all 15 experiments → Scout TTFT <50%
3. Re-evaluate with β₈ = 0 (ablation) → Scout TTFT 80-90%
4. Compare per-experiment: β₈ contribution = (TTFT with β₈) - (TTFT without β₈)

**Diagnostic Clause**: *If β₈ ablation shows <20pp difference, it indicates β₈ is not the primary Scout mechanism — investigate alternative bottlenecks (FP8, TP, model config).*

---

## H-boundary-dense-vs-moe: β₈ Effect Should Vanish for Dense Models

**Prediction**:
- **Dense models** (11 experiments): β₈ contribution = 0 (numMoELayers = 0)
  - Non-Scout TTFT change: <±10pp from iter7 (stable or slight improvement)
  - Non-Scout E2E change: <±10pp from iter7
- **MoE models** (4 Scout experiments): β₈ contribution = 26-130ms per request
  - Scout TTFT improvement: >40pp (90% → <50%)
  - Scout E2E improvement: >20pp (97% → <70%)

**Causal Mechanism**: β₈ basis function is `β₈ × (numMoELayers × totalTokens × numExpertsPerTok / TP)`. For dense models (numMoELayers = 0), this term vanishes mathematically, ensuring no spurious effect on non-MoE architectures.

**Validation**: After iter8 training, compare per-experiment β₈ contributions:
- Dense models: β₈ × 0 = 0μs (no contribution)
- Scout models: β₈ × (26 layers × 100 tokens × 1 expert / 2 TP) = β₈ × 1300 ≈ 13-65ms

**Diagnostic Clause**: *If non-Scout experiments degrade >10pp, it indicates β₈ is absorbing non-MoE error (zero-sum trade-off) — need architecture-specific handling or more training data.*

---

## H-error-pattern-scout: Scout Experiments Should Improve Uniformly

**Prediction**: All 4 Scout experiments should improve >40pp TTFT:
- **Scout general** (exp 17): 100% → **<60%** (>40pp improvement)
- **Scout reasoning-lite** (exp 48): 98% → **<58%** (>40pp improvement)
- **Scout codegen** (exp 20): 92% → **<52%** (>40pp improvement)
- **Scout roleplay** (exp 21): 79% → **<39%** (>40pp improvement, best Scout)

**Causal Mechanism**: All Scout workloads share the same MoE architecture (26 MoE layers, 16 experts, top-k routing). β₈ captures routing overhead proportional to numMoELayers × totalTokens, which scales uniformly across workloads. The only difference is total tokens (general > reasoning > codegen > roleplay), so improvement magnitude should correlate with sequence length.

**Validation**: After iter8 training, compare per-experiment Scout improvements:
1. Check if all 4 Scout experiments improved >40pp TTFT
2. Rank by improvement: Expect general > reasoning > codegen > roleplay (longer sequences = more MoE overhead)
3. Verify improvement correlates with total_tokens (prefill + decode)

**Diagnostic Clause**: *If any Scout experiment improves <20pp, it indicates workload-specific bottleneck beyond MoE routing (e.g., scheduler overhead, batching delay) — investigate per-workload patterns.*

---

## H-robustness-moe-generalization: β₈ Should Generalize to All MoE Architectures

**Prediction**: β₈ mechanism should generalize to all MoE architectures, not just Scout:
- **Scout** (26 MoE layers, 16 experts, top-k): β₈ = 10-50μs per routed token
- **Mixtral** (hypothetical, 32 MoE layers, 8 experts, top-2): β₈ should scale proportionally
- **DeepSeek-V3** (hypothetical, 60 layers, 256 experts, top-6): β₈ should scale proportionally

**Causal Mechanism**: Per-token expert routing overhead is a universal MoE property:
- Gating network forward pass (O(hidden_dim × num_experts) FLOPs)
- Top-k selection (O(num_experts × log(k)) per token)
- Expert dispatch (O(k) scatter/gather ops per token)
- Aggregation (O(k) weighted sum per token)

These operations scale with MoE architecture parameters (num_experts, k), captured by `numMoELayers × numExpertsPerTok`.

**Validation**:
1. After iter8 training, verify β₈ coefficient is physically plausible (10-50μs per routed token)
2. Test on future MoE models (Mixtral, DeepSeek-V3) once training data available
3. Check β₈ contribution scales proportionally with (numMoELayers × numExpertsPerTok)

**Diagnostic Clause**: *If β₈ coefficient is >100μs per routed token OR if future MoE models don't benefit from β₈, it indicates the basis function formulation is Scout-specific rather than MoE-universal — refine basis function or add architecture-specific terms.*

---

## H-decode-overhead-reversion: β₇ Should Converge Closer to 5-15ms

**Prediction**:
- **Iter7**: β₇ = 26.3ms (75% higher than 5-15ms predicted)
- **Iter8**: β₇ = **10-20ms** (closer to physical, but not full reversion to 5-15ms)

**Causal Mechanism**: Iter7's β₇ = 26.3ms likely absorbed Scout MoE error (4 experiments dominating optimization). Adding β₈ should offload Scout overhead, allowing β₇ to converge closer to physical decode overhead (output processing, TP coordination, result aggregation).

**Why Not Full Reversion to 5-15ms**: Other missing terms (batching delay, memory allocation, E2E residual) may still inflate β₇ slightly. Full reversion would require additional terms beyond β₈.

**Validation**: After iter8 training, compare β₇:
- Iter7: β₇ = 26.3ms
- Iter8: β₇ = 10-20ms (expect 20-40% reduction)
- If β₇ < 10ms: Decode overhead lower than expected OR absorbed by other terms
- If β₇ > 25ms: Still absorbing non-decode error (insufficient MoE capture OR other missing terms)

**Diagnostic Clause**: *If β₇ remains >25ms after β₈ addition, it indicates β₈ is insufficient to fully capture Scout overhead OR other missing terms (batching delay, memory allocation) need separate basis functions.*

---

## Summary of Predictions

| Hypothesis | Key Prediction | Success Threshold | Diagnostic If Failed |
|------------|----------------|-------------------|----------------------|
| **H-main** | Overall loss 155% → <80% | Loss <100% AND Scout TTFT <70% | β₈ coefficient ~0 OR >100μs OR non-Scout degraded >10pp |
| **H-ablation** | β₈ contributes >30pp to Scout improvement | Ablation shows >20pp difference | Alternative Scout bottleneck (FP8, TP, config) |
| **H-boundary** | β₈ = 0 for dense, 26-130ms for Scout | Non-Scout stable (<±10pp) | Zero-sum trade-off, need arch-specific handling |
| **H-error-pattern** | All 4 Scout improve >40pp TTFT | Each Scout <60% TTFT | Workload-specific bottleneck beyond MoE |
| **H-robustness** | β₈ generalizes to all MoE architectures | β₈ = 10-50μs per routed token | Scout-specific formulation, refine basis function |
| **H-decode-overhead** | β₇ converges to 10-20ms | β₇ = 10-20ms (vs iter7's 26.3ms) | β₈ insufficient OR other missing terms |

**Overall Success Criteria**: At least 4/6 hypotheses confirmed (✓) with H-main MANDATORY.

---

## Expected Coefficient Convergence (Iter8)

Based on iter7 results and β₈ addition:

| Coefficient | Iter7 | Iter8 Expected | Rationale |
|-------------|-------|----------------|-----------|
| α₀ (base) | 1.32ms | 1.0-2.0ms | Stable, minimal change expected |
| α₁ (input token) | 118μs | 100-150μs | Stable, achieved physical plausibility in iter7 |
| α₂ (output token) | 91μs | 60-100μs | May improve slightly as β₈ removes Scout pressure |
| β₀ (prefill compute) | 0.191 | 0.15-0.25 | Stable, expected range |
| β₁ (decode memory) | 1.108 | 1.00-1.15 | Stable after iter7 reversion ✓ |
| β₂ (TP comm) | 0.185 | 0.20-0.35 | May increase as β₈ removes Scout TP pressure |
| β₃ (KV mgmt) | 0.00440 | 0.0004-0.001 | Should revert after absorbing Scout error in iter7 |
| β₄ (decode compute) | 0.713 | 0.70-0.90 | Stable after iter7 reversion ✓ |
| β₅ (MoE gating) | 0.0411 | 0.010-0.020 | Should decrease as β₈ offloads routing overhead |
| β₆ (scheduler) | 0.0132 | 0.015-0.030 | May increase slightly |
| β₇ (decode overhead) | 0.0263 | 0.010-0.020 | Should decrease 20-40% as β₈ offloads Scout error |
| **β₈ (MoE routing)** | **N/A** | **0.000010-0.000050** | **NEW: 10-50μs per routed token** |

**Note**: β₈ is in **microseconds per routed token**, so coefficient in seconds is 0.000010-0.000050 (converted from 10-50μs).

---

## Experimental Validation Plan

After iter8 training, Agent 3 should run these validation experiments:

### 1. Ablation Study (H-ablation)
- Train iter8 with β₈ (9 beta coefficients)
- Evaluate on all 15 experiments
- Re-evaluate with β₈ = 0 (ablation)
- Compare per-experiment: TTFT with β₈ vs TTFT without β₈

### 2. Per-Experiment β₈ Contribution Analysis (H-boundary)
- Compute β₈ contribution per experiment: `β₈ × (numMoELayers × totalTokens × numExpertsPerTok / TP)`
- Dense models: Verify β₈ contribution = 0μs
- Scout models: Verify β₈ contribution = 26-130ms per request

### 3. Scout Improvement Uniformity (H-error-pattern)
- Compare all 4 Scout experiments: TTFT improvement (iter8 - iter7)
- Rank by improvement: Expect correlation with total_tokens
- Verify all improved >40pp TTFT

### 4. Coefficient Stability (H-decode-overhead)
- Compare iter7 vs iter8 coefficients: α₁, α₂, β₁, β₄, β₇
- Verify β₇ decreased 20-40% (26.3ms → 10-20ms)
- Verify β₃, β₅ decreased (stopped absorbing Scout error)

### 5. Non-Scout Stability (H-boundary)
- Compare iter7 vs iter8 non-Scout experiments (11 experiments)
- Verify TTFT change <±10pp, E2E change <±10pp
- Check for zero-sum trade-off (Scout improvement vs non-Scout degradation)

---

## Risk Assessment

**Primary Risk**: Zero-sum trade-off — helping Scout may hurt non-Scout experiments if optimization prioritizes Scout (49% of error budget).

**Mitigation**:
1. β₈ basis function mathematically vanishes for dense models (numMoELayers = 0)
2. If zero-sum observed, consider architecture-specific models OR exclude Scout from global training

**Secondary Risk**: β₈ formulation incorrect (wrong scaling, wrong units, wrong architecture parameters).

**Mitigation**:
1. After iter8 training, validate β₈ coefficient is physically plausible (10-50μs per routed token)
2. Profile Scout MoE overhead separately to verify β₈ matches measured routing latency
3. If β₈ converged to zero OR >100μs, refine basis function or investigate alternative bottlenecks

**Tertiary Risk**: Scout bottleneck is NOT MoE routing but alternative mechanism (FP8 dequantization, TP communication, model config).

**Mitigation**:
1. If H-main fails (Scout TTFT >70%), investigate diagnostic clause alternatives
2. Profile Scout with vLLM profiler to isolate bottleneck (routing vs FP8 vs TP)
3. Add architecture-specific terms (FP8 overhead, TP coordination) in iter9 if needed

---

## Success Definition

**Tier 1 (Full Success)**:
- Overall loss <80% ✓
- TTFT RMSE <40% ✓
- E2E RMSE <50% ✓
- All 4 Scout experiments <60% TTFT ✓
- Non-Scout experiments stable (<±10pp change) ✓
- **At least 5/6 hypotheses confirmed** ✓

**Tier 2 (Partial Success)**:
- Overall loss <100% (significant improvement, but not target)
- Scout experiments <70% TTFT (major improvement, but not <60%)
- Non-Scout experiments stable
- **At least 3/6 hypotheses confirmed**
- **Proceed to iter9 with refined β₈ OR additional MoE terms**

**Tier 3 (Failure)**:
- Overall loss >120% (minimal improvement)
- Scout experiments >80% TTFT (little to no improvement)
- **<3/6 hypotheses confirmed**
- **Diagnostic**: Profile Scout separately, investigate alternative bottlenecks (FP8, TP, config), consider architecture-specific models

---
