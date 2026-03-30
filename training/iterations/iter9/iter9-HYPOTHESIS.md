# Iteration 9: FP8 Dequantization Overhead Mechanism

## Motivation

**Critical Discovery from Iter8**: β₈ (MoE routing overhead) converged to 30μs per routed token (physically plausible, within predicted 10-50μs range) and contributes ~39ms per Scout prefill request. However, **Scout TTFT errors remained completely unchanged** (79-100% APE, 0pp improvement from iter7). Overall loss stayed at 155.35% (vs iter7's 155.37%). This proves β₈ captures a REAL mechanism but is INSUFFICIENT.

**Gap Analysis**:
- Roofline underestimates Scout general by **-99.88% MPE** (predicts 0.12ms, actual ~100ms)
- Missing overhead: ~99.88ms per Scout prefill request
- β₈ contribution: 39ms (only 39% of the gap)
- **Remaining gap: 61ms (61% of missing overhead)**

**Why FP8 Dequantization?**

Scout is the ONLY model in training data using **FP8 dynamic quantization** (all others use FP16/BF16). This is a smoking gun: if Scout fails uniformly across workloads while dense models succeed, and Scout has unique FP8 + MoE architecture, the bottleneck is likely architecture-specific.

**Evidence Trail**:
1. **Architecture-specific**: All 4 Scout experiments fail uniformly (79-100% TTFT) regardless of workload type or clean data
2. **FP8 unique to Scout**: Scout uses `torch.float8_e4m3fn` dynamic quantization (HuggingFace config: `"quant_method": "fp8"`)
3. **Roofline missing overhead**: -99% MPE on Scout = 100ms missing overhead, β₈ only closes 39ms
4. **β₃, β₅, β₇ inflation persists**: These coefficients remain inflated (4.4ms, 41μs, 26ms vs physical 0.4-1ms, 10-20μs, 10-20ms), suggesting they're still absorbing Scout error from a different source than β₈

**Iter9 Strategy**: Add β₉ to capture per-token FP8 dequantization overhead (mixed-precision coordination between FP8 weights and FP16/BF16 activations), train on all 15 experiments (including updated exp17 with clean general-lite data), and validate that β₉ absorbs Scout's remaining 61ms gap while leaving non-FP8 experiments unaffected.

---

## H-main: FP8 Dequantization Overhead Captures Scout Residual

**Prediction**:
- **Overall loss**: 155.35% → **<75%** (80pp improvement, 52% reduction)
- **TTFT RMSE**: 63.98% → **<35%** (29pp improvement, 45% reduction)
- **E2E RMSE**: 91.37% → **<45%** (46pp improvement, 50% reduction)
- **Scout TTFT error**: Avg 92% (range 79-100%) → **<40%** (>52pp improvement for all 4 Scout experiments)
- **Non-FP8 experiments**: Remain stable or improve slightly (< ±10pp change from iter8)
- **β₃, β₅, β₇ reversion**: β₃ (4.4ms → 0.4-1ms), β₅ (41μs → 10-20μs), β₇ (26ms → 10-20ms) as β₉ offloads Scout error

**Quantitative Threshold**: If overall loss does NOT reduce below 95%, or if Scout TTFT does NOT improve to <60%, then H-main is REJECTED.

**Causal Mechanism**:

**Physics Grounding**: FP8 dynamic quantization introduces per-token mixed-precision coordination overhead not captured by current model:

1. **FP8 Weight Dequantization** (10-30μs per token per layer):
   - Weights stored as `float8_e4m3fn` (1 byte/param)
   - Per-layer dequantization: FP8 → FP16/BF16 before matmul
   - Dynamic quantization: scales computed per-tensor (additional overhead vs static quantization)
   - Code: `vllm/model_executor/layers/quantization/fp8.py:Fp8LinearMethod.apply()` line ~100-150

2. **Mixed-Precision Coordination** (5-15μs per token per layer):
   - FP8 weights × FP16 activations → FP16 output (precision mismatch handling)
   - Tensor core mixed-precision mode requires additional synchronization
   - Dynamic scaling factor application per layer
   - Code: `vllm/model_executor/layers/linear.py:UnquantizedLinearMethod.apply_weights()` line ~200-250

3. **Quantization Scale Management** (2-5μs per token per layer):
   - Per-tensor scale factors stored and applied dynamically
   - Scale recomputation on every forward pass (dynamic quantization)
   - Cross-GPU scale synchronization when TP > 1
   - Code: `vllm/model_executor/layers/quantization/fp8.py:Fp8Config` line ~50-80

4. **Why β₅ (MoE gating) and current model insufficient**:
   - Current model captures compute time via FLOPs / (peak_TFLOPS × MFU)
   - FP8 dequantization is a **per-weight-access overhead**, not a compute operation
   - Happens BEFORE tensor cores are utilized (preprocessing step)
   - Cannot be modeled as efficiency factor (MFU) because it's a fixed latency per token

5. **Expected β₉ Contribution** (for Scout prefill):
   - Scout: 56 total layers (26 MoE + 30 dense), all use FP8 weights
   - Prefill tokens: ~100-500 tokens per request
   - FP8 overhead: 17-50μs per token per layer (dequant 10-30μs + coordination 5-15μs + scale mgmt 2-5μs)
   - Per-token total: 17-50μs × 56 layers = 950-2800μs per token
   - Per-request total: 950-2800μs × 100 tokens = 95-280ms per prefill request
   - **This matches Scout TTFT residual** (missing 61ms after β₈, actual full gap 100ms)

6. **Why β₉ Won't Affect Non-FP8 Models**:
   - Non-FP8 models have `BytesPerParam = 2` (FP16/BF16, no dequantization)
   - β₉ term: `β₉ × totalTokens × numLayers × isFP8` where `isFP8 = (BytesPerParam == 1.0)`
   - For non-FP8 models: `isFP8 = 0` → β₉ contribution = 0
   - **Non-FP8 experiments remain unaffected** (predicted <±10pp change)

**Code Citations**:

- **vLLM FP8 quantization**: `vllm/model_executor/layers/quantization/fp8.py`
  - Line ~100-150: `Fp8LinearMethod.apply()` — per-layer weight dequantization
  - Line ~50-80: `Fp8Config` — dynamic scale management
  - Line ~200-250: Mixed-precision coordination (FP8 weights × FP16 activations)

- **Scout model config**: HuggingFace `RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic`
  - `config.json`: `"quant_method": "fp8"`, `"torch_dtype": "float8_e4m3fn"`
  - 56 total layers (26 MoE + 30 dense), ALL use FP8 weights
  - TP=2 configuration (cross-GPU scale synchronization)

- **BLIS model config**: `sim/models.go` (ModelConfig struct)
  - `BytesPerParam = 1.0` for FP8 models (vs 2.0 for FP16/BF16)
  - `EffectiveWeightBytesPerParam()` distinguishes weight vs activation precision

**Diagnostic Clause**:

*If this hypothesis fails (overall loss remains >95% OR Scout TTFT >60%), it indicates:*

1. **β₉ coefficient converged to zero** → FP8 dequantization overhead negligible, investigate alternative Scout bottlenecks:
   - TP=2 MoE expert routing coordination (cross-GPU expert dispatch)
   - Scout model config error (InterleaveMoELayerStep=26, NumExpertsPerTok, or layer counts incorrect)
   - Batching inefficiency for hybrid MoE+dense architecture
   - Framework overhead specific to Scout (vLLM MoE + FP8 interaction)

2. **β₉ coefficient converged >100μs per token per layer** → Unrealistically high, investigate:
   - Absorbing other missing terms (batching delay, TP coordination beyond β₂)
   - Training data bias (Scout experiments dominating optimization due to high error)
   - Basis function formulation issue (scaling factor incorrect, double-counting with existing terms)

3. **β₉ is plausible (10-50μs per token per layer) but Scout errors remain >60%** → FP8 overhead is correct but INSUFFICIENT, investigate:
   - FP8 overhead is ONE component, but other mechanisms also missing (TP coordination, batching, config)
   - Need iter10 with additional Scout-specific term (e.g., β₁₀ for TP MoE coordination)
   - Consider architecture-specific models (separate model for MoE+FP8 vs dense)

4. **Non-FP8 experiments degraded >10pp** → β₉ absorbing non-FP8 error (zero-sum trade-off), investigate:
   - Basis function formulation allows non-zero contribution for non-FP8 models (isFP8 flag incorrect)
   - Training dynamics prioritizing Scout over dense models
   - Need architecture-specific training split

**Next Investigation**: If H-main fails, profile Scout FP8 overhead separately with vLLM profiler + CUDA events:
```bash
python -m vllm.profiler --model RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic \
  --workload general-lite --batch-size 1 --tp 2 --profile-fp8-dequant
```
Measure: Per-layer dequantization time, scale management overhead, mixed-precision coordination latency

---

## H-ablation-beta9: β₉ Accounts for Majority of Scout Improvement

**Prediction**:
- **With β₉ (full model)**: Scout TTFT avg 92% → <40% (>52pp improvement)
- **Without β₉ (ablated)**: Scout TTFT avg 92% → 80-92% (<12pp improvement, similar to iter8)
- **Difference**: β₉ contributes **>40pp** of Scout TTFT improvement

**Causal Mechanism**: β₉ is the only term that scales with FP8 architecture (BytesPerParam == 1.0). Removing β₉ should eliminate Scout-specific improvement while leaving non-FP8 experiments unaffected.

**Validation**: After iter9 training, run ablation experiment:
1. Train iter9 with β₉ (10 beta coefficients: β₀-β₈ + β₉)
2. Evaluate on all 15 experiments → Scout TTFT <40%
3. Re-evaluate with β₉ = 0 (ablation) → Scout TTFT 80-92% (similar to iter8)
4. Compare per-experiment: TTFT improvement = (TTFT without β₉) - (TTFT with β₉)

**Diagnostic Clause**: *If β₉ ablation shows <30pp difference, it indicates β₉ is not the primary Scout mechanism — investigate alternative bottlenecks (TP coordination, batching inefficiency, model config errors).*

---

## H-boundary-fp8-vs-non-fp8: β₉ Effect Should Vanish for Non-FP8 Models

**Prediction**:
- **Non-FP8 models** (11 experiments: all dense models use FP16/BF16): β₉ contribution = 0 (isFP8 = 0)
  - Non-FP8 TTFT change: <±10pp from iter8 (stable or slight improvement)
  - Non-FP8 E2E change: <±10pp from iter8
- **FP8 models** (4 Scout experiments): β₉ contribution = 95-280ms per request
  - Scout TTFT improvement: >52pp (92% → <40%)
  - Scout E2E improvement: >30pp (97% → <65%)

**Causal Mechanism**: β₉ basis function is `β₉ × totalTokens × numLayers × isFP8` where `isFP8 = 1 if BytesPerParam == 1.0 else 0`. For non-FP8 models (BytesPerParam = 2.0 for FP16/BF16), this term vanishes mathematically, ensuring no spurious effect on non-FP8 architectures.

**Validation**: After iter9 training, compare per-experiment β₉ contributions:
- Non-FP8 models: β₉ × 0 = 0μs (no contribution)
- Scout models: β₉ × (100 tokens × 56 layers × 1) = β₉ × 5600 ≈ 95-280ms per request (if β₉ = 17-50μs per token per layer)

**Diagnostic Clause**: *If non-FP8 experiments degrade >10pp, it indicates β₉ is absorbing non-FP8 error (zero-sum trade-off) — check basis function formulation (isFP8 flag implementation) or consider architecture-specific models.*

---

## H-error-pattern-scout: Scout Experiments Should Improve Uniformly

**Prediction**: All 4 Scout experiments should improve >52pp TTFT (proportional to sequence length):
- **Scout general-lite** (exp 17, NEW clean data): 100% → **<48%** (>52pp improvement, longest sequence)
- **Scout reasoning-lite** (exp 48): 98% → **<46%** (>52pp improvement, long sequence)
- **Scout codegen** (exp 20): 92% → **<40%** (>52pp improvement, moderate sequence)
- **Scout roleplay** (exp 21): 79% → **<27%** (>52pp improvement, shortest sequence, best Scout)

**Causal Mechanism**: All Scout workloads share the same FP8 architecture (56 layers, all FP8 weights). β₉ captures dequantization overhead proportional to `totalTokens × numLayers`, which scales uniformly across workloads. The only difference is total tokens (general > reasoning > codegen > roleplay), so improvement magnitude should correlate with sequence length.

**Validation**: After iter9 training, compare per-experiment Scout improvements:
1. Check if all 4 Scout experiments improved >52pp TTFT
2. Rank by improvement: Expect general-lite ≥ reasoning-lite ≥ codegen ≥ roleplay (longer sequences = more FP8 overhead)
3. Verify improvement correlates with (prefill_tokens + decode_tokens)

**Diagnostic Clause**: *If any Scout experiment improves <30pp, it indicates workload-specific bottleneck beyond FP8 dequantization (e.g., batching delay for long sequences, TP coordination issues) — investigate per-workload patterns.*

---

## H-robustness-fp8-generalization: β₉ Should Generalize to All FP8 Architectures

**Prediction**: β₉ mechanism should generalize to all FP8 architectures, not just Scout:
- **Scout** (56 layers FP8, MoE hybrid): β₉ = 17-50μs per token per layer
- **Hypothetical FP8 dense model** (e.g., Llama-3.1-70B FP8): β₉ should scale proportionally (same per-token overhead, more layers)
- **Hypothetical FP8 pure MoE** (e.g., Mixtral FP8): β₉ should scale proportionally

**Causal Mechanism**: FP8 dequantization overhead is a universal FP8 property (torch.float8_e4m3fn → torch.float16 conversion):
- Per-layer weight dequantization: O(weights_per_layer) operation before matmul
- Mixed-precision coordination: Independent of MoE/dense architecture
- Dynamic scale management: Scales with number of layers, not architecture type

**Validation**:
1. After iter9 training, verify β₉ coefficient is physically plausible (17-50μs per token per layer)
2. Test on future FP8 models (Llama-3.1-70B FP8, Mixtral FP8) once training data available
3. Check β₉ contribution scales proportionally with (totalTokens × numLayers × isFP8)

**Diagnostic Clause**: *If β₉ coefficient is >100μs per token per layer OR if future FP8 models don't benefit from β₉, it indicates the basis function formulation is Scout-specific rather than FP8-universal — refine basis function or add architecture-specific terms.*

---

## H-reversion-inflated-coefficients: β₃, β₅, β₇ Should Revert After β₉ Addition

**Prediction**:
- **Iter8**: β₃ = 4.40ms, β₅ = 41.1μs, β₇ = 26.3ms (inflated, absorbing Scout error)
- **Iter9**: β₃ = **0.4-1.0ms** (10× decrease), β₅ = **10-20μs** (2-4× decrease), β₇ = **10-20ms** (1.3-2.6× decrease)

**Causal Mechanism**: Iter8's β₃, β₅, β₇ inflation likely absorbed Scout FP8 error (β₈ only closed 39ms of 100ms gap). After adding β₉ to capture remaining 61ms, these coefficients should revert to physical values:
- β₃ (KV mgmt): Physical 0.4-0.5ms (PagedAttention block allocation per request)
- β₅ (MoE gating): Physical 10-20μs (gating network FLOPs)
- β₇ (decode overhead): Physical 10-20ms (output processing, TP coordination)

**Validation**: After iter9 training, compare iter8 vs iter9 coefficients:
- β₃: 4.40ms → 0.4-1.0ms (expect ≥4× decrease)
- β₅: 41.1μs → 10-20μs (expect ≥2× decrease)
- β₇: 26.3ms → 10-20ms (expect ≥1.3× decrease)

If reversion occurs, confirms β₉ offloaded excess error from β₃, β₅, β₇. If not, investigate:
- β₉ insufficient (FP8 overhead is partial, other terms still needed)
- Alternative error sources (batching delay, TP coordination beyond β₂)

**Diagnostic Clause**: *If any coefficient remains inflated (β₃ >2ms OR β₅ >30μs OR β₇ >25ms), it indicates β₉ is insufficient to fully capture Scout overhead OR other missing terms absorbing error — investigate complementary mechanisms.*

---

## H-data-update-exp17: New Exp17 (General-Lite) Should Show Similar Improvement

**Prediction**:
- **Old exp17** (general-2, saturated server): 100% TTFT in iter8 (worst performer)
- **New exp17** (general-lite-2-1, normal server): **<48%** TTFT in iter9 (>52pp improvement, similar to other Scout experiments)

**Rationale**: After iter8 analysis, exp17 was replaced with clean general-lite data (collected 2026-03-30 under normal operating conditions vs saturated server). The new data should have lower variance and cleaner signal, but Scout FP8 bottleneck persists (architecture-specific, not workload or server condition).

**Causal Mechanism**: Server saturation inflates TTFT (queueing delay, resource contention) but doesn't change the fundamental Scout FP8 overhead. β₉ captures per-token dequantization latency independent of server load. New exp17 should improve similarly to other Scout experiments once server noise removed.

**Validation**: After iter9 training, compare new exp17 TTFT vs other Scout experiments:
- New exp17 TTFT: <48% (expect similar to Scout reasoning-lite at <46%)
- Improvement magnitude: >52pp (same as other Scout experiments)
- Verify new exp17 doesn't show anomalous behavior (outlier error)

**Diagnostic Clause**: *If new exp17 TTFT remains >60% after β₉ addition, it indicates:*
1. New data still has quality issues (revalidate collection process)
2. General-lite workload has unique bottleneck vs other Scout workloads
3. β₉ formulation insufficient for longest-sequence workload (general-lite has ~500 prefill tokens vs ~100-200 for others)

---

## Summary of Predictions

| Hypothesis | Key Prediction | Success Threshold | Diagnostic If Failed |
|------------|----------------|-------------------|----------------------|
| **H-main** | Overall loss 155% → <75%, Scout TTFT <40% | Loss <95% AND Scout TTFT <60% | β₉ ~0 OR >100μs OR β₉ plausible but insufficient OR non-FP8 degraded >10pp |
| **H-ablation** | β₉ contributes >40pp to Scout improvement | Ablation shows >30pp difference | Alternative Scout bottleneck (TP, batching, config) |
| **H-boundary** | β₉ = 0 for non-FP8, 95-280ms for Scout | Non-FP8 stable (<±10pp) | Zero-sum trade-off, check isFP8 flag implementation |
| **H-error-pattern** | All 4 Scout improve >52pp TTFT uniformly | Each Scout <50% TTFT | Workload-specific bottleneck beyond FP8 |
| **H-robustness** | β₉ generalizes to all FP8 architectures | β₉ = 17-50μs per token per layer | Scout-specific formulation, refine basis function |
| **H-reversion** | β₃, β₅, β₇ revert to physical ranges | β₃ <2ms, β₅ <30μs, β₇ <25ms | β₉ insufficient OR other missing terms |
| **H-data-update** | New exp17 (general-lite) <48% TTFT | New exp17 improvement >52pp, similar to other Scout | Data quality issue OR general-lite bottleneck OR β₉ insufficient for longest sequences |

**Overall Success Criteria**: At least 5/7 hypotheses confirmed (✓) with H-main MANDATORY.

---

## Expected Coefficient Convergence (Iter9)

Based on iter8 results and β₉ addition:

| Coefficient | Iter8 | Iter9 Expected | Rationale |
|-------------|-------|----------------|-----------|
| α₀ (base) | 1.32ms | 1.0-2.0ms | Stable, minimal change expected |
| α₁ (input token) | 117.6μs | 100-150μs | Stable, achieved physical plausibility |
| α₂ (output token) | 90.5μs | 60-120μs | May improve as β₉ removes Scout pressure |
| β₀ (prefill compute) | 0.1912 | 0.15-0.25 | Stable, expected range |
| β₁ (decode memory) | 1.1076 | 1.00-1.15 | Stable after iter7 reversion ✓ |
| β₂ (TP comm) | 0.1846 | 0.20-0.35 | May increase as β₉ removes Scout TP pressure |
| β₃ (KV mgmt) | 0.00440 | 0.0004-0.001 | **Should revert 10× after β₉ offloads Scout error** |
| β₄ (decode compute) | 0.7132 | 0.70-0.90 | Stable after iter7 reversion ✓ |
| β₅ (MoE gating) | 0.04112 | 0.010-0.020 | **Should decrease 2-4× as β₉ offloads Scout error** |
| β₆ (scheduler) | 0.01316 | 0.015-0.030 | May increase slightly |
| β₇ (decode overhead) | 0.02626 | 0.010-0.020 | **Should decrease 1.3-2.6× as β₉ offloads Scout error** |
| β₈ (MoE routing) | 0.00003 | 0.000025-0.000035 | Stable (real but insufficient, keep for Scout MoE overhead) |
| **β₉ (FP8 dequant)** | **N/A** | **0.000017-0.000050** | **NEW: 17-50μs per token per layer** |

**Note**: β₉ is in **seconds per token per layer**, so coefficient in seconds is 0.000017-0.000050 (converted from 17-50μs).

---

## Risk Assessment

**Primary Risk**: β₉ insufficient — FP8 overhead is ONE component, but other mechanisms also missing (TP coordination, batching inefficiency).

**Mitigation**:
1. If iter9 achieves partial success (loss <100%, Scout <60% TTFT), β₉ is correct but needs complementary term (β₁₀ for TP MoE coordination or batching)
2. If H-reversion fails (β₃, β₅, β₇ remain inflated), confirms additional terms needed
3. Prepare iter10 design for complementary Scout terms

**Secondary Risk**: Model config error — InterleaveMoELayerStep=26, NumExpertsPerTok, or Scout layer counts incorrect, causing basis function underestimation.

**Mitigation**:
1. Before training, validate Scout model config against HuggingFace config.json:
   - `num_hidden_layers` (should be 56)
   - `num_local_experts` (should be 16)
   - `num_experts_per_tok` (should be 1 or 2)
   - `moe_layer_indices` or equivalent (should indicate 26 MoE layers)
2. If config wrong, fix before iter9 training (don't add β₉ with wrong config)

**Tertiary Risk**: Architecture-specific models needed — Universal model continues struggling with Scout due to zero-sum trade-offs.

**Mitigation**:
1. If iter9 fails (loss >120%, Scout >70%) AND non-FP8 experiments degrade >10pp, consider architecture-specific split:
   - **Dense model**: Trained on 11 non-FP8 experiments (β₀-β₇, no MoE or FP8 terms)
   - **MoE+FP8 model**: Trained on 4 Scout experiments (β₀-β₇, β₈ MoE routing, β₉ FP8 dequant, β₁₀ TP coordination)
2. Deploy: Route requests to appropriate model based on architecture metadata (NumLocalExperts, BytesPerParam)

---

## Success Definition

**Tier 1 (Full Success)**:
- Overall loss <75% ✓
- TTFT RMSE <35% ✓
- E2E RMSE <45% ✓
- All 4 Scout experiments <50% TTFT ✓
- Non-FP8 experiments stable (<±10pp change) ✓
- β₉ coefficient physically plausible (17-50μs per token per layer) ✓
- β₃, β₅, β₇ revert to physical ranges ✓
- **At least 6/7 hypotheses confirmed** ✓

**Tier 2 (Partial Success)**:
- Overall loss <95% (significant improvement, but not target)
- Scout experiments <60% TTFT (major improvement, but not <40%)
- Non-FP8 experiments stable
- β₉ coefficient plausible
- At least 2 of β₃, β₅, β₇ revert
- **At least 4/7 hypotheses confirmed**
- **Proceed to iter10 with complementary Scout terms (TP coordination, batching)**

**Tier 3 (Failure)**:
- Overall loss >120% (minimal improvement)
- Scout experiments >75% TTFT (<17pp improvement)
- β₉ coefficient implausible OR zero
- **<4/7 hypotheses confirmed**
- **Diagnostic**: Validate Scout model config first, then profile FP8 overhead separately, consider architecture-specific models

**If Tier 3**: Either (1) Scout model config wrong (fix and retry iter9), (2) FP8 overhead formulation incorrect (refine β₉ basis function), or (3) Universal model cannot handle Scout (split into architecture-specific models).

---

## Important Note: Data Update

**⚠️ CRITICAL**: After iter8 analysis was completed, **exp17 was replaced with clean data**:

- **Old exp17**: `17-llama-4-scout-17b-16e-tp2-general-2` — Collected under **saturated server conditions** (used in iter0-iter8)
- **New exp17**: `17-llama-4-scout-17b-16e-tp2-general-lite-2-1` — Collected on 2026-03-30 under **normal operating conditions** (reduced workload intensity)

**Impact on iter9**:
- ✅ Iter9 will train on 15 experiments with **NEW exp17** (general-lite-2-1)
- ✅ New exp17 should have cleaner signal (no server saturation noise)
- ✅ Scout bottleneck persists (architecture-specific, not server condition)
- ✅ Expect all 4 Scout experiments to improve uniformly with β₉ addition

**Rationale**: Similar to the reasoning → reasoning-lite replacement in iter7, Scout general workload caused server saturation. The new general-lite workload mirrors the strategy that fixed reasoning experiments (reduced intensity, normal server operation).
