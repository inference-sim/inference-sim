# Iteration 9: FP8 Dequantization Overhead Mechanism

## Status
✅ **Design Complete** — Ready for Agent 2 (Orchestration) to execute optimization

## Key Discovery from Iter8

**Critical Finding**: β₈ (MoE routing overhead) converged to 30μs per routed token (physically plausible) and contributes ~39ms per Scout prefill request, but Scout TTFT errors remained COMPLETELY UNCHANGED (79-100% APE, 0pp improvement). This proves β₈ captures a REAL mechanism but is INSUFFICIENT — Scout's bottleneck is 100-200ms, not 39ms.

**Gap Analysis**:
- Roofline underestimates Scout by **-99.88% MPE** (missing 99.88ms overhead)
- β₈ contribution: 39ms (only 39% of the gap)
- **Remaining gap: 61ms** (61% of missing overhead)
- β₃, β₅, β₇ remain inflated (4.4ms, 41μs, 26ms vs physical 0.4-1ms, 10-20μs, 10-20ms)

## Iter9 Hypothesis: FP8 Dequantization Overhead

**Root Cause**: Scout is the ONLY FP8 model (torch.float8_e4m3fn, all others FP16/BF16). FP8 dynamic quantization introduces per-token dequantization overhead:

1. **Weight dequantization** (10-30μs per token per layer): FP8 → FP16/BF16 conversion before matmul
2. **Mixed-precision coordination** (5-15μs per token per layer): FP8 weights × FP16 activations
3. **Dynamic scale management** (2-5μs per token per layer): Per-tensor scale factors per layer

**Total overhead**: 17-50μs per token per layer × 56 layers = 950-2800μs per token

**Expected β₉ contribution**: 950-2800μs × 100 tokens = **95-280ms per Scout prefill request**

## Predictions

### H-main (MANDATORY)
- **Overall loss**: 155.35% → **<75%** (80pp improvement, 52% reduction)
- **Scout TTFT**: Avg 92% → **<40%** (>52pp improvement for all 4 Scout experiments)
- **β₃, β₅, β₇ reversion**: β₃ (4.4ms → 0.4-1.0ms), β₅ (41μs → 10-20μs), β₇ (26ms → 10-20ms)
- **Non-FP8 stable**: <±10pp change (β₉ = 0 for non-FP8 models with BytesPerParam = 2.0)

**Success Threshold**: Overall loss <95% AND Scout TTFT <60% (else H-main REJECTED)

### Additional Hypotheses
- **H-ablation**: β₉ contributes >40pp to Scout improvement
- **H-boundary**: β₉ = 0 for non-FP8, 95-280ms for Scout
- **H-error-pattern**: All 4 Scout improve >52pp TTFT uniformly
- **H-robustness**: β₉ = 17-50μs per token per layer (generalizes to all FP8 models)
- **H-reversion**: β₃, β₅, β₇ revert to physical ranges
- **H-data-update**: New exp17 (general-lite) <48% TTFT (clean data from 2026-03-30)

**Overall Success**: At least 5/7 hypotheses confirmed (✓) with H-main MANDATORY

## Files Generated

1. **iter9-HYPOTHESIS.md** (7 hypotheses with quantitative predictions, causal mechanisms, diagnostic clauses)
2. **iteration_manifest.yaml** (metadata, reasoning, timestamp)
3. **coefficient_bounds.yaml** (10 beta coefficients with warm-start from iter8 optimal)
4. **sim/latency/evolved_model.go** (β₉ term added to StepTime method)

## Basis Function

```go
// β₉ × (totalTokens × numLayers × isFP8)
// where isFP8 = 1 if BytesPerParam == 1.0 else 0

if m.modelConfig.EffectiveWeightBytesPerParam() == 1.0 {
    totalTokens := float64(totalPrefillTokens + numDecodeRequests)
    numLayers := float64(m.modelConfig.NumLayers)
    tokenLayerOps := totalTokens * numLayers
    fp8DequantTimeUs = tokenLayerOps * m.Beta[9] * 1e6
}
```

**For Scout**: β₉ × (100 tokens × 56 layers × 1) = β₉ × 5600 ≈ 95-280ms
**For non-FP8**: β₉ × (... × 0) = 0 (unaffected)

## Training Data

15 experiments with **NEW exp17** (general-lite-2-1, clean data from 2026-03-30):
- Old exp17 collected under saturated server conditions (used in iter0-iter8)
- New exp17 collected under normal operating conditions (reduced workload intensity)
- Scout bottleneck persists (architecture-specific FP8, not server condition)

## Expected Coefficient Convergence

| Coefficient | Iter8 | Iter9 Expected | Change |
|-------------|-------|----------------|--------|
| α₀ (base) | 1.32ms | 1.0-2.0ms | Stable |
| α₁ (input token) | 117.6μs | 100-150μs | Stable |
| α₂ (output token) | 90.5μs | 60-120μs | Stable |
| β₀ (prefill compute) | 0.1912 | 0.15-0.25 | Stable |
| β₁ (decode memory) | 1.1076 | 1.00-1.15 | Stable |
| β₂ (TP comm) | 0.1846 | 0.20-0.35 | Stable |
| **β₃ (KV mgmt)** | **4.40ms** | **0.4-1.0ms** | **10× reversion** |
| β₄ (decode compute) | 0.7132 | 0.70-0.90 | Stable |
| **β₅ (MoE gating)** | **41.1μs** | **10-20μs** | **2-4× reversion** |
| β₆ (scheduler) | 13.2ms | 15-30ms | Stable |
| **β₇ (decode overhead)** | **26.3ms** | **10-20ms** | **1.3-2.6× reversion** |
| β₈ (MoE routing) | 30μs | 25-35μs | Stable (real but insufficient) |
| **β₉ (FP8 dequant)** | **N/A** | **17-50μs** | **NEW** |

## Success Criteria

**Tier 1 (Full Success)**:
- Overall loss <75% ✓
- TTFT RMSE <35% ✓
- All 4 Scout experiments <50% TTFT ✓
- β₉ coefficient physically plausible (17-50μs per token per layer) ✓
- β₃, β₅, β₇ revert to physical ranges ✓
- At least 6/7 hypotheses confirmed ✓

**Tier 2 (Partial Success)**:
- Overall loss <95%
- Scout experiments <60% TTFT
- At least 4/7 hypotheses confirmed
- **Proceed to iter10 with complementary Scout terms**

**Tier 3 (Failure)**:
- Overall loss >120%
- Scout experiments >75% TTFT
- <4/7 hypotheses confirmed
- **Diagnostic**: Validate Scout model config, profile FP8 overhead, consider architecture-specific models

## Next Steps for Agent 2

1. **Validate code compiles**: `go build -o blis main.go` ✅ (already verified)
2. **Run inner loop optimization**: Train on 15 experiments (20-30 trials expected with warm-start)
3. **Generate results**: Output to `inner_loop_results.json`
4. **Pass to Agent 3**: Analysis agent will validate hypotheses and extract principles

## Risk Assessment

**Primary Risk**: β₉ insufficient — FP8 overhead is ONE component, but other mechanisms also missing (TP coordination, batching inefficiency).

**Mitigation**: If Tier 2 (partial success), prepare iter10 with complementary Scout terms (β₁₀ for TP MoE coordination or batching efficiency).

**Secondary Risk**: Model config error — InterleaveMoELayerStep=26, NumExpertsPerTok, or Scout layer counts incorrect.

**Mitigation**: Validate Scout model config against HuggingFace config.json before training.

---

**Design completed by Agent 1 (Design Agent) on 2026-03-30**
**Ready for Agent 2 (Orchestration Agent) to execute optimization**
