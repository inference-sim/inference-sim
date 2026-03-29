# Iteration 4: Activation Memory Bandwidth + Continued Simplification

## Overview

Iter4 addresses two critical gaps identified in iter3:
1. **β₀ = 0.169 far below physical plausibility** (ideal: 0.40-0.55)
2. **β₇ (TP prefill comm) rejected by optimizer** (coefficient ≈ 0), eliminating communication as missing overhead

**Key hypothesis**: Activation memory bandwidth is the missing prefill term that's artificially suppressing β₀.

## Changes from Iter3

### Removed (continuing simplification):
- **β₂ (scheduler overhead)**: Coefficient 9.97e-05 ≈ 0, negligible
- **β₇ (TP prefill communication)**: Coefficient 2.78e-07 ≈ 0, rejected by optimizer

### Added:
- **β₆ (activation write bandwidth)**: NEW mechanism to capture HBM writes during prefill
  - Residual connections, attention QKV projections, layer norms
  - Scales linearly with prompt length: `activation_bytes = tokens × hidden_dim × num_layers × k_factor`
  - Expected coefficient: 3.0-6.0 (multiplier on theoretical write time)

### Result:
- **Parameter count**: 10 → 8 (3 alpha + 7 beta, using 0-indexed so β₀-β₆)
- **Physics coverage**: Compute, memory, communication, KV mgmt, MoE, activation bandwidth

## Basis Function Structure (7 Beta terms)

| Index | Term | Physics | Expected Coefficient | Notes |
|-------|------|---------|---------------------|-------|
| β₀ | Prefill compute MFU | FLOPs / (peak × MFU) | 0.25-0.35 | Should rise from 0.169 with activation term |
| β₁ | Decode memory-bound | KV cache reads | 1.00-1.10 | Stable at 1.037 from iter3 |
| β₂ | TP decode communication | All-reduce per layer | 0.318 | Stable from iter3 β₃ |
| β₃ | KV cache management | Block allocation | 0.00041 | Stable from iter3 β₄ |
| β₄ | Decode compute-bound | Large-batch tensor cores | 0.60-0.70 | May decrease from 0.796 |
| β₅ | MoE gating overhead | Routing computation | 0.008-0.010 | May decrease from 0.0117 |
| β₆ | Activation bandwidth | HBM writes (prefill) | 3.0-6.0 | NEW mechanism |

## Hypothesis Bundle

### H-main: Activation Memory Bandwidth
- **Prediction**: Loss <110% (from 133%), TTFT RMSE <55%, β₀ rises to 0.25-0.35
- **Mechanism**: Prefill writes activations to HBM (residual, QKV, layer norms) → bandwidth-limited for long prompts
- **Diagnostic**: If fails → try kernel launch overhead or O(n²) attention memory bandwidth

### H-simplification: Continue Removing Ineffective Terms
- **Prediction**: Removing β₂/β₇ won't degrade any experiments by >3%
- **Mechanism**: Iter3 validated simplification (removed β₇/β₈ → +3.06% improvement)
- **Diagnostic**: If degrades >5% → terms captured partial effects via lucky initialization

### H-coefficient-normalization: Physical Plausibility Recovery
- **Prediction**: β₀ rises to 0.25-0.35, β₄/β₅ may decrease (if were absorbing activation overhead)
- **Mechanism**: Current β₀ compensates for missing activation term by artificially lowering prefill MFU
- **Diagnostic**: If β₀ doesn't rise → activation bandwidth wrong, try kernel launch

### H-boundary: Activation Bandwidth Scales with Prompt Length
- **Prediction**: Long prompts (>4K) improve >25%, short prompts (<1K) <5%
- **Mechanism**: Activation writes scale linearly with prompt length
- **Diagnostic**: If reversed → formula wrong (nonlinear or batch-dependent)

### H-error-pattern: Which Experiments Should Improve Most?
- **Prediction**: Reasoning (100% → 70-85% TTFT), TP=4 general-lite (70.90% → 50-60%), Mistral TP=2 (79.61% → 65-75%)
- **Mechanism**: Activation overhead largest for long prompts + large models
- **Diagnostic**: If uniform → collinearity with β₀ (absorbing compute overhead)

## Expected Outcomes

**Overall loss**: 133% → 100-115%
**TTFT RMSE**: 70.59% → <55%
**E2E RMSE**: ~62-65% (stable, decode already well-modeled)

**Per-experiment predictions**:
- **Reasoning experiments**: ~100% TTFT → 70-85% TTFT (measurable progress, not full solve)
- **TP=4 Llama-3.1-70B general-lite**: 70.90% → 50-60% TTFT
- **Mistral TP=2 general-lite**: 79.61% → 65-75% TTFT
- **Scout experiments**: ~160-190% combined loss (minor improvement, still problematic)
- **Already-excellent experiments**: Minimal change (<10%)

**Coefficient movements**:
- β₀: 0.169 → 0.25-0.35 (rises as activation term captures missing overhead)
- β₁: 1.037 → 1.00-1.10 (stable)
- β₄: 0.796 → 0.60-0.70 (may decrease if was absorbing activation overhead)
- β₅: 0.0117 → 0.008-0.010 (may decrease if was absorbing activation overhead)
- β₆: 4.0 → 3.0-6.0 (converges to physics-based multiplier)

## Files Generated

1. `iter4-HYPOTHESIS.md` - Complete hypothesis bundle with H-main and 4 additional arms
2. `iteration_manifest.yaml` - Metadata (iteration 4, backend "evolved", changes summary)
3. `coefficient_bounds.yaml` - Bounds and warm-start values (7 beta terms)
4. `sim/latency/evolved_model.go` - Updated implementation (activation bandwidth term)

## Validation Checklist

- [x] Hypothesis document exists with H-main and diagnostic clauses
- [x] H-main has quantitative prediction (loss <110%, TTFT RMSE <55%, β₀ → 0.25-0.35)
- [x] H-main has causal mechanism (activation bandwidth captures HBM writes)
- [x] H-main has diagnostic clause (if fails → kernel launch or O(n²) attention)
- [x] Manifest declares backend="evolved" and lists modified files
- [x] Coefficient bounds have both bounds AND initial values (7 beta terms)
- [x] Initial values warm-started from iter3 best_params
- [x] evolved_model.go compiles (`go build -o blis main.go`)
- [x] StepTime() has physics comments for each basis function
- [x] All features workload-agnostic (no forbidden inputs)

## Next Steps (Agent 2: Orchestration)

1. Run inner loop optimization with iter4 artifacts
2. Expected convergence: ~20-30 trials (warm-started from iter3)
3. Monitor β₀ during optimization - should rise from 0.169 toward 0.25-0.35
4. If β₆ (activation bandwidth) converges to 0 → hypothesis rejected, try kernel launch overhead

## If Iter4 Fails

**Scenarios**:
1. **Loss >120%**: Activation bandwidth not the missing term → profile vLLM prefill
2. **β₀ <0.22**: Wrong hypothesis → try kernel launch overhead (~50μs per kernel × 100-200 kernels)
3. **β₆ ≈ 0**: Activation bandwidth rejected by optimizer → investigate O(n²) attention memory
4. **Short prompts improve >10%**: Formula bug → check activation term doesn't affect decode

**Candidate mechanisms if activation bandwidth fails**:
- Kernel launch overhead (per-layer CUDA kernel invocation)
- O(n²) attention memory bandwidth (quadratic working set for n>4K)
- KV cache preemption overhead (swapping blocks to CPU)
- Prefix cache miss rate (re-computing attention for repeated prompts)

**Action**: Profile vLLM reasoning experiments to identify actual bottleneck before iter5.
