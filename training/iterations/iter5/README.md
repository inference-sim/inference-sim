# Iteration 5: Per-Layer Fixed Overhead (Kernel Launch + Scheduler + Memory Allocation)

## Overview

Iter5 addresses the conclusive rejection of iter4's activation bandwidth hypothesis by replacing β₆ (activation bandwidth) with a new mechanism: **per-layer fixed overhead** that scales with prefill chunking.

## Rationale

Iter4 validation decisively rejected the activation bandwidth hypothesis:
- **0% improvement** in reasoning experiments (stayed at 99.98-99.99% TTFT)
- **Coefficient explosion**: β₁ +73.8%, β₂ +328%, β₅ +160%
- **β₆ converged to 1.818** (expected 3.0-6.0)
- **β₀ DECREASED** from 0.169 → 0.165 instead of rising
- The 3.93% overall improvement came entirely from simplification (removing β₂/β₇), not from adding β₆

## Root Cause Analysis

Reasoning experiments show **1000× underestimation** (predicted ~1ms, actual ~1000ms). This magnitude cannot be explained by continuous bottlenecks:
- Memory bandwidth: max 3-5× slowdown (HBM limit: 3.35 TB/s on H100)
- Compute throughput: max 2-3× slowdown (MFU limits: 40-55% typical)
- Communication: max 2× slowdown (NVLink limit: 900 GB/s)

A **1000× slowdown requires fixed per-operation overhead** that accumulates across layers:
1. **Kernel launch overhead** (~50-100μs per CUDA kernel): Each layer requires 10-20 kernel launches (QKV, attention, FFN, layer norms). For long contexts (>2K tokens), vLLM chunks prefill into 2048-token pieces, repeating kernel launches.
2. **Scheduler overhead** (batch formation, memory allocation): vLLM scheduler prepares KV cache blocks, allocates attention buffers.
3. **Memory allocator overhead** (prefix cache, KV block swapping): Reasoning workloads may trigger different allocation patterns.

## Changes from Iter4

### Removed
- **β₆ (activation write bandwidth)** — Misspecified, caused collinearity with β₀, destabilized coefficients

### Added
- **NEW β₆ (per-layer prefill overhead)** — Captures kernel launch + scheduler + memory allocation

### Formula
```go
// Prefill scale factor: captures chunking overhead
prefill_scale_factor = 1.0 + num_prefill_tokens / 2048.0

// Per-layer overhead: kernel launch + scheduler + memory allocation
overhead_us = β₆ × num_layers × prefill_scale_factor
```

### Warm-Start Strategy
- **Use iter3 coefficients** for β₀-β₅ (NOT iter4!)
- Rationale: Iter4 coefficients were destabilized by misspecified activation bandwidth term
- Iter3 coefficients were stable and physically plausible (β₁ = 1.037, β₂ = 0.318, β₅ = 0.0117)

## Hypothesis Bundle

### H-main: Per-Layer Fixed Overhead
- **Prediction**: Loss <110% (from 129%), TTFT RMSE <55%, β₀ rises to 0.25-0.35
- **Mechanism**: Fixed overhead per layer during prefill scales with chunking (kernel launch + scheduler + memory allocation)
- **Expected coefficient**: β₆ ~ 1000-3000μs (1-3ms per layer per chunk-equivalent)
- **Target experiments**: Reasoning TTFT 100% → 70-85%
- **Diagnostic**: If fails → try algorithmic switch, O(n²) attention, or KV preemption

### H-simplification-validated: Removing β₆ (Activation Bandwidth)
- **Prediction**: No degradation, coefficients stabilize (β₁ → 1.00-1.10, β₂ → 0.30-0.35, β₅ → 0.01-0.012)
- **Mechanism**: Removing collinear term eliminates gradient masking
- **Diagnostic**: If degrades >5% → activation BW was partially correct despite appearing ineffective

### H-coefficient-normalization: Physical Plausibility Recovery
- **Prediction**: β₀ rises to 0.25-0.35, other coefficients revert to iter3 ranges
- **Mechanism**: Correct prefill term allows β₀ to fit independently
- **Diagnostic**: If β₀ doesn't rise → per-layer overhead not the missing term

### H-boundary: Overhead Scales with Prompt Length
- **Prediction**: Long prompts (>4K) improve >25%, short prompts (<2K) <10%
- **Mechanism**: Overhead scales with chunking (1.0 + tokens/2048.0)
- **Diagnostic**: If reversed → functional form wrong; if uniform → collinearity with β₀

### H-error-pattern: Largest Improvements in Long-Prompt Experiments
- **Prediction**: Reasoning (99.98% → 70-85%), TP=4 general-lite (70.90% → 50-60%), Mistral TP=2 (76.90% → 60-70%)
- **Mechanism**: Overhead largest for long prompts + many layers
- **Diagnostic**: If no pattern → collinearity; if no improvement → wrong hypothesis

## Basis Function Structure (7 Beta Terms)

| Index | Term | Physics | Expected Coefficient | Status |
|-------|------|---------|---------------------|--------|
| β₀ | Prefill compute MFU | FLOPs / (peak × MFU) | 0.25-0.35 | Should rise from 0.165 |
| β₁ | Decode memory-bound | KV cache reads | 1.00-1.10 | Revert from 1.802 |
| β₂ | TP decode communication | All-reduce per layer | 0.30-0.35 | Revert from 1.360 |
| β₃ | KV cache management | Block allocation | ~0.0004-0.0005 | Stable |
| β₄ | Decode compute-bound | Large-batch tensor cores | 0.75-0.85 | Stable |
| β₅ | MoE gating overhead | Routing computation | 0.01-0.012 | Revert from 0.0304 |
| β₆ | Per-layer prefill overhead | Kernel launch + scheduler | 1000-3000μs | NEW (replaces activation BW) |

## Expected Outcomes

**Overall loss**: 129.20% (iter4) → <110% (target for iter5)
**TTFT RMSE**: 66.49% (iter4) → <55% (target)
**E2E RMSE**: ~62-65% (stable, decode already well-modeled)

**Per-experiment predictions**:
- **Reasoning experiments**: 99.98-99.99% TTFT → 70-85% TTFT (measurable progress, not full solve)
- **TP=4 Llama-3.1-70B general-lite**: 70.90% → 50-60% TTFT
- **Mistral TP=2 general-lite**: 76.90% → 60-70% TTFT
- **Scout experiments**: Minor improvement (10-20%), still problematic (~140-180% combined loss) due to interleaved MoE+dense architecture
- **Short-prompt experiments** (<1K tokens): Minimal change (<10%)

**Coefficient movements** (from iter4 → iter5):
- β₀: 0.165 → 0.25-0.35 (rises as per-layer term captures missing overhead)
- β₁: 1.802 → 1.00-1.10 (reverts to iter3 stability)
- β₂: 1.360 → 0.30-0.35 (reverts to iter3 stability)
- β₃: 0.0005 → 0.0004-0.0005 (stable)
- β₄: 0.918 → 0.75-0.85 (stable)
- β₅: 0.0304 → 0.01-0.012 (reverts to iter3 stability)
- β₆: 1.818 (activation BW) → 1000-3000μs (per-layer overhead) — completely different mechanism

## Example Calculation

**Reasoning experiment: 8K tokens, 80 layers, Llama-2-7B**
- prefill_scale_factor = 1.0 + 8192/2048 = 5.0
- If β₆ = 2000μs (2ms per layer-chunk): overhead = 2000 × 80 × 5.0 = 800ms
- Current TTFT underestimation: ~900ms
- **This captures the gap!**

## Files Generated

1. `iter5-HYPOTHESIS.md` - Complete hypothesis bundle with H-main and 4 additional arms
2. `iteration_manifest.yaml` - Metadata (iteration 5, backend "evolved", changes summary)
3. `coefficient_bounds.yaml` - Bounds and warm-start values (7 beta terms, warm-start from iter3)
4. `sim/latency/evolved_model.go` - Updated implementation (per-layer overhead term, removed activation BW)

## Validation Checklist

- [x] Hypothesis document exists with H-main and diagnostic clauses
- [x] H-main has quantitative prediction (loss <110%, TTFT RMSE <55%, β₀ → 0.25-0.35)
- [x] H-main has causal mechanism (per-layer overhead captures kernel launch + scheduler + memory)
- [x] H-main has diagnostic clause (if fails → algorithmic switch, O(n²) attention, or KV preemption)
- [x] Manifest declares backend="evolved" and lists modified files
- [x] Coefficient bounds have both bounds AND initial values (7 beta terms)
- [x] Initial values warm-started from iter3 (NOT iter4)
- [x] evolved_model.go compiles (`go build -o blis main.go`)
- [x] StepTime() has physics comments for each basis function
- [x] All features workload-agnostic (no forbidden inputs)

## Next Steps (Agent 2: Orchestration)

1. Run inner loop optimization with iter5 artifacts
2. Expected convergence: ~20-30 trials (warm-started from iter3)
3. Monitor β₀ during optimization - should rise from 0.169 toward 0.25-0.35
4. Monitor β₁, β₂, β₅ - should stabilize back to iter3 ranges after removing iter4's misspecified β₆
5. If β₆ (per-layer overhead) converges to <500μs or >5000μs → hypothesis rejected, functional form wrong

## If Iter5 Fails

**Scenarios**:
1. **Loss >120%**: Per-layer overhead not the missing term → profile vLLM prefill with `nsys`
2. **β₀ <0.22**: Wrong hypothesis → try algorithmic switch (different attention kernel for long contexts)
3. **β₆ <500μs or >5000μs**: Functional form wrong → try quadratic scaling or logarithmic scaling
4. **Reasoning doesn't improve (>95% TTFT)**: Try O(n²) attention memory bandwidth, KV cache preemption, or prefix cache misses

**Candidate mechanisms if per-layer overhead fails**:
- Algorithmic switch (different attention kernel for contexts >8K)
- O(n²) attention memory bandwidth (quadratic working set for large n)
- KV cache preemption overhead (swapping blocks to CPU)
- Prefix cache miss rate (re-computing attention for repeated prompts)

**Action**: Profile vLLM reasoning experiments to identify actual bottleneck before iter6.

## Key Learnings from Iter4

**What failed**:
1. **Activation bandwidth hypothesis was wrong** - β₆ = 1.818 (expected 3.0-6.0), 0% improvement
2. **Collinearity with β₀** - both terms scaled with tokens × layers, causing gradient masking
3. **Coefficient explosion** - β₁ +73.8%, β₂ +328%, β₅ +160% when misspecified term added

**What worked**:
1. **Simplification** - removing β₂/β₇ improved loss by 3.93% (continuing iter3's successful pattern)
2. **Diagnostic clauses** - β₀ < 0.22 correctly triggered "activation BW is wrong" diagnostic
3. **Warm-starting** - iter4 converged in 185 trials despite harder search space

**Strategy Evolution validation**:
- **Prediction errors are valuable** - H-main rejection revealed activation BW is NOT the bottleneck
- **Causal mechanisms must be testable** - H-boundary's prediction (long prompts improve >25%) provided clear test that refuted hypothesis
- **Diagnostic clauses guide next iteration** - Each diagnostic clause pointed to specific alternatives (kernel launch, O(n²) attention, KV preemption)
