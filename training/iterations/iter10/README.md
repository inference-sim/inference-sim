# Iteration 10: Sequence-Length-Dependent Batching Inefficiency

## Overview

Iteration 10 addresses the critical discovery from iter9: **Scout's bottleneck is sequence-length-dependent, NOT architecture-dependent (FP8)**. The FP8 dequantization hypothesis was rejected (β₉ converged to 0.14 μs vs expected 17-50 μs), but a powerful new pattern emerged showing inverse correlation between sequence length and error.

## Critical Discovery from Iter9

**Sequence-Length Correlation (inverse relationship)**:
- **Scout short-sequence**: Improved significantly (-53pp, -34pp TTFT from iter8)
  - Roleplay: 79% → 26% TTFT (short sequences ~50-100 tokens)
  - Codegen: 92% → 58% TTFT (moderate sequences ~100-200 tokens)
- **Scout long-sequence**: Failed completely (0pp, -8pp TTFT from iter8)
  - General-lite: 100% → 92% TTFT (long sequences ~400-600 tokens)
  - Reasoning-lite: 99% → 91% TTFT (long sequences ~200-400 tokens)

**Coefficient Explosions** (iter9 absorbed long-sequence overhead into existing terms):
- β₆ (scheduler): +654% (13ms → 99ms) — absorbing long-sequence queueing delays
- β₂ (TP communication): +343% (0.18 → 0.82) — absorbing sequence-length-dependent TP overhead
- β₈ (MoE routing): +143% (30μs → 73μs) — now above predicted 10-50μs range
- β₃ (KV management): +118% (4.4ms → 9.6ms) — moving away from physical 0.4-1ms range

## Iter10 Strategy

**Add β₁₀ (batching inefficiency)**: `β₁₀ × Σ(prefillTokens² / batchSize)`
- Captures queueing delays for long sequences that don't pack well into batches
- Quadratic penalty: prefillTokens² captures disproportionate impact on batch efficiency
- Division by batchSize: Amplifies effect when batch efficiency drops
- Expected contribution: 6-62ms for Scout long-sequence, 0.03-0.3ms for short-sequence (200× difference)

**Split β₃ (KV management)** into base + sequence-length components:
- **β₃** (base overhead): Per-request PagedAttention setup (expected 0.4-1.5ms, revert from iter9's 9.6ms)
- **β₃'** (sequence-length overhead): Block allocation scaling with KV cache size (expected 0.1-1.0μs per token×layer)

**Remove β₉ (FP8 dequantization)**: Hypothesis rejected (converged to 0.14 μs, essentially zero)

**Constrain alpha bounds**: Lower bounds on α₀ ≥ 0.5ms, α₁ ≥ 50μs, α₂ ≥ 40μs to prevent spurious reduction

## Key Files

- **`iter10-HYPOTHESIS.md`**: 7 hypotheses with quantitative predictions, causal mechanisms, and diagnostic clauses
  - H-main: Batching inefficiency captures long-sequence overhead (overall loss 160.6% → <90%)
  - H-kv-scaling: β₃ and β₃' capture base + sequence-length KV overhead
  - H-scheduler-reversion: β₆ reverts from 99ms to 15-40ms
  - H-alpha-stability: Constrained alpha bounds prevent spurious reduction
  - H-boundary: β₁₀ scales quadratically with sequence length
  - H-error-pattern-dense: Dense long-sequence experiments improve >20pp TTFT
  - H-robustness: β₁₀ generalizes across batch size distributions

- **`iteration_manifest.yaml`**: Backend name, modified files, reasoning, timestamp

- **`coefficient_bounds.yaml`**: Bounds and initial values for 3 alpha + 11 beta coefficients
  - Alpha bounds: Constrained lower bounds to prevent spurious reduction
  - Beta bounds: Adjusted based on iter9 learnings and expected reversions
  - Initial values: Warm-started from iter9 optimal with adjustments for new terms

- **`sim/latency/evolved_model.go`**: Evolved latency model implementation
  - 11 beta coefficients: β₀-β₈ (9 coefficients) + β₃' (1 coefficient) + β₁₀ (1 coefficient)
  - StepTime: 10 terms (β₀-β₅, β₇-β₈, β₃', β₁₀)
  - QueueingTime: α₀, α₁, β₆ (scheduler overhead)

## Expected Outcomes

**Success Criteria (Tier 1)**:
- Overall loss: 160.6% → <90% (>70pp improvement)
- TTFT RMSE: 64.8% → <40% (>25pp improvement)
- E2E RMSE: 95.8% → <55% (>41pp improvement)
- Scout long-sequence TTFT: 91.5% → <60% (>31pp improvement)
- Scout short-sequence TTFT: Maintain improvements (<30%, <65%)
- Dense long-sequence TTFT: <70% (all 5 experiments)
- β₁₀ coefficient: 0.1-1.0 ms per (token²/batch_request) — physically plausible
- β₆ reversion: 99ms → 15-40ms (scheduler overhead offloads queueing delay to β₁₀)
- β₃ reversion: 9.6ms → 0.4-1.5ms (base KV overhead)
- β₃' plausible: 0.1-1.0 μs per (token×layer)
- Alpha within constrained bounds (not hitting lower limits)

**Success Criteria (Tier 2 — Partial)**:
- Overall loss: <110% (significant improvement)
- Scout long-sequence: <70% TTFT (>20pp improvement)
- β₁₀ and β₃' coefficients plausible
- At least 2/3 coefficient explosions (β₆, β₂, β₈) decrease >30%

**Failure Criteria (Tier 3)**:
- Overall loss: >130% (minimal improvement)
- Scout long-sequence: >80% TTFT (<12pp improvement)
- β₁₀ converged to zero OR >5 ms (implausible)
- β₆ remains >80ms (no reversion)

## Physics Grounding

**Batching Inefficiency (β₁₀)**:
1. **Batch Packing Constraint**: vLLM scheduler packs requests subject to `Σ(prefill_tokens + kv_cache_blocks) ≤ max_num_batched_tokens`
   - Long sequences (500 tokens) consume 10× more capacity than short sequences (50 tokens)
   - Fewer requests fit in each batch → lower GPU utilization → increased wait time

2. **Quadratic Penalty**: prefillTokens² captures disproportionate impact on batch efficiency
   - Batch size penalty: long sequences → fewer requests per batch
   - Queueing amplification: low batch efficiency → longer queue waits

3. **Division by batchSize**: Amplifies effect for long sequences (lower batch efficiency → smaller denominator)

**KV Cache Management Split (β₃ + β₃')**:
- **β₃** (base): PagedAttention setup, block manager initialization, queue insertion (constant per request)
- **β₃'** (sequence-length): Block allocation/deallocation scaling with KV cache size (proportional to prefillTokens × numLayers)

## Next Steps

**If Tier 1 (Full Success)**:
- Achieved target accuracy, continue to iter11 with refinements or explore alternative architectures

**If Tier 2 (Partial Success)**:
- β₁₀ is correct but needs complementary term (β₁₁ for memory bandwidth saturation)
- Profile vLLM scheduler and batch formation separately to identify remaining bottlenecks

**If Tier 3 (Failure)**:
- Profile vLLM batch formation separately to validate batching inefficiency hypothesis
- Validate basis function formulation (quadratic scaling may be too aggressive)
- Consider architecture-specific models (separate model for MoE+FP8 vs dense)

## Risk Assessment

**Primary Risk**: β₁₀ insufficient — batching inefficiency is ONE component, but other long-sequence mechanisms also missing (memory bandwidth saturation, chunked prefill overhead)

**Mitigation**:
1. If partial success (loss <110%, Scout long-sequence <70%), β₁₀ is correct but needs complementary term
2. If β₆ doesn't revert from 99ms, confirms additional scheduler-specific overhead exists
3. Prepare iter11 design for memory bandwidth saturation or chunked prefill overhead

**Secondary Risk**: β₁₀ formulation incorrect — quadratic scaling may be too aggressive, causing spurious overhead for moderate sequences

**Mitigation**:
1. If moderate-sequence experiments (codegen) degrade >10pp, indicates β₁₀ overestimates overhead
2. Refine basis function to use sigmoid threshold or piecewise linear (only penalize sequences >300 tokens)

**Tertiary Risk**: Alpha constraints too restrictive — optimizer hits lower bounds, indicating beta terms still insufficient

**Mitigation**:
1. If alpha coefficients hit lower bounds, relax bounds slightly for iter11
2. Profile vLLM API overhead separately to measure actual physical alpha values

## References

- **Iter9 FINDINGS.md**: Complete analysis of iter9 results and coefficient explosions
- **Iter9 HYPOTHESIS.md**: FP8 dequantization hypothesis (rejected)
- **Design Agent Prompt**: `training/docs/design-agent-prompt.md` — methodology and requirements
- **Strategy Evolution**: `docs/methodology/strategy-evolution.md` — hypothesis bundle design process
- **Hypothesis Bundles**: `docs/methodology/hypothesis-bundles.md` — worked examples from previous iterations
