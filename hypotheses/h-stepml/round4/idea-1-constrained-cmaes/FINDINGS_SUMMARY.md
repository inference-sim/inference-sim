# FINDINGS SUMMARY: Idea 1 — Constrained CMA-ES

**Round:** 4
**Date:** 2026-03-02
**Status:** REFUTED (all tested sub-hypotheses)
**Session note:** H1 and H2 (alpha=0.7 only) completed. Session died (API ConnectionRefused) before H2 alpha=0.3, H3 LOMO, H4 LOWO. Short-circuit criterion was met (H1 E2E 51.2% >> 25% threshold), so H3/H4 would have been skipped regardless.

## Executive Summary

Constraining CMA-ES parameter bounds and adding an ITL penalty term **dramatically worsened** E2E accuracy compared to R3's unconstrained CMA-ES (51.2% vs 15.1% mean E2E). The parameter bounds prevented CMA-ES from finding the coefficient values that compensated for missing simulation dynamics. This confirms that R3's "implausible" parameter values (e.g., inflated `output_token_processing_time_us`) were doing necessary work — they were proxying for real system behaviors not captured in the BLIS model.

## Results Table

| Sub-Hypothesis | Status | Key Metric | Target | Result |
|---|---|---|---|---|
| H1: Constrained CMA-ES (α=0.5) | **REFUTED** | Mean E2E | <15% | 51.2% |
| H2: Pareto Sweep (α=0.7) | **REFUTED** | E2E<15% AND ITL<25% | At least 1 point | No (42.9% E2E) |
| H2: Pareto Sweep (α=0.3) | NOT RUN | — | — | Session died |
| H3: LOMO | NOT RUN | — | — | Short-circuited (H1 E2E >> 25%) |
| H4: LOWO | NOT RUN | — | — | Short-circuited (H1 E2E >> 25%) |

## H1: Constrained CMA-ES (α=0.5) — REFUTED

**Objective:** `f(x) = 0.5 * e2e + 0.5 * itl + penalty`

### Per-Model Optimization

| Model | Evals | Time (s) | Initial E2E | Initial ITL | Best E2E | Best ITL | Best Objective |
|---|---|---|---|---|---|---|---|
| codellama-34b | 88 | 1831 | 7.9% | 109.4% | 50.1% | 0.4% | 0.252 |
| llama-2-70b | 96 | 1952 | 8.3% | 98.9% | 52.3% | 2.8% | 0.276 |
| llama-2-7b | 104 | 553 | 22.2% | 91.1% | 57.0% | 5.6% | 0.313 |
| mixtral-8x7b-v0-1 | 96 | 1920 | 25.9% | 51.4% | 49.1% | 4.0% | 0.266 |

**Key observation:** CMA-ES converged to solutions with excellent ITL (0.4–5.6%) but terrible E2E (49–57%). The parameter bounds prevented it from trading ITL for E2E — the opposite of R3's problem. With α=0.5, the optimizer found that reducing ITL (easy within bounds) was more achievable than reducing E2E (impossible within bounds).

### Per-Experiment Results

| Experiment | Model | Workload | E2E % | TTFT % | ITL % | BLIS E2E (ms) | GT E2E (ms) |
|---|---|---|---|---|---|---|---|
| codellama-34b-tp2-general | codellama-34b | general | 50.2% | 13.5% | 0.4% | 2040 | 4093 |
| codellama-34b-tp2-codegen | codellama-34b | codegen | 50.3% | 4.5% | 0.6% | 1851 | 3723 |
| codellama-34b-tp2-roleplay | codellama-34b | roleplay | 49.9% | 5.3% | 0.1% | 1838 | 3670 |
| llama-2-70b-tp4-general | llama-2-70b | general | 51.2% | 25.7% | 0.0% | 2596 | 5321 |
| llama-2-70b-hf-tp4-codegen | llama-2-70b-hf | codegen | 52.6% | 33.1% | 3.6% | 2183 | 4605 |
| llama-2-70b-tp4-roleplay | llama-2-70b | roleplay | 53.0% | 34.3% | 4.9% | 2143 | 4562 |
| llama-2-7b-tp1-roleplay | llama-2-7b | roleplay | 57.0% | 14.4% | 5.6% | 891 | 2071 |
| mixtral-8x7b-v0-1-tp2-codegen | mixtral-8x7b-v0-1 | codegen | 48.6% | 6.7% | 5.3% | 2404 | 4675 |
| mixtral-8x7b-v0-1-tp2-general | mixtral-8x7b-v0-1 | general | 50.8% | 19.9% | 0.5% | 2480 | 5039 |
| mixtral-8x7b-v0-1-tp2-roleplay | mixtral-8x7b-v0-1 | roleplay | 48.0% | 9.5% | 6.2% | 2435 | 4685 |

**Aggregates:** Mean E2E = 51.2%, Mean TTFT = 16.7%, Mean ITL = 2.7%

**Pattern:** BLIS consistently under-predicts by ~50% — it produces E2E values roughly half the ground truth. This is uniform across all models and workloads, suggesting the bounded coefficients cannot represent the true step duration scaling.

## H2: Pareto Sweep (α=0.7) — REFUTED

**Objective:** `f(x) = 0.7 * e2e + 0.3 * itl + penalty`

### Per-Model Optimization

| Model | Evals | Time (s) | Best E2E | Best ITL | Best Objective |
|---|---|---|---|---|---|
| codellama-34b | 96 | 1955 | 41.0% | 17.9% | 0.341 |
| llama-2-70b | 96 | 1934 | 41.1% | 19.9% | 0.348 |
| llama-2-7b | 104 | 555 | 58.0% | 3.1% | 0.416 |
| mixtral-8x7b-v0-1 | 32 | 2256 | 41.5% | 19.6% | 0.349 |

### Per-Experiment Results

| Experiment | Model | Workload | E2E % | TTFT % | ITL % |
|---|---|---|---|---|---|
| codellama-34b-tp2-general | codellama-34b | general | 33.5% | 5.2% | 33.0% |
| codellama-34b-tp2-codegen | codellama-34b | codegen | 44.0% | 0.9% | 11.9% |
| codellama-34b-tp2-roleplay | codellama-34b | roleplay | 45.6% | 2.8% | 8.8% |
| llama-2-70b-tp4-general | llama-2-70b | general | 25.6% | 11.5% | 52.2% |
| llama-2-70b-hf-tp4-codegen | llama-2-70b-hf | codegen | 47.3% | 36.4% | 7.1% |
| llama-2-70b-tp4-roleplay | llama-2-70b | roleplay | 50.5% | 36.1% | 0.4% |
| llama-2-7b-tp1-roleplay | llama-2-7b | roleplay | 58.0% | 16.2% | 3.1% |
| mixtral-8x7b-v0-1-tp2-codegen | mixtral-8x7b-v0-1 | codegen | 41.0% | 2.7% | 20.8% |
| mixtral-8x7b-v0-1-tp2-general | mixtral-8x7b-v0-1 | general | 42.9% | 16.0% | 16.6% |
| mixtral-8x7b-v0-1-tp2-roleplay | mixtral-8x7b-v0-1 | roleplay | 40.6% | 5.5% | 21.5% |

**Aggregates:** Mean E2E = 42.9%, Mean TTFT = 13.3%, Mean ITL = 17.5%

**Pareto analysis:** Shifting weight toward E2E (α=0.7) reduced E2E from 51.2% → 42.9% but degraded ITL from 2.7% → 17.5%. Neither point achieves E2E < 15%, let alone the joint target of E2E < 15% AND ITL < 25%.

**α=0.3 was not tested** (session died), but extrapolating: it would push ITL lower and E2E higher (toward ~55%+). No Pareto knee exists within the constrained parameter space.

## H3: LOMO — NOT RUN

Short-circuited: H1 mean E2E (51.2%) exceeded the 25% short-circuit threshold.

## H4: LOWO — NOT RUN

Short-circuited: H1 mean E2E (51.2%) exceeded the 25% short-circuit threshold.

## Root Cause Analysis

### Why constraining parameters destroyed E2E accuracy

R3's unconstrained CMA-ES achieved 15.1% E2E by pushing parameters to "implausible" values. Specifically:
- `output_token_processing_time_us` was inflated to ~1,899μs for 7B models
- These values compensated for simulation dynamics not captured in the BLIS model

When we bounded these parameters to "physically reasonable" ranges:
- The optimizer could not find coefficient combinations that produce the correct E2E duration
- BLIS consistently under-predicted by ~50% (E2E values roughly half of ground truth)
- The optimizer settled for minimizing ITL (achievable within bounds) since E2E was unreachable

### The fundamental tension

The BLIS simulator's `output_token_processing_time_us` and related parameters are **not** physical quantities — they are regression coefficients in a simplified model. Constraining them to "physical" ranges assumes they represent physical quantities, but they actually absorb:
- CPU scheduling overhead
- Memory management latency
- Batch formation costs
- Synchronization overhead between decode steps
- Any other latency source not explicitly modeled

This confirms that parameter-level constraints are the wrong abstraction for improving ITL while maintaining E2E accuracy.

## Binding Constraints for Next Steps

1. **Parameter bounds are harmful:** Do not constrain CMA-ES parameters to "physical" ranges — the parameters compensate for unmodeled dynamics
2. **The CMA-ES approach needs a different formulation:** Instead of constraining the search space, the underlying model needs to be richer (as Idea 3's principled cycle-time approach demonstrates)
3. **α=0.5 on a bounded space collapses to ITL-only optimization:** The optimizer finds ITL much easier to minimize within bounds, so it ignores E2E

## vLLM-args Sensitivity

Not explicitly tested (short-circuited before generalization experiments). However, the fundamental finding — that bounded parameters cannot represent the full system — implies that vLLM-arg changes (which modify the unmodeled dynamics) would further degrade accuracy under this approach.
