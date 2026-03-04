# H30: CrossModel Fidelity on Training Set

**Status**: Partially Confirmed
**Date**: 2026-03-03

## Hypothesis

> When BLIS simulates statistically equivalent workloads for all 10 training-set experiments using Iter 3 crossmodel coefficients and the standard `./blis run` CLI, aggregate latency metrics (TTFT mean, E2E mean, throughput) will match real vLLM within 25% relative error, confirming the crossmodel backend is useful for capacity planning.

**Refuted if:** Throughput relative error exceeds 10% on any training experiment, or E2E mean relative error exceeds 25% on more than 2 experiments.

**Methodology note:** This is a same-specification comparison (same workload parameters, different Poisson realization), not per-request trace replay. The three-way comparison (crossmodel vs per-model blackbox via `--beta-coeffs` vs real) isolates coefficient error from BLIS scheduler model error.

## Context

The Iter 3 analytical evaluation computed TTFT/E2E by summing predicted step times over real journey step indices (train TTFT 5.3-17.6%, E2E 8.0-10.6%). This bypasses BLIS's scheduler — the batch compositions used were vLLM's actual decisions. BLIS replay is different: BLIS makes its own scheduling/batching decisions via VLLMBatchFormation, which will produce different batch compositions and thus different step times. This hypothesis tests whether the full integration (crossmodel + BLIS scheduler + BLIS KV cache) produces useful request-level predictions.
