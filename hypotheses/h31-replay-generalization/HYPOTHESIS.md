# H31: CrossModel Generalization to Near-Saturation Reasoning

**Status**: Refuted
**Date**: 2026-03-03

## Hypothesis

> For the codellama-34b-reasoning test-set experiment (0.1% real failure rate), BLIS replay with crossmodel coefficients will complete at least 90% of requests (fewer than 480 DroppedUnservable out of 4800), producing TTFT MAPE < 30% and E2E MAPE < 25% on the intersection set, because near-saturation amplifies step-time errors only linearly (unlike true overload's super-linear amplification).

**Refuted if:** BLIS throughput overestimates real vLLM by >10% (shifting utilization regime), or TTFT/E2E mean relative error exceeds 50% on the primary experiment. **Note:** The original DroppedUnservable gate tested the wrong failure mode — BLIS has no request timeout, so KV-OOM drops are not comparable to vLLM's queueing timeouts. The throughput regime test is the meaningful comparison.

## Context

Test-set reasoning experiments have high failure rates (85% for 7b, 33% for 70b). codellama-34b-reasoning is the cleanest test: 4796/4800 requests complete in real vLLM (0.1% failure). The crossmodel coefficients were trained on moderate-load profiles (general/codegen/roleplay, 0% failure). This tests whether the model + BLIS reproduce near-saturation behavior on an unseen workload regime. The other two test experiments (7b, 70b) are reported as informational but not gated due to >30% failure rate divergence (problem.md Section 5e).
