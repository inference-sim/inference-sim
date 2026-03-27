# H-MMK: Cross-Validate DES Against M/M/k Analytical Model

**Status**: Partially confirmed
**Date**: 2026-02-21

## Hypothesis

> Under matching assumptions (Poisson arrivals, approximately exponential service times, k servers, FCFS), the DES queue length distribution and mean wait time should match M/M/k predictions within 5%. Little's Law (L = lambda * W) should hold within 5%.

**Refuted if:** Little's Law fails (>5% error) for any configuration, or M/M/k wait time divergence exceeds 50% at rho <= 0.3 with least-loaded routing, indicating a fundamental DES architecture bug rather than a modeling approximation.
