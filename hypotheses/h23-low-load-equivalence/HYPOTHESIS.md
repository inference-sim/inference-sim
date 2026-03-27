# H23: Low-Load Routing Policy Equivalence

**Status**: Confirmed with nuance
**Date**: 2026-02-23

## Hypothesis

> Under very low load (1 req/s, 4 instances), all routing policies should produce equivalent TTFT because all instances are idle and no queue differentiates them.

**Refuted if:** Any pair of routing policies differs by more than 5% in TTFT mean at the low-load operating point (rate=1, 50 requests), in any seed.
