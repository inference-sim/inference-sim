# H9: Prefix Caching Reduces TTFT Monotonically with Prefix Length

**Status**: Confirmed
**Date**: 2026-02-20

## Hypothesis

> TTFT should decrease monotonically as prefix_length increases (holding total input constant at ~768 tokens), because more cached blocks means fewer new tokens to prefill. With all requests sharing the same prefix group, longer prefixes should produce higher cache hit rates and proportionally lower TTFT.

**Refuted if:** TTFT mean is non-monotonic with increasing prefix length (i.e., any inversion where a longer prefix produces higher TTFT), or the maximum prefix (512 tokens) reduces TTFT by less than 50% compared to no prefix, across all 3 seeds.
