# H-Perf-Wallclock: Simulator Wall-Clock Performance Optimization

**Status**: Confirmed
**Date**: 2026-03-04

## Hypothesis

> Combining three Phase 1 optimizations — O(1) LRU eviction, hash computation deduplication, and SHA256 hasher reuse — will reduce total wall-clock time by >50% (17s to <8.5s) on prefix-affinity-heavy workloads without changing any simulation output (INV-6 determinism preserved). With prefix-affinity disabled, the optimizations should have negligible (<5%) effect, confirming the bottleneck is prefix-affinity-specific.

**Refuted if:** Wall-clock reduction is less than 30% across all 3 seeds, or any simulation output differs between baseline and optimized (INV-6 violation), or the negative control (no prefix-affinity) shows more than 5% difference.
