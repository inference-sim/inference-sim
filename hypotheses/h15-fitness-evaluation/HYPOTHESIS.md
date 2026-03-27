# H15: Fitness Evaluation Ranks Prefix-Affinity Higher for Prefix Workloads

**Status**: Confirmed with nuance
**Date**: 2026-02-23

## Hypothesis

> Fitness evaluation should rank prefix-affinity-aware routing higher than load-only routing for prefix-heavy workloads when fitness weights favor TTFT.

**Refuted if:** Load-only routing achieves equal or higher fitness score than prefix-affinity-aware routing on the prefix-heavy workload with TTFT-heavy weights, in any seed.
