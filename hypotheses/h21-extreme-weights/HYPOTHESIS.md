# H21: Extreme Scorer Weights

**Status**: Refuted
**Date**: 2026-02-22

## Hypothesis

> Extreme scorer weight (weight=100:1) should behave identically to single-scorer routing. When one scorer's weight dominates by 100x, the minority scorer's contribution should be negligible and the routing behavior should be equivalent to using the dominant scorer alone.

**Refuted if:** TTFT mean or target distribution differs by more than 5% between the 100:1 two-scorer config and the single-scorer config, across all 3 seeds.
