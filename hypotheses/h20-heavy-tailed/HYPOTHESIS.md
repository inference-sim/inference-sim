# H20: Heavy-Tailed Input Distributions (ParetoLogNormal vs Gaussian)

**Status**: Refuted
**Date**: 2026-02-23

## Hypothesis

> Heavy-tailed input distributions (ParetoLogNormal) should produce more preemptions and HOL blocking than Gaussian at the same mean input length (~256 tokens), because occasional very long requests hold KV blocks for extended periods, starving short requests.

**Refuted if:** ParetoLogNormal produces fewer or equal preemptions compared to Gaussian in 2 or more of 3 seeds at the core operating point (rate=1000, KV=2000).
