# H4: Round-Robin Should Match or Outperform Least-Loaded for Uniform Workloads at Low Rates

**Status**: Confirmed with nuance
**Date**: 2026-02-23

## Hypothesis

> Round-robin should outperform or match least-loaded for uniform workloads at low rates. For perfectly uniform request sizes (constant 256 input / 128 output) at low utilization (~0.29x), round-robin distributes optimally with zero overhead. Least-loaded has the same distribution but with routing computation overhead and potential for minor oscillation due to PendingRequests tracking delays. Predicted outcome: nearly identical metrics (within 5%).

**Refuted if:** Least-loaded TTFT mean is more than 5% better than round-robin at low rate (rate=100, 4 instances) across all 3 seeds, indicating that LL's load-awareness provides measurable value even for uniform low-rate workloads.
