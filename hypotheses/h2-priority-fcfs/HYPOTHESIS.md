# H2: Priority-FCFS with SLO-Based Priority Should Reduce Realtime TTFT

**Status**: Refuted
**Date**: 2026-02-22

## Hypothesis

> Priority-FCFS with SLO-based priority should reduce realtime TTFT at the cost of batch TTFT. With three SLO classes (realtime, interactive, batch) at equal token sizes, the slo-based priority policy combined with priority-fcfs scheduling should give preferential treatment to realtime requests, reducing their TTFT while increasing batch TTFT.

**Refuted if:** Per-SLO-class TTFT is identical (0% difference) between the prioritized configuration and the FCFS baseline across all 3 seeds, indicating no differentiation.
