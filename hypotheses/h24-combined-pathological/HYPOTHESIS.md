# H24: Combined Pathological Anomalies

**Status**: Confirmed
**Date**: 2026-02-22

## Hypothesis

> Combining always-busiest routing with inverted-slo scheduling should produce maximum measurable anomalies. The pathological routing concentrates all traffic on one instance while the inverted priority starves older requests, producing both HOL blocking and priority inversions simultaneously.

**Refuted if:** The combined pathological configuration produces fewer anomaly counts (HOL blocking + priority inversions) than either single pathological configuration alone, across all 3 seeds.
