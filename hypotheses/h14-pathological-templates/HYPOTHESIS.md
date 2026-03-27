# H14: Pathological Templates -- Anomaly Detection Validation

**Status**: Partially confirmed
**Date**: 2026-02-20

## Hypothesis

> The pathological policies (`always-busiest`, `reverse-priority`, `inverted-slo`) exist specifically to test anomaly detection. `always-busiest` should produce HOL blocking (routes to most loaded instance). `reverse-priority` should produce priority inversions. If anomaly counters don't detect these, the detection logic has a bug.

**Refuted if:** Pathological configurations produce identical anomaly counter values as normal configurations across all 3 seeds.
