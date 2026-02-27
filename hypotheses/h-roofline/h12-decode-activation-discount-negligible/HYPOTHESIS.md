# H35: Decode Activation Memory Factor (0.75) Is Inconsequential

**Status**: Confirmed
**Date**: 2026-02-25

## Hypothesis

> The decode activation memory factor (0.75) in `calculateMemoryAccessBytes` is inconsequential for decode step time prediction because activation bytes (`nLayers * dModel * bytesPerParam * 0.75` per request) constitute less than 0.5% of total memory traffic across all evaluation operating points (bs=1..256, kvLen=128..8192), so replacing the 0.75 with any value in [0.5, 1.5] changes predicted decode step time by less than 0.05%.

**Refuted if:** There exists an evaluation operating point (batchSize, kvLen) at which changing the decode activation factor from 0.75 to 1.00 shifts the predicted decode step time by more than 0.1%, indicating activation memory is not negligible and the magic constant requires empirical calibration.
