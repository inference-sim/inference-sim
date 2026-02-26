# H34: Decode GEMM Time Underestimate Below MFU Grid Minimum (M < 8)

**Status**: Confirmed
**Date**: 2026-02-25

## Hypothesis

> For decode steps at batch sizes 1-7 (below the GEMM MFU grid minimum M=8), the roofline model's `computeTransformerGEMMTimes` predicts GEMM time proportional to batch size (ratio bs/8 relative to bs=8), whereas actual GPU GEMM kernels at these small M values exhibit a near-constant memory-bound latency floor, causing the model to underestimate GEMM compute time by up to 8x at bs=1 and potentially misclassifying the compute-memory regime at low batch sizes.

**Refuted if:** The predicted GEMM time ratio `computeTransformerGEMMTimes(bs=1) / computeTransformerGEMMTimes(bs=8)` exceeds 0.25 (i.e., the underestimate is less than 4x, not proportional to 1/8), indicating the MFU database already compensates for the memory-bound floor below M=8.
