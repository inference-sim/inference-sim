# H28: Decode Attention maxKVLen Overestimation

**Status**: Confirmed
**Date**: 2025-02-25

## Hypothesis

> In the roofline model's decode phase, the attention compute time overestimates true per-request-summed attention FLOPs by a factor of maxKVLen/meanKVLen. For a heterogeneous decode batch where the longest request's KV length is K times the mean KV length, the roofline attention FLOPs exceed the sum-of-individual-request FLOPs by exactly factor K. The total decode compute time (GEMM + attention) grows superlinearly when adding short-KV requests to a batch containing one long-KV request, because each added short request contributes maxKVLen (not its own KV length) to attention FLOPs.

**Refuted if:** The roofline decode attention FLOPs for a batch with KV lengths [4096, 64, 64, 64] do NOT simply multiply batchSize by maxKVLen, OR the total decode step time shows strictly linear scaling (constant marginal cost) when adding short-KV decode requests (KV=64) to a batch anchored by one long-KV request (KV=4096), meaning the attention overestimation is fully masked by GEMM or memory bandwidth dominance.
