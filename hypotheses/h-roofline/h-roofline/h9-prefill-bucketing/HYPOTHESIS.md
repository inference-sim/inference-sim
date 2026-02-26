# H29: Prefill Bucketing Overestimates Short Sequences

**Status**: Confirmed
**Date**: 2025-02-25

## Hypothesis

> For prefill requests with actual sequence lengths significantly below the 512-token minimum bucket, the power-of-2 bucketing in `rooflineStepTime` (sim/roofline_step.go lines 321-336) causes attention core FLOPs to be overestimated by a factor of bucketSeqLen/actualSeqLen (up to 512x for single-token prefills), while memory bandwidth uses actual values. This asymmetry makes the roofline model systematically predict compute-bound prefill for short sequences that should be memory-bound, producing prefill latency overestimates exceeding 2x for seqLen <= 100 compared to an unbucketed calculation using actual sequence lengths.

**Refuted if:** When comparing rooflineStepTime output for a prefill-only step with seqLen=100 (bucket=512) against a hypothetical unbucketed calculation using seqLen=100 for both attention FLOPs and MFU lookup, the bucketed latency is within 2x of the unbucketed latency. Alternatively, refuted if short-prefill steps are memory-bound under both bucketed and unbucketed calculations (meaning the attention FLOPs overestimate is absorbed by the max(compute, memory) roofline and does not affect the final step time).
