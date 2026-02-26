# H33: Decode Attention MFU Shape Mismatch for Heterogeneous KV Lengths

**Status**: Confirmed
**Date**: 2026-02-25

## Hypothesis

> In heterogeneous decode batches, using maxKVLen for the attention MFU lookup (roofline_step.go:298) while using per-request actual KV lengths for FLOPs (lines 287-295) systematically underestimates decode attention time, because the MFU at maxKVLen is higher than the effective per-request MFU at shorter KV lengths, and the underestimation magnitude scales with the KV length variance in the batch.

**Refuted if:** The decode attention time computed with the single maxKVLen MFU lookup is within 5% of the per-request MFU-weighted attention time (where each request's attention time uses its own MFU looked up at its actual KV length) for heterogeneous batches spanning a 10x+ KV length range (e.g., KV lengths from 128 to 4096 in the same batch).
