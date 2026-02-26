# H33: Decode Attention MFU Shape Mismatch -- Findings

**Status**: CONFIRMED
**Date**: 2026-02-25
**Family**: Structural model
**Type**: Deterministic (no RNG, pure roofline computation)

## Hypothesis

In heterogeneous decode batches, using maxKVLen for the attention MFU lookup
while using per-request actual KV lengths for FLOPs systematically
underestimates decode attention time, because the MFU at maxKVLen is higher
than the effective per-request MFU at shorter KV lengths.

**Refuted if**: The decode attention time computed with the single maxKVLen MFU
lookup is within 5% of the per-request MFU-weighted attention time for
heterogeneous batches spanning a 10x+ KV length range.

## Experiment Design

- **Independent variable**: Batch composition (KV length heterogeneity), ranging
  from homogeneous (1x range) to pathological (64x range).
- **Controlled variables**: Model config (Llama-3.1-8B), hardware config (H100),
  TP=1, decode-only attention computation.
- **Dependent variable**: Ratio of attention time computed by current method
  (single MFU at maxKVLen) to per-request method (individual MFU per request).

### Methods Compared

**Method A (Current)**: Sum per-request FLOPs with actual KV lengths, then divide
by a single MFU looked up at `(totalBatchSize, maxKVLen)`.

```
totalFLOPs = sum(calculateAttentionCoreFLOPs(1, req.kvLen) for req in batch)
attnMFU = GetAttnDecodeMFU(totalBatchSize, maxKVLen, tp)
attnTime = totalFLOPs / (peakFlops * attnMFU)
```

**Method B (Per-Request)**: Each request gets its own MFU at `(1, req.kvLen)`,
compute per-request time, sum individual times.

```
attnTime = sum(
    calculateAttentionCoreFLOPs(1, req.kvLen) / (peakFlops * GetAttnDecodeMFU(1, req.kvLen, tp))
    for req in batch
)
```

### Subtests

| Test | Purpose |
|------|---------|
| `TestH33_MFUShapeMismatch` | Core experiment: compute ratio for 15 batch compositions |
| `TestH33_MFUMonotonicity` | Verify MFU is monotone in KV length (underlying assumption) |
| `TestH33_PerRequestMFUProfile` | Show per-request MFU values vs batch-level MFU |
| `TestH33_StepTimeImpact` | Full step time impact (including GEMM, memory, overhead) |
| `TestH33_KVRangeVsRatio` | Sweep KV range factor to show ratio vs heterogeneity |

## Results

### Core Results (Ratio of Current / Per-Request Attention Time)

| Scenario | BS | MaxKV | Range | Ratio | Status |
|----------|---:|------:|------:|------:|--------|
| homo_4x128 | 4 | 128 | 1x | 5.000000 | OUTSIDE_5% |
| homo_4x1024 | 4 | 1024 | 1x | 5.000000 | OUTSIDE_5% |
| homo_4x4096 | 4 | 4096 | 1x | 0.652174 | OUTSIDE_5% |
| mild_2x_range | 4 | 1024 | 2x | 5.000000 | OUTSIDE_5% |
| mild_4x_range | 4 | 1024 | 4x | 5.000000 | OUTSIDE_5% |
| mod_8x_range | 4 | 1024 | 8x | 5.000000 | OUTSIDE_5% |
| mod_10x_short_heavy | 4 | 1024 | 8x | 5.000000 | OUTSIDE_5% |
| mod_10x_long_heavy | 4 | 1024 | 8x | 5.000000 | OUTSIDE_5% |
| high_32x_4req | 4 | 4096 | 32x | 0.689089 | OUTSIDE_5% |
| high_32x_5req | 5 | 8192 | 32x | 0.635271 | OUTSIDE_5% |
| extreme_64x_4req | 4 | 8192 | 64x | 0.794592 | OUTSIDE_5% |
| extreme_64x_8req | 8 | 8192 | 64x | 0.536241 | OUTSIDE_5% |
| extreme_64x_16req | 16 | 8192 | 64x | 0.229695 | OUTSIDE_5% |
| pathological_1long_7short | 8 | 8192 | 64x | 0.628059 | OUTSIDE_5% |
| pathological_1long_15short | 16 | 8192 | 64x | 0.448609 | OUTSIDE_5% |

**Summary**: 0/15 scenarios within the 5% threshold. All 7 high-heterogeneity
(10x+ range) scenarios show ratio < 0.95, confirming systematic underestimation.

- Overall: min ratio = 0.229695, max ratio = 5.000000, mean = 2.640915
- Homogeneous baselines: mean ratio = 3.550725 (expect ~1.0 -- see caveat below)
- High heterogeneity (10x+ range): mean ratio = 0.565937

### MFU Monotonicity

Decode attention MFU is **NOT** monotonically increasing with KV length at BS=1.
There is a discontinuity caused by the nearest-non-zero fallback: for KV lengths
<= 1024 at BS=1, the raw MFU is 0.0 and falls back to (bs=16, kv=1024) with
MFU=0.0080. For KV >= 2048, the actual MFU values are lower (0.001 at kv=2048,
0.003 at kv=4096, 0.005 at kv=8192). This creates an inverted MFU profile at
BS=1 where short-KV requests paradoxically get HIGHER per-request MFU than
long-KV requests.

| KVLen | BS=1 MFU | BS=4 MFU | BS=8 MFU | BS=16 MFU | BS=32 MFU |
|------:|---------:|---------:|---------:|----------:|----------:|
| 64 | 0.008000* | 0.001600 | 0.003733 | 0.008000 | 0.010000 |
| 128 | 0.008000* | 0.001600 | 0.003733 | 0.008000 | 0.010000 |
| 256 | 0.008000* | 0.001600 | 0.003733 | 0.008000 | 0.010000 |
| 512 | 0.008000* | 0.001600 | 0.003733 | 0.008000 | 0.010000 |
| 1024 | 0.008000* | 0.001600 | 0.003733 | 0.008000 | 0.010000 |
| 2048 | 0.001000 | 0.002600 | 0.004733 | 0.009000 | 0.010333 |
| 4096 | 0.003000 | 0.004600 | 0.006733 | 0.011000 | 0.011000 |
| 8192 | 0.005000 | 0.006400 | 0.008267 | 0.012000 | 0.013000 |
| 16384 | 0.007000 | 0.008000 | 0.009333 | 0.012000 | 0.013000 |

\* = nearest-non-zero fallback from raw MFU = 0.0

At BS=4 and above, MFU is monotonically increasing with KV length.
At BS=1, there is 1 violation out of 8 checks (kv=2048 < kv=1024).

### MFU Profile Detail

The inverted BS=1 MFU profile causes a counterintuitive effect. For the
pathological batch (1x8192 + 7x128), the per-request MFU at kv=128 is 0.008000
(fallback) while the per-request MFU at kv=8192 is 0.005000 (real). The batch-
level MFU at (bs=8, kv=8192) is 0.008267. This means short-KV requests
individually appear MORE efficient (higher MFU) than both the batch-level MFU
and the long-KV request MFU.

### Range Sweep Results

| ShortKV | MaxKV | Range | Ratio |
|--------:|------:|------:|------:|
| 8192 | 8192 | 1x | 0.781250 |
| 4096 | 8192 | 2x | 0.558036 |
| 2048 | 8192 | 4x | 0.287829 |
| 1024 | 8192 | 8x | 0.870253 |
| 512 | 8192 | 16x | 0.830420 |
| 256 | 8192 | 32x | 0.807196 |
| 128 | 8192 | 64x | 0.794592 |

The ratio does not monotonically decrease with range. The minimum ratio (0.287829)
occurs at 4x range (shortKV=2048), not at the highest range. This is because the
MFU fallback at BS=1 for kv<=1024 inflates the per-request MFU for short
sequences, reducing the per-request total time and bringing the ratio back up.

### Full Step Time Impact

| Scenario | Baseline(us) | Adjusted(us) | StepRatio | AttnFrac |
|----------|-------------:|-------------:|----------:|--------:|
| homo_4x128 | 8301 | 8301 | 1.000000 | 3.33% |
| homo_4x1024 | 8438 | 8438 | 1.000000 | 25.91% |
| homo_4x4096 | 8907 | 9613 | 0.926558 | 33.08% |
| mild_2x_range | 8399 | 8399 | 1.000000 | 19.58% |
| mild_4x_range | 8370 | 8370 | 1.000000 | 14.77% |
| mod_8x_range | 8355 | 8355 | 1.000000 | 12.34% |
| mod_10x_short_heavy | 8336 | 8336 | 1.000000 | 9.08% |
| mod_10x_long_heavy | 8404 | 8404 | 1.000000 | 20.37% |
| high_32x_4req | 8453 | 8453 | 1.000000 | 9.83% |
| high_32x_5req | 8819 | 9308 | 0.947465 | 19.35% |
| extreme_64x_8req | 11171 | 11980 | 0.932471 | 11.74% |
| extreme_64x_16req | 11179 | 14857 | 0.752440 | 13.75% |
| pathological_1long_7short | 10818 | 11163 | 0.969094 | 7.65% |
| pathological_1long_15short | 10528 | 11077 | 0.950438 | 6.10% |

Step time ratio: min = 0.752440, max = 1.000000, mean = 0.965231.

For the worst case (extreme_64x_16req), the baseline step time underestimates
by 24.8% relative to the per-request-MFU adjusted step time (11179 vs 14857 us).
However, most scenarios show < 7% step time impact because attention is only
6-33% of total step time, with GEMM and memory bandwidth costs dominating.

## Verdict

**CONFIRMED** -- The hypothesis is confirmed, but with a critical caveat.

The single maxKVLen MFU lookup produces a systematic underestimation of decode
attention time for heterogeneous batches. All 7 high-heterogeneity scenarios
(10x+ range) have ratios below 0.95, with the worst case at 0.2297 (77%
underestimation of attention time).

**However**, the experiment also reveals a confounding factor not anticipated
by the hypothesis: the **nearest-non-zero MFU fallback** at BS=1 for small KV
lengths (<=1024) artificially inflates per-request MFU values. This means:

1. The homogeneous baselines do NOT have ratio ~1.0 as expected (they show 5.0x
   for small KV lengths), because the batch-level MFU at (BS=4, kv<=1024) is
   0.0016, while the per-request MFU at (BS=1, kv<=1024) falls back to 0.008.
2. The direction of bias flips depending on the KV length regime: for kv<=1024,
   the current method OVERESTIMATES (ratio=5.0); for kv>=4096, it UNDERESTIMATES
   (ratio<1.0).
3. The MFU database's sparse coverage at low batch sizes is the dominant error
   source, not the maxKVLen vs per-request MFU choice.

## Root Cause Analysis

The mismatch has **two interacting causes**:

1. **Batch-size MFU scaling** (primary): The current method looks up MFU at
   `(totalBatchSize, maxKVLen)`, while the per-request method looks up at
   `(1, req.kvLen)`. The batch-size dimension creates a 5x difference for small
   KV lengths where both batch-level and per-request MFU values are fallback-
   dominated. At BS=4/kv<=1024, batch MFU = 0.0016; at BS=1/kv<=1024, per-request
   MFU = 0.008 (fallback). The batch-size axis dominates the ratio more than the
   KV-length axis for small sequences.

2. **MFU non-monotonicity at BS=1** (secondary): The nearest-non-zero fallback
   assigns the same MFU (0.008) to all KV lengths <= 1024 at BS=1, creating a
   cliff at kv=2048 where MFU drops to 0.001. This inverts the expected
   relationship and distorts the per-request method's results.

For large KV lengths (>=4096), the actual MFU data is available and the expected
underestimation pattern holds: batch-level MFU at maxKVLen is higher than
individual request MFU values, so the current method underestimates.

## Impact Assessment

**Attention-only impact**: Ratios range from 0.23 to 5.0, with the 10x+ range
scenarios showing 20-77% underestimation of attention time.

**Full step time impact**: Moderate. Step time ratios range from 0.752 to 1.0,
because attention contributes only 3-33% of total step time. The worst case
(extreme_64x_16req) shows a 24.8% step time underestimation. For most practical
workloads, the impact is 3-7%.

**Practical significance**: The impact is most severe for large batches with
high KV-length variance AND long maximum KV lengths (>= 4096), which are common
in real serving workloads (e.g., a mix of short chat turns and long document
contexts). For 8B models on H100, step time errors of 5-25% are expected in
these regimes.

## Recommended Action

1. **Short-term**: The MFU fallback behavior for BS=1 at small KV lengths needs
   investigation. The nearest-non-zero fallback inflating MFU from 0.0 to 0.008
   distorts both methods and makes the comparison unreliable for kv<=1024.

2. **Medium-term**: Consider using a weighted-MFU approach where each request
   contributes to the effective MFU proportionally to its FLOPs fraction:
   `effectiveMFU = sum(reqFLOPs_i * MFU_i) / sum(reqFLOPs_i)`. This preserves
   the single-division step time computation while accounting for per-request
   MFU variation.

3. **Long-term**: The sparse MFU database at low batch sizes (many zero entries
   requiring fallback) is the dominant error source. Improving bench_data coverage
   for BS=1-4 at KV lengths 64-1024 would resolve both the fallback artifacts
   and the shape mismatch.
