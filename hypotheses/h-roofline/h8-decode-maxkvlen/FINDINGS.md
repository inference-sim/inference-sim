# H28: Decode Attention maxKVLen Overestimation — Findings

**Status**: Confirmed
**Date**: 2026-02-25
**Family**: Structural model
**Type**: Deterministic (no RNG, pure roofline computation)

## Hypothesis

In the roofline model's decode phase, attention compute time overestimates true
per-request-summed attention FLOPs by a factor of `maxKVLen/meanKVLen` for
heterogeneous batches. The total decode compute time grows superlinearly when
adding short-KV requests to a batch containing one long-KV request.

**Refuted if**: The decode attention FLOPs do NOT simply multiply batchSize by
maxKVLen, OR total decode step time shows strictly linear scaling when adding
short-KV decode requests (KV=64) to a batch anchored by one long-KV request
(KV=4096).

## Experiment Design

- **Independent variable**: Number of short-KV (KV=64) decode requests added to
  a batch anchored by one long-KV (KV=4096) request. Range: 0 to 15 added.
- **Controlled variables**: Model config (Llama-3.1-8B), hardware config (H100),
  TP=1, decode-only batches (no prefill).
- **Dependent variables**: Total decode step time (us), attention FLOPs,
  marginal cost per added request (us).
- **Comparison baseline**: Homogeneous batches of short-KV (KV=64) requests at
  matching batch sizes (1-16).

## Results

### Check 1: Attention FLOPs use maxKVLen for all requests

**CONFIRMED.** The `calculateAttentionCoreFLOPs` function takes a single `seqLen`
parameter (the maximum KV length in the batch) and multiplies it by the total
`batchSize`. A heterogeneous batch [4096, 64, 64, 64] produces exactly the same
attention FLOPs as a homogeneous batch [4096, 4096, 4096, 4096] — both yield
8,589,934,592 FLOPs. The overestimation factor is consistently > 1.0 for all
batch sizes tested (range: 1.97x to 12.96x).

### Check 2: Overestimation ratio = maxKVLen/meanKVLen

**CONFIRMED.** The overestimation factor matches the predicted `maxKVLen/meanKVLen`
ratio within 1% tolerance for all tested batch sizes:

- batch_size=2: expected 1.9692, actual 1.9692 (match)
- batch_size=4: expected 3.8209, actual 3.8209 (match)
- batch_size=8: expected 7.2113, actual 7.2113 (match)
- batch_size=16: expected 12.9620, actual 12.9620 (match)

### Check 3: Total step time scaling behavior

**CONFIRMED.** The hetero/homo step time ratio grows from 1.019 (bs=1) to 1.305
(bs=16), demonstrating that the attention FLOPs overestimation does propagate into
measurably superlinear total step time growth. The marginal cost of adding the 1st
short request is 2 us, while the marginal cost of adding the 10th short request
is 60 us — a 30x increase.

Note: The marginal cost curve is non-monotonic due to MFU lookup grid boundaries
(a known effect from H6). Marginal costs jump at bs=4-8 then decrease at bs=9+,
reflecting transitions between MFU interpolation regions. Despite this, the
overall hetero/homo ratio increases consistently from bs=5 onward.

### Key Data

| Batch Size | Hetero Time (us) | Homo Time (us) | Ratio | Overestimation Factor |
|:----------:|:-----------------:|:---------------:|:-----:|:---------------------:|
| 1          | 8438              | 8284            | 1.019 | n/a (anchor only)     |
| 2          | 8440              | 8287            | 1.018 | 1.97x                 |
| 4          | 8606              | 8292            | 1.038 | 3.82x                 |
| 8          | 12815             | 10308           | 1.243 | 7.21x                 |
| 16         | 13240             | 10149           | 1.305 | 12.96x                |

### Attention FLOPs Overestimation Detail

| Batch Size | Roofline Attn FLOPs | Ideal Attn FLOPs | Overestimation |
|:----------:|:-------------------:|:-----------------:|:--------------:|
| 2          | 4,294,967,296       | 2,181,038,080     | 1.97x          |
| 4          | 8,589,934,592       | 2,248,146,944     | 3.82x          |
| 8          | 17,179,869,184      | 2,382,364,672     | 7.21x          |
| 16         | 34,359,738,368      | 2,650,800,128     | 12.96x         |

At batch_size=16, the roofline model computes 34.4 billion attention FLOPs when
the ideal (per-request-summed) value is only 2.65 billion — a 12.96x
overestimation. This means the roofline model attributes 64x more attention work
to each short-KV request than it actually requires (4096/64 = 64).

## Verdict

**CONFIRMED** — all three checks pass.

The roofline model's decode attention FLOPs overestimate by exactly
`maxKVLen/meanKVLen` for heterogeneous batches, and this overestimation is
visible in total step time as growing hetero/homo ratios (1.02x to 1.31x).

## Root Cause Analysis

The `calculateAttentionCoreFLOPs` function (`sim/roofline_step.go:210-231`) takes
a single `seqLen` parameter (the maximum KV length in the batch) and multiplies
it by the total `batchSize`. This means every request in the batch is attributed
`maxKVLen` worth of attention FLOPs, regardless of its actual KV length.

The decode phase (`roofline_step.go:259-311`) calls this with `maxKVLen` (line 289),
while memory bandwidth (`roofline_step.go:301-305`) correctly iterates over
individual requests and uses per-request actual KV lengths.

The root cause is an asymmetry in how the decode phase handles attention compute
vs. memory bandwidth:
- **Memory bandwidth**: correctly sums per-request `calculateMemoryAccessBytes`
  with each request's actual `ProgressIndex`
- **Attention compute**: uses a single `calculateAttentionCoreFLOPs` call with
  `batchSize * maxKVLen`, conflating all requests to the longest KV length

## Impact Assessment

- **Homogeneous decode batches** (all same KV length): zero impact — no overestimation
- **Heterogeneous decode batches**: overestimation factor = `maxKVLen/meanKVLen`
- **Worst case**: one very long request (e.g., 4096 KV) + many short requests (e.g., 64 KV) gives 12.96x overestimation at batch_size=16
- **Step time impact**: 30.5% higher step time (13,240 vs 10,149 us at bs=16) for heterogeneous vs homogeneous batches
- **Typical workloads**: moderate KV length variance in real workloads will see proportional overestimation; continuous batching naturally creates heterogeneous decode batches as requests at different completion stages share a batch
- **Practical significance**: The overestimation is partially masked by GEMM costs and memory bandwidth (which are correctly computed), limiting the total step time error to ~30% even when attention FLOPs are overestimated by ~13x

## Recommended Fix

Replace the batch-level `calculateAttentionCoreFLOPs(batchSize, maxKVLen)` call
with a sum of per-request calls:

```go
var attnCoreFLOPs float64
for _, req := range stepConfig.DecodeRequests {
    attnCoreFLOPs += calculateAttentionCoreFLOPs(
        modelConfig.NumHeads,
        modelConfig.NumKVHeads,
        modelConfig.HiddenDim,
        1,            // single request
        req.ProgressIndex,  // actual KV length
    ) * float64(modelConfig.NumLayers)
}
```

Note: The MFU lookup (`GetAttnDecodeMFU`) still uses `maxKVLen`, which is a
separate concern — MFU may legitimately depend on the maximum context length
in the batch for GPU utilization estimation.
