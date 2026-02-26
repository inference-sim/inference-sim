# H35: Decode Activation Memory Factor Is Inconsequential -- Findings

**Status**: CONFIRMED
**Date**: 2026-02-25
**Family**: Structural model
**Type**: Deterministic (no RNG, pure roofline computation)

## Hypothesis

The decode activation memory factor (0.75) is inconsequential because activation
bytes constitute less than 0.5% of total memory traffic across all evaluation
operating points (bs=1..256, kvLen=128..8192), so replacing 0.75 with any value
in [0.5, 1.5] changes predicted step time by less than 0.05%.

**Refuted if**: There exists an operating point where changing the decode
activation factor from 0.75 to 1.00 shifts predicted decode step time by more
than 0.1%.

## Experiment Design

- **Independent variable**: Decode activation memory factor [0.50, 0.75, 1.00, 1.50]
- **Controlled variables**: Model config (Llama-3.1-8B), hardware config (H100),
  TP=1, decode-only batches (uniform kvLen per operating point)
- **Dependent variables**:
  - Activation bytes as a fraction of total dynamic memory bytes
  - Decode step time delta (percentage) when factor changes from 0.75 to alternative
- **Operating point grid**: batchSize=[1, 4, 8, 16, 32, 64, 128, 256] x
  kvLen=[128, 256, 512, 1024, 2048, 4096, 8192] = 56 points

## Results

### Activation Fraction of Dynamic Memory Bytes

The activation fraction is independent of batch size (it cancels out since both
activation bytes and dynamic bytes scale linearly with bs). The fraction depends
only on kvLen, decreasing as KV cache access dominates at longer sequences.

| KV Length | Activation Fraction (all bs) |
|:---------:|:----------------------------:|
| 128       | 1.4299%                      |
| 256       | 0.7236%                      |
| 512       | 0.3640%                      |
| 1024      | 0.1825%                      |
| 2048      | 0.0914%                      |
| 4096      | 0.0457%                      |
| 8192      | 0.0229%                      |

**Max activation fraction of dynamic bytes: 1.4299%** (at kvLen=128).

Note: The original hypothesis predicted activation fraction < 0.5% of dynamic
bytes. At kvLen=128, it reaches 1.43%. However, this is the fraction of
*dynamic* memory bytes only -- when model weights (~13.96 GB) are included in
total memory traffic, activation bytes become vanishingly small (< 0.0014% of
total at bs=1, kvLen=128). The key question is whether this fraction is large
enough to measurably affect step time, which Phase 2 addresses directly.

### Step Time Delta: Factor 0.75 vs 1.00

Across all 56 operating points, the maximum step time delta when changing the
activation factor from 0.75 to 1.00 was **0.0120%**, occurring at bs=4,
kvLen=128. The vast majority of operating points (54 of 56) showed 0.0000%
delta due to integer microsecond rounding.

| BS  | KV Length | Step (0.75) us | Step (1.00) us | Delta %  | Regime |
|:---:|:---------:|:--------------:|:--------------:|:--------:|:------:|
| 1   | 128       | 8286           | 8286           | 0.0000%  | MEM    |
| 1   | 2048      | 8360           | 8360           | 0.0000%  | CMP    |
| 1   | 8192      | 8594           | 8594           | 0.0000%  | MEM    |
| 4   | 128       | 8301           | 8302           | 0.0120%  | MEM    |
| 4   | 256       | 8321           | 8321           | 0.0000%  | CMP    |
| 16  | 128       | 10217          | 10217          | 0.0000%  | CMP    |
| 64  | 2048      | 16609          | 16609          | 0.0000%  | CMP    |
| 128 | 4096      | 33842          | 33842          | 0.0000%  | CMP    |
| 256 | 8192      | 104026         | 104026         | 0.0000%  | CMP    |

Most operating points (bs >= 8) are compute-bound (regime=CMP), where the
activation factor change does not affect step time at all since memory time
remains below compute time. Only the smallest batch sizes (bs=1, bs=4) at short
kvLen are memory-bound, but even there the activation delta is negligible.

### Max Delta Across All Factors

| Factor | Max |Delta| % | Mean |Delta| % |
|:------:|:-----------------:|:------------------:|
| 0.50   | 0.0000%           | 0.0000%            |
| 0.75   | 0.0000%           | 0.0000%            |
| 1.00   | 0.0120%           | 0.0002%            |
| 1.50   | 0.0120%           | 0.0006%            |

**Global max delta across all factors: 0.0120%** (at bs=4, kvLen=128, factors
1.00 and 1.50).

Even at the most extreme factor tested (1.50, doubling the activation bytes
compared to 0.75), the maximum step time shift is only 0.012%.

## Verdict

**CONFIRMED.** The decode activation memory factor (0.75) is inconsequential
for predicted step time.

- Changing the factor from 0.75 to 1.00 shifts step time by at most **0.0120%**,
  which is 8.3x below the 0.1% refutation threshold.
- Across all factors in [0.50, 1.50], the maximum step time delta is **0.0120%**,
  which is 4.2x below the 0.05% confirmation threshold stated in the hypothesis.
- The only operating point where any step time difference appears (after
  microsecond rounding) is bs=4, kvLen=128 -- a small batch at the shortest
  sequence length in the grid.

## Root Cause Analysis

The decode activation memory term is computed in `calculateMemoryAccessBytes`
(`sim/roofline_step.go:98-99`):

```go
activationBytes = nLayers * dModel * BytesPerParam * newT * 0.75
```

For Llama-3.1-8B (nLayers=32, dModel=4096, BytesPerParam=2), this equals:
- Per decode token: 32 * 4096 * 2 * 1 * 0.75 = 196,608 bytes (~192 KB)

By comparison, KV cache access for a single token at kvLen=2048 is:
- kvReadPerToken = 2 * 32 * 8 * 128 * 2 = 131,072 bytes per token
- kv_cache_access = 131,072 * 2048 * 0.80 = 214,958,080 bytes (~205 MB)

The ratio is ~1000:1, explaining why activation bytes are negligible relative
to KV cache access at moderate-to-long KV lengths.

At the minimum kvLen=128, activation bytes are 1.43% of dynamic memory bytes,
but model weights (~13.96 GB) still dominate total memory traffic, so the
activation term has negligible impact on total memory time and hence step time.

**Why the factor is inconsequential despite activation fraction reaching 1.43%:**
The activation fraction measures the share of *dynamic* (non-weight) memory bytes.
But total memory bytes = weights + dynamic, and weights dominate at all decode
operating points. At bs=1, kvLen=128, dynamic bytes are only 0.10% of total
memory bytes, so even a 1.43% share of that 0.10% is vanishingly small (~0.001%
of total memory traffic).

## Impact Assessment

**No action required.** The 0.75 activation factor is a cosmetic constant that
has no measurable effect on predicted step time. It can remain hardcoded without
impacting simulation accuracy. If future models have significantly different
architectures (e.g., much larger hidden dimensions relative to KV cache), this
conclusion should be re-evaluated.

The experiment confirms that the roofline model's sensitivity to the activation
discount factor is negligible across the entire evaluation grid, and engineering
effort should be directed at other parameters (MFU calibration, KV cache access
patterns, per-layer overhead) that have materially larger effects on prediction
accuracy.
