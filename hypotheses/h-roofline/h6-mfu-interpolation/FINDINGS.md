# H6: MFU Grid-Boundary Discontinuities

**Status:** Confirmed
**Resolution:** Clean confirmation
**Family:** Structural model
**VV&UQ:** Verification
**Tier:** 0 (zero-config)
**Type:** Deterministic
**Date:** 2026-02-24
**Rounds:** 1

## Hypothesis

> When batch size or sequence length crosses an MFU grid boundary, the simulator's predicted latency jumps discontinuously. This produces excess per-request prediction variance that is an artifact of the lookup method, not of real hardware behavior. The MFU lookup returns a pure step function: across 3,062 fine-grained sweep points, only 37 unique MFU values (1.2%) are returned. The largest discontinuity is 72.7% between adjacent batch sizes.

## Experiment Design

- **Classification:** Deterministic (Part A — simulator-internal)
- **Configurations compared:** Nearest-neighbor MFU lookup across 9 sweep configurations
- **Controlled variables:** Model config (Llama 3.1 8B: 32 heads, 8 KV heads, 128 head_dim, 4096 hidden_dim, 14336 intermediate_dim), GPU (H100), benchmark data (InferSim bench_data)
- **Varied variable:** Input parameter (batch size M=1..512, batch_size=1..256, kv_len=128..16384, seq_len=512..32768 depending on sweep)
- **Seeds:** N/A — deterministic experiment (single path through lookup algorithm)
- **Preconditions verified:** MFU database loads successfully (5 prefill rows, 755 decode rows, 112 GEMM rows)

### Sweep Configurations

| Sweep | Lookup Type | Fixed Params | Swept Param | Range | Step |
|-------|------------|--------------|-------------|-------|------|
| 1 | GEMM | K=4096, N=6144 | M (batch) | 1-512 | 1 |
| 2 | GEMM | K=4096, N=11008 | M (batch) | 1-512 | 1 |
| 3a | Decode Attn | KV=1024, TP=1 | batch_size | 1-256 | 1 |
| 3b | Decode Attn | KV=4096, TP=1 | batch_size | 1-256 | 1 |
| 3c | Decode Attn | KV=8192, TP=1 | batch_size | 1-256 | 1 |
| 4a | Decode Attn | BS=1, TP=1 | kv_len | 128-16384 | 64 |
| 4b | Decode Attn | BS=32, TP=1 | kv_len | 128-16384 | 64 |
| 4c | Decode Attn | BS=128, TP=1 | kv_len | 128-16384 | 64 |
| 5 | Prefill Attn | - | seq_len | 512-32768 | 64 |

## Results

### Aggregate Summary

| Metric | Value |
|--------|-------|
| Total sweep points | 3,062 |
| Total unique MFU values returned | 37 (1.2%) |
| Total >=5% discontinuities | 29 |
| Total >=10% discontinuities | 21 |
| Total >=20% discontinuities | 19 |
| Discontinuity rate (>=5%) | 0.95% of adjacent pairs |

### Per-Sweep Results

| Sweep | Points | Unique MFU | Unique % | Disc >=5% | Disc >=10% | Disc >=20% | Max Disc | Longest Flat |
|-------|--------|-----------|----------|-----------|------------|------------|----------|-------------|
| GEMM (K=4096, N=6144) | 512 | 7 | 1.4% | 6 | 6 | 6 | 51.9% | 256 |
| GEMM (K=4096, N=11008) | 512 | 7 | 1.4% | 6 | 6 | 6 | 50.0% | 256 |
| Decode Attn (KV=1024) | 256 | 3 | 1.2% | 2 | 1 | 1 | 20.0% | 129 |
| Decode Attn (KV=4096) | 256 | 3 | 1.2% | 2 | 1 | 1 | 72.7% | 193 |
| Decode Attn (KV=8192) | 256 | 3 | 1.2% | 3 | 1 | 1 | 58.3% | 193 |
| Decode Attn (BS=1, KV sweep) | 255 | 4 | 1.6% | 3 | 3 | 3 | 62.5% | 128 |
| Decode Attn (BS=32, KV sweep) | 255 | 3 | 1.2% | 2 | 1 | 0 | 15.4% | 129 |
| Decode Attn (BS=128, KV sweep) | 255 | 2 | 0.8% | 1 | 0 | 0 | 8.3% | 193 |
| Prefill Attn (seq sweep) | 505 | 5 | 1.0% | 4 | 2 | 1 | 64.3% | 256 |

### Key Observations

1. **GEMM lookup is worst:** Only 7 unique values for 512 sweep points. The GEMM grid has M values at {8, 16, 32, 64, 128, 256, 512, 1024, 4096, ...}, so M=257 through M=512 all return the M=256 MFU value (0.495 for K=4096, N=6144), while M=1 through M=8 all return the M=8 value (0.013). The jump from M=256 (MFU=0.495) to M=512 would be the next step.

2. **Decode attention has zero-MFU problem:** For batch_size=1 at KV=1024, the benchmark data records MFU=0.0 (below the 0.0001 threshold), forcing a fallback to the nearest non-zero entry (bs=16, kv=1024, MFU=0.008). This means batch sizes 1-15 all return the same fallback value, then jump at bs=16 to the actual benchmarked value — creating a 72.7% discontinuity at kv=4096.

3. **Prefill lookup has 5 grid points covering 32K range:** The H100 prefill data for 32-8-128 has entries at seq_len={1024, 4096, 8192, 16384, 32768}. With floor-preference nearest-neighbor, seq_len=512 through seq_len=4095 all return the seq_len=1024 MFU (0.228), then jump to 0.638 at seq_len=4096 — a 64.3% discontinuity.

4. **Flat regions dominate:** The longest flat run is 256 consecutive identical values (GEMM and prefill sweeps). This means the simulator predicts identical latency for batch sizes 257-512 (GEMM) or sequence lengths 1024-4095 (prefill), which is physically unrealistic — larger GEMMs achieve higher MFU.

## Root Cause Analysis

### Why discontinuities exist

The root cause is the nearest-neighbor lookup algorithm combined with sparse benchmark grid points:

1. **GEMM lookup** (`sim/mfu_database.go:527-600`): Stage 2 finds `largest m <= target_m` for a fixed (K, N). The GEMM CSV (`InferSim/bench_data/gemm/h100/data.csv`) has M values at powers of 2: {8, 16, 32, 64, 128, 256, 512, 1024, 4096, 8192, ...}. For any M between grid points (e.g., M=200), the lookup returns the MFU at the floor grid point (M=128, MFU=0.176), creating a plateau from M=129 to M=255 where MFU doesn't change. At M=256, MFU jumps to 0.323 — an 83.5% increase.

2. **Decode attention lookup** (`sim/mfu_database.go:438-525`): Uses 2D Euclidean distance on (batch_size, kv_len) with floor preference. The decode CSV has batch sizes at {1, 16, 32, 64, 128, 256, 512} and KV lengths at {1024, 4096, 8192, 16384, 32768, 65536, 131072}. Between these grid points, the same MFU is returned for wide ranges.

3. **Prefill attention lookup** (`sim/mfu_database.go:362-436`): 1D nearest-neighbor with floor preference. Only 5 data points for the 32-8-128 config span seq_len 1024-32768.

4. **Zero-MFU fallback** (`sim/mfu_database.go:404-427`): When the nearest neighbor has MFU < 0.0001, the code finds the nearest non-zero MFU anywhere in the dataset. This creates long flat regions where many input values (e.g., batch_size=1 through 15) all map to the same fallback, then jump sharply when reaching an actual benchmarked point.

### Why this matters for prediction accuracy

The step-function behavior means:
- **Two requests with batch sizes 100 and 200 get identical predicted step time** (both map to GEMM M=64 MFU), when the real hardware would give different latencies.
- **A batch size change from 128 to 129 has zero effect on predicted latency**, but a change from 127 to 128 triggers a step — the opposite of physical reality where kernel performance varies smoothly.
- **Per-request prediction variance has artificial structure** tied to grid boundaries, not to real hardware behavior.

### Scale of impact

For GEMM-dominated steps (prefill with large batch), the MFU enters the `computeGEMMTime` formula (`sim/roofline_step.go:122-131`) as `time = flops / (peakFlops * mfu)`. A 50% MFU discontinuity translates to a ~33% latency discontinuity (`1/1.5 = 0.67`). This is amplified because GEMM time is summed across 7 projections × 32 layers = 224 lookups per step.

## Devil's Advocate (RCV-5)

**If "Confirmed," argue why it might be Refuted:**
The discontinuities, while large in relative terms (up to 72.7%), occur at low absolute MFU values (0.003-0.012 for decode attention). In these regimes, decode latency is memory-bandwidth-bound, not compute-bound, so the MFU lookup result may not meaningfully affect the final step time (the `max(compute, memory)` in the roofline formula selects the memory term). The GEMM discontinuities are more consequential since they affect compute-bound regimes. Additionally, in practice, batch sizes in vLLM tend to cluster at values close to the grid points (powers of 2 are common batch sizes), so the between-grid-point regions may be less exercised than the sweep suggests.

**If "Refuted," argue why it might be Confirmed:**
N/A — the experiment measures a deterministic property of the algorithm, not a statistical claim.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| MFU lookup is a step function: 37 unique values for 3,062 points | Confirmation | Enhancement: implement linear interpolation in `GetGEMMmfu`, `GetAttnDecodeMFU`, `GetAttnPrefillMFU` |
| GEMM lookup: 7 unique values for 512 batch sizes, max 51.9% jump | Confirmation | Same enhancement as above |
| Decode attention zero-MFU fallback creates 72.7% discontinuity | Bug/Design limitation | Enhancement: improve zero-MFU handling or add data points for small batch sizes |
| Prefill attention: only 5 grid points for seq_len 1024-32768 | Confirmation | Enhancement: interpolation would produce smooth curve |
| Longest flat run: 256 consecutive identical MFU values | Confirmation | Documented here |

## Standards Audit

- [x] Any violations of existing rules? None — nearest-neighbor is not incorrect, just suboptimal
- [x] Any new rules needed? Consider: "MFU lookup should be monotonic in batch size" — a batch size increase should never decrease predicted MFU
- [x] Any new invariants needed? Candidate: MFU(M+1, K, N) >= MFU(M, K, N) for GEMM — monotonicity in M
- [x] Any existing rules/invariants confirmed? INV-6 (determinism) — the sweep produces identical results on every run

## Scope and Limitations (RCV-6)

- **Operating point tested:** Llama 3.1 8B model config on H100 GPU, InferSim bench_data
- **Parameters findings depend on:** Benchmark data grid spacing (would be different for other GPUs if they have denser grids)
- **What was NOT tested:** (1) Part B of H6 — aggregate prediction accuracy impact on ground truth experiments (requires comparing BLIS predictions vs measured latency with and without interpolation). (2) Other model configs (different attention shapes would use different CSV files). (3) Bilinear interpolation vs linear interpolation vs spline. (4) Impact on end-to-end simulation latency (requires roofline integration test).
- **Generalizability:** The step-function behavior is inherent to the nearest-neighbor algorithm and will affect any model/GPU combination. The severity depends on grid density — sparser grids produce worse discontinuities. H100 data has relatively sparse grids (5 prefill points, ~7 decode batch sizes, ~14 GEMM M values).
- **Uncertainty quantification:** None needed — this is a deterministic property of the algorithm. The 37/3062 unique ratio and 29 discontinuities are exact counts.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| Unique MFU ratio | 37/3062 = 1.2% | High — exact deterministic count |
| Discontinuities >=5% | 29 across 9 sweeps | High — exact count, reproducible |
| Max discontinuity | 72.7% (decode attn KV=4096, bs=15→16) | High — traceable to zero-MFU fallback code path |
| Mechanism | Nearest-neighbor lookup on sparse grid | High — verified by code trace through `mfu_database.go:362-600` |
| Flat regions | Up to 256 consecutive identical values | High — directly observable in CSV output |

## Implications for Users

1. **BLIS roofline predictions exhibit step-function artifacts** at MFU grid boundaries. Users should be aware that predicted latency may jump discontinuously when batch size or sequence length crosses a boundary (approximately at powers of 2).

2. **The artifacts are most severe for GEMM-dominated workloads** (prefill with moderate-to-large batch sizes) where the GEMM MFU lookup is the primary accuracy determinant. Decode-dominated workloads are less affected because decode attention MFU values are very low (0.003-0.013) and decode latency is typically memory-bandwidth-bound.

3. **Linear interpolation would be the standard fix.** For GEMM lookup (1D in M), linear interpolation between adjacent grid points is straightforward. For decode attention (2D in batch_size × kv_len), bilinear interpolation is the natural extension. For prefill (1D in seq_len), linear interpolation between the 5 grid points would smooth the 64.3% jump at seq_len=4096.

4. **The zero-MFU fallback for batch_size=1 at small KV lengths is a separate data quality issue.** The benchmark records MFU=0.0 for these configurations, which the lookup code handles by finding the nearest non-zero value — but this creates large flat regions. This could be addressed by either: (a) adding benchmark data for these configurations, or (b) extrapolating from nearby non-zero values.

## Reproducing

```bash
cd hypotheses/h-roofline/h6-mfu-interpolation
./run.sh
```
