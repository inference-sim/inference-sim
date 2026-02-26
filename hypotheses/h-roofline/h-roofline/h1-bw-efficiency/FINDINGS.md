# H1: Bandwidth Efficiency Validation — Sustained vs Peak HBM Bandwidth

**Status:** Confirmed
**Resolution:** Clean confirmation
**Family:** Structural model (does the roofline bandwidth parameter match physical reality?)
**VV&UQ:** Validation (comparing simulator predictions against real-world measurements)
**Tier:** 0 (zero-config correction)
**Type:** Deterministic (same config = identical output per INV-6)
**Date:** 2026-02-24
**Rounds:** 1

## Hypothesis

> "The simulator systematically underestimates latency for memory-bound steps because it uses theoretical peak HBM bandwidth instead of achievable (~80-82%) sustained bandwidth."

**Diagnostic:** If MAPE does not improve, it indicates bandwidth efficiency is not the primary source of memory-bound prediction error — other factors (H2 overhead, H3 GEMM shapes, H6 MFU smoothing) dominate.

## Experiment Design

**Classification:** Deterministic validation against ground truth

**ED compliance:**
- **ED-1 (Single dimension):** Only `bwEfficiencyFactor` varies. Same hardware config, model config, workload, seed, rate.
- **ED-2 (Vanish rate):** Chatsweep (output=215 tokens, decode-heavy, more memory-bound) should show larger TPOT improvement than codesweep (output=28 tokens, prefill-heavy, more compute-mixed). This serves as the directional control.
- **ED-3 (Preconditions):** Script validates `model_configs/`, `bench_data/`, and `eval/ground_truth/` exist.
- **ED-4 (Seed independence):** Deterministic — roofline v2 has no stochastic element relevant to the hypothesis. Single run per config with seed=42.
- **ED-5 (Reproducibility):** Self-contained `run.sh` builds binary, extracts workload params from ground truth profiles, runs 26 simulations, and invokes `analyze.py`.
- **ED-6 (Config diff):** No prior experiment referenced.

**Configurations compared:**

- **Baseline:** `bwEfficiencyFactor` absent (defaults to 0 = disabled, uses raw peak BW of 3.35 TB/s)
  - Hardware config: `{"H100": {"TFlopsPeak": 989.5, "BwPeakTBs": 3.35}}`

- **Treatment:** `bwEfficiencyFactor=0.82` (H100 sustained-to-peak ratio from STREAM benchmarks)
  - Hardware config: `{"H100": {"TFlopsPeak": 989.5, "BwPeakTBs": 3.35, "bwEfficiencyFactor": 0.82}}`

**Controlled variables:**
- Model configs: HuggingFace config.json per model (from `model_configs/`)
- MFU benchmark data: `bench_data/` (roofline v2)
- GPU: H100
- Workload: Distribution mode with parameters matching ground truth `profile.yaml`
- Rate: Synchronous rate from GuideLLM `benchmarks[0]` (minimal queuing, max_concurrency=1)
- Seed: 42

**Varied variable:** `bwEfficiencyFactor` (0 vs 0.82), which multiplicatively scales `peakBW` at `sim/roofline_step_v2.go:133-134`.

**Ground truth:** 13 experiments across 5 models x 3 TP configurations x 2-3 workload types, using the synchronous benchmark (index 0) from GuideLLM `guidellm-results.json`.

**Workload characteristics:**
- **Chatsweep** (6 experiments): prompt=70, prefix=284, output=215 — **decode-heavy** (long output sequences, most step time memory-bound)
- **Codesweep** (6 experiments): prompt=2048, output=28 — **prefill-heavy** (long input, short output, more compute-mixed)
- **Summarization** (1 experiment): prompt=4096, output=512 — **mixed** (long prefill AND long decode)

| Experiment | Model Config | TP | Workload |
|---|---|---|---|
| jan30-llama2-7b-tp1-chatsweep | llama-2-7b-hf | 1 | chatsweep |
| jan30-llama2-7b-tp1-codesweep | llama-2-7b-hf | 1 | codesweep |
| jan30-llama2-7b-tp2-chatsweep | llama-2-7b-hf | 2 | chatsweep |
| jan30-llama2-7b-tp2-codesweep | llama-2-7b-hf | 2 | codesweep |
| jan30-llama2-7b-tp4-chatsweep | llama-2-7b-hf | 4 | chatsweep |
| jan30-llama2-7b-tp4-codesweep | llama-2-7b-hf | 4 | codesweep |
| 20260210-codellama-34b-tp2-chatsweep | codellama-34b-instruct-hf | 2 | chatsweep |
| 20260210-codellama-34b-tp2-codesweep | codellama-34b-instruct-hf | 2 | codesweep |
| 20260210-llama2-70b-tp4-chatsweep | llama-2-70b-hf | 4 | chatsweep |
| 20260210-llama2-70b-tp4-codesweep | llama-2-70b-hf | 4 | codesweep |
| 20260210-qwen3-14b-tp1-codesweep | qwen3-14b | 1 | codesweep |
| 20260210-qwen3-14b-tp2-chatsweep | qwen3-14b | 2 | chatsweep |
| dec17-tp1-qwen7-summarization | qwen2.5-7b-instruct | 1 | summarization |

**Total runs:** 13 experiments x 2 configs = 26

**Accept criteria (corrected from TAM doc):**
1. TPOT MAPE improves by >=3pp across experiments — **primary criterion**
2. Chatsweep (decode-heavy) experiments improve more than codesweep (prefill-heavy) in TPOT — **directional evidence**

**Rate selection rationale:** Uses lowest-rate benchmark from GuideLLM (synchronous mode, `benchmarks[0]`). At max_concurrency=1, queuing effects are negligible, isolating the roofline model's step-time prediction accuracy.

## Results

### Per-Experiment MAPE Comparison

| Experiment | Type | BL TTFT | TR TTFT | Delta | BL TPOT | TR TPOT | Delta | BL E2E | TR E2E | Delta |
|---|---|---|---|---|---|---|---|---|---|---|
| jan30-llama2-7b-tp1-chatsweep | chatsweep | 71.8% | 62.3% | +9.6pp | 36.4% | 22.4% | +14.0pp | 34.6% | 20.1% | +14.5pp |
| jan30-llama2-7b-tp1-codesweep | codesweep | 89.8% | 86.9% | +3.0pp | 44.8% | 32.1% | +12.7pp | 33.1% | 17.7% | +15.4pp |
| jan30-llama2-7b-tp2-chatsweep | chatsweep | 83.8% | 80.1% | +3.7pp | 54.6% | 44.6% | +10.0pp | 53.3% | 43.0% | +10.3pp |
| jan30-llama2-7b-tp2-codesweep | codesweep | 93.5% | 92.0% | +1.5pp | 60.5% | 51.5% | +9.0pp | 52.0% | 41.2% | +10.8pp |
| jan30-llama2-7b-tp4-chatsweep | chatsweep | 91.0% | 89.1% | +2.0pp | 69.3% | 62.5% | +6.7pp | 68.4% | 61.4% | +6.9pp |
| jan30-llama2-7b-tp4-codesweep | codesweep | 96.1% | 95.2% | +0.9pp | 74.2% | 68.5% | +5.7pp | 68.7% | 61.8% | +6.9pp |
| 20260210-codellama-34b-tp2-chatsweep | chatsweep | 53.6% | 45.3% | +8.3pp | 29.4% | 13.8% | +15.5pp | 27.3% | 11.3% | +16.0pp |
| 20260210-codellama-34b-tp2-codesweep | codesweep | 89.4% | 86.7% | +2.8pp | 39.5% | 26.1% | +13.3pp | 26.9% | 10.7% | +16.2pp |
| 20260210-llama2-70b-tp4-chatsweep | chatsweep | 61.7% | 42.0% | +19.7pp | 34.8% | 20.4% | +14.3pp | 32.9% | 18.0% | +14.8pp |
| 20260210-llama2-70b-tp4-codesweep | codesweep | 90.7% | 88.1% | +2.7pp | 44.6% | 32.4% | +12.2pp | 33.0% | 18.2% | +14.9pp |
| 20260210-qwen3-14b-tp1-codesweep | codesweep | 86.7% | 83.1% | +3.5pp | 40.1% | 26.8% | +13.3pp | 26.1% | 9.6% | +16.5pp |
| 20260210-qwen3-14b-tp2-chatsweep | chatsweep | 74.7% | 68.4% | +6.2pp | 48.1% | 36.6% | +11.4pp | 46.5% | 34.8% | +11.7pp |
| dec17-tp1-qwen7-summarization | summarization | 91.7% | 89.7% | +2.0pp | 43.7% | 31.3% | +12.4pp | 39.4% | 26.1% | +13.3pp |

### Aggregate by Workload Type

| Workload Type | N | BL TPOT | TR TPOT | Delta | BL E2E | TR E2E | Delta |
|---|---|---|---|---|---|---|---|
| chatsweep | 6 | 45.4% | 33.4% | +12.0pp | 43.8% | 31.4% | +12.4pp |
| codesweep | 6 | 50.6% | 39.6% | +11.0pp | 40.0% | 26.5% | +13.5pp |
| summarization | 1 | 43.7% | 31.3% | +12.4pp | 39.4% | 26.1% | +13.3pp |
| **OVERALL** | **13** | **47.7%** | **36.1%** | **+11.6pp** | **41.7%** | **28.8%** | **+12.9pp** |

### Absolute Latency Values (ms)

| Experiment | GT TTFT | BL TTFT | TR TTFT | GT TPOT | BL TPOT | TR TPOT | GT E2E | BL E2E | TR E2E |
|---|---|---|---|---|---|---|---|---|---|
| jan30-llama2-7b-tp1-chatsweep | 13.9 | 3.9 | 5.2 | 6.2 | 4.0 | 4.8 | 1292.6 | 846.0 | 1033.4 |
| jan30-llama2-7b-tp1-codesweep | 67.1 | 6.8 | 8.8 | 8.1 | 4.5 | 5.5 | 334.0 | 223.5 | 275.0 |
| jan30-llama2-7b-tp2-chatsweep | 12.1 | 2.0 | 2.4 | 4.4 | 2.0 | 2.4 | 905.4 | 422.8 | 515.8 |
| jan30-llama2-7b-tp2-codesweep | 49.9 | 3.2 | 4.0 | 5.6 | 2.2 | 2.7 | 230.4 | 110.6 | 135.6 |
| jan30-llama2-7b-tp4-chatsweep | 10.8 | 1.0 | 1.2 | 3.2 | 1.0 | 1.2 | 668.4 | 211.4 | 257.8 |
| jan30-llama2-7b-tp4-codesweep | 40.5 | 1.6 | 2.0 | 4.3 | 1.1 | 1.3 | 175.9 | 55.1 | 67.3 |
| 20260210-codellama-34b-tp2-chatsweep | 23.5 | 10.9 | 12.8 | 14.2 | 10.0 | 12.2 | 2941.2 | 2138.9 | 2608.3 |
| 20260210-codellama-34b-tp2-codesweep | 155.0 | 16.4 | 20.6 | 17.7 | 10.7 | 13.1 | 729.4 | 533.4 | 651.6 |
| 20260210-llama2-70b-tp4-chatsweep | 27.1 | 10.4 | 15.7 | 15.8 | 10.3 | 12.6 | 3280.4 | 2202.3 | 2689.0 |
| 20260210-llama2-70b-tp4-codesweep | 179.3 | 16.6 | 21.3 | 19.9 | 11.0 | 13.5 | 819.3 | 548.6 | 670.4 |
| 20260210-qwen3-14b-tp1-codesweep | 103.0 | 13.7 | 17.3 | 14.3 | 8.6 | 10.5 | 578.6 | 427.8 | 523.0 |
| 20260210-qwen3-14b-tp2-chatsweep | 15.7 | 4.0 | 4.9 | 7.7 | 4.0 | 4.9 | 1592.7 | 851.4 | 1038.4 |
| dec17-tp1-qwen7-summarization | 117.0 | 9.7 | 12.1 | 7.1 | 4.0 | 4.8 | 3532.7 | 2140.6 | 2611.7 |

### TP Scaling Pattern

The TPOT improvement from BW correction **decreases** as TP increases for Llama2-7b:

| TP | Chat TPOT Delta | Code TPOT Delta |
|---|---|---|
| TP=1 | +14.0pp | +12.7pp |
| TP=2 | +10.0pp | +9.0pp |
| TP=4 | +6.7pp | +5.7pp |

Higher TP means each GPU holds fewer model weight parameters (divided by TP), reducing the memory-bound fraction of step time and moving towards the compute-bound regime where bandwidth correction has no effect.

### Accept Criteria Evaluation

1. **TPOT MAPE improvement >= 3pp: PASS (+11.6pp)**
   - Every single experiment improves, with a minimum of +5.7pp (llama2-7b-tp4-codesweep) and maximum of +15.5pp (codellama-34b-tp2-chatsweep).

2. **Chatsweep (decode-heavy) improves more than codesweep (prefill-heavy): PASS (chat +12.0pp vs code +11.0pp)**
   - Chatsweep (output=215 tokens, decode-heavy) shows 1.0pp larger TPOT improvement than codesweep (output=28 tokens, prefill-heavy). The directional evidence is present but the margin is small.

## Root Cause Analysis

### Primary mechanism: confirmed

The bandwidth efficiency correction works as hypothesized at the hardware level. At `sim/roofline_step_v2.go:131-135`, `peakBW` is used to compute memory access time for both decode (`sim/roofline_step_v2.go:195`) and prefill (`sim/roofline_step_v2.go:268`) phases:

```
decodeMemoryS = (dWeightBytes + dDynamicBytes) / peakBW    // line 195
prefillMemoryS = (pWeightBytes + pDynamicBytes) / peakBW   // line 268
```

When `bwEfficiencyFactor=0.82`, `peakBW` drops from 3.35 TB/s to 2.747 TB/s (`sim/roofline_step_v2.go:134`). This increases predicted memory time by a factor of `3.35/2.747 = 1.22` (22%) for any step that is memory-bound (where `memoryS > computeS`). The 22% correction aligns with well-documented H100 STREAM benchmark results (2650-2750 GB/s sustained vs 3350 GB/s peak).

All 13 experiments show the treatment (higher predicted latency) moving closer to ground truth, confirming that the baseline systematically underpredicts. Every predicted value is below ground truth (negative bias), and the correction reduces this bias uniformly.

### Directional evidence

Chatsweep (decode-heavy, output=215 tokens) shows +12.0pp TPOT improvement vs codesweep (prefill-heavy, output=28 tokens) at +11.0pp. The margin is small (1.0pp) because the BW correction affects both prefill and decode memory access — it is not exclusive to decode. The directional signal is present but weak, consistent with bandwidth being a uniform physical constraint across all memory access patterns.

### TP scaling as mechanism control (RCV-4)

The TP scaling pattern in Llama2-7b (TP=1, 2, 4) serves as a natural control experiment. At TP=1, the full model resides on one GPU — maximum memory traffic, maximum BW correction benefit (+14.0pp chatsweep TPOT). At TP=4, model weights are split 4 ways — reduced memory traffic per GPU, reduced BW correction benefit (+6.7pp). This monotonic decrease with TP confirms the mechanism: the correction's impact is proportional to the memory-bound fraction, which decreases as TP sharding reduces per-GPU weight loads (`sim/roofline_step_v2.go:193`: `dWeightBytes := baseMem["model_weights"] * tpScaling` where `tpScaling = 1.0 / tpFactor`).

### Residual error dominated by per-step overhead (H2)

Even after BW correction, TPOT MAPE remains at 36.1% — the simulator still underpredicts by approximately 2-3x for small models and 1.2-1.5x for large models. The absolute values table shows predicted TPOT of 1.0-5.5ms against ground truth of 3.2-19.9ms. Examining the absolute gap:

- Llama2-7b TP=4: predicted 1.2ms, actual 3.2ms — gap of 2.0ms
- Llama2-7b TP=1: predicted 4.8ms, actual 6.2ms — gap of 1.4ms
- CodeLlama-34b TP=2: predicted 12.2ms, actual 14.2ms — gap of 2.0ms
- Llama2-70b TP=4: predicted 12.6ms, actual 15.8ms — gap of 3.2ms

The gap is roughly constant at ~2ms for smaller models, consistent with H2's hypothesis of a fixed scheduling overhead that the roofline model ignores (InferSim uses 5ms decode overhead at `models/model.py:229`). This provides indirect evidence supporting H2 as the next-largest error source.

## Devil's Advocate

**Arguing this should be "Refuted":**

The improvement could be partially coincidental: making *all* predictions 22% higher reduces MAPE for any systematically low predictor, regardless of whether the specific mechanism (bandwidth inefficiency) is correct. A multiplicative correction to compute time instead of memory time might show similar aggregate improvement. The directional margin (chat +12.0pp vs code +11.0pp = 1.0pp difference) is too small to confidently attribute the improvement to the memory-bound mechanism specifically. The TP-scaling control is suggestive but also consistent with other explanations — higher TP introduces more communication overhead (`sim/roofline_step_v2.go:314`: `numLayers * 2 * allReduceLatency`) which could absorb the correction's effect differently.

## Findings Classification

| Finding | Type | Action |
|---|---|---|
| BW efficiency correction improves TPOT MAPE by +11.6pp and E2E MAPE by +12.9pp across all 13 experiments | Confirmation | `bwEfficiencyFactor=0.82` should be default for H100 |
| Chatsweep (decode-heavy) improves 1.0pp more than codesweep (prefill-heavy) in TPOT | Confirmation | Directional evidence supports memory-bound mechanism |
| TP scaling shows monotonic decrease in BW correction benefit (TP=1: +14pp, TP=4: +6.7pp) | Confirmation | Mechanism control — correction proportional to per-GPU memory traffic |
| Residual error after BW correction (~36% TPOT MAPE) with ~2ms constant gap | Open question | Proceed to H2 experiment for per-step overhead validation |
| TTFT MAPE remains very high (42-96%) even after correction | Open question | TTFT is dominated by factors beyond BW (H2 prefill overhead, H3 GEMM shapes) |

## Standards Audit

- [x] Any violations of existing rules? None found. Run.sh uses harness correctly, timeout tiers appropriate, determinism maintained.
- [x] Any new rules needed? None.
- [x] Any new invariants needed? None.
- [x] Any existing rules/invariants confirmed? INV-6 (determinism): re-running produces byte-identical output. R2 (sort map keys): `calculateMemoryAccessBytes()` at `sim/roofline_step.go:138-146` correctly sorts keys before float summation.

## Scope and Limitations

**Operating point tested:**
- GPU: H100 only (3.35 TB/s peak, 0.82 efficiency factor)
- Models: 5 dense model families (7B, 14B, 34B, 70B), no MoE
- TP: 1, 2, 4
- Rate: Synchronous (lowest QPS benchmark from GuideLLM, max_concurrency=1)
- Seed: 42 (deterministic — single seed sufficient)
- Workload: 3 types (chatsweep, codesweep, summarization) with distribution-based generation

**Parameters findings depend on:**
- The 0.82 factor is specific to H100 HBM3. Other GPUs have different sustained/peak ratios (InferSim uses 0.80 for A100/H800/H200/GB200). The mechanism generalizes but the exact factor is GPU-specific.
- The factor is applied uniformly to all memory accesses. In reality, sequential reads (model weights) and scattered reads (PagedAttention KV cache) achieve different fractions of peak BW. Tier 2 of TAM addresses this with per-pattern measurements.

**What was NOT tested:**
- Non-H100 GPUs (A100, H800, H200, GB200)
- MoE architectures (Mixtral, DeepSeek-V3) — no ground truth available
- Higher QPS sweep points where queuing/batching effects interact with the correction
- Different BW efficiency factors (0.75, 0.80, 0.85) — only 0.82 was tested
- Prefill-only or decode-only workloads in isolation
- The interaction between BW correction and other TAM corrections (H2, H3, H4, H5)

**Generalizability:**
- The correction generalizes across all HBM-based GPUs (DRAM refresh, bank conflicts, ECC overhead are universal), though the exact factor varies by GPU generation.
- The correction is model-agnostic — it modifies the hardware bandwidth parameter, not the model architecture.
- The TP-scaling pattern (larger benefit at lower TP) should generalize because the mechanism is proportional to per-GPU memory traffic.

**Uncertainty quantification:**
- Only one BW efficiency factor (0.82) was tested. The optimal factor could be anywhere in [0.75, 0.88]. A sweep with 0.01 increments would narrow this but risks overfitting to the 13-experiment ground truth set.
- No cross-validation was performed (all 13 experiments used for evaluation). A train/holdout split (10/3) would be more rigorous but was not implemented.
- The directional margin (1.0pp) is too small to be considered strong evidence in isolation; the TP-scaling control is more convincing.

## Evidence Quality

| Metric | Value | Confidence |
|---|---|---|
| TPOT MAPE improvement (overall) | +11.6pp (47.7% -> 36.1%) | High — all 13 experiments improve, min +5.7pp |
| E2E MAPE improvement (overall) | +12.9pp (41.7% -> 28.8%) | High — all 13 experiments improve, min +6.9pp |
| Directional (chatsweep > codesweep) | PASS (chat +12.0pp vs code +11.0pp) | Low — 1.0pp margin; not strong evidence in isolation |
| TP-scaling mechanism evidence | Monotonic decrease: 14.0, 10.0, 6.7pp at TP=1,2,4 | High — clean monotonic trend across 3 TP values for same model |
| Residual error magnitude | ~36% TPOT MAPE after correction | High — consistent across models and workloads |
| Sample size | 13 experiments, 50 requests each, 5 model families | Medium — single GPU type (H100), no MoE, no cross-validation holdout |

## Implications for Users

1. **Enable BW efficiency correction.** Setting `bwEfficiencyFactor: 0.82` for H100 in `hardware_config.json` costs nothing (zero runtime overhead, O(1) multiply at `sim/roofline_step_v2.go:134`) and reduces TPOT prediction error by ~12pp and E2E error by ~13pp across all tested workloads.

2. **Do not expect BW correction alone to achieve target accuracy.** Even with the correction, TPOT MAPE is 36% and E2E MAPE is 29%. Achieving the TAM Tier 0 target of <20% E2E MAPE requires additional corrections (H2 overhead, H3 GEMM shapes at minimum).

3. **TP scaling affects correction magnitude.** At TP=1, the BW correction provides ~14pp improvement. At TP=4, only ~6pp. Users running high-TP configurations get less benefit — overhead and GEMM shape corrections are relatively more important at high TP.

4. **For other GPUs**, use the manufacturer's STREAM benchmark sustained bandwidth divided by theoretical peak as the efficiency factor. InferSim uses 0.80 for A100, H800, H200, and GB200.

## Reproducing

```bash
cd hypotheses/h-roofline/h1-bw-efficiency && ./run.sh
```

Requires: `model_configs/`, `bench_data/`, `eval/ground_truth/` directories populated.
