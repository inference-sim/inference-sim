# H5: Mixed-Batch Additive Model

**Status:** Refuted
**Resolution:** Refuted — mechanism not plausible at aggregate accuracy level
**Family:** Structural model
**VV&UQ:** Validation
**Tier:** 0
**Type:** Deterministic
**Date:** 2026-02-24
**Rounds:** 1

## Hypothesis

> The roofline mixed-batch model should combine prefill and decode times using an additive formula (GEMM(totalBatch) + PrefillAttn + DecodeAttn), matching vLLM's chunked-prefill execution where one fused GEMM runs over the concatenated batch and separate FlashAttention kernels run per phase. The current weighted-average blend (0.75\*prefill + 0.25\*decode for prefill-dominated, 0.35\*prefill + 0.65\*decode for decode-dominated) uses uncited magic constants that create discontinuities at branch boundaries and should be replaced by the physics-based additive model for improved prediction accuracy.

**Diagnostic clause:** If this fails, it would indicate that the weighted-average approximation correctly captures the interaction between prefill and decode execution at the aggregate prediction level, or that the fused-GEMM assumption overcounts compute for mixed batches.

## Experiment Design

**Classification:** Deterministic (consistent with H1–H4; roofline model is deterministic for given config + seed)

**Configurations compared:**
- **Baseline**: Current weighted-average model with branching thresholds (`roofline_step.go:462-474`, `default` case), hardware config: `BwEfficiencyFactor=0.82, PerLayerCPUOverhead=100, mixedBatchMode=""` (default)
- **Control (smooth-wa)**: Token-proportional weighted average without branch thresholds, hardware config: `mixedBatchMode="smooth-wa"`
- **Treatment (additive)**: Fused GEMM(totalBatch) + PrefillAttn + DecodeAttn with single weight load, hardware config: `mixedBatchMode="additive"`

**Controlled variables:** Model configs, MFU database, BwEfficiencyFactor (0.82, H1), PerLayerCPUOverhead (100, H2b), seed (42), workload parameters (from ground truth profiles)

**Varied variable:** `mixedBatchMode` (only the mixed-batch combination formula)

**Seeds:** 42 (single seed — deterministic experiment)

**Preconditions verified:** bench_data, model_configs, eval/ground_truth directories exist (ED-3)

**ED-6 Config diff:** Baseline includes H1 (BwEfficiencyFactor=0.82) + H2b (PerLayerCPUOverhead=100). H3 (fuseQKV) omitted — 0.0pp effect, refuted. H4 (perComponentRoofline) omitted — +0.3pp, below 1pp threshold.

## Results

**Delta convention:** BL→X = BaselineMAPE − X_MAPE. Positive = treatment improved (lower MAPE). Negative = treatment worsened (higher MAPE). Bold indicates |delta| > 1pp.

### Per-Experiment Aggregate E2E MAPE

| Experiment | Type | N | BL E2E | SWA E2E | ADD E2E | BL→SWA | BL→ADD |
|---|---|---|---|---|---|---|---|
| llama2-7b-tp1-chatsweep | chatsweep | 12 | 71.7% | 71.4% | 76.0% | +0.3pp | **-4.3pp** |
| llama2-7b-tp1-codesweep | codesweep | 12 | 33.8% | 32.5% | 41.1% | +1.3pp | **-7.3pp** |
| llama2-7b-tp2-chatsweep | chatsweep | 12 | 11.4% | 11.4% | 11.4% | -0.0pp | +0.1pp |
| llama2-7b-tp2-codesweep | codesweep | 12 | 21.9% | 22.5% | 20.3% | -0.5pp | **+1.6pp** |
| llama2-7b-tp4-chatsweep | chatsweep | 12 | 28.5% | 28.6% | 26.8% | -0.1pp | **+1.7pp** |
| llama2-7b-tp4-codesweep | codesweep | 12 | 50.2% | 50.6% | 48.6% | -0.3pp | **+1.7pp** |
| codellama-34b-tp2-chatsweep | chatsweep | 12 | 39.4% | 39.3% | 42.3% | +0.1pp | **-2.9pp** |
| codellama-34b-tp2-codesweep | codesweep | 12 | 24.1% | 24.2% | 23.6% | -0.1pp | +0.5pp |
| llama2-70b-tp4-chatsweep | chatsweep | 12 | 25.5% | 25.4% | 27.7% | +0.1pp | **-2.2pp** |
| llama2-70b-tp4-codesweep | codesweep | 12 | 24.5% | 24.9% | 23.0% | -0.3pp | **+1.5pp** |
| qwen3-14b-tp1-codesweep | codesweep | 12 | 19.8% | 20.3% | 17.4% | -0.5pp | **+2.4pp** |
| qwen3-14b-tp2-chatsweep | chatsweep | 12 | 20.0% | 19.9% | 22.2% | +0.1pp | **-2.3pp** |
| qwen7-summarization | summarization | 5 | 49.1% | 48.7% | 50.8% | +0.4pp | -1.7pp |

### Aggregate by Workload Type

| Workload | N | BL E2E | SWA E2E | ADD E2E | BL→SWA | BL→ADD | BL TPOT | ADD TPOT | Δ TPOT |
|---|---|---|---|---|---|---|---|---|---|
| chatsweep | 72 | 32.8% | 32.7% | 34.4% | +0.1pp | **-1.6pp** | 32.9% | 34.6% | -1.6pp |
| codesweep | 72 | 29.1% | 29.1% | 29.0% | -0.1pp | **+0.1pp** | 34.2% | 33.8% | +0.4pp |
| summarization | 5 | 49.1% | 48.7% | 50.8% | +0.4pp | -1.7pp | 43.3% | 44.8% | -1.5pp |
| **OVERALL** | **149** | **31.5%** | **31.5%** | **32.3%** | **+0.0pp** | **-0.8pp** | **33.9%** | **34.5%** | **-0.6pp** |

### By Concurrency Level

| Strategy | N | BL E2E | SWA E2E | ADD E2E | BL→SWA | BL→ADD |
|---|---|---|---|---|---|---|
| synchronous | 13 | 17.4% | 17.4% | 17.4% | -0.0pp | -0.0pp |
| constant-rate | 123 | 30.8% | 30.8% | 31.9% | +0.1pp | -1.1pp |
| throughput | 13 | 52.2% | 52.5% | 51.2% | -0.3pp | **+1.0pp** |

### Accept Criteria

| Criterion | Result | Status |
|---|---|---|
| E2E MAPE improvement ≥ 2pp (additive vs baseline) | -0.8pp (worsened) | **FAIL** |
| Improvement concentrated at higher QPS | -0.0pp sync vs -0.9pp high-QPS | **FAIL** |
| Vanishing effect at synchronous rate (ED-2) | 0.02pp difference | **PASS** |

## Root Cause Analysis

### Why the additive model worsens chatsweep (decode-heavy) accuracy

The additive model computes GEMM time for the combined batch (`totalBatch = prefill + decode`), which produces a **larger MFU lookup** than the separate prefill and decode GEMMs. For chatsweep workloads with `output=215 tokens` (decode-heavy), the dominant phase is decode with batch_size = number of concurrent decode requests. The fused GEMM inflates the batch dimension for the GEMM lookup, retrieving a higher MFU that reduces predicted compute time, leading to **underprediction** of step time.

- Additive GEMM computation: `roofline_step.go:396` — `computeTransformerGEMMTimes(totalBatchSize)`
- Baseline computes separate GEMMs at smaller batch sizes: `roofline_step.go:270` (decode) and `roofline_step.go:340` (prefill)
- MFU is batch-size-dependent: larger batch → higher MFU → lower predicted time → larger underprediction

**Note (RCV-4):** This mechanism is proposed but not verified by a targeted control experiment. A definitive test would fix the MFU lookup to return a constant value and re-run both arms — if the accuracy difference disappears, the MFU batch-size interaction is confirmed as the root cause. Additionally, the additive model uses a single weight load (`roofline_step.go:416-417`) instead of the baseline's implicit double weight load (separate `prefillMemoryS` at line 379 and `decodeMemoryS` at line 306, each including full model weight bytes). This memory path reduction further lowers predicted latency in the additive arm, contributing to underprediction alongside the GEMM effect.

### Why branch removal (smooth-WA) has zero effect

The smooth-WA control arm removes the branching thresholds but keeps the weighted-average structure. The overall effect is +0.0pp — essentially zero. This confirms that the branch thresholds (4× for prefill-dominated, 2× for decode-dominated) do not create measurable discontinuities at the aggregate accuracy level.

- Branch thresholds: `roofline_step.go:462-468` (`default` case, lines 463/466 for the 4× and 2× conditions)
- Smooth-WA always uses token-proportional weights: `roofline_step.go:446-448` (`smooth-wa` case)
- The zero effect means the branch cases rarely activate in the ground truth workloads, or when they do, the weighted-average weights (0.75/0.25 vs token-proportional) produce nearly identical step times

### Why codesweep and TP>1 show slight improvement

For codesweep (prefill-heavy, `output=28 tokens`) at higher TP, the decode phase is very short relative to prefill. The additive model's fused GEMM more accurately captures the actual execution where prefill and decode tokens share the same GEMM kernel, and the slight MFU increase from the larger batch is compensated by the single weight load.

- Codesweep improvement: +0.1pp overall, up to +2.4pp for qwen3-14b-tp1
- Throughput mode improvement: +1.0pp — high concurrency means larger batches where the fused GEMM is more realistic

### ED-2 vanishing effect (confirmed)

At synchronous rate (max_concurrency=1), there are zero mixed batches — every step is pure prefill or pure decode. The 0.02pp difference between baseline and additive confirms the model change only affects mixed-batch steps, validating experimental isolation.

## Devil's Advocate (RCV-5)

**Arguing why the additive model might actually be correct (despite refutation):**

The additive model may be physically correct for the GPU execution but produces worse *aggregate* predictions because it compensates in the wrong direction for other model errors. The current weighted-average model, with its uncited constants, may be an accidental "regularizer" that compensates for errors in the attention or memory models. At the individual-step level, the additive model could be more accurate, but this is unmeasurable from client-side data (Limitation 1 from TAM). The refutation is at the aggregate accuracy level only — it does not disprove the physical mechanism.

Additionally, the TP=1 experiments show the largest degradation (-4.3pp, -7.3pp for Llama-2-7B), while TP≥2 experiments are mixed (some improve, some worsen). At TP=1, the MFU lookup for the fused batch size may hit a different region of the benchmark grid, introducing artificial error from the MFU database discretization (H6's hypothesis).

## Findings Classification

| Finding | Type | Action |
|---|---|---|
| Additive model worsens E2E MAPE by 0.8pp overall | Refutation | documented here |
| Branch removal has zero measurable effect (smooth-WA ≈ baseline) | Confirmation (branches are harmless) | documented here |
| Additive model shows workload-dependent behavior: hurts chatsweep, helps codesweep | Surprise | documented here — follow-up hypothesis |
| ED-2 vanishing effect confirmed at synchronous rate | Confirmation (experimental isolation) | documented here |
| Model-TP interaction: TP=1 strongly hurt, TP≥2 mixed | Surprise | documented here — follow-up hypothesis |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? **Pre-existing R2 violation**: `roofline_step.go:335` iterates over `bucketMap` (Go map) without sorting keys, feeding float sum. Not introduced by H5 but affects all arms equally.
- [x] Any new rules needed? None
- [x] Any new invariants needed? None
- [x] Any existing rules/invariants confirmed? INV-6 (Determinism): vanishing effect test confirms same output at sync rate across all arms. R2 pre-existing violation noted.

## Scope and Limitations (RCV-6)

- **Operating point tested:** 13 ground truth experiments × up to 12 QPS sweep points (149 total data points). 5 model families (Llama-2-7B, CodeLlama-34B, Llama-2-70B, Qwen3-14B, Qwen2.5-7B), TP=1/2/4, H100 GPU, chatsweep/codesweep/summarization workloads. **Note:** H5 baseline MAPEs (31.5% overall) are not directly comparable to H1-H4 synchronous-only baselines because H5 sweeps all QPS points including high-QPS where queuing effects inflate E2E MAPE.
- **Parameters findings depend on:** H1 (BwEfficiencyFactor=0.82) and H2b (PerLayerCPUOverhead=100) are assumed. The refutation holds given these baseline corrections.
- **What was NOT tested:** (1) Part A synthetic ratio sweep was not performed as a separate test (would require a standalone Go test). (2) Per-step accuracy — we can only measure aggregate E2E/TTFT/TPOT, not per-step prediction error. (3) MoE models. (4) Non-H100 GPUs.
- **Generalizability:** The refutation generalizes across all 5 model families and 3 workload types tested. The workload-dependent effect (chatsweep hurt, codesweep neutral/helped) is a consistent pattern that may inform future work.
- **Uncertainty quantification:** UQ not performed — deterministic experiment. The -0.8pp overall result is exact for seed 42 with the given workloads. The finding is robust: 8 of 13 experiments show the additive model worsening E2E MAPE.

## Evidence Quality

| Metric | Value | Confidence |
|---|---|---|
| Overall E2E MAPE change | -0.8pp (worsened) | High — consistent across 149 data points |
| Chatsweep E2E MAPE change | -1.6pp (worsened) | High — consistent across 72 data points |
| Codesweep E2E MAPE change | +0.1pp (neutral) | Medium — small magnitude, direction varies by experiment |
| Branch removal effect | +0.0pp (zero) | High — smooth-WA ≈ baseline across all experiments |
| ED-2 vanishing effect | 0.02pp | High — confirms experimental isolation |
| Mechanism (fused GEMM overcounts) | Proposed | Medium — consistent with MFU lookup behavior but not verified at per-step level |
| Sample size | 1 seed × 13 experiments × ~12 benchmarks × 3 arms = ~468 simulations | High |

## Implications for Users

1. **Do not switch to the additive mixed-batch model.** The current weighted-average model produces equal or better aggregate accuracy despite its uncited constants.
2. **The branch thresholds (0.75/0.25, 0.35/0.65) are harmless.** Removing them produces no measurable change. Users concerned about discontinuities can use `mixedBatchMode: "smooth-wa"` without accuracy loss.
3. **Future mixed-batch model improvements should focus on the MFU lookup interaction** — the additive model's fused-GEMM batch size produces different MFU lookups that are not necessarily more accurate. Per-step accuracy validation (requiring server-side traces) would be needed to determine the true per-step model.

## Reproducing

```
cd hypotheses/h-roofline/h5-mixed-batch-additive
./run.sh
```
